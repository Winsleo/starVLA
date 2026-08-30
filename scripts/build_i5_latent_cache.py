"""Build the I5 offline latent cache from the LeRobot LIBERO conversion (`I5-S1-TOK`).

One `episode_*.npz` per episode, holding the normalised Wan 2.2 latents of every 9-frame window in
that episode, plus an `index.json` recording the split assignment and everything needed to reproduce
the cache. Orchestration only: the tensor contract lives in
`starVLA/model/modules/world_model/latent_tokenizer.py` and the split in
`starVLA/dataloader/i5_episode_split.py`, both unit-tested without weights.

Fixed by decision, not by flag default:

* window = 9 frames (`D-062`) -- aligned, so no raw frame is dropped, and `T_lat = 3` gives the two
  prediction steps that make the free-running gate measurable;
* single view (`I5` scope is fixed-camera single-view), the third-person `observation.images.image`;
* split = 80/10/10 by episode index within each task (`D-064`), and every task must populate all
  three splits or the build refuses to start;
* latents come from the posterior **mode**, never a sample, so the cache is reproducible.

Reproducibility caveat recorded in the index: determinism holds per configuration. Repeated runs at
the same batch size, device and dtype are bit-identical; a different batch size shifts results by
float32 epsilon (about 2e-6 absolute), because reduction order depends on it. Compare with `D-049`,
which reached the same shape of conclusion for the depth cache and its environment.

Decode path: frames are read sequentially with PyAV, the library the trainer's pinned
`video_backend="torchvision_av"` wraps. The reference lineage measured that a sequential pass is
byte-identical to the trainer's per-timestamp seek, which is what makes an absolute frame index a
valid cache key; that measurement is inherited here, not re-run. What this script does verify per
episode is the observable part: the decoded frame count equals `episodes.jsonl`'s `length`, and it
refuses the episode instead of substituting frames when it does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.dataloader.i5_episode_split import assign_splits, split_counts
from starVLA.model.modules.world_model.latent_tokenizer import (
    I5_WINDOW_FRAMES,
    LATENT_CHANNELS,
    FrozenLatentTokenizer,
    is_aligned_window,
    latent_frame_count,
    shard_of,
)

DEFAULT_VIEW = "observation.images.image"
EXPECTED_FPS = 20
EXPECTED_FRAME_SHAPE = [256, 256, 3]


def read_suite_meta(suite_dir: Path, view: str) -> tuple[dict, list[dict]]:
    """`info.json` plus the episode records, with the assumptions I5 relies on asserted."""
    info = json.loads((suite_dir / "meta" / "info.json").read_text())
    feature = info.get("features", {}).get(view)
    if feature is None:
        raise SystemExit(f"{suite_dir.name}: no feature {view!r}")
    if feature.get("shape") != EXPECTED_FRAME_SHAPE:
        raise SystemExit(
            f"{suite_dir.name}: {view} is {feature.get('shape')}, expected {EXPECTED_FRAME_SHAPE}"
        )
    fps = feature.get("info", {}).get("video.fps") or info.get("fps")
    if int(fps) != EXPECTED_FPS:
        raise SystemExit(f"{suite_dir.name}: fps is {fps}, expected {EXPECTED_FPS}")
    episodes = [
        json.loads(line)
        for line in (suite_dir / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return info, episodes


def decode_episode(path: Path, expected_frames: int) -> np.ndarray:
    """`[T, H, W, 3]` uint8, refusing the episode if the frame count disagrees with the metadata."""
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != expected_frames:
        raise SystemExit(
            f"{path}: decoded {len(frames)} frames, metadata says {expected_frames}. Refusing to "
            "cache: absolute frame index would no longer be a valid key."
        )
    return np.stack(frames)


def episode_video_path(suite_dir: Path, info: dict, view: str, episode: int) -> Path:
    """Resolve one episode's video through the dataset's own `video_path` template.

    Not reconstructed by hand: the chunk index is `episode_index // chunks_size`, and hardcoding
    `chunk-000` is only correct while every suite stays under one chunk. Upstream fixed exactly that
    class of bug in `get_video_path` (starVLA #359), and the trainer's own dataset reads the template
    from `info.json` (`_get_video_path_pattern`), so this follows the same source of truth.
    """
    template = info.get("video_path")
    if not template:
        raise SystemExit(f"{suite_dir.name}: info.json has no video_path template")
    chunk_size = int(info.get("chunks_size", 1000))
    relative = template.format(
        episode_chunk=episode // chunk_size, video_key=view, episode_index=episode
    )
    return suite_dir / relative


def window_starts(length: int, window: int, stride: int) -> list[int]:
    return list(range(0, max(0, length - window + 1), stride))


def encode_episode(
    tokenizer: FrozenLatentTokenizer, frames: np.ndarray, starts: list[int], window: int, batch: int
) -> np.ndarray:
    """`[N, C, T_lat, h, w]` float16 for every window start."""
    chunks: list[np.ndarray] = []
    for offset in range(0, len(starts), batch):
        group = starts[offset : offset + batch]
        stacked = np.stack([frames[s : s + window] for s in group])
        latents = tokenizer.encode_windows(stacked)
        chunks.append(latents.to(torch.float32).cpu().numpy().astype(np.float16))
    return np.concatenate(chunks, axis=0)


def check_contract(index: dict, *, require_complete: bool) -> None:
    """Refuse to write an index that contradicts the I5 latent contract.

    The process that declares a cache complete is the right place to enforce what "complete" means,
    so a cluster job does not have to re-derive these assertions in shell. Structural checks always
    run; coverage checks only when the caller asked for a full build.
    """
    problems: list[str] = []
    if index["window"] != I5_WINDOW_FRAMES:
        problems.append(f"window is {index['window']}, D-062 fixes it at {I5_WINDOW_FRAMES}")
    if index["latent_frames"] != latent_frame_count(I5_WINDOW_FRAMES):
        problems.append(f"latent_frames is {index['latent_frames']}, expected 3")
    shape = index.get("latent_shape")
    if shape is not None and shape[0] != LATENT_CHANNELS:
        problems.append(f"latent channels {shape[0]}, expected {LATENT_CHANNELS}")
    if shape is not None and tuple(shape[2:]) != (16, 16):
        problems.append(f"latent grid {tuple(shape[2:])}, expected (16, 16) to match I3/I4 targets")
    if index["posterior"] != "mode":
        problems.append(f"posterior is {index['posterior']!r}; the cache must be reproducible")
    if not index["normalized"]:
        problems.append("latents are not normalised; the DiT input space would disagree")
    if require_complete:
        missing = index["num_encodable_episodes"] - index["num_cached_episodes"]
        if missing:
            problems.append(
                f"{missing} of {index['num_encodable_episodes']} encodable episode(s) have no shard "
                "on disk"
            )
        elif index["subset"]:
            problems.append(
                "this is a --limit-episodes subset build, so it can never be a complete cache"
            )
        empty = [name for name, count in index["split_episode_counts"].items() if count == 0]
        if empty:
            problems.append(f"split(s) {', '.join(empty)} are empty (D-064 requires all three)")
    if problems:
        raise SystemExit("latent cache contract violated:\n  - " + "\n  - ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="LeRobot LIBERO conversion root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae-path", type=str, required=True, help="directory holding the Wan VAE")
    parser.add_argument("--vae-subfolder", default="vae")
    parser.add_argument("--suites", nargs="*", default=None, help="default: every suite under root")
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--window", type=int, default=I5_WINDOW_FRAMES)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-episodes", type=int, default=0, help="per suite; 0 = all. Subset runs")
    parser.add_argument(
        "--shard-count", type=int, default=1, help="split the episode list across this many workers"
    )
    parser.add_argument("--shard-index", type=int, default=0, help="which shard this process encodes")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="write index.json from what is already on disk, without encoding anything",
    )
    parser.add_argument(
        "--revision", default=None, help="submodule revision to stamp into the index for provenance"
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless every encodable episode is cached and every split is populated",
    )
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit(f"bad shard {args.shard_index}/{args.shard_count}")
    # Sharded workers write NPZ only; one --index-only pass afterwards writes the single index.json,
    # so the artifact contract stays one index per cache no matter how many workers built it. The
    # split assignment is computed from the full episode list in every process, so which shard
    # encodes an episode never changes which split it lands in.
    writes_index = args.index_only or args.shard_count == 1

    if not is_aligned_window(args.window):
        raise SystemExit(
            f"--window {args.window} is not 1 + 4k, so the VAE would drop trailing frames; "
            f"I5 is fixed at {I5_WINDOW_FRAMES} (D-062)"
        )
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    suites = sorted(args.suites or [p.name for p in args.data_root.iterdir() if (p / "meta").is_dir()])
    if not suites:
        raise SystemExit(f"no LeRobot suites under {args.data_root}")

    # The index pass reads metadata and file existence only, so it must not need 1.4 GB of weights.
    tokenizer = None
    if not args.index_only:
        tokenizer = FrozenLatentTokenizer.from_pretrained(args.vae_path, subfolder=args.vae_subfolder)
        tokenizer = tokenizer.to(args.device)
        tokenizer.enforce_frozen()

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    subset = args.limit_episodes > 0
    latent_shape: list[int] | None = None
    global_position = 0  # counts encodable episodes across suites, so sharding balances globally

    for suite in suites:
        suite_dir = args.data_root / suite
        info, episodes = read_suite_meta(suite_dir, args.view)
        if subset:
            episodes = episodes[: args.limit_episodes]
        by_task: dict[str, list[int]] = {}
        for record in episodes:
            task = record["tasks"][0] if record.get("tasks") else "<unknown>"
            by_task.setdefault(task, []).append(int(record["episode_index"]))
        # A subset cannot populate every split; only a full build is held to that gate.
        assignment = assign_splits(by_task, require_all_splits=not subset)

        for position, record in enumerate(episodes, start=1):
            episode = int(record["episode_index"])
            length = int(record["length"])
            starts = window_starts(length, args.window, args.stride)
            relative = Path(suite) / args.view / f"episode_{episode:06d}.npz"
            destination = args.output / relative
            entry = {
                "suite": suite,
                "episode_index": episode,
                "task": record["tasks"][0] if record.get("tasks") else None,
                "split": assignment[episode],
                "length": length,
                "num_windows": len(starts),
                "path": str(relative),
            }
            if not starts:
                entry["skipped"] = f"episode shorter than the {args.window}-frame window"
                records.append(entry)
                continue
            global_position += 1
            entry["cached"] = destination.exists()
            if entry["cached"] or args.index_only:
                records.append(entry)
                continue
            if shard_of(global_position, args.shard_count) != args.shard_index:
                records.append(entry)
                continue

            video = episode_video_path(suite_dir, info, args.view, episode)
            frames = decode_episode(video, length)
            latents = encode_episode(tokenizer, frames, starts, args.window, args.batch_size)
            latent_shape = list(latents.shape[1:])

            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(f".partial-{position}.npz")
            np.savez_compressed(
                partial, latents=latents, window_starts=np.asarray(starts, dtype=np.int32)
            )
            partial.replace(destination)
            entry["cached"] = True
            records.append(entry)
            if position % 25 == 0:
                print(f"{suite}: {position}/{len(episodes)}", flush=True)

    encodable = [r for r in records if not r.get("skipped")]
    cached = [r for r in encodable if r.get("cached")]
    if not writes_index:
        print(
            f"shard {args.shard_index}/{args.shard_count}: {len(cached)}/{len(encodable)} episode(s) "
            "on disk; run --index-only afterwards to write index.json"
        )
        return

    index = {
        # Complete means every encodable episode has a shard on disk -- checked, not assumed from
        # this process having finished.
        "complete": len(cached) == len(encodable) and not subset,
        "subset": subset,
        "num_episodes": len(records),
        "num_encodable_episodes": len(encodable),
        "num_cached_episodes": len(cached),
        "num_windows": sum(r["num_windows"] for r in cached),
        "split_episode_counts": split_counts({r["episode_index"]: r["split"] for r in records}),
        "suites": suites,
        "view": args.view,
        "window": args.window,
        "stride": args.stride,
        "latent_shape": latent_shape,
        "latent_frames": latent_frame_count(args.window),
        "posterior": "mode",
        "normalized": True,
        "dtype": "float16",
        # Determinism is per configuration: the same batch size, device and dtype reproduce this
        # cache bit-for-bit, a different batch size shifts it by float32 epsilon (about 2e-6).
        "batch_size": args.batch_size,
        "shard_count": args.shard_count,
        "device": args.device,
        "compute_dtype": str(next(tokenizer.vae.parameters()).dtype) if tokenizer else None,
        "vae_path": args.vae_path,
        "vae_subfolder": args.vae_subfolder,
        "split_rule": "80/10/10 interleaved by episode index within each task (D-064)",
        "decisions": ["D-062", "D-064"],
        # Travels with every number derived from this cache.
        "caveat": (
            "The step-6000 policy checkpoint was trained on all of libero_all, so held-out episodes "
            "are held out for the generator only and absolute generation quality reads "
            "optimistically. Condition-ablation comparisons are unaffected: their arms share one "
            "policy and one clip set."
        ),
        "starVLA_rebase_revision": args.revision,
        "episodes": records,
    }
    check_contract(index, require_complete=args.require_complete)
    (args.output / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(
        f"cached {len(cached)}/{len(encodable)} episode(s), {index['num_windows']} window(s), "
        f"complete={index['complete']}, "
        f"splits {index['split_episode_counts']}"
    )


if __name__ == "__main__":
    main()
