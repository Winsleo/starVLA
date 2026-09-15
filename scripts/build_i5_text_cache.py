"""Precompute the I5 text conditioning (`I5-S3-TF`, per D-064).

LIBERO has 40 task instructions and the UMT5 encoder is frozen, so the language condition is encoded
once here rather than loading 11 GB of text encoder at every training step. Orchestration only: the
encoding convention lives in `starVLA/i5/text_condition.py`, taken from the canonical inference path
and unit-tested.

Must run in the **pinned** environment, like the other two caches: it is the environment whose
numerics the recorded results correspond to.

The cache is keyed by the instruction string, not by task index, because the same instruction can
carry different `task_index` values across suites while the embedding is a pure function of the text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.i5.text_condition import (
    TEXT_EMBED_DIM,
    TEXT_SEQUENCE_LENGTH,
    assert_padding_is_zero,
    clean_prompt,
    encode_prompts,
)


def collect_instructions(data_root: Path, suites: list[str]) -> dict[str, list[str]]:
    """Instruction -> the suites it appears in, from each suite's `meta/tasks.jsonl`."""
    found: dict[str, list[str]] = {}
    for suite in suites:
        path = data_root / suite / "meta" / "tasks.jsonl"
        if not path.is_file():
            raise SystemExit(f"{suite}: no meta/tasks.jsonl")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)["task"]
            found.setdefault(task, []).append(suite)
    if not found:
        raise SystemExit("no instructions found")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wan-path", type=str, required=True, help="Wan checkpoint directory")
    parser.add_argument("--suites", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    suites = sorted(
        args.suites or [p.name for p in args.data_root.iterdir() if (p / "meta").is_dir()]
    )
    instructions = collect_instructions(args.data_root, suites)
    ordered = sorted(instructions)  # deterministic row order, independent of file order
    print(f"{len(ordered)} unique instruction(s) across {len(suites)} suite(s)", flush=True)

    from transformers import T5TokenizerFast, UMT5EncoderModel

    tokenizer = T5TokenizerFast.from_pretrained(args.wan_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.wan_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder = text_encoder.to(args.device).eval()
    text_encoder.requires_grad_(False)

    rows: list[torch.Tensor] = []
    for offset in range(0, len(ordered), args.batch_size):
        batch = ordered[offset : offset + args.batch_size]
        rows.append(
            encode_prompts(
                tokenizer, text_encoder, batch, device=args.device, dtype=torch.float32
            ).cpu()
        )
        print(f"  {min(offset + len(batch), len(ordered))}/{len(ordered)}", flush=True)
    embeds = torch.cat(rows, dim=0)

    # The zero-padded tail is part of the convention, so check it rather than trust it.
    lengths = [
        int(
            tokenizer(
                clean_prompt(text),
                padding="max_length",
                max_length=TEXT_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).attention_mask.sum()
        )
        for text in ordered
    ]
    assert_padding_is_zero(embeds, lengths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        embeds=embeds.numpy().astype(np.float16),
        valid_lengths=np.asarray(lengths, dtype=np.int32),
    )
    partial.replace(args.output)

    index = {
        "instructions": ordered,
        "instruction_to_row": {text: i for i, text in enumerate(ordered)},
        "instruction_suites": {text: sorted(set(instructions[text])) for text in ordered},
        "shape": list(embeds.shape),
        "sequence_length": TEXT_SEQUENCE_LENGTH,
        "embed_dim": TEXT_EMBED_DIM,
        "dtype": "float16",
        "valid_length_range": [min(lengths), max(lengths)],
        "convention": (
            "WanPipeline._get_t5_prompt_embeds with max_sequence_length=512: prompt_clean, "
            "padding to 512 with truncation and special tokens, encoder run with the attention "
            "mask, then valid positions kept and zero-padded back."
        ),
        "wan_path": args.wan_path,
        "suites": suites,
        "decisions": ["D-064"],
    }
    args.output.with_suffix(".index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(
        f"wrote {tuple(embeds.shape)} float16 to {args.output.name}; "
        f"valid lengths {min(lengths)}-{max(lengths)} of {TEXT_SEQUENCE_LENGTH}"
    )


if __name__ == "__main__":
    main()
