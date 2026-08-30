"""Gates for the I5 frozen video tokenizer and episode split (`I5-S1-TOK`).

What S1 has to guarantee before any latent cache is trusted, per D-062/D-064 and the plan's S1 row:

* the tokenizer is frozen and stays frozen through the parent's `train()` (AGENTS.md 6);
* encoding is **deterministic** -- the cache must be bit-identical across processes, which is why
  `mode()` is used and never `sample()`;
* the latent shape follows `T_lat = (T - 1) // 4 + 1` and the 9-frame window is aligned;
* **latent frame 0 is future-free**: raw frames 1..T-1 must not influence it at all, otherwise
  conditioning on it would leak the future into a policy-visible input (AGENTS.md 5);
* the split is deterministic and every task populates all three splits.

The tokenizer tests run two ways. A stub VAE covers the whole contract on CPU with no weights. The
causality and shape claims about the *real* pinned VAE are checked too, but only when the weights are
present -- those tests skip rather than fail on a machine without them, following the environment
fingerprint convention of the I2 parity suite (D-039).
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import torch
import torch.nn as nn

from starVLA.dataloader.i5_episode_split import (
    SPLITS,
    TEST,
    TRAIN,
    VAL,
    assign_splits,
    split_counts,
)
from starVLA.model.modules.world_model.latent_tokenizer import (
    I5_WINDOW_FRAMES,
    LATENT_CHANNELS,
    FrozenLatentTokenizer,
    frames_to_vae_input,
    is_aligned_window,
    latent_frame_count,
    normalize_latents,
    shard_of,
)


#: Set to a Wan-2.2-VAE directory to run the real-weight checks.
REAL_VAE_PATH = os.environ.get("I5_WAN_VAE_PATH")


class _StubConfig:
    scale_factor_spatial = 16
    scale_factor_temporal = 4
    z_dim = LATENT_CHANNELS
    latents_mean = [0.25] * LATENT_CHANNELS
    latents_std = [2.0] * LATENT_CHANNELS


class _StubDist:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator=None) -> torch.Tensor:  # must never be reached by encode()
        raise AssertionError("encode() must use mode(), not sample()")


class _StubOutput:
    def __init__(self, mean: torch.Tensor) -> None:
        self.latent_dist = _StubDist(mean)


class _StubVAE(nn.Module):
    """Causal stand-in: latent frame k is the mean of raw frames 0..4k, so the dependency
    structure the real VAE was measured to have is reproduced exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _StubConfig()
        self.proj = nn.Conv3d(3, LATENT_CHANNELS, kernel_size=1)
        self.dropout = nn.Dropout(0.5)  # makes train/eval mode observable

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, video: torch.Tensor) -> _StubOutput:
        projected = self.proj(video)  # [1, C, T, H, W]
        pooled = torch.nn.functional.avg_pool3d(projected, kernel_size=(1, 16, 16))
        frames = []
        for k in range(latent_frame_count(video.shape[2])):
            frames.append(pooled[:, :, : 4 * k + 1].mean(dim=2))
        return _StubOutput(torch.stack(frames, dim=2))


def _clip(num_frames: int = I5_WINDOW_FRAMES, size: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randint(
        0, 256, size=(num_frames, size, size, 3), dtype=np.uint8
    )


class LatentGeometryTest(unittest.TestCase):
    def test_latent_frame_count_and_alignment(self):
        self.assertEqual([latent_frame_count(t) for t in (1, 5, 8, 9, 12, 13)], [1, 2, 2, 3, 3, 4])
        # The I5 window is aligned; the policy path's 8-frame window is not, which is exactly why
        # D-062 gave the generator its own window.
        self.assertTrue(is_aligned_window(I5_WINDOW_FRAMES))
        self.assertFalse(is_aligned_window(8))
        self.assertEqual(latent_frame_count(I5_WINDOW_FRAMES), 3)

    def test_frames_to_vae_input_scaling(self):
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        frames[1] = 255
        video = frames_to_vae_input(frames)
        self.assertEqual(tuple(video.shape), (1, 3, 2, 4, 4))
        self.assertAlmostEqual(video[0, 0, 0].min().item(), -1.0, places=5)
        self.assertAlmostEqual(video[0, 0, 1].max().item(), 1.0, places=5)

    def test_frames_to_vae_input_rejects_wrong_layout(self):
        with self.assertRaisesRegex(ValueError, r"\[T, H, W, 3\]"):
            frames_to_vae_input(np.zeros((3, 4, 4), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8"):
            frames_to_vae_input(np.zeros((2, 4, 4, 3), dtype=np.float32))

    def test_normalize_latents_is_the_wan_convention(self):
        latents = torch.full((1, 2, 1, 1, 1), 3.0)
        out = normalize_latents(latents, torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(out, torch.full((1, 2, 1, 1, 1), 1.0))
        with self.assertRaisesRegex(ValueError, "do not match stats"):
            normalize_latents(latents, torch.tensor([1.0]), torch.tensor([2.0]))


class FrozenLatentTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FrozenLatentTokenizer(_StubVAE())

    def _assert_frozen(self, where: str) -> None:
        trainable = [n for n, p in self.tokenizer.vae.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, [], f"vae has trainable params {where}")
        for name, module in self.tokenizer.vae.named_modules():
            self.assertFalse(module.training, f"vae.{name or '<root>'} in train mode {where}")

    def test_frozen_after_construction_and_through_train(self):
        self._assert_frozen("after construction")
        self.tokenizer.train()
        self._assert_frozen("after train()")
        parent = nn.Module()
        parent.tok = self.tokenizer
        parent.head = nn.Linear(4, 4)
        parent.train()
        self._assert_frozen("after parent.train() recursion")
        self.assertTrue(parent.head.training)

    def test_rejects_a_vae_with_the_wrong_compression(self):
        vae = _StubVAE()
        vae.config.scale_factor_temporal = 8
        with self.assertRaisesRegex(ValueError, "expects the Wan 2.2 VAE compression"):
            FrozenLatentTokenizer(vae)

    def test_encode_shape_and_grid(self):
        latents = self.tokenizer.encode(_clip(size=32))
        self.assertEqual(tuple(latents.shape), (1, LATENT_CHANNELS, 3, 2, 2))
        self.assertEqual(self.tokenizer.latent_grid(256, 256), (16, 16))
        with self.assertRaisesRegex(ValueError, "not divisible"):
            self.tokenizer.latent_grid(250, 256)

    def test_encode_is_deterministic_and_never_samples(self):
        clip = _clip()
        first = self.tokenizer.encode(clip)
        second = self.tokenizer.encode(clip)
        # Bit-identical, not merely close: the offline cache is keyed on this.
        self.assertTrue(torch.equal(first, second))
        # _StubDist.sample() raises, so reaching this line at all proves mode() was used.

    def test_encode_does_not_build_a_graph(self):
        latents = self.tokenizer.encode(_clip())
        self.assertFalse(latents.requires_grad)

    def test_latent_frame_zero_is_future_free(self):
        """Conditioning on latent frame 0 must not see frames 1..T-1 (AGENTS.md 5)."""
        clip = _clip()
        base = self.tokenizer.encode(clip)
        for index in range(1, I5_WINDOW_FRAMES):
            perturbed = clip.copy()
            perturbed[index] = 255 - perturbed[index]
            latents = self.tokenizer.encode(perturbed)
            self.assertTrue(
                torch.equal(latents[:, :, 0], base[:, :, 0]),
                f"raw frame {index} changed latent frame 0",
            )
            self.assertFalse(
                torch.equal(latents[:, :, -1], base[:, :, -1]),
                f"raw frame {index} left the last latent frame untouched; causality test is vacuous",
            )

    def test_encode_is_deterministic_at_a_fixed_batch_size(self):
        """The determinism the cache actually needs: same configuration, bit-identical output."""
        clips = np.stack([_clip(seed=s) for s in range(3)])
        self.assertTrue(
            torch.equal(self.tokenizer.encode_windows(clips), self.tokenizer.encode_windows(clips))
        )
        self.assertTrue(torch.equal(self.tokenizer.encode(clips[0]), self.tokenizer.encode(clips[0])))

    def test_batched_encode_matches_single_within_float_tolerance(self):
        """Batching agrees with single encoding to float rounding, but **not** bitwise.

        Reduction order depends on batch size, so a batch-3 encode and three batch-1 encodes differ
        at float32 epsilon: measured on the pinned weights, max absolute difference 1.9e-6 and max
        relative 6.1e-7, against a latent scale of about 0.45 and a VAE posterior std of 8e-5. That
        is far below any signal I5 measures, but it does mean the cache is reproducible only
        *together with its batch size* -- the same shape of conclusion D-049 reached for the depth
        cache and its environment. The builder therefore records the batch size in the cache index.
        """
        clips = np.stack([_clip(seed=s) for s in range(3)])
        batched = self.tokenizer.encode_windows(clips)
        self.assertEqual(tuple(batched.shape), (3, LATENT_CHANNELS, 3, 2, 2))
        for index, clip in enumerate(clips):
            torch.testing.assert_close(
                batched[index : index + 1], self.tokenizer.encode(clip), rtol=0, atol=1e-5
            )

    def test_encode_windows_rejects_wrong_layout(self):
        with self.assertRaisesRegex(ValueError, r"\[B, T, H, W, 3\]"):
            self.tokenizer.encode_windows(_clip())

    def test_normalisation_is_applied_by_default(self):
        clip = _clip()
        raw = self.tokenizer.encode(clip, normalize=False)
        normalised = self.tokenizer.encode(clip)
        expected = (raw - 0.25) / 2.0
        torch.testing.assert_close(normalised, expected)


@unittest.skipUnless(REAL_VAE_PATH, "set I5_WAN_VAE_PATH to run the real-weight checks")
class RealWanVaeTest(unittest.TestCase):
    """Same claims, against the pinned weights. Skipped when they are not on this machine."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = FrozenLatentTokenizer.from_pretrained(REAL_VAE_PATH)

    def test_compression_and_shape_on_a_real_window(self):
        latents = self.tokenizer.encode(_clip(size=64))
        self.assertEqual(tuple(latents.shape), (1, LATENT_CHANNELS, 3, 4, 4))

    def test_frame_zero_is_future_free_on_real_weights(self):
        clip = _clip(size=64)
        base = self.tokenizer.encode(clip)
        for index in (1, 4, 5, I5_WINDOW_FRAMES - 1):
            perturbed = clip.copy()
            perturbed[index] = 255 - perturbed[index]
            latents = self.tokenizer.encode(perturbed)
            self.assertTrue(
                torch.equal(latents[:, :, 0], base[:, :, 0]),
                f"raw frame {index} changed latent frame 0",
            )

    def test_encode_is_bit_identical_across_calls(self):
        clip = _clip(size=64)
        self.assertTrue(torch.equal(self.tokenizer.encode(clip), self.tokenizer.encode(clip)))


class ShardingTest(unittest.TestCase):
    def test_every_position_lands_in_exactly_one_shard(self):
        for shard_count in (1, 3, 8):
            owners = [shard_of(position, shard_count) for position in range(200)]
            self.assertEqual(set(owners), set(range(shard_count)))
            # Round-robin: consecutive episodes go to different workers, so the long episodes of
            # libero_10 spread out instead of piling onto one shard.
            for shard in range(shard_count):
                count = owners.count(shard)
                self.assertLessEqual(abs(count - 200 / shard_count), 1)

    def test_shard_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "shard_count"):
            shard_of(0, 0)


class EpisodeSplitTest(unittest.TestCase):
    def test_deterministic_and_all_tasks_in_all_splits(self):
        episodes_by_task = {f"task{t}": list(range(t * 42, t * 42 + 42)) for t in range(40)}
        first = assign_splits(episodes_by_task)
        second = assign_splits(episodes_by_task)
        self.assertEqual(first, second)

        counts = split_counts(first.values())
        self.assertEqual(sum(counts.values()), 40 * 42)
        for task, episodes in episodes_by_task.items():
            per_task = {first[e] for e in episodes}
            self.assertEqual(per_task, set(SPLITS), f"{task} misses a split")

    def test_ratio_is_eighty_ten_ten(self):
        assignment = assign_splits({"t": list(range(100))})
        counts = split_counts(assignment.values())
        self.assertEqual(counts, {TRAIN: 80, VAL: 10, TEST: 10})

    def test_interleaved_not_contiguous(self):
        # A contiguous cut would put every held-out episode at the end of the task; interleaving
        # spreads them, so the split cannot inherit whatever order the episode index encodes.
        assignment = assign_splits({"t": list(range(30))})
        held_out = sorted(e for e, name in assignment.items() if name != TRAIN)
        self.assertEqual(held_out, [8, 9, 18, 19, 28, 29])

    def test_too_small_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot populate"):
            assign_splits({"tiny": [0, 1, 2]})
        relaxed = assign_splits({"tiny": [0, 1, 2]}, require_all_splits=False)
        self.assertEqual(split_counts(relaxed.values()), {TRAIN: 3, VAL: 0, TEST: 0})

    def test_counting_is_over_names_not_a_mapping(self):
        """LeRobot restarts episode indices per suite, so counting a mapping keyed on them collapses.

        The first full build reported 454 episodes across the splits instead of 1693 for exactly
        this reason: four suites reuse the low episode indices, and a dict keyed on `episode_index`
        kept only the last suite's value. Counting split names cannot express that.
        """
        names = ["train"] * 1389 + ["val"] * 154 + ["test"] * 150
        self.assertEqual(split_counts(names), {TRAIN: 1389, VAL: 154, TEST: 150})
        self.assertEqual(sum(split_counts(names).values()), 1693)
        with self.assertRaisesRegex(ValueError, "unknown split"):
            split_counts(["train", "holdout"])

    def test_duplicate_episode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "more than one task"):
            assign_splits({"a": list(range(10)), "b": [5]})


if __name__ == "__main__":
    unittest.main()
