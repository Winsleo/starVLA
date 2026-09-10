# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Frozen video tokenizer for the I5 generative baseline (`I5-S1-TOK`).

Wraps the Wan 2.2 VAE as a **frozen** RGB -> video-latent encoder. This is the tokenizer half of
I5: nothing here trains, nothing here generates. `Wan2.py` already loads the same VAE class as part
of the WM4A action-head path; this module exists because I5 needs the encoder on its own, driven by
its own clip window, with the firewall and the determinism contract asserted (D-062, D-064).

Measured properties this module relies on, all verified on the pinned VAE
(`scale_factor_spatial=16`, `scale_factor_temporal=4`, `z_dim=48`):

* `T_lat = (T - 1) // 4 + 1`, so an aligned window has `T = 1 + 4k` frames. The I5 window is
  **9 frames -> 3 latent frames** (D-062). An 8-frame window yields 2 latent frames and silently
  drops frames 5-7, which is why I5 does not reuse the 8-frame convention of the policy path.
* The encoder is **strictly causal**: latent frame `k` depends only on raw frames `0..4k`, and
  latent frame 0 depends on raw frame 0 *alone* -- perturbing frames 1..T-1 leaves it bit-identical.
  That is what lets the whole clip be encoded in one pass while latent frame 0 is still a
  future-free conditioning frame (AGENTS.md 5). `tests/test_i5_latent_tokenizer.py` pins it.
* The posterior is near-deterministic (`std` median about 8e-5 against a latent scale of 0.45), so
  `sample()` and `mode()` differ by well under a tenth of a percent. We nonetheless use `mode()`
  **always**: the offline cache has to be bit-identical across processes, and an unseeded `sample()`
  is not. The choice is about reproducibility, not signal quality.

Determinism is **per configuration**, not absolute. Repeated calls at a fixed batch size, device and
dtype are bit-identical, which is what the cache gate needs. Changing the batch size changes
reduction order and moves the result by float32 epsilon (measured: max absolute 1.9e-6, max relative
6.1e-7). So a cache is reproducible only together with the batch size that produced it, and the
builder records it -- the same shape of conclusion D-049 reached for the depth cache and its
environment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


#: Wan 2.2 VAE compression. Asserted against the loaded config rather than trusted.
LATENT_TEMPORAL_FACTOR = 4
LATENT_SPATIAL_FACTOR = 16
LATENT_CHANNELS = 48

#: I5 clip window (D-062). `1 + 4k` keeps the temporal grouping exact.
I5_WINDOW_FRAMES = 9


def shard_of(position: int, shard_count: int) -> int:
    """Which shard encodes the `position`-th episode of a build.

    Round-robin rather than contiguous blocks: episode lengths vary by a factor of a few, and
    `libero_10` episodes are about twice the corpus average, so contiguous blocks would leave one
    worker running long after the others finished.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    return position % shard_count


def latent_frame_count(num_frames: int) -> int:
    """Latent frames the VAE emits for `num_frames` raw frames."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    return (num_frames - 1) // LATENT_TEMPORAL_FACTOR + 1


def is_aligned_window(num_frames: int) -> bool:
    """True when the window is `1 + 4k`, i.e. no raw frame is dropped by the temporal grouping."""
    return num_frames >= 1 and (num_frames - 1) % LATENT_TEMPORAL_FACTOR == 0


def frames_to_vae_input(frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    """`[T, H, W, 3]` uint8 -> `[1, 3, T, H, W]` float in `[-1, 1]`.

    The VAE's own convention. Kept separate from the encoder so the scaling can be tested without
    loading 1.4 GB of weights.
    """
    tensor = torch.as_tensor(np.asarray(frames) if isinstance(frames, np.ndarray) else frames)
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"expected [T, H, W, 3], got {tuple(tensor.shape)}")
    if tensor.dtype != torch.uint8:
        raise ValueError(f"expected uint8 frames, got {tensor.dtype}")
    scaled = tensor.to(torch.float32).div_(127.5).sub_(1.0)
    return scaled.permute(3, 0, 1, 2).unsqueeze(0)


def normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    """Wan's per-channel latent normalisation: `(x - mean) / std`.

    `Wan2.py` spells the same transform as `(x - mean) * (1 / std)` inline. Sharing one helper keeps
    the cache and the DiT input in the same space -- a mismatch here would be silent.
    """
    if latents.shape[1] != latents_mean.numel():
        raise ValueError(
            f"latent channels {latents.shape[1]} do not match stats of size {latents_mean.numel()}"
        )
    view = (1, -1, 1, 1, 1)
    mean = latents_mean.to(latents.device, latents.dtype).view(view)
    std = latents_std.to(latents.device, latents.dtype).view(view)
    return (latents - mean) / std


class FrozenLatentTokenizer(nn.Module):
    """A frozen Wan VAE encoder, plus the invariants I5 depends on.

    The VAE is injected rather than loaded here, so tests can drive the whole contract with a stub
    and no checkpoint. Use :meth:`from_pretrained` for the real thing.
    """

    def __init__(self, vae: nn.Module, *, expect_channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.vae = vae
        config = vae.config
        spatial = int(getattr(config, "scale_factor_spatial", LATENT_SPATIAL_FACTOR))
        temporal = int(getattr(config, "scale_factor_temporal", LATENT_TEMPORAL_FACTOR))
        channels = int(getattr(config, "z_dim", expect_channels))
        # Assert rather than adapt: every downstream shape claim (the 16x16 grid matching the I3/I4
        # depth target grid, T_lat = (T-1)//4+1) is only true for this compression.
        if (spatial, temporal, channels) != (LATENT_SPATIAL_FACTOR, LATENT_TEMPORAL_FACTOR, expect_channels):
            raise ValueError(
                "I5 expects the Wan 2.2 VAE compression "
                f"(spatial {LATENT_SPATIAL_FACTOR}, temporal {LATENT_TEMPORAL_FACTOR}, "
                f"z_dim {expect_channels}); got ({spatial}, {temporal}, {channels})"
            )
        self.scale_factor_spatial = spatial
        self.scale_factor_temporal = temporal
        self.latent_channels = channels
        self.register_buffer(
            "latents_mean", torch.tensor(config.latents_mean, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(config.latents_std, dtype=torch.float32), persistent=False
        )
        self.enforce_frozen()

    @classmethod
    def from_pretrained(
        cls, model_path: str, *, subfolder: str = "vae", dtype: torch.dtype = torch.float32
    ) -> "FrozenLatentTokenizer":
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder=subfolder, torch_dtype=dtype)
        return cls(vae)

    def enforce_frozen(self) -> None:
        """Re-assert the firewall. Called again after every `train()` on the parent model."""
        self.vae.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True):
        """Keep the frozen VAE in eval mode when the parent switches to train (AGENTS.md 6)."""
        super().train(mode)
        self.enforce_frozen()
        return self

    def latent_grid(self, height: int, width: int) -> tuple[int, int]:
        """Spatial latent grid for a raw frame size."""
        if height % self.scale_factor_spatial or width % self.scale_factor_spatial:
            raise ValueError(
                f"frame size {height}x{width} is not divisible by {self.scale_factor_spatial}"
            )
        return height // self.scale_factor_spatial, width // self.scale_factor_spatial

    @torch.no_grad()
    def encode_windows(
        self, windows: np.ndarray | torch.Tensor, *, normalize: bool = True
    ) -> torch.Tensor:
        """`[B, T, H, W, 3]` uint8 -> normalised latents `[B, C, T_lat, h, w]`.

        Same contract as :meth:`encode`, batched. The offline cache builder needs this: encoding a
        273k-frame corpus one window at a time is hours of avoidable latency. Batching must not
        change results -- `tests/test_i5_latent_tokenizer.py` pins batch/single equality bitwise.
        """
        stacked = torch.as_tensor(np.asarray(windows) if isinstance(windows, np.ndarray) else windows)
        if stacked.ndim != 5:
            raise ValueError(f"expected [B, T, H, W, 3], got {tuple(stacked.shape)}")
        video = torch.cat([frames_to_vae_input(window) for window in stacked], dim=0)
        return self._encode_video(video, normalize=normalize)

    @torch.no_grad()
    def encode(self, frames: np.ndarray | torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """`[T, H, W, 3]` uint8 -> normalised latents `[1, C, T_lat, h, w]`.

        Always takes the posterior **mode**, never a sample: the offline cache must be bit-identical
        across processes (S1's determinism gate).
        """
        return self._encode_video(frames_to_vae_input(frames), normalize=normalize)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, normalized: bool = True) -> torch.Tensor:
        """Normalised latents `[B, C, T_lat, h, w]` -> `[B, T, H, W, 3]` uint8 frames.

        The inverse of :meth:`encode`, needed by the perceptual metrics: nothing in the tree called
        `vae.decode` before I5. Frames come back as uint8 in the same layout the encoder takes, so a
        round trip is directly comparable against the input clip.
        """
        if latents.ndim != 5:
            raise ValueError(f"expected [B, C, T_lat, h, w], got {tuple(latents.shape)}")
        if latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, got {latents.shape[1]}"
            )
        device, dtype = next(self.vae.parameters()).device, self.vae.dtype
        latents = latents.to(device, dtype)
        if normalized:
            view = (1, -1, 1, 1, 1)
            mean = self.latents_mean.to(device, dtype).view(view)
            std = self.latents_std.to(device, dtype).view(view)
            latents = latents * std + mean
        video = self.vae.decode(latents).sample  # [B, 3, T, H, W] in roughly [-1, 1]
        frames = video.permute(0, 2, 3, 4, 1).to(torch.float32)
        return frames.add_(1.0).mul_(127.5).clamp_(0, 255).round_().to(torch.uint8)

    def _encode_video(self, video: torch.Tensor, *, normalize: bool) -> torch.Tensor:
        video = video.to(next(self.vae.parameters()).device, self.vae.dtype)
        latents = self.vae.encode(video).latent_dist.mode()
        expected = latent_frame_count(video.shape[2])
        if latents.shape[2] != expected:
            raise AssertionError(
                f"VAE emitted {latents.shape[2]} latent frames for {video.shape[2]} raw frames, "
                f"expected {expected}"
            )
        if not normalize:
            return latents
        return normalize_latents(latents, self.latents_mean, self.latents_std)
