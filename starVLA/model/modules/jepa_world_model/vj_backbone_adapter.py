# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Adapter around the frozen V-JEPA video backbone.

Collects everything that depends on the HuggingFace V-JEPA API (`AutoModel` +
`AutoVideoProcessor`) and on the encoder's patch geometry, so the framework only talks to
`encode_video` / `split_teacher_forcing` and the geometry attributes. Swapping the teacher
(V-JEPA 2 -> 2.1) is then a change to this file, not to VLA_JEPA.

NOT an nn.Module on purpose: the encoder stays a direct submodule of the framework, so the
published checkpoints keep loading with `strict=True` under their existing `vj_encoder.*` keys.
Wrapping it in a module would either re-prefix or double-register those parameters.

Known upstream defect, preserved by default (`docs/provenance/upstream-conflicts.md`): upstream fuses
the two camera views with `torch.cat(torch.chunk(features, chunks=V, dim=0), dim=2)` after flattening
`[B, V, ...]` into `[B*V, ...]`. Views are the *minor* axis of that flatten, while `chunk` cuts the
batch axis into contiguous halves, so for `B > 1` a row receives one view from its own clip and
another from a different clip. It is exact at `B == 1`. `correct_view_fusion=True` selects the
per-clip pairing; the default is `False` because the I2 parity goldens were recorded at `BATCH_SIZE=2`
and therefore encode upstream's behaviour.

Gradient firewall (AGENTS.md section 6): the encoder is put in `eval()` and
`requires_grad_(False)` here, and `enforce_frozen()` is re-applied by `VLA_JEPA.train()`.
Numerically this is an identity operation on the pinned checkpoint -- its dropout and
drop-path rates are all 0.0 and the feature extraction already ran under `no_grad()` -- which
is why it can be introduced inside the I2 parity iteration.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from starVLA.model.modules.jepa_world_model.spatial_token_resampler import SpatialTokenResampler

# Config keys a teacher may state its square input size under, in order of preference. V-JEPA 2
# uses `image_size`; V-JEPA 2.1 has no `image_size` at all and states `crop_size` instead.
INPUT_SIZE_KEYS = ("image_size", "crop_size")


def resolve_input_size(config) -> int:
    """Square input edge the encoder was configured for, from whichever key it publishes.

    A missing key is not the same as a differing value: reading `config.image_size` on a V-JEPA 2.1
    config raises `AttributeError` deep inside the adapter, so the supported keys are tried in order
    and an unknown geometry is reported as such.
    """
    for key in INPUT_SIZE_KEYS:
        size = getattr(config, key, None)
        if isinstance(size, int):
            return size
    raise ValueError(f"encoder config states no input size; tried {INPUT_SIZE_KEYS}")


class VJBackboneAdapter:
    """Frozen video teacher: geometry contract + multi-view feature extraction.

    Attributes:
        image_size: square input edge the geometry is derived from, from `input_size` when given,
            else `config.image_size`, else `config.crop_size`.
        patch_size / tubelet_size / hidden_size: pinned encoder config values.
        native_grid_size / native_tokens_per_block: patch grid the encoder itself emits.
        grid_size: (height, width) patch grid of one temporal block as seen by consumers, i.e.
            after the optional resampler.
        tokens_per_block: tokens per temporal block on `grid_size`.
        num_temporal_blocks: temporal blocks per clip, `num_frames // tubelet_size`.
    """

    def __init__(
        self,
        encoder,
        processor,
        num_frames: int,
        resampler: Optional[SpatialTokenResampler] = None,
        input_size: Optional[int] = None,
        correct_view_fusion: bool = False,
    ) -> None:
        """Bind a frozen encoder to the geometry its features will be read with.

        Args:
            resampler: optional non-learned spatial pooling. `None` leaves features exactly as the
                encoder emits them.
            input_size: square input edge to derive the geometry from, overriding the config. Needed
                when a teacher runs at a resolution its config does not state: V-JEPA 2.1 pins
                `crop_size=384`, but its RoPE is interpolatable and the patch grid is taken from the
                input tensor, so the same weights also run natively at 256. `input_size` only
                *declares* which resolution the caller feeds; `encode_video`'s shape assertion is
                what verifies the processor actually delivers it.
            correct_view_fusion: pair each clip with its own views. `False` reproduces upstream's
                fusion, which is only correct at `batch_size == 1`, and is the default so that the
                I2 golden losses stay bitwise reproducible.
        """
        self.encoder = encoder
        self.processor = processor
        self.correct_view_fusion = correct_view_fusion

        config = encoder.config
        self.image_size: int = resolve_input_size(config) if input_size is None else input_size
        if not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError(f"input_size must be a positive int, got {input_size!r}")
        self.patch_size: int = config.patch_size
        self.tubelet_size: int = config.tubelet_size
        self.hidden_size: int = config.hidden_size

        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size {self.image_size} is not a multiple of patch_size {self.patch_size}")
        if num_frames % self.tubelet_size != 0:
            raise ValueError(f"num_frames {num_frames} is not a multiple of tubelet_size {self.tubelet_size}")

        grid = self.image_size // self.patch_size
        self.num_frames: int = num_frames
        self.num_temporal_blocks: int = num_frames // self.tubelet_size
        self.native_grid_size: Tuple[int, int] = (grid, grid)
        self.native_tokens_per_block: int = grid * grid

        # `resampler=None` is the pinned path: geometry and features are then exactly what the
        # encoder produces, so the V-JEPA 2 arm stays bitwise what I2 measured.
        self.resampler = resampler
        if resampler is None:
            self.grid_size: Tuple[int, int] = self.native_grid_size
        else:
            if tuple(resampler.grid_in) != self.native_grid_size:
                raise ValueError(
                    f"resampler expects a {tuple(resampler.grid_in)} grid, encoder emits {self.native_grid_size}"
                )
            self.grid_size = tuple(resampler.grid_out)
        self.tokens_per_block: int = self.grid_size[0] * self.grid_size[1]

        self.enforce_frozen()

    def enforce_frozen(self) -> None:
        """Re-assert the firewall. Called again after every `train()` on the parent model."""
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def encode_video(self, videos: np.ndarray) -> torch.Tensor:
        """Encode one multi-view clip batch into fused teacher features.

        Args:
            videos: [B, V, T, C, H, W] uint8, view order preserved.

        Returns:
            [B, num_temporal_blocks * tokens_per_block, V * hidden_size]. Views are fused by
            concatenation on the feature axis, so view `v` occupies channels
            `[v * hidden_size, (v + 1) * hidden_size)`.
        """
        batch_size, num_views, num_frames = videos.shape[0], videos.shape[1], videos.shape[2]
        if num_frames != self.num_frames:
            raise ValueError(f"clip has {num_frames} frames, encoder geometry expects {self.num_frames}")

        # The processor takes one clip at a time, so views are folded into the batch axis and
        # split apart again after encoding.
        flat = videos.reshape(batch_size * num_views, *videos.shape[2:])
        pixel_values = torch.cat(
            [
                self.processor(videos=flat[i], return_tensors="pt")["pixel_values_videos"].to(self.encoder.device)
                for i in range(batch_size * num_views)
            ],
            dim=0,
        )

        with torch.no_grad():
            features = self.encoder.get_vision_features(pixel_values_videos=pixel_values)
            features = self._fuse_views(features, batch_size, num_views)
            if self.resampler is not None:
                # After fusion, not before: the resampler leaves the channel axis alone, so the two
                # orders are equal (pinned by a test), and pooling once is the cheaper of them.
                features = self.resampler(features)

        expected = (
            batch_size,
            self.num_temporal_blocks * self.tokens_per_block,
            num_views * self.hidden_size,
        )
        if tuple(features.shape) != expected:
            raise AssertionError(f"teacher features {tuple(features.shape)}, geometry expects {expected}")
        return features

    def _fuse_views(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """Concatenate each clip's views on the feature axis: [B*V, tokens, D] -> [B, tokens, V*D].

        Two implementations, because the pinned upstream one is wrong for `B > 1` (see
        `correct_view_fusion`). Both are identical at `B == 1`, which is pinned by a test.
        """
        if self.correct_view_fusion:
            # `flat` was built as [B, V, ...] -> [B*V, ...], so V is the minor axis: split it back out
            # before moving it onto the channel axis. View v then occupies channels [v*D, (v+1)*D).
            unpacked = features.reshape(batch_size, num_views, features.shape[1], features.shape[2])
            return unpacked.permute(0, 2, 1, 3).reshape(batch_size, features.shape[1], num_views * features.shape[2])
        # Upstream: `chunk` cuts the batch axis into V contiguous halves, but V is the minor axis of
        # the flatten above, so for B > 1 row b receives one view from clip b and another from a
        # different clip entirely. Kept as the default so I2 parity stays bitwise.
        return torch.cat(torch.chunk(features, chunks=num_views, dim=0), dim=2)

    def split_teacher_forcing(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split fused features into the predictor input and its teacher-forced target.

        Blocks `[0, N-1)` are the input, blocks `[1, N)` the target, i.e. the predictor is asked
        for the next temporal block at every position.

        Args:
            features: [B, num_temporal_blocks * tokens_per_block, D] from `encode_video`.

        Returns:
            (input_states, gt_states), both [B, (num_temporal_blocks - 1) * tokens_per_block, D].
        """
        tokens = self.tokens_per_block
        blocks = self.num_temporal_blocks
        if features.shape[1] != tokens * blocks:
            raise AssertionError(
                f"{features.shape[1]} feature tokens, geometry expects {tokens} * {blocks} = {tokens * blocks}"
            )

        input_states = features[:, : tokens * (blocks - 1), :]
        gt_states = features[:, tokens:, :]
        assert input_states.shape == gt_states.shape
        return input_states, gt_states
