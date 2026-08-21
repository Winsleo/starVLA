# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Non-learned spatial pooling of teacher tokens onto a smaller patch grid.

V-JEPA 2 at 256 emits a 16x16 token grid per temporal block, V-JEPA 2.1 at 384 a 24x24 one. A
probe on 576 tokens has more than twice the input of a probe on 256, so a raw comparison of the
two teachers would partly measure token count rather than representation quality
(`docs/implementation-plan.md` section 4.1). Pooling 24x24 down to 16x16 makes the compared arms
share an output grid.

Deliberately parameter-free: a learned projection would add capacity to one arm only, which is the
confound this is meant to remove. Deliberately not an `nn.Module` either -- it holds no parameters
or buffers and needs no training mode, and staying out of the module tree keeps it out of every
`state_dict`, exactly like `VJBackboneAdapter`.

Averaging is `F.adaptive_avg_pool2d`. For a non-integer ratio such as 24 -> 16 its windows are
uneven and overlap (output cell `i` reads rows `floor(i * 24 / 16)` to `ceil((i + 1) * 24 / 16)`),
so adjacent output tokens share inputs. That is standard adaptive-pooling behaviour and is fine for
a probe target grid, but it does mean the pooled tokens are not disjoint patch averages.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialTokenResampler:
    """Average-pool `[B, blocks * grid_in, D]` teacher tokens to `[B, blocks * grid_out, D]`.

    The temporal block count is inferred from the token axis, so one instance works for any clip
    length. Channels are untouched, which is what makes it safe to apply to view-fused features:
    pooling the fused tensor equals pooling each view and concatenating again.

    Attributes:
        grid_in: (height, width) patch grid the teacher emits per temporal block.
        grid_out: (height, width) patch grid to pool onto. Equal grids make this the identity.
    """

    grid_in: Tuple[int, int]
    grid_out: Tuple[int, int]

    def __post_init__(self) -> None:
        for name, grid in (("grid_in", self.grid_in), ("grid_out", self.grid_out)):
            if len(grid) != 2 or not all(isinstance(side, int) and side > 0 for side in grid):
                raise ValueError(f"{name} must be two positive ints, got {grid!r}")
        if self.grid_out[0] > self.grid_in[0] or self.grid_out[1] > self.grid_in[1]:
            # Adaptive pooling would happily "upsample" by repeating inputs. That is not an
            # average of anything and would silently invent tokens, so refuse it.
            raise ValueError(f"grid_out {self.grid_out} is not smaller than or equal to grid_in {self.grid_in}")

    @property
    def tokens_in(self) -> int:
        return self.grid_in[0] * self.grid_in[1]

    @property
    def tokens_out(self) -> int:
        return self.grid_out[0] * self.grid_out[1]

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Pool the spatial axis of one feature batch.

        Args:
            features: [B, blocks * tokens_in, D], row-major within each temporal block.

        Returns:
            [B, blocks * tokens_out, D], same dtype and device, block order preserved. Returned
            unchanged (not a copy) when `grid_in == grid_out`.
        """
        if features.ndim != 3:
            raise ValueError(f"expected [B, tokens, D] features, got shape {tuple(features.shape)}")
        batch, total_tokens, dim = features.shape
        if total_tokens % self.tokens_in != 0:
            raise ValueError(f"{total_tokens} tokens is not a multiple of {self.tokens_in} tokens per block")
        if self.grid_in == self.grid_out:
            return features

        blocks = total_tokens // self.tokens_in
        # [B, blocks * h * w, D] -> [B * blocks, D, h, w]: pooling is spatial, so blocks and views
        # (which live in D) must not mix into it.
        grid = features.reshape(batch * blocks, *self.grid_in, dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid, self.grid_out)
        return pooled.permute(0, 2, 3, 1).reshape(batch, blocks * self.tokens_out, dim)
