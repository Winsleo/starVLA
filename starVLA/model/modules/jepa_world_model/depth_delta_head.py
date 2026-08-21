"""Small action-conditioned head for predictor-aligned log-depth deltas."""

import torch
from torch import nn


class DepthDeltaHead(nn.Module):
    """Predict one log-depth delta map from the current map and action-token condition.

    The head is deliberately independent of the VLM and teacher.  ``condition`` is
    ``[N, Q, hidden]`` and the current target-only state is ``[N, 1, H, W]``.
    """

    def __init__(self, hidden_size: int, channels: int = 64) -> None:
        super().__init__()
        if hidden_size < 1 or channels < 4:
            raise ValueError("hidden_size must be positive and channels must be at least 4")
        self.condition = nn.Linear(hidden_size, channels * 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, current: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if current.ndim != 4 or current.shape[1] != 1:
            raise ValueError(f"current must be [N,1,H,W], got {tuple(current.shape)}")
        if condition.ndim != 3 or condition.shape[0] != current.shape[0]:
            raise ValueError(f"condition must be [N,Q,H], got {tuple(condition.shape)}")
        features = self.encoder(current.float())
        scale, shift = self.condition(condition.float().mean(dim=1)).chunk(2, dim=-1)
        features = features * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.decoder(features).to(dtype=current.dtype)
