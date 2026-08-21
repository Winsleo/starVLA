"""Masked losses for the I4 depth auxiliary branch."""

import torch


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def depth_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gradient_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted loss plus raw pixel and spatial-gradient terms.

    ``target`` and ``mask`` are expected to be stop-gradient tensors.  Invalid pixels are excluded
    from both terms; gradient edges are valid only when all four participating pixels are valid.
    """
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target and mask must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"expected [N,1,H,W], got {tuple(prediction.shape)}")
    raw = masked_mean((prediction - target).abs(), mask)
    if prediction.shape[-1] < 2 and prediction.shape[-2] < 2:
        gradient = prediction.new_zeros(())
    else:
        dx = (prediction[..., :, 1:] - prediction[..., :, :-1]).abs()
        dy = (prediction[..., 1:, :] - prediction[..., :-1, :]).abs()
        mx = mask[..., :, 1:] & mask[..., :, :-1]
        my = mask[..., 1:, :] & mask[..., :-1, :]
        gradient = 0.5 * (masked_mean(dx, mx) + masked_mean(dy, my))
    return raw + gradient_weight * gradient, raw, gradient
