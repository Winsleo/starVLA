"""Depth target construction for the I3/I4 geometry branch.

Implements the depth target contract of `docs/implementation-plan.md` section 6: log-clipped metric
depth, adjacent deltas, tubelet alignment and mask propagation, plus the clip-level median/MAD
normalisation used for relative pseudo-depth. Pure tensor math with no filesystem, model or trainer
dependency (`section 5`: the target builder owns normalize / delta / mask / alignment validation).

Two invariants drive the interface:

* Metric and relative targets must never be mixed silently (AGENTS.md section 7), so every target
  carries its `target_type` and `units`, and combining two targets checks both.
* Invalid, range-clipped and usable pixels must stay separately countable (I3 gate condition c), so
  sensor validity (`mask`) and range clipping (`range_clip_mask`) are kept apart instead of being
  folded into one mask.

Axis convention follows section 4.1. Raw cache depth is `[B, V, T, 1, H, W]`; tubelet selection
returns `[B, Tp, V, 1, H, W]`; lag-`k` deltas return `[B, Tp - k, V, 1, H, W]`, with `k = 1` the default.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

TARGET_TYPE_METRIC = "sim_metric"
TARGET_TYPE_PSEUDO = "pseudo_relative"
# A monocular estimator that predicts metres rather than an arbitrary affine scale (S4's
# `DA3METRIC-LARGE`, `DA3NESTED-GIANT-LARGE`, `Metric-Video-Depth-Anything-Large`). It shares the
# metric pipeline and units, so `evaluate` reports the metric class for it, but it is a *distinct*
# label from `TARGET_TYPE_METRIC`: the numbers come from a network, not from the simulator's depth
# buffer, and AGENTS.md section 7 forbids letting the two look alike.
TARGET_TYPE_PSEUDO_METRIC = "pseudo_metric"

# Target types whose values are log metres, i.e. for which AbsRel / RMSE / delta1 have a referent.
METRIC_TARGET_TYPES = frozenset({TARGET_TYPE_METRIC, TARGET_TYPE_PSEUDO_METRIC})
TARGET_TYPES = METRIC_TARGET_TYPES | {TARGET_TYPE_PSEUDO}

UNITS_LOG_METER = "log_meter"
UNITS_LOG_METER_DELTA = "log_meter_delta"
UNITS_MAD = "mad"

# LIBERO clipping range in metres. Measured on the S1a recording (2026-08-02): agentview depth spans
# 0.70-3.07 m and the wrist camera 0.04-0.38 m, while the simulator planes themselves are far wider
# (znear 0.0106 m, zfar 530.5 m). These bounds therefore keep the whole scene and only clip
# degenerate pixels; anything they do clip is counted through `range_clip_mask`.
DEFAULT_D_MIN = 0.02
DEFAULT_D_MAX = 5.0

_EPS = 1e-6


@dataclass(frozen=True)
class DepthTarget:
    """A depth target plus the mask saying which of its elements are usable.

    Attributes:
        values: target tensor; invalid elements are zero-filled, never NaN, so downstream losses
            cannot propagate NaN through masked-out positions.
        mask: bool tensor, same shape as `values`, True where the target is usable.
        target_type: `TARGET_TYPE_METRIC` or `TARGET_TYPE_PSEUDO`.
        units: physical meaning of `values`, e.g. log metres or MAD-normalised units.
    """

    values: torch.Tensor
    mask: torch.Tensor
    target_type: str
    units: str

    def __post_init__(self) -> None:
        if self.values.shape != self.mask.shape:
            raise ValueError(f"mask shape {tuple(self.mask.shape)} != {tuple(self.values.shape)}")
        if self.mask.dtype is not torch.bool:
            raise TypeError(f"mask must be bool, got {self.mask.dtype}")

    @property
    def valid_fraction(self) -> float:
        return float(self.mask.float().mean())

    def replace(self, values: torch.Tensor, mask: torch.Tensor, units: Optional[str] = None):
        """Derive a new target of the same type, so the type label survives every transformation."""
        return DepthTarget(
            values=values,
            mask=mask,
            target_type=self.target_type,
            units=self.units if units is None else units,
        )


def require_same_target_type(left: DepthTarget, right: DepthTarget) -> None:
    """Guard against mixing metric and relative targets (AGENTS.md section 7)."""
    if left.target_type != right.target_type:
        raise ValueError(f"refusing to combine target types {left.target_type!r} and {right.target_type!r}")
    if left.units != right.units:
        raise ValueError(f"refusing to combine units {left.units!r} and {right.units!r}")


def sensor_valid_mask(depth_m: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Finite, strictly positive depth, intersected with the recorded sensor mask if given."""
    mask = torch.isfinite(depth_m) & (depth_m > 0)
    if valid is not None:
        mask = mask & valid
    return mask


def range_clip_mask(depth_m: torch.Tensor, d_min: float = DEFAULT_D_MIN, d_max: float = DEFAULT_D_MAX) -> torch.Tensor:
    """Pixels whose value `log_metric_depth` clips, reported separately from invalid pixels."""
    with torch.no_grad():
        outside = (depth_m < d_min) | (depth_m > d_max)
    return outside & torch.isfinite(depth_m)


def log_metric_depth(
    depth_m: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    d_min: float = DEFAULT_D_MIN,
    d_max: float = DEFAULT_D_MAX,
    target_type: str = TARGET_TYPE_METRIC,
) -> DepthTarget:
    """`D~ = log(clip(D, d_min, d_max))` on metric depth (implementation-plan section 6).

    Args:
        depth_m: metric depth in metres, any shape.
        valid: optional recorded sensor mask, broadcast-compatible with `depth_m`.
        d_min: lower clip bound in metres, must be > 0 so the log is defined.
        d_max: upper clip bound in metres.
        target_type: which metric source the metres came from; the maths is identical for the
            simulator buffer and for a metric estimator, only the provenance label differs.

    Returns:
        Log-depth target in `UNITS_LOG_METER`; invalid positions are zero.
    """
    if not d_min > 0:
        raise ValueError(f"d_min must be positive for a log target, got {d_min}")
    if not d_max > d_min:
        raise ValueError(f"d_max ({d_max}) must exceed d_min ({d_min})")
    if target_type not in METRIC_TARGET_TYPES:
        raise ValueError(f"target_type {target_type!r} is not one of the metric types {sorted(METRIC_TARGET_TYPES)}")

    mask = sensor_valid_mask(depth_m, valid)
    clipped = torch.clamp(torch.nan_to_num(depth_m, nan=d_min, posinf=d_max, neginf=d_min), d_min, d_max)
    values = torch.where(mask, torch.log(clipped), torch.zeros_like(clipped))
    return DepthTarget(values=values, mask=mask, target_type=target_type, units=UNITS_LOG_METER)


def normalize_clip_level(
    depth: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    clip_dims: Tuple[int, ...] = (-4, -3, -2, -1),
    eps: float = _EPS,
) -> DepthTarget:
    """Relative pseudo-depth target `(D_t - median(D_0:T)) / (MAD(D_0:T) + eps)`.

    Clip-level, never frame-wise (implementation-plan section 6): the statistics are reduced over
    `clip_dims`, which default to the trailing `(T, C, H, W)` axes of a `[B, V, T, 1, H, W]` cache
    tensor, i.e. one median/MAD per sample and view.
    """
    mask = sensor_valid_mask(depth, valid)
    masked = torch.where(mask, depth.to(torch.float32), torch.full_like(depth, float("nan"), dtype=torch.float32))

    dims = tuple(sorted(dim % masked.ndim for dim in clip_dims))
    median = _nanmedian_over(masked, dims)
    mad = _nanmedian_over((masked - median).abs(), dims)

    values = (masked - median) / (mad + eps)
    values = torch.where(mask, values, torch.zeros_like(values))
    return DepthTarget(values=values, mask=mask, target_type=TARGET_TYPE_PSEUDO, units=UNITS_MAD)


def _nanmedian_over(values: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    """Median over several dims ignoring NaN, keeping the reduced dims for broadcasting.

    `torch.nanmedian` only reduces one dim at a time, so the reduced axes are flattened first. All-NaN
    slices return 0, which is harmless because those positions are masked out anyway.
    """
    kept = [dim for dim in range(values.ndim) if dim not in dims]
    permuted = values.permute(*kept, *dims).reshape(*[values.shape[dim] for dim in kept], -1)
    median = torch.nanmedian(permuted, dim=-1).values
    median = torch.nan_to_num(median, nan=0.0)
    for dim in dims:
        median = median.unsqueeze(dim)
    return median


def tubelet_last_frame(cache_depth: torch.Tensor, tubelet_size: int) -> torch.Tensor:
    """Select each tubelet's last frame and move time in front of the view axis.

    `[B, V, T, 1, H, W] -> [B, T // tubelet_size, V, 1, H, W]`, matching the depth path of
    implementation-plan section 4.1. The last frame is the default aggregation there; any other
    aggregation is a separate ablation.
    """
    if cache_depth.ndim != 6:
        raise ValueError(f"expected [B,V,T,1,H,W], got {tuple(cache_depth.shape)}")
    num_frames = cache_depth.shape[2]
    if tubelet_size < 1 or num_frames % tubelet_size:
        raise ValueError(f"num_frames {num_frames} is not divisible by tubelet_size {tubelet_size}")

    selected = cache_depth[:, :, tubelet_size - 1 :: tubelet_size]
    return selected.permute(0, 2, 1, 3, 4, 5).contiguous()


def lagged_delta(target: DepthTarget, lag: int = 1, time_dim: int = 1) -> DepthTarget:
    """`dD~_k = D~_{k+lag} - D~_k` over tubelet states, with masks intersected pairwise.

    `[B, Tp, ...] -> [B, Tp - lag, ...]`. `lag = 1` is the adjacent-transition contract of section
    4.1; a larger lag widens the interval the target spans while leaving the teacher's input
    untouched, which is what makes the delta interval a separable axis rather than a property of the
    recording (I3 interval sweep, `docs/experiments/i3-geo-probes.md`).
    """
    if lag < 1:
        raise ValueError(f"lag must be at least 1, got {lag}")
    num_states = target.values.shape[time_dim]
    if num_states < lag + 1:
        raise ValueError(f"need at least {lag + 1} states on dim {time_dim} to form a lag-{lag} delta")

    span = num_states - lag
    later = target.values.narrow(time_dim, lag, span)
    earlier = target.values.narrow(time_dim, 0, span)
    mask = target.mask.narrow(time_dim, lag, span) & target.mask.narrow(time_dim, 0, span)

    values = torch.where(mask, later - earlier, torch.zeros_like(later))
    units = UNITS_LOG_METER_DELTA if target.units == UNITS_LOG_METER else target.units
    return target.replace(values=values, mask=mask, units=units)


def adjacent_delta(target: DepthTarget, time_dim: int = 1) -> DepthTarget:
    """`dD~_k = D~_{k+1} - D~_k`, i.e. the K transitions of section 4.1: `lagged_delta` at lag 1."""
    return lagged_delta(target, lag=1, time_dim=time_dim)


def pool_to_grid(target: DepthTarget, grid: Tuple[int, int]) -> DepthTarget:
    """Valid-weighted average pooling of a dense depth target onto a token grid.

    Probes compare teachers on a fixed token grid, so the dense target has to be reduced to it. Only
    valid pixels contribute; a cell with no valid pixel is masked out and zero-filled. Pooling is done
    per frame before any differencing, so a cell's weights come from that frame's own mask.

    Args:
        target: values shaped `[..., 1, H, W]`.
        grid: `(grid_h, grid_w)`, both must divide `H` and `W`.
    """
    values, mask = target.values, target.mask
    if values.shape[-3] != 1:
        raise ValueError(f"expected a single channel axis at -3, got {tuple(values.shape)}")
    height, width = values.shape[-2], values.shape[-1]
    grid_h, grid_w = grid
    if height % grid_h or width % grid_w:
        raise ValueError(f"grid {grid} does not divide the {height}x{width} depth map")

    lead = values.shape[:-3]
    flat_values = values.reshape(-1, 1, height, width).to(torch.float32)
    flat_mask = mask.reshape(-1, 1, height, width).to(torch.float32)

    weight_sum = F.avg_pool2d(flat_mask, kernel_size=(height // grid_h, width // grid_w))
    value_sum = F.avg_pool2d(flat_values * flat_mask, kernel_size=(height // grid_h, width // grid_w))

    pooled_mask = weight_sum > 0
    pooled = torch.where(pooled_mask, value_sum / weight_sum.clamp_min(_EPS), torch.zeros_like(value_sum))

    return target.replace(
        values=pooled.reshape(*lead, 1, grid_h, grid_w),
        mask=pooled_mask.reshape(*lead, 1, grid_h, grid_w),
    )


def build_metric_delta_targets(
    cache_depth: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    tubelet_size: int = 2,
    grid: Optional[Tuple[int, int]] = None,
    d_min: float = DEFAULT_D_MIN,
    d_max: float = DEFAULT_D_MAX,
    target_type: str = TARGET_TYPE_METRIC,
    delta_lag: int = 1,
) -> Tuple[DepthTarget, DepthTarget]:
    """The default metric pipeline: log-clip, tubelet-align, optionally pool, then difference.

    Args:
        cache_depth: `[B, V, T, 1, H, W]` metric depth in metres.
        valid: optional recorded sensor mask of the same shape.
        tubelet_size: teacher tubelet length; each tubelet keeps its last frame.
        grid: token grid to pool onto, or None to keep the dense resolution.
        target_type: `TARGET_TYPE_METRIC` for the simulator buffer, `TARGET_TYPE_PSEUDO_METRIC` for a
            metric estimator. Estimator targets must go through this same call rather than a parallel
            implementation, so the two only ever differ by their label.
        delta_lag: how many tubelet states the delta spans; 1 is the section 4.1 default. The states
            are unaffected, so a lag sweep changes only the transition targets.

    Returns:
        `(states, deltas)`: `[B, Tp, V, 1, h, w]` log-depth states and `[B, Tp - delta_lag, V, 1, h, w]`
        deltas.
    """
    states = log_metric_depth(cache_depth, valid=valid, d_min=d_min, d_max=d_max, target_type=target_type)
    aligned = DepthTarget(
        values=tubelet_last_frame(states.values, tubelet_size),
        mask=tubelet_last_frame(states.mask, tubelet_size),
        target_type=states.target_type,
        units=states.units,
    )
    if grid is not None:
        aligned = pool_to_grid(aligned, grid)
    return aligned, lagged_delta(aligned, lag=delta_lag)
