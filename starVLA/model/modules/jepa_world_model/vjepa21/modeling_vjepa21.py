"""V-JEPA 2.1 model implementation for HuggingFace Transformers.

Faithful port of the reference implementation in
``facebookresearch/vjepa2`` under ``app/vjepa_2_1/models/``.

Key differences from V-JEPA 2:
- Multi-modality: separate patch embedding for images (tubelet_size=1) + modality embeddings
- Hierarchical output: intermediate layer features with per-layer norms
- Interpolatable RoPE: variable input resolution support
- Dense predictor: hierarchical input fusion + context token prediction

Compatible with transformers >= 4.50 (both the 4.x and 5.x attention APIs).

Note on initialisation: the reference `VisionTransformer` and
`VisionTransformerPredictor` call `_rescale_blocks()` after `_init_weights`,
dividing `attn.proj.weight` and `mlp.fc2.weight` of layer *i* by `sqrt(2*(i+1))`.
That is deliberately **not** reproduced here: it only affects randomly
initialised models, and applying it inside `_init_weights` would risk touching
weights that `from_pretrained` has already loaded. Every published checkpoint
loads its weights, so the two agree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_vjepa21 import VJEPA21Config

logger = logging.get_logger(__name__)
_warn_once = getattr(logger, "warning_once", logger.warning)


# Keyword arguments the model forward understands. Anything else is reported
# once and dropped, instead of being swallowed by `**kwargs`: a typo such as
# `out_layer=[11]` used to be a silent no-op.
_ENCODER_FORWARD_KWARGS = frozenset(
    {"masks", "out_layers", "return_hierarchical", "output_attentions", "output_hidden_states"}
)
_MODEL_FORWARD_KWARGS = _ENCODER_FORWARD_KWARGS | frozenset(
    {"context_mask", "target_mask", "skip_predictor", "mask_index"}
)
# Injected by the Trainer or by generic HF plumbing; harmless and not worth a warning.
_SILENTLY_IGNORED_KWARGS = frozenset(
    {"num_items_in_batch", "return_dict", "return_loss", "interpolate_pos_encoding"}
)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VJEPA21EncoderOutput(ModelOutput):
    """Encoder output.

    Attributes:
        last_hidden_state: Final layer output, normalised with the last
            hierarchical norm, `(B, N, hidden_size)`.
        hierarchical_hidden_state: Distillation levels concatenated along the
            channel axis, `(B, N, n_output_distillation * hidden_size)`.
            Only returned when `return_hierarchical=True`.
        multilevel_hidden_states: Tuple of per-level normalised features, one
            entry per requested `out_layers` index, in the order the layers occur
            in the network (not in the order they were requested). Mirrors the
            `out_layers` argument of the reference implementation.
        hidden_states: Raw (un-normalised) outputs of the embedding layer and of
            every transformer layer, when `output_hidden_states=True`.
        attentions: Attention probabilities of every layer, when
            `output_attentions=True`.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hierarchical_hidden_state: Optional[torch.FloatTensor] = None
    multilevel_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class VJEPA21PredictorOutput(ModelOutput):
    """Predictor output.

    Attributes:
        last_hidden_state: Predicted target tokens `(B, N_target, proj_dim)`.
        context_hidden_state: Predicted context tokens, when the model was
            configured with `pred_return_all_tokens=True`.
        hidden_states: Raw outputs of every predictor layer.
        attentions: Attention probabilities of every predictor layer.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    context_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class VJEPA21ModelOutput(ModelOutput):
    """Full model output combining encoder and predictor.

    `masked_hidden_state` is the tensor the predictor actually consumed, gathered
    at `context_mask`. Its channel width therefore depends on the checkpoint:
    `hidden_size` when `n_output_distillation == 1` (ViT-B, ViT-L) and
    `n_output_distillation * hidden_size` when the predictor fuses several levels
    (ViT-g, ViT-G).
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hierarchical_hidden_state: Optional[torch.FloatTensor] = None
    multilevel_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    masked_hidden_state: Optional[torch.FloatTensor] = None
    predictor_output: Optional[VJEPA21PredictorOutput] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather tokens at mask indices.

    Args:
        tensor: `(B, N, D)` tensor.
        masks: List of `(B, K)` index tensors.

    Returns:
        `(len(masks)*B, K, D)` gathered tensor.
    """
    parts = []
    for mask in masks:
        mask = mask.to(tensor.device)
        idx = mask.unsqueeze(-1).expand(-1, -1, tensor.size(-1))
        parts.append(torch.gather(tensor, dim=1, index=idx))
    return torch.cat(parts, dim=0)


def _as_mask_list(masks) -> Optional[list[torch.Tensor]]:
    """Accept a single tensor or a list of tensors, as the reference does."""
    if masks is None:
        return None
    if isinstance(masks, torch.Tensor):
        return [masks]
    return list(masks)


def normalize_video_layout(pixel_values_videos: torch.Tensor, in_chans: int = 3) -> torch.Tensor:
    """Bring any accepted video layout to channels-first `(B, C, T, H, W)`.

    Accepted layouts, resolved in this order:
        - `(B, C, T, H, W)` channels-first (native)
        - `(B, T, C, H, W)` the layout produced by HF video processors
        - `(B, T, H, W, C)` channels-last
        - `(B, C, H, W)`    a batch of single images, promoted to `T = 1`

    The channel axis is identified by matching `config.in_chans`. When more than
    one axis matches — a 3-frame clip is the realistic case — the channels-first
    reading wins and a warning is emitted, because the alternative is a silent
    transposition.
    """
    x = pixel_values_videos
    if x.ndim == 4:
        # (B, C, H, W) -> (B, C, 1, H, W)
        if x.shape[1] != in_chans:
            raise ValueError(
                f"Expected a 4D tensor shaped (B, {in_chans}, H, W), got {tuple(x.shape)}."
            )
        return x.unsqueeze(2)

    if x.ndim != 5:
        raise ValueError(
            "pixel_values_videos must be a 4D or 5D tensor, got "
            f"{x.ndim} dimensions with shape {tuple(x.shape)}."
        )

    candidates = [axis for axis in (1, 2, 4) if x.shape[axis] == in_chans]
    if len(candidates) > 1:
        _warn_once(
            "ambiguous video layout: "
            f"pixel_values_videos of shape {tuple(x.shape)} has {len(candidates)} candidate "
            f"channel axes {candidates} of size in_chans={in_chans}; "
            "reading it as (B, C, T, H, W). "
            "Pass an unambiguous layout if that is not what you meant."
        )

    if x.shape[1] == in_chans:  # (B, C, T, H, W)
        return x
    if x.shape[2] == in_chans:  # (B, T, C, H, W)
        return x.permute(0, 2, 1, 3, 4)
    if x.shape[-1] == in_chans:  # (B, T, H, W, C)
        return x.permute(0, 4, 1, 2, 3)

    raise ValueError(
        f"Could not locate a channel axis of size {in_chans} in a tensor of shape "
        f"{tuple(x.shape)}. Supported layouts are (B, C, T, H, W), (B, T, C, H, W) "
        "and (B, T, H, W, C)."
    )


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep) * mask


class VJEPA21DropPath(nn.Module):
    def __init__(self, p: Optional[float] = None):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


def rotate_queries_or_keys(
    x: torch.Tensor,
    pos: torch.Tensor,
    n_registers: int = 0,
    has_cls_first: bool = False,
) -> torch.Tensor:
    """Apply rotary position embeddings with register/CLS token handling.

    Args:
        x: `(B, H, N, D)` query or key tensor.
        pos: Position ids broadcastable to `(..., N_ctx)`.
        n_registers: Number of register tokens at end of sequence (not rotated).
        has_cls_first: Whether first token is CLS (not rotated).
    """
    B, num_heads, N, D = x.size()
    if D % 2 != 0:
        raise ValueError(f"RoPE requires an even dimension per slice, got {D}.")

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    # Position ids are computed over the context tokens only. If they were built
    # over the full sequence, trim them so CLS/registers stay unrotated.
    if pos.shape[-1] == N and (n_cls or n_registers):
        pos = pos[..., start_ctx:end_ctx]

    # RoPE frequencies are computed in float32 and cast back to the input dtype.
    # The reference implementation builds them in `x.dtype`, which is equivalent in
    # float32 but promotes q/k to float32 when the weights are held in bf16/fp16 —
    # the fused attention kernels then reject the mismatch against v. Meta's training
    # code never hits this because it runs under `torch.autocast` rather than casting
    # the weights. Computing in float32 is also the numerically safer choice.
    omega = torch.arange(D // 2, dtype=torch.float32, device=x.device)
    omega = omega / (D / 2.0)
    omega = 1.0 / (10000.0**omega)

    freq = torch.einsum("..., f -> ... f", pos.to(torch.float32), omega)
    emb_sin = freq.sin().repeat_interleave(2, dim=-1).to(x.dtype)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1).to(x.dtype)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    out_ctx = x_ctx * emb_cos + y * emb_sin

    parts = []
    if x_cls is not None:
        parts.append(x_cls)
    parts.append(out_ctx)
    if x_reg is not None:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def resolve_attention_interface(config, output_attentions: bool = False) -> Callable:
    """Return the attention kernel for ``config._attn_implementation``.

    Works on transformers 4.x, where ``ALL_ATTENTION_FUNCTIONS`` is a plain
    mapping, and on 5.x, where it exposes ``get_interface``. When attention
    probabilities are requested we fall back to the eager kernel, since the
    fused kernels do not materialise them.
    """
    if output_attentions:
        return eager_attention_forward

    impl = getattr(config, "_attn_implementation", None) or "eager"
    if impl == "eager":
        return eager_attention_forward

    getter = getattr(ALL_ATTENTION_FUNCTIONS, "get_interface", None)
    if getter is not None:  # transformers >= 5
        return getter(impl, eager_attention_forward)
    return ALL_ATTENTION_FUNCTIONS.get(impl, eager_attention_forward)  # transformers 4.x


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------


class VJEPA21PatchEmbeddings3D(nn.Module):
    """3D patch embedding via Conv3d."""

    def __init__(self, config: VJEPA21Config, tubelet_size: Optional[int] = None):
        super().__init__()
        ts = tubelet_size if tubelet_size is not None else config.tubelet_size
        ps = config.patch_size
        self.proj = nn.Conv3d(
            config.in_chans,
            config.hidden_size,
            kernel_size=(ts, ps, ps),
            stride=(ts, ps, ps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        return self.proj(x).flatten(2).transpose(1, 2)


class VJEPA21Embeddings(nn.Module):
    """Patch embeddings with modality-aware processing."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config

        # Video patch embedding (tubelet_size from config)
        self.patch_embeddings = VJEPA21PatchEmbeddings3D(config)

        # Image patch embedding (tubelet_size=1) if img_temporal_dim_size is set
        self.patch_embeddings_img = None
        if config.img_temporal_dim_size is not None:
            self.patch_embeddings_img = VJEPA21PatchEmbeddings3D(config, tubelet_size=1)

        # Modality embeddings
        self.img_mod_embed = None
        self.video_mod_embed = None
        if config.modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, pixel_values_videos: torch.Tensor) -> tuple[torch.Tensor, str]:
        """
        Args:
            pixel_values_videos: `(B, C, T, H, W)`, already layout-normalised.

        Returns:
            embeddings: `(B, N, hidden_size)`.
            mode: "img" or "video".
        """
        target_dtype = self.patch_embeddings.proj.weight.dtype
        pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)

        T = pixel_values_videos.shape[2]
        is_image = (
            self.config.img_temporal_dim_size is not None
            and T == self.config.img_temporal_dim_size
        )

        if is_image and self.patch_embeddings_img is not None:
            embeddings = self.patch_embeddings_img(pixel_values_videos)
            mode = "img"
        else:
            # Ensure at least tubelet_size frames
            if T < self.config.tubelet_size:
                pixel_values_videos = pixel_values_videos.repeat(
                    1, 1, self.config.tubelet_size, 1, 1
                )
            embeddings = self.patch_embeddings(pixel_values_videos)
            mode = "video"

        if self.img_mod_embed is not None:
            if mode == "img":
                embeddings = embeddings + self.img_mod_embed
            else:
                embeddings = embeddings + self.video_mod_embed

        return embeddings, mode


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class VJEPA21RopeAttention(nn.Module):
    """RoPE-based multi-head attention with interpolation and register support."""

    def __init__(self, config: VJEPA21Config, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        # RoPE dimension split: depth, height, width
        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

        self.grid_size = config.crop_size // config.patch_size
        self.n_registers = config.n_registers
        self.has_cls_first = config.has_cls_first
        self.interpolate_rope = config.interpolate_rope
        self.pretrained_grid_size = config.pretrained_grid_size

    def _separate_positions(
        self,
        ids: torch.Tensor,
        H_patches: Optional[int] = None,
        W_patches: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose flat token ids into (depth, height, width) components."""
        hp = H_patches if H_patches is not None else self.grid_size
        wp = W_patches if W_patches is not None else self.grid_size
        tokens_per_frame = hp * wp
        frame_ids = ids // tokens_per_frame
        remainder = ids - tokens_per_frame * frame_ids
        height_ids = remainder // wp
        width_ids = remainder - wp * height_ids
        return frame_ids.float(), height_ids.float(), width_ids.float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
        H_patches: Optional[int] = None,
        W_patches: Optional[int] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = hidden_states.shape

        q = (
            self.query(hidden_states)
            .view(B, N, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        k = (
            self.key(hidden_states)
            .view(B, N, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        v = (
            self.value(hidden_states)
            .view(B, N, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        # Compute position ids
        if position_mask is not None:
            ids = position_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        else:
            ids = torch.arange(N, device=hidden_states.device)

        d_mask, h_mask, w_mask = self._separate_positions(ids, H_patches, W_patches)

        # Interpolate RoPE for variable resolution.
        # Mirrors app/vjepa_2_1/models/utils/modules.py (`interpolate_rope`).
        if self.interpolate_rope:
            hp = H_patches if H_patches is not None else self.grid_size
            wp = W_patches if W_patches is not None else self.grid_size
            h_mask = h_mask * (self.pretrained_grid_size - 1) / max(hp - 1, 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / max(wp - 1, 1)

        # Apply RoPE to each dimension slice
        s = 0
        qd = rotate_queries_or_keys(
            q[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first
        )
        kd = rotate_queries_or_keys(
            k[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first
        )
        s += self.d_dim
        qh = rotate_queries_or_keys(
            q[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first
        )
        kh = rotate_queries_or_keys(
            k[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first
        )
        s += self.h_dim
        qw = rotate_queries_or_keys(
            q[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first
        )
        kw = rotate_queries_or_keys(
            k[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first
        )
        s += self.w_dim

        if s < self.attention_head_size:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        attention_interface = resolve_attention_interface(self.config, output_attentions)

        context_layer, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            None,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        output = self.proj(context_layer.reshape(B, N, self.all_head_size))
        return output, attn_weights


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class VJEPA21MLP(nn.Module):
    """Standard GELU MLP."""

    def __init__(self, config: VJEPA21Config, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VJEPA21SwiGLUMLP(nn.Module):
    """SwiGLU FFN as used in V-JEPA 2.1."""

    def __init__(self, config: VJEPA21Config, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        if config.wide_silu:
            swiglu_hidden = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden = (swiglu_hidden + align_as - 1) // align_as * align_as
        else:
            swiglu_hidden = hidden_features
        self.fc1 = nn.Linear(hidden_size, swiglu_hidden)
        self.fc2 = nn.Linear(hidden_size, swiglu_hidden)
        self.fc3 = nn.Linear(swiglu_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


def _make_mlp(config: VJEPA21Config, hidden_size: int, mlp_ratio: float) -> nn.Module:
    if config.hidden_act == "silu":
        return VJEPA21SwiGLUMLP(config, hidden_size, mlp_ratio)
    return VJEPA21MLP(config, hidden_size, mlp_ratio)


# ---------------------------------------------------------------------------
# Transformer Layer
# ---------------------------------------------------------------------------


class VJEPA21Layer(nn.Module):
    """Single transformer block: LN -> Attention -> DropPath + Residual -> LN -> MLP -> DropPath + Residual."""

    def __init__(
        self,
        config: VJEPA21Config,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA21RopeAttention(config, hidden_size, num_attention_heads)
        self.drop_path = (
            VJEPA21DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = _make_mlp(config, hidden_size, mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
        H_patches: Optional[int] = None,
        W_patches: Optional[int] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        h = self.norm1(hidden_states)
        attn_out, attn_weights = self.attention(
            h, position_mask, T, H_patches, W_patches, output_attentions
        )
        hidden_states = residual + self.drop_path(attn_out)

        residual = hidden_states
        hidden_states = residual + self.drop_path(self.mlp(self.norm2(hidden_states)))
        return hidden_states, attn_weights


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class VJEPA21Encoder(nn.Module):
    """V-JEPA 2.1 encoder with hierarchical and multi-level output support."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA21Embeddings(config)

        dpr = [
            config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1)
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA21Layer(
                    config,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Per-layer norms for hierarchical outputs. One norm per hierarchical
        # level, exactly as in the reference VisionTransformer.
        hier_layers = config.encoder_hierarchical_layers
        self.norms_block = nn.ModuleList(
            [nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in hier_layers]
        )
        self._hier_layers = hier_layers
        self._distill_layers = config.encoder_distillation_layers

        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        masks: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        return_hierarchical: bool = False,
        out_layers: Optional[list[int]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> VJEPA21EncoderOutput:
        """
        Args:
            masks: Optional list of `(B, K)` index tensors. When given, the patch
                tokens are gathered at those indices *before* the transformer
                layers, so attention only ever sees the context tokens and RoPE
                receives their true positions. This is the JEPA training-time
                forward (`z = encoder(clips, masks_enc)` in the reference); the
                default `None` runs the full sequence, which is what feature
                extraction wants.
        """
        unexpected = set(kwargs) - _SILENTLY_IGNORED_KWARGS
        if unexpected:
            _warn_once(
                f"VJEPA21Encoder.forward received unexpected keyword arguments "
                f"{sorted(unexpected)}; they are ignored. Accepted arguments: "
                f"{sorted(_ENCODER_FORWARD_KWARGS)}."
            )

        pixel_values_videos = normalize_video_layout(
            pixel_values_videos, self.config.in_chans
        )

        embeddings, _ = self.embeddings(pixel_values_videos)

        B, C, T_raw, H, W = pixel_values_videos.shape
        is_image = (
            self.config.img_temporal_dim_size is not None
            and T_raw == self.config.img_temporal_dim_size
        )
        T_patches = T_raw if is_image else max(T_raw // self.config.tubelet_size, 1)
        H_patches = H // self.config.patch_size
        W_patches = W // self.config.patch_size

        if out_layers is not None:
            unknown = [i for i in out_layers if i not in self._hier_layers]
            if unknown:
                raise ValueError(
                    f"out_layers={out_layers} contains indices {unknown} that are not "
                    f"hierarchical layers of this model. Valid indices: {self._hier_layers}."
                )

        # Masked (JEPA) forward: drop tokens before the layers and carry their
        # original indices as RoPE positions, as the reference does.
        masks = _as_mask_list(masks)
        position_mask = None
        if masks is not None:
            n_tokens = embeddings.shape[1]
            for m in masks:
                if m.dim() != 2 or m.shape[0] != embeddings.shape[0]:
                    raise ValueError(
                        f"each mask must be a (B, K) index tensor with B={embeddings.shape[0]}, "
                        f"got {tuple(m.shape)}."
                    )
                if int(m.max()) >= n_tokens:
                    raise ValueError(
                        f"mask index {int(m.max())} is out of range for a sequence of "
                        f"{n_tokens} tokens."
                    )
            embeddings = apply_masks(embeddings, masks)
            position_mask = torch.cat([m.to(embeddings.device) for m in masks], dim=0)

        hidden_states = embeddings
        hier_outputs: list[torch.Tensor] = []
        multilevel_outputs: list[torch.Tensor] = []
        all_hidden_states: Optional[tuple[torch.Tensor, ...]] = () if output_hidden_states else None
        all_attentions: Optional[tuple[torch.Tensor, ...]] = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_out = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    position_mask,
                    T_patches,
                    H_patches,
                    W_patches,
                    output_attentions,
                )
            else:
                layer_out = layer_module(
                    hidden_states,
                    position_mask=position_mask,
                    T=T_patches,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_out[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_out[1],)

            if out_layers is not None and i in out_layers:
                idx = self._hier_layers.index(i)
                multilevel_outputs.append(self.norms_block[idx](hidden_states))

            if i in self._distill_layers:
                idx = self._hier_layers.index(i)
                hier_outputs.append(self.norms_block[idx](hidden_states))

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Reference implementation: `x = self.norms_block[-1](x)` on the final layer.
        last_hidden_state = self.norms_block[-1](hidden_states)

        hierarchical_hidden_state = None
        if return_hierarchical and hier_outputs:
            hierarchical_hidden_state = torch.cat(hier_outputs, dim=2)

        return VJEPA21EncoderOutput(
            last_hidden_state=last_hidden_state,
            hierarchical_hidden_state=hierarchical_hidden_state,
            multilevel_hidden_states=tuple(multilevel_outputs) if out_layers else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class VJEPA21PredictorEmbeddings(nn.Module):
    """Predictor embeddings with hierarchical input fusion."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        n_hier = len(config.predictor_hierarchical_layers)
        if n_hier <= 1:
            self.predictor_embed = nn.Linear(config.hidden_size, config.pred_hidden_size)
        else:
            act = nn.SiLU if config.hidden_act == "silu" else nn.GELU
            self.predictor_embed = nn.Sequential(
                nn.Linear(config.hidden_size * n_hier, config.hidden_size),
                act(),
                nn.Linear(config.hidden_size, config.pred_hidden_size),
            )

        self.num_mask_tokens = config.pred_num_mask_tokens
        self.mask_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, config.pred_hidden_size))
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.img_mod_embed = None
        self.video_mod_embed = None
        if config.img_temporal_dim_size is not None and config.modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, config.pred_hidden_size))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, config.pred_hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mask_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `hidden_states` already carries len(context_mask) * B rows.
        batch_size = hidden_states.size(0) // len(context_mask)

        context = self.predictor_embed(hidden_states)

        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index].repeat(
            batch_size, self._max_patches(target_mask), 1
        )
        pred_tokens = apply_masks(pred_tokens, target_mask)

        context = context.repeat(len(context_mask), 1, 1)
        embeddings = torch.cat([context, pred_tokens], dim=1)

        cm = torch.cat(context_mask, dim=0)
        tm = torch.cat(target_mask, dim=0)
        masks = torch.cat([cm, tm], dim=1)
        return embeddings, masks

    @staticmethod
    def _max_patches(masks: list[torch.Tensor]) -> int:
        return int(max(m.max().item() for m in masks)) + 1


class VJEPA21Predictor(nn.Module):
    """V-JEPA 2.1 predictor with hierarchical input and context projection."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA21PredictorEmbeddings(config)

        dpr = [
            config.drop_path_rate * i / max(config.pred_num_hidden_layers - 1, 1)
            for i in range(config.pred_num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA21Layer(
                    config,
                    hidden_size=config.pred_hidden_size,
                    num_attention_heads=config.pred_num_attention_heads,
                    mlp_ratio=config.pred_mlp_ratio,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.pred_num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.pred_hidden_size, eps=config.layer_norm_eps)

        n_hier = len(config.predictor_hierarchical_layers)
        if config.pred_teacher_embed_dim is not None:
            out_dim = config.pred_teacher_embed_dim // n_hier
        else:
            out_dim = config.hidden_size
        proj_out_dim = n_hier * out_dim

        self.proj = nn.Linear(config.pred_hidden_size, proj_out_dim)
        self.proj_context = None
        if config.pred_return_all_tokens:
            self.proj_context = nn.Linear(config.pred_hidden_size, proj_out_dim)

        self.gradient_checkpointing = False

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: Union[torch.Tensor, list[torch.Tensor]],
        target_mask: Union[torch.Tensor, list[torch.Tensor]],
        mode: str = "video",
        mask_index: int = 1,
        context_is_masked: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> VJEPA21PredictorOutput:
        """
        Args:
            encoder_hidden_states: Encoder output. By default this is the *full*
                token sequence and the predictor gathers the context itself at
                `context_mask`. Set `context_is_masked=True` when passing an
                encoder output that was already produced with
                `encoder(..., masks=context_mask)`, which is the reference
                convention.
            mask_index: Which learnable mask token to inject. The reference uses
                the index of the sequence-length group (`mask_index=i` in
                `PredictorMultiSeqWrapper`); its default is 1.
            mode: "video" or "img" ("image" is accepted as an alias of "img",
                since the reference spells it that way).
        """
        if kwargs:
            _warn_once(
                f"VJEPA21Predictor.forward received unexpected keyword arguments "
                f"{sorted(kwargs)}; they are ignored."
            )

        context_mask = _as_mask_list(context_mask)
        target_mask = _as_mask_list(target_mask)
        if context_mask is None or target_mask is None:
            raise ValueError("the predictor requires both context_mask and target_mask")
        if len(context_mask) != len(target_mask):
            raise ValueError(
                "context_mask and target_mask must have the same length, got "
                f"{len(context_mask)} and {len(target_mask)}."
            )
        if len(context_mask) > 1:
            raise NotImplementedError(
                "The predictor currently supports a single (context_mask, target_mask) "
                "pair. Call it once per mask pair instead."
            )

        if context_is_masked:
            masked_states = encoder_hidden_states
            if masked_states.shape[1] != context_mask[0].shape[1]:
                raise ValueError(
                    f"context_is_masked=True but the encoder output has "
                    f"{masked_states.shape[1]} tokens while context_mask has "
                    f"{context_mask[0].shape[1]}."
                )
        else:
            masked_states = apply_masks(encoder_hidden_states, context_mask)
        _, N_ctxt, _ = masked_states.shape

        hidden_states, position_masks = self.embeddings(
            masked_states, context_mask, target_mask, mask_index=mask_index
        )

        # Sort tokens by position so RoPE sees monotonically increasing ids
        argsort = torch.argsort(position_masks, dim=1)
        idx_expand = argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, 1, idx_expand.to(hidden_states.device))
        position_masks = torch.gather(position_masks, 1, argsort.to(position_masks.device))

        if self.embeddings.img_mod_embed is not None:
            if mode in ("img", "image"):
                hidden_states = hidden_states + self.embeddings.img_mod_embed
            else:
                hidden_states = hidden_states + self.embeddings.video_mod_embed

        all_hidden_states: Optional[tuple] = () if output_hidden_states else None
        all_attentions: Optional[tuple] = () if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_out = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    position_masks,
                    None,
                    None,
                    None,
                    output_attentions,
                )
            else:
                layer_out = layer_module(
                    hidden_states,
                    position_mask=position_masks,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_out[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_out[1],)

        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Unsort
        reverse = torch.argsort(argsort, dim=1)
        rev_expand = reverse.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, 1, rev_expand.to(hidden_states.device))

        pred = self.proj(hidden_states[:, N_ctxt:])
        ctx = None
        if self.config.pred_return_all_tokens:
            ctx = self.proj_context(hidden_states[:, :N_ctxt])

        return VJEPA21PredictorOutput(
            last_hidden_state=pred,
            context_hidden_state=ctx,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# ---------------------------------------------------------------------------
# Attentive Pooler (for downstream tasks)
# ---------------------------------------------------------------------------


class VJEPA21PoolerSelfAttention(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_pooler_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_probs_dropout_prob
        self.is_causal = False

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor, output_attentions: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface = resolve_attention_interface(self.config, output_attentions)
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            None,
            is_causal=False,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )
        return self.out_proj(attn_output.reshape(B, N, C)), attn_weights


class VJEPA21PoolerCrossAttention(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_pooler_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_probs_dropout_prob
        self.is_causal = False

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, queries: torch.Tensor, kv: torch.Tensor, output_attentions: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, Nq, C = queries.shape
        Nkv = kv.shape[1]
        q = self.q_proj(queries).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface = resolve_attention_interface(self.config, output_attentions)
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            None,
            is_causal=False,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )
        return attn_output.reshape(B, Nq, C), attn_weights


class VJEPA21PoolerSelfAttentionLayer(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VJEPA21PoolerSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA21MLP(config, hidden_size=config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = residual + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class VJEPA21PoolerCrossAttentionLayer(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = VJEPA21PoolerCrossAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA21MLP(config, hidden_size=config.hidden_size)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        residual = queries
        hidden, _ = self.cross_attn(queries, self.layer_norm1(kv))
        hidden = residual + hidden

        residual = hidden
        hidden = residual + self.mlp(self.layer_norm2(hidden))
        return hidden


class VJEPA21AttentivePooler(nn.Module):
    """Attentive pooler matching `AttentivePooler(depth=num_pooler_layers + 1)`.

    The reference frozen-probe configs under `configs/eval_2_1/` use
    `num_probe_blocks: 4` and `num_heads: 16`, i.e. three self-attention blocks
    followed by one cross-attention block, with 16 heads. Those are the defaults
    of `num_pooler_layers` and `num_pooler_heads`.
    """

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cross_attention_layer = VJEPA21PoolerCrossAttentionLayer(config)
        self.self_attention_layers = nn.ModuleList(
            [VJEPA21PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state)
        queries = self.query_tokens.expand(hidden_state.shape[0], -1, -1)
        return self.cross_attention_layer(queries, hidden_state).squeeze(1)


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------


class VJEPA21PreTrainedModel(PreTrainedModel):
    config_class = VJEPA21Config
    base_model_prefix = "vjepa21"
    main_input_name = "pixel_values_videos"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VJEPA21Layer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range

        if isinstance(module, VJEPA21AttentivePooler):
            nn.init.trunc_normal_(module.query_tokens, std=std)
        elif isinstance(module, VJEPA21PredictorEmbeddings):
            if self.config.pred_zero_init_mask_tokens:
                for mt in module.mask_tokens:
                    nn.init.zeros_(mt)
            else:
                for mt in module.mask_tokens:
                    nn.init.trunc_normal_(mt, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

        if isinstance(module, VJEPA21Embeddings):
            if module.img_mod_embed is not None:
                nn.init.normal_(module.img_mod_embed, std=1e-6)
                nn.init.normal_(module.video_mod_embed, std=1e-6)
        if isinstance(module, VJEPA21PredictorEmbeddings):
            if module.img_mod_embed is not None:
                nn.init.normal_(module.img_mod_embed, std=1e-6)
                nn.init.normal_(module.video_mod_embed, std=1e-6)


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------


class VJEPA21Model(VJEPA21PreTrainedModel):
    """V-JEPA 2.1 model (encoder + predictor).

    Example, feature extraction:

    ```python
    model = VJEPA21Model.from_pretrained("apiantonio/vjepa2.1-vit-base-384")
    outputs = model(pixel_values_videos, skip_predictor=True)
    features = outputs.last_hidden_state
    ```

    Example, multi-level features (the `out_layers` recipe of the official
    frozen-evaluation probes):

    ```python
    levels = model(
        pixel_values_videos,
        skip_predictor=True,
        out_layers=model.config.encoder_hierarchical_layers,
    ).multilevel_hidden_states
    ```

    Example, the JEPA masked forward (encoder sees only the context tokens):

    ```python
    out = model(
        pixel_values_videos,
        masks=[context_idx],          # (B, K_ctx)
        context_mask=[context_idx],
        target_mask=[target_idx],     # (B, K_tgt)
    )
    prediction = out.predictor_output.last_hidden_state
    ```
    """

    def __init__(self, config: VJEPA21Config):
        super().__init__(config)
        self.encoder = VJEPA21Encoder(config)
        self.predictor = VJEPA21Predictor(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        masks: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        context_mask: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        target_mask: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        skip_predictor: bool = False,
        return_hierarchical: bool = False,
        out_layers: Optional[list[int]] = None,
        mask_index: int = 1,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> VJEPA21ModelOutput:
        """
        Args:
            pixel_values_videos: Video tensor. Accepted layouts:
                `(B, C, T, H, W)`, `(B, T, C, H, W)`, `(B, T, H, W, C)`.
            masks: Optional list of `(B, K)` index tensors applied to the patch
                tokens *before* the encoder layers. This reproduces the JEPA
                training forward. When set, `context_mask` and `target_mask` must
                be given explicitly, since the encoder output no longer spans the
                full token grid.
            context_mask: List of `(B, K)` index tensors for context tokens.
                Defaults to all tokens when `masks` is None.
            target_mask: List of `(B, K)` index tensors for target tokens.
                Defaults to all tokens when `masks` is None.
            skip_predictor: Skip the predictor forward (encoder only).
            return_hierarchical: Return the concatenated distillation levels.
            out_layers: Encoder layer indices whose normalised features should be
                returned in `multilevel_hidden_states`. Must be a subset of
                `config.encoder_hierarchical_layers`.
            mask_index: Which learnable predictor mask token to inject.
        """
        if pixel_values_videos is None:
            raise ValueError("pixel_values_videos is required")

        unexpected = set(kwargs) - _SILENTLY_IGNORED_KWARGS
        if unexpected:
            _warn_once(
                f"VJEPA21Model.forward received unexpected keyword arguments "
                f"{sorted(unexpected)}; they are ignored. Accepted arguments: "
                f"{sorted(_MODEL_FORWARD_KWARGS)}."
            )

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        masks = _as_mask_list(masks)
        context_mask = _as_mask_list(context_mask)
        target_mask = _as_mask_list(target_mask)

        # When the predictor fuses several distillation levels (n_output_distillation > 1,
        # e.g. the ViT-g and ViT-G checkpoints) its input projection expects the
        # concatenated hierarchical features, not the last hidden state.
        needs_hierarchical_input = (
            not skip_predictor and len(self.config.predictor_hierarchical_layers) > 1
        )

        encoder_out = self.encoder(
            pixel_values_videos,
            masks=masks,
            return_hierarchical=return_hierarchical or needs_hierarchical_input,
            out_layers=out_layers,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        seq_output = encoder_out.last_hidden_state

        predictor_output = None
        masked_hidden_state = None

        if not skip_predictor:
            predictor_input = (
                encoder_out.hierarchical_hidden_state
                if needs_hierarchical_input
                else seq_output
            )
            batch_size = seq_output.size(0)
            num_tokens = seq_output.size(1)
            device = seq_output.device

            if masks is not None:
                # The encoder output only covers the context tokens, so the
                # "all tokens" default is meaningless here.
                if context_mask is None:
                    context_mask = list(masks)
                if target_mask is None:
                    raise ValueError(
                        "target_mask must be given explicitly when the encoder runs with "
                        "`masks`, since the encoder output no longer spans the full token "
                        "grid. Pass `skip_predictor=True` if you only want the encoder."
                    )
                context_is_masked = True
            else:
                context_is_masked = False
                if context_mask is None:
                    context_mask = [
                        torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
                    ]
                if target_mask is None:
                    target_mask = [
                        torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
                    ]

            mode = self._detect_mode(pixel_values_videos)
            predictor_output = self.predictor(
                predictor_input,
                context_mask,
                target_mask,
                mode=mode,
                mask_index=mask_index,
                context_is_masked=context_is_masked,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            masked_hidden_state = (
                predictor_input
                if context_is_masked
                else apply_masks(predictor_input, context_mask)
            )

        return VJEPA21ModelOutput(
            last_hidden_state=seq_output,
            hierarchical_hidden_state=(
                encoder_out.hierarchical_hidden_state if return_hierarchical else None
            ),
            multilevel_hidden_states=encoder_out.multilevel_hidden_states,
            masked_hidden_state=masked_hidden_state,
            predictor_output=predictor_output,
            hidden_states=encoder_out.hidden_states,
            attentions=encoder_out.attentions,
        )

    def _detect_mode(self, pixel_values_videos: torch.Tensor) -> str:
        x = normalize_video_layout(pixel_values_videos, self.config.in_chans)
        T = x.shape[2]
        if self.config.img_temporal_dim_size is not None and T == self.config.img_temporal_dim_size:
            return "img"
        return "video"

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Extract encoder features (convenience method for VLM integration)."""
        return self.forward(pixel_values_videos, skip_predictor=True).last_hidden_state


class VJEPA21ForVideoClassification(VJEPA21PreTrainedModel):
    """V-JEPA 2.1 with attentive pooler + classification head.

    The pooler and the classifier are always randomly initialised: this class is
    the frozen-probe / fine-tuning entry point, not a pretrained classifier.
    """

    def __init__(self, config: VJEPA21Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vjepa21 = VJEPA21Model(config)
        self.pooler = VJEPA21AttentivePooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)` or `(batch_size, num_labels)`, *optional*):
            Labels for computing the classification loss. Integer indices in
            `[0, ..., config.num_labels - 1]` give single-label classification;
            a float multi-hot tensor gives multi-label classification (BCE), which
            is what a dataset such as XD-Violence needs. Set
            `config.problem_type` explicitly to remove the ambiguity.
        """
        # Only forward the arguments the backbone understands; the Trainer injects
        # extras such as `num_items_in_batch` that would otherwise reach the encoder.
        forwarded = {k: v for k, v in kwargs.items() if k in _MODEL_FORWARD_KWARGS}
        ignored = set(kwargs) - set(forwarded) - _SILENTLY_IGNORED_KWARGS
        if ignored:
            _warn_once(
                f"VJEPA21ForVideoClassification.forward received unexpected keyword "
                f"arguments {sorted(ignored)}; they are ignored."
            )
        forwarded.pop("skip_predictor", None)

        outputs = self.vjepa21(
            pixel_values_videos,
            skip_predictor=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **forwarded,
        )
        pooled = self.pooler(outputs.last_hidden_state)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_function(pooled_logits=logits, labels=labels, config=self.config)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "VJEPA21Model",
    "VJEPA21PreTrainedModel",
    "VJEPA21ForVideoClassification",
]
