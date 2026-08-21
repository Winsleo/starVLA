"""V-JEPA 2.1 model configuration"""

from typing import Optional

from transformers import PretrainedConfig


# Encoder hierarchical layers, from `app/vjepa_2_1/models/vision_transformer.py`
# in `facebookresearch/vjepa2`.
_ENCODER_LAYER_MAP = {
    4: [0, 1, 2, 3],
    8: [1, 3, 5, 7],
    12: [2, 5, 8, 11],
    20: [4, 9, 14, 19],
    24: [5, 11, 17, 23],
    40: [9, 19, 29, 39],
    48: [11, 23, 37, 47],
}

# Predictor hierarchical layers, from `app/vjepa_2_1/models/predictor.py`.
# NOTE: this is *not* the same table as the encoder's. At depth 24 the reference
# predictor uses [4, 11, 17, 23] while the encoder uses [5, 11, 17, 23], and the
# predictor table has no entry for depth 48. Sharing one table happens to work
# today because only `len(...)` is consumed, but it would silently produce wrong
# indices as soon as multi-level predictor outputs are exposed.
_PREDICTOR_LAYER_MAP = {
    4: [0, 1, 2, 3],
    8: [1, 3, 5, 7],
    12: [2, 5, 8, 11],
    20: [4, 9, 14, 19],
    24: [4, 11, 17, 23],
    40: [9, 19, 29, 39],
}


def _get_hierarchical_layers(depth: int) -> list[int]:
    """Encoder hierarchical layer indices for a given depth."""
    if depth not in _ENCODER_LAYER_MAP:
        raise ValueError(
            f"Unsupported encoder depth {depth}. Supported depths: "
            f"{list(_ENCODER_LAYER_MAP.keys())}"
        )
    return _ENCODER_LAYER_MAP[depth]


def _get_predictor_hierarchical_layers(depth: int) -> list[int]:
    """Predictor hierarchical layer indices for a given depth."""
    if depth not in _PREDICTOR_LAYER_MAP:
        raise ValueError(
            f"Unsupported predictor depth {depth}. Supported depths: "
            f"{list(_PREDICTOR_LAYER_MAP.keys())}"
        )
    return _PREDICTOR_LAYER_MAP[depth]


class VJEPA21Config(PretrainedConfig):
    r"""
    Configuration class for the V-JEPA 2.1 model.

    V-JEPA 2.1 extends V-JEPA 2 with:
    - Multi-modality support (image + video with modality embeddings)
    - Hierarchical output distillation across intermediate layers
    - Interpolatable RoPE for variable input resolutions
    - Dense predictive loss with context token prediction

    Args:
        patch_size (`int`, defaults to 16):
            Spatial patch size.
        crop_size (`int`, defaults to 384):
            Input resolution of the model.
        frames_per_clip (`int`, defaults to 64):
            Number of frames in a video clip used during pre-training. This is
            informational: the model accepts any number of frames at inference.
        tubelet_size (`int`, defaults to 2):
            Temporal patch size (number of frames per tubelet).
        hidden_size (`int`, defaults to 1024):
            Encoder embedding dimension.
        in_chans (`int`, defaults to 3):
            Number of input channels.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads in the encoder.
        num_hidden_layers (`int`, defaults to 24):
            Number of encoder transformer layers.
        drop_path_rate (`float`, defaults to 0.0):
            Stochastic depth rate.
        mlp_ratio (`float`, defaults to 4.0):
            Ratio of MLP hidden dim to embedding dim.
        layer_norm_eps (`float`, defaults to 1e-6):
            Layer normalization epsilon.
        qkv_bias (`bool`, defaults to True):
            Whether to use bias in QKV projection.
        hidden_act (`str`, defaults to "gelu"):
            Activation function in MLP. "silu" enables SwiGLU.
        wide_silu (`bool`, defaults to True):
            Whether to use wide SwiGLU (2/3 hidden features) when hidden_act is "silu".
        initializer_range (`float`, defaults to 0.02):
            Standard deviation for weight initialization.
        attention_probs_dropout_prob (`float`, defaults to 0.0):
            Dropout probability for attention weights.
        img_temporal_dim_size (`int` or `None`, defaults to 1):
            Temporal dimension for image inputs. When set, a separate patch embedding
            with tubelet_size=1 is used for images. Set to None to disable.
        interpolate_rope (`bool`, defaults to True):
            Whether to interpolate RoPE frequencies for variable input resolutions.
        modality_embedding (`bool`, defaults to True):
            Whether to add learned modality embeddings (image vs video).
        n_output_distillation (`int`, defaults to 4):
            Number of intermediate encoder layers for hierarchical output.
            Set to 1 to only use the final layer output.
        n_registers (`int`, defaults to 0):
            Number of register tokens (appended to sequence).
        has_cls_first (`bool`, defaults to False):
            Whether the sequence starts with a CLS token.
        num_pooler_layers (`int`, defaults to 3):
            Number of self-attention layers in the attentive pooler. Together with
            the cross-attention layer this reproduces `AttentivePooler(depth=4)`,
            which is `num_probe_blocks: 4` in the reference evaluation configs.
        num_pooler_heads (`int` or `None`, defaults to 16):
            Number of attention heads in the attentive pooler. 16 is the value used
            by every frozen-probe config under `configs/eval_2_1/` in the reference
            repository (`classifier.num_heads: 16`), for all four model sizes, so it
            is the default here rather than `num_attention_heads`. The pooler is
            always trained from scratch, so this only affects the probe you train.
        pred_hidden_size (`int`, defaults to 384):
            Predictor embedding dimension.
        pred_num_attention_heads (`int`, defaults to 12):
            Number of attention heads in the predictor.
        pred_num_hidden_layers (`int`, defaults to 12):
            Number of predictor transformer layers.
        pred_num_mask_tokens (`int`, defaults to 8):
            Number of learnable mask tokens in the predictor.
        pred_zero_init_mask_tokens (`bool`, defaults to True):
            Whether to zero-initialize mask tokens.
        pred_mlp_ratio (`float`, defaults to 4.0):
            MLP ratio in the predictor.
        pred_teacher_embed_dim (`int` or `None`, defaults to None):
            Teacher embedding dimension for predictor output projection.
            When set, predictor projects to teacher_embed_dim // n_hierarchical_layers per layer.
        pred_return_all_tokens (`bool`, defaults to False):
            Whether the predictor returns predictions for both masked and context tokens.
    """

    model_type = "vjepa21"

    def __init__(
        self,
        patch_size: int = 16,
        crop_size: int = 384,
        frames_per_clip: int = 64,
        tubelet_size: int = 2,
        hidden_size: int = 1024,
        in_chans: int = 3,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        drop_path_rate: float = 0.0,
        mlp_ratio: float = 4.0,
        layer_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        hidden_act: str = "gelu",
        wide_silu: bool = True,
        initializer_range: float = 0.02,
        attention_probs_dropout_prob: float = 0.0,
        # V-JEPA 2.1 specific
        img_temporal_dim_size: Optional[int] = 1,
        interpolate_rope: bool = True,
        modality_embedding: bool = True,
        n_output_distillation: int = 4,
        n_registers: int = 0,
        has_cls_first: bool = False,
        # Pooler
        num_pooler_layers: int = 3,
        num_pooler_heads: Optional[int] = 16,
        # Predictor
        pred_hidden_size: int = 384,
        pred_num_attention_heads: int = 12,
        pred_num_hidden_layers: int = 12,
        pred_num_mask_tokens: int = 8,
        pred_zero_init_mask_tokens: bool = True,
        pred_mlp_ratio: float = 4.0,
        pred_teacher_embed_dim: Optional[int] = None,
        pred_return_all_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.crop_size = crop_size
        self.frames_per_clip = frames_per_clip
        self.tubelet_size = tubelet_size
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.hidden_act = hidden_act
        self.wide_silu = wide_silu
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # V-JEPA 2.1 specific
        self.img_temporal_dim_size = img_temporal_dim_size
        self.interpolate_rope = interpolate_rope
        self.modality_embedding = modality_embedding
        self.n_output_distillation = n_output_distillation
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first

        # Pooler
        self.num_pooler_layers = num_pooler_layers
        self.num_pooler_heads = num_pooler_heads if num_pooler_heads is not None else 16

        # Predictor
        self.pred_hidden_size = pred_hidden_size
        self.pred_num_attention_heads = pred_num_attention_heads
        self.pred_num_hidden_layers = pred_num_hidden_layers
        self.pred_num_mask_tokens = pred_num_mask_tokens
        self.pred_zero_init_mask_tokens = pred_zero_init_mask_tokens
        self.pred_mlp_ratio = pred_mlp_ratio
        self.pred_teacher_embed_dim = pred_teacher_embed_dim
        self.pred_return_all_tokens = pred_return_all_tokens

        self._validate()

    def _validate(self) -> None:
        n_levels = len(_get_hierarchical_layers(self.num_hidden_layers))
        n_pred_levels = len(_get_predictor_hierarchical_layers(self.pred_num_hidden_layers))
        if not 1 <= self.n_output_distillation <= n_levels:
            raise ValueError(
                f"n_output_distillation must be in [1, {n_levels}] for a model with "
                f"{self.num_hidden_layers} layers, got {self.n_output_distillation}."
            )
        if self.n_output_distillation > n_pred_levels:
            raise ValueError(
                f"n_output_distillation ({self.n_output_distillation}) exceeds the "
                f"{n_pred_levels} hierarchical levels available in a predictor with "
                f"{self.pred_num_hidden_layers} layers."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.hidden_size % self.num_pooler_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_pooler_heads ({self.num_pooler_heads})."
            )
        if self.pred_hidden_size % self.pred_num_attention_heads != 0:
            raise ValueError(
                f"pred_hidden_size ({self.pred_hidden_size}) must be divisible by "
                f"pred_num_attention_heads ({self.pred_num_attention_heads})."
            )
        if self.pred_teacher_embed_dim is not None:
            if self.pred_teacher_embed_dim % self.n_output_distillation != 0:
                raise ValueError(
                    f"pred_teacher_embed_dim ({self.pred_teacher_embed_dim}) must be "
                    f"divisible by n_output_distillation ({self.n_output_distillation})."
                )
        if self.tubelet_size < 1:
            raise ValueError(f"tubelet_size must be >= 1, got {self.tubelet_size}.")
        if self.pred_num_mask_tokens < 1:
            raise ValueError(
                f"pred_num_mask_tokens must be >= 1, got {self.pred_num_mask_tokens}."
            )

    @property
    def encoder_hierarchical_layers(self) -> list[int]:
        """Layer indices at which the encoder carries a per-level LayerNorm."""
        return _get_hierarchical_layers(self.num_hidden_layers)

    @property
    def encoder_distillation_layers(self) -> list[int]:
        """Encoder layer indices contributing to the hierarchical output."""
        all_layers = _get_hierarchical_layers(self.num_hidden_layers)
        return all_layers[-self.n_output_distillation :]

    @property
    def predictor_hierarchical_layers(self) -> list[int]:
        """Predictor layer indices for hierarchical output.

        Uses the predictor's own depth table, which differs from the encoder's at
        depth 24 (`[4, 11, 17, 23]` vs `[5, 11, 17, 23]`).
        """
        all_layers = _get_predictor_hierarchical_layers(self.pred_num_hidden_layers)
        return all_layers[-self.n_output_distillation :]

    @property
    def pretrained_grid_size(self) -> int:
        """Grid size used during pre-training (for RoPE interpolation)."""
        if self.patch_size == 14:
            return int(252 / self.patch_size)
        return int(256 / self.patch_size)


__all__ = ["VJEPA21Config"]
