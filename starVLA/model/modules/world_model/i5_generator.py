# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Action-conditioned latent generator skeleton for I5 (`I5-S2-SKEL`).

The three pieces I5 adds on top of the frozen tokenizer, per D-062:

1. **Condition projector.** The latent action is 24 tokens of 2048 dims (3 predictor transitions x 8
   tokens). They are projected into the DiT's cross-attention key/value space and appended after the
   text tokens. That path needs no change inside the DiT: `WanTransformer3DModel`'s condition
   embedder applies a per-token MLP to `encoder_hidden_states`, so a longer sequence just works, and
   the hardcoded 512-token text split in the attention processor only activates when the
   `added_kv_proj_dim` branch is enabled, which it is not.
2. **Segmented timesteps.** Wan 2.2 TI2V drives per-token timesteps. Latent frame 0 is the clean
   conditioning frame and gets timestep 0; the future frames get the sampled timestep. This is the
   mechanism that makes "given the current frame, denoise the future" the model's native task.
3. **Partial noising.** Noise is interpolated onto the future latent frames only, so the
   conditioning frame stays exactly the encoder's output.

**What "no action condition" means here.** Every arm keeps the *same* token layout; the arms differ
only in the *content* of the action segment. That is deliberate. Appending or dropping tokens would
change the attention denominator and give the arms different shapes, so an observed difference could
not be attributed to the condition alone. With a zero-initialised projector the action segment
carries only the modality embedding at step 0 -- no sample-dependent information -- so a
correct-condition model and a no-condition model start from bit-identical outputs.
`tests/test_i5_generator.py` asserts that equality rather than assuming it: an earlier phrasing of
this claim (D-062) said zero-init made the arms identical without pinning down that the layout has
to match for it to be true.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Latent action geometry, from the pinned VLA-JEPA config (`num_action_tokens_per_timestep` 8 across
#: `num_temporal_blocks - 1` = 3 transitions, `vl_hidden_dim` 2048).
ACTION_TOKENS = 24
ACTION_DIM = 2048

#: Wan 2.2 TI2V-5B cross-attention input width (`text_dim`).
CONDITION_DIM = 4096


class ActionConditionProjector(nn.Module):
    """Latent action tokens -> cross-attention condition tokens.

    Zero-initialised by default so training starts from the no-information state, and carrying a
    learnable modality embedding so the DiT can tell action tokens from text tokens sharing the same
    key/value space.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        condition_dim: int = CONDITION_DIM,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(action_dim, condition_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, condition_dim))
        self.zero_init = zero_init
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if action_tokens.ndim != 3:
            raise ValueError(f"expected [B, N, action_dim], got {tuple(action_tokens.shape)}")
        return self.proj(action_tokens) + self.modality_embedding

    def null_condition(self, batch_size: int, num_tokens: int = ACTION_TOKENS) -> torch.Tensor:
        """The masked-action segment: modality embedding only, no action content.

        This is what the masked arm feeds and what the no-condition arm trains against, so both keep
        the layout of the correct-condition arm.
        """
        marker = self.modality_embedding.expand(batch_size, num_tokens, -1)
        return marker + self.proj.bias


def concat_condition(text_embeds: torch.Tensor, action_embeds: torch.Tensor) -> torch.Tensor:
    """`[B, L_text + N, condition_dim]`, action tokens after the text tokens.

    Order matters if the `added_kv_proj_dim` branch is ever enabled: its processor splits the
    sequence as "everything before the last 512 tokens is the second modality", so the extra tokens
    would have to be *first*. With that branch disabled, appending keeps the pretrained text tokens
    at the positions the model was trained on.
    """
    if text_embeds.shape[0] != action_embeds.shape[0]:
        raise ValueError(
            f"batch mismatch: text {text_embeds.shape[0]} vs action {action_embeds.shape[0]}"
        )
    if text_embeds.shape[-1] != action_embeds.shape[-1]:
        raise ValueError(
            f"width mismatch: text {text_embeds.shape[-1]} vs action {action_embeds.shape[-1]}"
        )
    return torch.cat([text_embeds, action_embeds], dim=1)


def token_grid(
    latent_frames: int, latent_height: int, latent_width: int, patch_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    """DiT token grid for a latent shape, i.e. latent dims divided by the patch size."""
    p_t, p_h, p_w = patch_size
    for name, size, patch in (
        ("frames", latent_frames, p_t),
        ("height", latent_height, p_h),
        ("width", latent_width, p_w),
    ):
        if size % patch:
            raise ValueError(f"latent {name} {size} is not divisible by patch {patch}")
    return latent_frames // p_t, latent_height // p_h, latent_width // p_w


def clean_frame_mask(latent_frames: int, num_clean_frames: int = 1) -> torch.Tensor:
    """`[T_lat]` bool, True where the frame is the clean conditioning context."""
    if not 0 < num_clean_frames < latent_frames:
        raise ValueError(
            f"num_clean_frames must be in (0, {latent_frames}), got {num_clean_frames}: the "
            "generator needs at least one clean frame to condition on and one to predict"
        )
    mask = torch.zeros(latent_frames, dtype=torch.bool)
    mask[:num_clean_frames] = True
    return mask


def segmented_timesteps(
    timestep: torch.Tensor,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
    *,
    num_clean_frames: int = 1,
) -> torch.Tensor:
    """Per-token timesteps `[B, seq_len]`: 0 on the clean frames, `timestep` on the future ones.

    Wan 2.2 TI2V's `expand_timesteps` mode reads a timestep per token, which is how it conditions on
    a clean first frame. `Wan2.py`'s feature-extraction path sets every token to 0; generation needs
    this segmented form instead.
    """
    if timestep.ndim != 1:
        raise ValueError(f"expected a timestep per batch element, got {tuple(timestep.shape)}")
    frames, height, width = token_grid(latent_frames, latent_height, latent_width, patch_size)
    tokens_per_frame = height * width
    frame_is_clean = clean_frame_mask(frames, num_clean_frames)
    per_frame = torch.where(
        frame_is_clean.to(timestep.device),
        torch.zeros_like(timestep[:, None].expand(-1, frames)),
        timestep[:, None].expand(-1, frames),
    )
    return per_frame.repeat_interleave(tokens_per_frame, dim=1)


def apply_partial_noise(
    latents: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor, *, num_clean_frames: int = 1
) -> torch.Tensor:
    """Flow-matching interpolation on the future frames only.

    `x_sigma = (1 - sigma) * x0 + sigma * noise` where the frame is predicted, and exactly `x0`
    where it is the clean conditioning context -- the conditioning frame must stay bit-identical to
    the encoder output, otherwise the model is conditioned on something the tokenizer never emits.

    The velocity target that pairs with this interpolation is deferred to S3, where it can be checked
    against the scheduler rather than assumed.
    """
    if latents.shape != noise.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} != latents {tuple(latents.shape)}")
    if sigma.ndim != 1 or sigma.shape[0] != latents.shape[0]:
        raise ValueError(f"expected one sigma per batch element, got {tuple(sigma.shape)}")
    frame_is_clean = clean_frame_mask(latents.shape[2], num_clean_frames).to(latents.device)
    keep = frame_is_clean.view(1, 1, -1, 1, 1)
    scale = sigma.view(-1, 1, 1, 1, 1).to(latents.dtype)
    noisy = (1.0 - scale) * latents + scale * noise
    return torch.where(keep, latents, noisy)


class LatentGeneratorSkeleton(nn.Module):
    """Frozen-tokenizer-side generator: a DiT plus the condition projector, wired but untrained.

    The transformer is injected rather than loaded, so the whole contract is testable without the
    20 GB of Wan weights. S2 builds and asserts the wiring; S3 adds the loss.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        action_tokens: int = ACTION_TOKENS,
        action_dim: int = ACTION_DIM,
        condition_dim: int = CONDITION_DIM,
        num_clean_frames: int = 1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.projector = ActionConditionProjector(action_dim, condition_dim, zero_init=zero_init)
        self.action_tokens = action_tokens
        self.num_clean_frames = num_clean_frames
        patch = tuple(int(p) for p in transformer.config.patch_size)
        if len(patch) != 3:
            raise ValueError(f"expected a 3-tuple patch_size, got {patch}")
        self.patch_size = patch

    def condition(
        self, text_embeds: torch.Tensor, action_tokens: torch.Tensor | None
    ) -> torch.Tensor:
        """Build the cross-attention sequence. `None` action tokens give the null segment.

        Passing `None` is the masked arm and the no-condition arm; the layout is identical either
        way, so the arms differ only in content.
        """
        if action_tokens is None:
            action_embeds = self.projector.null_condition(text_embeds.shape[0], self.action_tokens)
        else:
            if action_tokens.shape[1] != self.action_tokens:
                raise ValueError(
                    f"expected {self.action_tokens} action tokens, got {action_tokens.shape[1]}"
                )
            action_embeds = self.projector(action_tokens)
        return concat_condition(text_embeds, action_embeds.to(text_embeds.dtype))

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        action_tokens: torch.Tensor | None,
    ):
        """Run the DiT with segmented timesteps and the assembled condition sequence."""
        if noisy_latents.ndim != 5:
            raise ValueError(f"expected [B, C, T_lat, h, w], got {tuple(noisy_latents.shape)}")
        _, _, frames, height, width = noisy_latents.shape
        per_token_timestep = segmented_timesteps(
            timestep,
            frames,
            height,
            width,
            self.patch_size,
            num_clean_frames=self.num_clean_frames,
        )
        return self.transformer(
            hidden_states=noisy_latents,
            timestep=per_token_timestep,
            encoder_hidden_states=self.condition(text_embeds, action_tokens),
        )
