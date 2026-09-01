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

import math

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

    The velocity target that pairs with this interpolation is :func:`flow_velocity_target`, verified
    against the real scheduler rather than assumed -- see that function's notes.
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


#: Wan 2.2's `num_train_timesteps`, from its scheduler config.
FLOW_NUM_TRAIN_TIMESTEPS = 1000


def flow_velocity_target(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """The flow-matching target that pairs with :func:`apply_partial_noise`: `noise - latents`.

    Not a convention taken on faith. Read off the pinned scheduler and then checked numerically:

    * `UniPCMultistepScheduler.convert_model_output` under `prediction_type="flow_prediction"`
      computes `x0 = sample - sigma * model_output`
      (`scheduling_unipc_multistep.py`, the `flow_prediction` branch);
    * so with `x_sigma = (1 - sigma) * x0 + sigma * noise`, consistency forces
      `v = noise - x0`;
    * feeding exactly that `v` back through the real scheduler recovers `x0` to float32 epsilon
      (max absolute error 2e-7 to 5e-7 across sigmas 0.63 to 1.0), while the opposite sign
      `x0 - noise` is off by 9.0 -- so the check is not vacuous.

    `tests/test_i5_generator.py` pins both directions against the real scheduler.
    """
    if latents.shape != noise.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} != latents {tuple(latents.shape)}")
    return noise - latents


def timestep_from_sigma(
    sigma: torch.Tensor, num_train_timesteps: int = FLOW_NUM_TRAIN_TIMESTEPS
) -> torch.Tensor:
    """`floor(sigma * num_train_timesteps)` as int64, matching the scheduler's own mapping.

    `set_timesteps` computes `timesteps = sigmas * num_train_timesteps` and stores them as int64, so
    the relation holds up to that truncation (verified: it equals `floor(sigma * 1000)` exactly, and
    the residual is below one timestep). The training loop has to use the same mapping, otherwise the
    per-token timestep the DiT sees disagrees with the sigma the latents were noised at.

    **Still open for S3**: the *training-time* distribution of sigma is not determined by the
    scheduler. `flow_shift=5.0` shapes the inference schedule, not the training sampler, and whether
    Wan trained with uniform sigma, a logit-normal in `u`, or a shifted distribution cannot be read
    off the config. That choice is an engineering default and must be recorded as one.
    """
    if not torch.is_floating_point(sigma):
        raise ValueError(f"sigma must be floating point, got {sigma.dtype}")
    return torch.floor(sigma.double() * num_train_timesteps).to(torch.long)


def aligned_sigma_mean(flow_shift: float = 5.0) -> float:
    """The logit-normal mean that reproduces the scheduler's `flow_shift`.

    The shift diffusers applies to build the inference schedule,
    `sigma' = shift * sigma / (1 + (shift - 1) * sigma)`, is exactly a constant translation in logit
    space: `sigma'/(1 - sigma') = shift * sigma/(1 - sigma)`, so `logit(sigma') = logit(sigma) +
    log(shift)`. Verified numerically to 1.8e-7 against the formula itself.

    So "shifted sigma" is not a separate distribution family -- it is logit-normal with
    `mean = log(flow_shift)`. At `flow_shift=5` that is 1.609, putting the median at 0.833, which is
    where the sampler actually spends its steps (measured: the 20-step schedule has median sigma
    0.860). A mean of 0 instead draws only 8% of its samples above 0.8, i.e. it would train the
    adapter mostly at noise levels inference barely visits.
    """
    if flow_shift <= 0:
        raise ValueError(f"flow_shift must be positive, got {flow_shift}")
    return math.log(flow_shift)


def wan_training_sigmas(
    num_steps: int = FLOW_NUM_TRAIN_TIMESTEPS, flow_shift: float = 5.0
) -> torch.Tensor:
    """The shifted sigma grid a Wan training run draws from, `[num_steps]` descending from 1.

    Reproduces `FlowMatchScheduler.set_timesteps_wan` in DiffSynth-Studio: a uniform grid on
    `[1, 0)` put through the same shift the inference schedule uses. Wan 2.2's own README points at
    DiffSynth for training, so this is the endorsed recipe rather than a guess (D-067).
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2, got {num_steps}")
    if flow_shift <= 0:
        raise ValueError(f"flow_shift must be positive, got {flow_shift}")
    raw = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    return flow_shift * raw / (1 + (flow_shift - 1) * raw)


def bell_timestep_weights(
    sigmas: torch.Tensor, num_train_timesteps: int = FLOW_NUM_TRAIN_TIMESTEPS
) -> torch.Tensor:
    """Per-sigma loss weight, reproducing DiffSynth's `set_training_weight`.

    A Gaussian bump on the timestep axis centred at `num_train_timesteps / 2`, min-subtracted and
    normalised so the weights average to 1. It **counteracts** the shift: the shifted grid puts most
    samples at high sigma (median 0.834), and this weight pulls the effective emphasis back to a
    median of 0.677. Taking the sampling without the weight would train noticeably higher on the
    sigma axis than the recipe that was actually validated -- 0.556 of the effective mass above
    sigma 0.8 instead of 0.280 -- so the two travel together (D-067).

    Upstream calls the shape "an empirical formula"; it is adopted as a validated pair, not because
    its form is derived from anything.
    """
    steps = float(num_train_timesteps)
    timesteps = sigmas * steps
    bump = torch.exp(-2 * ((timesteps - steps / 2) / steps) ** 2)
    shifted = bump - bump.min()
    return shifted * (steps / shifted.sum())


def sample_sigma(
    batch_size: int,
    *,
    distribution: str = "shifted_uniform",
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    mean: float = 0.0,
    std: float = 1.0,
    flow_shift: float = 5.0,
    num_steps: int = FLOW_NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """Training-time sigma, `[B]`.

    `shifted_uniform` is the default and is the endorsed recipe (D-067): draw an index uniformly from
    the shifted grid, exactly as DiffSynth's training loop does. Numerically its density is nearly
    the same as `logit_normal` with `mean = log(flow_shift)` (medians 0.834 and 0.833), which is why
    "shifted" was never a separate distribution family -- see :func:`aligned_sigma_mean`.

    `logit_normal` and `uniform` remain available as controls. What the *pretraining* used is still
    not public; the pinned scheduler fixes the interpolation, the target and the timestep mapping,
    but not the sampler.
    """
    if distribution == "shifted_uniform":
        grid = wan_training_sigmas(num_steps, flow_shift).to(device)
        index = torch.randint(0, grid.numel(), (batch_size,), generator=generator, device=device)
        return grid[index]
    if distribution == "logit_normal":
        z = torch.randn(batch_size, generator=generator, device=device)
        return torch.sigmoid(z * std + mean)
    if distribution == "uniform":
        return torch.rand(batch_size, generator=generator, device=device)
    raise ValueError(
        f"unknown sigma distribution {distribution!r}; "
        "expected shifted_uniform, logit_normal or uniform"
    )


def flow_matching_loss(
    prediction: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    *,
    num_clean_frames: int = 1,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked flow-matching loss: `(raw_loss, weighted_loss, per_element_squared_error)`.

    The loss covers **only the frames the model was asked to predict**. Including the clean
    conditioning frame would reward copying an input it was handed unnoised, which would both depress
    the loss and hide whether anything was learned.

    `weight` is the per-sample timestep weight from :func:`bell_timestep_weights`. Raw and weighted
    come back separately because `AGENTS.md` section 10 requires both to be logged, and because a
    weighted loss alone cannot be compared across sigma distributions. The unreduced squared error is
    returned too, so the error can be broken down per latent frame.
    """
    target = flow_velocity_target(latents, noise)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    predicted_frames = ~clean_frame_mask(latents.shape[2], num_clean_frames).to(latents.device)
    squared_error = (prediction.float() - target.float()) ** 2
    masked = squared_error[:, :, predicted_frames]
    raw = masked.mean()
    if weight is None:
        return raw, raw, squared_error
    if weight.shape != (latents.shape[0],):
        raise ValueError(f"expected one weight per batch element, got {tuple(weight.shape)}")
    per_sample = masked.flatten(1).mean(dim=1)
    weighted = (per_sample * weight.to(per_sample.dtype)).mean()
    return raw, weighted, squared_error


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
