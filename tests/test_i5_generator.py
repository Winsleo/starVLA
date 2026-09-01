"""Gates for the I5 generator skeleton (`I5-S2-SKEL`).

S2 wires the condition path and the segmented-timestep mechanism; it trains nothing. What has to hold
before S3 adds a loss:

* **the conditioning frame stays exactly the encoder's output** -- noise touches future frames only,
  otherwise the model is conditioned on something the tokenizer never emits;
* **per-token timesteps are 0 on the clean frame and the sampled value on the future frames**, which
  is what makes "given the current frame, denoise the future" the native task;
* **all arms share one token layout**, so correct / masked / no-condition differ only in content and
  an observed difference cannot be attributed to sequence length;
* **zero-init means no sample-dependent information flows at step 0**, so the correct-condition and
  no-condition arms start bit-identical;
* **masking the action segment is exactly the masked arm**, which is what makes that arm free;
* the generator is a separate module: the Fast Policy path neither imports nor loads it.

The transformer is stubbed, so this runs on CPU without the 20 GB of Wan weights. The stub mirrors the
real geometry that matters here: `config.patch_size` and a cross-attention that actually consumes the
condition sequence, so a change in that sequence is observable in the output.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.nn as nn

from starVLA.model.modules.world_model.i5_generator import (
    ACTION_DIM,
    ACTION_TOKENS,
    CONDITION_DIM,
    ActionConditionProjector,
    LatentGeneratorSkeleton,
    aligned_sigma_mean,
    bell_timestep_weights,
    apply_partial_noise,
    clean_frame_mask,
    concat_condition,
    flow_matching_loss,
    flow_velocity_target,
    sample_sigma,
    wan_training_sigmas,
    segmented_timesteps,
    timestep_from_sigma,
    token_grid,
)

PATCH = (1, 2, 2)
LATENT = (48, 3, 16, 16)  # C, T_lat, h, w -- the shape S1's cache actually holds
TEXT_TOKENS = 12
HIDDEN = 32


class _StubConfig:
    patch_size = list(PATCH)


class _StubTransformer(nn.Module):
    """Consumes both the per-token timesteps and the condition sequence, so both are observable."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _StubConfig()
        self.patchify = nn.Conv3d(LATENT[0], HIDDEN, kernel_size=PATCH, stride=PATCH)
        self.condition_proj = nn.Linear(CONDITION_DIM, HIDDEN)
        self.time_proj = nn.Embedding(1000, HIDDEN)
        self.unpatchify = nn.ConvTranspose3d(HIDDEN, LATENT[0], kernel_size=PATCH, stride=PATCH)

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        tokens = self.patchify(hidden_states)  # [B, HIDDEN, T, h', w']
        b, c, t, h, w = tokens.shape
        flat = tokens.flatten(2).transpose(1, 2)  # [B, seq, HIDDEN]
        flat = flat + self.time_proj(timestep)
        keys = self.condition_proj(encoder_hidden_states)  # [B, L, HIDDEN]
        attention = torch.softmax(flat @ keys.transpose(1, 2), dim=-1) @ keys
        merged = (flat + attention).transpose(1, 2).reshape(b, c, t, h, w)
        return self.unpatchify(merged)


def _latents(batch: int = 2, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, *LATENT, generator=generator)


def _text(batch: int = 2, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, TEXT_TOKENS, CONDITION_DIM, generator=generator)


def _actions(batch: int = 2, seed: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, ACTION_TOKENS, ACTION_DIM, generator=generator)


class TokenLayoutTest(unittest.TestCase):
    def test_token_grid_matches_the_cached_latent_shape(self):
        # 3 latent frames x 16x16 under patch (1, 2, 2) -> 3 x 8 x 8 = 192 tokens, far under the
        # transformer's rope_max_seq_len of 1024.
        self.assertEqual(token_grid(3, 16, 16, PATCH), (3, 8, 8))
        self.assertEqual(3 * 8 * 8, 192)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            token_grid(3, 15, 16, PATCH)

    def test_clean_frame_mask_needs_context_and_a_target(self):
        torch.testing.assert_close(
            clean_frame_mask(3, 1), torch.tensor([True, False, False])
        )
        torch.testing.assert_close(clean_frame_mask(3, 2), torch.tensor([True, True, False]))
        for bad in (0, 3, 4):
            with self.assertRaisesRegex(ValueError, "num_clean_frames"):
                clean_frame_mask(3, bad)

    def test_segmented_timesteps_are_zero_on_the_clean_frame(self):
        timestep = torch.tensor([700, 250])
        per_token = segmented_timesteps(timestep, 3, 16, 16, PATCH)
        self.assertEqual(tuple(per_token.shape), (2, 192))
        # 64 tokens per latent frame: frame 0 clean, frames 1 and 2 noised.
        torch.testing.assert_close(per_token[:, :64], torch.zeros(2, 64, dtype=torch.long))
        torch.testing.assert_close(
            per_token[:, 64:], timestep[:, None].expand(-1, 128).contiguous()
        )

    def test_segmented_timesteps_reject_a_scalar(self):
        with self.assertRaisesRegex(ValueError, "timestep per batch element"):
            segmented_timesteps(torch.tensor(700), 3, 16, 16, PATCH)


class PartialNoiseTest(unittest.TestCase):
    def test_conditioning_frame_is_bit_identical_to_the_encoder_output(self):
        latents, noise = _latents(), _latents(seed=9)
        sigma = torch.tensor([0.3, 0.85])
        noisy = apply_partial_noise(latents, noise, sigma)
        self.assertTrue(torch.equal(noisy[:, :, 0], latents[:, :, 0]))
        # And the future frames really are noised, so the check above is not vacuous.
        self.assertFalse(torch.equal(noisy[:, :, 1], latents[:, :, 1]))
        self.assertFalse(torch.equal(noisy[:, :, 2], latents[:, :, 2]))

    def test_interpolation_is_the_flow_matching_form(self):
        latents, noise = _latents(batch=1), _latents(batch=1, seed=9)
        sigma = torch.tensor([0.25])
        noisy = apply_partial_noise(latents, noise, sigma)
        expected = 0.75 * latents[:, :, 1:] + 0.25 * noise[:, :, 1:]
        torch.testing.assert_close(noisy[:, :, 1:], expected)

    def test_two_clean_frames_leave_only_the_last_noised(self):
        latents, noise = _latents(), _latents(seed=9)
        noisy = apply_partial_noise(latents, noise, torch.tensor([0.5, 0.5]), num_clean_frames=2)
        self.assertTrue(torch.equal(noisy[:, :, :2], latents[:, :, :2]))
        self.assertFalse(torch.equal(noisy[:, :, 2], latents[:, :, 2]))

    def test_shape_mismatches_are_rejected(self):
        latents = _latents()
        with self.assertRaisesRegex(ValueError, "noise shape"):
            apply_partial_noise(latents, _latents(batch=1), torch.tensor([0.5, 0.5]))
        with self.assertRaisesRegex(ValueError, "one sigma per batch element"):
            apply_partial_noise(latents, _latents(seed=9), torch.tensor(0.5))


class ConditionProjectorTest(unittest.TestCase):
    def test_zero_init_carries_no_sample_dependent_information(self):
        projector = ActionConditionProjector()
        first = projector(_actions(seed=3))
        second = projector(_actions(seed=4))
        # Different actions, identical output: at step 0 the segment cannot discriminate samples.
        self.assertTrue(torch.equal(first, second))
        # And it equals the null segment, which is what the masked arm feeds.
        self.assertTrue(torch.equal(first, projector.null_condition(first.shape[0])))

    def test_without_zero_init_the_action_content_does_flow(self):
        projector = ActionConditionProjector(zero_init=False)
        self.assertFalse(torch.equal(projector(_actions(seed=3)), projector(_actions(seed=4))))

    def test_modality_embedding_is_learnable(self):
        projector = ActionConditionProjector()
        self.assertTrue(projector.modality_embedding.requires_grad)
        self.assertEqual(tuple(projector.modality_embedding.shape), (1, 1, CONDITION_DIM))

    def test_concat_condition_appends_after_text(self):
        text = _text()
        action = torch.randn(2, ACTION_TOKENS, CONDITION_DIM)
        joined = concat_condition(text, action)
        self.assertEqual(tuple(joined.shape), (2, TEXT_TOKENS + ACTION_TOKENS, CONDITION_DIM))
        self.assertTrue(torch.equal(joined[:, :TEXT_TOKENS], text))
        self.assertTrue(torch.equal(joined[:, TEXT_TOKENS:], action))
        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            concat_condition(text, action[:1])


class GeneratorSkeletonTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.generator = LatentGeneratorSkeleton(_StubTransformer())

    def test_forward_shape_matches_the_latent_shape(self):
        out = self.generator(
            _latents(), torch.tensor([500, 500]), _text(), _actions()
        )
        self.assertEqual(tuple(out.shape), (2, *LATENT))

    def test_all_arms_share_one_token_layout(self):
        """Correct, masked and no-condition must differ only in content."""
        text = _text()
        with_action = self.generator.condition(text, _actions())
        without = self.generator.condition(text, None)
        self.assertEqual(with_action.shape, without.shape)
        self.assertEqual(with_action.shape[1], TEXT_TOKENS + ACTION_TOKENS)

    def test_zero_init_makes_the_correct_and_null_arms_bit_identical(self):
        """The precise form of D-062's zero-init claim, which holds because the layout matches.

        Appending or dropping tokens instead would change the attention denominator, so this equality
        is a property of the shared layout, not of zero-init alone.
        """
        latents, text, timestep = _latents(), _text(), torch.tensor([400, 400])
        correct = self.generator(latents, timestep, text, _actions())
        null = self.generator(latents, timestep, text, None)
        self.assertTrue(torch.equal(correct, null))

    def test_a_trained_projector_makes_the_arms_differ(self):
        """Without this, the equality above could hold for a generator that ignores the condition."""
        generator = LatentGeneratorSkeleton(_StubTransformer(), zero_init=False)
        latents, text, timestep = _latents(), _text(), torch.tensor([400, 400])
        correct = generator(latents, timestep, text, _actions())
        null = generator(latents, timestep, text, None)
        self.assertFalse(torch.equal(correct, null))
        # Shuffling the action across the batch must also change the output -- that is the G2 arm.
        shuffled = generator(latents, timestep, text, _actions().flip(0))
        self.assertFalse(torch.equal(correct, shuffled))

    def test_masked_arm_equals_zeroing_the_action_segment(self):
        """Makes the masked arm free: it is the null segment, not a separate model."""
        generator = LatentGeneratorSkeleton(_StubTransformer(), zero_init=False)
        text = _text()
        zeroed_input = generator.condition(text, torch.zeros(2, ACTION_TOKENS, ACTION_DIM))
        null = generator.condition(text, None)
        self.assertTrue(torch.equal(zeroed_input, null))

    def test_wrong_action_token_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, f"expected {ACTION_TOKENS} action tokens"):
            self.generator.condition(_text(), _actions()[:, :8])

    def test_generator_is_not_on_the_fast_policy_path(self):
        """The policy framework must not import or construct the generator (AGENTS.md 5, D-009)."""
        import inspect

        from starVLA.model.framework.VLM4A import VLA_JEPA

        source = inspect.getsource(VLA_JEPA)
        for name in ("i5_generator", "LatentGeneratorSkeleton", "ActionConditionProjector"):
            self.assertNotIn(name, source, f"the Fast Policy framework references {name}")


#: The pinned Wan 2.2 scheduler config, inline so this test needs no download. The on-disk config is
#: cross-checked against these values when it is present.
SCHEDULER_CONFIG = dict(
    num_train_timesteps=1000,
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    flow_shift=5.0,
    predict_x0=True,
    solver_order=2,
    solver_type="bh2",
    final_sigmas_type="zero",
)


class FlowConventionTest(unittest.TestCase):
    """Pin the noising and target convention against the real scheduler.

    This is the piece S2 deliberately did not hardcode: getting the sign or the interpolation wrong
    would train the generator against a target the sampler cannot integrate, and the failure would
    look like "it just does not learn" rather than like a bug.
    """

    def _scheduler(self):
        from diffusers import UniPCMultistepScheduler

        scheduler = UniPCMultistepScheduler(**SCHEDULER_CONFIG)
        scheduler.set_timesteps(num_inference_steps=8)
        return scheduler

    def test_velocity_target_lets_the_scheduler_recover_the_clean_latents(self):
        scheduler = self._scheduler()
        torch.manual_seed(0)
        latents = torch.randn(1, LATENT[0], 3, 4, 4)
        noise = torch.randn_like(latents)
        for index in (0, 3, 6):
            sigma = scheduler.sigmas[index]
            noisy = (1 - sigma) * latents + sigma * noise
            scheduler._step_index = index
            recovered = scheduler.convert_model_output(
                flow_velocity_target(latents, noise), sample=noisy
            )
            torch.testing.assert_close(recovered, latents, rtol=0, atol=1e-5)

    def test_the_opposite_sign_does_not_recover_them(self):
        """Without this the test above could pass for a target that happens to be near zero."""
        scheduler = self._scheduler()
        torch.manual_seed(0)
        latents = torch.randn(1, LATENT[0], 3, 4, 4)
        noise = torch.randn_like(latents)
        sigma = scheduler.sigmas[3]
        noisy = (1 - sigma) * latents + sigma * noise
        scheduler._step_index = 3
        wrong = scheduler.convert_model_output(latents - noise, sample=noisy)
        self.assertGreater((wrong - latents).abs().max().item(), 1.0)

    def test_timestep_mapping_matches_the_scheduler(self):
        scheduler = self._scheduler()
        sigmas = scheduler.sigmas[: len(scheduler.timesteps)]
        self.assertTrue(torch.equal(timestep_from_sigma(sigmas), scheduler.timesteps.long()))

    def test_timestep_mapping_rejects_an_integer_sigma(self):
        with self.assertRaisesRegex(ValueError, "floating point"):
            timestep_from_sigma(torch.tensor([1]))

    def test_velocity_target_shape_is_checked(self):
        with self.assertRaisesRegex(ValueError, "noise shape"):
            flow_velocity_target(torch.zeros(1, 2, 3, 4, 4), torch.zeros(1, 2, 3, 4, 5))

    @unittest.skipUnless(
        os.path.isdir(os.environ.get("I5_WAN_SCHEDULER_PATH", "")),
        "set I5_WAN_SCHEDULER_PATH to cross-check the on-disk scheduler config",
    )
    def test_on_disk_config_matches_the_inline_one(self):
        from diffusers import UniPCMultistepScheduler

        on_disk = UniPCMultistepScheduler.from_pretrained(os.environ["I5_WAN_SCHEDULER_PATH"])
        for key, value in SCHEDULER_CONFIG.items():
            self.assertEqual(on_disk.config[key], value, f"{key} disagrees with the pinned config")


class SigmaSamplingTest(unittest.TestCase):
    def test_default_is_the_endorsed_shifted_grid(self):
        """D-067: the default draws from the shifted grid, not from a distribution we invented."""
        sigma = sample_sigma(20000, generator=torch.Generator().manual_seed(0))
        grid = wan_training_sigmas()
        self.assertTrue(bool(torch.isin(sigma, grid).all()), "samples must land on the shifted grid")
        self.assertAlmostEqual(float(sigma.median()), 0.834, delta=0.02)
        self.assertGreater(float((sigma > 0.8).float().mean()), 0.5)

    def test_logit_normal_is_bounded_and_centred(self):
        sigma = sample_sigma(
            20000, distribution="logit_normal", generator=torch.Generator().manual_seed(0)
        )
        self.assertEqual(tuple(sigma.shape), (20000,))
        self.assertGreater(float(sigma.min()), 0.0)
        self.assertLess(float(sigma.max()), 1.0)
        # sigmoid of a standard normal is symmetric about 0.5.
        self.assertAlmostEqual(float(sigma.mean()), 0.5, delta=0.01)

    def test_logit_normal_concentrates_more_than_uniform(self):
        """The reason D-066 picked it: budget lands in the middle of the range, not at the ends."""
        seed = torch.Generator().manual_seed(1)
        logit_normal = sample_sigma(20000, distribution="logit_normal", generator=seed)
        uniform = sample_sigma(20000, distribution="uniform", generator=torch.Generator().manual_seed(1))
        middle = lambda s: float(((s > 0.25) & (s < 0.75)).float().mean())
        self.assertGreater(middle(logit_normal), middle(uniform))
        self.assertAlmostEqual(middle(uniform), 0.5, delta=0.02)

    def test_shift_is_exactly_a_logit_space_mean_shift(self):
        """`flow_shift` is not a separate distribution: it is this family with a shifted mean.

        Pins the identity `logit(shift-applied sigma) = logit(sigma) + log(shift)`, which is what
        makes `aligned_sigma_mean` the principled mean rather than one option among three.
        """
        import math

        shift = 5.0
        z = torch.randn(50000, generator=torch.Generator().manual_seed(0))
        sigma = torch.sigmoid(z)
        by_formula = shift * sigma / (1 + (shift - 1) * sigma)
        by_mean_shift = torch.sigmoid(z + aligned_sigma_mean(shift))
        torch.testing.assert_close(by_formula, by_mean_shift, rtol=0, atol=1e-6)
        self.assertAlmostEqual(aligned_sigma_mean(shift), math.log(shift), places=12)
        with self.assertRaisesRegex(ValueError, "flow_shift must be positive"):
            aligned_sigma_mean(0.0)

    def test_aligned_mean_covers_where_the_sampler_actually_steps(self):
        """The reason to prefer it: the 20-step schedule has median sigma 0.860."""
        centred = sample_sigma(
            50000, distribution="logit_normal", generator=torch.Generator().manual_seed(2)
        )
        aligned = sample_sigma(
            50000,
            distribution="logit_normal",
            generator=torch.Generator().manual_seed(2),
            mean=aligned_sigma_mean(),
        )
        above = lambda s: float((s > 0.8).float().mean())
        self.assertLess(above(centred), 0.15)
        self.assertGreater(above(aligned), 0.5)
        self.assertAlmostEqual(float(aligned.median()), 0.833, delta=0.02)

    def test_shifted_sampling_matches_the_aligned_logit_normal(self):
        """Why "shifted" was never a third family: the two densities nearly coincide."""
        shifted = sample_sigma(50000, generator=torch.Generator().manual_seed(3))
        aligned = sample_sigma(
            50000,
            distribution="logit_normal",
            generator=torch.Generator().manual_seed(3),
            mean=aligned_sigma_mean(),
        )
        self.assertAlmostEqual(float(shifted.median()), float(aligned.median()), delta=0.02)

    def test_bell_weight_pulls_the_effective_mass_back_down(self):
        """The half of the recipe that must not be dropped (D-067)."""
        grid = wan_training_sigmas()
        weights = bell_timestep_weights(grid)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=4)
        self.assertGreaterEqual(float(weights.min()), 0.0)
        sampling = torch.full_like(grid, 1.0 / grid.numel())
        effective = sampling * weights
        effective = effective / effective.sum()
        high_sampling = float(sampling[grid > 0.8].sum())
        high_effective = float(effective[grid > 0.8].sum())
        self.assertAlmostEqual(high_sampling, 0.556, delta=0.02)
        self.assertAlmostEqual(high_effective, 0.280, delta=0.02)
        self.assertLess(high_effective, high_sampling)

    def test_sampling_is_reproducible_and_validated(self):
        first = sample_sigma(16, generator=torch.Generator().manual_seed(7))
        second = sample_sigma(16, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(first, second))
        with self.assertRaisesRegex(ValueError, "unknown sigma distribution"):
            sample_sigma(4, distribution="beta")


class FlowLossTest(unittest.TestCase):
    def test_loss_ignores_the_clean_conditioning_frame(self):
        """Scoring the handed-over frame would reward copying an unnoised input."""
        torch.manual_seed(0)
        latents, noise = _latents(), _latents(seed=9)
        target = flow_velocity_target(latents, noise)
        prediction = target.clone()
        # Make the clean frame's prediction arbitrarily wrong: the loss must not notice.
        prediction[:, :, 0] += 100.0
        loss, _, _ = flow_matching_loss(prediction, latents, noise)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        # And a wrong future frame must be noticed, so the check above is not vacuous.
        prediction[:, :, 1] += 3.0
        worse, _, _ = flow_matching_loss(prediction, latents, noise)
        self.assertGreater(float(worse), 1.0)

    def test_loss_returns_unreduced_error_for_per_frame_logging(self):
        latents, noise = _latents(), _latents(seed=9)
        loss, _, squared = flow_matching_loss(
            flow_velocity_target(latents, noise) + 1.0, latents, noise
        )
        self.assertEqual(squared.shape, latents.shape)
        self.assertAlmostEqual(float(loss), 1.0, places=5)
        # Per-frame breakdown is available, including for the excluded frame.
        self.assertEqual(tuple(squared.mean(dim=(0, 1, 3, 4)).shape), (LATENT[1],))

    def test_raw_and_weighted_are_returned_separately(self):
        """AGENTS section 10 wants both; a weighted loss alone is not comparable across sigmas."""
        latents, noise = _latents(), _latents(seed=9)
        prediction = flow_velocity_target(latents, noise) + 1.0
        raw, weighted, _ = flow_matching_loss(
            prediction, latents, noise, weight=torch.tensor([0.5, 1.5])
        )
        self.assertAlmostEqual(float(raw), 1.0, places=5)
        self.assertAlmostEqual(float(weighted), 1.0, places=5)
        raw2, weighted2, _ = flow_matching_loss(
            prediction, latents, noise, weight=torch.tensor([0.25, 0.25])
        )
        self.assertAlmostEqual(float(raw2), 1.0, places=5)
        self.assertAlmostEqual(float(weighted2), 0.25, places=5)
        with self.assertRaisesRegex(ValueError, "one weight per batch element"):
            flow_matching_loss(prediction, latents, noise, weight=torch.tensor([1.0]))

    def test_shape_mismatch_is_rejected(self):
        latents, noise = _latents(), _latents(seed=9)
        with self.assertRaisesRegex(ValueError, "prediction shape"):
            flow_matching_loss(torch.zeros(2, 48, 3, 8, 8), latents, noise)


#: Set to a DiffSynth-Studio checkout to compare the recipe against its source implementation.
DIFFSYNTH_PATH = os.environ.get("I5_DIFFSYNTH_PATH")


@unittest.skipUnless(
    DIFFSYNTH_PATH and os.path.isfile(
        os.path.join(DIFFSYNTH_PATH, "diffsynth/diffusion/flow_match.py")
    ),
    "set I5_DIFFSYNTH_PATH to compare against DiffSynth's own scheduler",
)
class DiffSynthRecipeParityTest(unittest.TestCase):
    """Our reimplementation against the source it was adopted from (D-067).

    Wan 2.2's own README points at DiffSynth-Studio for training, so this is the endorsed recipe and
    the reimplementation has to match it exactly rather than approximately. The module is loaded by
    file path, not imported as a package: importing `diffsynth` pulls in deepspeed, which wants nvcc.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ds_flow_match", os.path.join(DIFFSYNTH_PATH, "diffsynth/diffusion/flow_match.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scheduler = module.FlowMatchScheduler()
        scheduler.set_timesteps_fn = module.FlowMatchScheduler.set_timesteps_wan
        scheduler.num_train_timesteps = 1000
        scheduler.set_timesteps(1000, training=True)
        cls.scheduler = scheduler

    def test_sigma_grid_is_bit_identical(self):
        self.assertTrue(torch.equal(self.scheduler.sigmas, wan_training_sigmas()))

    def test_loss_weights_are_bit_identical(self):
        ours = bell_timestep_weights(wan_training_sigmas())
        self.assertTrue(torch.equal(self.scheduler.linear_timesteps_weights, ours))

    def test_velocity_target_agrees(self):
        sample = torch.randn(2, 4)
        noise = torch.randn_like(sample)
        self.assertTrue(
            torch.equal(
                self.scheduler.training_target(sample, noise, None),
                flow_velocity_target(sample, noise),
            )
        )


def _real_dit(**overrides):
    """A tiny instance of the **real** `WanTransformer3DModel`.

    No checkpoint: the architecture is what has to be validated, and two layers at width 64 is 0.44M
    parameters. This exercises the real per-token timestep branch (`temb.ndim == 4`), the real
    attention processor, and the real patchify/unpatchify -- none of which a stub can vouch for.
    """
    from diffusers import WanTransformer3DModel

    config = dict(
        patch_size=list(PATCH),
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=LATENT[0],
        out_channels=LATENT[0],
        text_dim=CONDITION_DIM,
        freq_dim=256,
        ffn_dim=128,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
    )
    config.update(overrides)
    return WanTransformer3DModel(**config)


def _sample(output):
    return output.sample if hasattr(output, "sample") else output


class RealWanTransformerTest(unittest.TestCase):
    """The same contract, against the real transformer class rather than a stub."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.generator = LatentGeneratorSkeleton(_real_dit())

    def test_segmented_timesteps_and_longer_condition_are_accepted(self):
        latents, text = _latents(), _text()
        out = _sample(self.generator(latents, torch.tensor([500, 500]), text, _actions()))
        # The real model unpatchifies, so the prediction is shaped like the cached latents.
        self.assertEqual(tuple(out.shape), tuple(latents.shape))

    def test_zero_init_makes_the_arms_bit_identical_on_the_real_model(self):
        latents, text, timestep = _latents(), _text(), torch.tensor([400, 400])
        correct = _sample(self.generator(latents, timestep, text, _actions()))
        null = _sample(self.generator(latents, timestep, text, None))
        self.assertTrue(torch.equal(correct, null))

    def test_a_trained_projector_separates_the_arms_on_the_real_model(self):
        generator = LatentGeneratorSkeleton(_real_dit(), zero_init=False)
        latents, text, timestep = _latents(), _text(), torch.tensor([400, 400])
        correct = _sample(generator(latents, timestep, text, _actions()))
        null = _sample(generator(latents, timestep, text, None))
        shuffled = _sample(generator(latents, timestep, text, _actions().flip(0)))
        self.assertFalse(torch.equal(correct, null))
        self.assertFalse(torch.equal(correct, shuffled))


if __name__ == "__main__":
    unittest.main()
