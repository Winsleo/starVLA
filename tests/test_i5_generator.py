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

import unittest

import torch
import torch.nn as nn

from starVLA.model.modules.world_model.i5_generator import (
    ACTION_DIM,
    ACTION_TOKENS,
    CONDITION_DIM,
    ActionConditionProjector,
    LatentGeneratorSkeleton,
    apply_partial_noise,
    clean_frame_mask,
    concat_condition,
    segmented_timesteps,
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
