"""Gates for the I5 LoRA wiring (`I5-S3-TF`, D-064).

What has to hold before a training run is trusted:

* adapters land on the attention and feed-forward layers of every transformer block, resolved from
  the real module tree rather than a hardcoded list;
* the timestep and text embedders and the output head are **not** adapted -- adapting the timestep
  path would let the model relearn the noise schedule instead of the dynamics, and the output head is
  pinned by the flow target (D-067);
* **only** LoRA factors carry gradient; a run that silently starts updating the 5B backbone would
  still show a falling loss, so it would not announce itself;
* the adapted model starts numerically **identical** to the base model, which is what makes the
  no-condition and correct-condition arms coincide at step 0 (D-062).

Everything runs against a small instance of the real `WanTransformer3DModel`, so no checkpoint is
needed and the checks are about the real class, not a stub.
"""

from __future__ import annotations

import unittest

import torch

from starVLA.model.modules.world_model.i5_generator import (
    ACTION_DIM,
    ACTION_TOKENS,
    CONDITION_DIM,
    LatentGeneratorSkeleton,
)
from starVLA.model.modules.world_model.i5_lora import (
    WAN_LORA_TARGETS,
    assert_only_expected_parameters_train,
    attach_lora,
    trainable_parameter_summary,
    wan_lora_target_names,
)

PATCH = (1, 2, 2)
LATENT_CHANNELS = 48
NUM_LAYERS = 2


def _dit(num_layers: int = NUM_LAYERS):
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel(
        patch_size=list(PATCH),
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        text_dim=CONDITION_DIM,
        freq_dim=256,
        ffn_dim=128,
        num_layers=num_layers,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
    )


def _inputs(batch: int = 1):
    torch.manual_seed(0)
    return dict(
        hidden_states=torch.randn(batch, LATENT_CHANNELS, 3, 16, 16),
        timestep=torch.full((batch, 192), 500, dtype=torch.long),
        encoder_hidden_states=torch.randn(batch, 12, CONDITION_DIM),
    )


def _sample(output):
    if hasattr(output, "sample"):
        return output.sample
    return output[0] if isinstance(output, tuple) else output


class LoraTargetTest(unittest.TestCase):
    def test_targets_cover_every_block_and_nothing_else(self):
        names = wan_lora_target_names(_dit())
        self.assertEqual(len(names), NUM_LAYERS * len(WAN_LORA_TARGETS))
        for layer in range(NUM_LAYERS):
            for suffix in WAN_LORA_TARGETS:
                self.assertIn(f"blocks.{layer}.{suffix}", names)

    def test_embedders_and_output_head_are_not_targets(self):
        names = wan_lora_target_names(_dit())
        for excluded in ("condition_embedder", "proj_out", "time_embedder", "text_embedder"):
            self.assertFalse(
                any(excluded in name for name in names), f"{excluded} must not be adapted"
            )

    def test_a_changed_block_layout_is_reported(self):
        dit = _dit()
        dit.blocks = torch.nn.ModuleList()  # simulate a renamed or restructured block list
        with self.assertRaisesRegex(ValueError, "no LoRA targets found"):
            wan_lora_target_names(dit)


class LoraAttachmentTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.base = _dit()
        self.inputs = _inputs()
        with torch.no_grad():
            self.base_output = _sample(self.base(**self.inputs)).clone()

    def test_only_lora_factors_are_trainable(self):
        adapted = attach_lora(self.base, rank=8)
        trainable = assert_only_expected_parameters_train(adapted)
        self.assertNotEqual(trainable, [])
        self.assertTrue(all("lora_" in name for name in trainable))

    def test_adapted_model_starts_identical_to_the_base(self):
        """peft zero-initialises lora_B, so the adapter is an exact no-op at step 0."""
        adapted = attach_lora(self.base, rank=8)
        with torch.no_grad():
            output = _sample(adapted(**self.inputs))
        self.assertTrue(torch.equal(output, self.base_output))

    def test_a_nonzero_adapter_does_change_the_output(self):
        """Without this, the identity above could hold for adapters wired to nothing."""
        adapted = attach_lora(self.base, rank=8)
        changed = 0
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_B" in name:
                    parameter.add_(0.1)
                    changed += 1
        self.assertGreater(changed, 0, "no lora_B factors found")
        with torch.no_grad():
            output = _sample(adapted(**self.inputs))
        self.assertFalse(torch.equal(output, self.base_output))

    def test_parameter_summary_separates_lora_from_the_rest(self):
        adapted = attach_lora(self.base, rank=8)
        summary = trainable_parameter_summary(adapted)
        self.assertGreater(summary["trainable_lora"], 0)
        self.assertEqual(summary["trainable_other"], 0)
        self.assertEqual(summary["trainable"], summary["trainable_lora"])
        # The adapter is a small fraction of the model, which is the point of using one.
        self.assertLess(summary["trainable"] / summary["total"], 0.2)

    def test_guard_rejects_an_unfrozen_backbone(self):
        adapted = attach_lora(self.base, rank=8)
        # Simulate the failure this guard exists for: something re-enables a base weight.
        for name, parameter in adapted.named_parameters():
            if "lora_" not in name:
                parameter.requires_grad_(True)
                break
        with self.assertRaisesRegex(ValueError, "unexpected trainable parameter"):
            assert_only_expected_parameters_train(adapted)


class GeneratorWithLoraTest(unittest.TestCase):
    """The projector trains alongside the adapters, and both start as no-ops."""

    def test_generator_trainables_are_lora_plus_projector(self):
        torch.manual_seed(0)
        generator = LatentGeneratorSkeleton(attach_lora(_dit(), rank=8))
        trainable = assert_only_expected_parameters_train(
            generator, allowed_substrings=("lora_", "projector.")
        )
        self.assertTrue(any("projector." in name for name in trainable))
        self.assertTrue(any("lora_" in name for name in trainable))

    def test_action_condition_is_still_a_no_op_at_step_zero(self):
        torch.manual_seed(0)
        generator = LatentGeneratorSkeleton(attach_lora(_dit(), rank=8))
        latents = torch.randn(1, LATENT_CHANNELS, 3, 16, 16)
        text = torch.randn(1, 12, CONDITION_DIM)
        actions = torch.randn(1, ACTION_TOKENS, ACTION_DIM)
        timestep = torch.tensor([400])
        with torch.no_grad():
            correct = _sample(generator(latents, timestep, text, actions))
            null = _sample(generator(latents, timestep, text, None))
        self.assertTrue(torch.equal(correct, null))


if __name__ == "__main__":
    unittest.main()
