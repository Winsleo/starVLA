"""Gradient-firewall tests for the WM4A world-model wrappers.

`AGENTS.md` section 6 requires frozen target encoders to be `requires_grad=False`, in `eval()`,
and to stay in eval after the parent model's `train()`. Both wrappers under
`starVLA/model/modules/world_model/` hold two such encoders -- a VAE (observation/target ->
video latents) and a text encoder (language condition) -- while only the DiT transformer is a
trainable generator backbone.

What this pins, and why each part is not cosmetic:

  1. `build_param_lr_groups` excludes only params with `requires_grad=False`, so an unfrozen
     encoder lands in the optimizer's "base" group. Under ZeRO that allocates gradient buffers
     for it and AdamW's decoupled weight decay then updates weights that receive no gradient,
     so a nominally frozen tokenizer drifts during training, silently and without error. The
     same failure was measured once already in this repository for the V-JEPA teacher; see the
     comment in `starVLA/training/trainer_utils/trainer_tools.py`.
  2. `no_grad()` at a call site protects that call site only. Target encoding and
     decode-for-eval both still have to be written for I5, and a missing `no_grad()` at a new
     call site would let gradient reach the target, at which point collapsing the latent space
     becomes a valid way to lower the loss. A module-level freeze makes the guarantee
     independent of whoever writes the next call site.
  3. eval mode has to survive `train()`. `Wan2.py` originally had the freeze calls commented
     out and neither wrapper called `eval()` at all.

The diffusers/transformers loaders are patched, so this runs on CPU with no checkpoint: neither
the 34 GB of Wan weights nor the gated Cosmos weights are needed to verify the firewall.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn
from omegaconf import OmegaConf


HIDDEN_HEADS = 4
HIDDEN_HEAD_DIM = 8


class _StubVAE(nn.Module):
    """Stands in for AutoencoderKLWan. `temperal_downsample` drives the scale factors."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=1)
        self.dropout = nn.Dropout(0.5)  # makes train/eval mode observable
        self.temperal_downsample = [False, True, True]


class _StubTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.dropout = nn.Dropout(0.5)


class _StubBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(HIDDEN_HEADS * HIDDEN_HEAD_DIM, HIDDEN_HEADS * HIDDEN_HEAD_DIM)


class _StubTransformer(nn.Module):
    """Stands in for the DiT. The block-list attribute name differs per wrapper."""

    def __init__(self, blocks_attr: str) -> None:
        super().__init__()
        setattr(self, blocks_attr, nn.ModuleList([_StubBlock() for _ in range(2)]))
        self.config = OmegaConf.create(
            {"num_attention_heads": HIDDEN_HEADS, "attention_head_dim": HIDDEN_HEAD_DIM}
        )


def _wrapper_config() -> OmegaConf:
    return OmegaConf.create({"framework": {"world_model": {"base_wm": "stub://wm"}}})


class _FirewallContractMixin:
    """The invariant, asserted identically for every wrapper.

    Subclasses provide `_build()` returning a constructed wrapper with stubbed loaders.
    """

    FROZEN_ATTRS = ("vae", "text_encoder")

    def _build(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def setUp(self) -> None:
        self.wrapper = self._build()

    def _assert_frozen(self, where: str) -> None:
        for name in self.FROZEN_ATTRS:
            module = getattr(self.wrapper, name)
            trainable = [n for n, p in module.named_parameters() if p.requires_grad]
            self.assertEqual(trainable, [], f"{name} has trainable params {where}")
            for child_name, child in module.named_modules():
                label = f"{name}.{child_name}" if child_name else name
                self.assertFalse(child.training, f"{label} is in train mode {where}")

    def test_frozen_after_construction(self):
        self._assert_frozen("after construction")

    def test_generator_backbone_stays_trainable(self):
        # The firewall must not over-freeze: the DiT is the generator, not a teacher.
        trainable = [n for n, p in self.wrapper.transformer.named_parameters() if p.requires_grad]
        self.assertNotEqual(trainable, [], "transformer must remain trainable")

    def test_frozen_survives_parent_train(self):
        self.wrapper.train()
        self._assert_frozen("after wrapper.train()")
        self.assertTrue(self.wrapper.transformer.training, "transformer should follow train mode")

    def test_frozen_survives_train_through_module_recursion(self):
        # The real path: a framework owns the wrapper and calls train() on itself, which
        # recurses into children. The override has to be reached that way, not only when
        # called directly.
        parent = nn.Module()
        parent.backbone = self.wrapper
        parent.head = nn.Linear(4, 4)
        parent.train()

        self._assert_frozen("after parent.train() recursion")
        self.assertTrue(parent.head.training)
        self.assertTrue(self.wrapper.transformer.training)

    def test_eval_then_train_round_trip(self):
        self.wrapper.eval()
        self._assert_frozen("after wrapper.eval()")
        self.wrapper.train(True)
        self._assert_frozen("after eval() then train(True)")

    def test_frozen_encoders_stay_out_of_the_optimizer(self):
        """The consequence that made this a real bug rather than a style issue."""
        from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

        cfg = OmegaConf.create({"trainer": {"learning_rate": {"base": 1e-4}}})
        groups = build_param_lr_groups(self.wrapper, cfg)

        grouped_ids = {id(p) for group in groups for p in group["params"]}
        for name in self.FROZEN_ATTRS:
            module = getattr(self.wrapper, name)
            leaked = [n for n, p in module.named_parameters() if id(p) in grouped_ids]
            self.assertEqual(leaked, [], f"{name} params reached the optimizer: {leaked}")

        # The trainable backbone must still be optimized, otherwise nothing trains.
        dit_ids = {id(p) for p in self.wrapper.transformer.parameters()}
        self.assertTrue(dit_ids & grouped_ids, "transformer params never reached the optimizer")

    def test_no_grad_reaches_frozen_encoder_outputs(self):
        """A forward through the frozen VAE must not build a graph back to its weights."""
        self.wrapper.train()
        out = self.wrapper.vae.conv(torch.randn(1, 3, 2, 4, 4))
        self.assertFalse(out.requires_grad, "frozen VAE output still carries grad_fn")


class Wan2GradientFirewallTest(_FirewallContractMixin, unittest.TestCase):
    def _build(self):
        from starVLA.model.modules.world_model.Wan2 import _Wan2_Interface

        vae, text_encoder = _StubVAE(), _StubTextEncoder()
        transformer = _StubTransformer("blocks")
        with (
            mock.patch("diffusers.AutoencoderKLWan") as vae_cls,
            mock.patch("diffusers.WanTransformer3DModel") as dit_cls,
            mock.patch("diffusers.UniPCMultistepScheduler") as sched_cls,
            mock.patch("diffusers.video_processor.VideoProcessor") as proc_cls,
            mock.patch("transformers.T5TokenizerFast") as tok_cls,
            mock.patch("transformers.UMT5EncoderModel") as text_cls,
        ):
            vae_cls.from_pretrained.return_value = vae
            dit_cls.from_pretrained.return_value = transformer
            text_cls.from_pretrained.return_value = text_encoder
            sched_cls.from_pretrained.return_value = object()
            tok_cls.from_pretrained.return_value = object()
            proc_cls.return_value = object()
            return _Wan2_Interface(config=_wrapper_config())


class CosmoPredict2GradientFirewallTest(_FirewallContractMixin, unittest.TestCase):
    def _build(self):
        from starVLA.model.modules.world_model.CosmoPredict2 import _CosmoPredict2_Interface

        vae, text_encoder = _StubVAE(), _StubTextEncoder()
        # This wrapper reads `transformer.transformer_blocks`, not `.blocks`.
        transformer = _StubTransformer("transformer_blocks")
        with (
            mock.patch("diffusers.AutoencoderKLWan") as vae_cls,
            mock.patch("diffusers.CosmosTransformer3DModel") as dit_cls,
            mock.patch("diffusers.FlowMatchEulerDiscreteScheduler") as sched_cls,
            mock.patch("diffusers.video_processor.VideoProcessor") as proc_cls,
            mock.patch("transformers.T5TokenizerFast") as tok_cls,
            mock.patch("transformers.T5EncoderModel") as text_cls,
        ):
            vae_cls.from_pretrained.return_value = vae
            dit_cls.from_pretrained.return_value = transformer
            text_cls.from_pretrained.return_value = text_encoder
            sched_cls.from_pretrained.return_value = object()
            tok_cls.from_pretrained.return_value = object()
            proc_cls.return_value = object()
            return _CosmoPredict2_Interface(config=_wrapper_config())


if __name__ == "__main__":
    unittest.main()
