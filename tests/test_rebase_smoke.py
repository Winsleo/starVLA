"""Functional smoke tests for the upstream-rebase port of VLA_JEPA.

Scope: this is NOT a bit-wise parity gate. Checkpoint compatibility with the pre-rebase
VLA-JEPA lineage is explicitly not required (D-056) -- base behavior changes on purpose (adopted
upstream dispatch API, config-compat helpers, etc.). What this module verifies instead, per
docs/plans/upstream-rebase-experiment.md §5, is that the ported model is well-formed: it
constructs through the adopted registry path, produces named/finite losses, respects the
action-free and future-substitution information firewalls, keeps the frozen teacher frozen, and
keeps the Fast Policy graph free of world-model modules -- the same invariants
VLA-JEPA/tests/test_i1_smoke.py checks on the original, adapted here to run against this rebased
package instead.

Requires one visible GPU and the local checkpoints; skipped otherwise.
Run:  CUDA_VISIBLE_DEVICES=0 pytest tests/test_rebase_smoke.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from parity_probe import (
    BATCH_SIZE,
    CONFIG_PATH,
    NUM_VIEWS,
    PUBLISHED_LIBERO_CKPT,
    SEED,
    SHORT_RUN_STEPS,
    make_examples,
    probe_short_run,
    seeded_forward,
    split_losses,
)


def _require_gpu_and_weights(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    for key in (cfg.framework.qwenvl.base_vlm, cfg.framework.vj2_model.base_encoder):
        if not Path(key).exists():
            pytest.skip(f"missing local weights: {key}")


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def model(cfg):
    _require_gpu_and_weights(cfg)
    # build_framework(cfg), not VLA_JEPA(cfg) directly: exercises the adopted
    # _auto_import_framework_modules registry path end-to-end, per the scoping doc's
    # explicit "needs an actual build_framework(cfg) call" flag.
    from starVLA.model.framework.base_framework import build_framework

    torch.manual_seed(SEED)
    model = build_framework(cfg).to("cuda")
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# construction / registry
# --------------------------------------------------------------------------------------


def test_build_framework_resolves_vla_jepa(model):
    """The adopted auto-import scanner must resolve framework.name == 'VLA_JEPA' to this class."""
    assert type(model).__name__ == "VLA_JEPA"
    assert type(model).__module__ == "starVLA.model.framework.VLM4A.VLA_JEPA"


# --------------------------------------------------------------------------------------
# forward / loss contract
# --------------------------------------------------------------------------------------


def test_forward_losses_are_named_and_finite(model, cfg):
    out = seeded_forward(model, make_examples(cfg, video_seed=1))
    losses, metrics = split_losses(out)
    assert set(losses) == {"action_loss", "wm_loss"}
    for name, value in out.items():
        assert value.ndim == 0
        assert torch.isfinite(value), f"{name} is not finite: {value}"
        print(f"\n[{'metric' if name in metrics else 'loss'}] {name} = {value.item():.6f}")


def test_action_free_batch_masks_only_action_loss(model, cfg):
    """AGENTS §7 / D-012: an action-free sample must drop the action loss, not zero-pad actions."""
    assert "video_data" not in cfg.datasets, "fixture no longer exercises the robot-only config"
    examples = make_examples(cfg, video_seed=1)
    for e in examples:
        e.pop("action")
        e.pop("state")
    out = seeded_forward(model, examples)

    losses, _ = split_losses(out)
    assert set(losses) == {"wm_loss"}
    assert torch.isfinite(out["wm_loss"])


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------


def test_forward_is_deterministic_under_seed(model, cfg):
    examples = make_examples(cfg, video_seed=1)
    first = seeded_forward(model, examples)
    second = seeded_forward(model, examples)
    for name in first:
        assert torch.equal(first[name], second[name]), (
            f"{name} not bit-wise reproducible: {first[name].item()} vs {second[name].item()}"
        )


# --------------------------------------------------------------------------------------
# information firewall
# --------------------------------------------------------------------------------------


def test_future_substitution_does_not_move_action_loss(model, cfg):
    """Different future clip, identical deployment inputs => identical action loss, different wm loss."""
    a = seeded_forward(model, make_examples(cfg, video_seed=1))
    b = seeded_forward(model, make_examples(cfg, video_seed=999))
    assert torch.equal(a["action_loss"], b["action_loss"]), (
        f"action loss leaked future frames: {a['action_loss'].item()} vs {b['action_loss'].item()}"
    )
    assert not torch.equal(a["wm_loss"], b["wm_loss"]), "world loss ignored the future clip"


def test_fast_policy_graph_excludes_world_modules(model, cfg):
    """predict_action must not touch the V-JEPA encoder or the world predictor."""
    examples = make_examples(cfg, video_seed=1)

    calls = []

    def tripwire(name):
        def hook(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"Fast Policy path ran {name}")

        return hook

    enc_handle = model.vj_encoder.register_forward_pre_hook(tripwire("vj_encoder"))
    pred_handle = model.vj_predictor.register_forward_pre_hook(tripwire("vj_predictor"))
    try:
        out = model.predict_action(
            batch_images=[e["image"] for e in examples],
            instructions=[e["lang"] for e in examples],
            state=[e["state"] for e in examples],
        )
    finally:
        enc_handle.remove()
        pred_handle.remove()

    assert calls == []
    actions = out["normalized_actions"]
    chunk = cfg.framework.action_model.future_action_window_size + 1
    assert actions.shape == (BATCH_SIZE, chunk, cfg.framework.action_model.action_dim), actions.shape
    assert np.isfinite(actions).all()


# --------------------------------------------------------------------------------------
# gradient firewall (AGENTS §6)
# --------------------------------------------------------------------------------------


def test_target_encoder_receives_no_gradient(model, cfg):
    model.zero_grad(set_to_none=True)
    out = seeded_forward(model, make_examples(cfg, video_seed=1))
    (out["wm_loss"] + out["action_loss"]).backward()
    grads = [n for n, p in model.vj_encoder.named_parameters() if p.grad is not None]
    assert grads == [], f"gradient reached the frozen teacher: {grads[:5]}"
    model.zero_grad(set_to_none=True)


def test_target_encoder_params_are_frozen(model):
    trainable = [n for n, p in model.vj_encoder.named_parameters() if p.requires_grad]
    assert trainable == [], f"{len(trainable)} teacher params are trainable, e.g. {trainable[:3]}"


def test_target_encoder_stays_in_eval_after_parent_train(model):
    model.train()
    try:
        assert not model.vj_encoder.training, "teacher switched to train() with the parent"
    finally:
        model.eval()


# --------------------------------------------------------------------------------------
# checkpoint -- structural sanity only, NOT a compatibility claim (D-056)
# --------------------------------------------------------------------------------------


def test_state_dict_is_well_formed(model):
    """state_dict() succeeds and has a plausible, non-degenerate key set.

    Deliberately not a strict-load test against the pre-rebase published checkpoint: checkpoint
    compatibility with that lineage is explicitly not required for this experiment (D-056). This
    only guards against an accidentally empty/duplicate/crashing state_dict from the port itself.
    """
    state = model.state_dict()
    assert len(state) > 0
    assert len(state) == len(set(state)), "duplicate state_dict keys"
    assert any(k.startswith("vj_encoder.") for k in state)
    assert any(k.startswith("vj_predictor.") for k in state)
    assert any(k.startswith("qwen_vl_interface.") for k in state)
    assert any(k.startswith("action_model.") for k in state)
    print(f"\n[state_dict] {len(state)} keys, {sum(p.numel() for p in model.parameters())} parameters")


def test_published_checkpoint_load_is_reported_not_assumed(model):
    """Informational only: records whether the old lineage's checkpoint happens to strict-load.

    Not asserted either way -- a pass would be a bonus, a failure is expected and not a defect of
    this port, since D-056 does not require it. xfail(strict=False) keeps this visible in the
    report without turning a expected mismatch into a red build.
    """
    if not PUBLISHED_LIBERO_CKPT.exists():
        pytest.skip(f"missing published checkpoint: {PUBLISHED_LIBERO_CKPT}")
    checkpoint = torch.load(PUBLISHED_LIBERO_CKPT, map_location="cpu")
    state = model.state_dict()
    missing = sorted(set(state) - set(checkpoint))
    unexpected = sorted(set(checkpoint) - set(state))
    mismatched = [k for k in set(state) & set(checkpoint) if state[k].shape != checkpoint[k].shape]
    print(
        f"\n[ckpt, informational] missing={len(missing)} unexpected={len(unexpected)} "
        f"mismatched={len(mismatched)} (not a gate -- D-056 does not require checkpoint compat)"
    )
    del checkpoint


# --------------------------------------------------------------------------------------
# short real training run (scoping doc §5) -- single-process AdamW steps on a synthetic batch,
# via parity_probe.probe_short_run, mirroring train_starvla.py::_train_step's optimization
# semantics without needing accelerate/DeepSpeed process-group setup or a real dataset on disk.
# Not required to match the pre-rebase lineage's numbers (D-056) -- only non-degenerate.
# --------------------------------------------------------------------------------------


def test_short_training_run_is_non_degenerate(model, cfg):
    model.train()
    try:
        result = probe_short_run(model, cfg)
    finally:
        model.eval()

    assert result["optimizer_steps"] == SHORT_RUN_STEPS
    assert result["scheduler_last_epoch"] == SHORT_RUN_STEPS
    losses_by_step = [
        {name: float.fromhex(value) for name, value in step["losses"].items()} for step in result["step_losses"]
    ]
    for step_idx, losses in enumerate(losses_by_step):
        for name, value in losses.items():
            assert value == value and abs(value) != float("inf"), f"step {step_idx} {name} not finite: {value}"

    teacher = result["modules"].get("vj_encoder")
    assert teacher is not None
    grads = [n for n, p in model.vj_encoder.named_parameters() if p.grad is not None]
    assert grads == [], f"gradient reached the frozen teacher after training steps: {grads[:5]}"

    print(
        f"\n[short_run] {SHORT_RUN_STEPS} optimizer steps, last_lr={result['last_lr']}, "
        f"action_loss@0={losses_by_step[0].get('action_loss')}, "
        f"action_loss@last={losses_by_step[-1].get('action_loss')}"
    )


# --------------------------------------------------------------------------------------
# frozen teacher must stay out of the optimizer entirely, not just receive no gradient
# (docs/plans/upstream-rebase-experiment.md's matched-condition comparison found upstream's
# rewrite of trainer_tools.py dropped VLA-JEPA's explicit exclusion of requires_grad=False params
# from every LR group -- under DeepSpeed ZeRO this risks weight-decay drift on a param with no
# gradient, not just a wasted no-op the way it is under plain torch.optim.AdamW).
# --------------------------------------------------------------------------------------


def test_frozen_teacher_excluded_from_every_optimizer_group(model, cfg):
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    groups = build_param_lr_groups(model=model, cfg=cfg)
    teacher_ids = {id(p) for p in model.vj_encoder.parameters()}
    for group in groups:
        hit = [p for p in group["params"] if id(p) in teacher_ids]
        assert not hit, f"{len(hit)} frozen vj_encoder tensors leaked into optimizer group {group['name']!r}"
