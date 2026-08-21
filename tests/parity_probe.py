"""Deterministic probes of the VLA-JEPA training path, shared by the I2 parity harness.

`I2 Interface Parity` may only change interfaces, never numbers (DynaWeave AGENTS.md §9,
D-011). This module produces one JSON-serialisable dict that captures everything the parity
gate compares:

  - the two named losses of `VLA_JEPA.forward()` on a fixed synthetic batch
  - the Fast Policy output of `VLA_JEPA.predict_action()`
  - the measured world-path geometry (grid, temporal blocks, tokens per block)
  - the `state_dict` key set (checkpoint compatibility) and parameter counts
  - a short single-process AdamW run: per-module parameter fingerprints, per-step losses,
    optimizer/scheduler step counts

Floats are stored via `float.hex()`, which round-trips exactly, so the comparison is bit-wise
rather than "close enough" (the I2 tolerance decision, D-039).

The probe deliberately starts from the published LIBERO checkpoint instead of a freshly
initialised model: loading fixed weights removes any dependence on how much RNG module
construction happens to consume, so a parity failure means a real behavioural change.

Single process on purpose. Multi-GPU DeepSpeed runs are not bit-wise reproducible (D-034) and
are compared separately with a tolerance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "i1_libero_local.yaml"
PUBLISHED_LIBERO_CKPT = Path(
    "/vepfs/wangshilong/models/dynaweave/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
)

# Batch geometry. Values follow the published LIBERO config and are asserted against it in
# tests/test_i1_smoke.py, so a config change cannot silently invalidate the fixtures.
BATCH_SIZE = 2
NUM_VIEWS = 2
SEED = 1234

# Video seeds of the two probe batches. The pair doubles as the future-substitution fixture:
# identical deployment inputs, different future clip.
PROBE_VIDEO_SEEDS = (1, 999)

# Steps of the short optimisation run. Five is enough for the optimizer state to become
# non-trivial (AdamW bias correction, scheduler warmup) while keeping the probe under a minute.
SHORT_RUN_STEPS = 5

# Modules whose parameter fingerprints are compared after the short run. `vj_encoder` is the
# frozen V-JEPA teacher and must not move at all (AGENTS.md §6).
PROBE_MODULES = ("qwen_vl_interface", "action_model", "vj_predictor", "vj_encoder")


def make_examples(cfg, video_seed: int, batch_size: int = BATCH_SIZE):
    """Synthetic batch matching the dataloader contract of VLA_JEPA.forward().

    image  : list[PIL.Image]                       per view, obs resolution
    video  : np.ndarray [V, T, H, W, 3] uint8      per view clip, video resolution
    action : np.ndarray [T_chunk, action_dim]
    state  : np.ndarray [1, state_dim]
    """
    obs = cfg.datasets.vla_data.resolution_size
    vid = cfg.datasets.vla_data.video_resolution_size
    frames = cfg.framework.vj2_model.num_frames
    action_dim = cfg.framework.action_model.action_dim
    state_dim = cfg.framework.action_model.state_dim
    chunk = cfg.framework.action_model.future_action_window_size + 1

    img_rng = np.random.default_rng(SEED)
    vid_rng = np.random.default_rng(video_seed)
    examples = []
    for _ in range(batch_size):
        image = Image.fromarray(img_rng.integers(0, 255, (obs, obs, 3), dtype=np.uint8))
        examples.append(
            {
                "image": [image] * NUM_VIEWS,
                "video": vid_rng.integers(0, 255, (NUM_VIEWS, frames, vid, vid, 3), dtype=np.uint8),
                "lang": "pick up the black bowl and place it on the plate",
                "action": img_rng.uniform(-1, 1, size=(chunk, action_dim)).astype(np.float32),
                "state": img_rng.uniform(-1, 1, size=(1, state_dim)).astype(np.float32),
            }
        )
    return examples


def seeded_forward(model, examples, seed: int = SEED):
    """Seeded forward. The flow-matching head samples noise, so the seed is part of the contract."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return model(examples)


# --------------------------------------------------------------------------------------
# exact serialisation helpers
# --------------------------------------------------------------------------------------

def _hex(value: float) -> str:
    """Lossless float representation. `float.fromhex` recovers the exact same double."""
    return float(value).hex()


def _sha256_of(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _array_fingerprint(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def env_fingerprint() -> dict:
    """Recorded so a golden captured on different hardware is skipped, not reported as failure."""
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# --------------------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------------------

def split_losses(output: dict):
    """Loss terms that enter the backward sum, separated from log-only metric terms.

    C3 of I2 adds `metric/*` keys to the framework output and a shared splitter to
    trainer_tools. The fallback keeps this probe runnable on the pre-C3 revision that generates
    the golden, so the recorded totals stay comparable across the whole iteration.
    """
    try:
        from starVLA.training.trainer_utils.trainer_tools import split_loss_terms
    except ImportError:
        return dict(output), {}
    return split_loss_terms(output)


def load_config():
    from omegaconf import OmegaConf

    return OmegaConf.load(CONFIG_PATH)


def build_probe_model(cfg):
    """Model with the published LIBERO weights, i.e. a fixed starting point for every probe."""
    from starVLA.model.framework.VLM4A.VLA_JEPA import VLA_JEPA

    torch.manual_seed(SEED)
    model = VLA_JEPA(cfg).to("cuda")
    checkpoint = torch.load(PUBLISHED_LIBERO_CKPT, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    model.eval()
    return model


def build_optimizer_and_scheduler(model, cfg):
    """Mirror of train_starvla.py::setup_optimizer_and_scheduler.

    Duplicated rather than imported: importing that module constructs a global `Accelerator`
    at import time. Keep the two in sync; the step-count test below pins the semantics.
    """
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


# --------------------------------------------------------------------------------------
# individual probes
# --------------------------------------------------------------------------------------

def probe_forward(model, cfg) -> dict:
    """Named losses on both probe batches, bit-exact."""
    out = {}
    for video_seed in PROBE_VIDEO_SEEDS:
        output = seeded_forward(model, make_examples(cfg, video_seed=video_seed))
        losses, _ = split_losses(output)
        out[str(video_seed)] = {name: _hex(value.item()) for name, value in losses.items()}
    return out


def probe_loss_split(model, cfg) -> dict:
    """How the framework output splits into optimized losses and log-only metrics (C3).

    The optimized values themselves are compared by `probe_forward`; what is recorded here is the
    partition and the raw/weight/weighted relation, so an added metric key cannot silently end up
    in the backward sum. The equality is evaluated with torch, in whatever dtype the forward
    produced, instead of being recomputed from the serialised hex in double precision.
    """
    output = seeded_forward(model, make_examples(cfg, video_seed=PROBE_VIDEO_SEEDS[0]))
    losses, metrics = split_losses(output)
    weighted_matches = None
    raw, weight = metrics.get("metric/wm_loss_raw"), metrics.get("metric/wm_loss_weight")
    if raw is not None and weight is not None:
        weighted_matches = bool(torch.equal(losses["wm_loss"], raw * weight))
    return {
        "loss_names": sorted(losses),
        "metric_names": sorted(metrics),
        "wm_loss_equals_raw_times_weight": weighted_matches,
    }


def probe_predict_action(model, cfg) -> dict:
    """Fast Policy output. Must not move: I2 does not touch the action head."""
    examples = make_examples(cfg, video_seed=PROBE_VIDEO_SEEDS[0])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    out = model.predict_action(
        batch_images=[e["image"] for e in examples],
        instructions=[e["lang"] for e in examples],
        state=[e["state"] for e in examples],
    )
    return _array_fingerprint(out["normalized_actions"])


def probe_geometry(model, cfg) -> dict:
    """Measured world-path geometry. C2 makes these explicit; the numbers may not change."""
    encoder_cfg = model.vj_encoder.config
    tubelet = encoder_cfg.tubelet_size
    grid = encoder_cfg.image_size // encoder_cfg.patch_size
    num_temporal_blocks = cfg.framework.vj2_model.num_frames // tubelet

    predictor = model.vj_predictor
    return {
        "encoder": {
            "image_size": encoder_cfg.image_size,
            "patch_size": encoder_cfg.patch_size,
            "tubelet_size": tubelet,
            "hidden_size": encoder_cfg.hidden_size,
            "grid": [grid, grid],
            "num_temporal_blocks": num_temporal_blocks,
            "tokens_per_block": grid * grid,
        },
        "predictor": {
            "grid": [predictor.grid_height, predictor.grid_width],
            "num_frames": predictor.num_frames,
            "tubelet_size": predictor.tubelet_size,
            "attn_mask_shape": list(predictor.attn_mask.shape) if predictor.attn_mask is not None else None,
        },
    }


def probe_parameters(model) -> dict:
    """Checkpoint compatibility and the trainable parameter set."""
    state = model.state_dict()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    return {
        "state_dict_num_keys": len(state),
        "state_dict_keys_sha256": _sha256_of(sorted(state)),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "teacher_num_parameters": sum(p.numel() for p in model.vj_encoder.parameters()),
        "num_teacher_trainable_parameters": sum(
            p.numel() for p in model.vj_encoder.parameters() if p.requires_grad
        ),
        "trainable_keys_sha256": _sha256_of(sorted(trainable)),
    }


def probe_param_groups(model, cfg) -> dict:
    """Optimizer param groups as `build_param_lr_groups` builds them.

    Group names and learning rates are parity: I2 may not re-map a module to another lr. The
    parameter count of the `base` group is the one number C1 is allowed to change, because
    freezing the teacher removes it from the optimizer.
    """
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    groups = build_param_lr_groups(model=model, cfg=cfg)
    return {
        "groups": [
            {
                "name": group.get("name"),
                "lr": _hex(group["lr"]),
                "num_tensors": len(group["params"]),
                "numel": sum(p.numel() for p in group["params"]),
            }
            for group in groups
        ]
    }


def probe_short_run(model, cfg) -> dict:
    """`SHORT_RUN_STEPS` real AdamW steps, mirroring train_starvla.py::_train_step.

    One optimizer step and one scheduler step per batch: that is the upstream semantics I2 must
    preserve (D-011). Gradient clipping uses torch directly instead of accelerate because this
    probe runs without accelerate; the clip value is the configured one.
    """
    optimizer, lr_scheduler = build_optimizer_and_scheduler(model, cfg)
    examples = make_examples(cfg, video_seed=PROBE_VIDEO_SEEDS[0])

    model.train()
    step_losses = []
    optimizer_steps = 0
    for step in range(SHORT_RUN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = seeded_forward(model, examples, seed=SEED + step)
            losses, _ = split_losses(output)
            total_loss = sum(losses.values())
        total_loss.backward()
        if cfg.trainer.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
        optimizer.step()
        lr_scheduler.step()
        optimizer_steps += 1
        step_losses.append(
            {
                "losses": {name: _hex(value.item()) for name, value in losses.items()},
                "total": _hex(total_loss.item()),
            }
        )
    model.eval()

    modules = {}
    for name in PROBE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            continue
        stacked = torch.stack([p.detach().to(torch.float64).sum() for p in module.parameters()])
        modules[name] = {
            "param_sum": _hex(stacked.sum().item()),
            "param_abs_sum": _hex(stacked.abs().sum().item()),
            "num_params": sum(p.numel() for p in module.parameters()),
        }

    return {
        "steps": SHORT_RUN_STEPS,
        "optimizer_steps": optimizer_steps,
        "scheduler_last_epoch": lr_scheduler.last_epoch,
        "last_lr": [_hex(lr) for lr in lr_scheduler.get_last_lr()],
        "step_losses": step_losses,
        "modules": modules,
    }


def collect_probe() -> dict:
    """Full golden payload. Requires one visible GPU and the published LIBERO checkpoint."""
    cfg = load_config()
    model = build_probe_model(cfg)
    payload = {
        "env": env_fingerprint(),
        "forward": probe_forward(model, cfg),
        "loss_split": probe_loss_split(model, cfg),
        "predict_action": probe_predict_action(model, cfg),
        "geometry": probe_geometry(model, cfg),
        "parameters": probe_parameters(model),
        "param_groups": probe_param_groups(model, cfg),
        "short_run": probe_short_run(model, cfg),
    }
    return payload
