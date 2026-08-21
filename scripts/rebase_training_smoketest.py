"""Single-process DeepSpeed-launch smoke test for the upstream-rebase port.

Exercises the real production launch mechanism VLA-JEPA's own scripts/vlajepa_cotrain.sh uses
(`accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml`), just with
`--num_processes 1` instead of 8: real torch.distributed + NCCL init, the real ZeRO-2 DeepSpeed
config (not accelerate's bare default), model construction inside that context, `accelerator.
prepare()` wrapping the model as an actual `DeepSpeedEngine`, and a few real
accumulate/backward/clip_grad_norm_/optimizer.step() training steps.

Not exercised here: real dataset loading (see docs/plans/upstream-rebase-experiment.md's
"Execution update 3" -- upstream's Libero4in1DataConfig's default single-timestep `video`
modality does not currently produce the multi-frame clip VLA_JEPA.forward() expects; a synthetic
batch from parity_probe.make_examples is used instead, deliberately, pending that decision).
Not a checkpoint-compatibility or bit-wise-parity claim (D-056) -- only that the real launch path
constructs, wraps, and trains without error, with finite losses.

Run (single GPU, no multi-GPU required):

    CUDA_HOME=<path to the pinned conda env> \\
    accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \\
      --num_processes 1 scripts/rebase_training_smoketest.py

CUDA_HOME must point at the pinned conda env (its own bin/nvcc), not the default
/usr/local/cuda -- DeepSpeed's op-compatibility probe hardcodes the latter.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

import torch
from omegaconf import OmegaConf

from accelerate import Accelerator, DeepSpeedPlugin

from parity_probe import make_examples, split_losses
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

STEPS = 3


def main():
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "i1_libero_local.yaml")
    cfg.output_dir = "/tmp/rebase_ds_smoketest"
    cfg.trainer.gradient_accumulation_steps = 1
    cfg.datasets.vla_data.per_device_batch_size = 2

    deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    accelerator.print(accelerator.state)

    torch.manual_seed(1234)
    model = build_framework(cfg)
    accelerator.print("MODEL BUILT OK under launched accelerate/DeepSpeed context")

    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.trainer.learning_rate.base)

    # "auto" in ds_config.yaml normally resolves from a real DataLoader's .batch_size, which the
    # synthetic batch below does not have.
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
        cfg.datasets.vla_data.per_device_batch_size
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.print(f"accelerator.prepare() OK, model type: {type(model).__name__}")

    examples = make_examples(cfg, video_seed=1)
    for step in range(STEPS):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(examples)
                losses, _ = split_losses(out)
                total = sum(losses.values())
            accelerator.backward(total)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if accelerator.is_main_process:
            named = {k: round(v.item(), 4) for k, v in losses.items()}
            accelerator.print(f"step {step}: total_loss={total.item():.4f} {named}")
            assert total.item() == total.item(), f"step {step}: loss is NaN"

    accelerator.print("DEEPSPEED_LAUNCH_SMOKE_OK")


if __name__ == "__main__":
    main()
