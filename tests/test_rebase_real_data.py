"""Real LIBERO data through the ported VLA_JEPA, via dataloader.jepa_lerobot_datasets.

Separate from test_rebase_smoke.py (which uses only synthetic batches) because this needs real
data on disk and is slow (builds/reads dataset statistics caches). Confirms the resolution of
docs/plans/upstream-rebase-experiment.md's "Needs a decision" #3: upstream's own
`lerobot_datasets.py` cannot feed VLA_JEPA.forward() (its shared `_pack_sample` keeps only frame 0
of any modality regardless of DataConfig delta_indices) -- `jepa_lerobot_datasets.py` resolves this
via a JEPA-specific DataConfig (`Libero4in1JEPADataConfig`) plus a `LeRobotMixtureDataset` subclass
that packs the full multi-frame window, both additive and not touching upstream's own files.

Requires one visible GPU, the local checkpoints, and real LIBERO LeRobot data on disk; skipped
otherwise. Run:  CUDA_VISIBLE_DEVICES=0 pytest tests/test_rebase_real_data.py -v
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from parity_probe import CONFIG_PATH, split_losses


def _require_gpu_weights_and_data(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    for key in (cfg.framework.qwenvl.base_vlm, cfg.framework.vj2_model.base_encoder):
        if not Path(key).exists():
            pytest.skip(f"missing local weights: {key}")
    if not Path(cfg.datasets.vla_data.data_root_dir).exists():
        pytest.skip(f"missing LIBERO data: {cfg.datasets.vla_data.data_root_dir}")


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def real_batch(cfg):
    _require_gpu_weights_and_data(cfg)
    from starVLA.dataloader import build_dataloader

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    return next(iter(dataloader))


def test_real_batch_has_the_fields_forward_needs(real_batch):
    example = real_batch[0]
    assert set(example) >= {"action", "image", "lang", "video", "state"}
    assert example["video"].ndim == 5  # (V, T, H, W, C)
    assert example["video"].shape[0] == 2  # two camera views
    assert example["state"].shape[0] == 1  # current state only


def test_real_batch_forward_is_finite(cfg, real_batch):
    _require_gpu_weights_and_data(cfg)
    from starVLA.model.framework.base_framework import build_framework

    torch.manual_seed(1234)
    model = build_framework(cfg).to("cuda")
    model.eval()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(real_batch)

    losses, _ = split_losses(out)
    assert set(losses) == {"action_loss", "wm_loss"}
    for name, value in out.items():
        assert torch.isfinite(value), f"{name} is not finite on real data: {value}"
        print(f"\n[real data] {name} = {value.item():.6f}")
