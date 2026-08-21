"""LIBERO LeRobot dataloader producing the batch shape VLA_JEPA.forward() actually needs.

Upstream's own `lerobot_datasets.py` (adopted as-is elsewhere in this port) is correct for the
frameworks it was written for, but its shared `LeRobotMixtureDataset.__getitem__` ->
`LeRobotSingleDataset._pack_sample` path only ever keeps the first frame of whatever a modality's
`delta_indices` fetched -- so it cannot produce the multi-frame `video` clip VLA_JEPA.forward()
needs, regardless of DataConfig settings (see jepa_data_config.py and
docs/plans/upstream-rebase-experiment.md, "Needs a decision" #3, which this file resolves via
option (b): a JEPA-specific DataConfig plus a small `LeRobotMixtureDataset` subclass, rather than
porting VLA-JEPA's own diverged datasets.py wholesale).

`get_step_data`/`transforms` (the raw multi-frame fetch) are unmodified, shared upstream code --
only the final packing step below is new, and it mirrors the logic VLA-JEPA's own (diverged)
`LeRobotMixtureDataset.__getitem__` already uses in production.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.jepa_data_config import Libero4in1JEPADataConfig
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

# Robot types with a known JEPA-aware DataConfig. Extend as more embodiments need the same
# multi-frame contract; unlisted robot_types raise rather than silently falling back to the
# single-frame upstream config.
JEPA_ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1JEPADataConfig,
}


def collate_fn(batch):
    return batch


class JEPALeRobotMixtureDataset(LeRobotMixtureDataset):
    """Packs the multi-frame `video` clip and current `state` VLA_JEPA.forward() reads.

    Structurally identical to the parent class (sampling/weighting machinery is all inherited
    unchanged) except for `__getitem__`, which fetches the same raw per-step data upstream's own
    `_pack_sample` does but keeps the full `delta_indices` window for video instead of frame 0.
    """

    def __init__(
        self,
        *args,
        video_resolution_size: int = 256,
        resolution_size: int = 224,
        with_state: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.video_resolution_size = video_resolution_size
        self.resolution_size = resolution_size
        self.with_state = with_state

    @staticmethod
    def _resize_video(video: np.ndarray, size: int) -> np.ndarray:
        return np.stack([cv2.resize(frame, (size, size)) for frame in video])

    def __getitem__(self, index: int) -> dict:
        max_retries = 10
        last_exception = None
        for attempt in range(max_retries):
            try:
                dataset, trajectory_id, step = self.sample_step(index)
                data = dataset.transforms(dataset.get_step_data(trajectory_id, step))

                videos, images = [], []
                for video_key in dataset.modality_keys["video"]:
                    video = data[video_key]  # (T, H, W, C), T = observation_indices length
                    video = self._resize_video(video, self.video_resolution_size)
                    videos.append(video)
                    images.append(Image.fromarray(video[0]).resize((self.resolution_size, self.resolution_size)))
                videos = np.stack(videos, axis=0)  # (V, T, H, W, C)

                language = data[dataset.modality_keys["language"][0]][0]
                action = np.concatenate(
                    [data[key] for key in dataset.modality_keys["action"]], axis=1
                ).astype(np.float16)

                sample = {"action": action, "image": images, "lang": language, "video": videos}
                if self.with_state and dataset.modality_keys.get("state"):
                    state = np.concatenate(
                        [data[key] for key in dataset.modality_keys["state"]], axis=1
                    ).astype(np.float16)
                    sample["state"] = state[0:1]
                return sample
            except Exception as exc:  # noqa: BLE001 -- mirrors upstream's own retry-and-resample
                last_exception = exc
                if attempt < max_retries - 1:
                    index = random.randint(0, len(self) - 1)
                else:
                    raise last_exception


def get_jepa_vla_dataset(
    data_cfg,
    video_horizon: int = 8,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
) -> JEPALeRobotMixtureDataset:
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    dataset_mixture = []
    seen = set()
    for d_name, d_weight, robot_type in mixture_spec:
        if (d_name, robot_type) in seen:
            continue
        seen.add((d_name, robot_type))

        if robot_type not in JEPA_ROBOT_TYPE_CONFIG_MAP:
            raise NotImplementedError(
                f"No JEPA-aware DataConfig registered for robot_type={robot_type!r}; "
                f"add it to JEPA_ROBOT_TYPE_CONFIG_MAP in {__name__}."
            )
        data_config = JEPA_ROBOT_TYPE_CONFIG_MAP[robot_type](video_horizon=video_horizon)
        single_dataset = LeRobotSingleDataset(
            dataset_path=data_root_dir / d_name,
            modality_configs=data_config.modality_config(),
            transforms=data_config.transform(),
            embodiment_tag=data_config.embodiment_tag,
            video_backend=data_cfg.get("video_backend", "decord"),
            data_cfg=data_cfg,
        )
        dataset_mixture.append((single_dataset, d_weight))

    return JEPALeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        video_resolution_size=data_cfg.get("video_resolution_size", 256),
        resolution_size=data_cfg.get("resolution_size", 224),
        with_state=data_cfg.get("with_state", True),
    )
