"""LIBERO DataConfig for VLA_JEPA's multi-frame video contract.

`Libero4in1DataConfig` (data_config.py) requests a single timestep per modality
(`observation_indices = [0]`) -- correct for the single-frame VLM+action-head frameworks it was
written for, but VLA_JEPA.forward() needs a multi-frame video clip to predict future JEPA
features from (docs/plans/upstream-rebase-experiment.md, "Needs a decision" #3). This subclass
changes only the sampling window, reusing every other field (video/state/action/language keys,
transforms) from the parent unchanged.
"""

from starVLA.dataloader.gr00t_lerobot.data_config import Libero4in1DataConfig


class Libero4in1JEPADataConfig(Libero4in1DataConfig):
    def __init__(self, video_horizon: int = 8):
        # One window of `video_horizon` consecutive frames starting at the sampled step, matching
        # VLA-JEPA's own (diverged) dataloader/lerobot_datasets.py's
        # `observation_indices=list(range(video_horizon))` convention -- current frame plus
        # `video_horizon - 1` future frames, which is what a JEPA target needs.
        self.observation_indices = list(range(video_horizon))
        # Current proprioceptive state only: JEPALeRobotMixtureDataset takes state[0:1] regardless
        # (matching VLA-JEPA's own convention), so a single-index window is both sufficient and
        # avoids fetching 16 unused past steps the parent class's default requests.
        self.state_indices = [0]
