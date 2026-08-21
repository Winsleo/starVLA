"""Vendored V-JEPA 2.1 modeling code (see `VENDOR.md` for upstream, revision and licence).

Vendored rather than loaded with `trust_remote_code=True` so the teacher's definition is pinned,
diffable and reviewable in-tree: with remote code, a silent upstream edit would change what I3
measures. `configuration_vjepa21.py` and `modeling_vjepa21.py` are byte-identical to the upstream
port; only this file and `VENDOR.md` are ours.

`starVLA/model/modules/jepa_world_model/vj_backbone_adapter.py` is the only consumer, and it needs
`get_vision_features` - the same entry point `transformers.VJEPA2Model` exposes for V-JEPA 2, which
is why the 2.1 teacher fits behind the existing seam.
"""

from .configuration_vjepa21 import VJEPA21Config
from .modeling_vjepa21 import (
    VJEPA21ForVideoClassification,
    VJEPA21Model,
    VJEPA21PreTrainedModel,
)

__all__ = [
    "VJEPA21Config",
    "VJEPA21ForVideoClassification",
    "VJEPA21Model",
    "VJEPA21PreTrainedModel",
]
