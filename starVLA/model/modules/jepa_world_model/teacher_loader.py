# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""How each frozen video teacher is loaded from local weights.

One responsibility: turn `(teacher id, weight directory, input size)` into the `(encoder, processor)`
pair `VJBackboneAdapter` wraps. Nothing here knows about probe arms, training configs or geometry --
the caller decides those and passes `input_size` in.

The two teachers need different loaders, which is the whole reason this exists:

    vjepa2   `AutoModel` / `AutoVideoProcessor`, both read from the pinned checkpoint directory.
             The processor is used exactly as published, so this path stays the I2-measured one.
    vjepa21  no HuggingFace hub repo exists, so the vendored `VJEPA21Config` / `VJEPA21Model`
             (docs/provenance/teachers.md, D-044) load the community port and the processor is
             constructed here, because the checkpoint ships none.

Lives under `jepa_world_model/` rather than `probes/` because both the I3 probe bench
(`starVLA/probes/arms.py`) and the I4 training framework (`model/framework/VLM4A/VLA_JEPA.py`) need it.
The dependency runs probes -> jepa_world_model and framework -> jepa_world_model; `jepa_world_model`
imports neither, which is what keeps `tests/test_i3_probe_firewall.py` meaningful.
"""

from pathlib import Path
from typing import Optional, Tuple

from starVLA.model.modules.jepa_world_model.vj_backbone_adapter import resolve_input_size

TEACHER_VJEPA2 = "vjepa2"
TEACHER_VJEPA21 = "vjepa21"
TEACHERS: Tuple[str, ...] = (TEACHER_VJEPA2, TEACHER_VJEPA21)

# Shortest edge each resolution resizes to before the square centre crop. Both keep the crop ratio
# 0.8767 the pinned V-JEPA 2 processor uses, which is what makes two teachers see the same field of
# view and therefore be comparable at all.
SHORTEST_EDGE = {256: 292, 384: 438}


def load_teacher(teacher: str, root: Path, input_size: Optional[int] = None, device: Optional[str] = None):
    """Load one frozen teacher and the processor that feeds it.

    Args:
        teacher: `TEACHER_VJEPA2` or `TEACHER_VJEPA21`.
        root: local weight directory. Must exist; teachers are never fetched at run time.
        input_size: square edge the caller will feed, or `None` for the size the checkpoint states.
            Only V-JEPA 2.1 can be overridden -- its RoPE is interpolatable and its patch grid comes
            from the input tensor, so one checkpoint runs natively at both 256 and 384. V-JEPA 2
            states its grid as a config constant and has no RoPE interpolation (D-053), so a
            differing size is rejected here rather than silently re-cropped downstream.
        device: moved there when given. `None` leaves placement to the caller, which is what the
            training framework needs -- it registers the encoder as a submodule and lets the
            trainer place the whole model.

    Returns:
        `(encoder, processor)`. Freezing is *not* done here: `VJBackboneAdapter.__init__` owns the
        gradient firewall, so there is exactly one place that asserts it.
    """
    if teacher not in TEACHERS:
        raise ValueError(f"unknown teacher {teacher!r}; supported teachers are {list(TEACHERS)}")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"teacher {teacher!r}: missing weights at {root}")

    if teacher == TEACHER_VJEPA2:
        from transformers import AutoModel, AutoVideoProcessor

        encoder = AutoModel.from_pretrained(root)
        # As published, so this stays the path I1/I2 measured.
        processor = AutoVideoProcessor.from_pretrained(root)
        configured = resolve_input_size(encoder.config)
        if input_size is not None and input_size != configured:
            raise ValueError(
                f"{TEACHER_VJEPA2} is configured for input size {configured} and cannot run at {input_size}"
            )
    else:
        from transformers import VJEPA2VideoProcessor

        from starVLA.model.modules.jepa_world_model.vjepa21 import VJEPA21Config, VJEPA21Model

        config = VJEPA21Config.from_pretrained(root)
        size = resolve_input_size(config) if input_size is None else input_size
        if size not in SHORTEST_EDGE:
            raise ValueError(f"no crop ratio registered for input size {size}; have {sorted(SHORTEST_EDGE)}")
        encoder = VJEPA21Model.from_pretrained(root, config=config)
        processor = VJEPA2VideoProcessor(
            size={"shortest_edge": SHORTEST_EDGE[size]},
            crop_size={"height": size, "width": size},
        )

    if device is not None:
        encoder = encoder.to(device)
    return encoder, processor
