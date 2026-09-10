# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LoRA wiring for the I5 generator (`I5-S3-TF`, per D-064).

D-064 trains adapters on the Wan DiT plus the condition projector, and leaves the VAE and the text
encoder frozen. Full fine-tuning of the 5B transformer is declined there: the pretrained video prior
is the reason that backbone was chosen, and I5's gate is a relative judgement that does not need
absolute picture quality.

Target modules are the attention and feed-forward `Linear` layers inside the transformer blocks,
enumerated from the real model rather than guessed:

    blocks.*.attn1.{to_q,to_k,to_v,to_out.0}    self-attention over latent tokens
    blocks.*.attn2.{to_q,to_k,to_v,to_out.0}    cross-attention -- where the action condition enters
    blocks.*.ffn.net.{0.proj,2}                 feed-forward

Deliberately excluded:

* `condition_embedder.*` -- the timestep and text embedders. The timestep path is how the segmented
  per-token timesteps reach the model; adapting it would let the model relearn the noise schedule
  rather than the dynamics, and the text embedder is the pretrained projection our action tokens
  ride through.
* `proj_out` -- the unpatchify head. Adapting it changes the output parameterisation, and the flow
  target is pinned (D-067).

`peft` is used rather than a hand-rolled adapter so the artefacts stay loadable by the same tooling
Wan's README points at for training. It was installed with `--no-deps` and left torch, transformers
and accelerate untouched; recorded as a supplementary dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Suffixes of the `Linear` layers inside a `WanTransformerBlock`, verified against the real class.
WAN_LORA_TARGETS: tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)

DEFAULT_LORA_RANK = 32


def wan_lora_target_names(transformer: nn.Module) -> list[str]:
    """Fully qualified names of the `Linear` layers LoRA should wrap.

    Resolved against the actual module tree, so a change in the block layout surfaces as an empty or
    short list here rather than as adapters silently attached to nothing.
    """
    names = [
        name
        for name, module in transformer.named_modules()
        if isinstance(module, nn.Linear)
        and name.startswith("blocks.")
        and any(name.endswith(suffix) for suffix in WAN_LORA_TARGETS)
    ]
    if not names:
        raise ValueError(
            "no LoRA targets found; the transformer block layout does not match "
            f"{WAN_LORA_TARGETS}"
        )
    return names


def attach_lora(
    transformer: nn.Module,
    *,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Wrap the transformer's attention and feed-forward `Linear` layers in LoRA adapters.

    The base weights are frozen by peft, and peft zero-initialises the `lora_B` factor, so the
    adapted model starts numerically identical to the base model. Combined with the zero-initialised
    condition projector, that means the whole generator starts as the pretrained model with no
    action conditioning -- which is what makes the no-condition and correct-condition arms coincide
    at step 0 (D-062).
    """
    from peft import LoraConfig, get_peft_model

    targets = wan_lora_target_names(transformer)
    config = LoraConfig(
        r=rank,
        lora_alpha=rank if alpha is None else alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    return get_peft_model(transformer, config)


def trainable_parameter_summary(module: nn.Module) -> dict[str, int]:
    """Trainable and total parameter counts, split into LoRA and everything else.

    `AGENTS.md` section 10 wants gradient routes made explicit; the cheapest honest version of that
    for an adapter run is to be able to state exactly what carries gradient.
    """
    summary = {"total": 0, "trainable": 0, "trainable_lora": 0, "trainable_other": 0}
    for name, parameter in module.named_parameters():
        count = parameter.numel()
        summary["total"] += count
        if not parameter.requires_grad:
            continue
        summary["trainable"] += count
        key = "trainable_lora" if "lora_" in name else "trainable_other"
        summary[key] += count
    return summary


def assert_only_expected_parameters_train(
    module: nn.Module, *, allowed_substrings: tuple[str, ...] = ("lora_",)
) -> list[str]:
    """Return the trainable parameter names, raising if any is outside the allowed set.

    Used as a training-time guard: an adapter run that silently starts updating the frozen backbone
    would still produce a falling loss, so the failure would not announce itself.
    """
    trainable = [name for name, p in module.named_parameters() if p.requires_grad]
    unexpected = [
        name for name in trainable if not any(token in name for token in allowed_substrings)
    ]
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} unexpected trainable parameter(s), e.g. {unexpected[:5]}; "
            f"expected only names containing {allowed_substrings}"
        )
    return trainable


@torch.no_grad()
def lora_is_identity(
    transformer: nn.Module, sample_inputs: dict[str, torch.Tensor], base_output: torch.Tensor
) -> bool:
    """Whether the adapted model still reproduces a stored base-model output.

    Cheap way to assert the zero-init claim on the real model without keeping two copies of a 5B
    transformer resident.
    """
    output = transformer(**sample_inputs)
    output = output.sample if hasattr(output, "sample") else output
    if isinstance(output, tuple):
        output = output[0]
    return torch.equal(output, base_output)
