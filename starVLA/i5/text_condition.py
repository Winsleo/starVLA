# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Offline text conditioning for I5 (`I5-S3-TF`, per D-064).

The language condition is precomputed once and cached: LIBERO has 40 task instructions, the UMT5
encoder is frozen, so loading 11 GB of text encoder at every training step would buy nothing.

The encoding convention is taken from the **canonical inference path**, not invented and not copied
from `Wan2.py`. `WanPipeline.__call__` defaults `max_sequence_length` to 512 and passes it down to
`_get_t5_prompt_embeds`, which:

1. normalises the prompt with `prompt_clean` (ftfy plus double HTML unescape plus whitespace collapse);
2. tokenises with `padding="max_length"`, `truncation=True`, `add_special_tokens=True`;
3. runs the encoder **with the attention mask**;
4. keeps only the valid positions per sample and **zero-pads back** to 512.

Step 4 is the part worth naming: the padded tail is exact zeros, not encoder output on pad tokens.
Getting it wrong would feed the DiT a text field it never saw in training, and nothing would fail
loudly. `prompt_clean` is imported from diffusers rather than reimplemented, so it cannot drift.

The 226 that appears as `_get_t5_prompt_embeds`'s own default is never what runs -- every caller
overrides it with 512.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: Effective text context length at inference, from `WanPipeline.__call__`'s default.
TEXT_SEQUENCE_LENGTH = 512

#: UMT5-XXL hidden width, i.e. the DiT's `text_dim`.
TEXT_EMBED_DIM = 4096


def clean_prompt(text: str) -> str:
    """The pipeline's own prompt normalisation, imported so it cannot drift from it."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    return prompt_clean(text)


@torch.no_grad()
def encode_prompts(
    tokenizer,
    text_encoder,
    prompts: Sequence[str],
    *,
    max_length: int = TEXT_SEQUENCE_LENGTH,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """`[N, max_length, TEXT_EMBED_DIM]`, following the inference-path convention exactly."""
    if not prompts:
        raise ValueError("no prompts to encode")
    cleaned = [clean_prompt(p) for p in prompts]
    inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mask = inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    hidden = text_encoder(
        inputs.input_ids.to(device), mask.to(device)
    ).last_hidden_state.to(dtype)
    # Keep the valid positions, then zero-pad: the tail must be exact zeros.
    trimmed = [row[:length] for row, length in zip(hidden, seq_lens)]
    padded = torch.stack(
        [
            torch.cat([row, row.new_zeros(max_length - row.shape[0], row.shape[1])])
            for row in trimmed
        ],
        dim=0,
    )
    if padded.shape[1:] != (max_length, TEXT_EMBED_DIM):
        raise AssertionError(
            f"text embeds are {tuple(padded.shape[1:])}, expected {(max_length, TEXT_EMBED_DIM)}"
        )
    return padded


def assert_padding_is_zero(embeds: torch.Tensor, valid_lengths: Sequence[int]) -> None:
    """Raise unless every position past a prompt's valid length is exactly zero."""
    for index, length in enumerate(valid_lengths):
        tail = embeds[index, length:]
        if tail.numel() and bool(tail.abs().max() > 0):
            raise AssertionError(
                f"prompt {index}: padding is not zero past position {length} "
                f"(max abs {float(tail.abs().max()):.3e})"
            )
