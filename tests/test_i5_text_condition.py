"""Gates for the I5 text conditioning (`I5-S3-TF`, D-064).

The convention is not ours: it is `WanPipeline._get_t5_prompt_embeds` with `max_sequence_length=512`,
which is what `__call__` passes. Two parts of it matter enough to pin:

* the padded tail is **exact zeros**, not encoder output on pad tokens. Feeding the DiT encoder output
  on padding would be a text field it never saw in training, and nothing would fail loudly;
* `prompt_clean` is imported from diffusers rather than reimplemented, so normalisation cannot drift.

The encoder is stubbed, so the convention is checked on CPU with no weights. The real-weight checks
run only when the Wan checkpoint is present.
"""

from __future__ import annotations

import json
import os
import unittest

import torch

from starVLA.i5.text_condition import (
    TEXT_EMBED_DIM,
    TEXT_SEQUENCE_LENGTH,
    assert_padding_is_zero,
    clean_prompt,
    encode_prompts,
)

#: Set to the built text cache directory to check the artefact itself.
TEXT_CACHE = os.environ.get("I5_TEXT_CACHE")


class _StubTokenizerOutput:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _StubTokenizer:
    """One token per word plus an EOS, padded to `max_length`."""

    def __call__(self, texts, *, padding, max_length, truncation, return_tensors, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            length = min(len(text.split()) + 1, max_length)
            ids[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return _StubTokenizerOutput(ids, mask)


class _StubEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _StubTextEncoder:
    """Emits a nonzero value at *every* position, including padding.

    That is the point: if `encode_prompts` did not trim and re-pad, the tail would come back nonzero
    and the padding assertion would catch it.
    """

    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        hidden = torch.arange(1, length + 1, dtype=torch.float32)[None, :, None]
        return _StubEncoderOutput(hidden.expand(batch, length, TEXT_EMBED_DIM).clone())


class TextConventionTest(unittest.TestCase):
    PROMPTS = ["put the bowl on the plate", "open the top drawer"]

    def test_shape_and_dim(self):
        embeds = encode_prompts(_StubTokenizer(), _StubTextEncoder(), self.PROMPTS)
        self.assertEqual(
            tuple(embeds.shape), (2, TEXT_SEQUENCE_LENGTH, TEXT_EMBED_DIM)
        )

    def test_padding_is_exact_zero_past_the_valid_length(self):
        embeds = encode_prompts(_StubTokenizer(), _StubTextEncoder(), self.PROMPTS)
        lengths = [len(p.split()) + 1 for p in self.PROMPTS]
        assert_padding_is_zero(embeds, lengths)
        for row, length in enumerate(lengths):
            self.assertGreater(float(embeds[row, :length].abs().max()), 0.0)
            self.assertEqual(float(embeds[row, length:].abs().max()), 0.0)

    def test_the_padding_check_actually_fires(self):
        """Without trimming, the stub encoder's output would be nonzero everywhere."""
        embeds = encode_prompts(_StubTokenizer(), _StubTextEncoder(), self.PROMPTS)
        embeds[0, 100] = 1.0  # simulate encoder output leaking into the padded tail
        with self.assertRaisesRegex(AssertionError, "padding is not zero"):
            assert_padding_is_zero(embeds, [len(p.split()) + 1 for p in self.PROMPTS])

    def test_prompt_clean_is_the_pipeline_one(self):
        self.assertEqual(clean_prompt("  put   the  bowl on the plate "), "put the bowl on the plate")

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no prompts"):
            encode_prompts(_StubTokenizer(), _StubTextEncoder(), [])


@unittest.skipUnless(
    TEXT_CACHE and os.path.isfile(os.path.join(TEXT_CACHE, "text_embeds.index.json")),
    "set I5_TEXT_CACHE to check the built cache",
)
class TextCacheArtefactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np

        cls.index = json.loads(
            open(os.path.join(TEXT_CACHE, "text_embeds.index.json")).read()
        )
        with np.load(os.path.join(TEXT_CACHE, "text_embeds.npz")) as payload:
            cls.embeds = payload["embeds"]
            cls.lengths = payload["valid_lengths"]

    def test_covers_every_libero_task(self):
        self.assertEqual(len(self.index["instructions"]), 40)
        self.assertEqual(self.index["shape"], [40, TEXT_SEQUENCE_LENGTH, TEXT_EMBED_DIM])

    def test_rows_match_the_instruction_index(self):
        for text, row in self.index["instruction_to_row"].items():
            self.assertEqual(self.index["instructions"][row], text)

    def test_padding_is_zero_in_the_artefact(self):
        assert_padding_is_zero(torch.from_numpy(self.embeds.astype("float32")), self.lengths.tolist())

    def test_most_of_the_context_is_padding(self):
        """Recorded because it changes how the action-token dilution risk should be read.

        The instructions are 6 to 21 tokens of a 512-token context, so the action segment competes
        with roughly fifteen informative text tokens, not five hundred -- though the padded positions
        still take softmax mass, since their keys and values are the projection biases.
        """
        self.assertLessEqual(int(self.lengths.max()), 32)
        self.assertGreaterEqual(int(self.lengths.min()), 1)


if __name__ == "__main__":
    unittest.main()
