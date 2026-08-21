# Vendored V-JEPA 2.1 modeling code

## Upstream

| Field | Value |
|---|---|
| Source | HuggingFace repo [`apiantonio/vjepa2.1-vit-large-384`](https://huggingface.co/apiantonio/vjepa2.1-vit-large-384) |
| Revision | `d6cfdbdd818754f22eaa72e5320d97724765f099` |
| Converter | `convert_vjepa21_to_hf.py` from [`github.com/Dev-Jahn/vjepa2-hf`](https://github.com/Dev-Jahn/vjepa2-hf) |
| Original model | Meta V-JEPA 2.1 ViT-L/16 @384, distilled from ViT-G (`vjepa2_1_vitl_dist_vitG_384.pt`), section `ema_encoder` |
| Licence | MIT (port repo declares `license: mit`; it states the original V-JEPA 2 release is MIT) |

## Vendored files

Byte-identical copies of the upstream revision:

| File | Local sha256 | Upstream git blob id |
|---|---|---|
| `configuration_vjepa21.py` | `7e7c7172fc301704707672b4d829e44aa81a617319ff390b5c686beb7968edf0` | `7ab5234b38b04f94ef4ceb70aa1fb203275265aa` |
| `modeling_vjepa21.py` | `0d8eb50a21b879d73772354abf4ae8604d2e741af1fe775e043fb71ea5d62ff3` | `2d31e0afade20757ca913fddffa393271e8c67c9` |

The blob ids are what the HuggingFace API reports for these (non-LFS) files at the pinned revision;
the sha256 column is the local copy, so the pair lets either identity be re-checked independently.

`__init__.py` and this file are ours. **No other change was made to the upstream sources** - no
reformatting, no lint fixes, no renames. That is deliberate: an unmodified copy can be re-diffed
against upstream at any time, and the repository formatter must not touch it
(`AGENTS.md` section 8 already forbids batch-rewriting vendored/upstream code).

## Why vendored instead of `trust_remote_code=True`

The model definition is an experimental variable of I3: it decides what the teacher computes. With
remote code, a silent upstream edit would change what the probes measure with no diff and no version
bump. Vendoring pins it, makes it reviewable, and keeps the pinned environment free of a runtime
code-download path.

## Why this port and not the other one (measured 2026-08-02)

Two community ports publish a *byte-identical* `model.safetensors` (same sha256, verified by hashing
both files - see `docs/provenance/teachers.md`), because the second one converted the checkpoint with
the first one's script. Only their modeling code differs, and it differs in a way that decides the
question:

| | `Dev-Jahn/vjepa2.1-vitl-fpc64-384` | `apiantonio/vjepa2.1-vit-large-384` (vendored) |
|---|---|---|
| Attention dispatch | `ALL_ATTENTION_FUNCTIONS.get_interface(...)`, transformers 5.x only | `resolve_attention_interface`, explicit 4.x and 5.x branches |
| Runs on pinned `transformers==4.57.0` | **no** | yes |
| Declared support | `transformers_version 5.12.1` | `>= 4.50`, tested on 4.57.1 and 5.14.1 |
| Video layout | requires channels-first `(B, C, T, H, W)` | `normalize_video_layout` accepts either |

Dev-Jahn's code loads the weights without complaint and then fails on the first attention layer:

```text
AttributeError: 'AttentionInterface' object has no attribute 'get_interface'
  modeling_vjepa21.py:406 in VJEPA21RopeAttention.forward
```

`get_interface` does not exist in 4.57.0. Upgrading transformers is not an option - it would
invalidate the I1/I2 parity conclusions - so the port that supports the pinned major is the one that
gets vendored. Since the weights are identical, nothing about the weight verification changes: the
result in `docs/provenance/teachers.md` covers both ports.

Dev-Jahn's repository stays downloaded: it is the origin of the conversion script, and its weights
directory is what these vendored classes are pointed at.

## Compatibility with the pinned environment (measured 2026-08-02)

In `envs/dynaweave` (transformers 4.57.0, torch 2.6.0+cu124), loading the local weights through the
vendored classes and running a forward pass:

```text
VJEPA21Model.from_pretrained(port_dev_jahn, config=VJEPA21Config.from_pretrained(port_dev_jahn))
  -> 327,654,016 parameters
  -> missing_keys 0, unexpected_keys 0, mismatched_keys 0, error_msgs 0
get_vision_features, 8-frame clip, tubelet 2 -> 4 temporal blocks:
  384x384 -> (1, 2304, 1024)   = 4 blocks * 24*24 tokens
  256x256 -> (1, 1024, 1024)   = 4 blocks * 16*16 tokens   (config.interpolate_rope)
  (B, T, C, H, W) and (B, C, T, H, W) give bitwise-equal features; repeat calls are deterministic
```

Two consequences for I3. The clip can be fed exactly as the video processor returns it, so the
adapter needs no layout argument. And because RoPE is interpolatable, the 2.1 teacher also runs at
256 with a native 16x16 grid - which is a stronger control arm than resampling 24x24 down to 16x16,
since it holds input resolution *and* output grid fixed while swapping only the teacher.

## Config geometry (the one adapter-relevant difference)

| Key | V-JEPA 2 (`vjepa2-vitl-fpc64-256`) | V-JEPA 2.1 (this port) |
|---|---|---|
| `image_size` | `256` | **absent** |
| `crop_size` | `256` | `384` |
| `patch_size` / `tubelet_size` | `16` / `2` | `16` / `2` |
| `hidden_size` | `1024` | `1024` |
| token grid | 16x16 | 24x24 |

`image_size` being absent (not merely different) is why `vj_backbone_adapter.py` needs a geometry
fallback: read `image_size` first, fall back to `crop_size`, error if neither is present.

## Preprocessing

Neither the vendored code nor Dev-Jahn's repository ships a video processor; the second port does
(`video_processing_vjepa21.py`). It is deliberately **not** used. The pinned
`transformers.VJEPA2VideoProcessor` covers the 384 path with
`size={"shortest_edge": 438}, crop_size=384`, which keeps the crop ratio bit-identical to the
V-JEPA 2 arm (`292 -> 256` and `438 -> 384` are both `x1.5` apart, ratio `0.8767`). Using the same
processor class for both teachers means the A/B comparison is not confounded by a different resize
and crop rule. The port's own processor instead resizes the short side to exactly 384 (the reference
`EvalVideoTransform`), which would give the 2.1 arm a wider field of view than the 2 arm.

## Authenticity of the weights

The weights are **not** vendored; they stay outside the repository under
`/vepfs/wangshilong/models/dynaweave/vjepa21/`. `tests/tools/check_vjepa21_weights.py` verifies them
against Meta's own `.pt` without a name mapping. See `docs/provenance/teachers.md` for the result and
for the boundary of what that check does and does not prove.
