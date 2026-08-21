# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def reinitialize_predictor(predictor) -> int:
    """Reset the I4 world predictor after loading a checkpoint, preserving all other state."""
    predictor.apply(predictor._init_weights)
    predictor._rescale_blocks()
    return sum(parameter.numel() for parameter in predictor.parameters())

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.jepa_world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.jepa_world_model.vj_backbone_adapter import VJBackboneAdapter
from starVLA.model.modules.jepa_world_model.teacher_loader import TEACHER_VJEPA2, load_teacher
from starVLA.model.modules.jepa_world_model.depth_targets import build_metric_delta_targets
from starVLA.model.modules.jepa_world_model.depth_delta_head import DepthDeltaHead
from starVLA.model.modules.jepa_world_model.depth_losses import depth_delta_loss
from starVLA.training.trainer_utils.trainer_tools import METRIC_PREFIX, resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        teacher_cfg = self.config.framework.vj2_model
        teacher = teacher_cfg.get("teacher", TEACHER_VJEPA2)
        teacher_root = teacher_cfg.base_encoder if teacher == TEACHER_VJEPA2 else teacher_cfg.get("teacher_weights")
        if teacher_root is None:
            raise ValueError(f"teacher_weights is required for teacher={teacher!r}")
        self.vj_encoder, self.vj_processor = load_teacher(
            teacher=teacher,
            root=teacher_root,
            input_size=teacher_cfg.get("input_size"),
        )
        # Owns the frozen-teacher firewall and the patch geometry. Keeps `vj_encoder` as a direct
        # submodule so published checkpoints still load with strict=True.
        self.vj_backbone = VJBackboneAdapter(
            encoder=self.vj_encoder,
            processor=self.vj_processor,
            num_frames=self.config.framework.vj2_model.num_frames,
            input_size=teacher_cfg.get("input_size"),
            # Off by default: upstream's view fusion is wrong for batch > 1, and the I2 goldens
            # encode it. See the adapter's docstring and docs/provenance/upstream-conflicts.md.
            correct_view_fusion=bool(self.config.framework.vj2_model.get("correct_view_fusion", False)),
        )

        # One predictor token per teacher token: the geometry comes from the backbone adapter, and
        # `grid_size` / `num_temporal_blocks` make the predictor verify it against its own
        # img_size/num_frames derivation instead of silently disagreeing.
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.vj_backbone.num_temporal_blocks,
            img_size=(self.vj_backbone.image_size, self.vj_backbone.image_size),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_backbone.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
            grid_size=self.vj_backbone.grid_size,
            num_temporal_blocks=self.vj_backbone.num_temporal_blocks,
        )
        # One action-token group per predicted temporal block, i.e. all blocks but the first.
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.vj_backbone.num_temporal_blocks - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

        # World-model loss weights. The two defaults reproduce the pinned baseline, which weights
        # the world loss differently depending on the sample type: 0.1 next to an action loss, and
        # 1.0 for action-free samples, where it is the only loss. The asymmetry is upstream
        # behaviour, kept on purpose; changing it is a separate experiment.
        loss_scale = self.config.trainer.get("loss_scale", {}) if self.config and self.config.trainer else {}
        self.wm_loss_weight = loss_scale.get("wm", 0.1)
        self.wm_action_free_loss_weight = loss_scale.get("wm_action_free", 1.0)
        depth_cfg = self.config.framework.get("depth_head", {})
        self.depth_enabled = bool(depth_cfg.get("enabled", False))
        self.depth_loss_weight = float(depth_cfg.get("weight", 0.05))
        self.depth_gradient_weight = float(depth_cfg.get("gradient_weight", 0.5))
        self.depth_tubelet_size = int(depth_cfg.get("tubelet_size", 2))
        self.depth_delta_lag = int(depth_cfg.get("delta_lag", 1))
        self.depth_target_grid = tuple(depth_cfg.get("target_grid", (16, 16)))
        if self.depth_enabled:
            if self.depth_tubelet_size != 2 or self.depth_delta_lag != 1:
                raise ValueError("I4 training depth target is fixed to tubelet_size=2 and delta_lag=1")
            self.depth_delta_head = DepthDeltaHead(
                hidden_size=self.qwen_vl_interface.model.config.hidden_size,
                channels=int(depth_cfg.get("channels", 64)),
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a checkpoint and optionally reset only its world predictor for an I4 cell."""
        try:
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            # I4's opt-in depth head is intentionally absent from the pinned I2 checkpoint.  Keep
            # strict compatibility for every baseline path; only the explicitly enabled auxiliary
            # branch may bootstrap its new parameters from their initializer.
            if not (self.depth_enabled and strict):
                raise
            result = super().load_state_dict(state_dict, strict=False, assign=assign)
            logger.info(
                "Loaded legacy checkpoint with depth head initialized; missing=%s unexpected=%s",
                result.missing_keys,
                result.unexpected_keys,
            )
        if self.config.framework.vj2_model.get("reinit_predictor", False):
            count = reinitialize_predictor(self.vj_predictor)
            logger.info(f"Reinitialized vj_predictor after checkpoint load ({count} parameters)")
        return result

    def reinitialize_world_predictor(self) -> int:
        """Reset only the predictor for I4 cells that intentionally swap the frozen teacher."""
        count = reinitialize_predictor(self.vj_predictor)
        logger.info(f"Reinitialized vj_predictor ({count} parameters)")
        return count

    def train(self, mode: bool = True):
        """Keep the frozen teacher in eval mode when the parent switches to train (AGENTS.md 6)."""
        super().train(mode)
        self.vj_backbone.enforce_frozen()
        return self

    def expand_tokenizer(self,
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        """
        if self.action_model.device == torch.device("cuda:0") and "action" in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(actions[0].shape) # [T-1, action_dim]
            print(state[0].shape) if state is not None else print("No state") #[state_dim]
            print(len(batch_videos), len(instructions), len(actions), len(state) if state is not None else "No state")
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "data_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "data_view_1.mp4")
            batch_images[0][0].save("data_image_view_0.png")
            batch_images[0][1].save("data_image_view_1.png")
            #print(self.action_tokens)
            print(self.replace_prompt)
            print(self.action_token_ids)
        elif self.action_model.device == torch.device("cuda:0") and "action" not in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(len(batch_videos), len(instructions))
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "video_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "video_view_1.mp4")
            batch_images[0][0].save("video_image_view_0.png")
        exit()
        """
        
        

        #[print(each.shape, end=";") for each in batch_videos]
        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")) 
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                # `datasets.video_data` only exists in the cotrain configs, while an action-free
                # sample can reach this branch under any config; select() returns the same prompt
                # where the node exists and "" instead of raising where it does not.
                prompt_template=OmegaConf.select(self.config, "datasets.video_data.CoT_prompt", default=""))
        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        # TODO action condition tokens
        #embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder
            # [B, num_temporal_blocks * tokens_per_block, V * encoder_dim]
            video_embeddings = self.vj_backbone.encode_video(batch_videos)

            # Step 3: VJ Predictor
            # both [B, (num_temporal_blocks - 1) * tokens_per_block, V * encoder_dim]
            input_states, gt_states = self.vj_backbone.split_teacher_forcing(video_embeddings)
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_states,
                reduction="mean"
            )

        depth_loss = None
        depth_metrics = {}
        has_cached_depth = all(key in examples[0] for key in ("depth_states", "depth_targets", "depth_mask"))
        has_raw_depth = all("depth" in example for example in examples)
        if self.depth_enabled and "action" in examples[0] and (has_cached_depth or has_raw_depth):
            if has_cached_depth:
                states_values = torch.as_tensor(
                    np.stack([example["depth_states"] for example in examples]),
                    device=last_hidden.device, dtype=torch.float32,
                )
                target_values = torch.as_tensor(
                    np.stack([example["depth_targets"] for example in examples]),
                    device=last_hidden.device, dtype=torch.float32,
                )
                target_mask = torch.as_tensor(
                    np.stack([example["depth_mask"] for example in examples]),
                    device=last_hidden.device, dtype=torch.bool,
                )
            else:
                depth = torch.as_tensor(
                    np.stack([example["depth"] for example in examples]),
                    device=last_hidden.device, dtype=torch.float32,
                )
                with torch.no_grad():
                    state_target, delta_target = build_metric_delta_targets(
                        depth,
                        tubelet_size=self.depth_tubelet_size,
                        grid=self.depth_target_grid,
                        target_type=self.config.datasets.vla_data.get("depth_target_type", "pseudo_metric"),
                        delta_lag=self.depth_delta_lag,
                    )
                states_values, target_values, target_mask = state_target.values, delta_target.values, delta_target.mask
            # Three action-token groups condition the three predictor transitions; repeat each
            # condition over the two camera views without exposing future depth to the policy.
            groups = self.vj_backbone.num_temporal_blocks - 1
            if action_tokens.shape[1] % groups:
                raise ValueError(f"action token count {action_tokens.shape[1]} is not divisible by {groups}")
            tokens = action_tokens.view(action_tokens.shape[0], groups, -1, action_tokens.shape[-1])
            current = states_values[:, :-1]
            target = target_values
            valid = target_mask
            batch_size, transitions, views = current.shape[:3]
            condition = tokens[:, :, None].expand(batch_size, transitions, views, *tokens.shape[2:])
            predicted_depth = self.depth_delta_head(
                current.reshape(batch_size * transitions * views, 1, *current.shape[-2:]),
                condition.reshape(batch_size * transitions * views, *condition.shape[-2:]),
            )
            raw_depth_loss, raw_depth_pixel, raw_depth_gradient = depth_delta_loss(
                predicted_depth,
                target.reshape(batch_size * transitions * views, 1, *target.shape[-2:]).detach(),
                valid.reshape(batch_size * transitions * views, 1, *valid.shape[-2:]).detach(),
                gradient_weight=self.depth_gradient_weight,
            )
            depth_loss = raw_depth_loss * self.depth_loss_weight
            depth_metrics = {
                f"{METRIC_PREFIX}depth_loss_raw": raw_depth_loss.detach(),
                f"{METRIC_PREFIX}depth_pixel_l1_raw": raw_depth_pixel.detach(),
                f"{METRIC_PREFIX}depth_gradient_raw": raw_depth_gradient.detach(),
                f"{METRIC_PREFIX}depth_loss_weight": torch.as_tensor(
                    self.depth_loss_weight, device=raw_depth_loss.device, dtype=torch.float32
                ),
            }
        
        if "action" not in examples[0]:
            weight = self.wm_action_free_loss_weight
            return {
                "wm_loss": teacher_forcing_wm_loss * weight,
                **self._loss_metrics(teacher_forcing_wm_loss, weight),
                **depth_metrics,
            }

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            # Read from `trainer`, while every config declares the value under
            # `framework.action_model`, so the literal default is what actually applies. Kept as is
            # on purpose: correcting the node would change the effective batch repeat, i.e. break
            # I2 parity. Locked at 4 and scheduled for I4.5 (D-041); pinned by
            # tests/test_i2_parity.py::test_repeated_diffusion_steps_effective_value_is_pinned.
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                #print(state.shape)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            #print(embodied_action_repeated.shape, actions_target_repeated.shape, state_repeated.shape) if state_repeated is not None else print("No state for action model")
            #exit()
            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

        weight = self.wm_loss_weight
        losses = {
            "action_loss": action_loss,
            "wm_loss": teacher_forcing_wm_loss * weight,
            **self._loss_metrics(teacher_forcing_wm_loss, weight, action_loss=action_loss),
        }
        if depth_loss is not None:
            losses["depth_loss"] = depth_loss
        losses.update(depth_metrics)
        return losses

    def _loss_metrics(self, wm_loss, wm_weight, action_loss=None):
        """Log-only companions of the returned losses (AGENTS.md section 10, item 8).

        Keys carry METRIC_PREFIX, which trainer_tools.split_loss_terms excludes from the backward
        sum, so reporting raw values alongside the weighted ones cannot change optimization.
        """
        metrics = {
            f"{METRIC_PREFIX}wm_loss_raw": wm_loss.detach(),
            f"{METRIC_PREFIX}wm_loss_weight": torch.as_tensor(
                wm_weight, device=wm_loss.device, dtype=torch.float32
            ),
        }
        if action_loss is not None:
            metrics[f"{METRIC_PREFIX}action_loss_raw"] = action_loss.detach()
        return metrics

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
