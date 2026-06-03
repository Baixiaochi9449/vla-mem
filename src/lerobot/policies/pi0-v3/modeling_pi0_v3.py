#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import builtins
import copy
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    GemmaConfig,
    PaliGemmaWithExpertModel,
    PI05Policy,
    PI05Pytorch,
    CONFIG_MAPPING,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    modeling_gemma,
    pad_vector,
    resize_with_pad_torch,
)
from lerobot.policies.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma,
    PiGemmaForCausalLM,
    _gated_residual,
    layernorm_forward,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_pi0_v3 import PI0V3Config
from .memory_pi0_v3 import HybridSlotMemoryBank, MemoryReadResult
from .processor_pi0_v3 import (
    MEMORY_ACTION_PAD_KEY,
    MEMORY_ACTION_WINDOW_KEY,
    MEMORY_IMAGE_PAD_PREFIX,
    MEMORY_IMAGE_WINDOW_PREFIX,
    MEMORY_STATE_PAD_KEY,
    MEMORY_STATE_WINDOW_KEY,
)

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor


@dataclass(frozen=True)
class VariantSpec:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vision_hidden_size: int | None = None
    vision_intermediate_size: int | None = None
    vision_num_hidden_layers: int | None = None
    vision_num_attention_heads: int | None = None
    vision_projection_dim: int | None = None


def get_pi0_v3_variant_spec(variant: str) -> VariantSpec:
    if variant == "gemma_tiny":
        return VariantSpec(
            width=256,
            depth=4,
            mlp_dim=1024,
            num_heads=4,
            num_kv_heads=1,
            head_dim=64,
            vision_hidden_size=256,
            vision_intermediate_size=1024,
            vision_num_hidden_layers=4,
            vision_num_attention_heads=4,
            vision_projection_dim=256,
        )
    if variant == "gemma_300m":
        return VariantSpec(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            vision_hidden_size=768,
            vision_intermediate_size=3072,
            vision_num_hidden_layers=12,
            vision_num_attention_heads=8,
            vision_projection_dim=1024,
        )
    if variant == "gemma_2b":
        return VariantSpec(
            width=2048,
            depth=18,
            mlp_dim=16384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            vision_intermediate_size=4304,
            vision_projection_dim=2048,
        )
    raise ValueError(f"Unknown variant: {variant}")


def get_pi0_v3_gemma_config(variant: str) -> GemmaConfig:
    if variant in {"gemma_300m", "gemma_2b"}:
        return get_gemma_config(variant)
    spec = get_pi0_v3_variant_spec(variant)
    return GemmaConfig(
        width=spec.width,
        depth=spec.depth,
        mlp_dim=spec.mlp_dim,
        num_heads=spec.num_heads,
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
    )


def compute_layer_complete_v3(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    num_heads = None
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
        if num_heads is None:
            num_heads = query_state.shape[1]
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    if num_heads is None:
        raise ValueError("Unable to infer attention head count for PI0V3")
    att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        current_att_output = att_output[:, start_pos:end_pos]
        if current_att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            current_att_output = current_att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(current_att_output)
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class PaliGemmaWithTemporalMemoryModel(PaliGemmaWithExpertModel):
    def __init__(
        self,
        vlm_variant: str,
        action_variant: str,
        use_adarms=None,
        precision: str = "bfloat16",
        image_size: int = 224,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        nn.Module.__init__(self)
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_spec = get_pi0_v3_variant_spec(vlm_variant)
        expert_spec = get_pi0_v3_variant_spec(action_variant)

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_spec.width
        vlm_config_hf.text_config.intermediate_size = vlm_spec.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_spec.num_heads
        vlm_config_hf.text_config.head_dim = vlm_spec.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_spec.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_spec.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_spec.width if use_adarms[0] else None

        vlm_config_hf.vision_config.image_size = image_size
        if vlm_spec.vision_hidden_size is not None:
            vlm_config_hf.vision_config.hidden_size = vlm_spec.vision_hidden_size
        if vlm_spec.vision_intermediate_size is not None:
            vlm_config_hf.vision_config.intermediate_size = vlm_spec.vision_intermediate_size
        if vlm_spec.vision_num_hidden_layers is not None:
            vlm_config_hf.vision_config.num_hidden_layers = vlm_spec.vision_num_hidden_layers
        if vlm_spec.vision_num_attention_heads is not None:
            vlm_config_hf.vision_config.num_attention_heads = vlm_spec.vision_num_attention_heads
        if vlm_spec.vision_projection_dim is not None:
            vlm_config_hf.vision_config.projection_dim = vlm_spec.vision_projection_dim
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=expert_spec.head_dim,
            hidden_size=expert_spec.width,
            intermediate_size=expert_spec.mlp_dim,
            num_attention_heads=expert_spec.num_heads,
            num_hidden_layers=expert_spec.depth,
            num_key_value_heads=expert_spec.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=expert_spec.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None or inputs_embeds[0] is None:
            return super().forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
            )

        models = [self.paligemma.model.language_model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete_v3,
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )
            else:
                inputs_embeds = compute_layer_complete_v3(
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )

        def compute_final_norms(local_inputs_embeds, local_adarms_cond):
            outputs_embeds = []
            for i, hidden_states in enumerate(local_inputs_embeds):
                out_emb, _ = layernorm_forward(models[i].norm, hidden_states, local_adarms_cond[i])
                outputs_embeds.append(out_emb)
            return outputs_embeds

        if use_gradient_checkpointing:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                inputs_embeds,
                adarms_cond,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

        return [outputs_embeds[0], outputs_embeds[1]], None


class PI0V3Pytorch(PI05Pytorch):
    def __init__(self, config: PI0V3Config, rtc_processor: RTCProcessor | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_pi0_v3_gemma_config(config.paligemma_variant)
        action_expert_config = get_pi0_v3_gemma_config(config.action_expert_variant)
        self.prefix_width = paligemma_config.width
        self.expert_width = action_expert_config.width

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )
        if self.prefix_width % config.memory_temporal_heads != 0:
            raise ValueError(
                "memory_temporal_heads must divide the prefix hidden size for the temporal transformer"
            )

        self.paligemma_with_expert = PaliGemmaWithTemporalMemoryModel(
            config.paligemma_variant,
            config.action_expert_variant,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.slot_memory = HybridSlotMemoryBank(
            config=config,
            prefix_dim=self.prefix_width,
            expert_dim=self.expert_width,
            state_dim=config.max_state_dim,
            action_dim=config.max_action_dim,
        )

        self.history_state_proj = nn.Linear(config.max_state_dim, self.prefix_width)
        self.history_delta_proj = nn.Linear(config.max_state_dim, self.prefix_width)
        self.history_action_proj = nn.Linear(config.max_action_dim, self.prefix_width)
        self.history_task_proj = nn.Linear(self.prefix_width, self.prefix_width)
        self.history_fusion = nn.Sequential(
            nn.Linear(self.prefix_width * 5, self.prefix_width),
            nn.SiLU(),
            nn.Linear(self.prefix_width, self.prefix_width),
        )
        self.history_pos_embedding = nn.Parameter(
            torch.zeros(1, config.memory_sequence_lookback, self.prefix_width)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.prefix_width,
            nhead=config.memory_temporal_heads,
            dim_feedforward=max(self.prefix_width * 4, 512),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.memory_temporal_layers,
            enable_nested_tensor=False,
        )
        self.recent_token_proj = nn.Linear(self.prefix_width, self.prefix_width)
        self.recent_context_proj = nn.Linear(self.prefix_width, self.expert_width)
        self.recent_future_action_head = nn.Sequential(
            nn.Linear(self.prefix_width, self.expert_width),
            nn.SiLU(),
            nn.Linear(self.expert_width, config.max_action_dim),
        )
        self.slot_recent_context_gate = nn.Linear(self.expert_width * 2, self.expert_width)
        self.slot_context_gate = nn.Linear(self.expert_width * 2, self.expert_width)
        self.memory_time_gate = nn.Linear(self.expert_width * 2, self.expert_width)

        self.gradient_checkpointing_enabled = False
        self._align_memory_precision(config.dtype)

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _align_memory_precision(self, precision: str) -> None:
        if precision == "bfloat16":
            target_dtype = torch.bfloat16
        elif precision == "float32":
            target_dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision: {precision}")

        memory_modules = [
            self.slot_memory,
            self.history_state_proj,
            self.history_delta_proj,
            self.history_action_proj,
            self.history_task_proj,
            self.history_fusion,
            self.history_temporal_encoder,
            self.recent_token_proj,
            self.recent_context_proj,
            self.recent_future_action_head,
            self.slot_recent_context_gate,
            self.slot_context_gate,
            self.memory_time_gate,
        ]
        for module in memory_modules:
            module.to(dtype=target_dtype)
        self.history_pos_embedding.data = self.history_pos_embedding.data.to(dtype=target_dtype)

    def reset_memory(self) -> None:
        self.slot_memory.reset_online_state()

    def _masked_mean(self, embeddings: Tensor, pad_mask: Tensor) -> Tensor:
        weights = (~pad_mask).to(dtype=embeddings.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (embeddings * weights).sum(dim=1) / denom

    def _pool_prefix_summary(self, prefix_embs: Tensor, prefix_pad_masks: Tensor) -> Tensor:
        return self._masked_mean(prefix_embs, ~prefix_pad_masks)

    def _build_task_summary(self, tokens: Tensor, masks: Tensor) -> Tensor:
        task_embs = self.paligemma_with_expert.embed_language_tokens(tokens)
        task_embs = task_embs * math.sqrt(task_embs.shape[-1])
        return self._masked_mean(task_embs, ~masks)

    def _append_memory_to_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        memory_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if memory_tokens.numel() == 0:
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        memory_tokens = memory_tokens.to(dtype=prefix_embs.dtype)
        memory_pad_masks = torch.ones(
            memory_tokens.shape[0],
            memory_tokens.shape[1],
            dtype=torch.bool,
            device=memory_tokens.device,
        )
        memory_att_masks = torch.zeros(
            memory_tokens.shape[0],
            memory_tokens.shape[1],
            dtype=prefix_att_masks.dtype,
            device=prefix_att_masks.device,
        )

        return (
            torch.cat([prefix_embs, memory_tokens], dim=1),
            torch.cat([prefix_pad_masks, memory_pad_masks], dim=1),
            torch.cat([prefix_att_masks, memory_att_masks], dim=1),
        )

    def _combine_slot_results(
        self,
        sequence_result: MemoryReadResult | None,
        online_result: MemoryReadResult | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        zero_tokens = torch.zeros(batch_size, 0, self.prefix_width, device=device, dtype=dtype)
        zero_context = torch.zeros(batch_size, self.expert_width, device=device, dtype=dtype)
        zero_key = torch.zeros(batch_size, self.config.memory_key_dim, device=device, dtype=dtype)
        zero_action = torch.zeros(batch_size, self.config.max_action_dim, device=device, dtype=dtype)

        if sequence_result is None and online_result is None:
            return zero_tokens, zero_context, {
                "query": zero_key,
                "predicted_future_action": zero_action,
                "read_sparsity_loss": torch.zeros((), device=device, dtype=dtype),
            }

        if sequence_result is not None and online_result is not None:
            seq_tokens = sequence_result.tokens
            online_tokens = online_result.tokens
            half_tokens = max(self.config.memory_token_count // 2, 1)
            seq_tokens = seq_tokens[:, :half_tokens]
            online_tokens = online_tokens[:, : max(self.config.memory_token_count - half_tokens, 0)]
            slot_tokens = torch.cat([seq_tokens, online_tokens], dim=1)
            gate = torch.sigmoid(
                self.slot_context_gate(torch.cat([sequence_result.context, online_result.context], dim=-1))
            )
            slot_context = gate * sequence_result.context + (1.0 - gate) * online_result.context
            predicted_future_action = 0.5 * (sequence_result.future_action + online_result.future_action)
            read_sparsity_loss = 0.5 * (sequence_result.sparsity_loss + online_result.sparsity_loss)
            query = 0.5 * (sequence_result.query + online_result.query)
        else:
            source = sequence_result if sequence_result is not None else online_result
            assert source is not None
            slot_tokens = source.tokens
            slot_context = source.context
            predicted_future_action = source.future_action
            read_sparsity_loss = source.sparsity_loss
            query = source.query

        if slot_tokens.shape[1] > self.config.memory_token_count:
            slot_tokens = slot_tokens[:, : self.config.memory_token_count]

        return slot_tokens, slot_context, {
            "query": query,
            "predicted_future_action": predicted_future_action,
            "read_sparsity_loss": read_sparsity_loss,
        }

    def _align_action_history(
        self,
        state_window: Tensor,
        action_window: Tensor | None,
        action_pad: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, window_len = state_window.shape[:2]
        aligned_actions = state_window.new_zeros(batch_size, window_len, self.config.max_action_dim)
        aligned_pad = torch.ones(batch_size, window_len, dtype=torch.bool, device=state_window.device)

        if action_window is None or action_window.numel() == 0:
            aligned_pad[:, -1] = False
            return aligned_actions, aligned_pad

        padded_action_window = pad_vector(action_window, self.config.max_action_dim)
        hist_len = min(padded_action_window.shape[1], max(window_len - 1, 0))
        if hist_len > 0:
            aligned_actions[:, :hist_len] = padded_action_window[:, -hist_len:]
            if action_pad is None:
                aligned_pad[:, :hist_len] = False
            else:
                aligned_pad[:, :hist_len] = action_pad[:, -hist_len:]
        aligned_pad[:, -1] = False
        return aligned_actions, aligned_pad

    def _encode_image_history(
        self,
        image_windows: dict[str, Tensor] | None,
        image_pad_masks: dict[str, Tensor] | None,
        batch_size: int,
        window_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if not self.config.memory_use_image_history or not image_windows:
            return torch.zeros(batch_size, window_len, self.prefix_width, device=device, dtype=dtype)

        camera_summaries = []
        camera_valids = []
        for key, window in image_windows.items():
            if window.ndim == 4:
                window = window.unsqueeze(1)
            if window.shape[1] != window_len:
                raise ValueError(
                    f"History image window length mismatch for {key}: expected {window_len}, got {window.shape[1]}"
                )
            window = window.to(device=device)
            flat_window = window.reshape(batch_size * window_len, *window.shape[2:])

            def image_embed_func(flat_images):
                return self.paligemma_with_expert.embed_image(flat_images)

            image_embs = self._apply_checkpoint(image_embed_func, flat_window)
            step_summary = image_embs.mean(dim=1).reshape(batch_size, window_len, -1).to(dtype=dtype)
            camera_summaries.append(step_summary)

            if image_pad_masks is not None and key in image_pad_masks:
                valid = ~image_pad_masks[key].to(device=device)
            else:
                valid = torch.ones(batch_size, window_len, dtype=torch.bool, device=device)
            camera_valids.append(valid)

        stacked = torch.stack(camera_summaries, dim=0)
        valid_weights = torch.stack(camera_valids, dim=0).to(dtype=dtype).unsqueeze(-1)
        denom = valid_weights.sum(dim=0).clamp_min(1.0)
        return (stacked * valid_weights).sum(dim=0) / denom

    def _encode_history_steps(
        self,
        state_window: Tensor,
        state_pad: Tensor,
        action_window: Tensor | None,
        action_pad: Tensor | None,
        image_windows: dict[str, Tensor] | None,
        image_pad_masks: dict[str, Tensor] | None,
        tokens: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, window_len = state_window.shape[:2]
        padded_state_window = pad_vector(state_window, self.config.max_state_dim)
        prev_state = torch.cat([padded_state_window[:, :1], padded_state_window[:, :-1]], dim=1)
        delta_state = padded_state_window - prev_state
        aligned_actions, aligned_action_pad = self._align_action_history(
            padded_state_window,
            action_window,
            action_pad,
        )

        task_summary = self.history_task_proj(self._build_task_summary(tokens, masks)).unsqueeze(1)
        task_summary = task_summary.expand(-1, window_len, -1)
        image_summary = self._encode_image_history(
            image_windows=image_windows,
            image_pad_masks=image_pad_masks,
            batch_size=batch_size,
            window_len=window_len,
            dtype=padded_state_window.dtype,
            device=padded_state_window.device,
        )

        fused_inputs = torch.cat(
            [
                image_summary,
                self.history_state_proj(padded_state_window),
                self.history_delta_proj(delta_state),
                self.history_action_proj(aligned_actions),
                task_summary,
            ],
            dim=-1,
        )
        fused = self.history_fusion(fused_inputs)
        fused = fused + self.history_pos_embedding[:, -window_len:].to(dtype=fused.dtype, device=fused.device)

        causal_mask = torch.triu(
            torch.ones(window_len, window_len, dtype=torch.bool, device=fused.device),
            diagonal=1,
        )
        encoded = self.history_temporal_encoder(
            fused,
            mask=causal_mask,
            src_key_padding_mask=state_pad,
        )
        encoded = encoded.masked_fill(state_pad.unsqueeze(-1), 0.0)
        return encoded, aligned_actions, aligned_action_pad

    def _extract_recent_tokens(
        self,
        encoded_steps: Tensor,
        state_pad: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, _, hidden_size = encoded_steps.shape
        valid_lengths = (~state_pad).sum(dim=-1)

        recent_indices = []
        recent_valids = []
        for offset in range(self.config.memory_recent_tokens):
            index = (valid_lengths - self.config.memory_recent_tokens + offset).clamp(min=0)
            valid = valid_lengths > (self.config.memory_recent_tokens - offset - 1)
            recent_indices.append(index)
            recent_valids.append(valid)

        recent_indices = torch.stack(recent_indices, dim=1)
        recent_valids = torch.stack(recent_valids, dim=1)
        recent_tokens = torch.gather(
            encoded_steps,
            dim=1,
            index=recent_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        recent_tokens = recent_tokens * recent_valids.unsqueeze(-1).to(dtype=encoded_steps.dtype)
        recent_summary = recent_tokens.sum(dim=1) / recent_valids.sum(dim=1, keepdim=True).clamp_min(1).to(
            dtype=encoded_steps.dtype
        )
        return self.recent_token_proj(recent_tokens), self.recent_context_proj(recent_summary), recent_summary

    def _build_memory_conditioning(
        self,
        prefix_summary: Tensor,
        current_state: Tensor,
        state_window: Tensor,
        state_pad: Tensor,
        action_window: Tensor | None,
        action_pad: Tensor | None,
        image_windows: dict[str, Tensor] | None,
        image_pad_masks: dict[str, Tensor] | None,
        tokens: Tensor,
        masks: Tensor,
        use_online_memory: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        memory_dtype = next(self.slot_memory.parameters()).dtype
        prefix_summary = prefix_summary.to(dtype=memory_dtype)
        current_state = pad_vector(current_state, self.config.max_state_dim).to(dtype=memory_dtype)
        state_window = pad_vector(state_window, self.config.max_state_dim).to(dtype=memory_dtype)
        state_pad = state_pad.to(device=state_window.device)
        if action_window is not None:
            action_window = pad_vector(action_window, self.config.max_action_dim).to(dtype=memory_dtype)
        if action_pad is not None:
            action_pad = action_pad.to(device=state_window.device)

        encoded_steps, aligned_actions, _aligned_action_pad = self._encode_history_steps(
            state_window=state_window,
            state_pad=state_pad,
            action_window=action_window,
            action_pad=action_pad,
            image_windows=image_windows,
            image_pad_masks=image_pad_masks,
            tokens=tokens,
            masks=masks,
        )
        recent_tokens, recent_context, recent_summary = self._extract_recent_tokens(encoded_steps, state_pad)

        query = self.slot_memory.build_query(prefix_summary, current_state)

        sequence_result = None
        sequence_rollout = {
            "vq_loss": prefix_summary.new_zeros(()),
            "last_write_key": prefix_summary.new_zeros(prefix_summary.shape[0], self.config.memory_key_dim),
        }
        if self.config.memory_enabled and encoded_steps.shape[1] > 0:
            sequence_state, sequence_rollout = self.slot_memory.rollout_window(
                write_summaries=encoded_steps,
                state_window=state_window,
                pad_mask=state_pad,
                action_summaries=aligned_actions,
            )
            sequence_result = self.slot_memory.read_from_state(sequence_state, query)

        online_result = None
        if self.config.memory_enabled and self.config.memory_online_enabled and use_online_memory:
            online_result = self.slot_memory.read_online(query)

        slot_tokens, slot_context, slot_metrics = self._combine_slot_results(
            sequence_result=sequence_result,
            online_result=online_result,
            batch_size=prefix_summary.shape[0],
            device=prefix_summary.device,
            dtype=prefix_summary.dtype,
        )

        if slot_context.numel() == 0:
            slot_context = prefix_summary.new_zeros(prefix_summary.shape[0], self.expert_width)
        gate = torch.sigmoid(self.slot_recent_context_gate(torch.cat([slot_context, recent_context], dim=-1)))
        memory_context = gate * slot_context + (1.0 - gate) * recent_context
        memory_tokens = torch.cat([recent_tokens, slot_tokens], dim=1)

        recent_future_action = self.recent_future_action_head(recent_summary)
        if slot_metrics["predicted_future_action"].numel() == 0:
            predicted_future_action = recent_future_action
        else:
            predicted_future_action = 0.5 * (slot_metrics["predicted_future_action"] + recent_future_action)

        metrics = {
            "query": slot_metrics["query"],
            "predicted_future_action": predicted_future_action,
            "read_sparsity_loss": slot_metrics["read_sparsity_loss"],
            "vq_loss": sequence_rollout["vq_loss"],
            "last_write_key": sequence_rollout["last_write_key"],
            "memory_write_summary": recent_summary,
        }
        return memory_tokens, memory_context, metrics

    def embed_suffix(self, noisy_actions, timestep, memory_context: Tensor | None = None):
        embs, pad_masks, att_masks, adarms_cond = super().embed_suffix(noisy_actions, timestep)
        if memory_context is not None:
            memory_context = memory_context.to(dtype=adarms_cond.dtype)
            gate = torch.sigmoid(self.memory_time_gate(torch.cat([adarms_cond, memory_context], dim=-1)))
            adarms_cond = adarms_cond + gate * memory_context
        return embs, pad_masks, att_masks, adarms_cond

    def _compute_memory_losses(
        self,
        metrics: dict[str, Tensor],
        actions: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        total_aux_loss = actions.new_zeros(())
        logs: dict[str, float] = {}

        predicted_future_action = metrics.get("predicted_future_action")
        if predicted_future_action is not None and self.config.memory_future_loss_weight > 0:
            future_action_target = actions[:, 0].to(dtype=torch.float32)
            future_loss = F.mse_loss(predicted_future_action.to(dtype=torch.float32), future_action_target)
            total_aux_loss = total_aux_loss + self.config.memory_future_loss_weight * future_loss
            logs["memory_future_loss"] = future_loss.item()

        read_sparsity_loss = metrics.get("read_sparsity_loss")
        if read_sparsity_loss is not None and self.config.memory_sparsity_loss_weight > 0:
            total_aux_loss = total_aux_loss + self.config.memory_sparsity_loss_weight * read_sparsity_loss
            logs["memory_sparsity_loss"] = read_sparsity_loss.item()

        vq_loss = metrics.get("vq_loss")
        if vq_loss is not None and self.config.memory_vq_loss_weight > 0:
            total_aux_loss = total_aux_loss + self.config.memory_vq_loss_weight * vq_loss
            logs["memory_vq_loss"] = vq_loss.item()

        last_write_key = metrics.get("last_write_key")
        current_query = metrics.get("query")
        if (
            last_write_key is not None
            and current_query is not None
            and current_query.shape[0] > 1
            and self.config.memory_contrastive_loss_weight > 0
        ):
            logits = torch.matmul(current_query, last_write_key.transpose(0, 1)) / math.sqrt(
                current_query.shape[-1]
            )
            targets = torch.arange(logits.shape[0], device=logits.device)
            contrastive_loss = F.cross_entropy(logits, targets)
            total_aux_loss = total_aux_loss + self.config.memory_contrastive_loss_weight * contrastive_loss
            logs["memory_contrastive_loss"] = contrastive_loss.item()

        logs["memory_aux_loss"] = total_aux_loss.item()
        return total_aux_loss, logs

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        current_state,
        state_window,
        state_pad,
        image_windows,
        image_pad_masks,
        action_window=None,
        action_pad=None,
        noise=None,
        time=None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_summary = self._pool_prefix_summary(prefix_embs, prefix_pad_masks)
        memory_tokens, memory_context, memory_metrics = self._build_memory_conditioning(
            prefix_summary=prefix_summary,
            current_state=current_state,
            state_window=state_window,
            state_pad=state_pad,
            action_window=action_window,
            action_pad=action_pad,
            image_windows=image_windows,
            image_pad_masks=image_pad_masks,
            tokens=tokens,
            masks=masks,
            use_online_memory=False,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_memory_to_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            memory_tokens,
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t,
            time,
            memory_context,
        )

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        aux_loss, aux_logs = self._compute_memory_losses(memory_metrics, actions)
        losses = losses + aux_loss / max(losses.shape[1] * losses.shape[2], 1)

        aux = {
            "memory_aux_loss": losses.new_tensor(aux_logs.get("memory_aux_loss", 0.0)),
            "memory_future_loss": losses.new_tensor(aux_logs.get("memory_future_loss", 0.0)),
            "memory_sparsity_loss": losses.new_tensor(aux_logs.get("memory_sparsity_loss", 0.0)),
            "memory_contrastive_loss": losses.new_tensor(aux_logs.get("memory_contrastive_loss", 0.0)),
            "memory_vq_loss": losses.new_tensor(aux_logs.get("memory_vq_loss", 0.0)),
        }
        return losses, aux

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        current_state,
        state_window,
        state_pad,
        image_windows,
        image_pad_masks,
        action_window=None,
        action_pad=None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        batch_size = tokens.shape[0]
        device = tokens.device
        current_state = pad_vector(current_state, self.config.max_state_dim)
        state_window = pad_vector(state_window, self.config.max_state_dim)

        if noise is None:
            actions_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_summary = self._pool_prefix_summary(prefix_embs, prefix_pad_masks)
        memory_tokens, memory_context, memory_metrics = self._build_memory_conditioning(
            prefix_summary=prefix_summary,
            current_state=current_state,
            state_window=state_window,
            state_pad=state_pad,
            action_window=action_window,
            action_pad=action_pad,
            image_windows=image_windows,
            image_pad_masks=image_pad_masks,
            tokens=tokens,
            masks=masks,
            use_online_memory=True,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_memory_to_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            memory_tokens,
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    memory_context=memory_context,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        if self.config.memory_enabled and self.config.memory_online_enabled:
            action_summary = x_t.mean(dim=1)
            self.slot_memory.update_online(memory_metrics["memory_write_summary"], current_state, action_summary)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        memory_context: Tensor | None = None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t,
            timestep,
            memory_context,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        past_key_values = copy.deepcopy(past_key_values)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI0V3Policy(PI05Policy):
    config_class = PI0V3Config
    name = "pi0_v3"

    def __init__(self, config: PI0V3Config, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI0V3Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        return PreTrainedPolicy.from_pretrained.__func__(
            cls,
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def reset(self):
        super().reset()
        self.model.reset_memory()
        self._state_history: deque[Tensor] = deque(maxlen=self.config.memory_sequence_lookback)
        self._action_history: deque[Tensor] = deque(maxlen=self.config.memory_action_history_length)
        self._image_history: dict[str, deque[Tensor]] = {
            key: deque(maxlen=self.config.memory_sequence_lookback) for key in self.config.image_features
        }
        self._online_batch_size: int | None = None

    def _maybe_reset_online_buffers(self, batch_size: int) -> None:
        if self._online_batch_size is None:
            self._online_batch_size = batch_size
            return
        if self._online_batch_size != batch_size:
            self.reset()
            self._online_batch_size = batch_size

    def _preprocess_single_image_batch(self, img: Tensor) -> Tensor:
        device = next(self.parameters()).device
        if img.device != device:
            img = img.to(device)
        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        is_channels_first = img.shape[1] == 3
        if is_channels_first:
            img = img.permute(0, 2, 3, 1)

        if img.shape[1:3] != self.config.image_resolution:
            img = resize_with_pad_torch(img, *self.config.image_resolution)

        img = img * 2.0 - 1.0
        if is_channels_first:
            img = img.permute(0, 3, 1, 2)
        return img

    def _preprocess_named_image_windows(
        self,
        image_windows: dict[str, Tensor],
        image_pad_masks: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if not image_windows:
            return {}, {}

        device = next(self.parameters()).device
        processed_windows: dict[str, Tensor] = {}
        processed_pads: dict[str, Tensor] = {}
        for key, window in image_windows.items():
            if window.ndim == 4:
                window = window.unsqueeze(1)
            batch_size, window_len = window.shape[:2]
            flat_window = window.reshape(batch_size * window_len, *window.shape[2:])
            processed_flat = self._preprocess_single_image_batch(flat_window)
            processed_windows[key] = processed_flat.reshape(batch_size, window_len, *processed_flat.shape[1:])
            pad_mask = image_pad_masks.get(key)
            if pad_mask is None:
                pad_mask = torch.zeros(batch_size, window_len, dtype=torch.bool, device=device)
            else:
                pad_mask = pad_mask.to(device=device)
            processed_pads[key] = pad_mask

        return processed_windows, processed_pads

    def _stack_tensor_history(
        self,
        history: deque[Tensor],
        window_len: int,
        reference: Tensor,
        pad_mode: str,
    ) -> tuple[Tensor, Tensor]:
        batch_size = reference.shape[0]
        device = reference.device
        if len(history) == 0:
            pad_value = reference if pad_mode == "edge" else torch.zeros_like(reference)
            stacked = torch.stack([pad_value for _ in range(window_len)], dim=1)
            pad_mask = torch.ones(batch_size, window_len, dtype=torch.bool, device=device)
            return stacked, pad_mask

        values = list(history)[-window_len:]
        pad_count = max(window_len - len(values), 0)
        pad_value = values[0] if pad_mode == "edge" else torch.zeros_like(reference)
        padded_values = [pad_value for _ in range(pad_count)] + values
        padded_values = padded_values[-window_len:]
        stacked = torch.stack(padded_values, dim=1)
        pad_mask = torch.zeros(batch_size, window_len, dtype=torch.bool, device=device)
        if pad_count > 0:
            pad_mask[:, :pad_count] = True
        return stacked, pad_mask

    def _update_online_history(self, batch: dict[str, Tensor]) -> None:
        current_state = batch[OBS_STATE]
        if current_state.ndim == 3:
            current_state = current_state[:, -1]
        self._maybe_reset_online_buffers(current_state.shape[0])
        self._state_history.append(current_state.detach())

        for key in self.config.image_features:
            if key not in batch:
                continue
            current_image = batch[key]
            if current_image.ndim == 5:
                current_image = current_image[:, -1]
            self._image_history[key].append(current_image.detach())

    def _extract_training_memory_inputs(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], dict[str, Tensor], Tensor | None, Tensor | None]:
        current_state = batch[OBS_STATE]
        if current_state.ndim == 3:
            current_state = current_state[:, -1]

        state_window = batch.get(MEMORY_STATE_WINDOW_KEY)
        if state_window is None:
            state_window = current_state.unsqueeze(1)

        state_pad = batch.get(MEMORY_STATE_PAD_KEY)
        if state_pad is None:
            state_pad = torch.zeros(
                state_window.shape[0],
                state_window.shape[1],
                dtype=torch.bool,
                device=state_window.device,
            )

        image_windows = {
            key.removeprefix(MEMORY_IMAGE_WINDOW_PREFIX): value
            for key, value in batch.items()
            if key.startswith(MEMORY_IMAGE_WINDOW_PREFIX)
        }
        image_pad_masks = {
            key.removeprefix(MEMORY_IMAGE_PAD_PREFIX): value
            for key, value in batch.items()
            if key.startswith(MEMORY_IMAGE_PAD_PREFIX)
        }
        action_window = batch.get(MEMORY_ACTION_WINDOW_KEY)
        action_pad = batch.get(MEMORY_ACTION_PAD_KEY)
        return current_state, state_window, state_pad, image_windows, image_pad_masks, action_window, action_pad

    def _extract_online_memory_inputs(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], dict[str, Tensor], Tensor | None, Tensor | None]:
        current_state = batch[OBS_STATE]
        if current_state.ndim == 3:
            current_state = current_state[:, -1]

        state_window, state_pad = self._stack_tensor_history(
            self._state_history,
            self.config.memory_sequence_lookback,
            current_state.detach(),
            pad_mode="edge",
        )

        image_windows: dict[str, Tensor] = {}
        image_pad_masks: dict[str, Tensor] = {}
        for key in self.config.image_features:
            if key not in batch:
                continue
            current_image = batch[key]
            if current_image.ndim == 5:
                current_image = current_image[:, -1]
            stacked_window, pad_mask = self._stack_tensor_history(
                self._image_history[key],
                self.config.memory_sequence_lookback,
                current_image.detach(),
                pad_mode="edge",
            )
            image_windows[key] = stacked_window
            image_pad_masks[key] = pad_mask

        if self.config.memory_action_history_length > 0:
            reference_action = torch.zeros(
                current_state.shape[0],
                self.config.output_features[ACTION].shape[0],
                dtype=current_state.dtype,
                device=current_state.device,
            )
            action_window, action_pad = self._stack_tensor_history(
                self._action_history,
                self.config.memory_action_history_length,
                reference_action,
                pad_mode="zeros",
            )
        else:
            action_window, action_pad = None, None

        return current_state, state_window, state_pad, image_windows, image_pad_masks, action_window, action_pad

    def _predict_action_chunk_internal(
        self,
        batch: dict[str, Tensor],
        update_online_history: bool,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        self.eval()
        if update_online_history:
            self._update_online_history(batch)

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        (
            current_state,
            state_window,
            state_pad,
            image_windows,
            image_pad_masks,
            action_window,
            action_pad,
        ) = self._extract_online_memory_inputs(batch)
        image_windows, image_pad_masks = self._preprocess_named_image_windows(image_windows, image_pad_masks)

        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            current_state=current_state,
            state_window=state_window,
            state_pad=state_pad,
            image_windows=image_windows,
            image_pad_masks=image_pad_masks,
            action_window=action_window,
            action_pad=action_pad,
            **kwargs,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        return self._predict_action_chunk_internal(batch, update_online_history=True, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        self._update_online_history(batch)
        if len(self._action_queue) == 0:
            actions = self._predict_action_chunk_internal(batch, update_online_history=False, **kwargs)
            actions = actions[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        action = self._action_queue.popleft()
        self._action_history.append(action.detach())
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        (
            current_state,
            state_window,
            state_pad,
            image_windows,
            image_pad_masks,
            action_window,
            action_pad,
        ) = self._extract_training_memory_inputs(batch)
        image_windows, image_pad_masks = self._preprocess_named_image_windows(image_windows, image_pad_masks)
        actions = self.prepare_action(batch)

        losses, memory_aux = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            current_state=current_state,
            state_window=state_window,
            state_pad=state_pad,
            image_windows=image_windows,
            image_pad_masks=image_pad_masks,
            action_window=action_window,
            action_pad=action_pad,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "memory_aux_loss": memory_aux["memory_aux_loss"].item(),
            "memory_future_loss": memory_aux["memory_future_loss"].item(),
            "memory_sparsity_loss": memory_aux["memory_sparsity_loss"].item(),
            "memory_contrastive_loss": memory_aux["memory_contrastive_loss"].item(),
            "memory_vq_loss": memory_aux["memory_vq_loss"].item(),
        }

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict