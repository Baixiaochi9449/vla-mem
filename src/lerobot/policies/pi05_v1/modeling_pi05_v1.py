#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import builtins
import copy
import math
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
	ActionSelectKwargs,
	PI05Policy,
	PI05Pytorch,
	get_gemma_config,
	make_att_2d_masks,
	pad_vector,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_pi05_v1 import PI05V1Config
from .memory_pi05_v1 import EpisodicMemoryBank, MemoryReadResult
from .processor_pi05_v1 import MEMORY_STATE_PAD_KEY, MEMORY_STATE_WINDOW_KEY

if TYPE_CHECKING:
	from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class PI05V1Pytorch(PI05Pytorch):
	def __init__(self, config: PI05V1Config, rtc_processor: "RTCProcessor" | None = None):
		compile_requested = config.compile_model
		if compile_requested:
			config.compile_model = False
		super().__init__(config, rtc_processor=rtc_processor)
		config.compile_model = compile_requested

		paligemma_config = get_gemma_config(config.paligemma_variant)
		action_expert_config = get_gemma_config(config.action_expert_variant)
		self.prefix_width = paligemma_config.width
		self.expert_width = action_expert_config.width

		self.memory_bank = EpisodicMemoryBank(
			config=config,
			prefix_dim=self.prefix_width,
			expert_dim=self.expert_width,
			state_dim=config.max_state_dim,
			action_dim=config.max_action_dim,
		)
		self.memory_state_proj = nn.Linear(config.max_state_dim, self.prefix_width)
		self.memory_delta_proj = nn.Linear(config.max_state_dim, self.prefix_width)
		self.memory_task_proj = nn.Linear(self.prefix_width, self.prefix_width)
		self.memory_window_fusion = nn.Sequential(
			nn.Linear(self.prefix_width * 3, self.prefix_width),
			nn.SiLU(),
			nn.Linear(self.prefix_width, self.prefix_width),
		)
		self.memory_context_gate = nn.Linear(self.expert_width * 2, self.expert_width)
		self._align_memory_precision(config.dtype)

		if compile_requested:
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
			self.memory_bank,
			self.memory_state_proj,
			self.memory_delta_proj,
			self.memory_task_proj,
			self.memory_window_fusion,
			self.memory_context_gate,
		]
		for module in memory_modules:
			module.to(dtype=target_dtype)

	def reset_memory(self) -> None:
		self.memory_bank.reset_online_state()

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

	def _encode_window_summaries(
		self,
		state_window: Tensor,
		state_pad: Tensor,
		tokens: Tensor,
		masks: Tensor,
	) -> Tensor:
		padded_state_window = pad_vector(state_window, self.config.max_state_dim)
		prev_state = torch.cat([padded_state_window[:, :1], padded_state_window[:, :-1]], dim=1)
		delta_state = padded_state_window - prev_state
		task_summary = self.memory_task_proj(self._build_task_summary(tokens, masks)).unsqueeze(1)
		task_summary = task_summary.expand(-1, padded_state_window.shape[1], -1)
		features = torch.cat(
			[
				self.memory_state_proj(padded_state_window),
				self.memory_delta_proj(delta_state),
				task_summary,
			],
			dim=-1,
		)
		summaries = self.memory_window_fusion(features)
		return summaries.masked_fill(state_pad.unsqueeze(-1), 0.0)

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

	def _combine_memory_results(
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
			memory_tokens = torch.cat([seq_tokens, online_tokens], dim=1)

			gate = torch.sigmoid(
				self.memory_context_gate(torch.cat([sequence_result.context, online_result.context], dim=-1))
			)
			memory_context = gate * sequence_result.context + (1.0 - gate) * online_result.context
			predicted_future_action = 0.5 * (sequence_result.future_action + online_result.future_action)
			read_sparsity_loss = 0.5 * (sequence_result.sparsity_loss + online_result.sparsity_loss)
			query = 0.5 * (sequence_result.query + online_result.query)
		else:
			source = sequence_result if sequence_result is not None else online_result
			assert source is not None
			memory_tokens = source.tokens
			memory_context = source.context
			predicted_future_action = source.future_action
			read_sparsity_loss = source.sparsity_loss
			query = source.query

		if memory_tokens.shape[1] > self.config.memory_token_count:
			memory_tokens = memory_tokens[:, : self.config.memory_token_count]

		return memory_tokens, memory_context, {
			"query": query,
			"predicted_future_action": predicted_future_action,
			"read_sparsity_loss": read_sparsity_loss,
		}

	def _build_memory_conditioning(
		self,
		prefix_summary: Tensor,
		current_state: Tensor,
		state_window: Tensor,
		state_pad: Tensor,
		tokens: Tensor,
		masks: Tensor,
		use_online_memory: bool,
	) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
		memory_dtype = next(self.memory_bank.parameters()).dtype
		prefix_summary = prefix_summary.to(dtype=memory_dtype)
		current_state = pad_vector(current_state, self.config.max_state_dim).to(dtype=memory_dtype)
		state_window = pad_vector(state_window, self.config.max_state_dim).to(dtype=memory_dtype)
		query = self.memory_bank.build_query(prefix_summary, current_state)

		sequence_result = None
		sequence_rollout = {
			"vq_loss": prefix_summary.new_zeros(()),
			"last_write_key": prefix_summary.new_zeros(prefix_summary.shape[0], self.config.memory_key_dim),
		}
		if self.config.memory_enabled and state_window.shape[1] > 0:
			write_summaries = self._encode_window_summaries(state_window, state_pad, tokens, masks)
			sequence_state, sequence_rollout = self.memory_bank.rollout_window(
				write_summaries=write_summaries,
				state_window=state_window,
				pad_mask=state_pad,
			)
			sequence_result = self.memory_bank.read_from_state(sequence_state, query)

		online_result = None
		if self.config.memory_enabled and self.config.memory_online_enabled and use_online_memory:
			online_result = self.memory_bank.read_online(query)

		memory_tokens, memory_context, read_metrics = self._combine_memory_results(
			sequence_result=sequence_result,
			online_result=online_result,
			batch_size=prefix_summary.shape[0],
			device=prefix_summary.device,
			dtype=prefix_summary.dtype,
		)

		metrics = {
			**read_metrics,
			"vq_loss": sequence_rollout["vq_loss"],
			"last_write_key": sequence_rollout["last_write_key"],
		}
		return memory_tokens, memory_context, metrics

	def embed_suffix(self, noisy_actions, timestep, memory_context: Tensor | None = None):
		embs, pad_masks, att_masks, adarms_cond = super().embed_suffix(noisy_actions, timestep)
		if memory_context is not None:
			adarms_cond = adarms_cond + memory_context.to(dtype=adarms_cond.dtype)
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
		state_window=None,
		state_pad=None,
		noise=None,
		num_steps=None,
		**kwargs: Unpack[ActionSelectKwargs],
	) -> Tensor:
		if num_steps is None:
			num_steps = self.config.num_inference_steps

		batch_size = tokens.shape[0]
		device = tokens.device
		current_state = pad_vector(current_state, self.config.max_state_dim)
		if state_window is None:
			state_window = current_state.unsqueeze(1)
		else:
			state_window = pad_vector(state_window, self.config.max_state_dim)
		if state_pad is None:
			state_pad = torch.zeros(batch_size, state_window.shape[1], dtype=torch.bool, device=device)

		if noise is None:
			actions_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
			noise = self.sample_noise(actions_shape, device)

		prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
		prefix_summary = self._pool_prefix_summary(prefix_embs, prefix_pad_masks)
		memory_tokens, memory_context, _memory_metrics = self._build_memory_conditioning(
			prefix_summary=prefix_summary,
			current_state=current_state,
			state_window=state_window,
			state_pad=state_pad,
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
			self.memory_bank.update_online(prefix_summary, current_state, action_summary)

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


class PI05V1Policy(PI05Policy):
	config_class = PI05V1Config
	name = "pi05_v1"

	def __init__(self, config: PI05V1Config, **kwargs):
		PreTrainedPolicy.__init__(self, config)
		config.validate_features()
		self.config = config

		self.init_rtc_processor()
		self.model = PI05V1Pytorch(config, rtc_processor=self.rtc_processor)
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
		return super().from_pretrained(
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

	def _extract_memory_inputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
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

		return current_state, state_window, state_pad

	@torch.no_grad()
	def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
		self.eval()
		images, img_masks = self._preprocess_images(batch)
		tokens = batch[OBS_LANGUAGE_TOKENS]
		masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
		current_state, state_window, state_pad = self._extract_memory_inputs(batch)

		actions = self.model.sample_actions(
			images,
			img_masks,
			tokens,
			masks,
			current_state=current_state,
			state_window=state_window,
			state_pad=state_pad,
			**kwargs,
		)
		original_action_dim = self.config.output_features[ACTION].shape[0]
		return actions[:, :, :original_action_dim]

	def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
		images, img_masks = self._preprocess_images(batch)
		tokens = batch[OBS_LANGUAGE_TOKENS]
		masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
		current_state, state_window, state_pad = self._extract_memory_inputs(batch)
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
