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

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .configuration_pi05_v1 import PI05V1Config


@dataclass
class MemoryState:
	keys: Tensor
	values: Tensor
	phase_ids: Tensor
	ages: Tensor
	confidences: Tensor
	filled: Tensor


@dataclass
class MemoryReadResult:
	tokens: Tensor
	context: Tensor
	weights: Tensor
	indices: Tensor
	query: Tensor
	aggregate: Tensor
	future_action: Tensor
	progress_logits: Tensor
	sparsity_loss: Tensor


class PhaseCodebook(nn.Module):
	def __init__(self, enabled: bool, value_dim: int, codebook_size: int, embed_dim: int):
		super().__init__()
		self.enabled = enabled
		self.embed_dim = embed_dim
		if enabled:
			self.proj = nn.Linear(value_dim, embed_dim)
			self.embedding = nn.Embedding(codebook_size, embed_dim)
		else:
			self.proj = None
			self.embedding = None

	def quantize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		if not self.enabled or self.proj is None or self.embedding is None:
			phase_embed = values.new_zeros(*values.shape[:-1], self.embed_dim)
			phase_ids = torch.zeros(*values.shape[:-1], dtype=torch.long, device=values.device)
			return phase_embed, phase_ids, values.new_zeros(())

		projected = self.proj(values)
		codebook = self.embedding.weight
		distances = torch.cdist(projected.reshape(-1, self.embed_dim), codebook)
		phase_ids = distances.argmin(dim=-1).reshape(*values.shape[:-1])
		quantized = self.embedding(phase_ids)

		commitment_loss = F.mse_loss(projected, quantized.detach())
		codebook_loss = F.mse_loss(projected.detach(), quantized)
		vq_loss = codebook_loss + 0.25 * commitment_loss
		quantized = projected + (quantized - projected).detach()
		return quantized, phase_ids, vq_loss

	def lookup(self, phase_ids: Tensor, dtype: torch.dtype) -> Tensor:
		if not self.enabled or self.embedding is None:
			return torch.zeros(*phase_ids.shape, self.embed_dim, device=phase_ids.device, dtype=dtype)
		return self.embedding(phase_ids).to(dtype=dtype)


class EpisodicMemoryBank(nn.Module):
	def __init__(
		self,
		config: PI05V1Config,
		prefix_dim: int,
		expert_dim: int,
		state_dim: int,
		action_dim: int,
	):
		super().__init__()
		self.config = config
		self.prefix_dim = prefix_dim
		self.expert_dim = expert_dim
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.read_query_proj = nn.Sequential(
			nn.Linear(prefix_dim + state_dim, prefix_dim),
			nn.SiLU(),
			nn.Linear(prefix_dim, config.memory_key_dim),
		)
		self.write_key_proj = nn.Sequential(
			nn.Linear(prefix_dim + state_dim + action_dim, prefix_dim),
			nn.SiLU(),
			nn.Linear(prefix_dim, config.memory_key_dim),
		)
		self.write_value_proj = nn.Sequential(
			nn.Linear(prefix_dim + state_dim + action_dim, prefix_dim),
			nn.SiLU(),
			nn.Linear(prefix_dim, config.memory_value_dim),
		)

		self.phase_codebook = PhaseCodebook(
			enabled=config.memory_phase_enabled,
			value_dim=config.memory_value_dim,
			codebook_size=config.memory_phase_codebook_size,
			embed_dim=config.memory_phase_embed_dim,
		)

		token_in_dim = config.memory_value_dim + config.memory_phase_embed_dim + 2
		context_in_dim = config.memory_value_dim + config.memory_phase_embed_dim
		self.memory_token_proj = nn.Linear(token_in_dim, prefix_dim)
		self.aggregate_token_proj = nn.Linear(context_in_dim, prefix_dim)
		self.memory_context_proj = nn.Linear(context_in_dim, expert_dim)
		self.future_action_head = nn.Sequential(
			nn.Linear(context_in_dim, expert_dim),
			nn.SiLU(),
			nn.Linear(expert_dim, action_dim),
		)
		self.progress_head = nn.Sequential(
			nn.Linear(context_in_dim, max(expert_dim // 2, 128)),
			nn.SiLU(),
			nn.Linear(max(expert_dim // 2, 128), 1),
		)

		self._online_state: MemoryState | None = None

	def reset_online_state(self) -> None:
		self._online_state = None

	def new_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
		return MemoryState(
			keys=torch.zeros(batch_size, self.config.memory_slots, self.config.memory_key_dim, device=device, dtype=dtype),
			values=torch.zeros(
				batch_size,
				self.config.memory_slots,
				self.config.memory_value_dim,
				device=device,
				dtype=dtype,
			),
			phase_ids=torch.zeros(batch_size, self.config.memory_slots, device=device, dtype=torch.long),
			ages=torch.zeros(batch_size, self.config.memory_slots, device=device, dtype=torch.long),
			confidences=torch.zeros(batch_size, self.config.memory_slots, device=device, dtype=dtype),
			filled=torch.zeros(batch_size, self.config.memory_slots, device=device, dtype=torch.bool),
		)

	def ensure_online_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
		if self._online_state is None:
			self._online_state = self.new_state(batch_size, device, dtype)
		elif self._online_state.keys.shape[0] != batch_size:
			self._online_state = self.new_state(batch_size, device, dtype)
		elif self._online_state.keys.device != device or self._online_state.keys.dtype != dtype:
			self._online_state = self.new_state(batch_size, device, dtype)
		return self._online_state

	def build_query(self, prefix_summary: Tensor, current_state: Tensor) -> Tensor:
		query_inputs = torch.cat([prefix_summary, current_state], dim=-1)
		return F.normalize(self.read_query_proj(query_inputs), dim=-1, eps=1e-6)

	def encode_write(
		self,
		write_summary: Tensor,
		current_state: Tensor,
		action_summary: Tensor | None = None,
	) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
		if action_summary is None:
			action_summary = write_summary.new_zeros(write_summary.shape[0], self.action_dim)

		write_inputs = torch.cat([write_summary, current_state, action_summary], dim=-1)
		write_key = F.normalize(self.write_key_proj(write_inputs), dim=-1, eps=1e-6)
		write_value = self.write_value_proj(write_inputs)
		phase_embed, phase_ids, vq_loss = self.phase_codebook.quantize(write_value)
		return write_key, write_value, phase_embed, phase_ids, vq_loss

	def _gather_last_dim(self, tensor: Tensor, indices: Tensor) -> Tensor:
		expanded = indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
		return torch.gather(tensor, dim=1, index=expanded)

	def write_step(
		self,
		state: MemoryState,
		write_summary: Tensor,
		current_state: Tensor,
		action_summary: Tensor | None = None,
		valid_mask: Tensor | None = None,
	) -> tuple[MemoryState, dict[str, Tensor]]:
		batch_size = write_summary.shape[0]
		dtype = write_summary.dtype
		if valid_mask is None:
			valid_mask = torch.ones(batch_size, dtype=torch.bool, device=write_summary.device)

		write_key, write_value, _phase_embed, phase_ids, vq_loss = self.encode_write(
			write_summary=write_summary,
			current_state=current_state,
			action_summary=action_summary,
		)

		logits = torch.einsum("bd,bsd->bs", write_key, state.keys)
		logits = logits.masked_fill(~state.filled, -1e4)
		has_filled = state.filled.any(dim=-1)
		has_empty = (~state.filled).any(dim=-1)
		max_scores, best_match_idx = logits.max(dim=-1)
		first_empty_idx = (~state.filled).float().argmax(dim=-1)
		use_empty = (~has_filled) | ((max_scores < 0.5) & has_empty)
		slot_idx = torch.where(use_empty, first_empty_idx, best_match_idx)

		slot_mask = F.one_hot(slot_idx, num_classes=self.config.memory_slots).to(dtype=dtype)
		slot_mask = slot_mask * valid_mask[:, None].to(dtype=dtype)
		slot_mask_3d = slot_mask.unsqueeze(-1)
		selected_filled = (state.filled.to(dtype=dtype) * slot_mask).sum(dim=-1) > 0

		selected_keys = (state.keys * slot_mask_3d).sum(dim=1)
		selected_values = (state.values * slot_mask_3d).sum(dim=1)
		selected_confidences = (state.confidences * slot_mask).sum(dim=-1)

		ema = self.config.memory_write_ema
		updated_selected_keys = torch.where(
			selected_filled[:, None],
			F.normalize(ema * selected_keys + (1.0 - ema) * write_key, dim=-1, eps=1e-6),
			write_key,
		)
		updated_selected_values = torch.where(
			selected_filled[:, None],
			ema * selected_values + (1.0 - ema) * write_value,
			write_value,
		)

		new_keys = state.keys * (1.0 - slot_mask_3d) + updated_selected_keys[:, None, :] * slot_mask_3d
		new_values = state.values * (1.0 - slot_mask_3d) + updated_selected_values[:, None, :] * slot_mask_3d

		base_confidences = state.confidences * (1.0 - self.config.memory_forget_bias)
		updated_selected_confidences = torch.where(
			selected_filled,
			torch.clamp(selected_confidences + 0.5, max=1.0),
			torch.ones_like(selected_confidences),
		)
		new_confidences = base_confidences * (1.0 - slot_mask) + updated_selected_confidences[:, None] * slot_mask

		new_phase_ids = state.phase_ids * (1 - slot_mask.to(dtype=torch.long)) + phase_ids[:, None] * slot_mask.to(
			dtype=torch.long
		)
		new_ages = state.ages + valid_mask[:, None].to(dtype=torch.long)
		new_ages = new_ages * (1 - slot_mask.to(dtype=torch.long))
		new_filled = state.filled | slot_mask.bool()

		return (
			MemoryState(
				keys=new_keys,
				values=new_values,
				phase_ids=new_phase_ids,
				ages=new_ages,
				confidences=new_confidences,
				filled=new_filled,
			),
			{
				"write_key": write_key,
				"write_value": write_value,
				"vq_loss": vq_loss,
			},
		)

	def rollout_window(
		self,
		write_summaries: Tensor,
		state_window: Tensor,
		pad_mask: Tensor,
		action_summaries: Tensor | None = None,
	) -> tuple[MemoryState, dict[str, Tensor]]:
		batch_size = write_summaries.shape[0]
		state = self.new_state(batch_size, write_summaries.device, write_summaries.dtype)
		last_write_key = write_summaries.new_zeros(batch_size, self.config.memory_key_dim)
		vq_losses = []

		for step_idx in range(write_summaries.shape[1]):
			step_action_summary = None
			if action_summaries is not None:
				step_action_summary = action_summaries[:, step_idx]

			state, write_info = self.write_step(
				state=state,
				write_summary=write_summaries[:, step_idx],
				current_state=state_window[:, step_idx],
				action_summary=step_action_summary,
				valid_mask=~pad_mask[:, step_idx],
			)
			last_write_key = torch.where(
				(~pad_mask[:, step_idx]).unsqueeze(-1),
				write_info["write_key"],
				last_write_key,
			)
			vq_losses.append(write_info["vq_loss"])

		vq_loss = torch.stack(vq_losses).mean() if vq_losses else write_summaries.new_zeros(())
		return state, {"vq_loss": vq_loss, "last_write_key": last_write_key}

	def read_from_state(self, state: MemoryState, query: Tensor) -> MemoryReadResult:
		topk = min(self.config.memory_topk, self.config.memory_slots)
		logits = torch.einsum("bd,bsd->bs", query, state.keys) / math.sqrt(self.config.memory_key_dim)
		logits = logits.masked_fill(~state.filled, -1e4)
		has_memory = state.filled.any(dim=-1)
		safe_logits = torch.where(has_memory[:, None], logits, torch.zeros_like(logits))
		topk_scores, topk_idx = torch.topk(safe_logits, k=topk, dim=-1)

		topk_valid = torch.gather(state.filled, dim=1, index=topk_idx)
		masked_topk_scores = topk_scores.masked_fill(~topk_valid, -1e4)
		weights = torch.softmax(masked_topk_scores, dim=-1)
		weights = weights * topk_valid.to(dtype=query.dtype)
		weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

		selected_values = self._gather_last_dim(state.values, topk_idx)
		selected_phase_ids = torch.gather(state.phase_ids, dim=1, index=topk_idx)
		selected_ages = torch.gather(state.ages, dim=1, index=topk_idx)
		selected_confidences = torch.gather(state.confidences, dim=1, index=topk_idx)
		selected_phase_embeddings = self.phase_codebook.lookup(selected_phase_ids, query.dtype)

		max_age = torch.clamp(state.ages.max(dim=-1, keepdim=True).values + 1, min=1)
		normalized_ages = selected_ages.to(dtype=query.dtype) / max_age.to(dtype=query.dtype)
		token_features = torch.cat(
			[
				selected_values,
				selected_phase_embeddings,
				normalized_ages.unsqueeze(-1),
				selected_confidences.unsqueeze(-1),
			],
			dim=-1,
		)
		tokens = self.memory_token_proj(token_features)

		aggregate_feature = torch.sum(
			weights.unsqueeze(-1) * torch.cat([selected_values, selected_phase_embeddings], dim=-1),
			dim=1,
		)
		context = self.memory_context_proj(aggregate_feature)
		aggregate_token = self.aggregate_token_proj(aggregate_feature).unsqueeze(1)

		if self.config.memory_token_count > tokens.shape[1]:
			pad_count = self.config.memory_token_count - tokens.shape[1]
			tokens = torch.cat([tokens, aggregate_token.expand(-1, pad_count, -1)], dim=1)
		elif self.config.memory_token_count < tokens.shape[1]:
			tokens = tokens[:, : self.config.memory_token_count]

		entropy = -(weights * torch.log(weights.clamp_min(1e-6))).sum(dim=-1)
		normalizer = math.log(max(topk, 2))
		sparsity_loss = (entropy / normalizer).mean()

		return MemoryReadResult(
			tokens=tokens,
			context=context,
			weights=weights,
			indices=topk_idx,
			query=query,
			aggregate=aggregate_feature,
			future_action=self.future_action_head(aggregate_feature),
			progress_logits=self.progress_head(aggregate_feature).squeeze(-1),
			sparsity_loss=sparsity_loss,
		)

	def read_online(self, query: Tensor) -> MemoryReadResult:
		online_state = self.ensure_online_state(query.shape[0], query.device, query.dtype)
		return self.read_from_state(online_state, query)

	def update_online(self, write_summary: Tensor, current_state: Tensor, action_summary: Tensor | None = None) -> None:
		online_state = self.ensure_online_state(write_summary.shape[0], write_summary.device, write_summary.dtype)
		new_state, _ = self.write_step(
			state=online_state,
			write_summary=write_summary,
			current_state=current_state,
			action_summary=action_summary,
			valid_mask=torch.ones(write_summary.shape[0], dtype=torch.bool, device=write_summary.device),
		)
		self._online_state = new_state
