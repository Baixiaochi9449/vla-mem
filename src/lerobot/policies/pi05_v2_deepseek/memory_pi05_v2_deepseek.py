#!/usr/bin/env python

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.policies.pi05_v2_deepseek.configuration_pi05_v2_deepseek import PI05V2DeepseekConfig


class Pi05V2DeepseekMemoryEncoder(nn.Module):
    def __init__(self, config: PI05V2DeepseekConfig, vision_dim: int, expert_dim: int):
        super().__init__()
        self.config = config
        self.expert_dim = expert_dim
        self.vision_to_expert_proj = nn.Linear(vision_dim, expert_dim)
        self.state_proj = nn.Linear(config.max_state_dim, expert_dim)
        self.camera_embedding = nn.Embedding(config.memory_max_cameras, expert_dim)
        self.slot_embedding = nn.Embedding(len(config.memory_history_deltas), expert_dim)
        self.delta_mlp = nn.Sequential(
            nn.Linear(1, expert_dim),
            nn.SiLU(),
            nn.Linear(expert_dim, expert_dim),
        )

        view_layer = nn.TransformerEncoderLayer(
            d_model=expert_dim,
            nhead=config.memory_encoder_heads,
            dim_feedforward=config.memory_ff_dim,
            batch_first=True,
            activation="gelu",
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=expert_dim,
            nhead=config.memory_encoder_heads,
            dim_feedforward=config.memory_ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.view_encoder = nn.TransformerEncoder(view_layer, num_layers=config.memory_view_encoder_layers)
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.memory_temporal_encoder_layers,
        )
        self.output_norm = nn.LayerNorm(expert_dim)

    def _masked_mean(self, values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=values.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denom

    def forward(
        self,
        vision_tokens: Tensor,
        image_mask: Tensor,
        state_window: Tensor | None,
        state_pad: Tensor | None,
        delta_indices: Tensor,
    ) -> Tensor:
        batch_size, num_steps, num_cameras = image_mask.shape
        if num_steps == 0:
            return vision_tokens.new_zeros(batch_size, self.expert_dim)

        vision_tokens = self.vision_to_expert_proj(vision_tokens)
        image_summaries = vision_tokens.mean(dim=-2)
        image_summaries = image_summaries.view(batch_size, num_steps, num_cameras, self.expert_dim)

        camera_ids = torch.arange(num_cameras, device=image_summaries.device)
        camera_emb = self.camera_embedding(camera_ids)[None, None, :, :]
        slot_ids = torch.arange(num_steps, device=image_summaries.device)
        slot_emb = self.slot_embedding(slot_ids)[None, :, None, :]
        delta_emb = self.delta_mlp(delta_indices.to(dtype=image_summaries.dtype).unsqueeze(-1))[:, :, None, :]
        image_summaries = image_summaries + camera_emb + slot_emb + delta_emb

        view_inputs = image_summaries.view(batch_size * num_steps, num_cameras, self.expert_dim)
        view_padding = ~image_mask.view(batch_size * num_steps, num_cameras)
        encoded_views = self.view_encoder(view_inputs, src_key_padding_mask=view_padding)
        timestep_tokens = self._masked_mean(encoded_views, ~view_padding).view(batch_size, num_steps, self.expert_dim)

        timestep_valid = image_mask.any(dim=-1)
        if state_window is not None and state_window.shape[1] == num_steps:
            projected_state = self.state_proj(pad_vector(state_window, self.config.max_state_dim))
            timestep_tokens = timestep_tokens + projected_state
            if state_pad is not None:
                timestep_valid = timestep_valid & ~state_pad

        temporal_padding = ~timestep_valid
        encoded_time = self.temporal_encoder(timestep_tokens, src_key_padding_mask=temporal_padding)
        latent = self._masked_mean(self.output_norm(encoded_time), ~temporal_padding)
        return latent