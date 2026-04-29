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

from dataclasses import dataclass

from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PI05Config.register_subclass("pi05_v1")
@dataclass
class PI05V1Config(PI05Config):
	memory_enabled: bool = True
	memory_online_enabled: bool = True

	memory_slots: int = 4
	memory_key_dim: int = 256
	memory_value_dim: int = 1024
	memory_topk: int = 2
	memory_token_count: int = 2

	memory_sequence_lookback: int = 4
	memory_write_ema: float = 0.85
	memory_forget_bias: float = 0.05

	memory_phase_enabled: bool = True
	memory_phase_codebook_size: int = 64
	memory_phase_embed_dim: int = 128

	memory_future_horizon: int = 4
	memory_future_loss_weight: float = 0.2
	memory_progress_loss_weight: float = 0.05
	memory_contrastive_loss_weight: float = 0.05
	memory_sparsity_loss_weight: float = 0.001
	memory_vq_loss_weight: float = 0.01

	push_to_hub: bool = False

	def __post_init__(self):
		super().__post_init__()

		if self.memory_slots <= 0:
			raise ValueError("memory_slots must be positive")
		if self.memory_topk <= 0:
			raise ValueError("memory_topk must be positive")
		if self.memory_topk > self.memory_slots:
			raise ValueError("memory_topk cannot exceed memory_slots")
		if self.memory_token_count <= 0:
			raise ValueError("memory_token_count must be positive")
		if self.memory_sequence_lookback <= 0:
			raise ValueError("memory_sequence_lookback must be positive")
		if not 0.0 < self.memory_write_ema <= 1.0:
			raise ValueError("memory_write_ema must be in (0, 1]")
		if not 0.0 <= self.memory_forget_bias < 1.0:
			raise ValueError("memory_forget_bias must be in [0, 1)")
		if self.memory_future_horizon <= 0:
			raise ValueError("memory_future_horizon must be positive")

	@property
	def observation_delta_indices(self) -> list[int] | None:
		if not self.memory_enabled:
			return None
		return list(range(-self.memory_sequence_lookback + 1, 1))
