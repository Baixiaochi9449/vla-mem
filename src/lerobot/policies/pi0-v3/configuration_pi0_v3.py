#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05_v1.configuration_pi05_v1 import PI05V1Config


@PI05Config.register_subclass("pi0_v3")
@dataclass
class PI0V3Config(PI05V1Config):
    memory_recent_tokens: int = 4
    memory_temporal_layers: int = 2
    memory_temporal_heads: int = 4
    memory_use_action_history: bool = True
    memory_use_image_history: bool = True
    memory_observation_dropout: float = 0.0

    def __post_init__(self):
        PreTrainedConfig.__post_init__(self)

        allowed_variants = {"gemma_tiny", "gemma_300m", "gemma_2b"}

        if self.n_action_steps > self.chunk_size:
            if self.n_action_steps == PI05Config.n_action_steps:
                self.n_action_steps = self.chunk_size
            else:
                raise ValueError(
                    f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
                )

        if self.paligemma_variant not in allowed_variants:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in allowed_variants:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

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
        if self.memory_recent_tokens <= 0:
            raise ValueError("memory_recent_tokens must be positive")
        if self.memory_temporal_layers <= 0:
            raise ValueError("memory_temporal_layers must be positive")
        if self.memory_temporal_heads <= 0:
            raise ValueError("memory_temporal_heads must be positive")
        if not 0.0 < self.memory_write_ema <= 1.0:
            raise ValueError("memory_write_ema must be in (0, 1]")
        if not 0.0 <= self.memory_forget_bias < 1.0:
            raise ValueError("memory_forget_bias must be in [0, 1)")
        if self.memory_future_horizon <= 0:
            raise ValueError("memory_future_horizon must be positive")
        if not 0.0 <= self.memory_observation_dropout < 1.0:
            raise ValueError("memory_observation_dropout must be in [0, 1)")

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if not self.memory_enabled:
            return None
        return list(range(-self.memory_sequence_lookback + 1, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        if not self.memory_enabled or not self.memory_use_action_history:
            return list(range(self.chunk_size))
        return list(range(-self.memory_sequence_lookback + 1, self.chunk_size))

    @property
    def memory_action_history_length(self) -> int:
        return max(self.memory_sequence_lookback - 1, 0)

    @property
    def total_memory_token_count(self) -> int:
        return self.memory_recent_tokens + self.memory_token_count