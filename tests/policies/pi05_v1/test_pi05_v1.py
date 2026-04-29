#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.policies.pi05_v1.configuration_pi05_v1 import PI05V1Config
from lerobot.policies.pi05_v1.memory_pi05_v1 import EpisodicMemoryBank
from lerobot.policies.pi05_v1.processor_pi05_v1 import (
	MEMORY_STATE_PAD_KEY,
	MEMORY_STATE_WINDOW_KEY,
	Pi05V1PrepareMemoryProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def test_pi05_v1_config_factory_and_observation_deltas():
	config = make_policy_config(
		policy_type="pi05_v1",
		memory_sequence_lookback=4,
		memory_slots=6,
		max_state_dim=14,
		max_action_dim=7,
	)

	assert isinstance(config, PI05V1Config)
	assert config.observation_delta_indices == [-3, -2, -1, 0]
	assert config.action_delta_indices[:3] == [0, 1, 2]


def test_pi05_v1_prepare_memory_processor_step_splits_batched_windows():
	step = Pi05V1PrepareMemoryProcessorStep(max_state_dim=6)
	transition = {
		TransitionKey.OBSERVATION: {
			OBS_STATE: torch.tensor(
				[
					[[-1.0, -0.5, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.5, 1.0]],
					[[0.2, 0.1, 0.0], [0.3, 0.2, 0.1], [0.4, 0.3, 0.2]],
				],
				dtype=torch.float32,
			),
			"observation.images.base_0_rgb": torch.rand(2, 3, 3, 8, 8),
		},
		TransitionKey.COMPLEMENTARY_DATA: {
			"task": ["pick the cube", "open the drawer"],
			f"{OBS_STATE}_is_pad": torch.tensor([[True, False, False], [False, False, False]]),
		},
	}

	processed = step(transition)
	observation = processed[TransitionKey.OBSERVATION]
	complementary_data = processed[TransitionKey.COMPLEMENTARY_DATA]

	assert observation[OBS_STATE].shape == (2, 3)
	assert complementary_data[MEMORY_STATE_WINDOW_KEY].shape == (2, 3, 3)
	assert complementary_data[MEMORY_STATE_PAD_KEY].shape == (2, 3)
	assert observation["observation.images.base_0_rgb"].shape == (2, 3, 8, 8)
	assert complementary_data["task"][0].startswith("Task: pick the cube")


def test_pi05_v1_prepare_memory_processor_step_supports_unbatched_windows_and_obs_image():
	step = Pi05V1PrepareMemoryProcessorStep(max_state_dim=6)
	transition = {
		TransitionKey.OBSERVATION: {
			OBS_STATE: torch.tensor(
				[[-0.2, 0.0, 0.1, 0.3], [-0.1, 0.2, 0.2, 0.4], [0.0, 0.3, 0.4, 0.5]],
				dtype=torch.float32,
			),
			OBS_IMAGE: torch.rand(3, 3, 8, 8),
		},
		TransitionKey.COMPLEMENTARY_DATA: {
			"task": "reach target",
			f"{OBS_STATE}_is_pad": torch.tensor([True, False, False]),
		},
	}

	processed = step(transition)
	observation = processed[TransitionKey.OBSERVATION]
	complementary_data = processed[TransitionKey.COMPLEMENTARY_DATA]

	assert observation[OBS_STATE].shape == (1, 4)
	assert observation[OBS_IMAGE].shape == (1, 3, 8, 8)
	assert complementary_data[MEMORY_STATE_WINDOW_KEY].shape == (1, 3, 4)
	assert complementary_data[MEMORY_STATE_PAD_KEY].shape == (1, 3)


def test_pi05_v1_memory_bank_rollout_and_online_update():
	config = PI05V1Config(
		max_state_dim=6,
		max_action_dim=4,
		memory_slots=4,
		memory_topk=2,
		memory_token_count=2,
		memory_key_dim=8,
		memory_value_dim=12,
		memory_phase_embed_dim=4,
		memory_phase_codebook_size=8,
		memory_sequence_lookback=3,
	)
	config.input_features = {
		OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
	}
	config.output_features = {
		ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
	}
	config.validate_features()

	memory_bank = EpisodicMemoryBank(
		config=config,
		prefix_dim=16,
		expert_dim=12,
		state_dim=6,
		action_dim=4,
	)

	batch_size = 2
	seq_len = 3
	write_summaries = torch.randn(batch_size, seq_len, 16)
	state_window = torch.randn(batch_size, seq_len, 6)
	pad_mask = torch.tensor([[True, False, False], [False, False, False]])
	prefix_summary = torch.randn(batch_size, 16)
	current_state = torch.randn(batch_size, 6)

	state, rollout_info = memory_bank.rollout_window(write_summaries, state_window, pad_mask)
	query = memory_bank.build_query(prefix_summary, current_state)
	read_result = memory_bank.read_from_state(state, query)

	assert state.keys.shape == (batch_size, 4, 8)
	assert rollout_info["last_write_key"].shape == (batch_size, 8)
	assert read_result.tokens.shape == (batch_size, 2, 16)
	assert read_result.context.shape == (batch_size, 12)
	assert read_result.future_action.shape == (batch_size, 4)

	memory_bank.update_online(prefix_summary, current_state, torch.randn(batch_size, 4))
	online_read = memory_bank.read_online(query)
	assert online_read.tokens.shape == (batch_size, 2, 16)
