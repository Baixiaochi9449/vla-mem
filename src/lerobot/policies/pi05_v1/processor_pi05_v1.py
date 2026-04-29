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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05_v1.configuration_pi05_v1 import PI05V1Config
from lerobot.processor import (
	AddBatchDimensionProcessorStep,
	DeviceProcessorStep,
	NormalizerProcessorStep,
	PolicyAction,
	PolicyProcessorPipeline,
	ProcessorStep,
	ProcessorStepRegistry,
	RenameObservationsProcessorStep,
	TokenizerProcessorStep,
	UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
	OBS_IMAGE,
	OBS_IMAGES,
	OBS_STATE,
	POLICY_POSTPROCESSOR_DEFAULT_NAME,
	POLICY_PREPROCESSOR_DEFAULT_NAME,
)

MEMORY_STATE_WINDOW_KEY = "memory.state_window"
MEMORY_STATE_PAD_KEY = "memory.state_pad"
MEMORY_VALID_LENGTH_KEY = "memory.valid_length"


@ProcessorStepRegistry.register(name="pi05_v1_prepare_memory_processor_step")
@dataclass
class Pi05V1PrepareMemoryProcessorStep(ProcessorStep):
	max_state_dim: int = 32
	task_key: str = "task"

	def _get_batch_size(self, tasks: Any) -> int:
		if isinstance(tasks, str):
			return 1
		if isinstance(tasks, (list, tuple)):
			return len(tasks)
		if torch.is_tensor(tasks):
			return 1 if tasks.ndim == 0 else tasks.shape[0]
		raise ValueError(f"Unsupported task container for PI05V1: {type(tasks)!r}")

	def _split_state_window(self, state: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
		if state.ndim == 1:
			state_window = state.unsqueeze(0).unsqueeze(1)
		elif state.ndim == 2:
			if state.shape[0] == batch_size:
				state_window = state.unsqueeze(1)
			else:
				state_window = state.unsqueeze(0)
		elif state.ndim == 3:
			state_window = state
		else:
			raise ValueError(f"Unsupported state shape for PI05V1 memory preprocessing: {tuple(state.shape)}")
		return state_window, state_window[:, -1]

	def _split_image_window(self, image: torch.Tensor, batch_size: int) -> torch.Tensor:
		if image.ndim == 3:
			return image.unsqueeze(0)
		if image.ndim == 4:
			if image.shape[0] == batch_size:
				return image
			return image[-1].unsqueeze(0)
		if image.ndim == 5:
			return image[:, -1]
		raise ValueError(f"Unsupported image shape for PI05V1 memory preprocessing: {tuple(image.shape)}")

	def __call__(self, transition: EnvTransition) -> EnvTransition:
		transition = transition.copy()
		observation = dict(transition.get(TransitionKey.OBSERVATION, {}) or {})
		complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})

		state = observation.get(OBS_STATE)
		if state is None:
			raise ValueError("State is required for PI05V1")

		tasks = complementary_data.get(self.task_key)
		if tasks is None:
			raise ValueError("No task found in complementary data")

		batch_size = self._get_batch_size(tasks)
		state = deepcopy(state)
		state_window, current_state = self._split_state_window(state, batch_size)

		state_pad = complementary_data.get(f"{OBS_STATE}_is_pad")
		if state_pad is None:
			state_pad = torch.zeros(
				state_window.shape[0],
				state_window.shape[1],
				dtype=torch.bool,
				device=state_window.device,
			)
		elif state_pad.ndim == 1:
			if state_pad.shape[0] == state_window.shape[1]:
				state_pad = state_pad.unsqueeze(0)
			else:
				state_pad = state_pad.unsqueeze(1)

		for key, value in list(observation.items()):
			if key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}."):
				observation[key] = self._split_image_window(value, batch_size)

		observation[OBS_STATE] = current_state
		complementary_data[MEMORY_STATE_WINDOW_KEY] = state_window
		complementary_data[MEMORY_STATE_PAD_KEY] = state_pad
		complementary_data[MEMORY_VALID_LENGTH_KEY] = (~state_pad).sum(dim=-1)

		state_np = current_state.cpu().numpy()
		discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

		if isinstance(tasks, str):
			tasks = [tasks]
		elif isinstance(tasks, tuple):
			tasks = list(tasks)

		full_prompts = []
		for idx, task in enumerate(tasks):
			cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
			state_str = " ".join(map(str, discretized_states[idx]))
			full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

		complementary_data[self.task_key] = full_prompts
		transition[TransitionKey.OBSERVATION] = observation
		transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
		return transition

	def transform_features(
		self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
	) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
		return features


def make_pi05_v1_pre_post_processors(
	config: PI05V1Config,
	dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
	PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
	PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
	input_steps: list[ProcessorStep] = [
		RenameObservationsProcessorStep(rename_map={}),
		AddBatchDimensionProcessorStep(),
		NormalizerProcessorStep(
			features={**config.input_features, **config.output_features},
			norm_map=config.normalization_mapping,
			stats=dataset_stats,
		),
		Pi05V1PrepareMemoryProcessorStep(max_state_dim=config.max_state_dim),
		TokenizerProcessorStep(
			tokenizer_name="google/paligemma-3b-pt-224",
			max_length=config.tokenizer_max_length,
			padding_side="right",
			padding="max_length",
		),
		DeviceProcessorStep(device=config.device),
	]

	output_steps: list[ProcessorStep] = [
		UnnormalizerProcessorStep(
			features=config.output_features,
			norm_map=config.normalization_mapping,
			stats=dataset_stats,
		),
		DeviceProcessorStep(device="cpu"),
	]

	return (
		PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
			steps=input_steps,
			name=POLICY_PREPROCESSOR_DEFAULT_NAME,
		),
		PolicyProcessorPipeline[PolicyAction, PolicyAction](
			steps=output_steps,
			name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
			to_transition=policy_action_to_transition,
			to_output=transition_to_policy_action,
		),
	)
