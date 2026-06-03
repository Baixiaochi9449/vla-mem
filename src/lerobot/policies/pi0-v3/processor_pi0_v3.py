#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations
import os
_PALIGEMMA_TOKENIZER_NAME = os.environ.get("PALIGEMMA_TOKENIZER_PATH", "google/paligemma-3b-pt-224")


from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
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
    ACTION,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_pi0_v3 import PI0V3Config

MEMORY_STATE_WINDOW_KEY = "memory.state_window"
MEMORY_STATE_PAD_KEY = "memory.state_pad"
MEMORY_VALID_LENGTH_KEY = "memory.valid_length"
MEMORY_ACTION_WINDOW_KEY = "memory.action_window"
MEMORY_ACTION_PAD_KEY = "memory.action_pad"
MEMORY_IMAGE_WINDOW_PREFIX = "memory.image_window."
MEMORY_IMAGE_PAD_PREFIX = "memory.image_pad."


@ProcessorStepRegistry.register(name="pi0_v3_prepare_memory_processor_step")
@dataclass
class Pi0V3PrepareMemoryProcessorStep(ProcessorStep):
    max_state_dim: int = 32
    history_action_length: int = 0
    chunk_size: int = 50
    task_key: str = "task"

    def _get_batch_size(self, tasks: Any) -> int:
        if isinstance(tasks, str):
            return 1
        if isinstance(tasks, (list, tuple)):
            return len(tasks)
        if torch.is_tensor(tasks):
            return 1 if tasks.ndim == 0 else tasks.shape[0]
        raise ValueError(f"Unsupported task container for PI0V3: {type(tasks)!r}")

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
            raise ValueError(f"Unsupported state shape for PI0V3 memory preprocessing: {tuple(state.shape)}")
        return state_window, state_window[:, -1]

    def _split_image_window(self, image: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim == 3:
            image_window = image.unsqueeze(0).unsqueeze(1)
            current_image = image.unsqueeze(0)
        elif image.ndim == 4:
            if image.shape[0] == batch_size:
                image_window = image.unsqueeze(1)
                current_image = image
            else:
                image_window = image.unsqueeze(0)
                current_image = image[-1].unsqueeze(0)
        elif image.ndim == 5:
            image_window = image
            current_image = image[:, -1]
        else:
            raise ValueError(f"Unsupported image shape for PI0V3 memory preprocessing: {tuple(image.shape)}")
        return image_window, current_image

    def _normalize_pad_mask(self, pad_mask: torch.Tensor | None, batch_size: int, seq_len: int) -> torch.Tensor:
        if pad_mask is None:
            return torch.zeros(batch_size, seq_len, dtype=torch.bool)
        if pad_mask.ndim == 1:
            if pad_mask.shape[0] == seq_len:
                return pad_mask.unsqueeze(0)
            return pad_mask.unsqueeze(1)
        if pad_mask.ndim == 2:
            return pad_mask
        raise ValueError(f"Unsupported pad mask shape for PI0V3: {tuple(pad_mask.shape)}")

    def _split_action_sequence(
        self,
        action: torch.Tensor | None,
        batch_size: int,
        action_pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if action is None:
            return None, None, None, None

        if action.ndim == 1:
            action_seq = action.unsqueeze(0).unsqueeze(1)
        elif action.ndim == 2:
            if action.shape[0] == batch_size:
                action_seq = action.unsqueeze(1)
            else:
                action_seq = action.unsqueeze(0)
        elif action.ndim == 3:
            action_seq = action
        else:
            raise ValueError(f"Unsupported action shape for PI0V3 memory preprocessing: {tuple(action.shape)}")

        batch_size, seq_len = action_seq.shape[:2]
        pad_mask = self._normalize_pad_mask(action_pad, batch_size, seq_len)

        if self.history_action_length > 0 and seq_len > self.chunk_size:
            history_window = action_seq[:, : self.history_action_length]
            future_action = action_seq[:, self.history_action_length :]
            history_pad = pad_mask[:, : self.history_action_length]
            future_pad = pad_mask[:, self.history_action_length :]
        else:
            history_window = action_seq.new_zeros(batch_size, self.history_action_length, action_seq.shape[-1])
            future_action = action_seq
            history_pad = torch.ones(batch_size, self.history_action_length, dtype=torch.bool, device=action_seq.device)
            future_pad = pad_mask

        return history_window, history_pad, future_action, future_pad

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}) or {})
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})

        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI0V3")

        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        batch_size = self._get_batch_size(tasks)
        state = deepcopy(state)
        state_window, current_state = self._split_state_window(state, batch_size)

        state_pad = complementary_data.get(f"{OBS_STATE}_is_pad")
        state_pad = self._normalize_pad_mask(state_pad, state_window.shape[0], state_window.shape[1]).to(
            device=state_window.device
        )

        for key, value in list(observation.items()):
            if key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}."):
                image_window, current_image = self._split_image_window(deepcopy(value), batch_size)
                observation[key] = current_image
                complementary_data[f"{MEMORY_IMAGE_WINDOW_PREFIX}{key}"] = image_window

                image_pad_key = f"{key}_is_pad"
                image_pad = complementary_data.get(image_pad_key)
                if image_pad is None:
                    image_pad = state_pad.clone()
                else:
                    image_pad = self._normalize_pad_mask(image_pad, image_window.shape[0], image_window.shape[1]).to(
                        device=image_window.device
                    )
                complementary_data[f"{MEMORY_IMAGE_PAD_PREFIX}{key}"] = image_pad

        action = transition.get(TransitionKey.ACTION)
        action_pad = complementary_data.get(f"{ACTION}_is_pad")
        history_action, history_action_pad, future_action, future_action_pad = self._split_action_sequence(
            action,
            batch_size,
            action_pad,
        )
        if future_action is not None:
            transition[TransitionKey.ACTION] = future_action
            complementary_data[f"{ACTION}_is_pad"] = future_action_pad
        if history_action is not None:
            complementary_data[MEMORY_ACTION_WINDOW_KEY] = history_action
            complementary_data[MEMORY_ACTION_PAD_KEY] = history_action_pad

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


def make_pi0_v3_pre_post_processors(
    config: PI0V3Config,
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
        Pi0V3PrepareMemoryProcessorStep(
            max_state_dim=config.max_state_dim,
            history_action_length=config.memory_action_history_length,
            chunk_size=config.chunk_size,
        ),
        TokenizerProcessorStep(
            tokenizer_name=_PALIGEMMA_TOKENIZER_NAME,
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