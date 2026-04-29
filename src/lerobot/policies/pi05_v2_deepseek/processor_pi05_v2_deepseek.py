#!/usr/bin/env python

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05_v2_deepseek.configuration_pi05_v2_deepseek import PI05V2DeepseekConfig
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

MEMORY_IMAGES_WINDOW_KEY = "memory.images_window"
MEMORY_IMAGE_MASK_KEY = "memory.image_mask"
MEMORY_STATE_WINDOW_KEY = "memory.state_window"
MEMORY_STATE_PAD_KEY = "memory.state_pad"
MEMORY_DELTA_INDEX_KEY = "memory.delta_index"


@ProcessorStepRegistry.register(name="pi05_v2_deepseek_prepare_memory_processor_step")
@dataclass
class Pi05V2DeepseekPrepareMemoryProcessorStep(ProcessorStep):
    max_state_dim: int = 32
    history_deltas: tuple[int, ...] = (-12, -4, -1)
    max_cameras: int = 3
    task_key: str = "task"

    def _get_batch_size(self, tasks: str | list[str]) -> int:
        if isinstance(tasks, str):
            return 1
        return len(tasks)

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
            raise ValueError(f"Unsupported state shape for PI05V2Deepseek: {tuple(state.shape)}")
        return state_window[:, :-1], state_window[:, -1]

    def _split_image_window(self, image: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim == 3:
            current = image.unsqueeze(0)
            return current.new_zeros((1, 0, *current.shape[1:])), current

        if image.ndim == 4:
            is_batched_current = image.shape[0] == batch_size and (image.shape[1] in {1, 3} or image.shape[-1] in {1, 3})
            if is_batched_current:
                return image.new_zeros((image.shape[0], 0, *image.shape[1:])), image
            return image[:-1].unsqueeze(0), image[-1].unsqueeze(0)

        if image.ndim == 5:
            return image[:, :-1], image[:, -1]

        raise ValueError(f"Unsupported image shape for PI05V2Deepseek: {tuple(image.shape)}")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}) or {})
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})

        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05V2Deepseek")

        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")
        batch_size = self._get_batch_size(tasks)

        state = deepcopy(state)
        history_state_window, current_state = self._split_state_window(state, batch_size)

        state_pad = complementary_data.get(f"{OBS_STATE}_is_pad")
        if state_pad is None:
            state_pad = torch.zeros(
                history_state_window.shape[0],
                history_state_window.shape[1],
                dtype=torch.bool,
                device=history_state_window.device,
            )
        else:
            if state_pad.ndim == 1:
                state_pad = state_pad.unsqueeze(0)
            state_pad = state_pad[:, :-1]

        image_windows = []
        image_masks = []
        camera_keys = []
        for key, value in list(observation.items()):
            if key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}."):
                history_window, current_image = self._split_image_window(value, batch_size)
                observation[key] = current_image
                image_windows.append(history_window)
                image_masks.append(
                    torch.ones(
                        history_window.shape[0],
                        history_window.shape[1],
                        dtype=torch.bool,
                        device=history_window.device,
                    )
                )
                camera_keys.append(key)

        if image_windows:
            while len(image_windows) < self.max_cameras:
                reference = image_windows[0]
                image_windows.append(reference.new_zeros((reference.shape[0], reference.shape[1], *reference.shape[2:])))
                image_masks.append(
                    torch.zeros(reference.shape[0], reference.shape[1], dtype=torch.bool, device=reference.device)
                )
            image_windows = image_windows[: self.max_cameras]
            image_masks = image_masks[: self.max_cameras]
            complementary_data[MEMORY_IMAGES_WINDOW_KEY] = torch.stack(image_windows, dim=2)
            complementary_data[MEMORY_IMAGE_MASK_KEY] = torch.stack(image_masks, dim=2)
        else:
            complementary_data[MEMORY_IMAGES_WINDOW_KEY] = None
            complementary_data[MEMORY_IMAGE_MASK_KEY] = None

        observation[OBS_STATE] = current_state
        complementary_data[MEMORY_STATE_WINDOW_KEY] = history_state_window
        complementary_data[MEMORY_STATE_PAD_KEY] = state_pad
        complementary_data[MEMORY_DELTA_INDEX_KEY] = torch.tensor(
            self.history_deltas,
            dtype=torch.long,
            device=current_state.device,
        ).unsqueeze(0).expand(current_state.shape[0], -1)

        state_np = current_state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        if isinstance(tasks, str):
            tasks = [tasks]

        full_prompts = []
        for idx, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[idx]))
            full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

        complementary_data[self.task_key] = full_prompts
        complementary_data["memory.camera_keys"] = camera_keys
        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi05_v2_deepseek_pre_post_processors(
    config: PI05V2DeepseekConfig,
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
        Pi05V2DeepseekPrepareMemoryProcessorStep(
            max_state_dim=config.max_state_dim,
            history_deltas=tuple(config.memory_history_deltas),
            max_cameras=config.memory_max_cameras,
        ),
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
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
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