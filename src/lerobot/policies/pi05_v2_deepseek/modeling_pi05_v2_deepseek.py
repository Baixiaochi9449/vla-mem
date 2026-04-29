#!/usr/bin/env python

from __future__ import annotations

from collections import deque
from typing import Unpack

import torch
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    PI05Pytorch,
    get_gemma_config,
    resize_with_pad_torch,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_pi05_v2_deepseek import PI05V2DeepseekConfig
from .dynamic_lora_pi05_v2_deepseek import DynamicLoRABasisHyperNet, DynamicLoRALinear, DynamicLoRATargetSpec
from .memory_pi05_v2_deepseek import Pi05V2DeepseekMemoryEncoder
from .processor_pi05_v2_deepseek import (
    MEMORY_DELTA_INDEX_KEY,
    MEMORY_IMAGE_MASK_KEY,
    MEMORY_IMAGES_WINDOW_KEY,
    MEMORY_STATE_PAD_KEY,
    MEMORY_STATE_WINDOW_KEY,
)


class PI05V2DeepseekPytorch(PI05Pytorch):
    def __init__(self, config: PI05V2DeepseekConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)
        self.prefix_width = paligemma_config.width
        self.expert_width = action_expert_config.width

        self.memory_encoder = Pi05V2DeepseekMemoryEncoder(
            config=config,
            vision_dim=self.prefix_width,
            expert_dim=self.expert_width,
        )

        target_specs = self._install_dynamic_lora_wrappers(config)
        self.dynamic_lora_hypernet = DynamicLoRABasisHyperNet(
            latent_dim=self.expert_width,
            hidden_dim=config.lora_hidden_dim,
            rank=config.lora_rank,
            basis_count=config.lora_basis_count,
            target_specs=target_specs,
            scale_init=config.lora_scale_init,
        )

    def _install_dynamic_lora_wrappers(self, config: PI05V2DeepseekConfig) -> list[DynamicLoRATargetSpec]:
        target_specs: list[DynamicLoRATargetSpec] = []
        self._dynamic_lora_wrappers: dict[str, DynamicLoRALinear] = {}
        for layer_idx in config.lora_target_layers:
            layer = self.paligemma_with_expert.gemma_expert.model.layers[layer_idx].self_attn
            for module_name in config.lora_target_modules:
                source_linear = getattr(layer, module_name)
                target_key = f"expert.layers.{layer_idx}.{module_name}"
                wrapper = DynamicLoRALinear(source_linear, target_key=target_key)
                setattr(layer, module_name, wrapper)
                self._dynamic_lora_wrappers[target_key] = wrapper
                target_specs.append(
                    DynamicLoRATargetSpec(
                        key=target_key,
                        in_features=wrapper.in_features,
                        out_features=wrapper.out_features,
                    )
                )
        return target_specs

    def clear_dynamic_lora(self) -> None:
        for wrapper in self._dynamic_lora_wrappers.values():
            wrapper.set_dynamic_lora(None)

    def set_memory_context(
        self,
        memory_images_window: Tensor | None,
        memory_image_mask: Tensor | None,
        memory_state_window: Tensor | None,
        memory_state_pad: Tensor | None,
        memory_delta_indices: Tensor | None,
    ) -> Tensor:
        self.clear_dynamic_lora()
        if (
            memory_images_window is None
            or memory_image_mask is None
            or memory_delta_indices is None
            or memory_images_window.shape[1] == 0
        ):
            batch_size = 1 if memory_state_window is None else memory_state_window.shape[0]
            return self.action_in_proj.weight.new_zeros(batch_size, self.expert_width)

        batch_size, num_steps, num_cameras = memory_image_mask.shape
        flat_images = memory_images_window.reshape(
            batch_size * num_steps * num_cameras,
            *memory_images_window.shape[3:],
        )
        flat_mask = memory_image_mask.reshape(-1)
        if not flat_mask.any():
            return self.action_in_proj.weight.new_zeros(batch_size, self.expert_width)

        vision_tokens = self.paligemma_with_expert.embed_image(flat_images)
        if self.config.memory_detach_vision_features:
            vision_tokens = vision_tokens.detach()

        latent = self.memory_encoder(
            vision_tokens=vision_tokens,
            image_mask=memory_image_mask,
            state_window=memory_state_window,
            state_pad=memory_state_pad,
            delta_indices=memory_delta_indices,
        )
        params = self.dynamic_lora_hypernet(latent)
        for key, wrapper in self._dynamic_lora_wrappers.items():
            wrapper.set_dynamic_lora(params.get(key))
        return latent

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        memory_images_window: Tensor | None = None,
        memory_image_mask: Tensor | None = None,
        memory_state_window: Tensor | None = None,
        memory_state_pad: Tensor | None = None,
        memory_delta_indices: Tensor | None = None,
        noise=None,
        time=None,
    ) -> Tensor:
        self.set_memory_context(
            memory_images_window=memory_images_window,
            memory_image_mask=memory_image_mask,
            memory_state_window=memory_state_window,
            memory_state_pad=memory_state_pad,
            memory_delta_indices=memory_delta_indices,
        )
        return super().forward(images, img_masks, tokens, masks, actions, noise=noise, time=time)

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        memory_images_window: Tensor | None = None,
        memory_image_mask: Tensor | None = None,
        memory_state_window: Tensor | None = None,
        memory_state_pad: Tensor | None = None,
        memory_delta_indices: Tensor | None = None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        self.set_memory_context(
            memory_images_window=memory_images_window,
            memory_image_mask=memory_image_mask,
            memory_state_window=memory_state_window,
            memory_state_pad=memory_state_pad,
            memory_delta_indices=memory_delta_indices,
        )
        return super().sample_actions(images, img_masks, tokens, masks, noise=noise, num_steps=num_steps, **kwargs)


class PI05V2DeepseekPolicy(PI05Policy):
    config_class = PI05V2DeepseekConfig
    name = "pi05_v2_deepseek"

    def __init__(self, config: PI05V2DeepseekConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05V2DeepseekPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        history_size = max(abs(min(self.config.memory_history_deltas)), 1) + 1
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
            OBS_STATE: deque(maxlen=history_size),
        }
        for key in self.config.image_features:
            self._queues[key] = deque(maxlen=history_size)
        self.model.clear_dynamic_lora()

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        normalized_batch = batch
        updated_images: dict[str, Tensor] = {}
        for key in self.config.image_features:
            if key not in batch:
                continue
            image = batch[key]
            if isinstance(image, Tensor) and image.ndim == 5:
                updated_images[key] = image[:, -1]

        if updated_images:
            normalized_batch = dict(batch)
            normalized_batch.update(updated_images)

        return super()._preprocess_images(normalized_batch)

    def _preprocess_memory_images_window(
        self,
        memory_images_window: Tensor | None,
        memory_image_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if memory_images_window is None or memory_image_mask is None:
            return None, None
        images = memory_images_window
        if images.numel() == 0:
            return images, memory_image_mask

        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        if images.shape[-1] == 3:
            images = images.permute(0, 1, 2, 5, 3, 4)

        batch_size, num_steps, num_cameras = images.shape[:3]
        flat_images = images.reshape(batch_size * num_steps * num_cameras, *images.shape[3:])
        if flat_images.shape[-2:] != self.config.image_resolution:
            flat_images = resize_with_pad_torch(flat_images, *self.config.image_resolution)
        images = flat_images.reshape(batch_size, num_steps, num_cameras, *flat_images.shape[1:])

        images = images * 2.0 - 1.0
        return images, memory_image_mask

    def _extract_memory_from_batch(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        memory_images_window = batch.get(MEMORY_IMAGES_WINDOW_KEY)
        memory_image_mask = batch.get(MEMORY_IMAGE_MASK_KEY)
        memory_images_window, memory_image_mask = self._preprocess_memory_images_window(
            memory_images_window,
            memory_image_mask,
        )
        return (
            memory_images_window,
            memory_image_mask,
            batch.get(MEMORY_STATE_WINDOW_KEY),
            batch.get(MEMORY_STATE_PAD_KEY),
            batch.get(MEMORY_DELTA_INDEX_KEY),
        )

    def _build_online_memory_batch(self) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if OBS_STATE not in self._queues or len(self._queues[OBS_STATE]) == 0:
            return None, None, None, None, None

        state_history = torch.stack(list(self._queues[OBS_STATE]), dim=1)
        batch_size = state_history.shape[0]
        current_index = state_history.shape[1] - 1
        selected_indices = [max(current_index + delta, 0) for delta in self.config.memory_history_deltas]

        state_window = state_history[:, selected_indices]
        state_pad = torch.zeros(batch_size, len(selected_indices), dtype=torch.bool, device=state_window.device)
        delta_indices = torch.tensor(
            self.config.memory_history_deltas,
            dtype=torch.long,
            device=state_window.device,
        ).unsqueeze(0).expand(batch_size, -1)

        image_windows = []
        image_masks = []
        for key in self.config.image_features:
            if len(self._queues[key]) == 0:
                continue
            image_history = torch.stack(list(self._queues[key]), dim=1)
            image_windows.append(image_history[:, selected_indices])
            image_masks.append(
                torch.ones(batch_size, len(selected_indices), dtype=torch.bool, device=image_history.device)
            )

        if len(image_windows) == 0:
            return None, None, state_window, state_pad, delta_indices

        while len(image_windows) < self.config.memory_max_cameras:
            reference = image_windows[0]
            image_windows.append(reference.new_zeros(reference.shape))
            image_masks.append(torch.zeros_like(image_masks[0]))

        stacked_images = torch.stack(image_windows[: self.config.memory_max_cameras], dim=2)
        stacked_masks = torch.stack(image_masks[: self.config.memory_max_cameras], dim=2)
        stacked_images, stacked_masks = self._preprocess_memory_images_window(stacked_images, stacked_masks)
        return stacked_images, stacked_masks, state_window, state_pad, delta_indices

    def _update_online_memory_queues(self, batch: dict[str, Tensor]) -> None:
        queue_batch = {OBS_STATE: batch[OBS_STATE]}
        for key in self.config.image_features:
            if key in batch:
                queue_batch[key] = batch[key]
        self._queues = populate_queues(self._queues, queue_batch)

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        memory_inputs = self._build_online_memory_batch()
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            memory_images_window=memory_inputs[0],
            memory_image_mask=memory_inputs[1],
            memory_state_window=memory_inputs[2],
            memory_state_pad=memory_inputs[3],
            memory_delta_indices=memory_inputs[4],
            **kwargs,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"
        self.eval()
        self._update_online_memory_queues(batch)
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        memory_inputs = self._extract_memory_from_batch(batch)
        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            memory_images_window=memory_inputs[0],
            memory_image_mask=memory_inputs[1],
            memory_state_window=memory_inputs[2],
            memory_state_pad=memory_inputs[3],
            memory_delta_indices=memory_inputs[4],
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample_losses = losses.mean(dim=[1, 2])
            return per_sample_losses, loss_dict

        loss = losses.mean()
        return loss, loss_dict