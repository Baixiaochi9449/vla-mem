#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

#这个应该是废弃的，创建错的


from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROBOMME_TASKS = (
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
)
ROBOMME_SPLITS = ("train", "val", "test")
ROBOMME_ACTION_SPACE_SHAPES = {
    "ee_pose": 7,
    "joint_angle": 8,
}


def normalize_robomme_task_names(task: str | None) -> list[str]:
    if task is None or task == "all":
        return list(ROBOMME_TASKS)

    task_names = [name.strip() for name in task.split(",") if name.strip()]
    if not task_names:
        raise ValueError("task must contain at least one RoboMME task or 'all'.")

    invalid = sorted(set(task_names) - set(ROBOMME_TASKS))
    if invalid:
        raise ValueError(f"Unknown RoboMME tasks: {', '.join(invalid)}")

    return task_names


def _ensure_robomme_available() -> type:
    try:
        importlib.import_module("robomme.robomme_env")
        env_record_wrapper = importlib.import_module("robomme.env_record_wrapper")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboMME integration requires the external 'robomme' package. "
            "Install it with `pip install -e \".[robomme]\"` or install robomme_benchmark separately, "
            "then ensure it is importable."
        ) from exc

    builder_cls = getattr(env_record_wrapper, "BenchmarkEnvBuilder", None)
    if builder_cls is None:
        raise ImportError("Installed 'robomme' package does not expose BenchmarkEnvBuilder.")
    return builder_cls


def _to_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _last_item(observation: Mapping[str, Any], key: str) -> Any:
    values = observation.get(key)
    if values is None or len(values) == 0:
        raise KeyError(f"Missing or empty RoboMME observation field: {key}")
    return values[-1]


def _normalize_status(status: str | None, terminated: bool, truncated: bool, info: Mapping[str, Any]) -> str:
    if status and status != "ongoing":
        return status
    if bool(info.get("is_success", False)):
        return "success"
    if truncated:
        return "timeout"
    if terminated:
        return "fail"
    return "ongoing"


class RobommeEpisodeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_name: str,
        split: str = "test",
        action_space: str = "ee_pose",
        episode_indices: Sequence[int] | None = None,
        start_offset: int = 0,
        episode_stride: int = 1,
        episode_length: int = 1300,
        observation_height: int = 256,
        observation_width: int = 256,
        gui_render: bool = False,
        builder_cls: type | None = None,
    ):
        super().__init__()
        if split not in ROBOMME_SPLITS:
            raise ValueError(f"Unsupported split: {split}. Expected one of {ROBOMME_SPLITS}.")
        if action_space not in ROBOMME_ACTION_SPACE_SHAPES:
            raise ValueError(
                f"Unsupported action_space: {action_space}. Expected one of {tuple(ROBOMME_ACTION_SPACE_SHAPES)}."
            )

        self.task = task_name
        self.task_description = task_name
        self.split = split
        self.action_space_name = action_space
        self.gui_render = gui_render
        self.episode_length = episode_length
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._max_episode_steps = episode_length
        self._episode_stride = max(1, int(episode_stride))

        builder_type = builder_cls or _ensure_robomme_available()
        self._builder = builder_type(
            env_id=task_name,
            dataset=split,
            action_space=action_space,
            gui_render=gui_render,
            max_steps=episode_length,
        )
        self._episode_indices = self._resolve_episode_indices(episode_indices)
        self._episode_cursor = start_offset % len(self._episode_indices)
        self._current_episode_index: int | None = None
        self._env = None
        self._last_observation: dict[str, Any] | None = None
        self._last_frame: np.ndarray | None = None

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "image": spaces.Box(
                            low=0,
                            high=255,
                            shape=(observation_height, observation_width, 3),
                            dtype=np.uint8,
                        ),
                        "image2": spaces.Box(
                            low=0,
                            high=255,
                            shape=(observation_height, observation_width, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            }
        )
        action_dim = ROBOMME_ACTION_SPACE_SHAPES[action_space]
        self.action_space = spaces.Box(
            low=np.full((action_dim,), -np.inf, dtype=np.float32),
            high=np.full((action_dim,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _resolve_episode_indices(self, episode_indices: Sequence[int] | None) -> list[int]:
        total_episodes = int(self._builder.get_episode_num())
        if total_episodes <= 0:
            raise ValueError(f"RoboMME task '{self.task}' has no available episodes for split '{self.split}'.")

        if episode_indices is None:
            return list(range(total_episodes))

        resolved = [int(idx) for idx in episode_indices]
        invalid = sorted(idx for idx in resolved if idx < 0 or idx >= total_episodes)
        if invalid:
            raise ValueError(
                f"Episode indices out of range for task '{self.task}' split '{self.split}': {invalid}"
            )
        if not resolved:
            raise ValueError("episode_indices must not be empty.")
        return resolved

    def _open_episode_env(self) -> None:
        self.close()
        episode_idx = self._episode_indices[self._episode_cursor]
        self._episode_cursor = (self._episode_cursor + self._episode_stride) % len(self._episode_indices)
        self._current_episode_index = episode_idx
        self._env = self._builder.make_env_for_episode(episode_idx=episode_idx, max_steps=self.episode_length)

    def _convert_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        front = _to_numpy(_last_item(observation, "front_rgb_list"), dtype=np.uint8)
        wrist = _to_numpy(_last_item(observation, "wrist_rgb_list"), dtype=np.uint8)
        eef_state = _to_numpy(_last_item(observation, "eef_state_list"), dtype=np.float32).reshape(-1)
        gripper_state = _to_numpy(_last_item(observation, "gripper_state_list"), dtype=np.float32).reshape(-1)
        agent_pos = np.concatenate((eef_state, gripper_state), axis=0).astype(np.float32, copy=False)
        return {
            "pixels": {
                "image": front,
                "image2": wrist,
            },
            "agent_pos": agent_pos,
        }

    def _update_cached_frame(self) -> None:
        if self._last_observation is None:
            self._last_frame = None
            return
        pixels = self._last_observation["pixels"]
        self._last_frame = np.concatenate((pixels["image"], pixels["image2"]), axis=1)

    def _decorate_info(self, info: Mapping[str, Any] | None) -> dict[str, Any]:
        decorated = dict(info or {})
        decorated.setdefault("task", self.task)
        if self._current_episode_index is not None:
            decorated.setdefault("episode_index", self._current_episode_index)

        task_goal = decorated.get("task_goal")
        if isinstance(task_goal, list) and task_goal and isinstance(task_goal[0], str):
            self.task_description = task_goal[0]
        return decorated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._open_episode_env()
        observation, info = self._env.reset()
        converted = self._convert_observation(observation)
        self._last_observation = converted
        self._update_cached_frame()
        return converted, self._decorate_info(info)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("RoboMME environment must be reset before calling step().")

        observation, reward, terminated, truncated, info = self._env.step(np.asarray(action, dtype=np.float32))
        terminated = bool(terminated)
        truncated = bool(truncated)
        decorated_info = self._decorate_info(info)

        status = _normalize_status(decorated_info.get("status"), terminated, truncated, decorated_info)
        if status == "error":
            terminated = True
            truncated = False
        elif status == "timeout":
            truncated = True
        elif status in {"success", "fail"}:
            terminated = True

        if observation:
            converted_observation = self._convert_observation(observation)
            self._last_observation = converted_observation
            self._update_cached_frame()
        elif self._last_observation is None:
            raise RuntimeError("RoboMME step returned an empty observation before any valid reset observation.")

        converted_observation = self._last_observation
        is_success = status == "success"
        decorated_info["status"] = status
        decorated_info["is_success"] = is_success

        if terminated or truncated:
            final_info = dict(decorated_info.get("final_info") or {})
            final_info.update(
                {
                    "task": self.task,
                    "episode_index": self._current_episode_index,
                    "status": status,
                    "is_success": is_success,
                }
            )
            decorated_info["final_info"] = final_info

        reward_value = float(_to_numpy(reward).reshape(-1)[0])
        return converted_observation, reward_value, terminated, truncated, decorated_info

    def render(self):
        if self._last_frame is None:
            return np.zeros((self.observation_height, self.observation_width * 2, 3), dtype=np.uint8)
        return self._last_frame

    def close(self):
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None


def create_robomme_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    task_names = normalize_robomme_task_names(task)
    env_kwargs = dict(gym_kwargs or {})
    builder_cls = env_kwargs.pop("builder_cls", None)

    out: dict[str, dict[int, Any]] = {}
    for task_name in task_names:
        fns = [
            partial(
                RobommeEpisodeEnv,
                task_name=task_name,
                start_offset=start_offset,
                episode_stride=n_envs,
                builder_cls=builder_cls,
                **env_kwargs,
            )
            for start_offset in range(n_envs)
        ]
        out[task_name] = {0: env_cls(fns)}

    return out