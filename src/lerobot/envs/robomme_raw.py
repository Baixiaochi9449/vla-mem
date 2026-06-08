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

"""RoboMME raw gymnasium wrapper – bypasses DemonstrationWrapper / mplib.

This module provides ``RobommeRawEpisodeEnv``, a gymnasium wrapper around the
raw ManiSkill3 RoboMME environments.  It is designed for servers where mplib
(the motion-planning library used by ``DemonstrationWrapper``) is unavailable
or segfaults (e.g. on certain Intel Xeon platforms).

Key differences from ``RobommeEpisodeEnv`` (``robomme.py``):
- Does **not** call ``BenchmarkEnvBuilder.make_env_for_episode()`` (which
  triggers mplib).  Instead it creates the raw env directly via
  ``gym.make(task_id, control_mode="pd_ee_delta_pose", render_mode="none")``.
- The policy is expected to output **absolute** end-effector poses
  ``[x, y, z, roll, pitch, yaw, gripper]`` (7-D).  ``step()`` converts them
  to **delta** actions internally before forwarding to the raw env.
- ``BenchmarkEnvBuilder`` is used **only** for episode metadata (seed,
  episode count) – it never calls ``make_env_for_episode``.

Usage with lerobot-eval:
    PYTHONPATH=/path/to/robomme_benchmark/src:$PYTHONPATH \\
    SAPIEN_RENDER_DEVICE=cuda \\
    lerobot-eval \\
        --policy.path=outputs/train/pi05/checkpoints/030000/pretrained_model \\
        --env.type=robomme_raw \\
        --env.task=PickXtimes \\
        --env.split=test \\
        --eval.n_episodes=50 \\
        --eval.batch_size=1 \\
        --policy.device=cuda
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from lerobot.envs.robomme import (
    ROBOMME_ACTION_SPACE_SHAPES,
    ROBOMME_SPLITS,
    normalize_robomme_task_names,
)

# ManiSkill3 control mode that accepts delta EEF poses (no mplib required)
_CONTROL_MODE = "pd_ee_delta_pose"
_FRONT_CAM = "base_camera"
_WRIST_CAM = "hand_camera"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _ensure_robomme_env_registered() -> None:
    """Import robomme.robomme_env so all task gym IDs are registered."""
    try:
        importlib.import_module("robomme.robomme_env")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboMME raw env requires the 'robomme' package. "
            "Install it with: cd robomme_benchmark && pip install -e .\n"
            "Also ensure it is on PYTHONPATH: "
            "export PYTHONPATH=/path/to/robomme_benchmark/src:$PYTHONPATH"
        ) from exc


def _get_builder_cls() -> type:
    """Return BenchmarkEnvBuilder class (used only for episode metadata reads)."""
    try:
        mod = importlib.import_module("robomme.env_record_wrapper")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "robomme.env_record_wrapper not found. "
            "Install robomme_benchmark: cd robomme_benchmark && pip install -e ."
        ) from exc
    cls = getattr(mod, "BenchmarkEnvBuilder", None)
    if cls is None:
        raise ImportError("BenchmarkEnvBuilder not found in robomme.env_record_wrapper.")
    return cls


# ---------------------------------------------------------------------------
# Low-level observation / action helpers
# ---------------------------------------------------------------------------

def _eef_state(raw_env) -> np.ndarray:
    """Return current [x, y, z, roll, pitch, yaw] (6-D float32) of the TCP.

    SAPIEN Pose convention: ``tcp_pose.raw_pose[0]`` = ``[x, y, z, qw, qx, qy, qz]``
    scipy ``Rotation.from_quat`` expects                              ``[qx, qy, qz, qw]``

    RPY angles near -π are mapped to their +π equivalent to match the training
    dataset convention (eef_action in the HDF5 recordings always uses +π for the
    robot's nominal roll-≈π posture, and the conversion script aligns eef_state
    to the same branch).
    """
    raw = raw_env.unwrapped.agent.tcp_pose.raw_pose[0].cpu().numpy()
    qwxyz = raw[3:]  # [qw, qx, qy, qz]
    r = Rotation.from_quat([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])
    rpy = r.as_euler("xyz", degrees=False).astype(np.float32)
    # Align angles near -π to +π to be consistent with training data convention.
    rpy = np.where(rpy < -np.pi + 0.15, rpy + 2.0 * np.pi, rpy)
    return np.concatenate([raw[:3].astype(np.float32), rpy])  # (6,)


def _convert_obs(raw_obs: dict, raw_env) -> dict:
    """Convert raw ManiSkill3 obs dict to the format expected by preprocess_observation.

    Returns:
        {
            "pixels": {
                "image":  (H, W, 3) uint8 – front camera
                "image2": (H, W, 3) uint8 – wrist camera
            },
            "agent_pos": (8,) float32 – [x, y, z, roll, pitch, yaw, grip_l, grip_r]
        }
    """
    # Images: sensor_data[cam]["rgb"] is (1, H, W, C) uint8 on GPU
    front = raw_obs["sensor_data"][_FRONT_CAM]["rgb"][0].cpu().numpy().astype(np.uint8)
    wrist = raw_obs["sensor_data"][_WRIST_CAM]["rgb"][0].cpu().numpy().astype(np.uint8)

    eef = _eef_state(raw_env)  # (6,) xyz + rpy
    # agent.qpos shape: (1, 9)  →  [j0..j6, gripper_left, gripper_right]
    qpos = raw_obs["agent"]["qpos"][0].cpu().numpy()  # (9,)
    grip = qpos[7:9].astype(np.float32)  # (2,)
    agent_pos = np.concatenate([eef, grip])  # (8,)

    return {
        "pixels": {"image": front, "image2": wrist},
        "agent_pos": agent_pos,
    }


def _to_delta(target_ee: np.ndarray, raw_env) -> np.ndarray:
    """Convert absolute target ee_pose to pd_ee_delta_pose delta.

    Args:
        target_ee: (7,) float32 – [x, y, z, roll, pitch, yaw, gripper]
        raw_env: live ManiSkill3 env (for reading current TCP state)

    Returns:
        (7,) float32 – [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
    """
    current = _eef_state(raw_env)  # (6,)
    delta_xyz = target_ee[:3] - current[:3]
    delta_rpy = target_ee[3:6] - current[3:6]
    # Wrap angle deltas to [-π, π] to avoid large jumps
    delta_rpy = (delta_rpy + np.pi) % (2.0 * np.pi) - np.pi
    return np.concatenate([delta_xyz, delta_rpy, [target_ee[6]]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------

class RobommeRawEpisodeEnv(gym.Env):
    """RoboMME gym wrapper that bypasses DemonstrationWrapper / mplib.

    Designed to be used inside ``gymnasium.vector.SyncVectorEnv`` (one instance
    per parallel env slot) and driven by the lerobot eval/train loop.

    Observation space:
        pixels.image   (H, W, 3) uint8  – front camera
        pixels.image2  (H, W, 3) uint8  – wrist camera
        agent_pos      (8,)      float32 – [x, y, z, roll, pitch, yaw, grip_l, grip_r]

    Action space:
        (7,) float32 – absolute ee_pose [x, y, z, roll, pitch, yaw, gripper]
        (Converted to delta internally before calling the raw ManiSkill3 env.)

    Attributes:
        task_description (str): Human-readable task description. Updated after
            each ``reset()`` and ``step()`` from
            ``env.unwrapped.current_task_name_online`` (set by ManiSkill3's
            sub-goal evaluator).  Read by ``add_envs_task()`` in the eval loop.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_name: str,
        split: str = "test",
        episode_indices: Sequence[int] | None = None,
        start_offset: int = 0,
        episode_stride: int = 1,
        episode_length: int = 1300,
        observation_height: int = 256,
        observation_width: int = 256,
        builder_cls: type | None = None,
    ):
        super().__init__()
        if split not in ROBOMME_SPLITS:
            raise ValueError(f"Unsupported split: {split!r}. Expected one of {ROBOMME_SPLITS}.")

        self.task = task_name
        self.task_description: str = task_name  # updated after reset() / step()
        self.split = split
        self.episode_length = episode_length
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._max_episode_steps = episode_length
        self._episode_stride = max(1, int(episode_stride))

        # Builder is used ONLY for metadata (seed lookup), never for make_env_for_episode
        _builder_cls = builder_cls or _get_builder_cls()
        self._builder = _builder_cls(
            env_id=task_name,
            dataset=split,
            action_space="ee_pose",  # schema only; env created independently
            render_mode="none",
        )
        self._episode_indices = self._resolve_episode_indices(episode_indices)
        self._episode_cursor = int(start_offset) % len(self._episode_indices)
        self._current_episode_index: int | None = None

        # Raw ManiSkill3 env – created lazily in first reset(), reused across episodes
        self._raw_env: gym.Env | None = None
        self._last_obs: dict | None = None

        action_dim = ROBOMME_ACTION_SPACE_SHAPES["ee_pose"]  # 7
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "image": spaces.Box(
                            0, 255, (observation_height, observation_width, 3), np.uint8
                        ),
                        "image2": spaces.Box(
                            0, 255, (observation_height, observation_width, 3), np.uint8
                        ),
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, (8,), np.float32),
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, (action_dim,), np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_episode_indices(self, episode_indices: Sequence[int] | None) -> list[int]:
        total = int(self._builder.get_episode_num())
        if total <= 0:
            raise ValueError(
                f"No episodes found for task '{self.task}' split '{self.split}'."
            )
        if episode_indices is None:
            return list(range(total))
        resolved = [int(i) for i in episode_indices]
        invalid = [i for i in resolved if i < 0 or i >= total]
        if invalid:
            raise ValueError(
                f"Episode indices out of range for '{self.task}/{self.split}': {invalid} "
                f"(total={total})"
            )
        return resolved

    def _get_raw_env(self) -> gym.Env:
        """Lazily create the raw ManiSkill3 gym env (once per wrapper instance)."""
        if self._raw_env is None:
            _ensure_robomme_env_registered()
            self._raw_env = gym.make(
                self.task,
                obs_mode="rgb+depth+segmentation",
                control_mode=_CONTROL_MODE,
                render_mode="none",
                reward_mode="dense",
            )
        return self._raw_env

    def _update_task_description(self) -> None:
        """Sync task_description from the live env's current_task_name_online."""
        if self._raw_env is not None:
            live = getattr(self._raw_env.unwrapped, "current_task_name_online", None)
            if live:
                self.task_description = live

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw = self._get_raw_env()

        episode_idx = self._episode_indices[self._episode_cursor]
        self._episode_cursor = (self._episode_cursor + self._episode_stride) % len(
            self._episode_indices
        )
        self._current_episode_index = episode_idx

        meta_seed, _ = self._builder.resolve_episode(episode_idx)
        reset_seed = int(meta_seed) if meta_seed is not None else seed

        raw_obs, _ = raw.reset(seed=reset_seed)
        self._last_obs = _convert_obs(raw_obs, raw)
        self._update_task_description()

        info = {
            "task_goal": [self.task_description],
            "episode_index": episode_idx,
            "task": self.task,
        }
        return self._last_obs, info

    def step(self, action: np.ndarray):
        if self._raw_env is None:
            raise RuntimeError("reset() must be called before step().")

        delta = _to_delta(np.asarray(action, dtype=np.float32), self._raw_env)
        raw_obs, reward, terminated, truncated, info = self._raw_env.step(delta)

        terminated = bool(terminated)
        truncated = bool(truncated)

        # Success / fail from ManiSkill3 info tensors
        success_t = info.get("success")
        is_success = (
            bool(success_t.any().cpu().numpy()) if success_t is not None else False
        )
        fail_t = info.get("fail")
        if fail_t is not None and bool(fail_t.any().cpu().numpy()):
            terminated = True

        if raw_obs:
            self._last_obs = _convert_obs(raw_obs, self._raw_env)

        self._update_task_description()

        reward_val = float(np.asarray(reward).ravel()[0])
        info_out: dict[str, Any] = {
            "task": self.task,
            "episode_index": self._current_episode_index,
            "is_success": is_success,
        }

        if terminated or truncated:
            status = "success" if is_success else ("fail" if terminated else "timeout")
            info_out["status"] = status
            info_out["final_info"] = {
                "task": self.task,
                "episode_index": self._current_episode_index,
                "is_success": is_success,
                "status": status,
            }

        return self._last_obs, reward_val, terminated, truncated, info_out

    def render(self):
        if self._last_obs is None:
            return np.zeros(
                (self.observation_height, self.observation_width * 2, 3), dtype=np.uint8
            )
        p = self._last_obs["pixels"]
        return np.concatenate([p["image"], p["image2"]], axis=1)

    def close(self):
        if self._raw_env is not None:
            try:
                self._raw_env.close()
            except Exception:
                pass
            self._raw_env = None


# ---------------------------------------------------------------------------
# Factory function (called by lerobot.envs.factory.make_env)
# ---------------------------------------------------------------------------

def create_robomme_raw_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """Create ``{task_name: {0: VectorEnv}}`` for each requested RoboMME task.

    This follows the same pattern as ``create_robomme_envs`` and
    ``create_libero_envs``, returning a nested dict expected by
    ``eval_policy_all``.

    Args:
        task: Task name or ``"all"`` for all 16 tasks (comma-separated also works).
        n_envs: Number of parallel env instances per task.
        gym_kwargs: Extra kwargs forwarded to ``RobommeRawEpisodeEnv.__init__``
            (split, episode_length, observation_height, observation_width,
            episode_indices).
        env_cls: Vector-env class, e.g. ``gym.vector.SyncVectorEnv``.

    Returns:
        {task_name: {0: VectorEnv}}
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps env factories.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs!r}.")

    task_names = normalize_robomme_task_names(task)
    env_kwargs = dict(gym_kwargs or {})
    builder_cls = env_kwargs.pop("builder_cls", None)

    out: dict[str, dict[int, Any]] = {}
    for task_name in task_names:
        fns = [
            partial(
                RobommeRawEpisodeEnv,
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
