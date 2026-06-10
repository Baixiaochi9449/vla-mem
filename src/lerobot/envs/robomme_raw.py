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
import json
import os
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
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
# Panda pd_ee_delta_pose exposes a normalized [-1, 1] action space that maps
# to physical deltas of +/-0.1 m and +/-0.1 rad inside ManiSkill.
_PD_EE_DELTA_POS_LIMIT = 0.1
_PD_EE_DELTA_ROT_LIMIT = 0.1
_MAX_SCENE_RESET_ATTEMPTS = 10


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------



def _refresh_robomme_random_state(raw_env, seed: int | None, difficulty_hint: str | None) -> None:
    """Refresh RoboMME task RNG fields before a force-reconfigure reset.

    Several RoboMME tasks derive scene-generation state in ``__init__`` rather
    than in ``reset``. Since RobommeRawEpisodeEnv reuses the raw env to avoid
    leaking SAPIEN render devices, refresh those fields explicitly so each
    benchmark episode gets its own seed/difficulty semantics.
    """
    if seed is None:
        return

    unwrapped = raw_env.unwrapped
    seed = int(seed)
    if hasattr(unwrapped, "seed"):
        unwrapped.seed = seed
    np.random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    if hasattr(unwrapped, "generator"):
        unwrapped.generator = generator

    difficulty = difficulty_hint
    if difficulty is not None:
        try:
            from robomme.robomme_env.utils.difficulty import normalize_robomme_difficulty

            difficulty = normalize_robomme_difficulty(difficulty)
        except Exception:
            difficulty = str(difficulty).lower()
    elif hasattr(unwrapped, "difficulty"):
        seed_mod = seed % 3
        difficulty = "easy" if seed_mod == 0 else ("medium" if seed_mod == 1 else "hard")

    if difficulty is not None and hasattr(unwrapped, "difficulty"):
        unwrapped.difficulty = difficulty

    if hasattr(unwrapped, "num_repeats"):
        unwrapped.num_repeats = torch.randint(1, 4, (1,), generator=generator).item()

    configs = getattr(unwrapped, "configs", None)
    difficulty_key = getattr(unwrapped, "difficulty", None)
    if hasattr(unwrapped, "swap_times") and configs and difficulty_key in configs:
        cfg = configs[difficulty_key]
        if "swap_min" in cfg and "swap_max" in cfg:
            unwrapped.swap_times = torch.randint(
                cfg["swap_min"], cfg["swap_max"] + 1, (1,), generator=generator
            ).item()


def _is_scene_generation_error(exc: BaseException) -> bool:
    """Return True for RoboMME stochastic scene-generation failures."""
    if exc.__class__.__name__ == "SceneGenerationError":
        return True
    message = str(exc)
    return "spawn_random_cube" in message or "Failed to generate" in message


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



def _robomme_episode_task_instruction(raw_env, task_name: str) -> str:
    """Return the full episode-level RoboMME language goal when available."""
    try:
        from robomme.robomme_env.utils import task_goal

        goals = task_goal.get_language_goal(raw_env, task_name)
    except Exception:
        return task_name
    if isinstance(goals, (list, tuple)) and goals:
        first = goals[0]
        if isinstance(first, str) and first.strip():
            return first
    return task_name


def _desired_ee_delta(target_ee: np.ndarray, current_ee: np.ndarray) -> np.ndarray:
    """Return desired physical TCP delta [dx, dy, dz, droll, dpitch, dyaw]."""
    delta_xyz = target_ee[:3] - current_ee[:3]
    delta_rpy = target_ee[3:6] - current_ee[3:6]
    delta_rpy = (delta_rpy + np.pi) % (2.0 * np.pi) - np.pi
    return np.concatenate([delta_xyz, delta_rpy]).astype(np.float32)


def _to_delta(target_ee: np.ndarray, raw_env) -> np.ndarray:
    """Convert absolute target ee_pose to ManiSkill normalized pd_ee_delta_pose action.

    RoboMME/OpenPI actions are absolute ee targets. ManiSkill's Panda
    ``pd_ee_delta_pose`` controller, however, exposes a normalized action space:
    xyz inputs in [-1, 1] are scaled to +/-0.1 m and rotation inputs are scaled
    to +/-0.1 rad. Convert physical deltas into that normalized controller
    command before calling ``raw_env.step``.
    """
    current = _eef_state(raw_env)
    desired_delta = _desired_ee_delta(target_ee, current)
    delta_xyz_cmd = np.clip(desired_delta[:3] / _PD_EE_DELTA_POS_LIMIT, -1.0, 1.0)
    # ManiSkill's PDEEPoseController multiplies normalized rotation by rot_lower
    # (-0.1 for Panda), so negate here to request the intended physical sign.
    delta_rpy_cmd = -desired_delta[3:6] / _PD_EE_DELTA_ROT_LIMIT
    rot_norm = np.linalg.norm(delta_rpy_cmd)
    if rot_norm > 1.0:
        delta_rpy_cmd = delta_rpy_cmd / rot_norm
    return np.concatenate([delta_xyz_cmd, delta_rpy_cmd, [target_ee[6]]]).astype(np.float32)


def _clear_robomme_episode_flags(raw_env) -> None:
    """Clear RoboMME task-level sticky flags when reusing a raw env instance.

    The official BenchmarkEnvBuilder creates a fresh wrapped env per benchmark
    episode. RobommeRawEpisodeEnv intentionally reuses the raw ManiSkill env to
    avoid mplib, so clear failure/success state that some RoboMME tasks keep as
    Python attributes across steps.
    """
    unwrapped = raw_env.unwrapped
    for name in (
        "failureflag",
        "successflag",
        "current_task_failure",
        "swing_over_limit",
        "episode_success",
    ):
        if hasattr(unwrapped, name):
            try:
                delattr(unwrapped, name)
            except AttributeError:
                pass


def _debug_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_debug_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _debug_jsonable(item) for key, item in value.items()}
    return value


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
        task_instruction_mode: str = "subtask",
    ):
        super().__init__()
        if split not in ROBOMME_SPLITS:
            raise ValueError(f"Unsupported split: {split!r}. Expected one of {ROBOMME_SPLITS}.")

        if task_instruction_mode not in {"subtask", "episode"}:
            raise ValueError(
                "Unsupported task_instruction_mode: "
                f"{task_instruction_mode!r}. Expected 'subtask' or 'episode'."
            )

        self.task = task_name
        self.task_instruction_mode = task_instruction_mode
        self.task_description: str = task_name  # policy-facing instruction, see _update_task_description()
        self.current_subtask_instruction: str = task_name
        self.episode_task_instruction: str = task_name  # full episode-level language goal
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
        self._current_episode_seed: int | None = None
        self._current_reset_attempts = 0
        self._skipped_episode_indices: list[int] = []

        # Raw ManiSkill3 env - created once, then force-reconfigured on reset.
        # This gives each episode a fresh scene without leaking render devices.
        self._raw_env: gym.Env | None = None
        self._last_obs: dict | None = None
        self._step_index = 0

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
        """Create the raw ManiSkill3 gym env once per vector-env slot."""
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
        """Sync policy-facing task_description from the configured instruction source."""
        if self._raw_env is not None:
            live = getattr(self._raw_env.unwrapped, "current_task_name_online", None)
            if live:
                self.current_subtask_instruction = live

        if self.task_instruction_mode == "episode":
            self.task_description = self.episode_task_instruction
        else:
            self.task_description = self.current_subtask_instruction

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw = self._get_raw_env()

        raw_options = dict(options or {})
        raw_options["reconfigure"] = True
        last_exc: BaseException | None = None
        raw_obs = None
        episode_idx = None
        self._skipped_episode_indices = []

        for candidate_ix in range(len(self._episode_indices)):
            episode_idx = self._episode_indices[self._episode_cursor]
            self._episode_cursor = (self._episode_cursor + self._episode_stride) % len(
                self._episode_indices
            )

            meta_seed, difficulty_hint = self._builder.resolve_episode(episode_idx)
            reset_seed = int(meta_seed) if meta_seed is not None else seed

            for attempt in range(_MAX_SCENE_RESET_ATTEMPTS):
                attempt_seed = None if reset_seed is None else int(reset_seed) + attempt
                _refresh_robomme_random_state(raw, attempt_seed, difficulty_hint)
                try:
                    raw_obs, _ = raw.reset(seed=attempt_seed, options=raw_options)
                    self._current_episode_index = episode_idx
                    self._current_episode_seed = attempt_seed
                    self._current_reset_attempts = attempt + 1
                    break
                except Exception as exc:
                    if not _is_scene_generation_error(exc):
                        raise
                    last_exc = exc
            if raw_obs is not None:
                break
            self._skipped_episode_indices.append(int(episode_idx))
        else:
            raise RuntimeError(
                f"Failed to generate a valid RoboMME scene for {self.task} after "
                f"trying {len(self._episode_indices)} episode indices with "
                f"{_MAX_SCENE_RESET_ATTEMPTS} seeds each."
            ) from last_exc

        _clear_robomme_episode_flags(raw)
        self._step_index = 0
        if raw_obs is None or episode_idx is None:
            raise RuntimeError("RoboMME reset did not return an observation.")
        self._last_obs = _convert_obs(raw_obs, raw)
        self.episode_task_instruction = _robomme_episode_task_instruction(raw, self.task)
        self._update_task_description()

        info = {
            "task_goal": [self.task_description],
            "task_instruction": self.task_description,
            "subtask_instruction": self.current_subtask_instruction,
            "episode_task_instruction": self.episode_task_instruction,
            "episode_index": episode_idx,
            "episode_seed": self._current_episode_seed,
            "reset_attempts": self._current_reset_attempts,
            "skipped_episode_indices": list(self._skipped_episode_indices),
            "task": self.task,
        }
        return self._last_obs, info

    def step(self, action: np.ndarray):
        if self._raw_env is None:
            raise RuntimeError("reset() must be called before step().")

        target = np.asarray(action, dtype=np.float32).reshape(-1)
        current_before = _eef_state(self._raw_env)
        desired_delta = _desired_ee_delta(target, current_before)
        delta = _to_delta(target, self._raw_env)
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
        self._append_debug_trace(
            target=target,
            current_before=current_before,
            desired_delta=desired_delta,
            delta=delta,
            reward=reward_val,
            terminated=terminated,
            truncated=truncated,
            is_success=is_success,
            raw_info=info,
        )
        self._step_index += 1
        info_out: dict[str, Any] = {
            "task": self.task,
            "episode_index": self._current_episode_index,
            "episode_seed": self._current_episode_seed,
            "task_instruction": self.task_description,
            "subtask_instruction": self.current_subtask_instruction,
            "episode_task_instruction": self.episode_task_instruction,
            "skipped_episode_indices": list(self._skipped_episode_indices),
            "is_success": is_success,
        }

        if terminated or truncated:
            status = "success" if is_success else ("fail" if terminated else "timeout")
            info_out["status"] = status
            info_out["final_info"] = {
                "task": self.task,
                "episode_index": self._current_episode_index,
                "episode_seed": self._current_episode_seed,
                "task_instruction": self.task_description,
                "subtask_instruction": self.current_subtask_instruction,
                "episode_task_instruction": self.episode_task_instruction,
                "skipped_episode_indices": list(self._skipped_episode_indices),
                "is_success": is_success,
                "status": status,
            }

        return self._last_obs, reward_val, terminated, truncated, info_out

    def _append_debug_trace(
        self,
        *,
        target: np.ndarray,
        current_before: np.ndarray,
        desired_delta: np.ndarray,
        delta: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        is_success: bool,
        raw_info: dict[str, Any],
    ) -> None:
        debug_dir = os.environ.get("LEROBOT_ROBOMME_DEBUG_DIR")
        if not debug_dir:
            return

        max_steps = int(os.environ.get("LEROBOT_ROBOMME_DEBUG_MAX_STEPS", "0") or 0)
        if max_steps > 0 and self._step_index >= max_steps:
            return

        selected_episode = os.environ.get("LEROBOT_ROBOMME_DEBUG_EPISODE")
        if (
            selected_episode not in (None, "")
            and str(self._current_episode_index) != selected_episode
        ):
            return

        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_success = raw_info.get("success")
        raw_fail = raw_info.get("fail")
        record = {
            "source": "robomme_raw.step",
            "step": self._step_index,
            "episode_index": self._current_episode_index,
            "time_sec_at_30fps": self._step_index / 30.0,
            "task": self.task,
            "task_description": self.task_description,
            "subtask_instruction": self.current_subtask_instruction,
            "episode_task_instruction": self.episode_task_instruction,
            "task_instruction_mode": self.task_instruction_mode,
            "target_absolute_ee_pose": target,
            "current_ee_pose_before_step": current_before,
            "desired_physical_delta_ee_pose": desired_delta,
            "normalized_action_sent_to_pd_ee_delta_pose": delta,
            "desired_delta_translation_l2": float(np.linalg.norm(desired_delta[:3])),
            "desired_delta_rotation_l2": float(np.linalg.norm(desired_delta[3:6])),
            "normalized_command_translation_l2": float(np.linalg.norm(delta[:3])),
            "normalized_command_rotation_l2": float(np.linalg.norm(delta[3:6])),
            "target_translation_l2": float(np.linalg.norm(target[:3])),
            "target_rotation_l2": float(np.linalg.norm(target[3:6])),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "is_success": is_success,
            "raw_success": (
                bool(raw_success.any().cpu().numpy()) if raw_success is not None else None
            ),
            "raw_fail": bool(raw_fail.any().cpu().numpy()) if raw_fail is not None else None,
            "observation_state_after_step": (
                self._last_obs.get("agent_pos") if self._last_obs else None
            ),
        }
        path = out_dir / f"eval_episode_{self._current_episode_index}_step_trace.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_debug_jsonable(record), ensure_ascii=False) + "\n")

    def render(self):
        if self._last_obs is None:
            return np.zeros(
                (self.observation_height, self.observation_width * 2, 3), dtype=np.uint8
            )
        p = self._last_obs["pixels"]
        return np.concatenate([p["image"], p["image2"]], axis=1)

    def close(self):
        self._close_raw_env()

    def _close_raw_env(self) -> None:
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
