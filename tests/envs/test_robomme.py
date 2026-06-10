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

import numpy as np

from lerobot.envs import robomme_raw
from lerobot.envs.configs import RobommeEnv, RobommeRawEnv
from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.robomme import RobommeEpisodeEnv
from robomme.robomme_env.utils.subgoal_evaluate_func import correct_timestep


def _make_rgb(fill_value: int, height: int = 4, width: int = 6) -> np.ndarray:
    return np.full((height, width, 3), fill_value, dtype=np.uint8)


def _make_raw_obs(episode_idx: int, step_idx: int = 0) -> dict:
    return {
        "front_rgb_list": [_make_rgb(episode_idx), _make_rgb(episode_idx + step_idx + 1)],
        "wrist_rgb_list": [_make_rgb(episode_idx + 10), _make_rgb(episode_idx + step_idx + 11)],
        "eef_state_list": [
            np.zeros(6, dtype=np.float32),
            np.full(6, episode_idx + step_idx + 0.5, dtype=np.float32),
        ],
        "gripper_state_list": [
            np.zeros(2, dtype=np.float32),
            np.array([episode_idx + step_idx, episode_idx + step_idx + 0.25], dtype=np.float32),
        ],
    }


class FakeRobommeInnerEnv:
    def __init__(self, task_name: str, episode_idx: int):
        self.task_name = task_name
        self.episode_idx = episode_idx
        self.step_calls = 0
        self.closed = False

    def reset(self):
        return _make_raw_obs(self.episode_idx), {
            "task_goal": [f"{self.task_name} goal {self.episode_idx}"],
            "status": "ongoing",
        }

    def step(self, action):
        self.step_calls += 1
        if self.episode_idx == 1:
            return {}, 0.0, True, False, {
                "task_goal": [f"{self.task_name} goal {self.episode_idx}"],
                "status": "error",
                "error_message": "IK failed",
            }

        return _make_raw_obs(self.episode_idx, step_idx=self.step_calls), 1.0, True, False, {
            "task_goal": [f"{self.task_name} goal {self.episode_idx}"],
            "status": "success",
        }

    def close(self):
        self.closed = True


class FakeBenchmarkEnvBuilder:
    def __init__(self, env_id: str, dataset: str, action_space: str, gui_render: bool, max_steps: int):
        self.env_id = env_id
        self.dataset = dataset
        self.action_space = action_space
        self.gui_render = gui_render
        self.max_steps = max_steps

    def get_episode_num(self) -> int:
        return 4

    def make_env_for_episode(self, episode_idx: int, max_steps=None):
        return FakeRobommeInnerEnv(self.env_id, episode_idx)


def test_make_env_config_returns_robomme_config():
    cfg = make_env_config("robomme", action_space="joint_angle")

    assert isinstance(cfg, RobommeEnv)
    assert cfg.features["action"].shape == (8,)


def test_robomme_episode_env_converts_latest_observation_and_rotates_episodes():
    env = RobommeEpisodeEnv(
        task_name="PickXtimes",
        split="test",
        action_space="ee_pose",
        episode_indices=[0, 1, 2, 3],
        start_offset=0,
        episode_stride=2,
        episode_length=5,
        observation_height=4,
        observation_width=6,
        builder_cls=FakeBenchmarkEnvBuilder,
    )

    obs, info = env.reset()
    np.testing.assert_array_equal(obs["pixels"]["image"], _make_rgb(1))
    np.testing.assert_array_equal(obs["pixels"]["image2"], _make_rgb(11))
    np.testing.assert_allclose(obs["agent_pos"], np.array([0.5] * 6 + [0.0, 0.25], dtype=np.float32))
    assert info["episode_index"] == 0
    assert env.task_description == "PickXtimes goal 0"

    _, next_info = env.reset()
    assert next_info["episode_index"] == 2
    env.close()


def test_robomme_episode_env_maps_error_status_to_failed_episode():
    env = RobommeEpisodeEnv(
        task_name="PickXtimes",
        split="test",
        action_space="ee_pose",
        episode_indices=[1],
        episode_length=5,
        observation_height=4,
        observation_width=6,
        builder_cls=FakeBenchmarkEnvBuilder,
    )

    reset_obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward == 0.0
    np.testing.assert_array_equal(obs["pixels"]["image"], reset_obs["pixels"]["image"])
    assert info["status"] == "error"
    assert info["final_info"]["is_success"] is False
    env.close()


def test_make_env_builds_robomme_vector_env(monkeypatch):
    monkeypatch.setattr("lerobot.envs.robomme._ensure_robomme_available", lambda: FakeBenchmarkEnvBuilder)
    cfg = RobommeEnv(
        task="PickXtimes",
        action_space="ee_pose",
        episode_indices=[0, 1, 2, 3],
        episode_length=5,
        observation_height=4,
        observation_width=6,
    )

    envs = make_env(cfg, n_envs=2)
    vec_env = envs["PickXtimes"][0]
    obs, info = vec_env.reset()

    assert obs["agent_pos"].shape == (2, 8)
    assert len(info["episode_index"]) == 2
    vec_env.close()


def test_robomme_raw_absolute_ee_pose_maps_to_normalized_delta_command(monkeypatch):
    monkeypatch.setattr(
        robomme_raw,
        "_eef_state",
        lambda raw_env: np.array([0.0, 0.0, 0.2, np.pi, 0.0, 0.0], dtype=np.float32),
    )
    target = np.array([0.05, -0.2, 0.25, np.pi + 0.05, -0.2, 0.0, -1.0], dtype=np.float32)

    command = robomme_raw._to_delta(target, raw_env=object())

    # Position commands are normalized by Panda's +/-0.1 m controller bound.
    np.testing.assert_allclose(command[:3], np.array([0.5, -1.0, 0.5], dtype=np.float32))
    # Rotation commands are normalized by the +/-0.1 rad bound and negated to
    # compensate for ManiSkill's rot_lower=-0.1 scaling convention.
    np.testing.assert_allclose(
        command[3:6], np.array([-0.24253564, 0.97014254, -0.0], dtype=np.float32), atol=1e-6
    )
    assert command[-1] == -1.0
    assert np.linalg.norm(command[3:6]) <= 1.0 + 1e-6


class _FakeRawUnwrapped:
    def __init__(self):
        self.failureflag = object()
        self.successflag = object()
        self.current_task_failure = object()
        self.swing_over_limit = object()
        self.episode_success = object()


class _FakeRawEnv:
    def __init__(self):
        self.unwrapped = _FakeRawUnwrapped()


def test_robomme_raw_clears_sticky_episode_flags_when_reusing_env():
    raw = _FakeRawEnv()

    robomme_raw._clear_robomme_episode_flags(raw)

    for name in (
        "failureflag",
        "successflag",
        "current_task_failure",
        "swing_over_limit",
        "episode_success",
    ):
        assert not hasattr(raw.unwrapped, name)


class _FakeRawBuilder:
    def __init__(self, env_id: str, dataset: str, action_space: str, render_mode: str):
        self.env_id = env_id
        self.dataset = dataset
        self.action_space = action_space
        self.render_mode = render_mode

    def get_episode_num(self) -> int:
        return 2

    def resolve_episode(self, episode_idx: int):
        return episode_idx + 100, None


class _FakeResetRawEnv:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.closed = False
        self.reset_calls = []
        self.unwrapped = type("FakeUnwrapped", (), {})()

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, dict(options or {})))
        return {"env_id": self.env_id, "reset_count": len(self.reset_calls)}, {}

    def close(self):
        self.closed = True


def test_robomme_raw_config_passes_task_instruction_mode():
    cfg = RobommeRawEnv(task="SwingXtimes", task_instruction_mode="episode")

    assert cfg.gym_kwargs["task_instruction_mode"] == "episode"


def test_robomme_raw_can_switch_policy_instruction_between_subtask_and_episode(monkeypatch):
    created_envs = []

    def fake_make(*args, **kwargs):
        raw = _FakeResetRawEnv(len(created_envs))
        raw.unwrapped.current_task_name_online = "online subtask instruction"
        created_envs.append(raw)
        return raw

    monkeypatch.setattr(robomme_raw, "_ensure_robomme_env_registered", lambda: None)
    monkeypatch.setattr(robomme_raw.gym, "make", fake_make)
    monkeypatch.setattr(
        robomme_raw, "_convert_obs", lambda raw_obs, raw_env: {"env_id": raw_env.env_id}
    )
    monkeypatch.setattr(
        robomme_raw,
        "_robomme_episode_task_instruction",
        lambda raw_env, task_name: "global episode instruction",
    )

    subtask_env = robomme_raw.RobommeRawEpisodeEnv(
        task_name="SwingXtimes",
        split="test",
        episode_indices=[0],
        builder_cls=_FakeRawBuilder,
    )
    _, subtask_info = subtask_env.reset()

    episode_env = robomme_raw.RobommeRawEpisodeEnv(
        task_name="SwingXtimes",
        split="test",
        episode_indices=[0],
        builder_cls=_FakeRawBuilder,
        task_instruction_mode="episode",
    )
    _, episode_info = episode_env.reset()

    assert subtask_env.task_description == "online subtask instruction"
    assert subtask_info["task_instruction"] == "online subtask instruction"
    assert subtask_info["episode_task_instruction"] == "global episode instruction"
    assert subtask_info["subtask_instruction"] == "online subtask instruction"

    assert episode_env.task_description == "global episode instruction"
    assert episode_info["task_instruction"] == "global episode instruction"
    assert episode_info["episode_task_instruction"] == "global episode instruction"
    assert episode_info["subtask_instruction"] == "online subtask instruction"

    subtask_env.close()
    episode_env.close()


def test_robomme_raw_reset_force_reconfigures_without_recreating_env(monkeypatch):
    created_envs = []

    def fake_make(*args, **kwargs):
        raw = _FakeResetRawEnv(len(created_envs))
        created_envs.append(raw)
        return raw

    monkeypatch.setattr(robomme_raw, "_ensure_robomme_env_registered", lambda: None)
    monkeypatch.setattr(robomme_raw.gym, "make", fake_make)
    monkeypatch.setattr(
        robomme_raw, "_convert_obs", lambda raw_obs, raw_env: {"env_id": raw_env.env_id}
    )

    env = robomme_raw.RobommeRawEpisodeEnv(
        task_name="SwingXtimes",
        split="test",
        episode_indices=[0, 1],
        builder_cls=_FakeRawBuilder,
    )

    obs0, info0 = env.reset()
    obs1, info1 = env.reset()

    assert len(created_envs) == 1
    assert obs0 == {"env_id": 0}
    assert obs1 == {"env_id": 0}
    assert info0["episode_index"] == 0
    assert info1["episode_index"] == 1
    assert created_envs[0].closed is False
    assert created_envs[0].reset_calls == [
        (100, {"reconfigure": True}),
        (101, {"reconfigure": True}),
    ]

    env.close()
    assert created_envs[0].closed is True


def test_robomme_raw_reset_retries_scene_generation_failures(monkeypatch):
    class FlakyResetRawEnv(_FakeResetRawEnv):
        def reset(self, seed=None, options=None):
            self.reset_calls.append((seed, dict(options or {})))
            if len(self.reset_calls) == 1:
                raise RuntimeError("spawn_random_cube: Region crowded or constraints too tight")
            return {"env_id": self.env_id}, {}

    created_envs = []

    def fake_make(*args, **kwargs):
        raw = FlakyResetRawEnv(len(created_envs))
        created_envs.append(raw)
        return raw

    monkeypatch.setattr(robomme_raw, "_ensure_robomme_env_registered", lambda: None)
    monkeypatch.setattr(robomme_raw.gym, "make", fake_make)
    monkeypatch.setattr(
        robomme_raw, "_convert_obs", lambda raw_obs, raw_env: {"env_id": raw_env.env_id}
    )

    env = robomme_raw.RobommeRawEpisodeEnv(
        task_name="VideoRepick",
        split="test",
        episode_indices=[0],
        builder_cls=_FakeRawBuilder,
    )

    _, info = env.reset()

    assert info["episode_index"] == 0
    assert info["episode_seed"] == 101
    assert info["reset_attempts"] == 2
    assert created_envs[0].reset_calls == [
        (100, {"reconfigure": True}),
        (101, {"reconfigure": True}),
    ]

    env.close()


def test_robomme_raw_reset_skips_episode_after_exhausted_scene_generation(monkeypatch):
    class EpisodeSkippingRawEnv(_FakeResetRawEnv):
        def reset(self, seed=None, options=None):
            self.reset_calls.append((seed, dict(options or {})))
            if seed == 100:
                raise RuntimeError("spawn_random_cube: Region crowded or constraints too tight")
            return {"env_id": self.env_id}, {}

    created_envs = []

    def fake_make(*args, **kwargs):
        raw = EpisodeSkippingRawEnv(len(created_envs))
        created_envs.append(raw)
        return raw

    monkeypatch.setattr(robomme_raw, "_MAX_SCENE_RESET_ATTEMPTS", 1)
    monkeypatch.setattr(robomme_raw, "_ensure_robomme_env_registered", lambda: None)
    monkeypatch.setattr(robomme_raw.gym, "make", fake_make)
    monkeypatch.setattr(
        robomme_raw, "_convert_obs", lambda raw_obs, raw_env: {"env_id": raw_env.env_id}
    )

    env = robomme_raw.RobommeRawEpisodeEnv(
        task_name="VideoRepick",
        split="test",
        episode_indices=[0, 1],
        builder_cls=_FakeRawBuilder,
    )

    _, info = env.reset()

    assert info["episode_index"] == 1
    assert info["episode_seed"] == 101
    assert info["reset_attempts"] == 1
    assert info["skipped_episode_indices"] == [0]
    assert created_envs[0].reset_calls == [
        (100, {"reconfigure": True}),
        (101, {"reconfigure": True}),
    ]

    env.close()


def test_robomme_correct_timestep_handles_missing_stop_timestep():
    class DummyEnv:
        elapsed_steps = 0

    assert correct_timestep(DummyEnv(), time_range=(10, 20), stop_timestep=None) is False
    assert correct_timestep(DummyEnv(), time_range=None, stop_timestep=12) is False
    assert correct_timestep(DummyEnv(), time_range=(10, 20), stop_timestep=12) is True
    assert correct_timestep(DummyEnv(), time_range=(10, 20), stop_timestep=25) is False
