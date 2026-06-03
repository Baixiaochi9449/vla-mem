from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.convert_robomme_h5_to_lerobot import convert_robomme_h5_to_lerobot
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


try:
    import h5py
except ModuleNotFoundError:
    h5py = None


def _write_scalar_string(group, key: str, value: str) -> None:
    group.create_dataset(key, data=np.bytes_(value))


def _make_episode(h5_file, episode_group_name: str) -> None:
    episode_group = h5_file.create_group(episode_group_name)
    setup_group = episode_group.create_group("setup")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    setup_group.create_dataset("task_goal", data=np.array(["pick the highlighted cube"], dtype=string_dtype))

    timestep_demo = episode_group.create_group("timestep_1")
    obs_demo = timestep_demo.create_group("obs")
    obs_demo.create_dataset("front_rgb", data=np.zeros((256, 256, 3), dtype=np.uint8))
    obs_demo.create_dataset("wrist_rgb", data=np.zeros((256, 256, 3), dtype=np.uint8))
    obs_demo.create_dataset("eef_state", data=np.zeros((6,), dtype=np.float32))
    obs_demo.create_dataset("gripper_state", data=np.zeros((2,), dtype=np.float32))
    action_demo = timestep_demo.create_group("action")
    action_demo.create_dataset("eef_action", data=np.zeros((7,), dtype=np.float32))
    action_demo.create_dataset("joint_action", data=np.zeros((8,), dtype=np.float32))
    info_demo = timestep_demo.create_group("info")
    info_demo.create_dataset("is_video_demo", data=np.array(True, dtype=np.bool_))
    info_demo.create_dataset("is_completed", data=np.array(False, dtype=np.bool_))
    _write_scalar_string(info_demo, "simple_subgoal", "demo")

    timestep_exec = episode_group.create_group("timestep_2")
    obs_exec = timestep_exec.create_group("obs")
    obs_exec.create_dataset("front_rgb", data=np.full((256, 256, 3), 7, dtype=np.uint8))
    obs_exec.create_dataset("wrist_rgb", data=np.full((256, 256, 3), 17, dtype=np.uint8))
    obs_exec.create_dataset("eef_state", data=np.arange(6, dtype=np.float32))
    obs_exec.create_dataset("gripper_state", data=np.array([0.25, 0.75], dtype=np.float32))
    action_exec = timestep_exec.create_group("action")
    action_exec.create_dataset("eef_action", data=np.arange(7, dtype=np.float32))
    action_exec.create_dataset("joint_action", data=np.arange(8, dtype=np.float32))
    info_exec = timestep_exec.create_group("info")
    info_exec.create_dataset("is_video_demo", data=np.array(False, dtype=np.bool_))
    info_exec.create_dataset("is_completed", data=np.array(True, dtype=np.bool_))
    _write_scalar_string(info_exec, "simple_subgoal", "execute")


@pytest.mark.skipif(h5py is None, reason="h5py is required for RoboMME conversion tests")
def test_convert_robomme_h5_to_lerobot_creates_local_dataset(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    h5_path = raw_dir / "record_dataset_PickXtimes.h5"
    with h5py.File(h5_path, "w") as h5_file:
        _make_episode(h5_file, "episode_1")

    monkeypatch.setattr(
        "lerobot.scripts.convert_robomme_h5_to_lerobot._load_metadata_records",
        lambda split, task: [{"episode": 0, "seed": 1000, "difficulty": "easy"}],
    )

    output_root = tmp_path / "converted"
    outputs = convert_robomme_h5_to_lerobot(
        raw_dir=raw_dir,
        output_root=output_root,
        repo_id="local/robomme_ee_pose_train",
        split="train",
        tasks="PickXtimes",
        action_space="ee_pose",
        max_episodes_per_task=1,
        download_missing=False,
        overwrite=False,
    )

    assert outputs == [("local/robomme_ee_pose_train", output_root)]

    dataset = LeRobotDataset("local/robomme_ee_pose_train", root=output_root)
    raw_item = dataset.get_raw_item(0)

    assert dataset.num_episodes == 1
    assert len(dataset) == 1
    assert f"{OBS_IMAGES}.image" in dataset.features
    assert f"{OBS_IMAGES}.image2" in dataset.features
    assert OBS_STATE in dataset.features
    assert ACTION in dataset.features
    assert raw_item[ACTION].shape == (7,)
    assert raw_item[OBS_STATE].shape == (8,)
    assert list(dataset.meta.tasks.index) == ["pick the highlighted cube"]