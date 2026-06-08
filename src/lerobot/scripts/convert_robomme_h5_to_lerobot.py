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

from __future__ import annotations

import argparse
import importlib.resources
import json
import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.robomme import ROBOMME_ACTION_SPACE_SHAPES, ROBOMME_SPLITS, ROBOMME_TASKS
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)

ROBOMME_DATASET_REPO_ID = "Yinpei/robomme_data_h5"
ROBOMME_DEFAULT_RAW_DIR = HF_LEROBOT_HOME / "raw" / "robomme_data_h5"
ROBOMME_WORKSPACE_METADATA_DIR = (
    Path(__file__).resolve().parents[3] / "robomme_benchmark" / "src" / "robomme" / "env_metadata"
)
ROBOMME_FPS = 30
EPISODE_NAME_RE = re.compile(r"^episode_(\d+)$")
TIMESTEP_NAME_RE = re.compile(r"^timestep_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RoboMME HDF5 demos into a local LeRobot dataset.")
    parser.add_argument("--raw-dir", type=Path, default=ROBOMME_DEFAULT_RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--action-space", choices=tuple(ROBOMME_ACTION_SPACE_SHAPES), default="ee_pose")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--with-subtasks",
        action="store_true",
        help="Include subtask annotation fields (simple_subgoal, grounded_subgoal, "
             "simple_subgoal_online, grounded_subgoal_online, is_subgoal_boundary) "
             "as extra columns in the parquet files. These fields are skipped by the "
             "LeRobot training pipeline and do not affect policy learning.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _parse_names(value: str, valid_values: tuple[str, ...]) -> list[str]:
    if value == "all":
        return list(valid_values)

    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("Expected at least one value or 'all'.")

    invalid = sorted(set(names) - set(valid_values))
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return names


def _split_values(split_value: str) -> list[str]:
    return _parse_names(split_value, ROBOMME_SPLITS)


def _task_values(task_value: str) -> list[str]:
    return _parse_names(task_value, ROBOMME_TASKS)


def _decode_scalar(value: Any) -> Any:
    if hasattr(value, "asstr"):
        return value.asstr()[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_scalar(value.item())
        return [_decode_scalar(item) for item in value.tolist()]
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return value


def _load_metadata_records(split: str, task: str) -> list[dict[str, Any]]:
    workspace_metadata_file = ROBOMME_WORKSPACE_METADATA_DIR / split / f"record_dataset_{task}_metadata.json"
    if workspace_metadata_file.is_file():
        with workspace_metadata_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Invalid metadata file for task '{task}' split '{split}'.")
        return records

    try:
        metadata_file = (
            importlib.resources.files("robomme")
            .joinpath("env_metadata")
            .joinpath(split)
            .joinpath(f"record_dataset_{task}_metadata.json")
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboMME metadata is unavailable because neither the workspace clone at "
            f"'{ROBOMME_WORKSPACE_METADATA_DIR}' nor the installed 'robomme' package could be found. "
            "Install robomme or keep the robomme_benchmark clone in the workspace."
        ) from exc

    with metadata_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Invalid metadata file for task '{task}' split '{split}'.")
    return records


def _raw_h5_path(raw_dir: Path, task: str) -> Path:
    return raw_dir / f"record_dataset_{task}.h5"


def _raw_archive_path(raw_dir: Path, task: str) -> Path:
    return raw_dir / f"record_dataset_{task}.h5.tar.xz"


def _extract_raw_h5_archive(archive_path: Path, target_path: Path) -> None:
    with tarfile.open(archive_path, mode="r:xz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".h5")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one HDF5 file inside {archive_path}, found {len(members)}.")

        member = members[0]
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not extract {member.name} from {archive_path}.")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            shutil.copyfileobj(extracted, handle)


def _ensure_raw_h5_files(raw_dir: Path, tasks: list[str], download_missing: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing_tasks = []
    for task in tasks:
        h5_path = _raw_h5_path(raw_dir, task)
        archive_path = _raw_archive_path(raw_dir, task)
        if h5_path.is_file():
            continue
        if archive_path.is_file():
            _extract_raw_h5_archive(archive_path, h5_path)
            continue
        missing_tasks.append(task)

    if not missing_tasks:
        return

    if not download_missing:
        missing_files = ", ".join(_raw_archive_path(raw_dir, task).name for task in missing_tasks)
        raise FileNotFoundError(
            f"Missing RoboMME raw files under {raw_dir}: {missing_files}. "
            "Re-run with --download-missing to fetch them automatically."
        )

    allow_patterns = [f"record_dataset_{task}.h5.tar.xz" for task in missing_tasks]
    snapshot_download(
        repo_id=ROBOMME_DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=allow_patterns,
    )

    for task in missing_tasks:
        archive_path = _raw_archive_path(raw_dir, task)
        h5_path = _raw_h5_path(raw_dir, task)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Downloaded archive not found: {archive_path}")
        if not h5_path.is_file():
            _extract_raw_h5_archive(archive_path, h5_path)


def _build_features(action_space: str, with_subtasks: bool = False) -> dict[str, dict[str, Any]]:
    raw_shape = ROBOMME_ACTION_SPACE_SHAPES[action_space]
    action_dim = raw_shape[0] if isinstance(raw_shape, (tuple, list)) else int(raw_shape)
    features: dict[str, dict[str, Any]] = {
        f"{OBS_IMAGES}.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        f"{OBS_IMAGES}.image2": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper_left",
                "gripper_right",
            ],
        },
        ACTION: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{index}" for index in range(action_dim)],
        },
    }
    if with_subtasks:
        features["simple_subgoal"] = {"dtype": "string", "shape": (1,), "names": None}
        features["grounded_subgoal"] = {"dtype": "string", "shape": (1,), "names": None}
        features["simple_subgoal_online"] = {"dtype": "string", "shape": (1,), "names": None}
        features["grounded_subgoal_online"] = {"dtype": "string", "shape": (1,), "names": None}
        features["is_subgoal_boundary"] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


def _resolve_output_root(output_root: Path | None, action_space: str, split: str, multi_split: bool) -> Path:
    if output_root is None:
        return Path("outputs/datasets") / f"robomme_{action_space}_{split}"
    return output_root / split if multi_split else output_root


def _resolve_repo_id(repo_id: str | None, action_space: str, split: str, multi_split: bool) -> str:
    if repo_id is None:
        return f"local/robomme_{action_space}_{split}"
    return f"{repo_id}_{split}" if multi_split else repo_id


def _episode_name_lookup(h5_file: Any) -> dict[int, str]:
    lookup = {}
    for name in h5_file.keys():
        match = EPISODE_NAME_RE.match(name)
        if match is not None:
            lookup[int(match.group(1))] = name
    return lookup


def _resolve_episode_group_name(h5_file: Any, episode_index: int) -> str:
    lookup = _episode_name_lookup(h5_file)
    if episode_index in lookup:
        return lookup[episode_index]
    if 0 not in lookup and episode_index + 1 in lookup:
        return lookup[episode_index + 1]
    raise KeyError(f"Episode {episode_index} was not found in HDF5 file.")


def _sorted_timestep_names(episode_group: Any) -> list[str]:
    timestep_names = []
    for name in episode_group.keys():
        match = TIMESTEP_NAME_RE.match(name)
        if match is not None:
            timestep_names.append((int(match.group(1)), name))
    return [name for _, name in sorted(timestep_names)]


def _extract_task_prompt(setup_group: Any, fallback_task: str) -> str:
    if "task_goal" not in setup_group:
        return fallback_task
    decoded = _decode_scalar(setup_group["task_goal"][()])
    if isinstance(decoded, list) and decoded:
        first = decoded[0]
        return first if isinstance(first, str) else fallback_task
    if isinstance(decoded, str):
        return decoded
    return fallback_task


def _resolve_subgoal_text(raw: Any, last_known: str | None) -> str:
    """Decode a raw HDF5 subgoal value and fall back to the last-known non-sentinel text.

    A subgoal whose decoded text contains 'complete' or 'done' is treated as a
    terminal sentinel and replaced with the previous valid subgoal.
    """
    text = _decode_scalar(raw)
    if isinstance(text, list):
        text = text[0] if text else ""
    if not isinstance(text, str):
        text = str(text)
    # sentinel heuristic: the official code uses 'complete' as a terminal marker
    if last_known is not None and any(kw in text.lower() for kw in ("complete", "done", "finished", "task complete")):
        return last_known
    return text


def _iter_execution_frames(
    episode_group: Any,
    action_space: str,
    task_prompt: str,
    with_subtasks: bool = False,
):
    action_key = "eef_action" if action_space == "ee_pose" else "joint_action"
    last_simple_subgoal: str | None = None
    last_grounded_subgoal: str | None = None

    for timestep_name in _sorted_timestep_names(episode_group):
        timestep_group = episode_group[timestep_name]
        info_group = timestep_group["info"]
        is_video_demo = bool(np.asarray(info_group["is_video_demo"][()]).item())

        if with_subtasks and is_video_demo:
            # Still track the running subgoal state from demo frames so that
            # the first execution frame picks up the correct context.
            if "is_completed" in info_group:
                is_completed = bool(np.asarray(info_group["is_completed"][()]).item())
            else:
                is_completed = False
            if not is_completed and "simple_subgoal" in info_group:
                candidate = _resolve_subgoal_text(info_group["simple_subgoal"][()], last_simple_subgoal)
                if candidate:
                    last_simple_subgoal = candidate
                candidate_g = _resolve_subgoal_text(info_group["grounded_subgoal"][()], last_grounded_subgoal)
                if candidate_g:
                    last_grounded_subgoal = candidate_g
            continue

        if is_video_demo:
            continue

        obs_group = timestep_group["obs"]
        action_group = timestep_group["action"]
        eef_state = np.asarray(obs_group["eef_state"][()], dtype=np.float32).reshape(-1)
        gripper_state = np.asarray(obs_group["gripper_state"][()], dtype=np.float32).reshape(-1)
        # Align eef_state RPY angles to match eef_action convention.
        # The HDF5 recorder uses sign-tracking continuity for eef_state, which can
        # yield roll ≈ -π when the physical angle is actually ≈ +π (quaternion double-cover).
        # eef_action is computed independently and consistently yields roll ≈ +π.
        # robomme_raw.py._eef_state() (used at eval time) also yields +π for this pose.
        # Solution: map angles near -π to their equivalent +π value so that
        # training obs and eval obs are consistent, and obs/action pair in the same frame agree.
        eef_state = eef_state.copy()
        eef_state[3:6] = np.where(
            eef_state[3:6] < -np.pi + 0.15,
            eef_state[3:6] + 2.0 * np.pi,
            eef_state[3:6],
        )
        action_raw = np.asarray(action_group[action_key][()], dtype=np.float32).reshape(-1)
        # Apply the same canonical mapping to action angles so both obs and action
        # use the same branch of the periodic Euler angle representation.
        if action_raw.shape[0] >= 6:
            action_raw = action_raw.copy()
            action_raw[3:6] = np.where(
                action_raw[3:6] < -np.pi + 0.15,
                action_raw[3:6] + 2.0 * np.pi,
                action_raw[3:6],
            )
        frame = {
            f"{OBS_IMAGES}.image": np.asarray(obs_group["front_rgb"][()], dtype=np.uint8),
            f"{OBS_IMAGES}.image2": np.asarray(obs_group["wrist_rgb"][()], dtype=np.uint8),
            OBS_STATE: np.concatenate((eef_state, gripper_state), axis=0).astype(np.float32, copy=False),
            ACTION: action_raw,
            "task": task_prompt,
        }

        if with_subtasks:
            if "is_completed" in info_group:
                is_completed = bool(np.asarray(info_group["is_completed"][()]).item())
            else:
                is_completed = False

            if "is_subgoal_boundary" in info_group:
                is_subgoal_boundary = bool(np.asarray(info_group["is_subgoal_boundary"][()]).item())
            else:
                is_subgoal_boundary = False

            if not is_completed and "simple_subgoal" in info_group:
                simple_subgoal = _resolve_subgoal_text(
                    info_group["simple_subgoal"][()], last_simple_subgoal
                )
                grounded_subgoal = _resolve_subgoal_text(
                    info_group["grounded_subgoal"][()], last_grounded_subgoal
                )
                simple_subgoal_online = _resolve_subgoal_text(
                    info_group["simple_subgoal_online"][()], last_simple_subgoal
                )
                grounded_subgoal_online = _resolve_subgoal_text(
                    info_group["grounded_subgoal_online"][()], last_grounded_subgoal
                )
                if simple_subgoal:
                    last_simple_subgoal = simple_subgoal
                if grounded_subgoal:
                    last_grounded_subgoal = grounded_subgoal
            else:
                # Terminal / completed frame — use last known subgoal text
                simple_subgoal = last_simple_subgoal or ""
                grounded_subgoal = last_grounded_subgoal or ""
                simple_subgoal_online = simple_subgoal
                grounded_subgoal_online = grounded_subgoal

            frame["simple_subgoal"] = simple_subgoal
            frame["grounded_subgoal"] = grounded_subgoal
            frame["simple_subgoal_online"] = simple_subgoal_online
            frame["grounded_subgoal_online"] = grounded_subgoal_online
            frame["is_subgoal_boundary"] = np.array([is_subgoal_boundary], dtype=np.bool_)

        yield frame


def convert_robomme_h5_to_lerobot(
    *,
    raw_dir: Path,
    output_root: Path | None,
    repo_id: str | None,
    split: str,
    tasks: str,
    action_space: str,
    max_episodes_per_task: int | None,
    download_missing: bool = False,
    overwrite: bool = False,
    with_subtasks: bool = False,
) -> list[tuple[str, Path]]:
    import h5py

    splits = _split_values(split)
    task_names = _task_values(tasks)
    _ensure_raw_h5_files(raw_dir, task_names, download_missing)

    outputs: list[tuple[str, Path]] = []
    multi_split = len(splits) > 1
    for split_name in splits:
        split_output_root = _resolve_output_root(output_root, action_space, split_name, multi_split)
        split_repo_id = _resolve_repo_id(repo_id, action_space, split_name, multi_split)

        if split_output_root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output root already exists: {split_output_root}. Re-run with --overwrite to replace it."
                )
            shutil.rmtree(split_output_root)

        dataset = LeRobotDataset.create(
            repo_id=split_repo_id,
            root=split_output_root,
            fps=ROBOMME_FPS,
            features=_build_features(action_space, with_subtasks=with_subtasks),
            use_videos=False,
        )

        converted_episode_count = 0
        try:
            for task_name in task_names:
                records = _load_metadata_records(split_name, task_name)
                if max_episodes_per_task is not None:
                    records = records[:max_episodes_per_task]

                h5_path = _raw_h5_path(raw_dir, task_name)
                with h5py.File(h5_path, "r") as h5_file:
                    for record in records:
                        episode_index = int(record["episode"])
                        episode_group_name = _resolve_episode_group_name(h5_file, episode_index)
                        episode_group = h5_file[episode_group_name]
                        task_prompt = _extract_task_prompt(episode_group["setup"], task_name)

                        frame_count = 0
                        for frame in _iter_execution_frames(
                            episode_group, action_space, task_prompt, with_subtasks=with_subtasks
                        ):
                            dataset.add_frame(frame)
                            frame_count += 1

                        if frame_count == 0:
                            logger.warning(
                                "Skipping task=%s split=%s episode=%s because it contains no execution frames.",
                                task_name,
                                split_name,
                                episode_index,
                            )
                            continue

                        dataset.save_episode()
                        converted_episode_count += 1
        finally:
            dataset.finalize()

        logger.info(
            "Converted %s episodes to %s (repo_id=%s).",
            converted_episode_count,
            split_output_root,
            split_repo_id,
        )
        outputs.append((split_repo_id, split_output_root))

    return outputs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    outputs = convert_robomme_h5_to_lerobot(
        raw_dir=args.raw_dir,
        output_root=args.output_root,
        repo_id=args.repo_id,
        split=args.split,
        tasks=args.tasks,
        action_space=args.action_space,
        max_episodes_per_task=args.max_episodes_per_task,
        download_missing=args.download_missing,
        overwrite=args.overwrite,
        with_subtasks=args.with_subtasks,
    )
    for resolved_repo_id, resolved_root in outputs:
        print(f"repo_id={resolved_repo_id}")
        print(f"root={resolved_root}")


if __name__ == "__main__":
    main()