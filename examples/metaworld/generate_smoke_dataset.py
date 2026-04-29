#!/usr/bin/env python
#可以删掉
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.metaworld import MetaworldEnv
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny local MetaWorld dataset for smoke training.")
    parser.add_argument("--task", type=str, default="metaworld-push-v3")
    parser.add_argument("--repo-id", type=str, default="local/metaworld_pi05_v2_smoke")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/datasets/metaworld_pi05_v2_smoke"),
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use-expert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_features() -> dict[str, dict]:
    return {
        OBS_IMAGE: {
            "dtype": "image",
            "shape": (480, 480, 3),
            "names": ["height", "width", "channels"],
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "gripper"],
        },
        ACTION: {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
    }


def main() -> None:
    args = parse_args()
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset root already exists: {args.root}. Re-run with --overwrite.")
        shutil.rmtree(args.root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        fps=80,
        features=build_features(),
        use_videos=False,
    )

    env = MetaworldEnv(task=args.task, obs_type="pixels_agent_pos")

    try:
        for episode_idx in range(args.episodes):
            raw_obs, _ = env._env.reset(seed=args.seed + episode_idx)
            for _step_idx in range(args.max_steps):
                if args.use_expert:
                    action = env.expert_policy.get_action(raw_obs)
                else:
                    action = env.action_space.sample()

                frame = {
                    OBS_IMAGE: env.render(),
                    OBS_STATE: raw_obs[:4].astype(np.float32),
                    ACTION: np.asarray(action, dtype=np.float32),
                    "task": env.task_description,
                }
                dataset.add_frame(frame)

                raw_obs, _reward, terminated, truncated, info = env._env.step(action)
                if terminated or truncated or bool(info.get("success", 0)):
                    break

            dataset.save_episode()
    finally:
        dataset.finalize()
        env.close()

    print(f"Saved dataset to {args.root}")
    print(f"repo_id={args.repo_id}")
    print(f"episodes={args.episodes}")


if __name__ == "__main__":
    main()