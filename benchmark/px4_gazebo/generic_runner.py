"""Run a deterministic rule policy against any registered mock task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import Action, JointAction
from .task_registry import create_task


def run_episode(task_id: str, seed: int, *, task_spec=None) -> dict:
    task = create_task(task_id, task_spec)
    observation = task.reset(seed, task_spec)
    for _ in range(int(task_spec.get("episode_horizon", 240)) if task_spec else 240):
        del observation
        observation, _, done, _ = task.step(JointAction(uav_0=Action(), uav_1=Action()))
        if done:
            break
    return dict(task.episode_record())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [run_episode(args.task_id, seed) for seed in args.seeds]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(json.dumps([record["outcome"] for record in records], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
