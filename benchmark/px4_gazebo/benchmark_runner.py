"""Run a reference policy through the deterministic handoff benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from .communication import CommunicationChannel
from .contract import HandoffTask, JointAction
from .evaluator import aggregate
from .handoff_env import HandoffEnv
from .paths import get_storage_paths
from .policies import FixedRoleRule, HandoffRulePolicy, OracleHandoffPolicy


class Policy(Protocol):
    name: str

    def reset(self, task: HandoffTask) -> None: ...

    def act(self, observation: Mapping[str, Any]) -> JointAction: ...


POLICIES: dict[str, type[FixedRoleRule] | type[OracleHandoffPolicy] | type[HandoffRulePolicy]] = {
    "fixed": FixedRoleRule,
    "oracle": OracleHandoffPolicy,
    "rule": HandoffRulePolicy,
}


def run_episode(
    seed: int,
    *,
    task: HandoffTask | None = None,
    policy: Policy | None = None,
    bandwidth: int = 64,
    latency_ms: float = 100.0,
    drop_probability: float = 0.1,
) -> dict[str, Any]:
    task = task or HandoffTask()
    policy = policy or FixedRoleRule()
    channel = CommunicationChannel(
        bandwidth_bytes_per_second=bandwidth,
        latency_ms=latency_ms,
        drop_probability=drop_probability,
        control_hz=task.control_hz,
        seed=seed,
    )
    env = HandoffEnv(task, seed=seed, communication=channel)
    observation = env.reset(seed, task)
    policy.reset(task)
    for _ in range(task.horizon):
        joint_action = policy.act(observation)
        observation, _, terminated, _ = env.step(joint_action)
        if terminated:
            break
    return env.episode_record()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="fixed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bandwidth", type=int, default=64)
    parser.add_argument("--latency-ms", type=float, default=100.0)
    parser.add_argument("--drop-probability", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()

    policy_type = POLICIES[args.policy]
    default_output = get_storage_paths().episodes / f"{args.policy}_handoff_v1.jsonl"
    output = args.output or default_output
    records = [
        run_episode(
            seed,
            policy=policy_type(),
            bandwidth=args.bandwidth,
            latency_ms=args.latency_ms,
            drop_probability=args.drop_probability,
        )
        for seed in args.seeds
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    result = aggregate(records, policy_type.name)
    metrics_output = args.metrics_output or output.with_name(f"{output.stem}_metrics.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {len(records)} episodes to {output}")
    print(f"wrote aggregate metrics to {metrics_output}")


if __name__ == "__main__":
    main()
