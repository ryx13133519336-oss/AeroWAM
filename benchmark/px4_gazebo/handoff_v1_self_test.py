"""Non-flight acceptance test for Static Occlusion Handoff v1."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import jsonschema

from .benchmark_runner import run_episode
from .contract import HandoffTask
from .evaluator import aggregate, episode_metrics, replay_metric_differences
from .policies import FixedRoleRule, HandoffRulePolicy, OracleHandoffPolicy


ROOT = Path(__file__).resolve().parents[1]


def _events(record: dict, event_type: str) -> list[dict]:
    return [
        event
        for step in record["steps"]
        for event in step["events"]
        if event["event_type"] == event_type
    ]


def main() -> None:
    common = {"bandwidth": 4096, "latency_ms": 100.0, "drop_probability": 0.0}

    fixed = run_episode(0, policy=FixedRoleRule(), **common)
    assert fixed["outcome"]["termination_reason"] == "timeout"
    assert not fixed["outcome"]["success"]
    assert episode_metrics(fixed)["handoff_success"] == 0.0

    oracle = run_episode(0, policy=OracleHandoffPolicy(), **common)
    rule = run_episode(0, policy=HandoffRulePolicy(), **common)
    for record in (oracle, rule):
        assert record["outcome"]["termination_reason"] == "success"
        assert record["outcome"]["success"]
        metrics = episode_metrics(record)
        assert metrics["task_success"] == 1.0
        assert metrics["handoff_success"] == 1.0
        assert metrics["confirmation_success"] == 1.0
        assert metrics["collision_rate"] == 0.0
        assert metrics["visible_fraction_union"] == 1.0
        assert not replay_metric_differences(record)

    duplicate = run_episode(0, policy=HandoffRulePolicy(), **common)
    assert rule == duplicate, "same seed/config must produce byte-equivalent records"

    delayed = run_episode(
        1,
        policy=OracleHandoffPolicy(),
        bandwidth=4096,
        latency_ms=300.0,
        drop_probability=0.0,
    )
    deliveries = _events(delayed, "handoff_delivery")
    assert deliveries
    assert deliveries[0]["attributes"]["latency_steps"] == 3
    delivered_messages = [
        message
        for step in delayed["steps"]
        for message in step["communication"]["delivered"]
    ]
    assert any(
        message["deliver_timestamp_ms"] - message["sent_timestamp_ms"] == 300
        for message in delivered_messages
    )

    dropped = run_episode(
        2,
        policy=HandoffRulePolicy(),
        bandwidth=4096,
        latency_ms=100.0,
        drop_probability=1.0,
    )
    dropped_metrics = episode_metrics(dropped)
    assert dropped["outcome"]["termination_reason"] == "timeout"
    assert dropped_metrics["communication_dropped"] > 0
    assert dropped_metrics["communication_messages_delivered"] == 0

    collision_task = replace(
        HandoffTask(),
        horizon=10,
        initial_positions={"uav_0": (0.0, 0.0, 3.0), "uav_1": (0.0, 0.0, 3.0)},
    )
    collision = run_episode(3, task=collision_task, policy=FixedRoleRule(), **common)
    assert collision["outcome"]["termination_reason"] == "collision"
    assert episode_metrics(collision)["collision_rate"] == 1.0

    schema = json.loads((ROOT / "spec" / "episode_schema_v0_2.json").read_text(encoding="utf-8"))
    for record in (fixed, oracle, rule, delayed, dropped, collision):
        jsonschema.validate(record, schema)
        assert not replay_metric_differences(record)

    summary = aggregate((fixed, oracle, rule), method="handoff-v1-self-test")
    assert summary["schema_version"] == "0.2"
    assert summary["num_episodes"] == 3
    assert summary["metrics"]["task_success"] == 2.0 / 3.0

    print("Static Occlusion Handoff v1 non-flight self-test: PASS")


if __name__ == "__main__":
    main()
