"""Replay-safe metric computation from schema 0.2 episode JSONL only."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .contract import AGENT_IDS


def _position(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def _events(steps: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [event for step in steps for event in step.get("events", [])]


def _successful(events: list[Mapping[str, Any]], event_type: str) -> list[Mapping[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type and event.get("success")]


def _visible(step: Mapping[str, Any], agent_id: str) -> bool:
    truth = step.get("truth", {})
    visibility = truth.get("target_visibility")
    if isinstance(visibility, Mapping) and agent_id in visibility:
        return bool(visibility[agent_id])
    return agent_id in truth.get("target_visible_to", [])


def _longest_interruption(steps: list[Mapping[str, Any]]) -> int:
    longest = current = 0
    for step in steps:
        if any(_visible(step, agent_id) for agent_id in AGENT_IDS):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _step_position(
    step: Mapping[str, Any], agent_id: str
) -> tuple[float, float, float] | None:
    truth_positions = step.get("truth", {}).get("agent_positions", {})
    if isinstance(truth_positions, Mapping):
        truth_position = _position(truth_positions.get(agent_id))
        if truth_position is not None:
            return truth_position
    observations = step.get("observations", {})
    return _position(observations.get(agent_id, {}).get("own_state", []))


def _uses_common_truth(steps: list[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(step.get("truth", {}).get("agent_positions"), Mapping)
        for step in steps
    )


def _path_lengths(record: Mapping[str, Any], steps: list[Mapping[str, Any]]) -> dict[str, float]:
    initial_positions = record.get("task_spec", {}).get("initial_positions", {})
    previous: dict[str, tuple[float, float, float] | None]
    if _uses_common_truth(steps):
        previous = {agent_id: None for agent_id in AGENT_IDS}
    else:
        previous = {
            agent_id: _position(initial_positions.get(agent_id))
            for agent_id in AGENT_IDS
        }
    totals = {agent_id: 0.0 for agent_id in AGENT_IDS}
    for step in steps:
        for agent_id in AGENT_IDS:
            current = _step_position(step, agent_id)
            if current is not None and previous[agent_id] is not None:
                totals[agent_id] += _distance(previous[agent_id], current)
            if current is not None:
                previous[agent_id] = current
    return totals


def _separations(
    record: Mapping[str, Any], steps: list[Mapping[str, Any]]
) -> list[float]:
    positions: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    if not _uses_common_truth(steps):
        initial = record.get("task_spec", {}).get("initial_positions", {})
        initial_0 = _position(initial.get("uav_0"))
        initial_1 = _position(initial.get("uav_1"))
        if initial_0 is not None and initial_1 is not None:
            positions.append((initial_0, initial_1))
    for step in steps:
        position_0 = _step_position(step, "uav_0")
        position_1 = _step_position(step, "uav_1")
        if position_0 is not None and position_1 is not None:
            positions.append((position_0, position_1))
    if not positions:
        initial = record.get("task_spec", {}).get("initial_positions", {})
        initial_0 = _position(initial.get("uav_0"))
        initial_1 = _position(initial.get("uav_1"))
        if initial_0 is not None and initial_1 is not None:
            positions.append((initial_0, initial_1))
    return [_distance(a, b) for a, b in positions]


def _flight_time_s(steps: list[Mapping[str, Any]], control_hz: float) -> float:
    if _uses_common_truth(steps) and len(steps) >= 2:
        first_timestamp = steps[0].get("timestamp_ms")
        last_timestamp = steps[-1].get("timestamp_ms")
        if isinstance(first_timestamp, (int, float)) and isinstance(
            last_timestamp, (int, float)
        ):
            return max(0.0, (float(last_timestamp) - float(first_timestamp)) / 1000.0)
    if _uses_common_truth(steps):
        return 0.0
    return float(len(steps) / control_hz)


def _event_latency(events: list[Mapping[str, Any]], event_type: str, fallback: int) -> float:
    candidates = _successful(events, event_type)
    if not candidates:
        return float(fallback)
    return float(min(int(event.get("end_t", fallback)) - int(event.get("start_t", 0)) for event in candidates))


def episode_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    """Derive metrics exclusively from one serialized episode record."""

    steps = list(record.get("steps", []))
    events = _events(steps)
    task_spec = record.get("task_spec", {})
    horizon = int(task_spec.get("horizon", len(steps)))
    interruption_limit = int(task_spec.get("interruption_limit_steps", horizon))
    control_hz = float(record.get("environment", {}).get("control_hz", task_spec.get("control_hz", 10.0)))

    pre_position = bool(_successful(events, "pre_position"))
    handoff = bool(_successful(events, "handoff"))
    confirmation = bool(_successful(events, "verify"))
    collision = any(event.get("event_type") == "collision" for event in events) or any(
        bool(step.get("truth", {}).get("collision")) for step in steps
    )
    longest_interruption = _longest_interruption(steps)
    handoff_events = _successful(events, "handoff")
    handoff_step = (
        int(handoff_events[0].get("end_t", horizon))
        if handoff_events else horizon
    )
    handoff_step_record = next(
        (step for step in steps if int(step.get("t", -1)) == handoff_step),
        None,
    )
    target_visible_during_handoff = bool(
        handoff_step_record is not None
        and _visible(handoff_step_record, "uav_1")
    )
    request_events = _successful(events, "handoff_request")
    delivery_events = _successful(events, "handoff_delivery")
    verify_events = _successful(events, "verify")

    sent = [message for step in steps for message in step.get("communication", {}).get("sent", step.get("communication", {}).get("messages", []))]
    delivered = [message for step in steps for message in step.get("communication", {}).get("delivered", [])]
    dropped = [message for message in sent if message.get("dropped")]
    communication_bytes = sum(int(message.get("bytes", 0)) for message in sent if not message.get("dropped"))

    paths = _path_lengths(record, steps)
    visible_counts = {
        agent_id: sum(_visible(step, agent_id) for step in steps)
        for agent_id in AGENT_IDS
    }
    union_visible = sum(any(_visible(step, agent_id) for agent_id in AGENT_IDS) for step in steps)
    denominator = max(1, len(steps))
    identity_errors = len([event for event in events if event.get("event_type") == "identity_error"])
    identity_confirms = len(_successful(events, "identity_confirm"))
    policy_rows = [step["policy"] for step in steps if isinstance(step.get("policy"), Mapping)]
    inference_latencies = sorted(
        float(row["inference_latency_ms"])
        for row in policy_rows
        if isinstance(row.get("inference_latency_ms"), (int, float))
    )
    inference_p95 = 0.0
    if inference_latencies:
        inference_p95 = inference_latencies[
            min(len(inference_latencies) - 1, math.ceil(0.95 * len(inference_latencies)) - 1)
        ]
    separations = _separations(record, steps)

    return {
        "task_success": float(confirmation and not collision and longest_interruption <= interruption_limit),
        "pre_position_success": float(pre_position),
        "handoff_success": float(handoff),
        "confirmation_success": float(confirmation),
        "role_switch_success": float(handoff and confirmation),
        "target_visible_during_handoff": float(target_visible_during_handoff),
        "handoff_request_step": float(
            int(request_events[0].get("end_t", horizon)) if request_events else horizon
        ),
        "handoff_delivery_step": float(
            int(delivery_events[0].get("end_t", horizon)) if delivery_events else horizon
        ),
        "tracker_switch_step": float(handoff_step),
        "verify_step": float(
            int(verify_events[0].get("end_t", horizon)) if verify_events else horizon
        ),
        "recovery_latency_steps": _event_latency(events, "target_recovery", horizon),
        "handoff_latency_steps": _event_latency(events, "handoff", horizon),
        "longest_interruption_steps": float(longest_interruption),
        "communication_bytes": float(communication_bytes),
        "communication_messages_sent": float(len(sent)),
        "communication_messages_delivered": float(len(delivered)),
        "communication_dropped": float(len(dropped)),
        "collision_rate": float(collision),
        "minimum_separation_m": float(min(separations, default=float("nan"))),
        "mean_separation_m": float(
            sum(separations) / len(separations) if separations else float("nan")
        ),
        "maximum_separation_m": float(max(separations, default=float("nan"))),
        "path_length_uav_0": float(paths["uav_0"]),
        "path_length_uav_1": float(paths["uav_1"]),
        "path_length": float(sum(paths.values())),
        "flight_time_s": _flight_time_s(steps, control_hz),
        "visible_fraction_uav_0": float(visible_counts["uav_0"] / denominator),
        "visible_fraction_uav_1": float(visible_counts["uav_1"] / denominator),
        "visible_fraction_union": float(union_visible / denominator),
        "identity_error_rate": float(identity_errors / max(1, identity_errors + identity_confirms)),
        "inference_latency_ms_mean": float(sum(inference_latencies) / max(1, len(inference_latencies))),
        "inference_latency_ms_p95": float(inference_p95),
        "policy_timeout_count": float(sum(bool(row.get("timed_out", False)) for row in policy_rows)),
    }


def replay_metric_differences(record: Mapping[str, Any], *, absolute_tolerance: float = 1e-9) -> dict[str, tuple[float, float]]:
    """Return overlapping outcome metrics that disagree with replay."""

    replayed = episode_metrics(record)
    recorded = record.get("outcome", {}).get("metrics", {})
    differences: dict[str, tuple[float, float]] = {}
    for key in replayed.keys() & recorded.keys():
        actual = float(replayed[key])
        expected = float(recorded[key])
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=absolute_tolerance):
            differences[key] = (expected, actual)
    return differences


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate(records: Iterable[Mapping[str, Any]], method: str = "unknown") -> dict[str, Any]:
    rows = [episode_metrics(record) for record in records]
    if not rows:
        raise ValueError("cannot aggregate empty records")
    return {
        "schema_version": "0.2",
        "method": method,
        "num_episodes": len(rows),
        "metrics": {key: mean(row[key] for row in rows) for key in rows[0]},
        "episodes": rows,
    }
