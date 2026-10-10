"""Non-flight checks for the live PX4-Gazebo benchmark logger."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from .live_benchmark_runner import (
    append_episode_record,
    validate_episode_record,
    validate_runtime_arguments,
)


def _runtime_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "altitude": 2.0,
        "preposition_distance": 1.5,
        "duration": 10.0,
        "countdown": 0.0,
        "prestream": 0.0,
        "telemetry_timeout": 2.0,
        "camera_timeout": 2.0,
        "startup_timeout": 30.0,
        "takeoff_timeout": 30.0,
        "landing_timeout": 45.0,
        "uav1_spawn_offset_ned": [0.0, 3.0, 0.0],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _expect_invalid(**overrides: object) -> None:
    try:
        validate_runtime_arguments(_runtime_args(**overrides))
    except ValueError:
        return
    raise AssertionError(f"expected invalid runtime arguments: {overrides}")


def _check_runtime_argument_validation() -> None:
    assert validate_runtime_arguments(_runtime_args()) == (0.0, 3.0, 0.0)
    _expect_invalid(altitude=0.0)
    _expect_invalid(altitude=8.01)
    _expect_invalid(preposition_distance=12.01)
    _expect_invalid(duration=float("nan"))
    _expect_invalid(telemetry_timeout=0.0)
    _expect_invalid(camera_timeout=float("inf"))
    _expect_invalid(startup_timeout=-1.0)
    _expect_invalid(takeoff_timeout=0.0)
    _expect_invalid(landing_timeout=0.0)
    _expect_invalid(countdown=-0.1)
    _expect_invalid(prestream=-0.1)
    _expect_invalid(uav1_spawn_offset_ned=[0.0, float("nan"), 0.0])
    _expect_invalid(uav1_spawn_offset_ned=[0.0, 1.0, 0.0])


def _sample_record() -> dict:
    observation = {
        "rgb_uri": "ros2:///camera/image?seq=1&stamp_ns=1",
        "own_state": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        "health": "ok",
        "last_message": None,
    }
    action = {"control": [0.0, 0.0, 0.0, 0.0], "valid": True}
    return {
        "schema_version": "0.1",
        "episode_id": "live-self-test",
        "task_id": "live_dual_control_observation_diagnostic",
        "split": "live_diagnostic",
        "seed": 0,
        "agent_ids": ["uav_0", "uav_1"],
        "environment": {
            "simulator": "px4_gazebo",
            "version": "self-test",
            "map_id": "default",
            "reset_hash": "self-test",
            "ros_distro": "humble",
            "vehicle_namespaces": ["uav_0", "uav_1"],
            "transport": "px4_uxrce_dds",
            "control_hz": 10.0,
            "setpoint_hz": 20.0,
            "camera": {
                "width": 320,
                "height": 240,
                "encoding": "raw_uint8",
                "topic_template": "/world/default/model/x500_mono_cam_{index}/image",
            },
        },
        "steps": [
            {
                "t": 0,
                "timestamp_ms": 1,
                "observations": {"uav_0": observation, "uav_1": dict(observation)},
                "actions": {"uav_0": action, "uav_1": dict(action)},
                "communication": {
                    "messages": [],
                    "bytes_sent": 0,
                    "dropped": False,
                },
                "events": [],
            }
        ],
        "outcome": {
            "terminated": True,
            "termination_reason": "success",
            "success": True,
            "metrics": {
                "control_success": 1.0,
                "pre_position_success": 1.0,
                "minimum_separation_m": 3.0,
            },
        },
    }


def main() -> None:
    _check_runtime_argument_validation()
    record = _sample_record()
    validate_episode_record(record)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "episode.jsonl"
        append_episode_record(record, output)
        lines = output.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["episode_id"] == "live-self-test"
        assert loaded["steps"][0]["observations"]["uav_0"]["health"] == "ok"
    print("live benchmark non-flight self-test: PASS")


if __name__ == "__main__":
    main()
