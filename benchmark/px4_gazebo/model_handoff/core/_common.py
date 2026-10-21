"""Small validation helpers shared by the core schemas."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
EVENT = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
FORBIDDEN_PUBLIC_KEYS = frozenset({
    "truth", "target_pose", "target_position", "target_id", "target_identity",
    "instance_mask", "instance_masks", "occluder", "occluder_center",
    "occluder_half_extents", "global_map", "map", "future", "future_frames",
    "future_trajectory", "route_progress", "evaluator", "fixture_truth",
    "teammate_response_schedule", "private_visibility", "private_targets_enu",
    "candidate_actual_utility", "target_visibility_truth", "private_local_goals",
})


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def event_name(value: Any, name: str = "event") -> str:
    if not isinstance(value, str) or not EVENT.fullmatch(value):
        raise ValueError(f"{name} must be namespaced as task.event")
    return value


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def unique_strings(value: Any, name: str, *, min_items: int = 1) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < min_items:
        raise ValueError(f"{name} must contain at least {min_items} strings")
    result = tuple(identifier(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def assert_json_safe(value: Any, path: str = "value") -> None:
    """Reject NaN/Infinity and non-JSON values before persistence."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string key at {path}")
            assert_json_safe(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            assert_json_safe(item, f"{path}[{index}]")
        return
    raise ValueError(f"non-JSON value at {path}: {type(value).__name__}")


def assert_public(value: Any, path: str = "public") -> None:
    """Reject known evaluator-only fields recursively from model-visible data."""
    assert_json_safe(value, path)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"forbidden public field at {path}.{key}")
            assert_public(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_public(item, f"{path}[{index}]")


def canonical(value: Any) -> bytes:
    assert_json_safe(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
