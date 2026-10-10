"""Deterministic geometry helpers for the static-occlusion benchmark."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

Vec3 = tuple[float, float, float]


def as_vec3(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("expected a three-dimensional vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("vector contains a non-finite value")
    return result  # type: ignore[return-value]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    lhs, rhs = as_vec3(a), as_vec3(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(lhs, rhs)))


def target_at_step(start: Sequence[float], end: Sequence[float], step: int, horizon: int) -> Vec3:
    ratio = min(1.0, max(0.0, float(step) / max(1, int(horizon) - 1)))
    lhs, rhs = as_vec3(start), as_vec3(end)
    return tuple(a + ratio * (b - a) for a, b in zip(lhs, rhs))  # type: ignore[return-value]


def point_inside_aabb(point: Sequence[float], center: Sequence[float], half_extents: Sequence[float], *, epsilon: float = 1e-9) -> bool:
    p, c, h = as_vec3(point), as_vec3(center), as_vec3(half_extents)
    return all(abs(p[i] - c[i]) <= h[i] + epsilon for i in range(3))


def point_aabb_distance(
    point: Sequence[float], center: Sequence[float], half_extents: Sequence[float]
) -> float:
    """Euclidean clearance from a point to an axis-aligned box."""
    p, c, h = as_vec3(point), as_vec3(center), as_vec3(half_extents)
    offsets = [max(0.0, abs(p[index] - c[index]) - h[index]) for index in range(3)]
    return math.sqrt(sum(value * value for value in offsets))


def segment_intersects_aabb(start: Sequence[float], end: Sequence[float], center: Sequence[float], half_extents: Sequence[float], *, epsilon: float = 1e-9) -> bool:
    """Return whether the closed line segment intersects an axis-aligned box."""

    p0, p1, c, h = as_vec3(start), as_vec3(end), as_vec3(center), as_vec3(half_extents)
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        lower, upper = c[axis] - h[axis], c[axis] + h[axis]
        delta = p1[axis] - p0[axis]
        if abs(delta) <= epsilon:
            if p0[axis] < lower - epsilon or p0[axis] > upper + epsilon:
                return False
            continue
        inv = 1.0 / delta
        enter, leave = (lower - p0[axis]) * inv, (upper - p0[axis]) * inv
        if enter > leave:
            enter, leave = leave, enter
        t_min, t_max = max(t_min, enter), min(t_max, leave)
        if t_min > t_max + epsilon:
            return False
    return t_max >= -epsilon and t_min <= 1.0 + epsilon


def target_visible(observer: Sequence[float], target: Sequence[float], *, occluder_center: Sequence[float], occluder_half_extents: Sequence[float], max_range: float) -> bool:
    if distance(observer, target) > float(max_range):
        return False
    if point_inside_aabb(observer, occluder_center, occluder_half_extents):
        return False
    return not segment_intersects_aabb(observer, target, occluder_center, occluder_half_extents)


def visibility_truth(positions: Mapping[str, Sequence[float]], target: Sequence[float], *, occluder_center: Sequence[float], occluder_half_extents: Sequence[float], max_range: float) -> dict[str, bool]:
    return {
        agent_id: target_visible(
            position,
            target,
            occluder_center=occluder_center,
            occluder_half_extents=occluder_half_extents,
            max_range=max_range,
        )
        for agent_id, position in positions.items()
    }
