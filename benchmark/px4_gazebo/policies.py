"""Reference policies that consume only the public benchmark contract."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contract import Action, HandoffTask, JointAction
from .occlusion_geometry import distance, target_at_step, target_visible


TRACKING_LONGITUDINAL_OFFSET_M = 0.8
TRACKING_LATERAL_OFFSET_M = 1.5
STAGING_WALL_CLEARANCE_M = 1.2
UAV0_TRACKING_LEAD_M = 0.8
UAV0_TRACKING_LATERAL_M = 2.2
UAV0_WALL_CLEARANCE_M = 1.4
UAV0_SUPPORT_WALL_CLEARANCE_M = 1.2
UAV0_SUPPORT_LATERAL_CLEARANCE_M = 1.2
WAYPOINT_TOLERANCE_M = 0.35
WAYPOINT_SETTLE_STEPS = 5
MAX_POLICY_XY_SPEED_M_S = 1.2


def _target_direction(task: HandoffTask) -> float:
    return 1.0 if task.target_end[0] >= task.target_start[0] else -1.0


def _position_action(
    position: Sequence[float], target: Sequence[float], gain: float
) -> Action:
    velocity = [float(gain) * (float(target[i]) - float(position[i])) for i in range(3)]
    horizontal_speed = math.hypot(velocity[0], velocity[1])
    if horizontal_speed > MAX_POLICY_XY_SPEED_M_S:
        scale = MAX_POLICY_XY_SPEED_M_S / horizontal_speed
        velocity[0] *= scale
        velocity[1] *= scale
    return Action(
        vx=velocity[0],
        vy=velocity[1],
        vz=max(-0.6, min(0.6, velocity[2])),
    )


def _last_intent(observation: Mapping[str, Any], agent_id: str) -> str | None:
    message = observation["observations"][agent_id].get("last_message")
    return str(message.get("intent")) if isinstance(message, Mapping) and message.get("intent") else None


def _handoff_message(task: HandoffTask, *, sender: str, receiver: str, intent: str) -> dict[str, Any]:
    return {
        "sender": sender,
        "receiver": receiver,
        "payload": {"intent": intent, "target": task.target_identity},
    }


def _uav0_tracking_position(task: HandoffTask, step_index: int) -> tuple[float, float, float]:
    """Follow the target on the near side, then stop before the wall."""
    direction = _target_direction(task)
    target = target_at_step(task.target_start, task.target_end, step_index, task.horizon)
    desired_x = target[0] + direction * UAV0_TRACKING_LEAD_M
    near_wall_limit = (
        task.occluder_center[0]
        - direction * (task.occluder_half_extents[0] + UAV0_WALL_CLEARANCE_M)
    )
    tracking_x = (
        min(desired_x, near_wall_limit)
        if direction > 0.0
        else max(desired_x, near_wall_limit)
    )
    return (
        tracking_x,
        target[1] + UAV0_TRACKING_LATERAL_M,
        task.staging_position[2],
    )


def _uav0_startup_position(
    task: HandoffTask, position: Sequence[float], step_index: int
) -> tuple[float, float, float]:
    """Acquire the target by joining UAV0's moving observation lane."""
    _ = position
    return _uav0_tracking_position(task, step_index)


def _uav1_tracking_position(task: HandoffTask, step_index: int) -> tuple[float, float, float]:
    """Track from beyond the wall after UAV1 accepts responsibility."""
    target = target_at_step(task.target_start, task.target_end, step_index, task.horizon)
    return (
        target[0] + _target_direction(task) * TRACKING_LONGITUDINAL_OFFSET_M,
        target[1] - TRACKING_LATERAL_OFFSET_M,
        task.staging_position[2],
    )


def _uav1_transfer_position(task: HandoffTask) -> tuple[float, float, float]:
    """Return the first post-handoff point on UAV1's tracking lane."""
    step_index = _handoff_reference_step(task)
    return _uav1_tracking_position(task, step_index)


def _uav1_clearance_position(task: HandoffTask) -> tuple[float, float, float]:
    """A waypoint beyond the wall end where UAV1 must settle before turning."""
    direction = _target_direction(task)
    wall_end_x = task.occluder_center[0] + direction * (
        task.occluder_half_extents[0] + STAGING_WALL_CLEARANCE_M
    )
    reserve_crossing_lane = direction * (task.staging_position[0] - wall_end_x) > 1e-6
    return (
        # Use the staging lane for the entire crossing. A thin transverse
        # wall may require extra staging clearance to keep the moving target
        # away from UAV1 while it crosses from the near to the far side.
        (max if direction > 0.0 else min)(
            task.staging_position[0],
            wall_end_x,
        ),
        # With extra forward clearance the near-side observation lane leads
        # directly to the crossing point, outside the wall end. Avoid an
        # unnecessary lateral detour that lets the moving target catch UAV1.
        task.target_start[1] + UAV0_TRACKING_LATERAL_M
        if reserve_crossing_lane else max(
            task.target_start[1] + UAV0_TRACKING_LATERAL_M,
            task.occluder_center[1]
            + task.occluder_half_extents[1]
            + STAGING_WALL_CLEARANCE_M,
        ),
        task.staging_position[2],
    )


def _advance_uav1_route(
    task: HandoffTask,
    position: Sequence[float],
    phase: str,
    settled_steps: int,
) -> tuple[tuple[float, float, float], str, int]:
    """Advance only after UAV1 has remained inside a waypoint tolerance."""
    if phase == "clearance":
        target = _uav1_clearance_position(task)
    else:
        target = task.staging_position
    settled_steps = (
        settled_steps + 1
        if distance(position, target) <= WAYPOINT_TOLERANCE_M
        else 0
    )
    if settled_steps >= WAYPOINT_SETTLE_STEPS:
        if phase == "clearance":
            return task.staging_position, "staging", 0
        if phase == "staging":
            return task.staging_position, "ready", settled_steps
    return target, phase, settled_steps


def _uav0_support_position(
    task: HandoffTask, step_index: int
) -> tuple[float, float, float]:
    """Move to a visible near-side support point after responsibility transfers."""
    _ = step_index
    direction = _target_direction(task)
    return (
        task.occluder_center[0]
        - direction * (
            task.occluder_half_extents[0] + UAV0_SUPPORT_WALL_CLEARANCE_M
        ),
        max(
            task.target_start[1] + UAV0_TRACKING_LATERAL_M,
            task.occluder_center[1]
            + task.occluder_half_extents[1]
            + UAV0_SUPPORT_LATERAL_CLEARANCE_M,
        ),
        task.staging_position[2],
    )


def _handoff_reference_step(task: HandoffTask) -> int:
    """First deterministic step where UAV0 loses line of sight."""
    for step_index in range(task.horizon):
        target = target_at_step(
            task.target_start, task.target_end, step_index, task.horizon
        )
        if not target_visible(
            _uav0_tracking_position(task, step_index),
            target,
            occluder_center=task.occluder_center,
            occluder_half_extents=task.occluder_half_extents,
            max_range=task.visibility_range,
        ):
            return step_index
    return max(0, task.horizon // 2)


def handoff_demo_waypoints(task: HandoffTask) -> dict[str, tuple[float, float, float]]:
    """Shared source of truth for controller waypoints and demo markers."""
    handoff_step = _handoff_reference_step(task)
    return {
        "uav0_acquisition": _uav0_tracking_position(task, 0),
        "uav0_observation": _uav0_tracking_position(task, handoff_step),
        "uav0_support": _uav0_support_position(task, handoff_step),
        "uav1_clearance": _uav1_clearance_position(task),
        "uav1_staging": task.staging_position,
        "handoff": _uav1_tracking_position(task, handoff_step),
    }


class FixedRoleRule:
    """Keep UAV 0 fixed and pre-position UAV 1 without requesting a handoff."""

    name = "FixedRoleRule"

    def __init__(self, *, message_period_steps: int = 10, position_gain: float = 1.0) -> None:
        self.message_period_steps = max(1, int(message_period_steps))
        self.position_gain = float(position_gain)
        self.task = HandoffTask()
        self.step_index = 0
        self.uav1_route_phase = "clearance"
        self.uav1_settled_steps = 0

    def reset(self, task: HandoffTask) -> None:
        self.task = task
        self.step_index = 0
        self.uav1_route_phase = "clearance"
        self.uav1_settled_steps = 0

    def act(self, observation: Mapping[str, Any]) -> JointAction:
        position = observation["observations"]["uav_1"]["own_state"]["position"]
        route_target, self.uav1_route_phase, self.uav1_settled_steps = (
            _advance_uav1_route(
                self.task,
                position,
                self.uav1_route_phase,
                self.uav1_settled_steps,
            )
        )
        message = None
        if self.step_index % self.message_period_steps == self.message_period_steps - 1:
            message = _handoff_message(self.task, sender="uav_1", receiver="uav_0", intent="pre_position")
        self.step_index += 1
        return JointAction(
            uav_0=Action(),
            uav_1=_position_action(position, route_target, self.position_gain),
            message=message,
        )


class _HandoffPolicyBase:
    position_gain = 1.0
    retry_period_steps = 5

    def __init__(self) -> None:
        self.task = HandoffTask()
        self.step_index = 0
        self.uav0_tracking_ready = False
        self.uav0_tracking_settled_steps = 0
        self.uav1_route_phase = "clearance"
        self.uav1_settled_steps = 0

    def reset(self, task: HandoffTask) -> None:
        self.task = task
        self.step_index = 0
        self.uav0_tracking_ready = False
        self.uav0_tracking_settled_steps = 0
        self.uav1_route_phase = "clearance"
        self.uav1_settled_steps = 0

    def _should_request(self, observation: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def act(self, observation: Mapping[str, Any]) -> JointAction:
        uav_0_position = observation["observations"]["uav_0"]["own_state"]["position"]
        uav_1_position = observation["observations"]["uav_1"]["own_state"]["position"]
        route_target, self.uav1_route_phase, self.uav1_settled_steps = (
            _advance_uav1_route(
                self.task,
                uav_1_position,
                self.uav1_route_phase,
                self.uav1_settled_steps,
            )
        )
        staged = self.uav1_route_phase == "ready"
        uav_1_intent = _last_intent(observation, "uav_1")
        uav_0_intent = _last_intent(observation, "uav_0")
        tracking_position = _uav0_tracking_position(self.task, self.step_index)
        target = target_at_step(
            self.task.target_start,
            self.task.target_end,
            self.step_index,
            self.task.horizon,
        )
        uav0_tracking_now = (
            distance(uav_0_position, tracking_position) <= 0.55
            and target_visible(
                uav_0_position,
                target,
                occluder_center=self.task.occluder_center,
                occluder_half_extents=self.task.occluder_half_extents,
                max_range=self.task.visibility_range,
            )
        )
        self.uav0_tracking_settled_steps = (
            self.uav0_tracking_settled_steps + 1 if uav0_tracking_now else 0
        )
        if self.uav0_tracking_settled_steps >= 3:
            self.uav0_tracking_ready = True
        handoff_active = uav_1_intent == "handoff_request"
        message = None
        if handoff_active and uav_0_intent != "handoff_ack":
            message = _handoff_message(self.task, sender="uav_1", receiver="uav_0", intent="handoff_ack")
        elif staged and self.uav0_tracking_ready and self._should_request(observation) and uav_1_intent != "handoff_request" and self.step_index % self.retry_period_steps == 0:
            message = _handoff_message(self.task, sender="uav_0", receiver="uav_1", intent="handoff_request")
        if handoff_active:
            uav_1_target = _uav1_tracking_position(self.task, self.step_index)
        else:
            uav_1_target = route_target
        uav_0_target = (
            _uav0_support_position(self.task, self.step_index)
            if handoff_active
            else tracking_position
            if self.uav0_tracking_ready
            else _uav0_startup_position(
                self.task, uav_0_position, self.step_index
            )
        )
        self.step_index += 1
        return JointAction(
            uav_0=_position_action(uav_0_position, uav_0_target, self.position_gain),
            uav_1=_position_action(uav_1_position, uav_1_target, self.position_gain),
            message=message,
        )


class OracleHandoffPolicy(_HandoffPolicyBase):
    """Reference upper bound with exact task geometry and deterministic motion."""

    name = "OracleHandoffPolicy"
    retry_period_steps = 1

    def _should_request(self, observation: Mapping[str, Any]) -> bool:
        # Staging is the oracle's readiness condition. Waiting for UAV0 to
        # become occluded would create an avoidable blind interval.
        return True


class HandoffRulePolicy(_HandoffPolicyBase):
    """Simple planner using task timing, staging state, and delivered messages."""

    name = "HandoffRulePolicy"

    def _should_request(self, observation: Mapping[str, Any]) -> bool:
        uav_0_position = observation["observations"]["uav_0"]["own_state"]["position"]
        for offset in range(self.task.handoff_request_lead_steps + 1):
            target = target_at_step(
                self.task.target_start,
                self.task.target_end,
                self.step_index + offset,
                self.task.horizon,
            )
            if not target_visible(
                uav_0_position,
                target,
                occluder_center=self.task.occluder_center,
                occluder_half_extents=self.task.occluder_half_extents,
                max_range=self.task.visibility_range,
            ):
                return True
        return False
