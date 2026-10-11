"""Live PX4/Gazebo dual-UAV safe-route handoff benchmark.

Live Handoff v3 combines real PX4 control and camera streams with a
deterministic geometry-truth target/occluder protocol.  The camera streams are
logged as policy observations, but target visibility is not inferred from
pixels in this version.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import signal
import random
import time
from typing import Any, Mapping
import uuid

from .communication import CommunicationChannel, DeliveredMessage
from .contract import AGENT_IDS, Action, AgentObservation, HandoffTask, JointAction
from .gazebo_scene import (
    ACQUISITION_ZONE_ENTITY_NAME,
    CLEARANCE_ZONE_ENTITY_NAME,
    GazeboSceneController,
    OCCLUDER_ENTITY_NAME,
    OBSERVATION_ZONE_ENTITY_NAME,
    ROUTE_MARKER_ENTITY_NAME,
    ROUTE_MARKER_ENTITY_NAMES,
    STAGING_ZONE_ENTITY_NAME,
    SUPPORT_ZONE_ENTITY_NAME,
    TARGET_ENTITY_NAME,
    TRANSFER_ZONE_ENTITY_NAME,
)
from .handoff_env import _message_dict
from .occlusion_geometry import (
    distance,
    point_aabb_distance,
    segment_intersects_aabb,
    target_at_step,
    visibility_truth,
)
from .policies import (
    FixedRoleRule,
    HandoffRulePolicy,
    OracleHandoffPolicy,
    STAGING_WALL_CLEARANCE_M,
    TRACKING_LATERAL_OFFSET_M,
)
from .model_handoff.dataset import EpisodeDatasetWriter
from .model_handoff.fixtures import (
    HandoffFixture, load_fixture,
    WALL_BLOCKING_CENTER, WALL_BLOCKING_HALF_EXTENTS, WALL_BLOCKING_STAGING,
)
from .model_handoff.images import ros_image_to_jpeg
from .model_handoff.policy_client import PolicyClient, PolicyTimeoutError
from .model_handoff.protocol import (
    ModelJointAction,
    ModelObservation,
    VehicleState,
    WaypointDelta,
    hover_action,
)
from .ros2_px4_adapter import (
    PX4_NAMESPACES,
    PX4_SYSTEM_IDS,
    SafetyLimits,
    acceleration_limited_action,
    body_flu_to_ned,
    common_ned_position,
    distance_3d,
    finite_action,
    geofence_problem,
    ned_displacement_to_initial_body_flu,
)


CAMERA_TOPIC_TEMPLATE = (
    "/world/{world}/model/x500_mono_cam_{index}"
    "/link/camera_link/sensor/camera/image"
)

OBSTACLE_CLEARANCE_M = 0.5
TARGET_CLEARANCE_M = 0.8
MIN_ALTITUDE_MARGIN_M = 1.0


@dataclass
class CameraRuntime:
    topic: str
    frame_count: int = 0
    last_received_s: float = 0.0
    stamp_ns: int = 0
    width: int = 0
    height: int = 0
    latest_message: Any | None = None

    def uri(self) -> str:
        return f"ros2://{self.topic}?seq={self.frame_count}&stamp_ns={self.stamp_ns}"


@dataclass
class VehicleRuntime:
    namespace: str
    system_id: int
    spawn_offset_ned: tuple[float, float, float]
    odometry: Any | None = None
    status: Any | None = None
    last_odometry_s: float = 0.0
    last_status_s: float = 0.0
    origin: tuple[float, float, float] | None = None
    initial_yaw: float = 0.0
    takeoff_target: tuple[float, float, float] | None = None
    action: Action = field(default_factory=Action)
    previous_action: Action = field(default_factory=Action)


def _finite_vector(values: Any, size: int) -> bool:
    try:
        return len(values) == size and all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _yaw_from_wxyz(quaternion: Any) -> float:
    if not _finite_vector(quaternion, 4):
        return 0.0
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def evaluation_positions_in_uav0_flu(
    common_positions_ned: Mapping[str, tuple[float, float, float]],
    uav0_initial_yaw: float,
) -> dict[str, tuple[float, float, float]]:
    """Map common NED truth positions into UAV 0's initial body-FLU frame."""
    return {
        agent_id: ned_displacement_to_initial_body_flu(
            common_positions_ned[agent_id], uav0_initial_yaw
        )
        for agent_id in AGENT_IDS
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PX4/Gazebo live handoff v2: real flight/cameras plus deterministic "
            "geometry-truth handoff acceptance"
        )
    )
    parser.add_argument("--policy", choices=("fixed", "oracle", "rule", "model"), default="rule")
    parser.add_argument("--bandwidth", type=int, default=256)
    parser.add_argument("--latency-ms", type=float, default=100.0)
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world", default="lawn")
    parser.add_argument(
        "--visual-handoff",
        action="store_true",
        help=(
            "show demo-only acquisition/clearance/staging/handoff/support "
            "waypoints and the actual route segments in Gazebo"
        ),
    )
    parser.add_argument("--formal-model", action="store_true", help="enable the truth-isolated model benchmark contract")
    parser.add_argument("--hide-markers", action="store_true", help="hide staging/transfer/route aids without enabling model policy")
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--fixture-id")
    parser.add_argument("--policy-endpoint", default="127.0.0.1:8765")
    parser.add_argument("--model-method", choices=("joint_bc", "dual_vln_hier", "dual_wam_mpc"))
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--policy-timeout-ms", type=float, default=400.0)
    parser.add_argument("--max-policy-timeouts", type=int, default=3)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--oracle-perturb-probability", type=float, default=0.15)
    parser.add_argument("--countdown", type=float, default=5.0)
    parser.add_argument("--prestream", type=float, default=2.0)
    parser.add_argument("--telemetry-timeout", type=float, default=2.0)
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--takeoff-timeout", type=float, default=30.0)
    parser.add_argument("--landing-timeout", type=float, default=45.0)
    parser.add_argument(
        "--uav1-spawn-offset-ned", type=float, nargs=3, required=True,
        metavar=("N", "E", "D"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark/px4_gazebo/logs/live_handoff.jsonl"),
    )
    return parser


def validate_runtime_arguments(
    args: argparse.Namespace, limits: SafetyLimits = SafetyLimits()
) -> tuple[float, float, float]:
    numeric = {
        "latency-ms": args.latency_ms,
        "drop-probability": args.drop_probability,
        "altitude": args.altitude,
        "duration": args.duration,
        "countdown": args.countdown,
        "prestream": args.prestream,
        "telemetry-timeout": args.telemetry_timeout,
        "camera-timeout": args.camera_timeout,
        "startup-timeout": args.startup_timeout,
        "takeoff-timeout": args.takeoff_timeout,
        "landing-timeout": args.landing_timeout,
        "policy-timeout-ms": args.policy_timeout_ms,
        "oracle-perturb-probability": args.oracle_perturb_probability,
    }
    bad = [name for name, value in numeric.items() if not math.isfinite(float(value))]
    try:
        spawn = tuple(float(value) for value in args.uav1_spawn_offset_ned)
    except (TypeError, ValueError) as exc:
        raise ValueError("uav1-spawn-offset-ned must contain 3 finite numbers") from exc
    if len(spawn) != 3 or not all(math.isfinite(value) for value in spawn):
        bad.append("uav1-spawn-offset-ned")
    if bad:
        raise ValueError("non-finite arguments: " + ", ".join(bad))
    if args.bandwidth < 0:
        raise ValueError("bandwidth must be >= 0")
    if args.latency_ms < 0.0:
        raise ValueError("latency-ms must be >= 0")
    if not 0.0 <= args.drop_probability <= 1.0:
        raise ValueError("drop-probability must be between 0 and 1")
    for name in (
        "altitude", "duration", "telemetry_timeout", "camera_timeout",
        "startup_timeout", "takeoff_timeout", "landing_timeout",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be > 0")
    if args.countdown < 0.0 or args.prestream < 0.0:
        raise ValueError("countdown and prestream must be >= 0")
    if args.seed < 0:
        raise ValueError("seed must be >= 0")
    if args.max_policy_timeouts < 1:
        raise ValueError("max-policy-timeouts must be >= 1")
    if args.policy_timeout_ms <= 0.0:
        raise ValueError("policy-timeout-ms must be > 0")
    if not 0.0 <= args.oracle_perturb_probability <= 1.0:
        raise ValueError("oracle-perturb-probability must be between 0 and 1")
    if (args.fixture_manifest is None) != (args.fixture_id is None):
        raise ValueError("fixture-manifest and fixture-id must be provided together")
    if args.formal_model and (args.policy != "model" or args.fixture_manifest is None):
        raise ValueError("formal-model requires --policy model and a fixture")
    if args.formal_model and (args.model_method is None or args.training_seed is None):
        raise ValueError("formal-model requires model-method and training-seed")
    if args.formal_model and args.visual_handoff:
        raise ValueError("formal-model forbids visual-handoff markers")
    if args.altitude > limits.max_altitude:
        raise ValueError(f"altitude exceeds {limits.max_altitude:.1f} m safety limit")
    if math.hypot(spawn[0], spawn[1]) < limits.min_inter_vehicle_distance:
        raise ValueError(
            "declared horizontal spawn separation is below "
            f"{limits.min_inter_vehicle_distance:.1f} m"
        )
    return spawn


def build_task(args: argparse.Namespace, fixture: HandoffFixture | None = None) -> HandoffTask:
    altitude = float(args.altitude)
    horizon = max(1, int(round(float(args.duration) * 10.0)))
    task = replace(
        HandoffTask(),
        horizon=horizon,
        control_hz=10.0,
        target_start=(-2.0, -1.6, altitude),
        target_end=(8.0, -1.6, altitude),
        # A transverse wall crosses UAV0's forward observation lane (y=0.6).
        # Its lower end stays 0.5 m from the target lane (y=-1.6), so the
        # target passes beside the wall while UAV0 must stop and loses sight.
        # The existing geometry-derived clearance/staging points keep UAV1
        # outside the same physical wall used by the visibility evaluator.
        occluder_center=(*WALL_BLOCKING_CENTER[:2], altitude),
        occluder_half_extents=WALL_BLOCKING_HALF_EXTENTS,
        initial_positions={
            "uav_0": (0.0, 0.0, altitude),
            "uav_1": (0.0, 0.0, altitude),
        },
        staging_position=(3.5, -3.1, altitude),
        post_handoff_tracking_steps=min(30, max(3, horizon // 6)),
    )
    if fixture is not None:
        task = replace(
            task,
            split=fixture.split,
            instruction=fixture.instruction,
            target_identity=fixture.fixture_id,
            target_start=(fixture.target_start[0], fixture.target_start[1], altitude),
            target_end=(fixture.target_end[0], fixture.target_end[1], altitude),
            occluder_center=(fixture.occluder.center[0], fixture.occluder.center[1], altitude),
            occluder_half_extents=fixture.occluder.half_extents,
        )
    staging_x = task.occluder_center[0] + task.occluder_half_extents[0] + STAGING_WALL_CLEARANCE_M
    if fixture is None:
        # Keep the crossing at least as far forward as the original safe
        # staging lane. Shrinking wall thickness must not pull UAV1 across
        # the moving target when Gazebo follows commands more slowly.
        staging_x = max(WALL_BLOCKING_STAGING[0], staging_x)
    elif fixture.staging_position is not None:
        staging_x = fixture.staging_position[0]
    task = replace(
        task,
        staging_position=(
            staging_x,
            fixture.staging_position[1]
            if fixture is not None and fixture.staging_position is not None
            else task.target_start[1] - TRACKING_LATERAL_OFFSET_M,
            altitude,
        ),
    )
    return task


def build_policy(name: str, task: HandoffTask) -> Any:
    if name == "model":
        raise ValueError("model policy is provided by PolicyClient")
    policy = {
        "fixed": FixedRoleRule,
        "oracle": OracleHandoffPolicy,
        "rule": HandoffRulePolicy,
    }[name]()
    policy.reset(task)
    return policy


def _visual_position_action(
    position: tuple[float, float, float] | list[float],
    target: tuple[float, float, float],
    gain: float = 1.0,
) -> Action:
    """Bounded FLU velocity command used only by the visual demonstration."""
    return Action(
        vx=gain * (float(target[0]) - float(position[0])),
        vy=gain * (float(target[1]) - float(position[1])),
        vz=gain * (float(target[2]) - float(position[2])),
    ).clipped()


def _event(
    t: int,
    event_type: str,
    agent_ids: list[str],
    *,
    success: bool,
    start_t: int | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_type": event_type,
        "agent_ids": agent_ids,
        "start_t": t if start_t is None else int(start_t),
        "end_t": t,
        "success": bool(success),
    }
    if attributes:
        event["attributes"] = dict(attributes)
    return event


class LiveHandoffProtocol:
    """Pure deterministic communication and handoff acceptance protocol."""

    def __init__(
        self,
        task: HandoffTask,
        *,
        seed: int,
        bandwidth: int,
        latency_ms: float,
        drop_probability: float,
    ) -> None:
        self.task = task
        self.seed = int(seed)
        self.communication = CommunicationChannel(
            bandwidth_bytes_per_second=bandwidth,
            latency_ms=latency_ms,
            drop_probability=drop_probability,
            control_hz=task.control_hz,
            seed=seed,
        )
        self.t = 0
        self.last_messages: dict[str, dict[str, Any] | None] = {
            agent_id: None for agent_id in AGENT_IDS
        }
        self.identity_streak = {agent_id: 0 for agent_id in AGENT_IDS}
        self.visible_history: list[set[str]] = []
        self.steps: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.pre_position_t: int | None = None
        self.occlusion_start_t: int | None = None
        self.handoff_request_t: int | None = None
        self.handoff_delivery_t: int | None = None
        self.recovery_t: int | None = None
        self.handoff_t: int | None = None
        self.verify_t: int | None = None
        self.confirm_t: int | None = None
        self.require_explicit_confirm = False
        self.identity_errors = 0
        self.inference_latencies_ms: list[float] = []
        self.policy_timeout_count = 0
        self.minimum_separation = float("inf")
        self.maximum_separation = 0.0
        self.separation_sum = 0.0
        self.separation_samples = 0
        self.path_lengths = {agent_id: 0.0 for agent_id in AGENT_IDS}
        self._previous_positions: dict[str, tuple[float, float, float]] | None = None
        self.done = False
        self.termination_reason = "running"
        self.failure_reason: str | None = None

    def policy_observation(
        self, observations: Mapping[str, AgentObservation]
    ) -> dict[str, Any]:
        return {
            "task": {
                "task_id": self.task.task_id,
                "instruction": self.task.instruction,
                "target_identity": self.task.target_identity,
            },
            "timestamp_ms": int(round(self.t * 1000.0 / self.task.control_hz)),
            "observations": {
                agent_id: replace(
                    observations[agent_id], last_message=self.last_messages[agent_id]
                ).to_policy_dict()
                for agent_id in AGENT_IDS
            },
            "communication_budget": self.communication.bandwidth,
        }

    def abort(self, reason: str, termination: str = "safety_abort") -> None:
        if not self.done:
            self.done = True
            self.failure_reason = reason
            self.termination_reason = termination

    def _longest_interruption(self) -> int:
        longest = 0
        current = 0
        for visible in self.visible_history:
            if visible:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    def step(
        self,
        observations: Mapping[str, AgentObservation],
        evaluation_positions: Mapping[str, tuple[float, float, float]],
        joint_action: JointAction,
        policy_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("episode already terminated")
        joint_action = joint_action.clipped()
        step_events: list[dict[str, Any]] = []
        sent_messages: list[DeliveredMessage] = []
        delivered_messages: list[DeliveredMessage] = []
        confirm_requested = False

        previous_positions = self._previous_positions
        if previous_positions is not None:
            for agent_id in AGENT_IDS:
                self.path_lengths[agent_id] += distance(
                    previous_positions[agent_id], evaluation_positions[agent_id]
                )
        self._previous_positions = dict(evaluation_positions)
        separation = distance(evaluation_positions["uav_0"], evaluation_positions["uav_1"])
        self.minimum_separation = min(self.minimum_separation, separation)
        self.maximum_separation = max(self.maximum_separation, separation)
        self.separation_sum += separation
        self.separation_samples += 1

        if joint_action.message is not None:
            envelope = joint_action.message
            sender = str(envelope.get("sender", "uav_0"))
            receiver = str(envelope.get("receiver", "uav_1"))
            payload = envelope.get("payload", envelope)
            if sender not in AGENT_IDS or receiver not in {*AGENT_IDS, "broadcast"}:
                raise ValueError("invalid communication envelope")
            sent = self.communication.send(
                sender=sender, receiver=receiver, payload=payload, t=self.t
            )
            sent_messages.append(sent)
            intent = payload.get("intent") if isinstance(payload, Mapping) else None
            if intent == "handoff_request":
                if self.handoff_request_t is None:
                    self.handoff_request_t = self.t
                step_events.append(_event(
                    self.t, "handoff_request", [sender, receiver],
                    success=not sent.dropped,
                    attributes={"dropped": sent.dropped, "bytes": sent.bytes},
                ))
            elif intent == "handoff_ack":
                step_events.append(_event(
                    self.t, "handoff_ack_sent", [sender, receiver],
                    success=not sent.dropped,
                    attributes={"dropped": sent.dropped, "bytes": sent.bytes},
                ))
            elif intent == "confirm":
                confirm_requested = not sent.dropped

        for agent_id in AGENT_IDS:
            for delivered in self.communication.receive(receiver=agent_id, t=self.t):
                delivered_messages.append(delivered)
                payload = delivered.payload
                self.last_messages[agent_id] = (
                    dict(payload) if isinstance(payload, Mapping) else {"payload": payload}
                )
                intent = payload.get("intent") if isinstance(payload, Mapping) else None
                if intent == "handoff_request" and agent_id == "uav_1":
                    if self.handoff_delivery_t is None:
                        self.handoff_delivery_t = self.t
                    step_events.append(_event(
                        self.t, "handoff_delivery", [delivered.sender, agent_id],
                        success=True, start_t=delivered.sent_t,
                        attributes={"latency_steps": self.t - delivered.sent_t},
                    ))
                elif intent == "handoff_ack" and agent_id == "uav_0":
                    step_events.append(_event(
                        self.t, "handoff_ack_delivery", [delivered.sender, agent_id],
                        success=True, start_t=delivered.sent_t,
                        attributes={"latency_steps": self.t - delivered.sent_t},
                    ))

        target = target_at_step(
            self.task.target_start, self.task.target_end, self.t, self.task.horizon
        )
        staged = distance(
            evaluation_positions["uav_1"], self.task.staging_position
        ) <= self.task.staging_tolerance
        if staged and self.pre_position_t is None:
            self.pre_position_t = self.t
            step_events.append(_event(self.t, "pre_position", ["uav_1"], success=True))

        visibility = visibility_truth(
            evaluation_positions,
            target,
            occluder_center=self.task.occluder_center,
            occluder_half_extents=self.task.occluder_half_extents,
            max_range=self.task.visibility_range,
        )
        visible = {agent_id for agent_id, value in visibility.items() if value}
        if confirm_requested:
            if visibility["uav_1"] and self.confirm_t is None:
                self.confirm_t = self.t
                step_events.append(_event(
                    self.t, "identity_confirm", ["uav_1"], success=True,
                ))
            elif not visibility["uav_1"]:
                self.identity_errors += 1
                step_events.append(_event(
                    self.t, "identity_error", ["uav_1"], success=False,
                ))
        self.visible_history.append(visible)
        for agent_id in AGENT_IDS:
            self.identity_streak[agent_id] = (
                self.identity_streak[agent_id] + 1 if visibility[agent_id] else 0
            )
        if not visibility["uav_0"] and visibility["uav_1"] and self.occlusion_start_t is None:
            self.occlusion_start_t = self.t
            step_events.append(_event(
                self.t, "occlusion_start", ["uav_0"], success=True,
                attributes={"target_identity": self.task.target_identity},
            ))
        if (
            self.occlusion_start_t is not None
            and self.recovery_t is None
            and self.identity_streak["uav_1"] >= self.task.required_identity_frames
        ):
            self.recovery_t = self.t
            step_events.append(_event(
                self.t, "target_recovery", ["uav_1"], success=True,
                start_t=self.occlusion_start_t,
                attributes={
                    "recovery_latency_steps": self.t - self.occlusion_start_t,
                    "target_identity": self.task.target_identity,
                },
            ))
        if self.handoff_t is None and self.handoff_delivery_t is not None and self.recovery_t is not None:
            self.handoff_t = self.t
            start_t = self.occlusion_start_t if self.occlusion_start_t is not None else self.handoff_delivery_t
            step_events.append(_event(
                self.t, "handoff", list(AGENT_IDS), success=True, start_t=start_t,
                attributes={
                    "latency_steps": self.t - start_t,
                    "target_identity": self.task.target_identity,
                },
            ))
        passed = target[0] > self.task.occluder_center[0] + self.task.occluder_half_extents[0]
        if (
            self.verify_t is None and self.handoff_t is not None and passed
            and self.identity_streak["uav_1"] >= self.task.required_identity_frames
            and (not self.require_explicit_confirm or self.confirm_t is not None)
        ):
            self.verify_t = self.t
            step_events.append(_event(
                self.t, "verify", ["uav_1"], success=True, start_t=self.handoff_t,
                attributes={"target_identity": self.task.target_identity},
            ))

        obstacle_clearance = {
            agent_id: point_aabb_distance(
                evaluation_positions[agent_id],
                self.task.occluder_center,
                self.task.occluder_half_extents,
            )
            for agent_id in AGENT_IDS
        }
        obstacle_collisions = [
            agent_id
            for agent_id, clearance in obstacle_clearance.items()
            if clearance < OBSTACLE_CLEARANCE_M
        ]
        inflated_half_extents = tuple(
            float(value) + OBSTACLE_CLEARANCE_M
            for value in self.task.occluder_half_extents
        )
        swept_obstacle_collisions = (
            [
                agent_id
                for agent_id in AGENT_IDS
                if segment_intersects_aabb(
                    previous_positions[agent_id],
                    evaluation_positions[agent_id],
                    self.task.occluder_center,
                    inflated_half_extents,
                )
            ]
            if previous_positions is not None
            else []
        )
        obstacle_collisions = sorted(
            set(obstacle_collisions) | set(swept_obstacle_collisions)
        )
        target_clearance = {
            agent_id: distance(evaluation_positions[agent_id], target)
            for agent_id in AGENT_IDS
        }
        target_collisions = [
            agent_id
            for agent_id, clearance in target_clearance.items()
            if clearance < TARGET_CLEARANCE_M
        ]
        minimum_operational_altitude = max(
            0.0,
            min(
                self.task.target_start[2],
                self.task.target_end[2],
                self.task.staging_position[2],
            ) - MIN_ALTITUDE_MARGIN_M,
        )
        ground_collisions = [
            agent_id
            for agent_id in AGENT_IDS
            if evaluation_positions[agent_id][2] < minimum_operational_altitude
        ]
        collision = (
            separation < self.task.collision_distance
            or bool(obstacle_collisions)
            or bool(target_collisions)
            or bool(ground_collisions)
        )
        if collision:
            step_events.append(_event(
                self.t, "collision", list(AGENT_IDS), success=False,
                attributes={
                    "separation_m": separation,
                    "obstacle": (
                        "occluder" if obstacle_collisions
                        else "target" if target_collisions
                        else "ground" if ground_collisions
                        else None
                    ),
                    "obstacle_collision_agents": obstacle_collisions,
                    "swept_obstacle_collision_agents": swept_obstacle_collisions,
                    "obstacle_clearance_m": obstacle_clearance,
                    "minimum_obstacle_clearance_m": OBSTACLE_CLEARANCE_M,
                    "target_collision_agents": target_collisions,
                    "target_clearance_m": target_clearance,
                    "minimum_target_clearance_m": TARGET_CLEARANCE_M,
                    "ground_collision_agents": ground_collisions,
                    "minimum_operational_altitude_m": minimum_operational_altitude,
                },
            ))
        self.events.extend(step_events)
        interruption = self._longest_interruption()
        tracking_dwell_complete = (
            self.verify_t is not None
            and self.t - self.verify_t >= self.task.post_handoff_tracking_steps
        )
        success = tracking_dwell_complete and interruption <= self.task.interruption_limit_steps and not collision
        next_t = self.t + 1
        timed_out = next_t >= self.task.horizon and not success and not collision
        if timed_out:
            timeout_event = _event(self.t, "timeout", list(AGENT_IDS), success=False)
            step_events.append(timeout_event)
            self.events.append(timeout_event)
        truth = {
            "agent_positions": {
                agent_id: list(evaluation_positions[agent_id])
                for agent_id in AGENT_IDS
            },
            "target_pose": list(target),
            "target_visible_to": sorted(visible),
            "target_visibility": visibility,
            "target_identity": self.task.target_identity,
            "occluded_for_uav_0": not visibility["uav_0"],
            "collision": collision,
            "minimum_separation_m": float(self.minimum_separation),
        }
        policy_view = self.policy_observation(observations)
        sent = [_message_dict(message, self.task.control_hz) for message in sent_messages]
        delivered = [_message_dict(message, self.task.control_hz) for message in delivered_messages]
        step_record = {
            "t": self.t,
            "timestamp_ms": int(round(self.t * 1000.0 / self.task.control_hz)),
            "observations": {
                agent_id: {
                    "rgb_uri": policy_view["observations"][agent_id]["rgb_uri"],
                    "own_state": (
                        policy_view["observations"][agent_id]["own_state"]["position"]
                        + policy_view["observations"][agent_id]["own_state"]["velocity"]
                        + [policy_view["observations"][agent_id]["own_state"]["yaw"]]
                    ),
                    "health": policy_view["observations"][agent_id]["own_state"]["health"],
                    "last_message": policy_view["observations"][agent_id]["last_message"],
                }
                for agent_id in AGENT_IDS
            },
            "actions": {
                "uav_0": {"control": list(joint_action.uav_0.as_vector()), "valid": True},
                "uav_1": {"control": list(joint_action.uav_1.as_vector()), "valid": True},
            },
            "communication": {
                "messages": sent,
                "sent": sent,
                "delivered": delivered,
                "bytes_sent": int(self.communication.bytes_sent),
                "dropped": int(self.communication.dropped),
            },
            "events": step_events,
            "truth": truth,
        }
        if policy_metadata is not None:
            step_record["policy"] = dict(policy_metadata)
            latency = policy_metadata.get("inference_latency_ms")
            if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
                self.inference_latencies_ms.append(float(latency))
            if bool(policy_metadata.get("timed_out", False)):
                self.policy_timeout_count += 1
        self.steps.append(step_record)
        self.t = next_t
        if collision:
            self.done = True
            self.termination_reason = "collision"
        elif success:
            self.done = True
            self.termination_reason = "success"
        elif timed_out:
            self.done = True
            self.termination_reason = "timeout"
        return {
            "success": success,
            "collision": collision,
            "done": self.done,
            "termination_reason": self.termination_reason,
            "events": step_events,
            "truth": truth,
        }

    def metrics(
        self,
        *,
        flight_time_s: float,
        camera_frames: Mapping[str, int],
    ) -> dict[str, float]:
        recovery_latency = self.task.horizon
        if self.occlusion_start_t is not None and self.recovery_t is not None:
            recovery_latency = self.recovery_t - self.occlusion_start_t
        handoff_latency = self.task.horizon
        if self.occlusion_start_t is not None and self.handoff_t is not None:
            handoff_latency = self.handoff_t - self.occlusion_start_t
        collision = any(event["event_type"] == "collision" for event in self.events)
        sorted_latencies = sorted(self.inference_latencies_ms)
        latency_p95 = 0.0
        if sorted_latencies:
            latency_p95 = sorted_latencies[min(len(sorted_latencies) - 1, math.ceil(0.95 * len(sorted_latencies)) - 1)]
        minimum_separation = self.minimum_separation
        if not math.isfinite(minimum_separation):
            minimum_separation = 0.0
        handoff_step = self.handoff_t
        handoff_visible = False
        if handoff_step is not None:
            for step in self.steps:
                if int(step.get("t", -1)) == handoff_step:
                    truth = step.get("truth", {})
                    handoff_visible = "uav_1" in set(
                        truth.get("target_visible_to", [])
                    )
                    break
        successful_events = {
            event_type: [
                event for event in self.events
                if event.get("event_type") == event_type and event.get("success")
            ]
            for event_type in (
                "handoff_request", "handoff_delivery", "verify"
            )
        }
        return {
            "task_success": float(
                self.verify_t is not None
                and not collision
                and self._longest_interruption() <= self.task.interruption_limit_steps
            ),
            "pre_position_success": float(self.pre_position_t is not None),
            "handoff_success": float(self.handoff_t is not None),
            "confirmation_success": float(self.verify_t is not None),
            "role_switch_success": float(
                self.handoff_t is not None and self.verify_t is not None
            ),
            "target_visible_during_handoff": float(handoff_visible),
            "handoff_request_step": float(
                int(successful_events["handoff_request"][0].get("end_t", self.task.horizon))
                if successful_events["handoff_request"] else self.task.horizon
            ),
            "handoff_delivery_step": float(
                int(successful_events["handoff_delivery"][0].get("end_t", self.task.horizon))
                if successful_events["handoff_delivery"] else self.task.horizon
            ),
            "tracker_switch_step": float(
                self.handoff_t if self.handoff_t is not None else self.task.horizon
            ),
            "verify_step": float(
                int(successful_events["verify"][0].get("end_t", self.task.horizon))
                if successful_events["verify"] else self.task.horizon
            ),
            "recovery_latency_steps": float(recovery_latency),
            "handoff_latency_steps": float(handoff_latency),
            "longest_interruption_steps": float(self._longest_interruption()),
            "communication_bytes": float(self.communication.bytes_sent),
            "communication_dropped": float(self.communication.dropped),
            "collision_rate": float(collision),
            "minimum_separation_m": float(minimum_separation),
            "mean_separation_m": float(
                self.separation_sum / max(1, self.separation_samples)
            ),
            "maximum_separation_m": float(self.maximum_separation),
            "path_length_uav_0": float(self.path_lengths["uav_0"]),
            "path_length_uav_1": float(self.path_lengths["uav_1"]),
            "path_length": float(sum(self.path_lengths.values())),
            "flight_time_s": float(max(0.0, flight_time_s)),
            "identity_error_rate": float(self.identity_errors / max(1, self.identity_errors + int(self.verify_t is not None))),
            "inference_latency_ms_mean": float(sum(sorted_latencies) / max(1, len(sorted_latencies))),
            "inference_latency_ms_p95": float(latency_p95),
            "policy_timeout_count": float(self.policy_timeout_count),
            "camera_frames_uav_0": float(camera_frames.get("uav_0", 0)),
            "camera_frames_uav_1": float(camera_frames.get("uav_1", 0)),
        }


def validate_episode_record(
    record: dict[str, Any], schema_path: Path | None = None
) -> None:
    """Validate one Live Handoff record without importing ROS 2."""
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate live episode logs") from exc
    resolved_schema = schema_path or (
        Path(__file__).resolve().parents[1] / "spec" / "episode_schema_v0_2.json"
    )
    schema = json.loads(resolved_schema.read_text(encoding="utf-8"))
    jsonschema.validate(instance=record, schema=schema)


def append_episode_record(record: dict[str, Any], output: Path) -> None:
    """Validate and append exactly one schema 0.2 JSONL episode."""
    validate_episode_record(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_episode_record(
    *,
    task: HandoffTask,
    protocol: LiveHandoffProtocol,
    seed: int,
    world: str,
    policy_name: str,
    spawn_offset_ned: tuple[float, float, float],
    cameras: Mapping[str, CameraRuntime],
    flight_time_s: float,
    episode_id: str | None = None,
) -> dict[str, Any]:
    reset_material = {
        "task": task.to_dict(),
        "seed": int(seed),
        "world": world,
        "spawn_offset_ned": list(spawn_offset_ned),
        "visual_handoff": bool(getattr(protocol, "visual_handoff", False)),
    }
    reset_hash = hashlib.sha256(
        json.dumps(reset_material, sort_keys=True).encode("utf-8")
    ).hexdigest()
    task_spec = task.to_dict()
    if protocol.steps:
        first_positions = protocol.steps[0].get("truth", {}).get("agent_positions")
        if isinstance(first_positions, Mapping):
            task_spec["initial_positions"] = {
                agent_id: list(first_positions[agent_id])
                for agent_id in AGENT_IDS
                if agent_id in first_positions
            }
    task_spec.update({
        "policy": policy_name,
        "visual_handoff": bool(getattr(protocol, "visual_handoff", False)),
        "acceptance_source": "deterministic_geometry_truth",
        "camera_role": "real_observation_stream_with_synchronized_visual_scene",
        "visual_scene": True,
        "physical_entities": {
            "occluder": OCCLUDER_ENTITY_NAME,
            "target": TARGET_ENTITY_NAME,
            "acquisition_zone": ACQUISITION_ZONE_ENTITY_NAME,
            "observation_zone": OBSERVATION_ZONE_ENTITY_NAME,
            "clearance_zone": CLEARANCE_ZONE_ENTITY_NAME,
            "staging_zone": STAGING_ZONE_ENTITY_NAME,
            "handoff_zone": TRANSFER_ZONE_ENTITY_NAME,
            "support_zone": SUPPORT_ZONE_ENTITY_NAME,
            "route_marker": ROUTE_MARKER_ENTITY_NAME,
            "route_markers": list(ROUTE_MARKER_ENTITY_NAMES),
        },
        "camera_topics": {
            agent_id: cameras[agent_id].topic for agent_id in AGENT_IDS
        },
        "camera_frames": {
            agent_id: int(cameras[agent_id].frame_count) for agent_id in AGENT_IDS
        },
        "visibility_acceptance": "deterministic_geometry_truth",
    })
    fixture_spec = getattr(protocol, "fixture_spec", None)
    if isinstance(fixture_spec, Mapping):
        task_spec["fixture_id"] = fixture_spec.get("fixture_id")
        task_spec["fixture_split"] = fixture_spec.get("split")
        task_spec["target_appearance"] = fixture_spec.get("target")
        task_spec["distractor_appearances"] = fixture_spec.get("distractors")
    model_metadata = getattr(protocol, "model_metadata", None)
    if isinstance(model_metadata, Mapping):
        task_spec["model"] = dict(model_metadata)
    serialized_flight_time_s = 0.0
    if len(protocol.steps) >= 2:
        serialized_flight_time_s = max(
            0.0,
            (
                float(protocol.steps[-1]["timestamp_ms"])
                - float(protocol.steps[0]["timestamp_ms"])
            ) / 1000.0,
        )
    _ = flight_time_s  # Kept for call-site compatibility; replay uses serialized time.
    metrics = protocol.metrics(
        flight_time_s=serialized_flight_time_s,
        camera_frames={
            agent_id: cameras[agent_id].frame_count for agent_id in AGENT_IDS
        },
    )
    success = protocol.termination_reason == "success" and metrics["task_success"] == 1.0
    termination = protocol.termination_reason
    if termination not in {
        "success", "timeout", "collision", "safety_abort", "simulator_error"
    }:
        termination = "simulator_error"
    return {
        "schema_version": "0.2",
        "episode_id": episode_id or f"live_{task.task_id}__seed_{seed}",
        "task_id": task.task_id,
        "split": task.split,
        "seed": int(seed),
        "agent_ids": list(AGENT_IDS),
        "task_spec": task_spec,
        "environment": {
            "simulator": "px4_gazebo",
            "version": "live-px4-sitl-gazebo-sim-handoff-v3",
            "map_id": world,
            "reset_hash": reset_hash,
            "ros_distro": "humble",
            "vehicle_namespaces": list(PX4_NAMESPACES),
            "transport": "px4_uxrce_dds+deterministic_channel",
            "control_hz": float(task.control_hz),
            "setpoint_hz": 20.0,
        },
        "steps": protocol.steps,
        "outcome": {
            "terminated": bool(protocol.done),
            "termination_reason": termination,
            "success": bool(success),
            "metrics": metrics,
        },
    }


def main() -> None:
    args = _build_parser().parse_args()
    limits = SafetyLimits()
    try:
        spawn = validate_runtime_arguments(args, limits)
    except ValueError as exc:
        raise SystemExit(f"参数错误：{exc}") from exc
    fixture: HandoffFixture | None = None
    if args.fixture_manifest is not None:
        try:
            fixture = load_fixture(args.fixture_manifest, args.fixture_id)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise SystemExit(f"fixture 加载失败：{exc}") from exc

    try:
        import rclpy
        from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        from px4_msgs.msg import VehicleOdometry, VehicleStatus
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise SystemExit(
            "缺少 ROS 2 Humble、px4_msgs 或 sensor_msgs；请 source ROS 和 px4_ros2_ws"
        ) from exc

    px4_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    camera_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    camera_topics = {
        agent_id: CAMERA_TOPIC_TEMPLATE.format(world=args.world, index=index)
        for index, agent_id in enumerate(AGENT_IDS)
    }

    class LiveHandoffNode(Node):
        def __init__(self) -> None:
            super().__init__("aerowam_live_handoff")
            self.vehicles = {
                "uav_0": VehicleRuntime(
                    PX4_NAMESPACES["uav_0"], PX4_SYSTEM_IDS["uav_0"], (0.0, 0.0, 0.0)
                ),
                "uav_1": VehicleRuntime(
                    PX4_NAMESPACES["uav_1"], PX4_SYSTEM_IDS["uav_1"], spawn
                ),
            }
            self.cameras = {
                agent_id: CameraRuntime(camera_topics[agent_id]) for agent_id in AGENT_IDS
            }
            self.publishers_by_agent: dict[str, dict[str, Any]] = {}
            self._owned_subscription_handles: list[Any] = []
            for agent_id, vehicle in self.vehicles.items():
                input_prefix = f"/{vehicle.namespace}/fmu/in"
                output_prefix = f"/{vehicle.namespace}/fmu/out"
                self.publishers_by_agent[agent_id] = {
                    "mode": self.create_publisher(
                        OffboardControlMode, f"{input_prefix}/offboard_control_mode", px4_qos
                    ),
                    "setpoint": self.create_publisher(
                        TrajectorySetpoint, f"{input_prefix}/trajectory_setpoint", px4_qos
                    ),
                    "command": self.create_publisher(
                        VehicleCommand, f"{input_prefix}/vehicle_command", px4_qos
                    ),
                }
                self._owned_subscription_handles.extend([
                    self.create_subscription(
                        VehicleOdometry,
                        f"{output_prefix}/vehicle_odometry",
                        lambda msg, key=agent_id: self._on_odometry(key, msg),
                        px4_qos,
                    ),
                    self.create_subscription(
                        VehicleStatus,
                        f"{output_prefix}/vehicle_status_v4",
                        lambda msg, key=agent_id: self._on_status(key, msg),
                        px4_qos,
                    ),
                    self.create_subscription(
                        Image,
                        self.cameras[agent_id].topic,
                        lambda msg, key=agent_id: self._on_camera(key, msg),
                        camera_qos,
                    ),
                ])

            self.task = build_task(args, fixture)
            self.policy = None if args.policy == "model" else build_policy(args.policy, self.task)
            self.policy_client = (
                PolicyClient(args.policy_endpoint, timeout_s=args.policy_timeout_ms / 1000.0)
                if args.policy == "model" else None
            )
            self.protocol = LiveHandoffProtocol(
                self.task,
                seed=args.seed,
                bandwidth=args.bandwidth,
                latency_ms=args.latency_ms,
                drop_probability=args.drop_probability,
            )
            self.protocol.visual_handoff = bool(args.visual_handoff)
            self.protocol.require_explicit_confirm = bool(args.formal_model)
            self.protocol.fixture_spec = fixture.to_dict() if fixture is not None else None
            self.protocol.model_metadata = (
                {
                    "method": args.model_method,
                    "training_seed": args.training_seed,
                    "checkpoint_id": args.checkpoint_id,
                    "policy_hz": 2.0,
                    "timeout_ms": args.policy_timeout_ms,
                }
                if args.policy == "model" else None
            )
            self.visual_handoff = bool(args.visual_handoff)
            self.strict_scene = bool(args.formal_model or fixture is not None)
            self.scene = GazeboSceneController(args.world)
            self.limits = limits
            self.armed_state = int(getattr(VehicleStatus, "ARMING_STATE_ARMED", 2))
            self.offboard_state = int(getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14))
            self.episode_id = f"live_{self.task.task_id}_{uuid.uuid4().hex[:12]}"
            self.dataset_writer = None
            if args.record_dir is not None:
                if fixture is None:
                    raise ValueError("record-dir requires a frozen fixture")
                self.dataset_writer = EpisodeDatasetWriter(
                    args.record_dir,
                    episode_id=self.episode_id,
                    fixture=fixture.to_dict(),
                    mode=args.policy,
                )
            self._model_action = ModelJointAction()
            self._model_observation: ModelObservation | None = None
            self._model_decision_stride = max(1, int(round(self.task.control_hz / 2.0)))
            self._consecutive_policy_timeouts = 0
            self._action_history: deque[dict[str, list[float]]] = deque(maxlen=4)
            self._message_history: deque[dict[str, Any]] = deque(maxlen=16)
            self._perturb_rng = random.Random(args.seed)
            self._oracle_perturbation: dict[str, Action] | None = None
            self.state = "WAIT_STREAMS"
            self.state_started_s = time.monotonic()
            self.run_started_s: float | None = None
            self.run_finished_s: float | None = None
            self.next_command_s = 0.0
            self.next_policy_s = 0.0
            self.next_log_s = 0.0
            self.done = False
            self.episode_written = False
            self.failure_reason: str | None = None
            self._waiting_printed = False
            self._last_countdown_value: int | None = None
            self._stream_grace_until_s = 0.0
            self._reported_route_phase: str | None = None
            self._reported_uav0_tracking_ready = False
            self.create_timer(0.05, self._tick)
            self.get_logger().info(
                "实时 Handoff v3 已启动：安全绕墙双机控制/真实相机流 + 确定性几何真值验收。"
            )
            if self.visual_handoff:
                self.get_logger().info(
                    "可视化 Handoff 已启用：蓝色 UAV0 获取/跟踪，橙色 UAV1 "
                    "墙端安全点，绿色 staging，黄色接管点，青色 UAV0 支援点。"
                )

        @staticmethod
        def _now_us() -> int:
            return int(time.time_ns() // 1000)

        def _on_odometry(self, agent_id: str, message: Any) -> None:
            vehicle = self.vehicles[agent_id]
            vehicle.odometry = message
            vehicle.last_odometry_s = time.monotonic()
            if vehicle.origin is None:
                vehicle.origin = tuple(float(value) for value in message.position)
                vehicle.initial_yaw = _yaw_from_wxyz(message.q)
                vehicle.takeoff_target = (
                    vehicle.origin[0],
                    vehicle.origin[1],
                    vehicle.origin[2] - args.altitude,
                )
                self.get_logger().info(
                    f"{agent_id} 原点={vehicle.origin}，yaw={vehicle.initial_yaw:.3f}"
                )

        def _on_status(self, agent_id: str, message: Any) -> None:
            vehicle = self.vehicles[agent_id]
            vehicle.status = message
            vehicle.last_status_s = time.monotonic()

        def _on_camera(self, agent_id: str, message: Any) -> None:
            camera = self.cameras[agent_id]
            camera.frame_count += 1
            camera.last_received_s = time.monotonic()
            camera.width = int(message.width)
            camera.height = int(message.height)
            stamp = message.header.stamp
            camera.stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            camera.latest_message = message

        def _all_streams_ready(self) -> bool:
            return all(
                vehicle.origin is not None
                and vehicle.odometry is not None
                and vehicle.status is not None
                for vehicle in self.vehicles.values()
            ) and all(camera.frame_count > 0 for camera in self.cameras.values())

        def _all_armed_offboard(self) -> bool:
            return all(
                vehicle.status is not None
                and int(vehicle.status.arming_state) == self.armed_state
                and int(vehicle.status.nav_state) == self.offboard_state
                for vehicle in self.vehicles.values()
            )

        def _all_disarmed(self) -> bool:
            return all(
                vehicle.status is not None
                and int(vehicle.status.arming_state) != self.armed_state
                for vehicle in self.vehicles.values()
            )

        def _health(self, agent_id: str) -> str:
            now_s = time.monotonic()
            vehicle = self.vehicles[agent_id]
            camera = self.cameras[agent_id]
            if (
                now_s - vehicle.last_odometry_s > args.telemetry_timeout
                or now_s - vehicle.last_status_s > args.telemetry_timeout
                or now_s - camera.last_received_s > args.camera_timeout
            ):
                return "lost"
            if vehicle.status is not None and bool(vehicle.status.failsafe):
                return "degraded"
            return "ok"

        def _observation(self, agent_id: str) -> AgentObservation:
            vehicle = self.vehicles[agent_id]
            if vehicle.odometry is None or vehicle.origin is None:
                raise RuntimeError(f"{agent_id} observation requested before odometry")
            position = tuple(float(value) for value in vehicle.odometry.position)
            displacement = tuple(position[i] - vehicle.origin[i] for i in range(3))
            relative_flu = ned_displacement_to_initial_body_flu(
                displacement, vehicle.initial_yaw
            )
            velocity_flu = ned_displacement_to_initial_body_flu(
                tuple(float(value) for value in vehicle.odometry.velocity),
                vehicle.initial_yaw,
            )
            return AgentObservation(
                agent_id=agent_id,
                rgb_uri=self.cameras[agent_id].uri(),
                own_position=relative_flu,
                own_velocity=velocity_flu,
                own_yaw=_yaw_from_wxyz(vehicle.odometry.q) - vehicle.initial_yaw,
                health=self._health(agent_id),
                last_message=None,
            )

        def _observations(self) -> dict[str, AgentObservation]:
            return {agent_id: self._observation(agent_id) for agent_id in AGENT_IDS}

        def _common_positions(self) -> dict[str, tuple[float, float, float]]:
            result: dict[str, tuple[float, float, float]] = {}
            for agent_id, vehicle in self.vehicles.items():
                if vehicle.odometry is None or vehicle.origin is None:
                    raise RuntimeError(f"{agent_id} common position requested before odometry")
                result[agent_id] = common_ned_position(
                    tuple(float(value) for value in vehicle.odometry.position),
                    vehicle.origin,
                    vehicle.spawn_offset_ned,
                )
            return result

        def _validate_preflight(self) -> str | None:
            for agent_id, vehicle in self.vehicles.items():
                status = vehicle.status
                if status is None:
                    return f"{agent_id} missing vehicle_status_v4"
                if bool(status.failsafe):
                    return f"{agent_id} is in failsafe"
                # Headless SITL may report pre_flight_checks_pass=false when no
                # GCS is connected or simulated power input is unavailable. Keep
                # failsafe and arming-state checks elsewhere, but do not reject
                # startup solely on this non-actionable SITL preflight flag.
            return None

        def _stream_problem(self, now_s: float) -> str | None:
            for agent_id, vehicle in self.vehicles.items():
                if now_s - vehicle.last_odometry_s > args.telemetry_timeout:
                    return f"{agent_id} vehicle_odometry timeout"
                if now_s - vehicle.last_status_s > args.telemetry_timeout:
                    return f"{agent_id} vehicle_status timeout"
                if now_s - self.cameras[agent_id].last_received_s > args.camera_timeout:
                    return f"{agent_id} camera timeout"
            return None

        def _safety_problem(
            self, observations: Mapping[str, AgentObservation]
        ) -> str | None:
            common = self._common_positions()
            separation = distance_3d(common["uav_0"], common["uav_1"])
            if separation < self.limits.min_inter_vehicle_distance:
                return f"minimum separation violated: {separation:.2f} m"
            for agent_id in AGENT_IDS:
                problem = geofence_problem(observations[agent_id].own_position, self.limits)
                if problem:
                    return f"{agent_id}: {problem}"
            return None

        def _publish_position_setpoints(self, airborne: bool) -> None:
            for agent_id, vehicle in self.vehicles.items():
                if vehicle.origin is None or vehicle.takeoff_target is None:
                    continue
                desired = vehicle.takeoff_target if airborne else vehicle.origin
                timestamp = self._now_us()
                mode = OffboardControlMode()
                mode.timestamp = timestamp
                mode.position = True
                setpoint = TrajectorySetpoint()
                setpoint.timestamp = timestamp
                setpoint.position = [float(value) for value in desired]
                setpoint.velocity = [float("nan")] * 3
                setpoint.acceleration = [float("nan")] * 3
                setpoint.jerk = [float("nan")] * 3
                setpoint.yaw = float(vehicle.initial_yaw)
                setpoint.yawspeed = float("nan")
                self.publishers_by_agent[agent_id]["mode"].publish(mode)
                self.publishers_by_agent[agent_id]["setpoint"].publish(setpoint)

        def _publish_velocity_setpoints(self) -> None:
            for agent_id, vehicle in self.vehicles.items():
                timestamp = self._now_us()
                yaw = (
                    _yaw_from_wxyz(vehicle.odometry.q)
                    if vehicle.odometry is not None
                    else vehicle.initial_yaw
                )
                north, east, down, yaw_speed = body_flu_to_ned(vehicle.action, yaw)
                mode = OffboardControlMode()
                mode.timestamp = timestamp
                mode.velocity = True
                setpoint = TrajectorySetpoint()
                setpoint.timestamp = timestamp
                setpoint.position = [float("nan")] * 3
                setpoint.velocity = [north, east, down]
                setpoint.acceleration = [float("nan")] * 3
                setpoint.jerk = [float("nan")] * 3
                setpoint.yaw = float("nan")
                setpoint.yawspeed = yaw_speed
                self.publishers_by_agent[agent_id]["mode"].publish(mode)
                self.publishers_by_agent[agent_id]["setpoint"].publish(setpoint)

        def _send_command(
            self,
            agent_id: str,
            command: int,
            *,
            param1: float = 0.0,
            param2: float = 0.0,
        ) -> None:
            vehicle = self.vehicles[agent_id]
            message = VehicleCommand()
            message.timestamp = self._now_us()
            message.param1 = float(param1)
            message.param2 = float(param2)
            message.command = int(command)
            message.target_system = int(vehicle.system_id)
            message.target_component = 1
            message.source_system = 1
            message.source_component = 1
            message.from_external = True
            self.publishers_by_agent[agent_id]["command"].publish(message)

        def _request_offboard_and_arm(self) -> None:
            for agent_id in AGENT_IDS:
                self._send_command(
                    agent_id,
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    param1=1.0,
                    param2=6.0,
                )
                self._send_command(
                    agent_id,
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0,
                )

        def _send_land_commands(self) -> None:
            for agent_id in AGENT_IDS:
                self._send_command(agent_id, VehicleCommand.VEHICLE_CMD_NAV_LAND)

        def _flight_time_s(self) -> float:
            if self.run_started_s is None:
                return 0.0
            end_s = self.run_finished_s or time.monotonic()
            return max(0.0, end_s - self.run_started_s)

        def _capture_model_observation(
            self, observations: Mapping[str, AgentObservation]
        ) -> ModelObservation:
            if fixture is None:
                raise RuntimeError("model observation requires a frozen fixture")
            rgb: dict[str, bytes] = {}
            stamps: dict[str, int] = {}
            states: dict[str, VehicleState] = {}
            for agent_id in AGENT_IDS:
                camera = self.cameras[agent_id]
                if camera.latest_message is None:
                    raise RuntimeError(f"{agent_id} camera frame is unavailable")
                rgb[agent_id] = ros_image_to_jpeg(camera.latest_message)
                stamps[agent_id] = int(camera.stamp_ns)
                observation = observations[agent_id]
                states[agent_id] = VehicleState(
                    position=observation.own_position,
                    velocity=observation.own_velocity,
                    yaw=observation.own_yaw,
                    health=observation.health,
                )
            return ModelObservation(
                episode_id=self.episode_id,
                instruction=fixture.instruction,
                timestamp_ns=max(stamps.values()),
                image_timestamp_ns=stamps,
                rgb_jpeg=rgb,
                states=states,
                action_history=tuple(self._action_history),
                message_history=tuple(self._message_history),
            )

        @staticmethod
        def _waypoint_label(requested: JointAction, *, confirm: bool = False) -> ModelJointAction:
            period = 0.5
            waypoints = {
                agent_id: WaypointDelta(
                    dx=getattr(requested, agent_id).vx * period,
                    dy=getattr(requested, agent_id).vy * period,
                    dz=getattr(requested, agent_id).vz * period,
                    dyaw=getattr(requested, agent_id).yaw_rate * period,
                ).clipped()
                for agent_id in AGENT_IDS
            }
            intent = "confirm" if confirm else "none"
            if requested.message is not None:
                payload = requested.message.get("payload", requested.message)
                if payload.get("intent") == "handoff_request":
                    intent = "handoff"
            return ModelJointAction(waypoints["uav_0"], waypoints["uav_1"], intent)

        @staticmethod
        def _dataset_record_step(
            decision_step: bool, requested: JointAction
        ) -> bool:
            """Record ordinary 2 Hz rows and expert protocol-decision rows."""
            if decision_step:
                return True
            message = requested.message
            if not isinstance(message, Mapping):
                return False
            payload = message.get("payload", message)
            if not isinstance(payload, Mapping):
                return False
            return str(payload.get("intent", "")) in {
                "handoff_request", "confirm",
            }

        def _apply_oracle_perturbation(self, requested: JointAction, decision_step: bool) -> JointAction:
            if args.policy != "oracle" or fixture is None:
                return requested
            if decision_step:
                self._oracle_perturbation = None
                if self._perturb_rng.random() < args.oracle_perturb_probability:
                    self._oracle_perturbation = {
                        agent_id: Action(
                            vx=self._perturb_rng.uniform(-0.6, 0.6),
                            vy=self._perturb_rng.uniform(-0.6, 0.6),
                            vz=self._perturb_rng.uniform(-0.2, 0.2),
                            yaw_rate=self._perturb_rng.uniform(-0.15, 0.15),
                        )
                        for agent_id in AGENT_IDS
                    }
            if self._oracle_perturbation is None:
                return requested
            perturbed: dict[str, Action] = {}
            for agent_id in AGENT_IDS:
                base = getattr(requested, agent_id)
                noise = self._oracle_perturbation[agent_id]
                perturbed[agent_id] = Action(
                    base.vx + noise.vx, base.vy + noise.vy,
                    base.vz + noise.vz, base.yaw_rate + noise.yaw_rate,
                ).clipped()
            return JointAction(perturbed["uav_0"], perturbed["uav_1"], requested.message)

        def _planning_action_to_vehicle_body(self, requested: JointAction) -> JointAction:
            """Convert UAV0 task-frame velocities to each vehicle's current body FLU."""
            converted: dict[str, Action] = {}
            planning_yaw = self.vehicles["uav_0"].initial_yaw
            for agent_id in AGENT_IDS:
                action = getattr(requested, agent_id)
                north, east, down, _ = body_flu_to_ned(action, planning_yaw)
                vehicle = self.vehicles[agent_id]
                current_yaw = (
                    _yaw_from_wxyz(vehicle.odometry.q)
                    if vehicle.odometry is not None
                    else vehicle.initial_yaw
                )
                vx, vy, vz = ned_displacement_to_initial_body_flu(
                    (north, east, down), current_yaw
                )
                converted[agent_id] = Action(vx, vy, vz, action.yaw_rate).clipped()
            return JointAction(
                converted["uav_0"], converted["uav_1"], requested.message
            )

        def _apply_policy_step(self) -> None:
            observations = self._observations()
            problem = self._safety_problem(observations)
            if problem:
                self.request_land(problem, failed=True, termination="safety_abort")
                return
            common_positions = self._common_positions()
            evaluation_positions = evaluation_positions_in_uav0_flu(
                common_positions, self.vehicles["uav_0"].initial_yaw
            )
            decision_step = self.protocol.t % self._model_decision_stride == 0
            record_step = decision_step
            policy_metadata: dict[str, Any] | None = None
            dataset_observation: ModelObservation | None = None
            dataset_label: ModelJointAction | None = None
            if args.policy == "model":
                if decision_step:
                    try:
                        dataset_observation = self._capture_model_observation(observations)
                        assert self.policy_client is not None
                        self._model_action, latency_ms = self.policy_client.infer(dataset_observation)
                        self._consecutive_policy_timeouts = 0
                        policy_metadata = {
                            "decision": True, "method": "model", "timed_out": False,
                            "inference_latency_ms": latency_ms,
                        }
                    except PolicyTimeoutError as exc:
                        self._model_action = hover_action()
                        self._consecutive_policy_timeouts += 1
                        policy_metadata = {
                            "decision": True, "method": "model", "timed_out": True,
                        }
                        self.get_logger().warning(str(exc) + "; commanding hover")
                        if self._consecutive_policy_timeouts >= args.max_policy_timeouts:
                            self.request_land(
                                "three consecutive policy inference timeouts",
                                failed=True,
                                termination="safety_abort",
                            )
                            return
                    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                        self.request_land(
                            f"model policy failure: {exc}", failed=True,
                            termination="safety_abort",
                        )
                        return
                    self._action_history.append({
                        "uav_0": self._model_action.uav_0.vector(),
                        "uav_1": self._model_action.uav_1.vector(),
                    })
                    if self._model_action.intent != "none":
                        self._message_history.append({"intent": self._model_action.intent})
                requested = self._model_action.to_velocity_action(period_s=0.5)
                if not decision_step:
                    requested = JointAction(requested.uav_0, requested.uav_1, None)
                dataset_label = self._model_action
            else:
                assert self.policy is not None
                policy_input = self.protocol.policy_observation(observations)
                # Handcrafted baselines plan in the canonical task frame. The
                # recorded model observation retains per-vehicle local state.
                for agent_id in AGENT_IDS:
                    policy_input["observations"][agent_id]["own_state"]["position"] = list(
                        evaluation_positions[agent_id]
                    )
                requested = self.policy.act(policy_input)
                route_phase = getattr(self.policy, "uav1_route_phase", None)
                if isinstance(route_phase, str) and route_phase != self._reported_route_phase:
                    route_messages = {
                        "clearance": "UAV1 ROUTE: flying to the wall-end clearance waypoint",
                        "staging": "UAV1 CLEARANCE REACHED: turning toward staging",
                        "ready": "UAV1 STAGED: stable and ready to accept Handoff",
                    }
                    if route_phase in route_messages:
                        self.get_logger().info(route_messages[route_phase])
                    self._reported_route_phase = route_phase
                tracking_ready = bool(
                    getattr(self.policy, "uav0_tracking_ready", False)
                )
                if tracking_ready and not self._reported_uav0_tracking_ready:
                    self.get_logger().info(
                        "UAV0 TRACKING ESTABLISHED: following target on the near side"
                    )
                    self._reported_uav0_tracking_ready = True
                record_step = self._dataset_record_step(decision_step, requested)
                if record_step and self.dataset_writer is not None:
                    try:
                        dataset_observation = self._capture_model_observation(observations)
                    except (RuntimeError, ValueError) as exc:
                        self.request_land(
                            f"dataset camera synchronization failed: {exc}",
                            failed=True, termination="simulator_error",
                        )
                        return
                unperturbed_requested = requested
                requested = self._apply_oracle_perturbation(requested, decision_step)
                requested = self._planning_action_to_vehicle_body(requested)
                unperturbed_requested = self._planning_action_to_vehicle_body(
                    unperturbed_requested
                )
                if record_step and self.dataset_writer is not None:
                    dataset_label = self._waypoint_label(unperturbed_requested)
            limited_actions: dict[str, Action] = {}
            for agent_id in AGENT_IDS:
                action = getattr(requested, agent_id)
                if not finite_action(action):
                    self.request_land(
                        f"{agent_id} policy produced non-finite action",
                        failed=True,
                        termination="safety_abort",
                    )
                    return
                vehicle = self.vehicles[agent_id]
                limited_actions[agent_id] = acceleration_limited_action(
                    action,
                    vehicle.action,
                    1.0 / self.task.control_hz,
                    self.limits,
                )
            joint_action = JointAction(
                uav_0=limited_actions["uav_0"],
                uav_1=limited_actions["uav_1"],
                message=requested.message,
            )
            for agent_id in AGENT_IDS:
                vehicle = self.vehicles[agent_id]
                vehicle.previous_action = vehicle.action
                vehicle.action = getattr(joint_action, agent_id)
            target = target_at_step(
                self.task.target_start,
                self.task.target_end,
                self.protocol.t,
                self.task.horizon,
            )
            self.scene.update_target(target)
            result = self.protocol.step(
                observations, evaluation_positions, joint_action,
                policy_metadata=policy_metadata,
            )
            if self.dataset_writer is not None and record_step and dataset_observation is not None:
                if dataset_label is None:
                    dataset_label = self._waypoint_label(joint_action)
                if args.policy != "model" and any(
                    event.get("event_type") == "verify" for event in result["events"]
                ):
                    dataset_label = self._waypoint_label(unperturbed_requested, confirm=True)
                self.dataset_writer.append(
                    dataset_observation,
                    dataset_label,
                    events=result["events"],
                    truth=result["truth"],
                    inference_latency_ms=(
                        float(policy_metadata["inference_latency_ms"])
                        if policy_metadata and "inference_latency_ms" in policy_metadata else None
                    ),
                    timed_out=bool(policy_metadata and policy_metadata.get("timed_out")),
                )
            for event in result["events"]:
                event_type = str(event.get("event_type", ""))
                if event_type == "handoff_request":
                    self.get_logger().info("HANDOFF REQUEST: UAV0 -> UAV1")
                elif event_type == "handoff_ack_sent":
                    self.get_logger().info(
                        "HANDOFF ACK: UAV1 taking tracking responsibility"
                    )
                elif event_type == "handoff":
                    self.get_logger().info(
                        "UAV1 ACTIVE TRACKER: leaving staging point and following target"
                    )
                elif event_type == "verify":
                    self.get_logger().info(
                        "HANDOFF VERIFIED: maintaining visible UAV1 tracking dwell"
                    )
            if result["done"]:
                termination = str(result["termination_reason"])
                self.run_finished_s = time.monotonic()
                if termination == "success":
                    self.request_land("Handoff 验收成功")
                else:
                    self.request_land(
                        f"Handoff 终止：{termination}",
                        failed=True,
                        termination=termination,
                    )

        def request_land(
            self,
            reason: str,
            *,
            failed: bool = False,
            termination: str = "simulator_error",
        ) -> None:
            if self.done:
                return
            if failed and self.failure_reason is None:
                self.failure_reason = reason
            if failed:
                if self.protocol.done:
                    self.protocol.failure_reason = reason
                    self.protocol.termination_reason = termination
                else:
                    self.protocol.abort(reason, termination)
            any_armed = any(
                vehicle.status is not None
                and int(vehicle.status.arming_state) == self.armed_state
                for vehicle in self.vehicles.values()
            )
            if not any_armed and self.state in {
                "WAIT_STREAMS",
                "COUNTDOWN",
                "PRESTREAM",
                "REQUEST_OFFBOARD",
            }:
                if not self.protocol.done:
                    self.protocol.abort(reason, termination if failed else "safety_abort")
                level = self.get_logger().error if failed else self.get_logger().info
                level(f"起飞前终止：{reason}")
                self._finalize_episode()
                return
            if self.state != "LANDING":
                level = self.get_logger().error if failed else self.get_logger().info
                level(f"进入安全降落流程：{reason}")
                for vehicle in self.vehicles.values():
                    vehicle.previous_action = vehicle.action
                    vehicle.action = Action()
                self.run_finished_s = self.run_finished_s or time.monotonic()
                self.state = "LANDING"
                self.state_started_s = time.monotonic()
                self.next_command_s = 0.0
                self._send_land_commands()

        def _log_status(self, now_s: float) -> None:
            if now_s < self.next_log_s:
                return
            self.next_log_s = now_s + 1.0
            positions = []
            for agent_id, vehicle in self.vehicles.items():
                if vehicle.odometry is not None:
                    p = tuple(float(value) for value in vehicle.odometry.position)
                    positions.append(f"{agent_id}=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
            self.get_logger().info(
                f"状态={self.state}；" + "；".join(positions)
                + f"；active_tracker={'uav_1' if self.protocol.handoff_t is not None else 'uav_0'}"
                + f"；protocol_step={self.protocol.t}/{self.task.horizon}"
            )

        def _tick(self) -> None:
            if self.done:
                return
            now_s = time.monotonic()
            if (
                self.scene.last_error is not None
                and self.state != "LANDING"
                and self.strict_scene
            ):
                self.request_land(
                    f"Gazebo target pose synchronization failed: {self.scene.last_error}",
                    failed=True,
                    termination="simulator_error",
                )
                return
            if self.scene.last_error is not None and self.state != "LANDING":
                self.get_logger().warning(
                    "Gazebo visual scene update failed; disabling optional "
                    f"visual pose updates and continuing flight: {self.scene.last_error}"
                )
                self.scene.disable_updates()
            if self.state == "WAIT_STREAMS":
                if not self._all_streams_ready():
                    if not self._waiting_printed:
                        self.get_logger().info("等待 odometry/status/camera ...")
                        self._waiting_printed = True
                    if now_s - self.state_started_s > args.startup_timeout:
                        self.request_land(
                            "startup stream timeout",
                            failed=True,
                            termination=("simulator_error" if self.strict_scene else "timeout"),
                        )
                    return
                problem = self._validate_preflight()
                if problem:
                    self.request_land(problem, failed=True)
                    return
                try:
                    self.scene.spawn(
                        self.task,
                        self.vehicles["uav_0"].initial_yaw,
                        spawn_offset_ned=spawn,
                        target_appearance=fixture.target if fixture is not None else None,
                        distractor_appearances=fixture.distractors if fixture is not None else (),
                        show_markers=(
                            bool(args.visual_handoff)
                            and not (args.formal_model or args.hide_markers)
                        ),
                    )
                except Exception as exc:
                    self.request_land(
                        f"Gazebo visual scene setup failed: {exc}",
                        failed=True,
                        termination="simulator_error",
                    )
                    return
                # Gazebo entity creation can block for longer than the
                # telemetry freshness window. Refresh the clock before the
                # next state checks so a stale pre-spawn timestamp cannot
                # masquerade as a telemetry timeout.
                now_s = time.monotonic()
                self.state = "COUNTDOWN"
                self.state_started_s = now_s
                # Gazebo scene setup can temporarily starve the executor while
                # PX4 keeps publishing. Give subscriptions a short recovery
                # window before applying the normal telemetry freshness check.
                self._stream_grace_until_s = now_s + max(
                    3.0, float(args.telemetry_timeout)
                )
                self.get_logger().info(
                    "双机里程计、状态与真实相机流已就绪。请保持人员远离旋翼区域。"
                )
                # scene.spawn() performs blocking Gazebo service calls, so no
                # ROS subscription callbacks can run while it is executing.
                # Defer stream validation to the next timer tick and let the
                # executor refresh odometry/status/camera samples first.
                return

            if (
                self.state not in {"WAIT_STREAMS", "LANDING"}
                and now_s >= self._stream_grace_until_s
            ):
                problem = self._stream_problem(now_s) or self._validate_preflight()
                if problem:
                    self.request_land(problem, failed=True)
                    return

            if self.state == "COUNTDOWN":
                self._publish_position_setpoints(False)
                remaining = max(
                    0,
                    math.ceil(args.countdown - (now_s - self.state_started_s)),
                )
                if remaining != self._last_countdown_value:
                    self._last_countdown_value = remaining
                    self.get_logger().info(f"自动起飞倒计时：{remaining}")
                if now_s - self.state_started_s >= args.countdown:
                    self.state = "PRESTREAM"
                    self.state_started_s = now_s

            elif self.state == "PRESTREAM":
                self._publish_position_setpoints(False)
                if now_s - self.state_started_s >= args.prestream:
                    self.state = "REQUEST_OFFBOARD"
                    self.state_started_s = now_s
                    self.next_command_s = 0.0

            elif self.state == "REQUEST_OFFBOARD":
                self._publish_position_setpoints(False)
                if now_s >= self.next_command_s:
                    self._request_offboard_and_arm()
                    self.next_command_s = now_s + 1.0
                if self._all_armed_offboard():
                    self.state = "TAKEOFF"
                    self.state_started_s = now_s
                    self.get_logger().info(
                        f"双机进入 Offboard，起飞至 {args.altitude:.1f} m。"
                    )
                elif now_s - self.state_started_s > 12.0:
                    self.request_land("Offboard/arm timeout", failed=True)

            elif self.state == "TAKEOFF":
                self._publish_position_setpoints(True)
                reached = all(
                    vehicle.odometry is not None
                    and vehicle.takeoff_target is not None
                    and abs(
                        float(vehicle.odometry.position[2])
                        - vehicle.takeoff_target[2]
                    )
                    <= 0.35
                    for vehicle in self.vehicles.values()
                )
                self._log_status(now_s)
                if reached:
                    self.state = "RUN"
                    self.state_started_s = now_s
                    self.run_started_s = now_s
                    self.next_policy_s = 0.0
                    self.get_logger().info(
                        (
                            "到达高度，开始 2 Hz model policy / 20 Hz PX4 setpoint。"
                            if args.policy == "model"
                            else "到达高度，开始 10 Hz Handoff policy / 20 Hz PX4 setpoint。"
                        )
                    )
                elif now_s - self.state_started_s > args.takeoff_timeout:
                    self.request_land(
                        "takeoff timeout",
                        failed=True,
                        termination=("simulator_error" if self.strict_scene else "timeout"),
                    )

            elif self.state == "RUN":
                self._publish_velocity_setpoints()
                if now_s >= self.next_policy_s:
                    self._apply_policy_step()
                    self.next_policy_s = now_s + 1.0 / self.task.control_hz
                self._log_status(now_s)

            elif self.state == "LANDING":
                if self._all_disarmed():
                    self._finalize_episode()
                    if self.failure_reason:
                        self.get_logger().error(
                            f"双机已落地；Handoff 失败：{self.failure_reason}"
                        )
                    else:
                        self.get_logger().info(
                            f"双机已落地；Handoff episode 已写入 {args.output}"
                        )
                    return
                if now_s >= self.next_command_s:
                    self._send_land_commands()
                    self.next_command_s = now_s + 1.0
                if now_s - self.state_started_s > args.landing_timeout:
                    self.failure_reason = self.failure_reason or "landing timeout"
                    self.protocol.failure_reason = self.failure_reason
                    self.protocol.termination_reason = "timeout"
                    self.protocol.done = True
                    self._finalize_episode()
                    self.get_logger().error("降落超时，请立即在 QGroundControl 检查。")

        def episode_record(self) -> dict[str, Any]:
            return build_episode_record(
                task=self.task,
                protocol=self.protocol,
                seed=args.seed,
                world=args.world,
                policy_name=args.policy,
                spawn_offset_ned=spawn,
                cameras=self.cameras,
                flight_time_s=self._flight_time_s(),
                episode_id=self.episode_id,
            )

        def write_episode(self) -> None:
            if self.episode_written:
                return
            append_episode_record(self.episode_record(), args.output)
            self.episode_written = True

        def _finalize_episode(self) -> None:
            """Write one terminal schema-0.2 record and stop the spin loop."""
            if self.done:
                return
            self.scene.close()
            if self.strict_scene and self.scene.last_error is not None:
                self.failure_reason = (
                    "Gazebo target pose synchronization failed: "
                    f"{self.scene.last_error}"
                )
                self.protocol.failure_reason = self.failure_reason
                self.protocol.termination_reason = "simulator_error"
                self.protocol.done = True
            if not self.protocol.done:
                reason = self.failure_reason or "episode finalized before protocol completion"
                self.protocol.abort(reason, "safety_abort")
            try:
                self.write_episode()
            except Exception as exc:
                self.done = True
                self.get_logger().error(f"episode 校验或写入失败：{exc}")
                raise
            self.done = True

    rclpy.init()
    node = LiveHandoffNode()
    interrupted = False

    def _handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        node.request_land(
            "收到 Ctrl+C",
            failed=True,
            termination="safety_abort",
        )

    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception as exc:
        node.request_land(
            f"未处理异常：{exc}",
            failed=True,
            termination="simulator_error",
        )
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not node.done and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if not node.done:
            node.request_land(
                "节点退出前的最终安全降落",
                failed=True,
                termination="safety_abort",
            )
            deadline = time.monotonic() + 3.0
            while rclpy.ok() and not node.done and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        node.scene.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if interrupted:
        raise SystemExit(130)
    if node.failure_reason or not node.episode_record()["outcome"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
