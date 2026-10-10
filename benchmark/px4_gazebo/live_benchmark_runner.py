"""Live PX4-Gazebo dual-control / observation diagnostic benchmark.

This is deliberately not a visual handoff benchmark yet: it verifies the
live policy-to-PX4 control path, both camera streams, safety checks, landing,
and schema-compatible episode logging without exposing teammate ground truth
to either policy observation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import signal
import time
from typing import Any
import uuid

from .contract import AGENT_IDS, Action, AgentObservation, JointAction
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


@dataclass
class CameraRuntime:
    topic: str
    frame_count: int = 0
    last_received_s: float = 0.0
    stamp_ns: int = 0
    width: int = 0
    height: int = 0

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PX4-Gazebo 实时双机控制/观测诊断（不是完整视觉 handoff 任务）"
    )
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--preposition-distance", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--world",
        default="default",
        help="Gazebo world name used in camera topics, for example default or lawn",
    )
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
        help="第二架飞机相对第一架的 Gazebo 出生点 NED 偏移，必须与启动参数一致",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark/px4_gazebo/logs/live_fixed_role_diagnostic.jsonl"),
    )
    return parser


def validate_runtime_arguments(
    args: argparse.Namespace, limits: SafetyLimits = SafetyLimits()
) -> tuple[float, float, float]:
    """Reject invalid or unsafe settings before importing ROS or arming PX4."""
    numeric_fields = {
        "altitude": args.altitude,
        "preposition-distance": args.preposition_distance,
        "duration": args.duration,
        "countdown": args.countdown,
        "prestream": args.prestream,
        "telemetry-timeout": args.telemetry_timeout,
        "camera-timeout": args.camera_timeout,
        "startup-timeout": args.startup_timeout,
        "takeoff-timeout": args.takeoff_timeout,
        "landing-timeout": args.landing_timeout,
    }
    non_finite = [
        name for name, value in numeric_fields.items() if not math.isfinite(float(value))
    ]
    try:
        spawn = tuple(float(value) for value in args.uav1_spawn_offset_ned)
    except (TypeError, ValueError) as exc:
        raise ValueError("uav1-spawn-offset-ned must contain 3 finite numbers") from exc
    if len(spawn) != 3 or not all(math.isfinite(value) for value in spawn):
        non_finite.append("uav1-spawn-offset-ned")
    if non_finite:
        raise ValueError("non-finite arguments: " + ", ".join(non_finite))

    strictly_positive = (
        "altitude",
        "preposition-distance",
        "duration",
        "telemetry-timeout",
        "camera-timeout",
        "startup-timeout",
        "takeoff-timeout",
        "landing-timeout",
    )
    invalid_positive = [
        name for name in strictly_positive if float(numeric_fields[name]) <= 0.0
    ]
    if invalid_positive:
        raise ValueError("arguments must be > 0: " + ", ".join(invalid_positive))
    if args.countdown < 0.0 or args.prestream < 0.0:
        raise ValueError("countdown and prestream must be >= 0")
    if args.altitude > limits.max_altitude:
        raise ValueError(f"altitude exceeds {limits.max_altitude:.1f} m safety limit")
    if args.preposition_distance > limits.max_distance_from_origin:
        raise ValueError(
            "preposition-distance exceeds "
            f"{limits.max_distance_from_origin:.1f} m geofence"
        )
    if math.hypot(spawn[0], spawn[1]) < limits.min_inter_vehicle_distance:
        raise ValueError(
            "declared horizontal spawn separation is below "
            f"{limits.min_inter_vehicle_distance:.1f} m"
        )
    return spawn


def validate_episode_record(
    record: dict[str, Any], schema_path: Path | None = None
) -> None:
    """Validate one episode without importing ROS 2 or starting a vehicle."""
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate live episode logs") from exc
    resolved_schema = schema_path or (
        Path(__file__).resolve().parents[1] / "spec" / "episode_schema.json"
    )
    schema = json.loads(resolved_schema.read_text(encoding="utf-8"))
    jsonschema.validate(instance=record, schema=schema)


def append_episode_record(record: dict[str, Any], output: Path) -> None:
    """Validate and append exactly one JSONL record."""
    validate_episode_record(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class FixedRoleLivePolicy:
    """UAV 0 holds; UAV 1 moves away from the declared spawn baseline."""

    def __init__(
        self, direction_ned: tuple[float, float], distance_m: float, altitude_m: float
    ) -> None:
        norm = math.hypot(*direction_ned)
        self.direction_ned = (direction_ned[0] / norm, direction_ned[1] / norm)
        self.distance_m = distance_m
        self.altitude_m = altitude_m

    def act(
        self,
        observations: dict[str, AgentObservation],
        uav1_initial_yaw: float,
    ) -> JointAction:
        current = observations["uav_1"].own_position
        desired_ned = (
            self.direction_ned[0] * self.distance_m,
            self.direction_ned[1] * self.distance_m,
            0.0,
        )
        desired_flu = ned_displacement_to_initial_body_flu(desired_ned, uav1_initial_yaw)
        # Pre-position is scored horizontally while a small velocity controller
        # holds the requested takeoff altitude.
        error = (
            desired_flu[0] - current[0],
            desired_flu[1] - current[1],
            self.altitude_m - current[2],
        )
        return JointAction(
            uav_0=Action(),
            uav_1=Action(
                vx=max(-0.8, min(0.8, 0.7 * error[0])),
                vy=max(-0.8, min(0.8, 0.7 * error[1])),
                vz=max(-0.4, min(0.4, 0.7 * error[2])),
            ),
        )


def main() -> None:
    args = _build_parser().parse_args()
    try:
        spawn = validate_runtime_arguments(args)
    except ValueError as exc:
        raise SystemExit(f"argument error: {exc}") from exc

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

    class LiveBenchmarkNode(Node):
        def __init__(self) -> None:
            super().__init__("aerowam_live_dual_diagnostic")
            self.vehicles = {
                "uav_0": VehicleRuntime(PX4_NAMESPACES["uav_0"], PX4_SYSTEM_IDS["uav_0"], (0.0, 0.0, 0.0)),
                "uav_1": VehicleRuntime(PX4_NAMESPACES["uav_1"], PX4_SYSTEM_IDS["uav_1"], spawn),
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
                self._owned_subscription_handles.extend(
                    [
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
                    ]
                )

            self.armed_state = int(getattr(VehicleStatus, "ARMING_STATE_ARMED", 2))
            self.offboard_state = int(getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14))
            self.policy = FixedRoleLivePolicy(
                (spawn[0], spawn[1]), args.preposition_distance, args.altitude
            )
            self.limits = SafetyLimits()
            self.state = "WAIT_STREAMS"
            self.state_started_s = time.monotonic()
            self.next_command_s = 0.0
            self.next_policy_s = 0.0
            self.next_log_s = 0.0
            self.run_started_s: float | None = None
            self.done = False
            self.episode_written = False
            self.failure_reason: str | None = None
            self.termination_reason = "success"
            self.mission_completed = False
            self.step_index = 0
            self.steps: list[dict[str, Any]] = []
            self.minimum_separation_m = float("inf")
            self.preposition_error_m = float("inf")
            self.preposition_success = False
            self.episode_id = f"live-dual-diagnostic-{uuid.uuid4().hex[:12]}"
            self.started_wall_ms = int(time.time() * 1000)
            self._last_countdown_value: int | None = None
            self._waiting_printed = False
            self.create_timer(0.05, self._tick)
            self.get_logger().info(
                "实时双机诊断已启动，等待两机 odometry/status 和两路 camera。"
            )

        def _now_us(self) -> int:
            return int(self.get_clock().now().nanoseconds / 1000)

        def _on_odometry(self, agent_id: str, message: Any) -> None:
            if not _finite_vector(message.position, 3):
                self.request_land(f"{agent_id} 收到非有限 odometry", failed=True)
                return
            vehicle = self.vehicles[agent_id]
            vehicle.odometry = message
            vehicle.last_odometry_s = time.monotonic()
            if vehicle.origin is None:
                vehicle.origin = tuple(float(value) for value in message.position)
                vehicle.initial_yaw = _yaw_from_wxyz(message.q)
                vehicle.takeoff_target = (
                    vehicle.origin[0], vehicle.origin[1], vehicle.origin[2] - args.altitude
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

        def _all_streams_ready(self) -> bool:
            return all(
                vehicle.origin is not None and vehicle.odometry is not None and vehicle.status is not None
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
            yaw = _yaw_from_wxyz(vehicle.odometry.q) - vehicle.initial_yaw
            return AgentObservation(
                agent_id=agent_id,
                rgb_uri=self.cameras[agent_id].uri(),
                own_position=relative_flu,
                own_velocity=velocity_flu,
                own_yaw=yaw,
                health=self._health(agent_id),
                last_message=None,
            )

        def _observations(self) -> dict[str, AgentObservation]:
            return {agent_id: self._observation(agent_id) for agent_id in AGENT_IDS}

        def reset(self, seed: int, task_spec: dict[str, Any]) -> dict[str, AgentObservation]:
            """Expose the common environment API after all live streams are ready."""
            if seed != args.seed:
                raise ValueError("live runner seed must match the CLI seed")
            if not self._all_streams_ready():
                raise RuntimeError("live reset requested before all streams are ready")
            return self._observations()

        def _common_positions(self) -> dict[str, tuple[float, float, float]]:
            result: dict[str, tuple[float, float, float]] = {}
            for agent_id, vehicle in self.vehicles.items():
                assert vehicle.odometry is not None and vehicle.origin is not None
                result[agent_id] = common_ned_position(
                    tuple(float(value) for value in vehicle.odometry.position),
                    vehicle.origin,
                    vehicle.spawn_offset_ned,
                )
            return result

        def _safety_problem(self) -> str | None:
            common = self._common_positions()
            separation = distance_3d(common["uav_0"], common["uav_1"])
            self.minimum_separation_m = min(self.minimum_separation_m, separation)
            if separation < self.limits.min_inter_vehicle_distance:
                return f"minimum separation violated: {separation:.2f} m"
            for agent_id in AGENT_IDS:
                problem = geofence_problem(self._observation(agent_id).own_position, self.limits)
                if problem:
                    return f"{agent_id}: {problem}"
            return None

        def _schema_observation(self, observation: AgentObservation) -> dict[str, Any]:
            return {
                "rgb_uri": observation.rgb_uri,
                "own_state": [
                    *[float(value) for value in observation.own_position],
                    *[float(value) for value in observation.own_velocity],
                    float(observation.own_yaw),
                ],
                "health": observation.health,
                "last_message": observation.last_message,
            }

        def step(
            self, joint_action: JointAction, message: dict[str, Any] | None = None
        ) -> tuple[dict[str, AgentObservation], float, bool, dict[str, Any]]:
            """Apply one 10 Hz policy decision and append one public-data log step."""
            observations = self._observations()
            dt = 0.1
            for agent_id in AGENT_IDS:
                vehicle = self.vehicles[agent_id]
                requested = getattr(joint_action, agent_id)
                if not finite_action(requested):
                    raise ValueError(f"{agent_id} policy action is not finite")
                vehicle.action = acceleration_limited_action(
                    requested, vehicle.previous_action, dt, self.limits
                )
                vehicle.previous_action = vehicle.action

            safety = self._safety_problem()
            if safety:
                self.request_land(safety, failed=True, termination="safety_abort")

            uav1 = observations["uav_1"].own_position
            target_ned = (
                self.policy.direction_ned[0] * args.preposition_distance,
                self.policy.direction_ned[1] * args.preposition_distance,
                0.0,
            )
            target_flu = ned_displacement_to_initial_body_flu(
                target_ned, self.vehicles["uav_1"].initial_yaw
            )
            self.preposition_error_m = math.hypot(
                uav1[0] - target_flu[0], uav1[1] - target_flu[1]
            )
            just_succeeded = not self.preposition_success and self.preposition_error_m <= 0.35
            self.preposition_success = self.preposition_success or just_succeeded
            events = []
            if just_succeeded:
                events.append(
                    {
                        "event_type": "pre_position",
                        "agent_ids": ["uav_1"],
                        "start_t": 0,
                        "end_t": self.step_index,
                        "success": True,
                        "attributes": {"error_m": self.preposition_error_m},
                    }
                )
            communication = {"messages": [], "bytes_sent": 0, "dropped": False}
            if message is not None:
                payload_bytes = len(json.dumps(message, ensure_ascii=False).encode("utf-8"))
                communication["messages"] = [
                    {
                        "sender": "uav_0",
                        "receiver": "uav_1",
                        "payload": message,
                        "sent_timestamp_ms": int(time.time() * 1000),
                        "deliver_timestamp_ms": int(time.time() * 1000),
                    }
                ]
                communication["bytes_sent"] = payload_bytes
            self.steps.append(
                {
                    "t": self.step_index,
                    "timestamp_ms": int(time.time() * 1000),
                    "observations": {
                        key: self._schema_observation(value)
                        for key, value in observations.items()
                    },
                    "actions": {
                        agent_id: {
                            "control": list(self.vehicles[agent_id].action.as_vector()),
                            "valid": True,
                        }
                        for agent_id in AGENT_IDS
                    },
                    "communication": communication,
                    "events": events,
                }
            )
            self.step_index += 1
            reward = -self.preposition_error_m
            return observations, reward, self.state == "LANDING", {
                "preposition_error_m": self.preposition_error_m,
                "minimum_separation_m": self.minimum_separation_m,
            }

        def _validate_preflight(self) -> str | None:
            for agent_id, vehicle in self.vehicles.items():
                status = vehicle.status
                if status is None:
                    return f"{agent_id} missing vehicle_status_v4"
                if bool(status.failsafe):
                    return f"{agent_id} is in failsafe"
                if not bool(status.pre_flight_checks_pass):
                    return f"{agent_id} pre-flight checks failed"
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
                yaw = _yaw_from_wxyz(vehicle.odometry.q) if vehicle.odometry is not None else vehicle.initial_yaw
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
            self, agent_id: str, command: int, *, param1: float = 0.0, param2: float = 0.0
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
                    agent_id, VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
                )
                self._send_command(
                    agent_id, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
                )

        def _send_land_commands(self) -> None:
            for agent_id in AGENT_IDS:
                self._send_command(agent_id, VehicleCommand.VEHICLE_CMD_NAV_LAND)

        def request_land(
            self, reason: str, *, failed: bool = False, termination: str = "simulator_error"
        ) -> None:
            if self.done:
                return
            if failed and self.failure_reason is None:
                self.failure_reason = reason
                self.termination_reason = termination
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
                level = self.get_logger().error if failed else self.get_logger().info
                level(f"起飞前终止：{reason}")
                self._finalize_episode()
                return
            if self.state != "LANDING":
                level = self.get_logger().error if failed else self.get_logger().info
                level(f"进入安全降落流程：{reason}")
                for vehicle in self.vehicles.values():
                    vehicle.action = Action()
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
                + f"；preposition_error={self.preposition_error_m:.2f} m"
            )

        def _tick(self) -> None:
            if self.done:
                return
            now_s = time.monotonic()
            if self.state == "WAIT_STREAMS":
                if not self._all_streams_ready():
                    if not self._waiting_printed:
                        self.get_logger().info("等待 odometry/status/camera ...")
                        self._waiting_printed = True
                    if now_s - self.state_started_s > args.startup_timeout:
                        self.request_land(
                            "startup stream timeout", failed=True, termination="timeout"
                        )
                    return
                problem = self._validate_preflight()
                if problem:
                    self.request_land(problem, failed=True)
                    return
                self.reset(args.seed, {"task_id": "live_dual_control_diagnostic"})
                self.state = "COUNTDOWN"
                self.state_started_s = now_s
                self.get_logger().info("双机和双相机就绪。请保持人员远离旋翼区域。")

            if self.state not in {"WAIT_STREAMS", "LANDING"}:
                problem = self._stream_problem(now_s) or self._validate_preflight()
                if problem:
                    self.request_land(problem, failed=True)
                    return

            if self.state == "COUNTDOWN":
                self._publish_position_setpoints(False)
                remaining = max(0, math.ceil(args.countdown - (now_s - self.state_started_s)))
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
                    self.get_logger().info(f"双机进入 Offboard，起飞至 {args.altitude:.1f} m。")
                elif now_s - self.state_started_s > 12.0:
                    self.request_land("Offboard/arm timeout", failed=True)

            elif self.state == "TAKEOFF":
                self._publish_position_setpoints(True)
                reached = all(
                    vehicle.odometry is not None
                    and vehicle.takeoff_target is not None
                    and abs(float(vehicle.odometry.position[2]) - vehicle.takeoff_target[2]) <= 0.35
                    for vehicle in self.vehicles.values()
                )
                self._log_status(now_s)
                if reached:
                    self.state = "RUN"
                    self.state_started_s = now_s
                    self.run_started_s = now_s
                    self.next_policy_s = 0.0
                    self.get_logger().info("到达高度，开始 10 Hz policy / 20 Hz setpoint 诊断。")
                elif now_s - self.state_started_s > args.takeoff_timeout:
                    self.request_land("takeoff timeout", failed=True, termination="timeout")

            elif self.state == "RUN":
                self._publish_velocity_setpoints()
                if now_s >= self.next_policy_s:
                    observations = self._observations()
                    action = self.policy.act(observations, self.vehicles["uav_1"].initial_yaw)
                    self.step(action, action.message)
                    self.next_policy_s = now_s + 0.1
                self._log_status(now_s)
                if self.run_started_s is not None and now_s - self.run_started_s >= args.duration:
                    self.mission_completed = True
                    self.request_land("诊断时长完成")

            elif self.state == "LANDING":
                if self._all_disarmed():
                    self._finalize_episode()
                    if self.failure_reason:
                        self.get_logger().error(f"双机已落地；诊断失败：{self.failure_reason}")
                    else:
                        self.get_logger().info(f"双机已落地；episode 已写入 {args.output}")
                    return
                if now_s >= self.next_command_s:
                    self._send_land_commands()
                    self.next_command_s = now_s + 1.0
                if now_s - self.state_started_s > args.landing_timeout:
                    if self.failure_reason is None:
                        self.failure_reason = "landing timeout"
                        self.termination_reason = "timeout"
                    self._finalize_episode()
                    self.get_logger().error("降落超时，请立即在 QGroundControl 检查。")

        def episode_record(self) -> dict[str, Any]:
            success = bool(
                self.mission_completed and self.preposition_success and self.failure_reason is None
            )
            termination = self.termination_reason
            if self.failure_reason is None:
                termination = "success" if success else "timeout"
            camera = self.cameras["uav_0"]
            reset_payload = {
                "seed": args.seed,
                "spawn_offset_ned": list(spawn),
                "altitude": args.altitude,
                "preposition_distance": args.preposition_distance,
                "duration": args.duration,
            }
            reset_hash = hashlib.sha256(
                json.dumps(reset_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            communication_bytes = sum(
                int(step["communication"]["bytes_sent"]) for step in self.steps
            )
            flight_time = 0.0
            if self.run_started_s is not None:
                flight_time = max(0.0, time.monotonic() - self.run_started_s)
            minimum_separation = self.minimum_separation_m
            if not math.isfinite(minimum_separation):
                minimum_separation = math.hypot(spawn[0], spawn[1])
            preposition_error = self.preposition_error_m
            if not math.isfinite(preposition_error):
                preposition_error = float(args.preposition_distance)
            return {
                "schema_version": "0.1",
                "episode_id": self.episode_id,
                "task_id": "live_dual_control_observation_diagnostic",
                "split": "live_diagnostic",
                "seed": int(args.seed),
                "agent_ids": list(AGENT_IDS),
                "environment": {
                    "simulator": "px4_gazebo",
                    "version": "live-px4-sitl-gazebo-sim",
                    "map_id": args.world,
                    "reset_hash": reset_hash,
                    "ros_distro": "humble",
                    "vehicle_namespaces": list(AGENT_IDS),
                    "transport": "px4_uxrce_dds",
                    "control_hz": 10.0,
                    "setpoint_hz": 20.0,
                    "camera": {
                        "width": max(1, int(camera.width)),
                        "height": max(1, int(camera.height)),
                        "encoding": "raw_uint8",
                        "topic_template": CAMERA_TOPIC_TEMPLATE.format(
                            world=args.world, index="{index}"
                        ),
                    },
                },
                "steps": self.steps,
                "outcome": {
                    "terminated": True,
                    "termination_reason": termination,
                    "success": success,
                    "metrics": {
                        "control_success": float(self.mission_completed and self.failure_reason is None),
                        "pre_position_success": float(self.preposition_success),
                        "collision_rate": float(minimum_separation < self.limits.min_inter_vehicle_distance),
                        "minimum_separation_m": float(minimum_separation),
                        "preposition_error_m": float(preposition_error),
                        "communication_bytes": float(communication_bytes),
                        "flight_time_s": float(flight_time),
                        "camera_frames_uav_0": float(self.cameras["uav_0"].frame_count),
                        "camera_frames_uav_1": float(self.cameras["uav_1"].frame_count),
                    },
                },
            }

        def write_episode(self) -> None:
            if self.episode_written:
                return
            append_episode_record(self.episode_record(), args.output)
            self.episode_written = True

        def _finalize_episode(self) -> None:
            """Write the terminal record once, then stop the ROS spin loop."""
            if self.done:
                return
            try:
                self.write_episode()
            except Exception as exc:
                self.done = True
                self.get_logger().error(f"episode 校验或写入失败：{exc}")
                raise
            self.done = True

    rclpy.init()
    node = LiveBenchmarkNode()
    interrupted = False

    def _handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        node.request_land("收到 Ctrl+C", failed=True, termination="safety_abort")

    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception as exc:
        node.request_land(f"未处理异常：{exc}", failed=True)
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not node.done and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if not node.done:
            node.request_land("节点退出前的最终安全降落", failed=True)
            deadline = time.monotonic() + 3.0
            while rclpy.ok() and not node.done and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if interrupted:
        raise SystemExit(130)
    if node.failure_reason or not node.episode_record()["outcome"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
