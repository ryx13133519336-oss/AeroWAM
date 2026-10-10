"""Optional ROS 2 adapter for PX4 uXRCE-DDS.

The module imports ROS only inside ``main`` so fixture/evaluator tests work on
machines without ROS 2. It publishes the official PX4 setpoint topics under
``/px4_0`` and ``/px4_1`` at 20 Hz while policy decisions arrive at 10 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .contract import AGENT_IDS, Action, JointAction

PX4_NAMESPACES = {"uav_0": "px4_0", "uav_1": "px4_1"}
PX4_SYSTEM_IDS = {"uav_0": 1, "uav_1": 2}


@dataclass(frozen=True)
class SafetyLimits:
    min_inter_vehicle_distance: float = 1.5
    max_acceleration: float = 3.0
    max_xy_speed: float = 3.0
    max_z_speed: float = 1.5
    max_yaw_rate: float = 1.0
    max_distance_from_origin: float = 12.0
    max_altitude: float = 8.0


def safe_joint_action(action: JointAction, positions: dict[str, tuple[float, float, float]], limits: SafetyLimits = SafetyLimits()) -> JointAction:
    safe = action.clipped()
    a, b = positions["uav_0"], positions["uav_1"]
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    if distance < limits.min_inter_vehicle_distance:
        return JointAction(Action(), Action(), {"safety_abort": "minimum_separation"})
    return safe


def finite_action(action: Action) -> bool:
    """Return whether every command component is finite."""
    return all(math.isfinite(value) for value in action.as_vector())


def acceleration_limited_action(
    requested: Action,
    previous: Action,
    dt: float,
    limits: SafetyLimits = SafetyLimits(),
) -> Action:
    """Clip speed/yaw limits and limit translational acceleration."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not finite_action(requested):
        raise ValueError("requested action contains NaN or infinity")
    clipped = Action(
        vx=max(-limits.max_xy_speed, min(limits.max_xy_speed, requested.vx)),
        vy=max(-limits.max_xy_speed, min(limits.max_xy_speed, requested.vy)),
        vz=max(-limits.max_z_speed, min(limits.max_z_speed, requested.vz)),
        yaw_rate=max(-limits.max_yaw_rate, min(limits.max_yaw_rate, requested.yaw_rate)),
    )
    max_delta = limits.max_acceleration * dt

    def limited(value: float, old: float) -> float:
        return max(old - max_delta, min(old + max_delta, value))

    return Action(
        limited(clipped.vx, previous.vx),
        limited(clipped.vy, previous.vy),
        limited(clipped.vz, previous.vz),
        clipped.yaw_rate,
    )


def body_flu_to_ned(action: Action, yaw_ned: float = 0.0) -> tuple[float, float, float, float]:
    """Rotate body-FLU horizontal velocity to local NED and convert signs."""
    north = action.vx * math.cos(yaw_ned) + action.vy * math.sin(yaw_ned)
    east = action.vx * math.sin(yaw_ned) - action.vy * math.cos(yaw_ned)
    return (north, east, -action.vz, -action.yaw_rate)


def ned_displacement_to_initial_body_flu(
    displacement_ned: tuple[float, float, float], initial_yaw_ned: float
) -> tuple[float, float, float]:
    """Express local-NED displacement in the vehicle's initial body-FLU frame."""
    north, east, down = displacement_ned
    forward = north * math.cos(initial_yaw_ned) + east * math.sin(initial_yaw_ned)
    left = north * math.sin(initial_yaw_ned) - east * math.cos(initial_yaw_ned)
    return (forward, left, -down)


def common_ned_position(
    local_position: tuple[float, float, float],
    local_origin: tuple[float, float, float],
    spawn_offset_ned: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Approximate a shared NED position from a declared Gazebo spawn offset."""
    return tuple(
        float(spawn_offset_ned[index]) + float(local_position[index]) - float(local_origin[index])
        for index in range(3)
    )


def distance_3d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def geofence_problem(
    relative_position_flu: tuple[float, float, float],
    limits: SafetyLimits = SafetyLimits(),
) -> str | None:
    """Return a diagnostic string when the local flight envelope is violated."""
    x, y, z = relative_position_flu
    horizontal = math.hypot(x, y)
    if horizontal > limits.max_distance_from_origin:
        return f"horizontal geofence exceeded: {horizontal:.2f} m"
    if z < -0.5:
        return f"invalid below-origin altitude: {z:.2f} m"
    if z > limits.max_altitude:
        return f"altitude geofence exceeded: {z:.2f} m"
    return None


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
    except ImportError as exc:
        raise SystemExit("ROS 2 Humble with px4_msgs is required for the live adapter; fixture mode does not need ROS") from exc

    class PX4DualSetpointNode(Node):
        def __init__(self) -> None:
            super().__init__("aerowam_px4_dual_setpoint")
            self.publishers: dict[str, dict[str, Any]] = {}
            for agent_id in AGENT_IDS:
                prefix = f"/{PX4_NAMESPACES[agent_id]}/fmu/in"
                self.publishers[agent_id] = {
                    "mode": self.create_publisher(OffboardControlMode, f"{prefix}/offboard_control_mode", 10),
                    "setpoint": self.create_publisher(TrajectorySetpoint, f"{prefix}/trajectory_setpoint", 10),
                    "command": self.create_publisher(VehicleCommand, f"{prefix}/vehicle_command", 10),
                }
            self.actions = {agent_id: Action() for agent_id in AGENT_IDS}
            self.create_timer(0.05, self.publish_setpoints)

        def publish_setpoints(self) -> None:
            for agent_id in AGENT_IDS:
                action = self.actions[agent_id]
                mode = OffboardControlMode()
                mode.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                mode.velocity = True
                setpoint = TrajectorySetpoint()
                setpoint.timestamp = mode.timestamp
                setpoint.position = [float("nan")] * 3
                setpoint.velocity = list(body_flu_to_ned(action)[:3])
                setpoint.acceleration = [float("nan")] * 3
                setpoint.jerk = [float("nan")] * 3
                setpoint.yaw = float("nan")
                setpoint.yawspeed = body_flu_to_ned(action)[3]
                self.publishers[agent_id]["mode"].publish(mode)
                self.publishers[agent_id]["setpoint"].publish(setpoint)

    rclpy.init()
    node = PX4DualSetpointNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
