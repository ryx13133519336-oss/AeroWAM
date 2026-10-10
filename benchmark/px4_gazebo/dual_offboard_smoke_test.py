"""Two-vehicle PX4 ROS 2 Offboard takeoff/hover/land smoke test.

The test deliberately uses each vehicle's first local-NED odometry sample as
its own origin.  It never assumes that ``px4_0`` and ``px4_1`` share a global
coordinate frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import signal
import time
from typing import Any

from .ros2_px4_adapter import PX4_NAMESPACES, PX4_SYSTEM_IDS


AGENT_IDS = ("uav_0", "uav_1")


@dataclass
class VehicleRuntime:
    namespace: str
    system_id: int
    odometry: Any | None = None
    status: Any | None = None
    last_odometry_s: float = 0.0
    last_status_s: float = 0.0
    origin: tuple[float, float, float] | None = None
    target: tuple[float, float, float] | None = None
    hold_yaw: float = 0.0
    reached_target: bool = False


def _finite_vector(values: Any, size: int) -> bool:
    try:
        return len(values) == size and all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _yaw_from_wxyz(quaternion: Any) -> float:
    """Return NED yaw from a PX4 VehicleOdometry [w, x, y, z] quaternion."""
    if not _finite_vector(quaternion, 4):
        return 0.0
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PX4-Gazebo 双机 Offboard 自动起飞、悬停和降落冒烟测试"
    )
    parser.add_argument("--altitude", type=float, default=3.0, help="相对各自初始位置的起飞高度（米）")
    parser.add_argument("--countdown", type=float, default=5.0, help="开始预发布前的安全倒计时（秒）")
    parser.add_argument("--prestream", type=float, default=2.0, help="进入 Offboard 前预发布 setpoint 的时长（秒）")
    parser.add_argument("--hover", type=float, default=10.0, help="两机到达目标高度后的悬停时长（秒）")
    parser.add_argument("--telemetry-timeout", type=float, default=2.0, help="里程计或状态遥测超时（秒）")
    parser.add_argument("--takeoff-timeout", type=float, default=30.0, help="起飞阶段最大时长（秒）")
    parser.add_argument("--landing-timeout", type=float, default=45.0, help="自动降落最大等待时长（秒）")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.altitude <= 0.0:
        raise SystemExit("--altitude 必须大于 0")

    try:
        import rclpy
        from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        from px4_msgs.msg import VehicleOdometry, VehicleStatus
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise SystemExit(
            "缺少 ROS 2 Humble 或 px4_msgs。请先 source /opt/ros/humble/setup.bash "
            "和 ~/px4_ros2_ws/install/setup.bash"
        ) from exc

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    class DualOffboardSmokeTest(Node):
        def __init__(self) -> None:
            super().__init__("aerowam_dual_offboard_smoke_test")
            self.vehicles = {
                agent_id: VehicleRuntime(PX4_NAMESPACES[agent_id], PX4_SYSTEM_IDS[agent_id])
                for agent_id in AGENT_IDS
            }
            self.publishers_by_agent: dict[str, dict[str, Any]] = {}
            self.subscriptions_by_agent: dict[str, list[Any]] = {}
            for agent_id, vehicle in self.vehicles.items():
                input_prefix = f"/{vehicle.namespace}/fmu/in"
                output_prefix = f"/{vehicle.namespace}/fmu/out"
                self.publishers_by_agent[agent_id] = {
                    "mode": self.create_publisher(
                        OffboardControlMode, f"{input_prefix}/offboard_control_mode", qos
                    ),
                    "setpoint": self.create_publisher(
                        TrajectorySetpoint, f"{input_prefix}/trajectory_setpoint", qos
                    ),
                    "command": self.create_publisher(
                        VehicleCommand, f"{input_prefix}/vehicle_command", qos
                    ),
                }
                self.subscriptions_by_agent[agent_id] = [
                    self.create_subscription(
                        VehicleOdometry,
                        f"{output_prefix}/vehicle_odometry",
                        lambda message, key=agent_id: self._on_odometry(key, message),
                        qos,
                    ),
                    self.create_subscription(
                        VehicleStatus,
                        f"{output_prefix}/vehicle_status_v4",
                        lambda message, key=agent_id: self._on_status(key, message),
                        qos,
                    ),
                ]

            self.armed_state = int(getattr(VehicleStatus, "ARMING_STATE_ARMED", 2))
            self.offboard_state = int(getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14))
            self.state = "WAIT_TELEMETRY"
            self.state_started_s = time.monotonic()
            self.next_command_s = 0.0
            self.next_log_s = 0.0
            self.hover_started_s: float | None = None
            self.mission_completed = False
            self.done = False
            self.failure_reason: str | None = None
            self._waiting_message_printed = False
            self._last_countdown_value: int | None = None
            self.create_timer(0.05, self._tick)

            self.get_logger().info("双机冒烟测试节点已启动，等待 px4_0 和 px4_1 遥测。")

        def _now_us(self) -> int:
            return int(self.get_clock().now().nanoseconds / 1000)

        def _on_odometry(self, agent_id: str, message: Any) -> None:
            if not _finite_vector(message.position, 3):
                self.request_land(f"{agent_id} 收到非有限里程计位置", failed=True)
                return
            vehicle = self.vehicles[agent_id]
            vehicle.odometry = message
            vehicle.last_odometry_s = time.monotonic()
            if vehicle.origin is None:
                vehicle.origin = tuple(float(value) for value in message.position)
                vehicle.target = (
                    vehicle.origin[0],
                    vehicle.origin[1],
                    vehicle.origin[2] - float(args.altitude),
                )
                vehicle.hold_yaw = _yaw_from_wxyz(message.q)
                self.get_logger().info(
                    f"{agent_id} 本地 NED 原点={vehicle.origin}，目标={vehicle.target}"
                )

        def _on_status(self, agent_id: str, message: Any) -> None:
            vehicle = self.vehicles[agent_id]
            vehicle.status = message
            vehicle.last_status_s = time.monotonic()

        def _all_telemetry_ready(self) -> bool:
            return all(
                vehicle.origin is not None and vehicle.odometry is not None and vehicle.status is not None
                for vehicle in self.vehicles.values()
            )

        def _all_disarmed(self) -> bool:
            return all(
                vehicle.status is not None
                and int(vehicle.status.arming_state) != self.armed_state
                for vehicle in self.vehicles.values()
            )

        def _all_armed_offboard(self) -> bool:
            return all(
                vehicle.status is not None
                and int(vehicle.status.arming_state) == self.armed_state
                and int(vehicle.status.nav_state) == self.offboard_state
                for vehicle in self.vehicles.values()
            )

        def _validate_preflight(self) -> str | None:
            for agent_id, vehicle in self.vehicles.items():
                status = vehicle.status
                if status is None:
                    return f"{agent_id} 没有 vehicle_status_v4"
                if bool(status.failsafe):
                    return f"{agent_id} 已进入 failsafe"
                if not bool(status.pre_flight_checks_pass):
                    return f"{agent_id} 未通过 pre-flight checks"
            return None

        def _telemetry_problem(self, now_s: float) -> str | None:
            for agent_id, vehicle in self.vehicles.items():
                if now_s - vehicle.last_odometry_s > args.telemetry_timeout:
                    return f"{agent_id} vehicle_odometry 超时"
                if now_s - vehicle.last_status_s > args.telemetry_timeout:
                    return f"{agent_id} vehicle_status_v4 超时"
                if vehicle.odometry is None or not _finite_vector(vehicle.odometry.position, 3):
                    return f"{agent_id} vehicle_odometry 无效"
            return None

        def _publish_position_setpoints(self, airborne_target: bool) -> None:
            for agent_id, vehicle in self.vehicles.items():
                if vehicle.origin is None or vehicle.target is None:
                    continue
                desired = vehicle.target if airborne_target else vehicle.origin
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
                setpoint.yaw = float(vehicle.hold_yaw)
                setpoint.yawspeed = float("nan")

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

        def request_land(self, reason: str, *, failed: bool = False) -> None:
            if self.done:
                return
            if failed and self.failure_reason is None:
                self.failure_reason = reason
            if self.state != "LANDING":
                level = self.get_logger().error if failed else self.get_logger().info
                level(f"进入安全降落流程：{reason}")
                self.state = "LANDING"
                self.state_started_s = time.monotonic()
                self.next_command_s = 0.0
                self._send_land_commands()

        def _log_positions(self, now_s: float) -> None:
            if now_s < self.next_log_s:
                return
            self.next_log_s = now_s + 1.0
            details = []
            for agent_id, vehicle in self.vehicles.items():
                if vehicle.odometry is not None:
                    x, y, z = (float(value) for value in vehicle.odometry.position)
                    details.append(f"{agent_id}=({x:.2f}, {y:.2f}, {z:.2f})")
            if details:
                self.get_logger().info(f"状态={self.state}；" + "；".join(details))

        def _tick(self) -> None:
            if self.done:
                return
            now_s = time.monotonic()

            if self.state == "WAIT_TELEMETRY":
                if not self._all_telemetry_ready():
                    if not self._waiting_message_printed:
                        self.get_logger().info("等待双机 vehicle_odometry 和 vehicle_status_v4 ...")
                        self._waiting_message_printed = True
                    return
                problem = self._validate_preflight()
                if problem:
                    self.request_land(problem, failed=True)
                    return
                self.state = "COUNTDOWN"
                self.state_started_s = now_s
                self.get_logger().info("双机遥测和起飞前检查正常。请保持人员远离旋翼区域。")

            if self.state not in {"WAIT_TELEMETRY", "LANDING"}:
                problem = self._telemetry_problem(now_s)
                if problem:
                    self.request_land(problem, failed=True)
                    return
                problem = self._validate_preflight()
                if problem:
                    self.request_land(problem, failed=True)
                    return

            if self.state == "COUNTDOWN":
                self._publish_position_setpoints(airborne_target=False)
                remaining = max(0, math.ceil(args.countdown - (now_s - self.state_started_s)))
                if remaining != self._last_countdown_value:
                    self._last_countdown_value = remaining
                    self.get_logger().info(f"自动起飞倒计时：{remaining}")
                if now_s - self.state_started_s >= args.countdown:
                    self.state = "PRESTREAM"
                    self.state_started_s = now_s
                    self.get_logger().info("开始以 20 Hz 预发布 Offboard setpoint。")

            elif self.state == "PRESTREAM":
                self._publish_position_setpoints(airborne_target=False)
                if now_s - self.state_started_s >= args.prestream:
                    self.state = "REQUEST_OFFBOARD"
                    self.state_started_s = now_s
                    self.next_command_s = 0.0
                    self.get_logger().info("请求两架飞机进入 Offboard 并解锁。")

            elif self.state == "REQUEST_OFFBOARD":
                self._publish_position_setpoints(airborne_target=False)
                if now_s >= self.next_command_s:
                    self._request_offboard_and_arm()
                    self.next_command_s = now_s + 1.0
                if self._all_armed_offboard():
                    self.state = "TAKEOFF"
                    self.state_started_s = now_s
                    self.get_logger().info(f"两机已进入 Offboard 并解锁，起飞至各自相对高度 {args.altitude:.1f} m。")
                elif now_s - self.state_started_s > 12.0:
                    self.request_land("12 秒内未能让两机同时进入 Offboard 并解锁", failed=True)

            elif self.state == "TAKEOFF":
                self._publish_position_setpoints(airborne_target=True)
                all_reached = True
                for vehicle in self.vehicles.values():
                    assert vehicle.odometry is not None and vehicle.target is not None
                    error = abs(float(vehicle.odometry.position[2]) - vehicle.target[2])
                    vehicle.reached_target = error <= 0.35
                    all_reached = all_reached and vehicle.reached_target
                self._log_positions(now_s)
                if all_reached:
                    self.state = "HOVER"
                    self.state_started_s = now_s
                    self.hover_started_s = now_s
                    self.get_logger().info(f"两机到达目标高度，开始悬停 {args.hover:.1f} 秒。")
                elif now_s - self.state_started_s > args.takeoff_timeout:
                    self.request_land("起飞超时，未能同时到达目标高度", failed=True)

            elif self.state == "HOVER":
                self._publish_position_setpoints(airborne_target=True)
                self._log_positions(now_s)
                if now_s - self.state_started_s >= args.hover:
                    self.mission_completed = True
                    self.request_land("悬停测试完成")

            elif self.state == "LANDING":
                if self._all_disarmed():
                    self.done = True
                    if self.failure_reason:
                        self.get_logger().error(f"双机已落地并锁定；测试失败：{self.failure_reason}")
                    else:
                        self.get_logger().info("双机已落地并锁定；冒烟测试成功完成。")
                    return
                if now_s >= self.next_command_s:
                    self._send_land_commands()
                    self.next_command_s = now_s + 1.0
                if now_s - self.state_started_s > args.landing_timeout:
                    if self.failure_reason is None:
                        self.failure_reason = "等待自动降落并锁定超时"
                    self.get_logger().error(
                        "降落等待超时；NAV_LAND 已发送，请立即在 QGroundControl 检查两机状态。"
                    )
                    self.done = True

    rclpy.init()
    node = DualOffboardSmokeTest()
    interrupted = False

    def _handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        node.request_land("收到 Ctrl+C", failed=True)

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
    if node.failure_reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
