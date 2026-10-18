"""Independent SITL pilot control, with no oracle route or target inputs.

Model delta commands feed the same body-FLU conversion and acceleration
limiter as the legacy runner. Telemetry is receive-time state, not verified
exposure-time state or exact actuation. Only the isolated search pilot uses
this module; no hardware or old runner is addressed.
"""

import math
import threading
import time

from ..contract import Action
from ..ros2_px4_adapter import (SafetyLimits, acceleration_limited_action, body_flu_to_ned,
                              ned_displacement_to_initial_body_flu)
from .annotation_session_smoke import INSTANCES
from .protocol import AGENT_IDS, ModelJointAction, VehicleState
from .search_pilot_scene import SPAWNS_NED

LIMITS = SafetyLimits(max_xy_speed=1., max_z_speed=.5, max_yaw_rate=.5,
                      max_distance_from_origin=6., max_altitude=4.)


def px4_topic(index, direction, name, message_type):
    """PX4 appends _vN for versioned DDS messages (v0 has no suffix)."""
    version = int(getattr(message_type, "MESSAGE_VERSION", 0))
    suffix = f"_v{version}" if version else ""
    return f"/annotation_smoke_{index}/fmu/{direction}/{name}{suffix}"


def control_qos():
    """Commands are live writes, not cached telemetry for late subscribers.

    Use the reliable/volatile profile of ros2_px4_adapter's depth-10 writers.
    A matched best-effort/transient-local writer did not deliver the no-arm
    probe in this Agent 2.4.2 / Fast DDS 2.6 setup. Matching alone must still
    never bypass the fresh two-vehicle command ACK gate.
    """
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def public_state(odometry, origin, initial_yaw):
    pos = tuple(float(odometry.position[k]) - origin[k] for k in range(3))
    yaw = yaw_of(odometry.q) - initial_yaw
    return VehicleState(ned_displacement_to_initial_body_flu(pos, initial_yaw),
                        ned_displacement_to_initial_body_flu(tuple(float(v) for v in odometry.velocity), initial_yaw),
                        math.atan2(math.sin(yaw), math.cos(yaw)), "ok")


def safety_problem(odometry, status, received, origins, now):
    """Declared local envelope; separation is an odometry-based estimate."""
    common = []
    for index in (0, 1):
        if now - received.get(("odometry", index), -math.inf) > .5 or now - received.get(("status", index), -math.inf) > 2.:
            return "simulator_error", "stale_telemetry"
        msg, state = odometry[index], status[index]
        if not all(math.isfinite(float(v)) for values in (msg.position, msg.velocity, msg.q) for v in values):
            return "simulator_error", "nonfinite_telemetry"
        if state.failsafe:
            return "simulator_error", "px4_failsafe"
        p = tuple(float(msg.position[k]) - origins[index][k] for k in range(3))
        if math.hypot(p[0], p[1]) > LIMITS.max_distance_from_origin or not -.5 <= -p[2] <= LIMITS.max_altitude:
            return "safety_abort", "local_geofence"
        common.append(tuple(p[k] + SPAWNS_NED[index][k] for k in range(3)))
    if math.dist(*common) < LIMITS.min_inter_vehicle_distance:
        return "safety_abort", "minimum_separation"
    return None


class SearchFlightControl:
    def __init__(self, ros, odometry, status, received, control_link, *, use_telemetry_timestamps=False):
        if not control_link.verified or not all(n == 1 for n in control_link.reader_counts().values()):
            raise RuntimeError("Verified control link required before creating flight controller")
        from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        self.types = OffboardControlMode, TrajectorySetpoint, VehicleCommand
        self.ros, self.odometry, self.status, self.received = ros, odometry, status, received
        self.control_link = control_link
        self.use_telemetry_timestamps = use_telemetry_timestamps
        self.origins = {i: tuple(float(v) for v in odometry[i].position) for i in (0, 1)}
        self.yaws = {i: yaw_of(odometry[i].q) for i in (0, 1)}
        self.phase, self.started, self.run_started = "PRESTREAM", time.monotonic(), None
        self.failure = None
        self.landing_started = None
        self.last_command = -math.inf
        self.last_policy = None
        self.requested = ModelJointAction()
        self.previous = {agent: Action() for agent in AGENT_IDS}
        self.lock = threading.Lock()
        self.publications, self.commands = 0, 0
        self.last_tick = self.started
        self.max_publication_gap_s = 0.
        self.trace = []
        self.command_trace = []
        self.state_trace = []
        self.last_state_log = -math.inf
        # Reuse the exact six writers checked by the gate; never verify a
        # temporary writer and then replace it just before takeoff.
        self.publishers = control_link.publishers
        self.timer = ros.create_timer(.05, self.tick)

    def stamp_us(self, index):
        if not self.use_telemetry_timestamps:
            return self.ros.get_clock().now().nanoseconds // 1000
        # In slow lockstep SITL, Agent wall-clock sync can jump relative to
        # PX4 HRT. ROS wall time then maps backwards in PX4, even while 20Hz
        # writes keep arriving. Round-trip the latest PX4-synchronized outgoing
        # odometry stamp instead. Never advance an old stamp to fake freshness;
        # the telemetry watchdog and PX4 Offboard-loss check remain active.
        stamp = int(self.odometry[index].timestamp)
        if stamp <= 0:
            raise ValueError("Positive PX4 telemetry timestamp required")
        return stamp

    def command(self, index, command, param1=0., param2=0.):
        message = self.types[2]()
        message.timestamp = self.stamp_us(index)
        message.command = command
        message.param1, message.param2 = float(param1), float(param2)
        message.target_system, message.target_component = INSTANCES[index] + 1, 1
        message.source_system, message.source_component = 255, 1
        message.from_external = True
        self.publishers[index][2].publish(message)
        self.commands += 1
        self.command_trace.append({"agent": f"uav_{index}", "command": command, "param1": param1,
                                   "param2": param2, "target_system": int(message.target_system),
                                   "monotonic_ns": time.monotonic_ns()})

    def land(self, failure=None):
        if failure is not None and self.failure is None:
            self.failure = failure
        if self.phase not in {"LANDING", "DONE"}:
            self.phase, self.landing_started = "LANDING", time.monotonic()
            self.last_command = -math.inf

    def request(self, action):
        if not isinstance(action, ModelJointAction):
            raise TypeError("ModelJointAction required")
        with self.lock:
            self.requested, self.last_policy = action, time.monotonic()

    def states(self):
        return {f"uav_{i}": public_state(self.odometry[i], self.origins[i], self.yaws[i]) for i in (0, 1)}

    def tick(self):
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        if self.phase == "DONE":
            return
        try:
            if now - self.last_state_log >= 2.:
                states = {str(i): {"arming_state": int(self.status[i].arming_state),
                                   "nav_state": int(self.status[i].nav_state),
                                   "altitude_m": self.origins[i][2] - float(self.odometry[i].position[2]),
                                   "pre_flight_checks_pass": bool(getattr(self.status[i], "pre_flight_checks_pass", False))}
                          for i in (0, 1)}
                self.state_trace.append({"monotonic_ns": time.monotonic_ns(), "phase": self.phase, "vehicles": states})
                print(f"flight phase={self.phase} vehicles={states}", flush=True)
                self.last_state_log = now
            problem = safety_problem(self.odometry, self.status, self.received, self.origins, now)
            if not all(n == 1 for n in self.control_link.reader_counts().values()):
                problem = ("simulator_error", "control_link_reader_lost")
            if problem and self.phase != "LANDING":
                self.land(problem)
            if self.phase == "LANDING":
                if all(self.status[i].arming_state == 1 for i in (0, 1)):
                    self.phase = "DONE"
                    return
                if now - self.landing_started > 50.:
                    self.failure = ("simulator_error", "landing_not_verified")
                    self.phase = "DONE"
                    return
                if now - self.last_command > 1.:
                    for i in (0, 1):
                        self.command(i, 21)  # VEHICLE_CMD_NAV_LAND, no forced disarm.
                    self.last_command = now
                return
            if self.phase in {"PRESTREAM", "TAKEOFF"}:
                if now - self.started > 55.:
                    self.land(("simulator_error", "takeoff_not_verified"))
                    return
                if now - self.started > 2. and now - self.last_command > 1.:
                    for i in (0, 1):
                        self.command(i, 176, 1., 6.)
                        if self.status[i].nav_state == 14:
                            self.command(i, 400, 1.)
                    self.last_command, self.phase = now, "TAKEOFF"
                ready = all(self.status[i].arming_state == 2 and self.status[i].nav_state == 14
                            and float(self.odometry[i].position[2]) < self.origins[i][2] - 1.7 for i in (0, 1))
                if ready:
                    self.phase, self.run_started = "RUN", now
            if self.phase == "RUN":
                if any(self.status[i].arming_state != 2 or self.status[i].nav_state != 14 for i in (0, 1)):
                    self.land(("simulator_error", "offboard_or_armed_state_lost"))
                    return
                with self.lock:
                    requested, last_policy = self.requested, self.last_policy
                if (last_policy is None and now - self.run_started > 2.) or (last_policy is not None and now - last_policy > 1.):
                    self.land(("simulator_error", "policy_command_watchdog"))
                    return
                if dt > .5:
                    self.land(("simulator_error", "setpoint_publication_stall"))
                    return
                self.max_publication_gap_s = max(self.max_publication_gap_s, dt)
            for i, agent in enumerate(AGENT_IDS):
                mode, point = self.types[0](), self.types[1]()
                mode.timestamp = point.timestamp = self.stamp_us(i)
                point.acceleration = point.jerk = [float("nan")] * 3
                if self.phase != "RUN":
                    mode.position = True
                    point.position = [*self.origins[i][:2], self.origins[i][2] - 2.]
                    point.velocity, point.yawspeed, point.yaw = [float("nan")] * 3, float("nan"), self.yaws[i]
                else:
                    mode.velocity = True
                    action = getattr(requested.to_velocity_action(.5), agent)
                    limited = acceleration_limited_action(action, self.previous[agent], min(dt, .1), LIMITS)
                    self.previous[agent] = limited
                    velocity = body_flu_to_ned(limited, yaw_of(self.odometry[i].q))
                    point.position, point.velocity = [float("nan")] * 3, list(velocity[:3])
                    point.yaw, point.yawspeed = float("nan"), velocity[3]
                    self.trace.append({"monotonic_ns": time.monotonic_ns(), "agent": agent,
                                       "requested_delta": getattr(requested, agent).vector(),
                                       "published_velocity_ned": list(velocity[:3]), "published_yawspeed_ned": velocity[3]})
                self.publishers[i][0].publish(mode)
                self.publishers[i][1].publish(point)
                self.publications += 1
        except Exception as exc:
            self.land(("simulator_error", f"control_exception:{type(exc).__name__}:{exc}"))
