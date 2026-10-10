"""ROS-free regression checks for the live PX4/Gazebo Handoff runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

from .communication import CommunicationChannel
from .contract import AGENT_IDS, Action, AgentObservation, HandoffTask, JointAction
from .evaluator import episode_metrics, replay_metric_differences
from .gazebo_scene import task_flu_to_gazebo_enu
from .live_handoff_batch import _parser as batch_parser, _runner_kwargs as batch_runner_kwargs, runner_command as batch_runner_command
from .live_handoff_runner import (
    CameraRuntime,
    LiveHandoffProtocol,
    OBSTACLE_CLEARANCE_M,
    TARGET_CLEARANCE_M,
    build_episode_record,
    build_task,
    evaluation_positions_in_uav0_flu,
    validate_episode_record,
    validate_runtime_arguments,
)
from .model_handoff.fixtures import generate_fixtures
from .occlusion_geometry import (
    distance,
    point_aabb_distance,
    segment_intersects_aabb,
    target_visible,
    target_at_step,
    visibility_truth,
)
from .policies import (
    HandoffRulePolicy,
    _uav0_tracking_position,
    _uav1_tracking_position,
    OracleHandoffPolicy,
    WAYPOINT_SETTLE_STEPS,
    _uav1_clearance_position,
    handoff_demo_waypoints,
)
from .ros2_px4_adapter import SafetyLimits, acceleration_limited_action


def _runtime_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "policy": "rule",
        "bandwidth": 256,
        "latency_ms": 100.0,
        "drop_probability": 0.0,
        "altitude": 2.0,
        "duration": 3.0,
        "seed": 0,
        "world": "lawn",
        "countdown": 0.0,
        "prestream": 0.0,
        "telemetry_timeout": 2.0,
        "camera_timeout": 2.0,
        "startup_timeout": 30.0,
        "takeoff_timeout": 30.0,
        "landing_timeout": 45.0,
        "uav1_spawn_offset_ned": (0.0, 3.0, 0.0),
        "output": Path("unused.jsonl"),
        "max_policy_timeouts": 3,
        "policy_timeout_ms": 400.0,
        "oracle_perturb_probability": 0.15,
        "fixture_manifest": None,
        "fixture_id": None,
        "formal_model": False,
        "visual_handoff": False,
        "model_method": None,
        "training_seed": None,
        "hide_markers": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _expect_invalid(**overrides: object) -> None:
    try:
        validate_runtime_arguments(_runtime_args(**overrides))
    except ValueError:
        return
    raise AssertionError(f"expected invalid runtime arguments: {overrides}")


def _task(**overrides: object) -> HandoffTask:
    values: dict[str, object] = {
        "horizon": 8,
        "control_hz": 10.0,
        "target_start": (-3.0, 0.0, 0.0),
        "target_end": (3.0, 0.0, 0.0),
        "occluder_center": (0.0, 0.0, 0.0),
        "occluder_half_extents": (0.5, 0.5, 0.5),
        "visibility_range": 20.0,
        "initial_positions": {
            "uav_0": (-4.0, 0.0, 0.0),
            "uav_1": (4.0, 4.0, 0.0),
        },
        "staging_position": (4.0, 4.0, 0.0),
        "staging_tolerance": 0.1,
        "collision_distance": 0.5,
        "interruption_limit_steps": 8,
        "required_identity_frames": 1,
        "handoff_request_lead_steps": 1,
        "post_handoff_tracking_steps": 2,
    }
    values.update(overrides)
    return replace(HandoffTask(), **values)


def _observations(
    positions: dict[str, tuple[float, float, float]],
) -> dict[str, AgentObservation]:
    return {
        agent_id: AgentObservation(
            agent_id=agent_id,
            rgb_uri=f"ros2://camera/{agent_id}",
            own_position=position,
            own_velocity=(0.0, 0.0, 0.0),
            own_yaw=0.0,
        )
        for agent_id, position in positions.items()
    }


def _protocol(task: HandoffTask, **overrides: object) -> LiveHandoffProtocol:
    values: dict[str, object] = {
        "seed": 7,
        "bandwidth": 4096,
        "latency_ms": 100.0,
        "drop_probability": 0.0,
    }
    values.update(overrides)
    return LiveHandoffProtocol(task, **values)


def _cameras() -> dict[str, CameraRuntime]:
    return {
        "uav_0": CameraRuntime("/camera/0", frame_count=17, stamp_ns=100),
        "uav_1": CameraRuntime("/camera/1", frame_count=19, stamp_ns=200),
    }


def _success_protocol() -> LiveHandoffProtocol:
    task = _task()
    positions = dict(task.initial_positions)
    observations = _observations(positions)
    protocol = _protocol(task)
    for step in range(task.horizon):
        message = None
        if step == 0:
            message = {
                "sender": "uav_0",
                "receiver": "uav_1",
                "payload": {"intent": "handoff_request", "target_identity": "target_0"},
            }
        result = protocol.step(
            observations,
            positions,
            JointAction(message=message),
        )
        if step == 0:
            assert any(event["event_type"] == "handoff_request" for event in result["events"])
            assert not any(event["event_type"] == "handoff_delivery" for event in result["events"])
        if step == 1:
            deliveries = [
                event for event in result["events"]
                if event["event_type"] == "handoff_delivery"
            ]
            assert deliveries and deliveries[0]["attributes"]["latency_steps"] == 1
        if result["done"]:
            break
    assert protocol.termination_reason == "success", {
        "termination": protocol.termination_reason,
        "events": [(event["end_t"], event["event_type"]) for event in protocol.events],
        "longest_interruption": protocol._longest_interruption(),
        "invisible_steps": [
            index for index, visible in enumerate(protocol.visible_history) if not visible
        ],
        "positions": positions,
    }
    assert protocol.handoff_request_t == 0
    assert protocol.handoff_delivery_t == 1
    assert protocol.recovery_t is not None
    assert protocol.handoff_t is not None
    assert protocol.verify_t is not None
    assert {event["event_type"] for event in protocol.events} >= {
        "pre_position", "occlusion_start", "target_recovery", "handoff", "verify"
    }
    return protocol


def _visual_rule_protocol(task: HandoffTask) -> LiveHandoffProtocol:
    positions = {
        "uav_0": (
            task.occluder_center[0] - task.occluder_half_extents[0] - 0.8,
            task.occluder_center[1] + task.occluder_half_extents[1] + 0.8,
            task.target_start[2],
        ),
        "uav_1": task.staging_position,
    }
    initial_target = target_at_step(
        task.target_start, task.target_end, 0, task.horizon
    )
    assert visibility_truth(
        positions,
        initial_target,
        occluder_center=task.occluder_center,
        occluder_half_extents=task.occluder_half_extents,
        max_range=task.visibility_range,
    )["uav_0"]

    protocol = _protocol(task)
    policy = HandoffRulePolicy()
    policy.reset(task)
    # This protocol-only fixture starts with both acquisition and staging done.
    # The separate moving test and waypoint-order test exercise the full route.
    policy.uav0_tracking_ready = True
    policy.uav1_route_phase = "ready"
    saw_occluded_recovery = False
    saw_active_tracker_motion = False
    while not protocol.done:
        observations = _observations(positions)
        action = policy.act(protocol.policy_observation(observations))
        if protocol.handoff_delivery_t is not None:
            saw_active_tracker_motion |= (
                abs(action.uav_1.vx) + abs(action.uav_1.vy) + abs(action.uav_1.vz)
            ) > 0.1
        target = target_at_step(
            task.target_start,
            task.target_end,
            protocol.t,
            task.horizon,
        )
        visible = visibility_truth(
            positions,
            target,
            occluder_center=task.occluder_center,
            occluder_half_extents=task.occluder_half_extents,
            max_range=task.visibility_range,
        )
        saw_occluded_recovery |= not visible["uav_0"] and visible["uav_1"]
        protocol.step(observations, positions, action)

    assert saw_occluded_recovery
    assert saw_active_tracker_motion
    assert protocol.termination_reason == "success"
    assert protocol.t - 1 - protocol.verify_t >= task.post_handoff_tracking_steps
    assert {event["event_type"] for event in protocol.events} >= {
        "pre_position",
        "occlusion_start",
        "target_recovery",
        "handoff",
        "verify",
    }
    return protocol


def _moving_oracle_protocol(
    task: HandoffTask, *, policy_type: type = OracleHandoffPolicy,
    motion_scale: float = 1.0,
) -> LiveHandoffProtocol:
    """Exercise the real two-vehicle trajectory instead of static truth poses."""
    positions = {
        "uav_0": (0.0, 0.0, task.target_start[2]),
        "uav_1": (3.0, 0.35, task.target_start[2]),
    }
    velocities = {agent_id: Action() for agent_id in AGENT_IDS}
    protocol = _protocol(task, latency_ms=0.0)
    policy = policy_type()
    policy.reset(task)
    start_positions = dict(positions)
    minimum_target_clearance = math.inf
    minimum_wall_clearance = math.inf
    minimum_separation = math.inf
    maximum_separation = 0.0
    dt = 1.0 / task.control_hz
    while not protocol.done:
        observations = _observations(positions)
        requested = policy.act(protocol.policy_observation(observations))
        limited = {
            agent_id: acceleration_limited_action(
                getattr(requested, agent_id), velocities[agent_id], dt, SafetyLimits()
            )
            for agent_id in AGENT_IDS
        }
        action = JointAction(limited["uav_0"], limited["uav_1"], requested.message)
        target = target_at_step(
            task.target_start, task.target_end, protocol.t, task.horizon
        )
        minimum_target_clearance = min(
            minimum_target_clearance,
            *(distance(positions[agent_id], target) for agent_id in AGENT_IDS),
        )
        minimum_wall_clearance = min(
            minimum_wall_clearance,
            *(
                point_aabb_distance(
                    positions[agent_id],
                    task.occluder_center,
                    task.occluder_half_extents,
                )
                for agent_id in AGENT_IDS
            ),
        )
        separation = distance(positions["uav_0"], positions["uav_1"])
        minimum_separation = min(minimum_separation, separation)
        maximum_separation = max(maximum_separation, separation)
        protocol.step(observations, positions, action)
        for agent_id in AGENT_IDS:
            velocities[agent_id] = limited[agent_id]
            positions[agent_id] = tuple(
                positions[agent_id][axis]
                + getattr(limited[agent_id], ("vx", "vy", "vz")[axis]) * dt * motion_scale
                for axis in range(3)
            )
    assert protocol.termination_reason == "success", {
        "termination": protocol.termination_reason,
        "events": [(event["end_t"], event["event_type"]) for event in protocol.events],
        "longest_interruption": protocol._longest_interruption(),
        "invisible_steps": [
            index for index, visible in enumerate(protocol.visible_history) if not visible
        ],
        "positions": positions,
    }
    assert protocol._longest_interruption() <= task.interruption_limit_steps
    assert any(
        not step["truth"]["target_visibility"]["uav_0"]
        and segment_intersects_aabb(
            step["truth"]["agent_positions"]["uav_0"],
            step["truth"]["target_pose"],
            task.occluder_center,
            task.occluder_half_extents,
        )
        for step in protocol.steps
    ), {"task": task, "events": protocol.events}
    assert protocol.path_lengths["uav_0"] > 0.6, {
        "path_lengths": protocol.path_lengths,
        "final_positions": positions,
        "task": task,
    }
    assert distance(start_positions["uav_1"], positions["uav_1"]) > 1.0
    assert minimum_target_clearance >= TARGET_CLEARANCE_M
    assert minimum_wall_clearance >= OBSTACLE_CLEARANCE_M
    assert minimum_separation >= task.collision_distance
    assert maximum_separation <= 6.1, {
        "minimum_separation": minimum_separation,
        "maximum_separation": maximum_separation,
        "minimum_target_clearance": minimum_target_clearance,
        "minimum_wall_clearance": minimum_wall_clearance,
        "path_lengths": protocol.path_lengths,
        "final_positions": positions,
    }
    return protocol


def _assert_uav1_waypoint_order(task: HandoffTask) -> None:
    policy = OracleHandoffPolicy()
    policy.reset(task)
    waypoints = handoff_demo_waypoints(task)
    positions = {
        "uav_0": waypoints["uav0_observation"],
        "uav_1": task.staging_position,
    }
    protocol = _protocol(task, latency_ms=0.0)
    for _ in range(WAYPOINT_SETTLE_STEPS + 2):
        policy.act(protocol.policy_observation(_observations(positions)))
    assert policy.uav1_route_phase == "clearance"

    positions["uav_1"] = _uav1_clearance_position(task)
    for _ in range(WAYPOINT_SETTLE_STEPS):
        policy.act(protocol.policy_observation(_observations(positions)))
    assert policy.uav1_route_phase == "staging"

    positions["uav_1"] = task.staging_position
    for _ in range(WAYPOINT_SETTLE_STEPS):
        policy.act(protocol.policy_observation(_observations(positions)))
    assert policy.uav1_route_phase == "ready"


def _assert_route_geometry(task: HandoffTask) -> None:
    waypoints = handoff_demo_waypoints(task)
    uav1_start = (3.0, 0.35, task.staging_position[2])
    routes = (
        ((0.0, 0.0, task.staging_position[2]), waypoints["uav0_acquisition"]),
        (waypoints["uav0_acquisition"], waypoints["uav0_observation"]),
        (waypoints["uav0_observation"], waypoints["uav0_support"]),
        (uav1_start, waypoints["uav1_clearance"]),
        (waypoints["uav1_clearance"], waypoints["uav1_staging"]),
        (waypoints["uav1_staging"], waypoints["handoff"]),
    )
    inflated_half_extents = tuple(
        value + OBSTACLE_CLEARANCE_M for value in task.occluder_half_extents
    )
    for start, end in routes:
        assert not segment_intersects_aabb(
            start,
            end,
            task.occluder_center,
            inflated_half_extents,
        ), {"task": task, "start": start, "end": end}
    for waypoint in waypoints.values():
        assert point_aabb_distance(
            waypoint, task.occluder_center, task.occluder_half_extents
        ) >= OBSTACLE_CLEARANCE_M
    assert distance(
        waypoints["uav0_observation"], waypoints["uav1_staging"]
    ) <= 6.0


def main() -> None:
    headless_demo = batch_parser().parse_args(
        ["run-auto", "--visual-handoff", "--no-gui"]
    )
    assert headless_demo.visual_handoff and headless_demo.no_gui
    assert "--visual-handoff" in batch_runner_command(
        seed=0, output=Path("unused.jsonl"), **batch_runner_kwargs(headless_demo)
    )
    assert validate_runtime_arguments(_runtime_args()) == (0.0, 3.0, 0.0)
    _expect_invalid(bandwidth=-1)
    _expect_invalid(drop_probability=1.01)
    _expect_invalid(altitude=100.0)
    _expect_invalid(uav1_spawn_offset_ned=(0.0, 0.5, 0.0))

    built = build_task(_runtime_args(altitude=2.5, duration=3.0))
    assert built.horizon == 30
    assert built.target_start == (-2.0, -1.6, 2.5)
    assert built.target_end == (8.0, -1.6, 2.5)
    assert built.occluder_center == (2.0, -0.2, 2.5)
    assert built.occluder_half_extents == (0.25, 0.9, 2.0)
    # The physical wall crosses UAV0's forward flight lane; removing it
    # would open both the forward route and the target sight line.
    assert segment_intersects_aabb(
        (0.0, 0.6, 2.5), (4.0, 0.6, 2.5),
        built.occluder_center, built.occluder_half_extents,
    )
    # The target never crosses the physical wall, but the wall blocks the
    # near-side camera after the target passes it.
    before_wall = target_at_step(built.target_start, built.target_end, 8, built.horizon)
    after_wall = target_at_step(built.target_start, built.target_end, 20, built.horizon)
    assert built.occluder_center[1] - built.occluder_half_extents[1] - 0.34 > built.target_start[1]
    assert target_visible(
        _uav0_tracking_position(built, 8), before_wall,
        occluder_center=built.occluder_center,
        occluder_half_extents=built.occluder_half_extents,
        max_range=built.visibility_range,
    )
    assert not target_visible(
        _uav0_tracking_position(built, 20), after_wall,
        occluder_center=built.occluder_center,
        occluder_half_extents=built.occluder_half_extents,
        max_range=built.visibility_range,
    )
    assert target_visible(
        _uav1_tracking_position(built, 20), after_wall,
        occluder_center=built.occluder_center,
        occluder_half_extents=built.occluder_half_extents,
        max_range=built.visibility_range,
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(built.staging_position, (4.0, -3.1, 2.5))
    )
    _assert_uav1_waypoint_order(
        build_task(_runtime_args(altitude=2.5, duration=24.0))
    )
    obstacle_task = build_task(_runtime_args(altitude=2.0, duration=24.0))
    _assert_route_geometry(obstacle_task)
    for policy_type in (OracleHandoffPolicy, HandoffRulePolicy):
        obstacle_protocol = _moving_oracle_protocol(
            obstacle_task, policy_type=policy_type,
        )
        # Block the whole target body for a sustained interval, rather than
        # counting a grazing ray to its center as a convincing occlusion.
        fully_occluded_steps = 0
        for step in obstacle_protocol.steps:
            truth = step["truth"]
            target = truth["target_pose"]
            if truth["target_visibility"]["uav_1"] and all(
                segment_intersects_aabb(
                    truth["agent_positions"]["uav_0"],
                    (target[0] + dx, target[1] + dy, target[2] + dz),
                    obstacle_task.occluder_center,
                    obstacle_task.occluder_half_extents,
                )
                for dx in (-0.34, 0.34)
                for dy in (-0.34, 0.34)
                for dz in (-0.34, 0.34)
            ):
                fully_occluded_steps += 1
        assert fully_occluded_steps >= 2 * obstacle_task.control_hz, {
            "policy": policy_type.__name__, "fully_occluded_steps": fully_occluded_steps,
        }
        # The first layout passed ideal kinematics but approached the target
        # during the lateral crossing at the slower response seen in SITL.
        for motion_scale in (0.4, 0.45, 0.5, 0.6, 0.8):
            _moving_oracle_protocol(
                obstacle_task, policy_type=policy_type, motion_scale=motion_scale,
            )
    unsafe_crossing = replace(obstacle_task, staging_position=(3.45, -3.1, 2.0))
    try:
        _moving_oracle_protocol(
            unsafe_crossing, policy_type=HandoffRulePolicy, motion_scale=0.4,
        )
    except AssertionError as exc:
        assert isinstance(exc.args[0], dict) and exc.args[0].get("termination") == "collision"
    else:
        raise AssertionError("slow-response regression must reproduce the old target collision")
    fixture_tasks = [
        build_task(_runtime_args(altitude=2.0, duration=24.0), fixture)
        for fixture in generate_fixtures()
    ]
    for fixture_task in fixture_tasks:
        _assert_route_geometry(fixture_task)
        _moving_oracle_protocol(fixture_task)

    assert task_flu_to_gazebo_enu((1.0, 2.0, 3.0), 0.0) == (-2.0, 1.0, 3.0)
    gazebo_rotated = task_flu_to_gazebo_enu(
        (1.0, 2.0, 3.0), math.pi / 2.0
    )
    assert math.isclose(gazebo_rotated[0], 1.0, abs_tol=1e-12)
    assert math.isclose(gazebo_rotated[1], 2.0, abs_tol=1e-12)
    assert math.isclose(gazebo_rotated[2], 3.0, abs_tol=1e-12)

    converted = evaluation_positions_in_uav0_flu(
        {"uav_0": (1.0, 2.0, -3.0), "uav_1": (-1.0, 4.0, -2.0)},
        0.0,
    )
    assert converted == {"uav_0": (1.0, -2.0, 3.0), "uav_1": (-1.0, -4.0, 2.0)}
    rotated = evaluation_positions_in_uav0_flu(
        {"uav_0": (1.0, 0.0, 0.0), "uav_1": (0.0, 1.0, 0.0)},
        math.pi / 2.0,
    )
    assert math.isclose(rotated["uav_0"][0], 0.0, abs_tol=1e-12)
    assert math.isclose(rotated["uav_0"][1], 1.0, abs_tol=1e-12)

    channel = CommunicationChannel(
        bandwidth_bytes_per_second=4096,
        latency_ms=200.0,
        drop_probability=0.0,
        control_hz=10.0,
        seed=3,
    )
    sent = channel.send(sender="uav_0", receiver="uav_1", payload={"x": 1}, t=0)
    assert not sent.dropped and sent.deliver_t == 2
    assert not channel.receive(receiver="uav_1", t=1)
    assert channel.receive(receiver="uav_1", t=2) == [sent]
    dropped = CommunicationChannel(
        bandwidth_bytes_per_second=4096,
        latency_ms=0.0,
        drop_probability=1.0,
        control_hz=10.0,
        seed=3,
    ).send(sender="uav_0", receiver="uav_1", payload={"x": 1}, t=0)
    assert dropped.dropped and dropped.deliver_t is None

    success = _success_protocol()
    _visual_rule_protocol(built)
    _moving_oracle_protocol(
        build_task(_runtime_args(altitude=2.5, duration=24.0))
    )

    timeout_task = _task(
        horizon=3,
        target_start=(-3.0, 0.0, 0.0),
        target_end=(-2.0, 0.0, 0.0),
    )
    timeout = _protocol(timeout_task)
    timeout_positions = dict(timeout_task.initial_positions)
    while not timeout.done:
        timeout.step(
            _observations(timeout_positions), timeout_positions, JointAction()
        )
    assert timeout.termination_reason == "timeout"

    collision_task = _task(horizon=3)
    collision = _protocol(collision_task)
    collision_positions = {"uav_0": (4.0, 4.0, 0.0), "uav_1": (4.1, 4.0, 0.0)}
    collision.step(
        _observations(collision_positions), collision_positions, JointAction()
    )
    assert collision.termination_reason == "collision"

    obstacle_collision = _protocol(collision_task)
    obstacle_positions = {
        "uav_0": collision_task.occluder_center,
        "uav_1": (4.0, 4.0, 0.0),
    }
    obstacle_result = obstacle_collision.step(
        _observations(obstacle_positions), obstacle_positions, JointAction()
    )
    assert obstacle_collision.termination_reason == "collision"
    obstacle_events = [
        event for event in obstacle_result["events"]
        if event["event_type"] == "collision"
    ]
    assert obstacle_events[0]["attributes"]["obstacle"] == "occluder"

    swept_collision = _protocol(collision_task)
    swept_start = {
        "uav_0": (-2.0, 0.0, 0.0),
        "uav_1": (4.0, 4.0, 0.0),
    }
    first_swept = swept_collision.step(
        _observations(swept_start), swept_start, JointAction()
    )
    assert not first_swept["collision"]
    swept_end = {
        "uav_0": (2.0, 0.0, 0.0),
        "uav_1": (4.0, 4.0, 0.0),
    }
    second_swept = swept_collision.step(
        _observations(swept_end), swept_end, JointAction()
    )
    assert second_swept["collision"]
    swept_events = [
        event for event in second_swept["events"]
        if event["event_type"] == "collision"
    ]
    assert swept_events[0]["attributes"]["swept_obstacle_collision_agents"] == [
        "uav_0"
    ]

    target_collision = _protocol(collision_task)
    target_position = target_at_step(
        collision_task.target_start,
        collision_task.target_end,
        0,
        collision_task.horizon,
    )
    target_collision_positions = {
        "uav_0": target_position,
        "uav_1": (4.0, 4.0, 0.0),
    }
    target_result = target_collision.step(
        _observations(target_collision_positions),
        target_collision_positions,
        JointAction(),
    )
    assert target_collision.termination_reason == "collision"
    target_events = [
        event for event in target_result["events"]
        if event["event_type"] == "collision"
    ]
    assert target_events[0]["attributes"]["obstacle"] == "target"
    assert target_events[0]["attributes"]["target_collision_agents"] == ["uav_0"]

    aborted = _protocol(timeout_task)
    aborted.step(
        _observations(timeout_positions), timeout_positions, JointAction()
    )
    aborted.abort("operator interrupt", "safety_abort")
    assert aborted.done and aborted.termination_reason == "safety_abort"

    for name, protocol in {
        "success": success,
        "timeout": timeout,
        "collision": collision,
        "safety_abort": aborted,
    }.items():
        record = build_episode_record(
            task=protocol.task,
            protocol=protocol,
            seed=7,
            world="self_test",
            policy_name="rule",
            spawn_offset_ned=(0.0, 3.0, 0.0),
            cameras=_cameras(),
            flight_time_s=1.25,
            episode_id=f"live_handoff_self_test_{name}",
        )
        validate_episode_record(record)
        assert record["schema_version"] == "0.2"
        assert record["task_spec"]["acceptance_source"] == "deterministic_geometry_truth"
        assert record["task_spec"]["camera_role"] == (
            "real_observation_stream_with_synchronized_visual_scene"
        )
        assert record["outcome"]["termination_reason"] == protocol.termination_reason
        assert record["outcome"]["success"] == (name == "success")
        assert replay_metric_differences(record) == {}

        legacy_record = deepcopy(record)
        for step in legacy_record["steps"]:
            step["truth"].pop("agent_positions", None)
        episode_metrics(legacy_record)

    print("live Handoff PX4/Gazebo non-flight self-test: PASS")


if __name__ == "__main__":
    main()
