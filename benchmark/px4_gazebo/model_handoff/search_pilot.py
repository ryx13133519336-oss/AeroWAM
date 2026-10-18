"""Small paired PX4/Gazebo search pilot, independent of the frozen handoff.

Two fresh isolated SITL sessions receive the same scene and different text.
This is centralized diagnostic control, not restricted communication, a
compositional test, or capture-aligned world-model training data. Nothing is
promoted to a training dataset. No real hardware endpoint is supported.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
import uuid

from .annotation_models import LABEL_SENSOR, verify
from .annotation_session_smoke import INSTANCES, UDP_PROFILE, smoke_environment, stop_owned
from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .images import ros_image_to_jpeg
from .pixel_annotations import CaptureKey, annotate
from .policy_client import PolicyClient, PolicyTimeoutError
from .protocol import AGENT_IDS, ModelObservation
from .search_episode import SearchEpisode
from .search_flight_control import SearchFlightControl, px4_topic
from .search_control_link import LOOPBACK_PROFILE, PX4ControlLink
from .search_pilot_scene import LABELS, WORLD, manual_condition, paired_conditions, scene_digest, search_world
from .search_reporter import REPORTER_VERSION, RGBScanPolicy, requested_color
from .search_motion_policy import MOTION_POLICY_VERSION, RGBSearchPolicy
from .search_motion_gate import MOTION_RULE_VERSION, MotionSearchGate, private_local_goals
from .search_approach_reporter import APPROACH_REPORTER_VERSION
from .startup_evidence import StartupEvidence, validate_mode as validate_startup_evidence_mode
from .startup_clock import StartupClockMonitor, finalize_startup_clock
from .teammate_response import (RESPONSE_PROFILES, ResponseSpec, TeammateResponse,
                                audit_response_trace, request_control)

CHECKPOINT_POLICIES = {"checkpoint", "checkpoint_search"}
MOTION_POLICIES = {"rgb_search", "checkpoint_search"}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def agent_major_from_dependencies(dependencies):
    match = re.search(r"libmicroxrcedds_agent\.so\.(\d+)", dependencies)
    return int(match[1]) if match else None


def resolve_run_duration(policy, requested=None):
    """Bounded wall-clock RUN budget, not simulation time or evidence credit."""
    duration = (90. if policy in MOTION_POLICIES else 30.) if requested is None else requested
    maximum = 120. if policy in MOTION_POLICIES else 60.
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 5 <= duration <= maximum:
        raise ValueError(f"--duration for {policy} must be 5..{maximum:g} wall-clock seconds")
    return float(duration)


def run_progress(episode, policy):
    """Diagnostic only. Private acceptance progress never goes to the policy."""
    gate = episode.motion_gate
    return {"found_accepted": episode.evaluator.found,
            "confirm_accepted": episode.evaluator.completed,
            "motion_metrics": dict(gate.metrics) if gate is not None else {},
            "public_search_diagnostics": getattr(policy, "diagnostics", {})}


def init_pilot_ros(rclpy):
    # Python's SIGINT must reach our try/finally before ROS shuts down: the
    # executor must remain alive to publish ordinary landing commands.
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)


def interrupt_pilot(signum, frame):
    raise KeyboardInterrupt(f"pilot_signal_{signum}")


def save_control_link_diagnostics(link, summary, private):
    """Best-effort teardown diagnostics, never a replacement for gate evidence."""
    try:
        diagnostic = link.diagnostics()
        summary["control_link_verified"] = link.verified
        summary["input_reader_counts"] = diagnostic["reader_counts"]
        summary["preflight_command_publications"] = len(link.commands)
        if diagnostic.get("reader_counts_error"):
            summary["control_link_diagnostic_error"] = diagnostic["reader_counts_error"]
        write_json(private / "control_link.json", diagnostic)
    except Exception as exc:
        summary["control_link_diagnostic_error"] = f"{type(exc).__name__}:{exc}"


def inspect_agent(path):
    executable = path.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Agent executable not found/executable: {executable}")
    result = subprocess.run(["ldd", str(executable)], capture_output=True, text=True, timeout=5.)
    dependencies = result.stdout + result.stderr
    major = agent_major_from_dependencies(dependencies)
    if result.returncode != 0 or "not found" in dependencies or major != 2:
        raise ValueError(f"This PX4 2.x client pilot requires a verified Agent 2.x runtime; found major={major} at {executable}. "
                         "Select the separate 2.4.2 build with --agent; do not replace system libraries.")
    return {"path": str(executable), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "agent_major": major, "dependencies": dependencies}


def run_condition(args, bound, output):
    validate_startup_evidence_mode(getattr(args, "startup_evidence_frames", 0),
                                  execute_flight=args.execute_flight, control_link_check=args.control_link_check,
                                  policy=args.policy, teammate_response=args.teammate_response)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from px4_msgs.msg import VehicleOdometry, VehicleStatus, VehicleCommand, VehicleCommandAck
    from rclpy.serialization import serialize_message
    from .clock_probe import sample_clocks
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.contacts_pb2 import Contacts
    from ..live_sim_manager import is_px4_instance, read_process_table
    if any(is_px4_instance(p, i) for p in read_process_table() for i in INSTANCES):
        raise RuntimeError("Isolated instance 20/21 is occupied; no existing processes stopped")
    executable = args.px4_root / "build/px4_sitl_default/bin/px4"
    rootfs = executable.parent.parent
    agent = str(args.agent)
    if not executable.is_file() or not agent:
        raise FileNotFoundError("PX4 build and MicroXRCEAgent required")
    tree = search_world(args.px4_root, args.models)
    resources = verify(args.models)
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    public, private = output / "policy", output / "evaluator"
    public.mkdir()
    private.mkdir()
    tree.write(private / "world.sdf", encoding="utf-8", xml_declaration=True)
    episode_id = uuid.uuid4().hex
    response_spec = ResponseSpec(args.teammate_response)
    teammate_response, response_trace = TeammateResponse(response_spec), []
    episode = SearchEpisode(bound, episode_id)
    summary = {"schema": "aerowam-search-pilot/1-draft", "episode_id": episode_id,
               "scene_sha256": scene_digest(tree), "model_resources": resources,
               "policy": args.policy, "method": (args.method if args.policy in CHECKPOINT_POLICIES else
                                                 MOTION_POLICY_VERSION if args.policy == "rgb_search" else "rgb_scan_positive_control"),
               "reporter": REPORTER_VERSION, "execute_flight": args.execute_flight,
               "capture_aligned_telemetry": False, "communication": "centralized_unrestricted_diagnostic",
               "compositional_evaluation": False, "training_dataset_created": False,
               "live_multitask_ready": False, "initial_target_visible": None}
    summary["run_budget"] = {"clock": "wall_monotonic", "starts_at": "RUN", "limit_s": args.duration,
                             "automatic_extension": False}
    summary["agent_runtime"] = args.agent_runtime
    summary["teammate_response"] = response_spec.private_metadata()
    summary["controller_request_source"] = ("policy_request_unchanged" if response_spec.profile == "nominal"
                                             else "policy_request_with_declared_peer_response_perturbation")
    if args.policy in MOTION_POLICIES:
        summary.update(motion_acceptance=MOTION_RULE_VERSION, target_motion="static",
                       obstacle_avoidance=False, scene_variant="original_open_arena",
                       reporter=APPROACH_REPORTER_VERSION if args.policy == "checkpoint_search" else MOTION_POLICY_VERSION,
                       control_timestamp_source="latest_px4_odometry_roundtrip_no_extrapolation")
    if args.policy == "checkpoint_search":
        summary.update(action_source="checkpoint_only_no_rule_navigation_override",
                       event_source="auxiliary_public_approach_reporter_not_learned",
                       model_search_training="not_performed_existing_handoff_checkpoint_ood")
    write_json(private / "binding.json", {"target_id": bound.target_id,
                                         "object_ids": bound.object_ids, "task": bound.definition.public_context()})
    write_json(public / "metadata.json", {"schema": "aerowam-search-pilot-public/1-draft",
                                           "episode_id": episode_id, "instruction": bound.definition.instruction,
                                           "state_alignment": "latest_received_local_odometry",
                                           "action_history_semantics": "controller_requests_before_low_level_limits_not_verified_execution",
                                           "action_semantics": "body_FLU_delta_divided_by_0.5s_then_limited"})
    partition = "aerowam_search_" + uuid.uuid4().hex
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env_keys = ("GZ_PARTITION", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION",
                "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE")
    previous_env = {key: os.environ.get(key) for key in env_keys}
    profile = private / "dds_udp_only.xml"
    profile.write_text(LOOPBACK_PROFILE if args.dds_transport == "udp_loopback" else UDP_PROFILE, encoding="utf-8")
    summary["dds_transport"] = args.dds_transport
    os.environ.update(GZ_PARTITION=partition, ROS_DOMAIN_ID=str(args.ros_domain_id),
                      RMW_IMPLEMENTATION="rmw_fastrtps_cpp", FASTRTPS_DEFAULT_PROFILES_FILE=str(profile),
                      FASTDDS_DEFAULT_PROFILES_FILE=str(profile))
    processes, logs, subscriptions = [], [], []
    gui = None
    command_acks = []
    transport = ros = executor = control = control_link = None
    spin_thread = None
    rclpy_started = False
    previous_sigterm = None
    lock = threading.Lock()
    frames, odometry, status, received, counts = {}, {}, {}, {}, {}
    clock = {"stamp": -1, "received": -math.inf}
    contacts = []
    spin_error = []
    spin_stop = threading.Event()
    spin_health = {"iterations": 0}
    requests = []
    startup_monitor = evidence = None

    def spawn(name, argv, env, cwd):
        log = (private / f"{name}.log").open("xb")
        logs.append(log)
        process = subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((name, process))
        return process

    def check_processes():
        if startup_monitor is not None:
            startup_monitor.check()
        failed = [f"{name}:{p.returncode}" for name, p in processes if p.poll() is not None]
        if failed or spin_error:
            raise RuntimeError("Session process/executor failed: " + str(failed + spin_error))

    def stream_diagnostics():
        now = time.monotonic()
        with lock:
            camera = {topic: {"buffered": len(rows), "last_stamp_ns": max(rows) if rows else None,
                              "age_s": now - rows[max(rows)][1] if rows else None}
                      for topic, rows in frames.items()}
        return {"ros_counts": {f"{kind}_{i}": counts.get((kind, i), 0)
                               for kind in ("status", "odometry") for i in (0, 1)},
                "camera_streams": camera, "sim_stamp_ns": clock["stamp"],
                "clock_age_s": now - clock["received"] if math.isfinite(clock["received"]) else None}

    def snapshot(after):
        with lock:
            common = None
            for pair in pairs:
                stamps = set(frames.get(pair["rgb"], {})) & set(frames.get(pair["labels"], {}))
                common = stamps if common is None else common & stamps
            advancing = [s for s in common or () if s > after]
            if not advancing:
                return None
            stamp = max(advancing)
            chosen = {pair["agent"]: (frames[pair["rgb"]][stamp], frames[pair["labels"]][stamp], pair) for pair in pairs}
        now = time.monotonic()
        if now - clock["received"] > .5:
            raise RuntimeError("Simulation clock is stale")
        if any(now - row[0][1] > .25 or now - row[1][1] > .25 for row in chosen.values()):
            return None
        if not 0 <= clock["stamp"] - stamp <= 200_000_000:
            return None
        images, private_frames = {}, {}
        for agent_id, (rgb_row, label_row, pair) in chosen.items():
            rgb, labels = rgb_row[0], label_row[0]
            decoded = pair_rgb_instance(rgb_message=rgb, label_message=labels, rgb_topic=pair["rgb"],
                                        segmentation_topic=pair["seg"], label_topic=pair["labels"], encoding=GZ8_INSTANCE_ENCODING)
            transformed = resize_pair(decoded)
            key = CaptureKey(agent_id, stamp, stamp, f"{partition}:{agent_id}:{stamp}", 256, 256)
            bundle = annotate(task=bound, rgb_capture=key, label_capture=key,
                              instance_labels=transformed.object_labels, label_to_object=LABELS)
            private_frames[agent_id] = bundle.frame
            message = SimpleNamespace(width=rgb.width, height=rgb.height, step=rgb.step,
                                      data=rgb.data, encoding="rgb8")
            images[agent_id] = ros_image_to_jpeg(message)
        return stamp, images, private_frames

    try:
        if getattr(args, "startup_evidence_frames", 0):
            startup_monitor = StartupClockMonitor(private / "startup_host_clocks.json")
            startup_monitor.start()
        env = dict(os.environ)
        env["GZ_SIM_RESOURCE_PATH"] = str(args.px4_root / "Tools/simulation/gz/models") + os.pathsep + env.get("GZ_SIM_RESOURCE_PATH", "")
        env.pop("GZ_SIM_SERVER_CONFIG_PATH", None)
        if args.policy in CHECKPOINT_POLICIES:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                policy_port = reservation.getsockname()[1]
            summary["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            spawn("policy", [str(args.policy_python), "-B", "-u", "-m", "benchmark.px4_gazebo.model_handoff.policy_server",
                             "--checkpoint", str(args.checkpoint), "--device", args.device,
                             "--port", str(policy_port),
                             "--grounded-search-approach" if args.policy == "checkpoint_search" else "--grounded-search-pilot"], env, Path.cwd())
            deadline = time.monotonic() + 150.
            while time.monotonic() < deadline:
                check_processes()
                text = (private / "policy.log").read_text(errors="replace")
                if "serving " in text:
                    version = APPROACH_REPORTER_VERSION if args.policy == "checkpoint_search" else REPORTER_VERSION
                    if f"serving {args.method}+{version} " not in text:
                        raise ValueError("Checkpoint method differs from requested method")
                    break
                time.sleep(.2)
            else:
                raise TimeoutError("Policy initialization timed out")
            policy = PolicyClient(f"127.0.0.1:{policy_port}", timeout_s=.4)
        else:
            policy = RGBSearchPolicy() if args.policy == "rgb_search" else RGBScanPolicy()
        transport = Node()
        init_pilot_ros(rclpy)
        rclpy_started = True
        previous_sigterm = signal.signal(signal.SIGTERM, interrupt_pilot)
        ros = rclpy.create_node("aerowam_search_pilot")
        # Advertise all actual control writers before PX4/DDS discovery.
        # No control messages are sent until startup and the no-arm gate.
        if args.control_link_check or args.execute_flight:
            control_link = PX4ControlLink(ros, timeout_s=args.control_link_timeout)
        # PX4 uXRCE output writers are live/volatile telemetry. Requesting
        # transient-local here can leave the ROS reader unmatched: Gazebo
        # cameras keep flowing while status/odometry remain at zero. Keep the
        # command writers' separate verified QoS unchanged.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        for index in (0, 1):
            for key, msg_type, suffix in (("status", VehicleStatus, "vehicle_status_v4"),
                                          ("odometry", VehicleOdometry, "vehicle_odometry")):
                def receive(msg, k=key, i=index):
                    with lock:
                        (status if k == "status" else odometry)[i] = msg
                        received[k, i] = time.monotonic()
                        counts[k, i] = counts.get((k, i), 0) + 1
                ros.create_subscription(msg_type, f"/annotation_smoke_{index}/fmu/out/{suffix}", receive, qos)
            def receive_ack(msg, i=index):
                command_acks.append({"agent": f"uav_{i}", "command": int(msg.command), "result": int(msg.result),
                                     "target_system": int(msg.target_system), "target_component": int(msg.target_component),
                                     "timestamp": int(msg.timestamp), "monotonic_ns": time.monotonic_ns()})
                print(f"PX4 ACK uav_{i}: command={msg.command} result={msg.result}", flush=True)
            ros.create_subscription(VehicleCommandAck, px4_topic(index, "out", "vehicle_command_ack", VehicleCommandAck), receive_ack, qos)
        pairs = []
        for index in (0, 1):
            prefix = f"/world/{WORLD}/model/annotation_uav_{index}/link/camera_link/sensor"
            seg = prefix + f"/{LABEL_SENSOR}/segmentation"
            pair = {"agent": f"uav_{index}", "rgb": prefix + "/camera/image", "seg": seg, "labels": seg + "/labels_map"}
            pairs.append(pair)
            for topic in (pair["rgb"], pair["labels"]):
                def capture(message, key=topic):
                    stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
                    snapshot_msg = Image()
                    snapshot_msg.CopyFrom(message)
                    with lock:
                        rows = frames.setdefault(key, {})
                        rows[stamp] = snapshot_msg, time.monotonic()
                        while len(rows) > 8:
                            del rows[min(rows)]
                if not transport.subscribe(Image, topic, capture):
                    raise RuntimeError(f"Subscription failed: {topic}")
                subscriptions.append(topic)
        def receive_clock(message):
            with lock:
                clock.update(stamp=message.sim.sec * 1_000_000_000 + message.sim.nsec, received=time.monotonic())
        clock_topic = f"/world/{WORLD}/clock"
        if not transport.subscribe(Clock, clock_topic, receive_clock):
            raise RuntimeError("Cannot subscribe simulation clock")
        subscriptions.append(clock_topic)
        for identity in ("probe_red", "probe_blue"):
            topic = f"/aerowam/search_pilot/private/contacts/{identity}"
            def receive_contact(message):
                for contact in message.contact:
                    if "annotation_uav_" in contact.collision1.name or "annotation_uav_" in contact.collision2.name:
                        contacts.append({"sim_time_ns": clock["stamp"], "collision1": contact.collision1.name,
                                         "collision2": contact.collision2.name})
            if not transport.subscribe(Contacts, topic, receive_contact):
                raise RuntimeError("Cannot subscribe private contact sensor")
            subscriptions.append(topic)
        executor = SingleThreadedExecutor()
        executor.add_node(ros)
        def spin():
            try:
                # Periodically rebuild/check the wait set while DDS endpoints
                # appear. Only this thread spins the executor, including the
                # flight timer; no extra ROS CLI process is required.
                while not spin_stop.is_set():
                    executor.spin_once(timeout_sec=.1)
                    spin_health["iterations"] += 1
            except Exception as exc:
                spin_error.append(str(exc))
        spin_thread = threading.Thread(target=spin, daemon=True)
        spin_thread.start()
        spawn("dds_agent", [agent, "udp4", "-p", str(port)], env, output)
        spawn("gazebo", ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", str(private / "world.sdf")], env, output)
        if args.gui:
            log = (private / "gui.log").open("xb")
            logs.append(log)
            gui = subprocess.Popen(["gz", "sim", "-g", "-v", "3"], env=env, cwd=output,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for index, instance in enumerate(INSTANCES):
            working = private / f"px4_work_{instance}"
            working.mkdir()
            px4_env = smoke_environment(env, partition=partition, domain=args.ros_domain_id, port=port, index=index)
            px4_env["PX4_GZ_WORLD"] = WORLD
            # Match the old headless SITL's GCS-link setting, only in fresh
            # per-run parameters. Do not disable estimator/arming checks.
            px4_env["PX4_PARAM_NAV_DLL_ACT"] = "0"
            spawn(f"px4_{index}", [str(executable), str(rootfs / "etc"), "-i", str(instance), "-w", str(working), "-d"], px4_env, rootfs)
        startup_started = time.monotonic()
        deadline = startup_started + args.startup_timeout
        last_startup_log = -math.inf
        while time.monotonic() < deadline:
            check_processes()
            if time.monotonic() - last_startup_log >= 5.:
                diagnostic = stream_diagnostics()
                print(f"startup streams={json.dumps(diagnostic)}", flush=True)
                last_startup_log = time.monotonic()
            if len(status) == len(odometry) == 2 and min(counts.values()) >= 2 and clock["stamp"] > 0:
                if any(s.arming_state != 1 or s.failsafe for s in status.values()):
                    raise RuntimeError("Startup requires two disarmed healthy vehicles")
                if all(int(m.pose_frame) == 1 and int(m.velocity_frame) == 1
                       and all(math.isfinite(float(v)) for xs in (m.position, m.velocity, m.q) for v in xs) for m in odometry.values()):
                    candidate = snapshot(-1)
                    if candidate is not None:
                        break
            time.sleep(.05)
        else:
            raise TimeoutError("Isolated startup streams not ready")
        summary["startup_passed"] = True
        summary["startup_wait_s"] = time.monotonic() - startup_started
        if getattr(args, "startup_evidence_frames", 0):
            # No PX4ControlLink or SearchFlightControl exists in this mode.
            evidence = StartupEvidence(private / "startup_evidence", args.startup_evidence_frames,
                                       summary["scene_sha256"], require_session_clock=True)
            evidence_deadline, after = time.monotonic() + 15., -250_000_000
            while len(evidence.rows) < args.startup_evidence_frames:
                check_processes()
                if time.monotonic() >= evidence_deadline:
                    raise TimeoutError("Disarmed evidence collection incomplete")
                with lock:
                    common = None
                    for pair in pairs:
                        stamps = set(frames.get(pair["rgb"], {})) & set(frames.get(pair["labels"], {}))
                        common = stamps if common is None else common & stamps
                    advancing = [s for s in common or () if s - after >= 250_000_000]
                    if advancing:
                        stamp = max(advancing)
                        cameras = {p["agent"]: {"stamp_ns": stamp, "rgb": frames[p["rgb"]][stamp],
                                               "labels": frames[p["labels"]][stamp]} for p in pairs}
                        frozen = dict(odometry=odometry.copy(), status=status.copy(), received=received.copy(),
                                      clock=clock.copy(), cameras=cameras, now=time.monotonic())
                    else:
                        frozen = None
                if frozen is not None:
                    evidence.add(**frozen, serialize_ros=serialize_message, host_clock=sample_clocks())
                    after = stamp
                time.sleep(.02)
            summary["startup_evidence"] = evidence.finish()
        if control_link is not None:
            control_link.start()
            last_gate_log = -math.inf
            while not control_link.poll(command_acks, status, received):
                check_processes()
                if time.monotonic() - last_gate_log >= 2.:
                    print(f"control link phase={control_link.phase} readers={control_link.reader_counts()}", flush=True)
                    last_gate_log = time.monotonic()
                if control_link.phase == "FAILED":
                    raise RuntimeError(control_link.reason)
                time.sleep(.05)
            summary["control_link_verified"] = True
            print("control link VERIFIED: six writers matched; fresh accepted disarm ACK from both vehicles", flush=True)
        if args.control_link_check:
            summary["category"], summary["reason"] = "startup_only", "disarmed_command_roundtrip_verified"
        elif not args.execute_flight:
            summary["category"], summary["reason"] = "startup_only", "no_motion_commands_requested"
        else:
            control = SearchFlightControl(ros, odometry, status, received, control_link,
                                          use_telemetry_timestamps=args.policy in MOTION_POLICIES)
            print(f"[{episode_id[:8]}] isolated SITL takeoff; policy={summary['method']}", flush=True)
            while control.phase not in {"RUN", "DONE", "LANDING"}:
                check_processes()
                time.sleep(.05)
            if args.policy in MOTION_POLICIES:
                episode.motion_gate = MotionSearchGate(private_local_goals(tree, bound.target_id, control.yaws))
            after, last_decision, wait_started = -1, -math.inf, time.monotonic()
            run_sim_started = clock["stamp"]
            print(f"RUN budget={args.duration:g} wall-clock seconds; no automatic extension", flush=True)

            def record_run_stop(reason):
                summary.update(run_stop_reason=reason,
                               run_wall_elapsed_s=time.monotonic() - control.run_started,
                               run_sim_elapsed_s=(clock["stamp"] - run_sim_started) / 1e9,
                               evidence_progress=run_progress(episode, policy))
                print(f"RUN stop={reason}; wall={summary['run_wall_elapsed_s']:.2f}s "
                      f"sim={summary['run_sim_elapsed_s']:.2f}s; "
                      f"evidence={json.dumps(summary['evidence_progress'], ensure_ascii=False)}; normal landing", flush=True)

            action_history = []
            while control.phase == "RUN":
                check_processes()
                if contacts:
                    episode.evaluator.safety_violation("target_contact")
                    record_run_stop("target_contact")
                    control.land()
                    break
                if time.monotonic() - control.run_started > args.duration:
                    record_run_stop("run_wall_timeout")
                    control.land()
                    break
                if time.monotonic() - last_decision < .5:
                    time.sleep(.01)
                    continue
                try:
                    candidate = snapshot(after)
                except (ValueError, KeyError) as exc:
                    episode.annotation_error(str(exc))
                    control.land()
                    break
                if candidate is None:
                    if time.monotonic() - wait_started > 1.:
                        episode.observation_error("no_fresh_exact_capture_pair")
                        control.land()
                    time.sleep(.01)
                    continue
                stamp, images, private_frames = candidate
                try:
                    observation = ModelObservation(episode_id, bound.definition.instruction, stamp,
                                                   {a: stamp for a in AGENT_IDS}, images, control.states(),
                                                   action_history=action_history[-4:])
                    episode.validate_capture(observation, private_frames)
                except (ValueError, KeyError) as exc:
                    episode.observation_error(str(exc))
                    control.land()
                    break
                if summary["initial_target_visible"] is None:
                    summary["initial_target_visible"] = {a: private_frames[a].objects[bound.target_id].visible_bbox is not None for a in AGENT_IDS}
                try:
                    inference_started = time.monotonic()
                    if args.policy in CHECKPOINT_POLICIES:
                        response, latency_ms = policy.infer_grounded(observation)
                    else:
                        response = policy(observation)
                        latency_ms = (time.monotonic() - inference_started) * 1000
                        if latency_ms > 400:
                            raise PolicyTimeoutError("Positive-control inference exceeded 400 ms")
                except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                    summary["policy_error"] = f"{type(exc).__name__}:{exc}"
                    control.land(("model_failure", "policy_inference_error"))
                    break
                action = episode.delivered(observation, private_frames, response, now_sim_time_ns=clock["stamp"])
                requests.append({"observation": observation.to_wire(), "response": response.to_wire(),
                                 "inference_latency_ms": latency_ms})
                write_json(private / f"annotations_{len(requests):03d}.json",
                           {a: {identity: value.visible_bbox for identity, value in frame.objects.items()} for a, frame in private_frames.items()})
                issued_action = request_control(control, teammate_response, action,
                                                sim_stamp_ns=clock["stamp"], trace=response_trace)
                requests[-1]["controller_requested_action"] = issued_action.to_wire()
                action_history.append(issued_action.to_wire()["waypoints"])
                after, last_decision = stamp, time.monotonic()
                wait_started = last_decision
                print(f"[{episode_id[:8]}] step={len(requests)} reports={[r.event for r in response.reports]}", flush=True)
                if args.policy == "rgb_search":
                    print(f"search {json.dumps(policy.diagnostics, ensure_ascii=False)}", flush=True)
                    requests[-1]["public_search_diagnostics"] = policy.diagnostics
                if episode.evaluator.completed or episode.evaluator.failure is not None:
                    record_run_stop(episode.evaluator.failure[1] if episode.evaluator.failure else "task_evidence_complete")
                    control.land()
                    break
            if "run_stop_reason" not in summary and control.run_started is not None:
                failure = control.failure or episode.evaluator.failure
                record_run_stop(failure[1] if failure else "run_interrupted")
    except KeyboardInterrupt:
        episode.observation_error("pilot_interrupted")
    except Exception as exc:
        episode.observation_error(f"{type(exc).__name__}:{exc}")
    finally:
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if control is not None:
                control.land()
                deadline = time.monotonic() + 52.
                while control.phase != "DONE" and time.monotonic() < deadline and not spin_error:
                    if contacts:
                        episode.evaluator.safety_violation("target_contact")
                    time.sleep(.05)
                if control.phase != "DONE" or not all(status[i].arming_state == 1 for i in (0, 1)):
                    episode.observation_error("landing_not_verified")
                if control.failure:
                    category, reason = control.failure
                    if category in {"simulator_error", "evaluator_error"}:
                        episode.evaluator.invalidate(category, reason)
                    elif category == "model_failure":
                        episode.model_error(reason)
                    else:
                        episode.evaluator.safety_violation(reason)
                summary.update(setpoint_publications=control.publications, command_publications=control.commands,
                               max_publication_gap_s=control.max_publication_gap_s,
                               landing_verified=all(status[i].arming_state == 1 for i in (0, 1)))
                (public / "published_controls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in control.trace), encoding="utf-8")
                write_json(private / "flight_diagnostics.json", {"states": control.state_trace, "commands": control.command_trace})
            summary["command_acks"] = command_acks
            summary["stream_diagnostics"] = stream_diagnostics()
            summary["executor_iterations"] = spin_health["iterations"]
            if control_link is not None:
                save_control_link_diagnostics(control_link, summary, private)
            try:
                summary["teammate_response_audit"] = audit_response_trace(
                    response_spec, response_trace,
                    controller_records=[(r["response"]["action"], r["controller_requested_action"])
                                        for r in requests if "controller_requested_action" in r])
            except (ValueError, TypeError, KeyError) as exc:
                episode.annotation_error(f"teammate_response_trace:{exc}")
            (private / "teammate_response.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in response_trace), encoding="utf-8")
            if "category" not in summary:
                result = episode.finish()
                summary["evaluation_schema"] = result.pop("schema", None)
                summary.update(result)
            summary["contacts"] = contacts
            (public / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8")
            spin_stop.set()
            if spin_thread is not None:
                spin_thread.join(timeout=2.)
            if executor is not None:
                executor.shutdown(timeout_sec=2.)
            if transport is not None:
                for topic in subscriptions:
                    try:
                        transport.unsubscribe(topic)
                    except Exception as exc:
                        summary.setdefault("cleanup_errors", []).append(str(exc))
            if ros is not None:
                ros.destroy_node()
            if rclpy_started:
                rclpy.try_shutdown()
        except Exception as exc:
            summary.setdefault("cleanup_errors", []).append(f"session_cleanup:{type(exc).__name__}:{exc}")
            if startup_monitor is None:
                raise
        finally:
            # Even a log/disk/ROS shutdown error must not skip process cleanup.
            try:
                groups = [processes]
                if gui is not None:
                    groups.append([("gui", gui)])
                for group in groups:
                    try:
                        summary.setdefault("cleanup_errors", []).extend(stop_owned(group))
                    except Exception as exc:
                        summary.setdefault("cleanup_errors", []).append(f"process_cleanup:{type(exc).__name__}:{exc}")
                        if startup_monitor is None:
                            raise
            finally:
                for log in logs:
                    try:
                        log.close()
                    except Exception as exc:
                        summary.setdefault("cleanup_errors", []).append(f"log_close:{type(exc).__name__}:{exc}")
                        if startup_monitor is None:
                            raise
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                signal.signal(signal.SIGINT, previous_handler)
                if previous_sigterm is not None:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                summary["owned_processes_stopped"] = (all(p.poll() is not None for _, p in processes)
                                                       and (gui is None or gui.poll() is not None))
                if startup_monitor is not None:
                    finalize_startup_clock(startup_monitor, summary, evidence)
                write_json(private / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--agent", type=Path, help="Separate compatible Micro XRCE-DDS Agent 2.x executable; defaults to PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=("rgb_scan", "rgb_search", "checkpoint", "checkpoint_search"), default="rgb_scan",
                        help="checkpoint_search: learned actions + explicit approach evidence and movement acceptance; checkpoint: legacy scan reporter")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--method", choices=("joint_bc", "dual_vln_hier", "dual_wam_mpc"), default="joint_bc")
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute-flight", action="store_true", help="Actually arm/take off isolated SITL instances20/21; no hardware")
    parser.add_argument("--control-link-check", action="store_true", help="Verify disarm-command ACK while already disarmed; never take off")
    parser.add_argument("--startup-evidence-frames", type=int, default=0,
                        help="Private disarmed raw capture: 0 disabled or 2..20 frames; no flight/command check/checkpoint")
    parser.add_argument("--gui", action="store_true", help="Open GUI in the same isolated partition; close with the session")
    parser.add_argument("--duration", type=float, default=None,
                        help="Wall-clock RUN budget, never auto-extended: rgb_search/checkpoint_search default90/range5..120; other modes default30/range5..60")
    parser.add_argument("--startup-timeout", type=float, default=150.)
    parser.add_argument("--control-link-timeout", type=float, default=25., help="No-arm match/ACK gate budget, 5..60 seconds")
    parser.add_argument("--dds-transport", choices=("udp_all", "udp_loopback"), default="udp_all",
                        help="Scoped ROS-side DDS profile; PX4 participant XML may override Agent defaults, so loopback is diagnostic only")
    parser.add_argument("--ros-domain-id", type=int, default=78)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teammate-response", choices=RESPONSE_PROFILES, default="nominal",
                        help="Opt-in UAV1 command-response diagnostic, not an autonomous teammate; cautious gain0.5 or delayed0.75 simulation seconds; no safety/landing delay")
    parser.add_argument("--instruction", help="Your original text; if omitted, prompt in the terminal")
    parser.add_argument("--expected-target", choices=("red", "blue"),
                        help="Separate human scoring annotation; never sent to the model")
    parser.add_argument("--paired-check", action="store_true",
                        help="Developer-only two authored conditions; default is one manually entered instruction")
    args = parser.parse_args()
    if os.name == "nt":
        parser.error("Run in WSL/Linux with ROS and Gazebo Python bindings")
    if args.execute_flight and args.control_link_check:
        parser.error("--control-link-check cannot be combined with --execute-flight")
    try:
        validate_startup_evidence_mode(args.startup_evidence_frames, execute_flight=args.execute_flight,
                                      control_link_check=args.control_link_check, policy=args.policy,
                                      teammate_response=args.teammate_response)
    except ValueError as exc:
        parser.error(str(exc))
    if args.teammate_response != "nominal" and args.policy not in MOTION_POLICIES:
        parser.error("Non-nominal teammate response requires rgb_search or checkpoint_search; legacy modes unchanged")
    try:
        args.duration = resolve_run_duration(args.policy, args.duration)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.ros_domain_id <= 100 or not 10 <= args.startup_timeout <= 180:
        parser.error("Invalid domain or startup timeout")
    if not 5 <= args.control_link_timeout <= 60:
        parser.error("--control-link-timeout must be 5..60 seconds")
    if args.policy in CHECKPOINT_POLICIES and (args.checkpoint is None or not args.checkpoint.is_file()
                                        or args.policy_python is None or not args.policy_python.is_file()):
        parser.error("Checkpoint policy requires existing --checkpoint and --policy-python")
    try:
        selected_agent = args.agent or shutil.which("MicroXRCEAgent")
        if selected_agent is None:
            raise ValueError("MicroXRCEAgent not found; specify --agent")
        args.agent_runtime = inspect_agent(Path(selected_agent))
        args.agent = Path(args.agent_runtime["path"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    if args.paired_check:
        if args.instruction is not None or args.expected_target is not None:
            parser.error("Manual instruction cannot be combined with --paired-check")
        conditions = paired_conditions(args.seed)
    else:
        instruction = args.instruction
        if instruction is None:
            instruction = input("请输入本次搜索指令（原文传给模型）：")
        expected = args.expected_target
        if expected is None:
            choice = input("验收正确目标（仅裁判可见）：1=红色黑条纹球，2=蓝色白条纹方块：").strip()
            expected = {"1": "red", "2": "blue", "red": "red", "blue": "blue"}.get(choice)
        try:
            requested_color(instruction)  # Refuse unsupported pilot language before starting SITL.
            conditions = (manual_condition(instruction, expected),)
        except ValueError as exc:
            parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index, bound in enumerate(conditions):
        summary = run_condition(args, bound, args.output_dir / f"condition_{index:02d}")
        summaries.append(summary)
        print(f"condition_{index:02d}: {summary['category']} ({summary['reason']})", flush=True)
        if summary["category"] in {"simulator_error", "evaluator_error"} or not summary["owned_processes_stopped"] or summary.get("cleanup_errors"):
            break
    group = {"schema": "aerowam-search-pilot-pair/1-draft", "seed": args.seed, "independent_scene_groups": 1,
             "instruction_source": "authored_developer_pair" if args.paired_check else "manual_user_input",
             "conditions_expected": len(conditions),
             "conditions_completed": len(summaries), "scene_sha256": [s["scene_sha256"] for s in summaries],
             "pair_success": len(summaries) == 2 and len({s["scene_sha256"] for s in summaries}) == 1
                             and all(s["category"] == "success" for s in summaries),
             "categories": [s["category"] for s in summaries], "formal_benchmark_result": False}
    write_json(args.output_dir / "pair_summary.json", group)
    completed = len(summaries) == len(conditions) and all(s["category"] == ("success" if args.execute_flight else "startup_only") for s in summaries)
    raise SystemExit(0 if completed else 2)


if __name__ == "__main__":
    main()
