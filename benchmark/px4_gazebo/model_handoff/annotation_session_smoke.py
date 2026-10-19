"""Isolated two-aircraft startup diagnostic. Never arms or commands motion.

Reads Gazebo RGB/labels and ROS PX4 status/odometry, not policy inference.
No broad stale-process cleanup, default DDS agent, installed model edits or
old dataset writes. Passed startup is not a task/flight/calibration success.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from .annotation_models import AIRCRAFT_MODEL, LABEL_SENSOR, verify
from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair

WORLD = "aerowam_annotation_startup"
INSTANCES = (20, 21)
UDP_PROFILE = """<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors><transport_descriptor><transport_id>annotation_udp</transport_id>
    <type>UDPv4</type></transport_descriptor></transport_descriptors>
  <participant profile_name="annotation_udp" is_default_profile="true"><rtps>
    <userTransports><transport_id>annotation_udp</transport_id></userTransports>
    <useBuiltinTransports>false</useBuiltinTransports>
  </rtps></participant>
</profiles>
"""


def build_world(px4_root: Path, models: Path) -> ET.ElementTree:
    verify(models)
    tree = ET.parse(px4_root / "Tools/simulation/gz/worlds/lawn.sdf")
    scene = tree.getroot().find("world")
    if scene is None or scene.get("name") != "lawn" or scene.findall("include"):
        raise ValueError("Expected local lawn world without external includes")
    scene.set("name", WORLD)
    for plugin in scene.findall("plugin"):
        scene.remove(plugin)
    for filename, name in (
        ("physics", "Physics"), ("user-commands", "UserCommands"),
        ("scene-broadcaster", "SceneBroadcaster"), ("contact", "Contact"),
        ("imu", "Imu"), ("air-pressure", "AirPressure"),
        ("navsat", "NavSat"), ("magnetometer", "Magnetometer"), ("sensors", "Sensors"),
    ):
        plugin = ET.SubElement(scene, "plugin", filename=f"gz-sim-{filename}-system", name=f"gz::sim::systems::{name}")
        if name == "Sensors":
            ET.SubElement(plugin, "render_engine").text = "ogre2"
    reference = ET.parse(Path(__file__).with_name("annotation_probe.sdf")).getroot().find("world")
    for name in ("probe_red", "probe_blue", "probe_wall"):
        item = deepcopy(reference.find(f"model[@name='{name}']"))
        # Match resting aircraft camera height, without moving/arming aircraft.
        parts = item.findtext("pose").split()
        parts[2] = "0.6"
        item.find("pose").text = " ".join(parts)
        scene.append(item)
    for index, y in enumerate((0., -3.)):
        include = ET.SubElement(scene, "include")
        ET.SubElement(include, "uri").text = (models.resolve() / AIRCRAFT_MODEL / "model.sdf").as_uri()
        ET.SubElement(include, "name").text = f"annotation_uav_{index}"
        ET.SubElement(include, "pose").text = f"0 {y} 0.1 0 0 0"
    return tree


def smoke_environment(base, *, partition, domain, port, index):
    if (type(index) is not int or index not in (0, 1) or type(domain) is not int or not 1 <= domain <= 100
            or type(port) is not int or not 1024 <= port <= 65535 or not isinstance(partition, str) or not partition):
        raise ValueError("Invalid diagnostic vehicle/domain")
    env = dict(base)
    # rcS can load saved parameters: always use fresh -w directories below.
    for name in list(env):
        if name.startswith("PX4_"):
            env.pop(name)
    env.update(GZ_PARTITION=partition, ROS_DOMAIN_ID=str(domain), HEADLESS="1",
               PX4_SYS_AUTOSTART="4010", PX4_SIM_MODEL="gz_x500_mono_cam",
               PX4_GZ_STANDALONE="1", PX4_GZ_WORLD=WORLD,
               PX4_GZ_MODEL_NAME=f"annotation_uav_{index}",
               PX4_UXRCE_DDS_NS=f"annotation_smoke_{index}", PX4_UXRCE_DDS_PORT=str(port))
    return env


def stop_owned(processes):
    """Stop exactly our fresh process groups, even after an interrupt."""
    errors = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        for name, process in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    errors.append(f"{name}: {exc}")
        deadline = time.monotonic() + 3.
        while time.monotonic() < deadline and any(p.poll() is None for _, p in processes):
            time.sleep(.1)
    for name, process in processes:
        try:
            process.wait(timeout=1.)
        except subprocess.TimeoutExpired:
            errors.append(f"{name} did not stop")
    return errors


def odometry_snapshot(message):
    """ROS numeric arrays may be numpy.float32; keep strict JSON evidence."""
    vectors = {key: [float(value) for value in getattr(message, key)] for key in ("position", "velocity", "q")}
    finite = all(math.isfinite(value) for values in vectors.values() for value in values)
    vectors = {key: [value if math.isfinite(value) else None for value in values] for key, values in vectors.items()}
    return {"timestamp": int(message.timestamp), **vectors, "finite": finite,
            "pose_frame": int(message.pose_frame), "velocity_frame": int(message.velocity_frame)}


def run(*, px4_root: Path, models: Path, output: Path, timeout_s=120., domain=77, udp_only=False,
        prepared_scene=None, agent_path=None, gui=False):
    if not 10 <= timeout_s <= 180 or type(domain) is not int or not 1 <= domain <= 100:
        raise ValueError("Timeout must be 10..180 seconds; ROS domain must be 1..100")
    if os.name == "nt":
        raise RuntimeError("Run inside WSL/Linux with ROS and Gazebo bindings")
    from ..live_sim_manager import is_px4_instance, read_process_table
    if any(is_px4_instance(p, i) for p in read_process_table() for i in INSTANCES):
        raise RuntimeError("Diagnostic PX4 instance 20/21 is already running; no processes were stopped")
    # Import everything before starting any processes; missing packages must
    # not leave a half-started simulator behind.
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from px4_msgs.msg import VehicleOdometry, VehicleStatus
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    executable = px4_root / "build/px4_sitl_default/bin/px4"
    rootfs = px4_root / "build/px4_sitl_default"
    agent = str(agent_path) if agent_path is not None else shutil.which("MicroXRCEAgent")
    if not executable.is_file() or not agent:
        raise FileNotFoundError("PX4 executable and MicroXRCEAgent are required")
    if prepared_scene is None:
        tree, scene_resources = build_world(px4_root, models), None
    else:
        tree, scene_resources = prepared_scene
    world_name = tree.getroot().find("world").get("name")
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    tree.write(output / "world.sdf", encoding="utf-8", xml_declaration=True)
    partition = "aerowam_startup_" + uuid.uuid4().hex
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    previous = {key: os.environ.get(key) for key in ("GZ_PARTITION", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION",
                                                   "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE")}
    os.environ.update(GZ_PARTITION=partition, ROS_DOMAIN_ID=str(domain))
    if udp_only:
        profile = output / "dds_udp_only.xml"
        profile.write_text(UDP_PROFILE, encoding="utf-8")
        os.environ.update(RMW_IMPLEMENTATION="rmw_fastrtps_cpp", FASTRTPS_DEFAULT_PROFILES_FILE=str(profile),
                          FASTDDS_DEFAULT_PROFILES_FILE=str(profile))
    processes, logs, subscriptions = [], [], []
    node = ros = None
    rclpy_started = False
    lock = threading.Lock()
    frames, counts, latest_status, odometry, received_ros = {}, {}, {}, {}, {}
    summary = {"schema": "aerowam-annotation-startup/1-draft", "partition": partition,
               "ros_domain_id": domain, "dds_port": port, "px4_instances": list(INSTANCES),
               "startup_passed": False, "flight_ready": False, "live_multitask_ready": False,
               "motion_commands_sent": 0, "world_sha256": hashlib.sha256((output / "world.sdf").read_bytes()).hexdigest(),
               "model_resources": verify(models), "dds_transport": "udp_only_diagnostic" if udp_only else "inherited"}
    summary["world_name"] = world_name
    if scene_resources is not None:
        summary["scene_resources"] = scene_resources

    def spawn(name, argv, env, cwd):
        log = (output / f"{name}.log").open("xb")
        logs.append(log)
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((name, process))
        return process

    try:
        env = dict(os.environ)
        model_dir = px4_root / "Tools/simulation/gz/models"
        env["GZ_SIM_RESOURCE_PATH"] = str(model_dir) + os.pathsep + env.get("GZ_SIM_RESOURCE_PATH", "")
        plugin_dir = px4_root / "build/px4_sitl_default/src/modules/simulation/gz_plugins"
        if plugin_dir.is_dir():
            env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = str(plugin_dir) + os.pathsep + env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
            summary["px4_plugin_directory"] = str(plugin_dir)
        # Avoid inherited server.config injecting a different sensor system.
        env.pop("GZ_SIM_SERVER_CONFIG_PATH", None)
        node = Node()
        rclpy.init(args=[])
        rclpy_started = True
        ros = rclpy.create_node("aerowam_annotation_startup")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        for index in (0, 1):
            prefix = f"/annotation_smoke_{index}/fmu/out"
            for key, msg_type, suffix in (("status", VehicleStatus, "vehicle_status_v4"),
                                          ("odometry", VehicleOdometry, "vehicle_odometry")):
                counts[f"{key}_{index}"] = 0
                def receive(msg, k=key, i=index):
                    counts[f"{k}_{i}"] += 1
                    received_ros[k, i] = time.monotonic()
                    if k == "status":
                        latest_status[i] = {"arming_state": int(msg.arming_state), "failsafe": bool(msg.failsafe),
                                            "pre_flight_checks_pass": bool(msg.pre_flight_checks_pass)}
                    else:
                        odometry[i] = odometry_snapshot(msg)
                ros.create_subscription(msg_type, f"{prefix}/{suffix}", receive, qos)
        pairs = []
        for index in (0, 1):
            prefix = f"/world/{world_name}/model/annotation_uav_{index}/link/camera_link/sensor"
            rgb, seg = prefix + "/camera/image", prefix + f"/{LABEL_SENSOR}/segmentation"
            pairs.append({"agent_id": f"uav_{index}", "rgb": rgb, "segmentation": seg})
            for topic in (rgb, seg + "/labels_map"):
                def capture(message, key=topic):
                    stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
                    with lock:
                        values = frames.setdefault(key, {})
                        if stamp not in values:
                            snapshot = Image()
                            snapshot.CopyFrom(message)
                            values[stamp] = snapshot
                            # Keep a recent bounded window, not the first eight
                            # potentially disjoint stamps after discovery.
                            while len(values) > 8:
                                del values[min(values)]
                if not node.subscribe(Image, topic, capture):
                    raise RuntimeError(f"Cannot subscribe {topic}")
                subscriptions.append(topic)
        summary["camera_pairs"] = pairs
        spawn("dds_agent", [agent, "udp4", "-p", str(port)], env, output)
        spawn("gazebo", ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", str(output / "world.sdf")], env, output)
        if gui:
            spawn("gui", ["gz", "sim", "-g", "-v", "3"], env, output)
        for index, instance in enumerate(INSTANCES):
            working = output / f"px4_work_{instance}"
            working.mkdir()
            px4_env = smoke_environment(env, partition=partition, domain=domain, port=port, index=index)
            px4_env["PX4_GZ_WORLD"] = world_name
            spawn(f"px4_{index}", [str(executable), str(rootfs / "etc"), "-i", str(instance), "-w", str(working), "-d"], px4_env, rootfs)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            failed = [f"{name} exited {p.returncode}" for name, p in processes if p.poll() is not None]
            if failed:
                raise RuntimeError("; ".join(failed))
            rclpy.spin_once(ros, timeout_sec=.1)
            if any(status["arming_state"] != 1 for status in latest_status.values()):
                raise RuntimeError("Expected disarmed PX4 throughout diagnostic")
            with lock:
                pair_counts = {pair["agent_id"]: len(set(frames.get(pair["rgb"], {})) &
                               set(frames.get(pair["segmentation"] + "/labels_map", {}))) for pair in pairs}
            healthy = (len(latest_status) == 2 and all(not s["failsafe"] for s in latest_status.values()))
            finite_odometry = len(odometry) == 2 and all(item["finite"] for item in odometry.values())
            now = time.monotonic()
            fresh = all(now - received_ros.get((kind, i), -math.inf) <= limit
                        for kind, limit in (("status", 2.), ("odometry", .5)) for i in (0, 1))
            if all(n >= 3 for n in pair_counts.values()) and all(n >= 2 for n in counts.values()) and healthy and finite_odometry and fresh:
                break
        else:
            raise TimeoutError(f"Startup streams missing/insufficient: ROS={counts}, pairs={pair_counts}")
        # Snapshot once under lock; callbacks must not change the chosen join.
        with lock:
            saved = {topic: dict(rows) for topic, rows in frames.items()}
        summary["pair_counts"] = {}
        summary["saved_captures"] = {}
        for pair in pairs:
            rgb, labels = saved[pair["rgb"]], saved[pair["segmentation"] + "/labels_map"]
            common = sorted(set(rgb) & set(labels))
            summary["pair_counts"][pair["agent_id"]] = len(common)
            for stamp in common:
                decoded = pair_rgb_instance(rgb_message=rgb[stamp], label_message=labels[stamp], rgb_topic=pair["rgb"],
                                            segmentation_topic=pair["segmentation"], label_topic=pair["segmentation"] + "/labels_map",
                                            encoding=GZ8_INSTANCE_ENCODING)
                resize_pair(decoded)  # Strict decoder + integer-label transform.
            stamp = common[-1]
            summary["saved_captures"][pair["agent_id"]] = {"stamp_ns": stamp, "files": {}}
            for kind, msg in (("rgb", rgb[stamp]), ("labels", labels[stamp])):
                filename = f"{pair['agent_id']}_{kind}_{stamp}.pb"
                payload = msg.SerializeToString()
                (output / filename).write_bytes(payload)
                summary["saved_captures"][pair["agent_id"]]["files"][kind] = {
                    "file": filename, "sha256": hashlib.sha256(payload).hexdigest()}
        summary["startup_passed"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    except KeyboardInterrupt:
        summary["error"] = "KeyboardInterrupt: diagnostic interrupted"
    finally:
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            summary["ros_counts"] = dict(counts)
            summary["latest_status"] = dict(latest_status)
            summary["latest_odometry"] = dict(odometry)
            if node is not None:
                for topic in subscriptions:
                    try:
                        node.unsubscribe(topic)
                    except Exception as exc:
                        summary.setdefault("cleanup_errors", []).append(str(exc))
            summary.setdefault("cleanup_errors", []).extend(stop_owned(processes))
            if ros is not None:
                ros.destroy_node()
            if rclpy_started:
                rclpy.shutdown()
        finally:
            for log in logs:
                log.close()
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            signal.signal(signal.SIGINT, previous_handler)
            summary["owned_processes_stopped"] = all(p.poll() is not None for _, p in processes)
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=120.)
    parser.add_argument("--ros-domain-id", type=int, default=77)
    parser.add_argument("--udp-only", action="store_true", help="Scoped FastDDS UDP transport diagnostic; no system config change")
    args = parser.parse_args()
    result = run(px4_root=args.px4_root, models=args.models, output=args.output_dir,
                 timeout_s=args.timeout_s, domain=args.ros_domain_id, udp_only=args.udp_only)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["startup_passed"] and result["owned_processes_stopped"] and not result.get("cleanup_errors") else 2)


if __name__ == "__main__":
    main()
