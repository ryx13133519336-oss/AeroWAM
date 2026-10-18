"""One-world scripted camera sequence. Private diagnostic, never a flight."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from PIL import Image as PILImage, ImageDraw

from .annotation_models import verify
from .annotation_session_smoke import stop_owned
from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .search_occlusion_probe import measure
from .search_pilot_scene import OCCLUSION_WORLD, static_occlusion_world

SCHEMA = "aerowam-search-occlusion-sequence/1-draft"
PHASES = ("seen", "hidden", "reacquired")
MIN_SPAN_NS = 1_000_000_000
MAX_GAP_NS = 600_000_000
SETTLE_NS = 1_000_000_000
SAMPLE_NS = 250_000_000
TARGET_POSES = {"probe_red": (4., 1., 2.4, 0.), "probe_blue": (4., -1., 2.4, 0.),
                "probe_wall": (2., -.5, 2.4, 0.)}


def camera_poses(phase):
    if phase not in PHASES:
        raise ValueError("Unknown sequence phase")
    ys = {"seen": (6., -8.), "hidden": (0., -3.), "reacquired": (-8., 6.)}[phase]
    return {f"annotation_camera_{i}": (0., y, 2.4, math.atan2(-y, 4.))
            for i, y in enumerate(ys)}


def sequence_world(models):
    tree, pairs = static_occlusion_world(models, "search_revealed")
    ET.SubElement(tree.getroot().find("world"), "plugin",
                  filename="gz-sim-user-commands-system", name="gz::sim::systems::UserCommands")
    return tree, pairs


def stamp(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec


def check_scene(scene, phase):
    """Read-back evidence only; not a claim of pose at image exposure."""
    expected = {**TARGET_POSES, **camera_poses(phase)}
    models = {item.name: item for item in scene.model}
    if set(models) != set(expected) or len(scene.model) != len(expected):
        raise ValueError("Unexpected scene entities")
    for name, (x, y, z, yaw) in expected.items():
        pose = models[name].pose
        actual = (pose.position.x, pose.position.y, pose.position.z,
                  pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        target = (x, y, z, 0., 0., math.sin(yaw / 2), math.cos(yaw / 2))
        if any(not math.isfinite(a) or abs(a - b) > 1e-4 for a, b in zip(actual, target)):
            raise ValueError(f"Scene pose read-back mismatch: {name}")


def check_pose_packet(packet, phase):
    """SceneBroadcaster's ongoing pose stream, not cached scene/info."""
    expected = {**TARGET_POSES, **camera_poses(phase)}
    selected = [pose for pose in packet.pose if pose.name in expected]
    poses = {pose.name: pose for pose in selected}
    if set(poses) != set(expected) or len(selected) != len(expected):
        raise ValueError("Pose stream missing root entities")
    # Reuse the numeric check while retaining the raw stamped pose packet.
    from types import SimpleNamespace
    check_scene(SimpleNamespace(model=[SimpleNamespace(name=name, pose=pose)
                                       for name, pose in poses.items()]), phase)


def audit_timeline(phases):
    """Independently check order, sustained evidence and transition cutoffs."""
    if [p["phase"] for p in phases] != list(PHASES):
        raise ValueError("Exactly seen/hidden/reacquired phases are required")
    previous = -1
    spans = {}
    for phase in phases:
        boundary, gate = phase["boundary_clock_ns"], phase["gate_ns"]
        command_clock = phase["command_clock_ns"]
        if (type(boundary) is not int or type(gate) is not int
                or type(command_clock) is not int or command_clock < max(previous, 0)
                or boundary < command_clock or gate < boundary + SETTLE_NS):
            raise ValueError("Invalid transition/settling clock boundary")
        frames = phase["frames"]
        if len(frames) < 3:
            raise ValueError("Insufficient sustained frame evidence")
        first, last = frames[0]["stamp_ns"], -1
        for frame in frames:
            current = frame["stamp_ns"]
            if (type(current) is not int or current <= max(previous, last, gate)
                    or last >= 0 and current - last > MAX_GAP_NS):
                raise ValueError("Stale, reordered or gapped sequence evidence")
            if set(frame["agents"]) != {"uav_0", "uav_1"}:
                raise ValueError("Joint two-camera evidence required")
            for agent in frame["agents"].values():
                if type(agent["stamp_ns"]) is not int or agent["stamp_ns"] != current:
                    raise ValueError("Both cameras must share the exact capture timestamp")
            last = current
        if last - first < MIN_SPAN_NS:
            raise ValueError("Phase evidence must span at least one simulation second")
        spans[phase["phase"]] = {"frames": len(frames), "first_stamp_ns": first,
                                  "last_stamp_ns": last, "span_ns": last - first}
        previous = last
    return spans


def _dump_message(output, name, message):
    raw = message.SerializeToString()
    (output / name).write_bytes(raw)
    return {"file": name, "sha256": hashlib.sha256(raw).hexdigest()}


def _load_message(output, record, cls):
    name = record["file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Message must stay within diagnostic directory")
    raw = (output / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Raw message fingerprint mismatch")
    message = cls()
    message.ParseFromString(raw)
    return message


def run(models, output, timeout_s=90.):
    if not math.isfinite(timeout_s) or not 10 <= timeout_s <= 120:
        raise ValueError("Sequence timeout must be 10..120 wall seconds")
    if os.name == "nt":
        raise RuntimeError("Run capture in WSL/Linux with Gazebo bindings")
    from gz.transport13 import Node
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_v_pb2 import Pose_V

    tree, pairs = sequence_world(models)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tree.write(output / "world.sdf", encoding="utf-8", xml_declaration=True)
    resources = verify(models)
    partition = "aerowam_occlusion_sequence_" + uuid.uuid4().hex
    previous = os.environ.get("GZ_PARTITION")
    os.environ["GZ_PARTITION"] = partition
    summary = {"schema": SCHEMA, "partition": partition, "phases": [], "camera_pairs": pairs,
               "model_resources": resources, "capture_complete": False,
               "world_sha256": hashlib.sha256((output / "world.sdf").read_bytes()).hexdigest()}
    node, process, subscriptions = None, None, []
    buffers, clock, errors = {}, {"ns": -1, "received": -math.inf}, []
    pose_stream = {"message": None, "received": -math.inf}
    lock = threading.Lock()
    try:
        node = Node()
        def receive_clock(message):
            with lock:
                value = message.sim.sec * 1_000_000_000 + message.sim.nsec
                if value < clock["ns"]:
                    errors.append("Simulation clock moved backwards")
                clock.update(ns=value, received=time.monotonic())
        topic = f"/world/{OCCLUSION_WORLD}/clock"
        if not node.subscribe(Clock, topic, receive_clock):
            raise RuntimeError("Cannot subscribe clock")
        subscriptions.append(topic)
        def receive_pose(message):
            copied = Pose_V()
            copied.CopyFrom(message)
            with lock:
                pose_stream.update(message=copied, received=time.monotonic())
        topic = f"/world/{OCCLUSION_WORLD}/pose/info"
        if not node.subscribe(Pose_V, topic, receive_pose):
            raise RuntimeError("Cannot subscribe ongoing pose stream")
        subscriptions.append(topic)
        topics = [t for pair in pairs for t in (pair["rgb"], pair["segmentation"] + "/labels_map")]
        for topic in topics:
            buffers[topic] = {}
            def callback(message, key=topic):
                with lock:
                    rows = buffers[key]
                    value = stamp(message)
                    if value not in rows:
                        copied = Image()
                        copied.CopyFrom(message)
                        rows[value] = copied
                        while len(rows) > 12:
                            del rows[min(rows)]
            if not node.subscribe(Image, topic, callback):
                raise RuntimeError("Cannot subscribe paired image stream")
            subscriptions.append(topic)
        deadline = time.monotonic() + timeout_s
        with (output / "gazebo.log").open("xb") as log:
            process = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3",
                                        str(output / "world.sdf")], env=dict(os.environ), stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            summary.update(server_pid=process.pid, server_starts=1)
            def healthy():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Sequence wall-clock budget exhausted")
                if process.poll() is not None or errors:
                    raise RuntimeError(f"Gazebo exited or clock invalid: {process.poll()}, {errors}")
            while True:
                healthy()
                with lock:
                    ready = clock["ns"] > 0 and pose_stream["message"] is not None and all(buffers[t] for t in topics)
                if ready:
                    break
                time.sleep(.05)
            for phase in PHASES:
                healthy()
                record = {"phase": phase, "frames": [], "command": None}
                summary["phases"].append(record)
                with lock:
                    record["command_clock_ns"] = clock["ns"]
                if phase != "seen":
                    command = Pose_V()
                    for name, (x, y, z, yaw) in camera_poses(phase).items():
                        pose = command.pose.add(name=name)
                        pose.position.x, pose.position.y, pose.position.z = x, y, z
                        pose.orientation.z, pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                    command_file = _dump_message(output, f"{phase}_command.pb", command)
                    ok, response = node.request(f"/world/{OCCLUSION_WORLD}/set_pose_vector",
                                                command, Pose_V, Boolean, 2000)
                    record["command"] = {**command_file, "accepted": bool(ok and response.data)}
                    if not record["command"]["accepted"]:
                        raise RuntimeError("Scripted camera pose command rejected")
                # scene/info can retain authored poses after set_pose. Read
                # ongoing pose/info instead, then exclude a full sim second.
                while True:
                    healthy()
                    with lock:
                        packet = pose_stream["message"]
                        pose_age = time.monotonic() - pose_stream["received"]
                    try:
                        if packet is None or pose_age > 1. or stamp(packet) < record["command_clock_ns"]:
                            raise ValueError("Pose stream not fresh after command")
                        check_pose_packet(packet, phase)
                        break
                    except ValueError as exc:
                        record["last_scene_error"] = str(exc)
                        time.sleep(.05)
                record.pop("last_scene_error", None)
                record["poses_before"] = _dump_message(output, f"{phase}_poses_before.pb", packet)
                with lock:
                    record["boundary_clock_ns"] = max(clock["ns"], stamp(packet))
                record["gate_ns"] = record["boundary_clock_ns"] + SETTLE_NS
                last = -1
                while True:
                    healthy()
                    selected = None
                    with lock:
                        if time.monotonic() - clock["received"] > 1.:
                            raise RuntimeError("Stale simulation clock")
                        common = set.intersection(*(set(buffers[t]) for t in topics))
                        eligible = [s for s in common if s > record["gate_ns"] and s >= last + SAMPLE_NS
                                    and 0 <= clock["ns"] - s <= 250_000_000]
                        if eligible:
                            selected = max(eligible)
                            messages = {t: buffers[t][selected] for t in topics}
                    if selected is None:
                        time.sleep(.02)
                        continue
                    frame = {"stamp_ns": selected, "agents": {}}
                    for spec in pairs:
                        agent = spec["agent_id"]
                        rgb, labels = messages[spec["rgb"]], messages[spec["segmentation"] + "/labels_map"]
                        pair = pair_rgb_instance(rgb_message=rgb, label_message=labels, rgb_topic=spec["rgb"],
                                                 segmentation_topic=spec["segmentation"],
                                                 label_topic=spec["segmentation"] + "/labels_map",
                                                 encoding=GZ8_INSTANCE_ENCODING)
                        measure(pair, scene_case="search_blocked" if phase == "hidden" else "search_revealed",
                                agent_id=agent, render_id=f"{partition}-{agent}-{selected}")
                        frame["agents"][agent] = {"stamp_ns": selected,
                            "rgb": _dump_message(output, f"{phase}_{agent}_{selected}_rgb.pb", rgb),
                            "labels": _dump_message(output, f"{phase}_{agent}_{selected}_labels.pb", labels)}
                    record["frames"].append(frame)
                    last = selected
                    if len(record["frames"]) >= 3 and selected - record["frames"][0]["stamp_ns"] >= MIN_SPAN_NS:
                        break
                while True:
                    healthy()
                    with lock:
                        packet = pose_stream["message"]
                    check_pose_packet(packet, phase)
                    if stamp(packet) >= last:
                        break
                    time.sleep(.02)
                record["poses_after"] = _dump_message(output, f"{phase}_poses_after.pb", packet)
                print(f"phase={phase} joint_frames={len(record['frames'])} span_s="
                      f"{(last - record['frames'][0]['stamp_ns']) / 1e9:.3f}", flush=True)
            audit_timeline(summary["phases"])
            summary["capture_complete"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    except KeyboardInterrupt:
        summary["error"] = "KeyboardInterrupt: sequence interrupted"
    finally:
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if node is not None:
                for topic in subscriptions:
                    try:
                        node.unsubscribe(topic)
                    except Exception as exc:
                        summary.setdefault("cleanup_errors", []).append(str(exc))
            summary.setdefault("cleanup_errors", []).extend(stop_owned([("sequence", process)] if process else []))
            summary["owned_process_stopped"] = process is None or process.poll() is not None
        finally:
            if previous is None:
                os.environ.pop("GZ_PARTITION", None)
            else:
                os.environ["GZ_PARTITION"] = previous
            signal.signal(signal.SIGINT, previous_handler)
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def inspect(output):
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_v_pb2 import Pose_V
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("schema") != SCHEMA or not summary.get("capture_complete") or summary.get("error")
            or summary.get("cleanup_errors") or not summary.get("owned_process_stopped")):
        raise ValueError("Sequence capture/cleanup incomplete")
    if (summary.get("server_starts") != 1 or type(summary.get("server_pid")) is not int
            or summary["server_pid"] <= 0 or not summary.get("partition", "").startswith("aerowam_occlusion_sequence_")):
        raise ValueError("One owned server instance required")
    if hashlib.sha256((output / "world.sdf").read_bytes()).hexdigest() != summary["world_sha256"]:
        raise ValueError("Sequence world fingerprint mismatch")
    world = ET.parse(output / "world.sdf").getroot().find("world")
    expected = {**TARGET_POSES, **camera_poses("seen")}
    if (world is None or world.get("name") != OCCLUSION_WORLD or world.findall("include")
            or len(world.findall("model")) != len(expected)
            or {m.get("name") for m in world.findall("model")} != set(expected)):
        raise ValueError("Unexpected authored sequence world")
    for model in world.findall("model"):
        x, y, z, yaw = expected[model.get("name")]
        values = tuple(float(v) for v in model.findtext("pose", "").split())
        if (model.findtext("static") != "true" or len(values) != 6
                or any(not math.isfinite(a) or abs(a - b) > 1e-4
                       for a, b in zip(values, (x, y, z, 0., 0., yaw)))):
            raise ValueError("Authored initial pose/static configuration mismatch")
    spans = audit_timeline(summary["phases"])
    if (len(summary["camera_pairs"]) != 2
            or {s["agent_id"] for s in summary["camera_pairs"]} != {"uav_0", "uav_1"}):
        raise ValueError("Exactly two camera specs required")
    specs = {s["agent_id"]: s for s in summary["camera_pairs"]}
    result = {"schema": SCHEMA, "partition": summary["partition"], "world_sha256": summary["world_sha256"],
              "phase_spans": spans, "frames": [], "one_world_visibility_sequence_verified": False,
              "pose_at_exposure_verified": False, "continuous_motion_verified": False,
              "memory_use_verified": False, "flight_verified": False, "compositional_transfer_verified": False}
    sheet = PILImage.new("RGB", (512, 282 * 3), "white")
    for row, phase in enumerate(summary["phases"]):
        name = phase["phase"]
        for key in ("poses_before", "poses_after"):
            packet = _load_message(output, phase[key], Pose_V)
            check_pose_packet(packet, name)
            lower = phase["command_clock_ns"] if key == "poses_before" else phase["frames"][-1]["stamp_ns"]
            if stamp(packet) < lower or key == "poses_before" and stamp(packet) > phase["boundary_clock_ns"]:
                raise ValueError("Pose stream clock boundary mismatch")
        if name == "seen":
            if phase["command"] is not None:
                raise ValueError("Initial phase must use authored starting poses")
        else:
            command = _load_message(output, phase["command"], Pose_V)
            if not phase["command"]["accepted"] or len(command.pose) != 2:
                raise ValueError("Camera pose command not accepted")
            # Reuse numeric checks with the requested camera poses and the
            # fixed target/wall poses from the observed packet.
            packet = deepcopy(_load_message(output, phase["poses_before"], Pose_V))
            camera_commands = {p.name: p for p in command.pose}
            if set(camera_commands) != set(camera_poses(name)):
                raise ValueError("Only the two camera rigs may be commanded")
            for pose in packet.pose:
                if pose.name in camera_commands:
                    pose.CopyFrom(camera_commands[pose.name])
            check_pose_packet(packet, name)
        for index, frame in enumerate(phase["frames"]):
            measured = {"phase": name, "stamp_ns": frame["stamp_ns"], "agents": {}}
            for agent, record in frame["agents"].items():
                spec = specs[agent]
                expected = f"/world/{OCCLUSION_WORLD}/model/annotation_camera_{agent[-1]}/link/camera_link/sensor"
                if spec["rgb"] != expected + "/camera/image" or spec["segmentation"] != expected + "/private_instance/segmentation":
                    raise ValueError("Unexpected camera topic mapping")
                pair = pair_rgb_instance(rgb_message=_load_message(output, record["rgb"], Image),
                                         label_message=_load_message(output, record["labels"], Image),
                                         rgb_topic=spec["rgb"], segmentation_topic=spec["segmentation"],
                                         label_topic=spec["segmentation"] + "/labels_map", encoding=GZ8_INSTANCE_ENCODING)
                if pair.timestamp_ns != frame["stamp_ns"]:
                    raise ValueError("Raw image capture timestamp mismatch")
                measured["agents"][agent] = measure(pair,
                    scene_case="search_blocked" if name == "hidden" else "search_revealed",
                    agent_id=agent, render_id=f"{summary['partition']}-{agent}-{pair.timestamp_ns}")
                if index == 0:
                    preview = PILImage.fromarray(resize_pair(pair).rgb)
                    sheet.paste(preview, (int(agent[-1]) * 256, row * 282 + 26))
                    ImageDraw.Draw(sheet).text((int(agent[-1]) * 256 + 4, row * 282 + 5),
                                              f"{name} / {agent} / {pair.timestamp_ns / 1e9:.2f}s", fill="black")
            result["frames"].append(measured)
    result["one_world_visibility_sequence_verified"] = True
    sheet.save(output / "sequence.png")
    (output / "inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not args.inspect_only:
        if args.models is None:
            parser.error("--models required")
        captured = run(args.models, args.output_dir, args.timeout_s)
        if not captured["capture_complete"] or captured.get("error"):
            compact = {k: v for k, v in captured.items() if k not in {"phases", "model_resources"}}
            compact["phase_frames"] = {p["phase"]: len(p["frames"]) for p in captured["phases"]}
            print(json.dumps(compact, indent=2))
            raise SystemExit(2)
    result = inspect(args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
