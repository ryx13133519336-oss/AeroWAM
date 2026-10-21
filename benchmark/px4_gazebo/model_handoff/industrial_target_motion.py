"""Bounded kinematic target trolley motion; never sends PX4 flight commands.

This is a scripted environment perturbation, not wheel/drive dynamics or a
learned target policy. Mount and insert are changed by one Pose_V request.
Actual same-stamp poses and contacts are independently observed fail-closed.
"""

import argparse
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np

from .industrial_targets import IDENTITIES, POSITIONS, SUPPORT_BOUNDS, from_configuration, fingerprint
from .subt_urban_geometry import load_triangles, segment_hits, triangles_touch_box
from .warehouse_scene import sha256

ROUTES = {identity: ((xyz[0], xyz[1], 0.), (xyz[0] + 2., xyz[1], 0.)) for identity, xyz in POSITIONS.items()}
MAX_SPEED = .25


def sample_route(route, elapsed, *, speed=.15):
    if (not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(speed) or not 0 < speed <= MAX_SPEED):
        raise ValueError("Finite elapsed and target speed 0..0.25 m/s required")
    a, b = np.asarray(route[0], dtype=float), np.asarray(route[1], dtype=float)
    length = float(np.linalg.norm(b - a))
    if length <= 0:
        raise ValueError("Nonempty target route required")
    distance = elapsed * speed % (2 * length)
    distance = min(distance, 2 * length - distance)
    return a + (b - a) * distance / length


def audit_routes(package):
    triangles = load_triangles(package)
    bounds = np.asarray(SUPPORT_BOUNDS)
    boxes = []
    details = {}
    for identity, route in ROUTES.items():
        route = np.asarray(route)
        low = route.min(axis=0) + bounds[0] - .10
        high = route.max(axis=0) + bounds[1] + .10
        low[2] = .03  # Legitimate floor/wheel contact; not clearance for flight.
        clear = not triangles_touch_box(triangles, low, high)
        floor_samples = []
        for fraction in np.linspace(0., 1., 21):
            point = route[0] + fraction * (route[1] - route[0])
            hits = segment_hits(triangles, point + [0, 0, .3], point - [0, 0, .3])
            floor_samples.append(bool(len(hits)) and bool(np.any(np.abs(.3 - .6 * hits) <= .02)))
        if not clear or not all(floor_samples):
            raise ValueError(f"Target trolley route not admitted: {identity}; clear={clear}, floor={sum(floor_samples)}/21")
        boxes.append((low, high))
        details[identity] = {"waypoints_floor_enu": route.tolist(), "swept_low_enu": low.tolist(),
                             "swept_high_enu": high.tolist(), "surface_clear": clear, "floor_samples": len(floor_samples)}
    if np.all(boxes[0][1] >= boxes[1][0]) and np.all(boxes[1][1] >= boxes[0][0]):
        raise ValueError("Trolley routes overlap")
    return {"schema": "aerowam-industrial-motion-clearance/1", "routes": details,
            "target_motion_geometry_pass": True, "flight_admitted": False,
            "scope": "inflated_target_trolley_sweep_and_sampled_floor_support_only"}


def poses_for(elapsed, *, speed=.15):
    result = {}
    for index, identity in enumerate(IDENTITIES):
        xyz = sample_route(ROUTES[identity], elapsed, speed=speed)
        result[f"industrial_mount_{'ab'[index]}"] = xyz
        result[identity] = xyz + [0, 0, 2.]
    return result


def validate_feedback(row):
    values = row["positions_enu"]
    if set(values) != set(poses_for(0.)):
        raise ValueError("Incomplete target pose feedback")
    for index, identity in enumerate(IDENTITIES):
        mount = f"industrial_mount_{'ab'[index]}"
        xyz = np.asarray(values[mount], dtype=float)
        route = np.asarray(ROUTES[identity])
        if (xyz.shape != (3,) or not np.isfinite(xyz).all()
                or np.any(xyz < route.min(axis=0) - .01) or np.any(xyz > route.max(axis=0) + .01)):
            raise ValueError("Target mount left admitted route")
        delta = np.asarray(values[identity], dtype=float) - xyz
        if not np.allclose(delta, [0, 0, 2.], atol=.005, rtol=0):
            raise ValueError("Mount/component pose coherence failed")


def pose_worker():
    """Keep blocking transport requests outside the subscriber's interpreter."""
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.boolean_pb2 import Boolean
    node = Node()
    for line in sys.stdin:
        try:
            command = json.loads(line)
            positions = command["positions"]
            validate_feedback({"positions_enu": positions})
            request = Pose_V()
            for name, xyz in positions.items():
                pose = request.pose.add()
                pose.name = name
                pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
                pose.orientation.w = 1.
            ok, response = node.request(command["service"], request, Pose_V, Boolean, 1000)
            result = {"accepted": bool(ok and response.data)}
        except Exception as exc:
            result = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, allow_nan=False), flush=True)


def request_in_worker(worker, service, positions):
    if worker.poll() is not None:
        raise RuntimeError("Target pose helper exited")
    worker.stdin.write(json.dumps({"service": service, "positions": {k: list(map(float, v)) for k, v in positions.items()}},
                                  allow_nan=False) + "\n")
    worker.stdin.flush()
    if not select.select([worker.stdout], [], [], 2.)[0]:
        raise TimeoutError("Target pose helper response timeout")
    line = worker.stdout.readline()
    if not line or not json.loads(line).get("accepted"):
        raise RuntimeError("Target pose update rejected")


def inspect_world(contract_path):
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    from .subt_urban_scene import WORLD, verify
    if contract.get("world_name") != WORLD or sha256(contract_path.parent / "world.sdf") != contract.get("world_sha256"):
        raise ValueError("World contract mismatch")
    values = from_configuration(contract.get("industrial_targets"))
    if fingerprint(values) != contract.get("target_config_sha256"):
        raise ValueError("Target configuration mismatch")
    root = ET.parse(contract_path.parent / "world.sdf")
    mesh = root.findtext("world/model[@name='subt_two_story']/link/collision/geometry/mesh/uri")
    parsed = urlparse(mesh or "")
    if parsed.scheme != "file":
        raise ValueError("Local pinned building required")
    package = Path(unquote(parsed.path)).parents[2]
    if verify(package) != contract["subt"]:
        raise ValueError("SubT resource binding mismatch")
    return contract, audit_routes(package)


def run(contract_path, output, *, duration=30., speed=.15, feedback_timeout_s=8.):
    if (not 1 <= duration <= 600 or not 0 < speed <= MAX_SPEED
            or not 2 <= feedback_timeout_s <= 30 or not os.environ.get("GZ_PARTITION")):
        raise ValueError("Explicit Gazebo partition, duration 1..600s, speed <=0.25 m/s and feedback timeout 2..30s required")
    contract, clearance = inspect_world(contract_path)
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.msgs10.image_pb2 import Image
    from .industrial_target_capture import MotionCapture
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    node = Node()
    capture = MotionCapture(contract)
    lock = threading.Lock()
    snapshots, unsafe_contacts, subscriptions = [], [], []
    worker = None
    contact_messages = 0
    expected = poses_for(0.)
    world = contract["world_name"]
    started = time.monotonic()
    result = {"schema": "aerowam-industrial-target-motion/1", "clearance": clearance,
              "speed_m_s": speed, "duration_s": duration, "motion_verified": False,
              "flight_commands_sent": 0, "physics_driven_wheels": False, "requests": 0}
    result.update({"time_base": "gazebo_simulation_stamp", "command_trace": [],
                   "contact_fault_probe_verified": False, "flight_admitted": False})

    def receive_poses(message):
        values = {p.name: [p.position.x, p.position.y, p.position.z] for p in message.pose if p.name in expected}
        if set(values) == set(expected):
            with lock:
                snapshots.append({"stamp_ns": message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec,
                                  "wall_elapsed_s": time.monotonic() - started, "positions_enu": values})
                if len(snapshots) > 10000:
                    del snapshots[:1000]

    def receive_contacts(message):
        nonlocal contact_messages
        with lock:
            contact_messages += 1
        for contact in message.contact:
            names = contact.collision1.name + " " + contact.collision2.name
            if "industrial_" not in names:
                continue
            # Floor contact only: near z=0 with a vertical normal. The building
            # is one mesh containing floors AND walls; never blanket-ignore it.
            floor = ("subt_two_story" in names and "industrial_mount_" in names
                     and bool(contact.position) and len(contact.position) == len(contact.normal)
                     and all(abs(p.z) <= .04 and abs(n.z) >= .98 for p, n in zip(contact.position, contact.normal)))
            if not floor:
                with lock:
                    unsafe_contacts.append(names)

    try:
        for spec in contract["camera_pairs"]:
            for channel, topic in enumerate((spec["rgb"], spec["segmentation"] + "/labels_map")):
                def receive_image(message, agent=spec["agent_id"], channel=channel):
                    capture.receive(agent, channel, message)
                if not node.subscribe(Image, topic, receive_image):
                    raise RuntimeError("Cannot subscribe moving target image stream")
                subscriptions.append(topic)
        for topic, message_type, callback in ((f"/world/{world}/pose/info", Pose_V, receive_poses),
                                               *((binding["topic"], Contacts, receive_contacts) for binding in contract["contact_sensors"])):
            if not node.subscribe(message_type, topic, callback):
                raise RuntimeError("Cannot subscribe target pose/contact stream")
            subscriptions.append(topic)
        # Wait for real feedback, not a successful service reply alone.
        deadline = time.monotonic() + 20.
        while time.monotonic() < deadline:
            with lock:
                ready = bool(snapshots)
            if ready:
                break
            time.sleep(.05)
        else:
            raise TimeoutError("Target pose feedback missing")
        with lock:
            origin = snapshots[-1]["stamp_ns"]
        capture.start(origin)
        worker = subprocess.Popen([sys.executable, "-B", "-u", "-m", __spec__.name, "--worker"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        deadline = time.monotonic() + 30. + duration * 6
        last_command_stamp = origin - 50_000_000
        previous_stamp = origin
        while True:
            with lock:
                recent = snapshots[-1] if snapshots else None
                if unsafe_contacts:
                    raise RuntimeError("Unexpected target/support collision")
            if recent is None or time.monotonic() - started - recent["wall_elapsed_s"] > feedback_timeout_s:
                raise RuntimeError("Stale target pose feedback")
            validate_feedback(recent)
            stamp = recent["stamp_ns"]
            if stamp < previous_stamp:
                raise RuntimeError("Gazebo simulation timestamp regressed")
            previous_stamp = stamp
            elapsed = (stamp - origin) / 1e9
            if elapsed >= duration:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Target motion simulation clock did not progress")
            if stamp - last_command_stamp >= 50_000_000:
                positions = poses_for(elapsed, speed=speed)
                request_in_worker(worker, f"/world/{world}/set_pose_vector", positions)
                result["command_trace"].append({"stamp_ns": stamp, "elapsed_sim_s": elapsed,
                                                "positions_enu": {k: v.tolist() for k, v in positions.items()}})
                result["requests"] += 1
                last_command_stamp = stamp
            time.sleep(.01)
        result["elapsed_sim_s"] = elapsed
        with lock:
            saved = list(snapshots)
        for index, identity in enumerate(IDENTITIES):
            mount = f"industrial_mount_{'ab'[index]}"
            samples = np.array([row["positions_enu"][identity] for row in saved])
            if len(samples) < 3 or np.ptp(samples[:, 0]) < min(.20, duration * speed * .5):
                raise RuntimeError("Target did not move far enough in observed Gazebo poses")
            for row in saved:
                validate_feedback(row)
        result["rendering"] = capture.inspect(output, saved)
        result["motion_verified"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if worker is not None:
            worker.stdin.close()
            try:
                worker.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=3.)
        result["owned_worker_stopped"] = worker is None or worker.poll() is not None
        for topic in subscriptions:
            node.unsubscribe(topic)
        with lock:
            result["pose_samples"] = snapshots
            result["unexpected_contacts"] = unsafe_contacts
            result["contact_feedback_messages"] = contact_messages
        result["collision_safety_scope"] = "offline_swept_geometry_only; contact_fault_probe_not_verified"
        (output / "motion.private.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--speed", type=float, default=.15)
    parser.add_argument("--feedback-timeout-s", type=float, default=8.)
    args = parser.parse_args()
    result = run(args.contract, args.output_dir, duration=args.duration, speed=args.speed,
                 feedback_timeout_s=args.feedback_timeout_s)
    print(json.dumps({k: v for k, v in result.items() if k not in {"pose_samples", "clearance"}}, indent=2))
    raise SystemExit(0 if result["motion_verified"] else 2)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        pose_worker()
    else:
        main()
