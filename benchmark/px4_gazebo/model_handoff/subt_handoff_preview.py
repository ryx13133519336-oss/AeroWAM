"""GUI preview of the stair-crossing handoff route, without PX4 or flight.

This is intentionally separate from the industrial-target trolley preview:
the target follows the audited two-story stair route and both camera proxies
follow the role-free handoff plan. It is a visual demonstration, not evidence
of aircraft dynamics or a learned policy.
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
import uuid

from .industrial_targets import parse_component
from .subt_handoff_collect import NAMES, command_worker, poses_for, prepare_scene
from .subt_handoff_expert import audit_expert, duration, planned_state
from .subt_handoff_route import build_route, load_triangles, audit_route
from .subt_urban import stop_owned
from .subt_urban_scene import WORLD
from .warehouse_scene import sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def preview(package, models, output, *, component, speed=.4, swapped=False,
            teammate="normal", seconds=100., feedback_timeout_s=12.):
    if not 5 <= seconds <= 300 or not 0 < speed <= .4:
        raise ValueError("Preview requires 5..300 seconds and speed <=0.4 m/s")
    if not 2 <= feedback_timeout_s <= 30:
        raise ValueError("Feedback timeout must be 2..30 seconds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    route = build_route(load_triangles(package))
    geometry = audit_route(load_triangles(package), route)
    expert_audit = audit_expert(load_triangles(package), route, speed=speed,
                                swapped=swapped, teammate=teammate)
    summary = {
        "schema": "aerowam-subt-stair-gui-preview/1",
        "preview_only": True, "flight_commands_sent": 0,
        "flight_admitted": False, "policy_evaluated": False,
        "speed_m_s": speed, "preview_seconds": seconds,
        "route_length_m": route["distance_m"][-1],
        "target_vertical_span_m": max(p[2] for p in route["centers_enu"]) - min(p[2] for p in route["centers_enu"]),
        "geometry": geometry, "expert_audit": expert_audit,
        "commands_sent": 0, "reached_upper_floor": False,
        "owned_processes_stopped": False,
    }
    if not geometry["passed"] or not expert_audit["passed"]:
        summary["error"] = "Offline stair geometry or expert audit failed"
        write_json(output / "preview_summary.json", summary)
        return summary

    tree, contract = prepare_scene(package, models, route, component=component,
                                   swapped=swapped, teammate=teammate)
    tree.write(output / "world.sdf", encoding="utf8", xml_declaration=True)
    contract["world_sha256"] = sha256(output / "world.sdf")
    write_json(output / "scene_contract.private.json", contract)

    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V

    previous_partition = os.environ.get("GZ_PARTITION")
    env = dict(os.environ, GZ_PARTITION="aerowam_stair_gui_" + uuid.uuid4().hex)
    env.pop("GZ_SIM_SERVER_CONFIG_PATH", None)
    os.environ["GZ_PARTITION"] = env["GZ_PARTITION"]
    node = Node()
    lock = threading.Lock()
    feedback = {"stamp_ns": -1, "poses": None, "wall": 0.}
    subscriptions = []
    processes = []
    worker = None
    started = time.monotonic()

    def receive_pose(message):
        values = {p.name: [p.position.x, p.position.y, p.position.z]
                  for p in message.pose if p.name in NAMES}
        if set(values) == set(NAMES):
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
            with lock:
                feedback.update(stamp_ns=stamp, poses=values, wall=time.monotonic())

    try:
        topic = f"/world/{WORLD}/pose/info"
        if not node.subscribe(Pose_V, topic, receive_pose):
            raise RuntimeError("Pose subscription failed")
        subscriptions.append(topic)
        log = (output / "gazebo.log").open("x", encoding="utf8")
        simulator = subprocess.Popen(["gz", "sim", "-r", "-v", "3", str(output / "world.sdf")],
                                     env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        processes.append(simulator)
        worker = subprocess.Popen([sys.executable, "-B", "-u", "-m",
                                   "benchmark.px4_gazebo.model_handoff.subt_handoff_collect", "--worker"],
                                  env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=log, text=True, bufsize=1, start_new_session=True)
        processes.append(worker)
        deadline = time.monotonic() + max(60., seconds * 8.)
        while True:
            with lock:
                stamp, actual, last_wall = feedback["stamp_ns"], feedback["poses"], feedback["wall"]
            if simulator.poll() is not None:
                raise RuntimeError(f"Gazebo exited early with code {simulator.returncode}")
            if actual is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Stair preview pose stream did not start")
            time.sleep(.05)
        origin = stamp
        last_command = origin - 50_000_000
        last_elapsed = 0.
        while True:
            with lock:
                stamp, actual, last_wall = feedback["stamp_ns"], feedback["poses"], feedback["wall"]
            if simulator.poll() is not None:
                raise RuntimeError(f"Gazebo exited during preview with code {simulator.returncode}")
            if actual is None or time.monotonic() - last_wall > feedback_timeout_s:
                raise RuntimeError("Stair preview pose feedback became stale")
            elapsed = max(0., (stamp - origin) / 1e9)
            if stamp < origin or elapsed < last_elapsed:
                raise RuntimeError("Gazebo preview simulation timestamp regressed")
            last_elapsed = elapsed
            if elapsed >= seconds:
                break
            if stamp - last_command >= 50_000_000:
                state = planned_state(route, elapsed, speed=speed, swapped=swapped, teammate=teammate)
                command_worker(worker, poses_for(state))
                summary["commands_sent"] += 1
                last_command = stamp
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded stair preview timed out")
            time.sleep(.02)
        summary.update({"elapsed_sim_s": last_elapsed,
                        "reached_upper_floor": last_elapsed >= duration(route, speed)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}:{exc}"
    finally:
        if worker is not None:
            try:
                worker.stdin.close()
                worker.wait(timeout=3.)
            except Exception:
                try:
                    worker.kill()
                    worker.wait(timeout=3.)
                except Exception:
                    pass
        for process in reversed(processes):
            try:
                stop_owned(process)
            except Exception as exc:
                summary.setdefault("cleanup_errors", []).append(str(exc))
        for subscription in subscriptions:
            node.unsubscribe(subscription)
        if previous_partition is None:
            os.environ.pop("GZ_PARTITION", None)
        else:
            os.environ["GZ_PARTITION"] = previous_partition
        summary["owned_processes_stopped"] = all(p.poll() is not None for p in processes)
        summary["wall_seconds"] = time.monotonic() - started
        write_json(output / "preview_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--target", type=parse_component, default=parse_component("blue:cube"))
    parser.add_argument("--swapped", action="store_true")
    parser.add_argument("--teammate", choices=("normal", "delayed"), default="normal")
    parser.add_argument("--speed", type=float, default=.4)
    parser.add_argument("--preview-seconds", type=float, default=100.)
    parser.add_argument("--feedback-timeout-s", type=float, default=12.)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = preview(args.package, args.models, args.output_dir, component=args.target,
                     speed=args.speed, swapped=args.swapped, teammate=args.teammate,
                     seconds=args.preview_seconds, feedback_timeout_s=args.feedback_timeout_s)
    print(json.dumps({k: v for k, v in result.items() if k not in {"geometry", "expert_audit"}},
                     indent=2), flush=True)
    raise SystemExit(0 if result.get("owned_processes_stopped") and "error" not in result else 2)


if __name__ == "__main__":
    main()
