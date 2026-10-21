"""SubT Urban prepare / export / preview / static capture / disarmed startup."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .subt_urban_scene import CAMERAS, scene, prepare
from .warehouse_scene import sha256
from .industrial_targets import parse_component


def stop_owned(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.)
        except ProcessLookupError:
            process.wait(timeout=5.)


def preview(world, *, px4_root=None, seconds=None, move_targets=False, speed=.15, headless=False):
    if seconds is not None and not 1 <= seconds <= 60:
        raise ValueError("Bounded preview must be 1..60 seconds")
    if move_targets and (seconds is None or not 0 < speed <= .25):
        raise ValueError("Moving preview requires bounded seconds and speed 0..0.25 m/s")
    if headless and seconds is None:
        raise ValueError("Headless preview requires bounded seconds")
    env = dict(os.environ, GZ_PARTITION="aerowam_subt_preview_" + uuid.uuid4().hex)
    env.pop("GZ_SIM_SERVER_CONFIG_PATH", None)
    if px4_root is not None:
        root = Path(px4_root).resolve()
        plugins = root / "build/px4_sitl_default/src/modules/simulation/gz_plugins"
        if not plugins.is_dir():
            raise ValueError("PX4 Gazebo plugin directory missing")
        for key, extra in (("GZ_SIM_SYSTEM_PLUGIN_PATH", plugins),
                           ("GZ_SIM_RESOURCE_PATH", root / "Tools/simulation/gz/models")):
            env[key] = str(extra) + (os.pathsep + env[key] if env.get(key) else "")
    print("Gazebo preview; no PX4 or flight commands. Log: " + str(Path(world).parent / "gazebo_gui.log"), flush=True)
    command = ["gz", "sim", "-r", "-v", "3"]
    if headless:
        command += ["-s", "--headless-rendering"]
    motion = None
    code = 2
    with (Path(world).parent / "gazebo_gui.log").open("xb") as log:
        process = subprocess.Popen([*command, str(world)], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if move_targets:
            with (Path(world).parent / "target_motion.log").open("xb") as log:
                motion = subprocess.Popen([sys.executable, "-B", "-u", "-m",
                    "benchmark.px4_gazebo.model_handoff.industrial_target_motion",
                    "--contract", str(Path(world).parent / "scene_contract.private.json"),
                    "--output-dir", str(Path(world).parent / "motion"), "--duration", str(seconds),
                    "--speed", str(speed), "--feedback-timeout-s", "12"],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 60. + seconds * 6
            while motion.poll() is None:
                if process.poll() is not None:
                    raise RuntimeError(f"Gazebo exited before motion verification: {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded target motion preview timed out")
                time.sleep(.1)
            code = motion.returncode
            if code == 0:
                report = json.loads((Path(world).parent / "motion/motion.private.json").read_text())
                if not report.get("motion_verified") or not report.get("owned_worker_stopped"):
                    code = 2
        else:
            code = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        code = 0
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(f"Preview failed: {type(exc).__name__}: {exc}", flush=True)
        code = 2
    finally:
        stop_owned(motion)
        stop_owned(process)
        summary = {"exit_code": code, "move_targets": move_targets, "partition": env["GZ_PARTITION"],
                   "owned_simulator_stopped": process.poll() is not None,
                   "owned_motion_process_stopped": motion is None or motion.poll() is not None,
                   "flight_commands_sent": 0, "policy_evaluated": False}
        (Path(world).parent / "preview_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return code


def export(package, models, output, *, case="overview", aircraft=False, target_components=None):
    tree, contract = scene(package, models, case=case, aircraft=aircraft, target_components=target_components)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tree.write(output / "world.sdf", encoding="utf-8", xml_declaration=True)
    contract["world_sha256"] = sha256(output / "world.sdf")
    (output / "scene_contract.private.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "export", "preview", "capture", "startup"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case", choices=tuple(CAMERAS), default="overview")
    parser.add_argument("--aircraft", action="store_true", help="Export dynamic aircraft; not flight admission")
    parser.add_argument("--px4-root", type=Path)
    parser.add_argument("--agent", type=Path)
    parser.add_argument("--timeout-s", type=float, default=60.)
    parser.add_argument("--ros-domain-id", type=int, default=83)
    parser.add_argument("--gui", action="store_true", help="GUI during disarmed startup")
    parser.add_argument("--preview-seconds", type=float, help="Optional bounded GUI smoke; default runs until closed")
    parser.add_argument("--move-targets", action="store_true", help="Bounded kinematic trolley preview, no PX4")
    parser.add_argument("--target-speed", type=float, default=.15, help="Target speed in simulation m/s, <=0.25")
    parser.add_argument("--headless", action="store_true", help="Bounded preview without GUI")
    parser.add_argument("--target-a", type=parse_component, default=parse_component("red:sphere"), help="COLOR:SHAPE")
    parser.add_argument("--target-b", type=parse_component, default=parse_component("blue:cube"), help="COLOR:SHAPE")
    args = parser.parse_args()
    if (args.move_targets or args.headless) and args.mode != "preview":
        parser.error("--move-targets and --headless apply only to preview")
    if args.move_targets and args.preview_seconds is None:
        args.preview_seconds = 30.
    if args.headless and args.preview_seconds is None:
        parser.error("--headless requires --preview-seconds")
    if args.move_targets and args.aircraft:
        parser.error("Moving admission preview uses fixed cameras; aircraft/flight admission is separate")
    if args.mode == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        result = prepare(args.source, args.package)
        print(json.dumps({"schema": result["schema"], "files": len(result["files"])}, indent=2))
        return
    if args.models is None or args.output_dir is None:
        parser.error("--models and --output-dir required")
    if args.mode in {"export", "preview"}:
        if args.mode == "preview" and args.aircraft and args.px4_root is None:
            parser.error("aircraft preview requires --px4-root for Gazebo plugins; no PX4 is launched")
        contract = export(args.package, args.models, args.output_dir, case=args.case, aircraft=args.aircraft,
                          target_components=(args.target_a, args.target_b))
        print(json.dumps({"world_sha256": contract["world_sha256"], "flight_admitted": False}, indent=2))
        if args.mode == "preview":
            raise SystemExit(preview(args.output_dir / "world.sdf", px4_root=args.px4_root,
                                     seconds=args.preview_seconds, move_targets=args.move_targets,
                                     speed=args.target_speed, headless=args.headless))
    elif args.mode == "capture":
        if args.aircraft:
            parser.error("capture is static; use startup for disarmed aircraft")
        from .annotation_probe import run
        tree, contract = scene(args.package, args.models, case=args.case, target_components=(args.target_a, args.target_b))
        result = run(args.output_dir, args.timeout_s, args.case,
                     prepared_scene=(tree, contract["camera_pairs"], contract))
        print(json.dumps({k: result.get(k) for k in ("pair_counts", "error", "owned_process_stopped")}, indent=2))
        if result.get("error") or result.get("cleanup_errors") or not result.get("owned_process_stopped"):
            raise SystemExit(2)
        from .subt_urban_probe import inspect_capture
        inspect_capture(args.output_dir)
    elif args.mode == "startup":
        if args.px4_root is None or args.agent is None:
            parser.error("startup requires --px4-root and --agent")
        from .annotation_session_smoke import run
        from .search_pilot import inspect_agent
        inspect_agent(args.agent)
        result = run(px4_root=args.px4_root, models=args.models, output=args.output_dir,
                     timeout_s=args.timeout_s, domain=args.ros_domain_id, udp_only=True,
                     prepared_scene=scene(args.package, args.models, aircraft=True, target_components=(args.target_a, args.target_b)),
                     agent_path=args.agent, gui=args.gui)
        print(json.dumps({k: result.get(k) for k in ("startup_passed", "flight_ready", "motion_commands_sent",
                                                    "pair_counts", "ros_counts", "error", "owned_processes_stopped")}, indent=2))
        raise SystemExit(0 if result["startup_passed"] and result["owned_processes_stopped"] and not result.get("cleanup_errors") else 2)


if __name__ == "__main__":
    main()
