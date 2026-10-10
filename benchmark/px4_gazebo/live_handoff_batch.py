"""Safe manual and automatic launchers for live Handoff episodes.

The runner owns one episode. This module adds the repeatable outer loop:
inspect one output per seed, start a fresh PX4/Gazebo session, run the episode,
validate its terminal record, and only then continue to the next seed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable, Sequence

from .contract import HandoffTask
from .live_sim_manager import Px4GazeboConfig, Px4GazeboSession
from .paths import get_storage_paths


TASK_ID = HandoffTask().task_id
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class SeedStatus:
    seed: int
    path: Path
    state: str
    detail: str = ""


def seed_output_path(output_dir: Path, seed: int) -> Path:
    return Path(output_dir) / f"seed_{int(seed):03d}.jsonl"


def inspect_seed(output_dir: Path, seed: int) -> SeedStatus:
    path = seed_output_path(output_dir, seed)
    if not path.exists():
        return SeedStatus(seed, path, "pending")
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        return SeedStatus(seed, path, "invalid", f"cannot read: {exc}")
    if len(lines) != 1:
        return SeedStatus(seed, path, "invalid", f"expected 1 record, found {len(lines)}")
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return SeedStatus(seed, path, "invalid", f"invalid JSON: {exc}")
    if not isinstance(record, dict):
        return SeedStatus(seed, path, "invalid", "record is not an object")
    if record.get("task_id") != TASK_ID:
        return SeedStatus(seed, path, "invalid", f"unexpected task_id={record.get('task_id')!r}")
    if record.get("seed") != int(seed):
        return SeedStatus(seed, path, "invalid", f"record seed={record.get('seed')!r}")
    outcome = record.get("outcome")
    if not isinstance(outcome, dict) or not isinstance(outcome.get("success"), bool):
        return SeedStatus(seed, path, "invalid", "missing boolean outcome.success")
    state = "success" if outcome["success"] else "failed"
    return SeedStatus(seed, path, state, str(outcome.get("termination_reason", "unknown")))


def inspect_seeds(output_dir: Path, seeds: Iterable[int]) -> list[SeedStatus]:
    return [inspect_seed(output_dir, seed) for seed in seeds]


def next_pending_seed(statuses: Iterable[SeedStatus]) -> int | None:
    return next((status.seed for status in statuses if status.state == "pending"), None)


def unresolved_statuses(statuses: Iterable[SeedStatus]) -> list[SeedStatus]:
    return [status for status in statuses if status.state in {"failed", "invalid"}]


def ensure_new_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; inspect it or move it before rerunning"
        )


def runner_command(
    *, seed: int, output: Path, policy: str, bandwidth: int, latency_ms: float,
    drop_probability: float, world: str, altitude: float, duration: float,
    spawn_offset_ned: Sequence[float], countdown: float, prestream: float,
    telemetry_timeout: float, camera_timeout: float, startup_timeout: float,
    takeoff_timeout: float, landing_timeout: float,
    visual_handoff: bool = False,
    python_executable: str = "python3",
) -> list[str]:
    if len(spawn_offset_ned) != 3:
        raise ValueError("spawn_offset_ned must contain N E D")
    command = [
        python_executable, "-m", "benchmark.px4_gazebo.live_handoff_runner",
        "--policy", policy, "--bandwidth", str(int(bandwidth)),
        "--latency-ms", str(float(latency_ms)), "--drop-probability", str(float(drop_probability)),
        "--altitude", str(float(altitude)), "--duration", str(float(duration)),
        "--seed", str(int(seed)), "--world", world, "--countdown", str(float(countdown)),
        "--prestream", str(float(prestream)), "--telemetry-timeout", str(float(telemetry_timeout)),
        "--camera-timeout", str(float(camera_timeout)), "--startup-timeout", str(float(startup_timeout)),
        "--takeoff-timeout", str(float(takeoff_timeout)), "--landing-timeout", str(float(landing_timeout)),
        "--uav1-spawn-offset-ned", *(str(float(value)) for value in spawn_offset_ned),
        "--output", str(output),
    ]
    if visual_handoff:
        command.append("--visual-handoff")
    return command


def validate_episode_result(output_dir: Path, seed: int, returncode: int) -> SeedStatus:
    if returncode != 0:
        raise RuntimeError(f"seed {seed:03d} runner exited with code {returncode}")
    status = inspect_seed(output_dir, seed)
    if status.state != "success":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(f"seed {seed:03d} output state is {status.state}{detail}; stopping batch")
    return status


def _runner_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "policy": args.policy, "bandwidth": args.bandwidth,
        "latency_ms": args.latency_ms, "drop_probability": args.drop_probability,
        "world": args.world, "altitude": args.altitude, "duration": args.duration,
        "spawn_offset_ned": args.uav1_spawn_offset_ned, "countdown": args.countdown,
        "prestream": args.prestream, "telemetry_timeout": args.telemetry_timeout,
        "camera_timeout": args.camera_timeout, "startup_timeout": args.runner_startup_timeout,
        "takeoff_timeout": args.takeoff_timeout, "landing_timeout": args.landing_timeout,
        "visual_handoff": args.visual_handoff,
    }


def run_automatic_seed(args: argparse.Namespace, seed: int) -> SeedStatus:
    output = seed_output_path(args.output_dir, seed)
    ensure_new_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = runner_command(seed=seed, output=output, python_executable=sys.executable, **_runner_kwargs(args))
    config = Px4GazeboConfig(
        world=args.world, spawn_offset_ned=tuple(args.uav1_spawn_offset_ned),
        px4_root=args.px4_root, log_root=args.auto_log_dir / f"seed_{seed:03d}",
        startup_timeout_s=args.session_startup_timeout, shutdown_timeout_s=args.shutdown_timeout,
        show_gui=(args.show_gui or args.visual_handoff) and not args.no_gui,
        gui_renderer=args.gui_renderer,
    )
    print(f"auto-reset: preparing Handoff seed {seed:03d}")
    with Px4GazeboSession(config) as session:
        print(f"auto-reset: running seed {seed:03d}; session logs: {session.log_dir}")
        completed = subprocess.run(command, check=False)
    status = validate_episode_result(args.output_dir, seed, completed.returncode)
    print(f"auto-reset: Handoff seed {seed:03d} succeeded and simulator teardown completed")
    return status


def check_gazebo_gui(args: argparse.Namespace) -> None:
    """Verify the visual session without arming either vehicle."""
    config = Px4GazeboConfig(
        world=args.world, spawn_offset_ned=tuple(args.uav1_spawn_offset_ned),
        px4_root=args.px4_root, log_root=args.auto_log_dir / "gui_preflight",
        startup_timeout_s=args.session_startup_timeout,
        shutdown_timeout_s=args.shutdown_timeout, show_gui=True,
        gui_renderer=args.gui_renderer,
    )
    with Px4GazeboSession(config) as session:
        print(f"Gazebo GUI preflight ready; logs: {session.log_dir}")
        time.sleep(args.gui_hold_seconds)
    print("Gazebo GUI preflight completed; simulator teardown completed")


def run_automatic(args: argparse.Namespace, *, one_seed_only: bool) -> None:
    while True:
        statuses = inspect_seeds(args.output_dir, args.seeds)
        blockers = unresolved_statuses(statuses)
        if blockers:
            _print_status(statuses)
            blocked = ", ".join(f"{status.seed:03d}" for status in blockers)
            raise RuntimeError(f"failed/invalid seed blocks automatic batch: {blocked}")
        seed = next_pending_seed(statuses)
        if seed is None:
            _print_status(statuses)
            print("auto-reset: no pending Handoff seeds")
            return
        run_automatic_seed(args, seed)
        if one_seed_only:
            return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Handoff batch with manual or automatic PX4/Gazebo reset")
    parser.add_argument("action", choices=("status", "command", "run-next", "run-auto-next", "run-auto", "check-gui"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=get_storage_paths().episodes / "handoff_batch")
    parser.add_argument("--policy", choices=("fixed", "oracle", "rule"), default="rule")
    parser.add_argument("--bandwidth", type=int, default=256)
    parser.add_argument("--latency-ms", type=float, default=100.0)
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--world", default="lawn")
    parser.add_argument(
        "--visual-handoff", action="store_true",
        help="pass the explicit Gazebo handoff demonstration trajectory to the runner",
    )
    gui_options = parser.add_mutually_exclusive_group()
    gui_options.add_argument("--show-gui", action="store_true", help="show the Gazebo scene without enabling demo markers")
    gui_options.add_argument(
        "--no-gui", action="store_true",
        help="keep the visual handoff trajectory without starting the Gazebo GUI",
    )
    parser.add_argument("--gui-renderer", choices=("auto", "hardware", "software"), default="auto",
                        help="GUI OpenGL path; auto uses hardware with WSLg compatibility settings")
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--countdown", type=float, default=5.0)
    parser.add_argument("--prestream", type=float, default=2.0)
    parser.add_argument("--telemetry-timeout", type=float, default=2.0)
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--runner-startup-timeout", type=float, default=30.0)
    parser.add_argument("--takeoff-timeout", type=float, default=30.0)
    parser.add_argument("--landing-timeout", type=float, default=45.0)
    parser.add_argument("--uav1-spawn-offset-ned", type=float, nargs=3, default=(0.0, 3.0, 0.0), metavar=("N", "E", "D"))
    parser.add_argument("--px4-root", type=Path, default=Path.home() / "PX4-Autopilot")
    parser.add_argument("--auto-log-dir", type=Path, default=get_storage_paths().px4_logs / "handoff_auto")
    parser.add_argument("--session-startup-timeout", type=float, default=90.0)
    parser.add_argument("--shutdown-timeout", type=float, default=15.0)
    parser.add_argument("--gui-hold-seconds", type=float, default=30.0)
    return parser


def _print_status(statuses: Sequence[SeedStatus]) -> None:
    for status in statuses:
        suffix = f" ({status.detail})" if status.detail else ""
        print(f"seed {status.seed:03d}: {status.state:7s}  {status.path}{suffix}")
    counts = {state: sum(item.state == state for item in statuses) for state in ("pending", "success", "failed", "invalid")}
    print("summary: " + ", ".join(f"{name}={value}" for name, value in counts.items()))


def main() -> None:
    args = _parser().parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds contains duplicates")
    if any(seed < 0 for seed in args.seeds):
        raise SystemExit("--seeds must be non-negative")
    if args.gui_hold_seconds < 0 or args.gui_hold_seconds > 300:
        raise SystemExit("--gui-hold-seconds must be between 0 and 300")
    if args.no_gui and not args.visual_handoff:
        raise SystemExit("--no-gui requires --visual-handoff")
    if args.action == "check-gui":
        try:
            check_gazebo_gui(args)
        except (FileExistsError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise SystemExit(f"Gazebo GUI preflight failed: {exc}") from exc
        return
    statuses = inspect_seeds(args.output_dir, args.seeds)
    if args.action == "status":
        _print_status(statuses)
        return
    if args.action in {"run-auto-next", "run-auto"}:
        try:
            run_automatic(args, one_seed_only=args.action == "run-auto-next")
        except (FileExistsError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise SystemExit(f"automatic reset stopped: {exc}") from exc
        return
    blockers = unresolved_statuses(statuses)
    if blockers:
        _print_status(statuses)
        blocked = ", ".join(f"{item.seed:03d}" for item in blockers)
        raise SystemExit(f"failed/invalid seed output blocks further flights (seed {blocked}); inspect and archive it outside the formal Handoff directory")
    seed = next_pending_seed(statuses)
    if seed is None:
        _print_status(statuses)
        raise SystemExit("no pending seed; existing outputs are never overwritten")
    output = seed_output_path(args.output_dir, seed)
    ensure_new_output(output)
    command = runner_command(seed=seed, output=output, python_executable="python3" if args.action == "command" else sys.executable, **_runner_kwargs(args))
    if args.action == "command":
        print(shlex.join(command))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"running Handoff seed {seed:03d}; this launches exactly one flight episode")
    completed = subprocess.run(command, check=False)
    print(f"episode process exited with code {completed.returncode}. Confirm both vehicles are landed and disarmed before manually resetting for the next seed.")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
