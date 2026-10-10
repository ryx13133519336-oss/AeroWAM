"""Safe manual and automatic launchers for the live FixedRole baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Sequence

from .paths import get_storage_paths
from .live_sim_manager import Px4GazeboConfig, Px4GazeboSession


TASK_ID = "live_dual_control_observation_diagnostic"
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
    if record.get("task_id") != TASK_ID:
        return SeedStatus(seed, path, "invalid", "unexpected task_id")
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
    """Return outputs that require manual review before another flight."""
    return [
        status for status in statuses
        if status.state in {"failed", "invalid"}
    ]


def ensure_new_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; inspect it or move it before rerunning"
        )


def runner_command(
    *, seed: int, output: Path, world: str, altitude: float,
    preposition_distance: float, duration: float,
    spawn_offset_ned: Sequence[float], python_executable: str = "python3",
) -> list[str]:
    if len(spawn_offset_ned) != 3:
        raise ValueError("spawn_offset_ned must contain N E D")
    return [
        python_executable, "-m", "benchmark.px4_gazebo.live_benchmark_runner",
        "--seed", str(int(seed)), "--world", world,
        "--altitude", str(float(altitude)),
        "--preposition-distance", str(float(preposition_distance)),
        "--duration", str(float(duration)), "--uav1-spawn-offset-ned",
        *(str(float(value)) for value in spawn_offset_ned), "--output", str(output),
    ]


def validate_episode_result(output_dir: Path, seed: int, returncode: int) -> SeedStatus:
    """Require both a clean runner exit and one successful formal output."""
    if returncode != 0:
        raise RuntimeError(f"seed {seed:03d} runner exited with code {returncode}")
    status = inspect_seed(output_dir, seed)
    if status.state != "success":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(
            f"seed {seed:03d} output state is {status.state}{detail}; stopping batch"
        )
    return status


def run_automatic_seed(args: argparse.Namespace, seed: int) -> SeedStatus:
    output = seed_output_path(args.output_dir, seed)
    ensure_new_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = runner_command(
        seed=seed, output=output, world=args.world, altitude=args.altitude,
        preposition_distance=args.preposition_distance, duration=args.duration,
        spawn_offset_ned=args.uav1_spawn_offset_ned,
        python_executable=sys.executable,
    )
    config = Px4GazeboConfig(
        world=args.world,
        spawn_offset_ned=tuple(args.uav1_spawn_offset_ned),
        px4_root=args.px4_root,
        log_root=args.auto_log_dir / f"seed_{seed:03d}",
        startup_timeout_s=args.startup_timeout,
        shutdown_timeout_s=args.shutdown_timeout,
    )
    print(f"auto-reset: preparing seed {seed:03d}")
    with Px4GazeboSession(config) as session:
        print(f"auto-reset: running seed {seed:03d}; session logs: {session.log_dir}")
        completed = subprocess.run(command, check=False)
    status = validate_episode_result(args.output_dir, seed, completed.returncode)
    print(f"auto-reset: seed {seed:03d} succeeded and simulator teardown completed")
    return status


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
            print("auto-reset: no pending seeds")
            return
        run_automatic_seed(args, seed)
        if one_seed_only:
            return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FixedRole live baseline with manual or automatic simulator reset"
    )
    parser.add_argument(
        "action",
        choices=("status", "command", "run-next", "run-auto-next", "run-auto"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output-dir", type=Path,
        default=get_storage_paths().episodes / "fixed_role",
    )
    parser.add_argument("--world", default="lawn")
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--preposition-distance", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--uav1-spawn-offset-ned", type=float, nargs=3,
        default=(0.0, 3.0, 0.0), metavar=("N", "E", "D"),
    )
    parser.add_argument("--px4-root", type=Path, default=Path.home() / "PX4-Autopilot")
    parser.add_argument(
        "--auto-log-dir", type=Path,
        default=get_storage_paths().px4_logs / "fixed_role_auto",
    )
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--shutdown-timeout", type=float, default=15.0)
    return parser


def _print_status(statuses: Sequence[SeedStatus]) -> None:
    for status in statuses:
        suffix = f" ({status.detail})" if status.detail else ""
        print(f"seed {status.seed:03d}: {status.state:7s}  {status.path}{suffix}")
    counts = {
        state: sum(item.state == state for item in statuses)
        for state in ("pending", "success", "failed", "invalid")
    }
    print("summary: " + ", ".join(f"{name}={value}" for name, value in counts.items()))


def main() -> None:
    args = _parser().parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds contains duplicates")
    statuses = inspect_seeds(args.output_dir, args.seeds)
    if args.action == "status":
        _print_status(statuses)
        return
    if args.action in {"run-auto-next", "run-auto"}:
        try:
            run_automatic(args, one_seed_only=args.action == "run-auto-next")
        except (FileExistsError, OSError, RuntimeError, TimeoutError) as exc:
            raise SystemExit(f"automatic reset stopped: {exc}") from exc
        return
    blockers = unresolved_statuses(statuses)
    if blockers:
        _print_status(statuses)
        blocked_seeds = ", ".join(f"{item.seed:03d}" for item in blockers)
        raise SystemExit(
            "failed/invalid seed output blocks further flights (seed "
            f"{blocked_seeds}); inspect and archive the file outside the formal "
            "fixed_role directory before retrying the same seed"
        )
    seed = next_pending_seed(statuses)
    if seed is None:
        _print_status(statuses)
        raise SystemExit("no pending seed; existing failed/invalid outputs are never overwritten")
    output = seed_output_path(args.output_dir, seed)
    ensure_new_output(output)
    command = runner_command(
        seed=seed, output=output, world=args.world, altitude=args.altitude,
        preposition_distance=args.preposition_distance, duration=args.duration,
        spawn_offset_ned=args.uav1_spawn_offset_ned,
        python_executable="python3" if args.action == "command" else sys.executable,
    )
    if args.action == "command":
        print(shlex.join(command))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"running seed {seed:03d}; this command launches exactly one flight episode")
    completed = subprocess.run(command, check=False)
    print(
        f"episode process exited with code {completed.returncode}. Confirm both vehicles "
        "are landed and disarmed, then reset PX4/Gazebo before the next seed."
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
