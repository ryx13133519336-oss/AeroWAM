"""Fresh-session privileged-oracle data collection and fixture validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from ..live_sim_manager import Px4GazeboConfig, Px4GazeboSession
from ..paths import get_storage_paths
from .fixtures import HandoffFixture, read_manifest
from .dataset import validate_episode_directory


class SimulatorFixtureError(RuntimeError):
    """A fixture could not be completed because the simulator was unhealthy."""


class TraceQualityFixtureError(RuntimeError):
    """Task may succeed while receive timing is unsuitable for gated collection."""


class HostClockPreflightError(RuntimeError):
    """A host clock diagnostic failed before any simulator was launched."""


def clock_preflight(args: argparse.Namespace) -> Path:
    from .clock_probe import run_probe
    diagnostics = args.output_dir / ".diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    path = diagnostics / f"host_clock_probe_{uuid.uuid4().hex}.json"
    report = run_probe(path, duration_s=args.clock_preflight_seconds)
    if report["issues"]:
        raise HostClockPreflightError(
            f"host clock preflight rejected ({', '.join(report['issues'])}); "
            f"no simulator launched; evidence: {path}"
        )
    print(f"host clock preflight passed for {args.clock_preflight_seconds:g} s; "
          f"in-run trace checks remain required; evidence: {path}", flush=True)
    return path


def _trace_quality(args: argparse.Namespace, episode: Path, *, persist: bool) -> dict | None:
    if not getattr(args, "record_execution_trace", False):
        return None
    from .trace_quality import report_episode
    report = report_episode(episode, persist=persist)
    if getattr(args, "require_trace_receive_quality", False) and not report["receive_quality_pass"]:
        raise TraceQualityFixtureError(
            f"episode {report['episode_id']} receive quality rejected: {', '.join(report['issues'])}; "
            "task outcome preserved; precise transition alignment remains unverified"
        )
    return report

def _result(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "missing", ""
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", str(exc)
    if len(rows) != 1:
        return "invalid", f"expected one row, found {len(rows)}"
    outcome = rows[0].get("outcome", {})
    if outcome.get("termination_reason") == "simulator_error":
        return "invalid", "simulator_error"
    return ("success" if outcome.get("success") is True else "infeasible"), str(outcome.get("termination_reason"))


def _command(
    args: argparse.Namespace, fixture: HandoffFixture, output: Path,
    record_dir: Path | None,
) -> list[str]:
    command = [
        sys.executable, "-m", "benchmark.px4_gazebo.live_handoff_runner",
        "--policy", "oracle", "--hide-markers",
        "--fixture-manifest", str(args.manifest), "--fixture-id", fixture.fixture_id,
        "--bandwidth", "256", "--latency-ms", "0", "--drop-probability", "0",
        "--world", args.world, "--altitude", "2", "--duration", "24",
        "--seed", str(fixture.seed), "--uav1-spawn-offset-ned", "0", "3", "0",
        "--startup-timeout", str(args.runner_startup_timeout),
        "--oracle-perturb-probability", str(args.perturb_probability if args.mode == "collect" else 0.0),
        "--output", str(output),
    ]
    if record_dir is not None:
        command.extend(("--record-dir", str(record_dir)))
        if getattr(args, "record_execution_trace", False):
            command.append("--record-execution-trace")
    return command


def run_fixture(args: argparse.Namespace, fixture: HandoffFixture) -> None:
    final = args.output_dir / fixture.split / f"{fixture.fixture_id}.jsonl"
    state, detail = _result(final)
    if state == "success":
        if getattr(args, "record_execution_trace", False):
            episode_id = json.loads(final.read_text(encoding="utf-8"))["episode_id"]
            validate_episode_directory(args.dataset_dir / episode_id, require_expert_supervision=True, require_execution_trace=True)
            _trace_quality(args, args.dataset_dir / episode_id, persist=False)
        print(f"{fixture.fixture_id}: already verified")
        return
    if state == "infeasible":
        raise RuntimeError(f"oracle previously failed fixture {fixture.fixture_id}: {detail}")
    quality_rejected = False
    attempts_dir = args.output_dir / ".attempts" / fixture.fixture_id
    # Retain all prior attempts across restarts, including successful flights
    # rejected solely by the opt-in quality gate.
    existing = [int(path.stem.removeprefix("attempt_")) for path in attempts_dir.glob("attempt_*.jsonl")
                if path.stem.removeprefix("attempt_").isdigit()]
    existing.extend(int(path.name.removeprefix("dataset_")) for path in attempts_dir.glob("dataset_*")
                    if path.name.removeprefix("dataset_").isdigit())
    first_attempt = max(existing, default=-1) + 1
    for attempt in range(first_attempt, first_attempt + args.simulator_retries + 1):
        quality_rejected = False
        attempt_path = args.output_dir / ".attempts" / fixture.fixture_id / f"attempt_{attempt:02d}.jsonl"
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_dataset = attempt_path.parent / f"dataset_{attempt:02d}"
        config = Px4GazeboConfig(
            world=args.world, spawn_offset_ned=(0.0, 3.0, 0.0),
            px4_root=args.px4_root,
            log_root=args.auto_log_dir / fixture.fixture_id / f"attempt_{attempt:02d}",
            startup_timeout_s=args.session_startup_timeout,
            shutdown_timeout_s=args.shutdown_timeout,
            show_gui=args.show_gui, gui_renderer=args.gui_renderer,
            record_sim_clock=bool(getattr(args, "record_execution_trace", False)),
        )
        session_error: str | None = None
        try:
            with Px4GazeboSession(config):
                subprocess.run(
                    _command(
                        args, fixture, attempt_path,
                        attempt_dataset if args.mode == "collect" else None,
                    ),
                    check=False,
                )
        except (OSError, RuntimeError, TimeoutError) as exc:
            session_error = f"simulator session error: {exc}"
            print(
                f"{fixture.fixture_id}: attempt {attempt:02d} failed before a "
                f"valid episode ({session_error})",
                flush=True,
            )
        state, detail = _result(attempt_path)
        if state == "success" and session_error is None:
            if args.mode == "collect":
                episode_dirs = [path for path in attempt_dataset.iterdir() if path.is_dir()]
                if len(episode_dirs) != 1:
                    # A reused attempt directory can contain stale recordings from
                    # an earlier simulator retry. Select the recording referenced by
                    # the successful episode JSON instead of treating stale siblings
                    # as a new multi-episode recording.
                    episode_id = ""
                    try:
                        rows = [
                            json.loads(line)
                            for line in attempt_path.read_text(encoding="utf-8").splitlines()
                            if line
                        ]
                        if len(rows) == 1:
                            episode_id = str(rows[0].get("episode_id", ""))
                    except (OSError, json.JSONDecodeError):
                        pass
                    matching_dirs = [path for path in episode_dirs if path.name == episode_id]
                    if len(matching_dirs) == 1:
                        episode_dirs = matching_dirs
                    else:
                        raise RuntimeError(
                            f"expected one recorded dataset for {fixture.fixture_id}, "
                            f"found {len(episode_dirs)} (episode_id={episode_id or 'unknown'})"
                        )
                try:
                    validation = validate_episode_directory(episode_dirs[0], require_expert_supervision=True, require_execution_trace=bool(getattr(args, "record_execution_trace", False)))
                except (OSError, ValueError, KeyError) as exc:
                    detail = f"recorded dataset invalid: {exc}"
                    print(f"{fixture.fixture_id}: {detail}; retrying simulator", flush=True)
                    continue
                print(f"{fixture.fixture_id}: dataset validated ({validation['samples']} samples)", flush=True)
                try:
                    quality = _trace_quality(args, episode_dirs[0], persist=True)
                except TraceQualityFixtureError as exc:
                    quality_rejected = True
                    detail = str(exc)
                    print(f"{fixture.fixture_id}: {detail}; retaining attempt and retrying", flush=True)
                    continue
                if quality is not None:
                    print(f"{fixture.fixture_id}: receive_quality="
                          f"{'pass' if quality['receive_quality_pass'] else 'flagged'} "
                          f"(precise_transition_supervision_ready=false)", flush=True)
                assert args.dataset_dir is not None
                args.dataset_dir.mkdir(parents=True, exist_ok=True)
                destination = args.dataset_dir / episode_dirs[0].name
                if destination.exists():
                    raise RuntimeError(f"dataset destination already exists: {destination}")
                shutil.move(str(episode_dirs[0]), destination)
            final.parent.mkdir(parents=True, exist_ok=True)
            attempt_path.replace(final)
            print(f"{fixture.fixture_id}: oracle success", flush=True)
            return
        if state == "infeasible":
            raise RuntimeError(f"oracle cannot complete fixture {fixture.fixture_id}: {detail}")
        if session_error is not None:
            detail = session_error
    error_type = TraceQualityFixtureError if quality_rejected else SimulatorFixtureError
    raise error_type(
        f"fixture {fixture.fixture_id} exhausted simulator retries: {detail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect oracle demonstrations or validate test fixtures")
    parser.add_argument("--mode", choices=("collect", "validate"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--record-execution-trace", action="store_true", help="opt in to published-controls dataset v2")
    parser.add_argument("--require-trace-receive-quality", action="store_true",
                        help="keep receive-quality failures in attempts; does not certify precise transitions")
    parser.add_argument("--clock-preflight-seconds", type=float, default=20.,
                        help="1..60 s host clock check before gated batch (only with require-trace-receive-quality)")
    parser.add_argument("--perturb-probability", type=float, default=0.15)
    parser.add_argument("--simulator-retries", type=int, default=2)
    parser.add_argument("--world", default="lawn")
    parser.add_argument("--px4-root", type=Path, default=Path.home() / "PX4-Autopilot")
    parser.add_argument("--auto-log-dir", type=Path, default=get_storage_paths().px4_logs / "model_handoff_oracle")
    parser.add_argument("--session-startup-timeout", type=float, default=150.0)
    parser.add_argument("--runner-startup-timeout", type=float, default=45.0)
    parser.add_argument("--shutdown-timeout", type=float, default=15.0)
    parser.add_argument("--show-gui", action="store_true", help="watch oracle collection or fixture validation")
    parser.add_argument("--gui-renderer", choices=("auto", "hardware", "software"), default="auto")
    args = parser.parse_args()
    if args.mode == "collect" and args.dataset_dir is None:
        raise SystemExit("collect mode requires --dataset-dir")
    if args.record_execution_trace and args.mode != "collect":
        raise SystemExit("record-execution-trace requires collect mode")
    if args.require_trace_receive_quality and not args.record_execution_trace:
        raise SystemExit("require-trace-receive-quality requires record-execution-trace")
    if not math.isfinite(args.clock_preflight_seconds) or not 1 <= args.clock_preflight_seconds <= 60:
        raise SystemExit("clock-preflight-seconds must be 1..60")
    fixtures = [item for item in read_manifest(args.manifest) if item.split in set(args.splits)]
    if args.limit is not None:
        fixtures = fixtures[:args.limit]
    if args.require_trace_receive_quality and any(
        _result(args.output_dir / fixture.split / f"{fixture.fixture_id}.jsonl")[0] in {"missing", "invalid"}
        for fixture in fixtures
    ):
        try:
            clock_preflight(args)
        except HostClockPreflightError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            raise SystemExit(2) from exc
    completed = 0
    simulator_errors = 0
    quality_errors = 0
    for fixture in fixtures:
        try:
            run_fixture(args, fixture)
        except SimulatorFixtureError as exc:
            simulator_errors += 1
            print(f"{fixture.fixture_id}: simulator_error; continuing batch: {exc}", flush=True)
        except TraceQualityFixtureError as exc:
            quality_errors += 1
            print(f"{fixture.fixture_id}: trace_quality_error; continuing batch: {exc}", flush=True)
        else:
            completed += 1
    print(
        f"oracle {args.mode} complete: total={len(fixtures)}, "
        f"completed={completed}, simulator_error={simulator_errors}"
        + (f", trace_quality_error={quality_errors}" if args.record_execution_trace else "")
    )
    if args.require_trace_receive_quality and (simulator_errors or quality_errors):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
