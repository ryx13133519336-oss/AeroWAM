"""Fresh-simulator, paired-fixture model evaluation with invalid-run retries."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

from ..live_sim_manager import Px4GazeboConfig, Px4GazeboSession
from ..paths import get_storage_paths
from .fixtures import HandoffFixture, read_manifest


@dataclass(frozen=True)
class RunSpec:
    method: str
    training_seed: int
    endpoint: str
    checkpoint_id: str
    checkpoint: Path

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSpec":
        method = str(value["method"])
        if method not in {"joint_bc", "dual_vln_hier", "dual_wam_mpc"}:
            raise ValueError(f"unknown method: {method}")
        checkpoint = Path(str(value["checkpoint"]))
        return cls(
            method, int(value["training_seed"]),
            str(value.get("endpoint", "127.0.0.1:8765")),
            str(value.get("checkpoint_id", checkpoint.stem)), checkpoint,
        )

    @property
    def key(self) -> str:
        return f"{self.method}/seed_{self.training_seed}"


def load_run_specs(path: Path) -> list[RunSpec]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("runs config must be a non-empty JSON list")
    specs = [RunSpec.from_dict(item) for item in value]
    if len({item.key for item in specs}) != len(specs):
        raise ValueError("runs config contains duplicate method/training_seed pairs")
    return specs


def output_path(root: Path, spec: RunSpec, fixture: HandoffFixture) -> Path:
    return root / spec.method / f"seed_{spec.training_seed}" / fixture.split / f"{fixture.fixture_id}.jsonl"


def inspect(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "pending", ""
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", str(exc)
    if len(rows) != 1:
        return "invalid", f"expected one record, found {len(rows)}"
    outcome = rows[0].get("outcome", {})
    termination = str(outcome.get("termination_reason", "invalid"))
    if termination == "simulator_error":
        return "invalid", termination
    if not isinstance(outcome.get("success"), bool):
        return "invalid", "missing outcome.success"
    return ("success" if outcome["success"] else "model_failure"), termination


def schedule(fixtures: Iterable[HandoffFixture], specs: list[RunSpec], *, seed: int) -> list[tuple[HandoffFixture, RunSpec]]:
    result: list[tuple[HandoffFixture, RunSpec]] = []
    rng = random.Random(seed)
    for fixture in fixtures:
        block = list(specs)
        rng.shuffle(block)
        result.extend((fixture, spec) for spec in block)
    return result


@contextmanager
def policy_server(spec: RunSpec, args: argparse.Namespace):
    if args.external_policy_servers:
        yield
        return
    command = [
        sys.executable, "-m", "benchmark.px4_gazebo.model_handoff.policy_server",
        "--checkpoint", str(spec.checkpoint), "--device", args.model_device,
    ]
    host, separator, port_text = spec.endpoint.rpartition(":")
    if not separator:
        raise ValueError(f"invalid endpoint: {spec.endpoint}")
    command.extend(("--host", host, "--port", port_text))
    log_path = args.auto_log_dir / "policy_servers" / spec.method / f"seed_{spec.training_seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + args.policy_server_startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"policy server exited with code {process.returncode}; see {log_path}")
                try:
                    with socket.create_connection((host, int(port_text)), timeout=0.25):
                        break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError(f"policy server startup timeout; see {log_path}")
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def runner_command(args: argparse.Namespace, fixture: HandoffFixture, spec: RunSpec, output: Path) -> list[str]:
    return [
        sys.executable, "-m", "benchmark.px4_gazebo.live_handoff_runner",
        "--policy", "model", "--formal-model",
        "--fixture-manifest", str(args.manifest), "--fixture-id", fixture.fixture_id,
        "--model-method", spec.method, "--training-seed", str(spec.training_seed),
        "--checkpoint-id", spec.checkpoint_id, "--policy-endpoint", spec.endpoint,
        "--policy-timeout-ms", str(args.policy_timeout_ms),
        "--max-policy-timeouts", "3",
        "--bandwidth", "256", "--latency-ms", "0", "--drop-probability", "0",
        "--world", args.world, "--altitude", str(args.altitude), "--duration", str(args.duration),
        "--seed", str(fixture.seed), "--uav1-spawn-offset-ned", *(str(item) for item in args.uav1_spawn_offset_ned),
        "--startup-timeout", str(args.runner_startup_timeout),
        "--output", str(output),
    ]


def run_one(args: argparse.Namespace, fixture: HandoffFixture, spec: RunSpec) -> tuple[str, str]:
    final = output_path(args.output_dir, spec, fixture)
    state, detail = inspect(final)
    if state != "pending":
        return state, detail
    attempts = args.simulator_retries + 1
    for attempt in range(attempts):
        attempt_path = args.output_dir / ".attempts" / spec.method / f"seed_{spec.training_seed}" / fixture.fixture_id / f"attempt_{attempt:02d}.jsonl"
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        if attempt_path.exists():
            attempt_path.unlink()
        config = Px4GazeboConfig(
            world=args.world,
            spawn_offset_ned=tuple(args.uav1_spawn_offset_ned),
            px4_root=args.px4_root,
            log_root=args.auto_log_dir / spec.method / f"seed_{spec.training_seed}" / fixture.fixture_id / f"attempt_{attempt:02d}",
            startup_timeout_s=args.session_startup_timeout,
            shutdown_timeout_s=args.shutdown_timeout,
            show_gui=args.show_gui, gui_renderer=args.gui_renderer,
        )
        print(f"[{fixture.fixture_id}] {spec.key}: simulator attempt {attempt + 1}/{attempts}", flush=True)
        try:
            with policy_server(spec, args):
                with Px4GazeboSession(config):
                    subprocess.run(runner_command(args, fixture, spec, attempt_path), check=False)
        except (OSError, RuntimeError, TimeoutError) as exc:
            detail = f"session_error: {exc}"
            continue
        state, detail = inspect(attempt_path)
        if state == "invalid":
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        attempt_path.replace(final)
        return state, detail
    return "invalid", detail or "simulator retry budget exhausted"


def _fixtures(args: argparse.Namespace) -> list[HandoffFixture]:
    allowed = set(args.splits)
    result = [fixture for fixture in read_manifest(args.manifest) if fixture.split in allowed]
    if args.limit is not None:
        result = result[:args.limit]
    return result


def print_status(args: argparse.Namespace, fixtures: list[HandoffFixture], specs: list[RunSpec]) -> None:
    counts: dict[str, int] = {}
    for fixture, spec in schedule(fixtures, specs, seed=args.schedule_seed):
        state, _ = inspect(output_path(args.output_dir, spec, fixture))
        counts[state] = counts.get(state, 0) + 1
    print("summary: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired model-ready PX4/Gazebo Handoff batch")
    parser.add_argument("action", choices=("status", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("test_appearance", "test_occluder", "test_combined"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--schedule-seed", type=int, default=20261008)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulator-retries", type=int, default=2)
    parser.add_argument("--policy-timeout-ms", type=float, default=400.0)
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--external-policy-servers", action="store_true")
    parser.add_argument("--policy-server-startup-timeout", type=float, default=90.0)
    parser.add_argument("--world", default="lawn")
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--uav1-spawn-offset-ned", type=float, nargs=3, default=(0.0, 3.0, 0.0))
    parser.add_argument("--px4-root", type=Path, default=Path.home() / "PX4-Autopilot")
    parser.add_argument("--auto-log-dir", type=Path, default=get_storage_paths().px4_logs / "model_handoff")
    parser.add_argument("--session-startup-timeout", type=float, default=150.0)
    parser.add_argument("--runner-startup-timeout", type=float, default=45.0)
    parser.add_argument("--shutdown-timeout", type=float, default=15.0)
    parser.add_argument("--show-gui", action="store_true", help="watch the model episode in Gazebo")
    parser.add_argument("--gui-renderer", choices=("auto", "hardware", "software"), default="auto")
    args = parser.parse_args()
    fixtures, specs = _fixtures(args), load_run_specs(args.runs_config)
    if args.action == "status":
        print_status(args, fixtures, specs)
        return
    for fixture, spec in schedule(fixtures, specs, seed=args.schedule_seed):
        state, detail = run_one(args, fixture, spec)
        print(f"[{fixture.fixture_id}] {spec.key}: {state} ({detail})", flush=True)
        if state == "invalid":
            raise SystemExit("invalid simulator episode exhausted retry budget; formal batch stopped")
    print_status(args, fixtures, specs)


if __name__ == "__main__":
    main()
