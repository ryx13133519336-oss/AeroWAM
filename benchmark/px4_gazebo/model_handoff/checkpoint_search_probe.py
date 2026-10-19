"""Offline recorded-observation TCP probe, never a flight or task-success test.

Recorded states/actions came from another behavior policy. Inference on these
frames cannot establish the trajectory a checkpoint would produce in closed
loop. No private annotation/evaluator file is read or sent to the model.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .policy_client import PolicyClient
from .policy_server import InferencePolicy, ReusableThreadingTCPServer, _Handler, search_composite, validate_policy_seed
from .protocol import AGENT_IDS, ModelObservation
from .search_reporter import REPORTER_VERSION
from .search_approach_reporter import APPROACH_REPORTER_VERSION
from .role_free_search import ROLE_FREE_REPORTER_VERSION

PROBE_SCHEMA = "aerowam-checkpoint-search-probe/1"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_observations(path, limit):
    if type(limit) is not int or not 1 <= limit <= 120:
        raise ValueError("Probe limit must be 1..120 frames per recorded episode")
    observations = []
    previous = {a: -1 for a in AGENT_IDS}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            # Explicitly ignore recorded response and private diagnostic fields.
            observation = ModelObservation.from_wire(json.loads(line)["observation"])
            if observations and (observation.episode_id != observations[0].episode_id
                                 or observation.instruction != observations[0].instruction):
                raise ValueError("One constant episode/instruction required per input")
            for agent in AGENT_IDS:
                stamp = observation.image_timestamp_ns[agent]
                if stamp <= previous[agent]:
                    raise ValueError("Recorded camera frames must advance")
                previous[agent] = stamp
            observations.append(observation)
            if len(observations) == limit:
                break
    if not observations:
        raise ValueError("No recorded observations")
    return observations


def probe_observations(client, observations):
    rows, failure = [], None
    for index, observation in enumerate(observations, 1):
        try:
            response, latency = client.infer_grounded(observation)
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Invalid latency")
            rows.append({"step": index, "image_timestamp_ns": dict(observation.image_timestamp_ns),
                         "latency_ms": latency, "response": response.to_wire()})
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            # No retries on the same history and no concurrent call after a
            # timeout. Stop the entire probe; an error is not a model score.
            failure = {"step": index, "type": type(exc).__name__, "reason": str(exc)}
            break
    latencies = sorted(row["latency_ms"] for row in rows)
    intents = Counter(row["response"]["action"]["intent"] for row in rows)
    reports = Counter(r["event"] for row in rows for r in row["response"]["reports"])
    axes = {agent: [max((abs(row["response"]["action"]["waypoints"][agent][k]) for row in rows), default=0.)
                    for k in range(4)] for agent in AGENT_IDS}
    return rows, {"episode_id": observations[0].episode_id, "instruction": observations[0].instruction,
                  "frames_requested": len(observations), "frames_completed": len(rows), "failure": failure,
                  "transport_passed": failure is None,
                  "latency_ms": {"median": latencies[len(latencies) // 2] if latencies else None,
                                 "p95": latencies[max(0, math.ceil(.95 * len(latencies)) - 1)] if latencies else None,
                                 "max": max(latencies, default=None)},
                  "learned_intents": dict(intents), "auxiliary_report_events": dict(reports),
                  "max_abs_requested_delta": axes,
                  "task_success_evaluated": False, "closed_loop_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", default="joint_bc", choices=("joint_bc", "dual_vln_hier", "dual_wam_mpc"))
    parser.add_argument("--requests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-seed", type=int, help="Optional torch/planner seed")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--reporter", choices=("scan", "approach"), default="scan",
                        help="approach only changes auxiliary reports, never checkpoint actions")
    parser.add_argument("--role-mode", choices=("fixed", "unassigned"), default="fixed")
    args = parser.parse_args()
    if args.role_mode == "unassigned" and args.reporter != "approach":
        parser.error("Unassigned roles require --reporter approach")
    if args.output_dir.exists():
        parser.error("Output directory must be new; existing evidence is never overwritten")
    try:
        validate_policy_seed(args.policy_seed)
        inputs = [(path, load_observations(path, args.limit)) for path in args.requests]
        identities = [observations[0].episode_id for _, observations in inputs]
        if len(set(identities)) != len(identities):
            raise ValueError("Repeated episode IDs would contaminate model/reporter history")
        checkpoint_hash = digest(args.checkpoint)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"schema": PROBE_SCHEMA, "checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_sha256": checkpoint_hash, "requested_method": args.method,
               "reporter": (ROLE_FREE_REPORTER_VERSION if args.role_mode == "unassigned" else
                            APPROACH_REPORTER_VERSION if args.reporter == "approach" else REPORTER_VERSION),
               "role_mode": args.role_mode,
               "action_source": "checkpoint_only_no_rule_navigation_override",
               "timeout_ms": 400, "device": args.device, "policy_seed": args.policy_seed,
               "task_success_evaluated": False, "closed_loop_evaluated": False,
               "training_performed": False, "flight_started": False,
               "observation_history_source": "recorded_other_behavior_policy",
               "private_annotations_read": False, "episodes": [], "status": "initializing"}
    try:
        started = time.monotonic()
        model = InferencePolicy(args.checkpoint, device=args.device, seed=args.policy_seed)
        summary.update(model_config=asdict(model.config), initialization_wall_s=time.monotonic() - started)
        if model.config.method != args.method:
            raise ValueError("Checkpoint method differs from requested method")
        print(f"Loaded {model.config.method}; checkpoint read-only; no flight", flush=True)
        adapter, _ = search_composite(model, approach=args.reporter == "approach", role_mode=args.role_mode)
        with ReusableThreadingTCPServer(("127.0.0.1", 0), _Handler) as server:
            server.policy = adapter
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = PolicyClient(f"127.0.0.1:{server.server_address[1]}", timeout_s=.4)
                for index, (path, observations) in enumerate(inputs):
                    rows, result = probe_observations(client, observations)
                    result.update(source=str(path.resolve()), source_sha256=digest(path))
                    summary["episodes"].append(result)
                    output = args.output_dir / f"responses_{index:02d}.jsonl"
                    output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    if result["failure"]:
                        break
            finally:
                server.shutdown()
                thread.join(timeout=2.)
        summary["status"] = ("transport_passed" if len(summary["episodes"]) == len(inputs)
                             and all(r["transport_passed"] for r in summary["episodes"]) else "probe_failed")
    except Exception as exc:
        summary.update(status="probe_failed", error=f"{type(exc).__name__}:{exc}")
        print(summary["error"], flush=True)
    finally:
        summary["checkpoint_unchanged"] = digest(args.checkpoint) == checkpoint_hash
        if not summary["checkpoint_unchanged"]:
            summary["status"] = "probe_failed"
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    raise SystemExit(0 if summary["status"] == "transport_passed" else 2)


if __name__ == "__main__":
    main()
