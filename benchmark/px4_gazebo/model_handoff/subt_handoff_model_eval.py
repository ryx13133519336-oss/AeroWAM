"""Recorded-observation core adapter diagnostic. Does not execute any action.

This runner cannot produce a handoff success rate. The private labels are
opened only by the independent source audit, never as policy observations.
"""

import argparse
import copy
import importlib
import json
from pathlib import Path
import time
import uuid

from .core._common import canonical
from .policy_client import PolicyClient
from .subt_handoff_inspect import inspect, read_rows
from .subt_handoff_model_io import model_input, validate_output
from .warehouse_scene import sha256

SCHEMA = "aerowam-subt-model-replay/1"


def verify_source(episode):
    if (episode / "batch_audit_error.txt").exists():
        raise ValueError("Known failed batch audit cannot be bypassed on another platform")
    frozen = json.loads((episode / "inspection.json").read_text(encoding="utf8"))
    if frozen.get("passed") is not True or frozen.get("message_delay_s") != .5 or not frozen.get("condition"):
        raise ValueError("Requires modern independently audited fixed-0.5s episode")
    fresh = inspect(episode)
    if fresh["files_sha256"] != frozen.get("files_sha256"):
        raise ValueError("Episode changed after its independent data audit")
    return fresh


def run(episode, output, *, method_id, method_kind, manifest, policy, max_steps=None):
    if method_kind not in {"repository_model", "external_model", "reference_rule"}:
        raise ValueError("Explicit method_kind required")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("Positive decision limit required")
    episode, output, manifest = Path(episode).resolve(), Path(output).resolve(), Path(manifest).resolve()
    source = verify_source(episode)
    metadata = json.loads((episode / "public/metadata.json").read_text(encoding="utf8"))
    identity = json.loads(manifest.read_text(encoding="utf8"))
    if identity.get("method_id") != method_id or identity.get("method_kind") != method_kind:
        raise ValueError("Manifest method_id/method_kind must match the run")
    source_audit_sha = sha256(episode / "inspection.json")
    rows = read_rows(episode / "public/observations.jsonl")
    if not rows:
        raise ValueError("No public observations")
    output.mkdir(parents=True, exist_ok=False)
    run_id = "subt-replay-" + uuid.uuid4().hex
    config = {"schema": SCHEMA, "run_id": run_id, "source_episode": str(episode),
        "source_audit_sha256": source_audit_sha, "method_id": method_id, "method_kind": method_kind,
        "model_manifest_sha256": sha256(manifest), "split": source["split"], "max_steps": max_steps,
        "message_mode": "recorded_teacher_forced", "actions_executed": False,
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                          ("subt_handoff_model_eval.py", "subt_handoff_model_io.py")}}
    (output / "run.json").write_bytes(canonical(config) + b"\n")
    (output / "model_manifest.json").write_bytes(manifest.read_bytes())
    history, errors, latencies = [], [], []
    count = 0
    with (output / "io_trace.jsonl").open("xb") as stream:
        for row in rows[:max_steps]:
            request = model_input(episode, row, episode_id=run_id, instruction=metadata["instruction"],
                                  messages=row["messages"], history=history)
            started = time.perf_counter()
            record = {"sample_index": row["sample_index"], "request": request, "response": None,
                      "latency_ms": None, "error": None}
            try:
                # Preserve the actual request even if a local adapter mutates its argument.
                response = policy(copy.deepcopy(request))
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                canonical(response)
                record["response"] = response
                validated = validate_output(response, request)
                latencies.append(record["latency_ms"])
                history.append({"timestamp_ns": row["stamp_ns"], "actions": validated["actions"]})
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(record["error"])
            stream.write(canonical(record) + b"\n")
            stream.flush()
            count += 1
            if record["error"]:
                break
    result = {"schema": SCHEMA, "run_id": run_id, "status": "model_failure" if errors else "complete",
        "steps": count, "source_steps": len(rows), "truncated": count < len(rows),
        "success": None, "mean_latency_ms": sum(latencies)/len(latencies) if latencies else None,
        "errors": errors, "closed_loop": False, "public_input_only": True,
        "independent_audit": False, "flight_admitted": False, "formal_score": False,
        "action_conditioned_world_model": False, "kinematic_proxy_source": True,
        "teacher_forced_messages": True, "role_assignment": None,
        "files_sha256": {p: sha256(output / p) for p in ("run.json", "model_manifest.json", "io_trace.jsonl")},
        "limitations": ["recorded frames do not respond to model actions",
                        "recorded delivered messages come from the privileged demonstrator",
                        "centralized dual RGB; no decentralized-policy claim",
                        "protocol smoke only; no task success or world-model dynamics score"]}
    (output / "replay_summary.json").write_bytes(canonical(result) + b"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--method-kind", choices=("repository_model", "external_model", "reference_rule"), required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--adapter", help="MODULE:FUNCTION(core_request) -> core_response; local timing is diagnostic only")
    group.add_argument("--policy-endpoint", help="HOST:PORT length-prefixed JSON transport")
    parser.add_argument("--policy-timeout-s", type=float, default=.4)
    args = parser.parse_args()
    if args.adapter:
        module, function = args.adapter.split(":", 1)
        policy = getattr(importlib.import_module(module), function)
    else:
        client = PolicyClient(args.policy_endpoint, timeout_s=args.policy_timeout_s)
        policy = lambda request: client.request(canonical(request))[0]
    print(json.dumps(run(args.episode, args.output, method_id=args.method_id, method_kind=args.method_kind,
        manifest=args.model_manifest, policy=policy, max_steps=args.max_steps), indent=2))


if __name__ == "__main__":
    main()
