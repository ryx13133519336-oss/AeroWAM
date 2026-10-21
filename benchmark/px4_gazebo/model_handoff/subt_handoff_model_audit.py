"""Read-only reconstruction of SubT core model requests and response bindings."""

import argparse
import json
from pathlib import Path

from .core._common import canonical
from .subt_handoff_inspect import read_rows
from .subt_handoff_model_eval import SCHEMA, verify_source
from .subt_handoff_model_io import model_input, validate_output
from .warehouse_scene import sha256


def audit(run_dir):
    root = Path(run_dir).resolve()
    config = json.loads((root / "run.json").read_text(encoding="utf8"))
    saved = json.loads((root / "replay_summary.json").read_text(encoding="utf8"))
    if config["schema"] != SCHEMA or saved["schema"] != SCHEMA or config["run_id"] != saved["run_id"]:
        raise ValueError("Run schema/identity mismatch")
    for name, expected in config["source_sha256"].items():
        if name not in {"subt_handoff_model_eval.py", "subt_handoff_model_io.py"} or sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError("Run source changed")
    for name in ("run.json", "model_manifest.json", "io_trace.jsonl"):
        if saved["files_sha256"][name] != sha256(root / name):
            raise ValueError("Saved run artifact fingerprint changed")
    manifest = root / "model_manifest.json"
    if sha256(manifest) != config["model_manifest_sha256"]:
        raise ValueError("Model identity changed")
    identity = json.loads(manifest.read_text(encoding="utf8"))
    if any(identity.get(k) != config[k] for k in ("method_id", "method_kind")):
        raise ValueError("Model identity mismatch")
    episode = Path(config["source_episode"])
    source = verify_source(episode)
    if sha256(episode / "inspection.json") != config["source_audit_sha256"]:
        raise ValueError("Episode audit changed")
    rows = read_rows(episode / "public/observations.jsonl")
    trace = read_rows(root / "io_trace.jsonl")
    metadata = json.loads((episode / "public/metadata.json").read_text(encoding="utf8"))
    if (len(trace) != saved["steps"] or not trace or saved["source_steps"] != len(rows)
            or len(trace) > len(rows[:config["max_steps"]])):
        raise ValueError("Trace count mismatch")
    history, latencies, errors = [], [], []
    for index, (observation, record) in enumerate(zip(rows, trace)):
        expected = model_input(episode, observation, episode_id=config["run_id"], instruction=metadata["instruction"],
                               messages=observation["messages"], history=history)
        if record["sample_index"] != index or canonical(record["request"]) != canonical(expected):
            raise ValueError("Public request does not reconstruct from audited source")
        if record["error"]:
            if index != len(trace)-1 or saved["status"] != "model_failure":
                raise ValueError("Failure must terminate the run")
            errors.append(record["error"])
            continue
        response = validate_output(record["response"], expected)
        latency = record["latency_ms"]
        if isinstance(latency, bool) or not isinstance(latency, (int,float)) or latency < 0:
            raise ValueError("Invalid measured latency")
        canonical(latency)
        latencies.append(latency)
        history.append({"timestamp_ns": observation["stamp_ns"], "actions": response["actions"]})
    if not errors and len(trace) != len(rows[:config["max_steps"]]):
        raise ValueError("Unexplained partial run")
    mean = sum(latencies)/len(latencies) if latencies else None
    if (saved["mean_latency_ms"] != mean or saved["errors"] != errors
            or saved["truncated"] != (len(trace) < len(rows))
            or saved["status"] != ("model_failure" if errors else "complete")
            or saved["success"] is not None or saved["role_assignment"] is not None):
        raise ValueError("Summary does not match independently reconstructed trace")
    for field in ("closed_loop", "flight_admitted", "formal_score", "action_conditioned_world_model", "independent_audit"):
        if saved[field] is not False:
            raise ValueError("Replay evidence cannot grant flight, task success or formal score")
    if saved["public_input_only"] is not True or saved["teacher_forced_messages"] is not True:
        raise ValueError("Replay input provenance mismatch")
    return {"schema": "aerowam-subt-model-replay-audit/1", "passed": True,
        "run_id": config["run_id"], "requests_reconstructed": len(trace), "status": saved["status"],
        "protocol_valid_responses": len(latencies), "formal_score": False, "success": None,
        "flight_admitted": False, "source_split": source["split"], "teacher_forced_messages": True,
        "files_sha256": {name: sha256(root / name) for name in
                          ("run.json", "model_manifest.json", "io_trace.jsonl", "replay_summary.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
