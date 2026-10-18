"""Offline adapter identity and saved-contract audit; never imports a factory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "aerowam-adapter-manifest/1-draft"
REPORT_SCHEMA = "aerowam-adapter-preflight/1-draft"
PROFILE = "centralized_dual_rgb_local_flu_unassigned/1-draft"
INTERFACES = {"joint_action", "grounded_action", "candidate_prediction"}
BASE_INPUTS = {"instruction", "dual_rgb", "local_states", "action_history", "message_history"}
CANDIDATE_INPUTS = {"observed_history", "candidate_actions", "horizon_s", "task_kind"}
FACTORY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", re.ASCII)
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", re.ASCII)


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Expected exact fields: {sorted(expected)}")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty text required")


def _choice(value, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"Expected one of {sorted(choices)}")


def _unique_strings(value):
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value) or len(set(value)) != len(value):
        raise ValueError("Unique string list required")
    return set(value)


def validate_manifest(value):
    """Validate declarations, not model behavior, completeness or provenance truth."""
    _keys(value, {"schema", "method_id", "implementation", "factory", "interface", "input_profile",
                  "required_inputs", "policy_role_binding", "source", "components", "training", "runtime", "artifacts"})
    if value["schema"] != SCHEMA or value["input_profile"] != PROFILE:
        raise ValueError("Unsupported manifest schema or input profile")
    if not isinstance(value["method_id"], str) or not IDENTIFIER.fullmatch(value["method_id"]):
        raise ValueError("Invalid versioned method ID")
    if not isinstance(value["factory"], str) or not FACTORY.fullmatch(value["factory"]):
        raise ValueError("Expected module:function factory; never a shell command")
    _choice(value["implementation"], {"external_model", "repository_model", "rule_policy", "transport_example"})
    _choice(value["interface"], INTERFACES)
    allowed = BASE_INPUTS | (CANDIDATE_INPUTS if value["interface"] == "candidate_prediction" else set())
    required = {"instruction", "dual_rgb"} | (CANDIDATE_INPUTS if value["interface"] == "candidate_prediction" else set())
    inputs = _unique_strings(value["required_inputs"])
    if not required <= inputs <= allowed:
        raise ValueError("Missing protocol inputs or unavailable inputs in this RGB profile")
    # Fixed roles may be a method property; the task never supplies assigned roles.
    _choice(value["policy_role_binding"], {"unassigned", "method_fixed", "unknown"})
    for field, names in (("source", {"repository", "revision", "license", "native_output", "adaptation"}),
                         ("components", {"language_encoder", "planner", "reporter", "action_decoder"}),
                         ("runtime", {"device", "precision", "history_usage"})):
        _keys(value[field], names)
        for item in value[field].values():
            _text(item)
    _choice(value["runtime"]["history_usage"], {"current_only", "explicit_history", "stateful", "unknown"})
    training = value["training"]
    _keys(training, {"status", "data_id", "seed", "steps", "trainable_parameters"})
    _choice(training["status"], {"none", "finetuned", "from_scratch", "unknown"})
    _text(training["data_id"])
    for key in ("seed", "steps", "trainable_parameters"):
        if training[key] is not None and (type(training[key]) is not int or training[key] < 0):
            raise ValueError("Training counts/seeds must be nonnegative integers or null (unknown)")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Explicit identity artifacts required")
    identifiers, kinds = set(), set()
    for item in artifacts:
        _keys(item, {"id", "kind", "path", "sha256"})
        if not isinstance(item["id"], str) or not IDENTIFIER.fullmatch(item["id"]) or item["id"] in identifiers:
            raise ValueError("Invalid or duplicate artifact ID")
        identifiers.add(item["id"])
        _choice(item["kind"], {"adapter_code", "weights", "config", "environment"})
        kinds.add(item["kind"])
        _text(item["path"])
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Expected lowercase SHA256 for each named file")
    required_kinds = {"adapter_code"}
    if value["implementation"] != "transport_example":
        required_kinds |= {"config", "environment"}
    if value["implementation"] in {"external_model", "repository_model"}:
        required_kinds.add("weights")
    if not required_kinds <= kinds:
        raise ValueError(f"Missing artifact kinds: {sorted(required_kinds - kinds)}")


def _load(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _file_identity(path):
    digest, length = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            length += len(block)
            digest.update(block)
    return {"bytes": length, "sha256": digest.hexdigest()}


def _joint_action(payload):
    from .protocol import AGENT_IDS, ModelJointAction, MODEL_PROTOCOL_VERSION
    _keys(payload, {"protocol", "waypoints", "intent"})
    _keys(payload["waypoints"], AGENT_IDS)
    if payload["protocol"] != MODEL_PROTOCOL_VERSION or payload["intent"] != "none":
        raise ValueError("Unassigned profile requires explicit protocol and no legacy directed intent")
    action = ModelJointAction.from_wire(payload)
    if action.to_wire()["waypoints"] != payload["waypoints"]:
        raise ValueError("Do not silently clip an adapter output")


def check_saved_io(interface, request, response):
    """Check saved public bytes only; never invokes a model or a flight runner."""
    from .decision_interface import DECISION_PROTOCOL_VERSION, DecisionRequest
    from .grounded_interface import GroundedAction, validate_observed_reports
    from .interface_check import check
    from .protocol import ModelObservation, assert_truth_free
    _choice(interface, INTERFACES)
    assert_truth_free(response)
    if interface == "candidate_prediction":
        if request.get("protocol") != DECISION_PROTOCOL_VERSION:
            raise ValueError("Candidate prediction requires an explicit candidate request")
        parsed = DecisionRequest.from_wire(request)
        if any(step.action.intent != "none" for c in parsed.candidates for step in c.steps):
            raise ValueError("Legacy directed candidate intent unavailable in unassigned profile")
        check(request, response)
    else:
        observation = ModelObservation.from_wire(request)
        check(request)  # Decode original RGB; do not manufacture missing modalities.
        if interface == "grounded_action":
            envelope = GroundedAction.from_wire(response)
            _joint_action(response["action"])
            validate_observed_reports(envelope, observation)
        else:
            _joint_action(response)


def audit(path, *, request_path=None, response_path=None):
    path = Path(path).resolve()
    if (request_path is None) != (response_path is None):
        raise ValueError("Provide both saved request and response, or neither")
    raw = path.read_bytes()
    manifest = _load(raw)
    validate_manifest(manifest)
    verified, paths = [], {path}
    for item in manifest["artifacts"]:
        artifact = (path.parent / item["path"]).resolve()
        if artifact in paths or not artifact.is_file():
            raise ValueError("Duplicate, missing or non-file identity artifact")
        paths.add(artifact)
        identity = _file_identity(artifact)
        if identity["sha256"] != item["sha256"]:
            raise ValueError(f"Artifact digest mismatch: {item['id']}")
        verified.append({"id": item["id"], "kind": item["kind"], **identity})
    saved_io = None
    if request_path is not None:
        request_raw, response_raw = Path(request_path).read_bytes(), Path(response_path).read_bytes()
        check_saved_io(manifest["interface"], _load(request_raw), _load(response_raw))
        saved_io = {"request_sha256": hashlib.sha256(request_raw).hexdigest(),
                    "response_sha256": hashlib.sha256(response_raw).hexdigest()}
    return {"schema": REPORT_SCHEMA, "method_id": manifest["method_id"], "interface": manifest["interface"],
            "input_profile": PROFILE, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "declared_method": manifest, "verified_artifacts": verified,
            "manifest_files_verified": True, "saved_io_verified": saved_io is not None,
            "saved_io": saved_io, "online_deadline_ms": 400,
            "factory_import_verified": False, "model_executed": False, "runtime_latency_verified": False,
            "artifact_list_completeness_verified": False, "access_isolation_verified": False,
            "role_free_execution_verified": False, "prediction_quality_verified": False,
            "flight_ready": False, "formal_benchmark_result": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, request_path=args.request, response_path=args.response)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Adapter preflight verified (named files/saved contracts only; not runtime or model capability)")


if __name__ == "__main__":
    main()
