"""Synthetic identity/contract regression; no real model, training or flight."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .adapter_manifest import SCHEMA, PROFILE, audit, check_saved_io, validate_manifest
from .decision_interface import Candidate, CandidateStep, CandidatePrediction, DecisionRequest, DECISION_PROTOCOL_VERSION
from .grounded_interface import GroundedAction
from .language_tasks import TargetReport
from .protocol import ModelJointAction
from .self_test import _observation


def synthetic_manifest(directory):
    """Files and weight bytes are explicit test fixtures, never a real model."""
    entries = []
    for name, kind in (("adapter.py", "adapter_code"), ("synthetic.weights", "weights"),
                       ("config.json", "config"), ("environment.txt", "environment")):
        raw = b"raise RuntimeError('This fixture must never be imported or executed')\n"
        (directory / name).write_bytes(raw + kind.encode())
        entries.append({"id": kind, "kind": kind, "path": name,
                        "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()})
    return {"schema": SCHEMA, "method_id": "synthetic_not_model_v1", "implementation": "external_model",
            "factory": "never_import_this_adapter:create", "interface": "joint_action", "input_profile": PROFILE,
            "required_inputs": ["instruction", "dual_rgb"], "policy_role_binding": "unassigned",
            "source": {"repository": "synthetic_not_official", "revision": "synthetic_v1", "license": "not_applicable",
                       "native_output": "synthetic_fixture", "adaptation": "no_model_loaded"},
            "components": {k: "not_applicable_synthetic" for k in ("language_encoder", "planner", "reporter", "action_decoder")},
            "training": {"status": "none", "data_id": "not_applicable", "seed": None, "steps": 0, "trainable_parameters": 0},
            "runtime": {"device": "not_executed", "precision": "not_applicable", "history_usage": "unknown"},
            "artifacts": entries}


def prediction_io():
    obs = _observation()
    request = DecisionRequest("synthetic", "search", obs.instruction, (obs,),
        tuple(Candidate(n, (CandidateStep(ModelJointAction(), .5),)) for n in ("a", "b")), .5).to_wire()
    predictions = [CandidatePrediction(n, .2, {"uav_0": .3, "uav_1": .4}, 0., .2, .1).to_wire() for n in ("a", "b")]
    return request, {"protocol": DECISION_PROTOCOL_VERSION, "decision_id": "synthetic", "predictions": predictions}


class AdapterManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = synthetic_manifest(self.root)
        self.path = self.root / "manifest.json"

    def save(self, value=None):
        self.path.write_text(json.dumps(self.manifest if value is None else value), encoding="utf-8")
        return self.path

    def test_hashes_named_files_without_importing_factory_or_claiming_readiness(self):
        report = audit(self.save())
        self.assertEqual(len(report["verified_artifacts"]), 4)
        self.assertEqual(report["declared_method"], self.manifest)
        self.assertTrue(report["manifest_files_verified"])
        for field in ("saved_io_verified", "factory_import_verified", "model_executed", "runtime_latency_verified",
                      "artifact_list_completeness_verified", "access_isolation_verified", "role_free_execution_verified",
                      "prediction_quality_verified", "flight_ready", "formal_benchmark_result"):
            self.assertIs(report[field], False)
        self.assertEqual(report["online_deadline_ms"], 400)
        self.assertNotIn("never_import_this_adapter", sys.modules)

    def test_changed_or_missing_weights_refuse(self):
        self.save()
        weight = self.root / "synthetic.weights"
        weight.write_bytes(b"different weights")
        with self.assertRaises(ValueError):
            audit(self.path)
        weight.unlink()
        with self.assertRaises(ValueError):
            audit(self.path)

    def test_model_cannot_omit_identity_categories(self):
        for kind in ("adapter_code", "weights", "config", "environment"):
            with self.subTest(kind=kind):
                value = deepcopy(self.manifest)
                value["artifacts"] = [a for a in value["artifacts"] if a["kind"] != kind]
                with self.assertRaises(ValueError):
                    validate_manifest(value)

    def test_transport_example_needs_no_fake_weights(self):
        self.manifest["implementation"] = "transport_example"
        self.manifest["artifacts"] = self.manifest["artifacts"][:1]
        report = audit(self.save())
        self.assertEqual(len(report["verified_artifacts"]), 1)
        self.assertFalse(report["model_executed"])

    def test_factory_shell_commands_and_invalid_ids_refuse(self):
        for field, bad in (("factory", "python model.py"), ("factory", "x:y;run"),
                           ("method_id", "../other"), ("method_id", "")):
            value = deepcopy(self.manifest)
            value[field] = bad
            with self.assertRaises(ValueError):
                validate_manifest(value)

    def test_unknown_or_missing_fields_do_not_silently_pass(self):
        for mutate in (lambda v: v.update(flight_ready=True), lambda v: v.pop("source"),
                       lambda v: v["source"].update(official_verified=True),
                       lambda v: v["runtime"].update(timeout_ms=4000)):
            value = deepcopy(self.manifest)
            mutate(value)
            with self.assertRaises(ValueError):
                validate_manifest(value)

    def test_unavailable_modalities_or_assigned_task_roles_refuse(self):
        for name in ("depth", "bev", "target_pose", "global_map", "task_assigned_roles", "candidate_actions"):
            value = deepcopy(self.manifest)
            value["required_inputs"].append(name)
            with self.assertRaises(ValueError):
                validate_manifest(value)

    def test_method_fixed_is_declared_without_changing_task_profile(self):
        self.manifest["policy_role_binding"] = "method_fixed"
        report = audit(self.save())
        self.assertEqual(report["input_profile"], PROFILE)
        self.assertFalse(report["role_free_execution_verified"])

    def test_candidate_interface_requires_conditioning_fields(self):
        value = deepcopy(self.manifest)
        value["interface"] = "candidate_prediction"
        with self.assertRaises(ValueError):
            validate_manifest(value)
        value["required_inputs"] += ["observed_history", "candidate_actions", "horizon_s", "task_kind"]
        validate_manifest(value)

    def test_duplicate_artifact_id_path_or_directory_refuse(self):
        for mutate in (lambda v: v["artifacts"][1].update(id="adapter_code"),
                       lambda v: v["artifacts"][1].update(path="adapter.py"),
                       lambda v: v["artifacts"][1].update(path=".")):
            value = deepcopy(self.manifest)
            mutate(value)
            with self.assertRaises(ValueError):
                audit(self.save(value))

    def test_duplicate_json_and_nonfinite_or_bool_counts_refuse(self):
        self.path.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            audit(self.path)
        for raw in (self.save().read_text().replace('"steps": 0', '"steps": NaN'),
                    self.save().read_text().replace('"steps": 0', '"steps": true')):
            self.path.write_text(raw, encoding="utf-8")
            with self.assertRaises(ValueError):
                audit(self.path)

    def test_saved_joint_action_has_no_capability_or_latency_claim(self):
        request, response = self.root / "request.json", self.root / "response.json"
        request.write_text(json.dumps(_observation().to_wire()), encoding="utf-8")
        response.write_text(json.dumps(ModelJointAction().to_wire()), encoding="utf-8")
        report = audit(self.save(), request_path=request, response_path=response)
        self.assertTrue(report["saved_io_verified"])
        self.assertEqual(report["saved_io"]["request_sha256"], hashlib.sha256(request.read_bytes()).hexdigest())
        self.assertFalse(report["runtime_latency_verified"])
        with self.assertRaises(ValueError):
            audit(self.path, request_path=request)

    def test_grounded_contract_accepts_either_agent_report(self):
        obs = _observation()
        for agent in ("uav_0", "uav_1"):
            report = TargetReport(agent, obs.image_timestamp_ns[agent], (.1, .1, .3, .3), .8, "observe")
            check_saved_io("grounded_action", obs.to_wire(), GroundedAction(ModelJointAction(), (report,)).to_wire())

    def test_stale_grounded_report_and_action_envelope_confusion_refuse(self):
        obs = _observation()
        result = GroundedAction(ModelJointAction(), (TargetReport("uav_0", 7, (.1, .1, .3, .3), .8, "found"),)).to_wire()
        with self.assertRaises(ValueError):
            check_saved_io("grounded_action", obs.to_wire(), result)
        with self.assertRaises(ValueError):
            check_saved_io("joint_action", obs.to_wire(), GroundedAction(ModelJointAction()).to_wire())
        with self.assertRaises(ValueError):
            check_saved_io("grounded_action", obs.to_wire(), ModelJointAction().to_wire())

    def test_prediction_contract_is_separate_and_covers_all_candidates(self):
        request, response = prediction_io()
        check_saved_io("candidate_prediction", request, response)
        response["predictions"].pop()
        with self.assertRaises(ValueError):
            check_saved_io("candidate_prediction", request, response)
        with self.assertRaises(ValueError):
            check_saved_io("candidate_prediction", _observation().to_wire(), ModelJointAction().to_wire())

    def test_private_input_and_response_refuse(self):
        request, response = _observation().to_wire(), ModelJointAction().to_wire()
        request["target_pose"] = [1, 2, 3]
        with self.assertRaises(ValueError):
            check_saved_io("joint_action", request, response)
        response["truth"] = True
        with self.assertRaises(ValueError):
            check_saved_io("joint_action", _observation().to_wire(), response)

    def test_no_silent_clipping_or_legacy_directed_intents(self):
        for intent, delta in (("handoff", 0.), ("confirm", 0.), ("none", 99.)):
            response = ModelJointAction().to_wire()
            response["intent"] = intent
            response["waypoints"]["uav_0"][0] = delta
            with self.assertRaises(ValueError):
                check_saved_io("joint_action", _observation().to_wire(), response)

    def test_saved_prediction_not_certified_by_schema(self):
        request, response = prediction_io()
        self.manifest["interface"] = "candidate_prediction"
        self.manifest["required_inputs"] += ["observed_history", "candidate_actions", "horizon_s", "task_kind"]
        q, s = self.root / "q.json", self.root / "s.json"
        q.write_text(json.dumps(request), encoding="utf-8")
        s.write_text(json.dumps(response), encoding="utf-8")
        report = audit(self.save(), request_path=q, response_path=s)
        self.assertTrue(report["saved_io_verified"])
        self.assertFalse(report["prediction_quality_verified"])

    def test_cli_is_offline_and_refuses_output_overwrite(self):
        self.save()
        output = self.root / "audit.json"
        command = [sys.executable, "-B", "-m", "benchmark.px4_gazebo.model_handoff.adapter_manifest",
                   "--manifest", str(self.path), "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
