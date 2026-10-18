"""Offline probe tests without torch weights, training or flight."""

import json
from pathlib import Path
import tempfile
import unittest

from .checkpoint_search_probe import load_observations, probe_observations
from .grounded_interface import GroundedAction
from .policy_client import PolicyTimeoutError
from .protocol import ModelJointAction, WaypointDelta
from .search_pilot_self_test import observation


class CheckpointProbeTests(unittest.TestCase):
    def test_reads_only_public_observation_and_limits_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            row = {"observation": observation().to_wire(), "response": {"truth": "ignored"}}
            path.write_text(json.dumps(row) + "\nnot parsed after limit\n", encoding="utf-8")
            loaded = load_observations(path, 1)
            self.assertEqual(loaded[0].rgb_jpeg, observation().rgb_jpeg)
            self.assertEqual(loaded[0].instruction, observation().instruction)

    def test_stale_changed_instruction_or_private_payload_rejected(self):
        first = observation().to_wire()
        cases = [first, observation("找蓝方块", stamp=1_500_000_000).to_wire(),
                 dict(observation(stamp=1_500_000_000).to_wire(), truth="forbidden")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            for second in cases:
                path.write_text("\n".join(json.dumps({"observation": r}) for r in (first, second)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_observations(path, 2)

    def test_preserves_model_actions_and_never_claims_closed_loop_success(self):
        action = ModelJointAction(uav_0=WaypointDelta(.2, -.1, 0., .03), intent="confirm")
        class Client:
            def infer_grounded(self, obs):
                return GroundedAction(action), 50.
        rows, result = probe_observations(Client(), [observation()])
        self.assertEqual(rows[0]["response"]["action"], action.to_wire())
        self.assertTrue(result["transport_passed"])
        self.assertFalse(result["task_success_evaluated"])
        self.assertFalse(result["closed_loop_evaluated"])
        self.assertEqual(result["learned_intents"], {"confirm": 1})
        self.assertEqual(result["auxiliary_report_events"], {})

    def test_timeout_stops_without_retry_or_later_frames(self):
        calls = []
        class Client:
            def infer_grounded(self, obs):
                calls.append(obs)
                raise PolicyTimeoutError("400 ms")
        rows, result = probe_observations(Client(), [observation(), observation(stamp=1_500_000_000)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(rows, [])
        self.assertFalse(result["transport_passed"])
        self.assertEqual(result["failure"]["type"], "PolicyTimeoutError")


if __name__ == "__main__":
    unittest.main()
