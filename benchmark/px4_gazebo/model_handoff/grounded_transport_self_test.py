"""Grounded adapter/client transport tests; no task or flight claim."""

from copy import deepcopy
import threading
import unittest

from .grounded_interface import GroundedAction, GroundedAdapter
from .language_tasks import TargetReport
from .policy_client import PolicyClient
from .policy_server import ReusableThreadingTCPServer, _Handler
from .protocol import ModelJointAction, assert_truth_free
from .self_test import _observation


class GroundedTransportTests(unittest.TestCase):
    def test_roundtrip_through_actual_adapter_server_handler(self):
        observation = _observation()
        delivered = []
        def policy(public):
            delivered.append(public)
            assert_truth_free(public.to_wire())
            return GroundedAction(ModelJointAction(), (TargetReport("uav_0", public.image_timestamp_ns["uav_0"],
                                                                    (.1, .1, .4, .4), .8, "observe"),))
        with ReusableThreadingTCPServer(("127.0.0.1", 0), _Handler) as server:
            server.policy = GroundedAdapter(policy)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = PolicyClient(f"127.0.0.1:{server.server_address[1]}", timeout_s=2.)
                result, latency = client.infer_grounded(observation)
            finally:
                server.shutdown()
                thread.join(timeout=2.)
        self.assertEqual(result.reports[0].image_timestamp_ns, observation.image_timestamp_ns["uav_0"])
        self.assertEqual(delivered[0].rgb_jpeg, observation.rgb_jpeg)
        self.assertEqual(delivered[0].instruction, observation.instruction)
        self.assertGreaterEqual(latency, 0)

    def test_forbidden_private_input_never_reaches_policy(self):
        calls = []
        adapter = GroundedAdapter(lambda public: calls.append(public))
        wire = _observation().to_wire()
        for key in ("target_id", "truth", "evaluator", "instance_labels"):
            payload = deepcopy(wire)
            payload[key] = "private"
            with self.assertRaises(ValueError):
                adapter(payload)
        self.assertFalse(calls)

    def test_stale_or_wrong_camera_stamp_rejected_on_both_ends(self):
        observation = _observation()
        result = GroundedAction(ModelJointAction(), (TargetReport("uav_0", 7, (.1, .1, .3, .3), .9, "found"),))
        with self.assertRaises(ValueError):
            GroundedAdapter(lambda public: result)(observation.to_wire())
        client = PolicyClient("127.0.0.1:12345")
        client.request = lambda payload: (result.to_wire(), 1.)
        with self.assertRaises(ValueError):
            client.infer_grounded(observation)

    def test_schema_separation_and_legacy_client(self):
        observation = _observation()
        client = PolicyClient("127.0.0.1:12345")
        client.request = lambda payload: (ModelJointAction().to_wire(), 1.)
        self.assertEqual(client.infer(observation)[0], ModelJointAction())
        with self.assertRaises(ValueError):
            client.infer_grounded(observation)
        with self.assertRaises(TypeError):
            GroundedAdapter(lambda public: ModelJointAction())(observation.to_wire())


if __name__ == "__main__":
    unittest.main()
