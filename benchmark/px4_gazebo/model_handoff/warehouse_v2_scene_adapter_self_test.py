"""Self-tests for the live Warehouse v2 scene adapter."""

import base64
import unittest

from .core.model_io import ModelInput
from .protocol import AGENT_IDS, ModelObservation, VehicleState
from .warehouse_v2_scene_adapter import ADAPTER_ID, decode_rgb, to_model_input


class WarehouseV2SceneAdapterTests(unittest.TestCase):
    def observation(self):
        return ModelObservation(
            episode_id="warehouse_live_episode_01",
            instruction="Track the parcel and request grounded relay evidence.",
            timestamp_ns=1_000_000_000,
            image_timestamp_ns={agent: 1_000_000_000 for agent in AGENT_IDS},
            rgb_jpeg={agent: b"jpeg-bytes-" + agent.encode() for agent in AGENT_IDS},
            states={agent: VehicleState((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0, "ok") for agent in AGENT_IDS},
            action_history=({agent: [0.0, 0.0, 0.0, 0.0] for agent in AGENT_IDS},),
            message_history=({"event": "track", "confidence": 0.8, "agent_id": "uav_1"},),
        )

    def test_maps_live_public_packet_without_assigning_roles(self):
        item = to_model_input(self.observation())
        self.assertIsInstance(item, ModelInput)
        self.assertEqual(item.task_id, "warehouse_coop_prediction")
        self.assertEqual(item.scene_id, "aws_warehouse_handoff_v2")
        self.assertEqual(item.task_context["role_mode"], "unassigned")
        self.assertEqual(item.task_context["adapter_id"], ADAPTER_ID)
        self.assertNotIn("role_assignment", item.task_context)
        self.assertEqual(decode_rgb(item), {agent: b"jpeg-bytes-" + agent.encode() for agent in AGENT_IDS})

    def test_rejects_private_context_and_fixed_roles(self):
        with self.assertRaises(ValueError):
            to_model_input(self.observation(), task_context={"target_pose": [1, 2, 3]})
        with self.assertRaises(ValueError):
            to_model_input(self.observation(), task_context={"role_assignment": {"uav_0": "source"}})

    def test_public_input_has_no_truth_or_synthetic_score_flags(self):
        item = to_model_input(self.observation())
        wire = item.to_wire()
        text = str(wire)
        self.assertNotIn("target_pose", text)
        self.assertNotIn("private_targets_enu", text)
        self.assertNotIn("model_score_eligible", text)
        self.assertNotIn("flight_admitted", text)


if __name__ == "__main__":
    unittest.main()
