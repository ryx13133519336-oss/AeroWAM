"""Synthetic tests for the SubT grounded/Core bridge; no Gazebo or PX4."""

import base64
from io import BytesIO
import unittest

from PIL import Image

from .grounded_interface import GroundedAction
from .language_tasks import TargetReport
from .protocol import ModelJointAction, WaypointDelta
from .subt_handoff_grounded_bridge import core_to_legacy, grounded_to_core


def request():
    image = Image.new("RGB", (8, 8), (40, 60, 80))
    stream = BytesIO()
    image.save(stream, format="JPEG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return {"schema": "aerowam-model-input/1", "episode_id": "bridge_episode",
            "task_id": "handoff", "task_version": "1", "scene_id": "subt_urban_stair_v1",
            "scene_family_id": "subt_urban_2_story", "instruction": "Track the red sphere.",
            "timestamp_ns": 1_000_000_000, "agent_ids": ["uav_0", "uav_1"],
            "observations": {agent: {"rgb_jpeg_b64": encoded, "image_timestamp_ns": 1_000_000_000,
                                      "proprioception": {"position_local_m": [0., 0., 0.],
                                                          "quaternion_xyzw": [0., 0., 0., 1.]}}
                             for agent in ("uav_0", "uav_1")},
            "messages": [], "action_history": [], "task_context": {"role_assignment": None}}


class BridgeTests(unittest.TestCase):
    def test_core_input_becomes_legacy_observation(self):
        parsed = core_to_legacy(request())
        self.assertEqual(set(parsed.rgb_jpeg), {"uav_0", "uav_1"})
        self.assertEqual(parsed.states["uav_0"].yaw, 0.)

    def test_grounded_events_preserve_frame_and_default_to_hold(self):
        payload = request()
        grounded = GroundedAction(
            ModelJointAction(uav_0=WaypointDelta(dx=.8), uav_1=WaypointDelta(dy=-.4)),
            (TargetReport("uav_0", 1_000_000_000, (.1, .2, .4, .5), .9, "found"),))
        result = grounded_to_core(payload, grounded)
        self.assertEqual(result["actions"]["uav_0"]["delta_enu_m"], [0., 0., 0.])
        self.assertEqual(result["events"][0]["event"], "handoff.found")
        self.assertEqual(result["events"][0]["image_timestamp_ns"], 1_000_000_000)

    def test_proxy_enu_opt_in_is_explicit(self):
        payload = request()
        grounded = GroundedAction(ModelJointAction(uav_0=WaypointDelta(dx=.8)), ())
        result = grounded_to_core(payload, grounded, allow_proxy_enu_actions=True)
        self.assertEqual(result["actions"]["uav_0"]["delta_enu_m"], [.8, 0., 0.])


if __name__ == "__main__":
    unittest.main()
