"""Checkpoint adapter protocol tests without loading a trained checkpoint."""

import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .subt_handoff_checkpoint_adapter import public_features


class CheckpointAdapterTests(unittest.TestCase):
    def request(self):
        image = BytesIO()
        Image.fromarray(np.full((256, 256, 3), 80, dtype=np.uint8)).save(image, format="JPEG")
        encoded = base64.b64encode(image.getvalue()).decode("ascii")
        return {"schema": "aerowam-model-input/1", "episode_id": "ep", "timestamp_ns": 1,
                "observations": {agent: {"rgb_jpeg_b64": encoded,
                    "proprioception": {"position_local_m": [0., 0., 0.],
                                       "quaternion_xyzw": [0., 0., 0., 1.]}}
                    for agent in ("uav_0", "uav_1")},
                "messages": [{"agent_id": "uav_1", "event": "handoff.handoff_request"}],
                "task_context": {"dt_s": .5}}

    def test_feature_layout_matches_training_width(self):
        self.assertEqual(public_features(self.request()).shape, (419,))

    def test_only_public_fields_are_used(self):
        request = self.request()
        request["private_truth"] = {"visible": True}
        first = public_features(request)
        del request["private_truth"]
        np.testing.assert_array_equal(first, public_features(request))

    def test_message_history_is_namespaced(self):
        values = public_features(self.request())
        # The first agent's request count is zero; the peer's is one.
        self.assertEqual(values[398 + 5 + 2], 1.)


if __name__ == "__main__":
    unittest.main()
