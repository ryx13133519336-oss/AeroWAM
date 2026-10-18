"""Startup isolation and telemetry evidence tests; no simulator required."""

import json
from types import SimpleNamespace
import unittest

import numpy as np

from .annotation_session_smoke import INSTANCES, WORLD, odometry_snapshot, smoke_environment


class AnnotationSessionTests(unittest.TestCase):
    def test_no_inherited_px4_target_or_default_agent(self):
        base = {"PX4_GZ_MODEL_NAME": "old", "PX4_SYS_AUTOSTART": "4001", "PX4_UXRCE_DDS_PORT": "8888",
                "PX4_GZ_MODEL_POSE": "9,9", "ROS_DOMAIN_ID": "0", "GZ_PARTITION": "old", "PATH": "bin"}
        before = dict(base)
        for index in (0, 1):
            env = smoke_environment(base, partition="isolated", domain=77, port=45678, index=index)
            self.assertEqual(env["PX4_GZ_MODEL_NAME"], f"annotation_uav_{index}")
            self.assertEqual(env["PX4_UXRCE_DDS_PORT"], "45678")
            self.assertEqual(env["PX4_GZ_WORLD"], WORLD)
            self.assertNotIn("PX4_GZ_MODEL_POSE", env)
            self.assertEqual(env["PX4_GZ_STANDALONE"], "1")
            self.assertEqual(env["PATH"], "bin")
        self.assertEqual(base, before)
        self.assertTrue(set(INSTANCES).isdisjoint({0, 1}))

    def test_invalid_isolation_config_rejected(self):
        for changes in ({"domain": 0}, {"domain": True}, {"port": 1}, {"port": True},
                        {"index": True}, {"index": 2}, {"partition": ""}):
            args = dict(partition="isolated", domain=77, port=45678, index=0)
            args.update(changes)
            with self.assertRaises(ValueError):
                smoke_environment({}, **args)

    def test_ros_float32_arrays_are_strict_json(self):
        message = SimpleNamespace(timestamp=1, position=np.array([0, 0, -2], dtype=np.float32),
                                  velocity=np.zeros(3, dtype=np.float32), q=np.array([1, 0, 0, 0], dtype=np.float32),
                                  pose_frame=1, velocity_frame=1)
        result = odometry_snapshot(message)
        self.assertTrue(result["finite"])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False))["position"], [0, 0, -2])
        message.position[0] = np.nan
        result = odometry_snapshot(message)
        self.assertFalse(result["finite"])
        self.assertIsNone(result["position"][0])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
