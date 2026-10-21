"""Offline industrial target contracts, materials, geometry and motion regressions."""

from copy import deepcopy
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np

from . import industrial_targets as t
from .industrial_target_motion import poses_for, sample_route, validate_feedback, request_in_worker
from .industrial_target_capture import MotionCapture
from .subt_urban_geometry import triangles_touch_box


class TargetTests(unittest.TestCase):
    def test_roundtrip_all_color_shape_components(self):
        for color in t.COLORS:
            for shape in t.SHAPES:
                values = (t.parse_component(f"{color}:{shape}"), t.DEFAULT[1])
                self.assertEqual(t.from_configuration(t.configuration(values)), values)
                self.assertEqual(len(t.fingerprint(values)), 64)
        with self.assertRaises(ValueError):
            t.parse_component("blue:triangle")
        with self.assertRaises(ValueError):
            t.components([t.DEFAULT[0]])

    def test_config_tampering_and_ambiguous_judge_rejected(self):
        config = t.configuration()
        config["roughness"] = .5
        with self.assertRaises(ValueError):
            t.from_configuration(config)
        with self.assertRaises(ValueError):
            t.private_selection((t.DEFAULT[0], t.DEFAULT[0]), "red", "sphere")
        self.assertEqual(t.private_selection(tuple(reversed(t.DEFAULT)), "blue", "cube"), t.IDENTITIES[0])

    def test_public_request_has_no_private_truth_or_fixed_roles(self):
        self.assertEqual(t.public_request("寻找蓝色方块"), {"schema": "aerowam-industrial-request/1",
                         "instruction": "寻找蓝色方块", "role_mode": "unassigned"})
        with self.assertRaises(ValueError):
            t.public_request(" " )

    def test_matte_material_and_collision_visual_geometry_match(self):
        for shape in t.SHAPES:
            world = ET.Element("world", name="test")
            t.install(world, (t.Component("yellow", shape), t.Component("green", shape)))
            self.assertEqual(len(world.findall("model")), 4)
            for link in world.findall("model/link"):
                for visual in link.findall("visual"):
                    collision = link.find(f"collision[@name='{visual.get('name')}']")
                    self.assertEqual(ET.tostring(visual.find("geometry")), ET.tostring(collision.find("geometry")))
                    self.assertEqual(visual.findtext("pose"), collision.findtext("pose"))
                    self.assertEqual(visual.findtext("material/pbr/metal/roughness"), "1")
                    self.assertEqual(visual.findtext("material/pbr/metal/metalness"), "0")
                    self.assertEqual(visual.findtext("material/emissive"), "0 0 0 1")
            self.assertEqual([m.findtext("plugin/label") for m in world.findall("model")], ["100", "11", "100", "22"])
            with self.assertRaises(ValueError):
                t.install(world)

    def test_private_masks_exclude_mount_and_reject_unknown_ids(self):
        labels = np.full((20, 20), 100, dtype=np.uint8)
        labels[3:8, 4:9] = 11
        result = t.visible_components(labels)
        self.assertEqual(result[t.IDENTITIES[0]]["pixels"], 25)
        self.assertIsNone(result[t.IDENTITIES[1]]["bbox"])
        labels[0, 0] = 99
        with self.assertRaises(ValueError):
            t.visible_components(labels)

    def test_contacts_bind_each_collision(self):
        world = ET.Element("world", name="test")
        t.install(world)
        bindings = t.instrument_contacts(world)
        self.assertEqual(len(bindings), 4)
        self.assertEqual(sum(len(b["collisions"]) for b in bindings), 24)
        self.assertEqual({b["kind"] for b in bindings}, {"support", "component"})


class MotionTests(unittest.TestCase):
    def test_route_speed_period_and_invalid_inputs(self):
        route = ((9, -8, 0), (11, -8, 0))
        for seconds, expected in ((0, 9), (5, 10), (10, 11), (15, 10), (20, 9)):
            self.assertAlmostEqual(sample_route(route, seconds, speed=.2)[0], expected)
        for seconds, speed in ((-1, .2), (float("nan"), .2), (1, 0), (1, .26), (1, float("inf"))):
            with self.assertRaises(ValueError):
                sample_route(route, seconds, speed=speed)
        with self.assertRaises(ValueError):
            sample_route((route[0], route[0]), 1)

    def test_feedback_route_and_pair_coherence_fail_closed(self):
        values = {key: xyz.tolist() for key, xyz in poses_for(6).items()}
        validate_feedback({"positions_enu": values})
        for key, xyz in (("industrial_target_a", [9, -8, 3]),
                         ("industrial_mount_b", [12, -6, 0]),
                         ("industrial_target_b", [float("nan"), -6, 2])):
            bad = deepcopy(values)
            bad[key] = xyz
            with self.assertRaises(ValueError):
                validate_feedback({"positions_enu": bad})

    def test_worker_timeout_is_bounded(self):
        from unittest.mock import Mock
        worker = Mock()
        worker.poll.return_value = None
        with patch("benchmark.px4_gazebo.model_handoff.industrial_target_motion.select.select", return_value=([], [], [])):
            with self.assertRaises(TimeoutError):
                request_in_worker(worker, "/test", poses_for(0))

    def test_motion_feedback_timeout_is_bounded(self):
        from .industrial_target_motion import run
        with self.assertRaises(ValueError):
            run("contract", "output", duration=30., speed=.2, feedback_timeout_s=1.)
        with self.assertRaises(ValueError):
            run("contract", "output", duration=30., speed=.2, feedback_timeout_s=31.)

    def test_triangle_box_surface_touch_and_separation(self):
        triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        self.assertTrue(triangles_touch_box(triangle, [-.1, -.1, -.1], [.5, .5, .1]))
        self.assertTrue(triangles_touch_box(triangle, [0, 0, 0], [1, 1, 1]))
        self.assertFalse(triangles_touch_box(triangle, [.9, .9, -.1], [1, 1, .1]))
        with self.assertRaises(ValueError):
            triangles_touch_box(triangle, [0, 0, 0], [0, 1, 1])

    def test_capture_joins_exact_stamps_not_nearest(self):
        from types import SimpleNamespace
        capture = MotionCapture({"camera_pairs": [{"agent_id": "uav_0"}]})
        capture.start(1_000_000_000)
        def message(nsec):
            return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nsec=nsec)),
                                   SerializeToString=lambda: b"test")
        capture.receive("uav_0", 0, message(0))
        capture.receive("uav_0", 1, message(1))
        self.assertEqual(len(capture.frames["uav_0"]), 0)
        capture.receive("uav_0", 1, message(0))
        self.assertEqual(len(capture.frames["uav_0"]), 1)


if __name__ == "__main__":
    unittest.main()
