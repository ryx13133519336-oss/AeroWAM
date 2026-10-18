"""Occlusion controls and private pixel acceptance tests; no simulator."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .gazebo_rasters import PrivateRasterPair
from .search_occlusion_probe import measure, verify_controls
from .search_pilot_scene import OCCLUSION_CASES, static_occlusion_world


def raster(blocked=False):
    rgb = np.full((256, 256, 3), 180, dtype=np.uint8)
    labels = np.zeros((256, 256), dtype=np.uint8)
    if blocked:
        labels[40:210, 30:220] = 33
    else:
        labels[80:110, 80:110] = 11
        rgb[80:110, 80:110] = (220, 0, 0)
        labels[140:170, 140:170] = 22
        rgb[140:170, 140:170] = (0, 0, 220)
    return PrivateRasterPair(1_000_000_000, rgb, labels,
                             np.zeros((256, 256, 2), dtype=np.uint8))


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        with patch("benchmark.px4_gazebo.model_handoff.search_pilot_scene.verify"):
            self.trees = {case: static_occlusion_world(Path("camera-models"), case)[0]
                          for case in OCCLUSION_CASES}

    def test_only_wall_and_viewpoints_change(self):
        result = verify_controls(self.trees)
        self.assertTrue(result["matched_scene_controls"])
        for tree in self.trees.values():
            scene = tree.getroot().find("world")
            self.assertEqual(len(scene.findall(".//sensor[@type='contact']")), 2)
            for name in ("probe_red", "probe_blue"):
                self.assertIsNotNone(scene.find(f"model[@name='{name}']/link/visual[@name='stripe']"))
            self.assertFalse(scene.findall("include"))

    def test_reject_confounded_target_or_intrinsics(self):
        for selector in ("model[@name='probe_blue']/pose",
                         "model[@name='annotation_camera_0']/include/uri"):
            trees = deepcopy(self.trees)
            trees["search_blocked"].getroot().find("world/" + selector).text = "changed"
            with self.assertRaisesRegex(ValueError, "assets"):
                verify_controls(trees)

    def test_reject_camera_or_wall_mismatch(self):
        for case, selector in (("search_clear", "model[@name='annotation_camera_0']/pose"),
                               ("search_revealed", "model[@name='probe_wall']/pose")):
            trees = deepcopy(self.trees)
            trees[case].getroot().find("world/" + selector).text = "0 0 0 0 0 0"
            with self.assertRaises(ValueError):
                verify_controls(trees)

    def test_reject_missing_control_or_dynamic_entity(self):
        trees = deepcopy(self.trees)
        trees.pop("search_clear")
        with self.assertRaises(ValueError):
            verify_controls(trees)
        trees = deepcopy(self.trees)
        trees["search_blocked"].getroot().find("world/model/static").text = "false"
        with self.assertRaisesRegex(ValueError, "static"):
            verify_controls(trees)

    def test_visible_and_blocked_pixel_controls(self):
        for case in OCCLUSION_CASES:
            result = measure(raster(case == "search_blocked"), scene_case=case,
                             agent_id="uav_0", render_id="synthetic")
            for record in result.values():
                self.assertEqual(record["target_pixels"]["probe_blue"],
                                 0 if case == "search_blocked" else 900)

    def test_blocked_requires_zero_target_pixels_and_real_wall(self):
        for no_wall in (False, True):
            pair = raster(True)
            if no_wall:
                pair.object_labels[:] = 0
            else:
                pair.object_labels[100, 100] = 22
            with self.assertRaises(ValueError):
                measure(pair, scene_case="search_blocked", agent_id="uav_0", render_id="bad")

    def test_visible_requires_rgb_sanity_and_reportable_box(self):
        for wrong_rgb in (False, True):
            pair = raster()
            if wrong_rgb:
                pair.rgb[:] = 100
            else:
                pair.object_labels[:100, :100] = 11  # Border-clipped support.
            with self.assertRaises(ValueError):
                measure(pair, scene_case="search_clear", agent_id="uav_1", render_id="bad")

    def test_unknown_private_label_rejected(self):
        pair = raster()
        pair.object_labels[0, 0] = 99
        with self.assertRaises(ValueError):
            measure(pair, scene_case="search_revealed", agent_id="uav_1", render_id="bad")

    def test_native_support_does_not_guarantee_policy_resolution_support(self):
        rgb = np.full((512, 512, 3), 180, dtype=np.uint8)
        labels = np.zeros((512, 512), dtype=np.uint8)
        for label, x, color in ((11, 100, (220, 0, 0)), (22, 300, (0, 0, 220))):
            labels[100:116, x:x + 4] = label
            rgb[100:116, x:x + 4] = color
        pair = PrivateRasterPair(1, rgb, labels, np.zeros((512, 512, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "policy_256"):
            measure(pair, scene_case="search_clear", agent_id="uav_0", render_id="small")

    def test_unknown_case_rejected_before_resource_read(self):
        with self.assertRaises(ValueError):
            static_occlusion_world(Path("missing"), "wrong")


if __name__ == "__main__":
    unittest.main()
