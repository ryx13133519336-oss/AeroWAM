"""History diagnostic controls and shortcut rejection, without Gazebo."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .search_history_probe import (
    canonical, design, equal_hidden, label_order, request, rgb_order, verify_queries, verify_worlds,
)
from .search_occlusion_sequence import LAYOUTS, check_pose_packet, sequence_world, target_poses
from .search_occlusion_sequence_self_test import scene_for


def pixels():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    labels = np.zeros((32, 32), dtype=np.uint8)
    rgb[4:12, 2:8, 0], labels[4:12, 2:8] = 230, 11
    rgb[4:12, 22:28, 2], labels[4:12, 22:28] = 230, 22
    return rgb, labels


def queries():
    current = {a: "same-hidden-rgb" for a in ("uav_0", "uav_1")}
    return [{"current_only": request("uav_0", "red", current),
             "with_history": request("uav_0", "red", current,
                                     {a: visible for a in current}), "answer": answer}
            for visible, answer in (("red-left", "left"), ("red-right", "right"))]


class HistoryTests(unittest.TestCase):
    def test_seeded_pair_is_balanced_and_bounded_not_frame_replicates(self):
        self.assertEqual(design(20261018), design(20261018))
        self.assertEqual(set(design(17)["private_run_order"]), set(LAYOUTS))
        self.assertEqual(design(17)["scene_groups"], 1)
        for seed in (True, -1, 2**32, .5, "0"):
            with self.assertRaises(ValueError):
                design(seed)

    def test_only_target_positions_may_change(self):
        with patch("benchmark.px4_gazebo.model_handoff.search_pilot_scene.verify"):
            trees = {layout: sequence_world(Path("models"), layout)[0] for layout in LAYOUTS}
        verify_worlds(trees)
        for path in ("world/model[@name='probe_wall']/pose",
                     "world/model[@name='annotation_camera_0']/pose",
                     "world/model[@name='probe_red']/link/visual[@name='body']/geometry/sphere/radius"):
            changed = deepcopy(trees)
            changed["swapped"].getroot().find(path).text = "0"
            with self.assertRaises(ValueError):
                verify_worlds(changed)

    def test_wrong_or_missing_intervention_rejected(self):
        with patch("benchmark.px4_gazebo.model_handoff.search_pilot_scene.verify"):
            tree, _ = sequence_world(Path("models"))
        with self.assertRaises(ValueError):
            verify_worlds({"normal": tree, "swapped": tree})
        with self.assertRaises(ValueError):
            verify_worlds({"normal": tree})
        with self.assertRaises(ValueError):
            target_poses("bad")
        self.assertEqual(target_poses()["probe_red"], (4., 1., 2.4, 0.))
        self.assertEqual(target_poses("swapped")["probe_red"], target_poses()["probe_blue"])

    def test_swapped_live_readback_must_match_private_layout(self):
        from types import SimpleNamespace as NS
        poses = [NS(name=m.name, position=m.pose.position, orientation=m.pose.orientation)
                 for m in scene_for("hidden").model]
        with self.assertRaises(ValueError):
            check_pose_packet(NS(pose=poses), "hidden", "swapped")
        for pose in poses:
            if pose.name in ("probe_red", "probe_blue"):
                pose.position.y = -pose.position.y
        check_pose_packet(NS(pose=poses), "hidden", "swapped")

    def test_rgb_only_reference_and_private_order_reverse(self):
        rgb, labels = pixels()
        self.assertEqual(rgb_order(rgb), {"red": "left", "blue": "right"})
        self.assertEqual(rgb_order(rgb), label_order(labels))
        self.assertEqual(rgb_order(rgb[:, ::-1]), {"red": "right", "blue": "left"})
        self.assertEqual(rgb_order(rgb[:, ::-1]), label_order(labels[:, ::-1]))

    def test_ambiguous_missing_or_tiny_target_rejected(self):
        rgb, labels = pixels()
        for bad in (np.zeros_like(rgb), rgb[:5]):
            with self.assertRaises(ValueError):
                rgb_order(bad)
        with self.assertRaises(ValueError):
            label_order(np.zeros_like(labels))
        rgb[:] = 0
        rgb[4:12, 4:8, 0], rgb[15:23, 4:8, 2] = 230, 230
        with self.assertRaises(ValueError):
            rgb_order(rgb)

    def test_even_one_hidden_pixel_difference_rejected(self):
        rgb, _ = pixels()
        self.assertEqual(len(equal_hidden([rgb, rgb.copy()])), 64)
        changed = rgb.copy()
        changed[0, 0, 0] += 1
        for rows in ([rgb], [rgb, changed], [rgb, rgb[:, :-1]]):
            with self.assertRaises(ValueError):
                equal_hidden(rows)

    def test_identical_current_requests_opposite_answers(self):
        rows = queries()
        verify_queries(rows)
        self.assertEqual(canonical(rows[0]["current_only"]), canonical(rows[1]["current_only"]))
        self.assertEqual(set(rows[0]["with_history"]),
                         {"schema", "question", "current_rgb_png_base64", "past_rgb_snapshots_png_base64"})

    def test_metadata_or_pixels_must_not_distinguish_current(self):
        for field, value in (("timestamp", 22), ("layout", "swapped"),
                             ("current_rgb_png_base64", {"uav_0": "changed"})):
            rows = queries()
            rows[1]["current_only"][field] = value
            with self.assertRaises(ValueError):
                verify_queries(rows)

    def test_history_cannot_change_question_current_or_add_answer(self):
        for field, value in (("question", "changed"), ("answer", "right"),
                             ("current_rgb_png_base64", {})):
            rows = queries()
            rows[1]["with_history"][field] = value
            with self.assertRaises(ValueError):
                verify_queries(rows)

    def test_same_history_same_answer_or_missing_pair_rejected(self):
        for change in ("history", "answer", "count"):
            rows = queries()
            if change == "history":
                rows[1]["with_history"] = deepcopy(rows[0]["with_history"])
            elif change == "answer":
                rows[1]["answer"] = rows[0]["answer"]
            else:
                rows.pop()
            with self.assertRaises(ValueError):
                verify_queries(rows)

    def test_queries_require_both_camera_views(self):
        for agent, color, images in (("uav_9", "red", {}), ("uav_0", "green", {}),
                                     ("uav_0", "red", {"uav_0": "rgb"})):
            with self.assertRaises(ValueError):
                request(agent, color, images)


if __name__ == "__main__":
    unittest.main()
