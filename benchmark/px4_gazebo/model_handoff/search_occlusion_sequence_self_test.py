"""Causal sequence rejection tests, without Gazebo, policy or flight."""

from copy import deepcopy
import math
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from .search_occlusion_sequence import (
    PHASES, TARGET_POSES, audit_timeline, camera_poses, check_scene, check_pose_packet,
    sequence_world, _load_message,
)


def timeline():
    result = []
    boundary = 100_000_000
    for phase in PHASES:
        gate = boundary + 1_000_000_000
        frames = []
        for i in range(5):
            stamp = gate + 100_000_000 + i * 250_000_000
            frames.append({"stamp_ns": stamp,
                           "agents": {agent: {"stamp_ns": stamp} for agent in ("uav_0", "uav_1")}})
        result.append({"phase": phase, "command_clock_ns": boundary,
                       "boundary_clock_ns": boundary, "gate_ns": gate, "frames": frames})
        boundary = stamp + 100_000_000
    return result


def scene_for(phase):
    models = []
    for name, (x, y, z, yaw) in {**TARGET_POSES, **camera_poses(phase)}.items():
        models.append(NS(name=name, pose=NS(position=NS(x=x, y=y, z=z),
                         orientation=NS(x=0., y=0., z=math.sin(yaw / 2), w=math.cos(yaw / 2)))))
    return NS(model=models)


class SequenceTests(unittest.TestCase):
    def test_one_second_sustained_joint_evidence(self):
        spans = audit_timeline(timeline())
        self.assertEqual(set(spans), set(PHASES))
        self.assertTrue(all(s["span_ns"] == 1_000_000_000 and s["frames"] == 5 for s in spans.values()))

    def test_reject_missing_reordered_or_duplicated_phases(self):
        rows = timeline()
        for bad in (rows[:2], list(reversed(rows)), [rows[0], rows[0], rows[2]]):
            with self.assertRaises(ValueError):
                audit_timeline(bad)

    def test_reject_reused_reset_or_unsynchronized_stamps(self):
        for change in ("reused", "reset", "mixed", "bool"):
            rows = timeline()
            if change == "reused":
                rows[0]["frames"][1] = deepcopy(rows[0]["frames"][0])
            elif change == "reset":
                rows[1]["boundary_clock_ns"] = 1
            elif change == "mixed":
                rows[0]["frames"][1]["agents"]["uav_1"]["stamp_ns"] += 1
            else:
                rows[0]["frames"][1]["stamp_ns"] = True
            with self.assertRaises(ValueError):
                audit_timeline(rows)

    def test_reject_too_short_or_gapped_phase(self):
        for change in ("short", "few", "gap"):
            rows = timeline()
            if change == "short":
                rows[1]["frames"].pop()
            elif change == "few":
                rows[1]["frames"] = rows[1]["frames"][:2]
            else:
                rows[1]["frames"].pop(1)
                rows[1]["frames"].pop(1)
            with self.assertRaises(ValueError):
                audit_timeline(rows)

    def test_reject_pre_settling_frames(self):
        for change in ("gate", "frame"):
            rows = timeline()
            if change == "gate":
                rows[0]["gate_ns"] -= 1
            else:
                rows[0]["gate_ns"] = rows[0]["frames"][0]["stamp_ns"]
            with self.assertRaises(ValueError):
                audit_timeline(rows)

    def test_both_cameras_required(self):
        rows = timeline()
        del rows[1]["frames"][0]["agents"]["uav_1"]
        with self.assertRaises(ValueError):
            audit_timeline(rows)

    def test_actual_readback_required_for_both_cameras_and_targets(self):
        for phase in PHASES:
            check_scene(scene_for(phase), phase)
        for name in ("annotation_camera_0", "annotation_camera_1", *TARGET_POSES):
            scene = scene_for("hidden")
            next(m for m in scene.model if m.name == name).pose.position.x += .01
            with self.assertRaises(ValueError):
                check_scene(scene, "hidden")

    def test_pose_nonfinite_or_extra_entity_rejected(self):
        scene = scene_for("seen")
        scene.model[0].pose.orientation.w = math.nan
        with self.assertRaises(ValueError):
            check_scene(scene, "seen")
        scene = scene_for("seen")
        scene.model.append(deepcopy(scene.model[0]))
        with self.assertRaises(ValueError):
            check_scene(scene, "seen")

    def test_world_retains_wall_and_has_only_static_rigs(self):
        with patch("benchmark.px4_gazebo.model_handoff.search_pilot_scene.verify"):
            tree, pairs = sequence_world(Path("camera-models"))
        world = tree.getroot().find("world")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(world.findall("plugin[@name='gz::sim::systems::UserCommands']")), 1)
        self.assertIsNotNone(world.find("model[@name='probe_wall']"))
        self.assertTrue(all(m.findtext("static") == "true" for m in world.findall("model")))
        self.assertNotEqual(camera_poses("seen"), camera_poses("reacquired"))
        with self.assertRaises(ValueError):
            camera_poses("unknown")

    def test_missing_or_duplicated_stream_root_pose_rejected(self):
        poses = [NS(name=m.name, position=m.pose.position, orientation=m.pose.orientation)
                 for m in scene_for("hidden").model]
        check_pose_packet(NS(pose=poses), "hidden")
        for bad in (poses[:-1], poses + [poses[0]]):
            with self.assertRaises(ValueError):
                check_pose_packet(NS(pose=bad), "hidden")

    def test_command_boundary_must_precede_settling_and_follow_prior_phase(self):
        for offset in (-1, 1_000_000_000):
            rows = timeline()
            rows[1]["command_clock_ns"] = (rows[0]["frames"][-1]["stamp_ns"] - 1
                                                if offset < 0 else rows[1]["boundary_clock_ns"] + offset)
            with self.assertRaises(ValueError):
                audit_timeline(rows)

    def test_raw_evidence_tampering_or_external_path_rejected(self):
        class Message:
            def ParseFromString(self, data):
                self.data = data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frame.pb").write_bytes(b"recorded evidence")
            record = {"file": "frame.pb", "sha256": hashlib.sha256(b"recorded evidence").hexdigest()}
            self.assertEqual(_load_message(root, record, Message).data, b"recorded evidence")
            (root / "frame.pb").write_bytes(b"changed evidence")
            with self.assertRaises(ValueError):
                _load_message(root, record, Message)
            record["file"] = "../frame.pb"
            with self.assertRaises(ValueError):
                _load_message(root, record, Message)


if __name__ == "__main__":
    unittest.main()
