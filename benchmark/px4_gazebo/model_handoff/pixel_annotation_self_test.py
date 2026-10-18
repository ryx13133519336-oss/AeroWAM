"""Synthetic raster/geometry tests. No renderer or real RGB calibration claim."""

from dataclasses import replace
import unittest

import numpy as np

from .fixtures import Appearance
from .language_evaluator import AnnotationError, LanguageEvaluator
from .language_tasks import TargetReport, templated_task
from .pixel_annotations import CaptureKey, CaptureGeometry, VisibilityRules, ViewRules, annotate, inspection_view

OBJECTS = {"red": Appearance("red", "sphere", "black"), "blue": Appearance("blue", "cube", "white")}
MAP = {0: None, 1: "red", 2: "red", 3: "blue", 4: None}


def capture(t=0, agent="uav_0"):
    return CaptureKey(agent, t, t, f"render-{agent}-{t}", 32, 24)


def raster():
    image = np.zeros((24, 32), dtype=np.uint16)
    image[4:12, 4:12] = 1
    image[6:8, 4:12] = 2  # Stripe belongs to the same instance.
    image[14:20, 20:26] = 3
    return image


def task(kind="search"):
    return templated_task(kind, OBJECTS["red"]).bind(OBJECTS)


def build(**changes):
    args = dict(task=task(), rgb_capture=capture(), label_capture=capture(),
                instance_labels=raster(), label_to_object=MAP)
    args.update(changes)
    return annotate(**args)


def geometry(key, position=(2., 0., 0.), rotation=None):
    kwargs = {} if rotation is None else {"object_to_world": rotation}
    return {name: CaptureGeometry(key, position, (0., 0., 0.), **kwargs) for name in OBJECTS}


class PixelTests(unittest.TestCase):
    def test_visible_pixels_and_stripe_union(self):
        bundle = build()
        self.assertEqual(bundle.frame.objects["red"].visible_bbox, (4/32, 4/24, 12/32, 12/24))
        self.assertEqual(bundle.diagnostics["red"]["visible_pixels"], 64)
        self.assertNotIn("label_to_object", bundle.frame.__dict__)

    def test_occluded_pixels_not_amodal(self):
        image = raster()
        image[4:12, 4:8] = 4  # Known wall pixels obscure the left half.
        box = build(instance_labels=image).frame.objects["red"].visible_bbox
        self.assertEqual(box, (8/32, 4/24, 12/32, 12/24))
        image[(image == 1) | (image == 2)] = 4
        self.assertIsNone(build(instance_labels=image).frame.objects["red"].visible_bbox)

    def test_low_support_sparse_and_border(self):
        for image, expected in ((np.zeros((24, 32), dtype=np.uint16), "no_visible_pixels"),
                                (raster(), "border_clipped")):
            if expected == "border_clipped":
                image[4:12, :4] = 1
            self.assertEqual(build(instance_labels=image).diagnostics["red"]["reason"], expected)
        image = np.zeros((24, 32), dtype=np.int32)
        image[5, 5] = 1
        self.assertEqual(build(instance_labels=image).diagnostics["red"]["reason"], "insufficient_pixel_support")
        image[18, 26] = 1
        self.assertEqual(build(instance_labels=image, visibility=VisibilityRules(1, 1, 1)).diagnostics["red"]["reason"],
                         "sparse_pixel_support")

    def test_capture_mismatch_rejected(self):
        for change in ({"image_timestamp_ns": 1}, {"sim_time_ns": 1}, {"render_id": "other"},
                       {"agent_id": "uav_1"}, {"width": 64}):
            with self.assertRaises(AnnotationError):
                build(label_capture=replace(capture(), **change))

    def test_mapping_and_raster_rejected(self):
        for mapping in ({0: None, 1: "red"}, {**MAP, 0: "red"}, {**MAP, 5: "unknown"},
                        {key: value for key, value in MAP.items() if key != 2}):
            with self.assertRaises(AnnotationError):
                build(label_to_object=mapping)
        for image in (raster().astype(float), np.ones((24, 32), dtype=bool),
                      np.full((24, 32), -1, dtype=np.int32), np.zeros((12, 16), dtype=np.uint8)):
            with self.assertRaises(AnnotationError):
                build(instance_labels=image)

    def test_view_sectors_distance_elevation_rotation(self):
        key = capture()
        for position, expected in (((2., 0., 0.), "front"), ((0., 2., 0.), "side"),
                                   ((0., -2., 0.), "side"), ((-2., 0., 0.), None),
                                   ((2., 2., 0.), None), ((.1, 0., 0.), None),
                                   ((7., 0., 0.), None), ((1., 0., 2.), None)):
            self.assertEqual(inspection_view(CaptureGeometry(key, position, (0., 0., 0.))), expected)
        rotation = (0., -1., 0., 1., 0., 0., 0., 0., 1.)
        self.assertEqual(inspection_view(CaptureGeometry(key, (0., 2., 0.), (0., 0., 0.), rotation)), "front")
        for invalid in ((1.,)*9, (-1., 0., 0., 0., 1., 0., 0., 0., 1.)):
            with self.assertRaises(AnnotationError):
                CaptureGeometry(key, (2., 0., 0.), (0., 0., 0.), invalid)
        with self.assertRaises(AnnotationError):
            ViewRules(half_angle_deg=45.)

    def test_geometry_required_and_aligned(self):
        with self.assertRaises(AnnotationError):
            build(task=task("inspection"))
        with self.assertRaises(AnnotationError):
            build(task=task("inspection"), geometry=geometry(capture(1)))
        with self.assertRaises(AnnotationError):
            build(geometry={"red": geometry(capture())["red"]})
        bundle = build(task=task("inspection"), geometry=geometry(capture(), (2., 2., 0.)))
        self.assertTrue(bundle.frame.objects["red"].inspection_geometry_available)
        self.assertIsNone(bundle.frame.objects["red"].inspection_view)
        ev = LanguageEvaluator(task("inspection"))
        ev.register_observed_frame(bundle.frame)
        ev.report(TargetReport("uav_0", 0, bundle.frame.objects["red"].visible_bbox, .9, "inspect"), now_sim_time_ns=0)
        self.assertEqual(ev.finish()["reason"], "incomplete_task_evidence")

    def test_evaluator_integration_without_geometry_visibility_shortcut(self):
        ev = LanguageEvaluator(task("search"))
        for agent, t, event in (("uav_0", 0, "found"), ("uav_1", 100_000_000, "confirm")):
            key = capture(t, agent)
            bundle = build(rgb_capture=key, label_capture=key)
            ev.register_observed_frame(bundle.frame)
            ev.report(TargetReport(agent, t, bundle.frame.objects["red"].visible_bbox, .9, event), now_sim_time_ns=t)
        self.assertEqual(ev.finish()["category"], "success")
        image = raster()
        image[(image == 1) | (image == 2)] = 0
        bundle = build(instance_labels=image, geometry=geometry(capture()))
        self.assertIsNone(bundle.frame.objects["red"].visible_bbox)
        self.assertIsNone(bundle.frame.objects["red"].inspection_view)

    def test_no_view_frame_interrupts_inspection_dwell(self):
        for interrupted_position in ((2., 2., 0.), (0., 2., 0.)):
            ev = LanguageEvaluator(task("inspection"))
            for ns in (0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 450_000_000, 500_000_000):
                key = capture(ns)
                position = interrupted_position if ns == 450_000_000 else (2., 0., 0.)
                bundle = build(task=task("inspection"), rgb_capture=key, label_capture=key, geometry=geometry(key, position))
                ev.register_observed_frame(bundle.frame)
                if ns != 450_000_000:
                    ev.report(TargetReport("uav_0", ns, bundle.frame.objects["red"].visible_bbox, .9, "inspect"), now_sim_time_ns=ns)
            self.assertFalse(ev.views["uav_0"])


if __name__ == "__main__":
    unittest.main()
