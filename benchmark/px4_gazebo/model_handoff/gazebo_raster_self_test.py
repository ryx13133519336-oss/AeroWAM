"""Wire-decoder regressions; synthetic messages do not prove calibration."""

from types import SimpleNamespace
import unittest

import numpy as np

from .gazebo_rasters import GZ8_INSTANCE_ENCODING, decode_rgb8, pair_rgb_instance, resize_pair
from .language_evaluator import AnnotationError


def message(pixels=None, padding=0, **changes):
    if pixels is None:
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
    h, w, _ = pixels.shape
    rows = np.full((h, w * 3 + padding), 255, dtype=np.uint8)
    rows[:, :w * 3] = pixels.reshape(h, w * 3)
    value = dict(width=w, height=h, step=w * 3 + padding, pixel_format_type=3,
                 data=rows.tobytes(), header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nsec=2)))
    value.update(changes)
    return SimpleNamespace(**value)


def pair(**changes):
    args = dict(rgb_message=message(), label_message=message(), rgb_topic="/probe/rgb",
                segmentation_topic="/probe/seg", label_topic="/probe/seg/labels_map",
                encoding=GZ8_INSTANCE_ENCODING)
    args.update(changes)
    return pair_rgb_instance(**args)


class GazeboRasterTests(unittest.TestCase):
    def test_padding_and_label_channel(self):
        pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        decoded = decode_rgb8(message(pixels, padding=5))
        np.testing.assert_array_equal(decoded.pixels, pixels)
        result = pair(label_message=message(pixels, padding=5))
        np.testing.assert_array_equal(result.object_labels, pixels[:, :, 2])
        np.testing.assert_array_equal(result.instance_bytes, pixels[:, :, :2])
        self.assertEqual(result.timestamp_ns, 1_000_000_002)
        self.assertFalse(result.object_labels.flags.writeable)

    def test_malformed_buffers_dimensions_and_format(self):
        for changes in ({"width": 0}, {"height": 4097}, {"width": True},
                        {"step": 11}, {"step": 5000}, {"pixel_format_type": 4},
                        {"data": b""}, {"data": bytes(37)}, {"data": bytearray(36)}):
            with self.subTest(changes=changes), self.assertRaises(AnnotationError):
                decode_rgb8(message(**changes))

    def test_invalid_timestamp(self):
        for sec, nsec in ((-1, 0), (1, -1), (1, 1_000_000_000), (True, 0)):
            header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=nsec))
            with self.assertRaises(AnnotationError):
                decode_rgb8(message(header=header))

    def test_no_color_preview_or_unverified_packing(self):
        for changes in ({"label_topic": "/probe/seg/colored_map"},
                        {"rgb_topic": "/probe/seg/colored_map"},
                        {"rgb_topic": "/probe/seg/labels_map"},
                        {"encoding": "guess"}, {"segmentation_topic": ""}):
            with self.assertRaises(AnnotationError):
                pair(**changes)

    def test_no_approximate_stamp_or_dimension_join(self):
        with self.assertRaises(AnnotationError):
            pair(label_message=message(header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nsec=3))))
        with self.assertRaises(AnnotationError):
            pair(label_message=message(np.zeros((3, 5, 3), dtype=np.uint8)))

    def test_model_transform_preserves_integer_labels(self):
        pixels = np.zeros((24, 32, 3), dtype=np.uint8)
        pixels[4:12, 4:12, 2] = 11
        pixels[14:20, 20:26, 2] = 22
        native = pair(rgb_message=message(pixels), label_message=message(pixels))
        transformed = resize_pair(native, 16, 12)
        self.assertEqual(transformed.rgb.shape, (12, 16, 3))
        self.assertEqual(set(np.unique(transformed.object_labels)), {0, 11, 22})
        self.assertEqual(transformed.timestamp_ns, native.timestamp_ns)
        self.assertFalse(transformed.rgb.flags.writeable)
        self.assertTrue(np.any(transformed.rgb[:, :, 2] != transformed.object_labels))
        with self.assertRaises(AnnotationError):
            resize_pair(native, True, 12)


if __name__ == "__main__":
    unittest.main()
