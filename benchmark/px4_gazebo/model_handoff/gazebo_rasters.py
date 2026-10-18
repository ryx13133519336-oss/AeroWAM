"""Private Gazebo 8 instance-label decoder. Never infer IDs from color maps.

Only the verified gz-msgs10 RGB_INT8 wire format and Gazebo 8 panoptic
packing are supported. Scene-authored label IDs must uniquely identify each
object; renderer instance bytes are retained, not guessed to be entity IDs.
Equal stamps/dimensions alone do not prove two sensors are co-calibrated.
No Gazebo dependency is imported here; callers pass gz.msgs.Image messages.
"""

from dataclasses import dataclass

import numpy as np

from .language_evaluator import AnnotationError

GZ8_INSTANCE_ENCODING = "gz-sensors8-instance-label-last/1-draft"
RGB_INT8 = 3  # gz.msgs10 PixelFormatType; checked against installed descriptor.


def _require(condition, reason):
    if not condition:
        raise AnnotationError(reason)


@dataclass(frozen=True)
class DecodedRaster:
    timestamp_ns: int
    pixels: np.ndarray


@dataclass(frozen=True)
class PrivateRasterPair:
    timestamp_ns: int
    rgb: np.ndarray
    object_labels: np.ndarray
    instance_bytes: np.ndarray


def decode_rgb8(message) -> DecodedRaster:
    """Validate dimensions, timestamp, format, row padding and exact length."""
    width, height, step = message.width, message.height, message.step
    _require(type(width) is int and type(height) is int
             and 0 < width <= 4096 and 0 < height <= 4096, "Invalid Gazebo image dimensions")
    _require(type(step) is int and width * 3 <= step <= width * 3 + 4096,
             "Invalid Gazebo RGB row stride")
    _require(type(message.pixel_format_type) is int and message.pixel_format_type == RGB_INT8,
             "Unsupported Gazebo image format; RGB_INT8 required")
    _require(isinstance(message.data, bytes) and len(message.data) == step * height,
             "Gazebo image buffer length mismatch")
    sec, nsec = message.header.stamp.sec, message.header.stamp.nsec
    _require(type(sec) is int and sec >= 0 and type(nsec) is int and 0 <= nsec < 1_000_000_000,
             "Invalid Gazebo capture timestamp")
    pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)[:, :width * 3]
    pixels = pixels.reshape(height, width, 3).copy()
    pixels.setflags(write=False)
    return DecodedRaster(sec * 1_000_000_000 + nsec, pixels)


def pair_rgb_instance(*, rgb_message, label_message, rgb_topic: str,
                      segmentation_topic: str, label_topic: str,
                      encoding: str) -> PrivateRasterPair:
    """Exact-stamp join, without nearest-frame reuse or color-map fallback.

    Trusted caller binds topics to distinct, co-configured RGB/instance sensors.
    This performs a wire-level join only, not calibration/pose acceptance and
    not policy-delivery registration. `segmentation_topic` is its SDF base topic.
    """
    _require(encoding == GZ8_INSTANCE_ENCODING, "Unverified segmentation packing")
    _require(isinstance(segmentation_topic, str) and segmentation_topic.startswith("/")
             and not segmentation_topic.endswith("/"), "Invalid segmentation base topic")
    _require(label_topic == segmentation_topic + "/labels_map", "Actual labels_map required")
    _require(isinstance(rgb_topic, str) and rgb_topic.startswith("/")
             and rgb_topic != segmentation_topic and not rgb_topic.startswith(segmentation_topic + "/"),
             "Ordinary RGB topic required; segmentation previews are private")
    rgb, packed = decode_rgb8(rgb_message), decode_rgb8(label_message)
    _require(rgb.timestamp_ns == packed.timestamp_ns, "RGB/label capture stamp mismatch")
    _require(rgb.pixels.shape == packed.pixels.shape, "RGB/label raster dimensions mismatch")
    # Gazebo's panoptic output: first two channels = opaque instance count,
    # final channel = authored label. Do not rely on instance byte order.
    labels = packed.pixels[:, :, 2].copy()
    instances = packed.pixels[:, :, :2].copy()
    labels.setflags(write=False)
    instances.setflags(write=False)
    return PrivateRasterPair(rgb.timestamp_ns, rgb.pixels, labels, instances)


def resize_pair(pair: PrivateRasterPair, width: int = 256, height: int = 256) -> PrivateRasterPair:
    """Legacy full-image stretch: RGB bilinear, integer maps nearest.

    JPEG follows this transform. This aligns rasters, not exposure/pose state.
    """
    from PIL import Image
    _require(type(width) is int and type(height) is int and 0 < width <= 4096 and 0 < height <= 4096,
             "Invalid transformed raster dimensions")
    size = (width, height)
    rgb = np.asarray(Image.fromarray(pair.rgb).resize(size, Image.Resampling.BILINEAR)).copy()
    labels = np.asarray(Image.fromarray(pair.object_labels).resize(size, Image.Resampling.NEAREST)).copy()
    instances = np.stack([np.asarray(Image.fromarray(pair.instance_bytes[:, :, channel]).resize(
        size, Image.Resampling.NEAREST)) for channel in (0, 1)], axis=2)
    for array in (rgb, labels, instances):
        array.setflags(write=False)
    return PrivateRasterPair(pair.timestamp_ns, rgb, labels, instances)
