"""Private draft raster annotations; no projection-only visibility labels.

The trusted producer must deliver co-rendered RGB/instance labels in the same
model-image raster. Matching metadata is necessary, not proof of calibration.
This module neither generates segmentation nor feeds truth to a policy.
"""

from dataclasses import dataclass
import math
from typing import Mapping

from .language_evaluator import AnnotationError, ObjectAnnotation, ObservedFrame
from .language_tasks import BoundTask, finite_number, integer
from .protocol import AGENT_IDS

ANNOTATION_SCHEMA = "aerowam-pixel-annotation/1-draft"


def _require(condition, reason):
    if not condition:
        raise AnnotationError(reason)


def _vector(value, size, name):
    _require(isinstance(value, tuple) and len(value) == size, f"Invalid {name}")
    for item in value:
        finite_number(item, name)


@dataclass(frozen=True)
class CaptureKey:
    agent_id: str
    image_timestamp_ns: int
    sim_time_ns: int
    render_id: str
    width: int
    height: int

    def __post_init__(self):
        _require(self.agent_id in AGENT_IDS, "Unknown camera")
        integer(self.image_timestamp_ns, "image stamp")
        integer(self.sim_time_ns, "simulation capture time")
        _require(isinstance(self.render_id, str) and bool(self.render_id), "Render identity required")
        _require(type(self.width) is int and type(self.height) is int
                 and 0 < self.width <= 4096 and 0 < self.height <= 4096, "Invalid raster dimensions")


@dataclass(frozen=True)
class VisibilityRules:
    min_pixels: int = 16
    min_box_width: int = 3
    min_box_height: int = 3
    min_box_fill: float = .15
    reject_border: bool = True

    def __post_init__(self):
        for value in (self.min_pixels, self.min_box_width, self.min_box_height):
            _require(type(value) is int and value > 0, "Positive integer visibility thresholds required")
        finite_number(self.min_box_fill, "box fill")
        _require(0 < self.min_box_fill <= 1 and type(self.reject_border) is bool, "Invalid visibility thresholds")


@dataclass(frozen=True)
class ViewRules:
    min_distance_m: float = .5
    max_distance_m: float = 6.
    half_angle_deg: float = 25.
    max_elevation_deg: float = 30.

    def __post_init__(self):
        for value in (self.min_distance_m, self.max_distance_m, self.half_angle_deg, self.max_elevation_deg):
            finite_number(value, "view threshold")
        _require(0 < self.min_distance_m < self.max_distance_m, "Invalid view distance")
        _require(0 < self.half_angle_deg < 45 and 0 < self.max_elevation_deg < 90, "Overlapping or invalid view sectors")


@dataclass(frozen=True)
class CaptureGeometry:
    capture: CaptureKey
    camera_position_enu: tuple[float, float, float]
    object_position_enu: tuple[float, float, float]
    # Row-major rotation mapping object-local XYZ to world ENU. Local +X
    # defines front; +/-Y defines side. Not inferred from RGB or UAV yaw.
    object_to_world: tuple[float, ...] = (1., 0., 0., 0., 1., 0., 0., 0., 1.)

    def __post_init__(self):
        _require(isinstance(self.capture, CaptureKey), "Geometry capture required")
        _vector(self.camera_position_enu, 3, "camera position")
        _vector(self.object_position_enu, 3, "object position")
        _vector(self.object_to_world, 9, "object rotation")
        import numpy as np
        rotation = np.asarray(self.object_to_world, dtype=float).reshape(3, 3)
        _require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                 and abs(np.linalg.det(rotation) - 1.) <= 1e-6, "Rotation must be proper orthonormal")


def inspection_view(geometry: CaptureGeometry, rules: ViewRules = ViewRules()) -> str | None:
    import numpy as np
    rotation = np.asarray(geometry.object_to_world).reshape(3, 3)
    local = rotation.T @ (np.asarray(geometry.camera_position_enu) - np.asarray(geometry.object_position_enu))
    distance = float(np.linalg.norm(local))
    if not rules.min_distance_m <= distance <= rules.max_distance_m:
        return None
    azimuth = abs(math.degrees(math.atan2(float(local[1]), float(local[0]))))
    elevation = abs(math.degrees(math.atan2(float(local[2]), math.hypot(float(local[0]), float(local[1])))))
    if elevation > rules.max_elevation_deg:
        return None
    if azimuth <= rules.half_angle_deg:
        return "front"
    if abs(azimuth - 90.) <= rules.half_angle_deg:
        return "side"
    return None


@dataclass(frozen=True)
class AnnotationBundle:
    frame: ObservedFrame
    diagnostics: Mapping[str, dict]
    schema: str = ANNOTATION_SCHEMA


def annotate(*, task: BoundTask, rgb_capture: CaptureKey, label_capture: CaptureKey,
             instance_labels, label_to_object: Mapping[int, str | None],
             geometry: Mapping[str, CaptureGeometry] | None = None,
             visibility: VisibilityRules = VisibilityRules(), views: ViewRules = ViewRules()) -> AnnotationBundle:
    """Consume private visible-instance raster, not RGB color heuristics.

    Every label (including known scenery) needs an explicit mapping; zero is
    background. Multiple labels may map to one object's body/stripe. Raster
    resize/crop must already match the exact policy RGB transform. Caller
    registers returned frame only after RGB delivery and handles errors as
    evaluator_error, not model_failure.
    """
    import numpy as np
    _require(isinstance(task, BoundTask) and isinstance(rgb_capture, CaptureKey)
             and isinstance(label_capture, CaptureKey), "Task and capture metadata required")
    _require(rgb_capture == label_capture, "RGB/instance capture mismatch")
    _require(isinstance(visibility, VisibilityRules) and isinstance(views, ViewRules), "Annotation rules required")
    _require(isinstance(instance_labels, np.ndarray) and instance_labels.ndim == 2
             and instance_labels.shape == (rgb_capture.height, rgb_capture.width), "Instance raster shape mismatch")
    _require(instance_labels.dtype.kind in {"u", "i"} and bool(np.all(instance_labels >= 0)),
             "Instance raster must contain nonnegative integer labels")
    _require(isinstance(label_to_object, Mapping) and bool(label_to_object), "Private label map required")
    identities = set(task.object_ids)
    _require(all(type(key) is int and key >= 0 and (value is None or isinstance(value, str) and value in identities)
                 for key, value in label_to_object.items()), "Invalid scene label mapping")
    _require(0 in label_to_object and label_to_object[0] is None, "Zero must map to background")
    _require({value for value in label_to_object.values() if value is not None} == identities,
             "Every scene object needs a label mapping, even when invisible")
    _require(set(int(value) for value in np.unique(instance_labels)) <= set(label_to_object),
             "Unmapped renderer label")
    if geometry is not None:
        _require(isinstance(geometry, Mapping) and set(geometry) == identities, "Incomplete geometry mapping")
        _require(all(isinstance(item, CaptureGeometry) and item.capture == rgb_capture for item in geometry.values()),
                 "Geometry/capture mismatch")
    if task.definition.task == "inspection":
        _require(geometry is not None, "Inspection requires capture-aligned private geometry")
        _require(set(task.definition.rules.inspection_views) == {"front", "side"}, "Unsupported view geometry contract")
    annotations, diagnostics = {}, {}
    for identity in task.object_ids:
        labels = [key for key, value in label_to_object.items() if value == identity]
        ys, xs = np.nonzero(np.isin(instance_labels, labels))
        count = len(xs)
        box, reason = None, "no_visible_pixels"
        fill = 0.
        if count:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            width, height = x2 - x1, y2 - y1
            fill = count / (width * height)
            if count < visibility.min_pixels or width < visibility.min_box_width or height < visibility.min_box_height:
                reason = "insufficient_pixel_support"
            elif fill < visibility.min_box_fill:
                reason = "sparse_pixel_support"
            elif visibility.reject_border and (x1 == 0 or y1 == 0 or x2 == rgb_capture.width or y2 == rgb_capture.height):
                reason = "border_clipped"
            else:
                box = (x1 / rgb_capture.width, y1 / rgb_capture.height, x2 / rgb_capture.width, y2 / rgb_capture.height)
                reason = "visible_pixel_support"
        view = inspection_view(geometry[identity], views) if box is not None and geometry is not None else None
        annotations[identity] = ObjectAnnotation(box, view, geometry is not None)
        diagnostics[identity] = {"visible_pixels": count, "box_fill": fill, "reason": reason,
                                 "inspection_view": view}
    return AnnotationBundle(ObservedFrame(rgb_capture.agent_id, rgb_capture.image_timestamp_ns,
                                         rgb_capture.sim_time_ns, annotations), diagnostics)
