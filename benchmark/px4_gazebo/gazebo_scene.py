"""Gazebo scene synchronization for the live Handoff benchmark."""

from __future__ import annotations

import math
from queue import Empty, Full, Queue
import re
import subprocess
import time
from threading import Event, Thread
from typing import Any, Mapping, Sequence

from .contract import HandoffTask
from .policies import handoff_demo_waypoints


OCCLUDER_ENTITY_NAME = "aerowam_handoff_occluder"
TARGET_ENTITY_NAME = "aerowam_handoff_target"
STAGING_ZONE_ENTITY_NAME = "aerowam_handoff_staging_zone"
TRANSFER_ZONE_ENTITY_NAME = "aerowam_handoff_transfer_zone"
ACQUISITION_ZONE_ENTITY_NAME = "aerowam_handoff_acquisition_zone"
OBSERVATION_ZONE_ENTITY_NAME = "aerowam_handoff_observation_zone"
CLEARANCE_ZONE_ENTITY_NAME = "aerowam_handoff_clearance_zone"
SUPPORT_ZONE_ENTITY_NAME = "aerowam_handoff_support_zone"
ROUTE_MARKER_ENTITY_NAME = "aerowam_handoff_route_marker_0"
ROUTE_MARKER_ENTITY_NAMES = tuple(
    f"aerowam_handoff_route_marker_{index}" for index in range(6)
)
DISTRACTOR_ENTITY_NAMES = (
    "aerowam_handoff_distractor_same_color",
    "aerowam_handoff_distractor_same_shape",
)

_COLORS = {
    "red": (0.90, 0.04, 0.03),
    "green": (0.03, 0.72, 0.10),
    "blue": (0.03, 0.20, 0.92),
    "yellow": (0.95, 0.78, 0.03),
}


def _appearance_value(value: Any, key: str, default: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get(key, default))
    return str(getattr(value, key, default))


def _shape_geometry(shape: str) -> str:
    if shape == "sphere":
        return "<sphere><radius>0.32</radius></sphere>"
    if shape == "cube":
        return "<box><size>0.58 0.58 0.58</size></box>"
    if shape == "cylinder":
        return "<cylinder><radius>0.30</radius><length>0.68</length></cylinder>"
    raise ValueError(f"unsupported target shape: {shape}")


def _appearance_sdf(name: str, appearance: Any, pose: Sequence[float]) -> str:
    color = _appearance_value(appearance, "color", "red")
    shape = _appearance_value(appearance, "shape", "sphere")
    stripe = _appearance_value(appearance, "stripe", "black")
    if color not in _COLORS or stripe not in {"black", "white"}:
        raise ValueError(f"unsupported appearance: {color}/{shape}/{stripe}")
    red, green, blue = _COLORS[color]
    stripe_value = 0.02 if stripe == "black" else 0.96
    x, y, z = (float(item) for item in pose)
    return _compact_xml(
        f"""
        <sdf version='1.9'>
          <model name='{name}'>
            <static>true</static>
            <pose>{x} {y} {z} 0 0 0</pose>
            <link name='object'>
              <visual name='body'>
                <geometry>{_shape_geometry(shape)}</geometry>
                <material>
                  <ambient>{red} {green} {blue} 1</ambient>
                  <diffuse>{red} {green} {blue} 1</diffuse>
                  <emissive>{0.12 * red} {0.12 * green} {0.12 * blue} 1</emissive>
                </material>
              </visual>
              <visual name='wide_identity_stripe'>
                <pose>0 0 0 0 0 0</pose>
                <geometry><box><size>0.68 0.13 0.20</size></box></geometry>
                <material>
                  <ambient>{stripe_value} {stripe_value} {stripe_value} 1</ambient>
                  <diffuse>{stripe_value} {stripe_value} {stripe_value} 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
    )


def task_flu_to_gazebo_enu(
    point: Sequence[float], initial_yaw: float
) -> tuple[float, float, float]:
    """Convert UAV0-initial body FLU task coordinates to Gazebo ENU."""
    forward, left, up = (float(value) for value in point)
    north = forward * math.cos(initial_yaw) + left * math.sin(initial_yaw)
    east = forward * math.sin(initial_yaw) - left * math.cos(initial_yaw)
    return east, north, up


def _compact_xml(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _marker_sdf(
    name: str,
    pose: Sequence[float],
    color: Sequence[float],
) -> str:
    """Small non-colliding beacon placed below a flight waypoint."""
    x, y, _ = (float(value) for value in pose)
    red, green, blue = (float(value) for value in color)
    return _compact_xml(
        f"""
        <sdf version='1.9'>
          <model name='{name}'>
            <static>true</static>
            <pose>{x} {y} 0.18 0 0 0</pose>
            <link name='beacon'>
              <visual name='pin'>
                <geometry><cylinder><radius>0.18</radius><length>0.36</length></cylinder></geometry>
                <material>
                  <ambient>{red} {green} {blue} 0.9</ambient>
                  <diffuse>{red} {green} {blue} 0.9</diffuse>
                  <emissive>{0.18 * red} {0.18 * green} {0.18 * blue} 1</emissive>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
    )


def _route_segment_sdf(
    name: str,
    start: Sequence[float],
    end: Sequence[float],
    color: Sequence[float],
) -> str:
    """Non-colliding ground segment connecting two actual controller waypoints."""
    x0, y0, _ = (float(value) for value in start)
    x1, y1, _ = (float(value) for value in end)
    red, green, blue = (float(value) for value in color)
    length = max(0.05, math.hypot(x1 - x0, y1 - y0))
    yaw = math.atan2(y1 - y0, x1 - x0)
    return _compact_xml(
        f"""
        <sdf version='1.9'>
          <model name='{name}'>
            <static>true</static>
            <pose>{(x0 + x1) / 2.0} {(y0 + y1) / 2.0} 0.08 0 0 {yaw}</pose>
            <link name='route'>
              <visual name='route'>
                <geometry><box><size>{length} 0.07 0.04</size></box></geometry>
                <material>
                  <ambient>{red} {green} {blue} 0.75</ambient>
                  <diffuse>{red} {green} {blue} 0.75</diffuse>
                  <emissive>{0.12 * red} {0.12 * green} {0.12 * blue} 1</emissive>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
    )


def _spawn_offset_to_task_flu(
    offset_ned: Sequence[float], initial_yaw: float
) -> tuple[float, float, float]:
    north, east, down = (float(value) for value in offset_ned)
    return (
        north * math.cos(initial_yaw) + east * math.sin(initial_yaw),
        north * math.sin(initial_yaw) - east * math.cos(initial_yaw),
        -down,
    )


DEMO_MARKER_ENTITY_NAMES = (
    ACQUISITION_ZONE_ENTITY_NAME,
    OBSERVATION_ZONE_ENTITY_NAME,
    CLEARANCE_ZONE_ENTITY_NAME,
    STAGING_ZONE_ENTITY_NAME,
    TRANSFER_ZONE_ENTITY_NAME,
    SUPPORT_ZONE_ENTITY_NAME,
    *ROUTE_MARKER_ENTITY_NAMES,
)


class GazeboSceneController:
    """Create and update the benchmark wall and target through Gazebo services."""

    def __init__(self, world: str, *, timeout_ms: int = 3000) -> None:
        self.world = str(world)
        self.timeout_ms = int(timeout_ms)
        self.initial_yaw = 0.0
        self.last_error: str | None = None
        self._spawned = False
        self._closed = False
        self._updates_disabled = False
        self._stop = Event()
        self._poses: Queue[tuple[float, float, float] | None] = Queue(maxsize=1)
        self._worker = Thread(
            target=self._pose_worker,
            name="aerowam-gazebo-target-pose",
            daemon=True,
        )
        self._worker.start()

    def _service(
        self,
        service: str,
        reqtype: str,
        reptype: str,
        request: str,
        *,
        attempts: int = 1,
    ) -> None:
        last_error = "unknown error"
        for attempt in range(max(1, int(attempts))):
            try:
                completed = subprocess.run(
                    [
                        "gz", "service", "-s", service,
                        "--reqtype", reqtype, "--reptype", reptype,
                        "--timeout", str(self.timeout_ms), "--req", request,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(5.0, self.timeout_ms / 1000.0 + 2.0),
                )
                output = "\n".join(
                    part for part in (completed.stdout, completed.stderr) if part
                ).strip()
                if completed.returncode == 0 and re.search(
                    r"\bdata\s*:\s*true\b", output, re.IGNORECASE
                ):
                    return
                last_error = output or f"exit code {completed.returncode}"
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_error = str(exc)
            if attempt + 1 < max(1, int(attempts)):
                time.sleep(0.5)
        raise RuntimeError(f"Gazebo service {service} failed: {last_error}")

    def _remove(self, name: str) -> None:
        try:
            self._service(
                f"/world/{self.world}/remove",
                "gz.msgs.Entity",
                "gz.msgs.Boolean",
                f'name: "{name}" type: MODEL',
            )
        except Exception:
            pass

    def _create(self, name: str, sdf: str) -> None:
        self._service(
            f"/world/{self.world}/create",
            "gz.msgs.EntityFactory",
            "gz.msgs.Boolean",
            f'name: "{name}" allow_renaming: false sdf: "{sdf}"',
            attempts=3,
        )

    def spawn(
        self,
        task: HandoffTask,
        initial_yaw: float,
        spawn_offset_ned: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        target_appearance: Any | None = None,
        distractor_appearances: Sequence[Any] = (),
        show_markers: bool = True,
    ) -> None:
        """Spawn a physical wall and a visible target before flight starts."""
        if self._closed:
            raise RuntimeError("Gazebo scene controller is already closed")
        if len(spawn_offset_ned) != 3:
            raise ValueError("spawn_offset_ned must contain N E D")
        self.initial_yaw = float(initial_yaw)
        self.last_error = None
        self._updates_disabled = False
        for name in (
            TARGET_ENTITY_NAME,
            OCCLUDER_ENTITY_NAME,
            *DEMO_MARKER_ENTITY_NAMES,
            *DISTRACTOR_ENTITY_NAMES,
        ):
            self._remove(name)

        wall_x, wall_y, wall_z = task_flu_to_gazebo_enu(
            task.occluder_center, self.initial_yaw
        )
        wall_yaw = math.pi / 2.0 - self.initial_yaw
        size_x, size_y, size_z = (
            2.0 * float(value) for value in task.occluder_half_extents
        )
        wall_sdf = _compact_xml(
            f"""
            <sdf version='1.9'>
              <model name='{OCCLUDER_ENTITY_NAME}'>
                <static>true</static>
                <pose>{wall_x} {wall_y} {wall_z} 0 0 {wall_yaw}</pose>
                <link name='wall'>
                  <collision name='collision'>
                    <geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry>
                  </collision>
                  <visual name='visual'>
                    <geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry>
                    <material>
                      <ambient>0.30 0.32 0.34 1</ambient>
                      <diffuse>0.42 0.44 0.46 1</diffuse>
                      <specular>0.10 0.10 0.10 1</specular>
                    </material>
                  </visual>
                  <visual name='hazard_band'>
                    <pose>0 0 {-size_z / 2.0 + 0.12} 0 0 0</pose>
                    <geometry><box><size>{size_x + 0.03} {size_y + 0.03} 0.24</size></box></geometry>
                    <material>
                      <ambient>0.92 0.66 0.03 1</ambient>
                      <diffuse>0.98 0.72 0.04 1</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>
            """
        )
        target_x, target_y, target_z = task_flu_to_gazebo_enu(
            task.target_start, self.initial_yaw
        )
        target_sdf = _appearance_sdf(
            TARGET_ENTITY_NAME,
            target_appearance or {"color": "red", "shape": "sphere", "stripe": "black"},
            (target_x, target_y, target_z),
        )
        if show_markers and not distractor_appearances:
            distractor_appearances = (
                {"color": "red", "shape": "cube", "stripe": "black"},
                {"color": "blue", "shape": "sphere", "stripe": "black"},
            )
        distractor_sdfs: list[tuple[str, str]] = []
        if distractor_appearances:
            if len(distractor_appearances) != 2:
                raise ValueError("formal scene requires exactly two distractor appearances")
            distractor_task_positions = (
                (0.0, 2.4, task.target_start[2]),
                (6.0, 1.4, task.target_start[2]),
            )
            for name, appearance, task_position in zip(
                DISTRACTOR_ENTITY_NAMES, distractor_appearances, distractor_task_positions
            ):
                pose = task_flu_to_gazebo_enu(task_position, self.initial_yaw)
                distractor_sdfs.append((name, _appearance_sdf(name, appearance, pose)))
        # Demo-only beacons and paths use the exact controller waypoints.
        # Formal model episodes hide all of them.
        waypoint_task = handoff_demo_waypoints(task)
        waypoint_names = {
            "uav0_acquisition": ACQUISITION_ZONE_ENTITY_NAME,
            "uav0_observation": OBSERVATION_ZONE_ENTITY_NAME,
            "uav0_support": SUPPORT_ZONE_ENTITY_NAME,
            "uav1_clearance": CLEARANCE_ZONE_ENTITY_NAME,
            "uav1_staging": STAGING_ZONE_ENTITY_NAME,
            "handoff": TRANSFER_ZONE_ENTITY_NAME,
        }
        waypoint_colors = {
            "uav0_acquisition": (0.02, 0.22, 0.82),
            "uav0_observation": (0.05, 0.35, 0.95),
            "uav0_support": (0.10, 0.85, 0.90),
            "uav1_clearance": (0.98, 0.38, 0.04),
            "uav1_staging": (0.03, 0.78, 0.12),
            "handoff": (0.98, 0.78, 0.03),
        }
        waypoint_gazebo = {
            key: task_flu_to_gazebo_enu(value, self.initial_yaw)
            for key, value in waypoint_task.items()
        }
        marker_sdfs = [
            (
                waypoint_names[key],
                _marker_sdf(
                    waypoint_names[key], waypoint_gazebo[key], waypoint_colors[key]
                ),
            )
            for key in waypoint_names
        ]
        uav1_spawn = _spawn_offset_to_task_flu(spawn_offset_ned, self.initial_yaw)
        uav1_spawn = (uav1_spawn[0], uav1_spawn[1], task.staging_position[2])
        route_specs = (
            (
                (0.0, 0.0, task.staging_position[2]),
                waypoint_task["uav0_acquisition"],
                (0.02, 0.22, 0.82),
            ),
            (
                waypoint_task["uav0_acquisition"],
                waypoint_task["uav0_observation"],
                (0.05, 0.35, 0.95),
            ),
            (
                waypoint_task["uav0_observation"],
                waypoint_task["uav0_support"],
                (0.10, 0.85, 0.90),
            ),
            (
                uav1_spawn,
                waypoint_task["uav1_clearance"],
                (0.98, 0.38, 0.04),
            ),
            (
                waypoint_task["uav1_clearance"],
                waypoint_task["uav1_staging"],
                (0.03, 0.78, 0.12),
            ),
            (
                waypoint_task["uav1_staging"],
                waypoint_task["handoff"],
                (0.98, 0.78, 0.03),
            ),
        )
        route_sdfs = []
        for name, (start, end, color) in zip(ROUTE_MARKER_ENTITY_NAMES, route_specs):
            start_gazebo = task_flu_to_gazebo_enu(start, self.initial_yaw)
            end_gazebo = task_flu_to_gazebo_enu(end, self.initial_yaw)
            route_sdfs.append(
                (name, _route_segment_sdf(name, start_gazebo, end_gazebo, color))
            )
        try:
            self._create(OCCLUDER_ENTITY_NAME, wall_sdf)
            self._create(TARGET_ENTITY_NAME, target_sdf)
            for name, sdf in distractor_sdfs:
                self._create(name, sdf)
            if show_markers:
                for name, sdf in (*marker_sdfs, *route_sdfs):
                    self._create(name, sdf)
        except Exception:
            for name in (
                TARGET_ENTITY_NAME,
                OCCLUDER_ENTITY_NAME,
                *DEMO_MARKER_ENTITY_NAMES,
                *DISTRACTOR_ENTITY_NAMES,
            ):
                self._remove(name)
            raise
        self._spawned = True

    def update_target(self, task_position: Sequence[float]) -> None:
        """Queue only the newest target pose for asynchronous Gazebo updates."""
        if (
            not self._spawned
            or self._closed
            or self._updates_disabled
            or self.last_error is not None
        ):
            return
        pose = task_flu_to_gazebo_enu(task_position, self.initial_yaw)
        try:
            self._poses.put_nowait(pose)
        except Full:
            try:
                self._poses.get_nowait()
                self._poses.task_done()
            except Empty:
                pass
            try:
                self._poses.put_nowait(pose)
            except Full:
                pass

    def disable_updates(self) -> None:
        """Disable optional pose updates while keeping the scene entities."""
        self._updates_disabled = True
        self.last_error = None
        try:
            while True:
                self._poses.get_nowait()
                self._poses.task_done()
        except Empty:
            pass

    def _pose_worker(self) -> None:
        while not self._stop.is_set():
            try:
                pose = self._poses.get(timeout=0.2)
            except Empty:
                continue
            try:
                if pose is None:
                    return
                x, y, z = pose
                self._service(
                    f"/world/{self.world}/set_pose",
                    "gz.msgs.Pose",
                    "gz.msgs.Boolean",
                    (
                        f'name: "{TARGET_ENTITY_NAME}" '
                        f'position {{ x: {x} y: {y} z: {z} }} '
                        "orientation { w: 1 }"
                    ),
                )
            except Exception as exc:
                self.last_error = str(exc)
            finally:
                self._poses.task_done()

    def close(self) -> None:
        """Stop updates and remove scene entities; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            while True:
                self._poses.get_nowait()
                self._poses.task_done()
        except Empty:
            pass
        try:
            self._poses.put_nowait(None)
        except Full:
            pass
        self._worker.join(timeout=max(5.0, self.timeout_ms / 1000.0 + 2.0))
        if self._spawned:
            for name in (
                TARGET_ENTITY_NAME,
                OCCLUDER_ENTITY_NAME,
                *DEMO_MARKER_ENTITY_NAMES,
                *DISTRACTOR_ENTITY_NAMES,
            ):
                self._remove(name)
        self._spawned = False
