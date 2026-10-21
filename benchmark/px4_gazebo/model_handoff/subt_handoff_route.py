"""Private stair-route design and conservative mesh checks; no flight admission."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .subt_urban_geometry import load_triangles, segment_hits, triangles_touch_box

SCHEMA = "aerowam-subt-stair-handoff-route/1"
# Floor reference, not object centers. Both flights are in the original mesh.
WAYPOINTS = np.array([
    (-6., -6., 0.), (-9.03, -6., 0.), (-9.03, -8.4, 0.),
    (-9.03, -14.35, 5.), (-9.03, -15.3, 5.),
    (-10.93, -15.3, 5.), (-10.93, -14.25, 5.),
    (-10.93, -8.6, 10.), (-10.93, -6., 10.), (-6., -6., 10.),
], dtype=float)
TARGET_LOW = np.array([-.45, -.45, -.90])
TARGET_HIGH = np.array([.45, .45, .45])
TARGET_OFFSET = 1.15
AGENTS = ("uav_0", "uav_1")


def resample(points, spacing=.12):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or not np.isfinite(points).all() or not math.isfinite(spacing) or spacing <= 0):
        raise ValueError("Finite 3D route and positive spacing required")
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        distance = float(np.linalg.norm(b-a))
        if distance <= 0:
            raise ValueError("Duplicate route point")
        result.extend(a + (b-a)*f for f in np.linspace(0., 1., max(1, math.ceil(distance/spacing))+1)[1:])
    return np.asarray(result)


def local_triangles(triangles, low, high):
    return triangles[np.all(triangles.max(axis=1) >= low, axis=1)
                     & np.all(triangles.min(axis=1) <= high, axis=1)]


def surface_height(triangles, xy, reference):
    """Select a tread near nominal floor; never jump to overhead floors."""
    start = np.r_[xy, reference + .8]
    end = np.r_[xy, reference - .8]
    hits = segment_hits(triangles, start, end)
    if not len(hits):
        raise ValueError("No stair/floor support near reference")
    return float(start[2] - 1.6*hits[0])


def build_route(triangles, *, reverse=False):
    floor = resample(WAYPOINTS)
    centers, supports = [], []
    for point in floor:
        nearby = local_triangles(triangles, point+[-.5, -.5, -.8], point+[.5, .5, .8])
        # Footprint envelope is kinematic clearance, not tracked-robot dynamics.
        heights = [surface_height(nearby, point[:2]+[x,y], point[2])
                   for x in (-.45, 0., .45) for y in (-.45, 0., .45)]
        height = max(heights)
        centers.append([*point[:2], height+TARGET_OFFSET])
        supports.append(height)
    centers = np.asarray(centers)
    if reverse:
        centers, supports = centers[::-1].copy(), supports[::-1]
    lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return {"centers_enu": centers.tolist(), "distance_m": np.r_[0., np.cumsum(lengths)].tolist(),
            "support_height_m": supports, "reverse": reverse,
            "carrier": "kinematic_stair_carrier_not_wheel_dynamics", "floor_reference_enu": WAYPOINTS.tolist()}


def sample(route, distance):
    distances = np.asarray(route["distance_m"])
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Finite nonnegative route distance required")
    points = np.asarray(route["centers_enu"])
    return np.array([np.interp(distance, distances, points[:,i]) for i in range(3)])


def audit_route(triangles, route):
    points = np.asarray(route["centers_enu"])
    collisions = []
    for index, (a,b) in enumerate(zip(points, points[1:])):
        low = np.minimum(a,b)+TARGET_LOW-.02
        high = np.maximum(a,b)+TARGET_HIGH+.02
        if triangles_touch_box(triangles, low, high):
            collisions.append({"segment": index, "a": a.tolist(), "b": b.tolist()})
    return {"passed": not collisions, "collisions": collisions,
            "sweep_segments": len(points)-1, "target_aabb_low": TARGET_LOW.tolist(),
            "target_aabb_high": TARGET_HIGH.tolist(), "margin_m": .02,
            "scope": "conservative_swept_target_box_against_pinned_building",
            "flight_admitted": False, "contact_dynamics_verified": False}


def ray_visible(triangles, eye, target):
    low, high = np.minimum(eye, target)-.001, np.maximum(eye, target)+.001
    return not len(segment_hits(local_triangles(triangles, low, high), eye, target))


def plot_route(triangles, route, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    points = np.asarray(route["centers_enu"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, z in zip(axes[:2], (2., 12.)):
        selected = triangles[(triangles[:,:,2].min(1)<z) & (triangles[:,:,2].max(1)>z)]
        ax.add_collection(LineCollection(selected[:,:,:2], colors="#9b9b9b", linewidths=.35))
        colored = points[points[:,2] < 6] if z==2 else points[points[:,2] >= 6]
        ax.plot(colored[:,0], colored[:,1], color="#bc2337", linewidth=2)
        ax.set(xlim=(-17, 1), ylim=(-20, 0), xlabel="ENU X (m)", ylabel="ENU Y (m)",
               title="Lower floor" if z==2 else "Upper floor")
        ax.set_aspect("equal")
    axes[2].plot(route["distance_m"], points[:,2], color="#23765b")
    axes[2].set(xlabel="Route arc length (m)", ylabel="Target center ENU Z (m)", title="Stair ascent profile")
    fig.suptitle("SubT original switchback stair route / geometry design, not a flight score")
    fig.tight_layout()
    fig.savefig(Path(output)/"route.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    triangles = load_triangles(args.package)
    route = build_route(triangles, reverse=args.reverse)
    audit = audit_route(triangles, route)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"route.private.json").write_text(json.dumps({"schema": SCHEMA, "route": route,
        "audit": audit, "flight_admitted": False, "model_evaluated": False}, indent=2), encoding="utf8")
    plot_route(triangles, route, args.output_dir)
    print(json.dumps({"passed": audit["passed"], "length_m": route["distance_m"][-1],
                      "collisions": audit["collisions"][:8], "output": str(args.output_dir)}))
    raise SystemExit(0 if audit["passed"] else 2)


if __name__ == "__main__":
    main()
