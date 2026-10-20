"""Private offline ray checks for the pinned 2-story COLLADA, not navigation.

Only the audited Z-up, matrix-node, indexed-triangle subset is accepted.
This supplements rendered occlusion; it is not a UAV volume/contact proof.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def load_triangles(package):
    from .subt_urban_scene import verify
    verify(package)
    root = ET.parse(Path(package) / "urban_2_story/meshes/2story.dae").getroot()
    return collada_triangles(root)


def collada_triangles(root):
    if root.findtext("{*}asset/{*}up_axis") != "Z_UP":
        raise ValueError("Only audited Z-up COLLADA supported")
    unit = float(root.find("{*}asset/{*}unit").get("meter"))
    meshes = {}
    for geometry in root.findall("{*}library_geometries/{*}geometry"):
        mesh = geometry.find("{*}mesh")
        if mesh is None or mesh.findall("{*}polylist") or mesh.findall("{*}polygons"):
            raise ValueError("Only triangle meshes supported")
        sources = {}
        for source in mesh.findall("{*}source"):
            accessor = source.find("{*}technique_common/{*}accessor")
            stride = int(accessor.get("stride", "1"))
            sources[source.get("id")] = np.fromstring(source.findtext("{*}float_array"), sep=" ").reshape(-1, stride)
        vertices = {v.get("id"): v.find("{*}input[@semantic='POSITION']").get("source")[1:]
                    for v in mesh.findall("{*}vertices")}
        triangles = []
        for primitive in mesh.findall("{*}triangles"):
            inputs = primitive.findall("{*}input")
            width = max(int(i.get("offset", "0")) for i in inputs) + 1
            position = next(i for i in inputs if i.get("semantic") == "VERTEX")
            indices = np.fromstring(primitive.findtext("{*}p"), sep=" ", dtype=np.int64).reshape(-1, width)
            xyz = sources[vertices[position.get("source")[1:]]][:, :3]
            triangles.append(xyz[indices[:, int(position.get("offset", "0"))]].reshape(-1, 3, 3))
        if not triangles:
            raise ValueError("Empty geometry")
        meshes[geometry.get("id")] = np.concatenate(triangles)
    result = []

    def visit(node, parent):
        transform = parent.copy()
        for child in node:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"translate", "rotate", "scale", "lookat", "skew", "instance_node"}:
                raise ValueError("Unaudited COLLADA node transform")
            if tag == "matrix":
                transform = transform @ np.fromstring(child.text, sep=" ").reshape(4, 4)
        for instance in node.findall("{*}instance_geometry"):
            triangle = meshes[instance.get("url")[1:]]
            result.append((triangle @ transform[:3, :3].T + transform[:3, 3]) * unit)
        for child in node.findall("{*}node"):
            visit(child, transform)

    scene_id = root.find("{*}scene/{*}instance_visual_scene").get("url")[1:]
    scene = root.find(f"{{*}}library_visual_scenes/{{*}}visual_scene[@id='{scene_id}']")
    for node in scene.findall("{*}node"):
        visit(node, np.eye(4))
    triangles = np.concatenate(result)
    if not np.isfinite(triangles).all() or unit <= 0:
        raise ValueError("Invalid geometry")
    return triangles


def segment_hits(triangles, start, end):
    """Two-sided Moller-Trumbore intersection fractions, excluding endpoints."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    usable = np.abs(determinant) > 1e-10
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=usable)
    offset = start - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", offset, h)
    q = np.cross(offset, edge1)
    v = inverse * (q @ direction)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    mask = usable & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (t > 1e-6) & (t < 1 - 1e-6)
    return np.sort(t[mask])
