"""Procedural matte industrial targets; public RGB, private IDs/annotations.

Neutral hardware and replaceable colored inserts are separate static models.
Only the insert is a target. No AprilTag, text, emissive material or stripe.
No existing lawn / warehouse asset or language-task schema is changed.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = "aerowam-industrial-targets/1"
COLORS = {"red": (.52, .065, .045), "blue": (.045, .11, .50),
          "green": (.055, .38, .09), "yellow": (.65, .48, .035)}
SHAPES = ("sphere", "cube", "cylinder")
IDENTITIES = ("industrial_target_a", "industrial_target_b")
LABELS = {0: None, 100: None, 11: IDENTITIES[0], 22: IDENTITIES[1]}
POSITIONS = {IDENTITIES[0]: (9., -8., 2.), IDENTITIES[1]: (9., -6., 2.)}
# Whole trolley bounds in its floor-local frame, not just the colored insert.
SUPPORT_BOUNDS = ((-.5, -.73, 0.), (.5, .73, 2.7))


@dataclass(frozen=True)
class Component:
    color: str
    shape: str

    def __post_init__(self):
        if self.color not in COLORS or self.shape not in SHAPES:
            raise ValueError("Component requires red/blue/green/yellow and sphere/cube/cylinder")


DEFAULT = (Component("red", "sphere"), Component("blue", "cube"))


def parse_component(text):
    if not isinstance(text, str) or text.count(":") != 1:
        raise ValueError("Use COLOR:SHAPE, for example green:cylinder")
    return Component(*text.split(":"))


def components(values=None):
    values = DEFAULT if values is None else values
    if (not isinstance(values, (tuple, list)) or len(values) != 2
            or any(not isinstance(v, Component) for v in values)):
        raise ValueError("Exactly two Component objects required")
    return tuple(values)


def configuration(values=None):
    values = components(values)
    return {"schema": SCHEMA, "slots": {identity: asdict(value) for identity, value in zip(IDENTITIES, values)},
            "component_diameter_m": .8, "component_height_m": .8,
            "roughness": 1., "metalness": 0., "stripe": None,
            "bbox_scope": "replaceable_component_only", "mount": "wheeled_open_frame",
            "motion_mode": "bounded_kinematic_trolley", "role_assignment": None}


def fingerprint(values=None):
    return hashlib.sha256(json.dumps(configuration(values), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def from_configuration(config):
    if not isinstance(config, dict) or set(config.get("slots", {})) != set(IDENTITIES):
        raise ValueError("Unexpected industrial target slots")
    values = tuple(Component(**config["slots"][identity]) for identity in IDENTITIES)
    if config != configuration(values):
        raise ValueError("Unexpected industrial target configuration")
    return values


def private_selection(values, color, shape):
    """Explicit judge selection, not a language parser or a policy input."""
    expected = Component(color, shape)
    matches = [identity for identity, value in zip(IDENTITIES, components(values)) if value == expected]
    if len(matches) != 1:
        raise ValueError("Expected descriptor is absent or ambiguous")
    return matches[0]


def public_request(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Nonempty human instruction required")
    return {"schema": "aerowam-industrial-request/1", "instruction": instruction, "role_mode": "unassigned"}


def material(parent, color):
    result = ET.SubElement(parent, "material")
    rgba = " ".join(map(str, (*color, 1)))
    for name in ("ambient", "diffuse"):
        ET.SubElement(result, name).text = rgba
    ET.SubElement(result, "specular").text = "0 0 0 1"
    ET.SubElement(result, "emissive").text = "0 0 0 1"
    metal = ET.SubElement(ET.SubElement(result, "pbr"), "metal")
    ET.SubElement(metal, "roughness").text = "1"
    ET.SubElement(metal, "metalness").text = "0"


def geometry(parent, shape, size):
    node = ET.SubElement(parent, "geometry")
    if shape == "cube":
        ET.SubElement(ET.SubElement(node, "box"), "size").text = " ".join(map(str, size))
    elif shape == "sphere":
        ET.SubElement(ET.SubElement(node, "sphere"), "radius").text = str(size[0] / 2)
    elif shape == "cylinder":
        cylinder = ET.SubElement(node, "cylinder")
        ET.SubElement(cylinder, "radius").text = str(size[0] / 2)
        ET.SubElement(cylinder, "length").text = str(size[2])
    else:
        raise ValueError("Unsupported component geometry")


def part(link, name, shape, size, position, color, rotation=(0, 0, 0)):
    for kind in ("visual", "collision"):
        element = ET.SubElement(link, kind, name=name)
        ET.SubElement(element, "pose").text = " ".join(map(str, (*position, *rotation)))
        geometry(element, shape, size)
        if kind == "visual":
            material(element, color)


def static_model(world, name, xyz, label):
    model = ET.SubElement(world, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = " ".join(map(str, (*xyz, 0, 0, 0)))
    plugin = ET.SubElement(model, "plugin", filename="gz-sim-label-system", name="gz::sim::systems::Label")
    ET.SubElement(plugin, "label").text = str(label)
    return ET.SubElement(model, "link", name="object")


def instrument_contacts(world):
    bindings = []
    for model in world.findall("model"):
        kind = "component" if model.get("name") in IDENTITIES else "support" if model.get("name", "").startswith("industrial_mount_") else "structure"
        for link in model.findall("link"):
            collisions = link.findall("collision")
            if not collisions:
                continue
            sensor = ET.SubElement(link, "sensor", name="private_contact", type="contact")
            ET.SubElement(sensor, "always_on").text = "true"
            ET.SubElement(sensor, "update_rate").text = "30"
            topic = f"/world/{world.get('name')}/model/{model.get('name')}/link/{link.get('name')}/sensor/private_contact/contact"
            ET.SubElement(sensor, "topic").text = topic
            contact = ET.SubElement(sensor, "contact")
            for collision in collisions:
                ET.SubElement(contact, "collision").text = collision.get("name")
            bindings.append({"kind": kind, "model": model.get("name"), "link": link.get("name"),
                             "collisions": [c.get("name") for c in collisions], "topic": topic})
    return bindings


def install(world, values=None):
    values = components(values)
    for name in ("probe_red", "probe_blue"):
        previous = world.find(f"model[@name='{name}']")
        if previous is not None:
            world.remove(previous)
    if any(world.find(f"model[@name='{name}']") is not None for name in IDENTITIES):
        raise ValueError("Industrial targets already installed")
    for index, (identity, component) in enumerate(zip(IDENTITIES, values)):
        xyz = POSITIONS[identity]
        support = static_model(world, f"industrial_mount_{'ab'[index]}", (xyz[0], xyz[1], 0.), 100)
        # Identical powder-coated hardware for every color/shape combination.
        gray = (.15, .17, .18)
        for name, size, position in (
            ("base", (1., 1., .12), (0., 0., .20)),
            ("mast", (.12, .12, 1.04), (0., 0., .78)),
            ("frame_left", (.10, .10, 1.4), (0., -.68, 2.)),
            ("frame_right", (.10, .10, 1.4), (0., .68, 2.)),
            ("frame_lower", (.10, 1.4, .10), (0., 0., 1.35)),
            ("frame_upper", (.10, 1.4, .10), (0., 0., 2.65)),
            ("component_pin", (.08, .08, .25), (0., 0., 1.5)),
        ):
            part(support, name, "cube", size, position, gray)
        for wheel, (x, y) in enumerate(((-.35, -.42), (-.35, .42), (.35, -.42), (.35, .42))):
            part(support, f"wheel_{wheel}", "cylinder", (.20, .10, .10), (x, y, .10),
                 (.035, .035, .035), (1.5707963267948966, 0., 0.))
        body = static_model(world, identity, xyz, 11 + 11 * index)
        part(body, "component", component.shape, (.8, .8, .8), (0., 0., 0.), COLORS[component.color])
    return configuration(values)


def visible_components(labels):
    """Private mask-only boxes, same 16px/3px/.15/unclipped rules as legacy QA."""
    if (not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.dtype.kind not in "ui"
            or not set(map(int, np.unique(labels))) <= set(LABELS)):
        raise ValueError("Unknown industrial instance raster labels")
    height, width = labels.shape
    result = {}
    for label, identity in LABELS.items():
        if identity is None:
            continue
        ys, xs = np.nonzero(labels == label)
        box = None
        if len(xs):
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            if (len(xs) >= 16 and x2 - x1 >= 3 and y2 - y1 >= 3 and len(xs) / ((x2 - x1) * (y2 - y1)) >= .15
                    and x1 > 0 and y1 > 0 and x2 < width and y2 < height):
                box = (x1 / width, y1 / height, x2 / width, y2 / height)
        result[identity] = {"pixels": len(xs), "bbox": box}
    return result


def color_support(rgb, color):
    """QA within private masks only. Not a public detector or target oracle."""
    rgb = rgb.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "yellow":
        return (r > b + 25) & (g > b + 25)
    channel = {"red": r, "green": g, "blue": b}[color]
    other = {"red": (g, b), "green": (r, b), "blue": (r, g)}[color]
    return (channel > other[0] + 20) & (channel > other[1] + 20)
