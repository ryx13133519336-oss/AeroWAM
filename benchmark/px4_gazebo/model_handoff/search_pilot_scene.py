"""One private paired search scene. A pilot, not a compositional split."""

from copy import deepcopy
import hashlib
import random
import xml.etree.ElementTree as ET

from .annotation_session_smoke import build_world
from .fixtures import Appearance
from .language_tasks import LanguageTask, paired_target_tasks

WORLD = "aerowam_search_pilot"
OBJECTS = {"probe_red": Appearance("red", "sphere", "black"),
           "probe_blue": Appearance("blue", "cube", "white")}
LABELS = {0: None, 11: "probe_red", 22: "probe_blue"}
SPAWNS_NED = ((0., 0., 0.), (-3., 0., 0.))  # SDF ENU (0,0), (0,-3).


def manual_condition(instruction, expected_color):
    """Human-authored text + separate private scoring annotation.

    Do not infer the evaluator answer using the policy's color parser.
    The human must check that this text describes the selected target.
    """
    if expected_color not in {"red", "blue"}:
        raise ValueError("Choose the private expected red/blue object")
    descriptor = next(value for value in OBJECTS.values() if value.color == expected_color)
    return LanguageTask("search", instruction, descriptor).bind(OBJECTS)


def paired_conditions(seed=0):
    """Randomized instruction order, private answers, one scene-level group.

    Identical scene and initial conditions must be recreated per condition.
    Each method gets this same order; opaque run IDs must not encode answers.
    """
    conditions = list(paired_target_tasks("search", OBJECTS))
    random.Random(seed).shuffle(conditions)
    return tuple(conditions)


def search_world(px4_root, models):
    tree = build_world(px4_root, models)
    scene = tree.getroot().find("world")
    scene.set("name", WORLD)
    scene.remove(scene.find("model[@name='probe_wall']"))
    for index, y in enumerate((0., -3.)):
        # Require a view change: targets begin outside UAV0's forward view.
        scene.find(f"include[name='annotation_uav_{index}']/pose").text = f"0 {y} 0.1 0 0 1.6"
    for identity, y in (("probe_red", 1.), ("probe_blue", -1.)):
        scene.find(f"model[@name='{identity}']/pose").text = f"4 {y} 2.4 0 0 0"
    # The older diagnostic cube has no stripe; do not author a false instruction.
    blue = scene.find("model[@name='probe_blue']/link")
    stripe = deepcopy(scene.find("model[@name='probe_red']/link/visual[@name='stripe']"))
    stripe.find("geometry/box/size").text = "0.82 0.16 0.82"
    stripe.find("material/ambient").text = "0.95 0.95 0.95 1"
    stripe.find("material/diffuse").text = "0.95 0.95 0.95 1"
    blue.append(stripe)
    # Two target collisions for independent safety checks in this pilot.
    for identity in OBJECTS:
        link = scene.find(f"model[@name='{identity}']/link")
        collision = ET.SubElement(link, "collision", name="body_collision")
        collision.append(deepcopy(link.find("visual[@name='body']/geometry")))
        sensor = ET.SubElement(link, "sensor", name="private_contact", type="contact")
        ET.SubElement(sensor, "always_on").text = "true"
        ET.SubElement(sensor, "update_rate").text = "30"
        ET.SubElement(sensor, "topic").text = f"/aerowam/search_pilot/private/contacts/{identity}"
        ET.SubElement(ET.SubElement(sensor, "contact"), "collision").text = "body_collision"
    return tree


def scene_digest(tree):
    return hashlib.sha256(ET.tostring(tree.getroot(), encoding="utf-8")).hexdigest()
