"""Public-RGB color reporter for the two-object pilot ONLY.

This explicit auxiliary component is shared across methods. It checks color,
not shape/stripe or general language understanding; it never imports private
scene identities, labels or evaluator results. Not a learned grounding head.
"""

from io import BytesIO
import re

from .grounded_interface import GroundedAction, GroundedAdapter
from .language_tasks import TargetReport
from .protocol import AGENT_IDS, ModelJointAction, WaypointDelta

REPORTER_VERSION = "pilot_rgb_color_reporter/1"


class SearchTravelBudgetExceeded(RuntimeError):
    """Public movement guard, distinct from failed neural inference."""
    code = "search_local_travel_budget_exceeded"

    def __init__(self, distances):
        super().__init__("RGB search local travel budget exceeded")
        self.diagnostics = {"limit_m": 2.5, "displacement_m": distances,
                            "exceeded_agents": [a for a in AGENT_IDS if distances[a] > 2.5]}


def check_search_travel(observation, initial):
    import math
    distances = {a: math.dist(observation.states[a].position[:2], initial[a][:2]) for a in AGENT_IDS}
    if any(distance > 2.5 for distance in distances.values()):
        raise SearchTravelBudgetExceeded(distances)


def requested_color(instruction):
    # Original text still reaches the learned action model unchanged. This
    # deliberately limited auxiliary parser cannot resolve negation, multiple
    # colors, pronouns, relational language or a new task type.
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Nonempty manual instruction required")
    colors = [color for color, chinese in (("red", "红"), ("blue", "蓝"))
              if chinese in instruction or re.search(r"\b" + color + r"\b", instruction, re.IGNORECASE)]
    if len(colors) != 1:
        raise ValueError("Pilot color reporter needs exactly one explicit red/blue color (红/蓝); broader language is not supported")
    if re.search(r"\b(not|except|avoid|ignore|without)\b", instruction, re.IGNORECASE) or any(word in instruction for word in ("不要", "不是", "别找", "除了", "忽略", "避开")):
        raise ValueError("Pilot color reporter does not support negated target instructions")
    return colors[0]


def color_box(jpeg, color):
    import numpy as np
    from PIL import Image
    with Image.open(BytesIO(jpeg)) as image:
        if image.format != "JPEG" or image.size != (256, 256) or image.mode != "RGB":
            raise ValueError("Pilot expects actual 256x256 RGB JPEGs")
        pixels = np.asarray(image).astype(np.int16)
    channel, other = (0, (1, 2)) if color == "red" else (2, (0, 1)) if color == "blue" else (None, ())
    if channel is None:
        raise ValueError("Unsupported pilot color")
    main = pixels[:, :, channel]
    mask = (main > 60)
    for index in other:
        mask &= (main > 1.8 * pixels[:, :, index] + 15)
    ys, xs = np.nonzero(mask)
    # One object per color is a declared pilot restriction. Reject sparse or
    # widely scattered support rather than merging sky/reflection artifacts.
    if len(xs) < 20:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if min(x2 - x1, y2 - y1) < 3 or len(xs) / ((x2 - x1) * (y2 - y1)) < .25:
        return None
    if x1 == 0 or y1 == 0 or x2 == 256 or y2 == 256:
        return None
    return x1 / 256, y1 / 256, x2 / 256, y2 / 256


def reportable_box(box):
    """Public color support must be centered and clear of uncertain edges.

    Non-color material (e.g. white stripes) can extend past a color box.
    This conservative RGB gate is not independent visibility verification.
    """
    if box is None:
        return False
    x1, y1, x2, y2 = box
    clearance = max(8 / 256, .25 * max(x2 - x1, y2 - y1))
    return (abs((x1 + x2) / 2 - .5) <= .08
            and min(x1, y1, 1 - x2, 1 - y2) >= clearance)


class PilotReporter:
    def __init__(self):
        self.episode = None
        self.instruction = None
        self.found_stamp = None
        self.confirmed = False

    def reports(self, observation):
        color = requested_color(observation.instruction)
        if self.episode != observation.episode_id:
            self.episode, self.instruction = observation.episode_id, observation.instruction
            self.found_stamp, self.confirmed = None, False
        if self.instruction != observation.instruction:
            raise ValueError("Instruction changed within one episode")
        # This is the reporter's own emitted-event state, never evaluator feedback.
        found_before = self.found_stamp
        result = []
        for agent in AGENT_IDS:
            box = color_box(observation.rgb_jpeg[agent], color)
            if box is None:
                continue
            stamp = observation.image_timestamp_ns[agent]
            event = "observe"
            if agent == "uav_0" and self.found_stamp is None:
                event, self.found_stamp = "found", stamp
            if agent == "uav_1" and found_before is not None and stamp > found_before and not self.confirmed:
                event, self.confirmed = "confirm", True
            result.append(TargetReport(agent, stamp, box, 1., event))
        return tuple(result)


class GroundedCheckpointPolicy:
    """Legacy learned actions + explicitly declared shared pilot reporter.

    No navigation override or expert route. The old checkpoint remains an OOD
    pilot and has not been trained for search. This composite is not pure BC.
    """
    def __init__(self, action_policy, *, reporter=None):
        self.action_policy = action_policy
        self.reporter = PilotReporter() if reporter is None else reporter

    def __call__(self, observation):
        # Refuse unsupported language before mutating the learned history.
        requested_color(observation.instruction)
        action = ModelJointAction.from_wire(self.action_policy(observation.to_wire()))
        return GroundedAction(action, self.reporter.reports(observation))


class RGBScanPolicy:
    """Wiring positive control: rotate when target absent, hold when visible.

    No checkpoint, expert waypoint, scene geometry or private target binding.
    This is not one of the three learned benchmark methods.
    """
    def __init__(self):
        self.reporter = PilotReporter()

    def __call__(self, observation):
        reports = self.reporter.reports(observation)
        visible = {report.agent_id for report in reports}
        deltas = {agent: WaypointDelta(dyaw=0. if agent in visible else -.12) for agent in AGENT_IDS}
        return GroundedAction(ModelJointAction(**deltas), reports)


def create_rgb_scan_policy():
    return GroundedAdapter(RGBScanPolicy())
