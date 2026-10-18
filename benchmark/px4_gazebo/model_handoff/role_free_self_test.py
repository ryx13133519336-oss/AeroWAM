"""Synthetic task/score symmetry checks, not learned role adaptation or flight."""

from dataclasses import replace
import json
import unittest

from .language_evaluator import LanguageEvaluator, ObjectAnnotation
from .language_tasks import ROLE_FREE_TASK_SCHEMA, TargetReport, paired_target_tasks, role_free_task, templated_task
from .language_self_test import BOX, OBJECTS, emit
from .search_episode import SearchEpisode


def evaluator(task="search"):
    return LanguageEvaluator(role_free_task(task, OBJECTS["private_red"]).bind(OBJECTS))


def track(ev, agent, start=0.):
    for offset in (0., .1, .2, .3, .4):
        emit(ev, agent, start + offset)


def transfer(ev, source, receiver):
    track(ev, source)
    emit(ev, source, .5, "handoff_request")
    emit(ev, receiver, .6, "handoff_accept")
    track(ev, receiver, .7)
    emit(ev, receiver, 1.2, "confirm")


class RoleFreeTests(unittest.TestCase):
    def test_public_contract_has_no_agent_allocation(self):
        for task in ("search", "handoff", "inspection"):
            definition = role_free_task(task, OBJECTS["private_red"])
            wire = definition.public_context()
            self.assertEqual(wire["schema"], ROLE_FREE_TASK_SCHEMA)
            self.assertEqual(wire["role_mode"], "unassigned")
            self.assertNotIn("roles", wire)
            self.assertNotIn("uav_0", json.dumps(wire))
            self.assertNotIn("uav_1", json.dumps(wire))
            self.assertNotIn("UAV0", definition.instruction)
            self.assertNotIn("UAV1", definition.instruction)
            self.assertIsNone(definition.source_agent)
            self.assertIsNone(definition.receiver_agent)

    def test_authored_instruction_is_unchanged(self):
        instruction = "找到蓝色方块，让另一架确认；谁先发现由你们决定。"
        task = role_free_task("search", OBJECTS["private_blue"], instruction=instruction)
        self.assertEqual(task.public_context()["instruction"], instruction)
        self.assertEqual(task.bind(OBJECTS).target_id, "private_blue")
        with self.assertRaises(ValueError):
            role_free_task("unknown", OBJECTS["private_red"])
        with self.assertRaises(ValueError):
            role_free_task("search", OBJECTS["private_red"], instruction="")

    def test_role_modes_cannot_be_mixed(self):
        task = role_free_task("search", OBJECTS["private_red"])
        for kwargs in ({"source_agent": "uav_0"}, {"receiver_agent": "uav_1"},
                       {"role_mode": "fixed"}, {"role_mode": "dynamic_by_oracle"}):
            with self.assertRaises(ValueError):
                replace(task, **kwargs)

    def test_target_pairs_keep_role_free_contract_and_change_answer(self):
        pair = paired_target_tasks("search", OBJECTS, role_mode="unassigned")
        self.assertEqual({bound.target_id for bound in pair}, set(OBJECTS))
        self.assertEqual(len({bound.definition.instruction for bound in pair}), 2)
        for bound in pair:
            self.assertEqual(bound.definition.public_context()["schema"], ROLE_FREE_TASK_SCHEMA)
        with self.assertRaises(ValueError):
            paired_target_tasks("search", OBJECTS, role_mode="unknown")

    def test_search_both_discovery_directions_score_identically(self):
        results = []
        for source, receiver in (("uav_0", "uav_1"), ("uav_1", "uav_0")):
            ev = evaluator()
            # The first observer does not acquire a permanent task role.
            emit(ev, receiver, 0.)
            self.assertIsNone(ev.source_agent)
            emit(ev, source, .1, "found")
            emit(ev, receiver, .2, "confirm")
            result = ev.finish()
            self.assertEqual(result["category"], "success")
            self.assertEqual(result["schema"], "aerowam-language-evaluation/2-draft")
            self.assertEqual(result.pop("evidence_roles"), {"source": source, "receiver": receiver})
            results.append(result)
        self.assertEqual(*results)

    def test_one_agent_cannot_self_confirm(self):
        for source in ("uav_0", "uav_1"):
            ev = evaluator()
            emit(ev, source, 0., "found")
            emit(ev, source, .1, "confirm")
            self.assertEqual(ev.finish()["reason"], "invalid_search_event_order")

    def test_confirm_before_discovery_rejected_for_either_agent(self):
        for agent in ("uav_0", "uav_1"):
            ev = evaluator()
            emit(ev, agent, 0., "confirm")
            self.assertEqual(ev.finish()["reason"], "invalid_search_event_order")

    def test_wrong_target_does_not_allocate_roles(self):
        ev = evaluator()
        emit(ev, "uav_1", 0., "found", box=(.6, .6, .8, .8))
        result = ev.finish()
        self.assertEqual(result["reason"], "wrong_instruction_target")
        self.assertEqual(result["evidence_roles"], {"source": None, "receiver": None})

    def test_search_stale_evidence_still_rejected(self):
        for source, receiver in (("uav_0", "uav_1"), ("uav_1", "uav_0")):
            ev = evaluator()
            emit(ev, source, 1., "found", now=1.1)
            emit(ev, receiver, 1.05, "confirm", now=1.1)
            self.assertEqual(ev.finish()["category"], "model_failure")

    def test_handoff_both_directions_score_identically(self):
        results = []
        for source, receiver in (("uav_0", "uav_1"), ("uav_1", "uav_0")):
            ev = evaluator("handoff")
            # Both may observe/track before either requests a transfer.
            emit(ev, receiver, 0.)
            transfer(ev, source, receiver)
            result = ev.finish()
            self.assertEqual(result["category"], "success")
            self.assertEqual(result.pop("evidence_roles"), {"source": source, "receiver": receiver})
            results.append(result)
        self.assertEqual(*results)

    def test_handoff_request_requires_tracking_in_both_directions(self):
        for agent in ("uav_0", "uav_1"):
            ev = evaluator("handoff")
            emit(ev, agent, 0., "handoff_request")
            result = ev.finish()
            self.assertEqual(result["reason"], "request_before_source_tracking")
            self.assertIsNone(result["evidence_roles"]["source"])

    def test_handoff_accept_requires_request(self):
        for agent in ("uav_0", "uav_1"):
            ev = evaluator("handoff")
            emit(ev, agent, 0., "handoff_accept")
            self.assertEqual(ev.finish()["reason"], "accept_without_request")

    def test_handoff_requester_cannot_accept_own_transfer(self):
        for source in ("uav_0", "uav_1"):
            ev = evaluator("handoff")
            track(ev, source)
            emit(ev, source, .5, "handoff_request")
            emit(ev, source, .6, "handoff_accept")
            self.assertEqual(ev.finish()["reason"], "accept_without_request")

    def test_handoff_receiver_requires_post_accept_tracking(self):
        for source, receiver in (("uav_0", "uav_1"), ("uav_1", "uav_0")):
            ev = evaluator("handoff")
            track(ev, receiver)
            track(ev, source, .4)
            emit(ev, source, .9, "handoff_request")
            emit(ev, receiver, 1., "handoff_accept")
            emit(ev, receiver, 1.1, "confirm")
            self.assertEqual(ev.finish()["reason"], "confirm_before_receiver_tracking")

    def test_occlusion_breaks_tracking_for_either_requester(self):
        for source in ("uav_0", "uav_1"):
            ev = evaluator("handoff")
            track(ev, source)
            from .language_evaluator import ObservedFrame
            ev.register_observed_frame(ObservedFrame(source, 450_000_000, 450_000_000,
                {"private_red": ObjectAnnotation(None), "private_blue": ObjectAnnotation(BOX)}))
            emit(ev, source, .5, "handoff_request")
            self.assertEqual(ev.finish()["reason"], "request_before_source_tracking")

    def test_inspection_does_not_assign_views_by_identity(self):
        results = []
        for front, side in (("uav_0", "uav_1"), ("uav_1", "uav_0")):
            ev = evaluator("inspection")
            for t in (0., .1, .2, .3, .4, .5):
                emit(ev, front, t, "inspect", view="front")
                emit(ev, side, t, "inspect", view="side")
            emit(ev, front, .6, "confirm", view="front")
            emit(ev, side, .6, "confirm", view="side")
            results.append(ev.finish())
        self.assertEqual(results[0]["category"], "success")
        self.assertEqual(*results)

    def test_inspection_still_requires_two_contributors(self):
        for agent in ("uav_0", "uav_1"):
            ev = evaluator("inspection")
            for start, view in ((0., "front"), (.6, "side")):
                for offset in (0., .1, .2, .3, .4, .5):
                    emit(ev, agent, start + offset, "inspect", view=view)
            emit(ev, agent, 1.2, "confirm", view="side")
            self.assertEqual(ev.finish()["reason"], "confirm_before_dual_view_inspection")

    def test_grounded_episode_coordinator_preserves_either_discoverer(self):
        from .grounded_interface import GroundedAction
        from .protocol import AGENT_IDS, ModelJointAction
        from .search_pilot_self_test import observation
        from .language_evaluator import ObservedFrame
        for source, receiver in (("uav_0", "uav_1"), ("uav_1", "uav_0")):
            bound = role_free_task("search", OBJECTS["private_red"]).bind(OBJECTS)
            episode = SearchEpisode(bound, "opaque")
            for stamp, agent, event in ((1_000_000_000, source, "found"), (1_100_000_000, receiver, "confirm")):
                obs = observation(bound.definition.instruction, stamp)
                frames = {a: ObservedFrame(a, stamp, stamp,
                    {"private_red": ObjectAnnotation(BOX),
                     "private_blue": ObjectAnnotation((.6, .6, .8, .8))}) for a in AGENT_IDS}
                response = GroundedAction(ModelJointAction(), (TargetReport(agent, stamp, BOX, .9, event),))
                self.assertEqual(episode.delivered(obs, frames, response, now_sim_time_ns=stamp), response.action)
            result = episode.finish()
            self.assertEqual(result["category"], "success")
            self.assertEqual(result["evidence_roles"], {"source": source, "receiver": receiver})

    def test_late_safety_still_overrides_success(self):
        ev = evaluator()
        emit(ev, "uav_1", 0., "found")
        emit(ev, "uav_0", .1, "confirm")
        ev.safety_violation("collision")
        self.assertEqual(ev.finish()["reason"], "collision")

    def test_legacy_fixed_roles_and_wire_format_unchanged(self):
        task = templated_task("search", OBJECTS["private_red"])
        self.assertEqual(task.public_context()["schema"], "aerowam-language-task/1-draft")
        self.assertEqual(task.public_context()["roles"], {"source": "uav_0", "receiver": "uav_1"})
        ev = LanguageEvaluator(task.bind(OBJECTS))
        emit(ev, "uav_1", 0., "found")
        result = ev.finish()
        self.assertEqual(result["reason"], "invalid_search_event_order")
        self.assertNotIn("evidence_roles", result)

    def test_legacy_motion_gate_cannot_silently_score_new_task(self):
        task = role_free_task("search", OBJECTS["private_red"]).bind(OBJECTS)
        SearchEpisode(task, "opaque")  # Grounded evidence coordinator accepts /2.
        with self.assertRaisesRegex(ValueError, "fixed-role motion gate"):
            SearchEpisode(task, "opaque", motion_gate=object())


if __name__ == "__main__":
    unittest.main()
