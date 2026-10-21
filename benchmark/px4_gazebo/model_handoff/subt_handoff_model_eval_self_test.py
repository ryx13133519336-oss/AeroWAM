"""Negative and causal tests; synthetic fixtures are never benchmark evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from .core._common import canonical
from .subt_handoff_evaluator import SubTHandoffEvaluator
from .subt_handoff_model_eval import run
from .subt_handoff_model_audit import audit
from .subt_handoff_model_io import hold, validate_output
from .subt_handoff_proxy import propose_cameras, target_poses, full_occlusion
import numpy as np

PREFIX = "benchmark.px4_gazebo.model_handoff."
AGENTS = ("uav_0","uav_1")
BOX = [.2,.2,.6,.6]


def fixture(root):
    episode = root/"episode"
    (episode/"public/images").mkdir(parents=True)
    rows = []
    for index in range(3):
        rgb = {}
        for agent in AGENTS:
            name = f"images/{index}_{agent}.jpg"
            Image.new("RGB",(8,8),(index*30,20,40)).save(episode/"public"/name)
            rgb[agent] = name
        rows.append({"sample_index":index,"stamp_ns":index*500_000_000,"dt_s":None if index==0 else .5,
            "rgb":rgb,"proprioception":{a:{"position_local_m":[0.,0.,0.],"quaternion_xyzw":[0.,0.,0.,1.]} for a in AGENTS},
            "messages":[] if index==0 else [{"agent_id":"uav_1","event":"found"}]})
    (episode/"public/observations.jsonl").write_bytes(b"".join(canonical(row)+b"\n" for row in rows))
    (episode/"public/metadata.json").write_bytes(canonical({"instruction":"Track the red sphere and hand off."}))
    frozen = {"passed":True,"message_delay_s":.5,"condition":{"swapped":True},"split":"held_out_route_test","files_sha256":{"synthetic":"a"*64}}
    (episode/"inspection.json").write_bytes(canonical(frozen))
    manifest = root/"manifest.json"
    manifest.write_bytes(canonical({"method_id":"hold","method_kind":"reference_rule"}))
    return episode, manifest, frozen


class ProtocolTests(unittest.TestCase):
    def test_replay_and_separate_audit_then_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            episode, manifest, frozen=fixture(root)
            with patch(PREFIX+"subt_handoff_model_eval.inspect",return_value=frozen):
                result=run(episode,root/"run",method_id="hold",method_kind="reference_rule",manifest=manifest,policy=hold)
                checked=audit(root/"run")
                self.assertTrue(checked["passed"])
                self.assertIsNone(result["success"])
                self.assertFalse(result["formal_score"])
                trace=[json.loads(row) for row in (root/"run/io_trace.jsonl").read_text().splitlines()]
                self.assertEqual(trace[1]["request"]["messages"],[{"agent_id":"uav_1","event":"found"}])
                self.assertEqual(len(trace[2]["request"]["action_history"]),2)
                trace[0]["request"]["observations"]["uav_0"]["proprioception"]["position_local_m"][0]=100.
                (root/"run/io_trace.jsonl").write_bytes(b"".join(canonical(r)+b"\n" for r in trace))
                with self.assertRaisesRegex(ValueError,"fingerprint"):
                    audit(root/"run")

    def test_failure_is_recorded_and_audited_without_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            episode,manifest,frozen=fixture(root)
            def bad(request):
                response=hold(request)
                response["episode_id"]="wrong"
                return response
            with patch(PREFIX+"subt_handoff_model_eval.inspect",return_value=frozen):
                result=run(episode,root/"run",method_id="hold",method_kind="reference_rule",manifest=manifest,policy=bad)
                self.assertEqual(result["status"],"model_failure")
                self.assertEqual(audit(root/"run")["requests_reconstructed"],1)

    def test_stale_private_and_excessive_output_rejected(self):
        request={"episode_id":"x","timestamp_ns":1,"observations":{a:{"image_timestamp_ns":1} for a in AGENTS}}
        for case in ("excessive","private","stale","bad_event"):
            response=hold(request)
            if case=="excessive": response["actions"]["uav_1"]["delta_enu_m"][0]=2.
            if case=="private": response["actions"]["uav_1"]["truth"]=True
            if case=="stale": response["timestamp_ns"]=0
            if case=="bad_event": response["events"]=[{"agent_id":"uav_0","event":"handoff.confirm","image_timestamp_ns":0,"bbox_xyxy":BOX}]
            with self.subTest(case=case),self.assertRaises(ValueError):validate_output(response,request)


class EvaluatorTests(unittest.TestCase):
    def exercise(self, source, *, physical=True, skip_accept=False, gap=False):
        receiver=next(a for a in AGENTS if a!=source)
        evaluator=SubTHandoffEvaluator()
        def step(t,actor,event,hidden=False):
            views={a:{"target_bbox_xyxy":None if a==source and hidden else BOX,
                      "fully_occluded":bool(a==source and hidden and physical)} for a in AGENTS}
            evaluator.step(round(t*1e9),views,[{"agent_id":actor,"event":"handoff."+event,
                "image_timestamp_ns":round(t*1e9),"bbox_xyxy":None if hidden and actor==source else BOX}])
        step(0,source,"found")
        step(.5,source,"handoff_request")
        step(1,receiver,"confirm" if skip_accept else "handoff_accept")
        step(1.5,receiver,"found",True)
        step(2,receiver,"found",True)
        step(4 if gap else 2.5,receiver,"confirm",True)
        return evaluator

    def test_both_identity_assignments(self):
        for source in AGENTS:
            result=self.exercise(source).result()
            self.assertTrue(result["success"])
            self.assertEqual(result["source_agent"],source)

    def test_zero_pixels_missing_accept_or_tracking_gap_cannot_pass(self):
        for options in ({"physical":False},{"skip_accept":True},{"gap":True}):
            self.assertFalse(self.exercise("uav_1",**options).complete)

    def test_ambiguous_simultaneous_found_does_not_assign_by_number(self):
        evaluator=SubTHandoffEvaluator()
        views={a:{"target_bbox_xyxy":BOX,"fully_occluded":False} for a in AGENTS}
        evaluator.step(0,views,[{"agent_id":a,"event":"handoff.found","image_timestamp_ns":0,"bbox_xyxy":BOX} for a in AGENTS])
        self.assertIsNone(evaluator.source)


class ProxyTests(unittest.TestCase):
    def cameras(self):
        return {"uav_0":[0.,0.,2.,0.,0.,0.],"uav_1":[3.,0.,2.,0.,0.,0.]}

    def action(self):
        return hold({"episode_id":"x","timestamp_ns":0})

    def test_camera_moves_only_from_per_agent_model_action(self):
        action=self.action()
        action["actions"]["uav_1"]["delta_enu_m"]=[.2,0.,.1]
        result=propose_cameras(self.cameras(),action,np.empty((0,3,3)),[10.,10.,2.])
        self.assertEqual(result["uav_0"],self.cameras()["uav_0"])
        self.assertEqual(result["uav_1"][:3],[3.2,0.,2.1])

    def test_zero_model_action_does_not_follow_target(self):
        route={"centers_enu":[[10.,10.,1.],[11.,10.,2.]],"distance_m":[0.,2.]}
        first=target_poses(route,0.,.3,self.cameras())
        later=target_poses(route,8.,.3,self.cameras())
        self.assertNotEqual(first["handoff_target"],later["handoff_target"])
        self.assertEqual(first["annotation_camera_0"],later["annotation_camera_0"])
        self.assertEqual(first["annotation_camera_1"],later["annotation_camera_1"])

    def test_swept_collision_between_endpoints_rejected(self):
        wall=np.array([[[.5,-1.,1.],[.5,1.,1.],[.5,0.,3.]]])
        action=self.action()
        action["actions"]["uav_0"]["delta_enu_m"]=[1.,0.,0.]
        with self.assertRaisesRegex(RuntimeError,"mesh_sweep"):
            propose_cameras(self.cameras(),action,wall,[10.,10.,2.])

    def test_nonvisible_without_wall_is_not_physical_occlusion(self):
        self.assertFalse(full_occlusion(np.empty((0,3,3)),[0.,0.,2.],[5.,0.,2.],0))

    def test_common_opaque_triangle_certifies_whole_target(self):
        triangle=np.array([[[2.,-4.,-3.],[2.,4.,-3.],[2.,0.,8.]]])
        self.assertTrue(full_occlusion(triangle,[0.,0.,2.],[5.,0.,2.],0))
        self.assertFalse(full_occlusion(triangle,[0.,0.,2.],[5.,0.,2.],1))

    def test_camera_separation_and_target_proximity_stop(self):
        action=self.action()
        cameras=self.cameras()
        cameras["uav_1"][0]=.5
        with self.assertRaisesRegex(RuntimeError,"inter_camera"):
            propose_cameras(cameras,action,np.empty((0,3,3)),[10.,10.,2.])
        with self.assertRaisesRegex(RuntimeError,"target_separation"):
            propose_cameras(self.cameras(),action,np.empty((0,3,3)),[.1,0.,2.])


class SharedEventTests(unittest.TestCase):
    def test_self_peer_features_swap_all_agent_blocks(self):
        from .subt_handoff_learn import agent_features
        values=np.arange(419,dtype=float).reshape(1,-1)
        both=agent_features(values)
        np.testing.assert_array_equal(both[0],values[0])
        self.assertEqual(both[1,0],192)
        self.assertEqual(both[1,384],391)
        self.assertEqual(both[1,398],403)
        self.assertEqual(both[1,408],413)
        self.assertEqual(both[1,-1],418)
        twice=agent_features(both[1:2])
        np.testing.assert_array_equal(twice[1],values[0])

    def test_feature_shape_mismatch_refused(self):
        from .subt_handoff_learn import agent_features
        with self.assertRaises(ValueError):agent_features(np.zeros((1,418)))

    def test_incomplete_training_matrix_refused_before_fit(self):
        from .subt_handoff_learn import complete_development_paths
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"incomplete: 0/8"):
                complete_development_paths(Path(directory))

    def test_duplicate_episodes_refused_before_fit(self):
        from .subt_handoff_learn import train
        with self.assertRaisesRegex(ValueError,"Repeated training"):
            train([Path("same"),Path("same")],Path("test"),Path("out"))


if __name__=="__main__":unittest.main()
