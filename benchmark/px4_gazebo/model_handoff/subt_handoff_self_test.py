"""Offline stair demonstration regressions; do not require Gazebo or PX4."""

import math
import io
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import numpy as np

from .subt_handoff_collect import poses_for, visibility_stages, resolve_message_delay
from .subt_handoff_design import design
from .subt_handoff_expert import EventExpert, duration, minimum_segment_separation, planned_state
from .subt_handoff_route import resample, sample, surface_height, AGENTS
from .subt_handoff_data import learning_rows, condition_coverage
from . import subt_handoff_inspect as inspection


class StairTests(unittest.TestCase):
    def route(self):
        return {"centers_enu":[[0.,0.,1.],[15.,0.,6.],[30.,0.,11.]],"distance_m":[0.,16.,32.]}

    def test_resample_bounds_and_endpoints(self):
        points=resample([[0,0,0],[1,0,1],[1,2,1]])
        np.testing.assert_allclose(points[0],[0,0,0])
        np.testing.assert_allclose(points[-1],[1,2,1])
        self.assertLessEqual(np.linalg.norm(np.diff(points,axis=0),axis=1).max(),.120001)

    def test_nonfinite_and_duplicate_rejected(self):
        for points in ([[0,0,0],[0,0,0]],[[0,0,0],[float("nan"),1,2]]):
            with self.assertRaises(ValueError):resample(points)
        for value in (-1.,float("nan"),float("inf")):
            with self.assertRaises(ValueError):sample(self.route(),value)

    def test_surface_selects_local_floor_not_overhead(self):
        triangles=np.array([[[-1,-1,z],[1,-1,z],[0,1,z]] for z in (0.,5.,10.)])
        self.assertAlmostEqual(surface_height(triangles,[0,0],.1),0.)
        self.assertAlmostEqual(surface_height(triangles,[0,0],5.1),5.)
        with self.assertRaises(ValueError):surface_height(triangles,[0,0],2.)

    def test_trajectory_hold_clamp_and_finite_limits(self):
        route=self.route()
        start=planned_state(route,0.)
        np.testing.assert_allclose(start["target_enu"],planned_state(route,4.)["target_enu"])
        np.testing.assert_allclose(planned_state(route,duration(route))["target_enu"],[30,0,11])
        for value in (0.,-.1,float("nan"),float("inf"),.5):
            with self.assertRaises(ValueError):planned_state(route,1.,speed=value)

    def test_identity_permutation_does_not_change_physical_plan(self):
        first=planned_state(self.route(),50.)
        other=planned_state(self.route(),50.,swapped=True)
        self.assertEqual(first["camera_poses_enu"]["uav_0"],other["camera_poses_enu"]["uav_1"])
        self.assertEqual(first["target_enu"],other["target_enu"])

    def test_teammate_delay_changes_camera_not_target(self):
        first=planned_state(self.route(),30.)
        other=planned_state(self.route(),30.,teammate="delayed")
        self.assertEqual(first["target_enu"],other["target_enu"])
        self.assertEqual(first["camera_poses_enu"]["uav_0"],other["camera_poses_enu"]["uav_0"])
        self.assertNotEqual(first["camera_poses_enu"]["uav_1"],other["camera_poses_enu"]["uav_1"])

    def test_communication_delay_independent_of_motion_with_legacy_defaults(self):
        self.assertEqual(resolve_message_delay("normal"),.5)
        self.assertEqual(resolve_message_delay("delayed"),2.)
        for delay in (0.,.5,2.,10.):
            self.assertEqual(resolve_message_delay("normal",delay),delay)
            self.assertEqual(resolve_message_delay("delayed",delay),delay)
        for value in (-1.,10.1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):resolve_message_delay("normal",value)

    def test_factorial_plan_crosses_identity_target_motion_and_communication(self):
        plan=design()
        self.assertEqual(plan,design())
        self.assertNotEqual(plan["conditions"],design(7)["conditions"])
        rows=plan["conditions"]
        self.assertEqual(len(rows),16)
        self.assertEqual(len({r["condition_id"] for r in rows}),16)
        for target in ("red:sphere","blue:cube"):
            for swapped in (False,True):
                subset=[r for r in rows if r["target"]==target and r["swapped"]==swapped]
                self.assertEqual({(r["teammate"],r["message_delay_s"]) for r in subset},
                                 {("normal",.5),("normal",2.),("delayed",.5),("delayed",2.)})
        self.assertIsNone(plan["role_assignment"])
        self.assertFalse(plan["test_partition_created"])

    def test_continuous_separation_detects_between_endpoint_crossing(self):
        self.assertAlmostEqual(minimum_segment_separation([2,0,0],[-2,0,0]),0.)
        self.assertAlmostEqual(minimum_segment_separation([2,1,0],[-2,1,0]),1.)

    def test_pose_allowlist(self):
        self.assertEqual(set(poses_for(planned_state(self.route(),1.))),
                         {"handoff_target","handoff_carrier","annotation_camera_0","annotation_camera_1"})

    def exercise_expert(self,owner,delay=.5):
        expert=EventExpert(delay)
        receiver=next(a for a in AGENTS if a!=owner)
        events=[]
        for t in np.arange(0.,10.,.5):
            visible={owner:t<5.,receiver:t>=2.}
            current=expert.step(float(t),visible)
            events.extend(current)
            if current and current[0]["event"]=="handoff_accept":
                self.assertGreaterEqual(t,expert.request_stamp+delay)
            if t<=5.:self.assertFalse(expert.complete)
        self.assertTrue(expert.complete)
        self.assertEqual(expert.owner,owner)
        self.assertEqual([e["event"] for e in events],["found","handoff_request","handoff_accept","confirm"])
        self.assertEqual(events[-1]["agent_id"],receiver)

    def test_causal_handoff_both_role_assignments(self):
        for agent in AGENTS:self.exercise_expert(agent)

    def test_delayed_teammate(self):
        self.exercise_expert("uav_1",delay=2.)

    def test_no_confirm_without_visual_reacquisition(self):
        expert=EventExpert()
        for t in range(10):expert.step(t,{"uav_0":t<3,"uav_1":False})
        self.assertFalse(expert.complete)
        self.assertFalse(expert.accepted)

    def test_ambiguous_start_does_not_assign_by_number(self):
        expert=EventExpert()
        for t in range(5):self.assertEqual(expert.step(t,dict.fromkeys(AGENTS,True)),[])
        self.assertIsNone(expert.owner)

    def test_expert_rejects_reused_or_invalid_timestamp(self):
        expert=EventExpert()
        expert.step(0.,dict.fromkeys(AGENTS,False))
        for value in (0.,-1.,float("nan")):
            with self.assertRaises(ValueError):expert.step(value,dict.fromkeys(AGENTS,False))

    def test_rendered_stage_order_and_duration_gate(self):
        rows=[{"elapsed_s":t,"visible":{"uav_0":t<4,"uav_1":t>=2}} for t in range(7)]
        self.assertTrue(visibility_stages(rows,"uav_0")["passed"])
        self.assertFalse(visibility_stages(rows[::-1],"uav_0")["passed"])
        self.assertFalse(visibility_stages(rows,None)["passed"])


class ExportTests(unittest.TestCase):
    def test_condition_coverage_counts_episodes_and_does_not_infer_legacy_conditions(self):
        reports=[{'condition':None}]
        first=condition_coverage(reports)
        self.assertEqual(first['unresolved_legacy_episodes'],1)
        self.assertFalse(first['complete_2x2x2x2_coverage'])
        for row in design()['conditions']:
            color,shape=row['target'].split(':')
            reports.append({'condition':{'target':{'color':color,'shape':shape},
                'swapped':row['swapped'],'teammate':row['teammate'],
                'message_delay_s':row['message_delay_s']}})
        final=condition_coverage(reports)
        self.assertTrue(final['complete_2x2x2x2_coverage'])
        self.assertEqual(final['recorded_condition_episodes'],16)
        self.assertEqual(len(final['counts']),16)
        self.assertTrue(all(row['episodes']==1 for row in final['counts']))
        duplicate=condition_coverage(reports+[reports[-1]])
        self.assertEqual(duplicate['recorded_condition_episodes'],17)
        self.assertEqual(len(duplicate['counts']),16)

    def examples(self):
        public=[{'sample_index':i,'stamp_ns':1_000_000_000+i*500_000_000,
                 'dt_s':None if i==0 else .5,'proprioception':{},'messages':[],
                 'target_pose':[99,99,99],'private_visibility':{'uav_0':True}}
                for i in range(2)]
        labels=[{'sample_index':i,'events':[]} for i in range(2)]
        labels[1]['events']=[{'agent_id':'uav_1','event':'confirm'}]
        images=[{a:f'images/{i}_{a}.jpg' for a in AGENTS} for i in range(2)]
        return public,labels,images

    def test_inputs_exclude_private_fields_and_current_or_future_labels(self):
        public,labels,images=self.examples()
        inputs,targets,futures,groups=learning_rows(public,labels,'track blue cube','ep',images)
        for row in inputs:
            self.assertEqual(set(row['observation']),
                             {'instruction','rgb','proprioception','messages','dt_s'})
        self.assertEqual(inputs[1]['observation']['messages'],[])
        self.assertEqual(targets[1]['joint_event']['uav_1'],'confirm')
        self.assertEqual(len(futures),1)
        self.assertEqual(futures[0]['next_key'],inputs[1]['key'])
        self.assertEqual(futures[0]['next_rgb'],images[1])
        self.assertTrue(all(r['split']=='development_training_only' for r in groups))
        self.assertEqual(len({r['scene_family_id'] for r in groups}),1)

    def test_prediction_pairs_reject_gaps_and_nonincreasing_time(self):
        for delta in (0,-1,1_000_000_001):
            public,labels,images=self.examples()
            public[1]['stamp_ns']=public[0]['stamp_ns']+delta
            with self.assertRaises(ValueError):
                learning_rows(public,labels,'instruction','ep',images)

    def test_row_inventory_and_indices_must_align(self):
        public,labels,images=self.examples()
        with self.assertRaises(ValueError):learning_rows(public,labels[:-1],'task','ep',images)
        with self.assertRaises(ValueError):learning_rows(public,labels,'task','ep',images[:-1])
        for collection in (public,labels):
            collection[1]['sample_index']=9
            with self.assertRaises(ValueError):learning_rows(public,labels,'task','ep',images)
            collection[1]['sample_index']=1

    def test_duplicate_or_unknown_event_agent_rejected(self):
        public,labels,images=self.examples()
        for events in ([{'agent_id':'uav_2','event':'found'}],
                       [{'agent_id':'uav_0','event':'found'},
                        {'agent_id':'uav_0','event':'confirm'}]):
            labels[1]['events']=events
            with self.assertRaises(ValueError):learning_rows(public,labels,'task','ep',images)

    def test_read_only_inspection_never_writes_or_regenerates_contact_sheet(self):
        with patch.object(inspection,'inspect',return_value={'passed':True,'files_sha256':{}}) as audit, \
             patch.object(inspection,'contact_sheet') as sheet, \
             patch.object(inspection,'read_rows') as read, \
             patch.object(inspection.Path,'write_text') as write, \
             redirect_stdout(io.StringIO()) as output:
            inspection.main(['--episode','episode','--read-only'])
        audit.assert_called_once()
        sheet.assert_not_called()
        read.assert_not_called()
        write.assert_not_called()
        self.assertIn('true',output.getvalue())

    def test_color_support_fraction_handles_valid_one_to_three_pixel_mask(self):
        # A valid bbox can contain a few target pixels at the occlusion edge.
        # Color support is fractional; visibility still uses the stricter bbox gate.
        from .subt_handoff_inspect import color_support
        rgb=np.zeros((1,3,3),dtype=np.uint8)
        rgb[0,:,2]=100
        labels=np.full((1,3),11,dtype=np.uint8)
        support=int(np.count_nonzero((labels==11)&color_support(rgb,'blue')))
        self.assertEqual(support,3)
        self.assertGreaterEqual(support,max(1,int(3*.15)))

    def test_read_only_inspection_does_not_mask_audit_failures(self):
        with patch.object(inspection,'inspect',side_effect=ValueError('bad digest')), \
             patch.object(inspection.Path,'write_text') as write:
            with self.assertRaisesRegex(ValueError,'bad digest'):
                inspection.main(['--episode','episode','--read-only'])
        write.assert_not_called()


if __name__=="__main__":unittest.main()
