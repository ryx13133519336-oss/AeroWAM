"""Sensor-contract tests only; no simulator, clock changes or flight."""
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from . import warehouse_localization as loc


class LocalizationTests(unittest.TestCase):
    def scene(self):
        return ET.ElementTree(ET.fromstring('<sdf><world name="warehouse">'
            '<include><name>annotation_uav_0</name><pose>0 2 0 0 0 0</pose></include>'
            '<include><name>annotation_uav_1</name><pose>0 -2 0 0 0 0</pose></include>'
            '<model name="target"><static>true</static></model></world></sdf>'))

    def test_legacy_is_byte_identical(self):
        tree=self.scene(); before=ET.tostring(tree.getroot())
        loc.instrument(tree, loc.LEGACY)
        self.assertEqual(ET.tostring(tree.getroot()), before)

    def test_only_both_aircraft_receive_identical_sensor(self):
        tree=self.scene(); before=ET.tostring(tree.getroot())
        loc.instrument(tree, loc.SIM_ODOM)
        plugins=tree.findall('.//plugin')
        self.assertEqual(len(plugins), 2)
        self.assertEqual(ET.tostring(plugins[0]), ET.tostring(plugins[1]))
        for node in tree.findall('.//include'):
            self.assertEqual(node.find('plugin/dimensions').text, '3')
            node.remove(node.find('plugin'))
        self.assertEqual(ET.tostring(tree.getroot()), before)

    def test_contract_discloses_ideal_localization_not_vio_or_truth(self):
        value=loc.contract(loc.SIM_ODOM)
        for key in ('real_visual_odometry', 'target_truth_input', 'flight_admission_granted'):
            self.assertFalse(value[key])
        self.assertTrue(value['separate_localization_track'])
        self.assertEqual(value['px4_parameters']['EKF2_GPS_CTRL'], 0)
        value['px4_parameters']['EKF2_EV_CTRL']=0
        self.assertEqual(loc.contract(loc.SIM_ODOM)['px4_parameters']['EKF2_EV_CTRL'], 15)

    def test_environment_is_scoped_copy_without_health_bypass(self):
        env={'PATH':'fixture'}
        result=loc.scope_environment(env, loc.SIM_ODOM)
        self.assertEqual(env, {'PATH':'fixture'})
        self.assertEqual(result, dict(env, **{'PX4_PARAM_'+k:str(v) for k,v in loc.PARAMETERS.items()}))
        self.assertEqual(loc.scope_environment(env, loc.LEGACY), env)
        self.assertFalse(any(k.startswith(('COM_', 'CBRK_')) for k in loc.PARAMETERS))

    def test_unknown_missing_and_duplicate_fail_closed(self):
        for fn in (loc.validate, loc.contract):
            with self.assertRaises(ValueError): fn('implicit_vio')
        tree=self.scene(); tree.find('world').remove(tree.find('world/include'))
        with self.assertRaises(ValueError): loc.instrument(tree, loc.SIM_ODOM)

    def fixture(self):
        link=NS(phase='VERIFIED',verified=True,reader_counts=lambda:dict.fromkeys(range(6),1),
                publishers={i:tuple(Mock() for _ in range(3)) for i in (0,1)})
        stage=loc.DisarmedLocalReadiness(link,NS,NS)
        status={i:NS(arming_state=1,failsafe=False,nav_state=4,pre_flight_checks_pass=False) for i in (0,1)}
        odometry={i:NS(timestamp=1,pose_frame=1,velocity_frame=1,position=[0.,0.,0.],
                       velocity=[0.,0.,0.],q=[1.,0.,0.,0.]) for i in (0,1)}
        def received(now):return {(kind,i):now for i in (0,1) for kind in ('status','odometry')}
        return stage,link,status,odometry,received

    def test_heartbeat_before_mode_no_trajectory_arm_or_health_bypass(self):
        stage,link,status,odo,received=self.fixture()
        with patch('time.monotonic_ns',return_value=10):
            for now in (0.,.5,1.,1.2):stage.service(status,odo,received(now),[],now)
        self.assertEqual(len(stage.commands),2)
        self.assertEqual({x['command'] for x in stage.commands},{176})
        for i in (0,1):
            link.publishers[i][1].publish.assert_not_called()
            self.assertGreaterEqual(link.publishers[i][0].publish.call_count,3)
            command=link.publishers[i][2].publish.call_args.args[0]
            self.assertEqual((command.source_system,command.source_component),(255,2))
            self.assertEqual(command.param1,1.)
            self.assertEqual(command.param2,6.)
            self.assertTrue(command.from_external)
        acks=[dict(agent=f'uav_{i}',command=176,result=0,target_system=255,
                   target_component=2,monotonic_ns=10) for i in (0,1)]
        stage.service(status,odo,received(1.3),acks,1.3)
        self.assertFalse(stage.ready)  # ACK does not replace actual healthy status.
        for s in status.values():s.nav_state=14;s.pre_flight_checks_pass=True
        stage.service(status,odo,received(1.4),acks,1.4)
        self.assertTrue(stage.ready)

    def test_stale_wrong_origin_ack_and_unhealthy_or_armed_fail_closed(self):
        stage,link,status,odo,received=self.fixture()
        stage.service(status,odo,received(0.),[],0.)
        with self.assertRaises(RuntimeError):stage.service(status,odo,received(0.),[],2.1)
        for field,value in (('arming_state',2),('failsafe',True)):
            stage,link,status,odo,received=self.fixture()
            setattr(status[0],field,value)
            with self.assertRaises(RuntimeError):stage.service(status,odo,received(0.),[],0.)
        stage,link,status,odo,received=self.fixture()
        stage.first_request_ns={'uav_0':10,'uav_1':10}
        for s in status.values():s.nav_state=14;s.pre_flight_checks_pass=True
        for bad in (dict(target_system=1,target_component=2,monotonic_ns=10),
                    dict(target_system=255,target_component=1,monotonic_ns=10),
                    dict(target_system=255,target_component=2,monotonic_ns=9)):
            acks=[dict(agent=f'uav_{i}',command=176,result=0,**bad) for i in (0,1)]
            stage.service(status,odo,received(0.),acks,0.)
            self.assertFalse(stage.ready)
        with self.assertRaises(TimeoutError):stage.service(status,odo,received(26.),[],26.)

    def test_inactive_handoff_stops_all_readiness_publications(self):
        stage,link,status,odo,received=self.fixture();stage.active=False
        status[0].arming_state=2
        stage.service(status,odo,received(0.),[],0.)
        for pubs in link.publishers.values():
            for p in pubs:p.publish.assert_not_called()
        tree=self.scene(); loc.instrument(tree, loc.SIM_ODOM)
        with self.assertRaises(ValueError): loc.instrument(tree, loc.SIM_ODOM)

    def test_missing_or_extra_or_lost_control_readers_fail_before_publish(self):
        for counts in ({},dict.fromkeys(range(5),1),dict.fromkeys(range(7),1),
                       dict.fromkeys(range(6),0),dict.fromkeys(range(6),2)):
            stage,link,status,odo,received=self.fixture()
            link.reader_counts=lambda:counts
            with self.assertRaises(RuntimeError):
                stage.service(status,odo,received(0.),[],0.)
            for pubs in link.publishers.values():
                for publisher in pubs:publisher.publish.assert_not_called()

    def test_rejected_mode_ack_never_substitutes_for_acceptance(self):
        stage,link,status,odo,received=self.fixture()
        stage.first_request_ns={'uav_0':10,'uav_1':10}
        for state in status.values():
            state.nav_state=14;state.pre_flight_checks_pass=True
        for result in (1,2,3,4,5,6):
            acks=[dict(agent=f'uav_{i}',command=176,result=result,target_system=255,
                       target_component=2,monotonic_ns=10) for i in (0,1)]
            stage.service(status,odo,received(0.),acks,0.)
            self.assertFalse(stage.ready)


if __name__=='__main__':
    unittest.main()
