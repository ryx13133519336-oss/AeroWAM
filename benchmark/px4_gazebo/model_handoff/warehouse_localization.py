"""Explicit simulator self-localization profile, not an actual VIO system.

Only own-vehicle odometry enters PX4's existing estimator. Target truth,
labels, warehouse geometry and evaluator state never enter this channel.
Legacy GPS worlds remain byte-identical unless this profile is selected.
"""
from copy import deepcopy
import xml.etree.ElementTree as ET

LEGACY = 'gps_v1'
SIM_ODOM = 'sim_own_odometry_v1'
PROFILES = (LEGACY, SIM_ODOM)
PARAMETERS = dict(EKF2_EV_CTRL=15, EKF2_HGT_REF=3, EKF2_GPS_CTRL=0,
                  EKF2_EV_DELAY=0, SIM_GZ_EN_ODOM=1)
# Commander addresses each ACK to the command's source, not its destination.
# Keep readiness distinct from the component-1 disarm/flight controller.
READINESS_SOURCE_SYSTEM = 255
READINESS_SOURCE_COMPONENT = 2


def validate(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown explicit warehouse localization profile')
    return profile


def contract(profile):
    validate(profile)
    return dict(schema='aerowam-warehouse-localization/1', profile=profile,
        source='Gazebo_OdometryPublisher_ideal_simulated_own_pose_and_twist'
               if profile == SIM_ODOM else 'PX4_Gazebo_noisy_GPS_default',
        real_visual_odometry=False, target_truth_input=False,
        policy_observation='existing_local_body_state_only_no_global_pose_or_map',
        px4_parameters=deepcopy(PARAMETERS) if profile == SIM_ODOM else {},
        simulator_odometry=dict(dimensions=3, frequency_hz=50., gaussian_noise=0.)
                            if profile == SIM_ODOM else None,
        disarmed_readiness='fresh_disarm_ACK_then_OFFBOARD_heartbeat_and_mode_ACK_no_setpoints'
                            if profile == SIM_ODOM else 'legacy_default_mode',
        separate_localization_track=True, flight_admission_granted=False)


def instrument(tree, profile):
    validate(profile)
    if profile == LEGACY:
        return
    scene = tree.getroot().find('world')
    includes = [scene.find(f"include[name='annotation_uav_{i}']") for i in (0,1)]
    if any(node is None for node in includes):
        raise ValueError('Exactly two existing aircraft includes required')
    if any(node.findall("plugin[@name='gz::sim::systems::OdometryPublisher']") for node in includes):
        raise ValueError('Duplicate own-odometry instrumentation')
    for node in includes:
        plugin = ET.SubElement(node, 'plugin', filename='gz-sim-odometry-publisher-system',
                               name='gz::sim::systems::OdometryPublisher')
        for key,value in (('dimensions','3'),('odom_publish_frequency','50'),('gaussian_noise','0')):
            ET.SubElement(plugin,key).text=value


def scope_environment(env, profile):
    validate(profile)
    result = dict(env)
    if profile == SIM_ODOM:
        result.update({'PX4_PARAM_'+name:str(value) for name,value in PARAMETERS.items()})
    return result


class DisarmedLocalReadiness:
    """Select the actual local autonomous mode, never bypass its health checks.

    GPS-free indoor localization cannot make global Loiter armable. OFFBOARD
    needs a live heartbeat before mode selection. This stage sends no arm or
    trajectory messages, requires the original fresh disarm roundtrip first,
    and leaves the existing preflight flag/freshness checks mandatory.
    """
    def __init__(self, link, heartbeat_type, command_type):
        self.link=link
        self.heartbeat_type,self.command_type=heartbeat_type,command_type
        self.started=self.first_heartbeat=None
        self.last_heartbeat=self.last_command=-float('inf')
        self.first_request_ns={}
        self.commands=[]
        self.active=True
        self.ready=False

    def service(self, status, odometry, received, acks, now):
        import math
        import time
        from .annotation_session_smoke import INSTANCES
        if not self.active:
            return
        if self.link.phase=='NOT_STARTED':
            # Do not start the bounded roundtrip timer until both streams exist.
            if set(status)!={0,1} or set(odometry)!={0,1}:
                return
            self.link.start()
        if not self.link.verified:
            self.link.poll(acks,status,received)
            if self.link.phase=='FAILED':
                raise RuntimeError(self.link.reason)
            return
        if self.started is None:
            self.started=now
        readers=self.link.reader_counts()
        if (set(status)!={0,1} or set(odometry)!={0,1}
                or any(s.arming_state!=1 or s.failsafe for s in status.values())
                or any(not 0<=now-received.get((kind,i),-float('inf'))<=limit
                       for kind,limit in (('status',2.),('odometry',.5)) for i in (0,1))
                or any(m.pose_frame!=1 or m.velocity_frame!=1
                       or not all(math.isfinite(float(v)) for xs in (m.position,m.velocity,m.q) for v in xs)
                       for m in odometry.values())
                or len(readers)!=6 or any(n!=1 for n in readers.values())):
            raise RuntimeError('local_readiness_requires_fresh_disarmed_healthy_matched_vehicles')
        if now-self.last_heartbeat>=.1:
            for i in (0,1):
                message=self.heartbeat_type()
                message.timestamp=int(odometry[i].timestamp)
                message.position=True
                self.link.publishers[i][0].publish(message)
            self.last_heartbeat=now
            if self.first_heartbeat is None:
                self.first_heartbeat=now
        accepted={r['agent'] for r in acks if r.get('command')==176 and r.get('result')==0
                  and r.get('target_system')==READINESS_SOURCE_SYSTEM
                  and r.get('target_component')==READINESS_SOURCE_COMPONENT
                  and r.get('agent') in self.first_request_ns
                  and r.get('monotonic_ns',-1)>=self.first_request_ns[r['agent']]}
        if now-self.first_heartbeat>=1.1 and now-self.last_command>=1. and not self.ready:
            for i in (0,1):
                agent=f'uav_{i}'
                if agent in accepted:
                    continue
                message=self.command_type()
                message.timestamp=int(odometry[i].timestamp)
                message.command,message.param1,message.param2=176,1.,6.
                message.target_system,message.target_component=INSTANCES[i]+1,1
                message.source_system=READINESS_SOURCE_SYSTEM
                message.source_component=READINESS_SOURCE_COMPONENT
                message.from_external=True
                sent=time.monotonic_ns()
                self.first_request_ns.setdefault(agent,sent)
                self.link.publishers[i][2].publish(message)
                self.commands.append(dict(agent=agent,command=176,param1=1.,param2=6.,monotonic_ns=sent))
            self.last_command=now
        self.ready=(accepted=={'uav_0','uav_1'} and all(
            s.nav_state==14 and s.pre_flight_checks_pass for s in status.values()))
        if not self.ready and now-self.started>25.:
            raise TimeoutError('disarmed_local_OFFBOARD_readiness_failed')

    def diagnostics(self):
        return dict(ready=self.ready,commands=list(self.commands),arm_commands=0,
                    trajectory_setpoints=0,health_checks_bypassed=False)
