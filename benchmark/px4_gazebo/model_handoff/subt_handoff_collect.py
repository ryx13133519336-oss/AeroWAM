"""Gazebo-only stair handoff demonstrations with privileged teacher supervision.

Moving camera proxies are NOT flying aircraft. No PX4/ROS/controller imports.
Inputs, oracle labels and private geometry/instance maps are separate exports.
"""

import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from .industrial_targets import COLORS, IDENTITIES, parse_component, part, static_model, visible_components
from .subt_handoff_expert import EventExpert, audit_expert, duration, planned_state
from .subt_handoff_route import AGENTS, audit_route, build_route, load_triangles, plot_route
from .subt_urban import stop_owned
from .subt_urban_scene import WORLD, scene
from .warehouse_scene import sha256

NAMES = ("handoff_target", "handoff_carrier", "annotation_camera_0", "annotation_camera_1")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def poses_for(state):
    target = state["target_enu"]
    return {"handoff_target": [*target,0.,0.,0.],
            "handoff_carrier": [target[0],target[1],target[2]-.65,0.,0.,0.],
            **{f"annotation_camera_{i}": state["camera_poses_enu"][a] for i,a in enumerate(AGENTS)}}


def pose_worker():
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.boolean_pb2 import Boolean
    node = Node()
    for line in sys.stdin:
        try:
            command = json.loads(line)
            if set(command)!=set(NAMES):
                raise ValueError("Only owned scenario entities allowed")
            message = Pose_V()
            for name,values in command.items():
                if len(values)!=6 or not all(math.isfinite(v) for v in values) or values[3]!=0:
                    raise ValueError("Finite zero-roll pose required")
                p=message.pose.add()
                p.name=name
                p.position.x,p.position.y,p.position.z=values[:3]
                pitch,yaw=values[4:]
                p.orientation.x=-math.sin(yaw/2)*math.sin(pitch/2)
                p.orientation.y=math.cos(yaw/2)*math.sin(pitch/2)
                p.orientation.z=math.sin(yaw/2)*math.cos(pitch/2)
                p.orientation.w=math.cos(yaw/2)*math.cos(pitch/2)
            ok,response=node.request(f"/world/{WORLD}/set_pose_vector",message,Pose_V,Boolean,1000)
            result={"accepted":bool(ok and response.data)}
        except Exception as exc:
            result={"accepted":False,"error":f"{type(exc).__name__}:{exc}"}
        print(json.dumps(result),flush=True)


def command_worker(worker, poses):
    if worker.poll() is not None:
        raise RuntimeError("Pose helper exited")
    worker.stdin.write(json.dumps(poses,allow_nan=False)+"\n")
    worker.stdin.flush()
    if not select.select([worker.stdout],[],[],2.)[0]:
        raise TimeoutError("Pose helper timed out")
    line=worker.stdout.readline()
    if not line or not json.loads(line).get("accepted"):
        raise RuntimeError("Scene pose request rejected")


def prepare_scene(package, models, route, *, component, swapped=False, teammate="normal"):
    tree,contract=scene(package,models,case="revealed")
    world=tree.getroot().find("world")
    for name in (*IDENTITIES,"industrial_mount_a","industrial_mount_b"):
        world.remove(world.find(f"model[@name='{name}']"))
    state=planned_state(route,0.,swapped=swapped,teammate=teammate)
    poses=poses_for(state)
    for i,agent in enumerate(AGENTS):
        wrapper=world.find(f"model[@name='annotation_camera_{i}']")
        wrapper.find("pose").text=" ".join(map(str,state["camera_poses_enu"][agent]))
    body=static_model(world,"handoff_target",poses["handoff_target"][:3],11)
    part(body,"component",component.shape,(.8,.8,.8),(0,0,0),COLORS[component.color])
    carrier=static_model(world,"handoff_carrier",poses["handoff_carrier"][:3],100)
    part(carrier,"compact_base","cube",(.6,.6,.4),(0,0,0),(.14,.16,.17))
    part(carrier,"insert_support","cube",(.1,.1,.4),(0,0,.3),(.14,.16,.17))
    contract.update({"scenario":"stair_handoff_kinematic_proxies", "role_assignment":None,
                     "expert_is_privileged":True,"aircraft_spawn_poses_enu":None,
                     "private_targets_enu":{"handoff_target":state["target_enu"]},
                     "camera_poses_enu":state["camera_poses_enu"],"contact_sensors":[],
                     "industrial_targets":None,"target_config_sha256":None,
                     "motion_commands_sent":0,"flight_admitted":False})
    return tree,contract


def visibility_stages(rows, owner):
    if owner not in AGENTS:
        return {"passed":False,"stages":[],"reason":"no_initial_owner"}
    receiver=next(a for a in AGENTS if a!=owner)
    stages=[]
    previous=-float('inf')
    for row in rows:
        elapsed=row['elapsed_s']
        if not math.isfinite(elapsed) or elapsed<=previous or set(row['visible'])!=set(AGENTS):
            return {'passed':False,'stages':[], 'reason':'nonincreasing_or_invalid_rows'}
        if any(type(value) is not bool for value in row['visible'].values()):
            return {'passed':False,'stages':[], 'reason':'visibility_must_be_boolean'}
        previous=elapsed
        pattern=[row["visible"][a] for a in (owner,receiver)]
        if not stages or stages[-1]["pattern"]!=pattern:
            stages.append({"pattern":pattern,"start_s":row["elapsed_s"],"end_s":row["elapsed_s"]})
        else:
            stages[-1]["end_s"]=row["elapsed_s"]
    selected=[]
    expected=[[True,False],[True,True],[False,True]]
    for stage in stages:
        if (len(selected)<3 and stage["pattern"]==expected[len(selected)]
                and stage["end_s"]-stage["start_s"]>=1.):
            selected.append(stage)
    return {"passed":len(selected)==3,"stages":stages,"selected":selected}


def resolve_message_delay(teammate, message_delay_s=None):
    """Explicit communication is independent of motion; None preserves old runs."""
    if teammate not in {"normal", "delayed"}:
        raise ValueError("Declared teammate motion required")
    delay = (2. if teammate == "delayed" else .5) if message_delay_s is None else message_delay_s
    if isinstance(delay, bool) or not math.isfinite(delay) or not 0 <= delay <= 10:
        raise ValueError("Finite message delay in 0..10 seconds required")
    return float(delay)


def collect(package,models,output,*,component,speed=.3,swapped=False,teammate="normal",reverse=False,
            timeout=720.,message_delay_s=None):
    delay_s=resolve_message_delay(teammate,message_delay_s)
    if not math.isfinite(timeout) or not 30<=timeout<=1800:
        raise ValueError("Finite bounded wall timeout 30..1800 required")
    total_start=time.monotonic()
    triangles=load_triangles(package)
    route=build_route(triangles,reverse=reverse)
    geometry=audit_route(triangles,route)
    expert_audit=audit_expert(triangles,route,speed=speed,swapped=swapped,teammate=teammate)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    summary={"schema":"aerowam-subt-stair-demonstration/1","passed":False,
             "data_scope":"privileged_kinematic_RGB_event_demonstration_not_flight_controls",
             "dataset_admitted":False,"flight_admitted":False,"px4_started":False,
             "flight_commands_sent":0,"model_evaluated":False,"formal_score":False,
             "geometry":geometry,"expert_audit":expert_audit,"cleanup_errors":[]}
    summary['condition_private']={'target':{'color':component.color,'shape':component.shape},
        'speed_m_s':speed,'swapped':swapped,'teammate':teammate,'reverse':reverse,
        'message_delay_s':delay_s,
        'message_delay_selection':'legacy_motion_default' if message_delay_s is None else 'explicit_independent'}
    summary['source_sha256']={name:sha256(Path(__file__).with_name(name)) for name in
        ('subt_handoff_collect.py','subt_handoff_expert.py','subt_handoff_route.py')}
    write_json(output/"route.private.json",route)
    plot_route(triangles,route,output)
    if not geometry["passed"] or not expert_audit["passed"]:
        write_json(output/"summary.json",summary)
        return summary
    tree,contract=prepare_scene(package,models,route,component=component,swapped=swapped,teammate=teammate)
    tree.write(output/"world.sdf",encoding="utf8",xml_declaration=True)
    contract["world_sha256"]=sha256(output/"world.sdf")
    write_json(output/"scene_contract.private.json",contract)
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_v_pb2 import Pose_V
    from .gazebo_rasters import GZ8_INSTANCE_ENCODING,pair_rgb_instance,resize_pair
    from PIL import Image as PILImage
    previous_partition=os.environ.get("GZ_PARTITION")
    os.environ["GZ_PARTITION"]="aerowam_stair_teacher_"+uuid.uuid4().hex
    node=Node()
    lock=threading.Lock()
    buffers={}
    feedback=OrderedDict()
    subscriptions=[]
    processes=[]
    rows=[]
    labels=[]
    messages=[]
    origin=None
    expert=EventExpert(delay_s)
    for folder in ("public/images","supervision","private"):
        (output/folder).mkdir(parents=True)
    metadata={"schema":summary["schema"],"instruction":f"Track the {component.color} {component.shape} across floors; hand off the same target and confirm after your teammate loses sight.",
              "split":"development_training_only","scene_family":"subt_urban_2_story_stair_v1",
              "role_assignment":None,"expert_is_privileged":True,"px4_started":False,
              "observation_fields":["rgb","proprioception","messages"],
              "proprioception_source":"ideal_camera_proxy_pose_not_PX4_odometry",
              "action_scope":"event_and_oracle_camera_waypoint_not_flight_control",
              "history":"only_previous_labels_delivered_after_declared_delay",
              "dt_target_s":.5,"world_model_scope":"rendered_kinematic_transitions_not_UAV_dynamics",
              "camera_orientation_policy":"privileged_look_at_including_while_target_hidden_not_learned_tracking",
              "motion_supervision_semantics":"desired_next_pose_label_not_a_logged_executed_action",
              "training_eligibility":"requires_independent_inspection_json_pass_not_summary_alone",
              "dataset_admitted":False}
    write_json(output/"public/metadata.json",metadata)
    initial=planned_state(route,0.,speed=speed,swapped=swapped,teammate=teammate)

    def receive_pose(msg):
        stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nsec
        selected={p.name:{"xyz":[p.position.x,p.position.y,p.position.z],
                          "quaternion_xyzw":[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]}
                  for p in msg.pose if p.name in NAMES}
        if set(selected)==set(NAMES):
            with lock:
                feedback[stamp]=selected
                while len(feedback)>64:feedback.popitem(last=False)

    try:
        topic=f"/world/{WORLD}/pose/info"
        if not node.subscribe(Pose_V,topic,receive_pose):raise RuntimeError("Pose subscription failed")
        subscriptions.append(topic)
        for spec in contract["camera_pairs"]:
            for topic in (spec["rgb"],spec["segmentation"]+"/labels_map"):
                buffers[topic]=OrderedDict()
                def receive_image(msg,key=topic):
                    stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nsec
                    raw=msg.SerializeToString()
                    with lock:
                        buffers[key][stamp]=raw
                        while len(buffers[key])>8:buffers[key].popitem(last=False)
                if not node.subscribe(Image,topic,receive_image):raise RuntimeError("Image subscription failed")
                subscriptions.append(topic)
        with (output/"gazebo.log").open("xb") as log:
            env=dict(os.environ)
            env.pop("GZ_SIM_SERVER_CONFIG_PATH",None)
            simulator=subprocess.Popen(["gz","sim","-s","-r","--headless-rendering","-v","3",str(output/"world.sdf")],
                                       env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(simulator)
            worker=subprocess.Popen([sys.executable,"-B","-u","-m",__spec__.name,"--worker"],
                                    env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                                    text=True,bufsize=1,start_new_session=True)
            processes.append(worker)
            last_command=-1
            previous_feedback_stamp=-1
            previous_frame_stamp=-1
            last_frame=-1
            last_feedback_wall=time.monotonic()
            deadline=time.monotonic()+timeout
            with (output/"public/observations.jsonl").open("x",encoding="utf8") as public_stream, \
                 (output/"supervision/labels.jsonl").open("x",encoding="utf8") as label_stream, \
                 (output/"private/annotations.jsonl").open("x",encoding="utf8") as private_stream, \
                 (output/"private/pose_commands.jsonl").open("x",encoding="utf8") as command_stream:
                while time.monotonic()<deadline:
                    if any(p.poll() is not None for p in processes):raise RuntimeError("Owned process exited early")
                    with lock:
                        latest=max(feedback,default=-1)
                        common=set.intersection(*(set(v) for v in buffers.values())) if buffers else set()
                        stamp=max(common,default=-1)
                        image_raw={k:v[stamp] for k,v in buffers.items()} if stamp>=0 else None
                        pose_stamp=max((s for s in feedback if s<=stamp),default=-1)
                        actual=feedback.get(pose_stamp)
                    if latest<previous_feedback_stamp or stamp<previous_frame_stamp:
                        raise RuntimeError('Simulation timestamp regressed')
                    if latest>previous_feedback_stamp:
                        last_feedback_wall=time.monotonic()
                        previous_feedback_stamp=latest
                    previous_frame_stamp=stamp
                    if origin is None and actual is not None:
                        origin=stamp
                    if origin is None:
                        if time.monotonic()-last_feedback_wall>45:raise TimeoutError("Initial image/pose streams missing")
                        time.sleep(.02)
                        continue
                    if latest-origin<0 or stamp-origin<0:raise RuntimeError("Simulation clock regression")
                    elapsed=(latest-origin)/1e9
                    if latest-last_command>=50_000_000:
                        state=planned_state(route,elapsed,speed=speed,swapped=swapped,teammate=teammate)
                        command_worker(worker,poses_for(state))
                        command_stream.write(json.dumps({'command_observed_sim_stamp_ns':latest,
                            'elapsed_sim_s':elapsed,'requested_poses_enu':poses_for(state),
                            'service_accepted':True,'verified_actuation_at_image_exposure':False},allow_nan=False)+'\n')
                        last_command=latest
                    if time.monotonic()-last_feedback_wall>5:raise RuntimeError("Pose stream stale")
                    if (stamp>last_frame and (last_frame<0 or stamp-last_frame>=500_000_000)
                            and actual is not None and 0<=stamp-pose_stamp<=100_000_000):
                        t=(stamp-origin)/1e9
                        plan=planned_state(route,t,speed=speed,swapped=swapped,teammate=teammate)
                        expected=poses_for(plan)
                        error=max(math.dist(actual[n]["xyz"],expected[n][:3]) for n in NAMES)
                        if error>.15:raise RuntimeError(f"Actual scene pose deviates from expert plan: {error:.3f}m")
                        index=len(rows)
                        visible={}
                        annotations={}
                        paths={}
                        hashes={}
                        for spec in contract["camera_pairs"]:
                            agent=spec["agent_id"]
                            rgb,mask=Image(),Image()
                            rgb.ParseFromString(image_raw[spec["rgb"]])
                            mask.ParseFromString(image_raw[spec["segmentation"]+"/labels_map"])
                            pair=pair_rgb_instance(rgb_message=rgb,label_message=mask,rgb_topic=spec["rgb"],
                                segmentation_topic=spec["segmentation"],label_topic=spec["segmentation"]+"/labels_map",encoding=GZ8_INSTANCE_ENCODING)
                            reduced=resize_pair(pair)
                            findings=visible_components(reduced.object_labels)[IDENTITIES[0]]
                            visible[agent]=findings["bbox"] is not None
                            annotations[agent]=findings
                            relative=f"images/{index:05d}_{agent}.jpg"
                            PILImage.fromarray(reduced.rgb).save(output/"public"/relative,quality=95)
                            paths[agent]=relative
                            hashes[agent]=sha256(output/"public"/relative)
                            PILImage.fromarray(reduced.object_labels).save(output/f"private/{index:05d}_{agent}_labels.png")
                        public={"sample_index":index,"stamp_ns":stamp,"dt_s":None if last_frame<0 else (stamp-last_frame)/1e9,
                                "rgb":paths,"rgb_sha256":hashes,
                                "proprioception":{a:{"position_local_m":(np.array(actual[f'annotation_camera_{i}']["xyz"])
                                    -initial["camera_poses_enu"][a][:3]).tolist(),
                                    "quaternion_xyzw":actual[f'annotation_camera_{i}']["quaternion_xyzw"]} for i,a in enumerate(AGENTS)},
                                "messages":[m["claim"] for m in messages if m["deliver_s"]<=t]}
                        events=expert.step(t,visible)
                        after=planned_state(route,t+.5,speed=speed,swapped=swapped,teammate=teammate)
                        label={"sample_index":index,"stamp_ns":stamp,"events":events,
                               "oracle_camera_waypoint_delta_enu_m":{a:(np.array(after["camera_poses_enu"][a][:3])
                                   -actual[f'annotation_camera_{i}']["xyz"]).tolist() for i,a in enumerate(AGENTS)},
                               "flight_control_label":False}
                        for event in events:messages.append({"deliver_s":t+expert.delay_s,"claim":event})
                        row={"sample_index":index,"stamp_ns":stamp,"elapsed_s":t,"pose_stamp_ns":pose_stamp,
                             "actual":actual,"visible":visible,"annotations":annotations,"path_error_m":error}
                        public_stream.write(json.dumps(public,allow_nan=False)+"\n")
                        label_stream.write(json.dumps(label,allow_nan=False)+"\n")
                        private_stream.write(json.dumps(row,allow_nan=False)+"\n")
                        rows.append(row)
                        labels.append(label)
                        last_frame=stamp
                        if events or index%20==0:print(json.dumps({"sample":index,"sim_s":round(t,2),"visible":visible,"events":events}),flush=True)
                        if t>=duration(route,speed):break
                    time.sleep(.005)
                else:
                    raise TimeoutError("Bounded demonstration collection timed out")
            stages=visibility_stages(rows,expert.owner)
            gaps=[(b["stamp_ns"]-a["stamp_ns"])/1e9 for a,b in zip(rows,rows[1:])]
            heights=[row["actual"]["handoff_target"]["xyz"][2] for row in rows]
            summary.update({"rendered_visibility":stages,"expert_complete":expert.complete,
                            "initial_owner":expert.owner,"receiver":expert.receiver,"samples":len(rows),
                            "max_capture_gap_s":max(gaps,default=0.),"target_vertical_span_m":max(heights)-min(heights),
                            "events":[e for label in labels for e in label["events"]],
                            "elapsed_sim_s":rows[-1]["elapsed_s"]})
            summary["passed"]=(stages["passed"] and expert.complete and max(gaps,default=1.)<=1.
                                and max(heights)-min(heights)>=9. and rows[-1]["elapsed_s"]>=duration(route,speed))
    except Exception as exc:
        summary["error"]=f"{type(exc).__name__}:{exc}"
    finally:
        for process in reversed(processes):
            try:stop_owned(process)
            except Exception as exc:summary["cleanup_errors"].append(str(exc))
        for topic in subscriptions:
            try:node.unsubscribe(topic)
            except Exception as exc:summary['cleanup_errors'].append(f'unsubscribe:{exc}')
        if previous_partition is None:os.environ.pop("GZ_PARTITION",None)
        else:os.environ["GZ_PARTITION"]=previous_partition
        summary["owned_processes_stopped"]=all(p.poll() is not None for p in processes)
        summary["dataset_admitted"]=bool(summary["passed"] and summary["owned_processes_stopped"] and not summary["cleanup_errors"])
        metadata["dataset_admitted"]=summary["dataset_admitted"]
        write_json(output/"public/metadata.json",metadata)
        summary["wall_s"]=time.monotonic()-total_start
        write_json(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("package","models","output-dir"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--target",type=parse_component,default=parse_component("red:sphere"))
    parser.add_argument("--speed",type=float,default=.3)
    parser.add_argument("--swapped",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    parser.add_argument("--teammate",choices=("normal","delayed"),default="normal")
    parser.add_argument("--message-delay-s",type=float,
                        help="independent communication delay 0..10 s; unset preserves legacy motion-dependent defaults")
    parser.add_argument("--timeout",type=float,default=720.)
    args=parser.parse_args()
    result=collect(args.package,args.models,args.output_dir,component=args.target,speed=args.speed,
                   swapped=args.swapped,teammate=args.teammate,reverse=args.reverse,timeout=args.timeout,
                   message_delay_s=args.message_delay_s)
    print(json.dumps({k:v for k,v in result.items() if k not in {"geometry","expert_audit","rendered_visibility"}}),flush=True)
    raise SystemExit(0 if result["dataset_admitted"] else 2)


if __name__=="__main__":
    if sys.argv[1:]==["--worker"]:pose_worker()
    else:main()
