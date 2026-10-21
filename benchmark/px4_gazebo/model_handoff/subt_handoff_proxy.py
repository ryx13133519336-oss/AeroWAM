"""Action-driven Gazebo camera-proxy diagnostic; no PX4 or flight commands.

Only the target follows the private stair route. After initialization neither
camera is steered by the teacher: both translations/yaw come from the core
model outputs. This is centralized, kinematic and excluded from formal scores.
"""

import argparse
from collections import OrderedDict
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image as PILImage

from .core._common import canonical
from .industrial_targets import IDENTITIES, parse_component, visible_components
from .policy_client import PolicyClient
from .subt_handoff_collect import NAMES, command_worker, prepare_scene, write_json
from .subt_handoff_evaluator import SubTHandoffEvaluator
from .subt_handoff_expert import planned_state, minimum_segment_separation
from .subt_handoff_model_io import model_input, validate_output
from .subt_handoff_route import AGENTS, audit_route, build_route, local_triangles
from .subt_urban import stop_owned
from .subt_urban_geometry import load_triangles, triangles_touch_box, segment_hits
from .subt_urban_scene import WORLD
from .warehouse_scene import sha256


def propose_cameras(cameras, response, triangles, target):
    """Conservative proxy swept boxes and separation; never x500 admission."""
    proposals = {}
    for agent in AGENTS:
        action = response["actions"][agent]
        old = np.asarray(cameras[agent][:3], dtype=float)
        new = old + np.asarray(action["delta_enu_m"], dtype=float)
        if np.any(new < [-20.,-25.,.5]) or np.any(new > [20.,20.,18.]):
            raise RuntimeError("proxy_geofence")
        low, high = np.minimum(old,new)-[.4,.4,.25], np.maximum(old,new)+[.4,.4,.25]
        if triangles_touch_box(local_triangles(triangles,low,high),low,high):
            raise RuntimeError("proxy_mesh_sweep")
        if minimum_segment_separation(old-np.asarray(target),new-np.asarray(target)) < 1.:
            raise RuntimeError("proxy_target_separation")
        proposals[agent] = [*new.tolist(),0.,cameras[agent][4],
                            cameras[agent][5]+action["delta_yaw_rad"]]
    before = np.asarray(cameras[AGENTS[0]][:3])-np.asarray(cameras[AGENTS[1]][:3])
    after = np.asarray(proposals[AGENTS[0]][:3])-np.asarray(proposals[AGENTS[1]][:3])
    if minimum_segment_separation(before,after) < 1.:
        raise RuntimeError("proxy_inter_camera_separation")
    return proposals


def target_poses(route, elapsed, speed, cameras):
    # Environment target motion only. Never read the teacher camera trajectory
    # after initialization or steer a model camera toward the hidden target.
    from .subt_handoff_route import sample
    target = sample(route, max(0.,elapsed-4.)*speed).tolist()
    return {"handoff_target":[*target,0.,0.,0.],
            "handoff_carrier":[target[0],target[1],target[2]-.65,0.,0.,0.],
            **{f"annotation_camera_{i}":cameras[a] for i,a in enumerate(AGENTS)}}


def full_occlusion(triangles, eye, target, native_pixels):
    """One opaque triangle blocks ALL target-box corners, plus zero pixels.

    A common convex blocker is conservative: unrelated corner blockers could
    leave a hole through the middle and are insufficient to certify the box.
    """
    import itertools
    if native_pixels != 0:
        return False
    eye,target=np.asarray(eye),np.asarray(target)
    corners=[target+corner for corner in itertools.product((-.4,.4),repeat=3)]
    candidates=local_triangles(triangles,np.minimum(eye,target-.4)-.001,np.maximum(eye,target+.4)+.001)
    return any(all(len(segment_hits(np.asarray([triangle]),eye,corner)) for corner in corners)
               for triangle in candidates)


def run(package, models, output, *, manifest, policy, seconds=20., target="red:sphere", swapped=False,
        reverse=False, gui=False, wall_timeout_s=300.):
    if os.name == "nt":
        raise RuntimeError("Run in WSL/Linux with Gazebo bindings")
    if not math.isfinite(seconds) or not 5 <= seconds <= 180 or not 60 <= wall_timeout_s <= 1800:
        raise ValueError("Use 5..180 sim seconds and 60..1800 wall seconds")
    # No simultaneous simulator ownership, no cleanup of another task.
    table = subprocess.run(["ps","-eo","args="],check=True,capture_output=True,text=True).stdout
    if any(line.lstrip().startswith(("gz sim ","gz sim-server ")) for line in table.splitlines()):
        raise RuntimeError("Another Gazebo task is active; no processes were stopped")
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.image_pb2 import Image
    from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
    triangles = load_triangles(package)
    route = build_route(triangles,reverse=reverse)
    if not audit_route(triangles,route)["passed"]:
        raise ValueError("Target route mesh audit failed")
    identity_path = Path(manifest).resolve()
    identity = json.loads(identity_path.read_text(encoding="utf8"))
    if identity.get("method_kind") not in {"repository_model","external_model","reference_rule"} or not identity.get("method_id"):
        raise ValueError("Manifest requires method_id and method_kind")
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    for folder in ("public/images","private"):
        (output/folder).mkdir(parents=True)
    component = parse_component(target)
    tree, contract = prepare_scene(package,models,route,component=component,swapped=swapped,teammate="normal")
    tree.write(output/"world.sdf",encoding="utf8",xml_declaration=True)
    write_json(output/"scene_contract.private.json",contract)
    initial = planned_state(route,0.,swapped=swapped)["camera_poses_enu"]
    cameras = copy.deepcopy(initial)
    run_id = "subt-proxy-"+uuid.uuid4().hex
    summary = {"schema":"aerowam-subt-model-proxy/1","run_id":run_id,"method_id":identity["method_id"],
        "status":"simulator_error","success":None,"closed_loop":True,"kinematic_proxy":True,
        "flight_admitted":False,"px4_started":False,"formal_score":False,"independent_audit":False,
        "role_assignment":None,"message_delay_s":.5,"model_action_services_accepted":0,"samples":0,
        "world_model_score":False,"owned_processes_stopped":False,"cleanup_errors":[],
        "limitations":["camera proxies are not aircraft", "centralized joint public observations",
                       "privileged camera initialization only; no teacher camera updates",
                       "no changing teammate policy in this joint-policy diagnostic",
                       "no exposure-aligned flight action or dynamics evidence"]}
    write_json(output/"run.private.json",{"schema":summary["schema"],"package":str(Path(package).resolve()),
        "run_id":run_id,"target":target,"swapped":swapped,"reverse":reverse,"speed_m_s":.3,
        "duration_s":seconds,"model_manifest_sha256":sha256(identity_path),"world_sha256":sha256(output/"world.sdf"),
        "initial_camera_poses_enu":initial,
        "source_sha256":{name:sha256(Path(__file__).with_name(name)) for name in
            ("subt_handoff_proxy.py","subt_handoff_model_io.py","subt_handoff_evaluator.py")}})
    (output/"model_manifest.json").write_bytes(identity_path.read_bytes())
    partition = "aerowam_subt_model_"+uuid.uuid4().hex
    previous = os.environ.get("GZ_PARTITION")
    os.environ["GZ_PARTITION"] = partition
    node = Node()
    lock = threading.Lock()
    buffers, poses, topics, processes = {}, OrderedDict(), [], []
    evaluator = SubTHandoffEvaluator()
    origin = None
    history, pending = [], []
    last_pose_wall = time.monotonic()
    last_pose_stamp = last_frame = last_command = -1
    owned_log = None

    def receive_pose(message):
        stamp = message.header.stamp.sec*10**9+message.header.stamp.nsec
        values = {p.name:{"xyz":[p.position.x,p.position.y,p.position.z],
                          "q":[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]}
                  for p in message.pose if p.name in NAMES}
        if set(values)==set(NAMES):
            with lock:
                poses[stamp] = values
                while len(poses)>128:poses.popitem(last=False)

    try:
        topic=f"/world/{WORLD}/pose/info"
        if not node.subscribe(Pose_V,topic,receive_pose):raise RuntimeError("Pose subscription failed")
        topics.append(topic)
        for spec in contract["camera_pairs"]:
            for topic in (spec["rgb"],spec["segmentation"]+"/labels_map"):
                buffers[topic]=OrderedDict()
                def receive_image(message,key=topic):
                    stamp=message.header.stamp.sec*10**9+message.header.stamp.nsec
                    with lock:
                        buffers[key][stamp]=message.SerializeToString()
                        while len(buffers[key])>12:buffers[key].popitem(last=False)
                if not node.subscribe(Image,topic,receive_image):raise RuntimeError("Image subscription failed")
                topics.append(topic)
        env=dict(os.environ)
        env.pop("GZ_SIM_SERVER_CONFIG_PATH",None)
        owned_log=(output/"gazebo.log").open("xb")
        command=["gz","sim","-r","-v","3"]
        if not gui:command += ["-s","--headless-rendering"]
        processes.append(subprocess.Popen([*command,str(output/"world.sdf")],env=env,stdout=owned_log,
            stderr=subprocess.STDOUT,start_new_session=True))
        worker=subprocess.Popen([sys.executable,"-B","-u","-m",
            "benchmark.px4_gazebo.model_handoff.subt_handoff_collect","--worker"],env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=owned_log,text=True,bufsize=1,start_new_session=True)
        processes.append(worker)
        deadline=time.monotonic()+wall_timeout_s
        with (output/"io_trace.jsonl").open("xb") as trace:
            while time.monotonic()<deadline:
                if any(p.poll() is not None for p in processes):raise RuntimeError("Owned process exited")
                with lock:
                    latest=max(poses,default=-1)
                    common=set.intersection(*(set(v) for v in buffers.values()))
                    stamp=max(common,default=-1)
                    raw={key:value[stamp] for key,value in buffers.items()} if stamp>=0 else None
                    pose_stamp=max((s for s in poses if s<=stamp),default=-1)
                    actual=copy.deepcopy(poses.get(pose_stamp))
                if latest<last_pose_stamp:raise RuntimeError("Simulation clock regressed")
                if latest>last_pose_stamp:last_pose_stamp,last_pose_wall=latest,time.monotonic()
                if time.monotonic()-last_pose_wall > (45. if origin is None else 10.):
                    raise TimeoutError("Pose stream missing or stale")
                if origin is None and actual is not None:origin=stamp
                if origin is None:
                    time.sleep(.01)
                    continue
                elapsed=(latest-origin)/1e9
                if latest-last_command>=50_000_000:
                    desired=target_poses(route,elapsed,.3,cameras)
                    # Recheck the target as it moves toward stationary cameras.
                    zero={"actions":{a:{"delta_enu_m":[0.,0.,0.],"delta_yaw_rad":0.} for a in AGENTS}}
                    propose_cameras(cameras,zero,triangles,desired["handoff_target"][:3])
                    command_worker(worker,desired)
                    last_command=latest
                if stamp>last_frame and (last_frame<0 or stamp-last_frame>=500_000_000) and actual is not None and 0<=stamp-pose_stamp<=100_000_000:
                    paths, views = {}, {}
                    index=summary["samples"]
                    target_xyz=actual["handoff_target"]["xyz"]
                    for spec in contract["camera_pairs"]:
                        agent=spec["agent_id"]
                        rgb, labels=Image(),Image()
                        rgb.ParseFromString(raw[spec["rgb"]])
                        labels.ParseFromString(raw[spec["segmentation"]+"/labels_map"])
                        paired=pair_rgb_instance(rgb_message=rgb,label_message=labels,rgb_topic=spec["rgb"],
                            segmentation_topic=spec["segmentation"],label_topic=spec["segmentation"]+"/labels_map",encoding=GZ8_INSTANCE_ENCODING)
                        reduced=resize_pair(paired)
                        findings=visible_components(reduced.object_labels)[IDENTITIES[0]]
                        paths[agent]=f"images/{index:05d}_{agent}.jpg"
                        PILImage.fromarray(reduced.rgb).save(output/"public"/paths[agent],quality=95)
                        PILImage.fromarray(paired.object_labels).save(output/f"private/{index:05d}_{agent}_labels.png")
                        native_pixels=int(np.count_nonzero(paired.object_labels==11))
                        eye=actual[f"annotation_camera_{AGENTS.index(agent)}"]["xyz"]
                        views[agent]={"target_bbox_xyxy":findings["bbox"],
                            "fully_occluded":full_occlusion(triangles,eye,target_xyz,native_pixels)}
                    delivered=[m["event"] for m in pending if m["deliver_ns"]<=stamp]
                    row={"sample_index":index,"stamp_ns":stamp,"dt_s":None if last_frame<0 else (stamp-last_frame)/1e9,
                        "rgb":paths,"proprioception":{a:{"position_local_m":(np.asarray(actual[f"annotation_camera_{i}"]["xyz"])-initial[a][:3]).tolist(),
                              "quaternion_xyzw":actual[f"annotation_camera_{i}"]["q"]} for i,a in enumerate(AGENTS)}}
                    request=model_input(output,row,episode_id=run_id,
                        instruction=f"Track the {component.color} {component.shape}; hand off and confirm after the first observer loses sight.",
                        messages=delivered[-16:],history=history)
                    record={"sample_index":index,"request":request,"response":None,"latency_ms":None,
                        "error":None,"private":{"pose_stamp_ns":pose_stamp,"actual":actual,"views":views},
                        "command_elapsed_s":elapsed,"service_requested_poses_after_response":None,"service_accepted":False}
                    try:
                        started=time.perf_counter()
                        response=policy(copy.deepcopy(request))
                        record["latency_ms"]=(time.perf_counter()-started)*1000
                        canonical(response)
                        record["response"]=response
                        validate_output(response,request)
                        if record["latency_ms"]>400:raise TimeoutError("400ms model budget exceeded")
                        evaluator.step(stamp,views,response["events"])
                        proposals=propose_cameras(cameras,response,triangles,target_xyz)
                        desired=target_poses(route,elapsed,.3,proposals)
                        command_worker(worker,desired)
                        cameras=proposals
                        record["service_requested_poses_after_response"]=desired
                        record["service_accepted"]=True
                        summary["model_action_services_accepted"]+=1
                        history.append({"timestamp_ns":stamp,"actions":response["actions"]})
                        pending.extend({"deliver_ns":stamp+500_000_000,"event":event} for event in response["events"])
                    except Exception as exc:
                        record["error"]=f"{type(exc).__name__}: {exc}"
                        summary["status"]="safety_stop" if "proxy_" in str(exc) else "model_failure"
                    trace.write(canonical(record)+b"\n")
                    trace.flush()
                    summary["samples"]+=1
                    last_frame=stamp
                    if record["error"]:break
                    if (stamp-origin)/1e9>=seconds:
                        summary["status"]="complete"
                        break
                time.sleep(.005)
            else:raise TimeoutError("Bounded proxy run timed out")
    except Exception as exc:
        summary["error"]=f"{type(exc).__name__}: {exc}"
        if "proxy_" in str(exc):summary["status"]="safety_stop"
    finally:
        for process in reversed(processes):
            try:stop_owned(process)
            except Exception as exc:summary["cleanup_errors"].append(str(exc))
        for topic in topics:
            try:node.unsubscribe(topic)
            except Exception as exc:summary["cleanup_errors"].append(str(exc))
        if owned_log:owned_log.close()
        if previous is None:os.environ.pop("GZ_PARTITION",None)
        else:os.environ["GZ_PARTITION"]=previous
        summary["owned_processes_stopped"]=bool(processes) and all(p.poll() is not None for p in processes)
        write_json(output/"private/outcome.json",evaluator.result())
        summary["files_sha256"]={p.relative_to(output).as_posix():sha256(p)
            for p in output.rglob("*") if p.is_file() and p.name!="proxy_summary.json"}
        write_json(output/"proxy_summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("package","models","output","model-manifest"):
        parser.add_argument("--"+field,type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--adapter",help="MODULE:FUNCTION(core input) -> core output")
    group.add_argument("--policy-endpoint")
    parser.add_argument("--seconds",type=float,default=20.)
    parser.add_argument("--wall-timeout-s",type=float,default=300.)
    parser.add_argument("--target",default="red:sphere")
    parser.add_argument("--swapped",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    parser.add_argument("--gui",action="store_true")
    args=parser.parse_args()
    if args.adapter:
        module,function=args.adapter.split(":",1)
        policy=getattr(importlib.import_module(module),function)
    else:
        client=PolicyClient(args.policy_endpoint,timeout_s=.4)
        policy=lambda payload:client.request(canonical(payload))[0]
    result=run(args.package,args.models,args.output,manifest=args.model_manifest,policy=policy,
        seconds=args.seconds,target=args.target,swapped=args.swapped,reverse=args.reverse,gui=args.gui,
        wall_timeout_s=args.wall_timeout_s)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["status"]=="complete" and result["owned_processes_stopped"] and not result["cleanup_errors"] else 2)


if __name__=="__main__":main()
