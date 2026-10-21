"""Independent, read-only audit of Gazebo model-camera proxy diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core._common import canonical
from .industrial_targets import IDENTITIES, visible_components
from .subt_handoff_evaluator import SubTHandoffEvaluator
from .subt_handoff_inspect import read_rows
from .subt_handoff_model_io import model_input, validate_output
from .subt_handoff_proxy import full_occlusion, propose_cameras, target_poses
from .subt_handoff_route import AGENTS, build_route
from .subt_urban_geometry import load_triangles
from .warehouse_scene import sha256


def audit(root):
    root=Path(root).resolve()
    saved=json.loads((root/"proxy_summary.json").read_text(encoding="utf8"))
    config=json.loads((root/"run.private.json").read_text(encoding="utf8"))
    if saved["schema"]!="aerowam-subt-model-proxy/1" or config["run_id"]!=saved["run_id"]:
        raise ValueError("Proxy identity mismatch")
    for field in ("flight_admitted","px4_started","formal_score","independent_audit","world_model_score"):
        if saved[field] is not False:raise ValueError("Proxy cannot claim flight or formal scores")
    if saved["closed_loop"] is not True or saved["kinematic_proxy"] is not True:
        raise ValueError("Proxy mode declaration mismatch")
    if saved["success"] is not None or saved["role_assignment"] is not None or saved["message_delay_s"]!=.5:
        raise ValueError("Proxy summary evidence mismatch")
    if saved["status"]!="complete" or saved["owned_processes_stopped"] is not True or saved["cleanup_errors"]:
        raise ValueError("Only completed, cleaned proxy runs are admitted to diagnostic audit")
    for relative,digest in saved["files_sha256"].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or sha256(path)!=digest:raise ValueError("Proxy artifact hash mismatch")
    for name,digest in config["source_sha256"].items():
        if name not in {"subt_handoff_proxy.py","subt_handoff_model_io.py","subt_handoff_evaluator.py"} or sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError("Proxy source changed")
    if sha256(root/"world.sdf")!=config["world_sha256"] or sha256(root/"model_manifest.json")!=config["model_manifest_sha256"]:
        raise ValueError("Proxy world/model identity mismatch")
    triangles=load_triangles(config["package"])
    route=build_route(triangles,reverse=config["reverse"])
    records=read_rows(root/"io_trace.jsonl")
    if len(records)!=saved["samples"] or not records:raise ValueError("Missing model I/O records")
    evaluator=SubTHandoffEvaluator()
    initial=config["initial_camera_poses_enu"]
    cameras=initial
    history,pending=[],[]
    previous=None
    origin=records[0]["request"]["timestamp_ns"]
    public_root=root
    for index,record in enumerate(records):
        request=record["request"]
        stamp=request["timestamp_ns"]
        if record["sample_index"]!=index or previous is not None and not 490_000_000<=stamp-previous<=1_000_000_000:
            raise ValueError("Proxy frame sequence/gap invalid")
        truth=record["private"]
        if not 0<=stamp-truth["pose_stamp_ns"]<=100_000_000:raise ValueError("Stale pose alignment")
        actual=truth["actual"]
        views={}
        paths={}
        for i,agent in enumerate(AGENTS):
            paths[agent]=f"images/{index:05d}_{agent}.jpg"
            with Image.open(root/f"private/{index:05d}_{agent}_labels.png") as native:
                labels=np.asarray(native)
                resized=np.asarray(native.resize((256,256),Image.Resampling.NEAREST))
            found=visible_components(resized)[IDENTITIES[0]]
            views[agent]={"target_bbox_xyxy":found["bbox"],"fully_occluded":full_occlusion(
                triangles,actual[f"annotation_camera_{i}"]["xyz"],actual["handoff_target"]["xyz"],int(np.count_nonzero(labels==11)))}
        if canonical(views)!=canonical(truth["views"]):raise ValueError("Private visibility recomputation differs")
        row={"sample_index":index,"stamp_ns":stamp,"dt_s":None if previous is None else (stamp-previous)/1e9,
            "rgb":paths,"proprioception":{a:{"position_local_m":(np.asarray(actual[f"annotation_camera_{i}"]["xyz"])-initial[a][:3]).tolist(),
                "quaternion_xyzw":actual[f"annotation_camera_{i}"]["q"]} for i,a in enumerate(AGENTS)}}
        color,shape=config["target"].split(":")
        expected=model_input(public_root,row,episode_id=config["run_id"],
            instruction=f"Track the {color} {shape}; hand off and confirm after the first observer loses sight.",
            messages=[m["event"] for m in pending if m["deliver_ns"]<=stamp][-16:],history=history)
        if canonical(expected)!=canonical(request):raise ValueError("Public request reconstruction differs")
        response=validate_output(record["response"],request)
        if record["error"] is not None or record["service_accepted"] is not True or not 0<=record["latency_ms"]<=400:
            raise ValueError("Response/action service/budget failed")
        evaluator.step(stamp,views,response["events"])
        proposals=propose_cameras(cameras,response,triangles,actual["handoff_target"]["xyz"])
        desired=record["service_requested_poses_after_response"]
        expected_poses=target_poses(route,record["command_elapsed_s"],config["speed_m_s"],proposals)
        if canonical(expected_poses)!=canonical(desired):raise ValueError("Target/model pose command differs")
        for i,agent in enumerate(AGENTS):
            if desired[f"annotation_camera_{i}"]!=proposals[agent]:raise ValueError("Model camera action not traced")
        cameras=proposals
        pending.extend({"deliver_ns":stamp+500_000_000,"event":e} for e in response["events"])
        history.append({"timestamp_ns":stamp,"actions":response["actions"]})
        previous=stamp
    if (previous-origin)/1e9 < config["duration_s"] or saved["model_action_services_accepted"]!=len(records):
        raise ValueError("Proxy duration/action count differs")
    outcome=json.loads((root/"private/outcome.json").read_text(encoding="utf8"))
    if canonical(evaluator.result())!=canonical(outcome):raise ValueError("Private event replay differs")
    return {"schema":"aerowam-subt-model-proxy-audit/1","passed":True,"run_id":config["run_id"],
        "frames_verified":len(records),"model_actions_traced":len(records),"proxy_event_success":outcome["success"],
        "kinematic_proxy":True,"flight_admitted":False,"formal_score":False,
        "limitations":["service acceptance is not exposure-aligned actuation",
                       "no PX4 telemetry or flight admission", "privileged initial camera placement"],
        "summary_sha256":sha256(root/"proxy_summary.json")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.run_dir)
    with args.output.open("xb") as stream:stream.write(canonical(result)+b"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
