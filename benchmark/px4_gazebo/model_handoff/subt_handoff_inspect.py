"""File and causal-history audit; not flight or learned-policy evaluation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .industrial_targets import IDENTITIES, color_support, parse_component, visible_components
from .subt_handoff_expert import EventExpert, duration, planned_state
from .subt_handoff_route import AGENTS
from .subt_handoff_collect import visibility_stages, poses_for
from .warehouse_scene import sha256


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line]


def safe_image(root, relative):
    if not isinstance(relative,str) or not relative.startswith('images/') or chr(92) in relative:
        raise ValueError('Only episode-local public images allowed')
    path=(Path(root)/'public'/relative).resolve()
    if not path.is_relative_to((Path(root)/'public/images').resolve()) or not path.is_file():
        raise ValueError('Image path escape or missing image')
    return path


def validate_public(row):
    expected={'sample_index','stamp_ns','dt_s','rgb','rgb_sha256','proprioception','messages'}
    if set(row)!=expected or set(row['rgb'])!=set(AGENTS) or set(row['rgb_sha256'])!=set(AGENTS):
        raise ValueError('Unexpected policy-row fields or agent inventory')
    if set(row['proprioception'])!=set(AGENTS):raise ValueError('Both pose streams required')
    for values in row['proprioception'].values():
        if set(values)!={'position_local_m','quaternion_xyzw'}:
            raise ValueError('Private fields in proprioception')
        if len(values['position_local_m'])!=3 or len(values['quaternion_xyzw'])!=4:
            raise ValueError('Invalid pose dimensions')
        if not all(math.isfinite(x) for key in values for x in values[key]):raise ValueError('Nonfinite pose')
        if abs(np.linalg.norm(values['quaternion_xyzw'])-1.)>.001:raise ValueError('Unnormalized quaternion')
    for message in row['messages']:
        if set(message)!={'agent_id','event'} or message['agent_id'] not in AGENTS:
            raise ValueError('Private or malformed communication')


def validate_causal_rows(public, supervision, private, *, delay_s):
    if not (len(public)==len(supervision)==len(private)>2):raise ValueError('Incomplete row inventory')
    expert=EventExpert(delay_s)
    messages=[]
    previous=None
    for index,(observation,label,truth) in enumerate(zip(public,supervision,private)):
        validate_public(observation)
        if any(type(row['sample_index']) is not int or row['sample_index']!=index for row in (observation,label,truth)):
            raise ValueError('Contiguous integer sample indices required')
        stamp=observation['stamp_ns']
        if (type(stamp) is not int or stamp<0 or stamp!=label['stamp_ns'] or stamp!=truth['stamp_ns']
                or previous is not None and stamp<=previous):raise ValueError('Timestamp alignment/reversal')
        elapsed=truth['elapsed_s']
        if abs(elapsed-(stamp-public[0]['stamp_ns'])/1e9)>1e-7:raise ValueError('Elapsed clock mismatch')
        dt=None if previous is None else (stamp-previous)/1e9
        if observation['dt_s']!=dt:raise ValueError('Recorded dt mismatch')
        if dt is not None and not .49<=dt<=1.:raise ValueError('Image gap or frame duplication')
        if observation['messages']!=[m['claim'] for m in messages if m['deliver_s']<=elapsed]:
            raise ValueError('Noncausal, missing or altered communication history')
        if label['flight_control_label'] is not False:raise ValueError('Proxy waypoints cannot label flight controls')
        events=expert.step(elapsed,truth['visible'])
        if events!=label['events']:raise ValueError('Expert event replay mismatch')
        messages.extend({'deliver_s':elapsed+delay_s,'claim':e} for e in events)
        previous=stamp
    return expert


def contact_sheet(root, public, private, output):
    root=Path(root)
    indices=sorted(set([0,len(public)//4,len(public)//2,3*len(public)//4,len(public)-1]))
    sheet=Image.new('RGB',(768,len(indices)*310),'#f5f5f5')
    draw=ImageDraw.Draw(sheet)
    for y,index in enumerate(indices):
        for x,agent in enumerate(AGENTS):
            with Image.open(safe_image(root,public[index]['rgb'][agent])) as frame:
                sheet.paste(frame.resize((384,288)),(x*384,y*310+22))
            visible=private[index]['visible'][agent]
            draw.text((x*384+6,y*310+5),f'{agent} / {private[index]["elapsed_s"]:.1f}s / visible={visible}',fill='black')
    sheet.save(output)


def inspect(root, *, component=None, delay_s=None):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text(encoding='utf8'))
    metadata=json.loads((root/'public/metadata.json').read_text(encoding='utf8'))
    condition=summary.get('condition_private')
    legacy=condition is None
    if legacy and (component is None or delay_s is None):
        raise ValueError('Legacy capture requires explicit component and message delay')
    if condition is not None:
        actual=parse_component(condition['target']['color']+':'+condition['target']['shape'])
        if component is not None and component!=actual:raise ValueError('Component override refused')
        component=actual
        if delay_s is not None and delay_s!=condition['message_delay_s']:raise ValueError('Delay override refused')
        delay_s=condition['message_delay_s']
    required=('passed','dataset_admitted','owned_processes_stopped','expert_complete')
    if not all(summary.get(k) is True for k in required) or summary.get('cleanup_errors'):
        raise ValueError('Collection or cleanup acceptance missing')
    if any(summary.get(k) is not False for k in ('px4_started','flight_admitted','model_evaluated','formal_score')):
        raise ValueError('Nonflying development-only provenance required')
    split=metadata.get('split')
    if metadata.get('role_assignment') is not None or split not in {
            'development_training_only','held_out_route_test'}:
        raise ValueError('Role-unassigned development or held-out route data required')
    if not summary['geometry']['passed'] or not summary['expert_audit']['passed']:raise ValueError('Geometry audit failed')
    contract=json.loads((root/'scene_contract.private.json').read_text(encoding='utf8'))
    if sha256(root/'world.sdf')!=contract['world_sha256']:raise ValueError('World fingerprint changed')
    route=json.loads((root/'route.private.json').read_text(encoding='utf8'))
    if condition is not None:
        expected_route='stair_descent_v1' if condition['reverse'] else 'stair_ascent_v1'
        if (condition.get('split', 'development_training_only')!=split
                or condition.get('route_id', expected_route)!=expected_route
                or metadata.get('route_id', expected_route)!=expected_route
                or route.get('reverse') is not condition['reverse']
                or (split=='held_out_route_test')!=condition['reverse']):
            raise ValueError('Route identity and data partition disagree')
    public=read_rows(root/'public/observations.jsonl')
    supervision=read_rows(root/'supervision/labels.jsonl')
    private=read_rows(root/'private/annotations.jsonl')
    expert=validate_causal_rows(public,supervision,private,delay_s=delay_s)
    if not expert.complete or expert.owner!=summary['initial_owner'] or expert.receiver!=summary['receiver']:
        raise ValueError('Independent causal handoff replay did not complete')
    if summary['samples']!=len(public):raise ValueError('Summary row count mismatch')
    stages=visibility_stages(private,expert.owner)
    if not stages['passed']:raise ValueError('Rendered ordered visibility missing')
    image_hashes={}
    for observation,label,truth in zip(public,supervision,private):
        i=observation['sample_index']
        if not 0<=truth['stamp_ns']-truth['pose_stamp_ns']<=100_000_000:raise ValueError('Stale pose alignment')
        if truth['path_error_m']>.15:raise ValueError('Path mismatch')
        for agent in AGENTS:
            path=safe_image(root,observation['rgb'][agent])
            if sha256(path)!=observation['rgb_sha256'][agent]:raise ValueError('RGB digest mismatch')
            image_hashes[path.relative_to(root).as_posix()]=sha256(path)
            with Image.open(path) as frame:
                if frame.size!=(256,256) or frame.mode!='RGB' or frame.format!='JPEG':raise ValueError('JPEG format mismatch')
                if metadata.get('rgb_storage') is not None:
                    from .subt_handoff_collect import RGB_STORAGE
                    from PIL.JpegImagePlugin import get_sampling
                    if metadata['rgb_storage']!=RGB_STORAGE or get_sampling(frame)!=0:
                        raise ValueError('Declared JPEG 4:4:4 storage contract mismatch')
                rgb=np.asarray(frame)
            label_path=root/f'private/{i:05d}_{agent}_labels.png'
            image_hashes[label_path.relative_to(root).as_posix()]=sha256(label_path)
            with Image.open(label_path) as mask_frame:labels=np.asarray(mask_frame)
            if labels.shape!=(256,256):raise ValueError('Mask size mismatch')
            findings=visible_components(labels)[IDENTITIES[0]]
            if findings['pixels']!=truth['annotations'][agent]['pixels']:raise ValueError('Mask count mismatch')
            if (findings['bbox'] is not None)!=truth['visible'][agent]:raise ValueError('Visibility unsupported')
            if findings['bbox'] is not None:
                support=int(np.count_nonzero((labels==11)&color_support(rgb,component.color)))
                # Tiny edge-of-occlusion masks can yield only 1-3 color pixels;
                # bbox validity already gates detection, so keep the same 15%
                # color fraction without a second absolute four-pixel floor.
                if support<max(1,int(findings['pixels']*.15)):
                    from PIL import __version__ as pillow_version, features
                    raise ValueError(f'No declared RGB color support: sample={i}, agent={agent}, '
                        f'color={component.color}, support={support}, mask_pixels={findings["pixels"]}, '
                        f'required={max(1,int(findings["pixels"]*.15))}, '
                        f'Pillow={pillow_version}, JPEG={features.version("jpg")}')
        if condition is not None:
            t=truth['elapsed_s']
            plan=planned_state(route,t,speed=condition['speed_m_s'],swapped=condition['swapped'],teammate=condition['teammate'])
            expected=poses_for(plan)
            actual=truth['actual']
            error=max(math.dist(actual[n]['xyz'],expected[n][:3]) for n in expected)
            if abs(error-truth['path_error_m'])>1e-6:raise ValueError('Path error recomputation differs')
            after=planned_state(route,t+.5,speed=condition['speed_m_s'],swapped=condition['swapped'],teammate=condition['teammate'])
            for j,agent in enumerate(AGENTS):
                delta=np.array(after['camera_poses_enu'][agent][:3])-actual[f'annotation_camera_{j}']['xyz']
                if not np.allclose(delta,label['oracle_camera_waypoint_delta_enu_m'][agent],atol=1e-7,rtol=0):
                    raise ValueError('Waypoint label mismatch')
    heights=[r['actual']['handoff_target']['xyz'][2] for r in private]
    if max(heights)-min(heights)<9.:raise ValueError('Both floors not covered')
    commands_verified=0
    if not legacy:
        if private[-1]['elapsed_s']<duration(route,condition['speed_m_s']):raise ValueError('Trajectory incomplete')
        commands=read_rows(root/'private/pose_commands.jsonl')
        previous=-1
        for command in commands:
            stamp=command['command_observed_sim_stamp_ns']
            if stamp<=previous or command['service_accepted'] is not True:raise ValueError('Command clock invalid')
            if command['verified_actuation_at_image_exposure'] is not False:raise ValueError('Unsupported actuation claim')
            plan=planned_state(route,command['elapsed_sim_s'],speed=condition['speed_m_s'],swapped=condition['swapped'],teammate=condition['teammate'])
            expected=poses_for(plan)
            actual=command['requested_poses_enu']
            if set(expected)!=set(actual) or any(not np.allclose(expected[k],actual[k],atol=1e-10,rtol=0) for k in expected):
                raise ValueError('Pose request differs from plan')
            previous=stamp
        commands_verified=len(commands)
        if commands_verified<50:raise ValueError('Missing command log')
    inventory={name:sha256(root/name) for name in ('summary.json','route.private.json','world.sdf',
        'scene_contract.private.json','public/metadata.json','public/observations.jsonl',
        'supervision/labels.jsonl','private/annotations.jsonl')}
    inventory.update(image_hashes)
    if not legacy:inventory['private/pose_commands.jsonl']=sha256(root/'private/pose_commands.jsonl')
    report={'schema':'aerowam-subt-stair-data-inspection/1','passed':True,'samples':len(public),
        'images_verified':len(public)*2,'commands_verified':commands_verified,'source_capture_legacy':legacy,
        'condition':condition,'component':{'color':component.color,'shape':component.shape},'message_delay_s':delay_s,
        'message_delay_selection':None if legacy else condition.get('message_delay_selection','legacy_motion_default'),
        'causal_expert_replay':True,'ordered_rendered_visibility':stages['selected'],
        'vertical_span_m':max(heights)-min(heights),'files_sha256':inventory,
        'training_scope':'RGB_and_causal_joint_event_supervision_only',
        'motion_supervision_scope':'privileged_desired_camera_pose_not_executed_policy_action',
        'world_model_scope':'kinematic_observation_sequence_not_action_conditioned_UAV_dynamics',
        'split':split,'route_id':metadata.get('route_id',
            condition.get('route_id', 'stair_descent_v1' if condition.get('reverse') else 'stair_ascent_v1')
            if condition is not None else 'legacy_unresolved'),
        'same_scene_family_as_development':True,
        'decentralized_policy_evaluated':False,'model_score_eligible':False,'flight_admitted':False,
        'limitations':['oracle_camera_points_at_hidden_target','proxy_not_PX4_flight','same_scene_family_no_independent_scene',
            'sampled_preceding_pose_not_exact_exposure_actuation','no_contact_fault_validation']}
    if legacy:report['limitations'].append('legacy_capture_no_source_hash_or_pose_command_log')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--target',type=parse_component)
    parser.add_argument('--message-delay-s',type=float)
    parser.add_argument('--read-only',action='store_true',
                        help='audit without rewriting the episode report or contact sheet')
    args=parser.parse_args(argv)
    report=inspect(args.episode,component=args.target,delay_s=args.message_delay_s)
    if not args.read_only:
        contact_sheet(args.episode,read_rows(args.episode/'public/observations.jsonl'),
                      read_rows(args.episode/'private/annotations.jsonl'),args.episode/'handoff_contact_sheet.png')
        (args.episode/'inspection.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'}))


if __name__=='__main__':main()
