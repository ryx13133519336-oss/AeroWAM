"""Export inspected stair demonstrations for event BC / observation prediction.

Deliberately excludes oracle camera waypoints from learning targets: those are
desired poses, not logged controls which caused an independently observed next
state. All sources stay development-training-only, in one scene-family group.
"""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil

from .subt_handoff_inspect import inspect, read_rows, safe_image
from .industrial_targets import parse_component
from .subt_handoff_route import AGENTS
from .warehouse_scene import sha256


def condition_coverage(reports):
    """Offline provenance only; frames are not independent condition replicates."""
    counts=Counter()
    unresolved=0
    for report in reports:
        condition=report.get('condition')
        if condition is None:
            unresolved+=1
            continue
        target=condition['target']['color']+':'+condition['target']['shape']
        counts[(target,condition['swapped'],condition['teammate'],condition['message_delay_s'])]+=1
    expected=set(itertools.product(('red:sphere','blue:cube'),(False,True),('normal','delayed'),(.5,2.)))
    return {'unit':'episode_not_frame','recorded_condition_episodes':sum(counts.values()),
            'unresolved_legacy_episodes':unresolved,
            'counts':[{'target':key[0],'swapped':key[1],'teammate':key[2],
                       'message_delay_s':key[3],'episodes':count} for key,count in sorted(counts.items())],
            'complete_2x2x2x2_coverage':expected.issubset(counts),
            'coverage_is_not_independent_replication_or_model_performance':True}


def learning_rows(public, labels, instruction, group, image_paths):
    """Never expose current/future labels to observation features."""
    inputs, targets, futures, groups = [], [], [], []
    if len(public)!=len(labels) or len(public)!=len(image_paths):
        raise ValueError('Aligned episode rows and images required')
    for index,(row,label) in enumerate(zip(public,labels)):
        if row['sample_index']!=index or label['sample_index']!=index:
            raise ValueError('Aligned contiguous row indices required')
        key=f'{group}_{index:05d}'
        inputs.append({'key':key,'observation':{
            'instruction':instruction,'rgb':image_paths[index],
            'proprioception':row['proprioception'],'messages':row['messages'],
            'dt_s':row['dt_s']}})
        events=dict.fromkeys(AGENTS,'none')
        for report in label['events']:
            if report['agent_id'] not in AGENTS or events[report['agent_id']]!='none':
                raise ValueError('Malformed/duplicate agent event')
            events[report['agent_id']]=report['event']
        targets.append({'key':key,'joint_event':events})
        groups.append({'key':key,'scene_family_id':'subt_urban_2_story_stair_v1',
                       'parent_episode_id':group,'sample_index':index,
                       'split':'development_training_only'})
        if index+1<len(public):
            dt=(public[index+1]['stamp_ns']-row['stamp_ns'])/1e9
            if not 0<dt<=1.:raise ValueError('Prediction pair crosses capture gap')
            futures.append({'key':key,'next_key':f'{group}_{index+1:05d}',
                'dt_s':dt,'next_rgb':image_paths[index+1],
                'scope':'observation_prediction_only_no_causal_action_conditioning'})
    return inputs,targets,futures,groups


def export(episodes,output):
    episodes=[Path(p).resolve() for p in episodes]
    if not episodes or len(set(episodes))!=len(episodes):raise ValueError('Distinct episode directories required')
    prepared=[]
    for episode in episodes:
        frozen=json.loads((episode/'inspection.json').read_text(encoding='utf8'))
        if not frozen.get('passed'):raise ValueError('Independent inspection must pass')
        # Recompute, do not trust a bare success flag or stale file inventory.
        component=parse_component(frozen['component']['color']+':'+frozen['component']['shape'])
        current=inspect(episode,component=component,delay_s=frozen['message_delay_s'])
        if current['files_sha256']!=frozen['files_sha256']:raise ValueError('Capture changed since independent inspection')
        prepared.append((episode,current,read_rows(episode/'public/observations.jsonl'),
                         read_rows(episode/'supervision/labels.jsonl'),
                         json.loads((episode/'public/metadata.json').read_text(encoding='utf8'))))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'images').mkdir()
    all_inputs,all_targets,all_futures,all_groups=[],[],[],[]
    sources=[]
    image_hashes={}
    for number,(episode,report,public,labels,metadata) in enumerate(prepared):
        group=f'episode_{number:03d}'
        image_paths=[]
        for row in public:
            paths={}
            for agent in AGENTS:
                relative=f"images/{group}_{row['sample_index']:05d}_{agent}.jpg"
                source=safe_image(episode,row['rgb'][agent])
                shutil.copyfile(source,output/relative)
                digest=sha256(output/relative)
                if digest!=row['rgb_sha256'][agent]:raise ValueError('Image changed while exporting')
                image_hashes[relative]=digest
                paths[agent]=relative
            image_paths.append(paths)
        values=learning_rows(public,labels,metadata['instruction'],group,image_paths)
        for collection,rows in zip((all_inputs,all_targets,all_futures,all_groups),values):collection.extend(rows)
        sources.append({'parent_episode_id':group,'source_path':str(episode),
            'inspection_sha256':sha256(episode/'inspection.json'),
            'inspection_files_sha256':report['files_sha256'],
            'samples':len(public),'legacy_capture':report['source_capture_legacy'],
            'condition_private':report['condition'],
            'limitations':report['limitations']})
    tables={'inputs.jsonl':all_inputs,'event_supervision.jsonl':all_targets,
            'future_observation_supervision.jsonl':all_futures,'groups.jsonl':all_groups}
    for name,rows in tables.items():
        with (output/name).open('x',encoding='utf8') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    design={'schema':'aerowam-subt-stair-training-export/1',
        'split':'development_training_only','scene_families':1,'parent_episodes':len(episodes),
        'samples':len(all_inputs),'images':len(image_hashes),'prediction_pairs':len(all_futures),
        'role_assignment':None,'sources':sources,
        'condition_coverage':condition_coverage([item[1] for item in prepared]),
        'input_features':['instruction','rgb','proprioception','messages','dt_s'],
        'excluded_from_inputs':['target_pose','instance_masks','route_progress','private_visibility','future_frames','current_labels'],
        'BC_scope':'joint_event_prediction_with_privileged_teacher_not_decentralized_policy',
        'world_model_scope':'observation_sequence_prediction_not_control_conditioned_dynamics',
        'motion_supervision_exported':False,'model_evaluated':False,'flight_admitted':False,
        'test_partition_created':False,'compositional_transfer_established':False,
        'independence':'frames_and_color_or_id_variants_of_one_scene_are_not_independent_scene_replicates',
        'limitations':['oracle_look_at_uses_hidden_target','no_action_conditioned_counterfactual_branches',
            'no_actual_PX4_trajectory','no_independent_test_scene',
            'no_learned_teammate_adaptation_or_transfer_evaluation'],
        'files_sha256':{name:sha256(output/name) for name in tables},'image_sha256':image_hashes}
    (output/'design.json').write_text(json.dumps(design,indent=2,allow_nan=False),encoding='utf8')
    result={'samples':len(all_inputs),'images':len(image_hashes),'prediction_pairs':len(all_futures),
            'parent_episodes':len(episodes),'scene_families':1,'design_sha256':sha256(output/'design.json'),
            'split':'development_training_only','model_score_eligible':False,'flight_admitted':False}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.episodes,args.output_dir)))
