"""Frozen 2x2x2x2 development capture schedule, not a model evaluation suite.

One episode per condition is coverage, not independent replication. This command
only writes a plan; collection and independent inspection remain explicit steps.
"""

import argparse
import itertools
import json
from pathlib import Path
import random


def design(seed=20261021):
    if type(seed) is not int:
        raise ValueError("Integer randomization seed required")
    conditions=[]
    for target,swapped,motion,delay in itertools.product(
            ("red:sphere","blue:cube"),(False,True),("normal","delayed"),(.5,2.)):
        name=f"{target.replace(':','_')}_id{int(swapped)}_{motion}_comm{delay:g}"
        conditions.append({"condition_id":name,"target":target,"swapped":swapped,
                           "teammate":motion,"message_delay_s":delay})
    random.Random(seed).shuffle(conditions)
    return {"schema":"aerowam-subt-stair-development-design/1","seed":seed,
            "randomization":"seeded_shuffle_within_one_complete_scene_route_block",
            "unit":"whole_episode_not_frame","scene_family":"subt_urban_2_story_stair_v1",
            "route_direction":"ascent","target_speed_m_s":.3,
            "split":"development_training_only","role_assignment":None,
            "episodes_per_condition":1,"planned_episodes":len(conditions),
            "factors":{"target":["red:sphere","blue:cube"],"swapped":[False,True],
                       "teammate":["normal","delayed"],"message_delay_s":[.5,2.]},
            "conditions":[{"run_order":i+1,**row} for i,row in enumerate(conditions)],
            "admission":"each_episode_requires_mesh_rendering_causal_and_cleanup_inspection",
            "model_evaluated":False,"flight_admitted":False,"test_partition_created":False,
            "limitations":["one_scene_route_block_no_independent_scene_replication",
                           "target_color_and_shape_are_bundled_not_independently_varied",
                           "privileged_look_at_and_joint_event_teacher",
                           "no_inferential_effect_or_compositional_transfer_claim"]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20261021)
    args=parser.parse_args(argv)
    plan=design(args.seed)
    with args.output.open("x",encoding="utf8") as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
    print(json.dumps({"output":str(args.output),"planned_episodes":plan["planned_episodes"],
                      "collected_episodes":0,"seed":args.seed}))


if __name__=="__main__":main()
