"""Frozen development and held-out-route capture schedules, not a model score.

One episode per condition is coverage, not independent replication. The held-out
route reverses direction in the same scene family. This command only writes a
plan; collection and independent inspection remain explicit steps.
"""

import argparse
import itertools
import json
from pathlib import Path
import random


def _conditions(seed, *, reverse):
    conditions=[]
    for target,swapped,motion,delay in itertools.product(
            ("red:sphere","blue:cube"),(False,True),("normal","delayed"),(.5,)):
        name=f"{target.replace(':','_')}_id{int(swapped)}_{motion}_comm{delay:g}"
        conditions.append({"condition_id":name,"target":target,"swapped":swapped,
                           "teammate":motion,"message_delay_s":delay,"reverse":reverse})
    random.Random(seed).shuffle(conditions)
    return conditions


def design(seed=20261021):
    if type(seed) is not int:
        raise ValueError("Integer randomization seed required")
    conditions=_conditions(seed,reverse=False)
    test_conditions=_conditions(seed+1,reverse=True)
    return {"schema":"aerowam-subt-stair-development-design/2","seed":seed,
            "research_question":"Can dual-agent cooperation skills and world-model prediction compose and transfer to new tasks under partial observability and changing teammate behavior?",
            "communication_effects_evaluated":False,
            "controlled_constants":{"message_delay_s":.5},
            "randomization":"seeded_shuffle_within_each_complete_scene_route_block",
            "unit":"whole_episode_not_frame","scene_family":"subt_urban_2_story_stair_v1",
            "target_speed_m_s":.3,"role_assignment":None,
            "episodes_per_condition":1,"planned_episodes":len(conditions)+len(test_conditions),
            "development":{"split":"development_training_only","route_id":"stair_ascent_v1",
                "route_direction":"ascent","episodes":len(conditions),
                "conditions":[{"run_order":i+1,"route_id":"stair_ascent_v1",
                    "split":"development_training_only",**row} for i,row in enumerate(conditions)]},
            "held_out_route_test":{"split":"held_out_route_test","route_id":"stair_descent_v1",
                "route_direction":"descent","episodes":len(test_conditions),
                "independence":"direction-held-out task on same scene family; not an independent scene",
                "conditions":[{"run_order":i+1,"route_id":"stair_descent_v1",
                    "split":"held_out_route_test",**row} for i,row in enumerate(test_conditions)]},
            "factors":{"target":["red:sphere","blue:cube"],"swapped":[False,True],
                       "teammate":["normal","delayed"]},
            "admission":"each_episode_requires_mesh_rendering_causal_and_cleanup_inspection",
            "model_evaluated":False,"flight_admitted":False,"test_partition_created":True,
            "limitations":["one_scene_family_no_independent_scene_replication",
                           "target_color_and_shape_are_bundled_not_independently_varied",
                           "privileged_look_at_and_joint_event_teacher",
                           "held_out_route_is_direction_reversal_not_new_geometry",
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
