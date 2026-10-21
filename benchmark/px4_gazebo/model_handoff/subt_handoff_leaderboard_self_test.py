"""Synthetic tests for the SubT leaderboard evidence gates."""

from __future__ import annotations

from .subt_handoff_leaderboard import aggregate, validate_report


def report(method, *, success=True, proxy=False, status="complete", repeat="r0"):
    return {
        "schema": "aerowam-subt-handoff-evaluation/1", "benchmark_version": "subt-stair-v1",
        "method_id": method, "method_kind": "repository_model", "model_manifest_sha256": "a" * 64,
        "task_id": "handoff", "scene_id": "subt_urban_stair_v1", "scene_family_id": "subt_urban_2_story",
        "split": "held_out_route_test", "episode_id": f"ep-{method}-{repeat}",
        "condition": {"condition_id": "red_sphere_id0_normal", "target": "red:sphere", "swapped": False,
                      "teammate": "normal", "message_delay_s": 0.5, "route_id": "stair_descent_v1", "repeat_id": repeat},
        "status": status, "success": success if status == "complete" else None,
        "metrics": {"latency_ms": 30.0}, "evidence": {
            "closed_loop": True, "public_input_only": True, "independent_audit": True,
            "flight_admitted": not proxy, "privileged_teacher": proxy, "kinematic_proxy": proxy,
            "action_trace_verified": not proxy, "role_assignment": None, "external_model": True,
            "audit_sha256": "b" * 64, "trace_sha256": "c" * 64,
            "public_request_sha256": "d" * 64, "private_outcome_sha256": "e" * 64,
        }, "limitations": ["synthetic self-test"],
    }


def run():
    good = report("model_a")
    proxy = report("proxy_reference", proxy=True)
    assert validate_report(good)["_model_score_eligible"] is True
    assert validate_report(proxy)["_model_score_eligible"] is False
    result = aggregate([good, proxy])
    assert result["formal_score"] is True
    assert result["ranking"][0]["method_id"] == "model_a"
    assert result["all_methods"][1]["ranked"] is False
    failed = report("model_b", status="simulator_error")
    result = aggregate([failed])
    assert result["formal_score"] is False and result["all_methods"][0]["success_rate"] is None
    print({"passed": True, "formal_ranked": 1, "proxy_ranked": 0})


if __name__ == "__main__":
    run()
