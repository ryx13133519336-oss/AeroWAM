"""Regression checks for WAM v2 and event-derived identity metrics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import torch

from .models import ModelConfig, build_model
from .train import _loss, FeatureSequenceDataset


def main() -> None:
    torch.manual_seed(7)
    config = ModelConfig(method="dual_wam_mpc", feature_dim=16, model_dim=32, heads=4, wam_revision="expert_prior_v2")
    model = build_model(config).eval()
    batch = {
        "visual": torch.randn(2, 4, 2, 16), "text": torch.randn(2, 16),
        "state": torch.randn(2, 4, 2, 8), "previous_action": torch.randn(2, 4, 2, 4),
        "action": torch.randn(2, 2, 4).tanh(), "intent": torch.tensor([1, 2]), "phase": torch.tensor([1, 3]),
        "next_visual": torch.randn(2, 2, 16), "next_state": torch.randn(2, 2, 8),
        "next_previous_action": torch.randn(2, 2, 4), "transition_valid": torch.tensor([1., 0.]),
        "next_visibility": torch.ones(2), "next_handoff": torch.zeros(2), "next_collision": torch.zeros(2),
    }
    first = model(batch, transition_action=torch.zeros(2, 2, 4))
    second = model(batch, transition_action=torch.ones(2, 2, 4))
    assert torch.equal(first["action"], second["action"])
    assert not torch.allclose(first["next_latent"], second["next_latent"])
    history_changed = deepcopy(batch)
    history_changed["visual"][:, 0] += 3
    assert not torch.allclose(model.encode(batch), model.encode(history_changed))
    baseline_loss = _loss(model, batch, config.method)
    masked = deepcopy(batch)
    for name in ("next_visual", "next_state", "next_previous_action"):
        masked[name][1] += 99
    masked["next_visibility"][1] = 0
    masked["next_collision"][1] = 1
    assert torch.allclose(baseline_loss, _loss(model, masked, config.method))
    baseline_loss.backward()
    assert model.transition[0].weight.grad.abs().sum() > 0
    assert model.next_observation.weight.grad.abs().sum() > 0
    single = {key: value[:1] for key, value in batch.items()}
    for seed in (0, 13):
        torch.manual_seed(seed)
        action, intent = model.plan(single)
        assert torch.equal(action, model(single)["action"])
        assert int(intent) == int(model(single)["intent_logits"].argmax(-1))
    rewarded = build_model(replace(config, wam_visibility_reward=True)).eval()
    action, _ = rewarded.plan(single, horizon=3, candidates=8, iterations=2)
    prior = rewarded(single)["action"]
    assert (action - prior).abs().max() <= config.wam_prior_radius + 1e-6
    assert torch.isfinite(action).all() and action.abs().max() <= 1

    cache = {key: torch.zeros(3, *shape) for key, shape in {
        "visual": (2, 16), "state": (2, 8), "previous_action": (2, 4), "action": (2, 4),
        "text_index": (), "intent": (), "phase": (), "visibility": (), "handoff": (), "collision": (),
    }.items()}
    cache["rows"] = [{"episode_id": "a"}, {"episode_id": "a"}, {"episode_id": "b"}]
    cache["text"] = torch.zeros(1, 16)
    cache["visibility"] = torch.tensor([0., 1., 0.])
    dataset = FeatureSequenceDataset(cache, history=1)
    assert dataset[0]["transition_valid"] == 1 and dataset[0]["next_visibility"] == 1
    assert dataset[1]["transition_valid"] == 0 and dataset[2]["transition_valid"] == 0

    from ..live_handoff_runner import LiveHandoffProtocol, build_episode_record
    from ..live_handoff_self_test import _task, _runtime_args
    from ..evaluator import replay_metric_differences
    from .repair_metrics import repair_record
    protocol = LiveHandoffProtocol(_task(), seed=0, bandwidth=256, latency_ms=0., drop_probability=0.)
    events = [{"event_type": name, "agent_ids": ["uav_1"], "start_t": 0, "end_t": 0, "success": name == "identity_confirm"} for name in ("identity_error", "identity_confirm")]
    protocol.events = events
    protocol.identity_errors = 1
    protocol.steps = [{"t": 0, "events": events, "truth": {"target_visibility": {"uav_0": True, "uav_1": True}}}]
    assert protocol.verify_t is None
    assert protocol.metrics(flight_time_s=0., camera_frames={})["identity_error_rate"] == 0.5
    # Use an existing schema-valid synthetic live episode for copy migration.
    from ..live_handoff_self_test import _moving_oracle_protocol, _cameras
    from ..live_handoff_runner import build_task
    args = _runtime_args(duration=24.)
    task = build_task(args)
    p = _moving_oracle_protocol(task)
    record = build_episode_record(task=task, protocol=p, seed=0, world="self_test", policy_name="oracle",
                                  spawn_offset_ned=(0., 3., 0.), cameras=_cameras(), flight_time_s=len(p.steps)/10.)
    original = deepcopy(record)
    record["outcome"]["metrics"]["identity_error_rate"] = 1.
    fixed, diff = repair_record(record)
    assert diff and fixed["outcome"]["success"] == original["outcome"]["success"]
    assert fixed["steps"] == record["steps"] and not replay_metric_differences(fixed)
    assert record["outcome"]["metrics"]["identity_error_rate"] == 1.
    print("WAM action conditioning, temporal inputs, terminal mask, prior fallback, bounded search and identity repair: PASS")


if __name__ == "__main__":
    main()
