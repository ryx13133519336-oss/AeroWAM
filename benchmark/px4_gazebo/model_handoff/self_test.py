"""ROS-free regression tests for the model-ready Handoff pipeline."""

from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile

from .dataset import EpisodeDatasetWriter, validate_episode_directory
from .fixtures import generate_fixtures, manifest_digest, read_manifest, HandoffFixture
from .protocol import (
    ACTION_LIMITS, ModelJointAction, ModelObservation, VehicleState,
    WaypointDelta, assert_truth_free,
)


EXPECTED_MANIFEST_SHA256 = "1486d421895e8ca531e081545794d190881d147d551340f4ae0b43cebf31a2fe"
WALL_BLOCKING_MANIFEST_SHA256 = "667309f16248102da5010b6b3fc70e3982569a9d08e4cf7bb81fe9237690666f"


def _observation() -> ModelObservation:
    from PIL import Image
    images = {}
    for agent, color in (("uav_0", "red"), ("uav_1", "blue")):
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), color).save(buffer, format="JPEG")
        images[agent] = buffer.getvalue()
    states = {
        agent: VehicleState((float(index), 0.0, 2.0), (0.0, 0.0, 0.0), 0.0, "ok")
        for index, agent in enumerate(("uav_0", "uav_1"))
    }
    return ModelObservation(
        episode_id="episode",
        instruction="Track the red sphere with one black stripe.",
        timestamp_ns=1_050_000_000,
        image_timestamp_ns={"uav_0": 1_000_000_000, "uav_1": 1_050_000_000},
        rgb_jpeg=images,
        states=states,
    )


def _test_models() -> str:
    try:
        import torch
    except ImportError:
        return "SKIP (PyTorch unavailable)"
    from .models import ModelConfig, build_model, parameter_count
    batch = {
        "visual": torch.randn(2, 4, 2, 512),
        "text": torch.randn(2, 512),
        "state": torch.randn(2, 4, 2, 8),
        "previous_action": torch.randn(2, 4, 2, 4),
    }
    for method in ("joint_bc", "dual_vln_hier", "dual_wam_mpc"):
        model = build_model(ModelConfig(method=method))
        output = model(batch)
        assert output["action"].shape == (2, 2, 4)
        assert output["intent_logits"].shape == (2, 3)
        assert parameter_count(model) <= 20_000_000
    return "PASS"


def main() -> None:
    fixtures = generate_fixtures()
    assert len(fixtures) == 400
    assert manifest_digest(fixtures) == EXPECTED_MANIFEST_SHA256
    manifest = Path(__file__).resolve().parent / "manifests" / "handoff_v3.jsonl"
    assert manifest_digest(read_manifest(manifest)) == EXPECTED_MANIFEST_SHA256
    counts = {split: sum(item.split == split for item in fixtures) for split in {item.split for item in fixtures}}
    assert counts == {"train": 240, "validation": 40, "test_appearance": 40, "test_occluder": 40, "test_combined": 40}

    from ..live_handoff_runner import build_task
    from ..live_handoff_self_test import _runtime_args, _assert_route_geometry, _moving_oracle_protocol
    wall_fixtures = generate_fixtures("wall_blocking_v1")
    wall_manifest = manifest.with_name("handoff_wall_blocking_v1.jsonl")
    assert manifest_digest(wall_fixtures) == WALL_BLOCKING_MANIFEST_SHA256
    assert manifest_digest(read_manifest(wall_manifest)) == WALL_BLOCKING_MANIFEST_SHA256
    accepted_task = build_task(_runtime_args(duration=24.0))
    unique_layouts = {}
    for fixture in wall_fixtures:
        assert HandoffFixture.from_dict(fixture.to_dict()) == fixture
        task = build_task(_runtime_args(duration=24.0), fixture)
        if fixture.split in {"train", "validation", "test_appearance"}:
            for key in ("target_start", "target_end", "occluder_center", "occluder_half_extents", "staging_position"):
                assert getattr(task, key) == getattr(accepted_task, key)
        unique_layouts[fixture.occluder] = task
    assert len(unique_layouts) == 3
    for task in unique_layouts.values():
        _assert_route_geometry(task)
        _moving_oracle_protocol(task)

    observation = _observation()
    assert observation.canonical_bytes() == observation.canonical_bytes()
    try:
        assert_truth_free({"nested": {"target_pose": [0, 0, 0]}})
    except ValueError:
        pass
    else:
        raise AssertionError("truth leakage was accepted")
    try:
        ModelObservation(
            episode_id="bad-episode", instruction="bad", timestamp_ns=0,
            image_timestamp_ns={"uav_0": 0, "uav_1": 100_000_001},
            rgb_jpeg=observation.rgb_jpeg, states=observation.states,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("camera skew over 100 ms was accepted")

    action = ModelJointAction(
        WaypointDelta(10.0, -10.0, 4.0, 2.0),
        WaypointDelta(-10.0, 10.0, -4.0, -2.0),
        "handoff",
    )
    clipped = action.to_wire()["waypoints"]
    for value in clipped.values():
        assert abs(value[0]) <= ACTION_LIMITS["dx"]
        assert abs(value[1]) <= ACTION_LIMITS["dy"]
        assert abs(value[2]) <= ACTION_LIMITS["dz"]
        assert abs(value[3]) <= ACTION_LIMITS["dyaw"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = fixtures[0].to_dict()
        writer = EpisodeDatasetWriter(root, episode_id="episode", fixture=fixture, mode="oracle")
        writer.append(
            observation, ModelJointAction(), events=[],
            truth={"target_pose": [1, 2, 3], "target_visibility": {"uav_0": True, "uav_1": False}, "collision": False},
        )
        result = validate_episode_directory(root / "episode")
        assert result["samples"] == 1
        public_text = (root / "episode" / "policy" / "samples.jsonl").read_text(encoding="utf-8")
        assert "target_pose" not in public_text and "truth" not in public_text
        assert "target_pose" in (root / "episode" / "evaluator" / "truth.jsonl").read_text(encoding="utf-8")
        json.loads(public_text)
        sample_path = root / "episode" / "policy" / "samples.jsonl"
        sample = json.loads(public_text)
        sample["timestamp_ns"] += 100_000_001
        sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        try:
            validate_episode_directory(root / "episode")
        except ValueError as exc:
            assert "image/action synchronization" in str(exc)
        else:
            raise AssertionError("misaligned image/action timestamps were accepted")

    model_status = _test_models()
    print("model-ready Handoff self-test: PASS")
    print(f"model forward tests: {model_status}")


if __name__ == "__main__":
    main()
