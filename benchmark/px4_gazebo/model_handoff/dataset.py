"""Episode recorder with public policy data and evaluator truth isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..contract import AGENT_IDS
from .protocol import ModelJointAction, ModelObservation, assert_truth_free


class EpisodeDatasetWriter:
    def __init__(self, root: Path, *, episode_id: str, fixture: Mapping[str, Any], mode: str,
                 record_execution: bool = False, strict_image_pairs: bool = False) -> None:
        if strict_image_pairs and not record_execution:
            raise ValueError("Strict image sampling requires execution recording")
        self.root = Path(root) / episode_id
        self.public_dir = self.root / "policy"
        self.truth_dir = self.root / "evaluator"
        self.image_dir = self.public_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=False)
        self.truth_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.public_dir / "samples.jsonl"
        self.truth_path = self.truth_dir / "truth.jsonl"
        self.metadata_path = self.public_dir / "metadata.json"
        metadata = {
            "schema": "aerowam-model-dataset/1", "episode_id": episode_id,
            "fixture_id": str(fixture["fixture_id"]), "split": str(fixture["split"]),
            "instruction": str(fixture["instruction"]), "mode": str(mode),
        }
        if record_execution:
            metadata["schema"] = "aerowam-model-dataset/2"
            metadata["execution_trace"] = "published_controls.jsonl"
            metadata["execution_semantics"] = "published_setpoints_not_verified_actuation"
        if strict_image_pairs:
            from .sampling import SAMPLING_PROFILE
            metadata["sampling_profile"] = SAMPLING_PROFILE
        assert_truth_free(metadata)
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        (self.truth_dir / "fixture.json").write_text(
            json.dumps(dict(fixture), indent=2, sort_keys=True), encoding="utf-8"
        )
        self.index = 0
        self.record_execution = record_execution
        self.last_timestamp_ns = -1
        self.last_image_timestamp_ns = {agent: -1 for agent in AGENT_IDS}
        self.strict_image_pairs = strict_image_pairs

    def frame_pair_advanced(self, stamps: Mapping[str, int]) -> bool:
        """A protocol event must not consume an already recorded image pair."""
        return max(int(stamps[agent]) for agent in AGENT_IDS) > self.last_timestamp_ns

    def append(
        self, observation: ModelObservation, action: ModelJointAction, *,
        events: list[Mapping[str, Any]], truth: Mapping[str, Any],
        inference_latency_ms: float | None = None, timed_out: bool = False,
        protocol_t: int | None = None,
    ) -> None:
        if observation.timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("sample timestamps are not strictly increasing")
        if self.strict_image_pairs and any(observation.image_timestamp_ns[agent] <= self.last_image_timestamp_ns[agent]
                                           for agent in AGENT_IDS):
            raise ValueError("Each recorded camera stamp must advance")
        if self.record_execution and (not isinstance(protocol_t, int) or isinstance(protocol_t, bool) or protocol_t < 0):
            raise ValueError("Recorded controls require the sample's protocol step")
        image_paths: dict[str, str] = {}
        image_hashes: dict[str, str] = {}
        for agent_id in AGENT_IDS:
            relative = Path("images") / f"{self.index:06d}_{agent_id}.jpg"
            payload = observation.rgb_jpeg[agent_id]
            (self.public_dir / relative).write_bytes(payload)
            image_paths[agent_id] = relative.as_posix()
            image_hashes[agent_id] = hashlib.sha256(payload).hexdigest()
        sample = observation.to_wire(include_images=False)
        sample.update({
            "sample_index": self.index,
            "rgb_jpeg": image_paths,
            "rgb_sha256": image_hashes,
            "label": action.to_wire(),
            "events": [
                {key: value for key, value in event.items() if key != "attributes"}
                for event in events
            ],
            "inference_latency_ms": inference_latency_ms,
            "timed_out": bool(timed_out),
        })
        if self.record_execution:
            sample["protocol_t"] = protocol_t
        assert_truth_free(sample)
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        truth_row = {"sample_index": self.index, "timestamp_ns": observation.timestamp_ns, **dict(truth)}
        with self.truth_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(truth_row, sort_keys=True) + "\n")
        self.index += 1
        self.last_timestamp_ns = observation.timestamp_ns
        self.last_image_timestamp_ns = dict(observation.image_timestamp_ns)


def validate_episode_directory(path: Path, *, require_expert_supervision: bool = False, require_execution_trace: bool = False) -> dict[str, Any]:
    from PIL import Image

    path = Path(path)
    public = path / "policy"
    metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
    assert_truth_free(metadata)
    if metadata.get("schema") not in {"aerowam-model-dataset/1", "aerowam-model-dataset/2"}:
        raise ValueError("Unsupported dataset schema")
    samples = [json.loads(line) for line in (public / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not samples:
        raise ValueError(f"no samples in {path}")
    if require_expert_supervision:
        if metadata.get("mode") != "oracle":
            raise ValueError("expert supervision validation requires oracle mode")
        intents = {sample["label"]["intent"] for sample in samples}
        if not {"handoff", "confirm"}.issubset(intents):
            raise ValueError("expert dataset is missing handoff or confirm supervision")
        for index, sample in enumerate(samples):
            expected = [row["label"]["waypoints"] for row in samples[max(0, index - 4):index]]
            if sample["action_history"] != expected:
                raise ValueError("expert command history differs from preceding recorded labels")
    previous_timestamp = -1
    previous_stamps = {agent: -1 for agent in AGENT_IDS}
    strict_pairs = "sampling_profile" in metadata
    if strict_pairs:
        from .sampling import SAMPLING_PROFILE
        if metadata["sampling_profile"] != SAMPLING_PROFILE:
            raise ValueError("Unknown sampling profile")
    truth_rows = [json.loads(line) for line in (path / "evaluator" / "truth.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(truth_rows) != len(samples):
        raise ValueError("policy samples and evaluator truth counts differ")
    for expected_index, sample in enumerate(samples):
        assert_truth_free(sample)
        if int(sample["sample_index"]) != expected_index:
            raise ValueError("sample indices are not contiguous")
        timestamp = int(sample["timestamp_ns"])
        if timestamp <= previous_timestamp:
            raise ValueError("sample timestamps are not strictly increasing")
        previous_timestamp = timestamp
        stamps = sample["image_timestamp_ns"]
        if strict_pairs and any(int(stamps[agent]) <= previous_stamps[agent] for agent in AGENT_IDS):
            raise ValueError("Each recorded camera stamp must advance")
        previous_stamps = {agent: int(stamps[agent]) for agent in AGENT_IDS}
        if abs(int(stamps["uav_0"]) - int(stamps["uav_1"])) > 100_000_000:
            raise ValueError("camera pair exceeds 100 ms synchronization limit")
        if any(abs(timestamp - int(stamps[agent])) > 100_000_000 for agent in AGENT_IDS):
            raise ValueError("image/action synchronization exceeds 100 ms")
        ModelJointAction.from_wire(sample["label"])
        truth = truth_rows[expected_index]
        if int(truth["sample_index"]) != expected_index or int(truth["timestamp_ns"]) != timestamp:
            raise ValueError("policy sample and evaluator truth alignment differs")
        for agent_id in AGENT_IDS:
            image = public / sample["rgb_jpeg"][agent_id]
            if hashlib.sha256(image.read_bytes()).hexdigest() != sample["rgb_sha256"][agent_id]:
                raise ValueError(f"image digest mismatch: {image}")
            with Image.open(image) as frame:
                if frame.format != "JPEG" or frame.size != (256, 256) or frame.mode != "RGB":
                    raise ValueError(f"expected a 256x256 RGB JPEG: {image}")
                frame.load()
    result = {"samples": len(samples), "first_timestamp_ns": int(samples[0]["timestamp_ns"]), "last_timestamp_ns": previous_timestamp}
    declares_trace = metadata.get("schema") == "aerowam-model-dataset/2"
    if require_execution_trace and not declares_trace:
        raise ValueError("Published-control trace required; old samples cannot reconstruct it")
    if declares_trace:
        from .execution_trace import validate_execution_file
        if metadata.get("execution_trace") != "published_controls.jsonl":
            raise ValueError("Invalid trace path")
        if metadata.get("execution_semantics") != "published_setpoints_not_verified_actuation":
            raise ValueError("Invalid trace execution semantics")
        result["execution"] = validate_execution_file(public / "published_controls.jsonl", samples=samples, episode_id=metadata["episode_id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model-ready Handoff dataset episodes")
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--require-expert-supervision", action="store_true")
    parser.add_argument("--require-execution-trace", action="store_true")
    args = parser.parse_args()
    for episode in args.episodes:
        print(episode, validate_episode_directory(episode, require_expert_supervision=args.require_expert_supervision, require_execution_trace=args.require_execution_trace))


if __name__ == "__main__":
    main()
