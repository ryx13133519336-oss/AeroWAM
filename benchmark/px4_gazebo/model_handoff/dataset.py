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
    def __init__(self, root: Path, *, episode_id: str, fixture: Mapping[str, Any], mode: str) -> None:
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
        assert_truth_free(metadata)
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        (self.truth_dir / "fixture.json").write_text(
            json.dumps(dict(fixture), indent=2, sort_keys=True), encoding="utf-8"
        )
        self.index = 0

    def append(
        self, observation: ModelObservation, action: ModelJointAction, *,
        events: list[Mapping[str, Any]], truth: Mapping[str, Any],
        inference_latency_ms: float | None = None, timed_out: bool = False,
    ) -> None:
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
        assert_truth_free(sample)
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        truth_row = {"sample_index": self.index, "timestamp_ns": observation.timestamp_ns, **dict(truth)}
        with self.truth_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(truth_row, sort_keys=True) + "\n")
        self.index += 1


def validate_episode_directory(path: Path) -> dict[str, Any]:
    from PIL import Image

    path = Path(path)
    public = path / "policy"
    metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
    assert_truth_free(metadata)
    samples = [json.loads(line) for line in (public / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not samples:
        raise ValueError(f"no samples in {path}")
    previous_timestamp = -1
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
    return {"samples": len(samples), "first_timestamp_ns": int(samples[0]["timestamp_ns"]), "last_timestamp_ns": previous_timestamp}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model-ready Handoff dataset episodes")
    parser.add_argument("episodes", type=Path, nargs="+")
    args = parser.parse_args()
    for episode in args.episodes:
        print(episode, validate_episode_directory(episode))


if __name__ == "__main__":
    main()
