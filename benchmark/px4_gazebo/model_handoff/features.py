"""Frozen OpenCLIP ViT-B/32 feature caching for recorded episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from ..contract import AGENT_IDS


ENCODER_NAME = "ViT-B-32"
ENCODER_PRETRAINED = "laion2b_s34b_b79k"


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "feature caching requires torch, Pillow, and open_clip_torch; "
            "install them in the training environment"
        ) from exc
    return open_clip, torch, Image


def state_vector(state: Mapping[str, Any]) -> list[float]:
    health = {"ok": 0.0, "degraded": 0.5, "lost": 1.0}.get(str(state["health"]), 1.0)
    return [
        *[float(value) for value in state["position"]],
        *[float(value) for value in state["velocity"]],
        float(state["yaw"]), health,
    ]


def normalized_action(label: Mapping[str, Any]) -> tuple[list[list[float]], int]:
    from .protocol import ACTION_LIMITS, INTENTS
    limits = [ACTION_LIMITS["dx"], ACTION_LIMITS["dy"], ACTION_LIMITS["dz"], ACTION_LIMITS["dyaw"]]
    action = [[float(value) / limits[index] for index, value in enumerate(label["waypoints"][agent])] for agent in AGENT_IDS]
    return action, INTENTS.index(str(label.get("intent", "none")))


def _phase(events: Sequence[Mapping[str, Any]], intent: int) -> int:
    names = {str(event.get("event_type")) for event in events}
    if "verify" in names or intent == 2:
        return 3
    if "handoff" in names or intent == 1:
        return 2
    if "pre_position" in names:
        return 1
    return 0


class OpenCLIPEncoder:
    def __init__(self, *, device: str = "cuda", pretrained: str = ENCODER_PRETRAINED) -> None:
        open_clip, torch, _ = _dependencies()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            ENCODER_NAME, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(ENCODER_NAME)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_paths(self, paths: Sequence[Path]) -> Any:
        _, torch, Image = _dependencies()
        images = torch.stack([self.preprocess(Image.open(path).convert("RGB")) for path in paths]).to(self.device)
        with torch.inference_mode():
            value = self.model.encode_image(images)
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return value.cpu()

    def encode_jpegs(self, payloads: Sequence[bytes]) -> Any:
        from io import BytesIO
        _, torch, Image = _dependencies()
        images = torch.stack([self.preprocess(Image.open(BytesIO(payload)).convert("RGB")) for payload in payloads]).to(self.device)
        with torch.inference_mode():
            value = self.model.encode_image(images)
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return value.cpu()

    def encode_text(self, texts: Sequence[str]) -> Any:
        torch = self.torch
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode():
            value = self.model.encode_text(tokens)
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return value.cpu()


def episode_directories(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if (root / "policy" / "samples.jsonl").exists():
            found.append(root)
        elif root.exists():
            found.extend(path.parent.parent for path in root.rglob("policy/samples.jsonl"))
    return sorted(set(found))


def cache_features(episodes: Sequence[Path], output: Path, *, device: str, batch_size: int = 64) -> None:
    _, torch, _ = _dependencies()
    encoder = OpenCLIPEncoder(device=device)
    rows: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    truth_by_episode: dict[str, dict[int, Mapping[str, Any]]] = {}
    for episode in episodes:
        public = episode / "policy"
        metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
        truth_path = episode / "evaluator" / "truth.jsonl"
        truth_by_episode[metadata["episode_id"]] = {
            int(item["sample_index"]): item
            for item in (json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines() if line)
        }
        for sample in (json.loads(line) for line in (public / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line):
            sample["episode_id"] = metadata["episode_id"]
            sample["split"] = metadata["split"]
            rows.append(sample)
            image_paths.extend(public / sample["rgb_jpeg"][agent] for agent in AGENT_IDS)
    if not rows:
        raise ValueError("no dataset samples found")
    print(f"encoder={ENCODER_NAME} device={encoder.device} episodes={len(episodes)} samples={len(rows)} images={len(image_paths)}", flush=True)
    started = time.monotonic()
    visual_parts = []
    for start in range(0, len(image_paths), batch_size):
        visual_parts.append(encoder.encode_paths(image_paths[start:start + batch_size]))
        if (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(image_paths):
            print(f"images={min(start + batch_size, len(image_paths))}/{len(image_paths)} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    visual = torch.cat(visual_parts).view(len(rows), 2, -1).to(torch.float16)
    unique_text = sorted({str(row["instruction"]) for row in rows})
    text_features = encoder.encode_text(unique_text).to(torch.float16)
    text_lookup = {text: index for index, text in enumerate(unique_text)}
    states, actions, intents, phases, visibility, handoff, collision = [], [], [], [], [], [], []
    previous = [[0.0] * 4, [0.0] * 4]
    previous_episode = None
    for row in rows:
        if row["episode_id"] != previous_episode:
            previous = [[0.0] * 4, [0.0] * 4]
            previous_episode = row["episode_id"]
        states.append([state_vector(row["states"][agent]) for agent in AGENT_IDS])
        action, intent = normalized_action(row["label"])
        actions.append(action)
        intents.append(intent)
        phases.append(_phase(row.get("events", []), intent))
        row["previous_action"] = previous
        previous = action
        truth = truth_by_episode[row["episode_id"]][int(row["sample_index"])]
        visibility_map = truth.get("target_visibility", {})
        visibility.append(float(any(bool(visibility_map.get(agent, False)) for agent in AGENT_IDS)))
        handoff.append(float(any(event.get("event_type") == "handoff" for event in row.get("events", []))))
        collision.append(float(bool(truth.get("collision", False))))
    payload = {
        "schema": "aerowam-openclip-cache/1", "encoder": ENCODER_NAME,
        "pretrained": ENCODER_PRETRAINED, "rows": rows, "texts": unique_text,
        "visual": visual, "text": text_features,
        "state": torch.tensor(states, dtype=torch.float32),
        "action": torch.tensor(actions, dtype=torch.float32),
        "intent": torch.tensor(intents, dtype=torch.long),
        "phase": torch.tensor(phases, dtype=torch.long),
        "visibility": torch.tensor(visibility, dtype=torch.float32),
        "handoff": torch.tensor(handoff, dtype=torch.float32),
        "collision": torch.tensor(collision, dtype=torch.float32),
        "text_index": torch.tensor([text_lookup[str(row["instruction"])] for row in rows], dtype=torch.long),
        "previous_action": torch.tensor([row["previous_action"] for row in rows], dtype=torch.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen OpenCLIP ViT-B/32 Handoff features")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", nargs="+")
    args = parser.parse_args()
    episodes = episode_directories(args.inputs)
    if args.splits:
        allowed = set(args.splits)
        episodes = [
            episode for episode in episodes
            if json.loads((episode / "policy" / "metadata.json").read_text(encoding="utf-8"))["split"] in allowed
        ]
    cache_features(episodes, args.output, device=args.device, batch_size=args.batch_size)
    print(f"cached {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
