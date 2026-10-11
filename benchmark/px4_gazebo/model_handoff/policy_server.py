"""Local OpenCLIP + policy checkpoint inference server."""

from __future__ import annotations

import argparse
import base64
import json
import socketserver
import struct
from pathlib import Path
from typing import Any, Mapping

from .features import OpenCLIPEncoder, state_vector
from .protocol import ACTION_LIMITS, AGENT_IDS, INTENTS, MODEL_PROTOCOL_VERSION, ModelJointAction


def _receive_exact(connection: Any, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = connection.recv(size - len(value))
        if not chunk:
            raise ConnectionError("client closed request")
        value.extend(chunk)
    return bytes(value)


class InferencePolicy:
    def __init__(self, checkpoint: Path, *, device: str) -> None:
        import torch
        from .models import ModelConfig, build_model
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = ModelConfig(**payload["model_config"])
        self.device = torch.device(device)
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.encoder = OpenCLIPEncoder(device=device, pretrained=payload.get("metadata", {}).get("pretrained", "laion2b_s34b_b79k"))
        self.history: dict[str, list[Any]] = {}

    def reset(self, fixture_id: str) -> None:
        self.history[fixture_id] = []

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        if request.get("protocol") != MODEL_PROTOCOL_VERSION:
            raise ValueError("protocol mismatch")
        stream_id = str(request["episode_id"])
        payloads = [base64.b64decode(request["rgb_jpeg_b64"][agent], validate=True) for agent in AGENT_IDS]
        visual = self.encoder.encode_jpegs(payloads).float()
        text = self.encoder.encode_text([str(request["instruction"])])[0].float()
        state = torch.tensor([state_vector(request["states"][agent]) for agent in AGENT_IDS], dtype=torch.float32)
        previous = torch.zeros(2, 4)
        history_actions = request.get("action_history", [])
        if history_actions:
            last = history_actions[-1]
            previous = torch.tensor([last[agent] for agent in AGENT_IDS], dtype=torch.float32)
            limits = torch.tensor([ACTION_LIMITS["dx"], ACTION_LIMITS["dy"], ACTION_LIMITS["dz"], ACTION_LIMITS["dyaw"]])
            previous = previous / limits
        history = self.history.setdefault(stream_id, [])
        history.append((visual, state, previous))
        history[:] = history[-self.config.history:]
        padded = [history[0]] * (self.config.history - len(history)) + history
        batch = {
            "visual": torch.stack([item[0] for item in padded]).unsqueeze(0).to(self.device),
            "text": text.unsqueeze(0).to(self.device),
            "state": torch.stack([item[1] for item in padded]).unsqueeze(0).to(self.device),
            "previous_action": torch.stack([item[2] for item in padded]).unsqueeze(0).to(self.device),
        }
        with torch.inference_mode():
            if self.config.method == "dual_wam_mpc":
                action, intent = self.model.plan(batch, horizon=8, candidates=64, iterations=3)
                intent_index = int(intent.item())
            else:
                output = self.model(batch)
                action = output["action"]
                intent_index = int(output["intent_logits"].argmax(dim=-1).item())
        normalized = action[0].cpu().tolist()
        limits = [ACTION_LIMITS["dx"], ACTION_LIMITS["dy"], ACTION_LIMITS["dz"], ACTION_LIMITS["dyaw"]]
        waypoints = {agent: [normalized[index][axis] * limits[axis] for axis in range(4)] for index, agent in enumerate(AGENT_IDS)}
        return ModelJointAction.from_wire({"waypoints": waypoints, "intent": INTENTS[intent_index]}).to_wire()

class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            size = struct.unpack("!I", _receive_exact(self.request, 4))[0]
            if size > 16_000_000:
                raise ValueError("request exceeds 16 MB")
            request = json.loads(_receive_exact(self.request, size))
            response = self.server.policy(request)  # type: ignore[attr-defined]
        except Exception as exc:
            response = {"error": type(exc).__name__, "message": str(exc)}
        encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
        self.request.sendall(struct.pack("!I", len(encoded)) + encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a trained Handoff policy over localhost TCP")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    policy = InferencePolicy(args.checkpoint, device=args.device)
    with ReusableThreadingTCPServer((args.host, args.port), _Handler) as server:
        server.policy = policy  # type: ignore[attr-defined]
        print(f"serving {policy.config.method} on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
