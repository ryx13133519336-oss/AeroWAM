"""Common, budget-capped training entry point for all three methods."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Mapping

class FeatureSequenceDataset:
    def __init__(self, cache: Mapping[str, Any], history: int = 4) -> None:
        self.cache = cache
        self.history = history
        rows = cache["rows"]
        self.indices = []
        for index, row in enumerate(rows):
            start = max(0, index - history + 1)
            if all(rows[item]["episode_id"] == row["episode_id"] for item in range(start, index + 1)):
                self.indices.append(index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch
        index = self.indices[item]
        rows = self.cache["rows"]
        episode_id = rows[index]["episode_id"]
        valid = [candidate for candidate in range(max(0, index - self.history + 1), index + 1) if rows[candidate]["episode_id"] == episode_id]
        valid = [valid[0]] * (self.history - len(valid)) + valid
        text_index = int(self.cache["text_index"][index])
        next_index = index + 1 if index + 1 < len(rows) and rows[index + 1]["episode_id"] == episode_id else index
        return {
            "visual": self.cache["visual"][valid].float(), "text": self.cache["text"][text_index].float(),
            "state": self.cache["state"][valid].float(), "previous_action": self.cache["previous_action"][valid].float(),
            "action": self.cache["action"][index].float(), "intent": self.cache["intent"][index],
            "phase": self.cache["phase"][index], "visibility": self.cache["visibility"][index],
            "handoff": self.cache["handoff"][index], "collision": self.cache["collision"][index],
            "next_visual": self.cache["visual"][next_index].float(),
            "next_state": self.cache["state"][next_index].float(),
            "next_previous_action": self.cache["previous_action"][next_index].float(),
        }


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def class_weights(labels: Any, num_classes: int, device: Any, max_weight: float = 12.0) -> Any:
    import torch

    counts = torch.bincount(labels.to(torch.long), minlength=num_classes).float()
    weights = torch.ones(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = torch.sqrt(
        counts.sum() / counts[present]
    ).clamp(max=max_weight)
    return weights.to(device)


def _loss(
    model: Any,
    batch: Mapping[str, Any],
    method: str,
    intent_weight: Any = None,
    phase_weight: Any = None,
) -> Any:
    import torch.nn.functional as functional
    output = model(batch)
    total = functional.smooth_l1_loss(output["action"], batch["action"])
    total = total + 0.25 * functional.cross_entropy(
        output["intent_logits"],
        batch["intent"],
        weight=intent_weight,
    )
    total = total + 0.15 * functional.cross_entropy(
        output["phase_logits"],
        batch["phase"],
        weight=phase_weight,
    )
    if method == "dual_wam_mpc":
        next_batch = {
            "visual": torch_cat_history(batch["visual"], batch["next_visual"]),
            "text": batch["text"],
            "state": torch_cat_history(batch["state"], batch["next_state"]),
            "previous_action": torch_cat_history(batch["previous_action"], batch["next_previous_action"]),
        }
        target_latent = model.encode(next_batch).detach()
        total = total + 0.5 * functional.mse_loss(output["next_latent"], target_latent)
        total = total + 0.2 * functional.binary_cross_entropy_with_logits(output["visibility_logit"], batch["visibility"])
        total = total + 0.2 * functional.binary_cross_entropy_with_logits(output["handoff_logit"], batch["handoff"])
        total = total + 0.4 * functional.binary_cross_entropy_with_logits(output["collision_logit"], batch["collision"])
    return total


def torch_cat_history(history: Any, current: Any) -> Any:
    import torch
    return torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)


def evaluate(
    model: Any,
    loader: Any,
    device: Any,
    method: str,
    intent_weight: Any,
    phase_weight: Any,
) -> float:
    import torch
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            losses.append(float(_loss(
                model,
                _to_device(batch, device),
                method,
                intent_weight,
                phase_weight,
            ).item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model-ready dual-UAV Handoff policy")
    parser.add_argument("--method", choices=("joint_bc", "dual_vln_hier", "dual_wam_mpc"), required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.steps <= 50_000:
        raise SystemExit("--steps must be in [1, 50000]")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training; install model_handoff/requirements.txt") from exc
    from .models import ModelConfig, build_model, checkpoint_payload, parameter_count
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cache = torch.load(args.validation_cache, map_location="cpu", weights_only=False)
    config = ModelConfig(method=args.method, feature_dim=int(train_cache["visual"].shape[-1]))
    model = build_model(config).to(device)
    count = parameter_count(model)
    if count > 20_000_000:
        raise RuntimeError(f"trainable parameter budget exceeded: {count:,}")
    train_loader = torch.utils.data.DataLoader(FeatureSequenceDataset(train_cache), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(FeatureSequenceDataset(val_cache), batch_size=args.batch_size, shuffle=False)
    intent_weight = class_weights(train_cache["intent"], 3, device)
    phase_weight = class_weights(train_cache["phase"], 4, device)
    print(f"intent_weight={intent_weight.detach().cpu().tolist()}", flush=True)
    print(f"phase_weight={phase_weight.detach().cpu().tolist()}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    iterator = iter(train_loader)
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, args.method, intent_weight, phase_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            validation_loss = evaluate(
                model,
                val_loader,
                device,
                args.method,
                intent_weight,
                phase_weight,
            )
            print(f"step={step} train_loss={loss.item():.6f} validation_loss={validation_loss:.6f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(checkpoint_payload(
                    model, config, seed=args.seed, step=step, validation_loss=validation_loss,
                    encoder=train_cache["encoder"], pretrained=train_cache["pretrained"],
                    trainable_parameters=count,
                ), args.output)


if __name__ == "__main__":
    main()
