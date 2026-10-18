"""Compact model families for fair dual-UAV Handoff comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised by CLI diagnostics
    raise RuntimeError("PyTorch is required for model_handoff.models") from exc


METHODS = ("joint_bc", "dual_vln_hier", "dual_wam_mpc")


@dataclass(frozen=True)
class ModelConfig:
    method: str
    feature_dim: int = 512
    state_dim: int = 8
    action_dim: int = 4
    history: int = 4
    model_dim: int = 256
    layers: int = 3
    heads: int = 8
    dropout: float = 0.1
    phases: int = 4
    intents: int = 3
    wam_revision: str = "legacy_v1"
    wam_visibility_reward: bool = False
    wam_collision_reward: bool = False
    wam_prior_radius: float = 0.15

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.wam_revision not in {"legacy_v1", "expert_prior_v2"}:
            raise ValueError("unknown WAM revision")
        if not 0 < self.wam_prior_radius <= 0.25:
            raise ValueError("WAM prior radius must be in (0, 0.25]")


class PredictionHeads(nn.Module):
    def __init__(self, dimension: int, config: ModelConfig) -> None:
        super().__init__()
        self.action = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 2 * config.action_dim), nn.Tanh())
        self.intent = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, config.intents))
        self.phase = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, config.phases))

    def forward(self, latent: Tensor) -> dict[str, Tensor]:
        return {
            "action": self.action(latent).view(latent.shape[0], 2, 4),
            "intent_logits": self.intent(latent),
            "phase_logits": self.phase(latent),
        }


class JointBC(nn.Module):
    """Four-frame temporal Transformer directly predicting joint actions."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = 2 * config.feature_dim + config.feature_dim + 2 * config.state_dim + 2 * config.action_dim
        self.input = nn.Linear(input_dim, config.model_dim)
        self.position = nn.Parameter(torch.zeros(1, config.history, config.model_dim))
        layer = nn.TransformerEncoderLayer(
            config.model_dim, config.heads, 4 * config.model_dim,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, config.layers)
        self.heads = PredictionHeads(config.model_dim, config)

    def encode(self, batch: Mapping[str, Tensor]) -> Tensor:
        visual = batch["visual"]
        batch_size, history = visual.shape[:2]
        text = batch["text"].unsqueeze(1).expand(-1, history, -1)
        token = torch.cat((visual.flatten(2), text, batch["state"].flatten(2), batch["previous_action"].flatten(2)), dim=-1)
        encoded = self.temporal(self.input(token) + self.position[:, :history])
        return encoded[:, -1]

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return self.heads(self.encode(batch))


class DualVLNHier(nn.Module):
    """Dual-view cross-attention with explicit collaboration phase prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dim
        self.visual = nn.Linear(config.feature_dim, d)
        self.text = nn.Linear(config.feature_dim, d)
        self.state = nn.Linear(config.state_dim + config.action_dim, d)
        self.agent_embedding = nn.Parameter(torch.zeros(1, 1, 2, d))
        self.time_embedding = nn.Parameter(torch.zeros(1, config.history, 1, d))
        self.cross_attention = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(d, config.heads, 4 * d, dropout=config.dropout, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, config.layers)
        self.heads = PredictionHeads(d, config)

    def encode(self, batch: Mapping[str, Tensor]) -> Tensor:
        history = batch["visual"].shape[1]
        visual = self.visual(batch["visual"]) + self.agent_embedding + self.time_embedding[:, :history]
        state = self.state(torch.cat((batch["state"], batch["previous_action"]), dim=-1))
        agent_tokens = (visual + state).flatten(1, 2)
        text = self.text(batch["text"]).unsqueeze(1)
        attended, _ = self.cross_attention(agent_tokens, text, text, need_weights=False)
        fused = self.fusion(torch.cat((text, agent_tokens + attended), dim=1))
        return fused[:, 0]

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return self.heads(self.encode(batch))


class DualWAMMPC(nn.Module):
    """Latent joint dynamics model with visibility/safety predictions and CEM MPC."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dim
        context_dim = 2 * config.feature_dim + config.feature_dim + 2 * config.state_dim + 2 * config.action_dim
        if config.wam_revision == "expert_prior_v2":
            context_dim = config.history * (context_dim - config.feature_dim) + config.feature_dim
        self.context = nn.Sequential(nn.Linear(context_dim, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.transition = nn.Sequential(nn.Linear(d + 8, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.auxiliary_action = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 8), nn.Tanh())
        self.intent = nn.Linear(d, config.intents)
        self.phase = nn.Linear(d, config.phases)
        self.visibility = nn.Linear(d, 1)
        self.handoff = nn.Linear(d, 1)
        self.collision = nn.Linear(d, 1)
        if config.wam_revision == "expert_prior_v2":
            # Fixed observation targets prevent a collapsed learned latent from
            # making the transition loss artificially small.
            self.next_observation = nn.Linear(d, 2 * config.feature_dim + 2 * config.state_dim)

    def encode(self, batch: Mapping[str, Tensor]) -> Tensor:
        if self.config.wam_revision == "expert_prior_v2":
            return self.context(torch.cat((
                batch["visual"].flatten(1), batch["text"],
                batch["state"].flatten(1), batch["previous_action"].flatten(1),
            ), dim=-1))
        visual = batch["visual"][:, -1].flatten(1)
        state = batch["state"][:, -1].flatten(1)
        previous = batch["previous_action"][:, -1].flatten(1)
        return self.context(torch.cat((visual, batch["text"], state, previous), dim=-1))

    def forward(self, batch: Mapping[str, Tensor], *, transition_action: Tensor | None = None) -> dict[str, Tensor]:
        latent = self.encode(batch)
        action = self.auxiliary_action(latent).view(latent.shape[0], 2, 4)
        condition = action if transition_action is None else transition_action
        next_latent = self.transition(torch.cat((latent, condition.flatten(1)), dim=-1))
        output = {
            "latent": latent, "next_latent": next_latent, "action": action,
            "intent_logits": self.intent(latent), "phase_logits": self.phase(latent),
            "visibility_logit": self.visibility(next_latent).squeeze(-1),
            "handoff_logit": self.handoff(next_latent).squeeze(-1),
            "collision_logit": self.collision(next_latent).squeeze(-1),
        }
        if self.config.wam_revision == "expert_prior_v2":
            output["next_observation"] = self.next_observation(next_latent)
        return output

    @torch.no_grad()
    def plan(
        self, batch: Mapping[str, Tensor], *, horizon: int = 8,
        candidates: int = 64, iterations: int = 3,
    ) -> tuple[Tensor, Tensor]:
        if batch["visual"].shape[0] != 1:
            raise ValueError("MPC inference currently requires batch size 1")
        if min(horizon, candidates, iterations) < 1:
            raise ValueError("planner budgets must be positive")
        if self.config.wam_revision == "expert_prior_v2":
            return self._plan_with_prior(batch, horizon=horizon, candidates=candidates, iterations=iterations)
        device = batch["visual"].device
        latent0 = self.encode(batch)
        mean = torch.zeros(horizon, 8, device=device)
        std = torch.full_like(mean, 0.7)
        elite_count = max(4, candidates // 8)
        for _ in range(iterations):
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(candidates, horizon, 8, device=device)).clamp(-1.0, 1.0)
            latent = latent0.expand(candidates, -1)
            score = torch.zeros(candidates, device=device)
            for step in range(horizon):
                latent = self.transition(torch.cat((latent, actions[:, step]), dim=-1))
                visible = torch.sigmoid(self.visibility(latent).squeeze(-1))
                handoff = torch.sigmoid(self.handoff(latent).squeeze(-1))
                collision = torch.sigmoid(self.collision(latent).squeeze(-1))
                effort = actions[:, step].square().mean(dim=-1)
                score += 1.5 * visible + 0.5 * handoff - 4.0 * collision - 0.05 * effort
            elite = actions[score.topk(elite_count).indices]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(0.05)
        intent = self.intent(latent0).argmax(dim=-1)
        return mean[0].view(1, 2, 4), intent

    def _plan_with_prior(self, batch: Mapping[str, Tensor], *, horizon: int, candidates: int, iterations: int) -> tuple[Tensor, Tensor]:
        """Search near the imitation policy; singleton labels never rank plans.

        This is an expert-prior baseline. Its predicted transitions are proxies
        learned from recorded expert commands, not verified executed controls.
        """
        latent0 = self.encode(batch)
        if not (self.config.wam_visibility_reward or self.config.wam_collision_reward):
            # No validated action-dependent objective: do not manufacture a
            # benefit from random search. Expose this fallback in metadata.
            return self.auxiliary_action(latent0).view(1, 2, 4), self.intent(latent0).argmax(-1)
        latent = latent0
        references = []
        for _ in range(horizon):
            prior = self.auxiliary_action(latent)
            references.append(prior[0])
            latent = self.transition(torch.cat((latent, prior), dim=-1))
        reference = torch.stack(references)
        radius = self.config.wam_prior_radius
        lower, upper = (reference - radius).clamp(-1, 1), (reference + radius).clamp(-1, 1)
        mean = reference.clone()
        std = torch.full_like(mean, radius / 2)
        best_actions = reference.clone()
        best_score = float("-inf")
        for _ in range(iterations):
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(candidates, horizon, 8, device=mean.device)
            actions = actions.maximum(lower).minimum(upper)
            actions[0] = reference  # Always allow the learned prior itself.
            latent = latent0.expand(candidates, -1)
            score = torch.zeros(candidates, device=mean.device)
            for step in range(horizon):
                prior = self.auxiliary_action(latent)
                score -= (actions[:, step] - prior).square().mean(-1)
                latent = self.transition(torch.cat((latent, actions[:, step]), dim=-1))
                if self.config.wam_visibility_reward:
                    score += 1.5 * torch.sigmoid(self.visibility(latent).squeeze(-1))
                if self.config.wam_collision_reward:
                    score -= 4.0 * torch.sigmoid(self.collision(latent).squeeze(-1))
                # Handoff events are sparse protocol events, not a validated
                # action-conditioned progress reward, so they do not rank plans.
            winner = int(score.argmax())
            if float(score[winner]) > best_score:
                best_score = float(score[winner])
                best_actions = actions[winner].clone()
            elite = actions[score.topk(max(1, candidates // 8)).indices]
            mean = elite.mean(0)
            std = elite.std(0, unbiased=False).clamp_min(radius / 20)
        intent = self.intent(latent0).argmax(-1)
        return best_actions[0].view(1, 2, 4), intent


def build_model(config: ModelConfig) -> nn.Module:
    return {
        "joint_bc": JointBC,
        "dual_vln_hier": DualVLNHier,
        "dual_wam_mpc": DualWAMMPC,
    }[config.method](config)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_payload(model: nn.Module, config: ModelConfig, **metadata: Any) -> dict[str, Any]:
    return {"model_config": asdict(config), "model_state": model.state_dict(), "metadata": metadata}
