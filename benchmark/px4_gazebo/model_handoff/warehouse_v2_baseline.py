"""Public-input baselines and a reproducible v2 migration report.

The baseline deliberately receives only ``ModelInput`` objects.  The trusted
evaluator is the only component that receives the private truth branch.  This
keeps the first visible result useful for debugging without presenting
synthetic kinematics as a flight or model leaderboard result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .warehouse_v2_task import (
    AGENTS,
    HORIZONS_S,
    SPLIT_MATRIX,
    generate_episode,
    model_inputs,
    score_predictions,
)


def _track_from_input(item: Mapping[str, Any]) -> tuple[float, int] | None:
    """Return the newest public target bearing and its timestamp, if visible."""
    visible = []
    for agent in AGENTS:
        track = item.get("observations", {}).get(agent, {}).get("visual_track", {})
        if track.get("visible") and track.get("bearing_y") is not None:
            visible.append(float(track["bearing_y"]))
    if not visible:
        return None
    return sum(visible) / len(visible), int(item["timestamp_ns"])


def constant_velocity_predictions(record: Any) -> list[dict[str, Any]]:
    """Emit a deterministic public-only prediction/action stream.

    The predictor estimates velocity from the last two publicly visible
    bearings and uses a broad uncertainty when the target is hidden.  It does
    not inspect the episode truth, latent family labels, or private fields.
    """
    inputs = model_inputs(record)
    history: list[tuple[float, int]] = []
    pending: set[str] = set()
    confirmed: set[str] = set()
    predictions: list[dict[str, Any]] = []
    for item in inputs[:-1]:
        current = _track_from_input(item.to_wire())
        if current is not None:
            history.append(current)
        if history:
            y, stamp = history[-1]
        else:
            y, stamp = -0.2, int(item.timestamp_ns)
        velocity = 0.0
        if len(history) >= 2:
            previous_y, previous_stamp = history[-2]
            delta_s = max((stamp - previous_stamp) / 1e9, 1e-6)
            velocity = max(-0.6, min(0.6, (y - previous_y) / delta_s))
        hidden = current is None
        horizons: dict[str, dict[str, Any]] = {}
        for horizon in HORIZONS_S:
            predicted_y = y + velocity * horizon
            # Hidden periods are intentionally less certain than grounded
            # tracks; this is a useful calibration baseline, not an oracle.
            std = 0.65 if hidden else max(0.18, 0.25 + 0.08 * horizon)
            horizons[str(horizon)] = {
                "mean_y": round(predicted_y, 4),
                "std_y": round(std, 4),
                "visible_probability": {agent: (0.5 if hidden else 0.65) for agent in AGENTS},
            }

        observations = item.to_wire()["observations"]
        visible_agents = [agent for agent in AGENTS
                          if observations[agent]["visual_track"].get("visible")]
        event, actor = "wait", AGENTS[len(predictions) % len(AGENTS)]
        if len(visible_agents) == 1:
            receiver = visible_agents[0]
            source = next(agent for agent in AGENTS if agent != receiver)
            if receiver not in pending:
                event, actor = "request", source
                pending.add(receiver)
            elif receiver not in confirmed:
                event, actor = "confirm", receiver
                confirmed.add(receiver)
        predictions.append({
            "sample_index": int(item.to_wire().get("timestamp_ns", 0) // 500_000_000),
            "horizons": horizons,
            "action": {"event": event, "agent_id": actor,
                       "evidence": "public_visual_track" if current else "insufficient"},
        })
    return predictions


def run_baseline_suite(*, episodes_per_split: int = 2, seed: int = 20261021) -> dict[str, Any]:
    """Evaluate the baseline on fixed seeds and return an auditable summary."""
    if episodes_per_split < 1:
        raise ValueError("episodes_per_split must be positive")
    rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLIT_MATRIX):
        for offset in range(episodes_per_split):
            episode_seed = seed + split_index * 1000 + offset
            record, truth = generate_episode(episode_seed, split)
            predictions = constant_velocity_predictions(record)
            score = score_predictions(record, truth, predictions)
            rows.append({"split": split, "seed": episode_seed, "score": score, "predictions": predictions})

    by_split: dict[str, dict[str, Any]] = {}
    for split in SPLIT_MATRIX:
        selected = [row["score"] for row in rows if row["split"] == split]
        by_split[split] = _aggregate_scores(selected)
    return {
        "schema": "aerowam-warehouse-v2-baseline-report/1",
        "method_id": "constant_velocity_public_only_v1",
        "task_id": "warehouse_coop_prediction",
        "scene_id": "aws_warehouse_handoff_v2",
        "episodes_per_split": episodes_per_split,
        "seed": seed,
        "splits": by_split,
        "model_score_eligible": False,
        "flight_verified": False,
        "limitations": ["synthetic_kinematics", "public_only_baseline", "no_px4_flight", "not_a_leaderboard_result"],
        "episodes": rows,
    }


def _aggregate_scores(scores: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"episodes": 0}
    numeric = ("prediction_nll", "visibility_brier", "calibration_error", "unsafe_handoff_rate")
    result: dict[str, Any] = {"episodes": len(scores)}
    for key in numeric:
        values = [float(score[key]) for score in scores if score.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    result["handoff_success_rate"] = sum(bool(score.get("handoff_success")) for score in scores) / len(scores)
    result["communication_events_mean"] = sum(int(score.get("communication_events", 0)) for score in scores) / len(scores)
    result["model_score_eligible"] = False
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Warehouse v2 public-only baseline",
        "",
        "This is a deterministic offline diagnostic. It is not a PX4 flight result or a model leaderboard score.",
        "",
        "| Split | NLL | Visibility Brier | Calibration | Unsafe handoff | Handoff success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, score in report["splits"].items():
        lines.append(
            f"| {split} | {score.get('prediction_nll', float('nan')):.4f} | "
            f"{score.get('visibility_brier', float('nan')):.4f} | "
            f"{score.get('calibration_error', float('nan')):.4f} | "
            f"{score.get('unsafe_handoff_rate', float('nan')):.4f} | "
            f"{score.get('handoff_success_rate', float('nan')):.4f} |"
        )
    lines.extend(["", "`model_score_eligible=false`; synthetic truth is retained only for trusted evaluator diagnostics.", ""])
    return "\n".join(lines)


def write_report(output: Path, *, episodes_per_split: int = 2, seed: int = 20261021) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    report = run_baseline_suite(episodes_per_split=episodes_per_split, seed=seed)
    public_rows = []
    for episode in report["episodes"]:
        public_rows.append({"split": episode["split"], "seed": episode["seed"], "predictions": episode["predictions"]})
    (output / "baseline_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in public_rows), encoding="utf8"
    )
    (output / "baseline_report.json").write_text(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2, ensure_ascii=True), encoding="utf8")
    (output / "baseline_report.md").write_text(_markdown(report), encoding="utf8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    report = write_report(args.output, episodes_per_split=args.episodes_per_split, seed=args.seed)
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
