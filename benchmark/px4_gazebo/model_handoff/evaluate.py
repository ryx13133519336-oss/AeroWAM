"""Paired hierarchical-bootstrap summaries for model Handoff results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Mapping

from ..evaluator import episode_metrics, replay_metric_differences


METRICS = (
    "task_success", "handoff_success", "role_switch_success",
    "target_visible_during_handoff", "visible_fraction_union",
    "longest_interruption_steps", "identity_error_rate", "collision_rate",
    "minimum_separation_m", "mean_separation_m", "maximum_separation_m",
    "path_length", "inference_latency_ms_mean",
)


def records(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for value in inputs:
        paths = [value] if value.is_file() else sorted(value.rglob("*.jsonl"))
        for path in paths:
            if ".attempts" in path.parts:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                record = json.loads(line)
                if record.get("outcome", {}).get("termination_reason") == "simulator_error":
                    continue
                if replay_metric_differences(record):
                    raise ValueError(f"recorded/replay metric mismatch: {path}")
                model = record.get("task_spec", {}).get("model", {})
                if not model.get("method"):
                    continue
                record["_method"] = str(model["method"])
                record["_training_seed"] = int(model["training_seed"])
                record["_fixture_id"] = str(record.get("task_spec", {}).get("fixture_id"))
                record["_split"] = str(record.get("task_spec", {}).get("fixture_split", record.get("split")))
                record["_metrics"] = episode_metrics(record)
                result.append(record)
    if not result:
        raise ValueError("no valid model records found")
    return result


def _percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def hierarchical_bootstrap(rows: list[dict[str, Any]], metric: str, *, samples: int, rng: random.Random) -> tuple[float, float, float]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(row["_training_seed"], {})[row["_fixture_id"]] = row
    seeds = sorted(by_seed)
    fixtures = sorted(set.intersection(*(set(by_seed[seed]) for seed in seeds)))
    if not seeds or not fixtures:
        raise ValueError("hierarchical bootstrap requires paired fixtures across seeds")
    observed = mean(float(row["_metrics"][metric]) for row in rows if row["_fixture_id"] in fixtures)
    estimates = []
    for _ in range(samples):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled_fixtures = [rng.choice(fixtures) for _ in fixtures]
        estimates.append(mean(float(by_seed[seed][fixture]["_metrics"][metric]) for seed in sampled_seeds for fixture in sampled_fixtures))
    return observed, _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def summarize(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    methods = sorted({row["_method"] for row in rows})
    splits = sorted({row["_split"] for row in rows})
    summary: dict[str, Any] = {"methods": {}, "pairwise_task_success": {}, "bootstrap_samples": bootstrap_samples, "excluded_simulator_errors": True}
    for method in methods:
        summary["methods"][method] = {}
        for split in splits:
            subset = [row for row in rows if row["_method"] == method and row["_split"] == split]
            if not subset:
                continue
            result = {}
            for metric in METRICS:
                estimate, low, high = hierarchical_bootstrap(
                    subset, metric, samples=bootstrap_samples,
                    rng=random.Random(f"{seed}:{method}:{split}:{metric}"),
                )
                result[metric] = {"mean": estimate, "ci95": [low, high]}
            summary["methods"][method][split] = result
        id_rows = [row for row in rows if row["_method"] == method and row["_split"] in {"validation", "test_id"}]
        ood_rows = [row for row in rows if row["_method"] == method and row["_split"].startswith("test_") and row["_split"] != "test_id"]
        summary["methods"][method]["generalization_gap"] = (
            mean(float(row["_metrics"]["task_success"]) for row in id_rows)
            - mean(float(row["_metrics"]["task_success"]) for row in ood_rows)
            if id_rows and ood_rows else None
        )
    for split in splits:
        summary["pairwise_task_success"][split] = {}
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1:]:
                left_rows = [row for row in rows if row["_method"] == left and row["_split"] == split]
                right_rows = [row for row in rows if row["_method"] == right and row["_split"] == split]
                if not left_rows or not right_rows:
                    continue
                left_map = {(row["_training_seed"], row["_fixture_id"]): row for row in left_rows}
                right_map = {(row["_training_seed"], row["_fixture_id"]): row for row in right_rows}
                seeds = sorted({key[0] for key in left_map} & {key[0] for key in right_map})
                fixtures = sorted(
                    set.intersection(*(
                        {key[1] for key in left_map if key[0] == training_seed}
                        & {key[1] for key in right_map if key[0] == training_seed}
                        for training_seed in seeds
                    ))
                ) if seeds else []
                if not fixtures:
                    continue
                rng = random.Random(f"{seed}:{split}:{left}:{right}")
                estimates = []
                for _ in range(bootstrap_samples):
                    sampled_seeds = [rng.choice(seeds) for _ in seeds]
                    sampled_fixtures = [rng.choice(fixtures) for _ in fixtures]
                    estimates.append(mean(
                        float(left_map[(training_seed, fixture)]["_metrics"]["task_success"])
                        - float(right_map[(training_seed, fixture)]["_metrics"]["task_success"])
                        for training_seed in sampled_seeds for fixture in sampled_fixtures
                    ))
                observed = mean(
                    float(left_map[(training_seed, fixture)]["_metrics"]["task_success"])
                    - float(right_map[(training_seed, fixture)]["_metrics"]["task_success"])
                    for training_seed in seeds for fixture in fixtures
                )
                summary["pairwise_task_success"][split][f"{left}-{right}"] = {
                    "mean_delta": observed,
                    "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
                }
    return summary


def markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Model-ready Handoff Evaluation", "", "Simulator-error and invalid episodes are excluded and must be retried.", ""]
    for method, splits in summary["methods"].items():
        lines.extend([f"## {method}", "", "| Split | Task success (95% CI) | Collision | Visibility union | Mean / max separation m | Path length | Latency ms |", "|---|---:|---:|---:|---:|---:|---:|"])
        for split, values in splits.items():
            if split == "generalization_gap":
                continue
            task = values["task_success"]
            lines.append(
                f"| {split} | {task['mean']:.3f} [{task['ci95'][0]:.3f}, {task['ci95'][1]:.3f}] "
                f"| {values['collision_rate']['mean']:.3f} | {values['visible_fraction_union']['mean']:.3f} "
                f"| {values['mean_separation_m']['mean']:.2f} / {values['maximum_separation_m']['mean']:.2f} "
                f"| {values['path_length']['mean']:.2f} | {values['inference_latency_ms_mean']['mean']:.1f} |"
            )
        gap = splits.get("generalization_gap")
        lines.extend(["", f"Generalization gap (ID - OOD task success): {gap if gap is not None else 'N/A (no ID reference records)'}", ""])
    return "\n".join(lines) + "\n"


def write_csv(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "split", "metric", "mean", "ci95_low", "ci95_high"))
        for method, splits in summary["methods"].items():
            for split, metrics in splits.items():
                if split == "generalization_gap":
                    continue
                for metric, values in metrics.items():
                    writer.writerow((method, split, metric, values["mean"], *values["ci95"]))


def write_plot(summary: Mapping[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot-output") from exc
    methods = list(summary["methods"])
    splits = ["test_appearance", "test_occluder", "test_combined"]
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    figure, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    width = 0.22
    for method_index, method in enumerate(methods):
        x_values, means, lower, upper = [], [], [], []
        for split_index, split in enumerate(splits):
            if split not in summary["methods"][method]:
                continue
            value = summary["methods"][method][split]["task_success"]
            x_values.append(split_index + (method_index - (len(methods) - 1) / 2) * width)
            means.append(value["mean"])
            lower.append(value["mean"] - value["ci95"][0])
            upper.append(value["ci95"][1] - value["mean"])
        axis.errorbar(x_values, means, yerr=[lower, upper], fmt="o", capsize=4, linewidth=1.8, color=colors[method_index % len(colors)], label=method)
    axis.set_xticks(range(len(splits)), ("Appearance OOD", "Occluder OOD", "Combined OOD"))
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Task success")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=min(3, len(methods)))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired model Handoff fixtures")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    result = summarize(records(args.inputs), bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown(result), encoding="utf-8")
    if args.csv_output is not None:
        write_csv(result, args.csv_output)
    if args.plot_output is not None:
        write_plot(result, args.plot_output)


if __name__ == "__main__":
    main()
