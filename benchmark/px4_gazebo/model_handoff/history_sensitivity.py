"""Frozen JointBC visual-history sensitivity, not task accuracy or flight."""

import argparse
import base64
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

SCHEMA = "aerowam-history-sensitivity/1-draft"
PUBLIC_SCHEMA = "aerowam-search-history-probe/1-draft"
AGENTS = ("uav_0", "uav_1")
QUERIES = tuple(f"{a}_{c}" for a in AGENTS for c in ("red", "blue"))
MODES = ("current_only", "with_history", "other_history")
ATOL, RTOL = 1e-6, 1e-5


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def images(value):
    if not isinstance(value, dict) or set(value) != set(AGENTS):
        raise ValueError("Exactly two public camera images required")
    result = {}
    for agent in AGENTS:
        data = base64.b64decode(value[agent], validate=True)
        with Image.open(BytesIO(data)) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.size != (256, 256):
                raise ValueError("Require unmodified 256x256 RGB PNG")
            im.load()
        result[agent] = data
    return result


def validate_request(value, *, history):
    expected = {"schema", "question", "current_rgb_png_base64"}
    if history:
        expected.add("past_rgb_snapshots_png_base64")
    if not isinstance(value, dict) or set(value) != expected or value["schema"] != PUBLIC_SCHEMA:
        raise ValueError("Only declared public history diagnostic fields allowed")
    if not isinstance(value["question"], str) or not 1 <= len(value["question"]) <= 1000:
        raise ValueError("Invalid public question")
    images(value["current_rgb_png_base64"])
    if history:
        snapshots = value["past_rgb_snapshots_png_base64"]
        if not isinstance(snapshots, list) or len(snapshots) != 1:
            raise ValueError("Exactly one ordered past RGB snapshot required")
        images(snapshots[0])


def load_public(root):
    """Read only 16 named public requests, never design/inspection/world/labels."""
    requests, fingerprints = {}, {}
    reference_views = {}
    for query in QUERIES:
        rows = []
        for episode in range(2):
            row = {}
            for mode in ("current_only", "with_history"):
                name = f"episode_{episode}/{query}_{mode}.json"
                raw = (Path(root) / name).read_bytes()
                value = json.loads(raw)
                validate_request(value, history=mode == "with_history")
                row[mode] = value
                fingerprints[name] = hashlib.sha256(raw).hexdigest()
            past = dict(row["with_history"])
            past.pop("past_rgb_snapshots_png_base64")
            if canonical(past) != canonical(row["current_only"]):
                raise ValueError("Historical input changed nonhistory fields")
            # The same episode images must serve all four questions.
            views = {k: v for k, v in row["with_history"].items() if k != "question"}
            if episode in reference_views and canonical(views) != reference_views[episode]:
                raise ValueError("Question-dependent images would confound the probe")
            reference_views[episode] = canonical(views)
            rows.append(row)
        if canonical(rows[0]["current_only"]) != canonical(rows[1]["current_only"]):
            raise ValueError("Matched current-only requests must be identical")
        if rows[0]["with_history"]["past_rgb_snapshots_png_base64"] == rows[1]["with_history"]["past_rgb_snapshots_png_base64"]:
            raise ValueError("The paired past snapshots must differ")
        requests[query] = rows
    return requests, fingerprints


def schedule(seed=20261018):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be integer in [0, 2**32)")
    rows = [{"query": q, "episode": e, "mode": m, "repeat": repeat}
            for q in QUERIES for e in range(2) for m in MODES for repeat in range(2)]
    random.Random(seed).shuffle(rows)
    return rows


def feature_history(current, past, mode, history):
    """Mirror first-frame left padding, explicitly not elapsed physical time."""
    if mode not in MODES or type(history) is not int or not 2 <= history <= 16:
        raise ValueError("Unsupported mode/history budget")
    current, past = np.asarray(current), np.asarray(past)
    if (current.shape != past.shape or current.ndim != 2 or current.shape[0] != 2
            or not np.isfinite(current).all() or not np.isfinite(past).all()):
        raise ValueError("Finite paired feature matrices required")
    first = current if mode == "current_only" else past
    return np.stack([first] * (history - 1) + [current])[None]


def delta(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite equally shaped comparison required")
    return {"max_abs": float(np.abs(a - b).max()), "l2": float(np.linalg.norm(a - b)),
            "beyond_numeric_tolerance": not bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))}


def compare(rows):
    indexed = {}
    fields = ("latent", "action", "intent_logits", "phase_logits")
    for row in rows:
        key = (row["query"], row["episode"], row["mode"], row["repeat"])
        if key in indexed:
            raise ValueError("Duplicate model evaluation")
        indexed[key] = row
    if set(indexed) != {(r["query"], r["episode"], r["mode"], r["repeat"]) for r in schedule()}:
        raise ValueError("All 48 scheduled calls required, failures cannot be omitted")
    results = {}
    for q in QUERIES:
        comparisons = []
        def get(e, m, repeat=0):
            return indexed[q, e, m, repeat]
        for e in range(2):
            for m in MODES:
                for field in fields:
                    if delta(get(e, m)[field], get(e, m, 1)[field])["beyond_numeric_tolerance"]:
                        raise ValueError("Repeated identical model input is not stable")
            for field in fields:
                if delta(get(0, "current_only")[field], get(1, "current_only")[field])["beyond_numeric_tolerance"]:
                    raise ValueError("Identical paired current-only model inputs differ")
                if delta(get(e, "other_history")[field], get(1-e, "with_history")[field])["beyond_numeric_tolerance"]:
                    raise ValueError("History swap did not reproduce the other paired input")
            comparisons.append({"episode": e, "true_vs_current": {
                f: delta(get(e, "with_history")[f], get(e, "current_only")[f]) for f in fields},
                "true_vs_other_history": {
                f: delta(get(e, "with_history")[f], get(e, "other_history")[f]) for f in fields}})
        results[q] = comparisons
    return results


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def same_metrics(a, b):
    """Allow float64 norm roundoff across BLAS platforms, not changed controls."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_metrics(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_metrics(x, y) for x, y in zip(a, b))
    if type(a) is float and type(b) is float:
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return type(a) is type(b) and a == b


def run(root, checkpoint, output, *, device="cuda", seed=20261018):
    requests, source_hashes = load_public(root)
    calls = schedule(seed)
    checkpoint_hash = digest(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": SCHEMA, "seed": seed, "schedule": calls, "atol": ATOL, "rtol": RTOL,
            "checkpoint_sha256": checkpoint_hash, "public_source_sha256": source_hashes,
            "method": "joint_bc", "state": "constant_zero_placeholder_not_measured",
            "previous_action": "constant_zero_placeholder_not_executed",
            "history": "one seen snapshot left padded then current hidden; not continuous online history",
            "response_metric": "latent/action/logit sensitivity; no left-right mapping or task accuracy",
            "scene_groups": 1, "repeats_are_numeric_controls_not_independent_replicates": True,
            "no_automatic_retry": True}
    (output / "design.json").write_bytes(canonical(plan))
    summary = {"schema": SCHEMA, "status": "initializing", "rows_completed": 0,
               "task_accuracy_evaluated": False, "memory_correctness_verified": False,
               "world_model_prediction_verified": False, "compositional_transfer_verified": False,
               "flight_started": False, "training_performed": False,
               "private_annotations_read": False, "production_latency_evaluated": False}
    rows = []
    try:
        # No network downloads; use the already installed, trusted local checkpoint.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from .policy_server import InferencePolicy
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        started = time.monotonic()
        policy = InferencePolicy(checkpoint, device=device)
        if policy.config.method != "joint_bc" or not 2 <= policy.config.history <= 16:
            raise ValueError("This revision only supports temporal JointBC checkpoints")
        summary.update(model_config=asdict(policy.config), torch_version=torch.__version__,
                       device=str(policy.device), initialization_wall_s=time.monotonic() - started)
        # Fingerprint actual loaded frozen encoder parameters, not just its name.
        encoder_hash = hashlib.sha256()
        for name, value in policy.encoder.model.state_dict().items():
            encoder_hash.update(name.encode())
            encoder_hash.update(str((tuple(value.shape), str(value.dtype))).encode())
            encoder_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        summary["encoder_state_sha256"] = encoder_hash.hexdigest()
        encoded, preprocessing = {}, {}
        def encode_views(views):
            features = []
            for agent, data in images(views).items():
                key = hashlib.sha256(data).hexdigest()
                if key not in encoded:
                    processed = policy.encoder.preprocess(Image.open(BytesIO(data)).convert("RGB"))
                    preprocessing[key] = tensor_hash(processed)
                    # Existing decoder accepts encoded PNG as well as JPEG; no lossy conversion.
                    encoded[key] = policy.encoder.encode_jpegs([data])[0].float().numpy()
                features.append(encoded[key])
            return np.stack(features)
        inputs, feature_audit = {}, {}
        for q, pair in requests.items():
            current = [encode_views(r["current_only"]["current_rgb_png_base64"]) for r in pair]
            past = [encode_views(r["with_history"]["past_rgb_snapshots_png_base64"][0]) for r in pair]
            text = policy.encoder.encode_text([pair[0]["current_only"]["question"]]).float().to(policy.device)
            if not np.array_equal(current[0], current[1]):
                raise ValueError("Current encoded features must match exactly")
            feature_audit[q] = {"past_feature_delta": delta(past[0], past[1]),
                                "current_feature_sha256": hashlib.sha256(current[0].tobytes()).hexdigest()}
            for e in range(2):
                for mode in MODES:
                    visual = feature_history(current[e], past[1-e if mode == "other_history" else e],
                                             mode, policy.config.history)
                    h = policy.config.history
                    inputs[q, e, mode] = {
                        "visual": torch.from_numpy(visual).to(policy.device), "text": text,
                        "state": torch.zeros(1, h, 2, policy.config.state_dim, device=policy.device),
                        "previous_action": torch.zeros(1, h, 2, policy.config.action_dim, device=policy.device)}
        summary.update(preprocessed_image_sha256=preprocessing, feature_audit=feature_audit,
                       model_parameters_trainable_during_probe=False)
        for parameter in policy.model.parameters():
            parameter.requires_grad_(False)
        with (output / "responses.jsonl").open("x", encoding="utf-8") as stream:
            for call in calls:
                batch = inputs[call["query"], call["episode"], call["mode"]]
                with torch.inference_mode():
                    latent = policy.model.encode(batch)
                    predictions = policy.model.heads(latent)
                row = {**call, "latent": latent[0].cpu().tolist(),
                       **{k: v[0].cpu().tolist() for k, v in predictions.items()}}
                # No RNG/stateful policy history, answers, actions sent to the simulator or reporters.
                if any(not math.isfinite(v) for k in ("latent", "action", "intent_logits", "phase_logits")
                       for v in np.asarray(row[k]).ravel()):
                    raise ValueError("Model output contains nonfinite values")
                stream.write(canonical(row).decode() + "\n")
                stream.flush()
                rows.append(row)
                summary["rows_completed"] = len(rows)
        summary.update(status="sensitivity_measured", comparisons=compare(rows),
                       numeric_repeat_control_passed=True, swapped_history_control_passed=True)
    except Exception as exc:
        summary.update(status="probe_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            summary["checkpoint_unchanged"] = digest(checkpoint) == checkpoint_hash
            _, final_hashes = load_public(root)
            summary["public_sources_unchanged"] = final_hashes == source_hashes
        except Exception as exc:
            summary.update(checkpoint_unchanged=False, public_sources_unchanged=False,
                           final_integrity_error=f"{type(exc).__name__}: {exc}")
        if not summary["checkpoint_unchanged"] or not summary["public_sources_unchanged"]:
            summary["status"] = "probe_failed"
        (output / "summary.json").write_bytes(canonical(summary))
    return summary


def inspect(output):
    """Recompute comparisons from saved outputs, without model dependencies."""
    output = Path(output)
    plan = json.loads((output / "design.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (plan["schema"] != SCHEMA or plan["schedule"] != schedule(plan["seed"])
            or plan["atol"] != ATOL or plan["rtol"] != RTOL
            or summary.get("status") != "sensitivity_measured"
            or not summary.get("checkpoint_unchanged") or not summary.get("public_sources_unchanged")):
        raise ValueError("Incomplete/changed design or failed probe")
    rows = [json.loads(line) for line in (output / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    if ([{k: r[k] for k in ("query", "episode", "mode", "repeat")} for r in rows] != plan["schedule"]
            or len(rows) != summary["rows_completed"] or not same_metrics(compare(rows), summary["comparisons"])):
        raise ValueError("Saved response audit differs from declared order/comparisons")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        result = inspect(args.output_dir)
    else:
        if args.public_dir is None or args.checkpoint is None:
            parser.error("--public-dir and --checkpoint required for inference")
        result = run(args.public_dir, args.checkpoint, args.output_dir, device=args.device, seed=args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    raise SystemExit(0 if result["status"] == "sensitivity_measured" else 2)


if __name__ == "__main__":
    main()
