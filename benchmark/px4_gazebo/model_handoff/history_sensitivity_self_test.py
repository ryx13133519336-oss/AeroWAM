"""No-model history intervention checks and numeric-control rejection."""

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .history_sensitivity import (
    AGENTS, MODES, QUERIES, canonical, compare, delta, feature_history, inspect,
    load_public, schedule, validate_request, SCHEMA, ATOL, RTOL,
    same_metrics,
)
from .search_history_probe import request


def encoded(color):
    stream = BytesIO()
    Image.new("RGB", (256, 256), color).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def populate(root):
    current = {a: encoded("gray") for a in AGENTS}
    for e in range(2):
        (root / f"episode_{e}").mkdir()
        past = {a: encoded("red" if e == 0 else "blue") for a in AGENTS}
        for query in QUERIES:
            agent, color = query.rsplit("_", 1)
            for mode, history in (("current_only", None), ("with_history", past)):
                (root / f"episode_{e}/{query}_{mode}.json").write_bytes(
                    canonical(request(agent, color, current, history)))


def rows():
    result = []
    for call in schedule():
        e, mode = call["episode"], call["mode"]
        value = 0. if mode == "current_only" else float(1 + (1-e if mode == "other_history" else e))
        result.append({**call, "latent": [value] * 3, "action": [[value] * 4] * 2,
                       "intent_logits": [value] * 3, "phase_logits": [value] * 4})
    return result


class SensitivityTests(unittest.TestCase):
    def test_public_only_load_does_not_need_private_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate(root)
            requests, hashes = load_public(root)
            self.assertEqual(len(hashes), 16)
            self.assertEqual(set(requests), set(QUERIES))
            self.assertFalse((root / "inspection.json").exists())

    def test_reject_private_fields_or_missing_view(self):
        value = request("uav_0", "red", {a: encoded("gray") for a in AGENTS})
        for field in ("answer", "private_layout", "timestamp", "states"):
            bad = deepcopy(value)
            bad[field] = "secret"
            with self.assertRaises(ValueError):
                validate_request(bad, history=False)
        del value["current_rgb_png_base64"]["uav_1"]
        with self.assertRaises(ValueError):
            validate_request(value, history=False)

    def test_reject_wrong_raster_size_or_encoding(self):
        for size, fmt in (((32, 32), "PNG"), ((256, 256), "JPEG")):
            stream = BytesIO()
            Image.new("RGB", size).save(stream, format=fmt)
            data = base64.b64encode(stream.getvalue()).decode()
            with self.assertRaises(ValueError):
                validate_request(request("uav_0", "red", {a: data for a in AGENTS}), history=False)

    def test_current_image_or_question_change_rejected(self):
        for change in ("question", "image", "cross_query"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                populate(root)
                query = "uav_0_red" if change != "cross_query" else "uav_0_blue"
                path = root / f"episode_1/{query}_with_history.json"
                value = json.loads(path.read_text())
                if change == "question":
                    value["question"] += "different"
                else:
                    value["current_rgb_png_base64"]["uav_0"] = encoded("green")
                path.write_bytes(canonical(value))
                with self.assertRaises(ValueError):
                    load_public(root)

    def test_history_padding_changes_only_past_slots(self):
        current, past = np.zeros((2, 5), dtype=np.float32), np.ones((2, 5), dtype=np.float32)
        full = feature_history(current, past, "with_history", 4)
        blank = feature_history(current, past, "current_only", 4)
        self.assertEqual(full.shape, (1, 4, 2, 5))
        np.testing.assert_array_equal(full[0, -1], current)
        np.testing.assert_array_equal(full[0, :3], np.stack([past] * 3))
        self.assertFalse(blank.any())
        np.testing.assert_array_equal(full, feature_history(current, past, "other_history", 4))

    def test_invalid_history_features_rejected(self):
        a = np.zeros((2, 5))
        for mode, history, b in (("bad", 4, a), ("with_history", 1, a),
                                  ("with_history", True, a), ("with_history", 4, a[:1]),
                                  ("with_history", 4, a + np.nan)):
            with self.assertRaises(ValueError):
                feature_history(a, b, mode, history)

    def test_seeded_schedule_covers_all_48_calls(self):
        self.assertEqual(schedule(), schedule())
        self.assertNotEqual(schedule(0), schedule(1))
        self.assertEqual(len(schedule()), 48)
        self.assertEqual(len({tuple(r.values()) for r in schedule()}), 48)
        self.assertEqual({r["mode"] for r in schedule()}, set(MODES))
        for seed in (True, -1, 2**32):
            with self.assertRaises(ValueError):
                schedule(seed)

    def test_true_and_swapped_history_controls(self):
        result = compare(rows())
        self.assertEqual(set(result), set(QUERIES))
        self.assertTrue(result[QUERIES[0]][0]["true_vs_other_history"]["action"]["beyond_numeric_tolerance"])

    def test_repeat_noise_or_cross_episode_state_rejected(self):
        for change in ("repeat", "current", "swap"):
            records = rows()
            mode = {"repeat": "with_history", "current": "current_only", "swap": "other_history"}[change]
            targets = [r for r in records if r["query"] == QUERIES[0] and r["episode"] == 0 and r["mode"] == mode]
            # Corrupt both repeats to separately test paired/swap invariants,
            # rather than only failing the repeat-noise gate.
            for target in targets[:1 if change == "repeat" else 2]:
                target["latent"][0] += .01
            with self.assertRaises(ValueError):
                compare(records)

    def test_missing_duplicate_nonfinite_rows_rejected(self):
        records = rows()
        bad = deepcopy(records)
        bad[0]["action"][0][0] = float("nan")
        for value in (records[:-1], records + [records[0]], bad):
            with self.assertRaises(ValueError):
                compare(value)

    def test_numeric_tolerance_not_semantic_success(self):
        self.assertFalse(delta([0], [1e-8])["beyond_numeric_tolerance"])
        self.assertTrue(delta([0], [.01])["beyond_numeric_tolerance"])
        for a, b in (([1], [1, 2]), ([1], [float("inf")]), ([], [])):
            with self.assertRaises(ValueError):
                delta(a, b)

    def test_saved_response_independent_inspection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, records = Path(temporary), rows()
            plan = {"schema": SCHEMA, "seed": 20261018, "schedule": schedule(), "atol": ATOL, "rtol": RTOL}
            summary = {"status": "sensitivity_measured", "checkpoint_unchanged": True,
                       "public_sources_unchanged": True, "rows_completed": 48, "comparisons": compare(records)}
            (root / "design.json").write_bytes(canonical(plan))
            (root / "summary.json").write_bytes(canonical(summary))
            trace = root / "responses.jsonl"
            trace.write_text("\n".join(canonical(r).decode() for r in records))
            self.assertEqual(inspect(root), summary)
            records.reverse()
            trace.write_text("\n".join(canonical(r).decode() for r in records))
            with self.assertRaises(ValueError):
                inspect(root)

    def test_cross_platform_norm_roundoff_not_changed_metric_or_flag(self):
        a = {"l2": 3.1977598166382117, "changed": True, "episode": 0}
        b = dict(a, l2=a["l2"] + 1e-15)
        self.assertTrue(same_metrics(a, b))
        for bad in (dict(a, l2=a["l2"] + 1e-8), dict(a, changed=False),
                    dict(a, episode=False), dict(a, l2=float("nan"))):
            self.assertFalse(same_metrics(a, bad))


if __name__ == "__main__":
    unittest.main()
