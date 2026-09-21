import json
import os
import tempfile
import unittest

from model_performance import load_model_performance


class ModelPerformanceTests(unittest.TestCase):
    def write_json(self, directory, payload):
        path = os.path.join(directory, "model_accuracy.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def window_row(self):
        return {
            "key": "full",
            "label": "Full available window",
            "season_range": "2023/24–2025/26",
            "seasons": ["2023/24", "2024/25", "2025/26"],
            "matches_predicted": 1101,
            "avg_rps": 0.2092,
            "baseline_rps": 0.2371,
            "rps_reduction_pct": 11.8,
            "rps_ci_low": 0.198,
            "rps_ci_high": 0.220,
            "avg_brier": 0.2188,
            "outcome_accuracy": 51.0,
            "baseline_accuracy": 43.6,
            "accuracy_lift_pp": 7.4,
        }

    def base_payload(self):
        return {
            "season": "2023/24–2025/26 (combined)",
            "matches_predicted": 1101,
            "matches_skipped": 40,
            "avg_rps": 0.2092,
            "baseline_rps": 0.2371,
            "avg_brier": 0.2188,
            "outcome_accuracy": 51.0,
            "baseline_accuracy": 43.6,
            "window_breakdown": [self.window_row()],
            "season_breakdown": [],
            "gw_breakdown": [
                {
                    "gw": 1,
                    "fixtures": 28,
                    "outcome_acc": 64.3,
                    "avg_rps": 0.1486,
                    "ou_acc": 39.3,
                }
            ],
        }

    def test_enriches_valid_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, self.base_payload())
            result = load_model_performance(path)

        self.assertEqual(result["matches_predicted"], 1101)
        self.assertEqual(result["rps_reduction_pct"], 11.8)
        self.assertEqual(result["accuracy_lift_pp"], 7.4)
        self.assertEqual(result["chart_mode"], "windows")
        self.assertEqual(result["chart_labels"], ["Full available window"])
        self.assertEqual(result["chart_rps"], [0.2092])
        self.assertEqual(result["chart_baseline_rps"], [0.2371])
        self.assertEqual(result["window_breakdown"][0]["rps_ci_low"], 0.198)
        self.assertIsNotNone(result["updated_at"])

    def test_legacy_artifact_uses_gameweek_chart(self):
        payload = self.base_payload()
        payload.pop("window_breakdown")
        payload.pop("season_breakdown")

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, payload)
            result = load_model_performance(path)

        self.assertEqual(result["chart_mode"], "gameweeks")
        self.assertEqual(result["chart_labels"], ["GW1"])
        self.assertEqual(result["chart_rps"], [0.1486])

    def test_returns_none_for_missing_file(self):
        self.assertIsNone(load_model_performance("/path/that/does/not/exist.json"))

    def test_returns_none_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model_accuracy.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not-json")

            self.assertIsNone(load_model_performance(path))

    def test_returns_none_when_required_metric_is_missing(self):
        payload = self.base_payload()
        del payload["avg_rps"]

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, payload)
            self.assertIsNone(load_model_performance(path))

    def test_ignores_malformed_gameweek_rows(self):
        payload = self.base_payload()
        payload["gw_breakdown"].append({"gw": "bad"})

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, payload)
            result = load_model_performance(path)

        self.assertEqual(len(result["gw_breakdown"]), 1)


if __name__ == "__main__":
    unittest.main()
