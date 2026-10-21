import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fixture_prediction_export import (
    build_fixture_prediction, validate_against_csv, write_correct_score_csv,
    write_fixture_json,
)


class FixturePredictionExportTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.arange(1, 65, dtype=float).reshape(8, 8)
        self.matrix /= self.matrix.sum()
        grid = np.add.outer(np.arange(8), np.arange(8))
        self.fixture = {
            "id": 31251, "round_number": 8, "date": "24/10/2026 12:30",
            "home_team": "Aston Villa", "away_team": "Manchester City",
        }
        self.probs = {
            "home_win_prob": float(np.tril(self.matrix, -1).sum()),
            "draw_prob": float(np.trace(self.matrix)),
            "away_win_prob": float(np.triu(self.matrix, 1).sum()),
            "over_2_5_prob": float(self.matrix[grid > 2].sum()),
            "btts_prob": float(self.matrix[1:, 1:].sum()),
            "home_cs_prob": float(self.matrix[:, 0].sum()),
            "away_cs_prob": float(self.matrix[0, :].sum()),
            "home_xg": 1.05, "away_xg": 2.18,
        }

    def record(self):
        return build_fixture_prediction(
            self.fixture, self.probs, self.matrix,
            "static/heatmaps/Aston Villa_Manchester City_heatmap.png",
        )

    def test_matrix_is_lossless_and_kickoff_is_utc(self):
        record = self.record()
        self.assertEqual(record["fixture"]["kickoff_utc"], "2026-10-24T11:30:00Z")
        self.assertEqual(record["fixture"]["season"], "2026/27")
        self.assertEqual(record["fixture"]["id"], 31251)
        score = record["predictions"]["correct_score"]
        self.assertEqual(np.array(score["matrix"]).shape, (8, 8))
        self.assertAlmostEqual(sum(sum(row) for row in score["matrix"]), 1.0)
        self.assertEqual(score["top_scorelines"][0]["score"], "7-7")

    def test_market_mismatch_rejected(self):
        bad = dict(self.probs, btts_prob=0.123)
        with self.assertRaisesRegex(ValueError, "btts_prob differs"):
            build_fixture_prediction(self.fixture, bad, self.matrix, "image.png")

    def test_bad_matrix_rejected(self):
        with self.assertRaisesRegex(ValueError, "8x8"):
            build_fixture_prediction(self.fixture, self.probs, self.matrix[:6, :6], "image.png")
        bad = self.matrix.copy()
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid probabilities"):
            build_fixture_prediction(self.fixture, self.probs, bad, "image.png")

    def test_csv_json_roundtrip(self):
        record = self.record()
        frame = pd.DataFrame([dict(self.probs, home_team="Aston Villa", away_team="Manchester City")])
        validate_against_csv([record], frame)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_json(
                [record], root / "fixture_predictions.json",
                generated_at_utc="2026-10-21T10:00:00Z", form_weight=0.09,
                covariance=0.05, dispersion=1.0,
            )
            write_correct_score_csv([record], root / "correct_score_probabilities.csv")
            payload = json.loads((root / "fixture_predictions.json").read_text())
            self.assertEqual(payload["fixture_count"], 1)
            self.assertEqual(payload["fixtures"][0]["predictions"], record["predictions"])
            with (root / "correct_score_probabilities.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 64)
            self.assertAlmostEqual(sum(float(row["probability"]) for row in rows), 1.0)
            self.assertEqual(rows[0]["fixture_id"], "31251")

    def test_csv_json_row_mismatch_rejected(self):
        frame = pd.DataFrame([dict(self.probs, home_team="Aston Villa", away_team="Manchester City")])
        with self.assertRaisesRegex(ValueError, "JSON has"):
            validate_against_csv([], frame)
        frame.at[0, "home_win_prob"] = 0.2
        with self.assertRaisesRegex(ValueError, "CSV/JSON mismatch"):
            validate_against_csv([self.record()], frame)


if __name__ == "__main__":
    unittest.main()
