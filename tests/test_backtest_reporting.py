import unittest

import pandas as pd

from backtest_reporting import (
    add_season_column,
    build_season_breakdown,
    build_window_breakdown,
    iter_walk_forward_batches,
    season_label,
)


class BacktestReportingTests(unittest.TestCase):
    def prediction_frame(self):
        rows = []
        seasons = ["2023/24", "2024/25", "2025/26"]
        for season_index, season in enumerate(seasons):
            for match_index in range(4):
                actual = "home_win" if match_index < 2 else (
                    "draw" if match_index == 2 else "away_win"
                )
                rows.append({
                    "forecast_date": (
                        f"{2023 + season_index}-08-{10 + match_index:02d}"
                    ),
                    "season": season,
                    "rps": 0.24 - (season_index * 0.02) + (match_index * 0.001),
                    "naive_rps": 0.25,
                    "brier": 0.22,
                    "actual_outcome": actual,
                    "predicted_outcome": (
                        actual if match_index != 2 else "home_win"
                    ),
                    "outcome_correct": match_index != 2,
                    "ou_correct": match_index % 2 == 0,
                    "correct_score": match_index == 0,
                })
        return pd.DataFrame(rows)

    def test_season_label_uses_july_boundary(self):
        self.assertEqual(season_label("2024-06-30"), "2023/24")
        self.assertEqual(season_label("2024-07-01"), "2024/25")
        self.assertEqual(season_label("2025-01-10"), "2024/25")

    def test_walk_forward_batches_do_not_combine_repeated_gameweeks(self):
        frame = pd.DataFrame({
            "date_parsed": pd.to_datetime([
                "2023-08-12",
                "2024-08-17",
                "2025-08-16",
            ]),
            "Round Number": [1, 1, 1],
        })
        labelled = add_season_column(frame)
        batches = list(iter_walk_forward_batches(labelled))

        self.assertEqual(len(batches), 3)
        self.assertEqual(
            [batch[0].strftime("%Y-%m-%d") for batch in batches],
            ["2023-08-12", "2024-08-17", "2025-08-16"],
        )
        self.assertEqual(
            labelled["season"].tolist(),
            ["2023/24", "2024/25", "2025/26"],
        )

    def test_fixed_windows_are_predeclared_and_nested(self):
        summaries = build_window_breakdown(self.prediction_frame())

        self.assertEqual(
            [row["key"] for row in summaries],
            ["full", "recent_two", "latest"],
        )
        self.assertEqual(
            [row["matches_predicted"] for row in summaries],
            [12, 8, 4],
        )
        self.assertEqual(
            summaries[1]["seasons"],
            ["2024/25", "2025/26"],
        )
        self.assertLess(
            summaries[2]["avg_rps"],
            summaries[0]["avg_rps"],
        )
        self.assertLessEqual(
            summaries[0]["rps_ci_low"],
            summaries[0]["avg_rps"],
        )
        self.assertGreaterEqual(
            summaries[0]["rps_ci_high"],
            summaries[0]["avg_rps"],
        )

    def test_season_breakdown_keeps_each_season_separate(self):
        summaries = build_season_breakdown(self.prediction_frame())

        self.assertEqual(len(summaries), 3)
        self.assertEqual(
            [row["season_range"] for row in summaries],
            ["2023/24", "2024/25", "2025/26"],
        )
        self.assertTrue(
            all(row["matches_predicted"] == 4 for row in summaries)
        )


if __name__ == "__main__":
    unittest.main()
