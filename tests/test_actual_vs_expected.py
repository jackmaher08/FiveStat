import unittest

from actual_vs_expected import build_actual_vs_expected


class ActualVsExpectedTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "Team": "Brighton",
                "MP": 4,
                "G": 13,
                "xG": 10.8,
                "GA": 5,
                "xGA": 7.3,
                "PTS": 7,
                "xPTS": 7.1,
            },
            {
                "Team": "Arsenal",
                "MP": 4,
                "G": 8,
                "xG": 8.2,
                "GA": 1,
                "xGA": 3.4,
                "PTS": 12,
                "xPTS": 9.1,
            },
        ]

    def test_scoring_uses_goals_minus_xg(self):
        views = build_actual_vs_expected(self.records)
        brighton = next(row for row in views["scoring"]["rows"] if row["team"] == "Brighton")

        self.assertEqual(brighton["difference"], 2.2)
        self.assertEqual(brighton["status"], "above")
        self.assertEqual(brighton["actual_per_match"], 3.25)
        self.assertEqual(brighton["expected_per_match"], 2.7)

    def test_defending_uses_xga_minus_goals_against(self):
        views = build_actual_vs_expected(self.records)
        arsenal = next(row for row in views["defending"]["rows"] if row["team"] == "Arsenal")

        self.assertEqual(arsenal["difference"], 2.4)
        self.assertEqual(arsenal["status"], "above")

    def test_points_uses_points_minus_xpts(self):
        views = build_actual_vs_expected(self.records)
        arsenal = next(row for row in views["points"]["rows"] if row["team"] == "Arsenal")

        self.assertEqual(arsenal["difference"], 2.9)
        self.assertEqual(arsenal["status"], "above")

    def test_each_view_sorts_by_overperformance(self):
        views = build_actual_vs_expected(self.records)

        self.assertEqual(views["scoring"]["rows"][0]["team"], "Brighton")
        self.assertEqual(views["defending"]["rows"][0]["team"], "Arsenal")
        self.assertEqual(views["points"]["rows"][0]["team"], "Arsenal")

    def test_near_expectation_is_level(self):
        views = build_actual_vs_expected(self.records)
        brighton = next(row for row in views["points"]["rows"] if row["team"] == "Brighton")

        self.assertEqual(brighton["difference"], -0.1)
        self.assertEqual(brighton["status"], "level")

    def test_display_names_and_invalid_rows(self):
        records = self.records + [
            {"Team": "No Games", "MP": 0, "G": 0, "xG": 0},
            {"Team": "Missing Data", "MP": 4, "G": None, "xG": 1},
            None,
        ]
        views = build_actual_vs_expected(records, {"Brighton": "Brighton & Hove"})

        scoring = views["scoring"]["rows"]
        self.assertEqual(len(scoring), 2)
        self.assertEqual(
            next(row for row in scoring if row["team"] == "Brighton")["label"],
            "Brighton & Hove",
        )


if __name__ == "__main__":
    unittest.main()
