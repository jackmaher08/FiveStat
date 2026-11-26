import json
import unittest

import pandas as pd

from fpl_team_charts import build_team_chart_windows


class TeamChartTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = pd.DataFrame([
            (12, "A", "B", False),
            (12, "C", "A", False),  # A has a double GW; D blanks.
            (13, "B", "C", False),
            (16, "D", "A", False),
            (17, "A", "D", False),  # Outside next five.
            (38, "C", "D", False),
            (11, "B", "A", True),
            (13, "D", "C", True),
            (39, "B", "D", False),
        ], columns=["round_number", "home_team", "away_team", "isResult"])
        self.probs = pd.DataFrame([
            ("A", "B", 2.0, 1.0, .6, .2),
            ("C", "A", 1.5, 2.5, .1, .7),
            ("B", "C", .8, 1.1, .3, .4),
            ("D", "A", 1.2, 1.8, .25, .35),
            ("A", "D", 1.6, 1.4, .2, .3),
            ("C", "D", 1.7, .7, .45, .15),
            ("B", "A", 3, 4, .1, .2),
            ("D", "C", 3, 4, .1, .2),
            ("B", "D", 3, 4, .1, .2),
        ], columns=["home_team", "away_team", "home_xg", "away_xg", "home_cs_prob", "away_cs_prob"])

    def windows(self, gw=12):
        return build_team_chart_windows(
            self.fixtures, self.probs, ["A", "B", "C", "D"], {"A": "Team A"}, gw
        )

    def test_current_double_and_blank_gameweek(self):
        a, b, c, d = self.windows()["by_gw"]["12"]
        self.assertEqual(a, {"team": "Team A", "fixtures": 2, "xg": 4.5,
                             "xga": 2.5, "cs": 88.0})
        self.assertEqual(b["cs"], 20.0)
        self.assertEqual(c["xga"], 2.5)
        self.assertEqual(d, {"team": "D", "fixtures": 0, "xg": None, "xga": None, "cs": None})

    def test_next_five_uses_expected_clean_sheets(self):
        a, b, c, d = self.windows()["next5"]
        self.assertEqual(a["fixtures"], 3)
        self.assertEqual(a["xg"], 6.3)
        self.assertEqual(a["xga"], 3.7)
        self.assertEqual(a["cs"], 1.65)
        self.assertEqual(d["fixtures"], 1)

    def test_later_gameweeks_and_end_of_season(self):
        w = self.windows()
        self.assertEqual(w["weeks"], [12, 13, 16, 17, 38])
        self.assertEqual(w["by_gw"]["38"][2]["cs"], 45.0)
        end = self.windows(38)
        self.assertEqual(end["weeks"], [38])
        self.assertEqual(end["next5"][2]["cs"], .45)
        self.assertEqual(self.windows(39), {"weeks": [], "by_gw": {}, "next5": []})

    def test_missing_or_invalid_model_values_are_not_zero_projections(self):
        self.probs.loc[0, "home_xg"] = float("nan")
        self.probs.loc[5, "away_cs_prob"] = 1.1
        w = self.windows()
        self.assertNotIn(12, w["weeks"])
        self.assertNotIn(38, w["weeks"])
        self.assertIn(13, w["weeks"])
        self.assertIsNone(w["by_gw"]["12"][0]["xg"])
        self.assertIsNone(w["next5"][1]["xga"])
        json.dumps(w, allow_nan=False)

    def test_duplicate_probabilities_and_fixtures_are_not_double_counted(self):
        self.probs = pd.concat([self.probs, self.probs.iloc[[5]]])
        self.fixtures = pd.concat([self.fixtures, self.fixtures.iloc[[2]]])
        w = self.windows()
        self.assertNotIn(38, w["weeks"])
        self.assertNotIn(13, w["weeks"])
        self.assertIsNone(w["next5"][1]["xg"])
        self.assertIn(12, w["weeks"])


if __name__ == "__main__":
    unittest.main()
