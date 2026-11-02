"""Regression tests for the Understat-outage FPL fallback."""
import unittest

import pandas as pd

from understat_fallback import merge_fpl_and_understat_fixtures, refresh_basic_league_table


class UnderstatFallbackTests(unittest.TestCase):
    def setUp(self):
        self.fpl = pd.DataFrame([
            {"round_number": 9, "date": "02/11/2026 20:00", "home_team": "Arsenal",
             "away_team": "Hull City", "result": "2 - 1", "finished": True,
             "home_goals": 2, "away_goals": 1, "fpl_id": 93},
            {"round_number": 10, "date": "09/11/2026 20:00", "home_team": "Hull City",
             "away_team": "Arsenal", "result": "", "finished": False,
             "home_goals": None, "away_goals": None, "fpl_id": 104},
        ])
        self.cached = pd.DataFrame([
            {"round_number": 9, "home_team": "Arsenal", "away_team": "Hull City",
             "id": 31001, "home_xG": 1.4, "away_xG": 0.7},
        ])
        self.table = pd.DataFrame([
            {"Team": team, "xG": 1.4, "npxG": 1.2, "xGA": 0.7,
             "npxGA": 0.6, "xPTS": 1.8, "xG +/-": 0.4,
             "xGA +/-": -0.3, "xPTS +/-": 0.8}
            for team in ("Arsenal", "Hull City")
        ])

    def test_fpl_scores_override_stale_results_but_preserve_ids_and_xg(self):
        updated = merge_fpl_and_understat_fixtures(self.fpl, None, self.cached)
        actual = updated.iloc[0]
        self.assertEqual((actual["home_goals"], actual["away_goals"]), (2, 1))
        self.assertEqual(actual["id"], 31001)
        self.assertEqual((actual["home_xG"], actual["away_xG"]), (1.4, 0.7))

    def test_new_fixture_keeps_schema_and_does_not_invent_xg(self):
        updated = merge_fpl_and_understat_fixtures(self.fpl, None, self.cached)
        self.assertEqual(updated.iloc[1]["id"], 104)
        self.assertFalse(updated.iloc[1]["isResult"])
        self.assertTrue(pd.isna(updated.iloc[1]["home_xG"]))

    def test_recently_finished_fixture_without_xg_stays_missing(self):
        self.fpl.loc[1, ["finished", "result", "home_goals", "away_goals"]] = [True, "1 - 0", 1, 0]
        updated = merge_fpl_and_understat_fixtures(self.fpl, None, self.cached)
        self.assertEqual(updated.iloc[1]["home_goals"], 1)
        self.assertTrue(pd.isna(updated.iloc[1]["home_xG"]))

    def test_fresh_understat_xg_updates_without_overwriting_scores(self):
        fresh = pd.DataFrame([{"home_team": "Arsenal", "away_team": "Hull City",
                               "id": 31001, "home_xG": 2.3,
                               "away_xG": 0.3, "home_goals": 7}])
        updated = merge_fpl_and_understat_fixtures(self.fpl, fresh, self.cached)
        self.assertEqual(updated.iloc[0]["home_xG"], 2.3)
        self.assertEqual(updated.iloc[0]["home_goals"], 2)

    def test_fpl_finished_match_without_score_fails(self):
        self.fpl.loc[0, "home_goals"] = None
        with self.assertRaises(ValueError):
            merge_fpl_and_understat_fixtures(self.fpl, None, self.cached)

    def test_standings_refresh_and_caches_stale_xg_deltas(self):
        fixtures = merge_fpl_and_understat_fixtures(self.fpl, None, self.cached)
        table = refresh_basic_league_table(fixtures, self.table).set_index("Team")
        self.assertEqual(int(table.loc["Arsenal", "PTS"]), 3)
        self.assertEqual(int(table.loc["Hull City", "GA"]), 2)
        self.assertEqual(table.loc["Arsenal", "xG"], 1.4)
        self.assertEqual(table.loc["Arsenal", "xG +/-"], 0.4)

    def test_missing_cached_team_rejected(self):
        fixtures = merge_fpl_and_understat_fixtures(self.fpl, None, self.cached)
        with self.assertRaises(ValueError):
            refresh_basic_league_table(fixtures, self.table.iloc[:1])


if __name__ == "__main__":
    unittest.main()
