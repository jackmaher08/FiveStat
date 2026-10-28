"""Regression tests for projected FPL bonus and DEFCON components."""
import unittest
from fpl_projection import historical_extra_points


class HistoricalExtraPointsTests(unittest.TestCase):
    def test_no_minutes_or_data(self):
        self.assertEqual(historical_extra_points("DEF", 10, 100, 0, 90), (0.0, 0.0))
        self.assertEqual(historical_extra_points("MID", 0, 0, 720, 90), (0.0, 0.0))

    def test_bonus_is_season_points_adjusted_by_minutes(self):
        bonus, dc = historical_extra_points("FWD", 12, 0, 720, 90)
        self.assertEqual(bonus, 1.5)
        self.assertEqual(dc, 0.0)

    def test_defender_defcon_probability_never_exceeds_two(self):
        _, dc = historical_extra_points("DEF", 0, 96, 720, 90)
        self.assertGreater(dc, 1.0)
        self.assertLessEqual(dc, 2.0)

    def test_midfielder_has_higher_threshold(self):
        _, d = historical_extra_points("DEF", 0, 80, 720, 90)
        _, m = historical_extra_points("MID", 0, 80, 720, 90)
        self.assertGreater(d, m)

    def test_goalkeeper_not_defcon_eligible(self):
        self.assertEqual(historical_extra_points("GK", 0, 800, 720, 90)[1], 0.0)


if __name__ == "__main__":
    unittest.main()
