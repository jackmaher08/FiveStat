import unittest

from fpl_projection import (
    eligibility_minutes,
    recent_minutes_eligible,
    expected_goals_conceded_deduction,
    expected_minutes,
    projected_fpl_points,
    projection_confidence,
)


class FplProjectionTests(unittest.TestCase):
    def test_eligibility_gate_scales_early_in_season(self):
        self.assertEqual(eligibility_minutes(4, "MID"), 198)
        self.assertEqual(eligibility_minutes(4, "DEF"), 216)
        self.assertEqual(eligibility_minutes(12, "MID"), 450)
        self.assertEqual(eligibility_minutes(12, "DEF"), 648)

    def test_availability_reduces_expected_minutes(self):
        available = expected_minutes(450, 5, 5, 100)
        doubtful = expected_minutes(450, 5, 5, 50)
        self.assertEqual(available, 90)
        self.assertEqual(doubtful, 45)

    def test_recent_eligibility_requires_sustained_usage(self):
        self.assertTrue(recent_minutes_eligible(350, 5, 4))
        self.assertFalse(recent_minutes_eligible(260, 5, 3))
        self.assertFalse(recent_minutes_eligible(250, 3, 3))
        self.assertFalse(recent_minutes_eligible(None, None, None))

    def test_recent_minutes_blend_can_restore_emerging_starter(self):
        # 384/9 season minutes and five starts: season-only projection
        # is below 60; sustained recent 80-minute games can restore it.
        original = expected_minutes(384, 5, 9, 100)
        revised = expected_minutes(
            384, 5, 9, 100, recent_minutes=400, recent_games=5
        )
        self.assertLess(original, 60)
        self.assertGreaterEqual(revised, 60)
        self.assertEqual(expected_minutes(384, 5, 9, 100, None, None), original)
        self.assertLess(
            expected_minutes(384, 5, 9, 50, recent_minutes=400, recent_games=5),
            60,
        )

    def test_position_specific_clean_sheet_points(self):
        defender = projected_fpl_points("DEF", 0, 0, 1, 0, 90)
        midfielder = projected_fpl_points("MID", 0, 0, 1, 0, 90)
        forward = projected_fpl_points("FWD", 0, 0, 1, 0, 90)
        self.assertEqual(defender, 6)
        self.assertEqual(midfielder, 3)
        self.assertEqual(forward, 2)

    def test_defender_conceded_deduction_is_non_linear(self):
        self.assertEqual(expected_goals_conceded_deduction(0), 0)
        self.assertGreater(expected_goals_conceded_deduction(2), expected_goals_conceded_deduction(1))

    def test_confidence_labels(self):
        cases = [(720, 8, 80, "High"), (270, 5, 60, "Medium"), (90, 5, 30, "Low")]
        for minutes, games, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(projection_confidence(minutes, games, expected), label)


if __name__ == "__main__":
    unittest.main()
