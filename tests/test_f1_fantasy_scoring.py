"""Regression tests for F1 Fantasy weekend scoring and simplified transfer view."""
import unittest
from unittest.mock import patch

import f1_data_loader as f1


def driver(code, grid, win, podium, points, expected):
    return {
        "code": code, "name": code, "constructor": "Test",
        "constructor_id": "test", "grid": grid, "win_prob": win,
        "podium_prob": podium, "points_prob": points, "expected_pos": expected,
    }


class FantasyProjectionTests(unittest.TestCase):
    def setUp(self):
        f1.invalidate_f1_cache()
        self.drivers = [
            driver("A", 1, 45, 70, 90, 3.0),
            driver("B", 2, 30, 60, 85, 4.0),
            driver("C", 3, 25, 50, 80, 5.0),
        ]

    def test_empty_race(self):
        self.assertEqual(f1.calculate_xfp([]), [])

    def test_unboosted_driver_components_sum_to_xfp(self):
        result = f1.calculate_xfp(
            self.drivers, race_meta={"has_sprint": False, "circuit_id": "monza"},
        )
        self.assertEqual(len(result), 3)
        self.assertEqual({d["code"] for d in result}, {"A", "B", "C"})
        for d in result:
            components = d["xfp_components"]
            self.assertIn("qualifying", components)
            self.assertIn("race_overtakes_est", components)
            self.assertIn("race_driver_of_day", components)
            self.assertNotIn("sprint_finish", components)
            self.assertAlmostEqual(d["xfp"], round(sum(components.values()), 1), delta=0.1)
        self.assertAlmostEqual(
            sum(d["xfp_components"]["race_fastest_lap"] for d in result), 10, delta=0.03,
        )
        self.assertAlmostEqual(
            sum(d["xfp_components"]["race_driver_of_day"] for d in result), 10, delta=0.03,
        )

    def test_sprint_added_only_for_sprint_round(self):
        standard = {
            d["code"]: d for d in f1.calculate_xfp(
                self.drivers, race_meta={"has_sprint": False},
            )
        }
        sprint = {
            d["code"]: d for d in f1.calculate_xfp(
                self.drivers, race_meta={"has_sprint": True},
            )
        }
        for code in standard:
            self.assertFalse(standard[code]["xfp_includes_sprint"])
            self.assertTrue(sprint[code]["xfp_includes_sprint"])
            self.assertIn("sprint_finish", sprint[code]["xfp_components"])
        self.assertAlmostEqual(
            sum(d["xfp_components"]["sprint_fastest_lap"] for d in sprint.values()),
            5, delta=0.03,
        )

    def test_pre_and_post_qualifying(self):
        drivers = [dict(self.drivers[0], best_quali_time=None)]
        pre = f1.calculate_xfp(drivers, race_meta={}, has_quali=False)
        post = f1.calculate_xfp(drivers, race_meta={}, has_quali=True)
        self.assertEqual(pre[0]["xfp_components"]["qualifying"], 10)
        self.assertEqual(post[0]["xfp_components"]["qualifying"], -5)

    def test_five_race_targets_preserve_cumulative_ranking_and_short_season(self):
        races = [
            {"season": 2026, "round": 21 + i, "name": f"Race {i}"}
            for i in range(2)
        ]
        with patch.object(f1, "get_upcoming_f1_races", return_value=races), \
             patch.object(f1, "calculate_race_predictions", return_value={
                 "predictions": self.drivers, "race_meta": {}, "has_quali": False,
             }), \
             patch.object(f1, "calculate_xfp", side_effect=lambda predictions, **kwargs: [
                 dict(d, xfp={"A": 12.0, "B": 9.0, "C": 3.0}[d["code"]])
                 for d in predictions
             ]), \
             patch.object(f1, "load_f1_driver_standings", return_value=[]):
            result = f1.calculate_transfer_targets()
        self.assertEqual(list(result), [5])
        self.assertEqual(result[5][0]["code"], "A")
        self.assertEqual(result[5][0]["total_xfp"], 24.0)
        self.assertEqual(len(result[5][0]["races"]), 2)


if __name__ == "__main__":
    unittest.main()
