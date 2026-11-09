"""EPL week rollover regression tests, including stale Understat/FPL status."""
import unittest
from datetime import datetime, timezone

import pandas as pd

from epl_gameweek import select_epl_gameweek


def fixtures(last_finished=False, last_date="08/11/2026 16:30"):
    return pd.DataFrame([
        {"round_number": 10, "date": "07/11/2026 15:00", "isResult": True},
        {"round_number": 10, "date": last_date, "isResult": last_finished},
        {"round_number": 11, "date": "21/11/2026 12:30", "isResult": False},
    ])


def utc(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=timezone.utc)


class EPLGameweekTests(unittest.TestCase):
    def test_does_not_advance_at_final_kickoff(self):
        self.assertEqual(select_epl_gameweek(fixtures(), utc(8, 16, 30)), 10)

    def test_does_not_advance_while_final_match_can_be_in_progress(self):
        self.assertEqual(select_epl_gameweek(fixtures(), utc(8, 18, 30)), 10)

    def test_advances_immediately_when_all_fixtures_finished(self):
        self.assertEqual(select_epl_gameweek(fixtures(last_finished=True), utc(8, 18)), 11)

    def test_advances_after_3h_when_completion_flags_stale(self):
        self.assertEqual(select_epl_gameweek(fixtures(), utc(8, 19, 30)), 11)

    def test_next_week_remains_selected_until_its_matches_finish(self):
        self.assertEqual(select_epl_gameweek(fixtures(), utc(9, 10)), 11)

    def test_postponement_does_not_block_week_rollover(self):
        self.assertEqual(
            select_epl_gameweek(fixtures(last_date="20/12/2026 16:30"), utc(9, 10)), 11
        )

    def test_missing_kickoff_never_forces_rollover(self):
        self.assertEqual(
            select_epl_gameweek(fixtures(last_date=""), utc(9, 10)), 10
        )

    def test_empty_fixtures(self):
        self.assertIsNone(select_epl_gameweek(pd.DataFrame()))

if __name__ == "__main__":
    unittest.main()
