import unittest
from datetime import datetime, timezone
import pandas as pd
from fpl_deadlines import next_deadline_event, display_gameweek

NOW = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)


class FPLDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = pd.DataFrame({"round_number": [10, 10, 11, 11], "isResult": [True, False, False, False]})

    def test_future_deadline_beats_lagging_is_next(self):
        events = [
            {"id": 10, "deadline_time": "2026-11-06T18:30:00Z", "is_next": True},
            {"id": 11, "deadline_time": "2026-11-13T18:30:00Z"},
        ]
        self.assertEqual(next_deadline_event(events, NOW)["id"], 11)

    def test_expired_cached_deadline_advances_during_gameweek(self):
        old = {"gw": 10, "deadline_time": "2026-11-06T18:30:00Z"}
        self.assertEqual(display_gameweek(old, self.fixtures, NOW), 11)

    def test_future_deadline_remains_current(self):
        record = {"gw": 11, "deadline_time": "2026-11-13T18:30:00Z"}
        self.assertEqual(display_gameweek(record, self.fixtures, NOW), 11)

    def test_missing_deadline_uses_unplayed_fixtures(self):
        self.assertEqual(display_gameweek(None, self.fixtures, NOW), 10)

    def test_no_future_deadline(self):
        self.assertIsNone(next_deadline_event([{"id": 10, "deadline_time": "2026-11-06T18:30:00Z"}], NOW))


if __name__ == "__main__":
    unittest.main()
