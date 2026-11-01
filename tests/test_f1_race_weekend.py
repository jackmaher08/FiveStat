"""Regression coverage for race-day selection and the post-qualifying transition."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import f1_data_loader as f1
from scripts import refresh_f1_qualifying as refresh


MEXICO = {
    "season": 2026, "round": 19, "name": "Mexico City Grand Prix",
    "date": "2026-11-01", "time": "20:00:00Z", "circuit_id": "rodriguez",
}
BRAZIL = {
    "season": 2026, "round": 20, "name": "Brazilian Grand Prix",
    "date": "2026-11-08", "time": "17:00:00Z", "circuit_id": "interlagos",
}


def at(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


class RaceWeekendTests(unittest.TestCase):
    def setUp(self):
        f1.invalidate_f1_cache()

    def tearDown(self):
        f1.invalidate_f1_cache()

    def test_mexico_remains_next_on_sunday_before_and_during_race(self):
        with patch.object(f1, "load_f1_schedule", return_value=[MEXICO, BRAZIL]), \
             patch.object(f1, "load_f1_race_results", return_value=[]):
            for stamp in ("2026-10-31T23:59:00Z", "2026-11-01T00:12:00Z",
                          "2026-11-01T20:01:00Z"):
                self.assertEqual(f1.get_next_race_info(now=at(stamp))["round"], 19)

    def test_race_results_advance_to_brazil(self):
        with patch.object(f1, "load_f1_schedule", return_value=[MEXICO, BRAZIL]), \
             patch.object(f1, "load_f1_race_results", return_value=[{}] * 20):
            self.assertEqual(f1.get_next_race_info(now=at("2026-11-01T22:00:00Z"))["round"], 20)

    def test_unavailable_results_do_not_leave_stale_race_forever(self):
        with patch.object(f1, "load_f1_schedule", return_value=[MEXICO, BRAZIL]), \
             patch.object(f1, "load_f1_race_results", return_value=[]):
            self.assertEqual(f1.get_next_race_info(now=at("2026-11-02T02:00:00Z"))["round"], 20)

    def test_quali_ingestion_saves_only_published_complete_data(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [{
                "position": str(i), "Driver": {"driverId": f"id{i}", "code": f"D{i}",
                "givenName": "Driver", "familyName": str(i)},
                "Constructor": {"name": "Test", "constructorId": "test"},
                "Q1": "1:22.000", "Q2": None, "Q3": None,
            } for i in range(1, 21)]
            class Response:
                def __init__(self, data):
                    self.data = data
                def raise_for_status(self):
                    pass
                def json(self):
                    return {"MRData": {"RaceTable": {"Races": [
                        {"QualifyingResults": self.data}]}}}
            with patch.object(refresh, "load_f1_schedule", return_value=[MEXICO, BRAZIL]), \
                 patch.object(refresh, "get_upcoming_f1_races", return_value=[MEXICO]):
                path = Path(folder) / "2026_19_qualifying.json"
                fetch = lambda url, timeout: Response([])
                self.assertEqual(refresh.refresh_qualifying(at("2026-10-31T22:00:00Z"), fetch, Path(folder)), 0)
                self.assertFalse(path.exists())
                fetch = lambda url, timeout: Response(rows)
                self.assertEqual(refresh.refresh_qualifying(at("2026-10-31T22:00:00Z"), fetch, Path(folder)), 1)
                self.assertEqual(len(json.loads(path.read_text())), 20)
                self.assertEqual(refresh.refresh_qualifying(at("2026-10-31T22:00:00Z"), fetch, Path(folder)), 0)

    def test_prediction_switches_to_actual_qualifying_without_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(f1, "RACES_DIR", folder), \
                 patch.object(f1, "load_f1_schedule", return_value=[MEXICO]), \
                 patch.object(f1, "calculate_driver_pace_rating", return_value={
                     "AAA": {"name": "Driver A", "constructor": "Team A", "constructor_id": "a", "dpr": 0.0},
                     "BBB": {"name": "Driver B", "constructor": "Team B", "constructor_id": "b", "dpr": 0.1},
                 }), \
                 patch.object(f1, "calculate_constructor_pace_index", return_value={
                     "a": {"cpi": 0.0}, "b": {"cpi": 0.1}
                 }), \
                 patch.object(f1, "load_f1_driver_standings", return_value=[]):
                before = f1.calculate_race_predictions(2026, 19)
                self.assertEqual(before["mode"], "pre_quali")
                self.assertEqual(next(d["grid"] for d in before["predictions"] if d["code"] == "BBB"), 2)
                quali = [{
                    "code": code, "name": code, "constructor": "Team",
                    "constructor_id": constructor, "position": pos,
                    "q1": "1:22.000", "q2": None, "q3": None,
                } for code, constructor, pos in (("AAA", "a", 2), ("BBB", "b", 1))]
                Path(folder, "2026_19_qualifying.json").write_text(json.dumps(quali))
                after = f1.calculate_race_predictions(2026, 19)
                self.assertEqual(after["mode"], "post_quali")
                self.assertTrue(after["has_quali"])
                self.assertEqual(next(d["grid"] for d in after["predictions"] if d["code"] == "BBB"), 1)


if __name__ == "__main__":
    unittest.main()
