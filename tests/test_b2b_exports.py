import csv
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from scripts.export_b2b_data import (
    build_f1_exports, build_fpl_exports, export_fpl, save_exports,
)

STAMP = "2026-10-21T12:00:00Z"


class B2BExportTests(unittest.TestCase):
    def fpl_context(self):
        player = {
            "fpl_id": 12, "name": "Sample Player", "team": "Chelsea", "position": "MID",
            "price": 8.5, "ownership": 15.5, "expected_minutes": 78.0,
            "confidence": "High", "gw1_xg": 0.2, "gw3_xg": 0.5, "gw5_xg": 0.9,
            "gw1_xa": 0.1, "gw3_xa": 0.3, "gw5_xa": 0.4,
            "gw1_pts": 4.1, "gw3_pts": 12.5, "gw5_pts": 21.1, "gw5_value": 2.48,
            "gw1_cs": 0.3, "gw3_cs": 0.9, "gw5_cs": 1.5,
            "gw1_fixture": "ARS (H)", "gw1_diff": "medium",
            "fixtures": [{"gw": 8, "label": "ARS (H)", "proj_xg": 0.2,
                          "proj_xa": 0.1, "proj_cs": 0.3, "proj_fpl": 4.1, "diff": "medium"}],
        }
        return {
            "captaincy_status": "ready", "current_gw": 8,
            "next_deadline": "2026-10-23T17:30:00Z",
            "all_player_projections": [player], "captain_picks": [player],
            "xg_data": [{"team": "Chelsea", "gw1": 1.5, "gw3": 4.2, "gw5": 7.3}],
            "xga_data": [{"team": "Chelsea", "gw1": 1.0, "gw3": 2.5, "gw5": 4.5}],
            "cs_data": [{"team": "Chelsea", "gw1": 30.0, "gw3": 0.9, "gw5": 1.5}],
            "fixture_ticker": [{"team": "Chelsea", "fixtures": [{"gw": 8,
                "label": "ARS (H)", "difficulty": "medium", "is_dgw": False}]}],
        }

    def test_fpl_json_and_all_five_csvs_roundtrip(self):
        payload, tables = build_fpl_exports(self.fpl_context(), STAMP)
        self.assertEqual(len(tables), 5)
        self.assertEqual(payload["team_outlook"][0]["expected_clean_sheets_gw1"], 0.3)
        self.assertEqual(tables["player_fixture_projections.csv"][1][0]["fpl_id"], 12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_exports("fpl", payload, tables, root)
            saved = json.loads((root / "fpl/predictions.json").read_text())
            self.assertEqual(saved["player_projections"][0]["fpl_id"], 12)
            with (root / "fpl/player_projections.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            self.assertEqual(len(list((root / "fpl").glob("*.csv"))), 5)

    def test_fpl_rejects_empty_or_fallback_context(self):
        context = self.fpl_context()
        context["captaincy_status"] = "error"
        with self.assertRaisesRegex(ValueError, "not ready"):
            build_fpl_exports(context, STAMP)
        context["captaincy_status"] = "ready"
        context["all_player_projections"] = []
        with self.assertRaisesRegex(ValueError, "No eligible"):
            build_fpl_exports(context, STAMP)

    def test_f1_roundtrip_and_percentage_units(self):
        race = {"season": 2026, "round": 19, "name": "Example Grand Prix", "date": "2026-10-25"}
        driver = {"code": "ABC", "name": "Example Driver", "constructor": "Example Team",
                  "grid": 2, "dpr": -0.1, "cpi": 0.2, "win_prob": 23.5,
                  "podium_prob": 62.0, "points_prob": 94.0, "expected_pos": 4.2, "xfp": 30.2}
        fantasy = {"season_projection": [{"code": "ABC", "name": "Example Driver",
            "constructor": "Example Team", "projected_total_xfp": 100.0}],
            "transfer_targets": {1: [{"code": "ABC", "name": "Example Driver",
                "constructor": "Example Team", "total_xfp": 30.2, "per_race_xfp": 30.2}]}}
        payload, tables = build_f1_exports(
            [race], {"ABC": {"name": "Example Driver", "dpr": -0.1}},
            {"example": {"name": "Example Team", "cpi": 0.2}},
            [{"race": race, "mode": "pre_quali", "predictions": [driver]}], fantasy, STAMP,
        )
        self.assertEqual(len(tables), 4)
        self.assertEqual(tables["race_predictions.csv"][1][0]["win_probability_percent"], 23.5)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_exports("f1", payload, tables, root)
            saved = json.loads((root / "f1/predictions.json").read_text())
            self.assertEqual(saved["upcoming_race_predictions"][0]["predictions"][0]["xfp"], 30.2)
            with (root / "f1/fantasy_transfer_targets.csv").open() as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["horizon_races"], "1")

    def test_f1_no_upcoming_races_writes_headers(self):
        payload, tables = build_f1_exports(
            [{"season": 2026}], {}, {}, [], {"season_projection": [], "transfer_targets": {}}, STAMP,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_exports("f1", payload, tables, root)
            self.assertEqual(len((root / "f1/race_predictions.csv").read_text().splitlines()), 1)

    def test_f1_rejects_empty_forecast(self):
        with self.assertRaisesRegex(ValueError, "no predictions"):
            build_f1_exports([], {}, {}, [{"race": {"round": 3}, "mode": "pre_quali", "predictions": []}], {}, STAMP)

    def test_nan_is_rejected_before_files_written(self):
        payload, tables = build_fpl_exports(self.fpl_context(), STAMP)
        payload["player_projections"][0]["gw1_pts"] = float("nan")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                save_exports("fpl", payload, tables, root)
            self.assertFalse((root / "fpl/predictions.json").exists())

    def test_export_fpl_uses_live_view_context(self):
        view = Mock(return_value=self.fpl_context())
        with patch.dict("sys.modules", {"app": SimpleNamespace(fpl=view)}), \
             patch("scripts.export_b2b_data.save_exports") as save:
            export_fpl()
        view.assert_called_once_with(export_mode=True)
        self.assertEqual(save.call_args.args[0], "fpl")


if __name__ == "__main__":
    unittest.main()
