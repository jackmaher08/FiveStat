"""Export existing FiveStat model outputs to internal, versioned CSV/JSON datasets.

This is a packaging step, not a public API or a licence to redistribute source data.
Run after the corresponding football/F1 pipeline has refreshed its input files.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

EXPORT_ROOT = Path("data/exports")
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not silently emit non-standard JSON containing NaN or Infinity.
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(content + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Enforce JSON-compatible finite numeric values for every exported row.
            json.dumps(row, allow_nan=False)
            writer.writerow(row)


def build_fpl_exports(context: dict, generated_at: str) -> tuple[dict, dict[str, tuple[list[str], list[dict]]]]:
    if context.get("captaincy_status") != "ready":
        raise ValueError("FPL projections are not ready; refusing to publish an empty/error export")
    players = context.get("all_player_projections") or []
    if not players:
        raise ValueError("No eligible FPL player projections available")
    current_gw = int(context["current_gw"])
    if not all(p.get("fixtures") and p.get("fpl_id") is not None for p in players):
        raise ValueError("FPL player projections are missing player IDs or fixture rows")

    player_fields = [
        "fpl_id", "name", "team", "position", "price", "ownership", "expected_minutes",
        "confidence", "gw1_xg", "gw3_xg", "gw5_xg", "gw1_xa", "gw3_xa", "gw5_xa",
        "gw1_pts", "gw3_pts", "gw5_pts", "gw5_value", "gw1_cs", "gw3_cs", "gw5_cs",
        "gw1_fixture", "gw1_diff",
    ]
    player_rows = [{k: p.get(k) for k in player_fields} for p in players]
    fixture_fields = [
        "fpl_id", "name", "team", "position", "gw", "opponent_fixture", "projected_xg",
        "projected_xa", "clean_sheet_probability", "projected_fpl_points", "difficulty",
    ]
    fixture_rows = [
        {
            "fpl_id": p["fpl_id"], "name": p["name"], "team": p["team"],
            "position": p["position"], "gw": f["gw"], "opponent_fixture": f["label"],
            "projected_xg": f["proj_xg"], "projected_xa": f["proj_xa"],
            "clean_sheet_probability": f["proj_cs"],
            "projected_fpl_points": f["proj_fpl"], "difficulty": f["diff"],
        }
        for p in players for f in p["fixtures"]
    ]
    captain_fields = ["rank", *player_fields]
    captain_rows = [
        {"rank": rank, **{k: p.get(k) for k in player_fields}}
        for rank, p in enumerate(context.get("captain_picks", []), 1)
    ]

    xg = {r["team"]: r for r in context["xg_data"]}
    xga = {r["team"]: r for r in context["xga_data"]}
    cs = {r["team"]: r for r in context["cs_data"]}
    team_fields = [
        "team", "xg_gw1", "xg_gw3", "xg_gw5", "xga_gw1", "xga_gw3", "xga_gw5",
        "expected_clean_sheets_gw1", "expected_clean_sheets_gw3", "expected_clean_sheets_gw5",
    ]
    if set(xg) != set(xga) or set(xg) != set(cs):
        raise ValueError("FPL team outlook xG/xGA/CS teams do not match")
    team_rows = []
    for team in sorted(xg):
        team_rows.append({
            "team": team,
            **{f"xg_{gw}": xg[team][gw] for gw in ("gw1", "gw3", "gw5")},
            **{f"xga_{gw}": xga[team][gw] for gw in ("gw1", "gw3", "gw5")},
            # UI uses a percentage for GW1, but an expected count for GW3/5.
            "expected_clean_sheets_gw1": round(cs[team]["gw1"] / 100, 3),
            "expected_clean_sheets_gw3": cs[team]["gw3"],
            "expected_clean_sheets_gw5": cs[team]["gw5"],
        })

    difficulty_fields = ["team", "gw", "fixture", "difficulty", "is_double_gameweek", "is_blank_gameweek"]
    difficulty_rows = [
        {
            "team": team["team"], "gw": fixture["gw"], "fixture": fixture["label"],
            "difficulty": fixture["difficulty"], "is_double_gameweek": fixture["is_dgw"],
            "is_blank_gameweek": fixture["difficulty"] == "blank",
        }
        for team in context["fixture_ticker"] for fixture in team["fixtures"]
    ]
    payload = {
        "schema_version": SCHEMA_VERSION, "generated_at_utc": generated_at,
        "product": "fpl", "current_gameweek": current_gw,
        "next_deadline_utc": context.get("next_deadline"),
        "coverage": "Eligible outfield players; not the full FPL player universe",
        "units": {"ownership": "percent_0_to_100", "clean_sheet_probability": "fraction_0_to_1",
                  "projected_fpl_points": "points", "expected_clean_sheets": "expected_count"},
        "player_projections": players,
        "captaincy_rankings": context.get("captain_picks", []),
        "team_outlook": team_rows,
        "fixture_difficulty": difficulty_rows,
    }
    csv_outputs = {
        "player_projections.csv": (player_fields, player_rows),
        "player_fixture_projections.csv": (fixture_fields, fixture_rows),
        "captaincy_rankings.csv": (captain_fields, captain_rows),
        "team_outlook.csv": (team_fields, team_rows),
        "fixture_difficulty.csv": (difficulty_fields, difficulty_rows),
    }
    return payload, csv_outputs


def build_f1_exports(
    schedule: list[dict], dpr: dict, cpi: dict, races: list[dict], fantasy: dict,
    generated_at: str,
) -> tuple[dict, dict[str, tuple[list[str], list[dict]]]]:
    race_fields = [
        "season", "round", "race_name", "race_date", "mode", "driver_code", "driver_name",
        "constructor", "grid", "dpr", "cpi", "win_probability_percent", "podium_probability_percent",
        "points_probability_percent", "expected_position", "expected_fantasy_points",
    ]
    race_rows = []
    for race in races:
        meta = race["race"]
        if not race.get("predictions"):
            raise ValueError(f"F1 race {meta.get('round')} has no predictions")
        for p in race["predictions"]:
            race_rows.append({
                "season": meta["season"], "round": meta["round"], "race_name": meta["name"],
                "race_date": meta["date"], "mode": race["mode"],
                "driver_code": p["code"], "driver_name": p["name"], "constructor": p["constructor"],
                "grid": p.get("grid"), "dpr": p.get("dpr"), "cpi": p.get("cpi"),
                "win_probability_percent": p["win_prob"], "podium_probability_percent": p["podium_prob"],
                "points_probability_percent": p["points_prob"], "expected_position": p["expected_pos"],
                "expected_fantasy_points": p.get("xfp"),
            })
    pace_fields = ["type", "id", "name", "pace_delta_percent", "races_counted", "constructor"]
    pace_rows = [
        {"type": "driver", "id": code, "name": r.get("name", code),
         "pace_delta_percent": r["dpr"], "races_counted": r.get("races_counted"),
         "constructor": r.get("constructor")}
        for code, r in sorted(dpr.items())
    ] + [
        {"type": "constructor", "id": code, "name": r.get("name", code),
         "pace_delta_percent": r["cpi"], "races_counted": r.get("races_counted"),
         "constructor": r.get("name")}
        for code, r in sorted(cpi.items())
    ]
    season_fields = [
        "code", "name", "constructor", "championship_points", "championship_position",
        "completed_xfp", "projected_remaining_xfp", "projected_total_xfp", "races_remaining",
    ]
    season_rows = fantasy.get("season_projection", [])
    transfer_fields = [
        "horizon_races", "rank", "code", "name", "constructor", "total_xfp", "per_race_xfp",
        "championship_points",
    ]
    transfer_rows = [
        {"horizon_races": int(horizon), "rank": rank, **target}
        for horizon, targets in fantasy.get("transfer_targets", {}).items()
        for rank, target in enumerate(targets, 1)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION, "generated_at_utc": generated_at,
        "product": "f1", "season": schedule[0]["season"] if schedule else None,
        "units": {"race_probabilities": "percent_0_to_100", "pace_delta": "percent_vs_reference",
                  "expected_fantasy_points": "points"},
        "schedule": schedule,
        "driver_pace_ratings": dpr,
        "constructor_pace_index": cpi,
        "upcoming_race_predictions": races,
        "fantasy": fantasy,
    }
    csv_outputs = {
        "race_predictions.csv": (race_fields, race_rows),
        "pace_ratings.csv": (pace_fields, pace_rows),
        "fantasy_season_projections.csv": (season_fields, season_rows),
        "fantasy_transfer_targets.csv": (transfer_fields, transfer_rows),
    }
    return payload, csv_outputs


def save_exports(product: str, payload: dict, csv_outputs: dict, root: Path = EXPORT_ROOT) -> None:
    folder = root / product
    # Validate every payload before touching any on-disk export.
    json.dumps(payload, allow_nan=False)
    for _, (_, rows) in csv_outputs.items():
        for row in rows:
            json.dumps(row, allow_nan=False)
    write_json(folder / "predictions.json", payload)
    for filename, (fields, rows) in csv_outputs.items():
        write_csv(folder / filename, fields, rows)
    print(f"Exported {product.upper()} predictions to {folder} ({len(csv_outputs)} CSV tables + JSON)")


def export_fpl(root: Path = EXPORT_ROOT) -> None:
    # Reuse the exact model computations that drive the live /fpl page.
    # export_mode is an internal Python call, not a new HTTP endpoint.
    from app import fpl
    context = fpl(export_mode=True)
    if not isinstance(context, dict):
        raise ValueError("FPL view failed to return model context")
    payload, csv_outputs = build_fpl_exports(context, utc_now())
    save_exports("fpl", payload, csv_outputs, root)


def export_f1(root: Path = EXPORT_ROOT) -> None:
    from f1_data_loader import (
        calculate_constructor_pace_index, calculate_driver_pace_rating,
        get_f1_fantasy_data, get_f1_predictions_data, get_upcoming_f1_races, load_f1_schedule,
    )
    schedule = load_f1_schedule()
    if not schedule:
        raise ValueError("F1 schedule is empty")
    upcoming = get_upcoming_f1_races(schedule)
    races = []
    for race in upcoming:
        pred = get_f1_predictions_data(int(race["season"]), int(race["round"]))
        races.append({
            "race": race,
            "mode": pred["mode"],
            "has_qualifying": pred["has_quali"],
            "simulation_count": pred["n_simulations"],
            "predictions": pred["predictions"],
        })
    payload, csv_outputs = build_f1_exports(
        schedule, calculate_driver_pace_rating(), calculate_constructor_pace_index(),
        races, get_f1_fantasy_data(), utc_now(),
    )
    save_exports("f1", payload, csv_outputs, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=("fpl", "f1", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, default=EXPORT_ROOT)
    args = parser.parse_args()
    if args.sport in ("fpl", "all"):
        export_fpl(args.output_dir)
    if args.sport in ("f1", "all"):
        export_f1(args.output_dir)


if __name__ == "__main__":
    main()
