"""Fail the update workflow when the FPL feature inputs are unusable."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl_projection import eligibility_minutes


TABLES = Path("data/tables")


def fail(message: str) -> None:
    raise SystemExit(f"FPL validation failed: {message}")


def main() -> None:
    required_paths = {
        "players": TABLES / "fpl_player_data.csv",
        "fixtures": TABLES / "fixture_data.csv",
        "probabilities": TABLES / "fixture_probabilities.csv",
        "deadline": TABLES / "next_deadline.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        fail(f"missing files: {', '.join(missing)}")

    players = pd.read_csv(required_paths["players"])
    fixtures = pd.read_csv(required_paths["fixtures"])
    probabilities = pd.read_csv(required_paths["probabilities"])
    deadline = json.loads(required_paths["deadline"].read_text())

    player_columns = {
        "fpl_name", "web_name", "team", "position", "minutes", "starts",
        "status", "xg", "xa", "chance_of_playing_next_round",
    }
    missing_columns = sorted(player_columns - set(players.columns))
    if missing_columns:
        fail(f"player data missing columns: {missing_columns}")
    if len(players) < 300:
        fail(f"only {len(players)} FPL players were saved")

    current_gw = int(deadline["gw"])
    fixtures["round_number"] = pd.to_numeric(fixtures["round_number"], errors="coerce")
    fixtures["isResult"] = fixtures["isResult"].astype(str).str.lower() == "true"
    upcoming = fixtures[(~fixtures["isResult"]) & (fixtures["round_number"] == current_gw)]
    if upcoming.empty:
        fail(f"no unplayed fixtures found for GW{current_gw}")

    probability_columns = {"home_team", "away_team", "home_xg", "away_xg", "home_cs_prob", "away_cs_prob"}
    if probability_columns - set(probabilities.columns):
        fail("fixture probabilities are missing xG or clean-sheet fields")

    completed = fixtures[fixtures["isResult"]]
    games_by_team = pd.concat([completed["home_team"], completed["away_team"]]).value_counts().to_dict()
    eligible = 0
    for row in players.itertuples():
        games = int(games_by_team.get(row.team, 0))
        if row.status in ("a", "d") and row.minutes >= eligibility_minutes(games, row.position):
            eligible += 1
    if eligible < 20:
        fail(f"only {eligible} players pass the season-aware reliability gate")

    numeric = players[["minutes", "xg", "xa"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        fail("player data contains NaN or infinite core metrics")

    print(f"FPL validation passed: GW{current_gw}, {len(upcoming)} fixtures, {eligible} eligible players")


if __name__ == "__main__":
    main()
