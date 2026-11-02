"""Fail-safe inputs for a temporary Understat outage.

FPL is authoritative for the fixture calendar and actual results. Understat
provides match IDs and xG; never use stale goals or invent missing xG.
"""

import time

import pandas as pd


FIXTURE_COLUMNS = [
    "round_number", "date", "home_team", "away_team", "result", "id",
    "isResult", "home_goals", "away_goals", "home_xG", "away_xG",
]


def fetch_understat_league(method, season="2026", attempts=3, sleep=time.sleep):
    """Retry only the external provider call; return None after exhaustion."""
    from understatapi import UnderstatClient

    for attempt in range(1, attempts + 1):
        try:
            with UnderstatClient() as client:
                data = getattr(client.league(league="EPL"), method)(season=season)
            if not data:
                raise ValueError("Understat returned no records")
            return data
        except Exception as exc:
            print(f"Understat {method} attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                sleep(2 ** (attempt - 1))
    print(f"Understat {method} unavailable; using last known values where possible")
    return None


def merge_fpl_and_understat_fixtures(fpl_df, understat_df, cached_df):
    """Preserve old Understat IDs/xG while keeping FPL results current.

    A newly encountered match gets its FPL ID only until its real Understat
    ID is available. A missing new xG is NaN, not zero.
    """
    if fpl_df.empty or fpl_df.duplicated(["round_number", "home_team", "away_team"]).any():
        raise ValueError("FPL fixtures are empty or contain duplicate matches")

    fresh = {} if understat_df is None else {
        (row["home_team"], row["away_team"]): row
        for _, row in understat_df.iterrows()
    }
    old = {} if cached_df is None else {
        (int(row["round_number"]), row["home_team"], row["away_team"]): row
        for _, row in cached_df.iterrows()
    }

    def preferred(field, recent, previous):
        for source in (recent, previous):
            if source is not None and field in source and pd.notna(source[field]):
                return source[field]
        return float("nan")

    records = []
    for _, f in fpl_df.iterrows():
        recent = fresh.get((f["home_team"], f["away_team"]))
        previous = old.get((int(f["round_number"]), f["home_team"], f["away_team"]))
        finished = bool(f["finished"])
        match_id = preferred("id", recent, previous)
        if pd.isna(match_id):
            match_id = f["fpl_id"]
        records.append({
            "round_number": int(f["round_number"]),
            "date": f["date"],
            "home_team": f["home_team"],
            "away_team": f["away_team"],
            "result": f["result"],
            "id": int(match_id),
            "isResult": finished,
            "home_goals": f["home_goals"] if finished else float("nan"),
            "away_goals": f["away_goals"] if finished else float("nan"),
            "home_xG": preferred("home_xG", recent, previous) if finished else float("nan"),
            "away_xG": preferred("away_xG", recent, previous) if finished else float("nan"),
        })
    output = pd.DataFrame(records, columns=FIXTURE_COLUMNS)
    complete = output[output["isResult"]]
    if (complete[["home_goals", "away_goals"]].isna().any().any()
            or output["id"].duplicated().any() or output["date"].isna().any()):
        raise ValueError("FPL fixture scores, dates or IDs are incomplete/duplicated")
    return output


def refresh_basic_league_table(fixtures, cached_table):
    """Update W/D/L, scores and points from FPL; keep xG metrics cached.

    The accompanying status flag MUST mark the xG columns stale. We never
    create fake zero xG or xPTS for newly completed fixtures.
    """
    if cached_table is None or cached_table.empty:
        raise ValueError("No cached xG table to preserve during Understat outage")
    cache = cached_table.set_index("Team")
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    if not set(teams).issubset(cache.index):
        raise ValueError("Cached xG table is missing current-season teams")
    finished = fixtures[fixtures["isResult"].astype(str).str.lower() == "true"]
    rows = []
    for team in teams:
        home = finished[finished["home_team"] == team]
        away = finished[finished["away_team"] == team]
        gf = int(home["home_goals"].sum() + away["away_goals"].sum())
        ga = int(home["away_goals"].sum() + away["home_goals"].sum())
        wins = int((home["home_goals"] > home["away_goals"]).sum()
                   + (away["away_goals"] > away["home_goals"]).sum())
        draws = int((home["home_goals"] == home["away_goals"]).sum()
                    + (away["away_goals"] == away["home_goals"]).sum())
        played = len(home) + len(away)
        losses = played - wins - draws
        cached = cache.loc[team]
        row = {
            "Team": team, "MP": played, "W": wins, "D": draws, "L": losses,
            "G": gf, "GD": gf - ga, "GA": ga, "PTS": wins * 3 + draws,
        }
        for col in ("xG", "npxG", "xGA", "npxGA", "xPTS"):
            row[col] = cached[col]
        # Historical expected-vs-actual deltas must not mix stale expected
        # metrics with newer FPL actuals. Retain the entire cached deltas.
        for col in ("xG +/-", "xGA +/-", "xPTS +/-"):
            row[col] = cached[col]
        rows.append(row)
    columns = ["Team", "MP", "W", "D", "L", "G", "GD", "GA", "xG",
               "npxG", "xG +/-", "xGA", "npxGA", "xGA +/-",
               "PTS", "xPTS", "xPTS +/-"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["PTS", "GD", "G"], ascending=False
    ).reset_index(drop=True)
