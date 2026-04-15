import os
import re
import json
import requests
import pandas as pd
from understatapi import UnderstatClient
from io import StringIO

import sys
BOOKIE_ONLY = "--bookie-only" in sys.argv

FPL_SCHEDULE_TEAM_MAP = {
    1:  "Arsenal",
    2:  "Aston Villa",
    3:  "Bournemouth",
    4:  "Brentford",
    5:  "Brighton",
    6:  "Chelsea",
    7:  "Crystal Palace",
    8:  "Everton",
    9:  "Fulham",
    10: "Ipswich",
    11: "Leeds",
    12: "Leicester",
    13: "Liverpool",
    14: "Manchester City",
    15: "Manchester United",
    16: "Newcastle United",
    17: "Nottingham Forest",
    18: "Southampton",
    19: "Sunderland",
    20: "Tottenham Hotspur",
    21: "West Ham",
    22: "Wolverhampton Wanderers",
    23: "Burnley",
}

if not BOOKIE_ONLY:
    print("🔄 Fetching fixture schedule from FPL API...")
    fpl_fix_resp = requests.get(
        "https://fantasy.premierleague.com/api/fixtures/",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15
    )
    fpl_fix_resp.raise_for_status()
    fpl_fixtures_raw = fpl_fix_resp.json()

    fpl_bootstrap_resp = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15
    )
    fpl_bootstrap_resp.raise_for_status()
    fpl_bootstrap = fpl_bootstrap_resp.json()
    fpl_team_id_to_name = {
        t["id"]: t["name"] for t in fpl_bootstrap["teams"]
    }
    TEAM_NAME_NORMALISE = {
        "Man City":       "Manchester City",
        "Man Utd":        "Manchester United",
        "Spurs":          "Tottenham Hotspur",
        "Tottenham":      "Tottenham Hotspur",
        "Wolves":         "Wolverhampton Wanderers",
        "Newcastle":      "Newcastle United",
        "Nott'm Forest":  "Nottingham Forest",
    }

    fix_rows = []
    for f in fpl_fixtures_raw:
        gw = f.get("event")
        if not gw:
            continue
        home_raw = fpl_team_id_to_name.get(f["team_h"], str(f["team_h"]))
        away_raw = fpl_team_id_to_name.get(f["team_a"], str(f["team_a"]))
        home = TEAM_NAME_NORMALISE.get(home_raw, home_raw)
        away = TEAM_NAME_NORMALISE.get(away_raw, away_raw)
        kickoff = f.get("kickoff_time", "")
        date_str = ""
        if kickoff:
            try:
                from datetime import datetime as _dt
                dt = _dt.strptime(kickoff[:16], "%Y-%m-%dT%H:%M")
                date_str = dt.strftime("%d/%m/%Y %H:%M")
            except Exception:
                date_str = kickoff[:16]
        finished = bool(f.get("finished", False))
        h_score = f.get("team_h_score")
        a_score = f.get("team_a_score")
        result = ""
        if finished and h_score is not None and a_score is not None:
            result = f"{h_score} - {a_score}"
        fix_rows.append({
            "round_number": int(gw),
            "home_team":    home,
            "away_team":    away,
            "date":         date_str,
            "result":       result,
            "finished":     finished,
        })

    fixtures_df = pd.DataFrame(fix_rows)
    print(f"✅ FPL fixture schedule loaded ({len(fixtures_df)} fixtures, {fixtures_df['round_number'].nunique()} gameweeks)")

    # 📌 **Second Source: Understat** – fixtures/xG via understatapi
    understat_season = "2025"  # corresponds to the 2024/25 EPL season

    # Team name mapping for consistency
    team_name_mapping = {
        "Man City": "Manchester City",
        "Newcastle": "Newcastle United",
        "Spurs": "Tottenham Hotspur",
        "Tottenham": "Tottenham Hotspur",
        "Man Utd": "Manchester United",
        "Wolves": "Wolverhampton Wanderers",
        "Nott'm Forest": "Nottingham Forest",
    }

    # Pull all league matches from Understat
    with UnderstatClient() as understat_client:
        league_matches = understat_client.league(league="EPL").get_match_data(
            season=understat_season
        )

    # Parse fixture data
    fixture_data_temp = []
    for match in league_matches:
        fixture_entry = {
            "id": match.get("id"),
            "isResult": match.get("isResult"),
            "home_team": team_name_mapping.get(
                match["h"]["title"], match["h"]["title"]
            ),
            "away_team": team_name_mapping.get(
                match["a"]["title"], match["a"]["title"]
            ),
            "home_goals": int(match["goals"]["h"])
            if match.get("goals") and match["goals"].get("h") is not None
            else None,
            "away_goals": int(match["goals"]["a"])
            if match.get("goals") and match["goals"].get("a") is not None
            else None,
            "home_xG": round(float(match["xG"]["h"]), 2)
            if match.get("xG") and match["xG"].get("h") is not None
            else None,
            "away_xG": round(float(match["xG"]["a"]), 2)
            if match.get("xG") and match["xG"].get("a") is not None
            else None,
        }
        fixture_data_temp.append(fixture_entry)

    # Convert Understat data to DataFrame – make sure expected columns exist
    fixture_columns = [
        "id",
        "isResult",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xG",
        "away_xG",
    ]
    fixture_data_df = pd.DataFrame(fixture_data_temp, columns=fixture_columns)

    if fixture_data_df.empty:
        print(
            "⚠️ Understat fixture DataFrame is empty – "
            "continuing without Understat xG/goals."
        )

    # Merge fixturedownload fixtures with Understat xG/results
    fixture_data = pd.merge(
        fixtures_df[["round_number", "date", "home_team", "away_team", "result", "finished"]],
        fixture_data_df[
            [
                "id",
                "home_team",
                "away_team",
                "isResult",
                "home_goals",
                "away_goals",
                "home_xG",
                "away_xG",
            ]
        ],
        on=["home_team", "away_team"],
        how="left",
    )
    fixture_data["isResult"] = fixture_data["isResult"].fillna(fixture_data["finished"])
    fixture_data = fixture_data.drop(columns=["finished"])





    # Define the save directory
    save_dir = "data/tables"
    os.makedirs(save_dir, exist_ok=True)  # ✅ Ensure the directory exists

    # Define the file path
    file_path = os.path.join(save_dir, "fixture_data.csv")

    # ✅ Save the DataFrame as a CSV file
    fixture_data.to_csv(file_path, index=False)

    print(f"✅ fixture data saved to: {file_path}")

    # load next gw fixtures
    # Count how many fixtures per round have isResult == False
    round_counts = fixture_data[fixture_data["isResult"] == False].groupby("round_number").size()

    # Find the first round_number where at least 5 fixtures are still to be played
    next_round_number = round_counts[round_counts >= 5].index.min()

    # Filter the fixtures for that round
    next_gw_fixtures = fixture_data[
        (fixture_data["round_number"] == next_round_number) & (fixture_data["isResult"] == False)
    ][["round_number", "date", "home_team", "away_team"]]

    # Define the file path for saving
    next_gw_file_path = os.path.join(save_dir, "next_gw_fixtures.csv")

    # Save the next round of fixtures
    next_gw_fixtures.to_csv(next_gw_file_path, index=False)


    print(f"✅ next gw fixture data saved to: {next_gw_file_path}")

    # Historical fixture data

    # Load all seasons' data


    import io, sys, requests
    import pandas as pd

    start_year = 2016
    end_year   = 2025

    frames   = []
    failures = []

    # small helper to fetch CSVs with browser-like headers
    def fetch_csv(url: str) -> pd.DataFrame:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0 Safari/126.0"),
            "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url.strip(), headers=headers, timeout=30, allow_redirects=True)
        print(f"[debug] HTTP {r.status_code} | final_url={r.url} | redirects={len(r.history)}", file=sys.stderr)
        r.raise_for_status()  # will raise on 4xx/5xx so we can catch it
        # Try pandas from text; if it’s a binary CSV, fall back to content
        try:
            return pd.read_csv(io.StringIO(r.text))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(r.content))

    for year in range(start_year, end_year + 1):
        url = f"https://fixturedownload.com/download/epl-{year}-GMTStandardTime.csv"
        print(f"[info] fetching season {year}: {url!r}")

        try:
            frame = fetch_csv(url)

            # rename BEFORE appending (your previous code renamed after append)
            frame = frame.rename(columns={
                "Round Number": "round_number",
                "Home Team": "home_team",
                "Away Team": "away_team",
                "Date": "date",
                "Result": "result",
            })

            # sanity check: required columns present?
            required = {"home_team", "away_team", "date", "result"}
            missing  = required - set(map(str.lower, frame.columns))
            if missing:
                print(f"[warn] season {year}: missing expected columns: {missing}", file=sys.stderr)

            frame["Season"] = year
            frames.append(frame)

            print(f"[ok] season {year} loaded: {len(frame)} rows")

        except Exception as e:
            print(f"[ERROR] season {year} failed: {e}", file=sys.stderr)
            failures.append((year, url, repr(e)))
            continue  # keep going

    # Merge all season data that succeeded
    if not frames:
        raise RuntimeError("No seasons loaded; see failures above.")
    df = pd.concat(frames, ignore_index=True)
    print(f"[summary] loaded seasons: {len(frames)} ok, {len(failures)} failed")

    # If any failures, list them at the end (easy to scan)
    if failures:
        print("\n[failed seasons]")
        for yr, u, err in failures:
            print(f" - {yr}: {u} -> {err}")








    # Player data
    # 📌 Understat player stats via understatapi
    understat_season = "2025"  # same season as above

    with UnderstatClient() as understat_client:
        players_raw = understat_client.league(league="EPL").get_player_data(
            season=understat_season
        )

    # Parse data into a list of dicts
    player_rows = [
        {
            "Name": p.get("player_name"),
            "POS": p.get("position", ""),
            "Team": p.get("team_title", ""),
            "MP": int(p["games"]) if p.get("games") else 0,
            "Mins": int(p["time"]) if p.get("time") else 0,
            "G": int(p["goals"]) if p.get("goals") else 0,
            "xG": round(float(p["xG"]), 2) if p.get("xG") else 0.0,
            "NPG": int(p["npg"]) if p.get("npg") else 0,
            "NPxG": round(float(p["npxG"]), 2) if p.get("npxG") else 0.0,
            "A": int(p["assists"]) if p.get("assists") else 0,
            "xA": round(float(p["xA"]), 2) if p.get("xA") else 0.0,
            "YC": int(p["yellow_cards"]) if p.get("yellow_cards") else 0,
            "RC": int(p["red_cards"]) if p.get("red_cards") else 0,
        }
        for p in players_raw
    ]

    player_data = pd.DataFrame(player_rows)

    player_data["Team"] = player_data["Team"].replace(
        {"Tottenham": "Tottenham Hotspur"}
    )

    # Define the file path
    player_file_path = os.path.join(save_dir, "player_data.csv")

    # ✅ Save the DataFrame as a CSV file
    player_data.to_csv(player_file_path, index=False)

    print(f"✅ player data saved to: {player_file_path}")



    '''
    #TEMP USING THE FOLLOWING
    # ✅ Generate fresh league table (preseason alphabetical placeholder)

    # Get unique list of 2025 teams from fixtures
    teams_2025 = sorted(set(fixtures_df["home_team"]).union(set(fixtures_df["away_team"])))

    # Create empty table
    preseason_table = pd.DataFrame({
        "Team": teams_2025,
        "MP": 0, "W": 0, "D": 0, "L": 0, "G": 0, "GD": 0, "GA": 0,
        "xG": 0.0, "npxG": 0.0, "xG +/-": 0.0,
        "xGA": 0.0, "npxGA": 0.0, "xGA +/-": 0.0,
        "PTS": 0, "xPTS": 0.0, "xPTS +/-": 0.0
    })

    # Save the preseason table
    league_table_file_path = os.path.join(save_dir, "league_table_data.csv")
    preseason_table.to_csv(league_table_file_path, index=False)

    print("✅ League table reset for new season (preseason alphabetical order)")

    #END OF TEMP

    '''

    # Gathering league table data
    # Gathering league table data via understatapi (instead of scraping HTML)
    understat_season = "2025"  # same season you use elsewhere

    with UnderstatClient() as understat_client:
        # This returns the same kind of structure you were previously getting
        # from the `teamsData` JSON (teams with `history` lists etc.)
        fixture_results_df = understat_client.league(league="EPL").get_team_data(
            season=understat_season
        )



    # Prepare the list to store extracted data
    team_stats = []

    # Extract relevant fields for each team
    for team_id, team_info in fixture_results_df.items():
        team_name = team_info['title']  # Get the team name
        for match in team_info['history']:
            team_stats.append({
                "Team": team_name,
                "h_a": match["h_a"],
                "xG": round(float(match["xG"]), 1),
                "xGA": round(float(match["xGA"]), 1),  # Ensures rounding
                "npxG": round(float(match["npxG"]), 1),
                "npxGA": round(float(match["npxGA"]), 1),
                "G": int(match["scored"]),
                "Shots": int(match["missed"]),
                "W": int(match["wins"]),
                "D": int(match["draws"]),
                "L": int(match["loses"]),
                "PTS": int(match["pts"]),
                "xPTS": round(float(match["xpts"]), 1),
            })

    # Convert to DataFrame
    complete_fixture_results_df = pd.DataFrame(team_stats)

    complete_fixture_results_df["Team"] = complete_fixture_results_df["Team"].replace(team_name_mapping)

    # ✅ Calculate Matches Played (MP)
    matches_played = complete_fixture_results_df.groupby("Team").size().reset_index(name="MP")

    # Load fixture data
    fixture_data_file_path = os.path.join(save_dir, "fixture_data.csv")
    fixture_df = pd.read_csv(fixture_data_file_path)

    # ✅ Calculate Goals Against (GA)
    ga_home = fixture_df.groupby("home_team")["away_goals"].sum()
    ga_away = fixture_df.groupby("away_team")["home_goals"].sum()
    ga_total = ga_home.add(ga_away, fill_value=0).reset_index()
    ga_total.columns = ["Team", "GA"]
    ga_total["GA"] = ga_total["GA"].astype(int)

    # ✅ Aggregate team stats
    aggregated_results_df = complete_fixture_results_df.groupby("Team", as_index=False).sum()

    # ✅ Merge GA and Matches Played (MP)
    aggregated_results_df = aggregated_results_df.merge(ga_total, on="Team", how="left")
    aggregated_results_df = aggregated_results_df.merge(matches_played, on="Team", how="left")

    # add in gd
    aggregated_results_df["GD"] = aggregated_results_df["G"] - aggregated_results_df["GA"]

    # Sort by PTS, then GD, then Goals Scored
    aggregated_results_df = aggregated_results_df.sort_values(by=["PTS", "GD", "G"], ascending=[False, False, False])


    # ✅ Add new calculated columns
    aggregated_results_df["xG +/-"] = (aggregated_results_df["xG"] - aggregated_results_df["G"]).round(2)
    aggregated_results_df["xGA +/-"] = (aggregated_results_df["xGA"] - aggregated_results_df["GA"]).round(2)
    aggregated_results_df["xPTS +/-"] = (aggregated_results_df["xPTS"] - aggregated_results_df["PTS"]).round(2)

    # ✅ Reorder columns
    aggregated_results_df = aggregated_results_df[['Team', 'MP', 'W', 'D', 'L', 'G', 'GD', 'GA', 'xG', 'npxG', 'xG +/-', 'xGA', 'npxGA', 'xGA +/-', 'PTS', 'xPTS', 'xPTS +/-']]


    # ✅ Save final league table
    league_table_file_path = os.path.join(save_dir, "league_table_data.csv")
    aggregated_results_df.to_csv(league_table_file_path, index=False)

    print(f"✅ League table data saved to: {league_table_file_path}")







    import pandas as pd

    # Paths
    HISTORICAL_PATH = "data/tables/historical_fixture_data.csv"
    CURR_FIXTURES_PATH = "data/tables/fixture_data.csv"                      # Season 2025
    LAST_FIXTURES_PATH = "data/tables/24-25/fixture_data.csv"               # Season 2024

    # Load historical data
    historical = pd.read_csv(HISTORICAL_PATH)

    # Create xG columns if not present
    if 'home_xG' not in historical.columns:
        historical['home_xG'] = pd.NA
    if 'away_xG' not in historical.columns:
        historical['away_xG'] = pd.NA

    # Load fixture files
    curr_fixtures = pd.read_csv(CURR_FIXTURES_PATH)
    curr_fixtures['Season'] = 2025

    last_fixtures = pd.read_csv(LAST_FIXTURES_PATH)
    last_fixtures['Season'] = 2024

    # Combine both seasons
    fixtures = pd.concat([curr_fixtures, last_fixtures], ignore_index=True)

    # Standardize column names
    fixtures = fixtures.rename(columns={
        'home_team': 'Home Team',
        'away_team': 'Away Team',
        'Home Team': 'Home Team',
        'Away Team': 'Away Team',
        'home_xG': 'home_xG_temp',
        'away_xG': 'away_xG_temp'
    })

    # Trim and match team names
    fixtures['Home Team'] = fixtures['Home Team'].str.strip()
    fixtures['Away Team'] = fixtures['Away Team'].str.strip()
    historical['Home Team'] = historical['Home Team'].str.strip()
    historical['Away Team'] = historical['Away Team'].str.strip()

    # ── Add missing current-season completed rows to historical ──────────────
    # fixture_data.csv has all completed 25/26 results from Understat;
    # historical_fixture_data.csv may not have them if it was last rebuilt mid-season.
    curr_completed = curr_fixtures[
        curr_fixtures['isResult'].astype(str).str.lower() == 'true'
    ].copy()
    curr_completed = curr_completed.rename(columns={
        'home_team':   'Home Team',
        'away_team':   'Away Team',
        'home_goals':  'home_goals',
        'away_goals':  'away_goals',
        'home_xG':     'home_xG',
        'away_xG':     'away_xG',
        'round_number':'Round Number',
        'date':        'Date',
    })
    curr_completed['Season'] = 2025

    existing_keys = set(
        zip(historical['Home Team'], historical['Away Team'],
            historical['Season'].astype(str))
    )
    new_rows = curr_completed[
        ~curr_completed.apply(
            lambda r: (r.get('Home Team', ''), r.get('Away Team', ''), str(r['Season']))
                       in existing_keys, axis=1
        )
    ].copy()

    if len(new_rows) > 0:
        historical = pd.concat([historical, new_rows], ignore_index=True)
        print(f"✅ Added {len(new_rows)} new 25/26 match rows to historical data")
    else:
        print("✅ Historical data already up to date — no new rows to add")
    # ─────────────────────────────────────────────────────────────────────────

    # Merge based on Home Team, Away Team and Season
    merged = pd.merge(
        historical,
        fixtures[['Home Team', 'Away Team', 'Season', 'home_xG_temp', 'away_xG_temp']],
        on=['Home Team', 'Away Team', 'Season'],
        how='left'
    )

    # Fill only where missing
    merged['home_xG'] = merged['home_xG'].combine_first(merged['home_xG_temp'])
    merged['away_xG'] = merged['away_xG'].combine_first(merged['away_xG_temp'])

    # Drop temp columns
    merged.drop(columns=['home_xG_temp', 'away_xG_temp'], inplace=True)

    # Dedup before saving — prevents double-rows from repeated runs
    merged = merged.drop_duplicates(subset=['Home Team', 'Away Team', 'Season'], keep='last')
    merged.to_csv(HISTORICAL_PATH, index=False)
    print(f"✅ historical_fixture_data updated and deduped ({len(merged)} rows).")

if not BOOKIE_ONLY:
    pass  # end of main scrape block

# ── Shared setup (runs in both full and bookie-only mode) ─────────────────────
save_dir = "data/tables"
os.makedirs(save_dir, exist_ok=True)

if BOOKIE_ONLY:
    _fix = pd.read_csv(os.path.join(save_dir, "fixture_data.csv"))
    _fix["round_number"] = pd.to_numeric(_fix["round_number"], errors="coerce")
    _upcoming = _fix[_fix["isResult"].astype(str).str.lower() != "true"]
    _counts = _upcoming.groupby("round_number").size()
    next_round_number = _counts[_counts >= 5].index.min()
    print(f"🔄 Bookie-only mode — detected next GW: {int(next_round_number)}")

BOOKIE_NAME_MAP = {
    "Man Utd":                    "Manchester United",
    "Man City":                   "Manchester City",
    "Spurs":                      "Tottenham Hotspur",
    "Tottenham":                  "Tottenham Hotspur",
    "Wolves":                     "Wolverhampton Wanderers",
    "Nott'm Forest":              "Nottingham Forest",
    "Newcastle":                  "Newcastle United",
    "Brighton":                   "Brighton",
    "Bournemouth":                "Bournemouth",
    "Brentford":                  "Brentford",
    "Burnley":                    "Burnley",
    "Chelsea":                    "Chelsea",
    "Crystal Palace":             "Crystal Palace",
    "Everton":                    "Everton",
    "Fulham":                     "Fulham",
    "Liverpool":                  "Liverpool",
    "Arsenal":                    "Arsenal",
    "Aston Villa":                "Aston Villa",
    "West Ham":                   "West Ham",
    "Leeds":                      "Leeds",
    "Sunderland":                 "Sunderland",
    "Ipswich":                    "Ipswich",
    "Leicester":                  "Leicester",
    "Southampton":                "Southampton",
    "Manchester City":            "Manchester City",
    "Manchester United":          "Manchester United",
    "Newcastle United":           "Newcastle United",
    "Nottingham Forest":          "Nottingham Forest",
    "Tottenham Hotspur":          "Tottenham Hotspur",
    "Wolverhampton Wanderers":    "Wolverhampton Wanderers",
    "Brighton and Hove Albion":   "Brighton",
    "Ipswich Town":               "Ipswich",
    "Leicester City":             "Leicester",
    "West Ham United":            "West Ham",
}

gw_label = f"GW{int(next_round_number)}"

ODDS_API_KEY = "8b7c090a754d217aa867386ab87b9ff8"

print(f"🔄 Fetching bookie probabilities from The Odds API for {gw_label}...")
try:
    odds_resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/",
        params={
            "apiKey":     ODDS_API_KEY,
            "regions":    "uk",
            "markets":    "h2h,totals",
            "oddsFormat": "decimal",
        },
        timeout=15,
    )
    odds_resp.raise_for_status()
    remaining = odds_resp.headers.get("x-requests-remaining", "unknown")
    print(f"📊 Odds API credits remaining: {remaining}")
    odds_data = odds_resp.json()

    win_rows = []
    ou_rows  = []

    for game in odds_data:
        home_raw  = game["home_team"]
        away_raw  = game["away_team"]
        home_team = BOOKIE_NAME_MAP.get(home_raw, home_raw)
        away_team = BOOKIE_NAME_MAP.get(away_raw, away_raw)

        h2h_home_probs, h2h_draw_probs, h2h_away_probs = [], [], []
        over25_probs, under25_probs = [], []

        for bk in game.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    if home_raw in outcomes and "Draw" in outcomes and away_raw in outcomes:
                        raw_home = 1 / outcomes[home_raw]
                        raw_draw = 1 / outcomes["Draw"]
                        raw_away = 1 / outcomes[away_raw]
                        total = raw_home + raw_draw + raw_away
                        h2h_home_probs.append(raw_home / total * 100)
                        h2h_draw_probs.append(raw_draw / total * 100)
                        h2h_away_probs.append(raw_away / total * 100)
                elif market["key"] == "totals":
                    outcomes = {(o["name"], o.get("point")): o["price"] for o in market["outcomes"]}
                    over_price  = outcomes.get(("Over",  2.5))
                    under_price = outcomes.get(("Under", 2.5))
                    if over_price and under_price:
                        raw_over  = 1 / over_price
                        raw_under = 1 / under_price
                        total = raw_over + raw_under
                        over25_probs.append(raw_over  / total * 100)
                        under25_probs.append(raw_under / total * 100)

        if h2h_home_probs:
            win_rows.append({
                "gw":              gw_label,
                "home_team":       home_team,
                "away_team":       away_team,
                "bookie_home_win": round(sum(h2h_home_probs) / len(h2h_home_probs), 1),
                "bookie_draw":     round(sum(h2h_draw_probs) / len(h2h_draw_probs), 1),
                "bookie_away_win": round(sum(h2h_away_probs) / len(h2h_away_probs), 1),
            })
        if over25_probs:
            ou_rows.append({
                "gw":             gw_label,
                "home_team":      home_team,
                "away_team":      away_team,
                "bookie_over25":  round(sum(over25_probs)  / len(over25_probs),  1),
                "bookie_under25": round(sum(under25_probs) / len(under25_probs), 1),
            })

    if win_rows:
        win_path = os.path.join(save_dir, "bookie_win_by_gw.csv")
        win_df   = pd.read_csv(win_path)
        win_df   = win_df[win_df["gw"] != gw_label]
        win_df   = pd.concat([win_df, pd.DataFrame(win_rows)], ignore_index=True)
        win_df.to_csv(win_path, index=False)
        print(f"✅ Saved {len(win_rows)} win probability rows for {gw_label}")
    else:
        print(f"⚠️  No win probability data from Odds API for {gw_label}")

    if ou_rows:
        ou_path = os.path.join(save_dir, "bookie_ou_by_gw.csv")
        if os.path.exists(ou_path):
            ou_df = pd.read_csv(ou_path)
            ou_df = ou_df[ou_df["gw"] != gw_label]
        else:
            ou_df = pd.DataFrame(columns=["gw", "home_team", "away_team", "bookie_over25", "bookie_under25"])
        ou_df = pd.concat([ou_df, pd.DataFrame(ou_rows)], ignore_index=True)
        ou_df.to_csv(ou_path, index=False)
        print(f"✅ Saved {len(ou_rows)} O/U 2.5 rows for {gw_label}")
    else:
        print(f"⚠️  No O/U 2.5 data from Odds API for {gw_label}")

except Exception as e:
    print(f"⚠️  Could not fetch Odds API data for {gw_label}: {e}")


# ── Fetch FPL player data (bootstrap-static) ──────────────────────────────────
FPL_POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FPL_TEAM_NAME_MAP = {
    "Man City":    "Manchester City",
    "Man Utd":     "Manchester United",
    "Spurs":       "Tottenham Hotspur",
    "Wolves":      "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Newcastle":   "Newcastle United",
}

print("🔄 Fetching FPL bootstrap data...")
try:
    fpl_resp = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15
    )
    fpl_resp.raise_for_status()
    fpl_json = fpl_resp.json()

    fpl_team_id_map = {
        t["id"]: FPL_TEAM_NAME_MAP.get(t["name"], t["name"])
        for t in fpl_json["teams"]
    }

    fpl_rows = []
    for p in fpl_json["elements"]:
        fpl_rows.append({
            "fpl_id":       p["id"],
            "fpl_name":     f"{p['first_name']} {p['second_name']}",
            "web_name":     p["web_name"],
            "team":         fpl_team_id_map.get(p["team"], str(p["team"])),
            "position":     FPL_POSITION_MAP.get(p["element_type"], "UNK"),
            "price":        round(p["now_cost"] / 10, 1),
            "ownership":    float(p["selected_by_percent"]),
            "form":         float(p["form"]) if p["form"] else 0.0,
            "ep_next":      float(p["ep_next"]) if p["ep_next"] else 0.0,
            "minutes":      int(p["minutes"]),
            "goals":        int(p["goals_scored"]),
            "assists":      int(p["assists"]),
            "clean_sheets": int(p["clean_sheets"]),
            "status":       p["status"],
            "xg":           float(p.get("expected_goals", 0) or 0),
            "xa":           float(p.get("expected_assists", 0) or 0),
            "xgi":          float(p.get("expected_goal_involvements", 0) or 0),
        })

    fpl_df = pd.DataFrame(fpl_rows)
    fpl_path = os.path.join(save_dir, "fpl_player_data.csv")
    fpl_df.to_csv(fpl_path, index=False)
    print(f"✅ FPL player data saved ({len(fpl_df)} players) to {fpl_path}")



except Exception as e:
    print(f"⚠️  Could not fetch FPL bootstrap data: {e}")

from data_loader import collect_all_shot_data
collect_all_shot_data()