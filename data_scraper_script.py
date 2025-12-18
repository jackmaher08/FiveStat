import os
import re
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.colors as mcolors
from bs4 import BeautifulSoup
from mplsoccer import Pitch
from mplsoccer import Radar
from matplotlib.colors import LinearSegmentedColormap
from understatapi import UnderstatClient


fixturedownload_url = "https://fixturedownload.com/download/epl-2025-GMTStandardTime.csv"
fixtures_df = pd.read_csv(fixturedownload_url)

# Rename columns to match expected format
fixtures_df = fixtures_df.rename(columns={
    "Round Number": "round_number",
    "Home Team": "home_team",
    "Away Team": "away_team",
    "Date": "date",
    "Result": "result"
})

# change relevant team names
fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Nott'm Forest": "Nottingham Forest"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Nott'm Forest": "Nottingham Forest"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Man Utd": "Manchester United"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Man Utd": "Manchester United"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Man City": "Manchester City"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Man City": "Manchester City"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Spurs": "Tottenham Hotspur"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Spurs": "Tottenham Hotspur"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Tottenham": "Tottenham Hotspur"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Tottenham": "Tottenham Hotspur"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Wolves": "Wolverhampton Wanderers"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Wolves": "Wolverhampton Wanderers"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Newcastle": "Newcastle United"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Newcastle": "Newcastle United"})


# Convert 'round_number' to numeric
fixtures_df["round_number"] = pd.to_numeric(fixtures_df["round_number"], errors="coerce")

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
    fixtures_df[["round_number", "date", "home_team", "away_team", "result"]],
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





# Define the save directory
save_dir = "data/tables"
os.makedirs(save_dir, exist_ok=True)  # ✅ Ensure the directory exists

# Define the file path
file_path = os.path.join(save_dir, "fixture_data.csv")

# ✅ Save the DataFrame as a CSV file
fixture_data.to_csv(file_path, index=False)

print(f"✅ fixture data saved to: {file_path}")



'''
#TEMPORARILY USING THE FOLLOWING
import pandas as pd
import os

# 1️⃣ Download 2025 fixture list from fixturedownload
fixturedownload_url = "https://fixturedownload.com/download/epl-2025-GMTStandardTime.csv"
fixtures_df = pd.read_csv(fixturedownload_url)

# 2️⃣ Standardize column names
fixtures_df = fixtures_df.rename(columns={
    "Round Number": "round_number",
    "Home Team": "home_team",
    "Away Team": "away_team",
    "Date": "date",
    "Result": "result"
})

# 3️⃣ Normalize team names
team_name_mapping = {
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Man Utd": "Manchester United",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest"
}
fixtures_df["home_team"] = fixtures_df["home_team"].replace(team_name_mapping)
fixtures_df["away_team"] = fixtures_df["away_team"].replace(team_name_mapping)

# 4️⃣ Ensure numeric round column
fixtures_df["round_number"] = pd.to_numeric(fixtures_df["round_number"], errors="coerce")

# 5️⃣ Add empty columns that Understat normally fills
fixtures_df["id"] = None
fixtures_df["isResult"] = False
fixtures_df["home_goals"] = None
fixtures_df["away_goals"] = None
fixtures_df["home_xG"] = None
fixtures_df["away_xG"] = None

# 6️⃣ Rearrange to match expected structure
columns_order = [
    "id", "isResult", "round_number", "date", "home_team", "away_team", "result",
    "home_goals", "away_goals", "home_xG", "away_xG"
]
fixtures_df = fixtures_df[columns_order]

fixture_data = fixtures_df

# 7️⃣ Save the merged file
save_dir = "data/tables"
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "fixture_data.csv")
fixture_data.to_csv(file_path, index=False)

print(f"✅ Fixture data saved (Understat skipped): {file_path}")


#END OF TEMP
'''




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
'''
start_year=2016
end_year=2025
frames = [] 
for year in range(start_year, end_year + 1):
    url = f"https://fixturedownload.com/download/epl-{year}-GMTStandardTime.csv"
    frame = pd.read_csv(url)
    frame['Season'] = year
    frames.append(frame)

    frame = frame.rename(columns={
        "Round Number": "round_number",
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Date": "date",
        "Result": "result"
    })

# Merge all season data
df = pd.concat(frames)

# correct team names
df["Home Team"] = df["Home Team"].replace({"Nott'm Forest": "Nottingham Forest"})
df["Away Team"] = df["Away Team"].replace({"Nott'm Forest": "Nottingham Forest"})
                                          
df["Home Team"] = df["Home Team"].replace({"Spurs": "Tottenham Hotspur"})
df["Away Team"] = df["Away Team"].replace({"Spurs": "Tottenham Hotspur"})

df["Home Team"] = df["Home Team"].replace({"Man Utd": "Manchester United"})
df["Away Team"] = df["Away Team"].replace({"Man Utd": "Manchester United"})

df["Home Team"] = df["Home Team"].replace({"Man City": "Manchester City"})
df["Away Team"] = df["Away Team"].replace({"Man City": "Manchester City"})

df["Home Team"] = df["Home Team"].replace({"Newcastle": "Newcastle United"})
df["Away Team"] = df["Away Team"].replace({"Newcastle": "Newcastle United"})

df["Home Team"] = df["Home Team"].replace({"Wolves": "Wolverhampton Wanderers"})
df["Away Team"] = df["Away Team"].replace({"Wolves": "Wolverhampton Wanderers"})

df = df[pd.notnull(df["Result"])]  # Keep only matches with results

# Process result column
df[['home_goals', 'away_goals']] = df['Result'].str.split(' - ', expand=True).astype(float)
df['result'] = df.apply(lambda row: 'home_win' if row['home_goals'] > row['away_goals'] 
                        else 'away_win' if row['home_goals'] < row['away_goals'] else 'draw', axis=1)

# 📌 **Generate Team ID Dictionary**
unique_teams = sorted(set(df['Home Team'].unique()) | set(df['Away Team'].unique()))  # Get all unique teams
team_id_dict = {team: idx + 1 for idx, team in enumerate(unique_teams)}  # Assign numeric IDs

# Assign team IDs to DataFrame
df['home_team_id'] = df['Home Team'].map(team_id_dict)
df['away_team_id'] = df['Away Team'].map(team_id_dict)



# Define the file path
historical_fixture_file_path = os.path.join(save_dir, "historical_fixture_data.csv")

# ✅ Save the DataFrame as a CSV file
df.to_csv(historical_fixture_file_path, index=False)

print(f"✅ historical fixture data saved to: {historical_fixture_file_path}")
'''

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





# scraping fbref player data for player radar plots
'''
fbref_url = 'https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com"
}

session = requests.Session()
response = session.get(fbref_url, headers=headers)
response.raise_for_status()

from io import StringIO
html_data = StringIO(response.text)
fbref_df = pd.read_html(html_data, attrs={"id": "stats_standard"})[0]


print(f"🔍 Status code: {response.status_code}")


# getting rid of the per 90 columns and will recalculate the ones we're interested in
columns_to_drop = fbref_df.columns.get_level_values(0) == 'Per 90 Minutes'
fbref_df = fbref_df.drop(columns=fbref_df.columns[columns_to_drop])


# get rid of the first level of the multiindex on the columns
fbref_df = fbref_df.droplevel(0, axis=1)

# Convert the 'Min' column to numeric, coercing any non-numeric entries to NaN
fbref_df['Min'] = pd.to_numeric(fbref_df['Min'], errors='coerce')

# Drop rows where 'Min' is NaN (i.e., non-numeric values)
fbref_df = fbref_df.dropna(subset=['Min'])

# Convert 'Min' to integers
fbref_df['Min'] = fbref_df['Min'].astype(int)

# Filter for players who have played more than 400 minutes
fbref_df = fbref_df[fbref_df['Min'] > 400]

# Drop GKs
fbref_df = fbref_df[fbref_df['Pos'] != 'GK']





# let's also make sure that the columns are of the correct type
fbref_df[['90s', 'xG', 'xAG']] = fbref_df[['90s', 'xG', 'xAG']].astype(float)
fbref_df[['Gls', 'Ast', 'G+A', 'PrgC', 'PrgP', 'PrgR']] = fbref_df[['Gls', 'Ast', 'G+A', 'PrgC', 'PrgP', 'PrgR']].astype(int)

# Now let's calculate the per 90 stats for each of these columns
# name them as we want to see them in the radar plot
fbref_df['goals_per_90'] = fbref_df['Gls'] / fbref_df['90s']
fbref_df['assists_per_90'] = fbref_df['Ast'] / fbref_df['90s']
fbref_df['goals_assists_per_90'] = fbref_df['G+A'] / fbref_df['90s']
fbref_df['expected_goals_per_90'] = fbref_df['xG'] / fbref_df['90s']
fbref_df['expected_assists_per_90'] = fbref_df['xAG'] / fbref_df['90s']
fbref_df['progressive_carries_per_90'] = fbref_df['PrgC'] / fbref_df['90s']
fbref_df['progressive_passes_per_90'] = fbref_df['PrgP'] / fbref_df['90s']
fbref_df['progressive_receptions_per_90'] = fbref_df['PrgR'] / fbref_df['90s']

# We'll calculate the percentiles for each of these columns
# We will also name them as we want to see them in the radar plot
fbref_df['Goals'] = (fbref_df['goals_per_90'].rank(pct=True) * 100).astype(int)
fbref_df['Assists'] = (fbref_df['assists_per_90'].rank(pct=True) * 100).astype(int)
fbref_df['Goals + Assists'] = (fbref_df['goals_assists_per_90'].rank(pct=True) * 100).astype(int)
fbref_df['Expected Goals'] = (fbref_df['expected_goals_per_90'].rank(pct=True)  * 100).astype(int)
fbref_df['Expected Assists'] = (fbref_df['expected_assists_per_90'].rank(pct=True) * 100).astype(int)
fbref_df['Progressive Carries'] = (fbref_df['progressive_carries_per_90'].rank(pct=True) * 100).astype(int)
fbref_df['Progressive Passes'] = (fbref_df['progressive_passes_per_90'].rank(pct=True) * 100).astype(int)
fbref_df['Progressive Receptions'] = (fbref_df['progressive_receptions_per_90'].rank(pct=True) * 100).astype(int)

# ensure we only keep players who have per 90 stats populated

fbref_df = fbref_df[fbref_df['goals_per_90'] > 0]
fbref_df = fbref_df[fbref_df['assists_per_90'] > 0]
fbref_df = fbref_df[fbref_df['goals_assists_per_90'] > 0]
fbref_df = fbref_df[fbref_df['expected_goals_per_90'] > 0]
fbref_df = fbref_df[fbref_df['expected_assists_per_90'] > 0]
fbref_df = fbref_df[fbref_df['progressive_carries_per_90'] > 0]
fbref_df = fbref_df[fbref_df['progressive_passes_per_90'] > 0]
fbref_df = fbref_df[fbref_df['progressive_receptions_per_90'] > 0]

# sort df by goals desc
fbref_df = fbref_df.sort_values(by='Gls', ascending=False)


# ✅ Save final league table
player_stats_file_path = os.path.join(save_dir, "player_radar_data.csv")
fbref_df.to_csv(player_stats_file_path, index=False)

print(f"✅ Player Radar data saved to: {player_stats_file_path}")

'''




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

# Save updated file
merged.to_csv(HISTORICAL_PATH, index=False)
print("✅ xG data merged and historical_fixture_data updated successfully.")
