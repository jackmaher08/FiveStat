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
from matplotlib.colors import LinearSegmentedColormap

fixturedownload_url = "https://fixturedownload.com/download/epl-2024-GMTStandardTime.csv"
fixtures_df = pd.read_csv(fixturedownload_url)

# Rename columns to match expected format
fixtures_df = fixtures_df.rename(columns={
    "Round Number": "round_number",
    "Home Team": "home_team",
    "Away Team": "away_team",
    "Date": "date",
    "Result": "result"
})

# Convert 'round_number' to numeric
fixtures_df["round_number"] = pd.to_numeric(fixtures_df["round_number"], errors="coerce")

# 📌 **Second Source: Understat**
understat_url = "https://understat.com/league/EPL/2024"
response = requests.get(understat_url
                        )
soup = BeautifulSoup(response.content, 'html.parser')
ugly_soup = str(soup)

# Extract JSON fixture data
match = re.search("var datesData .*= JSON.parse\\('(.*)'\\)", ugly_soup)
if match:
    all_fixture_data = match.group(1).encode('utf8').decode('unicode_escape')
    all_fixture_df = json.loads(all_fixture_data)
else:
    print("⚠️ No fixture data found on Understat")
    all_fixture_df = []

# Team name mapping for consistency
team_name_mapping = {
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Manchester City": "Man City",
    "Tottenham": "Spurs",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest"
}

# Parse fixture data
fixture_data_temp = []
for fixture in all_fixture_df:
    fixture_entry = {
        "id": fixture.get("id"),
        "isResult": fixture.get("isResult"),
        "home_team": team_name_mapping.get(fixture["h"]["title"], fixture["h"]["title"]),
        "away_team": team_name_mapping.get(fixture["a"]["title"], fixture["a"]["title"]),
        "home_goals": int(fixture["goals"]["h"]) if fixture.get("goals") and fixture["goals"].get("h") is not None else None,
        "away_goals": int(fixture["goals"]["a"]) if fixture.get("goals") and fixture["goals"].get("a") is not None else None,
        "home_xG": round(float(fixture["xG"]["h"]), 2) if fixture.get("xG") and fixture["xG"].get("h") is not None else None,
        "away_xG": round(float(fixture["xG"]["a"]), 2) if fixture.get("xG") and fixture["xG"].get("a") is not None else None,
    }
    fixture_data_temp.append(fixture_entry)

# Convert Understat data to DataFrame
fixture_data_df = pd.DataFrame(fixture_data_temp)

# 🔄 **Merge DataFrames**
fixture_data = pd.merge(
    fixtures_df[["round_number", "date", "home_team", "away_team", "result"]],
    fixture_data_df[["id", "home_team", "away_team", "isResult", "home_goals", "away_goals", "home_xG", "away_xG"]],
    on=["home_team", "away_team"],  
    how="left"  # Keep all fixtures even if no match in fixture_data_df
)

# Define the save directory
save_dir = "tables"
os.makedirs(save_dir, exist_ok=True)  # ✅ Ensure the directory exists

# Define the file path
file_path = os.path.join(save_dir, "fixture_data.csv")

# ✅ Save the DataFrame as a CSV file
fixture_data.to_csv(file_path, index=False)

print(f"✅ fixture data saved to: {file_path}")







# load next gw fixtures
# Find the next round number by getting the minimum round_number where isResult is False
next_round_number = fixture_data.loc[fixture_data["isResult"] == False, "round_number"].min()

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
start_year=2016
end_year=2024
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








# Player data

player_url = 'https://understat.com/league/EPL/2024'
response = requests.get(player_url)
soup = BeautifulSoup(response.content, 'html.parser')
ugly_soup = str(soup)

# Extract JSON data
player_data = re.search(r"var\s+playersData\s*=\s*JSON.parse\('(.*)'\);", ugly_soup).group(1)
player_df = player_data.encode('utf8').decode('unicode_escape')
player_df = json.loads(player_df)

# Parse data into a list of dicts
player_data = [
    {
        "Name": fixture.get("player_name"),
        "POS": fixture.get("position", ""),
        "Team": fixture.get("team_title", ""),
        "MP": int(fixture["games"]) if fixture["games"] else 0,
        "Mins": int(fixture["time"]) if fixture["time"] else 0,
        "G": int(fixture["goals"]) if fixture["goals"] else 0,
        "xG": round(float(fixture["xG"]), 2) if fixture["xG"] else 0.0,
        "NPG": int(fixture["npg"]) if fixture["npg"] else 0.0,
        "NPxG": round(float(fixture["npxG"]), 2) if fixture["npxG"] else 0.0,
        "A": int(fixture["assists"]) if fixture["assists"] else 0,
        "xA": round(float(fixture["xA"]), 2) if fixture["xA"] else 0.0,
        "YC": int(fixture["yellow_cards"]) if fixture["yellow_cards"] else 0,
        "RC": int(fixture["red_cards"]) if fixture["red_cards"] else 0,
    }
    for fixture in player_df
]

player_data = pd.DataFrame(player_data)

# Define the file path
player_file_path = os.path.join(save_dir, "player_data.csv")

# ✅ Save the DataFrame as a CSV file
player_data.to_csv(player_file_path, index=False)

print(f"✅ player data saved to: {player_file_path}")