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

# change relevant team names
fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Nott'm Forest": "Nottingham Forest"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Nott'm Forest": "Nottingham Forest"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Man Utd": "Manchester United"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Man Utd": "Manchester United"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Man City": "Manchester City"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Man City": "Manchester City"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Spurs": "Tottenham Hotspur"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Spurs": "Tottenham Hotspur"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Wolves": "Wolverhampton Wanderers"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Wolves": "Wolverhampton Wanderers"})

fixtures_df["home_team"] = fixtures_df["home_team"].replace({"Newcastle": "Newcastle United"})
fixtures_df["away_team"] = fixtures_df["away_team"].replace({"Newcastle": "Newcastle United"})


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
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Man Utd": "Manchester United",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest"
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

player_data["Team"] = player_data["Team"].replace({"Tottenham": "Tottenham Hotspur"})

# Define the file path
player_file_path = os.path.join(save_dir, "player_data.csv")

# ✅ Save the DataFrame as a CSV file
player_data.to_csv(player_file_path, index=False)

print(f"✅ player data saved to: {player_file_path}")





# Gathering league table data
fixture_result_data = re.search("var teamsData .*= JSON.parse\('(.*)'\)", ugly_soup).group(1)
fixture_results_df = fixture_result_data.encode('utf8').decode('unicode_escape')
fixture_results_df = json.loads(fixture_results_df)

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

# ✅ Sort by points
aggregated_results_df = aggregated_results_df.sort_values(by="PTS", ascending=False)

# ✅ Add new calculated columns
aggregated_results_df["xG +/-"] = (aggregated_results_df["xG"] - aggregated_results_df["G"]).round(2)
aggregated_results_df["xGA +/-"] = (aggregated_results_df["xGA"] - aggregated_results_df["GA"]).round(2)
aggregated_results_df["xPTS +/-"] = (aggregated_results_df["xPTS"] - aggregated_results_df["PTS"]).round(2)

# ✅ Reorder columns
aggregated_results_df = aggregated_results_df[['Team', 'MP', 'W', 'D', 'L', 'G', 'xG', 'npxG', 'xG +/-', 'GA', 'xGA', 'npxGA', 'xGA +/-', 'PTS', 'xPTS', 'xPTS +/-']]

# ✅ Save final league table
league_table_file_path = os.path.join(save_dir, "league_table_data.csv")
aggregated_results_df.to_csv(league_table_file_path, index=False)

print(f"✅ League table data saved to: {league_table_file_path}")





# scraping fbref player data for player radar plots

fbref_url = 'https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats'
fbref_df = pd.read_html(fbref_url, attrs={"id": "stats_standard"})[0]

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