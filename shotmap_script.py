import os
import re
import json
import requests
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.colors as mcolors
from bs4 import BeautifulSoup
from mplsoccer import Pitch
from matplotlib.colors import LinearSegmentedColormap

# Fetch fixture data
url = 'https://understat.com/league/EPL/2024'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
ugly_soup = str(soup)

# Extract JSON fixture data
all_fixture_data = re.search("var datesData .*= JSON.parse\('(.*)'\)", ugly_soup).group(1)
all_fixture_df = all_fixture_data.encode('utf8').decode('unicode_escape')
all_fixture_df = json.loads(all_fixture_df)

team_name_mapping = {
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Manchester City": "Man City",
    "Tottenham": "Spurs",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest"
}

# Parse fixture data
fixture_data = []
for fixture in all_fixture_df:
    fixture_entry = {
        "id": fixture.get("id"),
        "isResult": fixture.get("isResult"),
        "home_team": fixture["h"]["title"] if "h" in fixture and "title" in fixture["h"] else None,
        "away_team": fixture["a"]["title"] if "a" in fixture and "title" in fixture["a"] else None,
        "home_goals": int(fixture["goals"]["h"]) if fixture.get("goals") and fixture["goals"].get("h") is not None else None,
        "away_goals": int(fixture["goals"]["a"]) if fixture.get("goals") and fixture["goals"].get("a") is not None else None,
        "home_xG": round(float(fixture["xG"]["h"]), 2) if fixture.get("xG") and fixture["xG"].get("h") is not None else None,
        "away_xG": round(float(fixture["xG"]["a"]), 2) if fixture.get("xG") and fixture["xG"].get("a") is not None else None,
    }
    fixture_data.append(fixture_entry)

# Create DataFrame
complete_all_fixture_df = pd.DataFrame(fixture_data)

# Filter only completed matches (ignore upcoming games)
completed_fixtures = complete_all_fixture_df.dropna(subset=["home_goals", "away_goals"])

# Directory to save shotmaps
shotmap_save_path = "static/shotmaps/"
os.makedirs(shotmap_save_path, exist_ok=True)

fixture_data_df = pd.DataFrame(fixture_data)
fixture_data_df[["home_team", "away_team"]] = fixture_data_df[["home_team", "away_team"]].replace(team_name_mapping)

new_fixture_df = pd.merge(
        fixtures_df[["Round Number", "date", "home_team", "away_team", "Result"]],
        fixture_data_df[["id", "home_team", "away_team", "isResult", "home_goals", "away_goals", "home_xG", "away_xG"]],
        on=["home_team", "away_team"],
        how="left"
    )
    
match_id_file = "data/match_ids.csv"
if os.path.exists(match_id_file):
    match_id_df = pd.read_csv(match_id_file)
else:
    match_id_df = pd.DataFrame(columns=["home_team", "away_team", "match_id"])

def generate_match_id():
    existing_ids = set(match_id_df["match_id"].dropna().astype(int))
    while True:
        match_id = random.randint(1000, 9999)
        if match_id not in existing_ids:
            existing_ids.add(match_id)
            return match_id

if "match_id" not in new_fixture_df.columns:
    new_fixture_df["match_id"] = new_fixture_df.apply(lambda row: match_id_df.loc[(match_id_df["home_team"] == row["home_team"]) & (match_id_df["away_team"] == row["away_team"]), "match_id"].values[0] if ((match_id_df["home_team"] == row["home_team"]) & (match_id_df["away_team"] == row["away_team"])).any() else generate_match_id(), axis=1)

match_id_df = new_fixture_df[["home_team", "away_team", "match_id"]]
match_id_df.to_csv(match_id_file, index=False)



# Function to generate and save shot map
def generate_shot_map(understat_match_id):
    try:
        url = f'https://understat.com/match/{understat_match_id}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        ugly_soup = str(soup)

        # Extract JSON shot data
        match = re.search("var shotsData .*= JSON.parse\('(.*)'\)", ugly_soup)
        if not match:
            print(f"Skipping match {understat_match_id}: No shot data found")
            return
        
        shots_data = match.group(1)
        data = shots_data.encode('utf8').decode('unicode_escape')
        data = json.loads(data)

        # Create DataFrames
        home_df = pd.DataFrame(data['h'])
        away_df = pd.DataFrame(data['a'])

        # Extract and update team names using the mapping
        home_team_name = home_df.iloc[0]['h_team'] if not home_df.empty else "Unknown"
        away_team_name = away_df.iloc[0]['a_team'] if not away_df.empty else "Unknown"
        home_team_name = team_name_mapping.get(home_team_name, home_team_name)
        away_team_name = team_name_mapping.get(away_team_name, away_team_name)

        # Scale coordinates to StatsBomb pitch (120x80)
        home_df['x_scaled'] = home_df['X'].astype(float) * 120
        home_df['y_scaled'] = home_df['Y'].astype(float) * 80
        away_df['x_scaled'] = away_df['X'].astype(float) * 120
        away_df['y_scaled'] = away_df['Y'].astype(float) * 80

        # Adjust positions for correct plotting
        home_df['x_scaled'] = 120 - home_df['x_scaled']
        away_df['y_scaled'] = 80 - away_df['y_scaled']

        # Calculate total goals and xG
        total_goals_home = home_df['result'].str.contains('Goal', case=False, na=False).sum()
        total_goals_away = away_df['result'].str.contains('Goal', case=False, na=False).sum()
        total_xg_home = home_df['xG'].astype(float).sum()
        total_xg_away = away_df['xG'].astype(float).sum()

        # Initialize pitch
        pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black', line_zorder=2)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set background color to #f4f4f9
        fig.patch.set_facecolor('#f4f4f9')  # Entire figure background
        ax.set_facecolor('#f4f4f9')  # Axis background

        # Plot heatmap
        all_shots = pd.concat([home_df, away_df])
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f4f4f9', '#3f007d'])
        pitch.kdeplot(
            all_shots['x_scaled'], all_shots['y_scaled'], ax=ax, fill=True, cmap=cmap, 
            n_levels=100, thresh=0, zorder=1
        )

        # Draw pitch
        pitch.draw(ax=ax)

        # Plot shots for both teams
        for df in [home_df, away_df]:  
            for _, shot in df.iterrows():
                x, y = shot['x_scaled'], shot['y_scaled']
                color = 'gold' if shot['result'] == 'Goal' else 'white'
                zorder = 3 if shot['result'] == 'Goal' else 2
                ax.scatter(x, y, s=1000 * float(shot['xG']) if pd.notna(shot['xG']) else 100, 
                           ec='black', c=color, zorder=zorder)

        # Add match info
        ax.text(30, 10, f"{home_team_name}", ha='center', va='center', fontsize=25, fontweight='bold', color='black')
        ax.text(90, 10, f"{away_team_name}", ha='center', va='center', fontsize=25, fontweight='bold', color='black')
        ax.text(30, 40, f"{total_goals_home}", ha='center', va='center', fontsize=180, fontweight='bold', color='black', alpha=0.5)
        ax.text(90, 40, f"{total_goals_away}", ha='center', va='center', fontsize=180, fontweight='bold', color='black', alpha=0.5)
        ax.text(30, 60, f"{total_xg_home:.2f}", ha='center', va='center', fontsize=45, fontweight='bold', color='black', alpha=0.6)
        ax.text(90, 60, f"{total_xg_away:.2f}", ha='center', va='center', fontsize=45, fontweight='bold', color='black', alpha=0.6)

        # Adjust layout to prevent cropping
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(shotmap_save_path, f"{understat_match_id}_shotmap.png"))
        plt.close(fig)  # Close to free memory

    except Exception as e:
        print(f"Error processing match {understat_match_id}: {e}")

# Loop through completed fixtures only
for _, row in complete_all_fixture_df.iterrows():
    match_id = row['id']
    shotmap_file = os.path.join(shotmap_save_path, f"{match_id}_shotmap.png")

    # Check if the shotmap already exists, if so, skip processing
    if os.path.exists(shotmap_file):
        continue

    if pd.notna(row['home_goals']) and pd.notna(row['away_goals']):  # Only process completed matches
        generate_shot_map(match_id)

print("Shotmaps completed!")