from data_loader import completed_fixtures
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
from mplsoccer import VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import random
import matplotlib.image as mpimg
import seaborn as sns




# Ensure directories exist
SHOTMAP_DIR = "static/shotmaps/"
ALL_SHOTMAP_DIR = os.path.join(SHOTMAP_DIR, "all/")
TEAM_SHOTMAP_DIR = os.path.join(SHOTMAP_DIR, "team/")

os.makedirs(ALL_SHOTMAP_DIR, exist_ok=True)
os.makedirs(TEAM_SHOTMAP_DIR, exist_ok=True)

# ✅ Define the path for saving shots_data.csv
SHOTS_DATA_PATH = "data/tables/shots_data.csv"

TEAM_NAME_MAPPING = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "ipswich": "Ipswich",
    "leicester": "Leicester",
    "liverpool": "Liverpool",
    "man city": "Manchester City",
    "man utd": "Manchester United",
    "newcastle": "Newcastle United",
    "nott'm forest": "Nottingham Forest",
    "southampton": "Southampton",
    "spurs": "Tottenham",
    "west ham": "West Ham",
    "wolves": "Wolverhampton Wanderers"
}


if os.path.exists(SHOTS_DATA_PATH):
    all_shots_df = pd.read_csv(SHOTS_DATA_PATH)
else:
    print("⚠️ No shot data found! Exiting...")
    exit()

# ✅ Ensure shot data is available before processing
if all_shots_df.empty or "team" not in all_shots_df.columns:
    print("⚠️ No shot data available. Skipping shotmap generation.")
    exit()

# ✅ Process **all** shots taken this season per team
team_shots = {team: all_shots_df[all_shots_df['team'] == team] for team in all_shots_df['team'].unique()}


def process_match_shots(understat_match_id):
    """Fetch and process shot data for a match."""
    try:
        url = f'https://understat.com/match/{understat_match_id}'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        match = re.search(r"var shotsData\s*=\s*JSON.parse\('(.*)'\)", str(soup))

        if not match:
            print(f"Skipping match {understat_match_id}: No shot data found")
            return
        
        data = json.loads(match.group(1).encode('utf8').decode('unicode_escape'))

        # Get team names
        match_info = completed_fixtures[completed_fixtures["id"] == understat_match_id]
        if match_info.empty:
            print(f"Match {understat_match_id} not found in fixture list")
            return
        
        home_team, away_team = match_info["home_team"].values[0], match_info["away_team"].values[0]

        # Process shots for both teams
        for team, shots in [('home', data['h']), ('away', data['a'])]:
            df = pd.DataFrame(shots)
            if df.empty:
                continue

            df['team'] = home_team if team == 'home' else away_team
            df['x_scaled'] = df['X'].astype(float) * 120
            df['y_scaled'] = df['Y'].astype(float) * 80

            # Flip coordinates **only for away teams** (so all shots face the same goal)
            if team == 'away':
                df['x_scaled'] = 120 - df['x_scaled']

            team_shots.setdefault(df['team'].iloc[0], pd.DataFrame())
            team_shots[df['team'].iloc[0]] = pd.concat([team_shots[df['team'].iloc[0]], df], ignore_index=True)

    except Exception as e:
        print(f"Error processing match {understat_match_id}: {e}")



SHOTS_DATA_PATH = "data/tables/shots_data.csv"
ALL_SHOTMAP_DIR = "static/shotmaps/all"




def process_shot_data(completed_fixtures, team_shots):
    """Processes shot data, merges with fixture info, and generates an all-shots shotmap."""
    global all_shots_df
    # ✅ Load existing shot data
    if os.path.exists(SHOTS_DATA_PATH):
        all_shots_df = pd.read_csv(SHOTS_DATA_PATH)
        processed_match_ids = set(all_shots_df["match_id"].astype(str).unique())  # Convert IDs to string for comparison
    else:
        processed_match_ids = set()

    # ✅ Process ONLY new matches instead of ALL matches
    new_match_ids = set(completed_fixtures["id"].astype(str).unique()) - processed_match_ids

    if new_match_ids:
        print(f"🔄 Processing {len(new_match_ids)} new matches...")
        for i, match_id in enumerate(new_match_ids, start=1):
            process_match_shots(match_id)  
            print(f"Progress: {i}/{len(new_match_ids)} matches processed.")
    else:
        print("✅ No new matches to process. Using existing shot data.")

    # ✅ Merge new shot data with existing data
    if team_shots:
        new_shots_df = pd.concat(team_shots.values(), ignore_index=True)
        all_shots_df = pd.concat([all_shots_df, new_shots_df], ignore_index=True)
    else:
        print("No new shot data found. Using existing shots_data.csv.")

    # ✅ Ensure no duplicate 'id' column before merging
    if 'id' in all_shots_df.columns:
        all_shots_df.drop(columns=['id'], inplace=True)

    all_shots_df['match_id'] = all_shots_df['match_id'].astype(str)
    completed_fixtures['id'] = completed_fixtures['id'].astype(str)

    # ✅ Merge completed fixtures with shot data
    all_shots_df = all_shots_df.merge(
        completed_fixtures[['id', 'home_team', 'away_team']], 
        left_on='match_id', 
        right_on='id', 
        how='left'
    )

    # ✅ Clean up unnecessary columns
    all_shots_df.drop(columns=['id'], errors='ignore', inplace=True)
    all_shots_df.rename(columns={'home_team_y': 'home_team', 'away_team_y': 'away_team'}, inplace=True)
    all_shots_df = all_shots_df.loc[:, ~all_shots_df.columns.duplicated()].copy()

    # ✅ Assign home/away indicator
    all_shots_df['h_a'] = all_shots_df.apply(
        lambda row: 'h' if str(row['team']).strip() == str(row['home_team']).strip() else 'a', axis=1
    )

    # ✅ Save updated shot data
    all_shots_df.to_csv(SHOTS_DATA_PATH, index=False, columns=['match_id', 'team', 'x_scaled', 'y_scaled', 'xG', 'result', 'h_a', 'home_team', 'away_team'])
    print(f"✅ Shot data saved to {SHOTS_DATA_PATH}")

    # ✅ Generate All-Shots Shotmap
    print("🔄 Generating All-Shots Shotmap...")

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#f4f4f9', line_color='black', line_zorder=2, half=True)
    fig, ax = pitch.draw(figsize=(13, 9))
    fig.patch.set_facecolor("#f4f4f9")
    
    # ✅ Filter for goals only
    goals_df = all_shots_df[all_shots_df['result'].str.lower() == 'goal']

    # ✅ Plot all shots
    for _, shot in all_shots_df.iterrows():
        x, y = shot['x_scaled'], shot['y_scaled']
        color = 'gold' if "goal" in str(shot['result']).lower() else 'white'
        zorder = 3 if shot['result'].lower() == 'goal' else 2
        size = 500 * float(shot['xG']) if pd.notna(shot['xG']) else 100
        pitch.scatter(x, y, s=size, c=color, edgecolors='black', ax=ax, zorder=zorder)

    # ✅ Save All-Shots Shotmap
    os.makedirs(ALL_SHOTMAP_DIR, exist_ok=True)
    plt.savefig(os.path.join(ALL_SHOTMAP_DIR, "all_shots.png"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print("✅ All-Shots Shotmap Saved!")

print(TEAM_NAME_MAPPING.get("Spurs", "Not Found"))  # Output: Not Found
print(TEAM_NAME_MAPPING.get("spurs", "Not Found"))  # Output: Tottenham

def plot_team_shotmap(team_name):
    # Convert all filenames to Title Case to maintain consistency
    standardized_team_name = TEAM_NAME_MAPPING.get(team_name.strip(), team_name)
    formatted_filename = standardized_team_name.title()  # Ensures "chelsea" → "Chelsea"

    df = all_shots_df[all_shots_df['team'] == team_name]

    if df.empty:
        print(f"No shots found for {team_name}")
        return
    
    # Create pitch
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#f4f4f9', line_color='black', line_zorder=2, half=True)
    fig, ax = pitch.draw(figsize=(12, 9))
    fig.patch.set_facecolor("#f4f4f9")

    existing_columns = set(df.columns)
    subset_columns = [col for col in ["match_id", "player", "x_scaled", "y_scaled"] if col in existing_columns]

    if subset_columns:
        df = df.drop_duplicates(subset=subset_columns)

    for _, shot in df.iterrows():
        x, y = shot['x_scaled'], shot['y_scaled']
        color = 'gold' if shot['result'].lower() == 'goal' else 'white'
        zorder = 3 if shot['result'].lower() == 'goal' else 2
        size = 500 * float(shot['xG']) if pd.notna(shot['xG']) else 100

        pitch.scatter(x, y, s=size, c=color, edgecolors='black', ax=ax, zorder=zorder)

    plt.title(f"{standardized_team_name} Shotmap", fontsize=15)

    # Save using standardized team name
    shotmap_filename = f"{formatted_filename}_shotmap.png"
    plt.savefig(os.path.join(TEAM_SHOTMAP_DIR, shotmap_filename))
    plt.close(fig)
    print(f"Saved {standardized_team_name} shotmap to {TEAM_SHOTMAP_DIR}{shotmap_filename}")


print(f"🔄 Generating shotmaps for {len(team_shots.keys())} teams...")

for team in team_shots.keys():
    print(f"🎯 Processing shotmap for {team}...")
    plot_team_shotmap(team)  # ✅ Now uses **all** available shots for the season

print("✅ All Team Shotmaps Generated! 🎯⚽")

