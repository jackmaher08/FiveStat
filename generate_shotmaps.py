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
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Man Utd": "Manchester United",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest"
}


if os.path.exists(SHOTS_DATA_PATH):
    all_shots_df = pd.read_csv(SHOTS_DATA_PATH)
    all_shots_df["team"] = all_shots_df["team"].replace(TEAM_NAME_MAPPING)
    all_shots_df["team"] = all_shots_df["team"].str.strip()
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
             
            # ✅ Flip coordinates for away teams (Ensuring all shots face the same goal)
            if team == 'away':
                df['X'] = 1 - df['X']  # Flip X

            
            df['x_scaled'] = df['X'].astype(float) * 120
            df['y_scaled'] = df['Y'].astype(float) * 80

            

            # ✅ Debugging: Print sample flipped shots for validation
            print(f"🏟️ Processed shots for {df['team'].iloc[0]} (Team: {team}) - First 3 shots:")
            print(df[['x_scaled', 'y_scaled']].head(3))

            # ✅ Ensure modified shot data is correctly stored
            standardized_name = TEAM_NAME_MAPPING.get(team_name, team_name)
            team_shots[standardized_name] = df.copy()


    except Exception as e:
        print(f"❌ Error processing match {understat_match_id}: {e}")









def plot_team_shotmap(team_name):
    standardized_team_name = TEAM_NAME_MAPPING.get(team_name.strip(), team_name)
    formatted_filename = standardized_team_name.title()

    df = all_shots_df[all_shots_df['team'] == team_name]

    if df.empty:
        print(f"No shots found for {team_name}")
        return

    # Flip away team shots
    df.loc[df["h_a"] == "a", "x_scaled"] = 120 - df["x_scaled"]
    df.loc[df["h_a"] == "a", "y_scaled"] = 80 - df["y_scaled"]

    home_shots = len(df[
        (df["h_a"] == "h") & (df["home_team"] == team_name)
    ])

    away_shots = len(df[
        (df["h_a"] == "a") & (df["away_team"] == team_name)
    ])

    deduped = all_shots_df.drop_duplicates(subset=["match_id", "x_scaled", "y_scaled", "team"])

    team_shots = deduped[
        ((deduped["h_a"] == "h") & (deduped["home_team"] == team_name)) |
        ((deduped["h_a"] == "a") & (deduped["away_team"] == team_name))
    ]
    total_shots = len(team_shots)

    # ✅ Load goals from league table
    league_table_path = "data/tables/league_table_data.csv"
    league_df = pd.read_csv(league_table_path)
    team_row = league_df[league_df["Team"] == team_name]
    total_goals = int(team_row.iloc[0]["G"]) if not team_row.empty else 0
    total_xg = float(team_row.iloc[0]["xG"]) if not team_row.empty else 0.0


    # Draw pitch
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#f4f4f9', line_color='black', line_zorder=2, half=True)
    fig, ax = pitch.draw(figsize=(12, 9))
    fig.patch.set_facecolor("#f4f4f9")

    # Remove duplicates
    subset_columns = [col for col in ["match_id", "player", "x_scaled", "y_scaled"] if col in df.columns]
    if subset_columns:
        df = df.drop_duplicates(subset=subset_columns)

    # Plot each shot
    for _, shot in df.iterrows():
        x, y = shot['x_scaled'], shot['y_scaled']
        color = 'gold' if shot['result'].lower() == 'goal' else 'white'
        zorder = 3 if shot['result'].lower() == 'goal' else 2
        size = 500 * float(shot['xG']) if pd.notna(shot['xG']) else 100
        pitch.scatter(x, y, s=size, c=color, edgecolors='black', ax=ax, zorder=zorder)

    # Title and labels
    ax.text(10, 55, f"Shots: {total_shots}", ha='left', va='center', fontsize=20)
    ax.text(40, 55, f"Goals: {total_goals}", ha='center', va='center', fontsize=20)
    ax.text(70, 55, f"xG: {total_xg:.2f}", ha='right', va='center', fontsize=20)
    ax.text(4, 119, "FiveStat", ha='right', va='center', fontsize=8, alpha=0.3)


    # Save
    shotmap_filename = f"{formatted_filename}_shotmap.png"
    plt.savefig(os.path.join(TEAM_SHOTMAP_DIR, shotmap_filename))
    plt.close(fig)
    print(f"Saved {standardized_team_name} shotmap to {TEAM_SHOTMAP_DIR}{shotmap_filename}")








def plot_match_shotmap(home_team, away_team, match_shots_df):
    if match_shots_df.empty:
        print(f"No shot data found for {home_team} vs {away_team}")
        return

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#f4f4f9', line_color='black', line_zorder=2, half=True)
    fig, ax = pitch.draw(figsize=(12, 9))
    fig.patch.set_facecolor("#f4f4f9")

    for _, shot in match_shots_df.iterrows():
        x, y = shot['x_scaled'], shot['y_scaled']
        color = 'gold' if str(shot['result']).lower() == 'goal' else 'white'
        zorder = 3 if str(shot['result']).lower() == 'goal' else 2
        size = 500 * float(shot['xG']) if pd.notna(shot['xG']) else 100
        pitch.scatter(x, y, s=size, c=color, edgecolors='black', ax=ax, zorder=zorder)

    match_filename = f"{home_team}_{away_team}_shotmap.png"
    filepath = os.path.join(SHOTMAP_DIR, match_filename)
    plt.title(f"{home_team} vs {away_team}", fontsize=15)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"✅ Saved match shotmap: {filepath}")






    



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

        # ✅ Get team names and standardize them
        match_info = completed_fixtures[completed_fixtures["id"] == understat_match_id]
        if match_info.empty:
            print(f"Match {understat_match_id} not found in fixture list")
            return

        raw_home_team = match_info["home_team"].values[0]
        raw_away_team = match_info["away_team"].values[0]

        home_team = TEAM_NAME_MAPPING.get(raw_home_team, raw_home_team)
        away_team = TEAM_NAME_MAPPING.get(raw_away_team, raw_away_team)

        # ✅ Process shots for both teams
        for team_side, shots in [('home', data['h']), ('away', data['a'])]:
            df = pd.DataFrame(shots)
            if df.empty:
                continue

            df['team'] = home_team if team_side == 'home' else away_team

            if team_side == 'away':
                df['X'] = 1 - df['X']  # Flip X for consistent pitch view

            df['x_scaled'] = df['X'].astype(float) * 120
            df['y_scaled'] = df['Y'].astype(float) * 80

            # Debug sample
            print(f"🏟️ Processed shots for {df['team'].iloc[0]} ({team_side}) - First 3 shots:")
            print(df[['x_scaled', 'y_scaled']].head(3))

            # ✅ Store standardized version
            team_name = df['team'].iloc[0].strip()
            team_name_standardized = TEAM_NAME_MAPPING.get(team_name, team_name)
            team_shots[team_name_standardized] = df.copy()

    except Exception as e:
        print(f"❌ Error processing match {understat_match_id}: {e}")



for team in team_shots.keys():
    plot_team_shotmap(team)  # ✅ Now uses **all** available shots for the season

print("✅ All Team Shotmaps Generated! 🎯⚽")