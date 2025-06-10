# generate_player_shots.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import numpy as np

TEAM_NAME_MAPPING = {
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest"
}

# Directory for saving shotmaps
PLAYER_SHOTMAP_DIR = "static/shotmaps/player/"
os.makedirs(PLAYER_SHOTMAP_DIR, exist_ok=True)

# Load shot data
SHOTS_DATA_PATH = "data/tables/shots_data.csv"
if not os.path.exists(SHOTS_DATA_PATH):
    raise FileNotFoundError("❌ shots_data.csv not found.")

df = pd.read_csv(SHOTS_DATA_PATH)

# Preprocess: Add scaled coordinates
df["x_scaled"] = df["X"].astype(float) * 120
df["y_scaled"] = df["Y"].astype(float) * 80

def plot_player_shotmap(player_name):
    player_df = df[df["player"].str.lower() == player_name.lower()].copy()
    if player_df.empty:
        print(f"⚠️ No shots found for {player_name}")
        return

    # Remove duplicate shots (same match, coords)
    player_df = player_df.drop_duplicates(subset=["match_id", "x_scaled", "y_scaled"])


    pitch = VerticalPitch(
    pitch_type='statsbomb',
    pitch_color='#f4f4f9',
    line_color='black',
    half=True,
    line_zorder=4  # ✅ ensure pitch lines are drawn on top
)

    # Define custom purple colormap
    purple_cmap = LinearSegmentedColormap.from_list('custom_purple', ['#f4f4f9', '#3f007d'])
    fig, ax = pitch.draw(figsize=(12, 9))
    fig.patch.set_facecolor("#f4f4f9")

    pitch.kdeplot(
        player_df['x_scaled'], player_df['y_scaled'], ax=ax,
        fill=True, cmap=purple_cmap, n_levels=100, thresh=0, zorder=1
    )

    for _, shot in player_df.iterrows():
        x, y = shot['x_scaled'], shot['y_scaled']
        color = 'gold' if str(shot['result']).lower() == 'goal' else 'white'
        zorder = 3 if str(shot['result']).lower() == 'goal' else 2
        size = 500 * float(shot['xG']) if pd.notna(shot['xG']) else 100
        pitch.scatter(x, y, s=size, c=color, edgecolors='black', ax=ax, zorder=zorder, alpha=0.8)

    # Compute player summary stats
    total_shots = len(player_df)
    total_goals = sum(player_df['result'].str.lower() == 'goal')
    total_xg = player_df['xG'].astype(float).sum()

    # Add summary labels
    ax.text(10, 55, f"Shots: {total_shots}", ha='left', va='center', fontsize=20)
    ax.text(40, 55, f"Goals: {total_goals}", ha='center', va='center', fontsize=20)
    ax.text(70, 55, f"xG: {total_xg:.2f}", ha='right', va='center', fontsize=20)
    ax.text(8, 118, "FiveStat", ha='right', va='center', fontsize=8, alpha=0.3)

    # Player name as title
    ax.text(40, 85, player_name, ha='center', va='center', fontsize=50, fontweight='bold', alpha=0.3)

    filename = f"{player_name.replace(' ', '_')}_shotmap.png"
    filepath = os.path.join(PLAYER_SHOTMAP_DIR, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"✅ Saved {player_name} shotmap to {filepath}")


# Example usage
if __name__ == "__main__":
    players = ["Nicolas Jackson", "Liam Delap", "Jean-Philippe Mateta"]  # Add desired players
    for player in players:
        plot_player_shotmap(player)


import io

def create_player_shotmap_image(player_name, shot_type="all"):
    player_df = df[df["player"].str.lower() == player_name.lower()].copy()
    if player_df.empty:
        return None

    # Fix penalty spots
    player_df.loc[
        (player_df["x_scaled"].between(13.7, 13.9)) & (player_df["y_scaled"] == 40.0),
        "x_scaled"
    ] = 12
    player_df.loc[
        (player_df["x_scaled"].between(106.1, 106.3)) & (player_df["y_scaled"] == 40.0),
        "x_scaled"
    ] = 108

    # Filter if only showing goals
    if shot_type == "goals":
        player_df = player_df[player_df['result'].str.lower() == 'goal']

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#f4f4f9', line_color='black', half=True, line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 8))
    fig.patch.set_facecolor("#f4f4f9")

    # Heatmap only if plotting goals
    #from matplotlib.colors import LinearSegmentedColormap
    #purple_cmap = LinearSegmentedColormap.from_list('custom_purple', ['#f4f4f9', '#3f007d'])
    #if len(player_df) >= 2:
    #    pitch.kdeplot(player_df['x_scaled'], player_df['y_scaled'], ax=ax, fill=True, cmap=purple_cmap, n_levels=100, thresh=0, zorder=1)

    for _, shot in player_df.iterrows():
        x, y = shot['x_scaled'], shot['y_scaled']
        color = 'gold' if str(shot['result']).lower() == 'goal' else 'white'
        pitch.scatter(x, y, s=700 * float(shot['xG']), c=color, edgecolors='black', ax=ax, zorder = 4 if shot['result'].lower() == 'goal' else 3, alpha=0.9)

    total_shots = len(player_df)
    total_goals = sum(player_df['result'].str.lower() == 'goal')
    total_xg = player_df['xG'].astype(float).sum()
    ax.text(10, 55, f"Shots: {total_shots}", ha='left', va='center', fontsize=20)
    ax.text(40, 55, f"Goals: {total_goals}", ha='center', va='center', fontsize=20)
    ax.text(70, 55, f"xG: {total_xg:.2f}", ha='right', va='center', fontsize=20)
    ax.text(5, 119, "FiveStat", ha='right', va='center', fontsize=8, alpha=0.3)
    ax.text(40, 85, player_name, ha='center', va='center', fontsize=50, fontweight='bold', alpha=0.5)

    # Get base path and normalize team name
    # Resolve script-relative logo path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Get and standardize team name
    raw_team_name = player_df["h_team"].iloc[0] if player_df["h_a"].iloc[0] == "h" else player_df["a_team"].iloc[0]
    clean_team_name = raw_team_name.strip()

    standardized_team = TEAM_NAME_MAPPING.get(clean_team_name, clean_team_name)
    standardized_filename = standardized_team.lower().replace("’", "").replace("'", "")
    logo_path = os.path.join(base_path, "static", "team_logos", f"{standardized_filename}_logo.png")


    def add_team_logo(ax, logo_path, y_min, y_max, x_center):
        if os.path.exists(logo_path):
            logo_img = mpimg.imread(logo_path)
            #logo_img = np.flipud(logo_img)
            aspect_ratio = logo_img.shape[0] / logo_img.shape[1]
            height = y_max - y_min
            width = height / aspect_ratio
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            ax.imshow(logo_img, extent=(x_min, x_max, y_min, y_max), alpha=0.05, zorder=1)
        else:
            print(f"⚠️ Logo not found for team: {standardized_team} at {logo_path}")


    # After pitch.draw(...)
    add_team_logo(ax, logo_path, y_min=65, y_max=105, x_center=40)


    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

