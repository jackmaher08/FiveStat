import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg




# Ensure directories exist
SHOTMAP_DIR = "static/shotmaps/"
ALL_SHOTMAP_DIR = os.path.join(SHOTMAP_DIR, "all/")
TEAM_SHOTMAP_DIR = os.path.join(SHOTMAP_DIR, "team/")

os.makedirs(ALL_SHOTMAP_DIR, exist_ok=True)
os.makedirs(TEAM_SHOTMAP_DIR, exist_ok=True)

# ✅ Define the path for saving shots_data.csv
SHOTS_DATA_PATH = "data/tables/shots_data.csv"

TEAM_NAME_MAPPING = {
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest"
}


if os.path.exists(SHOTS_DATA_PATH):
    all_shots_df = pd.read_csv(SHOTS_DATA_PATH)
    all_shots_df["team"] = all_shots_df.apply(
        lambda row: row["h_team"] if row["h_a"] == "h" else row["a_team"],
        axis=1
    )
    # Standardize team names
    all_shots_df["team"] = all_shots_df["team"].replace(TEAM_NAME_MAPPING).str.strip()
else:
    print("⚠️ No shot data found! Exiting...")
    exit()

# ✅ Ensure shot data is available before processing
if all_shots_df.empty or "team" not in all_shots_df.columns:
    print("⚠️ No shot data available. Skipping shotmap generation.")
    exit()

# ✅ Process **all** shots taken this season per team
team_shots = {team: all_shots_df[all_shots_df['team'] == team] for team in all_shots_df['team'].unique()}









def plot_team_shotmap(team_name):

    standardized_team_name = TEAM_NAME_MAPPING.get(team_name.strip(), team_name)
    df = all_shots_df[all_shots_df['team'] == standardized_team_name].copy()

    if df.empty:
        print(f"No shots found for {team_name}")
        return

    # Flip home team shots
    df.loc[df["h_a"] == "h", "x_scaled"] = 120 - df["x_scaled"]
    df.loc[df["h_a"] == "h", "y_scaled"] = 80 - df["y_scaled"]

    BG = '#f5f5f0'

    # Draw pitch
    pitch = VerticalPitch(
        pitch_type='statsbomb', pitch_color=BG,
        line_color='#888882', line_zorder=2, line_alpha=0.5, half=True
    )
    fig, ax = pitch.draw(figsize=(8, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # KDE density heatmap — matches home page style
    cmap = LinearSegmentedColormap.from_list('fivestat', [BG, '#0a2540'])
    if len(df) >= 5:
        pitch.kdeplot(
            df['x_scaled'], df['y_scaled'],
            ax=ax, fill=True, cmap=cmap,
            n_levels=100, thresh=0, zorder=1, alpha=0.85
        )

    # Club badge
    base_path = os.path.dirname(os.path.abspath(__file__))
    standardized_filename = standardized_team_name.lower().replace("'", "").replace("'", "")
    logo_path = os.path.join(base_path, "static", "team_logos", f"{standardized_filename}_logo.png")

    if os.path.exists(logo_path):
        logo_img = mpimg.imread(logo_path)
        aspect_ratio = logo_img.shape[0] / logo_img.shape[1]
        height = 30
        width  = height / aspect_ratio
        ax.imshow(logo_img,
                  extent=(40 - width/2, 40 + width/2, 75, 75 + height),
                  alpha=0.08, zorder=2)

    # Remove duplicates
    subset_columns = [col for col in ["match_id", "player", "x_scaled", "y_scaled"] if col in df.columns]
    if subset_columns:
        df = df.drop_duplicates(subset=subset_columns)

    goals_df     = df[df['result'].str.lower().str.contains('goal')]
    non_goals_df = df[~df['result'].str.lower().str.contains('goal')]

    # Non-goal shots — small white dots
    pitch.scatter(
        non_goals_df['x_scaled'], non_goals_df['y_scaled'],
        s=non_goals_df['xG'].fillna(0.05) * 120,
        c='white', edgecolors='#888882', linewidths=0.4,
        alpha=0.35, zorder=3, ax=ax
    )

    # Goals — gold, more prominent
    pitch.scatter(
        goals_df['x_scaled'], goals_df['y_scaled'],
        s=goals_df['xG'].fillna(0.1) * 400,
        c='#FFD700', edgecolors='#b8860b', linewidths=0.6,
        alpha=0.9, zorder=4, ax=ax
    )

    # Watermark only
    fig.text(0.92, 0.04, 'FiveStat', fontsize=8, color='#888882',
             fontweight='bold', ha='right', va='bottom', alpha=0.5)

    # Save
    shotmap_filename = f"{standardized_team_name}_shotmap.png"
    plt.savefig(os.path.join(TEAM_SHOTMAP_DIR, shotmap_filename),
                facecolor=BG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {standardized_team_name} shotmap to {TEAM_SHOTMAP_DIR}{shotmap_filename}")


 



for team in team_shots.keys():
    plot_team_shotmap(team)  # ✅ Now uses **all** available shots for the season

print("✅ All Team Shotmaps Generated! 🎯⚽")