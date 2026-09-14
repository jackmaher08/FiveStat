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
    "Nott'm Forest": "Nottingham Forest",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Coventry": "Coventry City",
}


if os.path.exists(SHOTS_DATA_PATH):
    all_shots_df = pd.read_csv(SHOTS_DATA_PATH)

    # Keep only matches from the current season's completed fixtures.
    # This prevents old-season shots being carried into the homepage image.
    current_match_ids = set(
        pd.to_numeric(completed_fixtures["id"], errors="coerce")
        .dropna()
        .astype(int)
    )

    all_shots_df["match_id"] = pd.to_numeric(
        all_shots_df["match_id"],
        errors="coerce"
    )

    all_shots_df = all_shots_df[
        all_shots_df["match_id"].isin(current_match_ids)
    ].copy()

    all_shots_df["team"] = all_shots_df.apply(
        lambda row: row["h_team"] if row["h_a"] == "h" else row["a_team"],
        axis=1
    )

    # Standardize team names
    all_shots_df["team"] = (
        all_shots_df["team"]
        .replace(TEAM_NAME_MAPPING)
        .str.strip()
    )

    print(
        f"✅ Loaded {len(all_shots_df)} shots "
        f"from {all_shots_df['match_id'].nunique()} current-season matches"
    )

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

    # Normalise away shots to attack same direction as home (toward x=120)
    away_mask = df["h_a"] == "h"
    df.loc[away_mask, "x_scaled"] = 120 - df.loc[away_mask, "x_scaled"]
    df.loc[away_mask, "y_scaled"] = 80  - df.loc[away_mask, "y_scaled"]

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


 def plot_all_shots():
    """
    Generate the homepage all-shots image using only the
    current season's completed Premier League fixtures.
    """

    df = all_shots_df.copy()

    if df.empty:
        print("⚠️ No current-season shots available for all_shots.png")
        return

    # Remove any accidental duplicate shots
    subset_columns = [
        col for col in
        ["match_id", "player", "x_scaled", "y_scaled"]
        if col in df.columns
    ]

    if subset_columns:
        df = df.drop_duplicates(subset=subset_columns)

    # Put every shot towards the same goal for the aggregate visual.
    #
    # data_loader already flips home x coordinates. Away shots therefore
    # need their x coordinate flipped here so both sides attack the same end.
    away_mask = df["h_a"] == "a"
    df.loc[away_mask, "x_scaled"] = 120 - df.loc[away_mask, "x_scaled"]

    pitch = VerticalPitch(
        pitch_type="statsbomb",
        pitch_color="#f4f4f9",
        line_color="black",
        line_zorder=2,
        half=True
    )

    fig, ax = pitch.draw(figsize=(14, 10))
    fig.patch.set_facecolor("#f4f4f9")

    for _, shot in df.iterrows():
        x = shot["x_scaled"]
        y = shot["y_scaled"]

        result = str(shot.get("result", "")).lower()

        if "goal" in result and "owngoal" not in result:
            color = "gold"
            zorder = 4
        else:
            color = "white"
            zorder = 3

        try:
            xg = float(shot["xG"])
        except (ValueError, TypeError):
            xg = 0.05

        size = max(30, 500 * xg)

        pitch.scatter(
            x,
            y,
            s=size,
            c=color,
            edgecolors="black",
            linewidth=0.6,
            alpha=0.65,
            ax=ax,
            zorder=zorder
        )

    total_shots = len(df)
    total_goals = (
        df["result"]
        .astype(str)
        .str.lower()
        .eq("goal")
        .sum()
    )

    total_xg = pd.to_numeric(
        df["xG"],
        errors="coerce"
    ).sum()

    ax.text(
        10, 55,
        f"Shots: {total_shots}",
        ha="left",
        va="center",
        fontsize=18
    )

    ax.text(
        40, 55,
        f"Goals: {total_goals}",
        ha="center",
        va="center",
        fontsize=18
    )

    ax.text(
        70, 55,
        f"xG: {total_xg:.1f}",
        ha="right",
        va="center",
        fontsize=18
    )

    ax.text(
        4, 119,
        "FiveStat | Premier League 2026/27",
        ha="right",
        va="center",
        fontsize=9,
        alpha=0.4
    )

    output_path = os.path.join(
        ALL_SHOTMAP_DIR,
        "all_shots.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150
    )

    plt.close(fig)

    print(
        f"✅ Current-season all-shots image generated: {output_path}"
    )



# Generate homepage aggregate shotmap
plot_all_shots()

# Generate individual team shotmaps
for team in team_shots.keys():
    plot_team_shotmap(team)

print("✅ All Current-Season Shotmaps Generated! 🎯⚽")
