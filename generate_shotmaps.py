import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from io import BytesIO
from urllib.request import urlopen


# Ensure directories exist
SHOTMAP_DIR = "static/shotmaps/"
ALL_SHOTMAP_DIR = os.path.join(SHOTMAP_DIR, "all/")
TEAM_SHOTMAP_DIR = os.path.join(SHOTMAP_DIR, "team/")

os.makedirs(ALL_SHOTMAP_DIR, exist_ok=True)
os.makedirs(TEAM_SHOTMAP_DIR, exist_ok=True)


# Data paths
SHOTS_DATA_PATH = "data/tables/shots_data.csv"
FIXTURE_DATA_PATH = "data/tables/fixture_data.csv"


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

# Crest sources do not always mirror provider team names. Keep the aliases
# explicit so canonical team names can be used everywhere else.
TEAM_LOGO_SOURCES = {
    "Coventry City": "https://raw.githubusercontent.com/luukhopman/football-logos/master/logos/England%20-%20Premier%20League/Coventry%20City.png",
    "Hull City": "https://raw.githubusercontent.com/luukhopman/football-logos/master/logos/England%20-%20Premier%20League/Hull%20City.png",
    "Ipswich Town": "ipswich_logo.png",
}


# ---------------------------------------------------------------------------
# LOAD CURRENT-SEASON SHOT DATA
# ---------------------------------------------------------------------------

if os.path.exists(SHOTS_DATA_PATH) and os.path.exists(FIXTURE_DATA_PATH):

    all_shots_df = pd.read_csv(SHOTS_DATA_PATH)
    fixtures_df = pd.read_csv(FIXTURE_DATA_PATH)

    # Convert isResult safely in case it is stored as text rather than bool
    if "isResult" in fixtures_df.columns:
        fixtures_df["isResult"] = (
            fixtures_df["isResult"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    # Only completed fixtures from the current fixture dataset.
    # fixture_data.csv represents the active/current EPL season.
    completed_fixtures = fixtures_df[
        fixtures_df["isResult"] == True
    ].copy()

    # Create set of current-season completed match IDs
    current_match_ids = set(
        pd.to_numeric(
            completed_fixtures["id"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
    )

    # Make shot match IDs comparable
    all_shots_df["match_id"] = pd.to_numeric(
        all_shots_df["match_id"],
        errors="coerce"
    )

    # Keep ONLY current-season completed fixtures
    all_shots_df = all_shots_df[
        all_shots_df["match_id"].isin(current_match_ids)
    ].copy()

    # Determine the shooting team
    all_shots_df["team"] = all_shots_df.apply(
        lambda row: (
            row["h_team"]
            if row["h_a"] == "h"
            else row["a_team"]
        ),
        axis=1
    )

    # Standardise team names
    all_shots_df["team"] = (
        all_shots_df["team"]
        .replace(TEAM_NAME_MAPPING)
        .str.strip()
    )

    print(
        f"✅ Loaded {len(all_shots_df)} shots "
        f"from {all_shots_df['match_id'].nunique()} "
        f"current-season matches"
    )

else:
    print("⚠️ Shot data or fixture data not found! Exiting...")
    exit()


# Ensure shot data is available before processing
if all_shots_df.empty or "team" not in all_shots_df.columns:
    print("⚠️ No current-season shot data available. Skipping shotmap generation.")
    exit()


# Process all current-season shots per team
team_shots = {
    team: all_shots_df[all_shots_df["team"] == team]
    for team in all_shots_df["team"].unique()
}


# ---------------------------------------------------------------------------
# TEAM SHOTMAP
# ---------------------------------------------------------------------------

def plot_team_shotmap(team_name):

    standardized_team_name = TEAM_NAME_MAPPING.get(
        team_name.strip(),
        team_name
    )

    df = all_shots_df[
        all_shots_df["team"] == standardized_team_name
    ].copy()

    if df.empty:
        print(f"No shots found for {team_name}")
        return

    # Normalise shot direction.
    #
    # Existing FiveStat shot data has home-team coordinates requiring
    # inversion here so all attacks point toward the same goal.
    home_mask = df["h_a"] == "h"

    df.loc[home_mask, "x_scaled"] = (
        120 - df.loc[home_mask, "x_scaled"]
    )

    df.loc[home_mask, "y_scaled"] = (
        80 - df.loc[home_mask, "y_scaled"]
    )

    BG = "#f5f5f0"

    pitch = VerticalPitch(
        pitch_type="statsbomb",
        pitch_color=BG,
        line_color="#888882",
        line_zorder=2,
        line_alpha=0.5,
        half=True
    )

    fig, ax = pitch.draw(figsize=(8, 10))

    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Remove duplicates before plotting
    subset_columns = [
        col
        for col in [
            "match_id",
            "player",
            "x_scaled",
            "y_scaled"
        ]
        if col in df.columns
    ]

    if subset_columns:
        df = df.drop_duplicates(
            subset=subset_columns
        )

    # KDE density heatmap
    cmap = LinearSegmentedColormap.from_list(
        "fivestat",
        [BG, "#0a2540"]
    )

    if len(df) >= 5:
        pitch.kdeplot(
            df["x_scaled"],
            df["y_scaled"],
            ax=ax,
            fill=True,
            cmap=cmap,
            n_levels=100,
            thresh=0,
            zorder=1,
            alpha=0.85
        )

    # Club badge
    base_path = os.path.dirname(
        os.path.abspath(__file__)
    )

    standardized_filename = (
        standardized_team_name
        .lower()
        .replace("'", "")
        .replace("’", "")
    )

    logo_source = TEAM_LOGO_SOURCES.get(
        standardized_team_name,
        f"{standardized_filename}_logo.png"
    )

    logo_img = None
    try:
        if logo_source.startswith(("http://", "https://")):
            with urlopen(logo_source, timeout=10) as response:
                logo_img = mpimg.imread(
                    BytesIO(response.read()),
                    format="png"
                )
        else:
            logo_path = os.path.join(
                base_path,
                "static",
                "team_logos",
                logo_source
            )
            if os.path.exists(logo_path):
                logo_img = mpimg.imread(logo_path)
    except (OSError, ValueError):
        # A crest should never prevent the shotmap itself from rendering.
        logo_img = None

    if logo_img is not None:

        aspect_ratio = (
            logo_img.shape[0]
            / logo_img.shape[1]
        )

        height = 30
        width = height / aspect_ratio

        ax.imshow(
            logo_img,
            extent=(
                40 - width / 2,
                40 + width / 2,
                75,
                75 + height
            ),
            alpha=0.08,
            zorder=2
        )

    # Ensure xG is numeric
    df["xG"] = pd.to_numeric(
        df["xG"],
        errors="coerce"
    ).fillna(0.05)

    result_text = (
        df["result"]
        .astype(str)
        .str.lower()
    )

    goals_df = df[
        result_text.str.contains(
            "goal",
            na=False
        )
        &
        ~result_text.str.contains(
            "owngoal",
            na=False
        )
    ]

    non_goals_df = df[
        ~df.index.isin(
            goals_df.index
        )
    ]

    # Non-goal shots
    pitch.scatter(
        non_goals_df["x_scaled"],
        non_goals_df["y_scaled"],
        s=non_goals_df["xG"] * 120,
        c="white",
        edgecolors="#888882",
        linewidths=0.4,
        alpha=0.35,
        zorder=3,
        ax=ax
    )

    # Goals
    pitch.scatter(
        goals_df["x_scaled"],
        goals_df["y_scaled"],
        s=goals_df["xG"] * 400,
        c="#FFD700",
        edgecolors="#b8860b",
        linewidths=0.6,
        alpha=0.9,
        zorder=4,
        ax=ax
    )

    # Watermark
    fig.text(
        0.92,
        0.04,
        "FiveStat",
        fontsize=8,
        color="#888882",
        fontweight="bold",
        ha="right",
        va="bottom",
        alpha=0.5
    )

    # Save
    shotmap_filename = (
        f"{standardized_team_name}_shotmap.png"
    )

    output_path = os.path.join(
        TEAM_SHOTMAP_DIR,
        shotmap_filename
    )

    plt.savefig(
        output_path,
        facecolor=BG,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"✅ Saved {standardized_team_name} "
        f"shotmap to {output_path}"
    )





# ---------------------------------------------------------------------------
# GENERATE SHOTMAPS
# ---------------------------------------------------------------------------



# Individual team shotmaps
for team in team_shots.keys():
    plot_team_shotmap(team)


print(
    "✅ All Current-Season Shotmaps Generated! 🎯⚽"
)
