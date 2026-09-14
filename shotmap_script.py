import os
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.colors import LinearSegmentedColormap


BG = '#f5f5f0'

SAVE_PATH = 'static/shotmaps/all/'
SHOTS_DATA_PATH = 'data/tables/shots_data.csv'
FIXTURE_DATA_PATH = 'data/tables/fixture_data.csv'

os.makedirs(SAVE_PATH, exist_ok=True)


# ---------------------------------------------------------------------------
# LOAD CURRENT-SEASON FIXTURES
# ---------------------------------------------------------------------------

fixtures_df = pd.read_csv(FIXTURE_DATA_PATH)

# Handle bools safely whether stored as True/False or strings
fixtures_df['isResult'] = (
    fixtures_df['isResult']
    .astype(str)
    .str.lower()
    .isin(['true', '1', 'yes'])
)

completed_fixtures = fixtures_df[
    fixtures_df['isResult']
].copy()

current_match_ids = set(
    pd.to_numeric(
        completed_fixtures['id'],
        errors='coerce'
    )
    .dropna()
    .astype(int)
)


# ---------------------------------------------------------------------------
# LOAD SHOT DATA
# ---------------------------------------------------------------------------

shots_df = pd.read_csv(SHOTS_DATA_PATH)

shots_df['match_id'] = pd.to_numeric(
    shots_df['match_id'],
    errors='coerce'
)

shots_df['xG'] = pd.to_numeric(
    shots_df['xG'],
    errors='coerce'
)

shots_df['x_scaled'] = pd.to_numeric(
    shots_df['x_scaled'],
    errors='coerce'
)

shots_df['y_scaled'] = pd.to_numeric(
    shots_df['y_scaled'],
    errors='coerce'
)


# ---------------------------------------------------------------------------
# FILTER TO CURRENT SEASON ONLY
# ---------------------------------------------------------------------------

shots_df = shots_df[
    shots_df['match_id'].isin(current_match_ids)
].copy()

shots_df = shots_df.dropna(
    subset=[
        'x_scaled',
        'y_scaled',
        'xG'
    ]
)

# Remove accidental duplicates
duplicate_cols = [
    col
    for col in [
        'match_id',
        'player',
        'x_scaled',
        'y_scaled'
    ]
    if col in shots_df.columns
]

if duplicate_cols:
    shots_df = shots_df.drop_duplicates(
        subset=duplicate_cols
    )


print(
    f"✅ All-shots plot using {len(shots_df)} shots "
    f"from {shots_df['match_id'].nunique()} "
    f"current-season matches"
)


if shots_df.empty:
    raise RuntimeError(
        "No current-season shots found. "
        "all_shots.png was not generated."
    )


# ---------------------------------------------------------------------------
# SPLIT GOALS / NON-GOALS
# ---------------------------------------------------------------------------

result_text = (
    shots_df['result']
    .astype(str)
    .str.lower()
)

goal_mask = (
    result_text.str.contains(
        'goal',
        na=False
    )
    &
    ~result_text.str.contains(
        'owngoal',
        na=False
    )
)

goals_df = shots_df[
    goal_mask
]

non_goals_df = shots_df[
    ~goal_mask
]


# ---------------------------------------------------------------------------
# DRAW PITCH
# ---------------------------------------------------------------------------

pitch = Pitch(
    pitch_type='statsbomb',
    pitch_color=BG,
    line_color='#888882',
    line_zorder=2,
    line_alpha=0.5
)

fig, ax = pitch.draw(
    figsize=(16, 10)
)

fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# ---------------------------------------------------------------------------
# DENSITY HEATMAP
# ---------------------------------------------------------------------------

cmap = LinearSegmentedColormap.from_list(
    'fivestat',
    [
        BG,
        '#0a2540'
    ]
)

if len(shots_df) >= 5:
    pitch.kdeplot(
        shots_df['x_scaled'],
        shots_df['y_scaled'],
        ax=ax,
        fill=True,
        cmap=cmap,
        n_levels=100,
        thresh=0,
        zorder=1,
        alpha=0.85
    )


# ---------------------------------------------------------------------------
# NON-GOAL SHOTS
# ---------------------------------------------------------------------------

ax.scatter(
    non_goals_df['x_scaled'],
    non_goals_df['y_scaled'],
    s=non_goals_df['xG'] * 120,
    c='white',
    ec='#888882',
    linewidths=0.4,
    alpha=0.35,
    zorder=2
)


# ---------------------------------------------------------------------------
# GOALS
# ---------------------------------------------------------------------------

ax.scatter(
    goals_df['x_scaled'],
    goals_df['y_scaled'],
    s=goals_df['xG'] * 400,
    c='#FFD700',
    ec='#b8860b',
    linewidths=0.6,
    alpha=0.9,
    zorder=3
)





# ---------------------------------------------------------------------------
# WATERMARK
# ---------------------------------------------------------------------------

fig.text(
    0.9,
    0.06,
    'FiveStat',
    fontsize=9,
    color='#888882',
    fontweight='bold',
    ha='right',
    va='bottom',
    alpha=0.5
)


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

output_path = os.path.join(
    SAVE_PATH,
    'all_shots.png'
)

plt.tight_layout()

plt.savefig(
    output_path,
    facecolor=BG,
    dpi=150,
    bbox_inches='tight'
)

plt.close(fig)


print(
    f"✅ all_shots.png saved to {output_path}"
)

print(
    f"📊 Current season: "
    f"{len(shots_df)} shots | "
    f"{len(goals_df)} goals | "
    f"{shots_df['xG'].sum():.1f} xG"
)
