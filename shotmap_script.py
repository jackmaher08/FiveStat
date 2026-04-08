import os
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.colors import LinearSegmentedColormap

BG = '#f5f5f0'
SAVE_PATH = 'static/shotmaps/all/'
os.makedirs(SAVE_PATH, exist_ok=True)

# Load all shot data built by data_loader.py
shots_df = pd.read_csv('data/tables/shots_data.csv')
shots_df['xG']      = pd.to_numeric(shots_df['xG'],      errors='coerce')
shots_df['x_scaled'] = pd.to_numeric(shots_df['x_scaled'], errors='coerce')
shots_df['y_scaled'] = pd.to_numeric(shots_df['y_scaled'], errors='coerce')
shots_df = shots_df.dropna(subset=['x_scaled', 'y_scaled', 'xG'])

goals_df    = shots_df[shots_df['result'].str.contains('Goal', case=False, na=False)]
non_goals_df = shots_df[~shots_df['result'].str.contains('Goal', case=False, na=False)]

pitch = Pitch(
    pitch_type='statsbomb',
    pitch_color=BG,
    line_color='#888882',
    line_zorder=2,
    line_alpha=0.5
)

fig, ax = pitch.draw(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Density heatmap of all shots
cmap = LinearSegmentedColormap.from_list('fivestat', [BG, '#0a2540'])
pitch.kdeplot(
    shots_df['x_scaled'], shots_df['y_scaled'],
    ax=ax, fill=True, cmap=cmap,
    n_levels=100, thresh=0, zorder=1, alpha=0.85
)

# Non-goal shots — small white dots
ax.scatter(
    non_goals_df['x_scaled'], non_goals_df['y_scaled'],
    s=non_goals_df['xG'] * 120,
    c='white', ec='#888882', linewidths=0.4,
    alpha=0.35, zorder=2
)

# Goals — gold, larger, more prominent
ax.scatter(
    goals_df['x_scaled'], goals_df['y_scaled'],
    s=goals_df['xG'] * 400,
    c='#FFD700', ec='#b8860b', linewidths=0.6,
    alpha=0.9, zorder=3
)

# Watermark
fig.text(
    0.9, 0.06, 'FiveStat',
    fontsize=9, color='#888882', fontweight='bold',
    ha='right', va='bottom', alpha=0.5
)

plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_PATH, 'all_shots.png'),
    facecolor=BG, dpi=150, bbox_inches='tight'
)
plt.close(fig)
print('✅ all_shots.png saved to', SAVE_PATH)