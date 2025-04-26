from flask import Flask, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Place this line BEFORE importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
from mplsoccer import Radar
import os
from matplotlib import rcParams
from math import pi

app = Flask(__name__)

# Load player data
radar_data_file_path = "data/tables/player_radar_data.csv"
df = pd.read_csv(radar_data_file_path)

# Filter only Premier League players and exclude goalkeepers
df_premier_league = df[(df['Comp'] == 'eng Premier League') & (df['Pos'] != 'GK')]

# Stats to compare
columns_to_plot = [
    'Goals', 'Assists', 'Goals + Assists', 'Expected Goals', 
    'Expected Assists', 'Progressive Carries', 'Progressive Passes', 'Progressive Receptions'
]

# Drop any players where the above are 0
df_premier_league = df_premier_league.dropna(subset=columns_to_plot)

# Calculate league average for comparison
average_stats = df[columns_to_plot].mean().values.flatten().tolist()

@app.route('/generate_radar', methods=['GET'])
def generate_radar_chart():
    player_name = request.args.get('player')
    
    # Ensure player is valid and exists in the dataset
    if player_name not in df_premier_league['Player'].values:
        return jsonify({"error": "Player not found"}), 404

    # Get player stats
    player_data = df_premier_league[df_premier_league['Player'] == player_name]

    if player_data[columns_to_plot].isnull().any().any():
        return jsonify({"error": "Incomplete player data"}), 400

    player_stats = player_data[columns_to_plot].values.flatten().tolist()

    # Create Radar Chart
    radar = Radar(
        params=columns_to_plot,
        min_range=[0 for _ in columns_to_plot],
        max_range=[100 for _ in columns_to_plot]
    )

    fig, ax = radar.setup_axis()
    fig.patch.set_facecolor('#f4f4f9')
    ax.set_facecolor('#f4f4f9')
    
    radar.draw_circles(ax=ax, facecolor='#f4f4f9', edgecolor='black', lw=1, zorder=1)

    radar.draw_radar_compare(
        ax=ax,
        values=player_stats,
        compare_values=average_stats,
        kwargs_radar={'facecolor': '#669bbc', 'alpha': 0.6},
        kwargs_compare={'facecolor': '#e63946', 'alpha': 0.6}
    )

    radar.draw_range_labels(ax=ax, fontsize=15, fontproperties="monospace")
    radar.draw_param_labels(ax=ax, fontsize=15, fontproperties="monospace")

    ax.text(0.2, 1.02, player_name, fontsize=15, ha='center', transform=ax.transAxes, color='#669bbc')
    ax.text(0.8, 1.02, 'League Avg', fontsize=15, ha='center', transform=ax.transAxes, color='#e63946')

    # Additional info text
    ax.text(
        x=0.5, y=0.05, 
        s='Metrics show per 90 stats\ncompared againt all players\nin The Premier League\n\n@Five_Stat', 
        fontsize=11, ha='left', va='center', transform=ax.transAxes, fontfamily='monospace'
    )

    # Save to BytesIO instead of file
    img_io = BytesIO()
    plt.savefig(img_io, format='png', facecolor=fig.get_facecolor(), dpi=300)
    img_io.seek(0)

    plt.close(fig)  # Free up memory

    return send_file(img_io, mimetype='image/png')



# Function to generate a radar chart comparing two players
def generate_comparison_radar_chart(player1, player2, player1_stats, player2_stats):
    radar = Radar(
        params=columns_to_plot,
        min_range=[0 for _ in columns_to_plot],
        max_range=[100 for _ in columns_to_plot]
    )

    # Set up the radar chart
    fig, ax = radar.setup_axis()
    
    # Set background color
    fig.patch.set_facecolor('#f4f4f9')
    ax.set_facecolor('#f4f4f9')

    # Draw radar chart circles
    radar.draw_circles(ax=ax, facecolor='white', edgecolor='black', lw=1, zorder=1)

    # Plot Player 1 (Red) vs Player 2 (Blue)
    radar.draw_radar_compare(
        ax=ax,
        values=player1_stats,
        compare_values=player2_stats,
        kwargs_radar={'facecolor': '#e63946', 'alpha': 0.6},
        kwargs_compare={'facecolor': '#669bbc', 'alpha': 0.6}
    )

    # Add axis labels (stats like Goals, Assists, etc.)
    radar.draw_range_labels(ax=ax, fontsize=15, fontproperties="monospace")
    radar.draw_param_labels(ax=ax, fontsize=15, fontproperties="monospace")

    # Add player names
    ax.text(0.2, 1.02, player1, fontsize=15, ha='center', transform=ax.transAxes, color='#e63946')
    ax.text(0.8, 1.02, player2, fontsize=15, ha='center', transform=ax.transAxes, color='#669bbc')

    # Additional info text
    ax.text(
        x=0.5, y=0.05, 
        s='Metrics show per 90 stats\ncompared againt all players\nin The Premier League\n\n@Five_Stat', 
        fontsize=11, ha='left', va='center', transform=ax.transAxes, fontfamily='monospace'
    )

    return fig, ax




# Load league table data
LEAGUE_DATA_PATH = "data/tables/league_table_data.csv"
SAVE_DIR = "static/radar/teams"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(LEAGUE_DATA_PATH)

# Sort by PTS descending to simulate actual table position
df = df.sort_values(by="PTS", ascending=False).reset_index(drop=True)

# Assign position as index + 1
df["Pos"] = df.index + 1


# Metrics to include and how to calculate them
METRICS = {
    "Win%": lambda row: row["W"] / row["MP"] if row["MP"] else 0,
    "Goals/Match": lambda row: float(row["G"]) / row["MP"] if row["MP"] else 0,
    "xG/Match": lambda row: float(row["xG"]) / row["MP"] if row["MP"] else 0,
    "PTS/Match": lambda row: row["PTS"] / row["MP"] if row["MP"] else 0,
    "xPTS/Match": lambda row: row["xPTS"] / row["MP"] if row["MP"] else 0,
    "Pos": lambda row: row["Pos"]  
}

# Load data
if not os.path.exists(LEAGUE_DATA_PATH):
    raise FileNotFoundError("League table data not found.")

df.dropna(subset=["G", "xG", "MP"], inplace=True)

# Compute raw metrics for all teams
for label, func in METRICS.items():
    df[label] = df.apply(func, axis=1)

# Convert metrics to percentiles
for label in METRICS.keys():
    if label == "Pos":
        df[label + "_pct"] = df[label].rank(pct=True, ascending=False) * 100
    else:
        df[label + "_pct"] = df[label].rank(pct=True, ascending=True) * 100


# Generate radar chart for each team
for _, row in df.iterrows():
    team = row["Team"]

    # 1. Define your labels and base values
    labels = list(METRICS.keys())               # 6 metrics
    values = [row[m + "_pct"] for m in labels]  # 6 values

    # 2. Close the loop for plotting
    values += values[:1]  # 7 points for plotting only

    # ✅ 3. Only create angles for 6 segments
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]  # 7 angles to match values

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f4f4f9")
    ax.set_facecolor("#f4f4f9")
    ax.set_ylim(0, 100)  # ✅ Fix the radar radius to full 0–100

    ax.plot(angles, values, linewidth=2, linestyle='solid', label=team, color='#669bbc')
    ax.fill(angles, values, alpha=0.25)

    # Add League Average line at 50% (styled like team radar)
    avg_values = [50] * len(labels)
    avg_values += avg_values[:1]  # Close the loop

    # Solid red line + filled area
    ax.plot(angles, avg_values, color='#e63946', linewidth=2, linestyle='solid', label='League Average')
    ax.fill(angles, avg_values, color='#e63946', alpha=0.25)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=8)


    # ✅ Only use first 6 angles for grid ticks and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_yticklabels([])

    # Additional info text
    ax.text(
        x=-0.1, y=0.1, 
        s='Metrics show Percentile\nstats compared with the\nrest of the league\n\n@FiveStat', 
        fontsize=6, ha='left', va='center', transform=ax.transAxes, fontfamily='monospace'
    )

    # 5. Save the image
    save_path = os.path.join(SAVE_DIR, f"{team}_team_radar.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ Saved: {save_path}")



#if __name__ == '__main__':
#    app.run(debug=True)