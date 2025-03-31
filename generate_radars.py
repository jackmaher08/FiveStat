from flask import Flask, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Place this line BEFORE importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
from mplsoccer import Radar

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
    ax.text(0.8, 1.02, 'Avg', fontsize=15, ha='center', transform=ax.transAxes, color='#e63946')

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
        x=0, y=0.05, 
        s='Metrics show per 90 stats\n\n@FiveStat', 
        fontsize=10, ha='left', va='center', transform=ax.transAxes, fontfamily='monospace'
    )

    return fig, ax

if __name__ == '__main__':
    app.run(debug=True)