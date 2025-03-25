import os
import pandas as pd
from mplsoccer import Radar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


# generating player radar charts

# Load your data (adjust the path if needed)
radar_data_file_path = "data/tables/player_radar_data.csv"
df = pd.read_csv(radar_data_file_path)

# Define the stats you want to compare
columns_to_plot = [
    'Goals', 'Assists', 'Goals + Assists', 'Expected Goals', 
    'Expected Assists', 'Progressive Carries', 'Progressive Passes', 'Progressive Receptions'
]

# Group by 'Player' and aggregate the stats by calculating the mean for each player
df_grouped = df.groupby('Player')[columns_to_plot].sum().reset_index()

# Calculate the average values for each stat
average_stats = df[columns_to_plot].mean().values.flatten().tolist()

# Ensure that the lengths match
assert len(average_stats) == len(columns_to_plot)

# Define the save path
radar_save_path = "static/radar/"
# Ensure the directory exists
if not os.path.exists(radar_save_path):
    os.makedirs(radar_save_path)

# Function to generate radar charts for each player
def generate_radar_chart(player_name, player_stats, average_stats):

    # Ensure player_stats and average_stats have the correct length
    if len(player_stats) != len(columns_to_plot) or len(average_stats) != len(columns_to_plot):
        print(f"Skipping {player_name} due to mismatch in stat length.")
        return None  # Skip this player

    radar = Radar(
        params=columns_to_plot,
        min_range=[0 for _ in columns_to_plot],
        max_range=[100 for _ in columns_to_plot]
    )

    # Set up the plot
    fig, ax = radar.setup_axis()

    # Set background color for the figure and axis
    fig.patch.set_facecolor('#f4f4f9')  # Background color for the entire figure
    ax.set_facecolor('#f4f4f9')  # Background for the radar plot area

    # Draw circles for the radar chart
    radar.draw_circles(ax=ax, facecolor='#f4f4f9', edgecolor='black', lw=1, zorder=1)

    # Plot the player's stats and average stats on the radar chart
    radar_output = radar.draw_radar_compare(
        ax=ax,
        values=player_stats,
        compare_values=average_stats,
        kwargs_radar={'facecolor': '#669bbc', 'alpha': 0.6},
        kwargs_compare={'facecolor': '#e63946', 'alpha': 0.6}
    )

    # Draw the range and parameter labels
    radar.draw_range_labels(ax=ax, fontsize=15, fontproperties="monospace")
    radar.draw_param_labels(ax=ax, fontsize=15, fontproperties="monospace")

    # Add player name text
    ax.text(
        x=0.2, y=1.02, s=player_name, fontsize=15,
        ha='center', va='center', transform=ax.transAxes,
        fontfamily='monospace', color='#669bbc'
    )

    # Add "Average" label text
    ax.text(
        x=0.8, y=1.02, s='Avg', fontsize=15,
        ha='center', va='center', transform=ax.transAxes,
        fontfamily='monospace', color='#e63946'
    )

    # Add additional info text
    ax.text(
        x=0, y=0.05, s='Metrics show per 90 stats\n\nComparing against all Players\nwith at least 400 minutes played\nin Europes Top 5 Domestic Leagues\n\n@FiveStat', 
        fontsize=10, ha='left', va='center', transform=ax.transAxes, fontfamily='monospace'
    )

    return fig, ax


# Generate player comparion radar charts
def generate_comparison_radar_chart(player1, player2, player1_stats, player2_stats):
    radar = Radar(
        params=columns_to_plot,
        min_range=[0 for _ in columns_to_plot],
        max_range=[100 for _ in columns_to_plot]
    )

    # Set background color for the figure and axis
    fig.patch.set_facecolor('#f4f4f9')  # Background color for the entire figure
    ax.set_facecolor('#f4f4f9')  # Background for the radar plot area

    fig, ax = radar.setup_axis()
    radar.draw_circles(ax=ax, facecolor='white', edgecolor='black', lw=1, zorder=1)

    # Plot Player 1 (Red) vs Player 2 (Blue)
    radar.draw_radar_compare(
        ax=ax,
        values=player1_stats,
        compare_values=player2_stats,
        kwargs_radar={'facecolor': '#e63946', 'alpha': 0.6},
        kwargs_compare={'facecolor': '#669bbc', 'alpha': 0.6}
    )

    # ✅ Add Axis Labels (Goals, Assists, etc.)
    radar.draw_range_labels(ax=ax, fontsize=15, fontproperties="monospace")
    radar.draw_param_labels(ax=ax, fontsize=15, fontproperties="monospace")

    # ✅ Add Player Names
    ax.text(0.2, 1.02, player1, fontsize=15, ha='center', transform=ax.transAxes, color='#e63946')
    ax.text(0.8, 1.02, player2, fontsize=15, ha='center', transform=ax.transAxes, color='#669bbc')

    # ✅ Add Additional Info Text
    ax.text(
        x=0, y=0.05, 
        s='Metrics show per 90 stats\n\nComparing against all Players\nwith at least 400 minutes played\nin Europes Top 5 Domestic Leagues\n\n@FiveStat', 
        fontsize=10, ha='left', va='center', transform=ax.transAxes, fontfamily='monospace'
    )

    return fig, ax





# Loop through each player and generate the radar chart
for player_name in df['Player'].unique():
    # Retrieve player stats and ensure they are valid (drop rows with NaN values in the stats columns)
    player_data = df[df['Player'] == player_name]
    
    # Ensure that player has valid data (no NaNs in the relevant columns)
    if player_data[columns_to_plot].isnull().any().any():
        print(f"Skipping {player_name} due to missing data.")
        continue

    # Get player stats as a list for radar plot
    player_stats = player_data[columns_to_plot].values.flatten().tolist()
    
    # Generate radar chart for the player compared to the average
    result = generate_radar_chart(player_name, player_stats, average_stats)
    
    # Check if result is None (which means the chart couldn't be generated)
    if result is None:
        continue
    
    fig, ax = result  # Only unpack if result is not None
    
    # Define the radar chart filename
    radar_filename = f"{player_name}_radar.png" 
    
    # Save the radar chart to the specified folder
    plt.savefig(os.path.join(radar_save_path, radar_filename), facecolor=fig.get_facecolor(), dpi=300)
    
    # Close the figure to free up memory
    plt.close()

print("Radar charts generated and saved.")
