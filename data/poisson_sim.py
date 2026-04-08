import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import poisson

def simulate_poisson_distribution(home_xg, away_xg, number_of_goals=6):
    """Simulate a Poisson-distributed score matrix."""
    score_matrix = np.zeros((number_of_goals, number_of_goals))
    for home_goals in range(number_of_goals):
        for away_goals in range(number_of_goals):
            home_prob = poisson.pmf(home_goals, home_xg)
            away_prob = poisson.pmf(away_goals, away_xg)
            score_matrix[home_goals][away_goals] = home_prob * away_prob
    return score_matrix

def generate_heatmap(result_matrix, home_team, away_team, file_path):
    # Create a DataFrame from the result_matrix for plotting
    result_df = pd.DataFrame(result_matrix, index=range(result_matrix.shape[0]), columns=range(result_matrix.shape[1]))
    
    # Set up the figure with the desired size
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the heatmap
    heatmap = ax.imshow(result_matrix, cmap="Purples", origin='upper')

    # Add titles and labels
    #ax.set_title(f"{home_team} vs {away_team} Score Prediction Heatmap", fontsize=15, pad=20)
    ax.set_xlabel(f"{away_team} Goals", labelpad=20)
    ax.set_ylabel(f"{home_team} Goals", labelpad=20)

    # Move the away team goals (x-axis) to the top
    ax.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax.set_xticks(range(result_matrix.shape[1]))  # Set the tick positions for the away team goals
    ax.set_xticklabels(range(result_matrix.shape[1]))  # Set the tick labels for the away team goals
    ax.xaxis.set_ticks_position('top')  # Ensure the x-axis ticks appear at the top
    
    # Remove the little lines next to the numbers on both axes
    ax.tick_params(axis='x', length=0)  # Remove x-axis tick marks
    ax.tick_params(axis='y', length=0)  # Remove y-axis tick marks
    
    # Remove the outer spines (lines around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add percentage values in each cell
    for i in range(result_matrix.shape[0]):
        for j in range(result_matrix.shape[1]):
            value = result_matrix[i, j]
            percentage = f"{value * 100:.0f}%"  # Convert to percentage & apply <1% formatting
            ax.text(j, i, percentage, ha='center', va='center', color='black', fontsize=8)

    # Ensure the directory exists
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the heatmap to the specified file path
    file_path = 'C:/Users/jmaher/Documents/flask_heatmap_app/static/heatmaps/' #
    file_name = f"{home_team}_vs_{away_team}.png"
    full_file_path = os.path.join(file_path, file_name)
    plt.savefig(full_file_path, bbox_inches='tight')

    # Close the plot to avoid display, since we're saving it
    plt.close()

