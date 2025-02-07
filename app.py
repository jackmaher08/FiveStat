from flask import Flask, render_template
import numpy as np
from scipy.stats import poisson
import os
import matplotlib.pyplot as plt
from data_loader import load_fixtures, load_match_data, calculate_team_statistics, calculate_recent_form

# Flask app initialization
app = Flask(__name__)

# Load data
match_data = load_match_data()
team_stats, home_field_advantage = calculate_team_statistics(match_data)

@app.route("/")
def home():
    fixtures = load_fixtures().to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
    return render_template("index.html", fixtures=fixtures)

@app.route("/about")
def about():
    return render_template("about.html")

# Function to simulate a match using Poisson distribution
def simulate_poisson_distribution(home_xg, away_xg, max_goals=12):
    result_matrix = np.zeros((max_goals, max_goals))
    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            home_prob = poisson.pmf(home_goals, home_xg)
            away_prob = poisson.pmf(away_goals, away_xg)
            result_matrix[home_goals, away_goals] = home_prob * away_prob
    result_matrix /= np.sum(result_matrix)
    return result_matrix

# Function to generate a heatmap
def display_heatmap(result_matrix, home_team, away_team, save_path):
    display_matrix = result_matrix[:6, :6]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display_matrix, cmap="Purples", origin='upper')
    ax.set_xlabel(f"{away_team} Goals")
    ax.set_ylabel(f"{home_team} Goals")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{display_matrix[i, j] * 100:.1f}%", ha='center', va='center', color='black', fontsize=8)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{home_team}_vs_{away_team}_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    app.run(debug=True)
