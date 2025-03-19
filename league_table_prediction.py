import pandas as pd
import numpy as np

# Load remaining fixtures
fixtures = pd.read_csv("C:/Users/jmaher/Documents/flask_heatmap_app/data/tables/fixture_data.csv")

# Load current league table
league_table = pd.read_csv("C:/Users/jmaher/Documents/flask_heatmap_app/data/tables/league_table_data.csv")

# Extract necessary columns
team_points = league_table.set_index("Team")["PTS"].to_dict()
teams = list(team_points.keys())

# Simulation Parameters
num_simulations = 10000
num_teams = len(teams)
num_positions = num_teams  # Positions 1 to last place

# Create a dictionary to store simulation results
position_counts = {team: np.zeros(num_positions) for team in teams}

# Run Monte Carlo Simulation
for _ in range(num_simulations):
    simulated_points = team_points.copy()

    for _, match in fixtures.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        home_prob = match["home_win_prob"]
        draw_prob = match["draw_prob"]
        away_prob = match["away_win_prob"]

        # Randomly simulate the match outcome
        outcome = np.random.choice(["home_win", "draw", "away_win"], p=[home_prob, draw_prob, away_prob])

        if outcome == "home_win":
            simulated_points[home_team] += 3
        elif outcome == "draw":
            simulated_points[home_team] += 1
            simulated_points[away_team] += 1
        else:  # away win
            simulated_points[away_team] += 3

    # Rank teams based on simulated points
    sorted_teams = sorted(simulated_points.items(), key=lambda x: x[1], reverse=True)

    # Record finishing positions
    for rank, (team, _) in enumerate(sorted_teams):
        position_counts[team][rank] += 1

# Convert to DataFrame with percentages
final_probabilities = pd.DataFrame(position_counts)
final_probabilities = final_probabilities.T  # Transpose
final_probabilities.columns = range(1, num_positions + 1)  # Rename columns to positions
final_probabilities /= num_simulations  # Convert counts to probabilities

# Save as CSV
final_probabilities.to_csv("flask_heatmap_app/data/tables/simulated_league_positions.csv")

print("Simulation complete! Results saved as simulated_league_positions.csv")
