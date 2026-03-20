from flask import Flask, render_template
import numpy as np
from scipy.stats import poisson
import os
import matplotlib.pyplot as plt
from data_loader import load_fixtures, load_match_data, calculate_team_statistics, calculate_recent_form, calculate_league_table, get_player_data

# Flask app initialization
app = Flask(__name__)

# Load data
match_data = load_match_data()
team_stats, home_field_advantage = calculate_team_statistics(match_data)
fixtures = load_fixtures().to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
@app.route("/")
def home():
    return render_template("index.html", fixtures=fixtures)  # Just renders, no heatmap generation


@app.route("/about")
def about():
    return render_template("about.html")


# Generate league table data at startup
league_table = calculate_league_table(match_data)

@app.route('/table')
def table():
    return render_template('table.html')  # Ensure table.html exists in templates folder

@app.route('/epl-players')
def epl_players():
    players = get_player_data()

    # Extract unique POS and Teams
    unique_positions = sorted(set(player["POS"] for player in players if player["POS"]))
    unique_teams = sorted(set(player["Team"] for player in players if player["Team"]))

    return render_template('epl_player.html', players=players, positions=unique_positions, teams=unique_teams)

@app.route('/filter-players', methods=['POST'])
def filter_players():
    players = get_player_data()
    
    pos_filter = request.json.get("pos", "")
    team_filter = request.json.get("team", "")

    # Apply filtering
    filtered_players = [
        player for player in players 
        if (not pos_filter or pos_filter in player["POS"]) and 
           (not team_filter or team_filter == player["Team"])
    ]

    return jsonify(filtered_players)

if __name__ == "__main__":
    app.run(debug=True)