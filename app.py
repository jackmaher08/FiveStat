import pandas as pd
from flask import Flask, render_template, request
import numpy as np
from scipy.stats import poisson
import os
import matplotlib.pyplot as plt
from data_loader import load_fixtures, load_match_data, calculate_team_statistics, load_next_gw_fixtures, get_player_data
import datetime
from datetime import datetime

# Flask app initialization
app = Flask(__name__)

# Load data
match_data = load_match_data()
team_stats, home_field_advantage = calculate_team_statistics(match_data)
fixtures = load_fixtures().to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
@app.route("/")
def home():
    next_gw_fixtures = load_next_gw_fixtures()

    # Get the current gameweek number dynamically
    current_gw = next_gw_fixtures[0]["round_number"] if next_gw_fixtures else "Unknown"

    return render_template("index.html", fixtures=next_gw_fixtures, current_gw=current_gw)



@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/table')
def table():
    return render_template('table.html')  # Ensure table.html exists in templates folder

@app.route('/epl-players')
def epl_players():
    try:
        players = get_player_data()  # Load from the saved file
        next_gw_fixtures = load_next_gw_fixtures()
        current_gw = next_gw_fixtures[0]["round_number"] if next_gw_fixtures else "Unknown"

        # Ensure data is properly structured
        if not isinstance(players, list) or not all(isinstance(player, dict) for player in players):
            print("⚠️ Unexpected player data format")
            players = []

        # Extract unique positions and teams
        unique_positions = sorted(set(player.get("POS", "Unknown") for player in players if isinstance(player, dict)))
        unique_teams = sorted(set(player.get("Team", "Unknown") for player in players if isinstance(player, dict)))

        return render_template('epl_player.html', players=players, positions=unique_positions, teams=unique_teams, current_gw=current_gw)

    except Exception as e:
        print(f"❌ Error loading player data: {e}")
        return render_template('epl_player.html', players=[], positions=[], teams=[], current_gw="Unknown")



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

# Load fixtures
fixtures_df = pd.DataFrame(load_fixtures())

# this lets us determine the current gameweek along with all completed gameweeks
def get_fixtures_for_week(week_offset=0):
    """ Returns fixtures for the selected completed round. """
    
    # ✅ Load fixture data
    fixture_file_path = "data/tables/fixture_data.csv"
    fixtures_df = pd.read_csv(fixture_file_path)

    # Ensure 'round_number' is numeric
    fixtures_df['round_number'] = pd.to_numeric(fixtures_df['round_number'], errors='coerce')

    # ✅ Get list of completed gameweeks (where isResult = TRUE)
    completed_rounds = fixtures_df.loc[fixtures_df['isResult'] == True, 'round_number'].unique()
    completed_rounds = sorted(completed_rounds)  # Ensure they are sorted

    if not completed_rounds:
        raise ValueError("No completed gameweeks found in fixture_data.")

    # ✅ Determine the latest completed gameweek
    latest_round = max(completed_rounds)  

    # ✅ Adjust for previous/next navigation
    selected_round = latest_round + week_offset

    # ✅ Prevent navigation past valid rounds
    if selected_round < min(completed_rounds):
        selected_round = min(completed_rounds)
    elif selected_round > max(completed_rounds):
        selected_round = max(completed_rounds)

    # ✅ Filter fixtures for selected round
    weekly_fixtures = fixtures_df[fixtures_df['round_number'] == selected_round].to_dict(orient='records')

    return weekly_fixtures, selected_round, min(completed_rounds), max(completed_rounds)





from datetime import datetime

@app.route('/results')
def results():
    """ Renders the results page with shotmaps for the selected completed gameweek """
    week_offset = int(request.args.get('week_offset', 0))

    # ✅ Get completed fixtures and limits
    weekly_fixtures, current_week, first_gw, last_gw = get_fixtures_for_week(week_offset)

    # ✅ Convert 'date' from string to datetime
    for fixture in weekly_fixtures:
        try:
            fixture["date"] = datetime.strptime(fixture["date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            fixture["date"] = None  

    return render_template(
        "results.html",
        fixtures=weekly_fixtures,
        week_offset=current_week,
        first_gw=first_gw,
        last_gw=last_gw
    )






if __name__ == "__main__":
    app.run(debug=True)