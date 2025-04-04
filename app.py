import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for
import numpy as np
from scipy.stats import poisson
import os
import matplotlib.pyplot as plt
from data_loader import load_fixtures, load_match_data, calculate_team_statistics, load_next_gw_fixtures, get_player_data, get_player_radar_data
from collections import defaultdict
from datetime import datetime
from generate_radars import generate_comparison_radar_chart, columns_to_plot
import subprocess


# Flask app initialization
app = Flask(__name__)

def get_last_updated_time():
    try:
        raw_date = subprocess.check_output(['git', 'log', '-1', '--format=%cd'], encoding='utf-8').strip()
        return datetime.strptime(raw_date, '%a %b %d %H:%M:%S %Y %z').strftime('%d %b %Y at %H:%M')
    except Exception:
        return "Unknown"


def get_team_form(fixtures_df, team_name, max_matches=5):
    team_matches = fixtures_df[
        ((fixtures_df["home_team"] == team_name) | (fixtures_df["away_team"] == team_name)) &
        (fixtures_df["isResult"] == True)
    ].sort_values("date", ascending=False).head(max_matches)

    form = []
    for _, match in team_matches.iterrows():
        is_home = match["home_team"] == team_name
        team_goals = match["home_goals"] if is_home else match["away_goals"]
        opp_goals = match["away_goals"] if is_home else match["home_goals"]

        if team_goals > opp_goals:
            form.append("w")
        elif team_goals == opp_goals:
            form.append("d")
        else:
            form.append("l")
    return form





@app.route('/')
def index():
    return render_template("index.html")



# Load data
match_data = load_match_data()
team_stats, home_field_advantage = calculate_team_statistics(match_data)
fixtures = load_fixtures().to_dict(orient="records")  # Convert DataFrame to a list of dictionaries


@app.route("/epl_fixtures")
def fixtures_redirect():
    fixture_path = "data/tables/fixture_data.csv"
    fixtures = pd.read_csv(fixture_path)
    fixtures["isResult"] = fixtures["isResult"].astype(str).str.lower() == "true"
    fixtures["round_number"] = pd.to_numeric(fixtures["round_number"], errors="coerce")

    #next_gw = fixtures[fixtures["isResult"] == False]["round_number"].min()
    next_gw = 31
    return redirect(url_for("epl_fixtures", gw=next_gw))



@app.route("/epl_fixtures/<int:gw>")
def epl_fixtures(gw):
    fixture_path = "data/tables/fixture_data.csv"
    fixtures = pd.read_csv(fixture_path)

    # Normalize boolean values (critical!)
    fixtures["isResult"] = fixtures["isResult"].astype(str).str.lower() == "true"
    fixtures["round_number"] = pd.to_numeric(fixtures["round_number"], errors="coerce")

    gw_fixtures = fixtures[(fixtures["round_number"] == gw) & (fixtures["isResult"] == False)].copy()

    # Extract date-only from datetime string
    gw_fixtures["match_date"] = gw_fixtures["date"].str[:10]  # e.g. '02/04/2025'

    # Load all fixtures
    all_fixtures_df = pd.read_csv("data/tables/fixture_data.csv")
    all_fixtures_df["isResult"] = all_fixtures_df["isResult"].astype(str).str.lower() == "true"
    all_fixtures_df["date"] = pd.to_datetime(all_fixtures_df["date"], dayfirst=True)

    gw_fixtures["home_form"] = None
    gw_fixtures["away_form"] = None

    # Add form to each fixture
    for idx, fixture in gw_fixtures.iterrows():
        gw_fixtures.at[idx, "home_form"] = get_team_form(all_fixtures_df, fixture["home_team"])
        gw_fixtures.at[idx, "away_form"] = get_team_form(all_fixtures_df, fixture["away_team"])


    # Group fixtures by match_date
    fixture_groups = defaultdict(list)
    for _, row in gw_fixtures.iterrows():
        fixture_groups[row["match_date"]].append(row.to_dict())

    gameweeks = sorted(fixtures[fixtures["isResult"] == False]["round_number"].dropna().unique().tolist())

    return render_template(
        "epl_fixtures.html",
        fixture_groups=dict(fixture_groups),
        current_gw=gw,
        gameweeks=gameweeks,
        last_updated=get_last_updated_time()
    )


@app.template_filter("format_date")
def format_date(value):
    try:
        dt = datetime.strptime(value, "%d/%m/%Y")
        return dt.strftime("%A %d %B %Y")
    except:
        return value








@app.route("/about")
def about():
    return render_template("about.html", last_updated=get_last_updated_time())








@app.route('/epl_table')
def epl_table():
    import pandas as pd
    
    # ✅ Load current league table
    league_table = pd.read_csv("data/tables/league_table_data.csv").to_dict(orient="records")

    # ✅ Load simulated league table
    simulated_table = pd.read_csv("data/tables/simulated_league_positions.csv")

    # ✅ Rename first column to "Team" if it is unnamed
    simulated_table.rename(columns={"Unnamed: 0": "Team"}, inplace=True)

    # ✅ Convert all column names to **strings** (so Jinja can access them)
    simulated_table.columns = simulated_table.columns.astype(str)

    # ✅ Convert back to dictionary format
    simulated_table = simulated_table.to_dict(orient="records")

    # ✅ Get number of positions (1-20)
    num_positions = len(simulated_table[0]) - 2  # Exclude "Team" and "Final xPTS"

    return render_template("epl_table.html", league_table=league_table, simulated_table=simulated_table, num_positions=num_positions, last_updated=get_last_updated_time())











@app.route('/epl-players')
def epl_players():
    try:
        # Load radar data for dropdowns
        radar_players = get_player_radar_data()

        # Filter out goalkeepers and players missing required stats for the radar comparison
        required_stats = [
            'Goals', 'Assists', 'Goals + Assists', 'Expected Goals',
            'Expected Assists', 'Progressive Carries', 'Progressive Passes', 'Progressive Receptions'
        ]
        dropdown_players = []
        for player in radar_players:
            if player.get("Pos") == "GK":
                continue
            # Only include if all required stats are present (even if they are 0)
            if all(player.get(stat) not in [None, ""] for stat in required_stats):
                dropdown_players.append(player)
                
        # Load player data for the main table (if you still want to use the original list)
        players = get_player_data()  # or load radar_players if you want the same list for both

        next_gw_fixtures = load_next_gw_fixtures()
        current_gw = next_gw_fixtures[0]["round_number"] if next_gw_fixtures else "Unknown"

        # Extract unique positions and teams (you can derive these from the full list)
        unique_positions = sorted(set(p.get("Pos", "Unknown") for p in players))
        unique_teams = sorted(set(p.get("Team", "Unknown") for p in players))

        return render_template('epl_player.html', 
                               players=players,               # For the main table
                               dropdown_players=dropdown_players,  # For the radar dropdowns
                               positions=unique_positions, 
                               teams=unique_teams, 
                               current_gw=current_gw,
                               last_updated=get_last_updated_time())
    except Exception as e:
        print(f"❌ Error loading player data: {e}")
        return render_template('epl_player.html', players=[], dropdown_players=[], positions=[], teams=[], current_gw="Unknown", last_updated=get_last_updated_time())



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
fixtures_df = load_fixtures()


# this lets us determine the current gameweek along with all completed gameweeks
def get_fixtures_for_week(week_offset=0):
    """ Returns fixtures for the selected completed round. """
    
    # Load fixture data
    fixture_file_path = "data/tables/fixture_data.csv"
    fixtures_df = pd.read_csv(fixture_file_path)

    # Ensure 'round_number' is numeric
    fixtures_df['round_number'] = pd.to_numeric(fixtures_df['round_number'], errors='coerce')

    # Count occurrences of isResult == True per round
    result_counts = fixtures_df[fixtures_df['isResult'] == True].groupby('round_number').size()

    # Get list of completed rounds (where at least 3 results exist)
    completed_rounds = result_counts[result_counts >= 3].index.tolist()
    completed_rounds.sort()  # Ensure they are sorted

    if not completed_rounds:
        raise ValueError("No completed gameweeks with at least 10 results found in fixture_data.")

    # Determine the latest completed gameweek
    latest_round = max(completed_rounds)

    # Adjust for previous/next navigation
    selected_round = latest_round + week_offset

    # Prevent navigation past valid rounds
    selected_round = max(min(completed_rounds), min(max(completed_rounds), selected_round))

    # Filter fixtures for selected round
    weekly_fixtures = fixtures_df[fixtures_df['round_number'] == selected_round].to_dict(orient='records')

    return weekly_fixtures, selected_round, min(completed_rounds), max(completed_rounds)







@app.route('/epl_results')
def results_redirect():
    """ Redirect to latest completed GW """
    _, current_week, *_ = get_fixtures_for_week(0)
    return redirect(url_for("epl_results", gw=current_week))


@app.route('/epl_results/<int:gw>')
def epl_results(gw):
    """ Renders the results page for a specific gameweek """
    try:
        # ✅ Get all valid completed gameweeks
        weekly_fixtures, _, first_gw, last_gw = get_fixtures_for_week(0)
        all_gws = list(range(first_gw, last_gw + 1))

        # ✅ Force the selected week to stay within valid bounds
        gw = max(first_gw, min(last_gw, gw))

        # ✅ Get fixtures for selected week
        weekly_fixtures, _, _, _ = get_fixtures_for_week(gw - last_gw)  # This reuses offset logic

        # ✅ Filter for fixtures with existing shotmaps
        shotmap_dir = os.path.join("static", "shotmaps")
        filtered_fixtures = []
        for fixture in weekly_fixtures:
            shotmap_filename = f"{fixture['home_team']}_{fixture['away_team']}_shotmap.png"
            shotmap_path = os.path.join(shotmap_dir, shotmap_filename)
            if os.path.exists(shotmap_path):
                filtered_fixtures.append(fixture)

        # ✅ Load all fixture results to calculate form
        all_results_df = pd.read_csv("data/tables/fixture_data.csv")
        all_results_df["isResult"] = all_results_df["isResult"].astype(str).str.lower() == "true"
        all_results_df["date"] = pd.to_datetime(all_results_df["date"], dayfirst=True)

        # ✅ Add form data to each fixture in this GW
        for fixture in filtered_fixtures:
            fixture["home_form"] = get_team_form(all_results_df, fixture["home_team"])
            fixture["away_form"] = get_team_form(all_results_df, fixture["away_team"])

        # ✅ Group fixtures by date (INSERT HERE)
        fixture_groups = defaultdict(list)
        for fixture in filtered_fixtures:
            date_str = fixture['date'][:10] if fixture['date'] else 'Unknown'
            fixture_groups[date_str].append(fixture)

        # ✅ Load League Table
        league_table_path = "data/tables/league_table_data.csv"
        league_table = pd.read_csv(league_table_path).to_dict(orient="records") if os.path.exists(league_table_path) else []

        return render_template(
            "epl_results.html",
            fixture_groups=dict(fixture_groups),
            current_gw=gw,
            gameweeks=all_gws,
            league_table=league_table,
            last_updated=get_last_updated_time()
        )



    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return render_template("epl_results.html", fixtures=[], current_gw=0, gameweeks=[], league_table=[], last_updated=get_last_updated_time())




    

@app.route('/generate_radar')
def generate_radar():
    player1 = request.args.get('player1')
    player2 = request.args.get('player2')

    if not player1 or not player2:
        return "Invalid player selection", 400

    # Load player data
    df = pd.read_csv("data/tables/player_radar_data.csv")

    if player1 not in df['Player'].values or player2 not in df['Player'].values:
        return "Player not found", 404

    # Get player stats
    player1_stats = df[df['Player'] == player1][columns_to_plot].values.flatten().tolist()
    player2_stats = df[df['Player'] == player2][columns_to_plot].values.flatten().tolist()

    # Generate radar chart comparing both players
    fig, ax = generate_comparison_radar_chart(player1, player2, player1_stats, player2_stats)

    # Save the radar image
    radar_image_path = f"static/radar/{player1}_vs_{player2}.png"
    fig.savefig(radar_image_path, dpi=300)
    plt.close(fig)

    return send_file(radar_image_path, mimetype='image/png')




@app.route("/methodology")
def methodology():
    return render_template("methodology.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if port is None:
        raise RuntimeError("PORT environment variable is not set.")
    app.run(host="0.0.0.0", port=port, debug=True)