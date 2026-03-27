import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import numpy as np
from scipy.stats import poisson
import os
import matplotlib.pyplot as plt
from data_loader import load_fixtures, load_match_data, calculate_team_statistics, load_next_gw_fixtures, get_player_data, get_player_radar_data, predict_player_goals, TEAM_NAME_MAPPING
from data_loader import calculate_recent_form, get_team_xg
from collections import defaultdict
from datetime import datetime
from generate_radars import generate_comparison_radar_chart, columns_to_plot
import subprocess
import json
import unicodedata
import requests
from bs4 import BeautifulSoup
from mplsoccer import Radar
from io import BytesIO
from generate_player_shots import create_player_shotmap_image

# Flask app initialization
app = Flask(__name__)


# ── Manual GW override — set to an int to force a specific GW, None for auto ──
GW_OVERRIDE = 32



@app.route('/sitemap.xml')
def sitemap():
    return send_file('static/sitemap.xml', mimetype='application/xml')




TEAM_NAME_MAPPING = {
    "Wolverhampton Wanderers": "Wolves",
    "Crystal Palace": "Crystal Palace",
    "Tottenham": "Spurs",
    "Man Utd": "Manchester United",
    "Newcastle United": "Newcastle",
    "Spurs": "Tottenham",
    "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
}



    
def get_last_updated_time():
    try:
        # Try git first
        raw_date = subprocess.check_output(['git', 'log', '-1', '--format=%cd'], encoding='utf-8').strip()
        return datetime.strptime(raw_date, '%a %b %d %H:%M:%S %Y %z').strftime('%d %b %Y at %H:%M')
    except Exception:
        # Fallback to file mod time
        try:
            file_path = "data/tables/fixture_data.csv"
            mtime = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mtime).strftime('%d %b %Y at %H:%M')
        except Exception:
            return "Unknown"


BOOKIE_TEAM_NAME_MAP = {
    "Man Utd":         "Manchester United",
    "Man City":        "Manchester City",
    "Spurs":           "Tottenham Hotspur",
    "Wolves":          "Wolverhampton Wanderers",
    "Nott'm Forest":   "Nottingham Forest",
    "Newcastle":       "Newcastle United",
    "Brighton":        "Brighton",
    "Leeds":           "Leeds",
    "Sunderland":      "Sunderland",
    "Bournemouth":     "Bournemouth",
    "Brentford":       "Brentford",
    "Burnley":         "Burnley",
    "Chelsea":         "Chelsea",
    "Crystal Palace":  "Crystal Palace",
    "Everton":         "Everton",
    "Fulham":          "Fulham",
    "Liverpool":       "Liverpool",
    "Arsenal":         "Arsenal",
    "Aston Villa":     "Aston Villa",
    "West Ham":        "West Ham",
}

TEAM_SHORT_NAMES = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Ipswich": "IPS", "Leicester": "LEI", "Liverpool": "LIV",
    "Manchester City": "MCI", "Manchester United": "MUN",
    "Newcastle United": "NEW", "Nottingham Forest": "NFO",
    "Southampton": "SOU", "Tottenham Hotspur": "TOT",
    "West Ham": "WHU", "Wolverhampton Wanderers": "WOL",
    "Sunderland": "SUN", "Leeds": "LEE", "Burnley": "BUR",
}

def scrape_bookie_win_probs(gw):
    """Scrape win probabilities from checkthechance.com for a given gameweek."""
    try:
        url = f"https://checkthechance.com/premier-league-round-{gw}/"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3","h4","h5","h6"])]

        results = []
        i = 0
        while i < len(headings):
            h = headings[i]
            if "%" in h and "|" in h:
                # Home team line: "Bournemouth | 31.0 %"
                parts = h.split("|")
                home_team_raw = parts[0].strip()
                home_prob = float(parts[1].replace("%","").strip())
                # Next heading is draw %
                if i + 1 < len(headings) and "%" in headings[i+1] and "|" not in headings[i+1]:
                    draw_prob = float(headings[i+1].replace("%","").strip())
                    # Next is away team
                    if i + 2 < len(headings) and "|" in headings[i+2]:
                        away_parts = headings[i+2].split("|")
                        away_team_raw = away_parts[0].strip()
                        away_prob = float(away_parts[1].replace("%","").strip())
                        home_team = BOOKIE_TEAM_NAME_MAP.get(home_team_raw, home_team_raw)
                        away_team = BOOKIE_TEAM_NAME_MAP.get(away_team_raw, away_team_raw)
                        results.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookie_home_win": home_prob,
                            "bookie_draw":     draw_prob,
                            "bookie_away_win": away_prob,
                        })
                        i += 3
                        continue
            i += 1
        return results
    except Exception as e:
        print(f"⚠️ Failed to scrape win probs: {e}")
        return []


def scrape_bookie_cs_probs(gw):
    """Scrape clean sheet probabilities from checkthechance.com for a given gameweek."""
    try:
        url = f"https://checkthechance.com/fpl-clean-sheet-gameweek-{gw}/"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3","h4","h5","h6"])]

        results = {}
        for h in headings:
            if "%" in h and "|" in h:
                parts = h.split("|")
                team_raw = parts[0].strip()
                prob = float(parts[1].replace("%","").strip())
                team = BOOKIE_TEAM_NAME_MAP.get(team_raw, team_raw)
                results[team] = prob
        return results
    except Exception as e:
        print(f"⚠️ Failed to scrape CS probs: {e}")
        return {}


def get_team_form(fixtures_df, team_name, max_matches=10):
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
            result = "w"
        elif team_goals == opp_goals:
            result = "d"
        else:
            result = "l"

        form.append({
            "result": result,
            "h_team": match["home_team"],
            "a_team": match["away_team"],
            "h_score": int(match["home_goals"]),
            "a_score": int(match["away_goals"])
        })
    return form




# league position visuals
@app.template_filter('get_position_tooltip')
def get_position_tooltip(pos):
    if 1 <= pos <= 4:
        return "Champions League"
    elif 5 <= pos <= 6:
        return "Europa League"
    elif pos == 7:
        return "Conference League"
    elif pos >= 18:
        return "Relegation"
    return ""





@app.route('/')
def index():
    accuracy = None
    accuracy_path = "data/tables/model_accuracy.json"
    if os.path.exists(accuracy_path):
        with open(accuracy_path, "r") as f:
            accuracy = json.load(f)

    gw_fixtures = []
    current_gw = None
    try:
        fixtures_df = pd.read_csv("data/tables/fixture_data.csv")
        fixtures_df["isResult"] = fixtures_df["isResult"].astype(str).str.lower() == "true"
        fixtures_df["round_number"] = pd.to_numeric(fixtures_df["round_number"], errors="coerce")

        if GW_OVERRIDE is not None:
            current_gw = GW_OVERRIDE
        else:
            upcoming = fixtures_df[fixtures_df["isResult"] == False]["round_number"]
            current_gw = int(upcoming.min()) if not upcoming.empty else None

        if current_gw:
            probs_lookup = {}
            if os.path.exists("data/tables/fixture_probabilities.csv"):
                for _, row in pd.read_csv("data/tables/fixture_probabilities.csv").iterrows():
                    probs_lookup[f"{row['home_team']}|{row['away_team']}"] = {
                        "home_win": int(round(row["home_win_prob"] * 100)),
                        "draw":     int(round(row["draw_prob"]     * 100)),
                        "away_win": int(round(row["away_win_prob"] * 100)),
                    }

            gw_df = fixtures_df[
                (fixtures_df["round_number"] == current_gw) &
                (fixtures_df["isResult"] == False)
            ]
            for _, row in gw_df.iterrows():
                key = f"{row['home_team']}|{row['away_team']}"
                if key not in probs_lookup:
                    continue
                entry = {"home_team": row["home_team"], "away_team": row["away_team"]}
                entry.update(probs_lookup[key])
                gw_fixtures.append(entry)
    except Exception as e:
        print(f"⚠️ Index fixture load failed: {e}")

    return render_template("index.html",
        accuracy=accuracy,
        gw_fixtures=gw_fixtures,
        current_gw=current_gw
    )



# Load data
match_data = load_match_data()
team_stats, home_field_advantage = calculate_team_statistics(match_data, save_csv_path=None)
fixtures = load_fixtures().to_dict(orient="records")  # Convert DataFrame to a list of dictionaries


@app.route("/epl_fixtures")
def fixtures_redirect():
    if GW_OVERRIDE is not None:
        return redirect(url_for("epl_fixtures", gw=GW_OVERRIDE))

    fixture_path = "data/tables/fixture_data.csv"
    fixtures = pd.read_csv(fixture_path)
    fixtures["isResult"] = fixtures["isResult"].astype(str).str.lower() == "true"
    fixtures["round_number"] = pd.to_numeric(fixtures["round_number"], errors="coerce")

    upcoming_rounds = fixtures[fixtures["isResult"] == False]["round_number"]
    next_gw = upcoming_rounds.min() if not upcoming_rounds.empty else 38

    return redirect(url_for("epl_fixtures", gw=int(next_gw)))




@app.route("/epl_fixtures/<int:gw>")
def epl_fixtures(gw):
    fixture_path = "data/tables/fixture_data.csv"
    fixtures = pd.read_csv(fixture_path)

    # Normalize boolean values (critical!)
    fixtures["isResult"] = fixtures["isResult"].astype(str).str.lower() == "true"
    fixtures["round_number"] = pd.to_numeric(fixtures["round_number"], errors="coerce")

    gw_fixtures = fixtures[fixtures["round_number"] == gw].copy()

    gw_fixtures["match_date"] = gw_fixtures["date"].str[:10]

    # Load all fixture results to calculate form
    all_fixtures_df = pd.read_csv(fixture_path)
    all_fixtures_df["isResult"] = all_fixtures_df["isResult"].astype(str).str.lower() == "true"
    all_fixtures_df["date"] = pd.to_datetime(all_fixtures_df["date"], dayfirst=True)

    gw_fixtures["home_form"] = None
    gw_fixtures["away_form"] = None

    for idx, row in gw_fixtures.iterrows():
        gw_fixtures.at[idx, "home_form"] = get_team_form(all_fixtures_df, row["home_team"])
        gw_fixtures.at[idx, "away_form"] = get_team_form(all_fixtures_df, row["away_team"])

        if row["isResult"]:
            gw_fixtures.at[idx, "asset_type"] = "shotmap"
            gw_fixtures.at[idx, "asset_path"] = f"shotmaps/{row['home_team']}_{row['away_team']}_shotmap.png"
        else:
            gw_fixtures.at[idx, "asset_type"] = "heatmap"
            gw_fixtures.at[idx, "asset_path"] = f"heatmaps/{row['home_team']}_{row['away_team']}_heatmap.png"

    # Group fixtures by match_date
    fixture_groups = defaultdict(list)
    for _, row in gw_fixtures.iterrows():
        fixture_groups[row["match_date"]].append(row.to_dict())

    gameweeks = sorted(fixtures["round_number"].dropna().unique().tolist())

    with open("data/team_metadata.json", "r") as f:
        team_metadata = json.load(f)

    all_teams = list(team_metadata.keys())
    team_display_names = {t: TEAM_NAME_MAPPING.get(t, t) for t in all_teams}

    league_table_path = "data/tables/league_table_data.csv"
    league_table = pd.read_csv(league_table_path).to_dict(orient="records") if os.path.exists(league_table_path) else []

    # Build result stats (Shots, SOT) for resulted fixtures from shots_data
    result_stats = {}
    shots_path = "data/tables/shots_data.csv"
    if os.path.exists(shots_path):
        shots_df = pd.read_csv(shots_path)
        goal_kw = ['Goal', 'OwnGoal']
        sot_kw  = ['Goal', 'SavedShot']
        for _, fix in gw_fixtures[gw_fixtures["isResult"]].iterrows():
            mid = str(fix["id"])
            match_shots = shots_df[shots_df["match_id"].astype(str) == mid]
            home_s = match_shots[match_shots["h_a"] == "h"]
            away_s = match_shots[match_shots["h_a"] == "a"]
            key = f"{fix['home_team']}|{fix['away_team']}"
            result_stats[key] = {
                "home_goals": int(fix["home_goals"]),
                "away_goals": int(fix["away_goals"]),
                "home_xg":   round(pd.to_numeric(home_s["xG"], errors="coerce").sum(), 2),
                "away_xg":   round(pd.to_numeric(away_s["xG"], errors="coerce").sum(), 2),
                "home_shots": len(home_s),
                "away_shots": len(away_s),
                "home_sot":  int(home_s["result"].isin(sot_kw).sum()),
                "away_sot":  int(away_s["result"].isin(sot_kw).sum()),
            }

    # Load fixture probabilities and build a lookup dict keyed by "home_team|away_team"
    fixture_stats = {}
    probs_path = "data/tables/fixture_probabilities.csv"
    if os.path.exists(probs_path):
        probs_df = pd.read_csv(probs_path)
        for _, row in probs_df.iterrows():
            key = f"{row['home_team']}|{row['away_team']}"
            fixture_stats[key] = {
                "home_win":  round(row["home_win_prob"] * 100, 1),
                "draw":      round(row["draw_prob"] * 100, 1),
                "away_win":  round(row["away_win_prob"] * 100, 1),
                "over_2_5":  round(row.get("over_2_5_prob", 0) * 100, 1),
                "home_cs":   round(row.get("home_cs_prob", 0) * 100, 1),
                "away_cs":   round(row.get("away_cs_prob", 0) * 100, 1),
                "home_xg":   round(float(row.get("home_xg", 0) or 0), 2),
                "away_xg":   round(float(row.get("away_xg", 0) or 0), 2),
            }

    # Load simulated table data
    sim_table_path = "data/tables/simulated_league_positions.csv"
    simulated_table = pd.read_csv(sim_table_path)
    simulated_table.rename(columns={"Unnamed: 0": "Team"}, inplace=True)
    simulated_table.columns = simulated_table.columns.astype(str)
    simulated_table = simulated_table.to_dict(orient="records")
    num_positions = len(simulated_table[0]) - 2  # Assuming "Team" + "Final xPTS" are extra columns


    # Construct team-by-team position probability data
    sim_position_dist = []
    for team in simulated_table:
        sim_position_dist.append({
            "team": team["Team"],
            "distribution": [
                {"position": int(pos), "probability": round(team[str(pos)] * 100, 2)}
                for pos in range(1, num_positions + 1)
            ]
        })

    return render_template(
        "epl_fixtures.html",
        fixture_groups=dict(fixture_groups),
        current_gw=gw,
        gameweeks=gameweeks,
        last_updated=get_last_updated_time(),
        all_teams=all_teams,
        team_display_names=team_display_names,
        league_table=league_table,
        simulated_table=simulated_table,
        sim_position_dist=sim_position_dist,
        num_positions=num_positions,
        fixture_stats=fixture_stats,
        result_stats=result_stats
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
    xg_table = sorted(league_table, key=lambda x: float(x.get("xPTS", 0)), reverse=True)

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

    return render_template("epl_table.html", league_table=league_table, xg_table=xg_table, simulated_table=simulated_table, num_positions=num_positions, last_updated=get_last_updated_time())




@app.route("/privacy.html")
def privacy():
    return render_template("privacy.html")


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

    # Get list of completed rounds (If there are fixtures from older rounds increase the value e.g. where at least 3 results exist)
    completed_rounds = result_counts[result_counts >= 1].index.tolist()
    completed_rounds.sort()  # Ensure they are sorted

    if not completed_rounds:
        return [], 0, 0, 0  # 🧯 Fallback when no results yet

    # Determine the latest completed gameweek
    latest_round = max(completed_rounds)

    # Adjust for previous/next navigation
    selected_round = latest_round + week_offset

    # Prevent navigation past valid rounds
    selected_round = max(min(completed_rounds), min(max(completed_rounds), selected_round))

    # Filter fixtures for selected round
    weekly_fixtures = fixtures_df[fixtures_df['round_number'] == selected_round].to_dict(orient='records')

    return weekly_fixtures, selected_round, min(completed_rounds), max(completed_rounds)










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


@app.route('/generate_single_radar')
def generate_single_radar():
    player = request.args.get('player')

    if not player:
        return "Player not specified", 400

    # Load player data
    df = pd.read_csv("data/tables/player_radar_data.csv")
    
    # Check if player exists
    if player not in df['Player'].values:
        return "Player not found", 404

    # Get player stats
    columns = [
        'Goals', 'Assists', 'Goals + Assists', 'Expected Goals',
        'Expected Assists', 'Progressive Carries', 'Progressive Passes', 'Progressive Receptions'
    ]
    player_stats = df[df['Player'] == player][columns].values.flatten().tolist()
    average_stats = df[columns].mean().values.flatten().tolist()

    # Create radar
    from mplsoccer import Radar
    radar = Radar(params=columns, min_range=[0]*len(columns), max_range=[100]*len(columns))

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

    ax.text(0.2, 1.02, player, fontsize=15, ha='center', transform=ax.transAxes, color='#669bbc')
    ax.text(0.8, 1.02, 'League Avg', fontsize=15, ha='center', transform=ax.transAxes, color='#e63946')

    ax.text(
        x=0, y=0.05, 
        s='Metrics show per 90 percentile\nstats compared againt all players\nin The Premier League\n\n@Five_Stat', 
        fontsize=11, ha='left', va='center', transform=ax.transAxes, fontfamily='monospace'
    )

    img_io = BytesIO()
    plt.savefig(img_io, format='png', facecolor=fig.get_facecolor(), dpi=300)
    img_io.seek(0)
    plt.close(fig)

    return send_file(img_io, mimetype='image/png')





@app.route("/methodology")
def methodology():
    accuracy = None
    accuracy_path = "data/tables/model_accuracy.json"
    if os.path.exists(accuracy_path):
        with open(accuracy_path, "r") as f:
            accuracy = json.load(f)
    return render_template("methodology.html", accuracy=accuracy)






@app.route("/team/<team_name>")
def team_page(team_name):
    import pandas as pd
    import json
    from collections import defaultdict

    # Load team metadata
    with open("data/team_metadata.json", "r") as f:
        team_metadata = json.load(f)

    all_teams = list(team_metadata.keys())
    team_display_names = {t: TEAM_NAME_MAPPING.get(t, t) for t in all_teams}
    team_data = team_metadata.get(team_name)
    if not team_data:
        return f"Team '{team_name}' not found.", 404

    team_data["name"] = team_name
    team_data["display_name"] = TEAM_NAME_MAPPING.get(team_name, team_name)

    # === Load fixture data ===
    fixture_df = pd.read_csv("data/tables/fixture_data.csv")
    fixture_df["isResult"] = fixture_df["isResult"].astype(str).str.lower() == "true"
    fixture_df["date"] = pd.to_datetime(fixture_df["date"], dayfirst=True)
    fixture_df["round_number"] = pd.to_numeric(fixture_df["round_number"], errors="coerce")

    # === Previous Fixture Shotmap ===
    past_games = fixture_df[
        ((fixture_df["home_team"] == team_name) | (fixture_df["away_team"] == team_name)) &
        (fixture_df["isResult"] == True)
    ].sort_values("date", ascending=False)

    if not past_games.empty:
        prev_game = past_games.iloc[0]
        prev_gw = int(prev_game["round_number"])
        last_opp = prev_game["away_team"] if prev_game["home_team"] == team_name else prev_game["home_team"]
        prev_fixture_image = f"{prev_game['home_team']}_{prev_game['away_team']}_shotmap.png"
    else:
        prev_gw = 1
        last_opp = "Unknown"
        prev_fixture_image = "placeholder.png"

    # === Next Fixture Heatmap ===
    upcoming_games = fixture_df[
        ((fixture_df["home_team"] == team_name) | (fixture_df["away_team"] == team_name)) &
        (fixture_df["isResult"] == False)
    ].sort_values("date", ascending=True)

    if not upcoming_games.empty:
        next_game = upcoming_games.iloc[0]
        next_gw = int(next_game["round_number"])
        next_opp = next_game["away_team"] if next_game["home_team"] == team_name else next_game["home_team"]
        next_fixture_image = f"{next_game['home_team']}_{next_game['away_team']}_heatmap.png"
    else:
        next_gw = 1
        next_opp = "Unknown"
        next_fixture_image = "placeholder.png"

    # === Create GW dropdown options ===
    # === Previous gameweek mapping {gw: [home_team, away_team]} ===
    past_fixtures_by_gw = {}
    for _, row in past_games.iterrows():
        gw = int(row["round_number"])
        past_fixtures_by_gw[gw] = {
            "home": row["home_team"],
            "away": row["away_team"]
        }

    upcoming_gameweeks = sorted(upcoming_games["round_number"].dropna().astype(int).unique())

    # Create JS dict to resolve opponents for next fixture heatmaps
    next_opponents_by_gw = {}
    for _, row in upcoming_games.iterrows():
        gw = int(row["round_number"])
        opp = row["away_team"] if row["home_team"] == team_name else row["home_team"]
        next_opponents_by_gw[gw] = opp

    # === League table window for context ===
    table_path = "data/tables/league_table_data.csv"
    league_df = pd.read_csv(table_path)
    league_df_reset = league_df.reset_index(drop=True)
    team_row = league_df_reset[league_df_reset["Team"] == team_name]
    if not team_row.empty:
        position = team_row.index[0]
        start_position = max(0, position - 3)
        end_position = position + 4
        partial_table = league_df_reset.iloc[start_position:end_position].to_dict(orient="records")
        team_stats_row = team_row.iloc[0].to_dict()
        team_position  = int(position) + 1
    else:
        start_position = 0
        partial_table  = []
        team_stats_row = {}
        team_position  = None

    # Form
    team_data["form"] = get_team_form(fixture_df, team_name)

    # Load simulated league table data
    simulated_path = "data/tables/simulated_league_positions.csv"
    simulated_df = pd.read_csv(simulated_path)
    simulated_df.rename(columns={"Unnamed: 0": "Team"}, inplace=True)

    # Ensure consistent team names
    simulated_df["Team"] = simulated_df["Team"].apply(str)
    sim_team_row = simulated_df[simulated_df["Team"] == team_name]

    if not sim_team_row.empty:
        sim_index = sim_team_row.index[0]
        sim_start = max(0, sim_index - 3)
        sim_end = sim_index + 4
        simulated_partial = simulated_df.iloc[sim_start:sim_end].to_dict(orient="records")
        num_sim_positions = len([col for col in simulated_df.columns if col.isnumeric()])
    else:
        simulated_partial = []
        num_sim_positions = 20  # default fallback

    # Extract full simulation probabilities for the team
    team_sim_row = simulated_df[simulated_df["Team"] == team_name]
    sim_position_dist = []
    if not team_sim_row.empty:
        sim_position_dist = [
            {
                "position": int(pos),
                "probability": round(team_sim_row.iloc[0][pos] * 100, 2)
            }
            for pos in map(str, range(1, 21))
            if pos in team_sim_row.columns
        ]

    # === Filter Player Data for This Team ===
    from data_loader import get_player_data
    players_all = get_player_data()
    team_players = [p for p in players_all if p.get("Team") == team_name]

    return render_template(
        "team_page.html",
        team=team_data,
        all_teams=all_teams,
        team_display_names=team_display_names,
        league_table=partial_table,
        start_position=start_position,
        team_players=team_players,
        last_updated=get_last_updated_time(),
        team_stats_row=team_stats_row,
        team_position=team_position,

        # Fixture visual data
        prev_fixture_image=prev_fixture_image,
        next_fixture_image=next_fixture_image,
        current_result_gw=prev_gw,
        current_fixture_gw=next_gw,
        past_fixtures_by_gw=past_fixtures_by_gw,
        previous_gameweeks=sorted(past_fixtures_by_gw.keys()),  # ✅ ADD THIS
        upcoming_gameweeks=upcoming_gameweeks,
        next_opponents_by_gw=next_opponents_by_gw,
        sim_start=sim_start,
        simulated_table=simulated_partial,
        num_sim_positions=num_sim_positions,
        sim_position_dist=sim_position_dist
    )



@app.route("/teams")
def teams_landing():
    with open("data/team_metadata.json", "r") as f:
        team_metadata = json.load(f)
    all_teams = list(team_metadata.keys())
    team_display_names = {t: TEAM_NAME_MAPPING.get(t, t) for t in all_teams}
    return render_template("teams.html",
        all_teams=all_teams,
        team_display_names=team_display_names,
        last_updated=get_last_updated_time()
    )


@app.route("/fpl")
def fpl():
    try:
        fixtures_df = pd.read_csv("data/tables/fixture_data.csv")
        fixtures_df["isResult"] = fixtures_df["isResult"].astype(str).str.lower() == "true"
        fixtures_df["round_number"] = pd.to_numeric(fixtures_df["round_number"], errors="coerce")

        if GW_OVERRIDE is not None:
            current_gw = GW_OVERRIDE
        else:
            upcoming = fixtures_df[fixtures_df["isResult"] == False]["round_number"]
            current_gw = int(upcoming.min()) if not upcoming.empty else 1

        probs_df   = pd.read_csv("data/tables/fixture_probabilities.csv")
        league_df  = pd.read_csv("data/tables/league_table_data.csv")
        teams      = league_df["Team"].tolist()
        team_rank  = {row["Team"]: idx + 1 for idx, row in league_df.iterrows()}

        upcoming_df = fixtures_df[
            (fixtures_df["isResult"] == False) &
            (fixtures_df["round_number"] >= current_gw) &
            (fixtures_df["round_number"] <  current_gw + 5)
        ].merge(probs_df, on=["home_team", "away_team"], how="left")

        cs_data = []
        for team in teams:
            home = upcoming_df[upcoming_df["home_team"] == team][["round_number", "home_cs_prob"]].rename(columns={"home_cs_prob": "cs_prob"})
            away = upcoming_df[upcoming_df["away_team"] == team][["round_number", "away_cs_prob"]].rename(columns={"away_cs_prob": "cs_prob"})
            tg   = pd.concat([home, away]).dropna(subset=["cs_prob"])

            gw1 = tg[tg["round_number"] == current_gw]["cs_prob"].sum()
            gw3 = tg[tg["round_number"] <  current_gw + 3]["cs_prob"].sum()
            gw5 = tg["cs_prob"].sum()

            cs_data.append({
                "team": team,
                "gw1": round(float(gw1) * 100, 1),
                "gw3": round(float(gw3), 3),
                "gw5": round(float(gw5), 3),
            })

        cs_data.sort(key=lambda x: x["gw1"], reverse=True)

        xg_data = []
        for team in teams:
            home = upcoming_df[upcoming_df["home_team"] == team][["round_number", "home_xg"]].rename(columns={"home_xg": "xg"})
            away = upcoming_df[upcoming_df["away_team"] == team][["round_number", "away_xg"]].rename(columns={"away_xg": "xg"})
            tg   = pd.concat([home, away]).dropna(subset=["xg"])

            xg1 = tg[tg["round_number"] == current_gw]["xg"].sum()
            xg3 = tg[tg["round_number"] <  current_gw + 3]["xg"].sum()
            xg5 = tg["xg"].sum()

            xg_data.append({
                "team": team,
                "gw1": round(float(xg1), 2),
                "gw3": round(float(xg3), 2),
                "gw5": round(float(xg5), 2),
            })

        xg_data.sort(key=lambda x: x["gw1"], reverse=True)

        xga_data = []
        for team in teams:
            # xGA = goals opponent is expected to score, so home team's xGA = away_xg
            home = upcoming_df[upcoming_df["home_team"] == team][["round_number", "away_xg"]].rename(columns={"away_xg": "xga"})
            away = upcoming_df[upcoming_df["away_team"] == team][["round_number", "home_xg"]].rename(columns={"home_xg": "xga"})
            tg   = pd.concat([home, away]).dropna(subset=["xga"])

            xga1 = tg[tg["round_number"] == current_gw]["xga"].sum()
            xga3 = tg[tg["round_number"] <  current_gw + 3]["xga"].sum()
            xga5 = tg["xga"].sum()

            xga_data.append({
                "team": team,
                "gw1": round(float(xga1), 2),
                "gw3": round(float(xga3), 2),
                "gw5": round(float(xga5), 2),
            })

        xga_data.sort(key=lambda x: x["gw1"])

        # ── Captain picks — multi-GW projected xG ─────────────────────────
        captain_picks = []
        fpl_path    = "data/tables/fpl_player_data.csv"
        player_path = "data/tables/player_data.csv"
        shots_path  = "data/tables/shots_data.csv"

        if os.path.exists(fpl_path) and os.path.exists(shots_path):
            try:
                import unicodedata as _ud
                def _norm(s):
                    return _ud.normalize("NFKD", str(s)).encode("ascii","ignore").decode().lower().strip()

                fpl_df   = pd.read_csv(fpl_path)
                shots_df = pd.read_csv(shots_path)
                fpl_df["team"] = fpl_df["team"].replace(BOOKIE_TEAM_NAME_MAP)

                # Upcoming fixtures for next 5 GWs with model xG
                upcoming5_df = fixtures_df[
                    (fixtures_df["isResult"] == False) &
                    (fixtures_df["round_number"] >= current_gw) &
                    (fixtures_df["round_number"] <  current_gw + 5)
                ].merge(probs_df, on=["home_team","away_team"], how="left")

                # Pre-build team fixture map: team -> [{gw, opp, is_home, fix_xg, fix_diff}]
                team_fix_map = {}
                for _, fix in upcoming5_df.iterrows():
                    ht, at = fix["home_team"], fix["away_team"]
                    gw_num = int(fix["round_number"])
                    h_xg   = float(fix.get("home_xg", 0) or 0)
                    a_xg   = float(fix.get("away_xg", 0) or 0)
                    h_wp   = float(fix.get("home_win_prob", 0.5) or 0.5) * 100
                    a_wp   = float(fix.get("away_win_prob", 0.5) or 0.5) * 100
                    h_diff = "easy" if h_wp >= 60 else "hard" if h_wp < 40 else "medium"
                    a_diff = "easy" if a_wp >= 60 else "hard" if a_wp < 40 else "medium"
                    for team, opp, is_home, fix_xg, diff in [
                        (ht, at, True,  h_xg, h_diff),
                        (at, ht, False, a_xg, a_diff),
                    ]:
                        team_fix_map.setdefault(team, []).append({
                            "gw":      gw_num,
                            "opp":     TEAM_SHORT_NAMES.get(opp, opp[:3].upper()),
                            "is_home": is_home,
                            "fix_xg":  round(fix_xg, 2),
                            "diff":    diff,
                        })

                # ── Build team totals from FPL (complete roster, no missing players) ──
                if "xg" not in fpl_df.columns or "xa" not in fpl_df.columns:
                    print("⚠️  fpl_player_data.csv missing xg/xa columns — re-run data_scraper_script.py --bookie-only")
                    raise ValueError("fpl_player_data.csv outdated — missing xg/xa columns")
                team_xg_totals = fpl_df.groupby("team")["xg"].sum().to_dict()
                team_xa_totals = fpl_df.groupby("team")["xa"].sum().to_dict()

                # ── Pre-compute shots lookups once — never re-scan inside the player loop ──
                has_match_id = "match_id" in shots_df.columns

                # Normalise shot player names once up front
                if has_match_id:
                    shots_df["player_norm"] = shots_df["player"].apply(_norm)
                    shots_df["match_id_str"] = shots_df["match_id"].astype(str)

                # Recent 5 GW match IDs per team
                recent5_ids = {}
                if has_match_id:
                    for team in teams:
                        r5 = fixtures_df[
                            ((fixtures_df["home_team"] == team) | (fixtures_df["away_team"] == team)) &
                            (fixtures_df["isResult"] == True)
                        ].sort_values("round_number", ascending=False).head(5)
                        if "id" in r5.columns:
                            recent5_ids[team] = set(r5["id"].astype(str).dropna())

                # Pre-group shots by team and by (player_norm, match_id_str) for O(1) lookup
                # team_recent_xg[team] = total xG in recent 5 GWs for that team
                # player_recent_xg[(player_norm, team)] = player xG in recent 5 GWs
                team_recent_xg    = {}
                player_recent_xg  = {}
                if has_match_id:
                    for team, ids in recent5_ids.items():
                        t_shots_rec = shots_df[
                            ((shots_df["h_team"] == team) | (shots_df["a_team"] == team)) &
                            shots_df["match_id_str"].isin(ids)
                        ]
                        t_r = float(t_shots_rec["xG"].sum())
                        team_recent_xg[team] = t_r
                        if t_r > 0:
                            for pnorm, grp in t_shots_rec.groupby("player_norm"):
                                player_recent_xg[(pnorm, team)] = float(grp["xG"].sum())

                # Build player map — no DataFrame scans inside this loop
                player_xg_map = {}
                for _, fp in fpl_df.iterrows():
                    fteam = fp["team"]
                    pname = _norm(fp["fpl_name"])
                    web   = _norm(fp["web_name"])
                    xg_s  = float(fp.get("xg", 0) or 0)
                    xa_s  = float(fp.get("xa", 0) or 0)
                    t_xg  = team_xg_totals.get(fteam, 0)
                    t_xa  = team_xa_totals.get(fteam, 0)
                    season_share    = xg_s / t_xg if t_xg > 0 and xg_s > 0 else 0
                    season_xa_share = xa_s / t_xa if t_xa > 0 and xa_s > 0 else 0

                    recently_active = True
                    recent_share    = season_share
                    t_r = team_recent_xg.get(fteam, 0)
                    if t_r > 0:
                        p_r = player_recent_xg.get((pname, fteam)) or \
                              player_recent_xg.get((web, fteam), 0)
                        if p_r > 0:
                            recent_share = p_r / t_r
                        else:
                            recently_active = False

                    adj_share = 0.7 * season_share + 0.3 * recent_share
                    entry = {
                        "adj_share":       adj_share,
                        "xa_share":        season_xa_share,
                        "team":            fteam,
                        "recently_active": recently_active,
                    }
                    player_xg_map[pname] = entry
                    if web != pname:
                        player_xg_map[web] = entry

                for _, fp in fpl_df.iterrows():
                    if fp["status"] != "a":
                        continue
                    if int(fp["minutes"]) < 450:
                        continue
                    if fp["position"] not in ("MID", "FWD"):
                        continue

                    fpl_team = fp["team"]
                    team_fixes = team_fix_map.get(fpl_team, [])
                    if not team_fixes:
                        continue

                    # Name match
                    match = player_xg_map.get(_norm(fp["fpl_name"]))
                    if not match:
                        web = _norm(fp["web_name"])
                        match = next(
                            (v for k, v in player_xg_map.items()
                             if web in k and v["team"] == fpl_team),
                            None
                        )
                    if not match or match["adj_share"] <= 0:
                        continue
                    if not match.get("recently_active", True):
                        continue
                    # Minutes availability scale
                    games_played = fixtures_df[
                        ((fixtures_df["home_team"] == fpl_team) | (fixtures_df["away_team"] == fpl_team)) &
                        (fixtures_df["isResult"] == True)
                    ].shape[0]
                    avg_mins = int(fp["minutes"]) / games_played if games_played > 0 else 90
                    mins_scale = float(np.clip(avg_mins / 90, 0.3, 1.0))

                    # Per-fixture projections
                    xa_share = match.get("xa_share", 0)
                    fix_projections = []
                    for fix in sorted(team_fixes, key=lambda x: x["gw"]):
                        proj_xg = round(match["adj_share"] * fix["fix_xg"] * mins_scale, 3)
                        proj_xa = round(xa_share * fix["fix_xg"] * mins_scale, 3)
                        fix_projections.append({
                            "gw":      fix["gw"],
                            "label":   f"{fix['opp']} ({'H' if fix['is_home'] else 'A'})",
                            "proj_xg": proj_xg,
                            "proj_xa": proj_xa,
                            "diff":    fix["diff"],
                        })

                    gw1_xg = sum(f["proj_xg"] for f in fix_projections if f["gw"] == current_gw)
                    gw3_xg = round(sum(f["proj_xg"] for f in fix_projections if f["gw"] < current_gw + 3), 3)
                    gw5_xg = round(sum(f["proj_xg"] for f in fix_projections), 3)
                    gw1_xa = round(sum(f["proj_xa"] for f in fix_projections if f["gw"] == current_gw), 3)
                    gw3_xa = round(sum(f["proj_xa"] for f in fix_projections if f["gw"] < current_gw + 3), 3)
                    gw5_xa = round(sum(f["proj_xa"] for f in fix_projections), 3)

                    if gw1_xg <= 0:
                        continue

                    captain_picks.append({
                        "name":         fp["web_name"],
                        "team":         fpl_team,
                        "position":     fp["position"],
                        "price":        float(fp["price"]),
                        "ownership":    float(fp["ownership"]),
                        "gw1_xg":       round(gw1_xg, 3),
                        "gw3_xg":       gw3_xg,
                        "gw5_xg":       gw5_xg,
                        "gw1_xa":       gw1_xa,
                        "gw3_xa":       gw3_xa,
                        "gw5_xa":       gw5_xa,
                        "fixtures":     fix_projections,
                        "gw1_fixture":  fix_projections[0]["label"] if fix_projections else "",
                        "gw1_diff":     fix_projections[0]["diff"]  if fix_projections else "medium",
                    })

                captain_picks.sort(key=lambda x: x["gw1_xg"], reverse=True)
                captain_picks = captain_picks[:20]

            except Exception as e:
                print(f"⚠️  Captain picks error: {e}")
                import traceback; traceback.print_exc()

        fixture_ticker = []

        for team in teams:
            row_data = {"team": team, "fixtures": []}
            for gw in range(current_gw, current_gw + 5):
                gw_fix = fixtures_df[
                    ((fixtures_df["home_team"] == team) | (fixtures_df["away_team"] == team)) &
                    (fixtures_df["round_number"] == gw) &
                    (fixtures_df["isResult"] == False)
                ]
                if gw_fix.empty:
                    row_data["fixtures"].append({"gw": gw, "label": "BGW", "difficulty": "blank"})
                else:
                    fix     = gw_fix.iloc[0]
                    is_home = fix["home_team"] == team
                    opp     = fix["away_team"] if is_home else fix["home_team"]
                    if is_home:
                        prob_row = probs_df[(probs_df["home_team"] == team) & (probs_df["away_team"] == opp)]
                        win_prob = float(prob_row["home_win_prob"].iloc[0]) * 100 if not prob_row.empty else 50.0
                    else:
                        prob_row = probs_df[(probs_df["home_team"] == opp) & (probs_df["away_team"] == team)]
                        win_prob = float(prob_row["away_win_prob"].iloc[0]) * 100 if not prob_row.empty else 50.0
                    diff    = "easy" if win_prob >= 60 else "hard" if win_prob < 40 else "medium"
                    short   = TEAM_SHORT_NAMES.get(opp, opp[:3].upper())
                    row_data["fixtures"].append({
                        "gw":         gw,
                        "label":      f"{short} ({'H' if is_home else 'A'})",
                        "difficulty": diff,
                    })
            fixture_ticker.append(row_data)

        return render_template("fpl.html",
            cs_data=cs_data,
            xg_data=xg_data,
            xga_data=xga_data,
            fixture_ticker=fixture_ticker,
            captain_picks=captain_picks,
            current_gw=current_gw,
            gw_range=list(range(current_gw, current_gw + 5)),
            last_updated=get_last_updated_time()
        )
    except Exception as e:
        print(f"❌ FPL page error: {e}")
        import traceback; traceback.print_exc()
        return render_template("fpl.html",
            cs_data=[], xg_data=[], xga_data=[], fixture_ticker=[], captain_picks=[],
            current_gw=1, gw_range=[], last_updated=get_last_updated_time()
        )


@app.route("/ev_checker")
def ev_checker():
    # ── Fixture data ──
    fixture_path = "data/tables/fixture_data.csv"
    fixtures     = pd.read_csv(fixture_path)
    fixtures["isResult"]     = fixtures["isResult"].astype(str).str.lower() == "true"
    fixtures["round_number"] = pd.to_numeric(fixtures["round_number"], errors="coerce")

    upcoming = fixtures[fixtures["isResult"] == False].copy()
    upcoming["round_number"] = upcoming["round_number"].astype(int)
    current_gw = GW_OVERRIDE if GW_OVERRIDE is not None else (int(upcoming["round_number"].min()) if not upcoming.empty else 1)

    # ── Model probabilities — current GW only ──
    probs_df = pd.read_csv("data/tables/fixture_probabilities.csv")
    current_gw_fixtures = upcoming[upcoming["round_number"] == current_gw][["home_team", "away_team"]]
    probs_df = probs_df.merge(current_gw_fixtures, on=["home_team", "away_team"], how="inner")

    # ── Bookie win probabilities ──
    bookie_win_lookup = {}
    win_path = "data/tables/bookie_win_by_gw.csv"
    if os.path.exists(win_path):
        win_df = pd.read_csv(win_path)
        gw_label = f"GW{current_gw}"
        gw_rows  = win_df[win_df["gw"] == gw_label]
        for _, b in gw_rows.iterrows():
            if pd.notna(b["home_team"]) and b["home_team"] != "":
                key = f"{b['home_team']}|{b['away_team']}"
                bookie_win_lookup[key] = {
                    "bookie_home_win": float(b["bookie_home_win"]) if pd.notna(b["bookie_home_win"]) and b["bookie_home_win"] != "" else None,
                    "bookie_draw":     float(b["bookie_draw"])     if pd.notna(b["bookie_draw"])     and b["bookie_draw"]     != "" else None,
                    "bookie_away_win": float(b["bookie_away_win"]) if pd.notna(b["bookie_away_win"]) and b["bookie_away_win"] != "" else None,
                }

    # ── Bookie O/U 2.5 probabilities ──
    bookie_ou_lookup = {}
    ou_path = "data/tables/bookie_ou_by_gw.csv"
    if os.path.exists(ou_path):
        ou_df    = pd.read_csv(ou_path)
        gw_label = f"GW{current_gw}"
        ou_rows  = ou_df[ou_df["gw"] == gw_label]
        for _, b in ou_rows.iterrows():
            if pd.notna(b["home_team"]) and b["home_team"] != "":
                key = f"{b['home_team']}|{b['away_team']}"
                bookie_ou_lookup[key] = {
                    "bookie_over25":  float(b["bookie_over25"])  if pd.notna(b["bookie_over25"])  and b["bookie_over25"]  != "" else None,
                    "bookie_under25": float(b["bookie_under25"]) if pd.notna(b["bookie_under25"]) and b["bookie_under25"] != "" else None,
                }

    # ── Build EV data per fixture ──
    def ev_val(model, bookie):
        if model is None or bookie is None:
            return None
        return round(model - bookie, 1)

    ev_fixtures = []
    for _, row in probs_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        key  = f"{home}|{away}"

        model_home_win = round(float(row["home_win_prob"]) * 100, 1)
        model_draw     = round(float(row["draw_prob"])     * 100, 1)
        model_away_win = round(float(row["away_win_prob"]) * 100, 1)
        model_over25   = round(float(row.get("over_2_5_prob", 0)) * 100, 1)
        model_under25  = round(100 - model_over25, 1)

        bookie_win = bookie_win_lookup.get(key, {})
        bookie_ou  = bookie_ou_lookup.get(key, {})

        bookie_home_win = bookie_win.get("bookie_home_win")
        bookie_draw     = bookie_win.get("bookie_draw")
        bookie_away_win = bookie_win.get("bookie_away_win")
        bookie_over25   = bookie_ou.get("bookie_over25")
        bookie_under25  = bookie_ou.get("bookie_under25")

        ev_fixtures.append({
            "home_team":       home,
            "away_team":       away,
            "model_home_win":  model_home_win,
            "model_draw":      model_draw,
            "model_away_win":  model_away_win,
            "model_over25":    model_over25,
            "model_under25":   model_under25,
            "bookie_home_win": bookie_home_win,
            "bookie_draw":     bookie_draw,
            "bookie_away_win": bookie_away_win,
            "bookie_over25":   bookie_over25,
            "bookie_under25":  bookie_under25,
            "ev_home_win":  ev_val(model_home_win, bookie_home_win),
            "ev_draw":      ev_val(model_draw,     bookie_draw),
            "ev_away_win":  ev_val(model_away_win, bookie_away_win),
            "ev_over25":    ev_val(model_over25,   bookie_over25),
            "ev_under25":   ev_val(model_under25,  bookie_under25),
        })

    return render_template(
        "ev_checker.html",
        ev_fixtures=ev_fixtures,
        current_gw=current_gw,
        last_updated=get_last_updated_time()
    )














if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if port is None:
        raise RuntimeError("PORT environment variable is not set.")
    app.run(host="0.0.0.0", port=port, debug=True)