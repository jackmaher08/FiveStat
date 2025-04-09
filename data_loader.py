import os
import re
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.colors as mcolors
from bs4 import BeautifulSoup
from mplsoccer import Pitch
from mplsoccer import VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import random
import matplotlib.image as mpimg
import seaborn as sns
import subprocess

TEAM_NAME_MAPPING = {
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest"
}


def run_data_scraper():
    """Runs data_scraper_script.py to update fixture data before loading."""
    script_path = os.path.join("data", "data_scraper_script.py") 

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"❌ data_scraper_script.py not found at {script_path}")

    print("🔄 Running data_scraper_script.py to update data...")
    subprocess.run(["python", script_path], check=True) 
    print("Data scraper completed.")

    # Run generate_shotmaps.py
    shotmaps_script_path = os.path.join("data", "generate_shotmaps.py")
    if os.path.exists(shotmaps_script_path):
        print("🔄 Running generate_shotmaps.py to update shotmaps...")
        subprocess.run(["python", shotmaps_script_path], check=True)
        print("✅ Shotmaps generated.")

    '''# Run generate_radars.py
    radars_script_path = os.path.join("data", "generate_radars.py")
    if os.path.exists(radars_script_path):
        print("🔄 Running generate_radars.py to update radar charts...")
        subprocess.run(["python", radars_script_path], check=True)
        print("✅ Radar charts generated.")'''
    
def get_player_radar_data():
    radar_file_path = "data/tables/player_radar_data.csv"
    if os.path.exists(radar_file_path):
        radar_df = pd.read_csv(radar_file_path)
    else:
        raise FileNotFoundError(f"⚠️ Player radar data file not found: {radar_file_path}.")
    # Filter to only include players from the Premier League
    radar_df = radar_df[radar_df['Comp'] == 'eng Premier League']
    return radar_df.to_dict(orient="records")


# Function to load fixture data from multiple sources
def load_fixtures():
    fixture_file_path = "data/tables/fixture_data.csv"
    if os.path.exists(fixture_file_path):
        fixtures_df = pd.read_csv(fixture_file_path)
    else:
        raise FileNotFoundError(f"⚠️ Fixture file not found: {fixture_file_path}. Ensure it's saved before running.")

    return fixtures_df

# Function to load historical match data
def load_match_data(start_year=2016, end_year=2024):
    historical_fixture_file_path = "data/tables/historical_fixture_data.csv"
    
    if os.path.exists(historical_fixture_file_path):
        historical_fixtures_df = pd.read_csv(historical_fixture_file_path)
    else:
        raise FileNotFoundError(f"⚠️ Fixture file not found: {historical_fixture_file_path}. Ensure it's saved before running.")

    return historical_fixtures_df

def load_next_gw_fixtures():
    """Loads the next gameweek fixtures from the saved file."""
    next_gw_file_path = "data/tables/next_gw_fixtures.csv"

    if os.path.exists(next_gw_file_path):
        next_gw_fixtures_df = pd.read_csv(next_gw_file_path)
        return next_gw_fixtures_df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
    else:
        raise FileNotFoundError(f"⚠️ Next gameweek fixtures file not found: {next_gw_file_path}. Ensure it's saved before running.")


def get_player_data():
    player_file_path = "data/tables/player_data.csv"
    
    if os.path.exists(player_file_path):
        player_data_df = pd.read_csv(player_file_path)
    else:
        raise FileNotFoundError(f"⚠️ Player data file not found: {player_file_path}. Ensure it's saved before running.")

    return player_data_df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries




# Function to calculate team statistics
def calculate_team_statistics(historical_fixture_data):
    team_names = historical_fixture_data['Home Team'].unique()
    home_field_advantage = historical_fixture_data['home_goals'].mean() - historical_fixture_data['away_goals'].mean()
    team_data = {}

    for team in team_names:
        home_games = historical_fixture_data[historical_fixture_data['Home Team'] == team]
        away_games = historical_fixture_data[historical_fixture_data['Away Team'] == team]

        avg_home_goals_for = home_games['home_goals'].mean()
        avg_away_goals_for = away_games['away_goals'].mean()
        avg_home_goals_against = home_games['away_goals'].mean()
        avg_away_goals_against = away_games['home_goals'].mean()

        att_rating = (avg_home_goals_for + avg_away_goals_for) / 2
        def_rating = (avg_home_goals_against + avg_away_goals_against) / 2

        team_data[team] = {
            'Home Goals For': avg_home_goals_for,
            'Away Goals For': avg_away_goals_for,
            'Home Goals Against': avg_home_goals_against,
            'Away Goals Against': avg_away_goals_against,
            'ATT Rating': att_rating,
            'DEF Rating': def_rating
        }
        
        print(f"=== Overall Historical Stats for {team} ===")
        print(f"Avg Home Goals For: {avg_home_goals_for}")
        print(f"Avg Away Goals For: {avg_away_goals_for}")
        print(f"Avg Home Goals Against: {avg_home_goals_against}")
        print(f"Avg Away Goals Against: {avg_away_goals_against}")
        print(f"ATT Rating: {att_rating}")
        print(f"DEF Rating: {def_rating}\n")

    return team_data, home_field_advantage



# Function to calculate recent form ratings
def calculate_recent_form(historical_fixture_data, team_data, recent_matches=20, alpha=0.65):
    recent_form_att = {}
    recent_form_def = {}

    for team in historical_fixture_data['Home Team'].unique():
        recent_matches_df = historical_fixture_data[(historical_fixture_data['Home Team'] == team) | (historical_fixture_data['Away Team'] == team)].tail(recent_matches)
        
        home_matches = recent_matches_df[recent_matches_df['Home Team'] == team]
        away_matches = recent_matches_df[recent_matches_df['Away Team'] == team]

        avg_home_att = home_matches['home_goals'].mean()
        avg_away_att = away_matches['away_goals'].mean()
        avg_home_def = home_matches['away_goals'].mean()
        avg_away_def = away_matches['home_goals'].mean()

        recent_att = ((1 - alpha) * team_data[team]['ATT Rating']) + (alpha * ((avg_home_att + avg_away_att) / 2))
        recent_def = ((1 - alpha) * team_data[team]['DEF Rating']) + (alpha * ((avg_home_def + avg_away_def) / 2))

        recent_form_att[team] = recent_att
        recent_form_def[team] = recent_def

        print(f"=== Recent Form Stats for {team} ===")
        print(f"Avg Home Att: {avg_home_att}")
        print(f"Avg Away Att: {avg_away_att}")
        print(f"Avg Home Def: {avg_home_def}")
        print(f"Avg Away Def: {avg_away_def}")
        print(f"Recent ATT: {recent_att}")
        print(f"Recent DEF: {recent_def}\n")

    return recent_form_att, recent_form_def



# Function to simulate a match using Poisson distribution
def simulate_poisson_distribution(home_xg, away_xg, max_goals=12):
    result_matrix = np.zeros((max_goals, max_goals))
    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            home_prob = poisson.pmf(home_goals, home_xg)
            away_prob = poisson.pmf(away_goals, away_xg)
            result_matrix[home_goals, away_goals] = home_prob * away_prob
    # Normalize the matrix so the probabilities sum to 1
    result_matrix /= result_matrix.sum()

    # Calculate overall probabilities
    home_win_prob = np.sum(np.tril(result_matrix, -1))  # Below diagonal
    away_win_prob = np.sum(np.triu(result_matrix, 1))   # Above diagonal
    draw_prob = np.sum(np.diag(result_matrix))          # Diagonal

    #print(f"XG: {home_xg:.2f} vs {away_xg:.2f} -> Home Win: {home_win_prob:.3f}, Draw: {draw_prob:.3f}, Away Win: {away_win_prob:.3f}")

    return result_matrix, home_win_prob, draw_prob, away_win_prob


# Function to generate a heatmap
def display_heatmap(result_matrix, home_team, away_team, gw_number, home_prob, draw_prob, away_prob, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]}, facecolor="#f4f4f9")

    # Heatmap
    heatmap_ax = axes[0]
    display_matrix = result_matrix[:6, :6]  # Limit to 6x6 grid
    heatmap_ax.imshow(display_matrix, cmap="Purples", origin='upper')

    # Move x-axis labels and ticks to the top
    heatmap_ax.xaxis.set_label_position('top')
    heatmap_ax.xaxis.tick_top()

    # Labeling
    heatmap_ax.set_xlabel(f"{away_team} Goals")
    heatmap_ax.set_ylabel(f"{home_team} Goals")
    
    # Add percentage text inside each cell
    for i in range(6):
        for j in range(6):
            heatmap_ax.text(j, i, f"{display_matrix[i, j] * 100:.1f}%", 
                            ha='center', va='center', color='black', fontsize=8)

    # Hide spines
    for spine in heatmap_ax.spines.values():
        spine.set_visible(False)

    # Bar Chart
    bar_ax = axes[1]
    bar_ax.set_facecolor('#f4f4f9') 

    categories = [f"{home_team}", "Draw", f"{away_team}"]
    values = [home_prob * 100, draw_prob * 100, away_prob * 100]

    # Vertical bars for the probabilities
    bars = bar_ax.bar(categories, values, color='#3f007d', alpha=0.9, width=0.6)

    # Title for the bar chart
    bar_ax.set_title("Projected Win %'s:")

    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        bar_ax.text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.1f}%", 
                    ha='center', fontsize=10, fontweight='bold')

    # Remove unnecessary spines and y-ticks
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_visible(False)
    bar_ax.spines['bottom'].set_visible(False)
    bar_ax.set_yticks([])

    # Add watermark
    fig.text(0.97, 0.60, "FiveStat", fontsize=8, color="black", fontweight="bold", 
             ha="left", va="bottom", alpha=0.4, rotation=90)
    
    # Adjust layout and save the figure unconditionally
    plt.tight_layout()
    heatmap_filename = f"{home_team}_{away_team}_heatmap.png"
    heatmap_path = os.path.join(save_path, heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap for {home_team} vs {away_team} at {heatmap_path}")



# Calc the XG we need to keep a teams att rating the same
def find_xg_to_match_att_rating(target_att, opp_def, is_home, tolerance=1e-3, max_iter=100):
    """Binary search to find xG that gives expected goals ≈ target_att."""
    from data_loader import simulate_poisson_distribution

    low, high = 0.1, 5.0  # Reasonable xG bounds
    for _ in range(max_iter):
        mid = (low + high) / 2
        home_xg = mid if is_home else opp_def
        away_xg = opp_def if is_home else mid

        result_matrix, _, _, _ = simulate_poisson_distribution(home_xg, away_xg)

        # Expected goals calculation
        expected_goals = 0.0
        for i in range(result_matrix.shape[0]):
            for j in range(result_matrix.shape[1]):
                prob = result_matrix[i, j]
                expected_goals += (i if is_home else j) * prob

        if abs(expected_goals - target_att) < tolerance:
            return mid  # Found matching xG
        elif expected_goals > target_att:
            high = mid
        else:
            low = mid

    return mid  # Best approximation

def get_team_xg(
    team, opponent, is_home, team_stats, recent_form_att, recent_form_def,
    alpha=0.65, beta=0.6, home_field_advantage=0.15
):
    """
    Returns the blended xG value for a given team against an opponent,
    combining Poisson-based calibration with intuitive attack × defense logic.

    Args:
        team (str): Team name.
        opponent (str): Opponent team name.
        is_home (bool): True if team is playing at home.
        team_stats (dict): Contains 'ATT Rating' and 'DEF Rating' for each team.
        recent_form_att (dict): Recent ATT ratings.
        recent_form_def (dict): Recent DEF ratings.
        alpha (float): Weight of recent form in ATT/DEF rating blend.
        beta (float): Weight of multiplicative xG vs Poisson-calibrated xG.
        home_field_advantage (float): Additive bonus if team is at home.

    Returns:
        float: Blended xG value.


    Tweaking beta
    beta = 0.0: only Poisson logic (status quo)
    beta = 1.0: only attack × defense model
    """
    # 1. Get blended ratings
    att_rating = (1 - alpha) * team_stats[team]['ATT Rating'] + alpha * recent_form_att[team]
    def_rating = (1 - alpha) * team_stats[opponent]['DEF Rating'] + alpha * recent_form_def[opponent]

    # 2. Poisson-based xG to preserve ATT rating
    poisson_matched_xg = find_xg_to_match_att_rating(att_rating, def_rating, is_home=is_home)

    # 3. Simple multiplicative xG model
    multiplicative_xg = att_rating * def_rating

    # 4. Blend both
    true_xg = (1 - beta) * poisson_matched_xg + beta * multiplicative_xg

    # 5. Home field bonus
    if is_home:
        true_xg += home_field_advantage

    return true_xg





def generate_all_heatmaps(team_stats, recent_form_att, recent_form_def, alpha=0.65, save_path="static/heatmaps/"):
    print("🔄 Running generate_all_heatmaps()...")

    fixture_file_path = "data/tables/fixture_data.csv"
    probabilities_file_path = "data/tables/fixture_probabilities.csv"

    if not os.path.exists(fixture_file_path):
        print("❌ Fixture file missing! Exiting...")
        raise FileNotFoundError(f"Fixture file not found: {fixture_file_path}. Ensure it's saved before running.")

    print("✅ Fixture file found, loading data...")
    fixtures_df = pd.read_csv(fixture_file_path)

    print("✅ Creating a new DataFrame for probabilities...")
    probabilities_df = fixtures_df[['home_team', 'away_team']].copy()

    print("✅ Initializing empty probability columns...")
    probabilities_df["home_win_prob"] = np.nan
    probabilities_df["draw_prob"] = np.nan
    probabilities_df["away_win_prob"] = np.nan

    # Normalize team keys in stats dictionaries
    team_stats = {TEAM_NAME_MAPPING.get(k, k): v for k, v in team_stats.items()}
    recent_form_att = {TEAM_NAME_MAPPING.get(k, k): v for k, v in recent_form_att.items()}
    recent_form_def = {TEAM_NAME_MAPPING.get(k, k): v for k, v in recent_form_def.items()}

    print("✅ Processing matches to calculate probabilities and generate heatmaps...")
    for index, fixture in fixtures_df.iterrows():
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        home_team = TEAM_NAME_MAPPING.get(home_team, home_team)
        away_team = TEAM_NAME_MAPPING.get(away_team, away_team)


        if pd.isna(home_team) or pd.isna(away_team):
            continue

        if home_team not in team_stats or away_team not in team_stats:
            continue

        home_xg = get_team_xg(home_team, away_team, is_home=True, team_stats=team_stats, 
                      recent_form_att=recent_form_att, recent_form_def=recent_form_def)

        away_xg = get_team_xg(away_team, home_team, is_home=False, team_stats=team_stats, 
                      recent_form_att=recent_form_att, recent_form_def=recent_form_def)




        # Capture the full result_matrix along with probabilities
        result_matrix, home_prob, draw_prob, away_prob = simulate_poisson_distribution(home_xg, away_xg)

        probabilities_df.at[index, "home_win_prob"] = home_prob
        probabilities_df.at[index, "draw_prob"] = draw_prob
        probabilities_df.at[index, "away_win_prob"] = away_prob

        # Call display_heatmap to generate and save the image (this will print a confirmation)
        display_heatmap(result_matrix, home_team, away_team, fixture.get('round_number', ''), home_prob, draw_prob, away_prob, save_path)

    print("🔄 Saving match probabilities to fixture_probabilities.csv...")
    probabilities_df.to_csv(probabilities_file_path, index=False)
    print("✅ fixture_probabilities.csv successfully created at:", probabilities_file_path)





# Directory to save shotmaps
shotmap_save_path = "static/shotmaps/"
os.makedirs(shotmap_save_path, exist_ok=True)

# Fetch the latest fixtures (Merged from FixtureDownload & Understat)
fixtures_df = load_fixtures()

# Filter only completed matches
completed_fixtures = fixtures_df[(fixtures_df["isResult"] == True)]




# Function to generate and save shot maps
def generate_shot_map(understat_match_id):
    try:
        url = f'https://understat.com/match/{understat_match_id}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        ugly_soup = str(soup)

        # Extract JSON shot data
        match = re.search("var shotsData .*= JSON.parse\\('(.*)'\\)", ugly_soup)
        if not match:
            print(f"Skipping match {understat_match_id}: No shot data found")
            return

        shots_data = match.group(1)
        data = shots_data.encode('utf8').decode('unicode_escape')
        data = json.loads(data)

        # Create DataFrames
        home_df = pd.DataFrame(data['h'])
        away_df = pd.DataFrame(data['a'])

        # Extract and update team names
        home_team_name = home_df.iloc[0]['h_team'] if not home_df.empty else "Unknown"
        away_team_name = away_df.iloc[0]['a_team'] if not away_df.empty else "Unknown"

        # Scale coordinates to StatsBomb pitch (120x80)
        home_df['x_scaled'] = home_df['X'].astype(float) * 120
        home_df['y_scaled'] = home_df['Y'].astype(float) * 80
        away_df['x_scaled'] = away_df['X'].astype(float) * 120
        away_df['y_scaled'] = away_df['Y'].astype(float) * 80

        # Adjust positions for correct plotting
        home_df['x_scaled'] = 120 - home_df['x_scaled']
        away_df['y_scaled'] = 80 - away_df['y_scaled']

        # Calculate total goals and xG
        goal_keywords = ['Goal', 'PenaltyGoal']
        own_goal_keyword = 'OwnGoal'

        # Count regular & penalty goals normally
        home_goals = home_df['result'].apply(lambda x: any(kw in str(x) for kw in goal_keywords)).sum()
        away_goals = away_df['result'].apply(lambda x: any(kw in str(x) for kw in goal_keywords)).sum()

        # Count own goals & assign to other team
        home_own_goals = home_df['result'].apply(lambda x: own_goal_keyword in str(x)).sum()
        away_own_goals = away_df['result'].apply(lambda x: own_goal_keyword in str(x)).sum()

        # Correct final goal counts
        total_goals_home = home_goals + away_own_goals - home_own_goals  # Home goals + own goals scored by away team
        total_goals_away = away_goals + home_own_goals - away_own_goals  # Away goals + own goals scored by home team

        total_xg_home = home_df['xG'].astype(float).sum()
        total_xg_away = away_df['xG'].astype(float).sum()
    
        # sum stats for table
        def calculate_match_stats(team_df):
            team_df['xG'] = pd.to_numeric(team_df['xG'], errors='coerce')  # Ensure xG is numeric
            return {
                'Goals': len(team_df[team_df['result'] == 'Goal']),  
                'xG': round(team_df['xG'].sum(), 2),  
                'Shots': len(team_df),  
                'SOT': len(team_df[team_df['result'].isin(['Goal', 'SavedShot'])])  
            }

        home_stats = calculate_match_stats(home_df)
        away_stats = calculate_match_stats(away_df)

        # Initialize pitch
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#f4f4f9', line_color='black', line_zorder=2)
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})


        # Set background color
        fig.patch.set_facecolor('#f4f4f9')
        axs[0].set_facecolor('#f4f4f9')

        # Plot heatmap
        all_shots = pd.concat([home_df, away_df])
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f4f4f9', '#3f007d'])
        pitch.kdeplot(
            all_shots['x_scaled'], all_shots['y_scaled'], ax=axs[0], fill=True, cmap=cmap,
            n_levels=100, thresh=0, zorder=1
        )

        # Draw pitch
        pitch.draw(ax=axs[0])

        # Plot shots for both teams
        for df in [home_df, away_df]:  
            for _, shot in df.iterrows():
                x, y = shot['x_scaled'], shot['y_scaled']
                color = 'gold' if "goal" in str(shot['result']).lower() else 'white'
                zorder = 3 if shot['result'] == 'Goal' else 2
                axs[0].scatter(x, y, s=1000 * float(shot['xG']) if pd.notna(shot['xG']) else 100, 
                           ec='black', c=color, zorder=zorder)

        # Define the base path (where data_loader.py is located)
        base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of data_loader.py

        # Construct full paths for the logos
        home_logo_path = os.path.join(base_path, "static", "team_logos", f"{home_team_name.lower()}_logo.png")
        away_logo_path = os.path.join(base_path, "static", "team_logos", f"{away_team_name.lower()}_logo.png")

        def add_team_logo(ax, logo_path, y_min, y_max, x_center):
            """Loads and displays a team logo at a given position, keeping aspect ratio and flipping it if necessary."""
            if os.path.exists(logo_path):
                logo_img = mpimg.imread(logo_path)
                
                # Flip the image vertically so it appears correctly
                logo_img = np.flipud(logo_img)  
                
                # Get image aspect ratio (height / width)
                aspect_ratio = logo_img.shape[0] / logo_img.shape[1] 
                
                # Set width dynamically based on height
                height = y_max - y_min  # Define height of the image
                width = height / aspect_ratio  # Maintain ratio
                
                x_min = x_center - (width / 2)  # Centered positioning
                x_max = x_center + (width / 2)

                # Display the flipped image with transparency (alpha)
                ax.imshow(logo_img, extent=(x_min, x_max, y_min, y_max), alpha=0.1, zorder=1)

        # Add team logos
        add_team_logo(axs[0], home_logo_path, y_min=20, y_max=60, x_center=30)  # Home team
        add_team_logo(axs[0], away_logo_path, y_min=20, y_max=60, x_center=90)  # Away team


        # Add match info
        axs[0].text(30, 40, f"{total_goals_home}", ha='center', va='center', fontsize=180, fontweight='bold', color='black', alpha=0.5)
        axs[0].text(90, 40, f"{total_goals_away}", ha='center', va='center', fontsize=180, fontweight='bold', color='black', alpha=0.5)
        axs[0].text(30, 60, f"{total_xg_home:.2f}", ha='center', va='center', fontsize=45, fontweight='bold', color='black', alpha=0.6)
        axs[0].text(90, 60, f"{total_xg_away:.2f}", ha='center', va='center', fontsize=45, fontweight='bold', color='black', alpha=0.6)
        axs[0].text(105,78, f"Respective Team XG values", ha='center', va='center', fontsize=8, fontweight='bold', color='black', alpha=0.4)
        axs[0].text(6,78,   f"FiveStat", ha='center', va='center', fontsize=8, fontweight='bold', color='black', alpha=0.4)


        # Generate Table
        ax_table = axs[1]
        column_labels = [f"{home_team_name}", "", f"{away_team_name}"]
        table_vals = [
            [total_goals_home, 'Goals', total_goals_away],
            [home_stats['xG'], 'xG', away_stats['xG']],  
            [home_stats['Shots'], 'Shots', away_stats['Shots']],  
            [home_stats['SOT'], 'SOT', away_stats['SOT']]  
        ]
        table = ax_table.table(
            cellText=table_vals,
            cellLoc='center',
            colLabels=column_labels,
            bbox=[0, 0, 1, 1]
        )

        for i in range(len(table_vals) + 1):  # +1 to include header row
            for j in range(len(column_labels)):
                cell = table[(i, j)]
                cell.set_facecolor("#f4f4f9")  # Background color

        column_widths = [0.4, 0.2, 0.4]

        for j, width in enumerate(column_widths):
            for i in range(len(table_vals) + 1):  # +1 includes header row
                cell = table[i, j]
                cell.set_width(width)

        for (i, j), cell in table.get_celld().items():
            if j == 0:
                table.get_celld()[(i, j)].visible_edges = 'R'
            elif j == 2:
                table.get_celld()[(i, j)].visible_edges = 'L'
            else:
                table.get_celld()[(i, j)].visible_edges = 'LR'

        ax_table.axis('off')  # Hide axes for the table
        table.set_fontsize(16)

        # Save figure
        plt.tight_layout()
        shotmap_file = os.path.join(shotmap_save_path, f"{home_team}_{away_team}_shotmap.png")
        plt.savefig(shotmap_file)
        plt.close(fig)

    except Exception as e:
        print(f"❌ Error processing match {understat_match_id}: {e}")

# Loop through completed fixtures only and generate shotmaps
for _, row in completed_fixtures.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    match_id = row['id']
    shotmap_file = os.path.join(shotmap_save_path, f"{home_team}_{away_team}_shotmap.png")

    # Skip if shotmap already exists
    if os.path.exists(shotmap_file):
        continue

    generate_shot_map(match_id)



# Function to generate and save goals vs. xG bar chart
def generate_team_goals_xg_charts():
    DATA_PATH = "data/tables/fixture_data.csv"
    SAVE_DIR = "static/chart_images/teams"
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Skipping chart generation.")
        return

    df = pd.read_csv(DATA_PATH)
    df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")

    required_cols = {"round_number", "home_team", "away_team", "home_goals", "away_goals", "home_xG", "away_xG"}
    if not required_cols.issubset(df.columns):
        print(f"❌ Missing columns in fixture_data.csv: {required_cols - set(df.columns)}")
        return

    # Get all unique team names
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))

    for team in teams:
        # Filter all matches where the team played
        team_games = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()

        # Calculate goals and xG for the team (home or away)
        team_games["goals"] = team_games.apply(
            lambda row: row["home_goals"] if row["home_team"] == team else row["away_goals"], axis=1
        )
        team_games["xG"] = team_games.apply(
            lambda row: row["home_xG"] if row["home_team"] == team else row["away_xG"], axis=1
        )

        # Group by gameweek
        weekly_stats = team_games.groupby("round_number").agg({
            "goals": "sum",
            "xG": "sum"
        }).sort_index()

        # Plot chart
        # ✅ Calculate average goals (only for played weeks)
        avg_goals = weekly_stats["goals"].mean()

        # Plot chart
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#f4f4f9")
        ax.set_facecolor("#f4f4f9")

        bar_width = 0.4
        x = weekly_stats.index
        ax.bar(x - bar_width / 2, weekly_stats["goals"], width=bar_width, label="Goals", color="#3f007d")
        ax.bar(x + bar_width / 2, weekly_stats["xG"], width=bar_width, label="xG", color="#9163cb")

        # ✅ Add average goals line
        ax.axhline(y=avg_goals, color="black", linestyle="dashed", linewidth=1.5, label=f"Avg Goals ({avg_goals:.2f})")

        ax.legend(frameon=False)
        ax.set_xlabel("Gameweek")
        ax.set_ylabel("Total")

        # Clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        chart_path = os.path.join(SAVE_DIR, f"{team}_goals_xg_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close()
        print(f"✅ Saved: {chart_path}")




os.makedirs("data/tables", exist_ok=True)
# Load remaining fixtures
fixtures = pd.read_csv("data/tables/fixture_data.csv")

# Load current league table
league_table = pd.read_csv("data/tables/league_table_data.csv")

# Extract necessary columns
team_points = league_table.set_index("Team")["PTS"].to_dict()
teams = list(team_points.keys())

# Simulation Parameters
num_simulations = 10000
num_teams = len(teams)
num_positions = num_teams  # Positions 1 to last place

# Create a dictionary to store simulation results
position_counts = {team: np.zeros(num_positions) for team in teams}






if __name__ == "__main__":
    print("🚀 Starting data_loader.py...")

    print("🔄 Running data scraper to update fixtures...")
    run_data_scraper()

    print("🔄 Loading match data...")
    historical_fixtures_df = load_match_data()
    team_data, home_field_advantage = calculate_team_statistics(historical_fixtures_df)

    print("🔄 Calculating recent form...")
    recent_form_att, recent_form_def = calculate_recent_form(
        historical_fixtures_df, team_data, recent_matches=20, alpha=0.65
    )

    print("🔄 Ensuring generate_all_heatmaps() runs before simulation...")
    generate_all_heatmaps(team_data, recent_form_att, recent_form_def)
    print("✅ generate_all_heatmaps() executed successfully!")

    
    probabilities_file_path = "data/tables/fixture_probabilities.csv"
    print("🔄 Loading match probabilities from fixture_probabilities.csv...")

    try:
        probabilities_df = pd.read_csv(probabilities_file_path)
    except FileNotFoundError:
        raise ValueError("❌ 'fixture_probabilities.csv' is missing! Check if generate_all_heatmaps() is running.")

    if "home_win_prob" not in probabilities_df.columns:
        raise ValueError("❌ 'home_win_prob' is missing! Ensure generate_all_heatmaps() ran properly.")

    print("✅ Match probabilities successfully loaded!")

    
    print("🔄 Running Monte Carlo simulation: 10,000 sims")

    # Load fixture data (contains results) and match probabilities
    fixture_data = pd.read_csv("data/tables/fixture_data.csv")  # Contains `result`
    probabilities_df = pd.read_csv("data/tables/fixture_probabilities.csv")  # Contains win probabilities

    # Merge the datasets to ensure we have probabilities + results
    fixtures = fixture_data.merge(probabilities_df, on=["home_team", "away_team"], how="left")

    # Filter only games that haven't been played (where `result` column is NULL)
    remaining_fixtures = fixtures[fixtures["result"].isna()]

    # Load current league table
    league_table = pd.read_csv("data/tables/league_table_data.csv")

    # Extract current points for each team
    team_points = league_table.set_index("Team")["PTS"].to_dict()
    teams = list(team_points.keys())

    # Simulation Parameters
    num_simulations = 10000
    num_teams = len(teams)
    num_positions = num_teams  # Positions 1 to last place

    # Create a dictionary to store simulation results
    num_positions = 20  # Ensure exactly 20 positions
    position_counts = {team: np.zeros(num_positions) for team in teams}

    simulated_remaining_points = {team: 0 for team in teams}  # ✅ Track only points from unplayed matches

    # Monte Carlo Simulation (Simulating Remaining Fixtures Only)
    for _ in range(num_simulations):
        simulated_points = team_points.copy()  # ✅ Start with real league points

        for _, match in remaining_fixtures.iterrows():  # ✅ Use only unplayed matches
            home_team = TEAM_NAME_MAPPING.get(match["home_team"], match["home_team"])
            away_team = TEAM_NAME_MAPPING.get(match["away_team"], match["away_team"])


            # Ensure team exists before simulating
            if home_team not in simulated_points or away_team not in simulated_points:
                print(f"⚠️ Warning: {home_team} or {away_team} not found in league table! Skipping match.")
                continue

            home_prob = match["home_win_prob"]
            draw_prob = match["draw_prob"]
            away_prob = match["away_win_prob"]

            # Simulate match result
            outcome = np.random.choice(["home_win", "draw", "away_win"], p=[home_prob, draw_prob, away_prob])

            # Update points only for remaining fixtures
            if outcome == "home_win":
                simulated_points[home_team] += 3
                simulated_remaining_points[home_team] += 3
            elif outcome == "draw":
                simulated_points[home_team] += 1
                simulated_points[away_team] += 1
                simulated_remaining_points[home_team] += 1
                simulated_remaining_points[away_team] += 1
            else:  # away win
                simulated_points[away_team] += 3
                simulated_remaining_points[away_team] += 3

        # Rank teams based on final simulated points
        sorted_teams = sorted(simulated_points.items(), key=lambda x: x[1], reverse=True)

        # Record finishing positions
        for rank, (team, _) in enumerate(sorted_teams):
            position_counts[team][rank] += 1

    # Compute Final xPTS (Current Points + Expected Simulated Points for Remaining Games)
    average_xPTS = {}
    for team in teams:
        avg_sim_points = simulated_remaining_points.get(team, 0) / num_simulations
        average_xPTS[team] = team_points.get(team, 0) + avg_sim_points  # Add to current points


    # Ensure all teams have 1-20 position keys before saving
    for team in position_counts:
        for pos in range(num_positions):  # Loop through indices (0-19)
            if pos >= len(position_counts[team]):  # If index out of range, fill it
                position_counts[team][pos] = 0




    # Convert position counts to percentages
    final_probabilities = pd.DataFrame(position_counts)
    final_probabilities = final_probabilities.T  # Transpose (Teams as rows)
    final_probabilities.columns = [str(i) for i in range(1, num_positions + 1)]  # Ensure column names are strings
    final_probabilities /= num_simulations  # Keep precise values (e.g., 0.0036, not 0.0)

    # Add Final xPTS Column
    final_probabilities["Final xPTS"] = final_probabilities.index.map(average_xPTS)

    # Rank teams based on Final xPTS
    final_probabilities = final_probabilities.sort_values(by="Final xPTS", ascending=False)

    # Save results to CSV
    output_file_path = "data/tables/simulated_league_positions.csv"
    final_probabilities.to_csv(output_file_path, index=True, float_format="%.6f")

    print(f"✅ Simulation results saved to: {output_file_path}")

    generate_team_goals_xg_charts()