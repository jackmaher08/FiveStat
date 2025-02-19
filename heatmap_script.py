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
from matplotlib.colors import LinearSegmentedColormap
import random

# Function to load fixture data from multiple sources
def load_fixtures():
    # 📌 **First Source: FixtureDownload**
    url1 = "https://fixturedownload.com/download/epl-2024-GMTStandardTime.csv"
    fixtures_df = pd.read_csv(url1)

    # Rename columns to match expected format
    fixtures_df = fixtures_df.rename(columns={
        "Round Number": "round_number",
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Date": "date",
        "Result": "result"
    })

    # Convert 'round_number' to numeric
    fixtures_df["round_number"] = pd.to_numeric(fixtures_df["round_number"], errors="coerce")

    # 📌 **Second Source: Understat**
    url2 = "https://understat.com/league/EPL/2024"
    response = requests.get(url2)
    soup = BeautifulSoup(response.content, 'html.parser')
    ugly_soup = str(soup)

    # Extract JSON fixture data
    match = re.search("var datesData .*= JSON.parse\\('(.*)'\\)", ugly_soup)
    if match:
        all_fixture_data = match.group(1).encode('utf8').decode('unicode_escape')
        all_fixture_df = json.loads(all_fixture_data)
    else:
        print("⚠️ No fixture data found on Understat")
        all_fixture_df = []

    # Team name mapping for consistency
    team_name_mapping = {
        "Manchester United": "Man Utd",
        "Newcastle United": "Newcastle",
        "Manchester City": "Man City",
        "Tottenham": "Spurs",
        "Wolverhampton Wanderers": "Wolves",
        "Nottingham Forest": "Nott'm Forest"
    }

    # Parse fixture data
    fixture_data = []
    for fixture in all_fixture_df:
        fixture_entry = {
            "id": fixture.get("id"),
            "isResult": fixture.get("isResult"),
            "home_team": team_name_mapping.get(fixture["h"]["title"], fixture["h"]["title"]),
            "away_team": team_name_mapping.get(fixture["a"]["title"], fixture["a"]["title"]),
            "home_goals": int(fixture["goals"]["h"]) if fixture.get("goals") and fixture["goals"].get("h") is not None else None,
            "away_goals": int(fixture["goals"]["a"]) if fixture.get("goals") and fixture["goals"].get("a") is not None else None,
            "home_xG": round(float(fixture["xG"]["h"]), 2) if fixture.get("xG") and fixture["xG"].get("h") is not None else None,
            "away_xG": round(float(fixture["xG"]["a"]), 2) if fixture.get("xG") and fixture["xG"].get("a") is not None else None,
        }
        fixture_data.append(fixture_entry)

    # Convert Understat data to DataFrame
    fixture_data_df = pd.DataFrame(fixture_data)

    # 🔄 **Merge DataFrames**
    merged_fixture_df = pd.merge(
        fixtures_df[["round_number", "date", "home_team", "away_team", "result"]],
        fixture_data_df[["id", "home_team", "away_team", "isResult", "home_goals", "away_goals", "home_xG", "away_xG"]],
        on=["home_team", "away_team"],  
        how="left"  # Keep all fixtures even if no match in fixture_data_df
    )

    return merged_fixture_df

print("Fixtures loaded successfully!")






# Function to load historical match data
def load_match_data(start_year=2016, end_year=2024):
    # Load all seasons' data
    frames = [] 
    for year in range(start_year, end_year + 1):
        url = f"https://fixturedownload.com/download/epl-{year}-GMTStandardTime.csv"
        frame = pd.read_csv(url)
        frame['Season'] = year
        frames.append(frame)

        frame = frame.rename(columns={
            "Round Number": "round_number",
            "Home Team": "home_team",
            "Away Team": "away_team",
            "Date": "date",
            "Result": "result"
        })

    # Merge all season data
    df = pd.concat(frames)
    df = df[pd.notnull(df["Result"])]  # Keep only matches with results

    # Process result column
    df[['home_goals', 'away_goals']] = df['Result'].str.split(' - ', expand=True).astype(float)
    df['result'] = df.apply(lambda row: 'home_win' if row['home_goals'] > row['away_goals'] 
                            else 'away_win' if row['home_goals'] < row['away_goals'] else 'draw', axis=1)

    # 📌 **Generate Team ID Dictionary**
    unique_teams = sorted(set(df['Home Team'].unique()) | set(df['Away Team'].unique()))  # Get all unique teams
    team_id_dict = {team: idx + 1 for idx, team in enumerate(unique_teams)}  # Assign numeric IDs

    # ✅ Print team IDs for debugging
    print("Generated Team IDs:", team_id_dict)

    # Assign team IDs to DataFrame
    df['home_team_id'] = df['Home Team'].map(team_id_dict)
    df['away_team_id'] = df['Away Team'].map(team_id_dict)

    return df, team_id_dict  # Return both the DataFrame and the dictionary

print("Match data loaded successfully!")









# Function to calculate team statistics
def calculate_team_statistics(df):
    team_names = df['Home Team'].unique()
    home_field_advantage = df['home_goals'].mean() - df['away_goals'].mean()
    team_data = {}

    for team in team_names:
        home_games = df[df['Home Team'] == team]
        away_games = df[df['Away Team'] == team]

        avg_home_goals_for = home_games['home_goals'].mean()
        avg_away_goals_for = away_games['away_goals'].mean()
        avg_home_goals_against = home_games['away_goals'].mean()
        avg_away_goals_against = away_games['home_goals'].mean()

        team_data[team] = {
            'Home Goals For': avg_home_goals_for,
            'Away Goals For': avg_away_goals_for,
            'Home Goals Against': avg_home_goals_against,
            'Away Goals Against': avg_away_goals_against,
            'ATT Rating': (avg_home_goals_for + avg_away_goals_for) / 2,
            'DEF Rating': (avg_home_goals_against + avg_away_goals_against) / 2
        }

    return team_data, home_field_advantage

print("Team statistics calculated!")















# Function to calculate recent form ratings
def calculate_recent_form(df, team_data, recent_matches=20, alpha=0.65):
    recent_form_att = {}
    recent_form_def = {}

    for team in df['Home Team'].unique():
        recent_matches_df = df[(df['Home Team'] == team) | (df['Away Team'] == team)].tail(recent_matches)
        
        home_matches = recent_matches_df[recent_matches_df['Home Team'] == team]
        away_matches = recent_matches_df[recent_matches_df['Away Team'] == team]

        avg_home_att = home_matches['home_goals'].mean()
        avg_away_att = away_matches['away_goals'].mean()
        avg_home_def = home_matches['away_goals'].mean()
        avg_away_def = away_matches['home_goals'].mean()

        recent_form_att[team] = ((1 - alpha) * team_data[team]['ATT Rating']) + (alpha * ((avg_home_att + avg_away_att) / 2))
        recent_form_def[team] = ((1 - alpha) * team_data[team]['DEF Rating']) + (alpha * ((avg_home_def + avg_away_def) / 2))

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
    draw_prob = np.sum(np.diag(result_matrix))          # Diagonal elements

    return result_matrix, home_win_prob, draw_prob, away_win_prob

import os
import matplotlib.pyplot as plt




















# Function to generate a heatmap
def display_heatmap(result_matrix, match_id, home_team, away_team, home_team_id, away_team_id, home_prob, draw_prob, away_prob, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]}, facecolor="#f4f4f9")

    # --- Heatmap (Top) ---
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

    # --- Bar Chart (Bottom) ---
    bar_ax = axes[1]
    bar_ax.set_facecolor('#f4f4f9')  # Background color

    categories = [f"{home_team}", "Draw", f"{away_team}"]
    values = [home_prob * 100, draw_prob * 100, away_prob * 100]

    # **Change from horizontal to vertical bars**
    bars = bar_ax.bar(categories, values, color='#3f007d', alpha=0.9, width=0.6)

    # Title for the bar chart
    bar_ax.set_title("Projected Win %'s:")

    # Add text labels on bars (above each bar)
    for bar in bars:
        height = bar.get_height()
        bar_ax.text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.1f}%", 
                    ha='center', fontsize=10, fontweight='bold')

    # Remove unnecessary spines
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_visible(False)
    bar_ax.spines['bottom'].set_visible(False)
    bar_ax.set_yticks([])

    # Adjust layout
    plt.tight_layout()

    # 📌 **Save the combined figure using Team IDs**
    heatmap_filename = f"{home_team_id}_{away_team_id}_heatmap.png"
    plt.savefig(os.path.join(save_path, heatmap_filename))
    plt.close()

    print(f"✅ Heatmap saved: {heatmap_filename}")



















def generate_all_heatmaps(new_fixture_df, team_stats, recent_form_att, recent_form_def, alpha=0.65, save_path="static/heatmaps/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for _, fixture in new_fixture_df.iterrows():
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        match_id = fixture.get("match_id")
        home_team_id = fixture.get("home_team_id")
        away_team_id = fixture.get("away_team_id")

        if pd.isna(home_team_id) or pd.isna(away_team_id):
            print(f"❌ Skipping match {home_team} vs {away_team}: Missing team IDs")
            continue  # Skip if IDs are missing

        if home_team not in team_stats or away_team not in team_stats:
            print(f"Skipping {home_team} vs {away_team} due to missing data.")
            continue

        # Blend overall ratings with recent form using alpha weighting
        home_att_rating = (1 - alpha) * team_stats[home_team]['ATT Rating'] + alpha * recent_form_att[home_team]
        away_att_rating = (1 - alpha) * team_stats[away_team]['ATT Rating'] + alpha * recent_form_att[away_team]
        home_def_rating = (1 - alpha) * team_stats[home_team]['DEF Rating'] + alpha * recent_form_def[home_team]
        away_def_rating = (1 - alpha) * team_stats[away_team]['DEF Rating'] + alpha * recent_form_def[away_team]

        # Adjusted expected goals (xG)
        home_xg = home_att_rating * away_def_rating
        away_xg = away_att_rating * home_def_rating
        
        result_matrix, home_prob, draw_prob, away_prob = simulate_poisson_distribution(home_xg, away_xg)
        
        # 🔥 Pass `home_team_id` & `away_team_id` to `display_heatmap()`
        display_heatmap(result_matrix, match_id, home_team, away_team, home_team_id, away_team_id, home_prob, draw_prob, away_prob, save_path)





print("Heatmaps generated successfully!")









if __name__ == "__main__":
    print("🔄 Generating heatmaps in advance...")
    fixtures = pd.DataFrame(load_fixtures())
    match_data, team_id_dict = load_match_data()  # ✅ Fix here
    team_stats, _ = calculate_team_statistics(match_data)

    # Calculate recent form
    recent_form_att, recent_form_def = calculate_recent_form(match_data, team_stats, recent_matches=20, alpha=0.65)

    # Generate heatmaps with blended ratings
    generate_all_heatmaps(fixtures, team_stats, recent_form_att, recent_form_def, alpha=0.65)

    print("✅ All heatmaps generated and saved in 'static/heatmaps/'")

