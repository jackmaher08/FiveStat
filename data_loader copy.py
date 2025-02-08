import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Function to load fixture data
def load_fixtures():
    url = "https://fixturedownload.com/download/epl-2024-GMTStandardTime.csv"
    fixtures_df = pd.read_csv(url)

    # Print actual column names for debugging
    print("CSV Columns:", fixtures_df.columns)

    # Rename columns to match expected format
    fixtures_df = fixtures_df.rename(columns={
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Date": "date"
    })

    # Ensure correct columns exist
    if not {"home_team", "away_team", "date"}.issubset(set(fixtures_df.columns)):
        raise ValueError(f"Missing expected columns! Found: {fixtures_df.columns}")

    # Find the next gameweek
    round_counts = fixtures_df[fixtures_df['Result'].isna()].groupby('Round Number').size()
    round_number = round_counts[round_counts >= 10].index.min()

    print(f"The next gameweek is GW {round_number}")

    return fixtures_df[fixtures_df['Round Number'] == round_number][['home_team', 'away_team', 'date']]

# Function to load historical match data
def load_match_data(start_year=2016, end_year=2024):
    frames = []
    for year in range(start_year, end_year + 1):
        url = f"https://fixturedownload.com/download/epl-{year}-GMTStandardTime.csv"
        frame = pd.read_csv(url)
        frame['Season'] = year
        frames.append(frame)
    df = pd.concat(frames)
    df = df[pd.notnull(df.Result)]
    
    # Process result column
    df[['home_goals', 'away_goals']] = df['Result'].str.split(' - ', expand=True).astype(float)
    df['result'] = df.apply(lambda row: 'home_win' if row['home_goals'] > row['away_goals'] 
                            else 'away_win' if row['home_goals'] < row['away_goals'] else 'draw', axis=1)
    return df

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

# Function to calculate recent form ratings
def calculate_recent_form(df, team_data, recent_matches=10, alpha=0.35):
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

def generate_all_heatmaps(fixtures, team_stats, save_path="static/heatmaps/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fixture in fixtures:
        home_team = fixture['home_team']
        away_team = fixture['away_team']

            # Check if both teams exist in team_stats
        if home_team not in team_stats or away_team not in team_stats:
            print(f"Skipping {home_team} vs {away_team} due to missing data.")
            continue  # Skips affected fixture
            
        home_xg = team_stats[home_team]['ATT Rating'] * team_stats[away_team]['DEF Rating']
        away_xg = team_stats[away_team]['ATT Rating'] * team_stats[home_team]['DEF Rating']
        
        result_matrix = simulate_poisson_distribution(home_xg, away_xg)
        display_heatmap(result_matrix, home_team, away_team, save_path)

    print("Heatmaps generated successfully!")
