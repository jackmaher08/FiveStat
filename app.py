import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template
from scipy.stats import poisson

# Flask app initialization
app = Flask(__name__)

# Function to load fixture data
def load_fixtures():
    url = "https://fixturedownload.com/download/epl-2024-GMTStandardTime.csv"
    fixtures_df = pd.read_csv(url)

    # Print actual column names for debugging
    print("CSV Columns:", fixtures_df.columns)

    # Rename columns to match expected format (adjust if needed)
    fixtures_df = fixtures_df.rename(columns={
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Date": "date"
    })

    # Ensure the correct columns exist
    if not {"home_team", "away_team", "date"}.issubset(set(fixtures_df.columns)):
        raise ValueError(f"Missing expected columns! Found: {fixtures_df.columns}")

    # Find the first round where at least 10 fixtures have no result
    round_counts = fixtures_df[fixtures_df['Result'].isna()].groupby('Round Number').size()
    round_number = round_counts[round_counts >= 10].index.min()

    print(f"The next gameweek is GW {round_number}")

    # Filter for the selected round
    fixtures_df = fixtures_df[fixtures_df['Round Number'] == round_number]

    return fixtures_df[['home_team', 'away_team', 'date']]





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
    
    # Split the 'Result' column into 'home_goals' and 'away_goals'
    df[['home_goals', 'away_goals']] = df['Result'].str.split(' - ', expand=True).astype(float)
    
    # Determine match outcomes
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

max_goals = 12

# Function to simulate a match using Poisson distribution
# Simulate match using Poisson distribution
def simulate_poisson_distribution(home_xg, away_xg, max_goals=max_goals):
    result_matrix = np.zeros((max_goals, max_goals))

    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            home_prob = poisson.pmf(home_goals, home_xg)
            away_prob = poisson.pmf(away_goals, away_xg)
            result_matrix[home_goals, away_goals] = home_prob * away_prob

    # Normalize to ensure total probability sums to 1
    result_matrix /= np.sum(result_matrix)

    return result_matrix


# Function to generate a heatmap
# Display heatmap (0-5 goals)
def display_heatmap(result_matrix, home_team, away_team, save_path):
    display_matrix = result_matrix[:6, :6]  # Extract only 0-5 goals

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#f4f4f9")
    ax.set_facecolor("#f4f4f9")
    ax.imshow(display_matrix, cmap="Purples", origin='upper')

    ax.set_xlabel(f"{away_team} Goals", labelpad=20)
    ax.set_ylabel(f"{home_team} Goals", labelpad=20)
    ax.xaxis.set_label_position('top')
    ax.set_xticks(range(6))
    ax.set_xticklabels(range(6))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
        # Remove the outer spines (lines around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{display_matrix[i, j] * 100:.1f}%", ha='center', va='center', color='black', fontsize=8)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = f"{home_team}_vs_{away_team}_heatmap.png"
    plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight', facecolor=fig.get_facecolor())  
    plt.close()


# Main simulation function
def run_simulation(fixtures, team_data, recent_form_att, recent_form_def, home_field_advantage):
    save_path = 'static/heatmaps/'
    for _, row in fixtures.iterrows():
        home_team, away_team = row['home_team'], row['away_team']
        
        if home_team not in team_data or away_team not in team_data:
            continue

        home_att, away_att = team_data[home_team]['ATT Rating'], team_data[away_team]['ATT Rating']
        home_def, away_def = team_data[home_team]['DEF Rating'], team_data[away_team]['DEF Rating']

        recent_home_att = recent_form_att.get(home_team, home_att)
        recent_away_att = recent_form_att.get(away_team, away_att)
        recent_home_def = recent_form_def.get(home_team, home_def)
        recent_away_def = recent_form_def.get(away_team, away_def)

        adj_home_att = (0.65 * home_att) + (0.35 * recent_home_att)
        adj_away_att = (0.65 * away_att) + (0.35 * recent_away_att)
        adj_home_def = (0.65 * home_def) + (0.35 * recent_home_def)
        adj_away_def = (0.65 * away_def) + (0.35 * recent_away_def)

        home_xg = (adj_home_att * adj_away_def) + home_field_advantage
        away_xg = (adj_away_att * adj_home_def)

        result_matrix = simulate_poisson_distribution(home_xg, away_xg)
        display_heatmap(result_matrix, home_team, away_team, save_path)

        print(f'Checking {home_team} vs {away_team}')

        # Sum of all values (should be close to 1)
        total_probability = np.sum(result_matrix)
        print("Total Probability Sum:", total_probability)

        # Probability of a draw (diagonal elements)
        draw_probability = np.sum([result_matrix[i, i] for i in range(max_goals)])
        print("Draw Probability:", draw_probability)

        # Probability of an away win (upper triangle)
        away_win_probability = np.sum([result_matrix[i, j] for i in range(max_goals) for j in range(i + 1, max_goals)])
        print("Away Win Probability:", away_win_probability)

        # Probability of a home win (lower triangle)
        home_win_probability = np.sum([result_matrix[i, j] for i in range(1, max_goals) for j in range(i)])
        print("Home Win Probability:", home_win_probability)

        # Sanity check (should be close to 1)
        print("Sanity Check:", draw_probability + home_win_probability + away_win_probability)

        print("------------------")

# Load data and run simulations
fixtures = load_fixtures()
df = load_match_data()
team_data, home_field_advantage = calculate_team_statistics(df)
recent_form_att, recent_form_def = calculate_recent_form(df, team_data)
run_simulation(fixtures, team_data, recent_form_att, recent_form_def, home_field_advantage)

@app.route("/")
def home():
    return render_template("index.html", fixtures=fixtures.to_dict(orient="records"))



if __name__ == "__main__":
    app.run(debug=True)
