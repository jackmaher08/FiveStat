import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.colors as mcolors

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

print("Fixtures loaded successfully!")

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
def calculate_recent_form(df, team_data, recent_matches=15, alpha=0.65):
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

# create league table
def calculate_league_table(df):
    teams = df['Home Team'].unique()
    table = {team: {'MP': 0, 'W': 0, 'D': 0, 'L': 0, 'G': 0, 'GA': 0, 'PTS': 0} for team in teams}

    for _, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goals = row['home_goals']
        away_goals = row['away_goals']

        # Update Matches Played (MP) and Goals
        table[home_team]['MP'] += 1
        table[away_team]['MP'] += 1
        table[home_team]['G'] += home_goals
        table[away_team]['G'] += away_goals
        table[home_team]['GA'] += away_goals
        table[away_team]['GA'] += home_goals

        # Update Wins, Draws, Losses, Points
        if home_goals > away_goals:
            table[home_team]['W'] += 1
            table[away_team]['L'] += 1
            table[home_team]['PTS'] += 3
        elif home_goals < away_goals:
            table[away_team]['W'] += 1
            table[home_team]['L'] += 1
            table[away_team]['PTS'] += 3
        else:
            table[home_team]['D'] += 1
            table[away_team]['D'] += 1
            table[home_team]['PTS'] += 1
            table[away_team]['PTS'] += 1

    # Convert to sorted list
    table_df = pd.DataFrame.from_dict(table, orient='index').reset_index()
    table_df.rename(columns={'index': 'Team'}, inplace=True)
    table_df.sort_values(by=['PTS', 'G'], ascending=[False, False], inplace=True)
    table_df.insert(0, 'Pos', range(1, len(table_df) + 1))  # Add Position column

    return table_df

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

# Function to generate a heatmap
def display_heatmap(result_matrix, home_team, away_team, home_prob, draw_prob, away_prob, save_path):
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

    # Set y-axis range from 0 to 100
    # bar_ax.set_ylim(0, 100)

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

    # Save the combined figure
    plt.savefig(os.path.join(save_path, f"{home_team}_vs_{away_team}_heatmap.png"))
    plt.close()





def generate_all_heatmaps(fixtures, team_stats, recent_form_att, recent_form_def, alpha=0.45, save_path="static/heatmaps/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fixture in fixtures:
        home_team = fixture['home_team']
        away_team = fixture['away_team']

        if home_team not in team_stats or away_team not in team_stats:
            print(f"Skipping {home_team} vs {away_team} due to missing data.")
            continue

        # Blend overall ratings with recent form using the same alpha weighting
        home_att_rating = (1 - alpha) * team_stats[home_team]['ATT Rating'] + alpha * recent_form_att[home_team]
        away_att_rating = (1 - alpha) * team_stats[away_team]['ATT Rating'] + alpha * recent_form_att[away_team]
        home_def_rating = (1 - alpha) * team_stats[home_team]['DEF Rating'] + alpha * recent_form_def[home_team]
        away_def_rating = (1 - alpha) * team_stats[away_team]['DEF Rating'] + alpha * recent_form_def[away_team]

        # Adjusted expected goals (xG)
        home_xg = home_att_rating * away_def_rating
        away_xg = away_att_rating * home_def_rating
        
        result_matrix, home_prob, draw_prob, away_prob = simulate_poisson_distribution(home_xg, away_xg)
        
        display_heatmap(result_matrix, home_team, away_team, home_prob, draw_prob, away_prob, save_path)



print("Heatmaps generated successfully!")
print("Data loading complete!")
print(r"""
 ______      ___   ____  _____  ________  
|_   _ `.  .'   `.|_   \|_   _||_   __  | 
  | | `. \/  .-.  \ |   \ | |    | |_ \_| 
  | |  | || |   | | | |\ \| |    |  _| _  
 _| |_.' /\  `-'  /_| |_\   |_  _| |__/ | 
|______.'  `.___.'|_____|\____||________| 
                                          
""")

if __name__ == "__main__":
    print("🔄 Generating heatmaps in advance...")
    fixtures = load_fixtures().to_dict(orient="records")
    match_data = load_match_data()
    team_stats, _ = calculate_team_statistics(match_data)
    
    # Calculate recent form
    recent_form_att, recent_form_def = calculate_recent_form(match_data, team_stats, recent_matches=15, alpha=0.65)

    # Generate heatmaps with blended ratings
    generate_all_heatmaps(fixtures, team_stats, recent_form_att, recent_form_def, alpha=0.65)

    print("✅ All heatmaps generated and saved in 'static/heatmaps/'")

