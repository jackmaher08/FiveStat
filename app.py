import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template
from scipy.stats import poisson

app = Flask(__name__)

# Load your fixtures data from the CSV
fixtures_by_gw = pd.read_csv("https://fixturedownload.com/download/epl-2024-GMTStandardTime.csv")
fixtures_by_gw = fixtures_by_gw[fixtures_by_gw['Round Number'] == 25]
fixtures_by_gw = fixtures_by_gw[['Home Team', 'Away Team', 'Date']]

# Convert it into a list of dictionaries
fixtures = []
for _, row in fixtures_by_gw.iterrows():
    fixtures.append({
        "home_team": row["Home Team"],
        "away_team": row["Away Team"],
        "date": row["Date"]
    })

import pandas as pd

# first download the match data. We'll use fixturedownload.com for data since 2016

frames = []

for year in range(2016,2025):
    url="https://fixturedownload.com/download/epl-%s-GMTStandardTime.csv" % year
    #print(url)
    frame = pd.read_csv(url)
    frame['Season']=year
    frames.append(frame)
df = pd.concat(frames)

df = df[pd.notnull(df.Result)]

# Split the 'Score' column into 'Home Goals' and 'Away Goals' columns
df[['home_goals', 'away_goals']] = df['Result'].str.split(' - ', expand=True).astype(float)

# Determine the match outcome from the home and away goals

def get_outcome(row):
    if row['home_goals'] > row['away_goals']:
        return 'home_win'
    elif row['home_goals'] < row['away_goals']:
        return 'away_win'
    else:
        return 'draw'

# Create an outcome column following the above to record match winner
df['result'] = df.apply(get_outcome, axis=1)

# Extract unique team names from df
team_names = df['Home Team'].unique()

#Now I want to work out the home field advantage expressed in goals

# Calc the average home and away goals scored per game
avg_home_goals_for = df['home_goals'].mean()
avg_away_goals_for = df['away_goals'].mean()

# Calc home-field advantage
home_field_advantage = avg_home_goals_for - avg_away_goals_for

# Dictionary to store data for each team
team_data = {}

# Loop through each team
for team in team_names:
    # Filter rows for the teams as Home Team
    home_team_rows = df[df['Home Team'] == team]

    # Calculate average goals for and against as Home Team
    avg_home_goals_for = home_team_rows['home_goals'].mean()
    avg_away_goals_against = home_team_rows['away_goals'].mean()

    # Filter rows for the specific team as Away Team
    away_team_rows = df[df['Away Team'] == team]

    # Calculate average goals for and against as Away Team
    avg_away_goals_for = away_team_rows['away_goals'].mean()
    avg_home_goals_against = away_team_rows['home_goals'].mean()

    # Calculate average goals for and against for each team
    ATT_rating = (avg_home_goals_for + avg_away_goals_for) / 2
    DEF_rating = (avg_home_goals_against + avg_away_goals_against) / 2

    # Store calculated data in the dictionary
    team_data[team] = {
        'Home Goals For': avg_home_goals_for,
        'Away Goals For': avg_away_goals_for,
        'Home Goals Against': avg_home_goals_against,
        'Away Goals Against': avg_away_goals_against,
        'ATT Rating': ATT_rating,
        'DEF Rating': DEF_rating
    }

#Going to now use our df data to predict the results for the next round of fixtures (gw 11 of the 24/25 season at time of writing)
# to start we gather the next round of fixtures (gw 23)

fixtures_by_gw=pd.read_csv("https://fixturedownload.com/download/epl-2024-GMTStandardTime.csv")
fixtures_by_gw = fixtures_by_gw[fixtures_by_gw['Round Number'] == 25]

# now refine to just the home and away team names

fixtures_by_gw=fixtures_by_gw[['Home Team', 'Away Team', 'Date']]

#Now time to run the match simulator for each fixture in this game week

# import relevant packages
# aware I've repeated some of these but I like to import them still to know what is needed

import pandas as pd
import numpy as np
from scipy.stats import poisson
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# Define the number of goals for the Poisson matrix - 7 will display 0 - 6 goals
number_of_goals = 6

# I want to account for a teams recent form when calculating XG

# Define weight factor for recent form influence
alpha = 0.35  # Adjust this value to control how much recent form influences xG

# Create a dictionary to store recent ATT & DEF Ratings
recent_form_att = {}
recent_form_def = {}

# Loop through each team
for team in team_names:
    # Get last 15 matches for a team
    recent_matches = df[(df['Home Team'] == team) | (df['Away Team'] == team)].tail(10)
    
    # Calculate ATT & DEF Rating for recent matches
    home_matches = recent_matches[recent_matches['Home Team'] == team]
    away_matches = recent_matches[recent_matches['Away Team'] == team]

    # Average ATT & DEF Rating in recent games
    avg_home_att_recent = home_matches['home_goals'].mean()  # Goals scored at home
    avg_away_att_recent = away_matches['away_goals'].mean()  # Goals scored away
    avg_home_def_recent = home_matches['away_goals'].mean()  # Goals conceded at home
    avg_away_def_recent = away_matches['home_goals'].mean()  # Goals conceded away
    
    # Compute recent ATT & DEF rating (if no matches, default to overall ATT)
    recent_att_rating = (avg_home_att_recent + avg_away_att_recent) / 2 if not np.isnan(avg_home_att_recent) else team_data[team]['ATT Rating']
    recent_def_rating = (avg_home_def_recent + avg_away_def_recent) / 2 if not np.isnan(avg_home_def_recent) else team_data[team]['DEF Rating']

    
    # Store in dictionary
    recent_form_att[team] = recent_att_rating
    recent_form_def[team] = recent_def_rating




# Define functions

# create 2 poisson dist for the number of goals each team will score
# the XG is the value that each team needs to score for their OFF rating to remain exactly the same post fixture
def simulate_poisson_distribution(home_team_xg, away_team_xg, home_team_defense, away_team_defense):
    score_matrix = np.zeros((number_of_goals, number_of_goals))
    
    for home_goals in range(number_of_goals):
        for away_goals in range(number_of_goals):
            # Adjusted for both offensive and defensive strengths
            home_prob = poisson.pmf(home_goals, home_team_xg * away_team_defense)  # Home team offense adjusted by away team defense
            away_prob = poisson.pmf(away_goals, away_team_xg * home_team_defense)  # Away team offense adjusted by home team defense
            
            score_matrix[home_goals][away_goals] = home_prob * away_prob
            
    return score_matrix

print("This is a test print.")
#Display the home and away goals on a heatmap grid
import os

# Display the home and away goals on a heatmap grid and save the image
def display_heatmap(result_matrix, home_team, away_team, file_path):
    # Create a DataFrame from the result_matrix for plotting
    result_df = pd.DataFrame(result_matrix, index=range(result_matrix.shape[0]), columns=range(result_matrix.shape[1]))
    
    # Set up the figure with the desired size
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the heatmap
    heatmap = ax.imshow(result_matrix, cmap="Purples", origin='upper')

    # Add titles and labels
    ax.set_title(f"{home_team} vs {away_team} Score Prediction Heatmap", fontsize=15, pad=20)
    ax.set_xlabel(f"{away_team} Goals", labelpad=20)
    ax.set_ylabel(f"{home_team} Goals", labelpad=20)

    # Move the away team goals (x-axis) to the top
    ax.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax.set_xticks(range(result_matrix.shape[1]))  # Set the tick positions for the away team goals
    ax.set_xticklabels(range(result_matrix.shape[1]))  # Set the tick labels for the away team goals
    ax.xaxis.set_ticks_position('top')  # Ensure the x-axis ticks appear at the top
    
    # Remove the little lines next to the numbers on both axes
    ax.tick_params(axis='x', length=0)  # Remove x-axis tick marks
    ax.tick_params(axis='y', length=0)  # Remove y-axis tick marks
    
    # Remove the outer spines (lines around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add percentage values in each cell
    for i in range(result_matrix.shape[0]):
        for j in range(result_matrix.shape[1]):
            value = result_matrix[i, j]
            percentage = f"{value * 100:.0f}%"  # Convert to percentage & apply <1% formatting
            ax.text(j, i, percentage, ha='center', va='center', color='black', fontsize=8)

    # Ensure the directory exists
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the heatmap to the specified file path
    file_name = f"{home_team}_vs_{away_team}_heatmap.png"
    full_file_path = os.path.join(file_path, file_name)
    plt.savefig(full_file_path, bbox_inches='tight')

    # Close the plot to avoid display, since we're saving it
    plt.close()

    print(f"Heatmap saved to {full_file_path}")


    


    


def calculate_match_result_probabilities(result_matrix):
    home_win_prob = np.sum(np.tril(result_matrix, -1))
    away_win_prob = np.sum(np.triu(result_matrix, 1))
    draw_prob = np.sum(np.diag(result_matrix))
    return home_win_prob, away_win_prob, draw_prob


# Define the simulate_poisson_distribution function to accept adjusted xG values
def simulate_poisson_distribution(home_team_xg, away_team_xg, number_of_goals=number_of_goals):
    score_matrix = np.zeros((number_of_goals, number_of_goals))
    for home_goals in range(number_of_goals):
        for away_goals in range(number_of_goals):
            home_prob = poisson.pmf(home_goals, home_team_xg)
            away_prob = poisson.pmf(away_goals, away_team_xg)
            score_matrix[home_goals][away_goals] = home_prob * away_prob
    return score_matrix



# Simulation loop for each fixture
for index, row in fixtures_by_gw.iterrows():
    home_team = row['Home Team']
    away_team = row['Away Team']
    date = row['Date']
    
    #I want the XG to be the number of goals that a team needs to score to keep their ATT rating the exact same
    
    # Retrieve the ATT and DEF Ratings for both teams
    home_team_att = team_data.get(home_team, {}).get('ATT Rating', np.nan)
    away_team_att = team_data.get(away_team, {}).get('ATT Rating', np.nan)
    home_team_def = team_data.get(home_team, {}).get('DEF Rating', np.nan)
    away_team_def = team_data.get(away_team, {}).get('DEF Rating', np.nan)
    
    # Ensure ATT and DEF Ratings are available
    if np.isnan(home_team_att) or np.isnan(away_team_att) or np.isnan(home_team_def) or np.isnan(away_team_def):
        print(f"Missing ATT or DEF Rating data for {home_team} vs {away_team}. Skipping this match.")
        continue

    # Retrieve recent ATT and DEF Ratings
    recent_home_att = recent_form_att.get(home_team, home_team_att)
    recent_away_att = recent_form_att.get(away_team, away_team_att)
    recent_home_def = recent_form_def.get(home_team, home_team_def)
    recent_away_def = recent_form_def.get(away_team, away_team_def)

    # Apply weighted moving average formula for ATT and DEF Ratings
    adjusted_home_att = ((1 - alpha) * home_team_att) + (alpha * recent_home_att)
    adjusted_away_att = ((1 - alpha) * away_team_att) + (alpha * recent_away_att)
    adjusted_home_def = ((1 - alpha) * home_team_def) + (alpha * recent_home_def)
    adjusted_away_def = ((1 - alpha) * away_team_def) + (alpha * recent_away_def)

    # Calculate expected goals (xG) with both ATT & DEF recent form adjustments
    home_team_xg = (adjusted_home_att * adjusted_away_def) + home_field_advantage
    away_team_xg = (adjusted_away_att * adjusted_home_def)
    
    # Run the Poisson simulation
    result_matrix = simulate_poisson_distribution(home_team_xg, away_team_xg)
    
    # Display the result matrix as a heatmap for the current match
    # Save the heatmap to the './static/heatmaps' directory
    file_path = 'C:/Users/jmaher/Documents/flask_heatmap_app/static/heatmaps/' #
    display_heatmap(result_matrix, home_team, away_team, file_path)
    


@app.route("/")
def home():
    return render_template("index.html", fixtures=fixtures)

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/')
def index():
    # Assuming you pass fixtures_by_gw to the template as 'fixtures'
    return render_template('index.html', fixtures=fixtures_by_gw)