import pandas as pd
import numpy as np

def get_outcome(row):
    """Determine match outcome from goals scored."""
    if row['home_goals'] > row['away_goals']:
        return 'home_win'
    elif row['home_goals'] < row['away_goals']:
        return 'away_win'
    else:
        return 'draw'

def calculate_team_ratings(df):
    """Calculate ATT & DEF ratings for each team."""
    df['result'] = df.apply(get_outcome, axis=1)
    team_names = df['Home Team'].unique()

    # Calculate home-field advantage
    avg_home_goals_for = df['home_goals'].mean()
    avg_away_goals_for = df['away_goals'].mean()
    home_field_advantage = avg_home_goals_for - avg_away_goals_for

    team_data = {}
    for team in team_names:
        home_games = df[df['Home Team'] == team]
        away_games = df[df['Away Team'] == team]

        avg_home_goals_for = home_games['home_goals'].mean()
        avg_away_goals_against = home_games['away_goals'].mean()
        avg_away_goals_for = away_games['away_goals'].mean()
        avg_home_goals_against = away_games['home_goals'].mean()

        ATT_rating = (avg_home_goals_for + avg_away_goals_for) / 2
        DEF_rating = (avg_home_goals_against + avg_away_goals_against) / 2

        team_data[team] = {
            'Home Goals For': avg_home_goals_for,
            'Away Goals For': avg_away_goals_for,
            'Home Goals Against': avg_home_goals_against,
            'Away Goals Against': avg_away_goals_against,
            'ATT Rating': ATT_rating,
            'DEF Rating': DEF_rating
        }

    return team_data, home_field_advantage
