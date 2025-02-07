import pandas as pd

def get_next_gameweek():
    """Retrieve the next gameweek with at least 10 unplayed matches."""
    url = "https://fixturedownload.com/download/epl-2024-GMTStandardTime.csv"
    fixtures = pd.read_csv(url)

    # Find first gameweek with 10+ unplayed fixtures
    gameweek_counts = fixtures[fixtures['Result'].isna()].groupby('Round Number').size()
    first_unplayed_gameweek = gameweek_counts[gameweek_counts >= 10].index.min()

    if pd.isna(first_unplayed_gameweek):
        return None, None

    fixtures_by_gw = fixtures[fixtures['Round Number'] == first_unplayed_gameweek]
    return first_unplayed_gameweek, fixtures_by_gw.to_dict(orient='records')

def load_past_results():
    """Load past results from 2016 to 2025."""
    frames = []
    for year in range(2016, 2025):
        url = f"https://fixturedownload.com/download/epl-{year}-GMTStandardTime.csv"
        frame = pd.read_csv(url)
        frame['Season'] = year
        frames.append(frame)

    df = pd.concat(frames)
    df = df[df['Result'].notna()]
    
    # Split score into home and away goals
    df[['home_goals', 'away_goals']] = df['Result'].str.split(' - ', expand=True).astype(float)

    return df
