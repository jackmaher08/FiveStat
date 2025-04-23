import sys
import os
import pandas as pd
import requests
import json
import re
from bs4 import BeautifulSoup

def load_existing_data():
    path = "data/tables/fixture_data.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def fetch_understat_fixtures():
    url = "https://understat.com/league/EPL/2024"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    match = re.search(r"var datesData\s*=\s*JSON.parse\('(.*)'\)", str(soup))

    if not match:
        return pd.DataFrame()

    data = json.loads(match.group(1).encode('utf8').decode('unicode_escape'))
    rows = []
    for entry in data:
        rows.append({
            "id": entry["id"],
            "home_team": entry["h"]["title"],
            "away_team": entry["a"]["title"],
            "isResult": entry["isResult"]
        })
    return pd.DataFrame(rows)

def new_data_available(existing_df, new_understat_df):
    if existing_df.empty:
        print("⚠️ No existing data found. Triggering full update.")
        return True

    merged = pd.merge(existing_df, new_understat_df, on=["home_team", "away_team"], how="outer", suffixes=('_old', '_new'))

    new_matches = merged[merged["id_new"].isna()].shape[0]
    result_updates = merged[
        (merged["isResult_old"] == False) & 
        (merged["isResult_new"] == True)
    ].shape[0]

    print(f"🔍 New matches found: {new_matches}")
    print(f"🔁 Result status updates: {result_updates}")

    return new_matches > 0 or result_updates > 0

def main():
    print("🧪 Script has started running")
    print("🔍 Checking for new data...")

    existing_df = load_existing_data()
    understat_df = fetch_understat_fixtures()

    if new_data_available(existing_df, understat_df):
        print("✅ New data detected — running update pipeline.")

        # Run Python scripts
        os.system(f"{sys.executable} data_scraper_script.py")
        os.system(f"{sys.executable} data_loader.py")
        os.system(f"{sys.executable} generate_radars.py")
        os.system(f"{sys.executable} generate_shotmaps.py")

        # Git commit and push
        os.system("git add .")
        os.system('git commit -m "Automated visual + data update"')
        os.system("git push origin main")
    else:
        print("⏳ No new data — skipping update.")

if __name__ == "__main__":
    main()
