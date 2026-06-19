import os
import json
import requests
import csv
import io
from datetime import datetime, timezone

DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")
GAA_ELO_PATH      = os.path.join(DATA_DIR, "gaa_elo.json")
GAA_RESULTS_PATH  = os.path.join(DATA_DIR, "gaa_results.json")
GAA_FIXTURES_PATH = os.path.join(DATA_DIR, "gaa_fixtures.json")

os.makedirs(DATA_DIR, exist_ok=True)

SHEET_ID   = "1y5VpAqogmLXSVOBYKaGLKX2YOaAOZOJK2SYIN2SpgrA"
ELO_GID    = 887483570   # "Elo values" tab
SEASON_GID = 1607142428  # "2026" tab

COUNTIES = {
    'Dublin', 'Kerry', 'Galway', 'Mayo', 'Tyrone', 'Donegal', 'Armagh',
    'Monaghan', 'Derry', 'Cork', 'Roscommon', 'Kildare', 'Meath', 'Louth',
    'Westmeath', 'Cavan', 'Fermanagh', 'Antrim', 'Down', 'Sligo', 'Leitrim',
    'Offaly', 'Laois', 'Clare', 'Tipperary', 'Limerick', 'Waterford',
    'Wicklow', 'Longford', 'Carlow', 'Wexford', 'London', 'New York'
}

# Column indices in the 2026 tab
COL_DATE   = 0
COL_GRADE  = 1
COL_ROUND  = 2
COL_TEAM1  = 3
COL_SC1    = 7
COL_TEAM2  = 9
COL_SC2    = 13
COL_HOME   = 15
COL_WEIGHT = 17


def fetch_csv(gid):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return list(csv.reader(io.StringIO(r.text)))


def fetch_elo_ratings():
    print("Fetching GAA ELO ratings from Google Sheets...")
    try:
        rows = fetch_csv(ELO_GID)
        ratings = {}
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            county = row[0].strip()
            if len(row) < 2 or not row[1].strip():
                continue
            try:
                ratings[county] = int(float(row[1].strip()))
            except ValueError:
                continue
        print(f"  {len(ratings)} county ratings fetched")
        return ratings
    except Exception as e:
        print(f"  Failed to fetch ELO ratings: {e}")
        return None


def safe_int(val):
    if not val or not val.strip():
        return None
    try:
        return int(float(val.strip()))
    except ValueError:
        return None


def fetch_season_data():
    """Pull all 2026 matches - both played results and upcoming fixtures."""
    print("Fetching 2026 season data from Google Sheets...")
    try:
        rows = fetch_csv(SEASON_GID)
        if not rows:
            return [], []

        headers = [h.strip() for h in rows[0]]
        print(f"  Headers: {headers[:16]}")

        results  = []
        fixtures = []

        for row in rows[1:]:
            if not row or len(row) <= COL_HOME:
                continue

            date_str = row[COL_DATE].strip()
            grade    = row[COL_GRADE].strip()
            rnd      = row[COL_ROUND].strip()
            team1    = row[COL_TEAM1].strip()
            team2    = row[COL_TEAM2].strip()
            sc1      = safe_int(row[COL_SC1])
            sc2      = safe_int(row[COL_SC2])
            home     = row[COL_HOME].strip()
            weight   = safe_int(row[COL_WEIGHT]) if len(row) > COL_WEIGHT else None

            if team1 not in COUNTIES or team2 not in COUNTIES:
                continue

            entry = {
                'date':   date_str,
                'grade':  grade,
                'round':  rnd,
                'team1':  team1,
                'team2':  team2,
                'home':   home == 'Y',
                'weight': weight if weight else 45,
            }

            # A match is "played" if both scores present and non-zero
            if sc1 is not None and sc2 is not None and (sc1 > 0 or sc2 > 0):
                entry['sc1'] = sc1
                entry['sc2'] = sc2
                entry['winner'] = team1 if sc1 > sc2 else (team2 if sc2 > sc1 else None)
                results.append(entry)
            else:
                fixtures.append(entry)

        print(f"  {len(results)} played, {len(fixtures)} upcoming")
        return results, fixtures
    except Exception as e:
        print(f"  Failed to fetch season data: {e}")
        return [], []


def main():
    print(f"\n{'='*50}")
    print(f"  GAA Scraper - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}\n")

    ratings = fetch_elo_ratings()
    if not ratings:
        if os.path.exists(GAA_ELO_PATH):
            with open(GAA_ELO_PATH) as f:
                ratings = json.load(f).get('ratings', {})
            print("  Using cached ELO file")
        else:
            print("  No ratings available - aborting")
            return

    results, fixtures = fetch_season_data()
    now = datetime.now(timezone.utc).isoformat()

    with open(GAA_ELO_PATH, 'w') as f:
        json.dump({'scraped_at': now, 'ratings': ratings}, f, indent=2)
    print(f"  ELO saved -> {GAA_ELO_PATH}")

    if results:
        with open(GAA_RESULTS_PATH, 'w') as f:
            json.dump({'scraped_at': now, 'results': results}, f, indent=2)
        print(f"  {len(results)} results saved -> {GAA_RESULTS_PATH}")

    with open(GAA_FIXTURES_PATH, 'w') as f:
        json.dump({'scraped_at': now, 'fixtures': fixtures}, f, indent=2)
    print(f"  {len(fixtures)} fixtures saved -> {GAA_FIXTURES_PATH}\n")

    print("Current GAA Football ELO Rankings:")
    sorted_ratings = sorted(
        [(k, v) for k, v in ratings.items() if k in COUNTIES],
        key=lambda x: x[1], reverse=True
    )
    for rank, (county, elo) in enumerate(sorted_ratings[:16], 1):
        print(f"  {rank:2d}. {county:<15} {elo}")

    print(f"\nGAA scraper complete\n")


if __name__ == "__main__":
    main()
