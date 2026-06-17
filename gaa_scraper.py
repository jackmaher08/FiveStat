import os
import json
import re
import requests
import csv
import io
from datetime import datetime, timezone

DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
GAA_ELO_PATH    = os.path.join(DATA_DIR, "gaa_elo.json")
GAA_RESULTS_PATH = os.path.join(DATA_DIR, "gaa_results.json")

os.makedirs(DATA_DIR, exist_ok=True)

SHEET_ID    = "1y5VpAqogmLXSVOBYKaGLKX2YOaAOZOJK2SYIN2SpgrA"
ELO_GID     = 887483570   # "Elo values" tab
SEASON_GID  = 1607142428  # "2026" tab - match results

HOME_ADV    = 130   # from Rules tab
K_AI        = 80
K_PROVINCE  = 70
K_QUALIFIER = 60
K_LEAGUE    = 45
K_TAILTEANN = 80
K_OTHER     = 15
D_FACTOR    = 500   # predictive multiplier
MAX_MARGIN  = 2.0   # margin cap

COUNTIES = {
    'Dublin', 'Kerry', 'Galway', 'Mayo', 'Tyrone', 'Donegal', 'Armagh',
    'Monaghan', 'Derry', 'Cork', 'Roscommon', 'Kildare', 'Meath', 'Louth',
    'Westmeath', 'Cavan', 'Fermanagh', 'Antrim', 'Down', 'Sligo', 'Leitrim',
    'Offaly', 'Laois', 'Clare', 'Tipperary', 'Limerick', 'Waterford',
    'Wicklow', 'Longford', 'Carlow', 'Wexford', 'London', 'New York'
}

COMP_WEIGHTS = {
    'All-Ireland SFC':          K_AI,
    'All-Ireland SFC Round':    K_AI,
    'All-Ireland SFC round':    K_AI,
    'All-Ireland SFC Qualifier':K_QUALIFIER,
    'All-Ireland SFC qualifier':K_QUALIFIER,
    'Ulster SFC':               K_PROVINCE,
    'Munster SFC':              K_PROVINCE,
    'Leinster SFC':             K_PROVINCE,
    'Connacht SFC':             K_PROVINCE,
    'Tailteann Cup':            K_TAILTEANN,
    'Allianz Football':         K_LEAGUE,
    'National Football League': K_LEAGUE,
    'Lidl National Football':   K_LEAGUE,
}


def fetch_elo_ratings():
    """Pull current ELO ratings from the Google Sheet Elo values tab."""
    print("📊 Fetching GAA ELO ratings from Google Sheets...")
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={ELO_GID}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        reader = csv.reader(io.StringIO(r.text))
        rows = list(reader)

        # Row 0 is header: ['', 'Today', 'end of 2025', ...]
        # Subsequent rows: ['Kerry', '1974', '2100', ...]
        ratings = {}
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            county = row[0].strip()
            if len(row) < 2 or not row[1].strip():
                continue
            try:
                elo = int(float(row[1].strip()))
                ratings[county] = elo
            except ValueError:
                continue

        print(f"  ✅ {len(ratings)} county ratings fetched")
        return ratings
    except Exception as e:
        print(f"  ⚠️  Failed to fetch ELO ratings: {e}")
        return None


def fetch_season_results():
    """Pull 2026 match results from the season tab."""
    print("📅 Fetching 2026 match results from Google Sheets...")
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SEASON_GID}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        reader = csv.reader(io.StringIO(r.text))
        rows = list(reader)

        # Expected columns from screenshot:
        # Date, Grade, Team1, Elo, G, P, Sc, Team2, Elo, G, P, Sc, Home?, Margin*, Weight, ...
        # Row 0 = headers
        if not rows:
            return []

        headers = [h.strip() for h in rows[0]]
        print(f"  Headers: {headers[:16]}")

        results = []
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            try:
                date_str = row[0].strip()
                grade    = row[1].strip() if len(row) > 1 else ''
                team1    = row[3].strip() if len(row) > 3 else ''
                sc1_raw  = row[7].strip() if len(row) > 7 else ''
                team2    = row[9].strip() if len(row) > 9 else ''
                sc2_raw  = row[13].strip() if len(row) > 13 else ''
                home     = row[15].strip() if len(row) > 15 else ''
                margin   = row[16].strip() if len(row) > 16 else ''
                weight   = row[17].strip() if len(row) > 17 else ''

                if not team1 or not team2 or not sc1_raw or not sc2_raw:
                    continue
                if team1 not in COUNTIES or team2 not in COUNTIES:
                    continue

                sc1 = int(float(sc1_raw)) if sc1_raw else None
                sc2 = int(float(sc2_raw)) if sc2_raw else None

                if sc1 is None or sc2 is None:
                    continue

                results.append({
                    'date':   date_str,
                    'grade':  grade,
                    'team1':  team1,
                    'team2':  team2,
                    'sc1':    sc1,
                    'sc2':    sc2,
                    'home':   home == 'Y',
                    'weight': int(float(weight)) if weight else K_LEAGUE,
                })
            except (ValueError, IndexError):
                continue

        print(f"  ✅ {len(results)} match results fetched")
        return results
    except Exception as e:
        print(f"  ⚠️  Failed to fetch season results: {e}")
        return []


def update_elo_from_results(ratings, results):
    """Apply ELO updates from match results."""
    updated = {k: float(v) for k, v in ratings.items()}
    applied = 0

    for m in results:
        t1, t2 = m['team1'], m['team2']
        if t1 not in updated or t2 not in updated:
            continue

        r1 = updated[t1]
        r2 = updated[t2]
        home_bonus = HOME_ADV if m['home'] else 0

        oe = 1 / (1 + 10 ** (-((r1 + home_bonus) - r2) / D_FACTOR))

        sc1, sc2 = m['sc1'], m['sc2']
        if sc1 > sc2:
            o = 1.0
        elif sc1 < sc2:
            o = 0.0
        else:
            o = 0.5

        margin_ratio = min(max(sc1, sc2) / max(min(sc1, sc2), 1), MAX_MARGIN)
        k = m['weight'] * margin_ratio

        updated[t1] = updated[t1] + k * (o - oe)
        updated[t2] = updated[t2] + k * ((1 - o) - (1 - oe))
        applied += 1

    print(f"  ✅ ELO updated from {applied} results")
    return {k: round(v) for k, v in updated.items()}


def main():
    print(f"\n{'='*50}")
    print(f"  GAA Scraper — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}\n")

    ratings = fetch_elo_ratings()

    if not ratings:
        if os.path.exists(GAA_ELO_PATH):
            with open(GAA_ELO_PATH) as f:
                existing = json.load(f)
            ratings = existing.get('ratings', {})
            print(f"  📂 Using cached ELO file")
        else:
            print("  ❌ No ratings available — aborting")
            return

    results = fetch_season_results()

    if results:
        with open(GAA_RESULTS_PATH, 'w') as f:
            json.dump({'scraped_at': datetime.now(timezone.utc).isoformat(), 'results': results}, f, indent=2)
        print(f"  💾 {len(results)} results saved → {GAA_RESULTS_PATH}\n")

    out = {
        'scraped_at': datetime.now(timezone.utc).isoformat(),
        'ratings': ratings
    }
    with open(GAA_ELO_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  💾 ELO saved → {GAA_ELO_PATH}\n")

    print("🏆 Current GAA Football ELO Rankings:")
    sorted_ratings = sorted(
        [(k, v) for k, v in ratings.items() if k in COUNTIES],
        key=lambda x: x[1], reverse=True
    )
    for rank, (county, elo) in enumerate(sorted_ratings[:16], 1):
        print(f"  {rank:2d}. {county:<15} {elo}")

    print(f"\n✅ GAA scraper complete\n")


if __name__ == "__main__":
    main()
