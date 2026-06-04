import os
import json
import requests
import time
from datetime import datetime, timezone

FD_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "b888513f418f4173a75525dc0bd75f92")
FD_BASE    = "https://api.football-data.org/v4"
FD_HEADERS = {"X-Auth-Token": FD_API_KEY}

DATA_DIR = os.path.join(os.path.dirname(__file__), "static", "data")
os.makedirs(DATA_DIR, exist_ok=True)

WC_ELO_PATH     = os.path.join(DATA_DIR, "wc_elo.json")
WC_MATCHES_PATH = os.path.join(DATA_DIR, "wc_matches.json")

BASELINE_ELO = {
    "Argentina":              2082,
    "France":                 2005,
    "England":                1993,
    "Brazil":                 1988,
    "Spain":                  1985,
    "Portugal":               1961,
    "Netherlands":            1958,
    "Belgium":                1942,
    "Germany":                1939,
    "Croatia":                1921,
    "Colombia":               1906,
    "Uruguay":                1896,
    "Denmark":                1889,
    "USA":                    1880,
    "Mexico":                 1876,
    "Switzerland":            1874,
    "Austria":                1861,
    "Japan":                  1856,
    "Senegal":                1851,
    "Morocco":                1849,
    "Turkey":                 1838,
    "South Korea":            1831,
    "Ecuador":                1820,
    "Poland":                 1818,
    "Australia":              1811,
    "Serbia":                 1808,
    "Czech Republic":         1805,
    "Hungary":                1801,
    "Romania":                1796,
    "Ukraine":                1793,
    "Sweden":                 1792,
    "Iran":                   1784,
    "Canada":                 1780,
    "Algeria":                1778,
    "Ivory Coast":            1775,
    "Nigeria":                1772,
    "Tunisia":                1768,
    "Mali":                   1755,
    "Egypt":                  1749,
    "Venezuela":              1741,
    "Paraguay":               1738,
    "Slovakia":               1736,
    "Scotland":               1731,
    "Bosnia and Herzegovina": 1726,
    "Iraq":                   1718,
    "South Africa":           1712,
    "Congo DR":               1707,
    "New Zealand":            1654,
    "Haiti":                  1642,
    "Curacao":                1638,
    "Cabo Verde":             1635,
    "Jordan":                 1628,
    "Qatar":                  1624,
    "Scotland":               1731,
    "Sweden":                 1792,
    "Norway":                 1810
}

K_FACTOR = 40

FD_NAME_MAP = {
    "United States":      "USA",
    "Korea Republic":     "South Korea",
    "Côte d'Ivoire":      "Ivory Coast",
    "Türkiye":            "Turkey",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Congo DR":           "Congo DR",
    "Czech Republic":     "Czech Republic",
    "IR Iran":            "Iran",
}


def elo_expected(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo_from_results(ratings, matches):
    finished = [m for m in matches if m["status"] == "FINISHED"]
    if not finished:
        return ratings

    updated = ratings.copy()
    for m in finished:
        home, away = m["home"], m["away"]
        hg, ag = m["home_goals"], m["away_goals"]
        if home not in updated or away not in updated:
            continue
        if hg is None or ag is None:
            continue

        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        exp    = elo_expected(updated[home], updated[away])

        updated[home] = round(updated[home] + K_FACTOR * (actual - exp))
        updated[away] = round(updated[away] + K_FACTOR * ((1 - actual) - (1 - exp)))

    changed = sum(1 for t in updated if updated[t] != ratings.get(t))
    print(f"  ✅ ELO updated from {len(finished)} finished matches ({changed} teams moved)")
    return updated


def load_or_init_elo():
    if os.path.exists(WC_ELO_PATH):
        with open(WC_ELO_PATH) as f:
            existing = json.load(f)
        print(f"  📂 Loaded existing ELO file (scraped {existing.get('scraped_at', 'unknown')})")
        return existing.get("ratings", BASELINE_ELO.copy())
    print("  🆕 No ELO file found — initialising from baseline")
    return BASELINE_ELO.copy()


def fetch_wc_matches():
    print("📅 Fetching WC 2026 fixtures and results from football-data.org...")
    matches = []

    for stage in ["GROUP_STAGE", "ROUND_OF_32", "ROUND_OF_16",
                  "QUARTER_FINALS", "SEMI_FINALS", "THIRD_PLACE", "FINAL"]:
        url = f"{FD_BASE}/competitions/WC/matches?season=2026&stage={stage}"
        try:
            resp = requests.get(url, headers=FD_HEADERS, timeout=15)
            if resp.status_code == 429:
                print("  ⏳ Rate limited — waiting 60s...")
                time.sleep(60)
                resp = requests.get(url, headers=FD_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            stage_matches = data.get("matches", [])
            matches.extend(stage_matches)
            print(f"  ✅ {stage}: {len(stage_matches)} matches")
            time.sleep(1)
        except Exception as e:
            print(f"  ⚠️  Failed to fetch {stage}: {e}")

    return matches


def normalise_match(raw):
    home = raw["homeTeam"]["name"]
    away = raw["awayTeam"]["name"]
    home = FD_NAME_MAP.get(home, home)
    away = FD_NAME_MAP.get(away, away)

    score = raw.get("score", {})
    ft    = score.get("fullTime", {})
    home_goals = ft.get("home")
    away_goals = ft.get("away")

    return {
        "id":         raw["id"],
        "utcDate":    raw["utcDate"],
        "stage":      raw["stage"],
        "group":      raw.get("group"),
        "matchday":   raw.get("matchday"),
        "status":     raw["status"],
        "home":       home,
        "away":       away,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def main():
    print(f"\n{'='*50}")
    print(f"  WC 2026 Scraper — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}\n")

    print("🌍 Loading ELO ratings...")
    current_elo = load_or_init_elo()

    raw_matches = fetch_wc_matches()
    if raw_matches:
        normalised = [normalise_match(m) for m in raw_matches]
        normalised.sort(key=lambda m: m["utcDate"])
        out = {
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "matches": normalised
        }
        with open(WC_MATCHES_PATH, "w") as f:
            json.dump(out, f, indent=2)
        finished = sum(1 for m in normalised if m["status"] == "FINISHED")
        print(f"  💾 {len(normalised)} matches saved ({finished} finished) → {WC_MATCHES_PATH}\n")

        print("📊 Updating ELO from finished results...")
        updated_elo = update_elo_from_results(current_elo, normalised)
    else:
        print("  ⚠️  Matches not updated — keeping existing file\n")
        updated_elo = current_elo

    elo_out = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "ratings": updated_elo
    }
    with open(WC_ELO_PATH, "w") as f:
        json.dump(elo_out, f, indent=2)
    print(f"  💾 ELO saved → {WC_ELO_PATH}\n")

    print("✅ WC scraper complete\n")


if __name__ == "__main__":
    main()
