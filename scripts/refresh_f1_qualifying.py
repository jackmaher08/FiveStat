"""Refresh published qualifying classifications before race day.

Run independently of the full F1 scraper, which only processes races whose
calendar date has arrived. Never replace a complete file with partial data.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from f1_data_loader import get_upcoming_f1_races, load_f1_schedule

RACES_DIR = Path("data/f1/races")
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


def convert_qualifying(payload):
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    rows = races[0].get("QualifyingResults", [])
    if len(rows) < 15:
        return []
    return [{
        "position": int(q["position"]),
        "driver_id": q["Driver"]["driverId"],
        "code": q["Driver"].get("code", ""),
        "name": f"{q['Driver']['givenName']} {q['Driver']['familyName']}",
        "constructor": q["Constructor"]["name"],
        "constructor_id": q["Constructor"]["constructorId"],
        "q1": q.get("Q1"),
        "q2": q.get("Q2"),
        "q3": q.get("Q3"),
    } for q in rows]


def refresh_qualifying(now=None, get=requests.get, races_dir=RACES_DIR):
    """Fetch qualifying for races within the next 72 hours or underway.

    Returns how many files changed. No file is created when the feed has not
    published a sufficiently complete classification; repeated calls are safe.
    """
    now = now or datetime.now(timezone.utc)
    races = get_upcoming_f1_races(load_f1_schedule(), now)
    changes = 0
    for race in races:
        # The next event is the only relevant candidate; skip distant rounds.
        from f1_data_loader import _race_start_utc
        start = _race_start_utc(race)
        if start is None or start > now + timedelta(hours=72):
            continue
        path = races_dir / f"{int(race['season'])}_{int(race['round']):02d}_qualifying.json"
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        # Qualifying order is stable once the full published classification exists.
        if len(existing) >= 15:
            continue
        endpoint = f"{JOLPICA_BASE}/{race['season']}/{race['round']}/qualifying.json"
        try:
            response = get(endpoint, timeout=20)
            response.raise_for_status()
            updated = convert_qualifying(response.json())
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            print(f"Could not refresh {race['name']} qualifying: {exc}")
            continue
        if len(updated) <= len(existing):
            continue
        races_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        print(f"Updated qualifying for {race['name']}: {len(updated)} drivers")
        changes += 1
    return changes


if __name__ == "__main__":
    print(f"Qualifying files updated: {refresh_qualifying()}")
