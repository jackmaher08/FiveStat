import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime

import fastf1
import logging
logging.getLogger("fastf1").setLevel(logging.ERROR)

HISTORICAL_MODE = "--historical" in sys.argv
CURRENT_YEAR = datetime.now().year
FASTF1_START_YEAR = 2018
OPENF1_START_YEAR = 2023

SAVE_DIR = "data/f1"
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
RACES_DIR = os.path.join(SAVE_DIR, "races")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RACES_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE = "https://api.openf1.org/v1"


def jolpica_get(endpoint):
    url = f"{JOLPICA_BASE}/{endpoint}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    time.sleep(0.3)
    return resp.json()


def openf1_get(endpoint, params=None):
    url = f"{OPENF1_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    time.sleep(0.3)
    return resp.json()


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def fetch_openf1_laps(openf1_path, laps_path, stints_path):
    existing = _load_openf1(openf1_path)
    if not existing:
        return False
    session_key = existing.get("session_key")
    if not session_key:
        return False
    try:
        raw_laps = openf1_get("laps", {"session_key": session_key})
        if not raw_laps:
            return False

        drivers = {d["driver_number"]: d.get("name_acronym", str(d["driver_number"]))
                   for d in existing.get("drivers", [])}

        def get_stint_info(driver_num, lap_num):
            for s in raw_stints:
                lap_start = s.get("lap_start") or 0
                lap_end = s.get("lap_end") or 9999
                tyre_age = s.get("tyre_age_at_start") or 0
                if s.get("driver_number") == driver_num and lap_start <= lap_num <= lap_end:
                    return s.get("stint_number", 1), s.get("compound", "UNKNOWN"), lap_num - lap_start + tyre_age
            return None, "UNKNOWN", 0

        laps = []
        for l in raw_laps:
            driver_num = l.get("driver_number")
            lap_num = l.get("lap_number", 0)
            stint_num, compound, tyre_life = get_stint_info(driver_num, lap_num)
            laps.append({
                "Driver":        drivers.get(driver_num, str(driver_num)),
                "LapNumber":     lap_num,
                "LapTime_s":     l.get("lap_duration"),
                "Sector1Time_s": l.get("duration_sector_1"),
                "Sector2Time_s": l.get("duration_sector_2"),
                "Sector3Time_s": l.get("duration_sector_3"),
                "Compound":      compound,
                "TyreLife":      tyre_life,
                "Stint":         stint_num,
                "PitInTime_s":   None,
                "PitOutTime_s":  None if not l.get("is_pit_out_lap") else 0,
                "IsPersonalBest": False,
                "SpeedI1":       l.get("i1_speed"),
                "SpeedI2":       l.get("i2_speed"),
                "SpeedFL":       None,
                "SpeedST":       l.get("st_speed"),
            })

        save_json(laps, laps_path)
        print(f"    ✅ Lap data saved via OpenF1 ({len(laps)} laps)")

        import pandas as pd
        laps_df = pd.DataFrame(laps)
        stints = (
            laps_df[laps_df["Stint"].notna() & laps_df["Compound"].notna()]
            .groupby(["Driver", "Stint", "Compound"], dropna=False)
            .agg(start_lap=("LapNumber", "min"), end_lap=("LapNumber", "max"),
                 lap_count=("LapNumber", "count"), tyre_life_at_end=("TyreLife", "max"))
            .reset_index()
        )
        save_json(stints.to_dict(orient="records"), stints_path)
        print(f"    ✅ Stint data saved via OpenF1 ({len(stints)} stints)")
        return True

    except Exception as e:
        print(f"    ⚠️  OpenF1 laps fallback: {e}")
        return False


def _load_openf1(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data if data else None


def td_to_seconds(td):
    try:
        return td.total_seconds()
    except Exception:
        return None


# ── 1. Schedule ───────────────────────────────────────────────────────────────
print("🔄 Fetching current season schedule...")
try:
    data = jolpica_get(f"{CURRENT_YEAR}.json")
    races_raw = data["MRData"]["RaceTable"]["Races"]
    schedule = []
    for r in races_raw:
        entry = {
            "season": int(r["season"]),
            "round": int(r["round"]),
            "name": r["raceName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "circuit_name": r["Circuit"]["circuitName"],
            "country": r["Circuit"]["Location"]["country"],
            "locality": r["Circuit"]["Location"]["locality"],
            "date": r["date"],
            "time": r.get("time"),
            "has_sprint": "Sprint" in r,
        }
        schedule.append(entry)
    save_json(schedule, os.path.join(SAVE_DIR, "schedule.json"))
    print(f"✅ Schedule saved ({len(schedule)} rounds)")
except Exception as e:
    print(f"⚠️  Could not fetch schedule: {e}")


# ── 2. Driver standings ───────────────────────────────────────────────────────
print("🔄 Fetching driver standings...")
try:
    data = jolpica_get("current/driverStandings.json")
    raw = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    driver_standings = []
    for s in raw:
        driver_standings.append({
            "position": int(s["position"]),
            "driver_id": s["Driver"]["driverId"],
            "code": s["Driver"].get("code", ""),
            "name": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
            "nationality": s["Driver"]["nationality"],
            "constructor": s["Constructors"][0]["name"],
            "constructor_id": s["Constructors"][0]["constructorId"],
            "points": float(s["points"]),
            "wins": int(s["wins"]),
        })
    save_json(driver_standings, os.path.join(SAVE_DIR, "driver_standings.json"))
    print(f"✅ Driver standings saved ({len(driver_standings)} drivers)")
except Exception as e:
    print(f"⚠️  Could not fetch driver standings: {e}")


# ── 3. Constructor standings ──────────────────────────────────────────────────
print("🔄 Fetching constructor standings...")
try:
    data = jolpica_get("current/constructorStandings.json")
    raw = data["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
    constructor_standings = []
    for s in raw:
        constructor_standings.append({
            "position": int(s["position"]),
            "constructor_id": s["Constructor"]["constructorId"],
            "name": s["Constructor"]["name"],
            "nationality": s["Constructor"]["nationality"],
            "points": float(s["points"]),
            "wins": int(s["wins"]),
        })
    save_json(constructor_standings, os.path.join(SAVE_DIR, "constructor_standings.json"))
    print(f"✅ Constructor standings saved ({len(constructor_standings)} constructors)")
except Exception as e:
    print(f"⚠️  Could not fetch constructor standings: {e}")


# ── 4. OpenF1 meetings cache (2023+) ─────────────────────────────────────────
openf1_meetings_by_year = {}

years_to_scrape = range(FASTF1_START_YEAR, CURRENT_YEAR + 1) if HISTORICAL_MODE else [CURRENT_YEAR]

for year in years_to_scrape:
    if year >= OPENF1_START_YEAR:
        try:
            meetings = openf1_get("meetings", {"year": year})
            openf1_meetings_by_year[year] = meetings
        except Exception as e:
            print(f"⚠️  Could not fetch OpenF1 meetings for {year}: {e}")
            openf1_meetings_by_year[year] = []


# ── 5. Per-race data ──────────────────────────────────────────────────────────
today = datetime.now().strftime("%Y-%m-%d")

for year in years_to_scrape:
    print(f"\n{'=' * 50}")
    print(f"🏎️  Processing {year} season...")

    try:
        sched_data = jolpica_get(f"{year}.json")
        year_races = sched_data["MRData"]["RaceTable"]["Races"]
    except Exception as e:
        print(f"⚠️  Could not fetch {year} schedule: {e}")
        continue

    for race in year_races:
        round_num = int(race["round"])
        race_name = race["raceName"]
        race_date = race.get("date", "")
        race_key = f"{year}_{round_num:02d}"

        if race_date and race_date > today:
            print(f"\n  ⏭️  Round {round_num} ({race_name}) — future race, skipping")
            continue

        print(f"\n  🔄 Round {round_num}: {race_name} ({race_date})")

        # ── 5a. Race results (Jolpica) ────────────────────────────────────────
        results_path = os.path.join(RACES_DIR, f"{race_key}_results.json")
        if not os.path.exists(results_path):
            try:
                r_data = jolpica_get(f"{year}/{round_num}/results.json")
                r_races = r_data["MRData"]["RaceTable"]["Races"]
                if r_races:
                    results = []
                    for r in r_races[0].get("Results", []):
                        results.append({
                            "position": r.get("position"),
                            "position_text": r.get("positionText"),
                            "driver_id": r["Driver"]["driverId"],
                            "code": r["Driver"].get("code", ""),
                            "name": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                            "constructor": r["Constructor"]["name"],
                            "constructor_id": r["Constructor"]["constructorId"],
                            "grid": int(r.get("grid", 0)),
                            "laps": int(r.get("laps", 0)),
                            "status": r.get("status"),
                            "points": float(r.get("points", 0)),
                            "finish_time": r.get("Time", {}).get("time") if r.get("Time") else None,
                            "finish_millis": int(r["Time"]["millis"]) if r.get("Time", {}).get("millis") else None,
                            "fastest_lap_rank": r.get("FastestLap", {}).get("rank") if r.get("FastestLap") else None,
                            "fastest_lap_time": r.get("FastestLap", {}).get("Time", {}).get("time") if r.get("FastestLap") else None,
                            "fastest_lap_speed": r.get("FastestLap", {}).get("AverageSpeed", {}).get("speed") if r.get("FastestLap") else None,
                        })
                    save_json(results, results_path)
                    print(f"    ✅ Race results saved ({len(results)} drivers)")
                else:
                    print(f"    ⚠️  No race result data returned")
            except Exception as e:
                print(f"    ⚠️  Race results: {e}")
        else:
            print(f"    ⏭️  Race results already saved")

        # ── 5b. Qualifying results (Jolpica) ──────────────────────────────────
        quali_path = os.path.join(RACES_DIR, f"{race_key}_qualifying.json")
        if not os.path.exists(quali_path):
            try:
                q_data = jolpica_get(f"{year}/{round_num}/qualifying.json")
                q_races = q_data["MRData"]["RaceTable"]["Races"]
                if q_races:
                    quali = []
                    for q in q_races[0].get("QualifyingResults", []):
                        quali.append({
                            "position": int(q["position"]),
                            "driver_id": q["Driver"]["driverId"],
                            "code": q["Driver"].get("code", ""),
                            "name": f"{q['Driver']['givenName']} {q['Driver']['familyName']}",
                            "constructor": q["Constructor"]["name"],
                            "constructor_id": q["Constructor"]["constructorId"],
                            "q1": q.get("Q1"),
                            "q2": q.get("Q2"),
                            "q3": q.get("Q3"),
                        })
                    save_json(quali, quali_path)
                    print(f"    ✅ Qualifying results saved ({len(quali)} drivers)")
                else:
                    print(f"    ⚠️  No qualifying data returned")
            except Exception as e:
                print(f"    ⚠️  Qualifying results: {e}")
        else:
            print(f"    ⏭️  Qualifying results already saved")

        # ── 5c. Sprint results (Jolpica, if applicable) ───────────────────────
        if race.get("Sprint"):
            sprint_path = os.path.join(RACES_DIR, f"{race_key}_sprint.json")
            if not os.path.exists(sprint_path):
                try:
                    s_data = jolpica_get(f"{year}/{round_num}/sprint.json")
                    s_races = s_data["MRData"]["RaceTable"]["Races"]
                    if s_races:
                        sprint = []
                        for s in s_races[0].get("SprintResults", []):
                            sprint.append({
                                "position": s.get("position"),
                                "position_text": s.get("positionText"),
                                "driver_id": s["Driver"]["driverId"],
                                "code": s["Driver"].get("code", ""),
                                "name": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                                "constructor": s["Constructor"]["name"],
                                "grid": int(s.get("grid", 0)),
                                "laps": int(s.get("laps", 0)),
                                "status": s.get("status"),
                                "points": float(s.get("points", 0)),
                            })
                        save_json(sprint, sprint_path)
                        print(f"    ✅ Sprint results saved ({len(sprint)} drivers)")
                except Exception as e:
                    print(f"    ⚠️  Sprint results: {e}")
            else:
                print(f"    ⏭️  Sprint results already saved")

        # ── 5d. FastF1: laps, stints, weather ────────────────────────────────
        laps_path = os.path.join(RACES_DIR, f"{race_key}_laps.json")
        stints_path = os.path.join(RACES_DIR, f"{race_key}_stints.json")
        weather_path = os.path.join(RACES_DIR, f"{race_key}_weather.json")

        if not os.path.exists(laps_path):
            try:
                ff1_session = fastf1.get_session(year, round_num, "R")
                ff1_session.load(telemetry=False, weather=True, messages=False)

                if ff1_session.laps is None or len(ff1_session.laps) == 0:
                    print(f"    ⚠️  FastF1: no lap data available for {year} R{round_num} (pre-2020 archive gap)")
                    continue

                laps_df = ff1_session.laps[[
                    "Driver", "LapNumber", "LapTime",
                    "Sector1Time", "Sector2Time", "Sector3Time",
                    "Compound", "TyreLife", "Stint",
                    "PitInTime", "PitOutTime", "IsPersonalBest",
                    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                ]].copy()

                laps_df["LapTime_s"] = laps_df["LapTime"].apply(td_to_seconds)
                laps_df["Sector1Time_s"] = laps_df["Sector1Time"].apply(td_to_seconds)
                laps_df["Sector2Time_s"] = laps_df["Sector2Time"].apply(td_to_seconds)
                laps_df["Sector3Time_s"] = laps_df["Sector3Time"].apply(td_to_seconds)
                laps_df["PitInTime_s"] = laps_df["PitInTime"].apply(td_to_seconds)
                laps_df["PitOutTime_s"] = laps_df["PitOutTime"].apply(td_to_seconds)

                laps_df = laps_df.drop(columns=[
                    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                    "PitInTime", "PitOutTime",
                ])

                save_json(laps_df.to_dict(orient="records"), laps_path)
                print(f"    ✅ Lap data saved ({len(laps_df)} laps)")

                if not os.path.exists(stints_path):
                    stints = (
                        laps_df
                        .groupby(["Driver", "Stint", "Compound"], dropna=False)
                        .agg(
                            start_lap=("LapNumber", "min"),
                            end_lap=("LapNumber", "max"),
                            lap_count=("LapNumber", "count"),
                            tyre_life_at_end=("TyreLife", "max"),
                        )
                        .reset_index()
                    )
                    save_json(stints.to_dict(orient="records"), stints_path)
                    print(f"    ✅ Stint data saved ({len(stints)} stints)")

                if not os.path.exists(weather_path):
                    try:
                        weather_df = ff1_session.weather_data.copy()
                        save_json(weather_df.to_dict(orient="records"), weather_path)
                        print(f"    ✅ Weather data saved ({len(weather_df)} readings)")
                    except Exception as e:
                        print(f"    ⚠️  Weather data: {e}")

            except Exception as e:
                print(f"    ⚠️  FastF1 race session: {e}")
                if "500 calls/h" in str(e) or "rate" in str(e).lower():
                    print(f"    ℹ️  Rate limited — will retry on next run (no sentinel written)")
                    time.sleep(60)
                else:
                    openf1_path = os.path.join(RACES_DIR, f"{race_key}_openf1.json")
                    print(f"    ℹ️  FastF1 unavailable — trying OpenF1 laps fallback...")
                    success = fetch_openf1_laps(openf1_path, laps_path, stints_path)
                    if not success:
                        save_json([], laps_path)
                        save_json([], stints_path)
                        print(f"    ℹ️  Sentinel files written — no lap data available")
        else:
            print(f"    ⏭️  FastF1 lap/stint data already saved")

        # ── 5e. FastF1: qualifying lap times ─────────────────────────────────
        quali_laps_path = os.path.join(RACES_DIR, f"{race_key}_qualifying_laps.json")
        if not os.path.exists(quali_laps_path):
            try:
                q_session = fastf1.get_session(year, round_num, "Q")
                q_session.load(telemetry=False, weather=False, messages=False)

                if q_session.laps is None or len(q_session.laps) == 0:
                    print(f"    ⚠️  FastF1: no qualifying lap data available for {year} R{round_num}")
                    continue

                q_laps_df = q_session.laps[[
                    "Driver", "LapNumber", "LapTime",
                    "Sector1Time", "Sector2Time", "Sector3Time",
                    "Compound", "IsPersonalBest",
                ]].copy()

                q_laps_df["LapTime_s"] = q_laps_df["LapTime"].apply(td_to_seconds)
                q_laps_df["Sector1Time_s"] = q_laps_df["Sector1Time"].apply(td_to_seconds)
                q_laps_df["Sector2Time_s"] = q_laps_df["Sector2Time"].apply(td_to_seconds)
                q_laps_df["Sector3Time_s"] = q_laps_df["Sector3Time"].apply(td_to_seconds)

                q_laps_df = q_laps_df.drop(columns=[
                    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"
                ])

                save_json(q_laps_df.to_dict(orient="records"), quali_laps_path)
                print(f"    ✅ Qualifying lap data saved ({len(q_laps_df)} laps)")
            except Exception as e:
                print(f"    ⚠️  FastF1 qualifying session: {e}")
                if "500 calls/h" in str(e) or "rate" in str(e).lower():
                    print(f"    ℹ️  Rate limited — will retry on next run (no sentinel written)")
                    time.sleep(60)
                else:
                    save_json([], quali_laps_path)
        else:
            print(f"    ⏭️  FastF1 qualifying lap data already saved")

        # ── 5f. OpenF1 supplementary data (2023+) ────────────────────────────
        if year >= OPENF1_START_YEAR:
            openf1_path = os.path.join(RACES_DIR, f"{race_key}_openf1.json")
            if not os.path.exists(openf1_path):
                try:
                    year_meetings = openf1_meetings_by_year.get(year, [])
                    meeting = next(
                        (m for m in year_meetings if m.get("meeting_name", "") in race_name
                         or race_name in m.get("meeting_official_name", "")),
                        None
                    )
                    if not meeting:
                        meeting = next(
                            (m for m in year_meetings
                             if str(m.get("year")) == str(year)
                             and str(m.get("meeting_number")) == str(round_num)),
                            None
                        )

                    if meeting:
                        meeting_key = meeting["meeting_key"]
                        all_sessions = openf1_get("sessions", {"meeting_key": meeting_key})
                        sessions_data = [
                            s for s in all_sessions
                            if s.get("session_name") in ("Race", "Grand Prix")
                            and "sprint" not in s.get("session_name", "").lower()
                        ]
                        if sessions_data:
                            session_key = sessions_data[0]["session_key"]
                            pit_data = openf1_get("pit", {"session_key": session_key})
                            stints_data = openf1_get("stints", {"session_key": session_key})
                            drivers_data = openf1_get("drivers", {"session_key": session_key})
                            intervals_data = openf1_get("intervals", {"session_key": session_key})

                            openf1_payload = {
                                "meeting_key": meeting_key,
                                "session_key": session_key,
                                "meeting_name": meeting.get("meeting_name"),
                                "pit_stops": pit_data,
                                "stints": stints_data,
                                "drivers": drivers_data,
                                "intervals": intervals_data,
                            }
                            save_json(openf1_payload, openf1_path)
                            print(f"    ✅ OpenF1 data saved ({len(pit_data)} pit stops, {len(stints_data)} stints)")
                    else:
                        print(f"    ⚠️  OpenF1: no matching meeting found for {race_name}")
                except Exception as e:
                    print(f"    ⚠️  OpenF1 data: {e}")
                    if "404" in str(e):
                        save_json({}, openf1_path)
                        print(f"    ℹ️  OpenF1 sentinel written — endpoint unavailable for this race")
            else:
                print(f"    ⏭️  OpenF1 data already saved")

        time.sleep(0.5)

print(f"\n{'=' * 50}")
print("✅ F1 scraper complete")