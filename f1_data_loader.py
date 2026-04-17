import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

F1_DATA_DIR = "data/f1"
RACES_DIR = os.path.join(F1_DATA_DIR, "races")

CURRENT_YEAR = datetime.now().year

# Weight multiplier per race back in time — 0.85 means race N-1 is worth 85%
# of race N, race N-2 is worth 72%, etc. Mirrors recency logic in EPL model.
RECENCY_DECAY = 0.85
DPR_RELIABLE_THRESHOLD = 8
DPR_CAUTION_THRESHOLD = 4

import time as _time
_cache = {}
_CACHE_TTL = 3600

def _cache_get(key):
    if key in _cache:
        ts, val = _cache[key]
        if _time.time() - ts < _CACHE_TTL:
            return val
    return None

def _cache_set(key, val):
    _cache[key] = (_time.time(), val)

def invalidate_f1_cache():
    _cache.clear()


# ── File helpers ───────────────────────────────────────────────────────────────

def _load_json(path, fallback=None):
    if not os.path.exists(path):
        return fallback
    with open(path) as f:
        return json.load(f)


def _race_key(year, round_num):
    return f"{year}_{int(round_num):02d}"


def _load_race_file(year, round_num, suffix, fallback=None):
    path = os.path.join(RACES_DIR, f"{_race_key(year, round_num)}_{suffix}.json")
    return _load_json(path, fallback)


def _laps_available(laps):
    return laps is not None and len(laps) > 0


# ── Basic loaders ──────────────────────────────────────────────────────────────

def load_f1_schedule():
    return _load_json(os.path.join(F1_DATA_DIR, "schedule.json"), fallback=[])


def load_f1_driver_standings():
    return _load_json(os.path.join(F1_DATA_DIR, "driver_standings.json"), fallback=[])


def load_f1_constructor_standings():
    return _load_json(os.path.join(F1_DATA_DIR, "constructor_standings.json"), fallback=[])


def load_f1_race_results(year, round_num):
    return _load_race_file(year, round_num, "results", fallback=[])


def load_f1_qualifying(year, round_num):
    return _load_race_file(year, round_num, "qualifying", fallback=[])


def load_f1_qualifying_laps(year, round_num):
    return _load_race_file(year, round_num, "qualifying_laps", fallback=[])


def load_f1_sprint(year, round_num):
    return _load_race_file(year, round_num, "sprint", fallback=[])


def load_f1_laps(year, round_num):
    return _load_race_file(year, round_num, "laps", fallback=[])


def load_f1_stints(year, round_num):
    return _load_race_file(year, round_num, "stints", fallback=[])


def load_f1_weather(year, round_num):
    return _load_race_file(year, round_num, "weather", fallback=[])


def load_f1_openf1(year, round_num):
    return _load_race_file(year, round_num, "openf1", fallback={})


def get_completed_races(year=None):
    """Return list of (year, round, name, date) for all completed races with results files."""
    schedule = load_f1_schedule()
    today = datetime.now().strftime("%Y-%m-%d")
    completed = []

    if year:
        path = os.path.join(F1_DATA_DIR, f"schedule_{year}.json")
        year_schedule = _load_json(path, fallback=None)
        if year_schedule is None:
            year_schedule = [r for r in schedule if r.get("season") == year]
        if not year_schedule:
            import glob
            pattern = os.path.join(RACES_DIR, f"{year}_*_results.json")
            found_rounds = sorted([
                int(os.path.basename(f).split("_")[1])
                for f in glob.glob(pattern)
            ])
            year_schedule = [{"season": year, "round": r, "date": "2000-01-01",
                              "name": f"Round {r}", "circuit_name": "", "country": ""}
                             for r in found_rounds]
    else:
        year_schedule = schedule

    for race in year_schedule:
        race_year = race.get("season", CURRENT_YEAR)
        round_num = race.get("round")
        race_date = race.get("date", "")
        if race_date and race_date <= today:
            results = load_f1_race_results(race_year, round_num)
            if results:
                completed.append({
                    "year": race_year,
                    "round": round_num,
                    "name": race.get("name"),
                    "circuit": race.get("circuit_name"),
                    "country": race.get("country"),
                    "date": race_date,
                })
    return completed


def get_last_race_info():
    completed = get_completed_races(year=CURRENT_YEAR)
    if not completed:
        return None
    last = completed[-1]
    results = load_f1_race_results(last["year"], last["round"])
    winner = next((r for r in results if r.get("position") == "1"), None)
    last["winner"] = winner
    last["results"] = results
    return last


def get_next_race_info():
    schedule = load_f1_schedule()
    today = datetime.now().strftime("%Y-%m-%d")
    upcoming = [r for r in schedule if r.get("date", "") > today]
    return upcoming[0] if upcoming else None


# ── Clean lap filtering ────────────────────────────────────────────────────────

def _get_clean_laps(laps_df):
    """
    Filter to representative race pace laps only.
    Removes: first lap, pit entry/exit laps, outliers via IQR per driver.
    This is the core data quality gate for DPR and stint efficiency.
    """
    df = laps_df.copy()
    df = df[df["LapNumber"] > 1]
    df = df[df["LapTime_s"].notna()]
    df = df[df["LapTime_s"] > 0]
    df = df[df["PitInTime_s"].isna()]
    df = df[df["PitOutTime_s"].isna()]

    clean_rows = []
    for driver, grp in df.groupby("Driver"):
        q1 = grp["LapTime_s"].quantile(0.25)
        q3 = grp["LapTime_s"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        filtered = grp[grp["LapTime_s"] <= upper]
        clean_rows.append(filtered)

    if not clean_rows:
        return pd.DataFrame()
    return pd.concat(clean_rows, ignore_index=True)


# ── Teammate pairs ─────────────────────────────────────────────────────────────

def _get_teammate_pairs(year, round_num):
    """
    Derive teammate pairs from race results for a given round.
    Returns dict: {driver_code: teammate_code}
    """
    results = load_f1_race_results(year, round_num)
    if not results:
        return {}

    constructor_to_drivers = {}
    for r in results:
        cid = r.get("constructor_id", r.get("constructor", "unknown"))
        code = r.get("code") or r.get("driver_id", "")
        constructor_to_drivers.setdefault(cid, []).append(code)

    pairs = {}
    for drivers in constructor_to_drivers.values():
        if len(drivers) == 2:
            pairs[drivers[0]] = drivers[1]
            pairs[drivers[1]] = drivers[0]
    return pairs


# ── FiveStat Model: Driver Pace Rating (DPR) ──────────────────────────────────

def calculate_driver_pace_rating(year=None, bust_cache=False):
    cache_key = f"dpr_{year or CURRENT_YEAR}"
    if not bust_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    """
    Driver Pace Rating — teammate-delta-normalised pace, recency-weighted.

    Method:
      For each race with valid lap data:
        1. Filter to clean laps.
        2. Calculate each driver's median clean lap time.
        3. Compute delta vs teammate: (driver - teammate) / teammate * 100
           Negative = faster than teammate (better).
        4. Weight each race by RECENCY_DECAY^i (i=0 is most recent race).
      Final DPR = weighted mean of per-race deltas.

    Returns dict: {driver_code: {"dpr": float, "races_counted": int,
                                  "constructor": str, "name": str}}
    """
    target_year = year or CURRENT_YEAR
    completed = get_completed_races(year=target_year)

    if target_year == CURRENT_YEAR:
        prior_y1 = get_completed_races(year=target_year - 1)
        prior_y2 = get_completed_races(year=target_year - 2)
        completed = prior_y2 + prior_y1 + completed

    if not completed:
        return {}

    completed_reversed = list(reversed(completed))

    driver_weighted_deltas = {}
    driver_weight_totals = {}
    driver_meta = {}
    driver_races_counted = {}

    for i, race in enumerate(completed_reversed):
        weight = RECENCY_DECAY ** i
        round_num = race["round"]
        race_year = race["year"]

        raw_laps = load_f1_laps(race_year, round_num)
        if not _laps_available(raw_laps):
            continue

        laps_df = pd.DataFrame(raw_laps)
        clean_df = _get_clean_laps(laps_df)
        if clean_df.empty:
            continue

        teammate_pairs = _get_teammate_pairs(race_year, round_num)
        results = load_f1_race_results(race_year, round_num)
        driver_info = {r.get("code"): r for r in results}

        medians = clean_df.groupby("Driver")["LapTime_s"].median().to_dict()

        for driver, driver_median in medians.items():
            teammate = teammate_pairs.get(driver)
            if not teammate or teammate not in medians:
                continue

            teammate_median = medians[teammate]
            if teammate_median == 0:
                continue

            delta = (driver_median - teammate_median) / teammate_median * 100

            driver_weighted_deltas.setdefault(driver, 0)
            driver_weight_totals.setdefault(driver, 0)
            driver_races_counted.setdefault(driver, 0)
            driver_weighted_deltas[driver] += delta * weight
            driver_weight_totals[driver] += weight
            driver_races_counted[driver] += 1

            if driver not in driver_meta:
                info = driver_info.get(driver, {})
                driver_meta[driver] = {
                    "name": info.get("name", driver),
                    "constructor": info.get("constructor", ""),
                    "constructor_id": info.get("constructor_id", ""),
                }

    ratings = {}
    for driver, total_weight in driver_weight_totals.items():
        if total_weight == 0:
            continue
        dpr = driver_weighted_deltas[driver] / total_weight
        races_counted = driver_races_counted.get(driver, 0)
        if races_counted >= DPR_RELIABLE_THRESHOLD:
            reliability = "high"
        elif races_counted >= DPR_CAUTION_THRESHOLD:
            reliability = "caution"
        else:
            reliability = "low"

        ratings[driver] = {
            "dpr": round(dpr, 3),
            "races_counted": races_counted,
            "reliability": reliability,
            **driver_meta.get(driver, {}),
        }

    result = dict(sorted(ratings.items(), key=lambda x: x[1]["dpr"]))
    _cache_set(cache_key, result)
    return result


# ── FiveStat Model: Constructor Pace Index (CPI) ──────────────────────────────

def calculate_constructor_pace_index(year=None, bust_cache=False):
    cache_key = f"cpi_{year or CURRENT_YEAR}"
    if not bust_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    """
    Constructor Pace Index — team median pace vs field median, recency-weighted.

    Method:
      For each race with valid lap data:
        1. Filter to clean laps.
        2. Calculate field median lap time.
        3. Each constructor's driver median averaged → constructor median.
        4. CPI delta = (constructor_median - field_median) / field_median * 100
           Negative = faster than field (better).
        5. Weight by RECENCY_DECAY^i.

    Returns dict: {constructor_id: {"cpi": float, "name": str, "races_counted": int}}
    """
    target_year = year or CURRENT_YEAR
    completed = get_completed_races(year=target_year)
    if not completed:
        return {}

    completed_reversed = list(reversed(completed))

    constructor_weighted_deltas = {}
    constructor_weight_totals = {}
    constructor_names = {}

    for i, race in enumerate(completed_reversed):
        weight = RECENCY_DECAY ** i
        round_num = race["round"]
        race_year = race["year"]

        raw_laps = load_f1_laps(race_year, round_num)
        if not _laps_available(raw_laps):
            continue

        laps_df = pd.DataFrame(raw_laps)
        clean_df = _get_clean_laps(laps_df)
        if clean_df.empty:
            continue

        results = load_f1_race_results(race_year, round_num)
        driver_to_constructor = {
            r.get("code"): (r.get("constructor_id"), r.get("constructor"))
            for r in results
        }

        field_median = clean_df["LapTime_s"].median()
        if field_median == 0:
            continue

        constructor_laps = {}
        for _, row in clean_df.iterrows():
            driver = row["Driver"]
            cinfo = driver_to_constructor.get(driver)
            if not cinfo:
                continue
            cid, cname = cinfo
            constructor_laps.setdefault(cid, {"times": [], "name": cname})
            constructor_laps[cid]["times"].append(row["LapTime_s"])

        for cid, data in constructor_laps.items():
            if not data["times"]:
                continue
            c_median = np.median(data["times"])
            delta = (c_median - field_median) / field_median * 100

            constructor_weighted_deltas.setdefault(cid, 0)
            constructor_weight_totals.setdefault(cid, 0)
            constructor_weighted_deltas[cid] += delta * weight
            constructor_weight_totals[cid] += weight
            constructor_names[cid] = data["name"]

    index = {}
    for cid, total_weight in constructor_weight_totals.items():
        if total_weight == 0:
            continue
        cpi = constructor_weighted_deltas[cid] / total_weight
        races_counted = sum(
            1 for race in completed
            if _laps_available(load_f1_laps(race["year"], race["round"]))
        )
        index[cid] = {
            "cpi": round(cpi, 3),
            "name": constructor_names.get(cid, cid),
            "races_counted": races_counted,
        }

    result = dict(sorted(index.items(), key=lambda x: x[1]["cpi"]))
    _cache_set(cache_key, result)
    return result


# ── FiveStat Model: Stint Efficiency Score ─────────────────────────────────────

def calculate_stint_efficiency(year, round_num):
    """
    Stint Efficiency Score — actual lap times vs expected from compound
    degradation model, per driver per stint.

    Method:
      For each compound in the race:
        1. Collect all laps on that compound across all drivers.
        2. Fit linear model: lap_time = base + slope * tyre_age (np.polyfit).
        3. For each driver stint: predicted = base + slope * tyre_age per lap.
        4. Efficiency = mean(predicted - actual) in seconds.
           Positive = driver is faster than the compound model predicts (good).
           Negative = driver is slower (tyre management issue or car issue).

    Returns list of dicts, one per driver stint.
    """
    raw_laps = load_f1_laps(year, round_num)
    if not _laps_available(raw_laps):
        return []

    laps_df = pd.DataFrame(raw_laps)
    clean_df = _get_clean_laps(laps_df)
    if clean_df.empty:
        return []

    clean_df = clean_df[clean_df["TyreLife"].notna() & clean_df["Compound"].notna()]

    compound_models = {}
    for compound, grp in clean_df.groupby("Compound"):
        if len(grp) < 5:
            continue
        x = grp["TyreLife"].values.astype(float)
        y = grp["LapTime_s"].values.astype(float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            compound_models[compound] = {"slope": slope, "intercept": intercept}
        except Exception:
            continue

    results = load_f1_race_results(year, round_num)
    driver_info = {r.get("code"): r for r in results}

    stint_scores = []
    for (driver, stint, compound), grp in clean_df.groupby(["Driver", "Stint", "Compound"]):
        model = compound_models.get(compound)
        if not model:
            continue

        x = grp["TyreLife"].values.astype(float)
        y_actual = grp["LapTime_s"].values.astype(float)
        y_predicted = model["intercept"] + model["slope"] * x
        residuals = y_predicted - y_actual
        efficiency = float(np.mean(residuals))

        info = driver_info.get(driver, {})
        stint_scores.append({
            "driver": driver,
            "name": info.get("name", driver),
            "constructor": info.get("constructor", ""),
            "stint": int(stint),
            "compound": compound,
            "lap_count": len(grp),
            "start_lap": int(grp["LapNumber"].min()),
            "end_lap": int(grp["LapNumber"].max()),
            "mean_lap_s": round(float(y_actual.mean()), 3),
            "efficiency_s": round(efficiency, 3),
        })

    return sorted(stint_scores, key=lambda x: x["efficiency_s"], reverse=True)


# ── Route data packages ────────────────────────────────────────────────────────

def get_f1_hub_data():
    """
    Full data package for the /f1 hub route.
    Returns everything the template needs in one call.
    """
    return {
        "schedule": load_f1_schedule(),
        "driver_standings": load_f1_driver_standings(),
        "constructor_standings": load_f1_constructor_standings(),
        "last_race": get_last_race_info(),
        "next_race": get_next_race_info(),
        "dpr": calculate_driver_pace_rating(),
        "cpi": calculate_constructor_pace_index(),
    }


def get_race_report(year, round_num):
    """
    Full data package for the /f1/race/<round> route.
    Returns everything the template needs in one call.
    """
    results = load_f1_race_results(year, round_num)
    qualifying = load_f1_qualifying(year, round_num)
    laps = load_f1_laps(year, round_num)
    stints = load_f1_stints(year, round_num)
    weather = load_f1_weather(year, round_num)
    stint_efficiency = calculate_stint_efficiency(year, round_num)

    teammate_pairs = _get_teammate_pairs(year, round_num)
    driver_info = {r.get("code"): r for r in results}

    teammate_deltas = []
    if _laps_available(laps):
        laps_df = pd.DataFrame(laps)
        clean_df = _get_clean_laps(laps_df)
        if not clean_df.empty:
            medians = clean_df.groupby("Driver")["LapTime_s"].median().to_dict()
            seen = set()
            for driver, teammate in teammate_pairs.items():
                pair_key = tuple(sorted([driver, teammate]))
                if pair_key in seen or driver not in medians or teammate not in medians:
                    continue
                seen.add(pair_key)
                d_med = medians[driver]
                t_med = medians[teammate]
                delta = (d_med - t_med) / t_med * 100
                d_info = driver_info.get(driver, {})
                t_info = driver_info.get(teammate, {})
                teammate_deltas.append({
                    "constructor": d_info.get("constructor", ""),
                    "driver_a": driver,
                    "driver_a_name": d_info.get("name", driver),
                    "driver_a_median_s": round(d_med, 3),
                    "driver_b": teammate,
                    "driver_b_name": t_info.get("name", teammate),
                    "driver_b_median_s": round(t_med, 3),
                    "delta_pct": round(delta, 3),
                })

    schedule = load_f1_schedule()
    race_meta = next(
        (r for r in schedule if r.get("round") == int(round_num)),
        {}
    )

    return {
        "year": year,
        "round": round_num,
        "meta": race_meta,
        "results": results,
        "qualifying": qualifying,
        "stints": stints,
        "weather": weather,
        "teammate_deltas": teammate_deltas,
        "stint_efficiency": stint_efficiency,
        "has_lap_data": _laps_available(laps),
    }


def get_f1_drivers_data():
    """
    Full data package for the /f1/drivers route.
    Returns everything the template needs in one call.
    """
    dpr = calculate_driver_pace_rating()
    cpi = calculate_constructor_pace_index()
    standings = load_f1_driver_standings()

    standings_map = {s["code"]: s for s in standings}

    drivers_combined = []
    for code, rating in dpr.items():
        standing = standings_map.get(code, {})
        drivers_combined.append({
            "code": code,
            "name": rating.get("name", code),
            "constructor": rating.get("constructor", ""),
            "constructor_id": rating.get("constructor_id", ""),
            "dpr": rating["dpr"],
            "races_counted": rating["races_counted"],
            "championship_position": standing.get("position"),
            "championship_points": standing.get("points"),
            "wins": standing.get("wins"),
        })

    drivers_combined.sort(key=lambda x: x["dpr"])

    return {
        "drivers": drivers_combined,
        "dpr": dpr,
        "cpi": cpi,
        "driver_standings": standings,
        "constructor_standings": load_f1_constructor_standings(),
    }



# ── FiveStat Model: Race Prediction ───────────────────────────────────────────
# Append this block to the bottom of f1_data_loader.py, above the closing line.

CIRCUIT_OVERTAKING_INDEX = {
    # 2026 calendar — 0.0 = grid position dominates, 1.0 = pure pace
    "albert_park":         0.55,
    "shanghai":            0.60,
    "suzuka":              0.50,
    "miami":               0.55,
    "villeneuve":          0.60,
    "monaco":              0.10,
    "catalunya":           0.45,
    "red_bull_ring":       0.65,
    "silverstone":         0.60,
    "spa":                 0.70,
    "hungaroring":         0.35,
    "zandvoort":           0.40,
    "monza":               0.80,
    "baku":                0.65,
    "marina_bay":          0.30,
    "suzuka":              0.50,
    "americas":            0.55,
    "rodriguez":           0.55,
    "interlagos":          0.65,
    "las_vegas":           0.65,
    "losail":              0.60,
    "yas_marina":          0.55,
}

DNF_RATE = 0.08
N_SIMULATIONS = 5000
PACE_SIGMA = 0.18


def _get_circuit_overtaking_index(circuit_id):
    circuit_id_lower = (circuit_id or "").lower().replace(" ", "_")
    for key, val in CIRCUIT_OVERTAKING_INDEX.items():
        if key in circuit_id_lower or circuit_id_lower in key:
            return val
    return 0.55


def _parse_laptime_to_seconds(t):
    if not t:
        return None
    try:
        parts = str(t).split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(t)
    except Exception:
        return None


def calculate_race_predictions(year, round_num, bust_cache=False):
    cache_key = f"pred_{year}_{round_num}"
    if not bust_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    """
    FiveStat Race Prediction Model.

    Two modes:
      Post-qualifying: actual grid positions + pace model
      Pre-qualifying:  pace model only, grid estimated from pace ranking

    Method:
      For each driver, combine Constructor Pace Index (CPI) and Driver Pace
      Rating (DPR) into a single pace score. Blend with grid position via
      a circuit-specific overtaking index. Run 10,000 Monte Carlo simulations
      drawing from a normal distribution of race randomness, applying an 8%
      DNF probability per driver. Tally win/podium/points probabilities.

    Returns dict with keys:
      predictions   — list of driver prediction dicts, sorted by win_prob desc
      has_quali     — bool, whether qualifying data was available
      race_meta     — dict with circuit name etc
      mode          — 'post_quali' or 'pre_quali'
    """
    dpr = calculate_driver_pace_rating(year)
    cpi = calculate_constructor_pace_index(year)

    schedule = load_f1_schedule()
    race_meta = next(
        (r for r in schedule if r.get("round") == int(round_num)), {}
    )
    circuit_id = race_meta.get("circuit_id", "")
    overtaking_index = _get_circuit_overtaking_index(circuit_id)

    qualifying = load_f1_qualifying(year, round_num)
    results = load_f1_race_results(year, round_num)
    has_quali = bool(qualifying)
    is_completed = bool(results)

    driver_standings = load_f1_driver_standings()
    standings_map = {d["code"]: d for d in driver_standings}

    drivers_in_race = []

    if has_quali:
        for q in qualifying:
            code = q.get("code", "")
            constructor_id = q.get("constructor_id", "")
            dpr_entry = dpr.get(code, {})
            cpi_entry = cpi.get(constructor_id, {})
            dpr_val = dpr_entry.get("dpr", 0.0)
            cpi_val = cpi_entry.get("cpi", 0.0)
            pace_score = cpi_val + dpr_val

            q3_time = _parse_laptime_to_seconds(q.get("q3"))
            q2_time = _parse_laptime_to_seconds(q.get("q2"))
            q1_time = _parse_laptime_to_seconds(q.get("q1"))
            best_quali_time = q3_time or q2_time or q1_time

            standing = standings_map.get(code, {})
            drivers_in_race.append({
                "code":             code,
                "name":             q.get("name", code),
                "constructor":      q.get("constructor", ""),
                "constructor_id":   constructor_id,
                "grid":             int(q.get("position", 20)),
                "pace_score":       pace_score,
                "dpr":              dpr_val,
                "cpi":              cpi_val,
                "best_quali_time":  best_quali_time,
                "championship_pos": standing.get("position"),
                "championship_pts": standing.get("points"),
            })
    else:
        for code, d in dpr.items():
            constructor_id = d.get("constructor_id", "")
            cpi_entry = cpi.get(constructor_id, {})
            dpr_val = d.get("dpr", 0.0)
            cpi_val = cpi_entry.get("cpi", 0.0)
            pace_score = cpi_val + dpr_val
            standing = standings_map.get(code, {})
            drivers_in_race.append({
                "code":             code,
                "name":             d.get("name", code),
                "constructor":      d.get("constructor", ""),
                "constructor_id":   constructor_id,
                "grid":             None,
                "pace_score":       pace_score,
                "dpr":              dpr_val,
                "cpi":              cpi_val,
                "best_quali_time":  None,
                "championship_pos": standing.get("position"),
                "championship_pts": standing.get("points"),
            })
        drivers_in_race.sort(key=lambda x: x["pace_score"])
        for i, d in enumerate(drivers_in_race):
            d["grid"] = i + 1

    if not drivers_in_race:
        return {
            "predictions": [],
            "has_quali": has_quali,
            "race_meta": race_meta,
            "mode": "post_quali" if has_quali else "pre_quali",
            "overtaking_index": overtaking_index,
            "is_completed": is_completed,
        }

    n = len(drivers_in_race)
    pace_scores = np.array([d["pace_score"] for d in drivers_in_race])
    grids = np.array([d["grid"] for d in drivers_in_race], dtype=float)

    pace_min, pace_max = pace_scores.min(), pace_scores.max()
    pace_range = pace_max - pace_min if pace_max != pace_min else 1.0
    pace_norm = (pace_scores - pace_min) / pace_range

    grid_norm = (grids - 1) / max(n - 1, 1)

    grid_weight = 1.0 - overtaking_index
    pace_weight = overtaking_index

    win_counts    = np.zeros(n)
    podium_counts = np.zeros(n)
    points_counts = np.zeros(n)
    position_sum  = np.zeros(n)

    rng = np.random.default_rng(seed=42)

    for _ in range(N_SIMULATIONS):
        dnf_mask = rng.random(n) < DNF_RATE
        noise = rng.normal(0, PACE_SIGMA, n)
        perf = pace_weight * pace_norm + grid_weight * grid_norm + noise

        finishers = np.where(~dnf_mask)[0]
        dnfers = np.where(dnf_mask)[0]

        finisher_perf = perf[finishers]
        finish_order = finishers[np.argsort(finisher_perf)]
        dnf_order = dnfers[np.argsort(perf[dnfers])[::-1]]
        full_order = np.concatenate([finish_order, dnf_order])

        for pos, idx in enumerate(full_order):
            position_sum[idx] += pos + 1
            if pos == 0:
                win_counts[idx] += 1
            if pos < 3:
                podium_counts[idx] += 1
            if pos < 10 and not dnf_mask[idx]:
                points_counts[idx] += 1

    predictions = []
    actual_result_map = {r.get("code"): r for r in results} if results else {}

    for i, driver in enumerate(drivers_in_race):
        actual = actual_result_map.get(driver["code"])
        predictions.append({
            "code":             driver["code"],
            "name":             driver["name"],
            "constructor":      driver["constructor"],
            "constructor_id":   driver["constructor_id"],
            "grid":             driver["grid"],
            "pace_score":       round(driver["pace_score"], 4),
            "dpr":              round(driver["dpr"], 3),
            "cpi":              round(driver["cpi"], 3),
            "best_quali_time":  driver["best_quali_time"],
            "championship_pos": driver["championship_pos"],
            "championship_pts": driver["championship_pts"],
            "win_prob":         round(win_counts[i]    / N_SIMULATIONS * 100, 1),
            "podium_prob":      round(podium_counts[i] / N_SIMULATIONS * 100, 1),
            "points_prob":      round(points_counts[i] / N_SIMULATIONS * 100, 1),
            "expected_pos":     round(position_sum[i]  / N_SIMULATIONS, 1),
            "actual_position":  actual.get("position") if actual else None,
            "actual_points":    actual.get("points") if actual else None,
        })

    predictions.sort(key=lambda x: x["win_prob"], reverse=True)

    result = {
        "predictions":      predictions,
        "has_quali":        has_quali,
        "race_meta":        race_meta,
        "mode":             "post_quali" if has_quali else "pre_quali",
        "overtaking_index": round(overtaking_index, 2),
        "is_completed":     is_completed,
        "n_simulations":    N_SIMULATIONS,
    }
    _cache_set(cache_key, result)
    return result


# ── F1 Fantasy scoring ────────────────────────────────────────────────────────

F1F_POINTS_FINISH = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8,  7: 6,  8: 4,  9: 2,  10: 1,
}
F1F_POINTS_DNF        = -20
F1F_POINTS_FASTEST_LAP = 10
F1F_POINTS_PER_PLACE_GAINED = 1
F1F_POINTS_PER_PLACE_LOST   = -1
F1F_POINTS_CONSTRUCTOR_WIN  = 10
F1F_FASTEST_LAP_PROB        = 0.05

F1F_SPRINT_POINTS_FINISH = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1,
}
F1F_SPRINT_POINTS_DNF              = -10
F1F_SPRINT_POINTS_PER_PLACE_GAINED = 1
F1F_SPRINT_POINTS_PER_PLACE_LOST   = -1
F1F_QUALI_POINTS = {
    1: 10, 2: 9, 3: 8, 4: 7, 5: 6,
    6: 5,  7: 4, 8: 3, 9: 2, 10: 1,
}
F1F_QUALI_NO_TIME = -5


def calculate_xfp(predictions, race_round=None):
    """
    Expected Fantasy Points per driver for a single race.

    Scoring:
      Finish points from F1F_POINTS_FINISH lookup
      DNF: -20 pts
      Fastest lap: +5 pts, applied at F1F_FASTEST_LAP_PROB per driver weighted by win_prob
      Positions gained vs grid: +2 per place
      Positions lost vs grid: -1 per place

    Returns predictions list with xfp field added, sorted by xfp desc.
    """
    if not predictions:
        return []

    total_win_prob = sum(d.get("win_prob", 0) for d in predictions) or 1

    result = []
    for d in predictions:
        grid = d.get("grid") or 10
        win_prob = d.get("win_prob", 0)
        podium_prob = d.get("podium_prob", 0)
        points_prob = d.get("points_prob", 0)
        expected_pos = d.get("expected_pos") or grid
        dnf_prob = DNF_RATE

        finish_xfp = 0
        for pos, pts in F1F_POINTS_FINISH.items():
            if pos == 1:
                p = win_prob / 100
            elif pos <= 3:
                p = (podium_prob - win_prob) / 100 / 2
            elif pos <= 10:
                p = (points_prob - podium_prob) / 100 / 7
            else:
                p = 0
            p = max(p, 0)
            finish_xfp += p * pts

        dnf_xfp = dnf_prob * F1F_POINTS_DNF

        fl_weight = (win_prob / total_win_prob) if total_win_prob else 0
        fl_xfp = fl_weight * F1F_FASTEST_LAP_PROB * F1F_POINTS_FASTEST_LAP * 20

        pos_delta = grid - expected_pos
        if pos_delta > 0:
            pos_xfp = pos_delta * F1F_POINTS_PER_PLACE_GAINED
        else:
            pos_xfp = pos_delta * abs(F1F_POINTS_PER_PLACE_LOST)

        xfp = round(finish_xfp + dnf_xfp + fl_xfp + pos_xfp, 1)

        result.append({**d, "xfp": xfp})

    result.sort(key=lambda x: x["xfp"], reverse=True)
    return result


def calculate_actual_fantasy_points(year, round_num):
    """
    Returns official F1 Fantasy points for a completed race weekend.
    Reads from fantasy_points.json if available (source of truth).
    Falls back to calculated values if the race isn't yet in the manual file.
    """
    manual_path = os.path.join(F1_DATA_DIR, "fantasy_points.json")
    manual_data = _load_json(manual_path, fallback={})
    race_key = _race_key(year, round_num)
    if race_key in manual_data:
        return {code: float(pts) for code, pts in manual_data[race_key].items()}

    results = load_f1_race_results(year, round_num)
    if not results:
        return {}

    driver_points = {}
    for r in results:
        code = r.get("code", "")
        if not code:
            continue

        position_text = r.get("position_text", "")
        try:
            position = int(r.get("position", 99))
        except (ValueError, TypeError):
            position = 99

        grid = int(r.get("grid", 0) or 0)
        is_dnf = position_text in ("R", "D", "E", "W", "F", "N")

        if is_dnf:
            pts = F1F_POINTS_DNF
        else:
            pts = F1F_POINTS_FINISH.get(position, 0)

        if r.get("fastest_lap_rank") == "1":
            pts += F1F_POINTS_FASTEST_LAP

        if not is_dnf and grid > 0 and position > 0:
            places = grid - position
            if places > 0:
                pts += places * F1F_POINTS_PER_PLACE_GAINED
            elif places < 0:
                pts += places * abs(F1F_POINTS_PER_PLACE_LOST)

        driver_points[code] = round(pts, 1)

    quali = load_f1_qualifying(year, round_num)
    for q in quali:
        code = q.get("code", "")
        if not code:
            continue
        try:
            pos = int(q.get("position", 99))
        except (ValueError, TypeError):
            pos = 99
        q_pts = F1F_QUALI_POINTS.get(pos, 0)
        if not q.get("q1"):
            q_pts = F1F_QUALI_NO_TIME
        if code in driver_points:
            driver_points[code] = round(driver_points[code] + q_pts, 1)

    schedule = load_f1_schedule()
    race_info = next((r for r in schedule if r.get("round") == round_num), None)
    if race_info and race_info.get("has_sprint"):
        sprint = load_f1_sprint(year, round_num)
        for s in sprint:
            code = s.get("code", "")
            if not code:
                continue
            s_pos_text = s.get("position_text", "")
            try:
                s_pos = int(s.get("position", 99))
            except (ValueError, TypeError):
                s_pos = 99
            s_grid = int(s.get("grid", 0) or 0)
            s_is_dnf = s_pos_text in ("R", "D", "E", "W", "F", "N")

            if s_is_dnf:
                s_pts = F1F_SPRINT_POINTS_DNF
            else:
                s_pts = F1F_SPRINT_POINTS_FINISH.get(s_pos, 0)

            if not s_is_dnf and s_grid > 0 and s_pos > 0:
                s_places = s_grid - s_pos
                if s_places > 0:
                    s_pts += s_places * F1F_SPRINT_POINTS_PER_PLACE_GAINED
                elif s_places < 0:
                    s_pts += s_places * abs(F1F_SPRINT_POINTS_PER_PLACE_LOST)

            if code in driver_points:
                driver_points[code] = round(driver_points[code] + s_pts, 1)

    return driver_points


def calculate_season_xfp_projection():
    """
    Project total fantasy points for each driver across remaining season races.
    Uses race predictions for each upcoming race, sums xFP.
    Returns list of dicts sorted by projected_total_xfp desc.
    """
    cache_key = "season_xfp_projection"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    schedule = load_f1_schedule()
    today = datetime.now().strftime("%Y-%m-%d")
    upcoming = [r for r in schedule if r.get("date", "") > today]
    completed_races = [r for r in schedule if r.get("date", "") <= today]

    driver_projected = {}
    driver_completed_xfp = {}
    driver_meta = {}

    dpr = calculate_driver_pace_rating()
    cpi = calculate_constructor_pace_index()
    standings = load_f1_driver_standings()
    driver_info_map = {d["code"]: d for d in standings}

    for race in completed_races:
        year = race.get("season", CURRENT_YEAR)
        round_num = race.get("round")
        results = load_f1_race_results(year, round_num)
        if not results:
            continue
        actual_pts = calculate_actual_fantasy_points(year, round_num)
        for r in results:
            code = r.get("code", "")
            if not code:
                continue
            driver_completed_xfp.setdefault(code, 0)
            driver_completed_xfp[code] += actual_pts.get(code, 0)
            dpr_entry = dpr.get(code, {})
            driver_meta[code] = {
                "name": r.get("name", code),
                "constructor": r.get("constructor", ""),
                "constructor_id": r.get("constructor_id", ""),
            }

    races_remaining = len(upcoming)
    for race in upcoming:
        year = race.get("season", CURRENT_YEAR)
        round_num = race.get("round")
        pred = calculate_race_predictions(year, round_num)
        xfp_list = calculate_xfp(pred.get("predictions", []))
        for d in xfp_list:
            code = d["code"]
            driver_projected.setdefault(code, 0)
            driver_projected[code] += d["xfp"]
            driver_meta[code] = {
                "name": d["name"],
                "constructor": d["constructor"],
                "constructor_id": d["constructor_id"],
            }

    standings = load_f1_driver_standings()
    standings_map = {s["code"]: s for s in standings}

    projection = []
    all_codes = set(driver_projected) | set(driver_completed_xfp)
    for code in all_codes:
        standing = standings_map.get(code, {})
        meta = driver_meta.get(code, {})
        projected_remaining = driver_projected.get(code, 0)
        completed_total = driver_completed_xfp.get(code, 0)
        projection.append({
            "code":                  code,
            "name":                  meta.get("name", code),
            "constructor":           meta.get("constructor", ""),
            "constructor_id":        meta.get("constructor_id", ""),
            "championship_points":   standing.get("points"),
            "championship_position": standing.get("position"),
            "completed_xfp":         round(completed_total, 1),
            "projected_remaining_xfp": round(projected_remaining, 1),
            "projected_total_xfp":   round(completed_total + projected_remaining, 1),
            "races_remaining":       races_remaining,
        })

    projection.sort(key=lambda x: x["projected_total_xfp"], reverse=True)
    _cache_set(cache_key, projection)
    return projection


def calculate_transfer_targets(horizons=(1, 3, 5)):
    """
    Transfer targets for each horizon (next 1, 3, 5 races).
    Returns dict: {1: [...], 3: [...], 5: [...]}
    Each list is drivers sorted by summed xFP over that horizon.
    """
    cache_key = f"transfer_targets_{horizons}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    schedule = load_f1_schedule()
    today = datetime.now().strftime("%Y-%m-%d")
    upcoming = [r for r in schedule if r.get("date", "") > today]

    result = {}
    for horizon in horizons:
        races = upcoming[:horizon]
        driver_xfp = {}
        driver_meta = {}

        for race in races:
            year = race.get("season", CURRENT_YEAR)
            round_num = race.get("round")
            race_name = race.get("name", f"Round {round_num}")
            pred = calculate_race_predictions(year, round_num)
            xfp_list = calculate_xfp(pred.get("predictions", []))
            for d in xfp_list:
                code = d["code"]
                driver_xfp.setdefault(code, {"total": 0, "races": []})
                driver_xfp[code]["total"] += d["xfp"]
                driver_xfp[code]["races"].append({
                    "round": round_num,
                    "name": race_name,
                    "xfp": d["xfp"],
                    "win_prob": d.get("win_prob"),
                    "podium_prob": d.get("podium_prob"),
                })
                driver_meta[code] = {
                    "name": d["name"],
                    "constructor": d["constructor"],
                    "constructor_id": d["constructor_id"],
                }

        standings = load_f1_driver_standings()
        standings_map = {s["code"]: s for s in standings}

        targets = []
        for code, data in driver_xfp.items():
            standing = standings_map.get(code, {})
            meta = driver_meta.get(code, {})
            targets.append({
                "code":            code,
                "name":            meta.get("name", code),
                "constructor":     meta.get("constructor", ""),
                "constructor_id":  meta.get("constructor_id", ""),
                "total_xfp":       round(data["total"], 1),
                "per_race_xfp":    round(data["total"] / max(len(races), 1), 1),
                "races":           data["races"],
                "championship_points": standing.get("points"),
            })

        targets.sort(key=lambda x: x["total_xfp"], reverse=True)
        result[horizon] = targets

    _cache_set(cache_key, result)
    return result


def get_next_race_predictions():
    """Convenience wrapper — predictions for the next upcoming race."""
    next_race = get_next_race_info()
    if not next_race:
        return None
    return calculate_race_predictions(next_race.get("season", CURRENT_YEAR), next_race["round"])


def get_f1_predictions_data(year, round_num):
    """Full data package for the /f1/predictions/<year>/<round> route."""
    pred = calculate_race_predictions(year, round_num)
    schedule = load_f1_schedule()

    prev_round = next(
        (r for r in reversed(schedule) if r.get("round", 0) < int(round_num)), None
    )
    next_round = next(
        (r for r in schedule if r.get("round", 0) > int(round_num)), None
    )

    xfp = calculate_xfp(pred.get("predictions", []))
    xfp_map = {d["code"]: d["xfp"] for d in xfp}
    predictions_with_xfp = [
        {**p, "xfp": xfp_map.get(p["code"], 0)}
        for p in pred.get("predictions", [])
    ]
    return {
        **pred,
        "predictions": predictions_with_xfp,
        "year":        year,
        "round":       round_num,
        "prev_round":  prev_round,
        "next_round":  next_round,
    }


def get_f1_fantasy_data():
    """Full data package for the /f1/fantasy route."""
    cache_key = "f1_fantasy_data"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    next_race = get_next_race_info()
    next_pred = None
    next_xfp = []
    if next_race:
        pred = calculate_race_predictions(
            next_race.get("season", CURRENT_YEAR), next_race["round"]
        )
        next_xfp = calculate_xfp(pred.get("predictions", []))
        next_pred = {**pred, "xfp": next_xfp}

    season_projection = calculate_season_xfp_projection()
    transfer_targets = calculate_transfer_targets(horizons=(1, 3, 5))

    result = {
        "next_race": next_race,
        "next_race_xfp": next_xfp,
        "season_projection": season_projection,
        "transfer_targets": transfer_targets,
        "races_remaining": len([
            r for r in load_f1_schedule()
            if r.get("date", "") > datetime.now().strftime("%Y-%m-%d")
        ]),
    }
    _cache_set(cache_key, result)
    return result