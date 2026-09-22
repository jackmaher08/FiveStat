"""
Immutable FiveStat paper-betting ledger.

Official strategy (declared before live tracking):
- flat 1-unit stake
- model probability > 50%
- model edge > 10 percentage points vs de-vigged market probability
- settle at the median captured UK bookmaker decimal price

The ledger only appends a recommendation once. Later model/odds refreshes do not
rewrite its entry price or probability. Completed fixtures are then settled.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

TABLE_DIR = "data/tables"
FIXTURES_PATH = os.path.join(TABLE_DIR, "fixture_data.csv")
PROBS_PATH = os.path.join(TABLE_DIR, "fixture_probabilities.csv")
BOOKIE_WIN_PATH = os.path.join(TABLE_DIR, "bookie_win_by_gw.csv")
BOOKIE_OU_PATH = os.path.join(TABLE_DIR, "bookie_ou_by_gw.csv")
RAW_ODDS_PATH = os.path.join(TABLE_DIR, "bookie_raw_odds_by_gw.csv")
LEDGER_PATH = os.path.join(TABLE_DIR, "paper_betting_ledger.csv")
SUMMARY_PATH = os.path.join(TABLE_DIR, "paper_betting_summary.json")

MODEL_MIN_PROB = 50.0
MIN_EDGE_PP = 10.0
STAKE_UNITS = 1.0

TEAM_MAP = {
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Manchester Utd": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
}


def canonical_team(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def current_gw(fixtures):
    upcoming = fixtures[~fixtures["isResult"]]
    if upcoming.empty:
        return None
    return int(pd.to_numeric(upcoming["round_number"], errors="coerce").dropna().min())


def make_market_rows(fixture, model, win_market, ou_market, raw):
    candidates = []
    pairs = [
        ("home_win", "home_win_prob", "bookie_home_win", "home_odds_median"),
        ("draw", "draw_prob", "bookie_draw", "draw_odds_median"),
        ("away_win", "away_win_prob", "bookie_away_win", "away_odds_median"),
    ]
    for market, model_col, market_col, odds_col in pairs:
        if model_col not in model or market_col not in win_market or odds_col not in raw:
            continue
        model_p = float(model[model_col]) * 100
        market_p = pd.to_numeric(win_market[market_col], errors="coerce")
        odds = pd.to_numeric(raw[odds_col], errors="coerce")
        if pd.isna(market_p) or pd.isna(odds):
            continue
        candidates.append((market, model_p, float(market_p), float(odds)))

    over_model = float(model.get("over_2_5_prob", 0) or 0) * 100
    ou_pairs = [
        ("over_2_5", over_model, "bookie_over25", "over25_odds_median"),
        ("under_2_5", 100 - over_model, "bookie_under25", "under25_odds_median"),
    ]
    for market, model_p, market_col, odds_col in ou_pairs:
        if market_col not in ou_market or odds_col not in raw:
            continue
        market_p = pd.to_numeric(ou_market[market_col], errors="coerce")
        odds = pd.to_numeric(raw[odds_col], errors="coerce")
        if pd.isna(market_p) or pd.isna(odds):
            continue
        candidates.append((market, model_p, float(market_p), float(odds)))

    placed_at = str(raw.get("captured_at") or datetime.now(timezone.utc).isoformat())
    rows = []
    for market, model_p, market_p, odds in candidates:
        edge = model_p - market_p
        if model_p <= MODEL_MIN_PROB or edge <= MIN_EDGE_PP:
            continue
        rows.append({
            "fixture_id": int(fixture["id"]),
            "gw": int(fixture["round_number"]),
            "kickoff": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "market": market,
            "model_probability": round(model_p, 2),
            "market_fair_probability": round(market_p, 2),
            "edge_pp": round(edge, 2),
            "decimal_odds": round(odds, 3),
            "stake_units": STAKE_UNITS,
            "placed_at": placed_at,
            "status": "OPEN",
            "won": "",
            "profit_units": "",
            "settled_at": "",
        })
    return rows


def settle_row(row, fixture):
    hg = int(float(fixture["home_goals"]))
    ag = int(float(fixture["away_goals"]))
    market = row["market"]
    if market == "home_win":
        won = hg > ag
    elif market == "draw":
        won = hg == ag
    elif market == "away_win":
        won = ag > hg
    elif market == "over_2_5":
        won = hg + ag > 2
    elif market == "under_2_5":
        won = hg + ag < 3
    else:
        return row

    odds = float(row["decimal_odds"])
    stake = float(row["stake_units"])
    row["status"] = "WON" if won else "LOST"
    row["won"] = bool(won)
    row["profit_units"] = round(stake * (odds - 1), 3) if won else round(-stake, 3)
    row["settled_at"] = datetime.now(timezone.utc).isoformat()
    return row


def build_summary(ledger):
    if ledger.empty:
        settled = ledger
    else:
        settled = ledger[ledger["status"].isin(["WON", "LOST"])].copy()

    bets = len(settled)
    stake = float(pd.to_numeric(settled.get("stake_units"), errors="coerce").fillna(0).sum()) if bets else 0.0
    profit = float(pd.to_numeric(settled.get("profit_units"), errors="coerce").fillna(0).sum()) if bets else 0.0
    wins = int((settled["status"] == "WON").sum()) if bets else 0
    roi = (profit / stake * 100) if stake else 0.0

    return {
        "strategy": {
            "stake_units": STAKE_UNITS,
            "model_probability_min": MODEL_MIN_PROB,
            "edge_min_pp": MIN_EDGE_PP,
            "price_basis": "median captured UK bookmaker decimal odds",
        },
        "bets_settled": bets,
        "wins": wins,
        "losses": bets - wins,
        "win_rate": round((wins / bets * 100), 1) if bets else 0.0,
        "total_staked_units": round(stake, 2),
        "profit_units": round(profit, 2),
        "roi_percent": round(roi, 1),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def run():
    fixtures = load_csv(FIXTURES_PATH)
    raw_odds = load_csv(RAW_ODDS_PATH)
    probs = load_csv(PROBS_PATH)
    win = load_csv(BOOKIE_WIN_PATH)
    ou = load_csv(BOOKIE_OU_PATH)

    if fixtures.empty:
        print("Paper betting: no fixture data; skipping.")
        return

    fixtures["isResult"] = fixtures["isResult"].astype(str).str.lower().eq("true")
    fixtures["round_number"] = pd.to_numeric(fixtures["round_number"], errors="coerce")
    for df in (fixtures, raw_odds, probs, win, ou):
        if not df.empty:
            if "home_team" in df:
                df["home_team"] = df["home_team"].map(canonical_team)
            if "away_team" in df:
                df["away_team"] = df["away_team"].map(canonical_team)

    ledger = load_csv(LEDGER_PATH)
    if ledger.empty:
        ledger = pd.DataFrame(columns=[
            "fixture_id", "gw", "kickoff", "home_team", "away_team", "market",
            "model_probability", "market_fair_probability", "edge_pp",
            "decimal_odds", "stake_units", "placed_at", "status", "won",
            "profit_units", "settled_at",
        ])

    gw = current_gw(fixtures)
    if gw is not None and not raw_odds.empty and not probs.empty:
        gw_label = f"GW{gw}"
        upcoming = fixtures[(~fixtures["isResult"]) & (fixtures["round_number"] == gw)]
        raw_gw = raw_odds[raw_odds["gw"] == gw_label]
        win_gw = win[win["gw"] == gw_label] if not win.empty else pd.DataFrame()
        ou_gw = ou[ou["gw"] == gw_label] if not ou.empty else pd.DataFrame()

        existing = set()
        if not ledger.empty:
            for _, row in ledger.iterrows():
                existing.add((str(row["fixture_id"]), str(row["market"])))

        additions = []
        for _, f in upcoming.iterrows():
            h, a = f["home_team"], f["away_team"]
            model_rows = probs[(probs["home_team"] == h) & (probs["away_team"] == a)]
            raw_rows = raw_gw[(raw_gw["home_team"] == h) & (raw_gw["away_team"] == a)]
            if model_rows.empty or raw_rows.empty:
                continue
            win_rows = win_gw[(win_gw["home_team"] == h) & (win_gw["away_team"] == a)]
            ou_rows = ou_gw[(ou_gw["home_team"] == h) & (ou_gw["away_team"] == a)]
            win_row = win_rows.iloc[-1].to_dict() if not win_rows.empty else {}
            ou_row = ou_rows.iloc[-1].to_dict() if not ou_rows.empty else {}
            for rec in make_market_rows(
                f.to_dict(), model_rows.iloc[-1].to_dict(), win_row, ou_row, raw_rows.iloc[-1].to_dict()
            ):
                key = (str(rec["fixture_id"]), rec["market"])
                if key not in existing:
                    additions.append(rec)
                    existing.add(key)

        if additions:
            ledger = pd.concat([ledger, pd.DataFrame(additions)], ignore_index=True)
            print(f"Paper betting: added {len(additions)} qualifying recommendations.")

    completed_lookup = {
        str(int(row["id"])): row.to_dict()
        for _, row in fixtures[fixtures["isResult"]].iterrows()
        if pd.notna(row.get("id"))
    }
    for idx, row in ledger.iterrows():
        if row.get("status") != "OPEN":
            continue
        fixture = completed_lookup.get(str(row["fixture_id"]))
        if not fixture:
            continue
        settled = settle_row(row.to_dict(), fixture)
        for key, value in settled.items():
            ledger.at[idx, key] = value

    os.makedirs(TABLE_DIR, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    summary = build_summary(ledger)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"Paper betting: {summary['bets_settled']} settled bets, "
        f"{summary['profit_units']:+.2f}u profit, ROI {summary['roi_percent']:+.1f}%."
    )


if __name__ == "__main__":
    run()
