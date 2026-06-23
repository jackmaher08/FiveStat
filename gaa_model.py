import os
import json
import random
from datetime import datetime, timezone

DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")
GAA_ELO_PATH      = os.path.join(DATA_DIR, "gaa_elo.json")
GAA_RESULTS_PATH  = os.path.join(DATA_DIR, "gaa_results.json")
GAA_FIXTURES_PATH = os.path.join(DATA_DIR, "gaa_fixtures.json")

N_SIMS   = 5000
HOME_ADV = 130
D_FACTOR = 500

# Round progression order for the All-Ireland SFC
# The 'round' value comes from column C in the sheet
ROUND_ORDER = ["1", "2A", "2B", "PQF", "3", "QF", "SF", "Final"]

# Friendly labels for display
ROUND_LABELS = {
    "1":     "Round 1",
    "2A":    "Round 2A",
    "2B":    "Round 2B",
    "PQF":   "Preliminary QF",
    "3":     "Round 3",
    "QF":    "Quarter-Final",
    "SF":    "Semi-Final",
    "Final": "Final",
}


def load_json(path, key, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f).get(key, default)


def load_ratings():
    return load_json(GAA_ELO_PATH, "ratings", {})


def load_results():
    return load_json(GAA_RESULTS_PATH, "results", [])


def load_fixtures():
    return load_json(GAA_FIXTURES_PATH, "fixtures", [])


def is_ai(entry):
    return entry.get("grade", "").startswith("All-Ireland")


def win_prob(r_home, r_away, neutral=False):
    bonus = 0 if neutral else HOME_ADV
    return 1 / (1 + 10 ** (-((r_home + bonus) - r_away) / D_FACTOR))


def match_probs(team_a, team_b, ratings, neutral=False):
    ra = ratings.get(team_a, 1400)
    rb = ratings.get(team_b, 1400)
    wp = win_prob(ra, rb, neutral)
    draw_prob = max(0.08, 0.20 - 0.15 * abs(wp - 0.5))
    win_a = wp * (1 - draw_prob)
    win_b = (1 - wp) * (1 - draw_prob)
    total = win_a + draw_prob + win_b
    return round(win_a / total, 4), round(draw_prob / total, 4), round(win_b / total, 4)


def simulate_match(team_a, team_b, ratings, neutral=False):
    pa, pd, pb = match_probs(team_a, team_b, ratings, neutral)
    r = random.random()
    if r < pa:
        return team_a
    elif r < pa + pd:
        return team_a if random.random() < 0.52 else team_b
    else:
        return team_b


def get_current_round(fixtures):
    """The earliest unplayed round in the AI championship is the 'current' one."""
    ai_fixtures = [f for f in fixtures if is_ai(f)]
    for rnd in ROUND_ORDER:
        if any(f.get("round") == rnd for f in ai_fixtures):
            return rnd
    return None


def teams_still_in(results, fixtures):
    """Determine which teams remain alive in the AI championship."""
    # A team is eliminated if it lost a 2B, PQF, 3, QF, SF, or Final match
    eliminated = set()
    knockout_rounds = {"2B", "PQF", "3", "QF", "SF", "Final"}
    for m in results:
        if not is_ai(m):
            continue
        if m.get("round") in knockout_rounds and m.get("winner"):
            loser = m["team1"] if m["winner"] == m["team2"] else m["team2"]
            eliminated.add(loser)

    # Teams appearing in any AI fixture or result, minus eliminated
    alive = set()
    for m in results + fixtures:
        if is_ai(m):
            alive.add(m["team1"])
            alive.add(m["team2"])
    return alive - eliminated


def simulate_from_round(current_round, alive_teams, fixtures, ratings):
    """Simulate the championship forward from the current round."""
    # Get the actual fixtures for the current round if known
    cur_fixtures = [f for f in fixtures if is_ai(f) and f.get("round") == current_round]

    reached = {}

    def advance(matchups, neutral):
        winners = []
        for a, b in matchups:
            winners.append(simulate_match(a, b, ratings, neutral))
        return winners

    # Build remaining bracket dynamically based on which round we're at
    # Strategy: simulate known fixtures first, then approximate later rounds
    # by random draw among survivors

    if current_round in ("2A", "2B"):
        # Use actual 2A and 2B fixtures
        r2a = [(f["team1"], f["team2"], f["home"]) for f in fixtures
               if is_ai(f) and f.get("round") == "2A"]
        r2b = [(f["team1"], f["team2"], f["home"]) for f in fixtures
               if is_ai(f) and f.get("round") == "2B"]

        r2a_winners, r2a_losers = [], []
        for a, b, home in r2a:
            w = simulate_match(a, b, ratings, neutral=not home)
            l = b if w == a else a
            r2a_winners.append(w)
            r2a_losers.append(l)

        r2b_winners = []
        for a, b, home in r2b:
            r2b_winners.append(simulate_match(a, b, ratings, neutral=not home))

        # Round 3: R2A losers vs R2B winners
        survivors_seeded = r2a_winners[:]            # straight to QF
        r3a = r2a_losers[:]
        r3b = r2b_winners[:]
        random.shuffle(r3a); random.shuffle(r3b)
        r3_winners = [simulate_match(r3a[i], r3b[i], ratings, neutral=True)
                      for i in range(min(len(r3a), len(r3b)))]

        qf_field = build_qf(survivors_seeded, r3_winners)

    elif current_round in ("PQF", "3"):
        # Round 3 fixtures known; need seeded teams (already-through R2A winners)
        # Approximate: alive teams not in current R3 fixtures are the seeds
        r3_fixtures = [(f["team1"], f["team2"], f["home"]) for f in cur_fixtures]
        r3_teams = set(t for f in r3_fixtures for t in (f[0], f[1]))
        seeded = [t for t in alive_teams if t not in r3_teams]

        r3_winners = [simulate_match(a, b, ratings, neutral=not home)
                      for a, b, home in r3_fixtures]
        qf_field = build_qf(seeded, r3_winners)

    elif current_round == "QF":
        qf_fixtures = [(f["team1"], f["team2"]) for f in cur_fixtures]
        if qf_fixtures:
            qf_field = qf_fixtures
        else:
            teams = list(alive_teams)
            random.shuffle(teams)
            qf_field = [(teams[i], teams[i+1]) for i in range(0, len(teams)-1, 2)]

    elif current_round == "SF":
        sf_fixtures = [(f["team1"], f["team2"]) for f in cur_fixtures]
        sf_winners = [simulate_match(a, b, ratings, neutral=True) for a, b in sf_fixtures]
        champion = simulate_match(sf_winners[0], sf_winners[1], ratings, True) if len(sf_winners) >= 2 else (sf_winners[0] if sf_winners else None)
        return {"qf_winners": [], "sf_winners": sf_winners, "champion": champion}

    elif current_round == "Final":
        f = cur_fixtures[0] if cur_fixtures else None
        if f:
            champion = simulate_match(f["team1"], f["team2"], ratings, True)
            return {"qf_winners": [], "sf_winners": [f["team1"], f["team2"]], "champion": champion}
        return {"qf_winners": [], "sf_winners": [], "champion": None}

    else:
        # Fallback: random draw among alive teams
        teams = list(alive_teams)
        random.shuffle(teams)
        qf_field = [(teams[i], teams[i+1]) for i in range(0, len(teams)-1, 2)]

    # From QF onward
    qf_winners = advance(qf_field, neutral=True)

    random.shuffle(qf_winners)
    sf_field = [(qf_winners[i], qf_winners[i+1]) for i in range(0, len(qf_winners)-1, 2)]
    sf_winners = advance(sf_field, neutral=True)

    if len(sf_winners) >= 2:
        champion = simulate_match(sf_winners[0], sf_winners[1], ratings, neutral=True)
    elif len(sf_winners) == 1:
        champion = sf_winners[0]
    else:
        champion = None

    return {"qf_winners": qf_winners, "sf_winners": sf_winners, "champion": champion}


def build_qf(seeded, unseeded):
    """4 seeded vs 4 unseeded, random draw."""
    s = seeded[:]; u = unseeded[:]
    random.shuffle(s); random.shuffle(u)
    pairs = []
    for i in range(min(len(s), len(u))):
        pairs.append((s[i], u[i]))
    # If uneven, pair leftovers
    leftover = s[len(u):] + u[len(s):]
    for i in range(0, len(leftover)-1, 2):
        pairs.append((leftover[i], leftover[i+1]))
    return pairs


def run_simulation(ratings, current_round, alive_teams, fixtures):
    reach = {t: {"qf": 0, "sf": 0, "final": 0, "winner": 0} for t in alive_teams}

    for _ in range(N_SIMS):
        result = simulate_from_round(current_round, alive_teams, fixtures, ratings)
        for t in result.get("qf_winners", []):
            if t in reach:
                reach[t]["qf"] += 1
        for t in result.get("sf_winners", []):
            if t in reach:
                reach[t]["sf"] += 1
        champ = result.get("champion")
        if champ and champ in reach:
            reach[champ]["winner"] += 1

    return {
        t: {k: round(c / N_SIMS * 100, 1) for k, c in counts.items()}
        for t, counts in reach.items()
    }


def get_current_fixtures_with_probs(current_round, fixtures, ratings):
    cur = [f for f in fixtures if is_ai(f) and f.get("round") == current_round]
    out = []
    for f in cur:
        neutral = not f["home"]
        pa, pd, pb = match_probs(f["team1"], f["team2"], ratings, neutral)
        out.append({
            "home":      f["team1"],
            "away":      f["team2"],
            "date":      f["date"],
            "round":     f["round"],
            "round_label": ROUND_LABELS.get(f["round"], f["round"]),
            "neutral":   neutral,
            "prob_home": round(pa * 100, 1),
            "prob_draw": round(pd * 100, 1),
            "prob_away": round(pb * 100, 1),
            "home_elo":  ratings.get(f["team1"], 1400),
            "away_elo":  ratings.get(f["team2"], 1400),
        })
    return out


def get_recent_results(results, limit=12):
    ai = [r for r in results if is_ai(r)]
    # Most recent at top — preserve sheet order (newest last), so reverse
    ai = ai[-limit:][::-1]
    out = []
    for r in ai:
        out.append({
            "home":       r["team1"],
            "away":       r["team2"],
            "home_score": r.get("sc1", 0),
            "away_score": r.get("sc2", 0),
            "winner":     r.get("winner"),
            "round_label": ROUND_LABELS.get(r.get("round"), r.get("round", "")),
        })
    return out


def backfill_blank_rounds(fixtures, results):
    """
    Workaround for the sheet occasionally leaving the Round column blank
    for the next batch of All-Ireland fixtures before it's filled in.
    Infers the round as whatever comes after the latest completed
    All-Ireland round found in the results.
    """
    completed_rounds = {r.get("round") for r in results if is_ai(r) and r.get("round")}
    last_idx = -1
    for i, rnd in enumerate(ROUND_ORDER):
        if rnd in completed_rounds:
            last_idx = i

    if last_idx == -1 or last_idx + 1 >= len(ROUND_ORDER):
        return fixtures

    inferred_round = ROUND_ORDER[last_idx + 1]
    for f in fixtures:
        if is_ai(f) and not f.get("round"):
            f["round"] = inferred_round

    return fixtures


def get_gaa_data():
    ratings  = load_ratings()
    results  = load_results()
    fixtures = load_fixtures()
    fixtures = backfill_blank_rounds(fixtures, results)

    current_round = get_current_round(fixtures)
    alive = teams_still_in(results, fixtures)

    if current_round and alive:
        probs = run_simulation(ratings, current_round, alive, fixtures)
    else:
        probs = {}

    winner_table = sorted(
        [{"team": t, "pct": p["winner"], "sf_pct": p["sf"], "qf_pct": p["qf"]}
         for t, p in probs.items() if p["winner"] > 0],
        key=lambda x: x["pct"], reverse=True
    )

    cur_fixtures = get_current_fixtures_with_probs(current_round, fixtures, ratings) if current_round else []
    recent = get_recent_results(results)

    elo_table = sorted(
        [{"team": t, "elo": round(r)} for t, r in ratings.items()
         if t not in ("London", "New York")],
        key=lambda x: x["elo"], reverse=True
    )

    return {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "current_round":  current_round,
        "current_label":  ROUND_LABELS.get(current_round, current_round) if current_round else "Championship Complete",
        "cur_fixtures":   cur_fixtures,
        "recent_results": recent,
        "winner_table":   winner_table,
        "elo_table":      elo_table,
        "probs":          probs,
        "ratings":        ratings,
    }


if __name__ == "__main__":
    print("Running GAA model test...")
    data = get_gaa_data()
    print(f"\nCurrent round: {data['current_label']}")
    print(f"\nTop 10 All-Ireland winner probabilities:")
    for row in data["winner_table"][:10]:
        print(f"  {row['team']:<15} {row['pct']}%")
    print(f"\n{data['current_label']} fixtures:")
    for m in data["cur_fixtures"]:
        print(f"  {m['home']:<12} {m['prob_home']}%  |  Draw {m['prob_draw']}%  |  {m['away']:<12} {m['prob_away']}%")
