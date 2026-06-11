import os
import json
import random
from datetime import datetime, timezone

DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")
GAA_ELO_PATH     = os.path.join(DATA_DIR, "gaa_elo.json")
GAA_RESULTS_PATH = os.path.join(DATA_DIR, "gaa_results.json")

N_SIMS   = 5000
HOME_ADV = 130
D_FACTOR = 500

# 2026 All-Ireland SFC Round 1 results (completed)
R1_RESULTS = [
    {"home": "Armagh",     "away": "Derry",     "home_score": 21, "away_score": 16, "winner": "Armagh"},
    {"home": "Dublin",     "away": "Louth",      "home_score": 27, "away_score": 30, "winner": "Louth"},
    {"home": "Galway",     "away": "Kildare",    "home_score": 30, "away_score": 17, "winner": "Galway"},
    {"home": "Cork",       "away": "Meath",      "home_score": 30, "away_score": 27, "winner": "Cork"},
    {"home": "Kerry",      "away": "Donegal",    "home_score": 16, "away_score": 26, "winner": "Donegal"},
    {"home": "Monaghan",   "away": "Mayo",       "home_score": 26, "away_score": 27, "winner": "Mayo"},
    {"home": "Roscommon",  "away": "Tyrone",     "home_score": 24, "away_score": 25, "winner": "Tyrone"},
    {"home": "Westmeath",  "away": "Cavan",      "home_score": 34, "away_score": 30, "winner": "Westmeath"},
]

# Round 2 fixtures - 13/14 June 2026
# 2A = winners bracket, 2B = losers bracket (eliminated if they lose)
R2_FIXTURES = [
    {"home": "Donegal",   "away": "Cork",      "stream": "2A", "neutral": False, "date": "2026-06-13"},
    {"home": "Monaghan",  "away": "Roscommon", "stream": "2B", "neutral": False, "date": "2026-06-13"},
    {"home": "Kildare",   "away": "Kerry",     "stream": "2B", "neutral": False, "date": "2026-06-13"},
    {"home": "Derry",     "away": "Meath",     "stream": "2B", "neutral": False, "date": "2026-06-13"},
    {"home": "Louth",     "away": "Armagh",    "stream": "2A", "neutral": False, "date": "2026-06-14"},
    {"home": "Cavan",     "away": "Dublin",    "stream": "2B", "neutral": False, "date": "2026-06-14"},
    {"home": "Galway",    "away": "Westmeath", "stream": "2A", "neutral": False, "date": "2026-06-14"},
    {"home": "Tyrone",    "away": "Mayo",      "stream": "2A", "neutral": False, "date": "2026-06-14"},
]

# Teams still in championship
R1_WINNERS = [m["winner"] for m in R1_RESULTS]
R1_LOSERS  = [m["home"] if m["winner"] == m["away"] else m["away"] for m in R1_RESULTS]


def load_ratings():
    with open(GAA_ELO_PATH) as f:
        data = json.load(f)
    return data["ratings"]


def win_prob(r_home, r_away, neutral=False):
    bonus = 0 if neutral else HOME_ADV
    return 1 / (1 + 10 ** (-((r_home + bonus) - r_away) / D_FACTOR))


def match_probs(team_a, team_b, ratings, neutral=False):
    ra = ratings.get(team_a, 1400)
    rb = ratings.get(team_b, 1400)
    # GAA has draws — use a draw band around 50%
    wp = win_prob(ra, rb, neutral)
    # Draw probability modelled as reduction from win/loss
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
        # In knockout, draw goes to extra time/replay - slight advantage to home
        return team_a if random.random() < 0.52 else team_b
    else:
        return team_b


def simulate_championship(ratings):
    # Round 2 simulation
    r2a_winners = []  # advance to QF
    r2b_winners = []  # survive to QF

    # 2A matches: both teams were R1 winners
    r2a_fixtures = [f for f in R2_FIXTURES if f["stream"] == "2A"]
    r2b_fixtures = [f for f in R2_FIXTURES if f["stream"] == "2B"]

    for f in r2a_fixtures:
        w = simulate_match(f["home"], f["away"], ratings, f["neutral"])
        r2a_winners.append(w)

    for f in r2b_fixtures:
        w = simulate_match(f["home"], f["away"], ratings, f["neutral"])
        r2b_winners.append(w)

    # Quarter-finals: 4 x 2A winners + 4 x 2B winners paired up
    # Exact QF draw TBD — for now pair by bracket position
    qf_field = []
    for i in range(len(r2a_winners)):
        if i < len(r2b_winners):
            qf_field.append((r2a_winners[i], r2b_winners[i]))

    qf_winners = []
    for a, b in qf_field:
        qf_winners.append(simulate_match(a, b, ratings, neutral=True))

    # Semi-finals (Croke Park — neutral)
    sf_winners = []
    for i in range(0, len(qf_winners), 2):
        if i + 1 < len(qf_winners):
            sf_winners.append(simulate_match(qf_winners[i], qf_winners[i+1], ratings, neutral=True))

    # Final
    if len(sf_winners) >= 2:
        champion = simulate_match(sf_winners[0], sf_winners[1], ratings, neutral=True)
    elif len(sf_winners) == 1:
        champion = sf_winners[0]
    else:
        champion = None

    return {
        "r2a_winners": r2a_winners,
        "r2b_winners": r2b_winners,
        "qf_winners":  qf_winners,
        "sf_winners":  sf_winners,
        "champion":    champion,
    }


def run_simulation(ratings):
    all_teams = list(set(
        [f["home"] for f in R2_FIXTURES] + [f["away"] for f in R2_FIXTURES]
    ))
    reach = {t: {"qf": 0, "sf": 0, "final": 0, "winner": 0} for t in all_teams}

    for _ in range(N_SIMS):
        result = simulate_championship(ratings)

        for t in result["qf_winners"]:
            if t in reach:
                reach[t]["qf"] += 1
        for t in result["sf_winners"]:
            if t in reach:
                reach[t]["sf"] += 1
        if result["champion"]:
            if result["champion"] in reach:
                reach[result["champion"]]["winner"] += 1

    probs = {}
    for team, counts in reach.items():
        probs[team] = {
            stage: round(c / N_SIMS * 100, 1)
            for stage, c in counts.items()
        }

    return probs


def get_r2_predictions(ratings):
    predictions = []
    for f in R2_FIXTURES:
        pa, pd, pb = match_probs(f["home"], f["away"], ratings, f["neutral"])
        predictions.append({
            **f,
            "prob_home": round(pa * 100, 1),
            "prob_draw": round(pd * 100, 1),
            "prob_away": round(pb * 100, 1),
            "home_elo":  ratings.get(f["home"], 1400),
            "away_elo":  ratings.get(f["away"], 1400),
        })
    return predictions


def get_gaa_data():
    ratings = load_ratings()

    probs       = run_simulation(ratings)
    predictions = get_r2_predictions(ratings)

    winner_table = sorted(
        [{"team": t, "pct": p["winner"], "sf_pct": p["sf"], "qf_pct": p["qf"]}
         for t, p in probs.items() if p["winner"] > 0],
        key=lambda x: x["pct"], reverse=True
    )

    elo_table = sorted(
        [{"team": t, "elo": round(r)} for t, r in ratings.items()
         if t not in ("London", "New York")],
        key=lambda x: x["elo"], reverse=True
    )

    return {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "r1_results":    R1_RESULTS,
        "r2_fixtures":   predictions,
        "winner_table":  winner_table,
        "elo_table":     elo_table,
        "probs":         probs,
        "ratings":       ratings,
    }


if __name__ == "__main__":
    print("Running GAA model test...")
    data = get_gaa_data()
    print(f"\nTop 10 All-Ireland winner probabilities:")
    for row in data["winner_table"][:10]:
        print(f"  {row['team']:<15} {row['pct']}%")
    print(f"\nRound 2 predictions:")
    for m in data["r2_fixtures"]:
        print(f"  {m['home']:<12} {m['prob_home']}%  |  Draw {m['prob_draw']}%  |  {m['away']:<12} {m['prob_away']}%  [{m['stream']}]")
