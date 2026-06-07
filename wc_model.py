import os
import json
import random
from datetime import datetime, timezone
from wc_scraper import BASELINE_ELO

DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
WC_ELO_PATH     = os.path.join(DATA_DIR, "wc_elo.json")
WC_MATCHES_PATH = os.path.join(DATA_DIR, "wc_matches.json")

N_SIMS      = 10000
HOME_BONUS  = 0       # neutral venue tournament — no home advantage
DRAW_BASE   = 0.30    # base draw probability at ELO parity
DRAW_DECAY  = 0.002   # draw probability falls as ELO gap widens

WC_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Knockout bracket seeding: which groups feed into which R32 slots
# Format: (winner_group, runner_up_group) per match
# Based on official FIFA 2026 bracket
R32_BRACKET = [
    ("A", "w", "B", "r"),
    ("C", "w", "D", "r"),
    ("E", "w", "F", "r"),
    ("G", "w", "H", "r"),
    ("I", "w", "J", "r"),
    ("K", "w", "L", "r"),
    ("A", "r", "B", "w"),
    ("C", "r", "D", "w"),
    ("E", "r", "F", "w"),
    ("G", "r", "H", "w"),
    ("I", "r", "J", "w"),
    ("K", "r", "L", "w"),
    ("3rd_1", None, "3rd_2", None),
    ("3rd_3", None, "3rd_4", None),
    ("3rd_5", None, "3rd_6", None),
    ("3rd_7", None, "3rd_8", None),
]


def load_data():
    with open(WC_ELO_PATH) as f:
        elo_data = json.load(f)
    with open(WC_MATCHES_PATH) as f:
        match_data = json.load(f)
    return elo_data["ratings"], match_data["matches"]


def match_probs(elo_a, elo_b, ratings):
    ra = ratings.get(elo_a, 1750)
    rb = ratings.get(elo_b, 1750)
    diff = ra - rb

    win_prob  = 1 / (1 + 10 ** (-diff / 400))
    draw_prob = max(0.05, DRAW_BASE - DRAW_DECAY * abs(diff))
    win_prob  = win_prob  * (1 - draw_prob)
    loss_prob = (1 - draw_prob) * (1 - win_prob / (1 - draw_prob))

    win_a  = win_prob
    draw   = draw_prob
    win_b  = loss_prob

    total = win_a + draw + win_b
    return round(win_a / total, 4), round(draw / total, 4), round(win_b / total, 4)


def simulate_match(team_a, team_b, ratings, allow_draw=True):
    pa, pd, pb = match_probs(team_a, team_b, ratings)
    if not allow_draw:
        pa_adj = pa / (pa + pb)
        r = random.random()
        return team_a if r < pa_adj else team_b, None
    r = random.random()
    if r < pa:
        return team_a, team_b
    elif r < pa + pd:
        return None, None
    else:
        return team_b, team_a


def simulate_group(group_teams, ratings):
    points  = {t: 0 for t in group_teams}
    gd      = {t: 0 for t in group_teams}
    gf      = {t: 0 for t in group_teams}

    fixtures = [
        (group_teams[0], group_teams[1]),
        (group_teams[2], group_teams[3]),
        (group_teams[0], group_teams[2]),
        (group_teams[1], group_teams[3]),
        (group_teams[0], group_teams[3]),
        (group_teams[1], group_teams[2]),
    ]

    for a, b in fixtures:
        winner, loser = simulate_match(a, b, ratings, allow_draw=True)
        if winner is None:
            points[a] += 1
            points[b] += 1
        else:
            points[winner] += 3

    standings = sorted(group_teams, key=lambda t: points[t], reverse=True)
    return standings, points


def simulate_tournament(ratings, finished_matches):
    group_results = {}
    third_place_teams = []

    for group, teams in WC_GROUPS.items():
        overrides = {t: {"w": 0, "d": 0, "l": 0, "pts": 0} for t in teams}

        for m in finished_matches:
            if m.get("group") not in (f"Group {group}", f"GROUP_{group}"):
                continue
            h, a = m["home"], m["away"]
            hg, ag = m["home_goals"], m["away_goals"]
            if h not in overrides or a not in overrides:
                continue
            if hg > ag:
                overrides[h]["pts"] += 3
            elif hg == ag:
                overrides[h]["pts"] += 1
                overrides[a]["pts"] += 1
            else:
                overrides[a]["pts"] += 3

        played_pairs = set()
        for m in finished_matches:
            if m.get("group") in (f"Group {group}", f"GROUP_{group}"):
                played_pairs.add((m["home"], m["away"]))

        sim_pts = {t: overrides[t]["pts"] for t in teams}

        remaining = [
            (a, b) for i, a in enumerate(teams)
            for b in teams[i+1:]
            if (a, b) not in played_pairs and (b, a) not in played_pairs
        ]

        for a, b in remaining:
            winner, _ = simulate_match(a, b, ratings, allow_draw=True)
            if winner is None:
                sim_pts[a] += 1
                sim_pts[b] += 1
            else:
                sim_pts[winner] += 3

        standings = sorted(teams, key=lambda t: (sim_pts[t], ratings.get(t, 1750)), reverse=True)
        group_results[group] = {
            "winner":      standings[0],
            "runner_up":   standings[1],
            "third":       standings[2],
            "third_pts":   sim_pts[standings[2]],
            "standings":   standings,
            "points":      sim_pts,
        }
        third_place_teams.append((standings[2], sim_pts[standings[2]], group))

    third_place_teams.sort(key=lambda x: (x[1], ratings.get(x[0], 1750)), reverse=True)
    best_thirds = [t[0] for t in third_place_teams[:8]]

    r32_field = []
    groups_list = list(WC_GROUPS.keys())

    for i in range(0, 12, 2):
        g1, g2 = groups_list[i], groups_list[i+1]
        r32_field.append((group_results[g1]["winner"],   group_results[g2]["runner_up"]))
        r32_field.append((group_results[g2]["winner"],   group_results[g1]["runner_up"]))

    for i in range(0, 8, 2):
        r32_field.append((best_thirds[i], best_thirds[i+1]))

    def play_round(matches):
        winners = []
        for a, b in matches:
            w, _ = simulate_match(a, b, ratings, allow_draw=False)
            winners.append(w)
        return winners

    r32_winners = play_round(r32_field)

    r16_field = [(r32_winners[i], r32_winners[i+1]) for i in range(0, 16, 2)]
    r16_winners = play_round(r16_field)

    qf_field = [(r16_winners[i], r16_winners[i+1]) for i in range(0, 8, 2)]
    qf_winners = play_round(qf_field)

    sf_field = [(qf_winners[i], qf_winners[i+1]) for i in range(0, 4, 2)]
    sf_winners = play_round(sf_field)

    finalist_a, finalist_b = sf_winners[0], sf_winners[1]
    champion, _ = simulate_match(finalist_a, finalist_b, ratings, allow_draw=False)

    return {
        "group_results":  group_results,
        "r32_winners":    r32_winners,
        "r16_winners":    r16_winners,
        "qf_winners":     qf_winners,
        "sf_winners":     sf_winners,
        "finalists":      [finalist_a, finalist_b],
        "champion":       champion,
    }


def run_simulation(ratings, finished_matches):
    reach = {
        "r32":    {},
        "r16":    {},
        "qf":     {},
        "sf":     {},
        "final":  {},
        "winner": {},
    }
    group_qualify = {g: {t: 0 for t in teams} for g, teams in WC_GROUPS.items()}

    all_teams = [t for teams in WC_GROUPS.values() for t in teams]
    for stage in reach:
        for team in all_teams:
            reach[stage][team] = 0

    for _ in range(N_SIMS):
        result = simulate_tournament(ratings, finished_matches)

        for group, gdata in result["group_results"].items():
            group_qualify[group][gdata["winner"]]    += 1
            group_qualify[group][gdata["runner_up"]] += 1

        for team in result["r32_winners"]:
            reach["r32"][team] += 1
        for team in result["r16_winners"]:
            reach["r16"][team] += 1
        for team in result["qf_winners"]:
            reach["qf"][team] += 1
        for team in result["sf_winners"]:
            reach["sf"][team] += 1
        for team in result["finalists"]:
            reach["final"][team] += 1
        reach["winner"][result["champion"]] += 1

    probs = {}
    for stage, counts in reach.items():
        probs[stage] = {t: round(c / N_SIMS * 100, 1) for t, c in counts.items()}

    qualify_probs = {}
    for group, counts in group_qualify.items():
        qualify_probs[group] = {t: round(c / N_SIMS * 100, 1) for t, c in counts.items()}

    return probs, qualify_probs


def get_match_predictions(ratings, matches):
    predictions = []
    for m in matches:
        if m["status"] == "FINISHED":
            predictions.append({**m, "predicted": False})
            continue
        pa, pd, pb = match_probs(m["home"], m["away"], ratings)
        predictions.append({
            **m,
            "predicted": True,
            "prob_home": round(pa * 100, 1),
            "prob_draw": round(pd * 100, 1),
            "prob_away": round(pb * 100, 1),
        })
    return predictions


def get_wc_data():
    ratings, matches = load_data()
    finished = [m for m in matches if m["status"] == "FINISHED"]

    probs, qualify_probs = run_simulation(ratings, finished)
    predictions = get_match_predictions(ratings, matches)

    winner_table = sorted(
        [{"team": t, "pct": p} for t, p in probs["winner"].items() if p > 0],
        key=lambda x: x["pct"], reverse=True
    )

    groups_out = {}
    for group, teams in WC_GROUPS.items():
        group_matches = [m for m in predictions if m.get("group") in (f"Group {group}", f"GROUP_{group}")]
        group_matches.sort(key=lambda m: m["utcDate"])

        standings = []
        pts  = {t: 0 for t in teams}
        played = {t: 0 for t in teams}
        for m in group_matches:
            if m["status"] != "FINISHED":
                continue
            h, a, hg, ag = m["home"], m["away"], m["home_goals"], m["away_goals"]
            played[h] += 1
            played[a] += 1
            if hg > ag:
                pts[h] += 3
            elif hg == ag:
                pts[h] += 1
                pts[a] += 1
            else:
                pts[a] += 3

        for team in teams:
            standings.append({
                "team":    team,
                "played":  played[team],
                "pts":     pts[team],
                "elo":     ratings.get(team, 0) or 0,
                "qualify": qualify_probs[group].get(team, 0),
                "win":     probs["winner"].get(team, 0),
            })
        standings.sort(key=lambda x: (x["pts"], x["elo"]), reverse=True)

        groups_out[group] = {
            "teams":   standings,
            "matches": group_matches,
        }

    return {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "winner_table":  winner_table,
        "groups":        groups_out,
        "probs":         probs,
        "qualify_probs": qualify_probs,
        "ratings":       ratings,
        "predictions":   predictions,
    }


if __name__ == "__main__":
    print("Running WC model test...")
    data = get_wc_data()
    print(f"\nTop 10 WC 2026 winner probabilities:")
    for row in data["winner_table"][:10]:
        print(f"  {row['team']:<25} {row['pct']}%")
    print(f"\nGroup A qualify probabilities:")
    for team in data["groups"]["A"]["teams"]:
        print(f"  {team['team']:<25} qualify: {team['qualify']}%")
