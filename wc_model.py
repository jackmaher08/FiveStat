import os
import json
import random
from datetime import datetime, timezone

DISPLAY_NAME_MAP = {
    "Cape Verde Islands": "Cabo Verde",
    "Czechia":            "Czech Republic",
}


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

# Teams still alive in the knockout stage - tb remove as they are eliminated.
# Set to None to show all teams (pre-knockout behaviour).
REMAINING_TEAMS = [
    "Brazil", "Norway", "Croatia", "Spain", "Paraguay", "France",
    "Canada", "Morocco", "USA", "Belgium", "Mexico", "England",
    "Argentina", "Australia", "Switzerland", "Colombia", "Portugal",
    "Croatia", "Switzerland", "Algeria", "Australia", "Egypt", "Argentina", 
    "Cape Verde", "Colombia", "Ghana"
    # R32 in progress — update after each round
]

BASELINE_ELO = {
    "Spain":                  2165,
    "Argentina":              2113,
    "France":                 2081,
    "England":                2020,
    "Brazil":                 1988,
    "Portugal":               1984,
    "Colombia":               1977,
    "Netherlands":            1944,
    "Ecuador":                1935,
    "Germany":                1925,
    "Norway":                 1917,
    "Croatia":                1908,
    "Turkey":                 1906,
    "Japan":                  1906,
    "Switzerland":            1894,
    "Uruguay":                1892,
    "Belgium":                1888,
    "Mexico":                 1867,
    "Senegal":                1867,
    "Denmark":                1864,
    "Paraguay":               1832,
    "Austria":                1830,
    "Morocco":                1824,
    "Canada":                 1793,
    "Ukraine":                1785,
    "Australia":              1774,
    "Scotland":               1770,
    "Nigeria":                1770,
    "Iran":                   1764,
    "Algeria":                1760,
    "South Korea":            1756,
    "Serbia":                 1742,
    "Czech Republic":         1733,
    "USA":                    1733,
    "Panama":                 1733,
    "Venezuela":              1727,
    "Sweden":                 1714,
    "Egypt":                  1699,
    "Slovenia":               1694,
    "Jordan":                 1685,
    "Ivory Coast":            1676,
    "Slovakia":               1674,
    "Congo DR":               1661,
    "Romania":                1630,
    "Cameroon":               1613,
    "Iraq":                   1608,
    "Bosnia and Herzegovina": 1591,
    "Cabo Verde":             1576,
    "Saudi Arabia":           1566,
    "New Zealand":            1563,
    "Haiti":                  1554,
    "South Africa":           1518,
    "Uzbekistan":             1718,
    "Ghana":                  1510,
    "Curacao":                1433,
    "Qatar":                  1423,
    "Kenya":                  1356,
}


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



def simulate_tournament(ratings, finished_matches):
    group_results = {}

    for group, teams in WC_GROUPS.items():
        overrides = {t: {"pts": 0} for t in teams}

        for m in finished_matches:
            if m.get("group") not in (f"Group {group}", f"GROUP_{group}"):
                continue
            h, a = m["home"], m["away"]
            hg, ag = m["home_goals"], m["away_goals"]
            if h not in overrides or a not in overrides:
                continue
            if hg is None or ag is None:
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
                if m.get("home_goals") is None or m.get("away_goals") is None:
                    continue
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
            "winner":    standings[0],
            "runner_up": standings[1],
            "third":     standings[2],
            "third_pts": sim_pts[standings[2]],
        }

    # Pick 8 best third-place teams by points then ELO
    all_thirds = sorted(
        [(g, d["third"], d["third_pts"]) for g, d in group_results.items()],
        key=lambda x: (x[2], ratings.get(x[1], 1750)),
        reverse=True
    )
    best_thirds = [t[1] for t in all_thirds[:8]]
    random.shuffle(best_thirds)

    def w(group, pos):
        return group_results[group]["winner"] if pos == "w" else group_results[group]["runner_up"]

    def t(i):
        return best_thirds[i]

    # Round of 32 — exact FIFA 2026 bracket
    r32 = [
        (w("A", "r"), w("B", "r")),   # M73
        (w("E", "w"), t(0)),           # M74 — 3rd ABCDF
        (w("F", "w"), w("C", "r")),   # M75
        (w("C", "w"), w("F", "r")),   # M76
        (w("I", "w"), t(1)),           # M77 — 3rd CDFGH
        (w("E", "r"), w("I", "r")),   # M78
        (w("A", "w"), t(2)),           # M79 — 3rd CEFHI
        (w("L", "w"), t(3)),           # M80 — 3rd EHIJK
        (w("D", "w"), t(4)),           # M81 — 3rd BEFIJ
        (w("G", "w"), t(5)),           # M82 — 3rd AEHIJ
        (w("K", "r"), w("L", "r")),   # M83
        (w("H", "w"), w("J", "r")),   # M84
        (w("B", "w"), t(6)),           # M85 — 3rd EFGIJ
        (w("J", "w"), w("H", "r")),   # M86
        (w("K", "w"), t(7)),           # M87 — 3rd DEIJL
        (w("D", "r"), w("G", "r")),   # M88
    ]

    def play_round(matches):
        return [
            simulate_match(a, b, ratings, allow_draw=False)[0]
            for a, b in matches
        ]

    r32_winners = play_round(r32)

    # Round of 16 — winners progress in bracket order
    # M89: W74 v W77 | M90: W73 v W75 | M91: W76 v W78 | M92: W79 v W80
    # M93: W83 v W84 | M94: W81 v W82 | M95: W86 v W88 | M96: W85 v W87
    r16 = [
        (r32_winners[1],  r32_winners[4]),   # M89: W74 v W77
        (r32_winners[0],  r32_winners[2]),   # M90: W73 v W75
        (r32_winners[3],  r32_winners[5]),   # M91: W76 v W78
        (r32_winners[6],  r32_winners[7]),   # M92: W79 v W80
        (r32_winners[10], r32_winners[11]),  # M93: W83 v W84
        (r32_winners[8],  r32_winners[9]),   # M94: W81 v W82
        (r32_winners[13], r32_winners[15]),  # M95: W86 v W88
        (r32_winners[12], r32_winners[14]),  # M96: W85 v W87
    ]
    r16_winners = play_round(r16)

    # Quarter-finals
    # M97: W89 v W90 | M98: W93 v W94 | M99: W91 v W92 | M100: W95 v W96
    qf = [
        (r16_winners[0], r16_winners[1]),  # M97
        (r16_winners[4], r16_winners[5]),  # M98
        (r16_winners[2], r16_winners[3]),  # M99
        (r16_winners[6], r16_winners[7]),  # M100
    ]
    qf_winners = play_round(qf)

    # Semi-finals
    # M101: W97 v W98 | M102: W99 v W100
    sf = [
        (qf_winners[0], qf_winners[1]),  # M101
        (qf_winners[2], qf_winners[3]),  # M102
    ]
    sf_winners = play_round(sf)

    finalist_a, finalist_b = sf_winners[0], sf_winners[1]
    champion, _ = simulate_match(finalist_a, finalist_b, ratings, allow_draw=False)

    return {
        "group_results": group_results,
        "r32_winners":   r32_winners,
        "r16_winners":   r16_winners,
        "qf_winners":    qf_winners,
        "sf_winners":    sf_winners,
        "finalists":     [finalist_a, finalist_b],
        "champion":      champion,
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
        m = {**m,
             "home": DISPLAY_NAME_MAP.get(m["home"], m["home"]),
             "away": DISPLAY_NAME_MAP.get(m["away"], m["away"])}
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
        [{"team": t, "pct": p} for t, p in probs["winner"].items()
         if p > 0 and (REMAINING_TEAMS is None or t in REMAINING_TEAMS)],
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
            if hg is None or ag is None:
                continue
            if h in played:
                played[h] += 1
            if a in played:
                played[a] += 1
            if hg > ag:
                if h in pts: pts[h] += 3
            elif hg == ag:
                if h in pts: pts[h] += 1
                if a in pts: pts[a] += 1
            else:
                if a in pts: pts[a] += 3

        for team in teams:
            standings.append({
                "team":    team,
                "played":  played[team],
                "pts":     pts[team],
                "elo":     ratings.get(team, 0) or 0,
                "qualify": qualify_probs[group].get(team, 0),
                "win":     probs["winner"].get(team, 0),
            })
        standings.sort(key=lambda x: (x["pts"], x["played"], x["elo"]), reverse=True)
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
