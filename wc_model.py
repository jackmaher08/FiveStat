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
"Spain", 
"France",
"England",
"Argentina", 
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



# Round of 32 winners in bracket order M73-M88 (results locked)
R32_WINNERS = [
    "Canada",       # M73: South Africa 0-1 Canada
    "Paraguay",     # M74: Germany 1(3)-1(4) Paraguay
    "Morocco",      # M75: Netherlands 1(2)-1(3) Morocco
    "Brazil",       # M76: Brazil 2-1 Japan
    "France",       # M77: France 3-0 Sweden
    "Norway",       # M78: Ivory Coast 1-2 Norway
    "Mexico",       # M79: Mexico 2-0 Ecuador
    "England",      # M80: England 2-1 DR Congo
    "USA",          # M81: USA 2-0 Bosnia and Herzegovina
    "Belgium",      # M82: Belgium 3-2 Senegal
    "Spain",        # M83: Spain 3-0 Austria
    "Portugal",     # M84: Portugal 2-1 Croatia
    "Switzerland",  # M85: Switzerland 2-0 Algeria
    "Egypt",        # M86: Australia 1(2)-1(4) Egypt
    "Argentina",    # M87: Argentina 3-2 Cape Verde
    "Colombia",     # M88: Colombia 1-0 Ghana
]


# Round of 16 winners, ordered in QF pairs (locked results)
R16_WINNERS = [
    "France",       # QF: France v Morocco
    "Morocco",
    "Spain",        # QF: Spain v Belgium
    "Belgium",
    "Norway",       # QF: Norway v England
    "England",
    "Argentina",    # QF: Argentina v Switzerland
    "Switzerland",
]


# Semi-final pairings (locked)
# M101: France v Spain | M102: England v Argentina
SF_PAIRINGS = [
    ("France", "Spain"),
    ("England", "Argentina"),
]


def simulate_tournament(ratings, finished_matches):
    def play_round(matches):
        return [
            simulate_match(a, b, ratings, allow_draw=False)[0]
            for a, b in matches
        ]

    sf_winners = play_round(SF_PAIRINGS)

    finalist_a, finalist_b = sf_winners[0], sf_winners[1]
    champion, _ = simulate_match(finalist_a, finalist_b, ratings, allow_draw=False)

    return {
        "r32_winners": [],
        "r16_winners": [],
        "qf_winners":  [],
        "sf_winners":  sf_winners,
        "finalists":   [finalist_a, finalist_b],
        "champion":    champion,
    }


def _unused_simulate_tournament(ratings, finished_matches):
    def play_round(matches):
        return [
            simulate_match(a, b, ratings, allow_draw=False)[0]
            for a, b in matches
        ]

    r32_winners = R32_WINNERS

    # Quarter-finals — pairs taken directly from R16_WINNERS order
    qf = [
        (r16_winners[0], r16_winners[1]),  # France v Morocco
        (r16_winners[2], r16_winners[3]),  # Spain v Belgium
        (r16_winners[4], r16_winners[5]),  # Norway v England
        (r16_winners[6], r16_winners[7]),  # Argentina v Switzerland
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

    eligible = [(t, p) for t, p in probs["winner"].items()
                if p > 0 and (REMAINING_TEAMS is None or t in REMAINING_TEAMS)]
    total_pct = sum(p for _, p in eligible)
    winner_table = sorted(
        [{"team": t, "pct": round(p / total_pct * 100, 1)} for t, p in eligible]
        if total_pct > 0 else [],
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
