"""
FPL Full Season Data Fetcher & Analyser
Manager ID: 980868

Outputs:
  fpl_full_data.json     — complete raw data, every GW
  fpl_gw_summary.csv     — one row per GW, spreadsheet-ready
  fpl_squad_by_gw.csv    — one row per player per GW (150 x 38 rows)

Run: pip install requests  →  python fpl_captain_fetch.py
"""

import requests, json, time, csv
from collections import Counter

MANAGER_ID   = 980868
TOTAL_GWS    = 38
HEADERS      = {"User-Agent": "Mozilla/5.0 (compatible; FPL-fetch/1.0)"}
POS          = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()

def fmt(n, prefix=""):
    return f"{prefix}{n:,}"

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print("Fetching bootstrap (players + teams)...")
boot    = fetch("https://fantasy.premierleague.com/api/bootstrap-static/")
players = {
    p["id"]: {
        "name":     p["web_name"],
        "full":     f"{p['first_name']} {p['second_name']}",
        "pos":      POS[p["element_type"]],
        "team_id":  p["team"],
        "cost":     p["now_cost"] / 10,
    }
    for p in boot["elements"]
}
teams = {t["id"]: t["short_name"] for t in boot["teams"]}
print(f"  {len(players)} players · {len(teams)} teams\n")

# ── Manager history (chips) ───────────────────────────────────────────────────
print("Fetching manager history...")
hist        = fetch(f"https://fantasy.premierleague.com/api/entry/{MANAGER_ID}/history/")
chips_by_gw = {c["event"]: c["name"] for c in hist.get("chips", [])}
print(f"  Chips: {chips_by_gw}\n")

# ── Per-GW fetch ──────────────────────────────────────────────────────────────
gw_results = []

for gw in range(1, TOTAL_GWS + 1):
    try:
        pd   = fetch(f"https://fantasy.premierleague.com/api/entry/{MANAGER_ID}/event/{gw}/picks/")
        live = fetch(f"https://fantasy.premierleague.com/api/event/{gw}/live/")

        live_pts   = {e["id"]: e["stats"]["total_points"]  for e in live["elements"]}
        live_stats = {e["id"]: e["stats"]                  for e in live["elements"]}

        eh           = pd["entry_history"]
        active_chip  = pd.get("active_chip") or ""
        picks        = pd["picks"]
        auto_subs    = pd.get("automatic_subs", [])

        gw_pts       = eh["points"]
        total_pts    = eh["total_points"]
        overall_rank = eh["overall_rank"]
        hit_cost     = eh["event_transfers_cost"]
        bench_pts    = eh["points_on_bench"]
        n_transfers  = eh["event_transfers"]
        bank         = eh["bank"]  / 10
        value        = eh["value"] / 10

        # ── Squad ──────────────────────────────────────────────────────────
        squad = []
        for p in picks:
            pid  = p["element"]
            info = players.get(pid, {})
            pts  = live_pts.get(pid, 0)
            squad.append({
                "element":       pid,
                "name":          info.get("name", "?"),
                "full":          info.get("full",  "?"),
                "pos":           info.get("pos",   "?"),
                "team":          teams.get(info.get("team_id"), "?"),
                "cost":          info.get("cost", 0),
                "slot":          p["position"],          # 1-11 = XI, 12-15 = bench
                "is_starter":    p["position"] <= 11,
                "is_cap":        p["is_captain"],
                "is_vc":         p["is_vice_captain"],
                "multiplier":    p["multiplier"],        # 0=benched, 1=playing, 2=cap, 3=TC
                "pts":           pts,                    # raw points (no multiplier)
                "pts_counted":   pts * p["multiplier"],  # what this player actually contributed
                "stats":         live_stats.get(pid, {}),
            })

        captain = next((p for p in squad if p["is_cap"]), None)
        vice    = next((p for p in squad if p["is_vc"]),  None)

        cap_raw  = captain["pts"]        if captain else 0
        cap_mult = captain["multiplier"] if captain else 2
        vc_raw   = vice["pts"]           if vice    else 0

        cap_contribution   = cap_raw * cap_mult
        armband_bonus      = cap_raw * (cap_mult - 1)   # extra vs playing normally as non-cap

        # Optimal captain = squad member with highest raw score this GW
        # (apply same chip multiplier to show comparable value)
        optimal      = max(squad, key=lambda x: x["pts"])
        opt_contrib  = optimal["pts"] * cap_mult
        cap_diff     = cap_contribution - opt_contrib    # 0 = optimal, negative = pts lost

        # What VC returned if captain blanked (i.e. cap scored ≤ vc_raw raw pts)
        vc_would_have_been_better = (vc_raw * cap_mult) > cap_contribution

        # XI / bench breakdown
        xi_pts       = sum(p["pts_counted"] for p in squad if p["is_starter"])
        bench_raw    = sum(p["pts"]         for p in squad if not p["is_starter"])

        # Auto-sub enrichment (add names)
        enriched_subs = [
            {
                "in":  players.get(s["element_in"],  {}).get("name", "?"),
                "out": players.get(s["element_out"], {}).get("name", "?"),
            }
            for s in auto_subs
        ]

        row = {
            "gw":            gw,
            "active_chip":   active_chip,
            # Points
            "gw_pts":        gw_pts,
            "gw_pts_gross":  gw_pts + hit_cost,     # before hit deduction
            "total_pts":     total_pts,
            "xi_pts":        xi_pts,
            "bench_pts":     bench_pts,
            # Rank
            "overall_rank":  overall_rank,
            # Transfers
            "transfers_made":  n_transfers,
            "hit_cost":        hit_cost,
            # Finance
            "bank":          bank,
            "squad_value":   value,
            # Captain
            "captain":       captain["name"] if captain else "?",
            "captain_full":  captain["full"] if captain else "?",
            "captain_pos":   captain["pos"]  if captain else "?",
            "cap_raw_pts":   cap_raw,
            "cap_mult":      cap_mult,
            "cap_contribution":  cap_contribution,
            "armband_bonus":     armband_bonus,
            # VC
            "vice":          vice["name"] if vice else "?",
            "vice_full":     vice["full"] if vice else "?",
            "vc_raw_pts":    vc_raw,
            "vc_better_than_cap": vc_would_have_been_better,
            # Optimal captain
            "optimal_cap":      optimal["name"],
            "optimal_cap_pts":  optimal["pts"],
            "optimal_contrib":  opt_contrib,
            "cap_differential": cap_diff,    # 0 or negative
            # Auto subs
            "auto_subs":     enriched_subs,
            # Full squad (15 players)
            "squad":         squad,
        }

        gw_results.append(row)

        # ── Console line ──────────────────────────────────────────────────
        chip_str = f"[{active_chip}]" if active_chip else ""
        opt_str  = f"| opt: {optimal['name']} {opt_contrib}" if cap_diff < 0 else "| ✓ optimal"
        print(
            f"GW{gw:02d} {chip_str:<10}"
            f"Cap: {(captain['name'] if captain else '?'):<17}"
            f"{cap_raw:>3}x{cap_mult}={cap_contribution:>3}"
            f"  diff {cap_diff:>+4}  {opt_str}"
            f"  GW:{gw_pts:>3}  OR:{overall_rank:>9,}"
        )

    except Exception as e:
        import traceback
        print(f"GW{gw:02d} ERROR: {e}")
        traceback.print_exc()
        gw_results.append({"gw": gw, "error": str(e)})

    time.sleep(0.4)

# ── Analysis ──────────────────────────────────────────────────────────────────
ok = [r for r in gw_results if "error" not in r]
print("\n" + "="*70)
print("FULL SEASON ANALYSIS")
print("="*70)

# Captain stats
total_cap_pts    = sum(r["cap_contribution"]  for r in ok)
total_bonus      = sum(r["armband_bonus"]     for r in ok)
total_cap_diff   = sum(r["cap_differential"]  for r in ok)
optimal_gws      = sum(1 for r in ok if r["cap_differential"] == 0)
suboptimal_gws   = [r for r in ok if r["cap_differential"] < 0]
vc_saves         = [r for r in ok if r["vc_better_than_cap"]]

print(f"\n── Captaincy ───────────────────────────────────────────────────────")
print(f"Total armband contribution : {total_cap_pts} pts")
print(f"Extra from doubling/tripling: {total_bonus} pts")
print(f"Optimal captain chosen     : {optimal_gws}/{len(ok)} GWs")
print(f"Pts lost to sub-optimal cap: {total_cap_diff} pts over {len(suboptimal_gws)} GWs")
print(f"GWs where VC > captain     : {len(vc_saves)} ({[r['gw'] for r in vc_saves]})")

print("\nTop 5 captain weeks:")
for r in sorted(ok, key=lambda x: x["cap_contribution"], reverse=True)[:5]:
    print(f"  GW{r['gw']:02d}  {r['captain']:<18} {r['cap_raw_pts']:>3} x{r['cap_mult']} = {r['cap_contribution']:>3} pts")

print("\nBottom 5 captain weeks:")
for r in sorted(ok, key=lambda x: x["cap_contribution"])[:5]:
    print(f"  GW{r['gw']:02d}  {r['captain']:<18} {r['cap_raw_pts']:>3} x{r['cap_mult']} = {r['cap_contribution']:>3} pts")

print("\nBiggest captaincy errors (pts lost vs in-squad optimal):")
for r in sorted(suboptimal_gws, key=lambda x: x["cap_differential"])[:7]:
    print(f"  GW{r['gw']:02d}  Capped {r['captain']:<16} {r['cap_contribution']:>3}pts"
          f"  |  Optimal: {r['optimal_cap']:<16} {r['optimal_contrib']:>3}pts"
          f"  |  Lost: {r['cap_differential']:>+4}")

print("\nMost captained players:")
cap_counts = Counter(r["captain"] for r in ok)
for name, count in cap_counts.most_common(8):
    pts  = sum(r["cap_contribution"]  for r in ok if r["captain"] == name)
    diff = sum(r["cap_differential"]  for r in ok if r["captain"] == name)
    avg  = pts / count
    print(f"  {name:<20} {count:>2}x   {pts:>4} pts   avg {avg:>5.1f}/GW   total diff {diff:>+5}")

print(f"\n── Bench ───────────────────────────────────────────────────────────")
total_bench = sum(r["bench_pts"] for r in ok)
print(f"Total pts left on bench : {total_bench}")
print("\nTop 5 bench hauls:")
for r in sorted(ok, key=lambda x: x["bench_pts"], reverse=True)[:5]:
    bench_players = [p for p in r.get("squad", []) if not p["is_starter"]]
    bp_str = ", ".join(f"{p['name']} {p['pts']}" for p in bench_players)
    print(f"  GW{r['gw']:02d}  {r['bench_pts']:>3} pts on bench  [{bp_str}]")

print(f"\n── Transfers & Hits ────────────────────────────────────────────────")
total_hits = sum(r["hit_cost"] for r in ok)
hit_gws    = [(r["gw"], r["hit_cost"]) for r in ok if r["hit_cost"] > 0]
total_xfers = sum(r["transfers_made"] for r in ok)
print(f"Total transfers     : {total_xfers}")
print(f"Total hit deductions: -{total_hits} pts")
print(f"Hit GWs             : {hit_gws}")

print(f"\n── Chips ───────────────────────────────────────────────────────────")
for r in ok:
    if r["active_chip"]:
        subs_str = " | AutoSubs: " + ", ".join(f"{s['out']}→{s['in']}" for s in r["auto_subs"]) if r["auto_subs"] else ""
        print(f"  GW{r['gw']:02d} [{r['active_chip']:<12}]"
              f"  {r['gw_pts']:>3}pts (gross {r['gw_pts_gross']})"
              f"  Cap: {r['captain']:<16} {r['cap_contribution']:>3}pts"
              f"  Bench: {r['bench_pts']:>3}pts"
              f"  OR: {r['overall_rank']:,}"
              f"{subs_str}")

print(f"\n── Season arc ──────────────────────────────────────────────────────")
peak_gw  = min(ok, key=lambda x: x["overall_rank"])
worst_gw = max(ok, key=lambda x: x["overall_rank"])
print(f"Peak rank   : GW{peak_gw['gw']:02d}  OR {peak_gw['overall_rank']:,}")
print(f"Lowest rank : GW{worst_gw['gw']:02d}  OR {worst_gw['overall_rank']:,}")
print(f"Best GW     : GW{max(ok, key=lambda x: x['gw_pts'])['gw']:02d}  {max(ok, key=lambda x: x['gw_pts'])['gw_pts']} pts")
print(f"Worst GW    : GW{min(ok, key=lambda x: x['gw_pts'])['gw']:02d}  {min(ok, key=lambda x: x['gw_pts'])['gw_pts']} pts")

# ── Save outputs ──────────────────────────────────────────────────────────────

# 1. Full JSON
with open("fpl_full_data.json", "w") as f:
    json.dump(gw_results, f, indent=2)
print("\nSaved: fpl_full_data.json")

# 2. GW summary CSV
gw_csv_fields = [
    "gw","active_chip",
    "gw_pts","gw_pts_gross","total_pts","xi_pts","bench_pts",
    "overall_rank","transfers_made","hit_cost","bank","squad_value",
    "captain","captain_pos","cap_raw_pts","cap_mult","cap_contribution","armband_bonus",
    "vice","vc_raw_pts","vc_better_than_cap",
    "optimal_cap","optimal_cap_pts","optimal_contrib","cap_differential",
]
with open("fpl_gw_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=gw_csv_fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(ok)
print("Saved: fpl_gw_summary.csv")

# 3. Squad-by-GW CSV (all 15 players per GW)
squad_fields = ["gw","active_chip","name","pos","team","cost","slot",
                "is_starter","is_cap","is_vc","multiplier","pts","pts_counted"]
with open("fpl_squad_by_gw.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=squad_fields, extrasaction="ignore")
    w.writeheader()
    for r in ok:
        for p in r.get("squad", []):
            row = {k: p.get(k, "") for k in squad_fields if k not in ("gw", "active_chip")}
            row["gw"] = r["gw"]
            row["active_chip"] = r["active_chip"]
            w.writerow(row)
print("Saved: fpl_squad_by_gw.csv")

print("\nDone.\n")
