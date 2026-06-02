"""
FPL Performance Review — FiveStat
Flask blueprint: fpl_review.py

Register in app.py:
    from fpl_review import fpl_review_bp
    app.register_blueprint(fpl_review_bp)

Add to your navbar in base.html:
    <a href="/fpl-review">FPL Review</a>
"""

from flask import Blueprint, jsonify, render_template, request
import requests
import concurrent.futures
from collections import Counter

fpl_review_bp = Blueprint("fpl_review", __name__)

FPL   = "https://fantasy.premierleague.com/api"
HDRS  = {"User-Agent": "Mozilla/5.0 (compatible; FiveStat/1.0)"}
POS   = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Simple in-memory cache — manager_id → processed result
# Results won't change once a season ends, so this is safe.
_cache: dict = {}


def fpl_get(url: str) -> dict:
    r = requests.get(url, headers=HDRS, timeout=15)
    r.raise_for_status()
    return r.json()


@fpl_review_bp.route("/fpl-review")
def fpl_review_page():
    return render_template("fpl_review.html")


@fpl_review_bp.route("/api/fpl/review/<int:manager_id>")
def fpl_review_api(manager_id: int):

    force = request.args.get("nocache") == "1"
    if manager_id in _cache and not force:
        print(f"[FPL] Cache hit for {manager_id}")
        return jsonify(_cache[manager_id])
    if manager_id in _cache and force:
        del _cache[manager_id]
        print(f"[FPL] Cache cleared for {manager_id}")

    try:
        # ── Phase 1: bootstrap + manager info (concurrent) ────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as pool:

            boot_f  = pool.submit(fpl_get, f"{FPL}/bootstrap-static/")
            hist_f  = pool.submit(fpl_get, f"{FPL}/entry/{manager_id}/history/")
            entry_f = pool.submit(fpl_get, f"{FPL}/entry/{manager_id}/")

            boot  = boot_f.result()
            hist  = hist_f.result()

            try:
                entry = entry_f.result()
            except Exception:
                entry = {}

            players = {
                p["id"]: {
                    "name": p["web_name"],
                    "pos":  POS.get(p["element_type"], "?"),
                    "team": p["team"],
                }
                for p in boot["elements"]
            }
            teams       = {t["id"]: t["short_name"] for t in boot["teams"]}
            chips_by_gw = {c["event"]: c["name"] for c in hist.get("chips", [])}
            prev_seasons = hist.get("past", [])
            avg_score_by_gw = {e["id"]: e.get("average_entry_score", 0) for e in boot.get("events", [])}

            current_hist = hist.get("current", [])
            total_gws    = len(current_hist) if current_hist else 38

            # ── Phase 2: all GW picks + live (concurrent) ─────────────────
            def fetch_gw(gw: int):
                picks = fpl_get(f"{FPL}/entry/{manager_id}/event/{gw}/picks/")
                live  = fpl_get(f"{FPL}/event/{gw}/live/")
                return gw, picks, live

            gw_futures = [pool.submit(fetch_gw, gw) for gw in range(1, total_gws + 1)]
            gw_raw: dict = {}
            for fut in concurrent.futures.as_completed(gw_futures):
                try:
                    gw, picks, live = fut.result()
                    gw_raw[gw] = (picks, live)
                except Exception:
                    pass

        # ── Process each GW ───────────────────────────────────────────────
        gw_results = []

        for gw in range(1, total_gws + 1):
            if gw not in gw_raw:
                continue

            picks_data, live_data = gw_raw[gw]
            live_pts = {e["id"]: e["stats"]["total_points"] for e in live_data["elements"]}

            eh          = picks_data["entry_history"]
            picks       = picks_data["picks"]
            active_chip = picks_data.get("active_chip") or ""
            auto_subs   = picks_data.get("automatic_subs", [])

            squad = []
            for p in picks:
                pid  = p["element"]
                info = players.get(pid, {})
                pts  = live_pts.get(pid, 0)
                squad.append({
                    "element":     pid,
                    "name":        info.get("name", "?"),
                    "pos":         info.get("pos",  "?"),
                    "team":        teams.get(info.get("team"), "?"),
                    "slot":        p["position"],
                    "is_starter":  p["position"] <= 11,
                    "is_cap":      p["is_captain"],
                    "is_vc":       p["is_vice_captain"],
                    "multiplier":  p["multiplier"],
                    "pts":         pts,
                    "pts_counted": pts * p["multiplier"],
                })

            captain = next((p for p in squad if p["is_cap"]), None)
            vice    = next((p for p in squad if p["is_vc"]),  None)

            cap_raw  = captain["pts"]        if captain else 0
            cap_mult = captain["multiplier"] if captain else 2
            vc_raw   = vice["pts"]           if vice    else 0

            # If multiplier is 0 (captain auto-subbed out), use 2 for comparisons
            eff_mult         = cap_mult if cap_mult > 0 else 2
            cap_contribution = cap_raw * cap_mult
            armband_bonus    = cap_raw * (cap_mult - 1)

            optimal     = max(squad, key=lambda x: x["pts"])
            opt_contrib = optimal["pts"] * eff_mult
            cap_diff    = cap_contribution - opt_contrib

            enriched_subs = [
                {
                    "in":  players.get(s["element_in"],  {}).get("name", "?"),
                    "out": players.get(s["element_out"], {}).get("name", "?"),
                }
                for s in auto_subs
            ]

            gw_results.append({
                "gw":             gw,
                "active_chip":    active_chip,
                "gw_pts":         eh["points"],
                "gw_pts_gross":   eh["points"] + eh["event_transfers_cost"],
                "total_pts":      eh["total_points"],
                "overall_rank":   eh["overall_rank"],
                "gw_rank":        eh.get("rank", 0),
                "hit_cost":       eh["event_transfers_cost"],
                "transfers_made": eh["event_transfers"],
                "bench_pts":      eh["points_on_bench"],
                "bank":           eh["bank"]  / 10,
                "squad_value":    eh["value"] / 10,
                "captain":        captain["name"] if captain else "?",
                "captain_pos":    captain["pos"]  if captain else "?",
                "cap_raw_pts":    cap_raw,
                "cap_mult":       cap_mult,
                "cap_contribution":  cap_contribution,
                "armband_bonus":     armband_bonus,
                "vice":           vice["name"] if vice else "?",
                "vc_raw_pts":     vc_raw,
                "vc_better":      (vc_raw * eff_mult) > cap_contribution,
                "optimal_cap":       optimal["name"],
                "optimal_cap_pts":   optimal["pts"],
                "optimal_contrib":   opt_contrib,
                "cap_differential":  cap_diff,
                "auto_subs":      enriched_subs,
                "squad":          squad,
                "avg_gw_pts":     avg_score_by_gw.get(gw, 0),
            })

        # ── Aggregate analysis ────────────────────────────────────────────
        ok = gw_results

        cap_counter  = Counter(r["captain"] for r in ok)
        most_captained = []
        for name, count in cap_counter.most_common(12):
            gws = [r for r in ok if r["captain"] == name]
            most_captained.append({
                "name":       name,
                "count":      count,
                "total_pts":  sum(r["cap_contribution"] for r in gws),
                "avg_pts":    round(sum(r["cap_contribution"] for r in gws) / count, 1),
                "total_diff": sum(r["cap_differential"] for r in gws),
            })

        suboptimal = sorted(
            [r for r in ok if r["cap_differential"] < 0],
            key=lambda x: x["cap_differential"]
        )

        fn = entry.get("player_first_name", "")
        ln = entry.get("player_last_name",  "")

        analysis = {
            "manager_name":  f"{fn} {ln}".strip(),
            "team_name":     entry.get("name", ""),
            "season":        "2025/26",
            "prev_seasons":  prev_seasons,
            "final_pts":     ok[-1]["total_pts"]    if ok else 0,
            "final_rank":    ok[-1]["overall_rank"] if ok else 0,
            "total_gws":     len(ok),
            # Captaincy
            "total_cap_pts":          sum(r["cap_contribution"]  for r in ok),
            "total_armband_bonus":    sum(r["armband_bonus"]     for r in ok),
            "total_cap_differential": sum(r["cap_differential"]  for r in ok),
            "optimal_gws":            sum(1 for r in ok if r["cap_differential"] == 0),
            "vc_better_gws":          [r["gw"] for r in ok if r.get("vc_better")],
            "most_captained":         most_captained,
            "top_captain_weeks":      sorted(ok, key=lambda x: x["cap_contribution"], reverse=True)[:5],
            "worst_captain_weeks":    sorted(ok, key=lambda x: x["cap_contribution"])[:5],
            "biggest_cap_errors": [
                {
                    "gw":               r["gw"],
                    "captain":          r["captain"],
                    "cap_contribution": r["cap_contribution"],
                    "optimal_cap":      r["optimal_cap"],
                    "optimal_contrib":  r["optimal_contrib"],
                    "lost":             r["cap_differential"],
                }
                for r in suboptimal[:8]
            ],
            # Bench
            "total_bench_pts": sum(r["bench_pts"] for r in ok),
            "top_bench_gws":   sorted(ok, key=lambda x: x["bench_pts"], reverse=True)[:5],
            # Transfers
            "total_transfers": sum(r["transfers_made"] for r in ok),
            "total_hits":      sum(r["hit_cost"]       for r in ok),
            "hit_gws":         [{"gw": r["gw"], "cost": r["hit_cost"]} for r in ok if r["hit_cost"] > 0],
            # Chips
            "chip_summary": [
                {
                    "gw":               r["gw"],
                    "chip":             r["active_chip"],
                    "gw_pts":           r["gw_pts"],
                    "overall_rank":     r["overall_rank"],
                    "gw_rank":          r["gw_rank"],
                    "captain":          r["captain"],
                    "cap_contribution": r["cap_contribution"],
                    "bench_pts":        r["bench_pts"],
                }
                for r in ok if r["active_chip"]
            ],
            # Season arc
            "peak_rank_gw": min(ok, key=lambda x: x["overall_rank"])["gw"]          if ok else 0,
            "peak_rank":    min(ok, key=lambda x: x["overall_rank"])["overall_rank"] if ok else 0,
            "best_gw_num":  max(ok, key=lambda x: x["gw_pts"])["gw"]                if ok else 0,
            "best_gw_pts":  max(ok, key=lambda x: x["gw_pts"])["gw_pts"]            if ok else 0,
            "worst_gw_num": min(ok, key=lambda x: x["gw_pts"])["gw"]                if ok else 0,
            "worst_gw_pts": min(ok, key=lambda x: x["gw_pts"])["gw_pts"]            if ok else 0,
        }

        print(f"[FPL] Built result: {len(gw_results)} GWs, final_pts={analysis.get('final_pts')}, final_rank={analysis.get('final_rank')}")
        result = {"success": True, "gw_results": gw_results, "analysis": analysis}
        _cache[manager_id] = result
        return jsonify(result)

    except requests.HTTPError as e:
        code = e.response.status_code if e.response else 500
        if code == 404:
            return jsonify({"success": False, "error": "Manager ID not found. Double-check your ID on the FPL website."}), 404
        return jsonify({"success": False, "error": f"FPL API returned an error ({code}). Try again shortly."}), 502
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Something went wrong: {str(e)}"}), 500
