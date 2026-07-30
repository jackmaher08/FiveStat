"""
backtest.py — Walk-forward backtest for FiveStat match prediction model.

Methodology:
  - Test window: 2023/24, 2024/25, 2025/26 combined (all completed fixtures)
  - For each gameweek in the test window, ratings are calculated using ONLY data
    available up to that point (all prior seasons + completed GWs in the window).
  - Predictions are generated for that GW's fixtures, then compared against
    actual results.
  - This mirrors exactly how the model operates in production — no future
    data leaks into any prediction.

Metrics computed:
  1. Outcome accuracy        — % correct most-likely outcome (H/D/A)
  2. Ranked Probability Score (RPS) — probabilistic accuracy, lower = better
  3. Brier Score             — calibration of win probability
  4. Over/Under 2.5 accuracy — binary market accuracy
  5. xG MAE                  — mean absolute error on expected goals vs actual
  6. Correct score hit rate  — % where top heatmap cell = actual scoreline

Output: data/tables/model_accuracy.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson

# ── Allow imports from data_loader without triggering its module-level code ──
# We import only the functions we need directly
from data_loader import (
    calculate_team_statistics,
    calculate_recent_form,
    get_team_xg,
    simulate_bivariate_poisson,
    simulate_bivariate_nb,
    dixon_coles_correction,
    TEAM_NAME_MAPPING,
    MANUAL_XG_ADJUSTMENTS,
    MANUAL_XGA_ADJUSTMENTS,
)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

TEST_SEASON_START = "2022-08-01"   # TUNING window: 2022/23 to 2024/25 — 25/26 held out
TEST_SEASON_END   = "2025-06-01"   # excludes 2025/26 entirely, clean holdout
MIN_TRAIN_MATCHES = 100            # Minimum training matches before predicting
OUTPUT_PATH       = "data/tables/model_accuracy.json"
ALPHA             = 0.30           # Form blending weight — matches production
COV_XY            = 0.05           # Bivariate Poisson covariance — matches production
RHO               = 0.0            # Dixon-Coles correction — matches production. Removed
                                    # in favour of NB dispersion handling low-score
                                    # correction (see DISPERSION note below).
DISPERSION        = 1.45            # NB overdispersion factor — matches production. Fixes
                                    # severe 1-1 heatmap over-concentration (87.6% -> 36.8%
                                    # on live fixtures) at a real calibration cost (0.1pp ->
                                    # 5.3pp on holdout) — judged worth it given how visible
                                    # and severe the unfixed state was in production.

# ══════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════

def ranked_probability_score(probs, outcome_idx):
    """
    Compute RPS for a single match.
    probs: [home_win_prob, draw_prob, away_win_prob]
    outcome_idx: 0=home win, 1=draw, 2=away win
    Lower RPS = better. Range: 0 to 1.
    """
    actual = np.zeros(3)
    actual[outcome_idx] = 1.0
    cum_pred   = np.cumsum(probs)
    cum_actual = np.cumsum(actual)
    return np.sum((cum_pred - cum_actual) ** 2) / 2.0


def brier_score(prob, outcome):
    """
    Brier score for a single binary outcome.
    prob: predicted probability of outcome occurring
    outcome: 1 if it occurred, 0 if not
    Lower = better. Range: 0 to 1.
    """
    return (prob - outcome) ** 2


def naive_rps(outcome_idx):
    """RPS for a naive equal-probability baseline (1/3, 1/3, 1/3)."""
    return ranked_probability_score([1/3, 1/3, 1/3], outcome_idx)


def home_always_wins_accuracy(matches_df):
    """Accuracy of always predicting home win — the simplest baseline."""
    return (matches_df["actual_outcome"] == "home_win").mean()


# ══════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════

def predict_fixture(home_team, away_team, training_data, team_name_map):
    """
    Generate match probabilities and scoreline matrix for a single fixture
    using only the provided training_data slice.

    Returns dict with home_xg, away_xg, result_matrix, home_win_prob,
    draw_prob, away_win_prob — or None if either team lacks sufficient data.
    """
    # Normalise team names
    home_team = team_name_map.get(home_team, home_team)
    away_team = team_name_map.get(away_team, away_team)

    # Check both teams have data
    teams_in_data = set(training_data["Home Team"].unique()) | set(training_data["Away Team"].unique())
    if home_team not in teams_in_data or away_team not in teams_in_data:
        return None

    try:
        team_stats, team_home_advantage = calculate_team_statistics(
            training_data, save_csv_path=None, verbose=False  # Don't save/print during backtest
        )
        recent_form_att, recent_form_def = calculate_recent_form(
            training_data, team_stats, recent_matches=15, alpha=ALPHA
        )
    except Exception as e:
        return None

    if home_team not in team_stats or away_team not in team_stats:
        return None

    try:
        home_xg = get_team_xg(
            team=home_team, opponent=away_team, is_home=True,
            team_stats=team_stats, recent_form_att=recent_form_att,
            recent_form_def=recent_form_def, alpha=ALPHA,
            team_home_advantage=team_home_advantage
        )
        away_xg = get_team_xg(
            team=away_team, opponent=home_team, is_home=False,
            team_stats=team_stats, recent_form_att=recent_form_att,
            recent_form_def=recent_form_def, alpha=ALPHA,
            team_home_advantage=team_home_advantage
        )
    except Exception as e:
        return None

    result_matrix, home_win_prob, draw_prob, away_win_prob = simulate_bivariate_nb(
        home_xg, away_xg, cov_xy=COV_XY, dispersion=DISPERSION
    )
    result_matrix = dixon_coles_correction(result_matrix, home_xg, away_xg, rho=RHO)
    home_win_prob = float(np.sum(np.tril(result_matrix, -1)))
    draw_prob     = float(np.sum(np.diag(result_matrix)))
    away_win_prob = float(np.sum(np.triu(result_matrix, 1)))

    return {
        "home_xg":       home_xg,
        "away_xg":       away_xg,
        "result_matrix": result_matrix,
        "home_win_prob": home_win_prob,
        "draw_prob":     draw_prob,
        "away_win_prob": away_win_prob,
    }


# ══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ══════════════════════════════════════════════════════════════

def run_backtest():
    print("=" * 60)
    print("FiveStat Walk-Forward Backtest")
    print("=" * 60)

    # ── Load data ──
    hist_path = "data/tables/historical_fixture_data.csv"
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Cannot find {hist_path}")

    all_data = pd.read_csv(hist_path)
    all_data["Home Team"] = all_data["Home Team"].replace(TEAM_NAME_MAPPING)
    all_data["Away Team"] = all_data["Away Team"].replace(TEAM_NAME_MAPPING)

    all_data["date_parsed"] = pd.to_datetime(all_data["Date"], dayfirst=True)
    all_data = all_data.dropna(subset=["home_goals", "away_goals", "date_parsed"])
    all_data = all_data.sort_values("date_parsed").reset_index(drop=True)

    # ── Split into pre-test and test ──
    test_start = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)

    pre_test = all_data[all_data["date_parsed"] < test_start].copy()
    test     = all_data[
        (all_data["date_parsed"] >= test_start) &
        (all_data["date_parsed"] <= test_end)
    ].copy()

    print(f"Training data (pre-test): {len(pre_test)} matches")
    print(f"Test window ({TEST_SEASON_START} to {TEST_SEASON_END}): {len(test)} matches")

    if len(test) == 0:
        print("❌ No test matches found. Check TEST_SEASON_START / END and your date column format.")
        return

    
    test["round_number"] = pd.to_numeric(test["Round Number"], errors="coerce")
    gameweeks = sorted(test["round_number"].dropna().unique())
    gw_col = "round_number"

    print(f"Gameweeks to test: {len(gameweeks)}")
    print()

    # ── Accumulators ──
    results = []
    skipped = 0

    for gw_idx, gw in enumerate(gameweeks):
        gw_fixtures = test[test[gw_col] == gw]

        # Training data = everything before this GW's fixtures
        gw_first_date = gw_fixtures["date_parsed"].min()
        training_data = all_data[all_data["date_parsed"] < gw_first_date].copy()

        if len(training_data) < MIN_TRAIN_MATCHES:
            print(f"  GW{int(gw):02d} — skipping (only {len(training_data)} training matches)")
            skipped += len(gw_fixtures)
            continue

        gw_results_list = []
        for _, fixture in gw_fixtures.iterrows():
            home_team   = fixture["Home Team"]
            away_team   = fixture["Away Team"]
            home_goals  = int(fixture["home_goals"])
            away_goals  = int(fixture["away_goals"])
            home_xg_act = float(fixture["home_xG"]) if "home_xG" in fixture and pd.notna(fixture.get("home_xG")) else None
            away_xg_act = float(fixture["away_xG"]) if "away_xG" in fixture and pd.notna(fixture.get("away_xG")) else None

            pred = predict_fixture(home_team, away_team, training_data, TEAM_NAME_MAPPING)
            if pred is None:
                skipped += 1
                continue

            # Actual outcome
            if home_goals > away_goals:
                actual_outcome = "home_win"
                outcome_idx    = 0
            elif home_goals == away_goals:
                actual_outcome = "draw"
                outcome_idx    = 1
            else:
                actual_outcome = "away_win"
                outcome_idx    = 2

            # Predicted outcome (highest probability)
            probs = [pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]]
            predicted_outcome = ["home_win", "draw", "away_win"][np.argmax(probs)]

            # Most likely scoreline from matrix
            matrix = pred["result_matrix"]
            top_i, top_j = np.unravel_index(np.argmax(matrix), matrix.shape)

            # Over/Under 2.5
            actual_over = (home_goals + away_goals) > 2
            pred_over_prob = sum(
                matrix[i, j]
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
                if i + j > 2
            )
            predicted_over = pred_over_prob >= 0.5

            # Metrics
            rps        = ranked_probability_score(probs, outcome_idx)
            naive_rps_ = naive_rps(outcome_idx)
            brier      = brier_score(pred["home_win_prob"], 1 if actual_outcome == "home_win" else 0)

            row = {
                "gw":               int(gw),
                "home_team":        home_team,
                "away_team":        away_team,
                "home_goals":       home_goals,
                "away_goals":       away_goals,
                "pred_home_xg":     round(pred["home_xg"], 3),
                "pred_away_xg":     round(pred["away_xg"], 3),
                "home_win_prob":    round(pred["home_win_prob"], 4),
                "draw_prob":        round(pred["draw_prob"], 4),
                "away_win_prob":    round(pred["away_win_prob"], 4),
                "actual_outcome":   actual_outcome,
                "predicted_outcome":predicted_outcome,
                "outcome_correct":  actual_outcome == predicted_outcome,
                "rps":              round(rps, 4),
                "naive_rps":        round(naive_rps_, 4),
                "brier":            round(brier, 4),
                "actual_over_2_5":  actual_over,
                "pred_over_prob":   round(pred_over_prob, 4),
                "ou_correct":       actual_over == predicted_over,
                "pred_top_score":   f"{top_i}-{top_j}",
                "actual_score":     f"{home_goals}-{away_goals}",
                "correct_score":    (top_i == home_goals and top_j == away_goals),
            }

            # xG MAE (if actual xG available)
            if home_xg_act is not None and away_xg_act is not None:
                row["xg_mae_home"] = abs(pred["home_xg"] - home_xg_act)
                row["xg_mae_away"] = abs(pred["away_xg"] - away_xg_act)

            results.append(row)
            gw_results_list.append(row)

        gw_acc = np.mean([r["outcome_correct"] for r in gw_results_list]) if gw_results_list else 0
        print(f"  GW{int(gw):02d} — {len(gw_results_list)} fixtures predicted  |  outcome acc: {gw_acc:.1%}")

    # ── Aggregate metrics ──
    print()
    print("=" * 60)
    print("Aggregating metrics...")

    if not results:
        print("❌ No predictions generated. Check data and team name mappings.")
        return

    df = pd.DataFrame(results)

    n                   = len(df)
    outcome_accuracy    = round(df["outcome_correct"].mean() * 100, 1)
    avg_rps             = round(df["rps"].mean(), 4)
    avg_naive_rps       = round(df["naive_rps"].mean(), 4)
    rps_improvement     = round(avg_naive_rps - avg_rps, 4)
    avg_brier           = round(df["brier"].mean(), 4)
    ou_accuracy         = round(df["ou_correct"].mean() * 100, 1)
    correct_score_rate  = round(df["correct_score"].mean() * 100, 1)
    baseline_accuracy   = round(home_always_wins_accuracy(df) * 100, 1)

    # Moneyline accuracy — exclude draws, measure home/away prediction only
    decisive_df         = df[df["actual_outcome"] != "draw"].copy()
    decisive_df["moneyline_correct"] = decisive_df["actual_outcome"] == decisive_df["predicted_outcome"]
    moneyline_accuracy  = round(decisive_df["moneyline_correct"].mean() * 100, 1)
    moneyline_n         = len(decisive_df)

    # Draw prediction rate — how often does the model predict draw as most likely
    draw_predicted_rate = round((df["predicted_outcome"] == "draw").mean() * 100, 1)
    draw_actual_rate    = round((df["actual_outcome"] == "draw").mean() * 100, 1)
    avg_model_draw_prob = round(df["draw_prob"].mean() * 100, 1)
    draw_calibration_error = round(abs(avg_model_draw_prob - draw_actual_rate), 1)

    # xG MAE (only for rows with actual xG)
    xg_rows = df.dropna(subset=["xg_mae_home", "xg_mae_away"]) if "xg_mae_home" in df.columns else pd.DataFrame()
    xg_mae  = round((xg_rows["xg_mae_home"].mean() + xg_rows["xg_mae_away"].mean()) / 2, 3) if len(xg_rows) > 0 else None

    # Per-outcome breakdown
    home_win_matches = df[df["actual_outcome"] == "home_win"]
    draw_matches     = df[df["actual_outcome"] == "draw"]
    away_win_matches = df[df["actual_outcome"] == "away_win"]

    home_win_acc = round(home_win_matches["outcome_correct"].mean() * 100, 1) if len(home_win_matches) > 0 else None
    draw_acc     = round(draw_matches["outcome_correct"].mean() * 100, 1)     if len(draw_matches) > 0     else None
    away_win_acc = round(away_win_matches["outcome_correct"].mean() * 100, 1) if len(away_win_matches) > 0 else None

    # Gameweek-by-gameweek accuracy
    gw_breakdown = (
        df.groupby("gw")
        .agg(
            fixtures=("outcome_correct", "count"),
            outcome_acc=("outcome_correct", lambda x: round(x.mean() * 100, 1)),
            avg_rps=("rps", lambda x: round(x.mean(), 4)),
            ou_acc=("ou_correct", lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    # ── Print summary ──
    print()
    print(f"  Matches predicted:        {n}  (skipped: {skipped})")
    print(f"  Outcome accuracy:         {outcome_accuracy}%  (baseline: {baseline_accuracy}%)")
    print(f"  Avg RPS:                  {avg_rps}  (naive baseline: {avg_naive_rps}, improvement: +{rps_improvement})")
    print(f"  Avg Brier score:          {avg_brier}")
    print(f"  Over/Under 2.5 accuracy:  {ou_accuracy}%")
    print(f"  Correct score hit rate:   {correct_score_rate}%")
    pred_1_1_rate = round((df["pred_top_score"] == "1-1").mean() * 100, 1)
    actual_1_1_rate = round((df["actual_score"] == "1-1").mean() * 100, 1)
    print(f"  Predicted 1-1 rate:       {pred_1_1_rate}%  (actual 1-1 rate: {actual_1_1_rate}%)")
    if xg_mae:
        print(f"  xG MAE:                   {xg_mae}")
    print()
    print(f"  Moneyline (excl. draws):")
    print(f"    Accuracy:               {moneyline_accuracy}%  ({moneyline_n} decisive matches)")
    print()
    print(f"  By outcome:")
    print(f"    Home win accuracy:      {home_win_acc}%  ({len(home_win_matches)} matches)")
    print(f"    Draw accuracy:          {draw_acc}%  ({len(draw_matches)} matches)")
    print(f"    Away win accuracy:      {away_win_acc}%  ({len(away_win_matches)} matches)")
    print()
    print(f"  Draw calibration:")
    print(f"    Model avg draw probability: {avg_model_draw_prob}%")
    print(f"    Actual draw rate:           {draw_actual_rate}%")
    print(f"    Calibration error:          {draw_calibration_error}pp")
    print(f"    Model predicted draw:       {draw_predicted_rate}% of matches")

    # ── Save results ──
    accuracy_output = {
        "season":              "2023/24–2025/26 (combined)",
        "matches_predicted":   n,
        "matches_skipped":     skipped,
        "outcome_accuracy":    outcome_accuracy,
        "baseline_accuracy":   baseline_accuracy,
        "avg_rps":             avg_rps,
        "baseline_rps":        avg_naive_rps,
        "rps_improvement":     rps_improvement,
        "avg_brier":           avg_brier,
        "ou_accuracy":         ou_accuracy,
        "ou_sample":           n,
        "correct_score_rate":  correct_score_rate,
        "xg_mae":              xg_mae,
        "home_win_accuracy":   home_win_acc,
        "draw_accuracy":       draw_acc,
        "away_win_accuracy":   away_win_acc,
        "moneyline_accuracy":  moneyline_accuracy,
        "moneyline_n":         moneyline_n,
        "draw_predicted_rate": draw_predicted_rate,
        "draw_actual_rate":    draw_actual_rate,
        "gw_breakdown":        gw_breakdown,
        "avg_model_draw_prob":    avg_model_draw_prob,
        "draw_calibration_error": draw_calibration_error,
    }

    os.makedirs("data/tables", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(accuracy_output, f, indent=2)

    print()
    print(f"✅ Results saved to {OUTPUT_PATH}")

    # ── Save detailed match-level results ──
    detail_path = "data/tables/backtest_detail.csv"
    df.to_csv(detail_path, index=False)
    print(f"✅ Match-level detail saved to {detail_path}")
    print("=" * 60)




def sweep_cov_xy():
    """
    Quick parameter sweep across cov_xy values to find optimal draw calibration.
    Tests cov_xy = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
    Reports outcome accuracy, RPS, draw accuracy and moneyline accuracy for each.
    Tests against the full combined test window (see TEST_SEASON_START/END).
    """
    print()
    print("=" * 60)
    print("cov_xy Parameter Sweep")
    print("=" * 60)

    hist_path = "data/tables/historical_fixture_data.csv"
    all_data = pd.read_csv(hist_path)
    all_data["Home Team"] = all_data["Home Team"].replace(TEAM_NAME_MAPPING)
    all_data["Away Team"] = all_data["Away Team"].replace(TEAM_NAME_MAPPING)
    all_data["date_parsed"] = pd.to_datetime(all_data["Date"], dayfirst=True)
    all_data = all_data.dropna(subset=["home_goals", "away_goals", "date_parsed"])
    all_data = all_data.sort_values("date_parsed").reset_index(drop=True)

    test_start = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)
    pre_test   = all_data[all_data["date_parsed"] < test_start].copy()
    test       = all_data[
        (all_data["date_parsed"] >= test_start) &
        (all_data["date_parsed"] <= test_end)
    ].copy()

    test["round_number"] = pd.to_numeric(test["Round Number"], errors="coerce")

    # Test against the full combined window (all seasons in TEST_SEASON_START/END)
    sample = test.copy()
    print(f"  Sample: {len(sample)} matches (full {TEST_SEASON_START} to {TEST_SEASON_END} window)")
    print()
    print(f"  {'cov_xy':>8}  {'outcome_acc':>12}  {'draw_acc':>10}  {'moneyline_acc':>14}  {'avg_rps':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*8}")

    for cov in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sweep_results = []

        for _, fixture in sample.iterrows():
            home_team  = fixture["Home Team"]
            away_team  = fixture["Away Team"]
            home_goals = int(fixture["home_goals"])
            away_goals = int(fixture["away_goals"])

            gw_date = fixture["date_parsed"]
            training = all_data[all_data["date_parsed"] < gw_date].copy()
            if len(training) < MIN_TRAIN_MATCHES:
                continue

            teams_in = set(training["Home Team"].unique()) | set(training["Away Team"].unique())
            if home_team not in teams_in or away_team not in teams_in:
                continue

            try:
                team_stats, team_hfa = calculate_team_statistics(training, save_csv_path=None, verbose=False)
                recent_att, recent_def = calculate_recent_form(training, team_stats, alpha=ALPHA)
            except Exception:
                continue

            if home_team not in team_stats or away_team not in team_stats:
                continue

            try:
                hxg = get_team_xg(home_team, away_team, True,  team_stats, recent_att, recent_def, team_home_advantage=team_hfa)
                axg = get_team_xg(away_team, home_team, False, team_stats, recent_att, recent_def, team_home_advantage=team_hfa)
            except Exception:
                continue

            matrix, hwp, dp, awp = simulate_bivariate_poisson(hxg, axg, cov_xy=cov)
            probs = [hwp, dp, awp]

            if home_goals > away_goals:
                actual, idx = "home_win", 0
            elif home_goals == away_goals:
                actual, idx = "draw", 1
            else:
                actual, idx = "away_win", 2

            predicted = ["home_win", "draw", "away_win"][np.argmax(probs)]
            rps_val   = ranked_probability_score(probs, idx)

            sweep_results.append({
                "actual":    actual,
                "predicted": predicted,
                "correct":   actual == predicted,
                "rps":       rps_val,
            })

        if not sweep_results:
            continue

        sdf             = pd.DataFrame(sweep_results)
        oacc            = sdf["correct"].mean() * 100
        draw_acc_s      = sdf[sdf["actual"] == "draw"]["correct"].mean() * 100 if len(sdf[sdf["actual"] == "draw"]) > 0 else 0
        decisive_s      = sdf[sdf["actual"] != "draw"]
        ml_acc          = decisive_s["correct"].mean() * 100 if len(decisive_s) > 0 else 0
        avg_rps_s       = sdf["rps"].mean()

        print(f"  {cov:>8.2f}  {oacc:>11.1f}%  {draw_acc_s:>9.1f}%  {ml_acc:>13.1f}%  {avg_rps_s:>8.4f}")

    print()
    print("  Recommendation: pick the cov_xy with the best RPS while")
    print("  maintaining reasonable draw accuracy improvement.")
    print("=" * 60)



def sweep_alpha():
    """
    Sweep alpha (form blending weight) from 0.3 to 0.8.
    Higher alpha = more weight on recent form vs long-run ratings.
    """
    print()
    print("=" * 60)
    print("Alpha Parameter Sweep (form blending weight)")
    print("=" * 60)

    hist_path = "data/tables/historical_fixture_data.csv"
    all_data = pd.read_csv(hist_path)
    all_data["Home Team"] = all_data["Home Team"].replace(TEAM_NAME_MAPPING)
    all_data["Away Team"] = all_data["Away Team"].replace(TEAM_NAME_MAPPING)
    all_data["date_parsed"] = pd.to_datetime(all_data["Date"], dayfirst=True)
    all_data = all_data.dropna(subset=["home_goals", "away_goals", "date_parsed"])
    all_data = all_data.sort_values("date_parsed").reset_index(drop=True)

    test_start = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)
    test       = all_data[
        (all_data["date_parsed"] >= test_start) &
        (all_data["date_parsed"] <= test_end)
    ].copy()
    test["round_number"] = pd.to_numeric(test["Round Number"], errors="coerce")
    sample = test.copy()

    print(f"  Sample: {len(sample)} matches (full {TEST_SEASON_START} to {TEST_SEASON_END} window)")
    print()
    print(f"  {'alpha':>8}  {'outcome_acc':>12}  {'draw_acc':>10}  {'moneyline_acc':>14}  {'avg_rps':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*8}")

    for alpha in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sweep_results = []

        for _, fixture in sample.iterrows():
            home_team  = fixture["Home Team"]
            away_team  = fixture["Away Team"]
            home_goals = int(fixture["home_goals"])
            away_goals = int(fixture["away_goals"])

            gw_date  = fixture["date_parsed"]
            training = all_data[all_data["date_parsed"] < gw_date].copy()
            if len(training) < MIN_TRAIN_MATCHES:
                continue

            teams_in = set(training["Home Team"].unique()) | set(training["Away Team"].unique())
            if home_team not in teams_in or away_team not in teams_in:
                continue

            try:
                team_stats, team_hfa = calculate_team_statistics(training, save_csv_path=None, verbose=False)
                recent_att, recent_def = calculate_recent_form(
                    training, team_stats, recent_matches=15, alpha=alpha
                )
            except Exception:
                continue

            if home_team not in team_stats or away_team not in team_stats:
                continue

            try:
                hxg = get_team_xg(
                    home_team, away_team, True, team_stats, recent_att, recent_def,
                    alpha=alpha, team_home_advantage=team_hfa
                )
                axg = get_team_xg(
                    away_team, home_team, False, team_stats, recent_att, recent_def,
                    alpha=alpha, team_home_advantage=team_hfa
                )
            except Exception:
                continue

            matrix, hwp, dp, awp = simulate_bivariate_poisson(hxg, axg, cov_xy=COV_XY)
            probs = [hwp, dp, awp]

            if home_goals > away_goals:
                actual, idx = "home_win", 0
            elif home_goals == away_goals:
                actual, idx = "draw", 1
            else:
                actual, idx = "away_win", 2

            predicted = ["home_win", "draw", "away_win"][np.argmax(probs)]
            rps_val   = ranked_probability_score(probs, idx)

            sweep_results.append({
                "actual":    actual,
                "predicted": predicted,
                "correct":   actual == predicted,
                "rps":       rps_val,
            })

        if not sweep_results:
            continue

        sdf       = pd.DataFrame(sweep_results)
        oacc      = sdf["correct"].mean() * 100
        draw_rows = sdf[sdf["actual"] == "draw"]
        draw_acc_s = draw_rows["correct"].mean() * 100 if len(draw_rows) > 0 else 0
        decisive_s = sdf[sdf["actual"] != "draw"]
        ml_acc    = decisive_s["correct"].mean() * 100 if len(decisive_s) > 0 else 0
        avg_rps_s = sdf["rps"].mean()

        marker = "  ← current" if abs(alpha - ALPHA) < 0.001 else ""
        print(f"  {alpha:>8.2f}  {oacc:>11.1f}%  {draw_acc_s:>9.1f}%  {ml_acc:>13.1f}%  {avg_rps_s:>8.4f}{marker}")

    print()
    print("  Recommendation: lowest RPS with highest moneyline accuracy.")
    print("=" * 60)


def sweep_rho():
    """
    Sweep Dixon-Coles rho parameter from 0.0 to -0.25.
    More negative rho = stronger correction toward draws.
    rho=0.0 means no correction applied.
    """
    print()
    print("=" * 60)
    print("Rho Parameter Sweep (Dixon-Coles correction strength)")
    print("=" * 60)

    hist_path = "data/tables/historical_fixture_data.csv"
    all_data = pd.read_csv(hist_path)
    all_data["Home Team"] = all_data["Home Team"].replace(TEAM_NAME_MAPPING)
    all_data["Away Team"] = all_data["Away Team"].replace(TEAM_NAME_MAPPING)
    all_data["date_parsed"] = pd.to_datetime(all_data["Date"], dayfirst=True)
    all_data = all_data.dropna(subset=["home_goals", "away_goals", "date_parsed"])
    all_data = all_data.sort_values("date_parsed").reset_index(drop=True)

    test_start = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)
    test       = all_data[
        (all_data["date_parsed"] >= test_start) &
        (all_data["date_parsed"] <= test_end)
    ].copy()
    test["round_number"] = pd.to_numeric(test["Round Number"], errors="coerce")
    sample = test.copy()

    print(f"  Sample: {len(sample)} matches (full {TEST_SEASON_START} to {TEST_SEASON_END} window)")
    print()
    print(f"  {'rho':>8}  {'outcome_acc':>12}  {'draw_prob':>10}  {'draw_cal_err':>13}  {'moneyline':>10}  {'avg_rps':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*13}  {'-'*10}  {'-'*8}")

    actual_draw_rate = (sample.apply(
        lambda r: r["home_goals"] == r["away_goals"], axis=1
    )).mean() * 100

    for rho in [0.0, -0.05, -0.10, -0.13, -0.15, -0.18, -0.20, -0.25]:
        sweep_results = []

        for _, fixture in sample.iterrows():
            home_team  = fixture["Home Team"]
            away_team  = fixture["Away Team"]
            home_goals = int(fixture["home_goals"])
            away_goals = int(fixture["away_goals"])

            gw_date  = fixture["date_parsed"]
            training = all_data[all_data["date_parsed"] < gw_date].copy()
            if len(training) < MIN_TRAIN_MATCHES:
                continue

            teams_in = set(training["Home Team"].unique()) | set(training["Away Team"].unique())
            if home_team not in teams_in or away_team not in teams_in:
                continue

            try:
                team_stats, team_hfa = calculate_team_statistics(training, save_csv_path=None, verbose=False)
                recent_att, recent_def = calculate_recent_form(
                    training, team_stats, recent_matches=20, alpha=ALPHA
                )
            except Exception:
                continue

            if home_team not in team_stats or away_team not in team_stats:
                continue

            try:
                hxg = get_team_xg(home_team, away_team, True,  team_stats, recent_att, recent_def, alpha=ALPHA, team_home_advantage=team_hfa)
                axg = get_team_xg(away_team, home_team, False, team_stats, recent_att, recent_def, alpha=ALPHA, team_home_advantage=team_hfa)
            except Exception:
                continue

            matrix, hwp, dp, awp = simulate_bivariate_poisson(hxg, axg, cov_xy=COV_XY)
            matrix = dixon_coles_correction(matrix, hxg, axg, rho=rho)
            hwp = float(np.sum(np.tril(matrix, -1)))
            dp  = float(np.sum(np.diag(matrix)))
            awp = float(np.sum(np.triu(matrix, 1)))
            probs = [hwp, dp, awp]

            if home_goals > away_goals:
                actual, idx = "home_win", 0
            elif home_goals == away_goals:
                actual, idx = "draw", 1
            else:
                actual, idx = "away_win", 2

            predicted = ["home_win", "draw", "away_win"][np.argmax(probs)]
            rps_val   = ranked_probability_score(probs, idx)

            sweep_results.append({
                "actual":    actual,
                "predicted": predicted,
                "correct":   actual == predicted,
                "rps":       rps_val,
                "draw_prob": dp,
            })

        if not sweep_results:
            continue

        sdf        = pd.DataFrame(sweep_results)
        oacc       = sdf["correct"].mean() * 100
        decisive_s = sdf[sdf["actual"] != "draw"]
        ml_acc     = decisive_s["correct"].mean() * 100 if len(decisive_s) > 0 else 0
        avg_rps_s  = sdf["rps"].mean()
        avg_draw_p = sdf["draw_prob"].mean() * 100
        cal_err    = abs(avg_draw_p - actual_draw_rate)

        marker = "  ← current" if abs(rho - RHO) < 0.001 else ""
        marker = "  ← no correction" if rho == 0.0 else marker
        print(f"  {rho:>8.2f}  {oacc:>11.1f}%  {avg_draw_p:>9.1f}%  {cal_err:>12.1f}pp  {ml_acc:>9.1f}%  {avg_rps_s:>8.4f}{marker}")

    print()
    print(f"  Actual draw rate in sample: {actual_draw_rate:.1f}%")
    print(f"  Target: minimise calibration error while keeping RPS low.")
    print("=" * 60)


def sweep_dispersion():
    """
    Sweep the NB overdispersion factor from 1.0 (pure Poisson) upward.
    Tracks RPS/accuracy/moneyline as usual, but also tracks the rate at which
    the model's top predicted scoreline is 1-1 — the metric this parameter
    specifically exists to address (see the 1-1 over-concentration finding).
    """
    print()
    print("=" * 60)
    print("Dispersion Parameter Sweep (NB overdispersion, 1.0 = pure Poisson)")
    print("=" * 60)

    hist_path = "data/tables/historical_fixture_data.csv"
    all_data = pd.read_csv(hist_path)
    all_data["Home Team"] = all_data["Home Team"].replace(TEAM_NAME_MAPPING)
    all_data["Away Team"] = all_data["Away Team"].replace(TEAM_NAME_MAPPING)
    all_data["date_parsed"] = pd.to_datetime(all_data["Date"], dayfirst=True)
    all_data = all_data.dropna(subset=["home_goals", "away_goals", "date_parsed"])
    all_data = all_data.sort_values("date_parsed").reset_index(drop=True)

    test_start = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)
    test       = all_data[
        (all_data["date_parsed"] >= test_start) &
        (all_data["date_parsed"] <= test_end)
    ].copy()
    test["round_number"] = pd.to_numeric(test["Round Number"], errors="coerce")
    sample = test.copy()

    print(f"  Sample: {len(sample)} matches (full {TEST_SEASON_START} to {TEST_SEASON_END} window)")
    print()
    print(f"  {'disp':>8}  {'outcome_acc':>12}  {'moneyline':>10}  {'avg_rps':>8}  {'pred_1-1_rate':>14}  {'pred_0-0_rate':>14}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*14}  {'-'*14}")

    actual_11_rate = (sample.apply(
        lambda r: r["home_goals"] == 1 and r["away_goals"] == 1, axis=1
    )).mean() * 100

    for disp in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0]:
        sweep_results = []

        for _, fixture in sample.iterrows():
            home_team  = fixture["Home Team"]
            away_team  = fixture["Away Team"]
            home_goals = int(fixture["home_goals"])
            away_goals = int(fixture["away_goals"])

            gw_date  = fixture["date_parsed"]
            training = all_data[all_data["date_parsed"] < gw_date].copy()
            if len(training) < MIN_TRAIN_MATCHES:
                continue

            teams_in = set(training["Home Team"].unique()) | set(training["Away Team"].unique())
            if home_team not in teams_in or away_team not in teams_in:
                continue

            try:
                team_stats, team_hfa = calculate_team_statistics(training, save_csv_path=None, verbose=False)
                recent_att, recent_def = calculate_recent_form(
                    training, team_stats, recent_matches=20, alpha=ALPHA
                )
            except Exception:
                continue

            if home_team not in team_stats or away_team not in team_stats:
                continue

            try:
                hxg = get_team_xg(home_team, away_team, True,  team_stats, recent_att, recent_def, alpha=ALPHA, team_home_advantage=team_hfa)
                axg = get_team_xg(away_team, home_team, False, team_stats, recent_att, recent_def, alpha=ALPHA, team_home_advantage=team_hfa)
            except Exception:
                continue

            matrix, hwp, dp, awp = simulate_bivariate_nb(hxg, axg, cov_xy=COV_XY, dispersion=disp)
            matrix = dixon_coles_correction(matrix, hxg, axg, rho=RHO)
            hwp = float(np.sum(np.tril(matrix, -1)))
            dp  = float(np.sum(np.diag(matrix)))
            awp = float(np.sum(np.triu(matrix, 1)))
            probs = [hwp, dp, awp]

            if home_goals > away_goals:
                actual, idx = "home_win", 0
            elif home_goals == away_goals:
                actual, idx = "draw", 1
            else:
                actual, idx = "away_win", 2

            predicted = ["home_win", "draw", "away_win"][np.argmax(probs)]
            rps_val   = ranked_probability_score(probs, idx)
            top_i, top_j = np.unravel_index(np.argmax(matrix), matrix.shape)

            sweep_results.append({
                "actual":    actual,
                "predicted": predicted,
                "correct":   actual == predicted,
                "rps":       rps_val,
                "pred_11":   (top_i == 1 and top_j == 1),
                "pred_00":   (top_i == 0 and top_j == 0),
            })

        if not sweep_results:
            continue

        sdf        = pd.DataFrame(sweep_results)
        oacc       = sdf["correct"].mean() * 100
        decisive_s = sdf[sdf["actual"] != "draw"]
        ml_acc     = decisive_s["correct"].mean() * 100 if len(decisive_s) > 0 else 0
        avg_rps_s  = sdf["rps"].mean()
        pred_11_rate = sdf["pred_11"].mean() * 100
        pred_00_rate = sdf["pred_00"].mean() * 100

        marker = "  ← current" if abs(disp - DISPERSION) < 0.001 else ""
        print(f"  {disp:>8.2f}  {oacc:>11.1f}%  {ml_acc:>9.1f}%  {avg_rps_s:>8.4f}  {pred_11_rate:>13.1f}%  {pred_00_rate:>13.1f}%{marker}")

    print()
    print(f"  Actual 1-1 rate in sample: {actual_11_rate:.1f}%")
    print(f"  Target: bring pred_1-1_rate down toward the actual rate without hurting RPS.")
    print("=" * 60)


def sweep_dispersion_rho():
    """
    Joint sweep of dispersion x rho, since introducing NB overdispersion changes
    the case for how much (if any) additional Dixon-Coles low-score correction
    is still appropriate — DC was designed to fix pure Poisson's under-estimate
    of low scores, but NB already inflates low scores on its own, so the two
    can double-count rather than being independent knobs.

    Tracks pred_1-1_rate AND pred_0-0_rate together, since dispersion fixing
    the former was found to create the latter — this sweep exists specifically
    to check whether some combination avoids both at once.

    Restructured vs the single-parameter sweeps: xG is computed ONCE per
    fixture (the expensive step — refitting team ratings via MLE) rather than
    once per parameter combination, since dispersion/rho only affect the final
    matrix construction, not the underlying xG. This keeps a 3x3 grid roughly
    as expensive as one 8-value single-parameter sweep, not 9x as expensive.
    """
    print()
    print("=" * 60)
    print("Joint Dispersion x Rho Sweep")
    print("=" * 60)

    hist_path = "data/tables/historical_fixture_data.csv"
    all_data = pd.read_csv(hist_path)
    all_data["Home Team"] = all_data["Home Team"].replace(TEAM_NAME_MAPPING)
    all_data["Away Team"] = all_data["Away Team"].replace(TEAM_NAME_MAPPING)
    all_data["date_parsed"] = pd.to_datetime(all_data["Date"], dayfirst=True)
    all_data = all_data.dropna(subset=["home_goals", "away_goals", "date_parsed"])
    all_data = all_data.sort_values("date_parsed").reset_index(drop=True)

    test_start = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)
    test       = all_data[
        (all_data["date_parsed"] >= test_start) &
        (all_data["date_parsed"] <= test_end)
    ].copy()
    test["round_number"] = pd.to_numeric(test["Round Number"], errors="coerce")
    sample = test.copy()

    print(f"  Sample: {len(sample)} matches (full {TEST_SEASON_START} to {TEST_SEASON_END} window)")

    actual_11_rate = (sample.apply(
        lambda r: r["home_goals"] == 1 and r["away_goals"] == 1, axis=1
    )).mean() * 100
    actual_00_rate = (sample.apply(
        lambda r: r["home_goals"] == 0 and r["away_goals"] == 0, axis=1
    )).mean() * 100

    dispersion_grid = [1.0, 1.15, 1.3]
    rho_grid        = [0.0, -0.05, -0.10]

    # Accumulate per-fixture results for every (dispersion, rho) combo, keyed
    # by combo, while only computing hxg/axg once per fixture.
    combo_results = {(d, r): [] for d in dispersion_grid for r in rho_grid}

    n_fixtures = 0
    for _, fixture in sample.iterrows():
        home_team  = fixture["Home Team"]
        away_team  = fixture["Away Team"]
        home_goals = int(fixture["home_goals"])
        away_goals = int(fixture["away_goals"])

        gw_date  = fixture["date_parsed"]
        training = all_data[all_data["date_parsed"] < gw_date].copy()
        if len(training) < MIN_TRAIN_MATCHES:
            continue

        teams_in = set(training["Home Team"].unique()) | set(training["Away Team"].unique())
        if home_team not in teams_in or away_team not in teams_in:
            continue

        try:
            team_stats, team_hfa = calculate_team_statistics(training, save_csv_path=None, verbose=False)
            recent_att, recent_def = calculate_recent_form(
                training, team_stats, recent_matches=20, alpha=ALPHA
            )
        except Exception:
            continue

        if home_team not in team_stats or away_team not in team_stats:
            continue

        try:
            hxg = get_team_xg(home_team, away_team, True,  team_stats, recent_att, recent_def, alpha=ALPHA, team_home_advantage=team_hfa)
            axg = get_team_xg(away_team, home_team, False, team_stats, recent_att, recent_def, alpha=ALPHA, team_home_advantage=team_hfa)
        except Exception:
            continue

        n_fixtures += 1

        if home_goals > away_goals:
            actual, idx = "home_win", 0
        elif home_goals == away_goals:
            actual, idx = "draw", 1
        else:
            actual, idx = "away_win", 2

        for disp in dispersion_grid:
            base_matrix, _, _, _ = simulate_bivariate_nb(hxg, axg, cov_xy=COV_XY, dispersion=disp)
            for rho in rho_grid:
                matrix = dixon_coles_correction(base_matrix, hxg, axg, rho=rho) if rho != 0.0 else base_matrix
                hwp = float(np.sum(np.tril(matrix, -1)))
                dp  = float(np.sum(np.diag(matrix)))
                awp = float(np.sum(np.triu(matrix, 1)))
                probs = [hwp, dp, awp]

                predicted = ["home_win", "draw", "away_win"][np.argmax(probs)]
                rps_val   = ranked_probability_score(probs, idx)
                top_i, top_j = np.unravel_index(np.argmax(matrix), matrix.shape)

                combo_results[(disp, rho)].append({
                    "actual":    actual,
                    "predicted": predicted,
                    "correct":   actual == predicted,
                    "rps":       rps_val,
                    "pred_11":   (top_i == 1 and top_j == 1),
                    "pred_00":   (top_i == 0 and top_j == 0),
                })

    print(f"  Fixtures evaluated: {n_fixtures}")
    print()
    print(f"  {'disp':>6}  {'rho':>6}  {'outcome_acc':>12}  {'moneyline':>10}  {'avg_rps':>8}  {'pred_1-1':>9}  {'pred_0-0':>9}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*9}")

    for disp in dispersion_grid:
        for rho in rho_grid:
            results = combo_results[(disp, rho)]
            if not results:
                continue
            sdf        = pd.DataFrame(results)
            oacc       = sdf["correct"].mean() * 100
            decisive_s = sdf[sdf["actual"] != "draw"]
            ml_acc     = decisive_s["correct"].mean() * 100 if len(decisive_s) > 0 else 0
            avg_rps_s  = sdf["rps"].mean()
            pred_11_rate = sdf["pred_11"].mean() * 100
            pred_00_rate = sdf["pred_00"].mean() * 100

            marker = "  ← current prod" if abs(disp-1.0) < 0.001 and abs(rho-RHO) < 0.001 else ""
            print(f"  {disp:>6.2f}  {rho:>6.2f}  {oacc:>11.1f}%  {ml_acc:>9.1f}%  {avg_rps_s:>8.4f}  {pred_11_rate:>8.1f}%  {pred_00_rate:>8.1f}%{marker}")

    print()
    print(f"  Actual 1-1 rate in sample: {actual_11_rate:.1f}%")
    print(f"  Actual 0-0 rate in sample: {actual_00_rate:.1f}%")
    print(f"  Target: both pred rates close to actual rates, without hurting RPS.")
    print("=" * 60)


def sweep_comprehensive():
    """
    Full grid sweep across ALPHA x COV_XY x DISPERSION x RHO in one run.

    Efficiency: calculate_team_statistics (the expensive MLE fit) does NOT
    depend on any of these four parameters, so it's computed ONCE per fixture.
    ALPHA affects calculate_recent_form/get_team_xg (cheap arithmetic, not an
    optimisation), so it's the only parameter requiring its own xG recompute;
    COV_XY, DISPERSION, and RHO only affect the final matrix construction and
    are nested cheaply inside that. Net cost: roughly one full backtest run,
    not a multiplicative blow-up across 240 combinations.

    Ranking: combinations are first filtered by a sensibility gate (predicted
    1-1 and 0-0 rates must stay within 2x their actual rates — this is what
    catches a dispersion value that "fixes" 1-1 by creating a 0-0 problem, as
    happened with dispersion=1.5 earlier). Survivors are then ranked by RPS
    and moneyline together (co-primary, matching the site's displayed
    metrics), with correct-score rate as the tiebreaker. The gate is what
    keeps correct-score rate honest here — without it, a combo could win by
    re-triggering the same over-concentration pattern that originally
    inflated this exact metric. Draw calibration is printed for disclosure
    but doesn't drive the ranking.
    """
    print()
    print("=" * 60)
    print("Comprehensive Grid Sweep (ALPHA x COV_XY x DISPERSION x RHO)")
    print("=" * 60)

    hist_path = "data/tables/historical_fixture_data.csv"
    all_data = pd.read_csv(hist_path)
    all_data["Home Team"] = all_data["Home Team"].replace(TEAM_NAME_MAPPING)
    all_data["Away Team"] = all_data["Away Team"].replace(TEAM_NAME_MAPPING)
    all_data["date_parsed"] = pd.to_datetime(all_data["Date"], dayfirst=True)
    all_data = all_data.dropna(subset=["home_goals", "away_goals", "date_parsed"])
    all_data = all_data.sort_values("date_parsed").reset_index(drop=True)

    test_start = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)
    test       = all_data[
        (all_data["date_parsed"] >= test_start) &
        (all_data["date_parsed"] <= test_end)
    ].copy()
    sample = test.copy()

    print(f"  Sample: {len(sample)} matches (full {TEST_SEASON_START} to {TEST_SEASON_END} window)")

    actual_11_rate = (sample.apply(lambda r: r["home_goals"] == 1 and r["away_goals"] == 1, axis=1)).mean() * 100
    actual_00_rate = (sample.apply(lambda r: r["home_goals"] == 0 and r["away_goals"] == 0, axis=1)).mean() * 100
    actual_draw_rate = (sample["home_goals"] == sample["away_goals"]).mean() * 100

    alpha_grid      = [0.20, 0.25, 0.30, 0.35, 0.40]
    cov_xy_grid     = [0.03, 0.05, 0.07, 0.10]
    dispersion_grid = [1.0, 1.15, 1.3, 1.45]
    rho_grid        = [-0.10, -0.05, 0.0]

    combo_results = {
        (a, c, d, r): []
        for a in alpha_grid for c in cov_xy_grid for d in dispersion_grid for r in rho_grid
    }

    n_fixtures = 0
    for _, fixture in sample.iterrows():
        home_team  = fixture["Home Team"]
        away_team  = fixture["Away Team"]
        home_goals = int(fixture["home_goals"])
        away_goals = int(fixture["away_goals"])

        gw_date  = fixture["date_parsed"]
        training = all_data[all_data["date_parsed"] < gw_date].copy()
        if len(training) < MIN_TRAIN_MATCHES:
            continue

        teams_in = set(training["Home Team"].unique()) | set(training["Away Team"].unique())
        if home_team not in teams_in or away_team not in teams_in:
            continue

        # Expensive step — computed ONCE per fixture, independent of all 4 params
        try:
            team_stats, team_hfa = calculate_team_statistics(training, save_csv_path=None, verbose=False)
        except Exception:
            continue

        if home_team not in team_stats or away_team not in team_stats:
            continue

        n_fixtures += 1

        if home_goals > away_goals:
            actual, idx = "home_win", 0
        elif home_goals == away_goals:
            actual, idx = "draw", 1
        else:
            actual, idx = "away_win", 2

        for alpha in alpha_grid:
            # Cheap step — recomputed per alpha, reused across cov_xy/dispersion/rho
            try:
                recent_att, recent_def = calculate_recent_form(training, team_stats, recent_matches=20, alpha=alpha)
                hxg = get_team_xg(home_team, away_team, True,  team_stats, recent_att, recent_def, alpha=alpha, team_home_advantage=team_hfa)
                axg = get_team_xg(away_team, home_team, False, team_stats, recent_att, recent_def, alpha=alpha, team_home_advantage=team_hfa)
            except Exception:
                continue

            for cov in cov_xy_grid:
                for disp in dispersion_grid:
                    base_matrix, _, _, _ = simulate_bivariate_nb(hxg, axg, cov_xy=cov, dispersion=disp)
                    for rho in rho_grid:
                        matrix = dixon_coles_correction(base_matrix, hxg, axg, rho=rho) if rho != 0.0 else base_matrix
                        hwp = float(np.sum(np.tril(matrix, -1)))
                        dp  = float(np.sum(np.diag(matrix)))
                        awp = float(np.sum(np.triu(matrix, 1)))
                        probs = [hwp, dp, awp]

                        predicted = ["home_win", "draw", "away_win"][np.argmax(probs)]
                        rps_val   = ranked_probability_score(probs, idx)
                        top_i, top_j = np.unravel_index(np.argmax(matrix), matrix.shape)

                        combo_results[(alpha, cov, disp, rho)].append({
                            "actual":    actual,
                            "predicted": predicted,
                            "correct":   actual == predicted,
                            "rps":       rps_val,
                            "draw_prob": dp,
                            "pred_11":   (top_i == 1 and top_j == 1),
                            "pred_00":   (top_i == 0 and top_j == 0),
                            "correct_score": (top_i == home_goals and top_j == away_goals),
                        })

    print(f"  Fixtures evaluated: {n_fixtures}  |  Combinations tested: {len(combo_results)}")
    print()

    rows = []
    for (alpha, cov, disp, rho), results in combo_results.items():
        if not results:
            continue
        sdf        = pd.DataFrame(results)
        oacc       = sdf["correct"].mean() * 100
        decisive_s = sdf[sdf["actual"] != "draw"]
        ml_acc     = decisive_s["correct"].mean() * 100 if len(decisive_s) > 0 else 0
        avg_rps_s  = sdf["rps"].mean()
        pred_11_rate = sdf["pred_11"].mean() * 100
        pred_00_rate = sdf["pred_00"].mean() * 100
        avg_draw_p   = sdf["draw_prob"].mean() * 100
        cal_err      = abs(avg_draw_p - actual_draw_rate)
        cs_rate      = sdf["correct_score"].mean() * 100

        rows.append({
            "alpha": alpha, "cov_xy": cov, "dispersion": disp, "rho": rho,
            "outcome_acc": oacc, "moneyline": ml_acc, "rps": avg_rps_s,
            "pred_11": pred_11_rate, "pred_00": pred_00_rate,
            "cal_err": cal_err, "cs_rate": cs_rate,
        })

    results_df = pd.DataFrame(rows)

    # Sensibility gate: reject combos where 1-1 or 0-0 over-concentrate relative
    # to their actual rates (the same failure mode dispersion=1.5 hit earlier)
    gate = (
        (results_df["pred_11"] <= max(actual_11_rate * 2, 15)) &
        (results_df["pred_00"] <= max(actual_00_rate * 2, 15))
    )
    survivors = results_df[gate].copy()
    rejected_n = len(results_df) - len(survivors)

    print(f"  Rejected {rejected_n}/{len(results_df)} combinations on sensibility gate")
    print(f"  (pred_1-1 or pred_0-0 rate > 2x actual — same failure mode as dispersion=1.5)")
    print()

    # Rank by RPS and moneyline together (co-primary, per site display priorities),
    # using correct-score rate as the tiebreaker. The sensibility gate above still
    # applies first — this prevents a combo from winning on correct-score rate by
    # re-triggering the same over-concentration pattern that inflated it originally.
    survivors["rps_rank"]       = survivors["rps"].rank(ascending=True)
    survivors["moneyline_rank"] = survivors["moneyline"].rank(ascending=False)
    survivors["combined_rank"]  = (survivors["rps_rank"] + survivors["moneyline_rank"]) / 2
    survivors = survivors.sort_values(["combined_rank", "cs_rate"], ascending=[True, False]).head(15)

    print(f"  Top 15 surviving combinations, ranked by RPS + Moneyline (co-primary),")
    print(f"  correct-score rate as tiebreaker:")
    print()
    print(f"  {'alpha':>6}  {'cov_xy':>6}  {'disp':>6}  {'rho':>6}  {'rps':>8}  {'moneyline':>10}  {'outcome_acc':>12}  {'pred_1-1':>9}  {'pred_0-0':>9}  {'cal_err':>8}  {'cs_rate':>8}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")
    for _, r in survivors.iterrows():
        print(f"  {r['alpha']:>6.2f}  {r['cov_xy']:>6.2f}  {r['dispersion']:>6.2f}  {r['rho']:>6.2f}  "
              f"{r['rps']:>8.4f}  {r['moneyline']:>9.1f}%  {r['outcome_acc']:>11.1f}%  "
              f"{r['pred_11']:>8.1f}%  {r['pred_00']:>8.1f}%  {r['cal_err']:>7.1f}pp  {r['cs_rate']:>7.1f}%")

    print()
    print(f"  Actual 1-1 rate: {actual_11_rate:.1f}%  |  Actual 0-0 rate: {actual_00_rate:.1f}%  |  Actual draw rate: {actual_draw_rate:.1f}%")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_cov_xy()
    elif "--sweep-alpha" in sys.argv:
        sweep_alpha()
    elif "--sweep-rho" in sys.argv:
        sweep_rho()
    elif "--sweep-dispersion" in sys.argv:
        sweep_dispersion()
    elif "--sweep-dispersion-rho" in sys.argv:
        sweep_dispersion_rho()
    elif "--sweep-comprehensive" in sys.argv:
        sweep_comprehensive()
    elif "--sweep-all" in sys.argv:
        sweep_cov_xy()
        sweep_alpha()
        sweep_rho()
        sweep_dispersion()
    else:
        run_backtest()




#python backtest.py                  # full backtest
#python backtest.py --sweep          # cov_xy sweep
#python backtest.py --sweep-alpha    # alpha sweep
#python backtest.py --sweep-rho      # rho sweep
#python backtest.py --sweep-dispersion  # NB dispersion sweep
#python backtest.py --sweep-dispersion-rho  # joint dispersion x rho sweep
#python backtest.py --sweep-comprehensive   # full 4-param grid, ranked by RPS
#python backtest.py --sweep-all      # all sweeps in one go
