"""Reporting helpers for chronological FiveStat backtests."""

import numpy as np
import pandas as pd


def season_label(value):
    """Return an English league-season label for a date-like value."""
    timestamp = pd.Timestamp(value)
    start_year = timestamp.year if timestamp.month >= 7 else timestamp.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def add_season_column(frame, date_column="date_parsed"):
    """Return a copy with a deterministic season label derived from match date."""
    result = frame.copy()
    result["season"] = result[date_column].map(season_label)
    return result


def iter_walk_forward_batches(frame, date_column="date_parsed"):
    """Yield same-day fixture batches in chronological order.

    All fixtures on a date share a training cutoff, preventing results from one
    match on that date leaking into another prediction from the same date.
    """
    ordered = frame.sort_values(date_column)
    for forecast_date, fixtures in ordered.groupby(date_column, sort=True):
        yield pd.Timestamp(forecast_date), fixtures.copy()


def _bootstrap_rps_interval(frame, iterations=2000, seed=20260921):
    """95% cluster-bootstrap interval, sampling forecast dates with replacement."""
    if frame.empty:
        return None, None

    if "forecast_date" in frame.columns:
        clusters = [
            group["rps"].astype(float).to_numpy()
            for _, group in frame.groupby("forecast_date", sort=True)
        ]
    else:
        clusters = [np.asarray([value], dtype=float) for value in frame["rps"]]

    if not clusters:
        return None, None

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    cluster_count = len(clusters)

    for index in range(iterations):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        values = np.concatenate([clusters[position] for position in sampled])
        means[index] = values.mean()

    low, high = np.quantile(means, [0.025, 0.975])
    return round(float(low), 4), round(float(high), 4)


def summarise_prediction_window(frame, key, label, seasons):
    """Summarise a fixed evaluation window from already-frozen predictions."""
    if frame.empty:
        return None

    decisive = frame[frame["actual_outcome"] != "draw"].copy()
    if decisive.empty:
        moneyline_accuracy = None
        moneyline_n = 0
    else:
        moneyline_accuracy = round(
            (
                decisive["actual_outcome"]
                == decisive["predicted_outcome"]
            ).mean() * 100,
            1,
        )
        moneyline_n = int(len(decisive))

    avg_rps = round(float(frame["rps"].mean()), 4)
    baseline_rps = round(float(frame["naive_rps"].mean()), 4)
    rps_reduction_pct = (
        round(((baseline_rps - avg_rps) / baseline_rps) * 100, 1)
        if baseline_rps > 0
        else None
    )
    outcome_accuracy = round(float(frame["outcome_correct"].mean()) * 100, 1)
    baseline_accuracy = round(
        float((frame["actual_outcome"] == "home_win").mean()) * 100,
        1,
    )
    rps_ci_low, rps_ci_high = _bootstrap_rps_interval(frame)

    return {
        "key": key,
        "label": label,
        "seasons": list(seasons),
        "season_range": "–".join(
            [seasons[0], seasons[-1]]
        ) if len(seasons) > 1 else seasons[0],
        "matches_predicted": int(len(frame)),
        "avg_rps": avg_rps,
        "baseline_rps": baseline_rps,
        "rps_reduction_pct": rps_reduction_pct,
        "rps_ci_low": rps_ci_low,
        "rps_ci_high": rps_ci_high,
        "avg_brier": round(float(frame["brier"].mean()), 4),
        "outcome_accuracy": outcome_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_lift_pp": round(
            outcome_accuracy - baseline_accuracy,
            1,
        ),
        "ou_accuracy": round(float(frame["ou_correct"].mean()) * 100, 1),
        "correct_score_rate": round(
            float(frame["correct_score"].mean()) * 100,
            1,
        ),
        "moneyline_accuracy": moneyline_accuracy,
        "moneyline_n": moneyline_n,
    }


def build_window_breakdown(frame):
    """Build pre-declared full, recent-two and latest-season comparisons."""
    if frame.empty or "season" not in frame.columns:
        return []

    seasons = list(dict.fromkeys(frame["season"].tolist()))
    specs = [
        ("full", "Full available window", seasons),
    ]
    if len(seasons) >= 2:
        specs.append(("recent_two", "Latest two seasons", seasons[-2:]))
    specs.append(("latest", "Latest season", seasons[-1:]))

    summaries = []
    seen = set()
    for key, label, included in specs:
        signature = tuple(included)
        if signature in seen:
            continue
        seen.add(signature)
        subset = frame[frame["season"].isin(included)]
        summary = summarise_prediction_window(
            subset,
            key=key,
            label=label,
            seasons=included,
        )
        if summary:
            summaries.append(summary)
    return summaries


def build_season_breakdown(frame):
    """Return one comparable summary for each evaluated season."""
    if frame.empty or "season" not in frame.columns:
        return []

    summaries = []
    for season in dict.fromkeys(frame["season"].tolist()):
        subset = frame[frame["season"] == season]
        summary = summarise_prediction_window(
            subset,
            key=season,
            label=season,
            seasons=[season],
        )
        if summary:
            summaries.append(summary)
    return summaries
