"""Helpers for presenting FiveStat's historical model-performance data."""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo


REQUIRED_METRICS = (
    "matches_predicted",
    "avg_rps",
    "baseline_rps",
    "avg_brier",
    "outcome_accuracy",
    "baseline_accuracy",
)


def load_model_performance(path):
    """Load and enrich the backtest summary used by the public scorecard."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict) or any(key not in data for key in REQUIRED_METRICS):
        return None

    try:
        model_rps = float(data["avg_rps"])
        baseline_rps = float(data["baseline_rps"])
        data["rps_reduction_pct"] = round(
            ((baseline_rps - model_rps) / baseline_rps) * 100,
            1,
        ) if baseline_rps > 0 else None

        data["accuracy_lift_pp"] = round(
            float(data["outcome_accuracy"]) - float(data["baseline_accuracy"]),
            1,
        )

        data["matches_predicted"] = int(data["matches_predicted"])
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    gameweeks = []
    for row in data.get("gw_breakdown", []):
        if not isinstance(row, dict):
            continue
        try:
            gameweeks.append({
                "gw": int(row["gw"]),
                "fixtures": int(row["fixtures"]),
                "outcome_acc": float(row["outcome_acc"]),
                "avg_rps": float(row["avg_rps"]),
                "ou_acc": float(row["ou_acc"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    data["gw_breakdown"] = gameweeks
    data["chart_labels"] = [f"GW{row['gw']}" for row in gameweeks]
    data["chart_rps"] = [row["avg_rps"] for row in gameweeks]
    data["chart_baseline_rps"] = [
        round(float(data["baseline_rps"]), 4) for _ in gameweeks
    ]

    try:
        modified = os.path.getmtime(path)
        data["updated_at"] = datetime.fromtimestamp(
            modified,
            tz=ZoneInfo("Europe/London"),
        ).strftime("%d %b %Y at %H:%M %Z")
    except OSError:
        data["updated_at"] = None

    return data
