"""Prepare season-to-date actual-versus-expected chart data."""

import math


VIEW_CONFIG = {
    "scoring": {
        "title": "Scoring",
        "actual_key": "G",
        "expected_key": "xG",
        "difference_sign": 1,
        "actual_label": "Goals",
        "expected_label": "Expected goals",
        "difference_label": "Goals above xG",
        "axis_label": "Goals",
        "explanation": "Positive values mean a team has scored more goals than its chances suggested.",
    },
    "defending": {
        "title": "Defending",
        "actual_key": "GA",
        "expected_key": "xGA",
        "difference_sign": -1,
        "actual_label": "Goals conceded",
        "expected_label": "Expected goals against",
        "difference_label": "Goals prevented vs xGA",
        "axis_label": "Goals conceded",
        "explanation": "Positive values mean a team has conceded fewer goals than expected.",
    },
    "points": {
        "title": "Points",
        "actual_key": "PTS",
        "expected_key": "xPTS",
        "difference_sign": 1,
        "actual_label": "Points",
        "expected_label": "Expected points",
        "difference_label": "Points above xPTS",
        "axis_label": "Points",
        "explanation": "Positive values mean a team has collected more points than expected.",
    },
}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _status(difference, tolerance=0.5):
    if difference > tolerance:
        return "above"
    if difference < -tolerance:
        return "below"
    return "level"


def build_actual_vs_expected(records, display_names=None):
    """Return serialisable chart views with a consistent positive-is-good delta."""
    display_names = display_names or {}
    views = {}

    for view_name, config in VIEW_CONFIG.items():
        rows = []

        for record in records or []:
            if not isinstance(record, dict):
                continue

            team = record.get("Team")
            matches = _number(record.get("MP"))
            actual = _number(record.get(config["actual_key"]))
            expected = _number(record.get(config["expected_key"]))

            if not team or matches is None or matches <= 0:
                continue
            if actual is None or expected is None:
                continue

            difference = (actual - expected) * config["difference_sign"]
            rows.append({
                "team": str(team),
                "label": display_names.get(team, team),
                "matches": int(matches),
                "actual": round(actual, 2),
                "expected": round(expected, 2),
                "difference": round(difference, 2),
                "actual_per_match": round(actual / matches, 2),
                "expected_per_match": round(expected / matches, 2),
                "status": _status(difference),
            })

        rows.sort(key=lambda row: (-row["difference"], row["label"]))

        views[view_name] = {
            "title": config["title"],
            "actual_label": config["actual_label"],
            "expected_label": config["expected_label"],
            "difference_label": config["difference_label"],
            "axis_label": config["axis_label"],
            "explanation": config["explanation"],
            "rows": rows,
        }

    return views
