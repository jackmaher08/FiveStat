"""Lossless EPL fixture forecast exports for internal analysis and future feeds.

The scoreline matrix is supplied by the production match model; it must never
be independently recomputed from rounded expected-goals figures.
"""

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np


CSV_MARKETS = {
    "home_win_prob": ("result_probability", "home_win"),
    "draw_prob": ("result_probability", "draw"),
    "away_win_prob": ("result_probability", "away_win"),
    "home_cs_prob": ("clean_sheet_probability", "home"),
    "away_cs_prob": ("clean_sheet_probability", "away"),
}


def _number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _optional_int(value):
    if value is None or str(value).strip().lower() in ("", "nan", "none"):
        return None
    return int(float(value))


def _kickoff(date_value):
    if date_value is None or str(date_value).strip().lower() in ("", "nan", "none"):
        return None, None, None
    # FixtureDownload dates are UK wall-clock times, not UTC. Preserve the
    # original string and expose an unambiguous UTC field for feed consumers.
    local = datetime.strptime(str(date_value), "%d/%m/%Y %H:%M").replace(
        tzinfo=ZoneInfo("Europe/London")
    )
    utc = local.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    start_year = local.year if local.month >= 7 else local.year - 1
    season = f"{start_year}/{str(start_year + 1)[-2:]}"
    return str(date_value), utc, season


def build_fixture_prediction(fixture, probabilities, matrix, heatmap_path):
    """Build a fixture record and verify every market against its 8x8 matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (8, 8):
        raise ValueError(f"Expected 8x8 scoreline matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Scoreline matrix contains invalid probabilities")
    if not np.isclose(float(matrix.sum()), 1.0, rtol=0, atol=1e-9):
        raise ValueError("Scoreline matrix must sum to 1")

    home = str(fixture["home_team"])
    away = str(fixture["away_team"])
    local_date, kickoff_utc, season = _kickoff(fixture.get("date"))
    result = {
        "home_win": _number(probabilities["home_win_prob"], "home_win_prob"),
        "draw": _number(probabilities["draw_prob"], "draw_prob"),
        "away_win": _number(probabilities["away_win_prob"], "away_win_prob"),
    }
    clean_sheets = {
        "home": _number(probabilities["home_cs_prob"], "home_cs_prob"),
        "away": _number(probabilities["away_cs_prob"], "away_cs_prob"),
    }
    over = _number(probabilities["over_2_5_prob"], "over_2_5_prob")
    btts = _number(probabilities["btts_prob"], "btts_prob")
    xg = {
        "home": _number(probabilities["home_xg"], "home_xg"),
        "away": _number(probabilities["away_xg"], "away_xg"),
    }
    if min(xg.values()) < 0:
        raise ValueError("Expected goals cannot be negative")

    totals = np.add.outer(np.arange(8), np.arange(8))
    expected = {
        "home_win_prob": float(np.tril(matrix, -1).sum()),
        "draw_prob": float(np.trace(matrix)),
        "away_win_prob": float(np.triu(matrix, 1).sum()),
        "over_2_5_prob": float(matrix[totals > 2].sum()),
        "btts_prob": float(matrix[1:, 1:].sum()),
        "home_cs_prob": float(matrix[:, 0].sum()),
        "away_cs_prob": float(matrix[0, :].sum()),
    }
    for column, computed in expected.items():
        supplied = _number(probabilities[column], column)
        if not math.isclose(supplied, computed, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{home} vs {away}: {column} differs from scoreline matrix")

    scorelines = sorted(
        (
            {"home_goals": i, "away_goals": j, "score": f"{i}-{j}",
             "probability": float(matrix[i, j])}
            for i in range(8) for j in range(8)
        ),
        key=lambda item: (-item["probability"], item["home_goals"], item["away_goals"]),
    )
    return {
        "fixture": {
            "id": _optional_int(fixture.get("id")),
            "competition": "Premier League",
            "season": season,
            "round_number": _optional_int(fixture.get("round_number")),
            "kickoff_local": local_date,
            "kickoff_timezone": "Europe/London" if local_date else None,
            "kickoff_utc": kickoff_utc,
            "home_team": home,
            "away_team": away,
        },
        "predictions": {
            "result_probability": result,
            "expected_goals": xg,
            "over_2_5_goals_probability": over,
            "both_teams_to_score_probability": btts,
            "clean_sheet_probability": clean_sheets,
            "correct_score": {
                "max_goals_per_team": 7,
                "matrix": matrix.tolist(),
                "top_scorelines": scorelines[:5],
            },
        },
        "assets": {"correct_score_heatmap": heatmap_path},
    }


def validate_against_csv(records, csv_frame):
    """Fail the pipeline rather than publishing an incomplete/misaligned feed."""
    if len(records) != len(csv_frame):
        raise ValueError(f"JSON has {len(records)} fixtures; CSV has {len(csv_frame)}")
    for record, (_, row) in zip(records, csv_frame.iterrows()):
        fixture = record["fixture"]
        if (fixture["home_team"], fixture["away_team"]) != (row["home_team"], row["away_team"]):
            raise ValueError("Fixture order/team mismatch between CSV and JSON")
        prediction = record["predictions"]
        for column, keys in CSV_MARKETS.items():
            if not math.isclose(float(row[column]), prediction[keys[0]][keys[1]], abs_tol=1e-12):
                raise ValueError(f"CSV/JSON mismatch: {column}")
        for column, key in (
            ("over_2_5_prob", "over_2_5_goals_probability"),
            ("btts_prob", "both_teams_to_score_probability"),
        ):
            if not math.isclose(float(row[column]), prediction[key], abs_tol=1e-12):
                raise ValueError(f"CSV/JSON mismatch: {column}")
        for column, key in (("home_xg", "home"), ("away_xg", "away")):
            if not math.isclose(float(row[column]), prediction["expected_goals"][key], abs_tol=1e-12):
                raise ValueError(f"CSV/JSON mismatch: {column}")


def _atomic_write(path, writer):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as file:
            temp_path = file.name
            writer(file)
        os.replace(temp_path, destination)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def write_fixture_json(records, path, *, generated_at_utc, form_weight, covariance, dispersion):
    payload = {
        "schema_version": "1.0",
        "competition": "Premier League",
        "generated_at_utc": generated_at_utc,
        "probability_scale": "0_to_1",
        "model": {
            "name": "FiveStat EPL Match Model",
            "scoreline_distribution": "bivariate_negative_binomial",
            "parameters": {
                "covariance": covariance,
                "dispersion": dispersion,
                "dixon_coles_correction_applied": False,
                "form_weight": form_weight,
            },
        },
        "scoreline_matrix_axes": {"rows": "home_goals", "columns": "away_goals"},
        "fixture_count": len(records),
        "fixtures": records,
    }
    _atomic_write(path, lambda file: json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False))


def write_correct_score_csv(records, path):
    fields = ["fixture_id", "home_team", "away_team", "home_goals", "away_goals", "probability"]

    def write(file):
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            fixture = record["fixture"]
            for home_goals, row in enumerate(record["predictions"]["correct_score"]["matrix"]):
                for away_goals, probability in enumerate(row):
                    writer.writerow({
                        "fixture_id": fixture["id"],
                        "home_team": fixture["home_team"],
                        "away_team": fixture["away_team"],
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "probability": probability,
                    })

    _atomic_write(path, write)
