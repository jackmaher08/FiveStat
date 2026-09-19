"""Small, testable helpers used by the FPL projection feature."""

from __future__ import annotations

import math

import numpy as np


def eligibility_minutes(games_played: int, position: str) -> int:
    """Return a season-aware sample threshold for a player projection.

    The old fixed 450/900 minute gates made the entire feature disappear early
    in a season. This gate grows with the available sample and then caps out.
    """
    available = max(int(games_played), 0) * 90
    if available == 0:
        return 0
    share = 0.60 if position == "DEF" else 0.55
    floor = 180 if position == "DEF" else 90
    cap = 720 if position == "DEF" else 450
    return int(min(cap, max(floor, round(available * share))))


def expected_minutes(
    total_minutes: float,
    starts: float,
    games_played: int,
    chance_of_playing: float | None = None,
) -> float:
    """Estimate next-fixture minutes from season usage and availability."""
    if games_played <= 0:
        return 0.0
    average_minutes = float(total_minutes) / games_played
    start_rate = float(np.clip(float(starts) / games_played, 0, 1))
    baseline = 0.7 * average_minutes + 0.3 * (90 * start_rate)
    availability = 1.0 if chance_of_playing is None else float(np.clip(chance_of_playing / 100, 0, 1))
    return round(float(np.clip(baseline * availability, 0, 90)), 1)


def expected_appearance_points(minutes: float) -> float:
    """Approximate one point for appearing plus another for reaching 60 mins."""
    p_appearance = float(np.clip(minutes / 30, 0, 1))
    p_sixty = float(np.clip((minutes - 30) / 30, 0, 1))
    return p_appearance + p_sixty


def expected_goals_conceded_deduction(expected_goals_against: float) -> float:
    """Expected FPL deduction for each two goals conceded by a defender."""
    lam = max(float(expected_goals_against), 0.0)
    total = 0.0
    # Twelve goals safely covers realistic football score distributions.
    for goals in range(13):
        probability = math.exp(-lam) * (lam**goals) / math.factorial(goals)
        total += (goals // 2) * probability
    return total


def projected_fpl_points(
    position: str,
    projected_xg: float,
    projected_xa: float,
    clean_sheet_probability: float,
    expected_goals_against: float,
    minutes: float,
) -> float:
    """Project the modelled components of official FPL scoring."""
    goal_points = {"DEF": 6, "MID": 5, "FWD": 4}.get(position, 0)
    clean_sheet_points = {"DEF": 4, "MID": 1, "FWD": 0}.get(position, 0)
    points = (
        expected_appearance_points(minutes)
        + goal_points * projected_xg
        + 3 * projected_xa
        + clean_sheet_points * clean_sheet_probability
    )
    if position == "DEF":
        points -= expected_goals_conceded_deduction(expected_goals_against)
    return round(points, 2)


def projection_confidence(total_minutes: float, games_played: int, expected_mins: float) -> str:
    """Simple, explainable reliability label for the UI."""
    available = max(games_played * 90, 1)
    sample_share = total_minutes / available
    if games_played >= 8 and sample_share >= 0.75 and expected_mins >= 70:
        return "High"
    if games_played >= 4 and sample_share >= 0.50 and expected_mins >= 50:
        return "Medium"
    return "Low"
