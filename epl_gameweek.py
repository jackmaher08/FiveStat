"""Select the EPL display week using official results, then a conservative kickoff fallback.

Understat is not required: fixture_data.csv uses FPL's completion flags and
kickoff times. The fallback changes *navigation only*, never scores/results.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

UK = ZoneInfo("Europe/London")
MATCH_GRACE = timedelta(hours=3)


def _kickoff(value):
    if pd.isna(value):
        return None
    try:
        return datetime.strptime(str(value), "%d/%m/%Y %H:%M").replace(tzinfo=UK)
    except (ValueError, TypeError):
        return None


def select_epl_gameweek(fixtures, now=None):
    """Stay on a GW until all results are final OR its last match is well past kickoff.

    The three-hour fallback prevents a stale FPL 'finished' feed from pinning
    navigation to the previous GW. If a match is rescheduled beyond the next
    GW's start, it does not block the rest of the season's navigation.
    """
    if fixtures.empty:
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(UK)

    frame = fixtures.copy()
    frame["round_number"] = pd.to_numeric(frame["round_number"], errors="coerce")
    frame = frame.dropna(subset=["round_number"])
    if frame.empty:
        return None
    frame["complete"] = frame["isResult"].astype(str).str.lower().eq("true")
    frame["kickoff"] = frame["date"].map(_kickoff)
    weeks = sorted(int(gw) for gw in frame["round_number"].unique())

    for position, gw in enumerate(weeks):
        week = frame[frame["round_number"] == gw]
        if week["complete"].all():
            continue

        kickoff_times = week["kickoff"].dropna().tolist()
        if not kickoff_times:
            return gw  # Do not advance if we have no reliable dates.

        next_kickoffs = []
        if position + 1 < len(weeks):
            following = frame[frame["round_number"] == weeks[position + 1]]
            next_kickoffs = following["kickoff"].dropna().tolist()

        if next_kickoffs:
            next_start = min(next_kickoffs)
            regular_kickoffs = [kick for kick in kickoff_times if kick < next_start]
            if regular_kickoffs:
                kickoff_times = regular_kickoffs  # Ignore a postponed late fixture.

        if now < max(kickoff_times) + MATCH_GRACE:
            return gw

    return weeks[-1]  # End of season: retain last GW.
