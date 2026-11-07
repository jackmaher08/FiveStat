"""Deadline-based FPL gameweek selection (independent of Understat)."""
from datetime import datetime, timezone


def parse_deadline(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def next_deadline_event(events, now=None):
    """Choose the earliest *future* deadline, not the API's lagging is_next flag."""
    now = now or datetime.now(timezone.utc)
    candidates = [(parse_deadline(e.get("deadline_time")), e) for e in events]
    future = [(deadline, e) for deadline, e in candidates if deadline and deadline > now]
    return min(future, key=lambda item: item[0])[1] if future else None


def display_gameweek(deadline_record, fixtures, now=None):
    """Advance a stale cached GW after its deadline, even if games remain unplayed."""
    now = now or datetime.now(timezone.utc)
    upcoming = fixtures.loc[~fixtures["isResult"], "round_number"].dropna()
    fallback = int(upcoming.min()) if not upcoming.empty else 1
    if not deadline_record:
        return fallback
    try:
        gw = int(deadline_record["gw"])
    except (KeyError, TypeError, ValueError):
        return fallback
    deadline = parse_deadline(deadline_record.get("deadline_time"))
    if deadline is None:
        return fallback
    if deadline <= now:
        return max(gw + 1, fallback)
    return gw
