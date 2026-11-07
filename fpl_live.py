"""Extract FPL event-live minutes from list (current) and dict (legacy) responses."""


def extract_live_minutes(elements):
    if isinstance(elements, dict):
        entries = elements.items()
    elif isinstance(elements, list):
        entries = ((player["id"], player) for player in elements)
    else:
        raise TypeError("Unexpected FPL live elements format")
    return {
        int(pid): int(player.get("stats", {}).get("minutes", 0) or 0)
        for pid, player in entries
    }
