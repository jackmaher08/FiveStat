"""Build team chart windows from FiveStat's existing match projections."""

from collections import defaultdict
import math

import pandas as pd


def build_team_chart_windows(fixtures, probabilities, teams, names, current_gw):
    """Return the next GW, next five GWs and any fully modelled future GWs.

    Keep blank weeks distinct from missing projections, and never multiply
    fixtures via an ambiguous join. Team projections do not depend on player
    eligibility or the player data feed.
    """
    keys = ["home_team", "away_team"]
    metrics = ["home_xg", "away_xg", "home_cs_prob", "away_cs_prob"]
    remaining = fixtures.loc[
        ~fixtures["isResult"] & fixtures["round_number"].between(current_gw, 38),
        keys + ["round_number"],
    ].copy()
    if remaining.empty or not teams:
        return {"weeks": [], "by_gw": {}, "next5": []}

    # An ambiguous prediction cannot be assigned to a particular match.
    model = probabilities.reindex(columns=keys + metrics)
    model = model.loc[~model.duplicated(keys, keep=False)]
    remaining["ambiguous"] = remaining.duplicated(keys, keep=False)
    remaining = remaining.merge(model, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        remaining[metric] = pd.to_numeric(remaining[metric], errors="coerce")
    valid = (
        remaining[metrics].apply(lambda column: column.map(math.isfinite)).all(axis=1)
        & remaining[metrics[:2]].ge(0).all(axis=1)
        & remaining[metrics[2:]].ge(0).all(axis=1)
        & remaining[metrics[2:]].le(1).all(axis=1)
        & ~remaining["ambiguous"]
    )
    remaining["valid"] = valid

    # Index once rather than rescanning the fixture table for every team/week.
    per_week = defaultdict(lambda: defaultdict(list))
    next_five = defaultdict(list)
    for fix in remaining.itertuples(index=False):
        gw = int(fix.round_number)
        for team, xg, xga, cs in [
            (fix.home_team, fix.home_xg, fix.away_xg, fix.home_cs_prob),
            (fix.away_team, fix.away_xg, fix.home_xg, fix.away_cs_prob),
        ]:
            projection = (float(xg), float(xga), float(cs)) if fix.valid else None
            per_week[gw][team].append(projection)
            if gw < current_gw + 5:
                next_five[team].append(projection)

    def summarize(window, single_gw):
        rows = []
        for team in teams:
            projections = window.get(team, [])
            count = len(projections)
            row = {"team": names.get(team, team), "fixtures": count,
                   "xg": None, "xga": None, "cs": None}
            if count and all(projection is not None for projection in projections):
                xg, xga, cs = zip(*projections)
                row.update(
                    xg=round(math.fsum(xg), 2),
                    xga=round(math.fsum(xga), 2),
                    # Single GW: P(at least one CS), assuming independent
                    # fixtures. Multi-GW: expected number of clean sheets.
                    cs=round(100 * (cs[0] if count == 1 else
                                   1 - math.prod(1 - p for p in cs)), 1)
                    if single_gw else round(math.fsum(cs), 3),
                )
            rows.append(row)
        return rows

    by_gw = {}
    weeks = []
    for gw in range(current_gw, 39):
        window = remaining.loc[remaining["round_number"] == gw]
        if not window.empty and window["valid"].all():
            weeks.append(gw)
        if gw == current_gw or gw in weeks:
            by_gw[str(gw)] = summarize(per_week[gw], single_gw=True)

    return {
        "weeks": weeks,
        "by_gw": by_gw,
        "next5": summarize(next_five, single_gw=False),
    }
