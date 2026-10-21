# FiveStat internal B2B exports (schema v1)

These files package existing model outputs. They are not a public API or a licence to redistribute third-party data. Audit upstream source rights before any commercial delivery.

## Generated files

FPL: `data/exports/fpl/predictions.json`, plus `player_projections.csv`, `player_fixture_projections.csv`, `captaincy_rankings.csv`, `team_outlook.csv`, and `fixture_difficulty.csv` in the same directory.

F1: `data/exports/f1/predictions.json`, plus `race_predictions.csv`, `pace_ratings.csv`, `fantasy_season_projections.csv`, and `fantasy_transfer_targets.csv` in the same directory.

EPL fixture forecasts and correct-score distributions already live in `data/tables/fixture_predictions.json` and `data/tables/correct_score_probabilities.csv`.

## Running

After merging, choose GitHub Actions > FiveStat Auto-Update > Run workflow > `full`. The football/full workflow exports FPL; the F1/full workflow exports F1 after refreshing and validating source files. The scheduled full workflow also exports both. Export files are committed by the existing Actions commit step.

## Schema notes

- JSON includes `schema_version`, `generated_at_utc`, `product` and `units`.
- FPL player projections cover eligible outfield players only. The captaincy list is the site's top-100 selection, not a complete player universe.
- FPL clean-sheet probabilities are fractions (0..1). Team expected clean-sheet totals are counts; the website displays the one-gameweek figure as a percentage.
- F1 race probabilities are percentages (0..100), not fractions. Race forecast `mode` distinguishes pre- and post-qualifying.
- F1 pace ratings are relative percentage deltas, where lower is faster.
- F1 exports include only races whose calendar date is after today, matching the current website selection; races on today's date are omitted.
- Forecasts are overwritten on each run; Git history is not an immutable client snapshot archive.
- Exports fail on missing/empty FPL projections, empty forecasts for upcoming F1 races, and non-finite JSON values. A failed workflow will not commit its data.

## Remaining commercial tasks

Validate data source licensing, set up client delivery/authentication and monitoring, establish stable schema contracts, and retain timestamped historical forecast snapshots.
