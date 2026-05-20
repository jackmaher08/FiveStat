# FiveStat

Live multi-sport analytics platform covering the Premier League and Formula 1. Built with Python and Flask, deployed at [fivestat.co.uk](https://www.fivestat.co.uk).

---

## What it does

**Premier League**
- Match predictions using a Bivariate Poisson model with Dixon-Coles low-score correction
- Opponent-adjusted attack and defence ratings fitted via MLE, updated each gameweek
- xTable — league table ranked by expected points (xPTS) rather than actual points
- Scoreline probability heatmaps and Monte Carlo season simulations
- Player shotmaps and stat dashboards pulling from Understat, FBref and StatsBomb
- EV checker — compares model-implied probabilities against margin-adjusted bookmaker odds to flag potential edges
- FPL captain pick projections (xFP) with confidence ranges
- Walk-forward backtesting evaluated by RPS

**Formula 1**
- Driver Pace Rating (DPR) — qualifying and race pace adjusted for car performance
- Constructor Pace Index (CPI) — team-level pace rating across the season
- Race result predictions and expected fantasy points (xFP) per driver
- Season standings projections via Monte Carlo simulation

---

## Stack

- **Backend:** Python, Flask, pandas, NumPy
- **Modelling:** SciPy (MLE fitting), matplotlib, mplsoccer, seaborn
- **Data:** Understat, FBref, StatsBomb (via statsbombpy), FastF1, The Odds API
- **Deployment:** Railway, GitHub Actions (CI/CD on push to main)
- **Frontend:** Vanilla JS, Jinja2 templating, custom CSS

---

## Project structure

```
app.py                  # Flask routes and app logic
data_loader.py          # EPL data pipeline and model fitting
data_scraper_script.py  # Weekly EPL data scraper
f1_data_loader.py       # F1 data pipeline and model logic
f1_scraper.py           # F1 data scraper
generate_shotmaps.py    # Shotmap generation pipeline
shotmap_script.py       # Individual shotmap rendering
backtest.py             # Walk-forward backtest framework
styles.css              # All site styles
```

---

## Data update workflow

**EPL (weekly post-GW):**
```bash
python data_scraper_script.py
python data_loader.py
python generate_shotmaps.py
python shotmap_script.py
git add .
git commit -m "GW[X] data update"
git push origin main
```

**F1 (post-qualifying and post-race):**
Manual `workflow_dispatch` trigger via GitHub Actions once JSON files are updated in `data/f1/`.

---

## Model notes

The EPL model is a Dixon-Coles Bivariate Poisson implementation. ATT and DEF ratings are fitted via maximum likelihood estimation on a rolling window of recent results, with a time-decay weighting to down-weight older matches. The rho correction adjusts for the observed under-frequency of 0-0 and 1-0 scorelines relative to independent Poisson expectations.

Model quality is tracked by Ranked Probability Score (RPS) on a walk-forward holdout — not in-sample accuracy. Lower RPS is better.

The EV checker pulls live odds from The Odds API, strips the bookmaker margin, and compares the resulting fair probabilities against the model output. Positive EV flags are where model probability exceeds the fair implied probability by a meaningful margin.

---

## Methodology

Full methodology documented at [fivestat.co.uk/methodology](https://www.fivestat.co.uk/methodology).
