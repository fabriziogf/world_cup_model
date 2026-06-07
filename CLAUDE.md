# World Cup Prediction Model — Claude Code Instructions

## Project Overview

A modular Python system for predicting World Cup match outcomes and simulating tournament brackets. The stack uses Elo ratings as a team-strength feature, Dixon-Coles Poisson regression for scoreline prediction, and Monte Carlo simulation for tournament win probabilities.

---

## Current State

The following files are already implemented and should be treated as stable:

| File | Status | Description |
|---|---|---|
| `src/elo.py` | ✅ Complete | Elo rating system with K-factor weighting, home advantage, time decay |
| `src/poisson.py` | ✅ Complete | Dixon-Coles Poisson model with MLE fitting and scoreline matrix |
| `src/__init__.py` | ✅ Complete | Package exports |
| `notebooks/explore.py` | ✅ Complete | End-to-end walkthrough script |
| `requirements.txt` | ✅ Complete | Python dependencies |
| `data/` | ⏳ Empty | Drop `results.csv` here (see Data section below) |
| `src/simulate.py` | ❌ Not built | Next priority — see Agent 2 task |

---

## Project Structure

```
world_cup_model/
├── data/
│   └── results.csv          # Historical match results (download separately)
├── src/
│   ├── __init__.py
│   ├── elo.py               # EloSystem class
│   ├── poisson.py           # DixonColes class
│   ├── simulate.py          # TO BUILD: bracket Monte Carlo simulator
│   ├── features.py          # TO BUILD: feature engineering pipeline
│   └── evaluate.py          # TO BUILD: model validation & calibration
├── notebooks/
│   └── explore.py           # End-to-end walkthrough
├── tests/
│   ├── test_elo.py          # TO BUILD
│   ├── test_poisson.py      # TO BUILD
│   └── test_simulate.py     # TO BUILD
└── requirements.txt
```

---

## Data

**Source:** [Kaggle — International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Download `results.csv` and place it at `data/results.csv`.

**Expected schema:**

| Column | Type | Notes |
|---|---|---|
| `date` | date string | YYYY-MM-DD |
| `home_team` | string | Country name |
| `away_team` | string | Country name |
| `home_score` | int | Goals scored |
| `away_score` | int | Goals scored |
| `tournament` | string | e.g. "FIFA World Cup", "Friendly" |
| `neutral` | bool | True if played at neutral venue |

---

## Agent Task Breakdown

Use these tasks to parallelize work across multiple Claude Code agents. Each agent should work in its own git branch.

---

### Agent 1 — `simulate.py` (Bracket Monte Carlo)

**Branch:** `feature/simulator`

**Goal:** Build `src/simulate.py` — a tournament bracket simulator that uses `DixonColes.predict_knockout()` to run N Monte Carlo simulations and return tournament win probabilities per team.

**Requirements:**

- `class TournamentSimulator`
- Accept a group-stage bracket (dict of groups → list of teams)
- Simulate group stage: round-robin within each group, top 2 advance
- Simulate knockout rounds: Round of 16, QF, SF, Final
- Run `n_simulations` (default 100,000) Monte Carlo iterations
- Return a DataFrame of win probabilities sorted descending
- Should be fast — vectorize where possible, avoid Python loops over simulations
- Expose a `simulate_from_current(results_so_far)` method for mid-tournament updates

**Interface sketch:**
```python
sim = TournamentSimulator(model=dc, n_simulations=100_000)
probs = sim.simulate(bracket)
# Returns: DataFrame(team, p_win, p_final, p_semi, p_quarter, p_group_exit)
```

---

### Agent 2 — `features.py` (Feature Engineering Pipeline)

**Branch:** `feature/features`

**Goal:** Build `src/features.py` — a pipeline that enriches a match DataFrame with model-ready features for the XGBoost layer.

**Requirements:**

- `class FeatureBuilder`
- Takes raw results DataFrame + fitted `EloSystem` as input
- Outputs a feature DataFrame aligned row-by-row with matches
- Features to implement:
  - `elo_diff` — Elo rating difference (home - away) at match time
  - `elo_home`, `elo_away` — absolute Elo ratings at match time
  - `form_home`, `form_away` — points per game over last 10 matches (exponentially weighted)
  - `rest_days_home`, `rest_days_away` — days since last match
  - `is_neutral` — boolean
  - `tournament_weight` — numeric importance weight (from `MATCH_WEIGHTS`)
  - `h2h_home_winrate` — historical head-to-head win rate (last 10 meetings)
- Must avoid lookahead: features at row `t` use only data from rows `< t`
- Include a `transform(df)` method that returns the feature matrix as a DataFrame

---

### Agent 3 — `evaluate.py` (Validation & Calibration)

**Branch:** `feature/evaluation`

**Goal:** Build `src/evaluate.py` — tools for backtesting the model on past World Cups and measuring calibration.

**Requirements:**

- `class ModelEvaluator`
- Backtest on held-out tournaments: train on data before tournament year, evaluate on tournament matches
- Metrics to compute:
  - **Brier score** (primary) — lower is better
  - **Log-loss**
  - **Accuracy** (predicted winner matches actual winner)
  - **Calibration curve** — plot predicted probability vs. actual frequency in bins
- `evaluate_tournament(year)` method — returns metrics dict for a given World Cup year
- `compare_models(elo_system, dc_model)` — side-by-side Brier score comparison
- Plot calibration curves using `matplotlib`
- Target tournaments for validation: 2014, 2018, 2022

---

### Agent 4 — Tests

**Branch:** `feature/tests`

**Goal:** Write `pytest` test suites for the three complete modules.

**Files to create:** `tests/test_elo.py`, `tests/test_poisson.py`, `tests/test_simulate.py`

**Coverage requirements:**

`test_elo.py`:
- Elo initializes all teams at 1500
- After a win, winner's Elo increases and loser's decreases
- Zero-sum: total Elo in the system is constant
- Higher-K matches produce larger rating swings
- `predict()` returns probabilities that sum to 1

`test_poisson.py`:
- Score matrix sums to ~1.0 (within tolerance for truncation)
- `predict_knockout()` probabilities sum to 1
- Stronger team (higher attack) gets higher win probability
- Fit runs without error on small synthetic dataset
- Dixon-Coles tau correction returns expected values for edge cases

`test_simulate.py` (once `simulate.py` exists):
- Win probabilities sum to 1 across all teams
- Clear favorite wins more often than outsider in large N
- Mid-tournament update with known results changes probabilities correctly

---

## Coding Conventions

- **Python 3.11+**
- Type hints on all public methods
- Docstrings on all classes and public methods
- No global state — everything through class instances
- `pandas` for tabular data, `numpy` for numeric ops, `scipy` for optimization
- Do not modify `elo.py` or `poisson.py` — treat them as stable APIs
- All file paths relative to project root

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run end-to-end explore script (requires data/results.csv)
cd notebooks && python explore.py

# Run tests (once Agent 4 is complete)
pytest tests/ -v
```

---

## Integration Order

Once all branches are ready, merge in this order to avoid conflicts:

1. `feature/features` (no dependencies on new modules)
2. `feature/simulator` (depends on `poisson.py` only)
3. `feature/evaluation` (depends on `elo.py`, `poisson.py`, `features.py`)
4. `feature/tests` (depends on all of the above)
