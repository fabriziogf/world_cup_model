# World Cup Prediction Model

A modular Python system for predicting World Cup match outcomes and simulating
tournament brackets. It combines **Elo ratings** as a team-strength feature,
**Dixon-Coles Poisson regression** for scoreline prediction, and **Monte Carlo
simulation** for tournament win probabilities — and it supports both the
32-team and the 48-team (2026) tournament formats.

The repository is also a worked example of an honest modelling process: the
[`blogs/`](blogs/) folder narrates the build end-to-end, including a fix that
took the model from a 10-hour fit to ~10 seconds, and a candid post-mortem on
why the predictions *still* don't fully make sense.

---

## What's inside

| Module | Description |
|--------|-------------|
| [`src/elo.py`](src/elo.py) | Elo rating system — K-factor by match importance, home advantage, zero-sum updates. |
| [`src/poisson.py`](src/poisson.py) | Dixon-Coles Poisson model — attack/defense strengths, low-score correction, scoreline matrix. |
| [`src/fast_poisson.py`](src/fast_poisson.py) | **Vectorized** Dixon-Coles fit (~1000× faster than `poisson.py`), with match-importance and confederation strength-of-schedule weighting, plus model caching. |
| [`src/confederations.py`](src/confederations.py) | Maps national teams to FIFA confederations with relative strength multipliers (strength-of-schedule weighting). |
| [`src/simulate.py`](src/simulate.py) | `TournamentSimulator` — Monte Carlo bracket engine for both 32-team and 48-team (R32) formats. |
| [`src/features.py`](src/features.py) | `FeatureBuilder` — leak-free feature pipeline (Elo, form, rest, H2H, …). |
| [`src/ensemble.py`](src/ensemble.py) | `EloXGBoostModel` — Elo + gradient-boosted-trees ensemble (XGBoost, or scikit-learn fallback). Fixes the Dixon-Coles strength bias. |
| [`src/evaluate.py`](src/evaluate.py) | `ModelEvaluator` — backtesting and calibration on past World Cups (Brier, log-loss, accuracy, calibration curves). |

---

## Project structure

```
world_cup_model/
├── data/
│   └── results.csv          # Historical match results (download separately)
├── src/
│   ├── elo.py               # EloSystem
│   ├── poisson.py           # DixonColes (reference fit)
│   ├── fast_poisson.py      # Vectorized fit + weighting + caching
│   ├── confederations.py    # Team -> confederation strength map
│   ├── simulate.py          # TournamentSimulator (32- and 48-team)
│   ├── features.py          # FeatureBuilder
│   └── evaluate.py          # ModelEvaluator
├── notebooks/
│   ├── explore.py           # End-to-end walkthrough
│   ├── predict_winner.py    # Fit + simulate a full tournament
│   └── tune_weights.py      # Backtest-tune the fit weights
├── tests/                   # pytest suite (elo, poisson, fast_poisson,
│                            #   confederations, simulate)
├── blogs/                   # 8-part build narrative
└── requirements.txt
```

---

## Data

**Source:** [Kaggle — International Football Results 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Download `results.csv` and place it at `data/results.csv`. Expected columns:
`date, home_team, away_team, home_score, away_score, tournament, neutral`.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+.

---

## Usage

### Predict a tournament winner

```bash
python notebooks/predict_winner.py
```

Fits the Dixon-Coles model (vectorized, ~10s; cached to `data/dc_model.pkl`
for instant reuse) and runs 100,000 Monte Carlo simulations of the bracket
defined in the script. Edit the `bracket` dict to change the draw. Output is a
DataFrame of per-team probabilities for every stage:

```
team        p_win  p_final  p_semi  p_quarter  p_r16  p_r32  p_group_exit
Japan      12.86%   19.86%  30.13%     45.18% 65.69% 91.27%        8.73%
Morocco     7.64%   13.19%  22.20%     36.71% 58.83% 88.22%       11.78%
...
```

The simulator auto-detects the format from the number of groups: **8 groups**
→ 32-team (R16 → QF → SF → Final); **12 groups** → 48-team 2026 format (top 2
+ 8 best third-placed teams → R32 → R16 → QF → SF → Final).

### Mid-tournament updates

```python
sim.simulate_from_current(bracket, results_so_far)
```

Locks in already-played results and re-simulates only the remaining matches.

### Tune the fit weights by backtesting

```bash
python notebooks/tune_weights.py
```

Grid-searches the importance / confederation weighting strengths against the
2014/2018/2022 World Cups, scoring each by Brier score.

### Run the tests

```bash
pytest tests/ -v
```

---

## Two models, two stories

**Dixon-Coles** (goals-based) produces a counterintuitive favourite — Japan
over Brazil, France, and England — because it only sees goals and rewards
running up scores. That's a known limitation, diagnosed in
[`blogs/08`](blogs/08_why_it_still_doesnt_make_sense.md).

**The Elo + XGBoost ensemble** ([`predict_winner_ensemble.py`](notebooks/predict_winner_ensemble.py))
fixes it. By swapping the thin goals signal for opponent-adjusted Elo plus a
learned classifier, it produces a sensible ranking — **Argentina, Spain,
France, Brazil** on the 2026 draw — and backtests better on 2014/2018/2022
(Brier 0.2012 vs 0.2084, accuracy 55.7% vs 46.9%). The full story is in
[`blogs/09`](blogs/09_the_ensemble.md). Run a head-to-head with
[`notebooks/compare_models.py`](notebooks/compare_models.py).

> **XGBoost note:** on macOS + Anaconda you may hit an OpenMP load error. Fix
> with `conda install -c conda-forge llvm-openmp`. The ensemble falls back to
> scikit-learn's `HistGradientBoostingClassifier` automatically if XGBoost
> can't load.

---

## The build narrative

The [`blogs/`](blogs/) folder documents the project as it was built:

1. [Building the simulator](blogs/01_building_the_simulator.md)
2. [Feature engineering](blogs/02_feature_engineering.md)
3. [Evaluation & calibration](blogs/03_evaluation_and_calibration.md)
4. [Testing the models](blogs/04_testing_the_models.md)
5. [Making it fast](blogs/05_making_it_fast.md) — the 10-hour → 10-second fix
6. [The first prediction](blogs/06_the_prediction.md)
7. [Fixing the favourite](blogs/07_fixing_the_favourite.md) — weighting & backtest tuning
8. [Why it still doesn't make sense](blogs/08_why_it_still_doesnt_make_sense.md) — the post-mortem
9. [The ensemble](blogs/09_the_ensemble.md) — Elo + XGBoost, and the fix that worked
10. [The final prediction](blogs/10_the_final_prediction.md) — the 2026 forecast & measuring what matters

---

## Roadmap

- [x] Elo ratings, Dixon-Coles model, Monte Carlo simulator
- [x] Vectorized fit, model caching
- [x] 48-team (2026) tournament format
- [x] Match-importance & confederation strength-of-schedule weighting
- [x] Backtest-based weight tuning
- [x] Elo + XGBoost ensemble (backtests better; sensible champion ranking)
- [x] Tournament-level evaluation — scores probability on the actual champion.
      Ensemble assigns 3× the probability to real winners (14.6% vs 4.9%) and
      ranks them ~3rd vs ~8th across 2014/2018/2022
      ([`notebooks/evaluate_tournament_level.py`](notebooks/evaluate_tournament_level.py)).
