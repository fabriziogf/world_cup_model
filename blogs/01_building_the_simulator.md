# Building a World Cup Prediction Model — Part 1: The Tournament Simulator

## Project Overview

This project is a modular Python system for predicting World Cup outcomes. The core idea is to combine two well-established statistical approaches:

1. **Elo ratings** — a simple, battle-tested measure of team strength that updates after every match
2. **Dixon-Coles Poisson regression** — a more expressive model that estimates attacking and defensive parameters per team, and uses those to predict full scoreline distributions

With those two models as the foundation, the top layer is a **Monte Carlo tournament simulator** that runs thousands of simulated tournaments end-to-end and accumulates win probabilities for each team at every stage — group exit, quarter-final, semi-final, final, and champion.

The data source is the [Kaggle International Football Results dataset](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017), which covers 49,412 international matches from 1872 to the present.

---

## Project Structure

```
world_cup_model/
├── data/
│   └── results.csv          # Historical match results
├── src/
│   ├── elo.py               # EloSystem — team strength ratings
│   ├── poisson.py           # DixonColes — scoreline prediction model
│   ├── simulate.py          # TournamentSimulator — Monte Carlo bracket engine
│   ├── features.py          # (coming) Feature engineering pipeline
│   └── evaluate.py          # (coming) Backtesting & calibration
├── notebooks/
│   └── explore.py           # End-to-end walkthrough script
├── blogs/
│   └── 01_building_the_simulator.md
└── requirements.txt
```

---

## The Underlying Models

### Elo Ratings (`src/elo.py`)

The `EloSystem` class processes all historical matches chronologically and maintains a live rating for every national team. A few key design choices:

- **K-factor by match importance** — World Cup matches update ratings more aggressively than friendlies (K=60 vs K=10). This mirrors FIFA's own weighting scheme.
- **Home advantage** — 100 Elo points are added to the home team's effective rating when the match is not at a neutral venue.
- **Zero-sum updates** — points gained by the winner are exactly lost by the loser, so the total Elo in the system stays constant.

### Dixon-Coles Poisson Model (`src/poisson.py`)

The `DixonColes` class models goals as Poisson random variables. For a match between team *i* (home) and team *j* (away):

```
λ = exp(attack_i + defense_j + home_advantage)   # expected home goals
μ = exp(attack_j + defense_i)                     # expected away goals
```

The "Dixon-Coles correction" adjusts joint probabilities for low-scoring results (0-0, 1-0, 0-1, 1-1), which pure Poisson underestimates. Parameters are estimated via maximum likelihood, with older matches down-weighted using exponential time decay.

The key method for the simulator is `predict_knockout(team_a, team_b)`, which returns the win probability for each team in a no-draw context (draws are split proportionally).

---

## Building the Simulator (`src/simulate.py`)

The simulator's job is to take a full 32-team bracket and run it forward thousands of times, recording how often each team reaches each stage.

### Class Design

```python
sim = TournamentSimulator(model=dc_model, n_simulations=100_000, seed=42)
probs = sim.simulate(bracket)
```

`bracket` is a dict mapping group labels to lists of four teams:

```python
bracket = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    ...
}
```

The output is a DataFrame sorted by championship probability:

| team | p_win | p_final | p_semi | p_quarter | p_r16 | p_group_exit |
|------|-------|---------|--------|-----------|-------|--------------|
| Portugal | 0.109 | 0.177 | 0.306 | 0.459 | 0.699 | 0.301 |
| Mexico | 0.062 | 0.109 | 0.185 | 0.329 | 0.562 | 0.438 |
| ... | ... | ... | ... | ... | ... | ... |

### Performance: The Win Probability Cache

The single biggest design decision was **pre-computing all pairwise win probabilities before the simulation loop**. With 32 teams there are 32×31 = 992 ordered pairs. Each `predict_knockout()` call involves building a full scoreline matrix, so doing it inside the loop would be extremely slow. Instead, `_build_win_prob_cache()` computes all of them once upfront and stores them in a dict:

```python
cache[(team_a, team_b)] = p_a_wins
cache[(team_b, team_a)] = p_b_wins
```

Every match resolution during simulation is then just a random draw against a cached float — essentially free.

### Group Stage

Each group runs a full round-robin (6 matches for 4 teams). For each match the simulator draws a random number against the pre-cached win probability. A 25% draw bucket is carved out, so results can be win/draw/loss rather than binary. Teams get 3 points for a win, 1 for a draw. Tiebreakers use approximated goal difference (±1 per simulated match) plus a random noise term to handle exact ties.

The top two teams from each group advance.

### Knockout Rounds

The bracket follows standard World Cup seeding — group winners face runners-up from the adjacent group (1A vs 2B, 1B vs 2A, etc.). The tournament then runs four knockout rounds:

```
Round of 16  (16 → 8)
Quarter-finals (8 → 4)
Semi-finals   (4 → 2)
Final         (2 → 1)
```

All knockout matches use `predict_knockout()` probabilities (neutral venue, no draw). One bug caught during development: an early version skipped the semi-final round entirely — jumping straight from QF winners to a 2-team final — which caused all teams from groups G and H to show 0% championship probability. Fixed by adding the explicit SF round.

### Mid-Tournament Updates

`simulate_from_current(bracket, results_so_far)` accepts a DataFrame of already-played matches. When simulating a group, any match found in `results_so_far` uses the real scoreline instead of a random draw. This allows the model to be updated mid-tournament as results come in.

---

## Tests We Ran

### Syntax Check

```
python3 -m py_compile src/simulate.py  →  Syntax OK
```

### Probability Sanity Checks (Synthetic Data)

To avoid waiting on the slow MLE fit over 49k rows, a synthetic dataset was generated: all 32 tournament teams playing each other once with Poisson-distributed goals. The model was fitted on this data, then a full 10,000-simulation run was executed against the 2022 World Cup bracket.

**Results:**

| Check | Expected | Actual |
|-------|----------|--------|
| `p_win` sum | 1.0 | **1.0** |
| `p_final` sum | 2.0 | **2.0** |
| `p_semi` sum | 4.0 | **4.0** |
| `p_quarter` sum | 8.0 | **8.0** |
| `p_r16` sum | 16.0 | **16.0** |
| `p_group_exit` sum | 16.0 | **16.0** |

Every stage sum matches exactly — meaning no teams are double-counted or dropped at any round of the bracket.

### Distribution Check

With a synthetic dataset where all teams have roughly similar strength (drawn from the same Poisson distribution), championship probabilities ranged from ~1.3% (Germany, Ghana) to ~10.9% (Portugal). The spread is real — even on synthetic data the model picks up parameter differences from the random training samples — and no team shows 0% or 100%, confirming the simulator is exploring the full probability space.

---

## What's Next

The next modules to build are:

- **`src/features.py`** — a feature engineering pipeline that enriches match data with Elo differences, recent form, rest days, and head-to-head records for an XGBoost prediction layer
- **`src/evaluate.py`** — backtesting tools to validate the model on past World Cups (2014, 2018, 2022) using Brier score, log-loss, and calibration curves
- **`tests/`** — a full pytest suite covering the Elo system, Dixon-Coles model, and simulator
