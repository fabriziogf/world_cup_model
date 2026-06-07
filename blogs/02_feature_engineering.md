# Building a World Cup Prediction Model — Part 2: Feature Engineering

## Recap

In [Part 1](01_building_the_simulator.md) we built the Monte Carlo tournament simulator — the engine that runs thousands of bracket simulations and returns championship probabilities. That simulator relies entirely on the Dixon-Coles Poisson model to resolve individual match outcomes.

In this part we build the **feature engineering pipeline** (`src/features.py`), which prepares structured, model-ready inputs for a future XGBoost prediction layer. The goal is to describe each match numerically — not just "Brazil vs. France" but "Brazil going in with a 120-point Elo advantage, 4 days of rest, 2.1 weighted PPG form, playing at a neutral venue in a World Cup qualifier."

---

## The Core Challenge: No Lookahead

The most important constraint in any sports prediction pipeline is **no lookahead bias** — you must never let the model see information from the future when making predictions about the past.

In practice this is easy to violate accidentally. For example, computing a team's average goals scored over a season and attaching it to every match in that season leaks future results into early rows. The same applies to rolling windows that don't respect match order.

Our approach: for every match at row `t`, the feature computation receives only `df.iloc[:t]` — strictly the rows that came before it.

```python
for idx, row in df.iterrows():
    records.append(self._compute_row(row, df.iloc[:idx]))
```

This slice is passed into every helper as `past_matches`, making it impossible for any feature to accidentally reference a future match.

---

## The `FeatureBuilder` Class

```python
from src.elo import EloSystem
from src.features import FeatureBuilder

elo = EloSystem()
df = elo.load_matches("data/results.csv")
elo.compute_ratings()

builder = FeatureBuilder(elo_system=elo)
features = builder.transform(df)
```

`transform(df)` returns a DataFrame with 10 columns, aligned row-for-row with the input:

| Column | Type | Description |
|--------|------|-------------|
| `elo_home` | float | Home team's Elo rating just before this match |
| `elo_away` | float | Away team's Elo rating just before this match |
| `elo_diff` | float | `elo_home - elo_away` |
| `form_home` | float | Exponentially weighted PPG for home team (last 10 matches) |
| `form_away` | float | Same for away team |
| `rest_days_home` | int | Days since home team's last match (-1 if debut) |
| `rest_days_away` | int | Days since away team's last match (-1 if debut) |
| `is_neutral` | int | 1 if played at a neutral venue, 0 otherwise |
| `tournament_weight` | float | Numeric importance of the match type |
| `h2h_home_winrate` | float | Home team's historical win rate in this head-to-head |

---

## Feature Design Decisions

### Elo Ratings at Match Time

The Elo system maintains a full audit trail — `elo.history` records the pre- and post-match ratings for every team in every game. To get the rating at match time, we look up the most recent `post_elo` entry for each team in the history log that falls strictly before the current match date:

```python
home_rows = hist[(hist["home_team"] == team) & (hist["date"] < date)]
away_rows = hist[(hist["away_team"] == team) & (hist["date"] < date)]
# Take the most recent appearance (either home or away)
return max(candidates, key=lambda x: x[0])[1]
```

This correctly handles teams that appear as both home and away in history, always picking the most recently updated rating.

### Form: Exponentially Weighted PPG

Recent form matters more than results from three months ago. We compute points per game (Win=3, Draw=1, Loss=0) over the last 10 matches and apply exponential weighting so the most recent match has the highest weight:

```python
weights = np.array([np.exp((i - n + 1) / ewm_span) for i in range(n)])
weights /= weights.sum()
form = np.dot(weights, points)
```

With `ewm_span=5`, a match played 5 games ago has roughly 37% the weight of the most recent one. Teams with no history return a neutral baseline of **1.0** (the midpoint of a typical PPG range for international football).

### Rest Days

Fixture congestion is real — a team playing on 3 days' rest has a measurably worse record than one with a full week off. `rest_days` is simply the calendar gap since the team's most recent match. For a team's debut appearance, it returns **-1** as a sentinel value (easily handled by imputation or a separate flag in downstream models).

### Tournament Weight

Not all matches carry the same stakes — a World Cup final matters far more than a January friendly. The `tournament_weight` feature maps the `tournament` column to the same numeric K-factor scale used by `EloSystem`:

| Tournament type | Weight |
|----------------|--------|
| Friendly | 10 |
| Qualification | 25 |
| Confederation | 35 |
| Confederation final | 40 |
| World Cup | 60 |

This gives the XGBoost layer a simple numeric signal for match importance without needing to one-hot encode hundreds of tournament strings.

### Head-to-Head Win Rate

Historical matchups between specific pairs of teams capture rivalry dynamics, psychological edges, and stylistic matchups that Elo and form can't fully express. We look at the last 10 meetings between the two teams (regardless of venue) and compute the home team's win rate:

```python
wins / total  # draws count as 0 wins for both sides
```

Pairs with no head-to-head history return **0.5** — a neutral prior.

---

## Sample Output

Running `transform()` on 20 matches from 2018 onwards:

```
   elo_home  elo_away  elo_diff  form_home  form_away  rest_days_home  rest_days_away  is_neutral  tournament_weight  h2h_home_winrate
   1689.43   1574.77    114.65       1.00       1.00              -1              -1           0               20.0              0.50
   1588.27   1697.22   -108.95       1.00       1.00              -1              -1           1               20.0              0.50
   1604.77   1668.70    -63.93       1.00       1.00              -1              -1           0               20.0              0.50
   ...
```

Key checks that passed:

- **Zero nulls** across all 10 columns — every feature has a defined fallback for cold-start teams
- **Elo diff range** of -166 to +168 on the 2018+ slice, consistent with realistic rating spreads
- **No exceptions** raised on any match in the dataset

---

## What's Next

With the feature pipeline in place, the remaining modules are:

- **`src/evaluate.py`** — backtesting on past World Cups (2014, 2018, 2022) using Brier score, log-loss, accuracy, and calibration curves
- **`tests/`** — pytest suites covering Elo, Dixon-Coles, and the simulator
- **XGBoost layer** — training a gradient boosted classifier on the `FeatureBuilder` output to predict match outcomes, sitting on top of the Dixon-Coles probabilistic base
