# Building a World Cup Prediction Model — Part 9: The Ensemble

## Recap

[Part 8](08_why_it_still_doesnt_make_sense.md) ended with a diagnosis, not a fix. Even after vectorizing the fit, adding match-importance and confederation weighting, and tuning those weights by backtest, the Dixon-Coles model still crowned Japan favourite ahead of Brazil, France, and England. We argued the problem was structural: a model that reduces each team to a goals-based attack/defense pair is too thin to express football strength, and no amount of reweighting fixes a feature that weak.

The prescription was an **Elo + gradient-boosted-trees ensemble**: replace the goals signal with opponent-adjusted Elo ratings, and let a learned classifier combine many features instead of two. This part builds it — and finds out whether the diagnosis was right.

---

## The Design

The new model, `EloXGBoostModel`, has two layers.

### Layer 1: Elo — strength that knows who you played

The [Elo system](../src/elo.py) was built back in the project's foundations and has sat unused in the prediction path ever since. Its whole point is the thing Dixon-Coles lacks: it doesn't care how many goals you score, only *whom you beat and by how much it surprised the ratings*. Thrashing a weak side earns almost nothing; beating a strong one earns a lot. Strength of schedule is baked in automatically, with no hand-set confederation table required.

### Layer 2: Boosted trees — many features, learned weights

On top of Elo, a gradient-boosted-trees classifier predicts a 3-way target (home win / draw / away win) from the [feature pipeline](02_feature_engineering.md) designed in Part 2:

- `elo_diff`, `elo_home`, `elo_away` — opponent-adjusted strength
- `form_home`, `form_away` — exponentially weighted recent points-per-game
- `rest_days_home`, `rest_days_away` — fixture congestion
- `is_neutral`, `tournament_weight` — match context
- `h2h_home_winrate` — historical matchups

Crucially, the model *learns* how much to trust each feature. It can discover that `elo_diff` matters far more than any raw goal total — exactly the correction Dixon-Coles structurally cannot make.

### Three engineering details that mattered

**Leak-free features, fast.** `FeatureBuilder` from Part 2 is correct but O(n²) — it re-scans history for every row. Applied to 49,000 matches that's hopeless. So the ensemble builds features in a single chronological pass, maintaining per-team form/rest state and per-pair head-to-head history as it goes, computing each row's features *before* folding in that match's result. Elo's pre-match ratings come free from `EloSystem.history`. The whole feature build runs in seconds.

**Neutral-venue symmetry.** A World Cup knockout has no home team, but the model is trained on home/away framing. So for a knockout we predict *both* orientations (A-home and B-home) and average them, cancelling any residual home-framing bias before splitting the draw mass.

**A pluggable booster — and an OpenMP war story.** The plan said XGBoost. XGBoost had other plans: on this machine (an x86_64 Anaconda Python under Rosetta) its native library refused to load, complaining about a missing OpenMP symbol. Homebrew's `libomp` was the wrong architecture (arm64); the fix was `conda install -c conda-forge llvm-openmp` to update Anaconda's bundled OpenMP runtime. To keep the code robust regardless, the booster is pluggable: it uses real XGBoost when the library loads, and falls back to scikit-learn's `HistGradientBoostingClassifier` — a functionally equivalent gradient-boosted-trees implementation with no native dependency — when it doesn't. Same `fit`/`predict_proba` API either way.

The fitted model exposes `predict()`, `predict_knockout()`, `teams_`, and `is_fitted_`, so it drops straight into the existing `TournamentSimulator` with no changes.

---

## The Result

Fit on the full match history (49,411 games, ~5 seconds) and simulated through the official 2026 bracket, here is the new top of the table — with the old Dixon-Coles ranking for contrast:

| Rank | Ensemble | Win % | Dixon-Coles had |
|------|----------|-------|-----------------|
| 1 | **Argentina** | 17.2% | 3rd (7.5%) |
| 2 | Spain | 12.8% | 4th |
| 3 | France | 8.9% | 8th (4.4%) |
| 4 | Brazil | 6.8% | 7th (4.7%) |
| 5 | Portugal | 4.7% | 5th |
| 6 | England | 4.7% | **21st (1.8%)** |
| 7 | Germany | 3.4% | 11th |
| 8 | Japan | 3.3% | **1st (12.9%)** |

This passes the smell test. The reigning champions are the favourites. The traditional powers — Spain, France, Brazil, England, Germany — fill the top. **Japan fell from 1st to 8th; England climbed from 21st to 6th.** Every glaring problem from Part 8 is gone, and for the right reason: Elo knows that Japan's record is padded against weaker AFC opposition while England's cautious 1-0 wins came against serious teams.

---

## But Is It Actually Better? The Backtest

A ranking that *looks* right is not proof. So we backtested both models the same way: train on everything before a tournament, predict its matches, score them. Three World Cups, 64 matches each.

| Tournament | Brier (DC → Ens) | Accuracy (DC → Ens) |
|------------|------------------|---------------------|
| 2014 | 0.186 → 0.198 | 56% → 56% |
| 2018 | 0.208 → 0.205 | 47% → 53% |
| 2022 | 0.232 → **0.201** | 38% → **58%** |
| **Overall** | **0.2084 → 0.2012** | **46.9% → 55.7%** |

The ensemble wins on every overall metric: Brier **+3.5%**, log-loss 1.051 → 1.021, and accuracy jumping nearly **nine points** to 55.7%. It picks the correct match outcome more than half the time, against a model barely better than a coin-three-way.

### The honest asterisks

This is not a clean sweep, and pretending otherwise would undo the whole point of Part 8.

- **The ensemble lost in 2014.** On the oldest tournament, Dixon-Coles posted a better Brier (0.186 vs 0.198). The ensemble's edge is concentrated in **recent** tournaments, where Elo and form features rest on denser, more reliable data — exactly where the time-decay weighting points it.
- **64 matches is a small sample.** Per-tournament differences carry real variance; the overall average is the more trustworthy signal.
- **The per-match improvement (+3.5%) understates the real gain.** This is the subtle point from Part 8 restated: match-outcome Brier is dominated by lopsided group games, so it barely moves when the *champion ranking* goes from absurd to sensible. The metric and the eye agree on direction here, but the metric vastly under-reports the magnitude of what changed.

---

## Why It Worked

The fix wasn't cleverer tuning — it was a better signal. Two features did the heavy lifting:

1. **Elo encodes opponent quality**, so beating up on a weak confederation no longer inflates a rating. Japan's flattering goal difference stops mattering once the model sees *who* those goals came against.
2. **A learned classifier can prefer `elo_diff` to goals.** Dixon-Coles is structurally forced to explain results through scoring rates. The boosted trees are free to discover that the Elo gap predicts outcomes better than any goal total, and to weight it accordingly.

Everything we did in Parts 6 and 7 reweighted *which matches* informed a thin feature. Part 9 changed *what the model measures*. That's the difference between polishing a flawed answer and fixing it.

---

## What's Still Imperfect

Honesty has been the throughline of this series, so: the ensemble is better, not perfect.

- It still leans on a 25%-draw approximation in the group stage (a simulator simplification, not the model's fault).
- It was validated on **per-match** Brier, which we've now argued three times is the wrong yardstick for a tournament model. The right test scores the model on its *tournament-level* output — log-loss against the actual champions and finalists across many simulated runs. Building that metric is the natural next step, and it would let us claim superiority on the thing we actually care about, not just a proxy.
- Small-sample backtests and a 2014 loss mean the "+3.5%" headline deserves humility, not a victory lap.

But the core question from Part 8 — *can a richer feature set finally produce a favourite you'd put money on?* — now has an answer. Argentina, Spain, France, Brazil. Yes.

Next: a tournament-level evaluation that scores the champion ranking directly, so the metric finally measures the same thing our eyes have been measuring all along.
