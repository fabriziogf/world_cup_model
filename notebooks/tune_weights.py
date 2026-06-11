"""
tune_weights.py
---------------
Backtest-tune the importance and confederation weighting strengths.

Rather than hand-picking the weighting magnitudes and eyeballing whether
the predictions "look right", this grid-searches the two alpha knobs
(importance_alpha, confederation_alpha) and scores each configuration by
its mean multi-class Brier score on the 2014, 2018, and 2022 World Cups.
Lower Brier = better-calibrated predictions. The winning config is the
one that actually predicts past tournaments best.

Run from the project root:
    python notebooks/tune_weights.py
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_poisson import fit_fast
from src.evaluate import ModelEvaluator

YEARS = [2014, 2018, 2022]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # grid for each knob
TIME_DECAY = 0.005


def brier3(probs, actuals):
    """Multi-outcome Brier score for a 3-class (home/draw/away) prediction."""
    return sum((p - o) ** 2 for p, o in zip(probs, actuals)) / 3.0


def score_config(df, evaluator, imp_alpha, conf_alpha):
    """Mean Brier score across all target tournaments for one weight config."""
    briers = []
    for year in YEARS:
        train_df, eval_df = evaluator._split(year)
        if eval_df.empty:
            continue
        model = fit_fast(
            train_df,
            time_decay=TIME_DECAY,
            importance_alpha=imp_alpha,
            confederation_alpha=conf_alpha,
        )
        for _, m in eval_df.iterrows():
            home, away = m["home_team"], m["away_team"]
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            actual = [int(hs > as_), int(hs == as_), int(as_ > hs)]
            try:
                p = model.predict(home, away, neutral=True)
                probs = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
            except Exception:
                probs = [1 / 3, 1 / 3, 1 / 3]
            briers.append(brier3(probs, actual))
    return float(np.mean(briers))


def main():
    df = pd.read_csv("data/results.csv")
    evaluator = ModelEvaluator(df)

    print(f"Grid-searching {len(ALPHAS)}x{len(ALPHAS)} configs "
          f"over World Cups {YEARS}...\n")

    results = []
    for imp_alpha, conf_alpha in itertools.product(ALPHAS, ALPHAS):
        brier = score_config(df, evaluator, imp_alpha, conf_alpha)
        results.append((imp_alpha, conf_alpha, brier))
        print(f"  importance_alpha={imp_alpha:.2f}  "
              f"confederation_alpha={conf_alpha:.2f}  ->  Brier={brier:.5f}")

    results.sort(key=lambda r: r[2])
    best = results[0]
    baseline = next(r for r in results if r[0] == 0.0 and r[1] == 0.0)

    print("\n=== Best configurations (lowest Brier) ===")
    for imp_a, conf_a, brier in results[:5]:
        print(f"  importance_alpha={imp_a:.2f}  "
              f"confederation_alpha={conf_a:.2f}  ->  Brier={brier:.5f}")

    print(f"\nNo weighting (0, 0) baseline Brier: {baseline[2]:.5f}")
    print(f"Best config: importance_alpha={best[0]:.2f}, "
          f"confederation_alpha={best[1]:.2f}  ->  Brier={best[2]:.5f}")
    improvement = (baseline[2] - best[2]) / baseline[2] * 100
    print(f"Improvement over no weighting: {improvement:+.2f}%")


if __name__ == "__main__":
    main()
