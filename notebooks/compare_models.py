"""
compare_models.py
-----------------
Backtest the Elo + boosted-trees ensemble against the (tuned) Dixon-Coles
model on past World Cups. For each target tournament, both models are
trained on all matches before it and scored on the tournament's matches.

Metrics (3-way home/draw/away):
    Brier score  (lower better, primary)
    Log-loss     (lower better)
    Accuracy     (higher better)

Run from the project root:
    python notebooks/compare_models.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_poisson import fit_fast
from src.ensemble import EloXGBoostModel
from src.evaluate import ModelEvaluator

YEARS = [2014, 2018, 2022]


def brier3(probs, actual):
    return sum((p - o) ** 2 for p, o in zip(probs, actual)) / 3.0


def logloss3(probs, actual):
    eps = 1e-7
    return -sum(o * np.log(max(p, eps)) for p, o in zip(probs, actual))


def evaluate(model, eval_df):
    """Return (mean_brier, mean_logloss, accuracy) for a fitted model."""
    briers, loglosses, correct = [], [], 0
    for _, m in eval_df.iterrows():
        home, away = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        neutral = bool(m.get("neutral", True))
        actual = [int(hs > as_), int(hs == as_), int(as_ > hs)]
        try:
            p = model.predict(home, away, neutral=neutral)
            probs = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
        except Exception:
            probs = [1 / 3, 1 / 3, 1 / 3]
        briers.append(brier3(probs, actual))
        loglosses.append(logloss3(probs, actual))
        if int(np.argmax(probs)) == int(np.argmax(actual)):
            correct += 1
    n = len(eval_df)
    return np.mean(briers), np.mean(loglosses), correct / n


def main():
    df = pd.read_csv("data/results.csv")
    evaluator = ModelEvaluator(df)

    rows = []
    for year in YEARS:
        train_df, eval_df = evaluator._split(year)
        print(f"{year}: training both models on {len(train_df):,} matches, "
              f"evaluating on {len(eval_df)} World Cup matches...")

        dc = fit_fast(train_df, time_decay=0.005,
                      importance_alpha=0.0, confederation_alpha=1.0)
        ens = EloXGBoostModel().fit(train_df)

        b_dc, l_dc, a_dc = evaluate(dc, eval_df)
        b_en, l_en, a_en = evaluate(ens, eval_df)
        rows.append({
            "year": year, "n": len(eval_df),
            "brier_dc": b_dc, "brier_ens": b_en,
            "logloss_dc": l_dc, "logloss_ens": l_en,
            "acc_dc": a_dc, "acc_ens": a_en,
        })

    res = pd.DataFrame(rows)

    print("\n=== Per-tournament results ===")
    for _, r in res.iterrows():
        print(f"\n{int(r['year'])}  ({int(r['n'])} matches)")
        print(f"  Brier    Dixon-Coles {r['brier_dc']:.4f}   Ensemble {r['brier_ens']:.4f}")
        print(f"  Log-loss Dixon-Coles {r['logloss_dc']:.4f}   Ensemble {r['logloss_ens']:.4f}")
        print(f"  Accuracy Dixon-Coles {r['acc_dc']:.3f}    Ensemble {r['acc_ens']:.3f}")

    print("\n=== Overall (mean across tournaments) ===")
    print(f"  Brier    Dixon-Coles {res['brier_dc'].mean():.4f}   "
          f"Ensemble {res['brier_ens'].mean():.4f}")
    print(f"  Log-loss Dixon-Coles {res['logloss_dc'].mean():.4f}   "
          f"Ensemble {res['logloss_ens'].mean():.4f}")
    print(f"  Accuracy Dixon-Coles {res['acc_dc'].mean():.3f}    "
          f"Ensemble {res['acc_ens'].mean():.3f}")

    improvement = (res['brier_dc'].mean() - res['brier_ens'].mean()) / res['brier_dc'].mean() * 100
    print(f"\n  Ensemble Brier improvement over Dixon-Coles: {improvement:+.2f}%")


if __name__ == "__main__":
    main()
