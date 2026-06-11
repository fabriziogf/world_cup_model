"""
evaluate_tournament_level.py
----------------------------
Tournament-level backtest: score each model on how much probability it
assigned to the teams that actually reached the final stages of the
2014, 2018, and 2022 World Cups — the metric per-match Brier can't see.

For each tournament, both models are trained on all matches before that
year, then the actual bracket is simulated and scored against the real
champion / finalists / semifinalists.

Run from the project root:
    python notebooks/evaluate_tournament_level.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_poisson import fit_fast
from src.ensemble import EloXGBoostModel
from src.simulate import TournamentSimulator
from src.tournaments import HISTORICAL_TOURNAMENTS, score_tournament

N_SIMS = 30_000


def fit_dixon_coles(train_df):
    return fit_fast(train_df, time_decay=0.005,
                    importance_alpha=0.0, confederation_alpha=1.0)


def fit_ensemble(train_df):
    return EloXGBoostModel().fit(train_df)


def main():
    df = pd.read_csv("data/results.csv")
    df["date"] = pd.to_datetime(df["date"])

    records = []
    for year, info in HISTORICAL_TOURNAMENTS.items():
        train_df = df[df["date"].dt.year < year].copy()
        bracket = info["bracket"]
        print(f"{year}: training on {len(train_df):,} matches, "
              f"actual champion = {info['champion']}")

        for name, fitter in [("Dixon-Coles", fit_dixon_coles), ("Ensemble", fit_ensemble)]:
            model = fitter(train_df)
            sim = TournamentSimulator(model=model, n_simulations=N_SIMS, seed=42)
            probs = sim.simulate(bracket)
            s = score_tournament(probs, info)
            s.update({"year": year, "model": name})
            records.append(s)
            print(f"    {name:12s}  champion {info['champion']} "
                  f"p={s['champion_prob']:.3f}  rank={s['champion_rank']}  "
                  f"logloss={s['champion_logloss']:.3f}")

    res = pd.DataFrame(records)

    print("\n=== Champion probability (higher = better) ===")
    print(_pivot(res, "champion_prob").to_string())
    print("\n=== Champion rank in predicted table (lower = better) ===")
    print(_pivot(res, "champion_rank").to_string())
    print("\n=== Champion log-loss (lower = better) ===")
    print(_pivot(res, "champion_logloss").to_string())
    print("\n=== Finalist log-loss (lower = better) ===")
    print(_pivot(res, "finalist_logloss").to_string())
    print("\n=== Semifinalist log-loss (lower = better) ===")
    print(_pivot(res, "semifinal_logloss").to_string())

    print("\n=== Overall means ===")
    summary = res.groupby("model")[
        ["champion_prob", "champion_rank", "champion_logloss",
         "finalist_logloss", "semifinal_logloss"]
    ].mean().round(4)
    print(summary.to_string())


def _pivot(res, col):
    return res.pivot(index="year", columns="model", values=col)


if __name__ == "__main__":
    main()
