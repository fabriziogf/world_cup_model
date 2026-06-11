"""
predict_winner.py
-----------------
End-to-end script: fit the Dixon-Coles model on historical results and
run the Monte Carlo simulator to estimate World Cup win probabilities.

Run from the project root:
    python notebooks/predict_winner.py
"""

import sys
import os
import pandas as pd

# Make src and this folder importable when run from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fast_poisson import fit_fast, save_model, load_model
from src.simulate import TournamentSimulator
from bracket_2026 import BRACKET_2026


MODEL_CACHE = "data/dc_model.pkl"


# ----------------------------------------------------------------------
# 1 & 2. Load/fit the Dixon-Coles model (vectorized + cached)
#
# fit_fast() vectorizes the MLE objective, so fitting ~12k matches takes
# ~10 seconds instead of hours. The fitted model is cached to disk and
# reused on subsequent runs — delete data/dc_model.pkl to refit.
# ----------------------------------------------------------------------
if os.path.exists(MODEL_CACHE):
    model = load_model(MODEL_CACHE)
    print(f"Loaded cached model from {MODEL_CACHE} ({len(model.teams_)} teams).")
else:
    df = pd.read_csv("data/results.csv")
    df = df[df["date"] >= "2014-01-01"].copy()   # recent form, faster fit
    print(f"Fitting on {len(df):,} matches since 2014...")
    # Weights chosen by backtesting against the 2014/2018/2022 World Cups
    # (notebooks/tune_weights.py): confederation strength-of-schedule weighting
    # helps (alpha=1.0), match-importance weighting hurts and is disabled
    # (alpha=0.0). Best Brier 0.2073 vs 0.2122 unweighted (+2.3%).
    model = fit_fast(
        df,
        time_decay=0.005,
        importance_alpha=0.0,
        confederation_alpha=1.0,
    )
    save_model(model, MODEL_CACHE)
    print(f"Model fitted and cached to {MODEL_CACHE}.")


# ----------------------------------------------------------------------
# 3. The tournament bracket — official 2026 World Cup draw (see
#    notebooks/bracket_2026.py). 48 teams in 12 groups; top 2 per group
#    plus the 8 best third-placed teams advance to a Round of 32.
# ----------------------------------------------------------------------
bracket = BRACKET_2026


# ----------------------------------------------------------------------
# 4. Run the Monte Carlo simulation
# ----------------------------------------------------------------------
sim = TournamentSimulator(model=model, n_simulations=100_000, seed=42)
probs = sim.simulate(bracket)

print("\n=== World Cup win probabilities ===")
print(probs.to_string(index=False, float_format=lambda x: f"{x:6.2%}"))

winner = probs.iloc[0]
print(f"\nMost likely champion: {winner['team']} ({winner['p_win']:.1%})")
