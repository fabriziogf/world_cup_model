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

# Make src importable when run from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fast_poisson import fit_fast, save_model, load_model
from src.simulate import TournamentSimulator


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
    model = fit_fast(df, time_decay=0.005)
    save_model(model, MODEL_CACHE)
    print(f"Model fitted and cached to {MODEL_CACHE}.")


# ----------------------------------------------------------------------
# 3. Define the tournament bracket (8 groups of 4)
#    Replace these with the actual draw. Team names must match the
#    spelling used in results.csv.
# ----------------------------------------------------------------------
bracket = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}


# ----------------------------------------------------------------------
# 4. Run the Monte Carlo simulation
# ----------------------------------------------------------------------
sim = TournamentSimulator(model=model, n_simulations=100_000, seed=42)
probs = sim.simulate(bracket)

print("\n=== World Cup win probabilities ===")
print(probs.to_string(index=False, float_format=lambda x: f"{x:6.2%}"))

winner = probs.iloc[0]
print(f"\nMost likely champion: {winner['team']} ({winner['p_win']:.1%})")
