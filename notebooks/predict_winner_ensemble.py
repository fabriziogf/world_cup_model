"""
predict_winner_ensemble.py
--------------------------
End-to-end prediction using the Elo + gradient-boosted-trees ensemble
instead of Dixon-Coles. The ensemble swaps the goals-based attack/defense
signal for opponent-adjusted Elo plus a learned multi-feature classifier,
which fixes the confederation/strength-of-schedule bias documented in
blogs/08.

Run from the project root:
    python notebooks/predict_winner_ensemble.py
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ensemble import EloXGBoostModel
from src.simulate import TournamentSimulator
from bracket_2026 import BRACKET_2026


# ----------------------------------------------------------------------
# 1. Fit the ensemble on the full match history
#
#    Elo benefits from the full record (ratings converge), and the
#    gradient-boosted fit is fast. Recent matches are up-weighted via the
#    model's time_decay.
# ----------------------------------------------------------------------
df = pd.read_csv("data/results.csv")
print(f"Fitting Elo + boosted-trees ensemble on {len(df):,} matches...")
model = EloXGBoostModel().fit(df)
print(f"Fitted (booster backend: {model.backend_}).")


# ----------------------------------------------------------------------
# 2. Simulate the 2026 tournament
# ----------------------------------------------------------------------
sim = TournamentSimulator(model=model, n_simulations=100_000, seed=42)
probs = sim.simulate(BRACKET_2026)

print("\n=== World Cup win probabilities (Elo + boosted-trees ensemble) ===")
print(probs.to_string(index=False, float_format=lambda x: f"{x:6.2%}"))

winner = probs.iloc[0]
print(f"\nMost likely champion: {winner['team']} ({winner['p_win']:.1%})")
