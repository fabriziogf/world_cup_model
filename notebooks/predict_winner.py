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

from src.poisson import DixonColes
from src.simulate import TournamentSimulator


# ----------------------------------------------------------------------
# 1. Load historical match data
# ----------------------------------------------------------------------
df = pd.read_csv("data/results.csv")

# Optional but recommended: focus on recent matches so the fit reflects
# current team strength (and so the slow MLE fit runs faster).
df = df[df["date"] >= "2014-01-01"].copy()
print(f"Training on {len(df):,} matches since 2014...")


# ----------------------------------------------------------------------
# 2. Fit the Dixon-Coles scoreline model
# ----------------------------------------------------------------------
model = DixonColes(time_decay=0.005)   # older matches count less
model.fit(df)
print("Model fitted.")


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
