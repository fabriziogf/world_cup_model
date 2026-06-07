"""
explore.py  (convert to Jupyter with: jupytext --to notebook explore.py)
----------
End-to-end walkthrough: load data → fit Elo → fit Dixon-Coles → predict.

Data source: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Download results.csv and place it in ../data/results.csv
"""

import sys
sys.path.append("..")

import pandas as pd
import matplotlib.pyplot as plt
from src.elo import EloSystem
from src.poisson import DixonColes

# -----------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------
df = pd.read_csv("../data/results.csv", parse_dates=["date"])

# Keep only matches from 1990 onward — earlier data is noisy
df = df[df["date"].dt.year >= 1990].copy()

# Standardize tournament column to match MATCH_WEIGHTS keys
def classify_tournament(t: str) -> str:
    t = str(t).lower()
    if "world cup" in t and "qualification" not in t:
        return "world_cup"
    if "qualification" in t:
        return "qualification"
    if "friendly" in t:
        return "friendly"
    # UEFA Euros, Copa America, AFCON, etc.
    return "confederation"

df["tournament"] = df["tournament"].apply(classify_tournament)

print(f"Loaded {len(df):,} matches | {df['date'].min().date()} → {df['date'].max().date()}")
print(df["tournament"].value_counts())

# -----------------------------------------------------------------------
# 2. Fit Elo ratings
# -----------------------------------------------------------------------
elo = EloSystem(k_base=20, home_advantage=100)
elo.load_matches("../data/results.csv")  # re-reads, applies classify inside compute

# For Elo we compute on the enriched df directly
elo.matches = df
elo.compute_ratings()

ratings = elo.get_ratings(min_matches=20)
print("\nTop 20 teams by Elo:")
print(ratings.head(20).to_string(index=False))

# -----------------------------------------------------------------------
# 3. Fit Dixon-Coles model
# -----------------------------------------------------------------------
# Use last 4 years of data for the Poisson model
# (recent form matters more than decade-old results)
recent = df[df["date"].dt.year >= df["date"].dt.year.max() - 4]

dc = DixonColes(time_decay=0.005)
dc.fit(recent)

strengths = dc.get_team_strengths()
print("\nTop 20 team strengths (attack - defense):")
print(strengths.head(20).to_string(index=False))

# -----------------------------------------------------------------------
# 4. Sample prediction
# -----------------------------------------------------------------------
match = dc.predict("Brazil", "Argentina", neutral=True)
print(f"\nBrazil vs Argentina (neutral):")
print(f"  Brazil win : {match['p_home_win']:.1%}")
print(f"  Draw       : {match['p_draw']:.1%}")
print(f"  Argentina  : {match['p_away_win']:.1%}")
print(f"  xG Brazil  : {match['expected_home']:.2f}  xG Argentina: {match['expected_away']:.2f}")

# Knockout version (no draws)
ko = dc.predict_knockout("Brazil", "Argentina")
print(f"\n  Knockout — Brazil: {ko['p_a_win']:.1%}  Argentina: {ko['p_b_win']:.1%}")

# -----------------------------------------------------------------------
# 5. Scoreline heatmap
# -----------------------------------------------------------------------
matrix = dc.score_matrix("Brazil", "Argentina", neutral=True)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(matrix[:6, :6], cmap="Blues")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(range(6)); ax.set_yticklabels(range(6))
ax.set_xlabel("Argentina goals"); ax.set_ylabel("Brazil goals")
ax.set_title("Scoreline probabilities: Brazil vs Argentina")

for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{matrix[i, j]:.2%}", ha="center", va="center", fontsize=8)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("brazil_vs_argentina_scoreline.png", dpi=150)
plt.show()
print("\nHeatmap saved.")
