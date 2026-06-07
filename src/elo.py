"""
elo.py
------
Elo rating system for international football.

Key design decisions:
- K-factor varies by match importance (friendly vs. World Cup)
- Home advantage is modeled as a fixed Elo bonus for the home team
- Ratings are initialized at 1500 for all teams, then converge with history
- Margin of victory is NOT factored in (keeps it simple and robust)

Usage:
    from src.elo import EloSystem
    elo = EloSystem()
    elo.load_matches("data/results.csv")
    elo.compute_ratings()
    ratings = elo.get_ratings()
"""

import pandas as pd
import numpy as np
from typing import Optional


# Match importance weights (K-factor multipliers)
# Based on FIFA's own weighting scheme — a reasonable prior
MATCH_WEIGHTS = {
    "friendly":              10,
    "qualification":         25,
    "confederation":         35,
    "confederation_final":   40,
    "world_cup":             60,
    "world_cup_final":       60,
}

HOME_ADVANTAGE = 100   # Elo points added to home team's effective rating
BASE_RATING    = 1500  # Starting Elo for all teams


class EloSystem:
    def __init__(
        self,
        k_base: int = 20,
        home_advantage: float = HOME_ADVANTAGE,
        base_rating: float = BASE_RATING,
    ):
        """
        Parameters
        ----------
        k_base        : Base K-factor (scaled by match weight)
        home_advantage: Elo bonus for the home team
        base_rating   : Initial Elo for unseen teams
        """
        self.k_base         = k_base
        self.home_advantage = home_advantage
        self.base_rating    = base_rating
        self.ratings: dict[str, float] = {}
        self.history: list[dict]       = []   # one row per match, post-update ratings

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_matches(self, path: str) -> pd.DataFrame:
        """
        Load a CSV of historical match results.

        Expected columns (football-data.co.uk / Kaggle style):
            date, home_team, away_team, home_score, away_score, tournament, neutral

        'neutral' should be 1/True if played at a neutral venue (no home advantage).
        'tournament' maps to MATCH_WEIGHTS keys (friendly, qualification, world_cup, …).
        """
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self.matches = df
        return df

    def compute_ratings(self, matches: Optional[pd.DataFrame] = None) -> None:
        """
        Iterate through matches chronologically and update Elo ratings.
        Populates self.ratings (current) and self.history (full audit trail).
        """
        df = matches if matches is not None else self.matches
        self.ratings  = {}
        self.history  = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            r_home = self.ratings.get(home, self.base_rating)
            r_away = self.ratings.get(away, self.base_rating)

            # Apply home advantage unless neutral venue
            is_neutral   = bool(row.get("neutral", 0))
            ha_bonus     = 0 if is_neutral else self.home_advantage
            r_home_adj   = r_home + ha_bonus

            # Expected scores (logistic)
            e_home, e_away = self._expected(r_home_adj, r_away)

            # Actual scores
            s_home, s_away = self._actual(row["home_score"], row["away_score"])

            # K-factor for this match type
            k = self._k_factor(row.get("tournament", "friendly"))

            # Rating updates
            delta         = k * (s_home - e_home)
            self.ratings[home] = r_home + delta
            self.ratings[away] = r_away - delta   # zero-sum

            self.history.append({
                "date":          row["date"],
                "home_team":     home,
                "away_team":     away,
                "home_score":    row["home_score"],
                "away_score":    row["away_score"],
                "tournament":    row.get("tournament", "friendly"),
                "neutral":       is_neutral,
                "pre_elo_home":  r_home,
                "pre_elo_away":  r_away,
                "post_elo_home": self.ratings[home],
                "post_elo_away": self.ratings[away],
                "p_home_win":    self._win_prob(r_home_adj, r_away),
            })

    def get_ratings(self, min_matches: int = 0) -> pd.DataFrame:
        """Return current ratings as a sorted DataFrame."""
        df = pd.DataFrame(
            self.ratings.items(), columns=["team", "elo"]
        ).sort_values("elo", ascending=False).reset_index(drop=True)

        if min_matches > 0:
            counts = self._match_counts()
            df = df[df["team"].map(counts).fillna(0) >= min_matches]

        return df

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def predict(self, team_a: str, team_b: str, neutral: bool = True) -> dict:
        """
        Predict win/draw/loss probabilities for a single upcoming match.

        Returns a dict with keys: p_a_win, p_draw, p_b_win, elo_a, elo_b.
        Draw probability is estimated from historical base rate (~25% in intl football).
        """
        r_a = self.ratings.get(team_a, self.base_rating)
        r_b = self.ratings.get(team_b, self.base_rating)

        ha  = 0 if neutral else self.home_advantage
        p_a = self._win_prob(r_a + ha, r_b)
        p_b = self._win_prob(r_b, r_a + ha)

        # Shrink win probs toward a draw bucket (simple linear adjustment)
        draw_base = 0.25
        scale     = 1 - draw_base
        p_a_adj   = p_a * scale
        p_b_adj   = p_b * scale
        p_draw    = 1 - p_a_adj - p_b_adj

        return {
            "team_a":    team_a,
            "team_b":    team_b,
            "elo_a":     round(r_a, 1),
            "elo_b":     round(r_b, 1),
            "p_a_win":   round(p_a_adj, 4),
            "p_draw":    round(p_draw,  4),
            "p_b_win":   round(p_b_adj, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expected(r_a: float, r_b: float) -> tuple[float, float]:
        """Standard Elo expected score (logistic with 400-point scale)."""
        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
        return e_a, 1 - e_a

    @staticmethod
    def _actual(home_score: int, away_score: int) -> tuple[float, float]:
        """Convert scoreline to Elo outcome (1 / 0.5 / 0)."""
        if home_score > away_score:
            return 1.0, 0.0
        elif home_score < away_score:
            return 0.0, 1.0
        else:
            return 0.5, 0.5

    @staticmethod
    def _win_prob(r_a: float, r_b: float) -> float:
        return 1 / (1 + 10 ** ((r_b - r_a) / 400))

    def _k_factor(self, tournament: str) -> float:
        t = str(tournament).lower().strip()
        for key, weight in MATCH_WEIGHTS.items():
            if key in t:
                return weight
        return self.k_base   # fallback

    def _match_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.history:
            counts[row["home_team"]] = counts.get(row["home_team"], 0) + 1
            counts[row["away_team"]] = counts.get(row["away_team"], 0) + 1
        return counts
