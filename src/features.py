"""
features.py
-----------
Feature engineering pipeline for the World Cup prediction model.

Takes a raw match DataFrame and a fitted EloSystem and produces a
model-ready feature matrix aligned row-by-row with the input matches.

All features are computed with strict no-lookahead: the feature value
at row t uses only data from matches with index < t (i.e., earlier dates).

Usage:
    from src.elo import EloSystem
    from src.features import FeatureBuilder

    elo = EloSystem()
    df = elo.load_matches("data/results.csv")
    elo.compute_ratings()

    builder = FeatureBuilder(elo_system=elo)
    features = builder.transform(df)
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.elo import EloSystem, MATCH_WEIGHTS


class FeatureBuilder:
    """
    Builds a feature matrix from historical match data.

    Parameters
    ----------
    elo_system  : Fitted EloSystem instance (compute_ratings() must have been called)
    form_window : Number of past matches to include in form calculation (default 10)
    h2h_window  : Number of past head-to-head meetings to include (default 10)
    ewm_span    : Exponential weighting span for form calculation (default 5)
    """

    def __init__(
        self,
        elo_system: EloSystem,
        form_window: int = 10,
        h2h_window: int = 10,
        ewm_span: float = 5.0,
    ):
        if not elo_system.history:
            raise RuntimeError(
                "EloSystem has no history. Call compute_ratings() before passing to FeatureBuilder."
            )
        self.elo = elo_system
        self.form_window = form_window
        self.h2h_window = h2h_window
        self.ewm_span = ewm_span

        # Build lookup tables from elo history for fast access
        self._elo_history_df = pd.DataFrame(elo_system.history)
        self._elo_history_df["date"] = pd.to_datetime(self._elo_history_df["date"])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for every row in df.

        Parameters
        ----------
        df : DataFrame with columns: date, home_team, away_team, tournament, neutral
             Must be sorted chronologically (ascending date).

        Returns
        -------
        DataFrame of features, same row order as df, with columns:
            elo_home, elo_away, elo_diff,
            form_home, form_away,
            rest_days_home, rest_days_away,
            is_neutral,
            tournament_weight,
            h2h_home_winrate
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        records = []
        for idx, row in df.iterrows():
            records.append(self._compute_row(row, df.iloc[:idx]))

        return pd.DataFrame(records, index=df.index)

    # ------------------------------------------------------------------
    # Row-level feature computation
    # ------------------------------------------------------------------

    def _compute_row(self, row: pd.Series, past_matches: pd.DataFrame) -> dict:
        """Compute all features for a single match using only past_matches."""
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        elo_home, elo_away = self._get_elo_at(home, away, date)
        form_home = self._get_form(home, date, past_matches)
        form_away = self._get_form(away, date, past_matches)
        rest_home = self._get_rest_days(home, date, past_matches)
        rest_away = self._get_rest_days(away, date, past_matches)
        h2h_rate  = self._get_h2h_winrate(home, away, date, past_matches)
        t_weight  = self._tournament_weight(row.get("tournament", "friendly"))
        is_neutral = bool(row.get("neutral", False))

        return {
            "elo_home":         round(elo_home, 2),
            "elo_away":         round(elo_away, 2),
            "elo_diff":         round(elo_home - elo_away, 2),
            "form_home":        round(form_home, 4),
            "form_away":        round(form_away, 4),
            "rest_days_home":   rest_home,
            "rest_days_away":   rest_away,
            "is_neutral":       int(is_neutral),
            "tournament_weight": t_weight,
            "h2h_home_winrate": round(h2h_rate, 4),
        }

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _get_elo_at(self, home: str, away: str, date: pd.Timestamp) -> tuple[float, float]:
        """
        Return the pre-match Elo ratings for both teams at the given date.
        Uses the last recorded post-match Elo before this date from elo history.
        Falls back to EloSystem.base_rating for unseen teams.
        """
        hist = self._elo_history_df

        def latest_elo(team: str, as_home: bool) -> float:
            col = "post_elo_home" if as_home else "post_elo_away"
            team_col = "home_team" if as_home else "away_team"
            mask = (hist[team_col] == team) & (hist["date"] < date)
            rows = hist[mask]
            if rows.empty:
                return self.elo.base_rating
            return float(rows.iloc[-1][col])

        # Try both home and away appearance in history
        def elo_for(team: str) -> float:
            home_rows = hist[(hist["home_team"] == team) & (hist["date"] < date)]
            away_rows = hist[(hist["away_team"] == team) & (hist["date"] < date)]

            candidates = []
            if not home_rows.empty:
                r = home_rows.iloc[-1]
                candidates.append((r["date"], r["post_elo_home"]))
            if not away_rows.empty:
                r = away_rows.iloc[-1]
                candidates.append((r["date"], r["post_elo_away"]))

            if not candidates:
                return self.elo.base_rating
            # Return the rating from the most recent appearance
            return max(candidates, key=lambda x: x[0])[1]

        return elo_for(home), elo_for(away)

    def _get_form(
        self,
        team: str,
        date: pd.Timestamp,
        past_matches: pd.DataFrame,
    ) -> float:
        """
        Exponentially weighted points per game over the last `form_window` matches.
        Win = 3 pts, Draw = 1 pt, Loss = 0 pts.
        Returns 1.0 (neutral baseline) if no past matches are found.
        """
        if past_matches.empty:
            return 1.0

        mask = (past_matches["home_team"] == team) | (past_matches["away_team"] == team)
        team_matches = past_matches[mask].tail(self.form_window)

        if team_matches.empty:
            return 1.0

        points = []
        for _, m in team_matches.iterrows():
            hs = m["home_score"]
            as_ = m["away_score"]
            if m["home_team"] == team:
                result = 3 if hs > as_ else (1 if hs == as_ else 0)
            else:
                result = 3 if as_ > hs else (1 if hs == as_ else 0)
            points.append(float(result))

        if not points:
            return 1.0

        # Exponential weights: more recent matches weight more
        n = len(points)
        weights = np.array([np.exp((i - n + 1) / self.ewm_span) for i in range(n)])
        weights /= weights.sum()
        return float(np.dot(weights, points))

    def _get_rest_days(
        self,
        team: str,
        date: pd.Timestamp,
        past_matches: pd.DataFrame,
    ) -> int:
        """
        Days since the team's last match before this date.
        Returns -1 if no previous match found (first appearance).
        """
        if past_matches.empty:
            return -1

        mask = (past_matches["home_team"] == team) | (past_matches["away_team"] == team)
        team_matches = past_matches[mask]

        if team_matches.empty:
            return -1

        last_date = pd.to_datetime(team_matches["date"].max())
        return int((date - last_date).days)

    def _get_h2h_winrate(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        past_matches: pd.DataFrame,
    ) -> float:
        """
        Home team's win rate in head-to-head meetings against away team.
        Considers the last `h2h_window` meetings regardless of venue.
        Returns 0.5 (neutral baseline) if no H2H history exists.
        """
        if past_matches.empty:
            return 0.5

        mask = (
            ((past_matches["home_team"] == home) & (past_matches["away_team"] == away)) |
            ((past_matches["home_team"] == away) & (past_matches["away_team"] == home))
        )
        h2h = past_matches[mask].tail(self.h2h_window)

        if h2h.empty:
            return 0.5

        wins = 0
        total = 0
        for _, m in h2h.iterrows():
            hs = m["home_score"]
            as_ = m["away_score"]
            total += 1
            if m["home_team"] == home and hs > as_:
                wins += 1
            elif m["home_team"] == away and as_ > hs:
                wins += 1

        return wins / total if total > 0 else 0.5

    def _tournament_weight(self, tournament: str) -> float:
        """
        Map tournament string to numeric importance weight.
        Uses the same MATCH_WEIGHTS dict as EloSystem._k_factor().
        """
        t = str(tournament).lower().strip()
        for key, weight in MATCH_WEIGHTS.items():
            if key in t:
                return float(weight)
        return float(self.elo.k_base)  # fallback = base K
