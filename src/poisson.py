"""
poisson.py
----------
Dixon-Coles Poisson model for predicting match scorelines.

The model estimates:
    - attack_i  : attacking strength for team i
    - defense_i : defensive weakness for team i
    - home_adv  : global home advantage in goals

Goals scored by team i vs. team j at home are modeled as:
    Poisson(exp(attack_i + defense_j + home_adv))

The Dixon-Coles correction adjusts the joint probability for low-scoring
results (0-0, 1-0, 0-1, 1-1) which Poisson underestimates.

Usage:
    from src.poisson import DixonColes
    model = DixonColes()
    model.fit(df)                          # df: date, home_team, away_team, home_score, away_score
    probs = model.predict("Brazil", "Argentina", neutral=True)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Optional


class DixonColes:
    def __init__(self, time_decay: float = 0.005):
        """
        Parameters
        ----------
        time_decay : exponential decay weight per day (older matches count less).
                     0.005 ≈ a match from 18 months ago has ~half the weight.
                     Set to 0 to disable.
        """
        self.time_decay  = time_decay
        self.params_     = None   # fitted parameter dict
        self.teams_      = None   # sorted list of teams
        self.is_fitted_  = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> "DixonColes":
        """
        Fit the Dixon-Coles model via MLE.

        Parameters
        ----------
        df             : DataFrame with columns date, home_team, away_team, home_score, away_score
        reference_date : Date to compute time weights from (defaults to max date in df)
        """
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["date"] = pd.to_datetime(df["date"])

        ref  = reference_date or df["date"].max()
        df["weight"] = np.exp(-self.time_decay * (ref - df["date"]).dt.days)

        teams       = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams_ = teams
        n           = len(teams)
        idx         = {t: i for i, t in enumerate(teams)}

        # Initial parameter vector:
        # [attack_0..n-1, defense_0..n-1, home_adv, rho]
        x0 = np.concatenate([
            np.zeros(n),    # attack  (log scale, relative)
            np.zeros(n),    # defense (log scale)
            [0.1],          # home advantage (goals, log scale)
            [-0.1],         # rho (Dixon-Coles correction)
        ])

        # Constraint: sum of attack params = 0 (identifiability)
        def neg_log_likelihood(x):
            attack  = x[:n]
            defense = x[n:2*n]
            home    = x[2*n]
            rho     = x[2*n + 1]

            # Enforce sum-to-zero on attack (pin first team)
            attack = attack - attack.mean()

            ll = 0.0
            for _, row in df.iterrows():
                i  = idx[row["home_team"]]
                j  = idx[row["away_team"]]
                hg = int(row["home_score"])
                ag = int(row["away_score"])
                w  = row["weight"]

                lam = np.exp(attack[i] + defense[j] + home)  # home expected goals
                mu  = np.exp(attack[j] + defense[i])          # away expected goals

                # Dixon-Coles low-score correction
                tau = self._tau(lam, mu, hg, ag, rho)

                ll += w * (np.log(tau + 1e-10)
                           + poisson.logpmf(hg, lam)
                           + poisson.logpmf(ag, mu))

            return -ll

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )

        x       = result.x
        attack  = x[:n] - x[:n].mean()
        defense = x[n:2*n]
        home    = x[2*n]
        rho     = x[2*n + 1]

        self.params_ = {
            "attack":   {t: attack[idx[t]]  for t in teams},
            "defense":  {t: defense[idx[t]] for t in teams},
            "home_adv": home,
            "rho":      rho,
        }
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def score_matrix(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        max_goals: int = 8,
    ) -> np.ndarray:
        """
        Returns (max_goals+1) x (max_goals+1) matrix of scoreline probabilities.
        Rows = home goals, columns = away goals.
        """
        self._check_fitted()
        lam, mu = self._lambdas(home_team, away_team, neutral)

        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = self._tau(lam, mu, i, j, self.params_["rho"])
                matrix[i, j] = (tau
                                * poisson.pmf(i, lam)
                                * poisson.pmf(j, mu))
        # Normalize to sum to 1 (truncation artifact)
        matrix /= matrix.sum()
        return matrix

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        max_goals: int = 8,
    ) -> dict:
        """
        Predict match outcome probabilities.

        Returns dict with p_home_win, p_draw, p_away_win, expected_home, expected_away.
        """
        matrix = self.score_matrix(home_team, away_team, neutral, max_goals)

        p_home = float(np.sum(np.tril(matrix, -1)))   # home goals > away
        p_away = float(np.sum(np.triu(matrix,  1)))   # away goals > home
        p_draw = float(np.trace(matrix))

        lam, mu = self._lambdas(home_team, away_team, neutral)

        return {
            "home_team":     home_team,
            "away_team":     away_team,
            "p_home_win":    round(p_home, 4),
            "p_draw":        round(p_draw, 4),
            "p_away_win":    round(p_away, 4),
            "expected_home": round(lam, 3),
            "expected_away": round(mu,  3),
        }

    def predict_knockout(self, team_a: str, team_b: str, max_goals: int = 8) -> dict:
        """
        Predict win probabilities for a knockout (no draw) match.
        Draws are resolved proportionally between the two teams.
        """
        result    = self.predict(team_a, team_b, neutral=True, max_goals=max_goals)
        p_draw    = result["p_draw"]
        p_a       = result["p_home_win"] + p_draw / 2
        p_b       = result["p_away_win"] + p_draw / 2
        return {
            "team_a":  team_a,
            "team_b":  team_b,
            "p_a_win": round(p_a, 4),
            "p_b_win": round(p_b, 4),
        }

    def get_team_strengths(self) -> pd.DataFrame:
        """Return attack/defense parameters for all teams, sorted by net strength."""
        self._check_fitted()
        rows = []
        for t in self.teams_:
            atk = self.params_["attack"][t]
            dfn = self.params_["defense"][t]
            rows.append({"team": t, "attack": atk, "defense": dfn, "net": atk - dfn})
        return (pd.DataFrame(rows)
                  .sort_values("net", ascending=False)
                  .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lambdas(self, home_team: str, away_team: str, neutral: bool) -> tuple[float, float]:
        p    = self.params_
        home = 0.0 if neutral else p["home_adv"]
        lam  = np.exp(p["attack"][home_team] + p["defense"][away_team] + home)
        mu   = np.exp(p["attack"][away_team] + p["defense"][home_team])
        return lam, mu

    @staticmethod
    def _tau(lam: float, mu: float, x: int, y: int, rho: float) -> float:
        """Dixon-Coles low-score correction factor."""
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet. Call .fit(df) first.")
