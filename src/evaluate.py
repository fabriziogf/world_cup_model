"""
evaluate.py
-----------
Backtesting and calibration tools for the World Cup prediction model.

Trains on all data before a target tournament year, evaluates predictions
on tournament matches, and measures:
    - Brier score (primary — lower is better)
    - Log-loss
    - Accuracy (predicted winner matches actual winner)
    - Calibration curves (predicted probability vs. actual frequency)

Usage:
    from src.elo import EloSystem
    from src.poisson import DixonColes
    from src.evaluate import ModelEvaluator

    elo = EloSystem()
    df = elo.load_matches("data/results.csv")
    elo.compute_ratings()

    dc = DixonColes()
    dc.fit(df)

    evaluator = ModelEvaluator(df)
    metrics = evaluator.evaluate_tournament(year=2022)
    evaluator.plot_calibration(year=2022)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional

from src.elo import EloSystem
from src.poisson import DixonColes


# Tournament name fragments that identify World Cup group/knockout matches
WORLD_CUP_LABELS = ["fifa world cup", "world cup"]

# Calibration bins
N_BINS = 10


class ModelEvaluator:
    """
    Backtesting and calibration evaluator for Elo and Dixon-Coles models.

    For a given World Cup year, trains both models on all matches strictly
    before the tournament start date, then evaluates on the tournament matches.

    Parameters
    ----------
    df          : Full historical match DataFrame (date, home_team, away_team,
                  home_score, away_score, tournament, neutral)
    elo_params  : Optional kwargs forwarded to EloSystem constructor
    dc_params   : Optional kwargs forwarded to DixonColes constructor
    """

    def __init__(
        self,
        df: pd.DataFrame,
        elo_params: Optional[dict] = None,
        dc_params: Optional[dict] = None,
    ):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)
        self.elo_params = elo_params or {}
        self.dc_params = dc_params or {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate_tournament(self, year: int) -> dict:
        """
        Train on data before the tournament, evaluate on tournament matches.

        Parameters
        ----------
        year : World Cup year (e.g. 2014, 2018, 2022)

        Returns
        -------
        dict with keys:
            year, n_matches,
            brier_elo, brier_dc,
            logloss_elo, logloss_dc,
            accuracy_elo, accuracy_dc,
            predictions  (DataFrame with per-match detail)
        """
        train_df, eval_df = self._split(year)

        if eval_df.empty:
            raise ValueError(f"No World Cup matches found for year {year}.")

        # Fit models on training data
        elo = EloSystem(**self.elo_params)
        elo.compute_ratings(train_df)

        dc = DixonColes(**self.dc_params)
        dc.fit(train_df)

        # Generate predictions for each tournament match
        rows = []
        for _, match in eval_df.iterrows():
            home = match["home_team"]
            away = match["away_team"]
            hs   = int(match["home_score"])
            as_  = int(match["away_score"])

            actual_home_win = 1 if hs > as_ else 0
            actual_draw     = 1 if hs == as_ else 0
            actual_away_win = 1 if as_ > hs else 0

            # Elo prediction
            try:
                ep = elo.predict(home, away, neutral=True)
                p_h_elo = ep["p_a_win"]
                p_d_elo = ep["p_draw"]
                p_a_elo = ep["p_b_win"]
            except Exception:
                p_h_elo, p_d_elo, p_a_elo = 1/3, 1/3, 1/3

            # Dixon-Coles prediction
            try:
                dp = dc.predict(home, away, neutral=True)
                p_h_dc = dp["p_home_win"]
                p_d_dc = dp["p_draw"]
                p_a_dc = dp["p_away_win"]
            except Exception:
                p_h_dc, p_d_dc, p_a_dc = 1/3, 1/3, 1/3

            # Brier score per match (3-outcome)
            brier_elo = self._brier3(
                [p_h_elo, p_d_elo, p_a_elo],
                [actual_home_win, actual_draw, actual_away_win]
            )
            brier_dc = self._brier3(
                [p_h_dc, p_d_dc, p_a_dc],
                [actual_home_win, actual_draw, actual_away_win]
            )

            # Log-loss per match (3-outcome)
            logloss_elo = self._logloss3(
                [p_h_elo, p_d_elo, p_a_elo],
                [actual_home_win, actual_draw, actual_away_win]
            )
            logloss_dc = self._logloss3(
                [p_h_dc, p_d_dc, p_a_dc],
                [actual_home_win, actual_draw, actual_away_win]
            )

            # Accuracy: predicted winner == actual winner
            pred_winner_elo = self._predicted_winner(p_h_elo, p_d_elo, p_a_elo)
            pred_winner_dc  = self._predicted_winner(p_h_dc,  p_d_dc,  p_a_dc)
            actual_winner   = "home" if hs > as_ else ("away" if as_ > hs else "draw")

            rows.append({
                "date":           match["date"],
                "home_team":      home,
                "away_team":      away,
                "home_score":     hs,
                "away_score":     as_,
                # Elo
                "p_home_elo":     round(p_h_elo, 4),
                "p_draw_elo":     round(p_d_elo, 4),
                "p_away_elo":     round(p_a_elo, 4),
                "brier_elo":      round(brier_elo, 4),
                "logloss_elo":    round(logloss_elo, 4),
                "correct_elo":    int(pred_winner_elo == actual_winner),
                # Dixon-Coles
                "p_home_dc":      round(p_h_dc, 4),
                "p_draw_dc":      round(p_d_dc, 4),
                "p_away_dc":      round(p_a_dc, 4),
                "brier_dc":       round(brier_dc, 4),
                "logloss_dc":     round(logloss_dc, 4),
                "correct_dc":     int(pred_winner_dc == actual_winner),
                # Actuals
                "actual_winner":  actual_winner,
            })

        preds = pd.DataFrame(rows)

        return {
            "year":         year,
            "n_matches":    len(preds),
            "brier_elo":    round(preds["brier_elo"].mean(), 4),
            "brier_dc":     round(preds["brier_dc"].mean(), 4),
            "logloss_elo":  round(preds["logloss_elo"].mean(), 4),
            "logloss_dc":   round(preds["logloss_dc"].mean(), 4),
            "accuracy_elo": round(preds["correct_elo"].mean(), 4),
            "accuracy_dc":  round(preds["correct_dc"].mean(), 4),
            "predictions":  preds,
        }

    def compare_models(self, years: Optional[list[int]] = None) -> pd.DataFrame:
        """
        Side-by-side Brier score comparison across multiple World Cup years.

        Parameters
        ----------
        years : List of World Cup years to evaluate (default: [2014, 2018, 2022])

        Returns
        -------
        DataFrame with columns: year, n_matches, brier_elo, brier_dc,
                                logloss_elo, logloss_dc, accuracy_elo, accuracy_dc
        """
        years = years or [2014, 2018, 2022]
        rows = []
        for year in years:
            try:
                m = self.evaluate_tournament(year)
                rows.append({k: v for k, v in m.items() if k != "predictions"})
            except ValueError as e:
                print(f"Skipping {year}: {e}")

        return pd.DataFrame(rows)

    def plot_calibration(
        self,
        year: int,
        model: str = "dc",
        outcome: str = "home",
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plot a calibration curve: predicted probability vs. actual frequency.

        Parameters
        ----------
        year    : World Cup year
        model   : "dc" (Dixon-Coles) or "elo"
        outcome : "home", "draw", or "away"
        ax      : Optional existing matplotlib Axes to draw on
        save_path : If provided, save the figure to this path

        Returns
        -------
        matplotlib Axes object
        """
        metrics = self.evaluate_tournament(year)
        preds   = metrics["predictions"]

        prob_col   = f"p_{outcome}_{model}"
        actual_col = "actual_winner"
        actual_val = outcome  # "home", "draw", or "away"

        if prob_col not in preds.columns:
            raise ValueError(f"Column '{prob_col}' not found. Choose model='elo'/'dc' and outcome='home'/'draw'/'away'.")

        probs   = preds[prob_col].values
        actuals = (preds[actual_col] == actual_val).astype(int).values

        bin_edges   = np.linspace(0, 1, N_BINS + 1)
        bin_centers = []
        bin_freqs   = []
        bin_counts  = []

        for i in range(N_BINS):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bin_centers.append(probs[mask].mean())
                bin_freqs.append(actuals[mask].mean())
                bin_counts.append(mask.sum())

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

        # Calibration curve
        scatter = ax.scatter(
            bin_centers, bin_freqs,
            s=[max(30, c * 5) for c in bin_counts],
            alpha=0.75,
            color="#1f77b4",
            zorder=3,
            label="Predicted vs. actual",
        )
        ax.plot(bin_centers, bin_freqs, color="#1f77b4", linewidth=1.5, alpha=0.6)

        model_label = "Dixon-Coles" if model == "dc" else "Elo"
        ax.set_title(f"Calibration — {model_label} | {year} World Cup | outcome: {outcome}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Actual frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(True, alpha=0.3)

        if save_path:
            ax.get_figure().savefig(save_path, dpi=150, bbox_inches="tight")

        return ax

    def plot_calibration_grid(
        self,
        years: Optional[list[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a grid of calibration curves: one row per year, one column per model.

        Parameters
        ----------
        years     : World Cup years to include (default: [2014, 2018, 2022])
        save_path : Optional path to save the figure

        Returns
        -------
        matplotlib Figure
        """
        years = years or [2014, 2018, 2022]
        fig, axes = plt.subplots(
            len(years), 2,
            figsize=(12, 5 * len(years)),
            sharex=True, sharey=True,
        )
        if len(years) == 1:
            axes = [axes]  # ensure 2D indexing

        for row_idx, year in enumerate(years):
            for col_idx, model in enumerate(["elo", "dc"]):
                ax = axes[row_idx][col_idx]
                try:
                    self.plot_calibration(year=year, model=model, outcome="home", ax=ax)
                except Exception as e:
                    ax.set_title(f"{year} — {model} (error: {e})")

        fig.suptitle("Model Calibration — Home Win Probability", fontsize=14, y=1.01)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------

    def _split(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train (before tournament) and eval (tournament matches).
        Identifies tournament matches by year and World Cup label in the tournament column.
        """
        df = self.df

        # Training: everything strictly before the tournament year
        train_df = df[df["date"].dt.year < year].copy()

        # Eval: matches in the tournament year that are World Cup matches
        # Exclude qualifications — match "FIFA World Cup" but not "qualification"
        year_df = df[df["date"].dt.year == year].copy()
        t_lower = year_df["tournament"].str.lower()
        is_wc   = t_lower.str.contains("|".join(WORLD_CUP_LABELS), na=False)
        is_qual = t_lower.str.contains("qualif", na=False)
        eval_df = year_df[is_wc & ~is_qual].copy()

        return train_df, eval_df

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _brier3(probs: list[float], actuals: list[int]) -> float:
        """
        Multi-outcome Brier score for a 3-class prediction.
        BS = (1/3) * sum((p_i - o_i)^2)
        """
        return sum((p - o) ** 2 for p, o in zip(probs, actuals)) / len(probs)

    @staticmethod
    def _logloss3(probs: list[float], actuals: list[int]) -> float:
        """
        Multi-outcome log-loss for a 3-class prediction.
        Clips probabilities to avoid log(0).
        """
        eps = 1e-7
        return -sum(
            o * np.log(max(p, eps))
            for p, o in zip(probs, actuals)
        )

    @staticmethod
    def _predicted_winner(p_home: float, p_draw: float, p_away: float) -> str:
        """Return the most likely outcome as a string."""
        probs = {"home": p_home, "draw": p_draw, "away": p_away}
        return max(probs, key=probs.get)
