"""
ensemble.py
-----------
Elo + gradient-boosted-trees ensemble for match outcome prediction.

The Dixon-Coles model reduces each team to a goals-based attack/defense
pair, which rewards running up scores and is blind to opponent quality
(see blogs/08). This ensemble replaces that thin signal with a learned,
multi-feature classifier on top of opponent-adjusted Elo ratings:

    elo_diff, elo_home, elo_away,
    form_home, form_away,          (exp-weighted PPG over last 10)
    rest_days_home, rest_days_away,
    is_neutral, tournament_weight,
    h2h_home_winrate

A gradient-boosted-trees model learns a 3-class target
(home win / draw / away win) from historical matches, with strictly
leak-free features (row t uses only matches before t).

Booster: XGBoost is used when its native library loads; otherwise the
model falls back to scikit-learn's HistGradientBoostingClassifier, which
is a functionally equivalent gradient-boosted-trees implementation with
no OpenMP-dylib dependency. Both expose the same fit / predict_proba API.

The fitted object exposes predict(), predict_knockout(), teams_, and
is_fitted_, so it is a drop-in for DixonColes inside TournamentSimulator.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.elo import EloSystem, MATCH_WEIGHTS

# Ordered feature columns fed to the booster.
FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "form_home", "form_away",
    "rest_days_home", "rest_days_away",
    "is_neutral", "tournament_weight", "h2h_home_winrate",
]

# Class labels for the 3-way target.
HOME_WIN, DRAW, AWAY_WIN = 0, 1, 2

# At prediction time we don't know the real fixture schedule, so rest days
# are set to a neutral constant (symmetric, non-discriminating).
DEFAULT_PRED_REST = 4
# Tournament importance assumed for predicted (World Cup) matches.
WORLD_CUP_WEIGHT = float(MATCH_WEIGHTS["world_cup"])

FORM_WINDOW = 10
H2H_WINDOW = 10
EWM_SPAN = 5.0
NEUTRAL_FORM = 1.0
NEUTRAL_H2H = 0.5


def _make_booster(n_estimators: int, max_depth: int, learning_rate: float, seed: int):
    """
    Return (classifier, backend_name). Prefer XGBoost; fall back to sklearn's
    HistGradientBoostingClassifier if XGBoost's native library can't load.
    """
    try:
        import xgboost as xgb  # raises XGBoostError if libxgboost can't load
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
        )
        return clf, "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        return clf, "sklearn-histgb"


def _ewm_ppg(points: list[float]) -> float:
    """Exponentially weighted points-per-game over the recent results list."""
    if not points:
        return NEUTRAL_FORM
    pts = points[-FORM_WINDOW:]
    n = len(pts)
    w = np.array([np.exp((i - n + 1) / EWM_SPAN) for i in range(n)])
    w /= w.sum()
    return float(np.dot(w, pts))


def _h2h_winrate(history: list[Optional[str]], team: str) -> float:
    """Win rate of `team` over recent head-to-head meetings (draws count as 0)."""
    recent = history[-H2H_WINDOW:]
    if not recent:
        return NEUTRAL_H2H
    wins = sum(1 for w in recent if w == team)
    return wins / len(recent)


def _tournament_weight(tournament: str) -> float:
    """Map a tournament label to its importance weight (MATCH_WEIGHTS, default 20)."""
    t = str(tournament).lower().strip()
    for key, weight in MATCH_WEIGHTS.items():
        if key in t:
            return float(weight)
    return 20.0


class EloXGBoostModel:
    """
    Elo + gradient-boosted-trees ensemble.

    Parameters
    ----------
    time_decay    : Per-day exponential decay for training sample weights
                    (recent matches matter more). 0 disables.
    n_estimators  : Boosting rounds.
    max_depth     : Tree depth.
    learning_rate : Boosting learning rate.
    seed          : RNG seed.
    elo_params    : Optional kwargs for EloSystem.
    """

    def __init__(
        self,
        time_decay: float = 0.0008,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        seed: int = 42,
        elo_params: Optional[dict] = None,
    ):
        self.time_decay = time_decay
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.elo_params = elo_params or {}

        self.booster_ = None
        self.backend_ = None
        self.teams_ = None
        self.is_fitted_ = False

        # Final state captured at the end of fit, used for prediction.
        self._elo: dict[str, float] = {}
        self._form: dict[str, list[float]] = {}
        self._last_date: dict[str, pd.Timestamp] = {}
        self._h2h: dict[tuple[str, str], list[Optional[str]]] = {}
        self._elo_base = 1500.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "EloXGBoostModel":
        """Fit Elo ratings and the gradient-boosted classifier on match history."""
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Elo ratings: history gives pre-match ratings per row (leak-free),
        # and final self.ratings is used for prediction.
        elo = EloSystem(**self.elo_params)
        elo.compute_ratings(df)
        self._elo_base = elo.base_rating
        self._elo = dict(elo.ratings)
        hist = elo.get_history()  # aligned row-for-row with df

        X, y, weights = self._build_training_matrix(df, hist)

        clf, backend = _make_booster(
            self.n_estimators, self.max_depth, self.learning_rate, self.seed
        )
        clf.fit(X, y, sample_weight=weights)

        self.booster_ = clf
        self.backend_ = backend
        self.teams_ = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.is_fitted_ = True
        return self

    def _build_training_matrix(self, df: pd.DataFrame, hist: pd.DataFrame):
        """Single chronological pass building leak-free features for every match."""
        elo_home = hist["pre_elo_home"].to_numpy()
        elo_away = hist["pre_elo_away"].to_numpy()

        rows = np.zeros((len(df), len(FEATURES)), dtype=float)
        labels = np.zeros(len(df), dtype=int)

        ref = df["date"].max()
        weights = np.exp(-self.time_decay * (ref - df["date"]).dt.days.to_numpy())

        form: dict[str, list[float]] = {}
        last_date: dict[str, pd.Timestamp] = {}
        h2h: dict[tuple[str, str], list[Optional[str]]] = {}

        homes = df["home_team"].to_numpy()
        aways = df["away_team"].to_numpy()
        hs_arr = df["home_score"].astype(int).to_numpy()
        as_arr = df["away_score"].astype(int).to_numpy()
        dates = df["date"].to_numpy()
        tournaments = df["tournament"].to_numpy() if "tournament" in df.columns else np.array(["friendly"] * len(df))
        neutrals = df["neutral"].to_numpy() if "neutral" in df.columns else np.zeros(len(df), dtype=bool)

        for i in range(len(df)):
            home, away = homes[i], aways[i]
            hs, ag = hs_arr[i], as_arr[i]
            date = pd.Timestamp(dates[i])
            key = tuple(sorted((home, away)))

            fh = _ewm_ppg(form.get(home, []))
            fa = _ewm_ppg(form.get(away, []))
            rd_home = (date - last_date[home]).days if home in last_date else -1
            rd_away = (date - last_date[away]).days if away in last_date else -1
            h2h_rate = _h2h_winrate(h2h.get(key, []), home)

            rows[i, 0] = elo_home[i]
            rows[i, 1] = elo_away[i]
            rows[i, 2] = elo_home[i] - elo_away[i]
            rows[i, 3] = fh
            rows[i, 4] = fa
            rows[i, 5] = rd_home
            rows[i, 6] = rd_away
            rows[i, 7] = int(bool(neutrals[i]))
            rows[i, 8] = _tournament_weight(tournaments[i])
            rows[i, 9] = h2h_rate

            if hs > ag:
                labels[i] = HOME_WIN
                ph, pa, winner = 3.0, 0.0, home
            elif ag > hs:
                labels[i] = AWAY_WIN
                ph, pa, winner = 0.0, 3.0, away
            else:
                labels[i] = DRAW
                ph, pa, winner = 1.0, 1.0, None

            form.setdefault(home, []).append(ph)
            form.setdefault(away, []).append(pa)
            last_date[home] = date
            last_date[away] = date
            h2h.setdefault(key, []).append(winner)

        # Capture final state (trimmed) for prediction.
        self._form = {t: lst[-FORM_WINDOW:] for t, lst in form.items()}
        self._last_date = last_date
        self._h2h = {k: lst[-H2H_WINDOW:] for k, lst in h2h.items()}

        return rows, labels, weights

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _feature_vector(self, home: str, away: str, neutral: bool) -> np.ndarray:
        eh = self._elo.get(home, self._elo_base)
        ea = self._elo.get(away, self._elo_base)
        fh = _ewm_ppg(self._form.get(home, []))
        fa = _ewm_ppg(self._form.get(away, []))
        key = tuple(sorted((home, away)))
        h2h_rate = _h2h_winrate(self._h2h.get(key, []), home)

        return np.array([[
            eh, ea, eh - ea,
            fh, fa,
            DEFAULT_PRED_REST, DEFAULT_PRED_REST,
            int(bool(neutral)), WORLD_CUP_WEIGHT, h2h_rate,
        ]], dtype=float)

    def _proba(self, home: str, away: str, neutral: bool) -> np.ndarray:
        """Return [p_home_win, p_draw, p_away_win] for the given orientation."""
        self._check_fitted()
        raw = self.booster_.predict_proba(self._feature_vector(home, away, neutral))[0]
        # Map booster column order to [HOME_WIN, DRAW, AWAY_WIN].
        classes = list(getattr(self.booster_, "classes_", [HOME_WIN, DRAW, AWAY_WIN]))
        out = np.zeros(3)
        for col, cls in enumerate(classes):
            out[int(cls)] = raw[col]
        s = out.sum()
        return out / s if s > 0 else np.array([1 / 3, 1 / 3, 1 / 3])

    def predict(self, home: str, away: str, neutral: bool = True, **kwargs) -> dict:
        """Predict 3-way outcome probabilities for a match (given orientation)."""
        p = self._proba(home, away, neutral)
        return {
            "home_team": home,
            "away_team": away,
            "p_home_win": round(float(p[HOME_WIN]), 4),
            "p_draw": round(float(p[DRAW]), 4),
            "p_away_win": round(float(p[AWAY_WIN]), 4),
        }

    def predict_knockout(self, team_a: str, team_b: str, **kwargs) -> dict:
        """
        Win probabilities for a neutral knockout match. Averages both
        home/away orientations to remove home-framing bias, then splits the
        draw mass evenly between the teams.
        """
        self._check_fitted()
        p1 = self._proba(team_a, team_b, neutral=True)   # a as home
        p2 = self._proba(team_b, team_a, neutral=True)   # b as home

        p_a = (p1[HOME_WIN] + p2[AWAY_WIN]) / 2
        p_b = (p1[AWAY_WIN] + p2[HOME_WIN]) / 2
        p_d = (p1[DRAW] + p2[DRAW]) / 2

        p_a += p_d / 2
        p_b += p_d / 2
        total = p_a + p_b
        if total > 0:
            p_a, p_b = p_a / total, p_b / total

        return {
            "team_a": team_a,
            "team_b": team_b,
            "p_a_win": round(float(p_a), 4),
            "p_b_win": round(float(p_b), 4),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet. Call .fit(df) first.")
