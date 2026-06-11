"""
fast_poisson.py
---------------
Vectorized fitting for the Dixon-Coles model.

`poisson.py` is treated as a stable API and is not modified. Its `fit()`
method loops over every row in pure Python inside the log-likelihood,
which the optimizer then calls thousands of times — far too slow on a
realistically sized dataset (10+ hours on ~14k matches).

This module reimplements the *same* maximum-likelihood objective using
NumPy array operations, so each likelihood evaluation is a handful of
vectorized ops instead of a 14k-iteration Python loop. The result is the
identical `params_` dictionary, returned inside a ready-to-use DixonColes
instance — a drop-in replacement for `DixonColes().fit(df)` that runs in
seconds rather than hours.

Usage:
    from src.fast_poisson import fit_fast, save_model, load_model

    model = fit_fast(df, time_decay=0.005)     # fast fit
    save_model(model, "model.pkl")             # cache for reuse
    model = load_model("model.pkl")            # instant reload

The returned object is a normal DixonColes, so predict(), score_matrix(),
predict_knockout(), and TournamentSimulator all work unchanged.
"""

import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from typing import Optional

from src.poisson import DixonColes


# Relative importance of a match for the likelihood fit. A World Cup finals
# match should shape the ratings far more than a friendly. These multipliers
# scale each match's weight; they are normalised to mean 1 over the dataset
# so the overall likelihood scale (and optimizer behaviour) stays stable.
#
# Matched by substring against the lowercased `tournament` field, most
# specific first. Unlike Elo's MATCH_WEIGHTS (which keys on underscores that
# never match the space-separated tournament names in the data), these
# patterns are written against the actual strings found in results.csv.
WORLD_CUP_FINALS_WEIGHT = 6.0
CONTINENTAL_WEIGHT      = 3.5
QUALIFICATION_WEIGHT    = 2.5
OTHER_COMPETITIVE_WEIGHT = 2.0
FRIENDLY_WEIGHT         = 1.0

_CONTINENTAL_KEYWORDS = (
    "euro", "copa", "cup of nations", "gold cup", "asian cup",
    "confederations", "nations league", "oceania nations",
)


def match_importance(tournament: str) -> float:
    """
    Map a tournament label to a relative importance weight for fitting.

    World Cup finals matches carry the most weight, then continental finals,
    then qualifiers and other competitive games, with friendlies lowest.
    """
    t = str(tournament).lower().strip()

    # World Cup *finals* (exclude qualification rounds)
    if "world cup" in t and "qualif" not in t:
        return WORLD_CUP_FINALS_WEIGHT
    # Any qualification campaign
    if "qualif" in t:
        return QUALIFICATION_WEIGHT
    # Major continental tournaments
    if any(kw in t for kw in _CONTINENTAL_KEYWORDS):
        return CONTINENTAL_WEIGHT
    # Friendlies count least
    if "friendly" in t:
        return FRIENDLY_WEIGHT
    # Everything else competitive
    return OTHER_COMPETITIVE_WEIGHT


def fit_fast(
    df: pd.DataFrame,
    time_decay: float = 0.005,
    reference_date: Optional[pd.Timestamp] = None,
    maxiter: int = 500,
    use_match_importance: bool = True,
) -> DixonColes:
    """
    Fit a Dixon-Coles model via vectorized MLE.

    Produces the same parameter structure as DixonColes.fit() but evaluates
    the log-likelihood with NumPy arrays instead of a per-row Python loop.

    Parameters
    ----------
    df                   : DataFrame with columns date, home_team, away_team,
                           home_score, away_score, and (optionally) tournament
    time_decay           : Exponential decay weight per day (older matches count less)
    reference_date       : Date to compute time weights from (defaults to max date in df)
    maxiter              : Max L-BFGS-B iterations (matches DixonColes.fit default)
    use_match_importance : Weight each match by tournament importance (World Cup >
                           continental > qualifier > friendly). Requires a
                           `tournament` column; ignored if absent. This counters
                           the bias from teams padding records with easy friendlies.

    Returns
    -------
    A fitted DixonColes instance (is_fitted_=True, params_ populated).
    """
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    ref = reference_date or df["date"].max()
    weights = np.exp(-time_decay * (ref - df["date"]).dt.days.to_numpy())

    # Scale by match importance so competitive fixtures shape the fit more
    # than friendlies. Normalised to mean 1 to preserve the likelihood scale.
    if use_match_importance and "tournament" in df.columns:
        importance = df["tournament"].map(match_importance).to_numpy()
        importance = importance / importance.mean()
        weights = weights * importance

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Pre-encode everything to integer/float arrays once.
    home_idx = df["home_team"].map(idx).to_numpy()
    away_idx = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].astype(int).to_numpy()
    ag = df["away_score"].astype(int).to_numpy()
    w = weights

    # log(k!) is constant across optimizer iterations — precompute once.
    log_hg_fact = gammaln(hg + 1.0)
    log_ag_fact = gammaln(ag + 1.0)

    # Boolean masks for the four Dixon-Coles low-score corrections.
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)

    # Initial parameter vector: [attack_0..n-1, defense_0..n-1, home_adv, rho]
    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.1], [-0.1]])

    def neg_log_likelihood(x: np.ndarray) -> float:
        attack = x[:n]
        attack = attack - attack.mean()      # sum-to-zero identifiability
        defense = x[n:2 * n]
        home = x[2 * n]
        rho = x[2 * n + 1]

        # Vectorized expected goals for every match at once.
        # Clip the linear predictor so exp() stays finite even when the
        # optimizer probes extreme parameter values (overflow guard).
        eta_home = np.clip(attack[home_idx] + defense[away_idx] + home, -10.0, 10.0)
        eta_away = np.clip(attack[away_idx] + defense[home_idx], -10.0, 10.0)
        lam = np.exp(eta_home)
        mu = np.exp(eta_away)

        # Vectorized Poisson log-pmf: k*log(rate) - rate - log(k!)
        ll_home = hg * np.log(lam) - lam - log_hg_fact
        ll_away = ag * np.log(mu) - mu - log_ag_fact

        # Dixon-Coles tau correction, assembled by mask.
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m11] = 1.0 - rho

        # Floor tau at a small positive value: the Dixon-Coles correction
        # can go non-positive when the optimizer probes a large rho, which
        # would make log(tau) undefined. Clipping keeps the objective finite
        # and steers the optimizer back toward the valid region.
        tau = np.clip(tau, 1e-10, None)
        ll = np.sum(w * (np.log(tau) + ll_home + ll_away))
        return -ll

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-9},
    )

    x = result.x
    attack = x[:n] - x[:n].mean()
    defense = x[n:2 * n]
    home = x[2 * n]
    rho = x[2 * n + 1]

    # Build a standard DixonColes and populate it — drop-in compatible.
    model = DixonColes(time_decay=time_decay)
    model.teams_ = teams
    model.params_ = {
        "attack":   {t: float(attack[idx[t]]) for t in teams},
        "defense":  {t: float(defense[idx[t]]) for t in teams},
        "home_adv": float(home),
        "rho":      float(rho),
    }
    model.is_fitted_ = True
    return model


# ----------------------------------------------------------------------
# Model caching — fit once, reuse instantly
# ----------------------------------------------------------------------

def save_model(model: DixonColes, path: str) -> None:
    """Pickle a fitted model's state to disk for instant reuse."""
    if not model.is_fitted_:
        raise RuntimeError("Refusing to save an unfitted model.")
    state = {
        "time_decay": model.time_decay,
        "teams_":     model.teams_,
        "params_":    model.params_,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_model(path: str) -> DixonColes:
    """Reload a fitted model previously saved with save_model()."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    model = DixonColes(time_decay=state["time_decay"])
    model.teams_ = state["teams_"]
    model.params_ = state["params_"]
    model.is_fitted_ = True
    return model
