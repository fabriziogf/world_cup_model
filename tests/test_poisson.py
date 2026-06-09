"""
test_poisson.py
---------------
Tests for the DixonColes Poisson model.
"""

import numpy as np
import pandas as pd
import pytest

from src.poisson import DixonColes


@pytest.fixture(scope="module")
def fitted_model():
    """
    Fit a small DixonColes model once for the whole module.

    Four teams with engineered scorelines so that strength ordering is
    clear: Brazil > Germany > France > Chile.
    """
    rng = np.random.default_rng(0)
    teams = ["Brazil", "Germany", "France", "Chile"]
    strength = {"Brazil": 2.2, "Germany": 1.7, "France": 1.3, "Chile": 0.8}

    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(12):
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i == j:
                    continue
                h, a = teams[i], teams[j]
                rows.append({
                    "date": date,
                    "home_team": h,
                    "away_team": a,
                    "home_score": int(rng.poisson(strength[h])),
                    "away_score": int(rng.poisson(strength[a])),
                })
                date += pd.Timedelta(days=1)

    df = pd.DataFrame(rows)
    model = DixonColes(time_decay=0)
    model.fit(df)
    return model


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------

def test_fit_runs_and_sets_params(fitted_model):
    """Fitting completes and populates the parameter dict."""
    assert fitted_model.is_fitted_
    assert "attack" in fitted_model.params_
    assert "defense" in fitted_model.params_
    assert "home_adv" in fitted_model.params_
    assert "rho" in fitted_model.params_


# ----------------------------------------------------------------------
# Score matrix
# ----------------------------------------------------------------------

def test_score_matrix_sums_to_one(fitted_model):
    """The scoreline probability matrix is normalised to ~1.0."""
    matrix = fitted_model.score_matrix("Brazil", "Chile")
    assert matrix.sum() == pytest.approx(1.0, abs=1e-6)


def test_score_matrix_is_nonnegative(fitted_model):
    """All scoreline probabilities are non-negative."""
    matrix = fitted_model.score_matrix("Brazil", "Chile")
    assert (matrix >= 0).all()


def test_predict_outcome_probabilities_sum_to_one(fitted_model):
    """predict() home/draw/away probabilities sum to ~1."""
    p = fitted_model.predict("Brazil", "Chile")
    total = p["p_home_win"] + p["p_draw"] + p["p_away_win"]
    assert total == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------------------
# Knockout prediction
# ----------------------------------------------------------------------

def test_predict_knockout_sums_to_one(fitted_model):
    """predict_knockout() splits draws so the two win probs sum to ~1."""
    p = fitted_model.predict_knockout("Brazil", "Chile")
    assert p["p_a_win"] + p["p_b_win"] == pytest.approx(1.0, abs=1e-3)


def test_stronger_team_higher_win_probability(fitted_model):
    """The stronger team (Brazil) beats the weaker one (Chile) more often."""
    p = fitted_model.predict_knockout("Brazil", "Chile")
    assert p["p_a_win"] > p["p_b_win"]


def test_strength_ordering_recovered(fitted_model):
    """The fitted net strengths preserve the engineered ordering."""
    strengths = fitted_model.get_team_strengths()
    order = list(strengths["team"])
    assert order.index("Brazil") < order.index("Chile")


# ----------------------------------------------------------------------
# Dixon-Coles tau correction
# ----------------------------------------------------------------------

def test_tau_edge_cases():
    """The tau correction returns the documented values for low-score cases."""
    lam, mu, rho = 1.5, 1.2, -0.1

    # 0-0: 1 - lam*mu*rho
    assert DixonColes._tau(lam, mu, 0, 0, rho) == pytest.approx(1 - lam * mu * rho)
    # 1-0: 1 + mu*rho
    assert DixonColes._tau(lam, mu, 1, 0, rho) == pytest.approx(1 + mu * rho)
    # 0-1: 1 + lam*rho
    assert DixonColes._tau(lam, mu, 0, 1, rho) == pytest.approx(1 + lam * rho)
    # 1-1: 1 - rho
    assert DixonColes._tau(lam, mu, 1, 1, rho) == pytest.approx(1 - rho)
    # Any other scoreline: 1.0 (no correction)
    assert DixonColes._tau(lam, mu, 2, 3, rho) == 1.0


def test_predict_unfitted_raises():
    """Calling predict before fit raises a clear error."""
    model = DixonColes()
    with pytest.raises(RuntimeError):
        model.predict("Brazil", "Chile")
