"""
test_ensemble.py
----------------
Tests for the Elo + gradient-boosted-trees ensemble.

Uses a small synthetic dataset with a clear strength ordering so the
learned model's head-to-head probabilities are checkable. The booster is
whichever backend is available (xgboost or sklearn-histgb); tests assert
behaviour, not the backend.
"""

import numpy as np
import pandas as pd
import pytest

from src.ensemble import EloXGBoostModel, _ewm_ppg, _tournament_weight, _h2h_winrate
from src.simulate import TournamentSimulator


@pytest.fixture(scope="module")
def fitted_model():
    """Fit once on synthetic data with strengths Brazil > Germany > France > Chile."""
    rng = np.random.default_rng(0)
    teams = ["Brazil", "Germany", "France", "Chile"]
    strength = {"Brazil": 2.3, "Germany": 1.8, "France": 1.3, "Chile": 0.7}
    rows = []
    date = pd.Timestamp("2018-01-01")
    for _ in range(40):
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                h, a = teams[i], teams[j]
                rows.append({
                    "date": date,
                    "home_team": h,
                    "away_team": a,
                    "home_score": int(rng.poisson(strength[h])),
                    "away_score": int(rng.poisson(strength[a])),
                    "tournament": "Friendly",
                    "neutral": False,
                })
                date += pd.Timedelta(days=2)
    df = pd.DataFrame(rows)
    return EloXGBoostModel(n_estimators=80, max_depth=3).fit(df)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def test_ewm_ppg_baseline_and_recency():
    assert _ewm_ppg([]) == 1.0                       # neutral baseline
    assert _ewm_ppg([3, 3, 3]) == pytest.approx(3.0)  # all wins -> 3 PPG
    # recent results dominate: a recent win outweighs old losses
    assert _ewm_ppg([0, 0, 3]) > _ewm_ppg([3, 0, 0])


def test_tournament_weight_mapping():
    assert _tournament_weight("FIFA World Cup") > _tournament_weight("Friendly")
    assert _tournament_weight("anything else") == 20.0


def test_h2h_winrate():
    assert _h2h_winrate([], "Brazil") == 0.5
    assert _h2h_winrate(["Brazil", "Brazil", None], "Brazil") == pytest.approx(2 / 3)


# ----------------------------------------------------------------------
# Fitting & prediction
# ----------------------------------------------------------------------

def test_fit_sets_state(fitted_model):
    assert fitted_model.is_fitted_
    assert set(fitted_model.teams_) == {"Brazil", "Germany", "France", "Chile"}
    assert fitted_model.backend_ in ("xgboost", "sklearn-histgb")


def test_predict_probabilities_sum_to_one(fitted_model):
    p = fitted_model.predict("Brazil", "Chile", neutral=True)
    total = p["p_home_win"] + p["p_draw"] + p["p_away_win"]
    assert total == pytest.approx(1.0, abs=1e-3)


def test_predict_knockout_sums_to_one(fitted_model):
    r = fitted_model.predict_knockout("Brazil", "Chile")
    assert r["p_a_win"] + r["p_b_win"] == pytest.approx(1.0, abs=1e-3)


def test_stronger_team_favoured(fitted_model):
    r = fitted_model.predict_knockout("Brazil", "Chile")
    assert r["p_a_win"] > r["p_b_win"]


def test_knockout_is_symmetric(fitted_model):
    """Swapping the argument order should swap the probabilities (neutral venue)."""
    r1 = fitted_model.predict_knockout("Brazil", "Chile")
    r2 = fitted_model.predict_knockout("Chile", "Brazil")
    assert r1["p_a_win"] == pytest.approx(r2["p_b_win"], abs=1e-6)


def test_predict_unfitted_raises():
    with pytest.raises(RuntimeError):
        EloXGBoostModel().predict("Brazil", "Chile")


# ----------------------------------------------------------------------
# Integration with the simulator
# ----------------------------------------------------------------------

def test_drops_into_simulator(fitted_model):
    """The ensemble works as a drop-in model for TournamentSimulator."""
    # 32 distinct teams across 8 groups; the four known teams are seeded in,
    # the rest are unknown to the model (cache falls back to 50/50).
    known = ["Brazil", "Germany", "France", "Chile"]
    teams = known + [f"T{i}" for i in range(28)]
    bracket = {chr(65 + g): teams[g * 4:(g + 1) * 4] for g in range(8)}

    sim = TournamentSimulator(model=fitted_model, n_simulations=500, seed=1)
    probs = sim.simulate(bracket)
    assert probs["p_win"].sum() == pytest.approx(1.0, abs=1e-9)
    assert len(probs) == 32
    assert (probs["p_win"] >= 0).all() and (probs["p_win"] <= 1).all()
