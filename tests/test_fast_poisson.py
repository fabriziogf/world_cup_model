"""
test_fast_poisson.py
--------------------
Tests for the vectorized Dixon-Coles fitter and match-importance weighting.
"""

import numpy as np
import pandas as pd
import pytest

from src.poisson import DixonColes
from src.fast_poisson import (
    fit_fast,
    match_importance,
    save_model,
    load_model,
    WORLD_CUP_FINALS_WEIGHT,
    QUALIFICATION_WEIGHT,
    CONTINENTAL_WEIGHT,
    FRIENDLY_WEIGHT,
)


def _synthetic(tournament="Friendly", reps=8):
    """Four teams with engineered strengths Brazil > Germany > France > Chile."""
    rng = np.random.default_rng(0)
    teams = ["Brazil", "Germany", "France", "Chile"]
    strength = {"Brazil": 2.2, "Germany": 1.7, "France": 1.3, "Chile": 0.8}
    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(reps):
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                rows.append({
                    "date": date,
                    "home_team": teams[i],
                    "away_team": teams[j],
                    "home_score": int(rng.poisson(strength[teams[i]])),
                    "away_score": int(rng.poisson(strength[teams[j]])),
                    "tournament": tournament,
                })
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# match_importance mapping
# ----------------------------------------------------------------------

def test_world_cup_finals_highest():
    assert match_importance("FIFA World Cup") == WORLD_CUP_FINALS_WEIGHT


def test_qualification_excluded_from_finals():
    """A qualification match must not be treated as a finals match."""
    assert match_importance("FIFA World Cup qualification") == QUALIFICATION_WEIGHT


def test_continental_tournaments():
    for label in ["UEFA Euro", "Copa América", "African Cup of Nations", "UEFA Nations League"]:
        assert match_importance(label) == CONTINENTAL_WEIGHT


def test_friendly_lowest():
    assert match_importance("Friendly") == FRIENDLY_WEIGHT


def test_importance_ordering():
    """World Cup > continental > qualification > friendly."""
    assert (match_importance("FIFA World Cup")
            > match_importance("Copa América")
            > match_importance("FIFA World Cup qualification")
            > match_importance("Friendly"))


# ----------------------------------------------------------------------
# fit_fast correctness
# ----------------------------------------------------------------------

def test_fit_fast_produces_fitted_model():
    model = fit_fast(_synthetic(), time_decay=0)
    assert model.is_fitted_
    assert set(model.teams_) == {"Brazil", "Germany", "France", "Chile"}
    for key in ("attack", "defense", "home_adv", "rho"):
        assert key in model.params_


def test_fit_fast_recovers_strength_ordering():
    model = fit_fast(_synthetic(), time_decay=0)
    order = list(model.get_team_strengths()["team"])
    assert order.index("Brazil") < order.index("Chile")


def test_fit_fast_matches_slow_fit_without_importance():
    """
    With importance weighting off and a single tournament type, fit_fast
    should reproduce the original DixonColes.fit() parameters.
    """
    df = _synthetic()
    slow = DixonColes(time_decay=0).fit(df)
    fast = fit_fast(
        df, time_decay=0,
        use_match_importance=False,
        use_confederation_strength=False,
    )
    for team in slow.teams_:
        assert fast.params_["attack"][team] == pytest.approx(
            slow.params_["attack"][team], abs=1e-4
        )


def test_importance_weighting_changes_fit():
    """
    Mixing a high-importance result that contradicts the friendly record
    should shift the fitted parameters when importance weighting is on.
    """
    base = _synthetic(tournament="Friendly")
    # Add World Cup matches where the weakest team (Chile) thrashes Brazil
    upsets = pd.DataFrame([
        {"date": pd.Timestamp("2021-01-01"), "home_team": "Chile", "away_team": "Brazil",
         "home_score": 5, "away_score": 0, "tournament": "FIFA World Cup"},
        {"date": pd.Timestamp("2021-01-05"), "home_team": "Chile", "away_team": "Brazil",
         "home_score": 4, "away_score": 0, "tournament": "FIFA World Cup"},
    ])
    df = pd.concat([base, upsets], ignore_index=True)

    weighted   = fit_fast(df, time_decay=0, use_match_importance=True)
    unweighted = fit_fast(df, time_decay=0, use_match_importance=False)

    # The upset is amplified under importance weighting, so Chile's net
    # strength relative to Brazil should be higher than without weighting.
    def net(m, t):
        return m.params_["attack"][t] - m.params_["defense"][t]

    gap_weighted   = net(weighted, "Brazil")   - net(weighted, "Chile")
    gap_unweighted = net(unweighted, "Brazil") - net(unweighted, "Chile")
    assert gap_weighted < gap_unweighted


# ----------------------------------------------------------------------
# Caching round-trip
# ----------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    model = fit_fast(_synthetic(), time_decay=0)
    path = tmp_path / "model.pkl"
    save_model(model, str(path))
    reloaded = load_model(str(path))

    assert reloaded.teams_ == model.teams_
    assert reloaded.params_ == model.params_
    assert (reloaded.predict_knockout("Brazil", "Chile")
            == model.predict_knockout("Brazil", "Chile"))


def test_save_unfitted_raises(tmp_path):
    with pytest.raises(RuntimeError):
        save_model(DixonColes(), str(tmp_path / "x.pkl"))
