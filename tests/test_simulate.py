"""
test_simulate.py
----------------
Tests for the TournamentSimulator.

To keep tests fast we bypass the slow MLE fit and inject engineered
Dixon-Coles parameters directly. The simulator only reads params_,
teams_, and is_fitted_, so this exercises the bracket logic in isolation.
"""

import pandas as pd
import pytest

from src.poisson import DixonColes
from src.simulate import TournamentSimulator


# 32 teams across 8 groups
GROUPS = {
    "A": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "B": ["England", "Iran", "USA", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    "H": ["Netherlands", "Ecuador", "Senegal", "Tonga"],
}

ALL_TEAMS = [t for teams in GROUPS.values() for t in teams]
FAVOURITE = "Brazil"
OUTSIDER  = "Tonga"


def _engineered_model() -> DixonColes:
    """Build a DixonColes with hand-set params so Brazil dominates and Tonga is weak."""
    model = DixonColes()
    attack  = {}
    defense = {}
    for t in ALL_TEAMS:
        attack[t]  = 0.0
        defense[t] = 0.0
    # Strong favourite: high attack, strong (negative) defensive value
    attack[FAVOURITE]  = 1.2
    defense[FAVOURITE] = -1.2
    # Clear outsider: weak attack, leaky defence
    attack[OUTSIDER]  = -1.2
    defense[OUTSIDER] = 1.2

    model.teams_ = sorted(ALL_TEAMS)
    model.params_ = {
        "attack":   attack,
        "defense":  defense,
        "home_adv": 0.1,
        "rho":      -0.1,
    }
    model.is_fitted_ = True
    return model


@pytest.fixture(scope="module")
def sim_result():
    model = _engineered_model()
    sim = TournamentSimulator(model=model, n_simulations=3000, seed=42)
    return sim.simulate(GROUPS)


# ----------------------------------------------------------------------
# Probability structure
# ----------------------------------------------------------------------

def test_win_probabilities_sum_to_one(sim_result):
    """Exactly one champion per simulation -> p_win sums to 1."""
    assert sim_result["p_win"].sum() == pytest.approx(1.0, abs=1e-9)


def test_stage_sums_are_consistent(sim_result):
    """Each stage has the right number of teams across simulations."""
    assert sim_result["p_final"].sum()      == pytest.approx(2.0,  abs=1e-9)
    assert sim_result["p_semi"].sum()       == pytest.approx(4.0,  abs=1e-9)
    assert sim_result["p_quarter"].sum()    == pytest.approx(8.0,  abs=1e-9)
    assert sim_result["p_r16"].sum()        == pytest.approx(16.0, abs=1e-9)
    assert sim_result["p_group_exit"].sum() == pytest.approx(16.0, abs=1e-9)


def test_all_teams_present(sim_result):
    """Every team in the bracket appears exactly once in the output."""
    assert set(sim_result["team"]) == set(ALL_TEAMS)
    assert len(sim_result) == len(ALL_TEAMS)


def test_probabilities_in_valid_range(sim_result):
    """All probabilities lie within [0, 1]."""
    for col in ["p_win", "p_final", "p_semi", "p_quarter", "p_r16", "p_group_exit"]:
        assert (sim_result[col] >= 0).all()
        assert (sim_result[col] <= 1).all()


# ----------------------------------------------------------------------
# Favourite vs outsider
# ----------------------------------------------------------------------

def test_favourite_wins_more_than_outsider(sim_result):
    """The clear favourite wins the tournament more often than the outsider."""
    p_fav = sim_result.set_index("team").loc[FAVOURITE, "p_win"]
    p_out = sim_result.set_index("team").loc[OUTSIDER, "p_win"]
    assert p_fav > p_out


def test_favourite_is_top_ranked(sim_result):
    """The favourite should be the single most likely champion."""
    top_team = sim_result.iloc[0]["team"]
    assert top_team == FAVOURITE


# ----------------------------------------------------------------------
# Mid-tournament update
# ----------------------------------------------------------------------

def test_mid_tournament_update_changes_probabilities():
    """
    If the favourite loses all its group matches, its championship
    probability should collapse relative to the clean-slate simulation.
    """
    model = _engineered_model()
    sim = TournamentSimulator(model=model, n_simulations=3000, seed=7)

    baseline = sim.simulate(GROUPS)
    p_fav_baseline = baseline.set_index("team").loc[FAVOURITE, "p_win"]

    # Brazil loses all three group matches heavily
    results_so_far = pd.DataFrame([
        {"home_team": "Serbia",   "away_team": FAVOURITE,      "home_score": 3, "away_score": 0},
        {"home_team": FAVOURITE,  "away_team": "Switzerland",  "home_score": 0, "away_score": 3},
        {"home_team": "Cameroon", "away_team": FAVOURITE,      "home_score": 3, "away_score": 0},
    ])

    updated = sim.simulate_from_current(GROUPS, results_so_far)
    p_fav_updated = updated.set_index("team").loc[FAVOURITE, "p_win"]

    # Brazil was eliminated in the group stage -> can't win
    assert p_fav_updated < p_fav_baseline
    assert p_fav_updated == pytest.approx(0.0, abs=1e-9)


def test_known_result_respected_in_group():
    """A team handed three group wins should reliably advance from its group."""
    model = _engineered_model()
    sim = TournamentSimulator(model=model, n_simulations=2000, seed=11)

    # Tonga (the outsider) wins all three group H matches
    results_so_far = pd.DataFrame([
        {"home_team": OUTSIDER,  "away_team": "Netherlands", "home_score": 2, "away_score": 0},
        {"home_team": "Ecuador", "away_team": OUTSIDER,      "home_score": 0, "away_score": 2},
        {"home_team": OUTSIDER,  "away_team": "Senegal",     "home_score": 2, "away_score": 0},
    ])

    updated = sim.simulate_from_current(GROUPS, results_so_far)
    p_out_r16 = updated.set_index("team").loc[OUTSIDER, "p_r16"]

    # With 9 points, the outsider should advance in the vast majority of sims
    assert p_out_r16 > 0.9
