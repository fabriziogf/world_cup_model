"""
test_elo.py
-----------
Tests for the EloSystem class.
"""

import pandas as pd
import pytest

from src.elo import EloSystem, BASE_RATING


def _match(home, away, hs, as_, date="2020-01-01", tournament="Friendly", neutral=False):
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": tournament,
        "neutral": neutral,
    }


@pytest.fixture
def simple_matches():
    """A small chronological set of matches between four teams."""
    return pd.DataFrame([
        _match("Brazil", "Chile", 3, 0, "2020-01-01"),
        _match("Germany", "France", 1, 1, "2020-01-08"),
        _match("Brazil", "Germany", 2, 1, "2020-01-15"),
        _match("France", "Chile", 0, 2, "2020-01-22"),
    ])


# ----------------------------------------------------------------------
# Initialisation
# ----------------------------------------------------------------------

def test_teams_initialise_at_base_rating():
    """Before any update, an unseen team is rated at BASE_RATING (1500)."""
    elo = EloSystem()
    assert elo.ratings.get("Brazil", elo.base_rating) == BASE_RATING
    assert elo.base_rating == 1500


def test_single_match_first_appearance_starts_from_base():
    """The first match's pre-match ratings should both be the base rating."""
    elo = EloSystem()
    df = pd.DataFrame([_match("Brazil", "Chile", 1, 0)])
    elo.compute_ratings(df)
    hist = elo.get_history().iloc[0]
    assert hist["pre_elo_home"] == BASE_RATING
    assert hist["pre_elo_away"] == BASE_RATING


# ----------------------------------------------------------------------
# Win / loss dynamics
# ----------------------------------------------------------------------

def test_winner_gains_loser_loses():
    """After a decisive result, the winner's Elo rises and the loser's falls."""
    elo = EloSystem()
    df = pd.DataFrame([_match("Brazil", "Chile", 4, 0)])
    elo.compute_ratings(df)
    assert elo.ratings["Brazil"] > BASE_RATING
    assert elo.ratings["Chile"] < BASE_RATING


def test_draw_between_equal_teams_no_change():
    """A draw between two equal-rated teams at a neutral venue leaves ratings unchanged."""
    elo = EloSystem()
    df = pd.DataFrame([_match("Brazil", "Chile", 1, 1, neutral=True)])
    elo.compute_ratings(df)
    assert elo.ratings["Brazil"] == pytest.approx(BASE_RATING)
    assert elo.ratings["Chile"] == pytest.approx(BASE_RATING)


# ----------------------------------------------------------------------
# Zero-sum property
# ----------------------------------------------------------------------

def test_total_elo_is_constant(simple_matches):
    """Elo updates are zero-sum: total points in the system stay constant."""
    elo = EloSystem()
    elo.compute_ratings(simple_matches)
    total = sum(elo.ratings.values())
    n_teams = len(elo.ratings)
    assert total == pytest.approx(n_teams * BASE_RATING)


# ----------------------------------------------------------------------
# K-factor scaling
# ----------------------------------------------------------------------

def test_higher_k_produces_larger_swing():
    """A World Cup match (K=60) moves ratings more than a friendly (K=10)."""
    df_friendly = pd.DataFrame([_match("Brazil", "Chile", 1, 0, tournament="Friendly")])
    df_wc       = pd.DataFrame([_match("Brazil", "Chile", 1, 0, tournament="FIFA World Cup")])

    elo_f = EloSystem(); elo_f.compute_ratings(df_friendly)
    elo_w = EloSystem(); elo_w.compute_ratings(df_wc)

    swing_friendly = elo_f.ratings["Brazil"] - BASE_RATING
    swing_wc       = elo_w.ratings["Brazil"] - BASE_RATING

    assert swing_wc > swing_friendly


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------

def test_predict_probabilities_sum_to_one():
    """predict() returns home/draw/away probabilities that sum to 1."""
    elo = EloSystem()
    df = pd.DataFrame([_match("Brazil", "Chile", 2, 0)])
    elo.compute_ratings(df)
    pred = elo.predict("Brazil", "Chile")
    total = pred["p_a_win"] + pred["p_draw"] + pred["p_b_win"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_stronger_team_higher_win_prob():
    """A higher-rated team has a greater win probability."""
    elo = EloSystem()
    df = pd.DataFrame([
        _match("Brazil", "Chile", 5, 0),
        _match("Brazil", "Chile", 4, 0),
        _match("Brazil", "Chile", 3, 0),
    ])
    elo.compute_ratings(df)
    pred = elo.predict("Brazil", "Chile")
    assert pred["p_a_win"] > pred["p_b_win"]
