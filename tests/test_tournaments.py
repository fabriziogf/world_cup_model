"""
test_tournaments.py
-------------------
Tests for the tournament-level scoring function and historical data.
"""

import numpy as np
import pandas as pd
import pytest

from src.tournaments import HISTORICAL_TOURNAMENTS, score_tournament


def _probs(d):
    """Build a probs-style DataFrame from {team: (p_win, p_final, p_semi)}."""
    rows = [{"team": t, "p_win": pw, "p_final": pf, "p_semi": ps}
            for t, (pw, pf, ps) in d.items()]
    return pd.DataFrame(rows)


def test_historical_data_well_formed():
    for year, info in HISTORICAL_TOURNAMENTS.items():
        teams = [t for ts in info["bracket"].values() for t in ts]
        assert len(info["bracket"]) == 8        # 32-team format
        assert len(teams) == 32 and len(set(teams)) == 32
        assert info["champion"] in info["finalists"]
        assert all(f in info["semifinalists"] for f in info["finalists"])
        # the champion must actually be one of the bracket teams
        assert info["champion"] in teams


def test_score_champion_metrics():
    probs = _probs({
        "Germany":   (0.25, 0.40, 0.60),
        "Argentina": (0.20, 0.35, 0.55),
        "Brazil":    (0.15, 0.25, 0.45),
        "Spain":     (0.40, 0.50, 0.70),  # model's favourite, but didn't win
    })
    outcome = {
        "champion": "Germany",
        "finalists": ["Germany", "Argentina"],
        "semifinalists": ["Germany", "Argentina", "Brazil", "Spain"],
    }
    s = score_tournament(probs, outcome)

    assert s["champion_prob"] == pytest.approx(0.25)
    # Germany has the 2nd-highest p_win (behind Spain)
    assert s["champion_rank"] == 2
    assert s["champion_logloss"] == pytest.approx(-np.log(0.25 + 1e-9), abs=1e-4)
    # champion log-loss falls as champion probability rises
    assert s["champion_logloss"] > 0


def test_higher_champion_prob_scores_better():
    outcome = {"champion": "Germany", "finalists": ["Germany"], "semifinalists": ["Germany"]}
    good = score_tournament(_probs({"Germany": (0.30, 0.5, 0.7)}), outcome)
    bad = score_tournament(_probs({"Germany": (0.03, 0.1, 0.2)}), outcome)
    assert good["champion_logloss"] < bad["champion_logloss"]
    assert good["champion_prob"] > bad["champion_prob"]


def test_missing_champion_handled():
    """If the champion never appears in the table, scoring still returns finite-ish values."""
    probs = _probs({"Spain": (0.5, 0.6, 0.8)})
    outcome = {"champion": "Germany", "finalists": ["Germany"], "semifinalists": ["Germany"]}
    s = score_tournament(probs, outcome)
    assert s["champion_prob"] == 0.0
    assert s["champion_logloss"] > 10        # heavy penalty
    assert s["champion_rank"] == 1           # only one team in the table
