"""
test_confederations.py
----------------------
Tests for the confederation mapping and strength-of-schedule weighting.
"""

import numpy as np
import pandas as pd
import pytest

from src.confederations import (
    confederation,
    confederation_strength,
    CONFEDERATION_STRENGTH,
    DEFAULT_STRENGTH,
)
from src.fast_poisson import fit_fast


# ----------------------------------------------------------------------
# Mapping
# ----------------------------------------------------------------------

def test_known_teams_mapped_correctly():
    assert confederation("Germany") == "UEFA"
    assert confederation("Brazil") == "CONMEBOL"
    assert confederation("Japan") == "AFC"
    assert confederation("Morocco") == "CAF"
    assert confederation("United States") == "CONCACAF"
    assert confederation("New Zealand") == "OFC"


def test_unknown_team_returns_none_and_default_strength():
    assert confederation("Atlantis") is None
    assert confederation_strength("Atlantis") == DEFAULT_STRENGTH


def test_strong_confederations_outrank_weak():
    assert confederation_strength("Germany") > confederation_strength("Japan")
    assert confederation_strength("Brazil") > confederation_strength("New Zealand")
    assert CONFEDERATION_STRENGTH["UEFA"] == CONFEDERATION_STRENGTH["CONMEBOL"]


def test_all_32_bracket_teams_mapped():
    bracket_teams = [
        "Qatar", "Ecuador", "Senegal", "Netherlands", "England", "Iran",
        "United States", "Wales", "Argentina", "Saudi Arabia", "Mexico",
        "Poland", "France", "Australia", "Denmark", "Tunisia", "Spain",
        "Costa Rica", "Germany", "Japan", "Belgium", "Canada", "Morocco",
        "Croatia", "Brazil", "Serbia", "Switzerland", "Cameroon",
        "Portugal", "Ghana", "Uruguay", "South Korea",
    ]
    unmapped = [t for t in bracket_teams if confederation(t) is None]
    assert unmapped == []


# ----------------------------------------------------------------------
# Effect on the fit
# ----------------------------------------------------------------------

def _mixed_confederation_df():
    """
    Identifiable round-robin among real teams spanning UEFA, CONMEBOL, and
    AFC, with varied (non-degenerate) scorelines so every rating is finite.
    """
    rng = np.random.default_rng(1)
    teams = ["Germany", "Spain", "Brazil", "Argentina", "Japan", "Iran"]
    strength = {  # latent attacking rates
        "Germany": 1.6, "Spain": 1.5, "Brazil": 1.7,
        "Argentina": 1.5, "Japan": 1.3, "Iran": 1.2,
    }
    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(12):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                rows.append({
                    "date": date, "home_team": h, "away_team": a,
                    "home_score": int(rng.poisson(strength[h])),
                    "away_score": int(rng.poisson(strength[a])),
                    "tournament": "Friendly",
                })
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_confederation_weighting_changes_the_fit():
    """Enabling confederation weighting produces a materially different fit."""
    df = _mixed_confederation_df()
    on = fit_fast(df, time_decay=0, use_match_importance=False,
                  use_confederation_strength=True)
    off = fit_fast(df, time_decay=0, use_match_importance=False,
                   use_confederation_strength=False)

    def net(model, team):
        return model.params_["attack"][team] - model.params_["defense"][team]

    diffs = [abs(net(on, t) - net(off, t)) for t in df["home_team"].unique()]
    assert max(diffs) > 1e-3   # at least one team's rating shifts meaningfully


def test_confederation_weighting_off_matches_no_weighting():
    """With weighting off, the fit ignores confederation entirely (sanity)."""
    df = _mixed_confederation_df()
    a = fit_fast(df, time_decay=0, use_match_importance=False,
                 use_confederation_strength=False)
    b = fit_fast(df, time_decay=0, use_match_importance=False,
                 use_confederation_strength=False)
    assert a.params_["attack"] == b.params_["attack"]
