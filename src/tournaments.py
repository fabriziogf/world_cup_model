"""
tournaments.py
--------------
Historical World Cup brackets and outcomes, plus a tournament-level
scoring function.

Per-match Brier score is dominated by lopsided group games and barely
moves when the champion ranking goes from absurd to sensible (see
blogs/08). This module scores a model on what we actually care about:
how much probability it assigned to the teams that actually reached the
final stages.

Team names use the spellings in results.csv. All three tournaments are
the 32-team format (8 groups), which the simulator handles directly.
"""

import numpy as np


HISTORICAL_TOURNAMENTS = {
    2014: {
        "bracket": {
            "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
            "B": ["Spain", "Netherlands", "Chile", "Australia"],
            "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
            "D": ["Uruguay", "Costa Rica", "England", "Italy"],
            "E": ["Switzerland", "Ecuador", "France", "Honduras"],
            "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
            "G": ["Germany", "Portugal", "Ghana", "United States"],
            "H": ["Belgium", "Algeria", "Russia", "South Korea"],
        },
        "champion": "Germany",
        "finalists": ["Germany", "Argentina"],
        "semifinalists": ["Germany", "Argentina", "Brazil", "Netherlands"],
    },
    2018: {
        "bracket": {
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
        "champion": "France",
        "finalists": ["France", "Croatia"],
        "semifinalists": ["France", "Croatia", "Belgium", "England"],
    },
    2022: {
        "bracket": {
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
        "champion": "Argentina",
        "finalists": ["Argentina", "France"],
        "semifinalists": ["Argentina", "France", "Croatia", "Morocco"],
    },
}


def score_tournament(probs, outcome: dict) -> dict:
    """
    Score a simulated probability table against the actual outcome.

    Parameters
    ----------
    probs   : DataFrame from TournamentSimulator.simulate() with columns
              team, p_win, p_final, p_semi, ...
    outcome : dict with keys champion, finalists, semifinalists.

    Returns
    -------
    dict of tournament-level metrics:
        champion_prob     : probability assigned to the actual champion
        champion_logloss  : -log(p_win[champion])  (lower better)
        champion_brier    : multi-class Brier over the "who won" one-hot
        champion_rank     : rank of the actual champion in the p_win ordering
        finalist_logloss  : mean -log(p_final) over the two finalists
        semifinal_logloss : mean -log(p_semi) over the four semifinalists
    """
    eps = 1e-9
    pw = dict(zip(probs["team"], probs["p_win"]))
    pf = dict(zip(probs["team"], probs["p_final"]))
    ps = dict(zip(probs["team"], probs["p_semi"]))

    champ = outcome["champion"]
    champ_prob = pw.get(champ, 0.0)

    champ_brier = float(sum(
        (pw.get(t, 0.0) - (1.0 if t == champ else 0.0)) ** 2 for t in probs["team"]
    ))

    ranking = list(probs.sort_values("p_win", ascending=False)["team"])
    champ_rank = ranking.index(champ) + 1 if champ in ranking else len(ranking)

    finalist_ll = float(np.mean([-np.log(pf.get(t, 0.0) + eps) for t in outcome["finalists"]]))
    semifinal_ll = float(np.mean([-np.log(ps.get(t, 0.0) + eps) for t in outcome["semifinalists"]]))

    return {
        "champion_prob":     round(champ_prob, 4),
        "champion_logloss":  round(-np.log(champ_prob + eps), 4),
        "champion_brier":    round(champ_brier, 4),
        "champion_rank":     champ_rank,
        "finalist_logloss":  round(finalist_ll, 4),
        "semifinal_logloss": round(semifinal_ll, 4),
    }
