"""
simulate.py
-----------
Monte Carlo tournament bracket simulator for World Cup prediction.

Uses DixonColes.predict_knockout() to resolve each match and runs N
simulations to estimate win probabilities at each stage.

Usage:
    from src.simulate import TournamentSimulator
    from src.poisson import DixonColes

    model = DixonColes()
    model.fit(df)

    bracket = {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "USA", "Wales"],
        ...
    }

    sim = TournamentSimulator(model=model, n_simulations=100_000)
    probs = sim.simulate(bracket)
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.poisson import DixonColes


STAGES = ["p_win", "p_final", "p_semi", "p_quarter", "p_r16", "p_group_exit"]


class TournamentSimulator:
    def __init__(self, model: DixonColes, n_simulations: int = 100_000, seed: Optional[int] = None):
        """
        Parameters
        ----------
        model         : Fitted DixonColes instance
        n_simulations : Number of Monte Carlo iterations
        seed          : Optional RNG seed for reproducibility
        """
        if not model.is_fitted_:
            raise RuntimeError("DixonColes model must be fitted before simulating.")
        self.model = model
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate(self, bracket: dict[str, list[str]]) -> pd.DataFrame:
        """
        Run Monte Carlo simulation over a full tournament bracket.

        Parameters
        ----------
        bracket : dict mapping group label -> list of 4 team names
                  e.g. {"A": ["Brazil", "Serbia", "Switzerland", "Cameroon"], ...}

        Returns
        -------
        DataFrame with columns: team, p_win, p_final, p_semi, p_quarter, p_r16, p_group_exit
        Sorted descending by p_win.
        """
        all_teams = [t for teams in bracket.values() for t in teams]
        counts = {stage: {t: 0 for t in all_teams} for stage in STAGES}
        win_prob_cache = self._build_win_prob_cache(all_teams)

        for _ in range(self.n_simulations):
            result = self._run_once(bracket, win_prob_cache)
            for stage, teams_reached in result.items():
                for t in teams_reached:
                    counts[stage][t] += 1

        n = self.n_simulations
        rows = []
        for team in all_teams:
            rows.append({
                "team":         team,
                "p_win":        counts["p_win"][team] / n,
                "p_final":      counts["p_final"][team] / n,
                "p_semi":       counts["p_semi"][team] / n,
                "p_quarter":    counts["p_quarter"][team] / n,
                "p_r16":        counts["p_r16"][team] / n,
                "p_group_exit": counts["p_group_exit"][team] / n,
            })

        return (pd.DataFrame(rows)
                  .sort_values("p_win", ascending=False)
                  .reset_index(drop=True))

    def simulate_from_current(
        self,
        bracket: dict[str, list[str]],
        results_so_far: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Mid-tournament update: fix known results and simulate remaining matches.

        Parameters
        ----------
        bracket        : Full original bracket dict
        results_so_far : DataFrame with columns home_team, away_team, home_score, away_score

        Returns
        -------
        Same DataFrame shape as simulate(), probabilities updated for remaining matches.
        """
        all_teams = [t for teams in bracket.values() for t in teams]
        counts = {stage: {t: 0 for t in all_teams} for stage in STAGES}
        win_prob_cache = self._build_win_prob_cache(all_teams)

        for _ in range(self.n_simulations):
            result = self._run_once(bracket, win_prob_cache, known_results=results_so_far)
            for stage, teams_reached in result.items():
                for t in teams_reached:
                    counts[stage][t] += 1

        n = self.n_simulations
        rows = []
        for team in all_teams:
            rows.append({
                "team":         team,
                "p_win":        counts["p_win"][team] / n,
                "p_final":      counts["p_final"][team] / n,
                "p_semi":       counts["p_semi"][team] / n,
                "p_quarter":    counts["p_quarter"][team] / n,
                "p_r16":        counts["p_r16"][team] / n,
                "p_group_exit": counts["p_group_exit"][team] / n,
            })

        return (pd.DataFrame(rows)
                  .sort_values("p_win", ascending=False)
                  .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Single simulation run
    # ------------------------------------------------------------------

    def _run_once(
        self,
        bracket: dict[str, list[str]],
        win_prob_cache: dict[tuple[str, str], float],
        known_results: Optional[pd.DataFrame] = None,
    ) -> dict[str, list[str]]:
        """Run one full tournament simulation. Returns dict of stage -> teams that reached it."""

        r16_teams  = []
        group_exit = []

        for group_label, teams in bracket.items():
            standings = self._simulate_group(teams, win_prob_cache, known_results, group_label)
            r16_teams.extend(standings[:2])
            group_exit.extend(standings[2:])

        r16_matches = self._seed_r16(r16_teams, list(bracket.keys()))

        # R16 -> QF participants (8 teams)
        qf_participants = [self._knockout_winner(a, b, win_prob_cache) for a, b in r16_matches]
        # QF -> SF participants (4 teams)
        qf_pairs = [(qf_participants[i], qf_participants[i + 1]) for i in range(0, len(qf_participants), 2)]
        sf_participants = [self._knockout_winner(a, b, win_prob_cache) for a, b in qf_pairs]
        # SF -> Finalists (2 teams)
        sf_pairs = [(sf_participants[i], sf_participants[i + 1]) for i in range(0, len(sf_participants), 2)]
        finalists = [self._knockout_winner(a, b, win_prob_cache) for a, b in sf_pairs]
        # Final -> Champion (1 team)
        champion = [self._knockout_winner(finalists[0], finalists[1], win_prob_cache)]

        return {
            "p_win":        champion,
            "p_final":      finalists,
            "p_semi":       sf_participants,
            "p_quarter":    qf_participants,
            "p_r16":        r16_teams,
            "p_group_exit": group_exit,
        }

    # ------------------------------------------------------------------
    # Group stage
    # ------------------------------------------------------------------

    def _simulate_group(
        self,
        teams: list[str],
        win_prob_cache: dict[tuple[str, str], float],
        known_results: Optional[pd.DataFrame],
        group_label: str,
    ) -> list[str]:
        """
        Simulate a round-robin group. Returns teams sorted by points desc.
        Tiebreaker: approximated goal difference + random noise.
        """
        points = {t: 0 for t in teams}
        gd     = {t: 0.0 for t in teams}

        matchups = [
            (teams[i], teams[j])
            for i in range(len(teams))
            for j in range(i + 1, len(teams))
        ]

        for home, away in matchups:
            known_outcome = None
            if known_results is not None:
                mask = (
                    ((known_results["home_team"] == home) & (known_results["away_team"] == away)) |
                    ((known_results["home_team"] == away) & (known_results["away_team"] == home))
                )
                played = known_results[mask]
                if not played.empty:
                    row = played.iloc[0]
                    if row["home_team"] == home:
                        hs, ag = int(row["home_score"]), int(row["away_score"])
                    else:
                        hs, ag = int(row["away_score"]), int(row["home_score"])
                    known_outcome = (hs, ag)

            if known_outcome is not None:
                hs, ag = known_outcome
                winner = home if hs > ag else (away if ag > hs else None)
                gd[home] += hs - ag
                gd[away] += ag - hs
            else:
                p_home_win = win_prob_cache.get((home, away), 0.5)
                draw_share = 0.25
                p_home_adj = p_home_win * (1 - draw_share)
                p_away_adj = (1 - p_home_win) * (1 - draw_share)

                r = self.rng.random()
                if r < p_home_adj:
                    winner = home
                    gd[home] += 1; gd[away] -= 1
                elif r < p_home_adj + p_away_adj:
                    winner = away
                    gd[away] += 1; gd[home] -= 1
                else:
                    winner = None

            if winner == home:
                points[home] += 3
            elif winner == away:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

        noise = {t: self.rng.random() for t in teams}
        return sorted(teams, key=lambda t: (points[t], gd[t], noise[t]), reverse=True)

    # ------------------------------------------------------------------
    # Bracket seeding
    # ------------------------------------------------------------------

    def _seed_r16(self, r16_teams: list[str], group_labels: list[str]) -> list[tuple[str, str]]:
        """
        Pair group winners vs runners-up from adjacent groups.
        r16_teams layout: [1A, 2A, 1B, 2B, ...] after group loop ordering.
        Standard WC pairing: 1A vs 2B, 1B vs 2A, 1C vs 2D, 1D vs 2C, ...
        """
        n_groups = len(group_labels)
        winners  = [r16_teams[i * 2]     for i in range(n_groups)]
        runners  = [r16_teams[i * 2 + 1] for i in range(n_groups)]

        matches = []
        for i in range(0, n_groups, 2):
            matches.append((winners[i],     runners[i + 1]))
            matches.append((winners[i + 1], runners[i]))
        return matches

    # ------------------------------------------------------------------
    # Knockout helper
    # ------------------------------------------------------------------

    def _knockout_winner(
        self,
        team_a: str,
        team_b: str,
        win_prob_cache: dict[tuple[str, str], float],
    ) -> str:
        p_a = win_prob_cache.get((team_a, team_b), 0.5)
        return team_a if self.rng.random() < p_a else team_b

    # ------------------------------------------------------------------
    # Win probability cache
    # ------------------------------------------------------------------

    def _build_win_prob_cache(self, teams: list[str]) -> dict[tuple[str, str], float]:
        """
        Pre-compute p(A beats B) for all ordered team pairs via predict_knockout().
        All matches treated as neutral venue. Unknown teams fall back to 0.5.
        """
        cache: dict[tuple[str, str], float] = {}
        known_teams = set(self.model.teams_ or [])

        for a in teams:
            for b in teams:
                if a == b or (a, b) in cache:
                    continue
                if a not in known_teams or b not in known_teams:
                    cache[(a, b)] = 0.5
                    cache[(b, a)] = 0.5
                    continue
                result = self.model.predict_knockout(a, b)
                cache[(a, b)] = result["p_a_win"]
                cache[(b, a)] = result["p_b_win"]

        return cache
