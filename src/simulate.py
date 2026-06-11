"""
simulate.py
-----------
Monte Carlo tournament bracket simulator for World Cup prediction.

Supports two formats, detected automatically from the number of groups:

  * 32-team format (8 groups of 4): top 2 per group advance to a Round of
    16, then QF, SF, Final.
  * 48-team format (12 groups of 4, e.g. the 2026 World Cup): top 2 per
    group plus the 8 best third-placed teams advance to a Round of 32,
    then R16, QF, SF, Final.

Uses DixonColes.predict_knockout() to resolve each match and runs N
simulations to estimate the probability of each team reaching each stage.

Usage:
    from src.simulate import TournamentSimulator
    from src.fast_poisson import fit_fast

    model = fit_fast(df)
    bracket = { "A": [...4 teams...], ..., "L": [...] }   # 12 groups for 2026
    sim = TournamentSimulator(model=model, n_simulations=100_000)
    probs = sim.simulate(bracket)
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.poisson import DixonColes


# Knockout round sizes mapped to their output column names. A team is
# credited with a stage if it is one of the teams *entering* that round.
_ROUND_NAMES = {
    32: "p_r32",
    16: "p_r16",
    8:  "p_quarter",
    4:  "p_semi",
    2:  "p_final",
    1:  "p_win",
}

# Output stage columns per format (number of groups -> ordered column list).
_STAGES_32 = ["p_win", "p_final", "p_semi", "p_quarter", "p_r16", "p_group_exit"]
_STAGES_48 = ["p_win", "p_final", "p_semi", "p_quarter", "p_r16", "p_r32", "p_group_exit"]


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
        bracket : dict mapping group label -> list of 4 team names.
                  8 groups -> 32-team format; 12 groups -> 48-team (2026) format.

        Returns
        -------
        DataFrame sorted descending by p_win, with one probability column per
        stage. The 48-team format additionally includes p_r32.
        """
        return self._simulate(bracket, known_results=None)

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
        results_so_far : DataFrame with columns home_team, away_team,
                         home_score, away_score

        Returns
        -------
        Same shape as simulate(), with probabilities updated for the matches
        that have not yet been played.
        """
        return self._simulate(bracket, known_results=results_so_far)

    # ------------------------------------------------------------------
    # Core driver
    # ------------------------------------------------------------------

    def _simulate(
        self,
        bracket: dict[str, list[str]],
        known_results: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        self._validate_bracket(bracket)

        all_teams = [t for teams in bracket.values() for t in teams]
        stages = self._stage_columns(len(bracket))
        counts = {stage: {t: 0 for t in all_teams} for stage in stages}
        win_prob_cache = self._build_win_prob_cache(all_teams)

        for _ in range(self.n_simulations):
            result = self._run_once(bracket, win_prob_cache, known_results)
            for stage, teams_reached in result.items():
                for t in teams_reached:
                    counts[stage][t] += 1

        n = self.n_simulations
        rows = []
        for team in all_teams:
            row = {"team": team}
            for stage in stages:
                row[stage] = counts[stage][team] / n
            rows.append(row)

        return (pd.DataFrame(rows, columns=["team"] + stages)
                  .sort_values("p_win", ascending=False)
                  .reset_index(drop=True))

    @staticmethod
    def _validate_bracket(bracket: dict[str, list[str]]) -> None:
        n_groups = len(bracket)
        if n_groups not in (8, 12):
            raise ValueError(
                f"Unsupported bracket: {n_groups} groups. "
                "Use 8 groups (32-team) or 12 groups (48-team)."
            )
        for label, teams in bracket.items():
            if len(teams) != 4:
                raise ValueError(f"Group {label} has {len(teams)} teams; expected 4.")

    @staticmethod
    def _stage_columns(n_groups: int) -> list[str]:
        return _STAGES_48 if n_groups == 12 else _STAGES_32

    # ------------------------------------------------------------------
    # Single simulation run
    # ------------------------------------------------------------------

    def _run_once(
        self,
        bracket: dict[str, list[str]],
        win_prob_cache: dict[tuple[str, str], float],
        known_results: Optional[pd.DataFrame] = None,
    ) -> dict[str, list[str]]:
        """Run one full tournament simulation. Returns dict of stage -> teams reaching it."""
        # standings[group] = list of (team, points, gd) ordered best-first
        standings = {
            g: self._simulate_group(teams, win_prob_cache, known_results, g)
            for g, teams in bracket.items()
        }
        if len(bracket) == 12:
            return self._knockout_48(standings, win_prob_cache)
        return self._knockout_32(standings, win_prob_cache)

    # ------------------------------------------------------------------
    # Group stage
    # ------------------------------------------------------------------

    def _simulate_group(
        self,
        teams: list[str],
        win_prob_cache: dict[tuple[str, str], float],
        known_results: Optional[pd.DataFrame],
        group_label: str,
    ) -> list[tuple[str, int, float]]:
        """
        Simulate a round-robin group.

        Returns a list of (team, points, goal_difference) ordered best-first.
        Ordering is by points, then approximated goal difference, then a random
        tiebreaker.
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
        ordered = sorted(teams, key=lambda t: (points[t], gd[t], noise[t]), reverse=True)
        return [(t, points[t], gd[t]) for t in ordered]

    # ------------------------------------------------------------------
    # Knockout — 32-team format (8 groups, top 2 -> Round of 16)
    # ------------------------------------------------------------------

    def _knockout_32(
        self,
        standings: dict[str, list[tuple[str, int, float]]],
        cache: dict[tuple[str, str], float],
    ) -> dict[str, list[str]]:
        group_labels = list(standings.keys())

        r16_teams = []   # ordered [1A, 2A, 1B, 2B, ...]
        group_exit = []
        for g in group_labels:
            st = standings[g]
            r16_teams.append(st[0][0])
            r16_teams.append(st[1][0])
            group_exit.extend([st[2][0], st[3][0]])

        matches = self._seed_r16(r16_teams, group_labels)
        order = [team for pair in matches for team in pair]

        result = self._play_bracket(order, cache)
        result["p_group_exit"] = group_exit
        return result

    def _seed_r16(self, r16_teams: list[str], group_labels: list[str]) -> list[tuple[str, str]]:
        """
        Pair group winners vs runners-up from adjacent groups.
        r16_teams layout: [1A, 2A, 1B, 2B, ...].
        Standard pairing: 1A vs 2B, 1B vs 2A, 1C vs 2D, 1D vs 2C, ...
        """
        n_groups = len(group_labels)
        winners = [r16_teams[i * 2]     for i in range(n_groups)]
        runners = [r16_teams[i * 2 + 1] for i in range(n_groups)]

        matches = []
        for i in range(0, n_groups, 2):
            matches.append((winners[i],     runners[i + 1]))
            matches.append((winners[i + 1], runners[i]))
        return matches

    # ------------------------------------------------------------------
    # Knockout — 48-team format (12 groups, top 2 + 8 best thirds -> R32)
    # ------------------------------------------------------------------

    def _knockout_48(
        self,
        standings: dict[str, list[tuple[str, int, float]]],
        cache: dict[tuple[str, str], float],
    ) -> dict[str, list[str]]:
        winners, runners, thirds = [], [], []
        group_exit = []
        for g, st in standings.items():
            winners.append(st[0])
            runners.append(st[1])
            thirds.append(st[2])
            group_exit.append(st[3][0])   # 4th place is always eliminated

        # The 8 best third-placed teams advance; the other 4 are eliminated.
        thirds_ranked = self._rank(thirds)
        best_thirds = thirds_ranked[:8]
        group_exit.extend([t[0] for t in thirds_ranked[8:]])

        # Seed the 32 qualifiers by tier (winners strongest, then runners,
        # then thirds), best-to-worst within each tier. This approximates
        # FIFA's predetermined R32 bracket, which we can't replicate exactly
        # without the official slotting chart.
        ranked = self._rank(winners) + self._rank(runners) + best_thirds
        seed_teams = [r[0] for r in ranked]

        positions = self._standard_seed_positions(32)
        order = [seed_teams[p - 1] for p in positions]

        result = self._play_bracket(order, cache)
        result["p_group_exit"] = group_exit
        return result

    def _rank(self, entries: list[tuple[str, int, float]]) -> list[tuple[str, int, float]]:
        """Rank (team, points, gd) entries best-first, with a random tiebreaker."""
        return sorted(entries, key=lambda e: (e[1], e[2], self.rng.random()), reverse=True)

    @staticmethod
    def _standard_seed_positions(n: int) -> list[int]:
        """
        Standard single-elimination seeding for a bracket of size n.
        Returns the seed numbers (1=strongest) in bracket-position order, so
        consecutive pairs are the first-round matchups (seed 1 vs n, etc.) and
        the top two seeds are kept in opposite halves.
        """
        pos = [1, 2]
        while len(pos) < n:
            m = len(pos) * 2
            new = []
            for s in pos:
                new.append(s)
                new.append(m + 1 - s)
            pos = new
        return pos

    # ------------------------------------------------------------------
    # Generic knockout runner
    # ------------------------------------------------------------------

    def _play_bracket(
        self,
        order: list[str],
        cache: dict[tuple[str, str], float],
    ) -> dict[str, list[str]]:
        """
        Play a single-elimination bracket given teams in bracket-position order
        (consecutive pairs are matchups). Returns a dict mapping each stage
        column to the list of teams that entered that round.
        """
        rounds: dict[str, list[str]] = {}
        current = list(order)
        while True:
            size = len(current)
            if size in _ROUND_NAMES:
                rounds[_ROUND_NAMES[size]] = list(current)
            if size == 1:
                break
            winners = [
                self._knockout_winner(current[i], current[i + 1], cache)
                for i in range(0, size, 2)
            ]
            current = winners
        return rounds

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
