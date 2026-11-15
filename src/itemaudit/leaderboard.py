"""Leaderboard-level inference: which rank differences are supported by the data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .matrix import ResultsMatrix
from .stats import PairedComparison, compare, holm, wilson_interval


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    system: str
    n_tasks: int
    n_solved: int
    accuracy: float
    ci_low: float
    ci_high: float
    # Rank interval implied by Holm-corrected tests against every other system.
    rank_low: int
    rank_high: int
    # Percentile interval of the rank when tasks are resampled.
    boot_rank_low: int
    boot_rank_high: int
    tier: int


@dataclass(frozen=True)
class AdjacentPair:
    """Comparison of the system at ``rank`` with the one directly below it."""

    rank: int
    comparison: PairedComparison
    p_holm: float
    distinguishable: bool


@dataclass(frozen=True)
class Leaderboard:
    rows: tuple[LeaderboardRow, ...]
    adjacent: tuple[AdjacentPair, ...]
    alpha: float
    p_values: np.ndarray
    p_holm: np.ndarray
    discordance: np.ndarray

    @property
    def systems(self) -> tuple[str, ...]:
        return tuple(r.system for r in self.rows)

    @property
    def n_adjacent_indistinguishable(self) -> int:
        return sum(not pair.distinguishable for pair in self.adjacent)

    @property
    def n_pairs(self) -> int:
        k = len(self.rows)
        return k * (k - 1) // 2

    @property
    def n_pairs_indistinguishable(self) -> int:
        upper = np.triu_indices(len(self.rows), k=1)
        return int(np.sum(~(self.p_holm[upper] < self.alpha)))

    @property
    def n_tiers(self) -> int:
        return max((r.tier for r in self.rows), default=0)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.rows])

    def adjacent_frame(self) -> pd.DataFrame:
        records = []
        for pair in self.adjacent:
            c = pair.comparison
            records.append(
                {
                    "rank": pair.rank,
                    "system_upper": c.system_a,
                    "system_lower": c.system_b,
                    "diff": c.diff,
                    "ci_low": c.ci_low,
                    "ci_high": c.ci_high,
                    "boot_ci_low": c.boot_ci_low,
                    "boot_ci_high": c.boot_ci_high,
                    "n_only_upper": c.n_only_a,
                    "n_only_lower": c.n_only_b,
                    "discordance": c.discordance,
                    "p_value": c.p_value,
                    "p_holm": pair.p_holm,
                    "distinguishable": pair.distinguishable,
                }
            )
        return pd.DataFrame(records)

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "n_systems": len(self.rows),
            "n_adjacent_pairs": len(self.adjacent),
            "n_adjacent_indistinguishable": self.n_adjacent_indistinguishable,
            "n_pairs": self.n_pairs,
            "n_pairs_indistinguishable": self.n_pairs_indistinguishable,
            "n_tiers": self.n_tiers,
            "rows": [asdict(r) for r in self.rows],
            "adjacent": self.adjacent_frame().to_dict(orient="records"),
        }


def pairwise_counts(matrix: ResultsMatrix) -> tuple[np.ndarray, np.ndarray]:
    """``only[i, j]`` = tasks system i solved and j failed; ``shared[i, j]`` = tasks both ran."""
    obs = matrix.observed
    solved = np.where(obs, matrix.values, 0.0)
    failed = np.where(obs, 1.0 - matrix.values, 0.0)
    only = np.rint(solved @ failed.T).astype(int)
    shared = np.rint(obs.astype(float) @ obs.astype(float).T).astype(int)
    return only, shared


def pairwise_mcnemar(only: np.ndarray) -> np.ndarray:
    """Matrix of two-sided exact McNemar p-values from the discordant counts."""
    total = only + only.T
    smaller = np.minimum(only, only.T)
    p = np.minimum(1.0, 2.0 * stats.binom.cdf(smaller, total, 0.5))
    p[total == 0] = 1.0
    np.fill_diagonal(p, np.nan)
    return p


def _holm_symmetric(p: np.ndarray) -> np.ndarray:
    k = p.shape[0]
    upper = np.triu_indices(k, k=1)
    adjusted = np.full((k, k), np.nan)
    adjusted[upper] = holm(p[upper])
    adjusted.T[upper] = adjusted[upper]
    return adjusted


def _bootstrap_ranks(
    matrix: ResultsMatrix, confidence: float, n_boot: int, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile rank intervals over task resamples.

    Ties are frequent with pass/fail data, so the lower bound uses the best
    rank a tied system could take and the upper bound the worst.
    """
    k, n = matrix.values.shape
    if n == 0 or n_boot <= 0:
        return np.ones(k, dtype=int), np.full(k, k, dtype=int)
    rng = np.random.default_rng(seed)
    obs = matrix.observed.astype(float)
    solved = np.where(matrix.observed, matrix.values, 0.0)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    seen = weights @ obs.T
    acc = np.divide(weights @ solved.T, seen, out=np.full(seen.shape, -1.0), where=seen > 0)
    best = np.empty((n_boot, k), dtype=int)
    worst = np.empty((n_boot, k), dtype=int)
    chunk = max(1, 4_000_000 // (k * k))  # bounds the (chunk, k, k) comparison array
    for start in range(0, n_boot, chunk):
        part = acc[start : start + chunk]
        best[start : start + chunk] = 1 + (part[:, None, :] > part[:, :, None]).sum(axis=2)
        worst[start : start + chunk] = (part[:, None, :] >= part[:, :, None]).sum(axis=2)
    tail = (1.0 - confidence) / 2.0
    low = np.floor(np.quantile(best, tail, axis=0)).astype(int)
    high = np.ceil(np.quantile(worst, 1.0 - tail, axis=0)).astype(int)
    return low, high


def leaderboard(
    matrix: ResultsMatrix,
    *,
    top: int | None = None,
    alpha: float = 0.05,
    n_boot: int = 2_000,
    seed: int | None = 0,
) -> Leaderboard:
    """Rank systems by accuracy and test which differences are distinguishable.

    Adjacent ranks are compared with exact McNemar tests, Holm-corrected over
    the adjacent pairs. Rank intervals and tiers use Holm correction over all
    pairs: a system's rank interval runs from one plus the number of systems
    significantly better than it to the number of systems not significantly
    worse. A tier is a run of systems that cannot be separated from the best
    system in that run.

    With ``top`` set, only the ``top`` most accurate systems are ranked and
    tested (the multiple-comparison family is restricted to them).

    Each pair is compared on the tasks both systems attempted. If systems
    attempted different task sets, the accuracy ranking itself is not
    like-for-like and should be read with care.
    """
    ranked = matrix.top(matrix.n_systems if top is None else min(top, matrix.n_systems))
    k = ranked.n_systems
    confidence = 1.0 - alpha
    obs = ranked.observed
    n_seen = obs.sum(axis=1)
    n_solved = np.where(obs, ranked.values, 0.0).sum(axis=1).astype(int)
    accuracy = ranked.accuracy()

    only, shared = pairwise_counts(ranked)
    p = pairwise_mcnemar(only)
    p_adj = _holm_symmetric(p) if k > 1 else np.full((k, k), np.nan)
    significant = np.nan_to_num(p_adj, nan=1.0) < alpha
    # better[i, j]: j beats i on discordant tasks and the difference is significant.
    better = significant & (only.T > only)
    worse = significant & (only.T < only)
    rank_low = 1 + better.sum(axis=1)
    rank_high = k - worse.sum(axis=1)
    boot_low, boot_high = _bootstrap_ranks(ranked, confidence, n_boot, seed)

    tiers = np.zeros(k, dtype=int)
    leader, tier = 0, 1
    for i in range(k):
        if i != leader and significant[leader, i]:
            leader, tier = i, tier + 1
        tiers[i] = tier

    rows = []
    for i, system in enumerate(ranked.systems):
        low, high = wilson_interval(int(n_solved[i]), int(n_seen[i]), confidence)
        rows.append(
            LeaderboardRow(
                rank=i + 1,
                system=system,
                n_tasks=int(n_seen[i]),
                n_solved=int(n_solved[i]),
                accuracy=float(accuracy[i]),
                ci_low=low,
                ci_high=high,
                rank_low=int(rank_low[i]),
                rank_high=int(rank_high[i]),
                boot_rank_low=int(boot_low[i]),
                boot_rank_high=int(boot_high[i]),
                tier=int(tiers[i]),
            )
        )

    comparisons = [
        compare(
            ranked.values[i],
            ranked.values[i + 1],
            names=(ranked.systems[i], ranked.systems[i + 1]),
            confidence=confidence,
            n_boot=n_boot,
            seed=seed,
        )
        for i in range(k - 1)
    ]
    adjusted = holm([c.p_value for c in comparisons])
    adjacent = tuple(
        AdjacentPair(rank=i + 1, comparison=c, p_holm=float(q), distinguishable=bool(q < alpha))
        for i, (c, q) in enumerate(zip(comparisons, adjusted))
    )
    return Leaderboard(
        rows=tuple(rows),
        adjacent=adjacent,
        alpha=alpha,
        p_values=p,
        p_holm=p_adj,
        discordance=(only + only.T) / np.maximum(shared, 1),
    )
