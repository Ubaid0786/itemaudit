"""Paired statistics for two systems evaluated on the same tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import optimize, stats


def mcnemar_exact(n_only_a: int, n_only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Under the null the ``n_only_a + n_only_b`` discordant tasks split 50/50,
    so this is an exact binomial test. With no discordant tasks the p-value is 1.
    """
    if n_only_a < 0 or n_only_b < 0:
        raise ValueError("discordant counts must be non-negative")
    n = n_only_a + n_only_b
    if n == 0:
        return 1.0
    tail = stats.binom.cdf(min(n_only_a, n_only_b), n, 0.5)
    return float(min(1.0, 2.0 * tail))


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order.

    Controls the family-wise error rate under any dependence between tests.
    nan inputs stay nan and do not count towards the family size.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = valid.size
    if m == 0:
        return adjusted
    order = valid[np.argsort(p[valid], kind="stable")]
    scaled = (m - np.arange(m)) * p[order]
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted


def paired_se(n: int, n_only_a: int, n_only_b: int) -> float:
    """Standard error of the paired difference in accuracy.

    The per-task difference is +1 on ``n_only_a`` tasks, -1 on ``n_only_b``
    tasks and 0 elsewhere; this is its sample standard deviation (n - 1
    denominator) divided by sqrt(n).
    """
    if n < 2:
        return float("nan")
    mean = (n_only_a - n_only_b) / n
    variance = ((n_only_a + n_only_b) - n * mean**2) / (n - 1)
    return float(np.sqrt(max(variance, 0.0) / n))


def paired_bootstrap_ci(
    n: int,
    n_only_a: int,
    n_only_b: int,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the paired difference in accuracy.

    Resampling tasks with replacement only changes how many of each outcome
    type are drawn, so the resample counts are drawn directly from a
    multinomial. This is the same distribution as resampling task indices.
    """
    if n < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    probs = np.array([n_only_a, n_only_b, n - n_only_a - n_only_b], dtype=float) / n
    draws = rng.multinomial(n, probs, size=n_boot)
    diffs = (draws[:, 0] - draws[:, 1]) / n
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(diffs, [tail, 1.0 - tail])
    return float(low), float(high)


def min_detectable_difference(
    n_tasks: int,
    discordance: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest true accuracy difference a paired test on ``n_tasks`` detects.

    ``discordance`` is the fraction of tasks on which the two systems disagree
    (one passes, the other fails). Uses the normal approximation to McNemar's
    test: variance ``discordance / n`` under the null and
    ``(discordance - delta**2) / n`` under the alternative. Returns nan when
    even the largest difference compatible with ``discordance`` is undetectable.
    """
    if not 0.0 <= discordance <= 1.0:
        raise ValueError("discordance must be between 0 and 1")
    if n_tasks < 1:
        raise ValueError("n_tasks must be positive")
    if discordance == 0.0:
        return 0.0
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_power = stats.norm.ppf(power)

    def gap(delta: float) -> float:
        need = z_alpha * np.sqrt(discordance) + z_power * np.sqrt(
            max(discordance - delta**2, 0.0)
        )
        return delta * np.sqrt(n_tasks) - need

    # A difference cannot exceed the discordance rate (all disagreements one way).
    if gap(discordance) < 0:
        return float("nan")
    return float(optimize.brentq(gap, 0.0, discordance))


def min_significant_difference(n_tasks: int, discordance: float, alpha: float = 0.05) -> float:
    """Smallest observed difference that reaches two-sided significance.

    This is the power = 0.5 case of :func:`min_detectable_difference`: a true
    difference of this size is detected only half the time.
    """
    return min_detectable_difference(n_tasks, discordance, alpha=alpha, power=0.5)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass(frozen=True)
class PairedComparison:
    """Result of comparing two systems on the tasks both attempted."""

    system_a: str
    system_b: str
    n_tasks: int
    accuracy_a: float
    accuracy_b: float
    diff: float
    n_both: int
    n_only_a: int
    n_only_b: int
    n_neither: int
    discordance: float
    se_paired: float
    se_unpaired: float
    confidence: float
    ci_low: float
    ci_high: float
    boot_ci_low: float
    boot_ci_high: float
    p_value: float

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        pct = 100.0
        return (
            f"{self.system_a}: {self.accuracy_a * pct:.2f}%  "
            f"{self.system_b}: {self.accuracy_b * pct:.2f}%  on {self.n_tasks} shared tasks\n"
            f"difference (a - b): {self.diff * pct:+.2f} points, "
            f"paired SE {self.se_paired * pct:.2f} "
            f"(unpaired SE would be {self.se_unpaired * pct:.2f})\n"
            f"{self.confidence:.0%} CI analytic [{self.ci_low * pct:+.2f}, {self.ci_high * pct:+.2f}], "
            f"bootstrap [{self.boot_ci_low * pct:+.2f}, {self.boot_ci_high * pct:+.2f}]\n"
            f"solved by a only: {self.n_only_a}, by b only: {self.n_only_b}, "
            f"both: {self.n_both}, neither: {self.n_neither}\n"
            f"McNemar exact p = {self.p_value:.4g}"
        )


def compare(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    names: tuple[str, str] = ("a", "b"),
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int | None = 0,
) -> PairedComparison:
    """Compare two systems' per-task outcomes on the tasks both attempted.

    ``a`` and ``b`` are equal-length sequences of 1/0 (or booleans) with nan
    where a system was not run. Tasks missing for either system are dropped.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("a and b must be one-dimensional and the same length")
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if not (np.isin(x, (0.0, 1.0)).all() and np.isin(y, (0.0, 1.0)).all()):
        raise ValueError("outcomes must be 0, 1 or nan")
    n = int(x.size)
    n_both = int(np.sum((x == 1) & (y == 1)))
    n_only_a = int(np.sum((x == 1) & (y == 0)))
    n_only_b = int(np.sum((x == 0) & (y == 1)))
    n_neither = n - n_both - n_only_a - n_only_b
    if n == 0:
        nan = float("nan")
        return PairedComparison(
            names[0], names[1], 0, nan, nan, nan, 0, 0, 0, 0, nan, nan, nan,
            confidence, nan, nan, nan, nan, 1.0,
        )
    acc_a = (n_both + n_only_a) / n
    acc_b = (n_both + n_only_b) / n
    diff = (n_only_a - n_only_b) / n
    se = paired_se(n, n_only_a, n_only_b)
    se_unpaired = float(np.sqrt(acc_a * (1 - acc_a) / n + acc_b * (1 - acc_b) / n))
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    half = z * se if not np.isnan(se) else float("nan")
    boot_low, boot_high = paired_bootstrap_ci(n, n_only_a, n_only_b, confidence, n_boot, seed)
    return PairedComparison(
        system_a=names[0],
        system_b=names[1],
        n_tasks=n,
        accuracy_a=acc_a,
        accuracy_b=acc_b,
        diff=diff,
        n_both=n_both,
        n_only_a=n_only_a,
        n_only_b=n_only_b,
        n_neither=n_neither,
        discordance=(n_only_a + n_only_b) / n,
        se_paired=se,
        se_unpaired=se_unpaired,
        confidence=confidence,
        ci_low=diff - half,
        ci_high=diff + half,
        boot_ci_low=boot_low,
        boot_ci_high=boot_high,
        p_value=mcnemar_exact(n_only_a, n_only_b),
    )
