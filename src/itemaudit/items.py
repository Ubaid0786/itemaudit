"""Classical test theory: per-task difficulty and discrimination, per-system fit."""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .irt import IRTFit, fit_2pl
from .matrix import ResultsMatrix
from .stats import holm

FLAG_ALL = "solved_by_all"
FLAG_NONE = "solved_by_none"
FLAG_NEGATIVE = "negative_discrimination"
FLAG_NEAR_ZERO = "near_zero_discrimination"


def rest_scores(matrix: ResultsMatrix) -> np.ndarray:
    """For each cell, the system's solve rate on its *other* observed tasks.

    Leaving the task itself out removes the spurious positive correlation a
    task has with a total that includes it. nan where the cell is missing or
    the system has no other observed task.
    """
    obs = matrix.observed
    values = np.where(obs, matrix.values, 0.0)
    total = values.sum(axis=1, keepdims=True)
    count = obs.sum(axis=1, keepdims=True)
    others = count - obs
    rest = np.divide(total - values, others, out=np.full(values.shape, np.nan), where=others > 0)
    rest[~obs] = np.nan
    return rest


def _column_correlation(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of x and y down each column over ``valid`` cells."""
    n = valid.sum(axis=0)
    safe_n = np.maximum(n, 1)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    mean_x = xv.sum(axis=0) / safe_n
    mean_y = yv.sum(axis=0) / safe_n
    dx = np.where(valid, x - mean_x, 0.0)
    dy = np.where(valid, y - mean_y, 0.0)
    var_x = (dx**2).sum(axis=0)
    var_y = (dy**2).sum(axis=0)
    denom = np.sqrt(var_x * var_y)
    # A constant column has a variance of exactly zero up to rounding.
    defined = (n >= 3) & (var_x > 1e-12) & (var_y > 1e-12)
    r = np.divide((dx * dy).sum(axis=0), denom, out=np.full(n.shape, np.nan), where=defined)
    return np.clip(r, -1.0, 1.0), n


def item_stats(
    matrix: ResultsMatrix,
    *,
    near_zero: float = 0.1,
    confidence: float = 0.95,
    tail_fraction: float = 0.27,
) -> pd.DataFrame:
    """Difficulty and discrimination for every task.

    Columns
    -------
    n, n_solved, solve_rate
        How many systems ran the task, how many solved it, and the ratio.
    n_minority
        ``min(n_solved, n - n_solved)``. Discrimination estimated from a
        handful of systems is unstable; filter on this before acting on a flag.
    discrimination
        Point-biserial correlation between solving the task and the system's
        solve rate on its other tasks (corrected item-rest correlation). nan
        for tasks every system or no system solved.
    disc_ci_low, disc_ci_high
        Fisher-z interval for ``discrimination``. It treats systems as
        independent draws, which leaderboard submissions usually are not, so
        it is too narrow when many systems share a base model.
    upper_lower
        Solve rate in the top ``tail_fraction`` of systems minus the solve
        rate in the bottom ``tail_fraction`` (the classical D index).
    flag
        ``solved_by_all`` / ``solved_by_none`` (no information),
        ``negative_discrimination`` (weaker systems solve it more often),
        ``near_zero_discrimination`` (0 <= discrimination < ``near_zero``),
        or empty.
    """
    obs = matrix.observed
    values = np.where(obs, matrix.values, 0.0)
    n = obs.sum(axis=0)
    n_solved = values.sum(axis=0).astype(int)
    solve_rate = matrix.solve_rate()

    rest = rest_scores(matrix)
    valid = ~np.isnan(rest)
    disc, n_used = _column_correlation(matrix.values, rest, valid)

    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fisher = np.arctanh(np.clip(disc, -0.999999, 0.999999))
        half = np.where(n_used > 3, z / np.sqrt(np.maximum(n_used - 3, 1)), np.nan)
    ci_low = np.tanh(fisher - half)
    ci_high = np.tanh(fisher + half)

    accuracy = matrix.accuracy()
    order = np.argsort(-np.nan_to_num(accuracy, nan=-1.0), kind="stable")
    tail = max(1, int(np.ceil(tail_fraction * matrix.n_systems)))
    upper, lower = order[:tail], order[-tail:]

    def group_rate(rows: np.ndarray) -> np.ndarray:
        seen = obs[rows].sum(axis=0)
        return np.divide(values[rows].sum(axis=0), seen, out=np.full(seen.shape, np.nan), where=seen > 0)

    upper_lower = group_rate(upper) - group_rate(lower)

    flags = np.full(matrix.n_tasks, "", dtype=object)
    flags[(n > 0) & (n_solved == n)] = FLAG_ALL
    flags[(n > 0) & (n_solved == 0)] = FLAG_NONE
    flags[disc < 0] = FLAG_NEGATIVE
    flags[(disc >= 0) & (disc < near_zero)] = FLAG_NEAR_ZERO

    frame = pd.DataFrame(
        {
            "n": n,
            "n_solved": n_solved,
            "solve_rate": solve_rate,
            "n_minority": np.minimum(n_solved, n - n_solved),
            "discrimination": disc,
            "disc_ci_low": ci_low,
            "disc_ci_high": ci_high,
            "upper_lower": upper_lower,
            "flag": flags,
        },
        index=pd.Index(matrix.tasks, name="task"),
    )
    return frame


def kr20(matrix: ResultsMatrix) -> float:
    """Kuder-Richardson 20 reliability of the total score.

    Defined for complete data only; returns nan if any cell is missing, if
    there are fewer than two tasks, or if all systems have the same total.
    """
    if matrix.n_tasks < 2 or matrix.n_systems < 2 or not matrix.observed.all():
        return float("nan")
    k = matrix.n_tasks
    p = matrix.values.mean(axis=0)
    total_var = matrix.values.sum(axis=1).var(ddof=0)
    if total_var <= 0:
        return float("nan")
    return float(k / (k - 1) * (1.0 - (p * (1 - p)).sum() / total_var))


def _guttman_counts(values: np.ndarray, obs: np.ndarray, rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Guttman errors and their maximum per row, with task difficulty given by ``rate``."""
    usable = ~np.isnan(rate)
    order = np.argsort(-rate[usable], kind="stable")
    obs_sorted = obs[:, usable][:, order]
    vals = values[:, usable][:, order]
    ones = np.where(obs_sorted, vals, 0.0)
    zeros = np.where(obs_sorted, 1.0 - vals, 0.0)
    if ones.shape[1] == 0:
        return np.zeros(values.shape[0]), np.zeros(values.shape[0])
    sorted_rate = rate[usable][order]
    starts = np.flatnonzero(np.r_[True, sorted_rate[1:] != sorted_rate[:-1]])
    ones_g = np.add.reduceat(ones, starts, axis=1)
    zeros_g = np.add.reduceat(zeros, starts, axis=1)
    easier_failed = np.cumsum(zeros_g, axis=1) - zeros_g
    return (ones_g * easier_failed).sum(axis=1), ones.sum(axis=1) * zeros.sum(axis=1)


def guttman_errors(matrix: ResultsMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Count Guttman errors per system.

    A Guttman error is a pair of tasks where the system failed the easier one
    (higher solve rate across all systems) and solved the harder one. Pairs of
    equally difficult tasks are never errors. Returns ``(errors, maximum)``
    where ``maximum`` is solved x failed, the count if every pair were an error.
    """
    errors, maximum = _guttman_counts(matrix.values, matrix.observed, matrix.solve_rate())
    return np.rint(errors).astype(int), np.rint(maximum).astype(int)


def system_fit(
    matrix: ResultsMatrix,
    *,
    fit: IRTFit | None = None,
    model_reference: bool = True,
    n_sim: int = 200,
    rare: float = 0.1,
    z_threshold: float = 3.0,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Guttman-style response-pattern statistics for every system.

    Columns
    -------
    n, n_solved, accuracy
        Tasks attempted, tasks solved and the ratio.
    guttman_errors, guttman_norm
        Guttman error count and the count divided by its maximum
        (solved x failed). 0 means the solved tasks are exactly the easiest
        ones; larger means hard tasks solved while easy ones were failed.
    guttman_expected, guttman_z
        What ``guttman_norm`` would be for a system of the same ability that
        behaves as the 2PL model says, and how many standard deviations the
        observed value lies above that. Both come from ``n_sim`` response
        patterns drawn from the fitted model for each system. Some errors are
        always expected, more so at very low or very high scores, which is why
        the raw count is not compared across systems directly.
    n_hard_solved, n_easy_failed
        Tasks solved that at most ``rare`` of systems solve, and tasks failed
        that at least ``1 - rare`` of systems solve.
    flag
        ``unusual_pattern`` when ``guttman_z`` exceeds ``z_threshold``. This is
        a prompt to look, not a verdict: an unusual pattern can come from a
        different scaffold, a different task subset, tasks that were never
        run, a newer model, or a model that does not fit the data well.

    ``fit`` is a 2PL fit of this same matrix; it is computed if not given.
    With ``model_reference=False`` no model is fitted and the two reference
    columns and the flag are left empty.
    """
    obs = matrix.observed
    values = np.where(obs, matrix.values, 0.0)
    n = obs.sum(axis=1)
    n_solved = values.sum(axis=1).astype(int)
    rate = matrix.solve_rate()
    errors, maximum = guttman_errors(matrix)
    norm = np.divide(errors, maximum, out=np.full(len(errors), np.nan), where=maximum > 0)

    expected = np.full(len(norm), np.nan)
    z = np.full(len(norm), np.nan)
    if model_reference and matrix.n_systems >= 2 and matrix.n_tasks >= 2 and n_sim >= 2:
        if fit is None:
            fit = fit_2pl(matrix)
        if fit.systems != matrix.systems or fit.tasks != matrix.tasks:
            raise ValueError("fit was not computed from this matrix")
        prob = fit.prob(fit.ability).T
        rng = np.random.default_rng(seed)
        draws = np.full((n_sim, matrix.n_systems), np.nan)
        for b in range(n_sim):
            simulated = (rng.uniform(size=prob.shape) < prob).astype(float)
            sim_errors, sim_max = _guttman_counts(simulated, obs, rate)
            np.divide(sim_errors, sim_max, out=draws[b], where=sim_max > 0)
        enough = (~np.isnan(draws)).sum(axis=0) >= 2
        with warnings.catch_warnings():
            # Rows with no usable draw (all tasks solved or none) are handled below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(draws, axis=0)
            sd = np.nanstd(draws, axis=0, ddof=1)
        expected = np.where(enough, mean, np.nan)
        usable = enough & (np.nan_to_num(sd, nan=0.0) > 0) & ~np.isnan(norm)
        z = np.divide(norm - mean, sd, out=np.full(len(norm), np.nan), where=usable)

    hard = np.nan_to_num(rate, nan=1.0) <= rare
    easy = np.nan_to_num(rate, nan=0.0) >= 1.0 - rare
    n_hard_solved = values[:, hard].sum(axis=1).astype(int)
    n_easy_failed = np.where(obs, 1.0 - matrix.values, 0.0)[:, easy].sum(axis=1).astype(int)

    flags = np.where(np.nan_to_num(z, nan=0.0) > z_threshold, "unusual_pattern", "")
    return pd.DataFrame(
        {
            "n": n,
            "n_solved": n_solved,
            "accuracy": matrix.accuracy(),
            "guttman_errors": errors,
            "guttman_norm": norm,
            "guttman_expected": expected,
            "guttman_z": z,
            "n_hard_solved": n_hard_solved,
            "n_easy_failed": n_easy_failed,
            "flag": flags,
        },
        index=pd.Index(matrix.systems, name="system"),
    )


def dif(
    matrix: ResultsMatrix,
    groups: Sequence[str] | Mapping[str, str],
    *,
    focal: str | None = None,
    n_strata: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mantel-Haenszel test for tasks that behave differently in two groups of systems.

    ``groups`` labels every system with one of exactly two group names (a
    sequence in system order, or a mapping from system name). Systems are
    stratified by overall accuracy, and within each stratum the odds of
    solving a task are compared between the ``focal`` group and the other
    (reference) group. A task that one group solves far more often than
    equally able systems in the other group is said to show differential item
    functioning: its result depends on something other than ability, such as
    the scaffold, the evaluation environment or the submission period.

    Columns
    -------
    n_focal, n_reference, rate_focal, rate_reference
        Systems that ran the task in each group and the share that solved it.
    odds_ratio
        Mantel-Haenszel common odds ratio (focal over reference); below 1
        means the focal group solves the task less often at equal ability.
    chi2, p_value, p_holm
        Continuity-corrected Mantel-Haenszel statistic, its p-value on one
        degree of freedom, and the Holm-adjusted p-value across tasks.
    flag
        ``dif`` when ``p_holm < alpha``.

    The test treats systems as independent. Submissions sharing a base model
    are not, so p-values are optimistic; treat a flag as a prompt to look.
    """
    if isinstance(groups, Mapping):
        labels = np.array([str(groups[s]) for s in matrix.systems], dtype=object)
    else:
        labels = np.array([str(g) for g in groups], dtype=object)
    if labels.shape != (matrix.n_systems,):
        raise ValueError("groups must give one label per system")
    names = sorted(set(labels))
    if len(names) != 2:
        raise ValueError(f"dif needs exactly two groups, got {len(names)}")
    if focal is None:
        focal = min(names, key=lambda g: (int(np.sum(labels == g)), g))
    if focal not in names:
        raise ValueError(f"focal group {focal!r} is not one of {names}")
    is_focal = labels == focal

    accuracy = np.nan_to_num(matrix.accuracy(), nan=0.0)
    edges = np.quantile(accuracy, np.linspace(0.0, 1.0, n_strata + 1)[1:-1]) if n_strata > 1 else []
    stratum = np.searchsorted(edges, accuracy, side="right")

    obs = matrix.observed
    solved = np.where(obs, matrix.values, 0.0)
    seen = obs.astype(float)
    n_tasks = matrix.n_tasks
    sum_a = np.zeros(n_tasks)
    sum_expected = np.zeros(n_tasks)
    sum_var = np.zeros(n_tasks)
    or_num = np.zeros(n_tasks)
    or_den = np.zeros(n_tasks)
    for k in np.unique(stratum):
        f_rows = (stratum == k) & is_focal
        r_rows = (stratum == k) & ~is_focal
        a = solved[f_rows].sum(axis=0)
        n_f = seen[f_rows].sum(axis=0)
        c = solved[r_rows].sum(axis=0)
        n_r = seen[r_rows].sum(axis=0)
        total = n_f + n_r
        m1 = a + c
        usable = (n_f > 0) & (n_r > 0)
        safe = np.where(total > 0, total, 1.0)
        sum_a += np.where(usable, a, 0.0)
        sum_expected += np.where(usable, n_f * m1 / safe, 0.0)
        var = n_f * n_r * m1 * (total - m1) / (safe**2 * np.maximum(total - 1.0, 1.0))
        sum_var += np.where(usable & (total > 1), var, 0.0)
        or_num += np.where(usable, a * (n_r - c) / safe, 0.0)
        or_den += np.where(usable, (n_f - a) * c / safe, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        odds_ratio = np.where(
            or_den > 0, or_num / np.where(or_den > 0, or_den, 1.0), np.where(or_num > 0, np.inf, np.nan)
        )
        corrected = np.maximum(np.abs(sum_a - sum_expected) - 0.5, 0.0)
        chi2 = np.where(sum_var > 0, corrected**2 / np.where(sum_var > 0, sum_var, 1.0), np.nan)
    p_value = stats.chi2.sf(chi2, 1)
    p_holm = holm(p_value)

    def rate(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = seen[rows].sum(axis=0)
        return n.astype(int), np.divide(solved[rows].sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)

    n_focal, rate_focal = rate(is_focal)
    n_reference, rate_reference = rate(~is_focal)
    return pd.DataFrame(
        {
            "n_focal": n_focal,
            "n_reference": n_reference,
            "rate_focal": rate_focal,
            "rate_reference": rate_reference,
            "odds_ratio": odds_ratio,
            "chi2": chi2,
            "p_value": p_value,
            "p_holm": p_holm,
            "flag": np.where(np.nan_to_num(p_holm, nan=1.0) < alpha, "dif", ""),
        },
        index=pd.Index(matrix.tasks, name="task"),
    )
