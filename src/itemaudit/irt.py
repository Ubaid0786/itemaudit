"""Two-parameter logistic item response model, fitted by marginal maximum likelihood.

The model is ``P(system i solves task j) = sigmoid(a_j * (theta_i - b_j))``.
``theta`` is the system's ability, ``b`` the task's difficulty on the same
scale and ``a`` its discrimination: how sharply the solve probability rises
with ability. A negative ``a`` means stronger systems solve the task less often.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .matrix import ResultsMatrix


@dataclass(frozen=True)
class IRTFit:
    """Fitted 2PL parameters. Arrays follow the order of ``systems`` and ``tasks``."""

    systems: tuple[str, ...]
    tasks: tuple[str, ...]
    discrimination: np.ndarray  # a_j
    intercept: np.ndarray  # d_j, with logit = a_j * theta + d_j
    ability: np.ndarray  # posterior mean of theta_i
    ability_sd: np.ndarray  # posterior standard deviation of theta_i
    log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def difficulty(self) -> np.ndarray:
        """``b_j = -d_j / a_j``; nan where the slope is too close to zero to define it."""
        a = self.discrimination
        return np.divide(-self.intercept, a, out=np.full(a.shape, np.nan), where=np.abs(a) > 1e-6)

    def prob(self, theta: np.ndarray | float) -> np.ndarray:
        """Solve probability, shape (n_tasks, len(theta))."""
        t = np.atleast_1d(np.asarray(theta, dtype=float))
        return expit(self.discrimination[:, None] * t[None, :] + self.intercept[:, None])

    def item_information(self, theta: np.ndarray | float) -> np.ndarray:
        """Fisher information each task carries about ability, (n_tasks, len(theta))."""
        p = self.prob(theta)
        return self.discrimination[:, None] ** 2 * p * (1.0 - p)

    def test_information(self, theta: np.ndarray | float, *, positive_only: bool = True) -> np.ndarray:
        """Sum of item information at each ``theta``.

        Tasks with a negative slope formally carry information too, but they
        reward lower ability, so by default they are left out.
        """
        info = self.item_information(theta)
        if positive_only:
            info = info[self.discrimination > 0]
        return info.sum(axis=0)

    def mean_item_information(self, theta: np.ndarray | None = None) -> np.ndarray:
        """Each task's information averaged over ``theta`` (default: the fitted abilities).

        Tasks with a non-positive slope get zero.
        """
        t = self.ability if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
        info = self.item_information(t).mean(axis=1)
        return np.where(self.discrimination > 0, info, 0.0)

    def effective_n_items(self, theta: np.ndarray | None = None) -> float:
        """Effective number of informative tasks over the given abilities.

        ``(sum w)^2 / sum(w^2)`` with ``w`` the mean item information: the
        number of equally informative tasks that would spread information the
        same way. It equals the task count when all tasks are equally
        informative and approaches 1 when one task carries everything.
        """
        w = self.mean_item_information(theta)
        total_sq = float((w**2).sum())
        return float(w.sum() ** 2 / total_sq) if total_sq > 0 else 0.0

    def items_for_information_share(self, share: float = 0.9, theta: np.ndarray | None = None) -> int:
        """Fewest tasks that together hold ``share`` of the total information."""
        w = np.sort(self.mean_item_information(theta))[::-1]
        total = w.sum()
        if total <= 0:
            return 0
        return int(np.searchsorted(np.cumsum(w), share * total - 1e-12) + 1)

    def items_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "irt_discrimination": self.discrimination,
                "irt_difficulty": self.difficulty,
                "irt_mean_information": self.mean_item_information(),
            },
            index=pd.Index(self.tasks, name="task"),
        )

    def systems_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"ability": self.ability, "ability_sd": self.ability_sd},
            index=pd.Index(self.systems, name="system"),
        )


def _quadrature(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.linspace(-6.0, 6.0, n_quad)
    log_w = -0.5 * nodes**2
    return nodes, log_w - logsumexp(log_w)


def fit_2pl(
    matrix: ResultsMatrix,
    *,
    slope_prior: tuple[float, float] = (1.0, 1.0),
    intercept_prior: tuple[float, float] = (0.0, 3.0),
    n_quad: int = 61,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> IRTFit:
    """Fit a 2PL model with the EM algorithm (Bock and Aitkin's marginal likelihood).

    Abilities are integrated out against a standard normal, which fixes the
    scale. Missing cells are ignored. Each task's slope and intercept have a
    normal prior, given as ``(mean, standard deviation)``: this keeps estimates
    finite for tasks that every system or no system solved, and it lets a slope
    go negative when the data ask for it, which is what marks a suspect task.
    A lognormal slope prior, the usual choice, would hide exactly those tasks.

    The slope prior centred on 1 also fixes the sign of the ability scale.
    Iteration stops when no parameter moves by more than ``tol``.
    """
    obs = matrix.observed
    solved = np.where(obs, matrix.values, 0.0)
    failed = np.where(obs, 1.0 - matrix.values, 0.0)
    seen = obs.astype(float)
    n_tasks = matrix.n_tasks
    nodes, log_w = _quadrature(n_quad)
    mu_a, sd_a = slope_prior
    mu_d, sd_d = intercept_prior
    prec_a, prec_d = 1.0 / sd_a**2, 1.0 / sd_d**2

    rate = np.nan_to_num(matrix.solve_rate(), nan=0.5)
    a = np.full(n_tasks, float(mu_a))
    d = np.log(np.clip(rate, 0.02, 0.98) / (1.0 - np.clip(rate, 0.02, 0.98)))

    def posterior(a: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, float]:
        logits = a[:, None] * nodes[None, :] + d[:, None]
        log_p = -np.logaddexp(0.0, -logits)
        log_q = -np.logaddexp(0.0, logits)
        joint = solved @ log_p + failed @ log_q + log_w[None, :]
        norm = logsumexp(joint, axis=1, keepdims=True)
        return np.exp(joint - norm), float(norm.sum())

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        post, _ = posterior(a, d)
        expected_n = seen.T @ post
        expected_r = solved.T @ post

        new_a, new_d = a.copy(), d.copy()
        # The M-step objective is a ridge-penalised logistic regression per
        # task, which is concave, so a few damped Newton steps are enough.
        for _ in range(25):
            p = expit(new_a[:, None] * nodes[None, :] + new_d[:, None])
            resid = expected_r - expected_n * p
            w = expected_n * p * (1.0 - p)
            g_a = resid @ nodes - prec_a * (new_a - mu_a)
            g_d = resid.sum(axis=1) - prec_d * (new_d - mu_d)
            h_aa = w @ nodes**2 + prec_a
            h_ad = w @ nodes
            h_dd = w.sum(axis=1) + prec_d
            det = h_aa * h_dd - h_ad**2
            step_a = np.clip((h_dd * g_a - h_ad * g_d) / det, -1.0, 1.0)
            step_d = np.clip((h_aa * g_d - h_ad * g_a) / det, -1.0, 1.0)
            new_a += step_a
            new_d += step_d
            if max(np.abs(step_a).max(initial=0.0), np.abs(step_d).max(initial=0.0)) < 1e-8:
                break

        change = max(np.abs(new_a - a).max(initial=0.0), np.abs(new_d - d).max(initial=0.0))
        a, d = new_a, new_d
        if change < tol:
            converged = True
            break

    post, log_likelihood = posterior(a, d)
    ability = post @ nodes
    ability_sd = np.sqrt(np.maximum(post @ nodes**2 - ability**2, 0.0))
    return IRTFit(
        systems=matrix.systems,
        tasks=matrix.tasks,
        discrimination=a,
        intercept=d,
        ability=ability,
        ability_sd=ability_sd,
        log_likelihood=log_likelihood,
        n_iter=n_iter,
        converged=converged,
    )
