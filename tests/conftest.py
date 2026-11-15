from __future__ import annotations

import numpy as np
import pytest

from itemaudit import ResultsMatrix


def simulate_2pl(
    n_systems: int,
    n_tasks: int,
    seed: int,
    *,
    slope_range: tuple[float, float] = (0.6, 2.0),
    difficulty_range: tuple[float, float] = (-2.0, 2.0),
) -> tuple[ResultsMatrix, np.ndarray, np.ndarray, np.ndarray]:
    """Draw a results matrix from a known 2PL model. Returns (matrix, a, b, theta)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n_systems)
    a = rng.uniform(*slope_range, size=n_tasks)
    b = rng.uniform(*difficulty_range, size=n_tasks)
    p = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    values = (rng.uniform(size=p.shape) < p).astype(float)
    matrix = ResultsMatrix(
        tuple(f"s{i}" for i in range(n_systems)), tuple(f"t{j}" for j in range(n_tasks)), values
    )
    return matrix, a, b, theta


@pytest.fixture
def small_matrix() -> ResultsMatrix:
    """Five systems, six tasks, one missing cell; small enough to check by hand."""
    nan = np.nan
    values = np.array(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, nan, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    return ResultsMatrix(("a", "b", "c", "d", "e"), ("t1", "t2", "t3", "t4", "t5", "t6"), values)
