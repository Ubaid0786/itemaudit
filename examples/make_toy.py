"""Generate examples/toy_results.csv: simulated results, not a real benchmark.

Twelve systems on forty tasks drawn from a 2PL model with a fixed seed. Three
defects are planted so the report has something to find:

    task_07   reverse keyed (stronger systems fail it)
    task_19   solved by no system
    task_31   solved by every system

    python examples/make_toy.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from itemaudit import ResultsMatrix

HERE = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(2026)
    n_systems, n_tasks = 12, 40
    ability = np.linspace(-1.5, 2.0, n_systems)
    slope = rng.uniform(1.0, 2.5, n_tasks)
    difficulty = rng.uniform(-1.5, 2.0, n_tasks)
    prob = 1.0 / (1.0 + np.exp(-slope * (ability[:, None] - difficulty)))
    values = (rng.uniform(size=prob.shape) < prob).astype(float)
    values[:, 7] = 1.0 - (ability > 0.2)
    values[:, 19] = 0.0
    values[:, 31] = 1.0
    values[3, 12] = np.nan  # one task that one system was not run on
    matrix = ResultsMatrix(
        tuple(f"system_{chr(ord('a') + i)}" for i in range(n_systems)),
        tuple(f"task_{j:02d}" for j in range(n_tasks)),
        values,
    )
    matrix.to_csv(HERE / "toy_results.csv")
    print(f"wrote {HERE / 'toy_results.csv'}")


if __name__ == "__main__":
    main()
