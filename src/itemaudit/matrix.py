"""The results matrix: systems in rows, tasks in columns, pass/fail in cells."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_TRUE = {"1", "1.0", "true", "t", "yes", "y", "pass", "passed", "resolved", "solved"}
_FALSE = {"0", "0.0", "false", "f", "no", "n", "fail", "failed", "unresolved", "unsolved"}
_MISSING = {"", "na", "nan", "n/a", "none", "null", "?"}


def _parse_cell(value: object) -> float:
    """Map one CSV cell to 1.0, 0.0 or nan. Anything unrecognised is an error."""
    if value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return np.nan
        if value in (0, 1):
            return float(value)
        raise ValueError(f"result cells must be 0, 1 or missing, got {value!r}")
    text = str(value).strip().lower()
    if text in _TRUE:
        return 1.0
    if text in _FALSE:
        return 0.0
    if text in _MISSING:
        return np.nan
    raise ValueError(f"cannot read {value!r} as pass, fail or missing")


@dataclass(frozen=True)
class ResultsMatrix:
    """Pass/fail outcomes for ``systems`` (rows) on ``tasks`` (columns).

    ``values`` is float64 with 1.0 for a pass, 0.0 for a fail and nan where the
    system was not run on the task.
    """

    systems: tuple[str, ...]
    tasks: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("values must be two-dimensional (systems x tasks)")
        if values.shape != (len(self.systems), len(self.tasks)):
            raise ValueError(
                f"values has shape {values.shape}, expected "
                f"({len(self.systems)}, {len(self.tasks)})"
            )
        finite = values[~np.isnan(values)]
        if finite.size and not np.isin(finite, (0.0, 1.0)).all():
            raise ValueError("values must contain only 0, 1 and nan")
        if len(set(self.systems)) != len(self.systems):
            raise ValueError("system names must be unique")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("task names must be unique")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "systems", tuple(str(s) for s in self.systems))
        object.__setattr__(self, "tasks", tuple(str(t) for t in self.tasks))

    # -- construction -----------------------------------------------------

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ResultsMatrix":
        """Build from a wide DataFrame: index = systems, columns = tasks."""
        parsed = np.array(
            [[_parse_cell(v) for v in row] for row in frame.to_numpy(dtype=object)],
            dtype=float,
        ).reshape(frame.shape)
        return cls(
            tuple(str(s) for s in frame.index),
            tuple(str(t) for t in frame.columns),
            parsed,
        )

    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        system: str = "system",
        task: str = "task",
        result: str = "result",
    ) -> "ResultsMatrix":
        """Build from one row per (system, task) attempt. Absent pairs are missing."""
        if frame.duplicated([system, task]).any():
            raise ValueError("duplicate (system, task) rows; aggregate repeats first")
        parsed = frame.assign(**{result: frame[result].map(_parse_cell)})
        wide = parsed.pivot(index=system, columns=task, values=result)
        return cls(
            tuple(str(s) for s in wide.index),
            tuple(str(t) for t in wide.columns),
            wide.to_numpy(dtype=float),
        )

    @classmethod
    def from_resolved(
        cls,
        resolved: Mapping[str, Iterable[str]],
        tasks: Sequence[str],
    ) -> "ResultsMatrix":
        """Build from ``{system: ids it solved}``; every other task is a fail."""
        index = {t: j for j, t in enumerate(tasks)}
        values = np.zeros((len(resolved), len(tasks)), dtype=float)
        for i, solved in enumerate(resolved.values()):
            for task in solved:
                if task in index:
                    values[i, index[task]] = 1.0
        return cls(tuple(resolved.keys()), tuple(tasks), values)

    # -- views ------------------------------------------------------------

    @property
    def n_systems(self) -> int:
        return len(self.systems)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def observed(self) -> np.ndarray:
        """Boolean mask of cells that are not missing."""
        return ~np.isnan(self.values)

    def row(self, system: str) -> np.ndarray:
        try:
            return self.values[self.systems.index(system)]
        except ValueError:
            raise KeyError(f"unknown system {system!r}") from None

    def accuracy(self) -> np.ndarray:
        """Fraction of observed tasks each system solved (nan if none observed)."""
        obs = self.observed
        n = obs.sum(axis=1)
        solved = np.where(obs, self.values, 0.0).sum(axis=1)
        return np.divide(solved, n, out=np.full(len(n), np.nan), where=n > 0)

    def solve_rate(self) -> np.ndarray:
        """Fraction of observing systems that solved each task (nan if none)."""
        obs = self.observed
        n = obs.sum(axis=0)
        solved = np.where(obs, self.values, 0.0).sum(axis=0)
        return np.divide(solved, n, out=np.full(len(n), np.nan), where=n > 0)

    def select_systems(self, systems: Sequence[str]) -> "ResultsMatrix":
        idx = [self._system_index(s) for s in systems]
        return ResultsMatrix(tuple(systems), self.tasks, self.values[idx, :])

    def select_tasks(self, tasks: Sequence[str]) -> "ResultsMatrix":
        lookup = {t: j for j, t in enumerate(self.tasks)}
        missing = [t for t in tasks if t not in lookup]
        if missing:
            raise KeyError(f"unknown tasks: {missing[:5]}")
        idx = [lookup[t] for t in tasks]
        return ResultsMatrix(self.systems, tuple(tasks), self.values[:, idx])

    def drop_tasks(self, tasks: Iterable[str]) -> "ResultsMatrix":
        drop = set(tasks)
        return self.select_tasks([t for t in self.tasks if t not in drop])

    def top(self, k: int) -> "ResultsMatrix":
        """The ``k`` systems with the highest accuracy, best first.

        Ties are broken by the original row order so the result is deterministic.
        """
        acc = self.accuracy()
        order = sorted(range(self.n_systems), key=lambda i: (-np.nan_to_num(acc[i], nan=-1.0), i))
        return self.select_systems([self.systems[i] for i in order[:k]])

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.values, index=list(self.systems), columns=list(self.tasks))
        frame.index.name = "system"
        return frame

    def to_csv(self, path: str | Path) -> None:
        """Write the wide CSV that :func:`load_results` reads (missing = empty)."""
        frame = self.to_frame().astype("Int64")
        frame.to_csv(path, lineterminator="\n")

    def _system_index(self, system: str) -> int:
        try:
            return self.systems.index(system)
        except ValueError:
            raise KeyError(f"unknown system {system!r}") from None


def load_results(path: str | Path) -> ResultsMatrix:
    """Load a wide CSV: first column is the system name, the header names tasks.

    Cells may be 1/0, true/false, pass/fail (any case) or empty for "not run".
    """
    frame = pd.read_csv(path, index_col=0, dtype=str, keep_default_na=False)
    if frame.shape[1] == 0:
        raise ValueError(f"{path}: no task columns found")
    return ResultsMatrix.from_frame(frame)
