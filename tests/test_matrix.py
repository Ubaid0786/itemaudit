from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itemaudit import ResultsMatrix, load_results


def test_load_accepts_common_spellings_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        "system,t1,t2,t3,t4\n"
        "alpha,1,true,PASS,\n"
        "beta,0,False,fail,NA\n"
        "gamma,1.0,yes,resolved,0\n",
        encoding="utf-8",
    )
    matrix = load_results(path)
    assert matrix.systems == ("alpha", "beta", "gamma")
    assert matrix.tasks == ("t1", "t2", "t3", "t4")
    expected = np.array([[1, 1, 1, np.nan], [0, 0, 0, np.nan], [1, 1, 1, 0]])
    np.testing.assert_array_equal(matrix.values, expected)
    assert matrix.observed.sum() == 10


def test_load_rejects_unknown_cell(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("system,t1\nalpha,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maybe"):
        load_results(path)


def test_load_rejects_file_without_tasks(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("system\nalpha\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_results(path)


def test_csv_round_trip_keeps_missing(tmp_path: Path, small_matrix: ResultsMatrix) -> None:
    path = tmp_path / "round.csv"
    small_matrix.to_csv(path)
    again = load_results(path)
    assert again.systems == small_matrix.systems and again.tasks == small_matrix.tasks
    np.testing.assert_array_equal(again.values, small_matrix.values)


def test_accuracy_and_solve_rate_ignore_missing(small_matrix: ResultsMatrix) -> None:
    np.testing.assert_allclose(small_matrix.accuracy(), [5 / 6, 4 / 6, 3 / 6, 2 / 5, 1 / 6])
    np.testing.assert_allclose(small_matrix.solve_rate(), [1.0, 0.6, 0.6, 0.4, 0.5, 0.0])


def test_all_missing_row_and_column_give_nan() -> None:
    values = np.array([[1.0, np.nan], [np.nan, np.nan]])
    matrix = ResultsMatrix(("a", "b"), ("t1", "t2"), values)
    assert np.isnan(matrix.accuracy()[1])
    assert np.isnan(matrix.solve_rate()[1])


def test_from_long_marks_absent_pairs_missing() -> None:
    long = pd.DataFrame(
        {
            "system": ["a", "a", "b"],
            "task": ["t1", "t2", "t1"],
            "result": [True, False, "pass"],
        }
    )
    matrix = ResultsMatrix.from_long(long)
    np.testing.assert_array_equal(matrix.values, np.array([[1, 0], [1, np.nan]]))
    with pytest.raises(ValueError):
        ResultsMatrix.from_long(pd.concat([long, long]))


def test_from_resolved_treats_unlisted_as_failed_and_ignores_unknown_ids() -> None:
    matrix = ResultsMatrix.from_resolved({"a": ["t2", "zzz"], "b": []}, ["t1", "t2"])
    np.testing.assert_array_equal(matrix.values, np.array([[0, 1], [0, 0]]))


def test_validation() -> None:
    with pytest.raises(ValueError):
        ResultsMatrix(("a",), ("t1", "t2"), np.array([[1.0]]))
    with pytest.raises(ValueError):
        ResultsMatrix(("a",), ("t1",), np.array([[0.5]]))
    with pytest.raises(ValueError):
        ResultsMatrix(("a", "a"), ("t1",), np.array([[1.0], [0.0]]))
    with pytest.raises(ValueError):
        ResultsMatrix(("a",), ("t1", "t1"), np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        ResultsMatrix(("a",), ("t1",), np.array([1.0]))


def test_values_are_read_only(small_matrix: ResultsMatrix) -> None:
    with pytest.raises(ValueError):
        small_matrix.values[0, 0] = 0.0


def test_select_drop_and_row(small_matrix: ResultsMatrix) -> None:
    sub = small_matrix.select_systems(["c", "a"]).select_tasks(["t3", "t1"])
    np.testing.assert_array_equal(sub.values, np.array([[0, 1], [1, 1]]))
    assert small_matrix.drop_tasks(["t1", "t6"]).tasks == ("t2", "t3", "t4", "t5")
    np.testing.assert_array_equal(small_matrix.row("e"), [1, 0, 0, 0, 0, 0])
    with pytest.raises(KeyError):
        small_matrix.row("nobody")
    with pytest.raises(KeyError):
        small_matrix.select_tasks(["nope"])


def test_top_orders_by_accuracy_and_breaks_ties_by_position() -> None:
    values = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=float)
    matrix = ResultsMatrix(("w", "x", "y", "z"), ("t1", "t2", "t3"), values)
    assert matrix.top(3).systems == ("z", "x", "y")
    assert matrix.top(10).systems == ("z", "x", "y", "w")
