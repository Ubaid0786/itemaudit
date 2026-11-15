from __future__ import annotations

import numpy as np
import pytest

from itemaudit import ResultsMatrix, holm, leaderboard, mcnemar_exact
from itemaudit.leaderboard import pairwise_counts, pairwise_mcnemar


def _three_clear_levels(n: int = 300) -> ResultsMatrix:
    """strong solves the first 80%, mid the first 50%, weak the first 20%."""
    tasks = tuple(f"t{j}" for j in range(n))
    values = np.zeros((3, n))
    values[0, : int(0.8 * n)] = 1
    values[1, : int(0.5 * n)] = 1
    values[2, : int(0.2 * n)] = 1
    # Row order is deliberately not the rank order.
    return ResultsMatrix(("mid", "weak", "strong"), tasks, values[[1, 2, 0]])


def test_clearly_separated_systems_get_their_own_ranks_and_tiers() -> None:
    board = leaderboard(_three_clear_levels(), n_boot=500)
    assert board.systems == ("strong", "mid", "weak")
    assert [r.rank for r in board.rows] == [1, 2, 3]
    assert [(r.rank_low, r.rank_high) for r in board.rows] == [(1, 1), (2, 2), (3, 3)]
    assert [(r.boot_rank_low, r.boot_rank_high) for r in board.rows] == [(1, 1), (2, 2), (3, 3)]
    assert [r.tier for r in board.rows] == [1, 2, 3]
    assert board.n_adjacent_indistinguishable == 0
    assert board.n_pairs == 3 and board.n_pairs_indistinguishable == 0
    assert all(pair.distinguishable for pair in board.adjacent)


def test_noise_level_differences_are_one_tier() -> None:
    rng = np.random.default_rng(0)
    base = (rng.uniform(size=400) < 0.7).astype(float)
    rows = []
    for _ in range(6):
        row = base.copy()
        flip = rng.choice(400, 20, replace=False)
        row[flip] = 1 - row[flip]
        rows.append(row)
    matrix = ResultsMatrix(tuple(f"s{i}" for i in range(6)), tuple(f"t{j}" for j in range(400)), np.array(rows))
    board = leaderboard(matrix, n_boot=500)
    assert board.n_adjacent_indistinguishable == 5
    assert board.n_tiers == 1
    assert all((r.rank_low, r.rank_high) == (1, 6) for r in board.rows)
    assert all(r.boot_rank_low <= r.rank <= r.boot_rank_high for r in board.rows)


def test_two_systems_only() -> None:
    values = np.array([[1, 1, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0]], dtype=float)
    matrix = ResultsMatrix(("a", "b"), tuple(f"t{j}" for j in range(8)), values)
    board = leaderboard(matrix, n_boot=200)
    assert len(board.rows) == 2 and len(board.adjacent) == 1
    pair = board.adjacent[0]
    assert pair.comparison.n_only_a == 3 and pair.comparison.n_only_b == 0
    assert pair.comparison.p_value == pytest.approx(0.25)
    # One comparison: Holm leaves it unchanged.
    assert pair.p_holm == pytest.approx(0.25)
    assert not pair.distinguishable
    assert board.to_dict()["n_adjacent_indistinguishable"] == 1


def test_single_system() -> None:
    matrix = ResultsMatrix(("only",), ("t1", "t2"), np.array([[1.0, 0.0]]))
    board = leaderboard(matrix, n_boot=50)
    assert len(board.rows) == 1 and board.adjacent == ()
    assert board.rows[0].rank_low == board.rows[0].rank_high == 1


def test_identical_systems_are_indistinguishable() -> None:
    row = np.array([1, 0, 1, 1, 0, 1], dtype=float)
    matrix = ResultsMatrix(("a", "b", "c"), tuple(f"t{j}" for j in range(6)), np.array([row, row, row]))
    board = leaderboard(matrix, n_boot=100)
    assert board.n_adjacent_indistinguishable == 2
    assert all(pair.comparison.p_value == 1.0 for pair in board.adjacent)
    assert board.n_tiers == 1


def test_pairwise_matrix_matches_scalar_tests() -> None:
    rng = np.random.default_rng(4)
    values = (rng.uniform(size=(7, 120)) < np.linspace(0.3, 0.8, 7)[:, None]).astype(float)
    values[rng.uniform(size=values.shape) < 0.1] = np.nan
    matrix = ResultsMatrix(tuple(f"s{i}" for i in range(7)), tuple(f"t{j}" for j in range(120)), values)
    only, shared = pairwise_counts(matrix)
    p = pairwise_mcnemar(only)
    for i in range(7):
        for j in range(7):
            if i == j:
                assert np.isnan(p[i, j])
                continue
            both = ~np.isnan(values[i]) & ~np.isnan(values[j])
            assert shared[i, j] == both.sum()
            assert only[i, j] == ((values[i] == 1) & (values[j] == 0) & both).sum()
            assert p[i, j] == pytest.approx(mcnemar_exact(int(only[i, j]), int(only[j, i])))


def test_adjacent_holm_uses_only_adjacent_pairs() -> None:
    rng = np.random.default_rng(8)
    values = (rng.uniform(size=(5, 200)) < np.array([0.8, 0.7, 0.55, 0.5, 0.3])[:, None]).astype(float)
    matrix = ResultsMatrix(tuple("abcde"), tuple(f"t{j}" for j in range(200)), values)
    board = leaderboard(matrix, n_boot=100)
    raw = [pair.comparison.p_value for pair in board.adjacent]
    assert [pair.p_holm for pair in board.adjacent] == pytest.approx(list(holm(raw)))
    # The all-pairs matrix is corrected over 10 comparisons, so it is never smaller.
    for i, pair in enumerate(board.adjacent):
        assert board.p_holm[i, i + 1] >= pair.p_holm - 1e-12
    assert np.allclose(board.p_holm, board.p_holm.T, equal_nan=True)


def test_top_restricts_ranking_and_the_test_family() -> None:
    rng = np.random.default_rng(9)
    values = (rng.uniform(size=(8, 150)) < np.linspace(0.9, 0.2, 8)[:, None]).astype(float)
    matrix = ResultsMatrix(tuple(f"s{i}" for i in range(8)), tuple(f"t{j}" for j in range(150)), values)
    board = leaderboard(matrix, top=3, n_boot=100)
    assert len(board.rows) == 3 and len(board.adjacent) == 2
    assert board.p_values.shape == (3, 3)
    accuracy = matrix.accuracy()
    assert [r.accuracy for r in board.rows] == sorted(accuracy, reverse=True)[:3]


def test_missing_cells_compare_on_shared_tasks() -> None:
    nan = np.nan
    values = np.array([[1, 1, 1, nan, nan, 1], [1, 0, 1, 1, 0, nan]], dtype=float)
    matrix = ResultsMatrix(("a", "b"), tuple(f"t{j}" for j in range(6)), values)
    board = leaderboard(matrix, n_boot=100)
    comparison = board.adjacent[0].comparison
    assert comparison.n_tasks == 3
    assert board.rows[0].n_tasks == 4 and board.rows[1].n_tasks == 5


def test_frames_have_one_row_per_system_and_pair() -> None:
    board = leaderboard(_three_clear_levels(), n_boot=100)
    assert list(board.to_frame()["system"]) == ["strong", "mid", "weak"]
    adjacent = board.adjacent_frame()
    assert len(adjacent) == 2 and adjacent["distinguishable"].all()
    assert (adjacent["diff"] > 0).all()


def test_seed_makes_the_bootstrap_reproducible() -> None:
    rng = np.random.default_rng(2)
    values = (rng.uniform(size=(5, 100)) < 0.6).astype(float)
    matrix = ResultsMatrix(tuple("abcde"), tuple(f"t{j}" for j in range(100)), values)
    assert leaderboard(matrix, n_boot=300, seed=1).to_dict() == leaderboard(matrix, n_boot=300, seed=1).to_dict()
