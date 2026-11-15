from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from conftest import simulate_2pl
from itemaudit import ResultsMatrix, dif, guttman_errors, item_stats, kr20, system_fit
from itemaudit.items import FLAG_ALL, FLAG_NEAR_ZERO, FLAG_NEGATIVE, FLAG_NONE, rest_scores


def _manual_discrimination(values: np.ndarray, j: int) -> float:
    """Corrected item-rest correlation computed the slow, obvious way."""
    x, rest = [], []
    for row in values:
        if np.isnan(row[j]):
            continue
        others = np.delete(row, j)
        others = others[~np.isnan(others)]
        if others.size == 0:
            continue
        x.append(row[j])
        rest.append(others.mean())
    return float(np.corrcoef(x, rest)[0, 1])


class TestItemStats:
    def test_difficulty_counts(self, small_matrix: ResultsMatrix) -> None:
        items = item_stats(small_matrix)
        assert list(items["n"]) == [5, 5, 5, 5, 4, 5]
        assert list(items["n_solved"]) == [5, 3, 3, 2, 2, 0]
        assert list(items["n_minority"]) == [0, 2, 2, 2, 2, 0]
        np.testing.assert_allclose(items["solve_rate"], [1.0, 0.6, 0.6, 0.4, 0.5, 0.0])

    def test_discrimination_matches_manual_computation_with_missing_data(
        self, small_matrix: ResultsMatrix
    ) -> None:
        items = item_stats(small_matrix)
        for j, task in enumerate(small_matrix.tasks[1:5], start=1):
            expected = _manual_discrimination(small_matrix.values, j)
            assert items.loc[task, "discrimination"] == pytest.approx(expected)

    def test_hand_computed_point_biserial(self) -> None:
        # Task t1 against the rest score (mean of t2, t3):
        # x = [1, 1, 0, 0], rest = [1, .5, .5, 0] -> r = .5 / sqrt(1 * .5) = .7071
        values = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        matrix = ResultsMatrix(tuple("abcd"), ("t1", "t2", "t3"), values)
        assert item_stats(matrix).loc["t1", "discrimination"] == pytest.approx(np.sqrt(0.5))

    def test_rest_score_excludes_the_item_itself(self, small_matrix: ResultsMatrix) -> None:
        rest = rest_scores(small_matrix)
        assert rest[0, 5] == pytest.approx(1.0)  # system a solved all five other tasks
        assert rest[0, 0] == pytest.approx(4 / 5)
        assert np.isnan(rest[3, 4])  # missing cell
        assert rest[3, 0] == pytest.approx(1 / 4)  # d: four other observed tasks, one solved

    def test_constant_columns_are_flagged_and_have_no_discrimination(
        self, small_matrix: ResultsMatrix
    ) -> None:
        items = item_stats(small_matrix)
        assert items.loc["t1", "flag"] == FLAG_ALL
        assert items.loc["t6", "flag"] == FLAG_NONE
        assert np.isnan(items.loc["t1", "discrimination"])
        assert np.isnan(items.loc["t6", "discrimination"])

    def test_constant_rows_do_not_break_anything(self) -> None:
        values = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]], dtype=float)
        matrix = ResultsMatrix(tuple("abcd"), tuple(f"t{j}" for j in range(4)), values)
        items = item_stats(matrix)
        assert items["discrimination"].notna().all()
        assert (items["discrimination"].abs() <= 1).all()

    def test_two_systems_only_gives_undefined_discrimination(self) -> None:
        matrix = ResultsMatrix(("a", "b"), ("t1", "t2", "t3"), np.array([[1, 1, 0], [0, 1, 0]], dtype=float))
        items = item_stats(matrix)
        assert items["discrimination"].isna().all()
        assert items.loc["t2", "flag"] == FLAG_ALL and items.loc["t3", "flag"] == FLAG_NONE
        assert items.loc["t1", "flag"] == ""

    def test_confidence_interval_is_fisher_z(self) -> None:
        matrix, *_ = simulate_2pl(80, 30, seed=2)
        items = item_stats(matrix)
        r = items["discrimination"].iloc[0]
        half = stats.norm.ppf(0.975) / np.sqrt(80 - 3)
        assert items["disc_ci_low"].iloc[0] == pytest.approx(np.tanh(np.arctanh(r) - half))
        assert items["disc_ci_high"].iloc[0] == pytest.approx(np.tanh(np.arctanh(r) + half))

    def test_upper_lower_index(self) -> None:
        # Top 27% of 10 systems is 3 systems, likewise the bottom.
        values = np.zeros((10, 3))
        for i in range(10):
            values[i, : 3 - min(i // 3, 3)] = 1
        values[:, 2] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        matrix = ResultsMatrix(tuple(f"s{i}" for i in range(10)), ("t1", "t2", "t3"), values)
        items = item_stats(matrix)
        order = np.argsort(-matrix.accuracy(), kind="stable")
        top, bottom = order[:3], order[-3:]
        expected = values[top].mean(axis=0) - values[bottom].mean(axis=0)
        np.testing.assert_allclose(items["upper_lower"], expected)

    def test_planted_reverse_keyed_tasks_are_detected(self) -> None:
        matrix, a, _, _ = simulate_2pl(200, 60, seed=7, slope_range=(0.8, 2.0), difficulty_range=(-1.5, 1.5))
        values = matrix.values.copy()
        planted = [3, 17, 29, 41, 58]
        values[:, planted] = 1 - values[:, planted]
        items = item_stats(ResultsMatrix(matrix.systems, matrix.tasks, values))
        flagged = set(np.flatnonzero((items["flag"] == FLAG_NEGATIVE).to_numpy()))
        assert set(planted) <= flagged
        assert len(flagged - set(planted)) == 0
        # The planted tasks are significantly negative, not just below zero.
        assert (items["disc_ci_high"].iloc[planted] < 0).all()

    def test_pure_noise_task_is_near_zero(self) -> None:
        matrix, *_ = simulate_2pl(400, 40, seed=5)
        values = matrix.values.copy()
        values[:, 0] = np.random.default_rng(0).integers(0, 2, 400)
        items = item_stats(ResultsMatrix(matrix.systems, matrix.tasks, values))
        assert abs(items["discrimination"].iloc[0]) < 0.12
        assert items["flag"].iloc[0] in (FLAG_NEAR_ZERO, FLAG_NEGATIVE)
        assert (items["flag"].iloc[1:] == "").mean() > 0.9


class TestReliability:
    def test_kr20_hand_formula(self) -> None:
        values = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=float)
        matrix = ResultsMatrix(tuple("abcde"), tuple(f"t{j}" for j in range(4)), values)
        p = values.mean(axis=0)
        total_var = values.sum(axis=1).var()
        expected = 4 / 3 * (1 - (p * (1 - p)).sum() / total_var)
        assert kr20(matrix) == pytest.approx(expected)
        assert 0.7 < kr20(matrix) < 1.0

    def test_kr20_undefined_cases(self, small_matrix: ResultsMatrix) -> None:
        assert np.isnan(kr20(small_matrix))  # has a missing cell
        same = ResultsMatrix(("a", "b"), ("t1", "t2"), np.array([[1, 0], [0, 1]], dtype=float))
        assert np.isnan(kr20(same))  # every system has the same total


class TestGuttman:
    def test_perfect_scale_has_no_errors(self) -> None:
        values = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
        matrix = ResultsMatrix(tuple("abcd"), tuple(f"t{j}" for j in range(4)), values)
        errors, maximum = guttman_errors(matrix)
        assert list(errors) == [0, 0, 0, 0]
        assert list(maximum) == [0, 3, 4, 3]

    def test_hand_counted_errors(self) -> None:
        # Solve rates: t1 .75, t2 .5, t3 .25. System d solves only the hardest
        # task: it fails t1 and t2, both easier than t3 -> 2 errors.
        values = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        matrix = ResultsMatrix(tuple("abcd"), ("t1", "t2", "t3"), values)
        errors, maximum = guttman_errors(matrix)
        assert list(errors) == [0, 0, 0, 2]
        assert list(maximum) == [2, 2, 2, 2]

    def test_equally_difficult_tasks_are_never_errors(self) -> None:
        values = np.array([[1, 0], [0, 1]], dtype=float)
        matrix = ResultsMatrix(("a", "b"), ("t1", "t2"), values)
        errors, _ = guttman_errors(matrix)
        assert list(errors) == [0, 0]

    def test_missing_cells_are_skipped(self, small_matrix: ResultsMatrix) -> None:
        errors, maximum = guttman_errors(small_matrix)
        # System d: solved t1, t3; failed t2, t4, t6; t5 missing.
        # Rates: t1 1.0, t2 .6, t3 .6, t5 .5, t4 .4, t6 0. t2 and t3 tie, so no error.
        assert errors[3] == 0
        assert maximum[3] == 2 * 3
        # System c: solved t1, t2, t4; failed t3 (.6), t5 (.5), t6. t4 (.4) is
        # harder than t3 and t5 -> 2 errors.
        assert errors[2] == 2

    def test_planted_inverted_system_is_flagged(self) -> None:
        matrix, _, b, theta = simulate_2pl(120, 80, seed=3, slope_range=(1.5, 2.5))
        values = matrix.values.copy()
        order = np.argsort(matrix.solve_rate())
        values[0] = 0.0
        values[0, order[:30]] = 1.0  # solves the 30 hardest tasks, fails the rest
        fits = system_fit(ResultsMatrix(matrix.systems, matrix.tasks, values))
        assert fits["flag"].iloc[0] == "unusual_pattern"
        assert fits["guttman_norm"].iloc[0] > 0.9
        assert fits["guttman_norm"].iloc[0] > 3 * fits["guttman_expected"].iloc[0]
        assert fits["guttman_z"].iloc[0] == fits["guttman_z"].max()
        assert fits["guttman_z"].iloc[0] > 10
        assert (fits["flag"].iloc[1:] == "unusual_pattern").sum() <= 1

    def test_flag_is_calibrated_on_data_that_follow_the_model(self) -> None:
        # The raw error rate depends on the score, so it is compared with
        # simulations from the fitted model. On clean 2PL data almost nothing
        # should be flagged and the z-scores should look standard normal.
        flagged, z = 0, []
        for seed in range(4):
            matrix, *_ = simulate_2pl(150, 60, seed=100 + seed)
            fits = system_fit(matrix)
            flagged += int((fits["flag"] != "").sum())
            z.append(fits["guttman_z"].dropna().to_numpy())
        z = np.concatenate(z)
        assert flagged <= 4  # 600 systems; a nominal 3-sigma rule expects about 1
        assert abs(z.mean()) < 0.35
        assert 0.75 < z.std() < 1.25

    def test_without_a_model_reference_nothing_is_flagged(self, small_matrix: ResultsMatrix) -> None:
        fits = system_fit(small_matrix, model_reference=False)
        assert fits["guttman_expected"].isna().all() and fits["guttman_z"].isna().all()
        assert (fits["flag"] == "").all()
        assert fits["guttman_norm"].notna().any()

    def test_rejects_a_fit_from_another_matrix(self) -> None:
        from itemaudit import fit_2pl

        matrix, *_ = simulate_2pl(20, 8, seed=1)
        other, *_ = simulate_2pl(20, 9, seed=2)
        with pytest.raises(ValueError):
            system_fit(matrix, fit=fit_2pl(other))

    def test_system_fit_columns(self, small_matrix: ResultsMatrix) -> None:
        fits = system_fit(small_matrix, rare=0.4)
        assert list(fits["n"]) == [6, 6, 6, 5, 6]
        assert list(fits["n_solved"]) == [5, 4, 3, 2, 1]
        # Tasks solved by at most 40% of systems: t4 (.4) and t6 (0).
        assert list(fits["n_hard_solved"]) == [1, 0, 1, 0, 0]
        # Tasks solved by at least 60%: t1, t2, t3.
        assert list(fits["n_easy_failed"]) == [0, 0, 1, 1, 2]

    def test_all_or_nothing_systems_have_undefined_norm(self) -> None:
        values = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
        matrix = ResultsMatrix(tuple("abc"), ("t1", "t2", "t3"), values)
        fits = system_fit(matrix)
        assert np.isnan(fits["guttman_norm"].iloc[0]) and np.isnan(fits["guttman_norm"].iloc[1])
        assert (fits["flag"] == "").all()


class TestDif:
    def test_single_stratum_hand_computation(self) -> None:
        # Task t1: focal 2 of 10 solve it, reference 8 of 10.
        # A=2, B=8, C=8, D=2, N=20: OR = (2*2)/(8*8) = .0625
        # E[A] = 10*10/20 = 5, Var = 10*10*10*10 / (400*19) = 1.3158
        # chi2 = (|2 - 5| - .5)^2 / 1.3158 = 4.75
        values = np.zeros((20, 2))
        values[:2, 0] = 1
        values[10:18, 0] = 1
        values[:, 1] = 1
        matrix = ResultsMatrix(tuple(f"s{i}" for i in range(20)), ("t1", "t2"), values)
        groups = ["focal"] * 10 + ["ref"] * 10
        result = dif(matrix, groups, focal="focal", n_strata=1)
        row = result.loc["t1"]
        assert (row["n_focal"], row["n_reference"]) == (10, 10)
        assert row["rate_focal"] == pytest.approx(0.2) and row["rate_reference"] == pytest.approx(0.8)
        assert row["odds_ratio"] == pytest.approx(0.0625)
        assert row["chi2"] == pytest.approx(6.25 / (10000 / 7600))
        assert row["p_value"] == pytest.approx(stats.chi2.sf(4.75, 1))
        # A task everyone solves carries no evidence either way.
        assert np.isnan(result.loc["t2", "chi2"]) and result.loc["t2", "flag"] == ""

    def test_planted_group_dependent_tasks_are_detected(self) -> None:
        matrix, a, b, theta = simulate_2pl(300, 50, seed=13, slope_range=(1.0, 2.0))
        rng = np.random.default_rng(13)
        groups = np.where(np.arange(300) % 3 == 0, "new", "old")
        values = matrix.values.copy()
        planted = [5, 20, 44]
        focal = groups == "new"
        # In the "new" group the planted tasks fail 90% of the time regardless of ability.
        values[np.ix_(focal, planted)] = (rng.uniform(size=(focal.sum(), 3)) < 0.1).astype(float)
        result = dif(ResultsMatrix(matrix.systems, matrix.tasks, values), list(groups), focal="new")
        flagged = set(np.flatnonzero((result["flag"] == "dif").to_numpy()))
        assert set(planted) <= flagged
        assert len(flagged - set(planted)) <= 1
        assert (result["odds_ratio"].iloc[planted] < 0.3).all()

    def test_no_group_effect_flags_almost_nothing(self) -> None:
        matrix, *_ = simulate_2pl(300, 50, seed=21)
        groups = ["x" if i % 2 else "y" for i in range(300)]
        result = dif(matrix, groups)
        assert (result["flag"] == "dif").sum() <= 1
        assert (result["p_holm"].dropna() >= result["p_value"].dropna()).all()

    def test_groups_as_mapping_and_default_focal_is_the_smaller_group(self) -> None:
        matrix, *_ = simulate_2pl(60, 10, seed=1)
        mapping = {s: ("small" if i < 15 else "big") for i, s in enumerate(matrix.systems)}
        result = dif(matrix, mapping)
        assert (result["n_focal"] == 15).all() and (result["n_reference"] == 45).all()

    def test_rejects_wrong_group_specification(self) -> None:
        matrix, *_ = simulate_2pl(12, 5, seed=1)
        with pytest.raises(ValueError):
            dif(matrix, ["a"] * 12)
        with pytest.raises(ValueError):
            dif(matrix, ["a", "b", "c"] * 4)
        with pytest.raises(ValueError):
            dif(matrix, ["a", "b"] * 5)
        with pytest.raises(ValueError):
            dif(matrix, ["a", "b"] * 6, focal="z")
