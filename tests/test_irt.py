from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from conftest import simulate_2pl
from itemaudit import IRTFit, ResultsMatrix, fit_2pl


def _fit_with(a: np.ndarray, d: np.ndarray, ability: np.ndarray | None = None) -> IRTFit:
    ability = np.zeros(3) if ability is None else ability
    return IRTFit(
        systems=tuple(f"s{i}" for i in range(len(ability))),
        tasks=tuple(f"t{j}" for j in range(len(a))),
        discrimination=np.asarray(a, dtype=float),
        intercept=np.asarray(d, dtype=float),
        ability=ability,
        ability_sd=np.ones(len(ability)),
        log_likelihood=0.0,
        n_iter=1,
        converged=True,
    )


@pytest.fixture(scope="module")
def simulated() -> tuple[IRTFit, np.ndarray, np.ndarray, np.ndarray]:
    matrix, a, b, theta = simulate_2pl(1500, 40, seed=42)
    return fit_2pl(matrix), a, b, theta


class TestRecovery:
    def test_converges(self, simulated: tuple[IRTFit, np.ndarray, np.ndarray, np.ndarray]) -> None:
        fit = simulated[0]
        assert fit.converged
        assert np.isfinite(fit.log_likelihood)

    def test_recovers_discrimination(self, simulated: tuple[IRTFit, np.ndarray, np.ndarray, np.ndarray]) -> None:
        fit, a, _, _ = simulated
        assert np.corrcoef(fit.discrimination, a)[0, 1] > 0.93
        assert np.sqrt(np.mean((fit.discrimination - a) ** 2)) < 0.15
        assert abs(np.mean(fit.discrimination - a)) < 0.08

    def test_recovers_difficulty(self, simulated: tuple[IRTFit, np.ndarray, np.ndarray, np.ndarray]) -> None:
        fit, _, b, _ = simulated
        assert np.corrcoef(fit.difficulty, b)[0, 1] > 0.99
        assert np.sqrt(np.mean((fit.difficulty - b) ** 2)) < 0.15

    def test_recovers_ability(self, simulated: tuple[IRTFit, np.ndarray, np.ndarray, np.ndarray]) -> None:
        fit, _, _, theta = simulated
        assert np.corrcoef(fit.ability, theta)[0, 1] > 0.93
        # Posterior standard deviations should be honest about the error.
        z = (fit.ability - theta) / fit.ability_sd
        assert 0.8 < z.std() < 1.25

    def test_small_sample_still_orders_tasks_and_systems(self) -> None:
        matrix, a, b, theta = simulate_2pl(150, 60, seed=9)
        fit = fit_2pl(matrix)
        assert fit.converged
        assert stats.spearmanr(fit.difficulty, b).statistic > 0.95
        assert stats.spearmanr(fit.ability, theta).statistic > 0.93
        assert stats.spearmanr(fit.ability, matrix.accuracy()).statistic > 0.97

    def test_missing_cells_are_ignored_not_scored_as_failures(self) -> None:
        matrix, a, b, _ = simulate_2pl(1200, 30, seed=17)
        values = matrix.values.copy()
        values[np.random.default_rng(1).uniform(size=values.shape) < 0.25] = np.nan
        fit = fit_2pl(ResultsMatrix(matrix.systems, matrix.tasks, values))
        assert fit.converged
        assert np.corrcoef(fit.difficulty, b)[0, 1] > 0.98
        assert abs(np.mean(fit.difficulty - b)) < 0.1


class TestSuspectTasks:
    def test_reverse_keyed_tasks_get_negative_slopes(self) -> None:
        matrix, a, _, _ = simulate_2pl(300, 50, seed=4, slope_range=(0.8, 2.0), difficulty_range=(-1.5, 1.5))
        values = matrix.values.copy()
        planted = [2, 11, 30, 47]
        values[:, planted] = 1 - values[:, planted]
        fit = fit_2pl(ResultsMatrix(matrix.systems, matrix.tasks, values))
        negative = set(np.flatnonzero(fit.discrimination < 0))
        assert negative == set(planted)
        # They are excluded from the information totals.
        assert (fit.mean_item_information()[planted] == 0).all()

    def test_constant_columns_stay_finite(self) -> None:
        matrix, *_ = simulate_2pl(100, 20, seed=6)
        values = matrix.values.copy()
        values[:, 0] = 1.0
        values[:, 1] = 0.0
        fit = fit_2pl(ResultsMatrix(matrix.systems, matrix.tasks, values))
        assert np.isfinite(fit.discrimination).all() and np.isfinite(fit.intercept).all()
        # Everyone solves task 0 and nobody solves task 1 at any plausible ability.
        assert fit.prob(0.0)[0, 0] > 0.9 and fit.prob(0.0)[1, 0] < 0.1
        assert fit.mean_item_information()[0] < np.median(fit.mean_item_information())

    def test_constant_rows_and_two_systems(self) -> None:
        values = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=float)
        fit = fit_2pl(ResultsMatrix(("all", "none"), tuple(f"t{j}" for j in range(5)), values))
        assert np.isfinite(fit.ability).all()
        assert fit.ability[0] > fit.ability[1]

    def test_no_discrimination_prior_limit(self) -> None:
        # A coin-flip task should get a slope near zero, far below the others.
        matrix, a, _, _ = simulate_2pl(800, 30, seed=8, slope_range=(1.0, 2.0))
        values = matrix.values.copy()
        values[:, 0] = np.random.default_rng(2).integers(0, 2, 800)
        fit = fit_2pl(ResultsMatrix(matrix.systems, matrix.tasks, values))
        assert abs(fit.discrimination[0]) < 0.25
        assert fit.discrimination[1:].min() > 0.6


class TestInformation:
    def test_item_information_formula(self) -> None:
        fit = _fit_with([1.0, 2.0], [0.0, -2.0])
        info = fit.item_information([0.0, 1.0])
        # a=1, b=0 at theta=0: p=.5 -> 1 * .25
        assert info[0, 0] == pytest.approx(0.25)
        # a=2, b=1 at theta=1: p=.5 -> 4 * .25
        assert info[1, 1] == pytest.approx(1.0)
        p = 1 / (1 + np.exp(2.0))
        assert info[1, 0] == pytest.approx(4 * p * (1 - p))
        assert fit.test_information([0.0])[0] == pytest.approx(0.25 + 4 * p * (1 - p))

    def test_difficulty_is_minus_intercept_over_slope(self) -> None:
        fit = _fit_with([1.0, 2.0, 0.0], [0.5, -2.0, 1.0])
        assert fit.difficulty[0] == pytest.approx(-0.5)
        assert fit.difficulty[1] == pytest.approx(1.0)
        assert np.isnan(fit.difficulty[2])

    def test_negative_slopes_are_left_out_by_default(self) -> None:
        fit = _fit_with([1.0, -1.0], [0.0, 0.0])
        assert fit.test_information([0.0])[0] == pytest.approx(0.25)
        assert fit.test_information([0.0], positive_only=False)[0] == pytest.approx(0.5)

    def test_effective_number_of_tasks(self) -> None:
        equal = _fit_with(np.ones(12), np.zeros(12))
        assert equal.effective_n_items() == pytest.approx(12.0)
        # One sharp task among eleven nearly flat ones carries almost everything.
        lopsided = _fit_with([3.0] + [0.05] * 11, np.zeros(12))
        assert 1.0 < lopsided.effective_n_items() < 1.1
        # Half the tasks carry no information at this ability.
        half = _fit_with([1.0] * 6 + [-1.0] * 6, np.zeros(12))
        assert half.effective_n_items() == pytest.approx(6.0)
        assert _fit_with([-1.0, -2.0], [0.0, 0.0]).effective_n_items() == 0.0

    def test_effective_number_depends_on_where_the_systems_are(self) -> None:
        # Six tasks centred at ability 0, six at ability 6: systems near 0 only use the first six.
        a = np.full(12, 2.0)
        d = np.r_[np.zeros(6), np.full(6, -12.0)]
        fit = _fit_with(a, d)
        assert fit.effective_n_items(np.array([0.0])) == pytest.approx(6.0, abs=0.01)
        assert fit.effective_n_items(np.array([3.0])) == pytest.approx(12.0, abs=0.01)

    def test_items_for_information_share(self) -> None:
        fit = _fit_with([2.0, 1.0, 1.0, 1.0, 1.0], np.zeros(5))
        # Information 1, .25, .25, .25, .25 (total 2): the first task alone holds half.
        assert fit.items_for_information_share(0.5) == 1
        assert fit.items_for_information_share(0.6) == 2
        assert fit.items_for_information_share(1.0) == 5
        assert _fit_with([-1.0], [0.0]).items_for_information_share(0.9) == 0

    def test_frames(self) -> None:
        matrix, *_ = simulate_2pl(50, 8, seed=0)
        fit = fit_2pl(matrix)
        assert list(fit.items_frame().index) == list(matrix.tasks)
        assert list(fit.systems_frame().columns) == ["ability", "ability_sd"]
        assert fit.prob(np.array([-1.0, 0.0, 1.0])).shape == (8, 3)
