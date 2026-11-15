from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from itemaudit import (
    compare,
    holm,
    mcnemar_exact,
    min_detectable_difference,
    min_significant_difference,
    paired_bootstrap_ci,
    paired_se,
)
from itemaudit.stats import wilson_interval


class TestMcNemar:
    def test_hand_computed(self) -> None:
        # 6 discordant tasks split 1/5: P(X <= 1 | n=6, p=.5) = (1 + 6) / 64, doubled.
        assert mcnemar_exact(1, 5) == pytest.approx(14 / 64)
        assert mcnemar_exact(5, 1) == pytest.approx(14 / 64)
        # 0/3: P(X = 0) = 1/8, doubled.
        assert mcnemar_exact(0, 3) == pytest.approx(0.25)

    def test_no_discordant_tasks(self) -> None:
        assert mcnemar_exact(0, 0) == 1.0

    def test_balanced_split_is_capped_at_one(self) -> None:
        assert mcnemar_exact(5, 5) == 1.0
        assert mcnemar_exact(1, 1) == 1.0

    @pytest.mark.parametrize("b,c", [(0, 1), (2, 9), (7, 3), (15, 30), (40, 41), (0, 12), (100, 60)])
    def test_matches_scipy_binomial_test(self, b: int, c: int) -> None:
        reference = stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        assert mcnemar_exact(b, c) == pytest.approx(reference, rel=1e-12)

    def test_negative_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            mcnemar_exact(-1, 2)


class TestPairedSE:
    def test_hand_computed(self) -> None:
        # d = [+1, +1, +1, -1, 0 x 6]: mean .2, sum of squares 4,
        # variance (4 - 10 * .04) / 9 = .4, SE = sqrt(.4 / 10) = .2
        assert paired_se(10, 3, 1) == pytest.approx(0.2)

    def test_matches_numpy_on_explicit_differences(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.integers(0, 2, 300)
        b = rng.integers(0, 2, 300)
        d = a - b
        expected = d.std(ddof=1) / np.sqrt(len(d))
        assert paired_se(300, int((d == 1).sum()), int((d == -1).sum())) == pytest.approx(expected)

    def test_identical_systems_have_zero_se(self) -> None:
        assert paired_se(50, 0, 0) == 0.0

    def test_single_task_is_undefined(self) -> None:
        assert np.isnan(paired_se(1, 1, 0))


class TestHolm:
    def test_hand_computed(self) -> None:
        # sorted: .005 .01 .03 .04 -> x4 x3 x2 x1 = .02 .03 .06 .04 -> running max .02 .03 .06 .06
        adjusted = holm([0.01, 0.04, 0.03, 0.005])
        assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_matches_textbook_loop(self) -> None:
        rng = np.random.default_rng(3)
        p = rng.uniform(size=40) ** 3
        order = np.argsort(p)
        expected = np.empty_like(p)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (len(p) - rank) * p[idx])
            expected[idx] = min(1.0, running)
        assert holm(p) == pytest.approx(expected)

    def test_single_value_unchanged_and_capped(self) -> None:
        assert holm([0.2]) == pytest.approx([0.2])
        assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])

    def test_never_below_raw_and_monotone(self) -> None:
        p = np.array([0.001, 0.2, 0.03, 0.5, 0.04])
        adjusted = holm(p)
        assert (adjusted >= p).all()
        assert (np.diff(adjusted[np.argsort(p)]) >= 0).all()

    def test_nan_ignored_in_family_size(self) -> None:
        adjusted = holm([0.01, np.nan, 0.04])
        assert np.isnan(adjusted[1])
        assert adjusted[[0, 2]] == pytest.approx([0.02, 0.04])

    def test_empty(self) -> None:
        assert holm([]).shape == (0,)


class TestCompare:
    def test_counts_and_estimates(self) -> None:
        a = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        b = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
        result = compare(a, b, names=("x", "y"))
        assert (result.n_both, result.n_only_a, result.n_only_b, result.n_neither) == (3, 3, 1, 3)
        assert result.accuracy_a == pytest.approx(0.6)
        assert result.accuracy_b == pytest.approx(0.4)
        assert result.diff == pytest.approx(0.2)
        assert result.se_paired == pytest.approx(0.2)
        assert result.discordance == pytest.approx(0.4)
        assert result.p_value == pytest.approx(mcnemar_exact(3, 1))
        z = stats.norm.ppf(0.975)
        assert result.ci_low == pytest.approx(0.2 - z * 0.2)
        assert result.ci_high == pytest.approx(0.2 + z * 0.2)
        assert result.system_a == "x" and result.system_b == "y"

    def test_unpaired_se_formula(self) -> None:
        a = [1] * 30 + [0] * 10
        b = [1] * 20 + [0] * 20
        result = compare(a, b)
        assert result.se_unpaired == pytest.approx(np.sqrt(0.75 * 0.25 / 40 + 0.5 * 0.5 / 40))

    def test_paired_se_is_smaller_when_systems_agree(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 500).astype(float)
        b = a.copy()
        flip = rng.choice(500, 30, replace=False)
        b[flip] = 1 - b[flip]
        result = compare(a, b)
        assert result.se_paired < result.se_unpaired

    def test_missing_tasks_are_dropped_pairwise(self) -> None:
        a = [1, 0, np.nan, 1, 1]
        b = [1, 1, 1, np.nan, 0]
        result = compare(a, b)
        assert result.n_tasks == 3
        assert (result.n_both, result.n_only_a, result.n_only_b) == (1, 1, 1)

    def test_booleans_accepted(self) -> None:
        result = compare([True, False, True], [False, False, True])
        assert result.n_only_a == 1 and result.diff == pytest.approx(1 / 3)

    def test_no_shared_tasks(self) -> None:
        result = compare([1, np.nan], [np.nan, 0])
        assert result.n_tasks == 0 and result.p_value == 1.0 and np.isnan(result.diff)

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError):
            compare([1, 0], [1])
        with pytest.raises(ValueError):
            compare([1, 2], [0, 1])

    def test_identical_systems(self) -> None:
        result = compare([1, 0, 1, 1], [1, 0, 1, 1])
        assert result.diff == 0 and result.p_value == 1.0
        assert (result.boot_ci_low, result.boot_ci_high) == (0.0, 0.0)

    def test_summary_and_dict(self) -> None:
        result = compare([1, 0, 1, 1], [0, 0, 1, 1], names=("alpha", "beta"))
        assert "alpha" in result.summary() and "McNemar" in result.summary()
        assert result.to_dict()["n_only_a"] == 1
        assert result.significant(0.05) is False


class TestBootstrap:
    def test_agrees_with_analytic_interval_for_large_n(self) -> None:
        low, high = paired_bootstrap_ci(2000, 220, 150, n_boot=40_000, seed=1)
        se = paired_se(2000, 220, 150)
        z = stats.norm.ppf(0.975)
        diff = (220 - 150) / 2000
        assert low == pytest.approx(diff - z * se, abs=0.002)
        assert high == pytest.approx(diff + z * se, abs=0.002)

    def test_multinomial_shortcut_matches_resampling_task_indices(self) -> None:
        rng = np.random.default_rng(5)
        d = np.array([1] * 40 + [-1] * 22 + [0] * 138)
        means = np.array([d[rng.integers(0, 200, 200)].mean() for _ in range(20_000)])
        low, high = paired_bootstrap_ci(200, 40, 22, n_boot=20_000, seed=6)
        assert low == pytest.approx(np.quantile(means, 0.025), abs=0.01)
        assert high == pytest.approx(np.quantile(means, 0.975), abs=0.01)

    def test_seed_makes_it_reproducible(self) -> None:
        assert paired_bootstrap_ci(100, 12, 5, seed=3) == paired_bootstrap_ci(100, 12, 5, seed=3)

    def test_interval_contains_point_estimate(self) -> None:
        low, high = paired_bootstrap_ci(500, 40, 25)
        assert low <= (40 - 25) / 500 <= high


class TestDetectableDifference:
    def test_power_half_closed_form(self) -> None:
        expected = stats.norm.ppf(0.975) * np.sqrt(0.1 / 500)
        assert min_significant_difference(500, 0.1) == pytest.approx(expected, rel=1e-9)
        assert expected == pytest.approx(0.0277, abs=1e-4)

    def test_solves_the_sample_size_equation(self) -> None:
        n, psi = 500, 0.12
        delta = min_detectable_difference(n, psi, 0.05, 0.8)
        z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
        needed_n = (z_a * np.sqrt(psi) + z_b * np.sqrt(psi - delta**2)) ** 2 / delta**2
        assert needed_n == pytest.approx(n, rel=1e-6)

    def test_simulated_power_is_close_to_nominal(self) -> None:
        n, psi = 500, 0.12
        delta = min_detectable_difference(n, psi, 0.05, 0.8)
        rng = np.random.default_rng(11)
        p_a, p_b = (psi + delta) / 2, (psi - delta) / 2
        counts = rng.multinomial(n, [p_a, p_b, 1 - psi], size=4000)
        b, c = counts[:, 0], counts[:, 1]
        z = (b - c) / np.sqrt(b + c)
        power = np.mean(np.abs(z) > stats.norm.ppf(0.975))
        assert power == pytest.approx(0.8, abs=0.04)

    def test_more_tasks_or_more_agreement_detects_smaller_differences(self) -> None:
        assert min_detectable_difference(2000, 0.1) < min_detectable_difference(500, 0.1)
        assert min_detectable_difference(500, 0.05) < min_detectable_difference(500, 0.2)

    def test_edge_cases(self) -> None:
        assert min_detectable_difference(500, 0.0) == 0.0
        assert np.isnan(min_detectable_difference(5, 0.1))
        with pytest.raises(ValueError):
            min_detectable_difference(500, 1.5)
        with pytest.raises(ValueError):
            min_detectable_difference(0, 0.1)


def test_wilson_interval_reference_value() -> None:
    # Standard worked example: 8 successes in 10 trials at 95%.
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=5e-4)
    assert high == pytest.approx(0.9433, abs=5e-4)
    assert all(np.isnan(v) for v in wilson_interval(0, 0))
