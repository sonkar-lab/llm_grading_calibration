"""
Unit tests for statistical utilities.

Tests power analysis functions for edge cases and correctness.
"""

import pytest
import numpy as np
from utils.statistics import (
    correlation_power,
    required_sample_size_correlation,
    compute_spearman_with_perm_and_bootstrap,
    estimate_power_spearman_simulation,
    detect_feature_type,
    estimate_feature_effect,
    compute_power_curve_mixed_model_lrt,
    compute_empirical_prior_mixed_model_beta,
    _try_import_lme4,
    _compute_bca_ci_from_jackknife,
    analyze_feature_moderated_calibration,
)


class TestCorrelationPower:
    """Tests for correlation_power() function."""

    def test_power_increases_with_n(self):
        """Power should increase as sample size increases."""
        r = 0.3
        powers = [correlation_power(r, n) for n in [10, 50, 100, 500]]
        # Each power should be greater than the previous
        for i in range(len(powers) - 1):
            assert powers[i] < powers[i+1], f"Power should increase with n: {powers}"

    def test_power_increases_with_r(self):
        """Power should increase as effect size increases."""
        n = 50
        powers = [correlation_power(r, n) for r in [0.1, 0.3, 0.5, 0.7]]
        # Each power should be greater than the previous
        for i in range(len(powers) - 1):
            assert powers[i] < powers[i+1], f"Power should increase with r: {powers}"

    def test_r_near_one_returns_nan(self):
        """r extremely close to 1 should return NaN with warning."""
        with pytest.warns(UserWarning, match="too close to"):
            power = correlation_power(r=0.99999999999, n=50)
            assert np.isnan(power), "Power should be NaN for r very close to 1"

    def test_r_exactly_one_returns_nan(self):
        """r exactly 1 should return NaN with warning."""
        with pytest.warns(UserWarning, match="too close to"):
            power = correlation_power(r=1.0, n=50)
            assert np.isnan(power), "Power should be NaN for r=1.0"

    def test_r_negative_near_minus_one_returns_nan(self):
        """r close to -1 should return NaN with warning."""
        with pytest.warns(UserWarning, match="too close to"):
            power = correlation_power(r=-0.99999999999, n=50)
            assert np.isnan(power), "Power should be NaN for r very close to -1"

    def test_small_n_returns_zero(self):
        """n < 4 should return 0."""
        assert correlation_power(r=0.5, n=3) == 0.0
        assert correlation_power(r=0.5, n=2) == 0.0
        assert correlation_power(r=0.5, n=1) == 0.0

    def test_moderate_correlation_moderate_n(self):
        """Test known value: r=0.3, n=50 should give reasonable power."""
        power = correlation_power(r=0.3, n=50)
        # From G*Power or similar: should be around 0.5-0.6
        assert 0.5 <= power <= 0.7, f"Power {power} outside expected range"

    def test_strong_correlation_large_n(self):
        """Strong correlation with large n should have high power."""
        power = correlation_power(r=0.7, n=100)
        # Should be very high power (>0.99)
        assert power > 0.99, f"Power {power} should be very high"

    def test_weak_correlation_small_n(self):
        """Weak correlation with small n should have low power."""
        power = correlation_power(r=0.1, n=20)
        # Should be low power (<0.2)
        assert power < 0.2, f"Power {power} should be very low"


class TestMixedModelPowerAndAssurance:
    def _skip_if_no_lme4(self):
        ok, _ = _try_import_lme4()
        if not ok:
            pytest.skip("rpy2/lme4 not available; skipping mixed-model power/assurance tests")

    def _make_crossed_data(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_students = 40
        questions = np.array([1, 2, 3, 4])
        rubrics_by_q = {
            1: ["r1a", "r1b"],
            2: ["r2a", "r2b"],
            3: ["r3a", "r3b"],
            4: ["r4a", "r4b"],
        }

        rows = []
        for s in range(n_students):
            sid = f"s{s:03d}"
            student_re = rng.normal(0, 0.6)
            for q in questions:
                ans_re = rng.normal(0, 0.4)
                q_fe = {1: -0.3, 2: 0.0, 3: 0.2, 4: 0.1}[int(q)]
                for rid in rubrics_by_q[int(q)]:
                    rubric_re = rng.normal(0, 0.5)
                    x = rng.normal()
                    rows.append((sid, int(q), rid, x, student_re, ans_re, rubric_re, q_fe))

        df = np.array(rows, dtype=object)
        student_ids = df[:, 0]
        question_ids = df[:, 1].astype(int)
        rubric_ids = df[:, 2]
        x = df[:, 3].astype(float)
        student_re = df[:, 4].astype(float)
        ans_re = df[:, 5].astype(float)
        rubric_re = df[:, 6].astype(float)
        q_fe = df[:, 7].astype(float)

        true_beta = 0.25
        y = true_beta * x + student_re + ans_re + rubric_re + q_fe + rng.normal(0, 1.0, size=len(x))
        return x, y, student_ids, question_ids, rubric_ids

    def test_power_curve_runs_and_orders_beta(self):
        self._skip_if_no_lme4()
        x, y, sid, qid, rid = self._make_crossed_data(seed=0)
        pc = compute_power_curve_mixed_model_lrt(
            x,
            y,
            student_ids=sid,
            question_ids=qid,
            rubric_ids=rid,
            beta_targets=[0.0, 0.25],
            n_students_target=int(len(np.unique(sid))),
            n_sim=6,
            seed_students=1,
            seed_outcomes=2,
        )
        assert {"beta_target", "beta_target_abs", "power", "n_success", "n_failed"} <= set(pc.columns)
        p0 = float(pc.loc[pc["beta_target_abs"] == 0.0, "power"].iloc[0])
        p1 = float(pc.loc[pc["beta_target_abs"] == 0.25, "power"].iloc[0])
        assert p1 >= p0

    def test_empirical_prior_bootstrap_runs(self):
        self._skip_if_no_lme4()
        x, y, sid, qid, rid = self._make_crossed_data(seed=1)
        prior = compute_empirical_prior_mixed_model_beta(
            x,
            y,
            student_ids=sid,
            question_ids=qid,
            rubric_ids=rid,
            n_boot=12,
            seed=3,
        )
        assert prior["n_attempted"] == 12
        beta_samples = np.asarray(prior["beta_samples"], dtype=float)
        assert beta_samples.ndim == 1
        assert beta_samples.size >= 1


class TestRequiredSampleSize:
    """Tests for required_sample_size_correlation() function."""

    def test_higher_power_needs_larger_n(self):
        """Higher power requirement needs larger sample size."""
        r = 0.3
        n_80 = required_sample_size_correlation(r, power=0.8)
        n_90 = required_sample_size_correlation(r, power=0.9)
        assert n_90 > n_80, f"Higher power should need larger n: {n_80} vs {n_90}"

    def test_weaker_r_needs_larger_n(self):
        """Weaker correlation needs larger sample size."""
        n_strong = required_sample_size_correlation(r=0.5, power=0.8)
        n_weak = required_sample_size_correlation(r=0.2, power=0.8)
        assert n_weak > n_strong, f"Weaker r should need larger n: {n_strong} vs {n_weak}"

    def test_r_near_one_raises_error(self):
        """r very close to 1 should raise ValueError."""
        with pytest.raises(ValueError, match="too close to"):
            required_sample_size_correlation(r=0.99999999999, power=0.8)

    def test_r_exactly_one_raises_error(self):
        """r exactly 1 should raise ValueError."""
        with pytest.raises(ValueError, match="too close to"):
            required_sample_size_correlation(r=1.0, power=0.8)

    def test_r_too_small_raises_error(self):
        """r < 0.0001 should raise ValueError."""
        with pytest.raises(ValueError, match="too weak"):
            required_sample_size_correlation(r=0.00005, power=0.8)

    def test_r_very_weak_raises_error(self):
        """r that needs > 10M samples should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds"):
            required_sample_size_correlation(r=0.0005, power=0.8)


class TestCalibrationRegressionHelpers:
    def test_bca_ci_from_jackknife_basic(self):
        rng = np.random.default_rng(0)
        boot_stats = rng.normal(loc=0.0, scale=1.0, size=200)
        jack_stats = rng.normal(loc=0.0, scale=1.0, size=50)
        ci_low, ci_high = _compute_bca_ci_from_jackknife(boot_stats, 0.0, jack_stats, alpha=0.05)
        assert np.isfinite(ci_low)
        assert np.isfinite(ci_high)
        assert ci_low < ci_high

    def test_calibration_regression_insufficient_data(self):
        is_correct = np.array([1, 0, 1, 0, 1])
        confidence = np.linspace(0.2, 0.8, 5)
        feature = np.linspace(0.0, 1.0, 5)
        student_ids = np.array(["s1", "s2", "s3", "s4", "s5"])
        res = analyze_feature_moderated_calibration(
            is_correct=is_correct,
            confidence=confidence,
            feature=feature,
            student_ids=student_ids,
            mode="linear",
        )
        assert res["curve"] is None
        assert any("Insufficient valid observations" in w for w in res["warnings"])

    def test_calibration_regression_zero_variance_feature(self):
        rng = np.random.default_rng(1)
        n = 12
        is_correct = rng.integers(0, 2, size=n)
        confidence = rng.uniform(0.1, 0.9, size=n)
        feature = np.ones(n)
        student_ids = np.array([f"s{i:02d}" for i in range(n)])
        res = analyze_feature_moderated_calibration(
            is_correct=is_correct,
            confidence=confidence,
            feature=feature,
            student_ids=student_ids,
            mode="linear",
        )
        assert res["curve"] is None
        assert any("zero variance" in w for w in res["warnings"])

    def test_achieved_power_meets_target(self):
        """Returned n should actually achieve target power."""
        r = 0.3
        target_power = 0.8
        n = required_sample_size_correlation(r, power=target_power)
        achieved_power = correlation_power(r, n, alpha=0.05)

        # Should be within 1% of target (binary search tolerance)
        assert achieved_power >= target_power - 0.01, \
            f"Achieved power {achieved_power} below target {target_power}"
        assert achieved_power <= target_power + 0.05, \
            f"Achieved power {achieved_power} much higher than target {target_power}"

    def test_known_value_r_03(self):
        """Test against known power analysis result for r=0.3."""
        # From G*Power or similar tools:
        # r=0.3, power=0.8, alpha=0.05 (two-tailed) → n ≈ 84
        n = required_sample_size_correlation(r=0.3, power=0.8, alpha=0.05)
        assert 82 <= n <= 86, f"n={n} outside expected range [82, 86]"

    def test_known_value_r_05(self):
        """Test against known power analysis result for r=0.5."""
        # From G*Power or similar tools:
        # r=0.5, power=0.8, alpha=0.05 → n ≈ 29
        n = required_sample_size_correlation(r=0.5, power=0.8, alpha=0.05)
        assert 28 <= n <= 31, f"n={n} outside expected range [28, 31]"

    def test_known_value_r_07(self):
        """Test against known power analysis result for r=0.7."""
        # From G*Power or similar tools:
        # r=0.7, power=0.8, alpha=0.05 → n ≈ 13
        n = required_sample_size_correlation(r=0.7, power=0.8, alpha=0.05)
        assert 12 <= n <= 15, f"n={n} outside expected range [12, 15]"

    def test_invalid_power(self):
        """Power outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="Power must be"):
            required_sample_size_correlation(r=0.3, power=0.0)
        with pytest.raises(ValueError, match="Power must be"):
            required_sample_size_correlation(r=0.3, power=1.0)
        with pytest.raises(ValueError, match="Power must be"):
            required_sample_size_correlation(r=0.3, power=1.5)

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="Alpha must be"):
            required_sample_size_correlation(r=0.3, alpha=0.0)
        with pytest.raises(ValueError, match="Alpha must be"):
            required_sample_size_correlation(r=0.3, alpha=1.0)

    def test_weak_but_detectable_correlation(self):
        """Test that weak correlations (r=0.02-0.05) can be computed."""
        # These should NOT raise errors (within 10M upper bound)
        n = required_sample_size_correlation(r=0.05, power=0.8)
        assert n > 0, f"Should return valid n for r=0.05"
        assert n < 10_000_000, f"n={n:,} should be within 10M bound"

        n = required_sample_size_correlation(r=0.02, power=0.8)
        assert n > 0, f"Should return valid n for r=0.02"
        assert n < 10_000_000, f"n={n:,} should be within 10M bound"

    def test_negative_correlation(self):
        """Negative correlations should work (using abs value)."""
        n_pos = required_sample_size_correlation(r=0.3, power=0.8)
        n_neg = required_sample_size_correlation(r=-0.3, power=0.8)
        # Should be identical (uses abs(r))
        assert n_pos == n_neg, f"Positive and negative r should give same n: {n_pos} vs {n_neg}"

    def test_upper_bound_warning(self):
        """Should warn if hitting upper bound."""
        # This test would need a very weak correlation that maxes out at 10M
        # For now, we'll test with a smaller max_n
        with pytest.warns(UserWarning, match="upper search bound"):
            n = required_sample_size_correlation(r=0.005, power=0.8, max_n=10000)
            assert n == 10000, "Should return max_n when hitting bound"


class TestIntegration:
    """Integration tests checking consistency between functions."""

    def test_power_sample_size_consistency(self):
        """Verify that required_n actually achieves target power."""
        test_cases = [
            (0.2, 0.8, 0.05),
            (0.3, 0.8, 0.05),
            (0.4, 0.9, 0.05),
            (0.5, 0.8, 0.01),
        ]

        for r, power_target, alpha in test_cases:
            n = required_sample_size_correlation(r, power=power_target, alpha=alpha)
            achieved_power = correlation_power(r, n, alpha=alpha)

            # Allow 2% tolerance
            assert achieved_power >= power_target - 0.02, \
                f"For r={r}, power={power_target}: achieved {achieved_power} < target"

    def test_symmetry_positive_negative_r(self):
        """Positive and negative r should behave symmetrically."""
        test_rs = [0.2, 0.3, 0.5, 0.7]

        for r in test_rs:
            n_pos = required_sample_size_correlation(r, power=0.8)
            n_neg = required_sample_size_correlation(-r, power=0.8)
            assert n_pos == n_neg, f"r={r} and r={-r} should give same n"

            power_pos = correlation_power(r, n=100)
            power_neg = correlation_power(-r, n=100)
            assert np.isclose(power_pos, power_neg), \
                f"r={r} and r={-r} should give same power"


class TestSpearmanPermutationBootstrap:
    """Tests for Spearman permutation + bootstrap correlation helper."""

    def test_strong_monotonic_signal(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 40)
        y = x + rng.normal(0, 0.05, size=len(x))
        res = compute_spearman_with_perm_and_bootstrap(
            x,
            y,
            n_perm=300,
            n_boot=300,
            alpha=0.05,
            random_state=rng,
        )
        assert res['rho'] > 0.9, f"Expected strong positive rho, got {res['rho']}"
        assert res['p_perm'] < 0.01, f"Permutation p-value should be small, got {res['p_perm']}"
        assert res['ci_low'] < res['rho'] < res['ci_high'], "Observed rho should fall within bootstrap CI"

    def test_no_signal(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=60)
        y = rng.normal(size=60)
        res = compute_spearman_with_perm_and_bootstrap(
            x,
            y,
            n_perm=300,
            n_boot=200,
            alpha=0.05,
            random_state=rng,
        )
        assert abs(res['rho']) < 0.3, "No-signal data should have small rho magnitude"
        assert 0.0 < res['p_perm'] < 1.0, "p-value should be finite and between 0 and 1"

    def test_insufficient_data_flag(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=2)
        y = rng.normal(size=2)
        res = compute_spearman_with_perm_and_bootstrap(
            x,
            y,
            n_perm=10,
            n_boot=10,
            alpha=0.05,
            random_state=rng,
        )
        assert res['insufficient_data'] is True
        assert np.isnan(res['rho'])
        assert np.isnan(res['p_perm'])

    def test_constant_input_undefined(self):
        x = np.ones(10)
        y = np.arange(10)
        res = compute_spearman_with_perm_and_bootstrap(
            x,
            y,
            n_perm=50,
            n_boot=0,
            alpha=0.05,
            random_state=np.random.default_rng(5),
        )
        assert res['undefined_correlation'] is True
        assert np.isnan(res['p_perm'])


class TestSpearmanPowerSimulation:
    """Tests for simulation-based power estimation."""

    def test_power_low_for_noise(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        power = estimate_power_spearman_simulation(
            x,
            y,
            n_target=40,
            alpha=0.05,
            n_power_sim=200,
            n_perm=200,
            random_state=rng,
        )
        assert power < 0.3, f"Power should be low for uncorrelated data, got {power}"

    def test_power_high_for_strong_signal(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=80)
        y = x + rng.normal(0, 0.1, size=80)
        power = estimate_power_spearman_simulation(
            x,
            y,
            n_target=60,
            alpha=0.05,
            n_power_sim=200,
            n_perm=200,
            random_state=rng,
        )
        assert power > 0.7, f"Power should be reasonably high for strong signal, got {power}"


class TestStep3MixedModelUtilities:
    def _make_nested_rows(self, n_students: int = 12):
        rng = np.random.default_rng(0)
        questions = np.array([1, 2, 3, 4])
        rubrics_by_q = {q: [f"r{q}a", f"r{q}b"] for q in questions}

        student_ids = []
        question_ids = []
        rubric_ids = []
        for s in range(n_students):
            for q in questions:
                for r in rubrics_by_q[q]:
                    student_ids.append(s)
                    question_ids.append(q)
                    rubric_ids.append(r)

        student_ids = np.asarray(student_ids, dtype=object)
        question_ids = np.asarray(question_ids, dtype=object)
        rubric_ids = np.asarray(rubric_ids, dtype=object)
        answer_ids = np.asarray([f"{s}:{q}" for s, q in zip(student_ids, question_ids)], dtype=object)

        x = rng.normal(size=len(student_ids))
        return student_ids, question_ids, rubric_ids, answer_ids, x

    def test_detect_feature_type_question_constant(self):
        student_ids, question_ids, rubric_ids, answer_ids, _ = self._make_nested_rows()
        q_effect = {1: -0.2, 2: 0.0, 3: 0.1, 4: 0.3}
        xq = np.asarray([q_effect[int(q)] for q in question_ids], dtype=float)
        ft = detect_feature_type(xq, student_ids, question_ids, rubric_ids, answer_ids, eps=1e-12)
        assert ft == "question_constant"

    def test_estimate_feature_effect_builds_answer_id_after_dropna(self):
        student_ids, question_ids, rubric_ids, _, x = self._make_nested_rows(n_students=10)
        rng = np.random.default_rng(1)
        y = 0.4 * x + rng.normal(size=len(x))

        # Inject missingness so the analytic subset differs by feature/target.
        x = x.copy()
        y = y.copy()
        x[:5] = np.nan
        y[10:12] = np.nan

        res = estimate_feature_effect(
            feature=x,
            outcome=y,
            student_ids=student_ids,
            question_ids=question_ids,
            rubric_ids=rubric_ids,
            outcome_type="continuous",
            run_gee_sensitivity=False,
            alpha=0.05,
        )
        assert res.n_obs == int(np.isfinite(x).astype(int).dot(np.isfinite(y).astype(int)))
        assert np.isfinite(res.beta)

    def test_gee_rubric_fe_guardrail_for_rubric_constant(self):
        student_ids, question_ids, rubric_ids, _, _ = self._make_nested_rows(n_students=10)
        rng = np.random.default_rng(2)
        y = rng.normal(size=len(student_ids))

        # Rubric-constant feature: exactly one value per rubric_id.
        rubric_map = {rid: i for i, rid in enumerate(np.unique(rubric_ids))}
        x = np.asarray([rubric_map[r] for r in rubric_ids], dtype=float)

        res = estimate_feature_effect(
            feature=x,
            outcome=y,
            student_ids=student_ids,
            question_ids=question_ids,
            rubric_ids=rubric_ids,
            outcome_type="continuous",
            run_gee_sensitivity=False,
            gee_mode="rubric_fe",
            alpha=0.05,
        )
        assert any("Switched gee_mode" in w for w in res.warnings), "Expected rubric_fe guardrail warning"
