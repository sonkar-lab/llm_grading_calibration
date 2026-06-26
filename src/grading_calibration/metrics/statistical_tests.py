"""
Hypothesis testing framework for calibration research.

This module provides statistical hypothesis testing tools for comparing metrics,
computing effect sizes, performing power analysis, and applying multiple testing
corrections. It supports both parametric and non-parametric tests.

Classes:
    HypothesisTestResult: Result of a hypothesis test with p-value and effect size
    HypothesisTester: Framework for running various statistical tests

Key Features:
    - Paired and independent comparison tests
    - Bootstrap-based tests for arbitrary metrics
    - Effect size computation (Cohen's d, Cliff's delta)
    - Multiple testing correction (Bonferroni, Holm, FDR)
    - Power analysis (post-hoc and a priori)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class HypothesisTestResult:
    """
    Result of a hypothesis test.

    This class stores all relevant information from a hypothesis test, including
    test statistic, p-value, effect size, and interpretability information.

    Attributes:
        test_name: Name of test (e.g., "paired_t_test", "wilcoxon", "bootstrap_test")
        statistic: Test statistic value
        p_value: Two-tailed p-value
        effect_size: Effect size measure (Cohen's d, Cliff's delta, etc.)
        effect_size_name: Name of effect size measure
        null_hypothesis: Description of null hypothesis
        significant: Whether result is significant at alpha level
        alpha: Significance level used
        metadata: Additional test information (DOF, CI, sample sizes, etc.)

    Example:
        >>> result = HypothesisTestResult(
        ...     test_name="paired_t_test",
        ...     statistic=2.54,
        ...     p_value=0.012,
        ...     effect_size=0.45,
        ...     effect_size_name="Cohen's d",
        ...     significant=True,
        ...     alpha=0.05
        ... )
        >>> print(result.interpretation())
        "Significant difference detected (p=0.012, Cohen's d=0.45, small effect)"
    """
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    null_hypothesis: str = ""
    significant: bool = False
    alpha: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)

    def interpretation(self) -> str:
        """
        Generate human-readable interpretation of test results.

        Returns:
            String like "Significant difference detected (p=0.001, Cohen's d=0.85, large effect)"
        """
        if self.significant:
            result = "Significant difference detected"
        else:
            result = "No significant difference"

        parts = [result]
        parts.append(f"p={self.p_value:.4f}")

        if self.effect_size is not None and self.effect_size_name is not None:
            parts.append(f"{self.effect_size_name}={self.effect_size:.2f}")

            # Interpret effect size magnitude for Cohen's d
            if self.effect_size_name == "Cohen's d":
                abs_effect = abs(self.effect_size)
                if abs_effect < 0.2:
                    magnitude = "negligible"
                elif abs_effect < 0.5:
                    magnitude = "small"
                elif abs_effect < 0.8:
                    magnitude = "medium"
                else:
                    magnitude = "large"
                parts.append(f"{magnitude} effect")

            # Interpret Cliff's delta
            elif self.effect_size_name == "Cliff's delta":
                abs_effect = abs(self.effect_size)
                if abs_effect < 0.147:
                    magnitude = "negligible"
                elif abs_effect < 0.33:
                    magnitude = "small"
                elif abs_effect < 0.474:
                    magnitude = "medium"
                else:
                    magnitude = "large"
                parts.append(f"{magnitude} effect")

        return f"{parts[0]} ({', '.join(parts[1:])})"


class HypothesisTester:
    """
    Framework for statistical hypothesis testing on predictions.

    This class provides a comprehensive suite of hypothesis tests for comparing
    metrics between groups or conditions. It supports parametric and non-parametric
    tests, effect size computation, multiple testing correction, and power analysis.

    Attributes:
        alpha: Significance level (default: 0.05)
        random_seed: Random seed for reproducibility

    Example:
        >>> tester = HypothesisTester(alpha=0.05, random_seed=42)
        >>> result = tester.paired_t_test(
        ...     predictions_a,
        ...     predictions_b,
        ...     metric_func=lambda p: p.raw_confidence
        ... )
        >>> print(result.interpretation())
    """

    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """
        Initialize hypothesis tester.

        Args:
            alpha: Significance level (default: 0.05)
            random_seed: Random seed for reproducible bootstrap/permutation tests
        """
        self.alpha = alpha
        self.random_seed = random_seed
        np.random.seed(random_seed)

    # ===========================================================================
    # Paired Comparison Tests
    # ===========================================================================

    def paired_t_test(
        self,
        predictions_a: List[Any],
        predictions_b: List[Any],
        metric_func: Callable,
        alternative: str = "two-sided"
    ) -> HypothesisTestResult:
        """
        Paired t-test comparing metric values.

        Tests if the mean difference between paired observations is significantly
        different from zero. Assumes normally distributed differences.

        Args:
            predictions_a: First set of predictions
            predictions_b: Second set of predictions (matched instances)
            metric_func: Function that computes metric on single prediction
                        Example: lambda p: p.raw_confidence
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            HypothesisTestResult with t-statistic, p-value, and Cohen's d

        Example:
            >>> result = tester.paired_t_test(
            ...     preds_raw,
            ...     preds_calibrated,
            ...     lambda p: abs(p.raw_confidence - float(p.is_correct()))
            ... )
        """
        # Match instances by ID
        matched_a, matched_b = self._match_predictions(predictions_a, predictions_b)

        # Compute metric for each prediction
        values_a = np.array([metric_func(p) for p in matched_a])
        values_b = np.array([metric_func(p) for p in matched_b])

        # Paired t-test
        statistic, p_value = stats.ttest_rel(values_a, values_b, alternative=alternative)

        # Effect size (Cohen's d for paired samples)
        differences = values_a - values_b
        effect_size = np.mean(differences) / np.std(differences, ddof=1)

        return HypothesisTestResult(
            test_name="paired_t_test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name="Cohen's d",
            null_hypothesis="Mean difference = 0",
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            metadata={
                'n_pairs': len(matched_a),
                'mean_a': float(np.mean(values_a)),
                'mean_b': float(np.mean(values_b)),
                'mean_diff': float(np.mean(differences)),
                'alternative': alternative
            }
        )

    def wilcoxon_test(
        self,
        predictions_a: List[Any],
        predictions_b: List[Any],
        metric_func: Callable
    ) -> HypothesisTestResult:
        """
        Wilcoxon signed-rank test (non-parametric paired test).

        More robust alternative to paired t-test when metric distribution is
        non-normal. Good for ECE, Brier, and other bounded metrics.

        Args:
            predictions_a: First set of predictions
            predictions_b: Second set of predictions (matched instances)
            metric_func: Metric function for single prediction

        Returns:
            HypothesisTestResult with Wilcoxon statistic, p-value, and Cliff's delta
        """
        matched_a, matched_b = self._match_predictions(predictions_a, predictions_b)

        values_a = np.array([metric_func(p) for p in matched_a])
        values_b = np.array([metric_func(p) for p in matched_b])

        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(values_a, values_b)

        # Cliff's delta (non-parametric effect size)
        effect_size = self.cliffs_delta(values_a, values_b)

        return HypothesisTestResult(
            test_name="wilcoxon_signed_rank",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name="Cliff's delta",
            null_hypothesis="Median difference = 0",
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            metadata={'n_pairs': len(matched_a)}
        )

    # ===========================================================================
    # Independent Comparison Tests
    # ===========================================================================

    def independent_t_test(
        self,
        predictions_a: List[Any],
        predictions_b: List[Any],
        metric_func: Callable,
        equal_var: bool = False
    ) -> HypothesisTestResult:
        """
        Independent samples t-test (Welch's t-test by default).

        Compares means between two independent groups. Use for comparing
        different groups (e.g., different rubrics, questions).

        Args:
            predictions_a: First group
            predictions_b: Second group
            metric_func: Metric function for single prediction
            equal_var: If False (default), uses Welch's t-test (unequal variances)

        Returns:
            HypothesisTestResult with t-statistic, p-value, and Cohen's d
        """
        values_a = np.array([metric_func(p) for p in predictions_a])
        values_b = np.array([metric_func(p) for p in predictions_b])

        # Independent t-test
        statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=equal_var)

        # Cohen's d for independent samples
        effect_size = self.cohens_d(values_a, values_b, paired=False)

        test_name = "independent_t_test" if equal_var else "welch_t_test"

        return HypothesisTestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name="Cohen's d",
            null_hypothesis="Mean_A = Mean_B",
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            metadata={
                'n_a': len(values_a),
                'n_b': len(values_b),
                'mean_a': float(np.mean(values_a)),
                'mean_b': float(np.mean(values_b))
            }
        )

    def mann_whitney_test(
        self,
        predictions_a: List[Any],
        predictions_b: List[Any],
        metric_func: Callable
    ) -> HypothesisTestResult:
        """
        Mann-Whitney U test (non-parametric independent samples test).

        Tests if two independent samples have different distributions.
        More robust than t-test for non-normal data.

        Args:
            predictions_a: First group
            predictions_b: Second group
            metric_func: Metric function for single prediction

        Returns:
            HypothesisTestResult with U statistic, p-value, and Cliff's delta
        """
        values_a = np.array([metric_func(p) for p in predictions_a])
        values_b = np.array([metric_func(p) for p in predictions_b])

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')

        # Cliff's delta effect size
        effect_size = self.cliffs_delta(values_a, values_b)

        return HypothesisTestResult(
            test_name="mann_whitney_u",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name="Cliff's delta",
            null_hypothesis="Distributions are equal",
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            metadata={
                'n_a': len(values_a),
                'n_b': len(values_b)
            }
        )

    # ===========================================================================
    # Bootstrap-Based Tests
    # ===========================================================================

    def bootstrap_test(
        self,
        predictions_a: List[Any],
        predictions_b: List[Any],
        metric_func: Callable,
        n_bootstrap: int = 10000,
        paired: bool = True
    ) -> HypothesisTestResult:
        """
        Bootstrap hypothesis test for arbitrary metrics.

        Useful when metric has unknown sampling distribution (e.g., ECE,
        calibration metrics, AUC-ROC). Tests null hypothesis that
        metric(A) = metric(B).

        Args:
            predictions_a: First set of predictions
            predictions_b: Second set of predictions
            metric_func: Function computing metric on List[predictions]
                        Example: lambda preds: expected_calibration_error(preds)
            n_bootstrap: Number of bootstrap iterations
            paired: If True, maintain pairing during resampling

        Returns:
            HypothesisTestResult with observed difference, p-value, standardized effect

        Example:
            >>> from grading_calibration.metrics import expected_calibration_error
            >>> result = tester.bootstrap_test(
            ...     preds_raw,
            ...     preds_cal,
            ...     lambda p: expected_calibration_error(p),
            ...     n_bootstrap=10000,
            ...     paired=True
            ... )
        """
        np.random.seed(self.random_seed)

        # Observed difference
        metric_a = metric_func(predictions_a)
        metric_b = metric_func(predictions_b)
        observed_diff = metric_a - metric_b

        # Bootstrap under null hypothesis
        if paired:
            matched_a, matched_b = self._match_predictions(predictions_a, predictions_b)
            n = len(matched_a)

            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                # Resample pairs
                indices = np.random.choice(n, size=n, replace=True)
                boot_a = [matched_a[i] for i in indices]
                boot_b = [matched_b[i] for i in indices]

                try:
                    diff = metric_func(boot_a) - metric_func(boot_b)
                    bootstrap_diffs.append(diff)
                except Exception:
                    # Skip if metric computation fails
                    continue
        else:
            # Pool and split randomly (permutation under null)
            combined = predictions_a + predictions_b
            n_a = len(predictions_a)

            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                # Shuffle and split
                perm_indices = np.random.permutation(len(combined))
                boot_a = [combined[i] for i in perm_indices[:n_a]]
                boot_b = [combined[i] for i in perm_indices[n_a:]]

                try:
                    diff = metric_func(boot_a) - metric_func(boot_b)
                    bootstrap_diffs.append(diff)
                except Exception:
                    continue

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value: proportion of bootstrap diffs as extreme as observed
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        # Effect size (standardized difference)
        effect_size = observed_diff / np.std(bootstrap_diffs) if np.std(bootstrap_diffs) > 0 else 0.0

        return HypothesisTestResult(
            test_name="bootstrap_test",
            statistic=float(observed_diff),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name="standardized_diff",
            null_hypothesis="metric(A) = metric(B)",
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            metadata={
                'n_bootstrap': len(bootstrap_diffs),
                'metric_a': float(metric_a),
                'metric_b': float(metric_b),
                'observed_diff': float(observed_diff),
                'paired': paired
            }
        )

    # ===========================================================================
    # Effect Size Computation
    # ===========================================================================

    def cohens_d(
        self,
        values_a: np.ndarray,
        values_b: np.ndarray,
        paired: bool = False
    ) -> float:
        """
        Compute Cohen's d effect size (standardized mean difference).

        Cohen's d quantifies the difference between two means in terms of
        standard deviations. Common interpretations:
        - |d| < 0.2: negligible
        - |d| < 0.5: small
        - |d| < 0.8: medium
        - |d| >= 0.8: large

        Args:
            values_a: First group values
            values_b: Second group values
            paired: If True, use paired formula

        Returns:
            Cohen's d value
        """
        if paired:
            # Paired: d = mean(diff) / sd(diff)
            diff = values_a - values_b
            return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
        else:
            # Independent: d = (mean_a - mean_b) / pooled_sd
            mean_diff = np.mean(values_a) - np.mean(values_b)
            n_a, n_b = len(values_a), len(values_b)
            var_a = np.var(values_a, ddof=1)
            var_b = np.var(values_b, ddof=1)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

            return mean_diff / pooled_std if pooled_std > 0 else 0.0

    def cliffs_delta(
        self,
        values_a: np.ndarray,
        values_b: np.ndarray
    ) -> float:
        """
        Compute Cliff's delta (non-parametric effect size).

        Cliff's delta is the probability that a randomly selected value from
        group A is greater than a randomly selected value from group B, minus
        the reverse probability. Robust to outliers.

        Returns value in [-1, 1]:
        - -1: all values in A < all values in B
        - 0: no difference
        - +1: all values in A > all values in B

        Interpretation:
        - |δ| < 0.147: negligible
        - |δ| < 0.33: small
        - |δ| < 0.474: medium
        - |δ| >= 0.474: large

        Args:
            values_a: First group values
            values_b: Second group values

        Returns:
            Cliff's delta value
        """
        n_a, n_b = len(values_a), len(values_b)

        # Count pairs where a > b and a < b
        greater = 0
        less = 0

        for a in values_a:
            for b in values_b:
                if a > b:
                    greater += 1
                elif a < b:
                    less += 1

        delta = (greater - less) / (n_a * n_b)
        return delta

    # ===========================================================================
    # Multiple Testing Correction
    # ===========================================================================

    def bonferroni_correction(
        self,
        test_results: List[HypothesisTestResult]
    ) -> List[HypothesisTestResult]:
        """
        Apply Bonferroni correction to family of tests.

        Most conservative correction. Adjusted alpha = original_alpha / n_tests.
        Controls family-wise error rate (FWER).

        Args:
            test_results: List of hypothesis test results

        Returns:
            List of corrected results with updated significance
        """
        n_tests = len(test_results)
        adjusted_alpha = self.alpha / n_tests

        corrected = []
        for result in test_results:
            # Create new result with adjusted alpha
            corrected_result = HypothesisTestResult(
                test_name=result.test_name,
                statistic=result.statistic,
                p_value=result.p_value,
                effect_size=result.effect_size,
                effect_size_name=result.effect_size_name,
                null_hypothesis=result.null_hypothesis,
                significant=(result.p_value < adjusted_alpha),
                alpha=adjusted_alpha,
                metadata={
                    **result.metadata,
                    'correction': 'bonferroni',
                    'n_tests': n_tests,
                    'original_alpha': self.alpha
                }
            )
            corrected.append(corrected_result)

        return corrected

    def holm_correction(
        self,
        test_results: List[HypothesisTestResult]
    ) -> List[HypothesisTestResult]:
        """
        Apply Holm-Bonferroni correction (step-down procedure).

        Less conservative than Bonferroni. Tests are evaluated sequentially
        with decreasing threshold.

        Args:
            test_results: List of hypothesis test results

        Returns:
            List of corrected results
        """
        # Sort by p-value
        sorted_results = sorted(test_results, key=lambda t: t.p_value)
        n_tests = len(sorted_results)

        corrected = []
        for i, result in enumerate(sorted_results):
            adjusted_alpha = self.alpha / (n_tests - i)

            corrected_result = HypothesisTestResult(
                test_name=result.test_name,
                statistic=result.statistic,
                p_value=result.p_value,
                effect_size=result.effect_size,
                effect_size_name=result.effect_size_name,
                null_hypothesis=result.null_hypothesis,
                significant=(result.p_value < adjusted_alpha),
                alpha=adjusted_alpha,
                metadata={
                    **result.metadata,
                    'correction': 'holm',
                    'rank': i + 1,
                    'original_alpha': self.alpha
                }
            )
            corrected.append(corrected_result)

        return corrected

    def fdr_correction(
        self,
        test_results: List[HypothesisTestResult],
        method: str = "bh"
    ) -> List[HypothesisTestResult]:
        """
        Apply FDR (False Discovery Rate) correction.

        Controls expected proportion of false discoveries. Less conservative
        than FWER methods. Good for exploratory analysis.

        Args:
            test_results: List of hypothesis test results
            method: "bh" for Benjamini-Hochberg (default)

        Returns:
            List of corrected results
        """
        try:
            from statsmodels.stats.multitest import multipletests

            p_values = [t.p_value for t in test_results]
            _, p_adjusted, _, _ = multipletests(p_values, alpha=self.alpha, method=method)

            corrected = []
            for result, p_adj in zip(test_results, p_adjusted):
                corrected_result = HypothesisTestResult(
                    test_name=result.test_name,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    effect_size=result.effect_size,
                    effect_size_name=result.effect_size_name,
                    null_hypothesis=result.null_hypothesis,
                    significant=(p_adj < self.alpha),
                    alpha=self.alpha,
                    metadata={
                        **result.metadata,
                        'correction': method,
                        'p_adjusted': float(p_adj),
                        'original_alpha': self.alpha
                    }
                )
                corrected.append(corrected_result)

            return corrected

        except ImportError:
            # Fallback if statsmodels not available
            print("Warning: statsmodels not available, using Bonferroni instead")
            return self.bonferroni_correction(test_results)

    # ===========================================================================
    # Power Analysis
    # ===========================================================================

    def post_hoc_power(
        self,
        effect_size: float,
        n_samples: int,
        test_type: str = "paired_t"
    ) -> float:
        """
        Compute post-hoc statistical power.

        Given observed effect size and sample size, calculates the probability
        that the test would detect this effect.

        Args:
            effect_size: Observed effect size (Cohen's d)
            n_samples: Sample size per group
            test_type: Type of test ("paired_t", "independent_t")

        Returns:
            Statistical power (0 to 1)
        """
        try:
            from statsmodels.stats.power import tt_ind_solve_power

            power = tt_ind_solve_power(
                effect_size=effect_size,
                nobs1=n_samples,
                alpha=self.alpha,
                ratio=1.0
            )
            return float(power)

        except ImportError:
            # Rough approximation if statsmodels not available
            import warnings
            warnings.warn("statsmodels not available, returning approximate power")
            # Very rough approximation
            return 0.8 if abs(effect_size) > 0.5 else 0.5

    def a_priori_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        test_type: str = "paired_t"
    ) -> int:
        """
        Compute required sample size for desired power.

        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired power (default: 0.8)
            test_type: Type of test

        Returns:
            Required sample size per group
        """
        try:
            from statsmodels.stats.power import tt_ind_solve_power

            n = tt_ind_solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=power,
                ratio=1.0
            )
            return int(np.ceil(n))

        except ImportError:
            # Rough approximation
            import warnings
            warnings.warn("statsmodels not available, returning approximate sample size")
            # Very rough approximation: n ≈ 16 / d^2 for power=0.8, alpha=0.05
            return max(30, int(np.ceil(16 / (effect_size ** 2))))

    # ===========================================================================
    # Helper Methods
    # ===========================================================================

    def _match_predictions(
        self,
        predictions_a: List[Any],
        predictions_b: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Match predictions by instance_id for paired tests.

        Args:
            predictions_a: First set of predictions
            predictions_b: Second set of predictions

        Returns:
            Tuple of (matched_a, matched_b) with same order

        Raises:
            ValueError: If no common instance_ids found
        """
        # Create lookup dicts
        dict_a = {p.instance_id: p for p in predictions_a}
        dict_b = {p.instance_id: p for p in predictions_b}

        # Find common IDs
        common_ids = set(dict_a.keys()) & set(dict_b.keys())

        if len(common_ids) == 0:
            raise ValueError("No common instance_ids found for pairing")

        # Extract matched pairs in consistent order
        matched_a = [dict_a[id] for id in sorted(common_ids)]
        matched_b = [dict_b[id] for id in sorted(common_ids)]

        return matched_a, matched_b
