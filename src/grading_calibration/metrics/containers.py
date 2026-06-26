"""
Containers for organizing metrics with statistical metadata.

This module provides dataclasses and utilities for storing metrics with confidence
intervals, stratified analysis results, and comparison results. These containers
make it easy to track statistical metadata alongside metric values and enable
rigorous hypothesis testing and reporting.

Classes:
    MetricResult: Single metric value with CI and metadata
    StratifiedMetrics: Metrics computed across groups (by rubric, question, etc.)
    ComparisonMetrics: Comparison between two conditions with statistical tests

Functions:
    bootstrap_ci: Compute bootstrap confidence intervals for any metric
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


@dataclass
class MetricResult:
    """
    Container for a single metric value with statistical metadata.

    This class stores a metric value along with confidence intervals, standard
    error, and sample size information. It provides methods for serialization
    and human-readable formatting.

    Attributes:
        name: Metric name (e.g., "ece", "auc_roc", "accuracy")
        value: Computed metric value
        n_samples: Number of samples used in computation
        confidence_interval: Optional (lower, upper) CI bounds
        stderr: Optional standard error
        metadata: Additional information (e.g., n_bins, bootstrap_iterations)

    Example:
        >>> result = MetricResult(
        ...     name="ece",
        ...     value=0.0542,
        ...     n_samples=1000,
        ...     confidence_interval=(0.0421, 0.0663),
        ...     metadata={"n_bins": 10}
        ... )
        >>> print(result.summary_string())
        'ECE: 0.0542 (95% CI: 0.0421-0.0663) [n=1000]'
    """
    name: str
    value: float
    n_samples: int
    confidence_interval: Optional[Tuple[float, float]] = None
    stderr: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON export.

        Returns:
            Dict with all attributes as serializable types
        """
        return {
            'name': self.name,
            'value': float(self.value),
            'n_samples': int(self.n_samples),
            'confidence_interval': self.confidence_interval,
            'stderr': float(self.stderr) if self.stderr is not None else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricResult':
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary with metric result data

        Returns:
            MetricResult instance
        """
        return cls(
            name=data['name'],
            value=data['value'],
            n_samples=data['n_samples'],
            confidence_interval=tuple(data['confidence_interval']) if data.get('confidence_interval') else None,
            stderr=data.get('stderr'),
            metadata=data.get('metadata', {})
        )

    def summary_string(self, show_ci: bool = True, decimals: int = 4) -> str:
        """
        Generate human-readable summary.

        Args:
            show_ci: Whether to include confidence interval
            decimals: Number of decimal places

        Returns:
            Formatted string like 'ECE: 0.0542 (95% CI: 0.0421-0.0663) [n=1000]'
        """
        s = f"{self.name.upper()}: {self.value:.{decimals}f}"

        if show_ci and self.confidence_interval is not None:
            ci_lower, ci_upper = self.confidence_interval
            s += f" (95% CI: {ci_lower:.{decimals}f}-{ci_upper:.{decimals}f})"

        s += f" [n={self.n_samples}]"
        return s


@dataclass
class StratifiedMetrics:
    """
    Container for metrics computed across multiple groups.

    This class stores metrics stratified by some grouping variable (e.g., rubric_id,
    question_id, confidence level). It provides methods for sorting, visualization,
    and summarization of grouped results.

    Attributes:
        metric_name: Name of the metric being stratified
        groups: Dict mapping group_key -> MetricResult
        stratification_key: Variable used for stratification (e.g., "rubric_id")
        overall: Optional overall metric computed on all data

    Example:
        >>> groups = {
        ...     'R1': MetricResult(name='ece', value=0.05, n_samples=200),
        ...     'R2': MetricResult(name='ece', value=0.08, n_samples=200)
        ... }
        >>> stratified = StratifiedMetrics(
        ...     metric_name='ece',
        ...     groups=groups,
        ...     stratification_key='rubric_id'
        ... )
        >>> df = stratified.summary_table()
        >>> print(df)
    """
    metric_name: str
    groups: Dict[str, 'MetricResult']
    stratification_key: str
    overall: Optional['MetricResult'] = None

    def get_group_values(self) -> Dict[str, float]:
        """
        Extract just the metric values for each group.

        Returns:
            Dict mapping group_key -> metric_value

        Example:
            >>> {'R1': 0.05, 'R2': 0.08, 'R3': 0.06}
        """
        return {k: v.value for k, v in self.groups.items()}

    def get_sorted_groups(self, ascending: bool = True) -> List[Tuple[str, 'MetricResult']]:
        """
        Return groups sorted by metric value.

        Args:
            ascending: If True, sort from lowest to highest value

        Returns:
            List of (group_key, MetricResult) tuples sorted by value
        """
        items = list(self.groups.items())
        items.sort(key=lambda x: x[1].value, reverse=not ascending)
        return items

    def summary_table(self) -> pd.DataFrame:
        """
        Create DataFrame summary table for analysis.

        Returns:
            DataFrame with columns: group_id, value, n_samples, ci_lower, ci_upper
        """
        rows = []
        for group_id, result in self.groups.items():
            row = {
                'group_id': group_id,
                'value': result.value,
                'n_samples': result.n_samples
            }
            if result.confidence_interval:
                row['ci_lower'] = result.confidence_interval[0]
                row['ci_upper'] = result.confidence_interval[1]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by group_id for consistent display
        if not df.empty:
            df = df.sort_values('group_id')

        return df

    def plot(self, figsize=(10, 6), **kwargs) -> matplotlib.figure.Figure:
        """
        Create bar plot of metric across groups with error bars.

        Args:
            figsize: Figure size tuple
            **kwargs: Additional arguments passed to plt.bar

        Returns:
            matplotlib Figure object
        """
        df = self.summary_table()

        if df.empty:
            raise ValueError("No groups to plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate error bars if CIs available
        if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
            errors = np.array([
                df['value'] - df['ci_lower'],
                df['ci_upper'] - df['value']
            ])
        else:
            errors = None

        ax.bar(df['group_id'], df['value'], yerr=errors, capsize=5, **kwargs)
        ax.set_xlabel(self.stratification_key)
        ax.set_ylabel(self.metric_name)
        ax.set_title(f"{self.metric_name} by {self.stratification_key}")
        ax.grid(axis='y', alpha=0.3)

        # Rotate x-labels if many groups
        if len(df) > 10:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        return fig


@dataclass
class ComparisonMetrics:
    """
    Container for comparing two sets of predictions.

    This class stores comparison results between two conditions (e.g., raw vs
    calibrated confidence, method A vs method B). It includes delta values,
    percentage changes, and statistical test results.

    Attributes:
        baseline_name: Name of baseline condition (e.g., "raw_confidence")
        comparison_name: Name of comparison condition (e.g., "isotonic_calibrated")
        metrics: Dict of metric_name -> (baseline_value, comparison_value, delta)
        statistical_tests: Dict of metric_name -> HypothesisTestResult
        n_samples: Number of samples compared

    Example:
        >>> comparison = ComparisonMetrics(
        ...     baseline_name="raw",
        ...     comparison_name="calibrated",
        ...     metrics={'ece': (0.08, 0.03, -0.05)},
        ...     n_samples=1000
        ... )
        >>> improvements = comparison.get_improvements(
        ...     ['ece'],
        ...     lower_is_better={'ece': True}
        ... )
        >>> print(improvements)
        {'ece': -62.5}  # 62.5% improvement
    """
    baseline_name: str
    comparison_name: str
    metrics: Dict[str, Tuple[float, float, float]]  # (baseline, comparison, delta)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)  # metric_name -> HypothesisTestResult
    n_samples: int = 0

    def get_improvements(
        self,
        metric_names: List[str],
        lower_is_better: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Calculate percentage improvements for specified metrics.

        Positive percentage means improvement in the desired direction.

        Args:
            metric_names: List of metrics to analyze
            lower_is_better: Dict specifying whether lower values are better
                            (e.g., {'ece': True, 'accuracy': False})

        Returns:
            Dict mapping metric_name -> percentage_improvement

        Example:
            >>> # ECE went from 0.08 to 0.03 (lower is better)
            >>> improvements = comparison.get_improvements(
            ...     ['ece'],
            ...     {'ece': True}
            ... )
            >>> print(improvements['ece'])
            62.5  # 62.5% improvement
        """
        improvements = {}

        for metric in metric_names:
            if metric not in self.metrics:
                continue

            baseline, comparison, delta = self.metrics[metric]

            if baseline == 0:
                pct_change = 0.0
            else:
                pct_change = (delta / baseline) * 100

            # Flip sign if lower is better
            if lower_is_better.get(metric, False):
                pct_change = -pct_change

            improvements[metric] = pct_change

        return improvements

    def summary_table(self, include_tests: bool = True) -> pd.DataFrame:
        """
        Create comprehensive comparison table.

        Args:
            include_tests: Whether to include hypothesis test results

        Returns:
            DataFrame with columns: metric, baseline, comparison, delta,
                                   pct_change, p_value, significant, effect_size
        """
        rows = []

        for metric_name, (baseline, comparison, delta) in self.metrics.items():
            row = {
                'metric': metric_name,
                'baseline': baseline,
                'comparison': comparison,
                'delta': delta,
                'pct_change': (delta / baseline * 100) if baseline != 0 else 0
            }

            if include_tests and metric_name in self.statistical_tests:
                test = self.statistical_tests[metric_name]
                row['p_value'] = test.p_value
                row['significant'] = test.significant
                if test.effect_size is not None:
                    row['effect_size'] = test.effect_size
                    row['effect_size_name'] = test.effect_size_name

            rows.append(row)

        df = pd.DataFrame(rows)

        return df

    def significant_improvements(self, alpha: float = 0.05) -> List[str]:
        """
        Return list of metrics with statistically significant improvements.

        Args:
            alpha: Significance level (default: 0.05)

        Returns:
            List of metric names with p-value < alpha
        """
        significant = []

        for metric_name, test in self.statistical_tests.items():
            if hasattr(test, 'p_value') and test.p_value < alpha:
                significant.append(metric_name)

        return significant


def bootstrap_ci(
    predictions: List[Any],
    metric_func: Callable,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for any metric.

    This function uses the percentile bootstrap method with resampling
    to estimate confidence intervals for arbitrary metrics. It's particularly
    useful for metrics with unknown sampling distributions (like ECE).

    Args:
        predictions: List of predictions (PredictionResult objects or similar)
        metric_func: Function that computes metric on List[predictions]
                    Example: lambda preds: expected_calibration_error(preds)
        n_bootstrap: Number of bootstrap iterations (default: 10000)
        ci_level: Confidence level (default: 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound) for confidence interval

    Example:
        >>> from grading_calibration.metrics import expected_calibration_error
        >>> ci_lower, ci_upper = bootstrap_ci(
        ...     predictions,
        ...     lambda p: expected_calibration_error(p),
        ...     n_bootstrap=1000
        ... )
        >>> print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    Notes:
        - Uses percentile method (simpler than BCa but may be less accurate)
        - Resamples with replacement
        - For small sample sizes (n < 30), increase n_bootstrap or use exact methods
    """
    np.random.seed(random_seed)

    bootstrap_values = []
    n = len(predictions)

    if n == 0:
        raise ValueError("Cannot compute bootstrap CI on empty prediction list")

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        resampled = [predictions[i] for i in indices]

        # Compute metric on resampled data
        try:
            value = metric_func(resampled)
            bootstrap_values.append(value)
        except Exception as e:
            # Skip iterations that fail (e.g., empty groups after resampling)
            continue

    if len(bootstrap_values) == 0:
        raise ValueError("All bootstrap iterations failed")

    bootstrap_values = np.array(bootstrap_values)

    # Percentile method
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))


def bootstrap_ci_with_metadata(
    predictions: List[Any],
    metric_func: Callable,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute bootstrap CI with additional metadata.

    Similar to bootstrap_ci but returns additional information useful for
    diagnostics and reporting.

    Args:
        predictions: List of predictions
        metric_func: Metric function
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level
        random_seed: Random seed

    Returns:
        Dict with keys:
            - 'lower': Lower CI bound
            - 'upper': Upper CI bound
            - 'mean': Bootstrap mean (may differ from observed)
            - 'stderr': Bootstrap standard error
            - 'n_iterations': Number of successful iterations
            - 'distribution': Array of bootstrap values (for visualization)
    """
    np.random.seed(random_seed)

    bootstrap_values = []
    n = len(predictions)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        resampled = [predictions[i] for i in indices]

        try:
            value = metric_func(resampled)
            bootstrap_values.append(value)
        except Exception:
            continue

    bootstrap_values = np.array(bootstrap_values)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

    return {
        'lower': float(lower),
        'upper': float(upper),
        'mean': float(np.mean(bootstrap_values)),
        'stderr': float(np.std(bootstrap_values)),
        'n_iterations': len(bootstrap_values),
        'distribution': bootstrap_values
    }
