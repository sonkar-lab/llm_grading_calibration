"""
Evaluation metrics, statistical testing, and visualization.
"""

from .metrics import (
    expected_calibration_error,
    brier_score,
    auc_roc_correctness,
    spearman_correlation,
    risk_coverage_curve,
    accuracy_at_coverage,
    coverage_at_risk,
    check_calibratability,
    evaluate_predictions,
)

from .plots import plot_calibration_curve

from .calibration_bins import CalibrationBinData, compute_calibration_bins

# New: Statistical analysis framework
from .containers import (
    MetricResult,
    StratifiedMetrics,
    ComparisonMetrics,
    bootstrap_ci,
    bootstrap_ci_with_metadata,
)

from .statistical_tests import (
    HypothesisTestResult,
    HypothesisTester,
)

__all__ = [
    # Metrics
    "expected_calibration_error",
    "brier_score",
    "auc_roc_correctness",
    "spearman_correlation",
    "risk_coverage_curve",
    "accuracy_at_coverage",
    "coverage_at_risk",
    "check_calibratability",
    "evaluate_predictions",
    # Visualization
    "plot_calibration_curve",
    # Calibration binning
    "CalibrationBinData",
    "compute_calibration_bins",
    # Containers
    "MetricResult",
    "StratifiedMetrics",
    "ComparisonMetrics",
    "bootstrap_ci",
    "bootstrap_ci_with_metadata",
    # Statistical testing
    "HypothesisTestResult",
    "HypothesisTester",
]
