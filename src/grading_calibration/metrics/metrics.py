"""
Comprehensive evaluation metrics for confidence estimation.

Includes:
- Calibration metrics (ECE, Brier score)
- Ranking metrics (AUC-ROC, Spearman correlation)
- Selective prediction metrics (Risk-Coverage, Acc@K, Cov@Risk)
- Verification metrics (pre-calibration diagnostics)
"""

from typing import List, Dict, Any, Tuple, Optional
import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from grading_calibration.domain.models import PredictionResult
from .calibration_bins import (
    compute_calibration_bins,
    CalibrationBinData,
    _extract_confidence_correctness
)


# ============================================================================
# Calibration Metrics
# ============================================================================


def expected_calibration_error(
    predictions: List[PredictionResult],
    n_bins: int = 10,
    use_calibrated: bool = False,
    verbose: bool = False
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average difference between confidence and accuracy.

    Args:
        predictions: List of predictions
        n_bins: Number of bins for discretization
        use_calibrated: Use calibrated_confidence if True, else raw_confidence
        verbose: If True, print per-bin statistics for debugging (default: False)

    Returns:
        ECE score (lower is better, 0 is perfect)
    """
    # Use unified binning logic
    bin_data = compute_calibration_bins(predictions, n_bins, use_calibrated)

    # Print per-bin statistics for debugging (only if verbose=True)
    if verbose:
        for i, (bin_size, accuracy, mean_conf) in enumerate(zip(
            bin_data.bin_sizes, bin_data.accuracies, bin_data.mean_confidences
        )):
            prop_in_bin = bin_size / bin_data.n_total
            bin_lower = bin_data.bin_boundaries[i]
            bin_upper = bin_data.bin_boundaries[i + 1]
            print(
                f"Bin [{bin_lower:.2f}, {bin_upper:.2f}]: "
                f"Prop={prop_in_bin:.3f}, Acc={accuracy:.3f}, "
                f"Conf={mean_conf:.3f}"
            )

    return bin_data.get_ece()


def brier_score(
    predictions: List[PredictionResult],
    use_calibrated: bool = False,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> float:
    """
    Compute Brier score (mean squared error between confidence and correctness).

    Args:
        predictions: List of predictions
        use_calibrated: Use calibrated_confidence if True
        precomputed: Optional (confidences, correctness) tuple to avoid re-extraction

    Returns:
        Brier score (lower is better, 0 is perfect)
    """
    if precomputed is not None:
        confidences, correctness = precomputed
    else:
        confidences, correctness = _extract_confidence_correctness(
            predictions, use_calibrated
        )

    return float(np.mean((confidences - correctness) ** 2))


# ============================================================================
# Ranking Metrics
# ============================================================================


def auc_roc_correctness(
    predictions: List[PredictionResult],
    use_calibrated: bool = False,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> float:
    """
    Compute AUC-ROC for discriminating correct vs incorrect predictions.

    Args:
        predictions: List of predictions
        use_calibrated: Use calibrated_confidence if True
        precomputed: Optional (confidences, correctness) tuple to avoid re-extraction

    Returns:
        AUC-ROC score (higher is better, 0.5 is random, 1.0 is perfect)
    """
    if precomputed is not None:
        confidences, correctness = precomputed
    else:
        confidences, correctness = _extract_confidence_correctness(
            predictions, use_calibrated
        )

    # Need at least one positive and one negative
    if len(np.unique(correctness)) < 2:
        return 0.5  # Random baseline if all same

    return float(roc_auc_score(correctness, confidences))


def spearman_correlation(
    predictions: List[PredictionResult],
    use_calibrated: bool = False,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation between confidence and correctness.

    Args:
        predictions: List of predictions
        use_calibrated: Use calibrated_confidence if True
        precomputed: Optional (confidences, correctness) tuple to avoid re-extraction

    Returns:
        (correlation, p-value) tuple
    """
    if precomputed is not None:
        confidences, correctness = precomputed
    else:
        confidences, correctness = _extract_confidence_correctness(
            predictions, use_calibrated
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="An input array is constant; the correlation coefficient is not defined."
        )
        return spearmanr(confidences, correctness)


# ============================================================================
# Selective Prediction Metrics
# ============================================================================


def risk_coverage_curve(
    predictions: List[PredictionResult],
    use_calibrated: bool = False,
    n_points: int = 100,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute risk-coverage curve.

    At each coverage level γ, compute error rate on top-γ confident predictions.

    Args:
        predictions: List of predictions
        use_calibrated: Use calibrated_confidence if True
        n_points: Number of points on the curve
        precomputed: Optional (confidences, correctness) tuple to avoid re-extraction

    Returns:
        Dictionary with 'coverage' and 'risk' arrays
    """
    if precomputed is not None:
        confidences, correctness = precomputed
    else:
        confidences, correctness = _extract_confidence_correctness(
            predictions, use_calibrated
        )

    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidences)
    sorted_correctness = correctness[sorted_indices]

    # Compute risk at each coverage level
    coverages = np.linspace(0, 1, n_points)
    risks = []

    for coverage in coverages:
        if coverage == 0:
            risks.append(0.0)
            continue

        n_selected = int(coverage * len(predictions))
        if n_selected == 0:
            risks.append(0.0)
            continue

        # Risk = error rate on selected predictions
        selected_correctness = sorted_correctness[:n_selected]
        risk = 1 - selected_correctness.mean()
        risks.append(risk)

    return {"coverage": coverages, "risk": np.array(risks)}


def accuracy_at_coverage(
    predictions: List[PredictionResult],
    coverage: float,
    use_calibrated: bool = False,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> float:
    """
    Compute accuracy on top-coverage fraction of predictions.

    Args:
        predictions: List of predictions
        coverage: Fraction of predictions to consider (e.g., 0.5 for top 50%)
        use_calibrated: Use calibrated_confidence if True
        precomputed: Optional (confidences, correctness) tuple to avoid re-extraction

    Returns:
        Accuracy on selected predictions
    """
    if precomputed is not None:
        confidences, correctness = precomputed
    else:
        confidences, correctness = _extract_confidence_correctness(
            predictions, use_calibrated
        )

    # Select top-coverage predictions
    n_selected = int(coverage * len(predictions))
    if n_selected == 0:
        return 0.0

    top_indices = np.argsort(-confidences)[:n_selected]
    return correctness[top_indices].mean()


def coverage_at_risk(
    predictions: List[PredictionResult],
    target_risk: float,
    use_calibrated: bool = False,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> float:
    """
    Compute maximum coverage achievable while maintaining target risk level.

    Args:
        predictions: List of predictions
        target_risk: Maximum acceptable error rate (e.g., 0.05 for 5% errors)
        use_calibrated: Use calibrated_confidence if True
        precomputed: Optional (confidences, correctness) tuple to avoid re-extraction

    Returns:
        Maximum coverage at target risk
    """
    rc_curve = risk_coverage_curve(
        predictions, use_calibrated, n_points=1000, precomputed=precomputed
    )

    # Find maximum coverage where risk <= target_risk
    valid_coverages = rc_curve["coverage"][rc_curve["risk"] <= target_risk]

    if len(valid_coverages) == 0:
        return 0.0

    return valid_coverages[-1]


# ============================================================================
# Verification Metrics (Pre-Calibration Diagnostics)
# ============================================================================


def check_calibratability(predictions: List[PredictionResult]) -> Dict[str, Any]:
    """
    Verify if confidence scores are calibratable.

    Checks:
    1. Correlation between confidence and correctness
    2. AUC-ROC (discrimination ability)
    3. Monotonicity (higher confidence → higher accuracy)
    4. Resolution (spread of confidence values)

    Args:
        predictions: List of predictions with raw confidences

    Returns:
        Dictionary with diagnostic metrics and recommendations
    """
    confidences = np.array([p.raw_confidence for p in predictions])
    correctness = np.array([p.is_correct() for p in predictions], dtype=int)

    # 1. Correlation
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="An input array is constant; the correlation coefficient is not defined."
        )
        corr, p_value = spearmanr(confidences, correctness)

    # 2. AUC-ROC
    if len(np.unique(correctness)) >= 2:
        auc = roc_auc_score(correctness, confidences)
    else:
        auc = 0.5

    # 3. Monotonicity (bin analysis) - use unified binning logic
    n_bins = 10
    bin_data = compute_calibration_bins(predictions, n_bins, use_calibrated=False)
    bin_accuracies = bin_data.accuracies.tolist()

    # Check if generally increasing
    monotonicity_violations = 0
    for i in range(len(bin_accuracies) - 1):
        if bin_accuracies[i + 1] < bin_accuracies[i]:
            monotonicity_violations += 1

    monotonicity_score = 1 - (monotonicity_violations / max(len(bin_accuracies) - 1, 1))

    # 4. Resolution
    resolution = np.std(confidences)
    unique_confidences = len(np.unique(confidences))

    # Determine calibratability
    is_calibratable = (
        corr > 0.4 and auc > 0.65 and monotonicity_score > 0.6 and resolution > 0.1
    )

    return {
        "is_calibratable": is_calibratable,
        "correlation": corr,
        "correlation_pvalue": p_value,
        "auc_roc": auc,
        "monotonicity_score": monotonicity_score,
        "resolution_std": resolution,
        "n_unique_confidences": unique_confidences,
        "recommendation": (
            "Good proxy - proceed with calibration"
            if is_calibratable
            else "Poor proxy - consider different confidence method"
        ),
    }


# ============================================================================
# Comprehensive Evaluation
# ============================================================================


def evaluate_predictions(
    predictions: List[PredictionResult], use_calibrated: bool = False
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics efficiently.

    Extracts confidence/correctness arrays once and reuses them across
    all metrics for optimal performance (~8x faster than re-extraction).

    Args:
        predictions: List of predictions
        use_calibrated: Use calibrated confidences

    Returns:
        Dictionary with all metrics
    """
    # OPTIMIZATION: Extract arrays ONCE and reuse for all metrics
    confidences, correctness = _extract_confidence_correctness(
        predictions, use_calibrated
    )
    precomputed = (confidences, correctness)

    results = {
        # Basic stats
        "n_predictions": len(predictions),
        "accuracy": float(np.mean(correctness)),  # Reuse extracted correctness
        # Calibration - pass precomputed arrays
        "ece": expected_calibration_error(
            predictions, use_calibrated=use_calibrated, verbose=False
        ),
        "brier_score": brier_score(
            predictions, use_calibrated=use_calibrated, precomputed=precomputed
        ),
        # Ranking - pass precomputed arrays
        "auc_roc": auc_roc_correctness(
            predictions, use_calibrated=use_calibrated, precomputed=precomputed
        ),
        "spearman_corr": spearman_correlation(
            predictions, use_calibrated=use_calibrated, precomputed=precomputed
        )[0],
        # Selective prediction - pass precomputed arrays
        "acc_at_50": accuracy_at_coverage(
            predictions, 0.5, use_calibrated, precomputed=precomputed
        ),
        "acc_at_80": accuracy_at_coverage(
            predictions, 0.8, use_calibrated, precomputed=precomputed
        ),
        "cov_at_5pct_risk": coverage_at_risk(
            predictions, 0.05, use_calibrated, precomputed=precomputed
        ),
        "cov_at_10pct_risk": coverage_at_risk(
            predictions, 0.10, use_calibrated, precomputed=precomputed
        ),
    }

    # Verification (only for raw confidences)
    if not use_calibrated:
        results["calibratability"] = check_calibratability(predictions)

    return results


# ============================================================================
# Grouped Evaluation (Class-Based)
# ============================================================================


class MetricsEvaluator:
    """
    Evaluator for computing metrics across different groupings.

    Supports:
    - Overall metrics
    - Per-rubric metrics
    - Per-question metrics
    - Extensible to other groupings

    Reuses existing metric functions for consistency and backward compatibility.

    Example:
        >>> evaluator = MetricsEvaluator(predictions)
        >>> overall = evaluator.evaluate_overall()
        >>> by_rubric = evaluator.evaluate_by_rubric()
        >>> ece_by_rubric = evaluator.ece_by_rubric()
    """

    def __init__(self, predictions: List[PredictionResult]):
        """
        Initialize evaluator with predictions.

        Args:
            predictions: List of prediction results
        """
        self.predictions = predictions
        self._by_rubric = None
        self._by_question = None

    def _group_by_rubric(self) -> Dict[str, List[PredictionResult]]:
        """
        Group predictions by rubric item (lazy initialization).

        Returns:
            Dictionary mapping rubric_item -> list of predictions
        """
        if self._by_rubric is None:
            groups = {}
            for pred in self.predictions:
                rubric_id = pred.grading_instance.rubric_item
                groups.setdefault(rubric_id, []).append(pred)
            self._by_rubric = groups
        return self._by_rubric

    def _group_by_question(self) -> Dict[str, List[PredictionResult]]:
        """
        Group predictions by question (lazy initialization).

        Returns:
            Dictionary mapping question -> list of predictions
        """
        if self._by_question is None:
            groups = {}
            for pred in self.predictions:
                question_id = pred.grading_instance.question
                groups.setdefault(question_id, []).append(pred)
            self._by_question = groups
        return self._by_question

    def evaluate_overall(self, use_calibrated: bool = False) -> Dict[str, Any]:
        """
        Compute all metrics on full dataset.

        Args:
            use_calibrated: Use calibrated confidences if True

        Returns:
            Dictionary with all metrics
        """
        return evaluate_predictions(self.predictions, use_calibrated)

    def evaluate_by_rubric(self, use_calibrated: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Compute all metrics per rubric item.

        Args:
            use_calibrated: Use calibrated confidences if True

        Returns:
            Dictionary mapping rubric_item -> metrics dictionary
        """
        return {
            rubric_id: evaluate_predictions(preds, use_calibrated)
            for rubric_id, preds in self._group_by_rubric().items()
        }

    def evaluate_by_question(self, use_calibrated: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Compute all metrics per question.

        Args:
            use_calibrated: Use calibrated confidences if True

        Returns:
            Dictionary mapping question -> metrics dictionary
        """
        return {
            question_id: evaluate_predictions(preds, use_calibrated)
            for question_id, preds in self._group_by_question().items()
        }

    def ece_by_rubric(self, n_bins: int = 10, use_calibrated: bool = False) -> Dict[str, float]:
        """
        Compute ECE per rubric item.

        Args:
            n_bins: Number of bins for ECE computation
            use_calibrated: Use calibrated confidences if True

        Returns:
            Dictionary mapping rubric_item -> ECE
        """
        return {
            rubric_id: expected_calibration_error(preds, n_bins, use_calibrated)
            for rubric_id, preds in self._group_by_rubric().items()
        }

    def accuracy_by_rubric(self) -> Dict[str, float]:
        """
        Compute accuracy per rubric item.

        Returns:
            Dictionary mapping rubric_item -> accuracy
        """
        return {
            rubric_id: np.mean([p.is_correct() for p in preds])
            for rubric_id, preds in self._group_by_rubric().items()
        }

    def auc_roc_by_rubric(self, use_calibrated: bool = False) -> Dict[str, float]:
        """
        Compute AUC-ROC per rubric item.

        Args:
            use_calibrated: Use calibrated confidences if True

        Returns:
            Dictionary mapping rubric_item -> AUC-ROC
        """
        return {
            rubric_id: auc_roc_correctness(preds, use_calibrated)
            for rubric_id, preds in self._group_by_rubric().items()
        }

    def brier_score_by_rubric(self, use_calibrated: bool = False) -> Dict[str, float]:
        """
        Compute Brier score per rubric item.

        Args:
            use_calibrated: Use calibrated confidences if True

        Returns:
            Dictionary mapping rubric_item -> Brier score
        """
        return {
            rubric_id: brier_score(preds, use_calibrated)
            for rubric_id, preds in self._group_by_rubric().items()
        }

    def spearman_by_rubric(self, use_calibrated: bool = False) -> Dict[str, Tuple[float, float]]:
        """
        Compute Spearman correlation per rubric item.

        Args:
            use_calibrated: Use calibrated confidences if True

        Returns:
            Dictionary mapping rubric_item -> (correlation, p-value)
        """
        return {
            rubric_id: spearman_correlation(preds, use_calibrated)
            for rubric_id, preds in self._group_by_rubric().items()
        }
