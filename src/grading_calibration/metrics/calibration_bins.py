"""
Calibration binning utilities for computing calibration curves and metrics.

This module provides a unified implementation of confidence binning that ensures
consistency between metrics computation (ECE, MCE) and visualization (calibration curves).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from grading_calibration.domain.models import PredictionResult


def _extract_confidence_correctness(
    predictions: List[PredictionResult],
    use_calibrated: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract confidence and correctness arrays from predictions.

    This is the single source of truth for extracting raw data from
    PredictionResult objects for metric computation. Using this helper
    eliminates code duplication and ensures consistent dtype handling.

    Args:
        predictions: List of PredictionResult objects
        use_calibrated: If True, use calibrated_confidence; else raw_confidence

    Returns:
        Tuple of (confidences, correctness) as numpy float arrays

    Example:
        >>> confidences, correctness = _extract_confidence_correctness(predictions)
        >>> brier = np.mean((confidences - correctness) ** 2)
    """
    confidences = np.array([
        p.calibrated_confidence if use_calibrated else p.raw_confidence
        for p in predictions
    ])
    correctness = np.array([p.is_correct() for p in predictions], dtype=float)
    return confidences, correctness


@dataclass
class CalibrationBinData:
    """
    Stores calibration curve data computed from binning predictions.

    This data structure contains all information needed to:
    - Compute calibration metrics (ECE, MCE, Brier score)
    - Plot calibration curves
    - Analyze per-bin statistics
    - Reuse raw arrays for efficient batch metric computation

    Attributes:
        bin_boundaries: Bin boundaries, shape (n_bins + 1,). E.g., [0, 0.1, 0.2, ..., 1.0]
        bin_centers: Midpoint of each bin, shape (n_bins,). E.g., [0.05, 0.15, ...]
                    Used for plotting with evenly-spaced x-axis.
        mean_confidences: Actual mean confidence within each bin, shape (n_bins,).
                         This is the TRUE x-coordinate for calibration curves.
        accuracies: Fraction of correct predictions in each bin, shape (n_bins,).
        bin_sizes: Number of predictions in each bin, shape (n_bins,).
        n_total: Total number of predictions across all bins.
        confidences: Optional raw confidence array, shape (n_total,). Stored for metric reuse.
        correctness: Optional raw correctness array, shape (n_total,). Stored for metric reuse.

    Note:
        Arrays only include non-empty bins. If a bin has no predictions, it is excluded
        from all arrays (bin_centers, mean_confidences, accuracies, bin_sizes).
    """
    bin_boundaries: np.ndarray
    bin_centers: np.ndarray
    mean_confidences: np.ndarray
    accuracies: np.ndarray
    bin_sizes: np.ndarray
    n_total: int

    # Optional: Store raw arrays for efficient metric reuse
    confidences: Optional[np.ndarray] = None
    correctness: Optional[np.ndarray] = None

    def get_ece(self) -> float:
        """
        Compute Expected Calibration Error (ECE) from bin data.

        ECE is the weighted average of absolute calibration errors across bins:
        ECE = sum_i (n_i / n_total) * |mean_confidence_i - accuracy_i|

        Returns:
            ECE value in [0, 1], where 0 is perfect calibration.
        """
        weights = self.bin_sizes / self.n_total
        cal_errors = np.abs(self.mean_confidences - self.accuracies)
        return float(np.sum(weights * cal_errors))

    def get_mce(self) -> float:
        """
        Compute Maximum Calibration Error (MCE) from bin data.

        MCE is the maximum absolute calibration error across all bins:
        MCE = max_i |mean_confidence_i - accuracy_i|

        Returns:
            MCE value in [0, 1], where 0 is perfect calibration.
        """
        cal_errors = np.abs(self.mean_confidences - self.accuracies)
        return float(np.max(cal_errors))

    def get_brier_score(self) -> float:
        """
        Compute Brier score from stored raw arrays.

        Brier score is the mean squared error between confidences and correctness:
        Brier = mean((confidence - correctness)^2)

        Returns:
            Brier score in [0, 1], where 0 is perfect.

        Raises:
            ValueError: If raw arrays were not stored (store_raw_arrays=False).
        """
        if self.confidences is None or self.correctness is None:
            raise ValueError(
                "Raw arrays not stored in CalibrationBinData. "
                "Set store_raw_arrays=True in compute_calibration_bins()."
            )
        return float(np.mean((self.confidences - self.correctness) ** 2))


def compute_calibration_bins(
    predictions: List[PredictionResult],
    n_bins: int = 10,
    use_calibrated: bool = False,
    binning_strategy: str = "equal_width",
    store_raw_arrays: bool = True
) -> CalibrationBinData:
    """
    Compute calibration curve data by binning predictions.

    This is the single source of truth for binning logic used across:
    - Metrics computation (expected_calibration_error, check_calibratability)
    - Plotting (plot_calibration_curve)
    - Analysis (mechanistic_analysis)

    Binning Strategy:
        - First bin [0, bin_width] is INCLUSIVE on both sides
        - All other bins (bin_i, bin_i+1] are EXCLUSIVE on left, INCLUSIVE on right
        - This ensures predictions with exactly 0.0 confidence are captured

    Args:
        predictions: List of PredictionResult objects to bin
        n_bins: Number of bins to create (default: 10)
        use_calibrated: If True, use calibrated_confidence; else use raw_confidence
        binning_strategy: "equal_width" for uniform bin widths, "equal_frequency"
                         for quantile-based bins
        store_raw_arrays: If True, store raw confidences/correctness arrays in bin_data
                         for efficient metric reuse (default: True)

    Returns:
        CalibrationBinData object with bin statistics

    Raises:
        ValueError: If predictions list is empty or n_bins < 1

    Example:
        >>> bin_data = compute_calibration_bins(predictions, n_bins=10, use_calibrated=False)
        >>> print(f"ECE: {bin_data.get_ece():.4f}")
        >>> print(f"Mean confidence in bin 0: {bin_data.mean_confidences[0]:.4f}")
        >>> # Reuse raw arrays for other metrics
        >>> if bin_data.confidences is not None:
        >>>     brier = bin_data.get_brier_score()
    """
    if not predictions:
        raise ValueError("Predictions list cannot be empty")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    # Extract confidence scores and correctness labels using helper function
    confidences, correctness = _extract_confidence_correctness(predictions, use_calibrated)

    # Create bin boundaries
    if binning_strategy == "equal_width":
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif binning_strategy == "equal_frequency":
        bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning_strategy: {binning_strategy}")

    # Initialize lists for non-empty bins
    bin_centers = []
    mean_confidences = []
    accuracies = []
    bin_sizes = []

    # Compute statistics per bin
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Binning rule: First bin [0, 0.1] inclusive both sides, others (0.1, 0.2] exclusive left
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        # Only include non-empty bins
        if in_bin.sum() > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            mean_confidences.append(confidences[in_bin].mean())
            accuracies.append(correctness[in_bin].mean())
            bin_sizes.append(in_bin.sum())

    return CalibrationBinData(
        bin_boundaries=bin_boundaries,
        bin_centers=np.array(bin_centers),
        mean_confidences=np.array(mean_confidences),
        accuracies=np.array(accuracies),
        bin_sizes=np.array(bin_sizes, dtype=int),
        n_total=len(predictions),
        # Optionally store raw arrays for metric reuse
        confidences=confidences if store_raw_arrays else None,
        correctness=correctness if store_raw_arrays else None
    )


def assign_bin_calibration_errors(
    predictions: List[PredictionResult],
    n_bins: int = 10,
    use_calibrated: bool = False
) -> np.ndarray:
    """
    Assign each prediction its bin's calibration error.

    This eliminates mechanical correlation with confidence by using group-level
    calibration error instead of per-instance error.

    Calibration is a group property: predictions in the same confidence bin
    should have the same calibration error (|mean_confidence - accuracy| for that bin).

    **Methodological Note**:
    Per-instance calibration error (|confidence - correctness|) mechanically correlates
    with confidence because:
    - High confidence wrong prediction → large error (0.9)
    - Low confidence wrong prediction → small error (0.1)
    - Features correlating with confidence spuriously correlate with per-instance error

    Bin-based calibration error eliminates this by assigning each prediction the
    calibration error of its group (bin), not its individual error.

    Args:
        predictions: List of PredictionResult objects
        n_bins: Number of bins for calibration computation (default: 10)
        use_calibrated: If True, use calibrated_confidence; else raw_confidence

    Returns:
        Array of shape (n_predictions,) where each element is the calibration
        error of the bin that prediction belongs to.

    Example:
        >>> bin_errors = assign_bin_calibration_errors(predictions, n_bins=10)
        >>> # Use bin_errors for feature correlation analysis
        >>> features_df['calibration_error'] = bin_errors
        >>> # This measures which features affect calibration quality,
        >>> # not which features merely correlate with confidence
    """
    # Compute bin statistics using existing infrastructure
    bin_data = compute_calibration_bins(predictions, n_bins, use_calibrated)

    # Extract confidences for bin assignment
    confidences, _ = _extract_confidence_correctness(predictions, use_calibrated)

    # Compute per-bin calibration errors (only for non-empty bins)
    per_bin_errors = np.abs(bin_data.mean_confidences - bin_data.accuracies)

    # Build mapping from bin_idx → error
    # Note: bin_data arrays only include non-empty bins, but bin_boundaries is full
    # We need to iterate through all bins and track which are non-empty
    bin_error_map = {}
    non_empty_idx = 0
    n_bins_actual = len(bin_data.bin_boundaries) - 1

    for bin_idx in range(n_bins_actual):
        bin_lower = bin_data.bin_boundaries[bin_idx]
        bin_upper = bin_data.bin_boundaries[bin_idx + 1]

        # Check if this bin has predictions
        if bin_idx == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            # This bin is non-empty, use the corresponding error
            bin_error_map[bin_idx] = per_bin_errors[non_empty_idx]
            non_empty_idx += 1

    # Assign each prediction to its bin's error
    instance_errors = np.full(len(predictions), np.nan)

    for idx, conf in enumerate(confidences):
        # Find which bin this confidence falls into
        for bin_idx in range(n_bins_actual):
            bin_lower = bin_data.bin_boundaries[bin_idx]
            bin_upper = bin_data.bin_boundaries[bin_idx + 1]

            # Use same binning rule as compute_calibration_bins
            if bin_idx == 0:
                in_bin = (conf >= bin_lower) and (conf <= bin_upper)
            else:
                in_bin = (conf > bin_lower) and (conf <= bin_upper)

            if in_bin:
                # Assign this prediction the bin's calibration error
                instance_errors[idx] = bin_error_map.get(bin_idx, np.nan)
                break

    return instance_errors
