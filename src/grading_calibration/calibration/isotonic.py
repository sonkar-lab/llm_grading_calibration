"""
Isotonic Regression Calibration.

Learns a monotonic mapping from raw confidences to calibrated probabilities.
"""

from typing import List
import numpy as np
from sklearn.isotonic import IsotonicRegression
from grading_calibration.domain.models import PredictionResult
from .base import BaseCalibrator


class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic regression calibration.

    Fits a monotonic function that minimizes squared error between
    confidences and empirical accuracy.

    This is the most flexible calibration method that preserves ranking.
    """

    def __init__(self):
        """Initialize isotonic calibrator."""
        super().__init__()
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, predictions: List[PredictionResult]) -> None:
        """Fit isotonic regression on validation set."""
        confidences, correctness = self._extract_arrays(predictions)

        # Fit isotonic regression
        self.calibrator.fit(confidences, correctness)
        self.is_fitted = True

    def transform(self, predictions: List[PredictionResult]) -> List[PredictionResult]:
        """Apply isotonic calibration."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")

        confidences = np.array([p.raw_confidence for p in predictions])
        calibrated = self.calibrator.predict(confidences)

        # Create new predictions with calibrated confidences
        calibrated_predictions = []
        for pred, cal_conf in zip(predictions, calibrated):
            pred.calibrated_confidence = float(cal_conf)
            calibrated_predictions.append(pred)

        return calibrated_predictions
