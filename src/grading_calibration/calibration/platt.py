"""
Platt Scaling Calibration.

Fits a logistic regression model to map confidences to calibrated probabilities.
"""

from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from grading_calibration.domain.models import PredictionResult
from .base import BaseCalibrator


class PlattScalingCalibrator(BaseCalibrator):
    """
    Platt scaling calibration.

    Fits a logistic regression: P(correct) = sigmoid(a * conf + b)

    More flexible than temperature scaling but less than isotonic.
    """

    def __init__(self):
        """Initialize Platt scaling calibrator."""
        super().__init__()
        self.model = LogisticRegression()

    def fit(self, predictions: List[PredictionResult]) -> None:
        """Fit logistic regression on validation set."""
        confidences, correctness = self._extract_arrays(predictions)

        # Reshape for sklearn
        X = confidences.reshape(-1, 1)
        y = correctness

        # Fit logistic regression
        self.model.fit(X, y)
        self.is_fitted = True

    def transform(self, predictions: List[PredictionResult]) -> List[PredictionResult]:
        """Apply Platt scaling."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")

        confidences = np.array([p.raw_confidence for p in predictions]).reshape(-1, 1)
        calibrated = self.model.predict_proba(confidences)[:, 1]

        # Create calibrated predictions
        calibrated_predictions = []
        for pred, cal_conf in zip(predictions, calibrated):
            pred.calibrated_confidence = float(cal_conf)
            calibrated_predictions.append(pred)

        return calibrated_predictions
