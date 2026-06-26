"""
Temperature Scaling Calibration.

Learns a single temperature parameter to rescale logits/confidences.
"""

from typing import List
import numpy as np
from scipy.optimize import minimize_scalar
from grading_calibration.domain.models import PredictionResult
from .base import BaseCalibrator


class TemperatureScalingCalibrator(BaseCalibrator):
    """
    Temperature scaling calibration.

    Learns a single parameter T that rescales confidences:
    calibrated_conf = sigmoid(logit(raw_conf) / T)

    Simple but effective for neural networks.
    """

    def __init__(self):
        """Initialize temperature scaling calibrator."""
        super().__init__()
        self.temperature = 1.0

    def fit(self, predictions: List[PredictionResult]) -> None:
        """Find optimal temperature on validation set."""
        confidences, correctness = self._extract_arrays(predictions)

        # Convert to logits (inverse sigmoid)
        logits = self._to_logits(confidences)

        # Find temperature that minimizes negative log-likelihood
        def nll(T):
            scaled_logits = logits / T
            scaled_probs = self._sigmoid(scaled_logits)
            # Negative log-likelihood
            return -np.sum(
                correctness * np.log(scaled_probs + 1e-9)
                + (1 - correctness) * np.log(1 - scaled_probs + 1e-9)
            )

        # Optimize temperature
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        self.is_fitted = True

    def transform(self, predictions: List[PredictionResult]) -> List[PredictionResult]:
        """Apply temperature scaling."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")

        confidences = np.array([p.raw_confidence for p in predictions])

        # Scale confidences
        logits = self._to_logits(confidences)
        scaled_logits = logits / self.temperature
        calibrated = self._sigmoid(scaled_logits)

        # Create calibrated predictions
        calibrated_predictions = []
        for pred, cal_conf in zip(predictions, calibrated):
            pred.calibrated_confidence = float(cal_conf)
            calibrated_predictions.append(pred)

        return calibrated_predictions

    @staticmethod
    def _to_logits(probs: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits."""
        probs = np.clip(probs, 1e-9, 1 - 1e-9)
        return np.log(probs / (1 - probs))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))
