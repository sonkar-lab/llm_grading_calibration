"""
Base calibrator interface.
"""

from abc import ABC, abstractmethod
from typing import List
from grading_calibration.domain.models import PredictionResult
import numpy as np


class BaseCalibrator(ABC):
    """
    Abstract base class for calibration methods.

    Calibrators learn a mapping from raw confidence scores to
    calibrated probabilities using a validation set.
    """

    def __init__(self):
        """Initialize calibrator."""
        self.is_fitted = False

    @abstractmethod
    def fit(self, predictions: List[PredictionResult]) -> None:
        """
        Learn calibration mapping from validation predictions.

        Args:
            predictions: List of predictions with raw confidences and ground truth
        """
        pass

    @abstractmethod
    def transform(self, predictions: List[PredictionResult]) -> List[PredictionResult]:
        """
        Apply calibration to predictions.

        Args:
            predictions: List of predictions with raw confidences

        Returns:
            List of predictions with calibrated confidences
        """
        pass

    def fit_transform(
        self,
        val_predictions: List[PredictionResult],
        test_predictions: List[PredictionResult],
    ) -> List[PredictionResult]:
        """
        Fit on validation set and transform test set.

        Args:
            val_predictions: Validation predictions for fitting
            test_predictions: Test predictions to calibrate

        Returns:
            Calibrated test predictions
        """
        self.fit(val_predictions)
        return self.transform(test_predictions)

    def _extract_arrays(self, predictions: List[PredictionResult]) -> tuple:
        """
        Extract numpy arrays from predictions.

        Returns:
            (confidences, correctness) arrays
        """
        confidences = np.array([p.raw_confidence for p in predictions])
        correctness = np.array([p.is_correct() for p in predictions], dtype=int)
        return confidences, correctness
