"""
Method 3: Hybrid Confidence Estimator.

Combines multiple confidence signals (token probs, self-consistency, perturbation).
"""

from typing import Callable
import numpy as np
from grading_calibration.domain.models import GradingInstance, PredictionResult
from .base import BaseConfidenceEstimator
from .token_probability import TokenProbabilityConfidence
from .self_consistency_confidence import SelfConsistencyConfidence
from .perturbation_confidence import PerturbationConfidence


class HybridConfidence(BaseConfidenceEstimator):
    """
    Method 3: Hybrid confidence estimation.

    Combines signals from:
    1. Token probabilities (fast, but may not capture uncertainty well)
    2. Self-consistency (captures aleatoric uncertainty)
    3. Perturbation stability (captures epistemic uncertainty)

    Combination strategies:
    - "simple": Weighted average of all signals
    - "learned": Learn weights on validation set (requires training)
    """

    def __init__(
        self,
        inference_engine,
        include_rubric: bool = True,
        include_answer_key: bool = True,
        seed: int = 42,
        combination_strategy: str = "simple",
        weights: dict[str, float] = None,
        use_token_probs: bool = True,
        use_self_consistency: bool = True,
        use_perturbation: bool = True,
        n_samples: int = 10,
        n_perturbations: int = 5,
    ):
        """
        Initialize hybrid confidence estimator.

        Args:
            inference_engine: LLM inference engine
            include_rubric: Whether to include rubric
            include_answer_key: Whether to include answer key
            seed: Random seed
            combination_strategy: "simple" or "learned"
            weights: Manual weights for each signal (if simple strategy)
            use_token_probs: Whether to use token probabilities
            use_self_consistency: Whether to use self-consistency
            use_perturbation: Whether to use perturbation
            n_samples: Number of samples for self-consistency
            n_perturbations: Number of perturbations
        """
        super().__init__(inference_engine, include_rubric, include_answer_key, seed)
        self.combination_strategy = combination_strategy
        self.use_token_probs = use_token_probs
        self.use_self_consistency = use_self_consistency
        self.use_perturbation = use_perturbation

        # Initialize individual methods
        self.methods = {}

        if use_token_probs:
            self.methods["token_prob"] = TokenProbabilityConfidence(
                inference_engine, include_rubric, include_answer_key, seed
            )

        if use_self_consistency:
            self.methods["self_consistency"] = SelfConsistencyConfidence(
                inference_engine,
                include_rubric,
                include_answer_key,
                seed,
                n_samples=n_samples,
            )

        if use_perturbation:
            self.methods["perturbation"] = PerturbationConfidence(
                inference_engine,
                include_rubric,
                include_answer_key,
                seed,
                n_perturbations=n_perturbations,
            )

        # Set weights
        if weights is None:
            # Default: equal weights
            n_methods = len(self.methods)
            self.weights = {name: 1.0 / n_methods for name in self.methods}
        else:
            self.weights = weights

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def get_name(self) -> str:
        methods_used = "_".join(self.methods.keys())
        return f"hybrid_{methods_used}"

    def estimate(
        self,
        instances: list[GradingInstance],
        show_progress: bool = False,
        checkpoint_callback: (
            Callable[[list[PredictionResult], int, int], None] | None
        ) = None,
        checkpoint_chunk_size: int | None = None,
        log_manager=None,
    ) -> list[PredictionResult]:
        """Estimate using hybrid approach.

        Args:
            instances: List of instances to process
            show_progress: Show progress bar
            checkpoint_callback: Optional callback(results_so_far, chunk_idx, total_chunks) after each chunk
            checkpoint_chunk_size: Process instances in chunks of this size (for checkpointing)

        Returns:
            List of prediction results
        """
        # Collect results from each method
        method_results = {}
        for name, method in self.methods.items():
            # Pass checkpoint_callback only to the last method to avoid multiple callbacks
            method_results[name] = method.estimate(
                instances,
                show_progress=show_progress,
                checkpoint_callback=None,  # Handle checkpointing at hybrid level
                checkpoint_chunk_size=checkpoint_chunk_size,
                log_manager=log_manager,
            )

        # Combine results for each instance
        results = []
        for i, instance in enumerate(instances):
            # Collect confidence scores from each method
            confidences = {}
            predictions = {}

            for name in self.methods:
                result = method_results[name][i]
                confidences[name] = result.raw_confidence
                predictions[name] = result.prediction

            # Combine confidences
            if self.combination_strategy == "simple":
                combined_confidence = sum(
                    self.weights[name] * confidences[name] for name in self.methods
                )
            else:
                raise NotImplementedError(
                    f"Strategy {self.combination_strategy} not implemented"
                )

            # Determine final prediction (majority vote)
            pred_counts = {0: 0, 1: 0}
            for name, pred in predictions.items():
                pred_counts[pred] += self.weights[name]

            final_prediction = 1 if pred_counts[1] > pred_counts[0] else 0

            result = self._create_prediction_result(
                instance,
                final_prediction,
                combined_confidence,
                metadata={
                    "individual_confidences": confidences,
                    "individual_predictions": predictions,
                    "weights": self.weights,
                    "method": "hybrid",
                },
            )
            results.append(result)

        # Call checkpoint callback at the end (hybrid processes all at once through sub-methods)
        if checkpoint_callback:
            checkpoint_callback(results, 0, 1)

        return results
