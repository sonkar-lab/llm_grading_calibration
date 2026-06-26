"""
Method 2: Perturbation-Based Confidence.

Applies semantics-preserving perturbations and measures output stability.
"""

from typing import Callable
from grading_calibration.domain.models import GradingInstance, PredictionResult
from grading_calibration.inference.prompts import create_grading_conversation
from grading_calibration.inference.perturbation import PerturbationInference
from .base import CachedConfidenceEstimator


class PerturbationConfidence(CachedConfidenceEstimator):
    """
    Method 2: Perturbation stability-based confidence.

    Generates perturbed versions of the instance and measures
    output stability: proportion of perturbed outputs matching original.

    Hypothesis: Robust predictions are stable under perturbations.
    """

    def __init__(
        self,
        inference_engine,
        include_rubric: bool = True,
        include_answer_key: bool = True,
        seed: int = 42,
        cache_dir: str = None,
        n_perturbations: int = 5,
    ):
        """Initialize perturbation confidence estimator."""
        super().__init__(
            inference_engine, include_rubric, include_answer_key, seed, cache_dir
        )
        self.n_perturbations = n_perturbations
        self.pert_inference = PerturbationInference(inference_engine, seed=seed)

    def get_name(self) -> str:
        return f"perturbation_n{self.n_perturbations}"

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
        """Estimate using perturbation stability.

        Args:
            instances: List of instances to process
            show_progress: Show progress bar
            checkpoint_callback: Optional callback(results_so_far, chunk_idx, total_chunks) after each chunk
            checkpoint_chunk_size: Process instances in chunks of this size (for checkpointing)

        Returns:
            List of prediction results
        """
        # Determine chunk size
        chunk_size = (
            checkpoint_chunk_size
            if checkpoint_chunk_size
            else len(instances)
        )
        total_chunks = (len(instances) + chunk_size - 1) // chunk_size

        # Define conversation function
        def conv_fn(inst):
            return create_grading_conversation(
                inst,
                prompt_type="binary",
                include_rubric=self.include_rubric,
                include_answer_key=self.include_answer_key,
            )

        # Process in chunks
        results = []
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(instances))
            chunk_instances = instances[start_idx:end_idx]

            # Process each instance in chunk
            results_iter = chunk_instances
            if show_progress:
                try:
                    from tqdm import tqdm

                    results_iter = tqdm(
                        results_iter,
                        total=len(chunk_instances),
                        desc=f"Processing {self.get_name()} (chunk {chunk_idx+1}/{total_chunks})",
                    )
                except ImportError:
                    pass

            chunk_results = []
            for instance in results_iter:
                # Generate with perturbations
                result_pert = self.pert_inference.generate_with_perturbations(
                    instance,
                    conv_fn,
                    n_perturbations=self.n_perturbations,
                    include_original=True,
                )

                # Extract answers
                original_answer = self._parse_binary_response(result_pert["original_response"])

                # Use output stability as confidence
                raw_confidence = result_pert["output_stability"]

                result = self._create_prediction_result(
                    instance,
                    original_answer,
                    raw_confidence,
                    metadata={
                        "n_perturbations": self.n_perturbations,
                        "output_stability": result_pert["output_stability"],
                        "method": "perturbation",
                    },
                )

                results.append(result)
                chunk_results.append(result)

            # Call checkpoint callback after each chunk with only the chunk's results
            if checkpoint_callback:
                checkpoint_callback(chunk_results, chunk_idx, total_chunks)

        return results
