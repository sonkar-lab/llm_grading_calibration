"""
Baseline 2: Self-Reported Confidence Estimator.

Asks the LLM to explicitly report its confidence level (1-10 scale).
"""

from typing import Callable
from grading_calibration.domain.models import GradingInstance, PredictionResult
from grading_calibration.inference.prompts import SelfReportedConfidencePrompt
from .base import CachedConfidenceEstimator


class SelfReportedConfidence(CachedConfidenceEstimator):
    """
    Baseline 2: Ask LLM to self-report confidence.

    Prompts the model to provide both answer and confidence (1-10).
    Normalizes confidence to [0, 1].

    Hypothesis: The model's self-reported confidence correlates with accuracy.
    """

    def __init__(
        self,
        inference_engine,
        include_rubric: bool = True,
        include_answer_key: bool = True,
        seed: int = 42,
        cache_dir: str = None,
        confidence_scale: int = 10,
    ):
        """Initialize self-reported confidence estimator."""
        super().__init__(
            inference_engine, include_rubric, include_answer_key, seed, cache_dir
        )
        self.confidence_scale = confidence_scale

    def get_name(self) -> str:
        return "self_reported"

    def estimate(
        self,
        instances: list[GradingInstance],
        show_progress: bool = False,
        checkpoint_callback: (
            Callable[[list[PredictionResult], int, int], None] | None
        ) = None,
        checkpoint_chunk_size: int | None = None,
        log_manager=None,
        raw_response_saver=None,
    ) -> list[PredictionResult]:
        """Estimate using self-reported confidence.

        Args:
            instances: List of instances to process
            show_progress: Show progress bar
            checkpoint_callback: Optional callback(results_so_far, chunk_idx, total_chunks) after each chunk
            checkpoint_chunk_size: Process instances in chunks of this size (for checkpointing)
            log_manager: Optional LogManager for logging parse errors
            raw_response_saver: Optional RawResponseSaver to save responses before parsing

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

        # Process in chunks
        results = []
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(instances))
            chunk_instances = instances[start_idx:end_idx]

            # Create conversation for this chunk
            conversations = SelfReportedConfidencePrompt.create_conversation(
                chunk_instances,
                include_rubric=self.include_rubric,
                include_answer_key=self.include_answer_key,
                confidence_scale=self.confidence_scale,
            )

            # Get response
            responses = self.engine.generate(conversations, seed=self.seed)

            # CRITICAL: Save raw responses IMMEDIATELY before parsing
            if raw_response_saver is not None:
                instance_ids = [inst.id for inst in chunk_instances]
                raw_response_saver.save_responses(instance_ids, responses, chunk_idx=chunk_idx)

            # Parse responses
            results_iter = zip(chunk_instances, responses)
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
            for instance, response in results_iter:
                answer, confidence = SelfReportedConfidencePrompt.parse_response(
                    response,
                    self.confidence_scale,
                    logger=log_manager,
                    instance_id=instance.id,
                )

                prediction = 1 if answer == "yes" else 0

                result = self._create_prediction_result(
                    instance,
                    prediction,
                    confidence,
                    metadata={
                        "raw_response": response,
                        "parsed_answer": answer,
                        "method": "self_reported",
                    },
                )
                results.append(result)
                chunk_results.append(result)

            # Call checkpoint callback after each chunk with only the chunk's results
            if checkpoint_callback:
                checkpoint_callback(chunk_results, chunk_idx, total_chunks)

        return results
