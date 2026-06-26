"""
Base class for confidence estimation methods.

All confidence estimators must inherit from BaseConfidenceEstimator
and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional
from grading_calibration.domain.models import GradingInstance, PredictionResult
from grading_calibration.inference.inference_llm import BatchedLLMInference


class BaseConfidenceEstimator(ABC):
    """
    Abstract base class for confidence estimation methods.

    All confidence methods must:
    1. Take a GradingInstance as input
    2. Produce a prediction (0 or 1) and raw confidence score [0, 1]
    3. Return a PredictionResult object
    """

    def __init__(
        self,
        inference_engine: BatchedLLMInference,
        include_rubric: bool = True,
        include_answer_key: bool = True,
        seed: int = 42,
    ):
        """
        Initialize confidence estimator.

        Args:
            inference_engine: LLM inference engine
            include_rubric: Whether to include rubric items in prompts
            include_answer_key: Whether to include answer keys in prompts
            seed: Random seed for reproducibility
        """
        self.engine = inference_engine
        self.include_rubric = include_rubric
        self.include_answer_key = include_answer_key
        self.seed = seed

    @abstractmethod
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
        """
        Estimate confidence for a list of grading instances.

        Args:
            instances: List of grading instances to evaluate
            show_progress: Show progress bar
            checkpoint_callback: Optional callback(chunk_results, chunk_idx, total_chunks) after each chunk.
                                 Receives only the results from the current chunk, not accumulated results.
            checkpoint_chunk_size: Process instances in chunks of this size (for checkpointing)
            log_manager: Optional LogManager instance for logging parse errors

        Returns:
            List of PredictionResults with prediction and raw confidence
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this confidence method.

        Returns:
            Method name (e.g., "token_probability", "self_consistency")
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get configuration of this confidence method.

        Returns:
            Dictionary with method configuration
        """
        return {
            "method": self.get_name(),
            "include_rubric": self.include_rubric,
            "include_answer_key": self.include_answer_key,
            "seed": self.seed,
        }

    def _parse_binary_response(self, response: str) -> int:
        """
        Parse binary response (yes/no, 1/0, true/false) to prediction.

        Args:
            response: LLM response text

        Returns:
            Prediction (1 or 0)
        """
        response_lower = response.strip().lower()

        # Check for positive answers
        if any(
            word in response_lower
            for word in ["yes", "true", "1", "correct", "satisfies"]
        ):
            return 1

        # Check for negative answers
        if any(
            word in response_lower
            for word in ["no", "false", "0", "incorrect", "does not"]
        ):
            return 0

        # Default to 0 if unclear
        return 0

    def _create_prediction_result(
        self,
        instance: GradingInstance,
        prediction: int,
        raw_confidence: float,
        metadata: dict[str, Any] = None,
    ) -> PredictionResult:
        """
        Create a PredictionResult object.

        Args:
            instance: Original grading instance
            prediction: Predicted label (0 or 1)
            raw_confidence: Raw confidence score [0, 1]
            metadata: Additional method-specific information

        Returns:
            PredictionResult object
        """
        return PredictionResult(
            instance_id=instance.id,
            prediction=prediction,
            raw_confidence=raw_confidence,
            ground_truth=instance.ground_truth_label,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(method={self.get_name()})"


class CachedConfidenceEstimator(BaseConfidenceEstimator):
    """
    Base class for confidence estimators with caching support.

    Caches LLM outputs to avoid redundant API calls.
    """

    def __init__(
        self,
        inference_engine: BatchedLLMInference,
        include_rubric: bool = True,
        include_answer_key: bool = True,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize cached confidence estimator.

        Args:
            inference_engine: LLM inference engine
            include_rubric: Whether to include rubric items
            include_answer_key: Whether to include answer keys
            seed: Random seed
            cache_dir: Directory for caching LLM outputs (if None, no caching)
        """
        super().__init__(inference_engine, include_rubric, include_answer_key, seed)
        self.cache_dir = cache_dir
        self._cache = {}

    def _get_cache_key(self, instance: GradingInstance) -> str:
        """
        Generate cache key for an instance.

        Args:
            instance: Grading instance

        Returns:
            Cache key string
        """
        import hashlib

        # Create unique key from instance content and settings
        content = (
            f"{instance.id}|{instance.question}|{instance.student_answer}|"
            f"{instance.rubric_item}|{instance.answer_key}|"
            f"{self.include_rubric}|{self.include_answer_key}"
        )

        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)

    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add value to cache."""
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache = {}

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.cache_dir is not None,
        }
