"""
Baseline 1: Token Probability Confidence Estimator.

Uses the model's output probability distribution to estimate confidence.
Confidence = max(P(yes), P(no)) from the first generated token.
"""

from typing import Callable
from grading_calibration.domain.models import GradingInstance, PredictionResult
from grading_calibration.inference.prompts import create_grading_conversation
from .base import CachedConfidenceEstimator
from grading_calibration.inference.token_probs import compute_normalized_yes_no_probabilities
from utils.helpers import parse_json_response


class TokenProbabilityConfidence(CachedConfidenceEstimator):
    """
    Baseline 1: Token probability-based confidence.

    Extracts P(yes) and P(no) from the model's output distribution
    and uses max(P(yes), P(no)) as the confidence score.

    Hypothesis: When the model assigns high probability to one answer,
    it is more likely to be correct.
    """

    def __init__(
        self,
        inference_engine,
        include_rubric: bool = True,
        include_answer_key: bool = True,
        seed: int = 42,
        cache_dir: str = None,
        response_format: str = "yes_no",
        yes_tokens: list[str] = None,
        no_tokens: list[str] = None,
    ):
        """
        Initialize token probability confidence estimator.

        Args:
            inference_engine: LLM inference engine
            include_rubric: Whether to include rubric items
            include_answer_key: Whether to include answer keys
            seed: Random seed
            cache_dir: Cache directory (optional)
            response_format: "yes_no", "true_false", or "0_1"
        """
        super().__init__(
            inference_engine, include_rubric, include_answer_key, seed, cache_dir
        )
        self.response_format = response_format
        self.inference_engine = inference_engine
        # Default tokens for yes/no
        default_yes_tokens = ["yes", "Yes", "YES", "true", "True", "1"]
        default_yes_tokens += [f"_{token}" for token in default_yes_tokens]
        default_yes_tokens += [f"={token}" for token in default_yes_tokens]
        default_no_tokens = ["no", "No", "NO", "false", "False", "0"]
        default_no_tokens += [f"_{token}" for token in default_no_tokens]
        default_no_tokens += [f"={token}" for token in default_no_tokens]
        self.yes_tokens = yes_tokens or default_yes_tokens
        self.no_tokens = no_tokens or default_no_tokens

    def get_name(self) -> str:
        """Get method name."""
        return "token_probability"

    def estimate(
        self,
        instances: list[GradingInstance],
        show_progress: bool = False,
        top_k_log_probs: int = 10,
        predict_by_probability: bool = False,
        checkpoint_callback: (
            Callable[[list[PredictionResult], int, int], None] | None
        ) = None,
        checkpoint_chunk_size: int | None = None,
        log_manager=None,
        raw_response_saver=None,
    ) -> list[PredictionResult]:
        """
        Estimate confidence using token probabilities.

        Args:
            instances: List of grading instances
            show_progress: Show progress bar
            top_k_log_probs: Number of top tokens to retrieve for log probability analysis
            predict_by_probability: Use probability-based prediction instead of parsed verdict
            checkpoint_callback: Optional callback(results_so_far, chunk_idx, total_chunks) after each chunk
            checkpoint_chunk_size: Process instances in chunks of this size (for checkpointing)
            log_manager: Optional LogManager for logging parse errors
            raw_response_saver: Optional RawResponseSaver to save responses before parsing

        Returns:
            List of PredictionResults with prediction and confidence
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
            conversations = create_grading_conversation(
                chunk_instances,
                prompt_type="binary",
                include_rubric=self.include_rubric,
                include_answer_key=self.include_answer_key,
                response_format=self.response_format,
            )

            responses = self.inference_engine.generate(
                conversations, seed=self.seed, top_k_log_probs=top_k_log_probs
            )

            # CRITICAL: Save raw responses IMMEDIATELY before parsing
            # For token_probability, responses are dicts with "text" and "logprobs"
            if raw_response_saver is not None:
                instance_ids = [inst.id for inst in chunk_instances]
                text_responses = [r["text"] if isinstance(r, dict) else r for r in responses]
                raw_response_saver.save_responses(instance_ids, text_responses, chunk_idx=chunk_idx)

            # Extract token probabilities
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
                result = self._process_single_response(instance, response, predict_by_probability, log_manager)
                results.append(result)
                chunk_results.append(result)

            # Call checkpoint callback after each chunk with only the chunk's results
            if checkpoint_callback:
                checkpoint_callback(chunk_results, chunk_idx, total_chunks)

        return results

    def _process_single_response(
        self, instance: GradingInstance, response: dict, predict_by_probability: bool, log_manager=None
    ) -> PredictionResult:
        """Process a single response and create a PredictionResult."""
        text = response["text"]
        logprobs = response["logprobs"]
        parsed_json, token_indices = parse_json_response(
            text,
            fallback_dict={"verdict": "unknown"},
            log_probs=logprobs,
            logger=log_manager,
            instance_id=instance.id,
        )

        # Find the first position where a yes/no token appears
        # This handles cases where the model reasons before answering
        answer_position = token_indices.get("verdict", None)
        if answer_position is None:
            for pos_idx, token_alternatives in enumerate(logprobs[::-1]):
                # Check if any yes/no token appears in the top-k alternatives
                pred_token = token_alternatives[0]["token"].lower()
                if pred_token in self.yes_tokens + self.no_tokens:
                    answer_position = len(logprobs) - 1 - pos_idx

                if answer_position is not None:
                    break

        # Use the answer position's alternatives, or fallback to first token
        if answer_position is not None:
            target_alternatives = logprobs[answer_position]
        else:
            # Fallback: use first token if no yes/no found
            target_alternatives = logprobs[0] if logprobs else []

        yes_prob, no_prob = compute_normalized_yes_no_probabilities(
            target_alternatives,
            self.yes_tokens,
            self.no_tokens,
        )

        # Determine prediction and confidence
        if predict_by_probability:
            prediction_str = "yes" if yes_prob >= no_prob else "no"
        else:
            prediction_str = parsed_json.get("verdict", "unknown").strip().lower()
            if prediction_str not in ["yes", "no"]:
                # Fallback to probability-based prediction
                prediction_str = "yes" if yes_prob >= no_prob else "no"

        prediction = 1 if prediction_str == "yes" else 0

        # Confidence is P(no) if prediction is no, else P(yes)
        raw_confidence = no_prob.item() if prediction == 0 else yes_prob.item()

        # Create result
        result = self._create_prediction_result(
            instance,
            prediction,
            raw_confidence,
            metadata={
                "yes_prob": yes_prob.item(),
                "no_prob": no_prob.item(),
                "method": "token_probability",
                "raw_response": text,
                "parsed_answer": prediction_str,
            },
        )

        # Cache result
        cache_key = self._get_cache_key(instance)
        self._add_to_cache(cache_key, result)

        return result


if __name__ == "__main__":
    # Example usage
    from grading_calibration.domain.models import GradingInstance
    from grading_calibration.inference.inference_llm import BatchedLLMInference

    # Create test instance
    test_instance = GradingInstance(
        id="test_1",
        question="What is photosynthesis?",
        student_answer="Plants use sunlight to make food.",
        rubric_item="Mentions that plants convert light energy",
        ground_truth_label=1,
    )

    # Create inference engine
    engine = BatchedLLMInference(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        backend="vllm",
        max_model_len=20000,
    )

    # Create confidence estimator
    estimator = TokenProbabilityConfidence(engine)

    # Estimate confidence
    print("=" * 80)
    print("TOKEN PROBABILITY CONFIDENCE TEST")
    print("=" * 80)

    result = estimator.estimate([test_instance])[0]

    print(f"\nInstance: {test_instance.id}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.raw_confidence:.4f}")
    print(f"Ground truth: {result.ground_truth}")
    print(f"Correct: {result.is_correct()}")
    print(f"\nMetadata:")
    print(f"  P(yes): {result.metadata['yes_prob']:.4f}")
    print(f"  P(no): {result.metadata['no_prob']:.4f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
