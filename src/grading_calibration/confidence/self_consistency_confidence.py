"""
Method 1: Self-Consistency Based Confidence.

Samples multiple responses with temperature and uses agreement as confidence.
"""

from typing import Callable
from grading_calibration.domain.models import GradingInstance, PredictionResult
from grading_calibration.inference.prompts import create_grading_conversation
from grading_calibration.inference.self_consistency import SelfConsistencySampler
from .base import CachedConfidenceEstimator
from utils.helpers import parse_json_response


class SelfConsistencyConfidence(CachedConfidenceEstimator):
    """
    Method 1: Self-consistency based confidence.

    Samples n responses with temperature and computes agreement.
    Confidence = proportion agreeing with majority answer.

    Hypothesis: Consistent outputs across samples indicate certainty.
    """

    def __init__(
        self,
        inference_engine,
        include_rubric: bool = True,
        include_answer_key: bool = True,
        seed: int = 42,
        cache_dir: str = None,
        n_samples: int = 10,
    ):
        """Initialize self-consistency confidence estimator."""
        super().__init__(
            inference_engine, include_rubric, include_answer_key, seed, cache_dir
        )
        self.n_samples = n_samples
        self.sampler = SelfConsistencySampler(inference_engine)
        self.inference_engine = inference_engine

    def get_name(self) -> str:
        return f"self_consistency_n{self.n_samples}"

    def estimate(
        self,
        instances: list[GradingInstance],
        show_progress: bool = False,
        renormalize: bool = False,
        checkpoint_callback: (
            Callable[[list[PredictionResult], int, int], None] | None
        ) = None,
        checkpoint_chunk_size: int | None = None,
        log_manager=None,
        raw_response_saver=None,
    ) -> list[PredictionResult]:
        """Estimate using self-consistency.

        Args:
            instances: List of instances to process
            show_progress: Show progress bar
            renormalize: Renormalize confidence scores
            checkpoint_callback: Optional callback(results_so_far, chunk_idx, total_chunks) after each chunk
            checkpoint_chunk_size: Process instances in chunks of this size (for checkpointing)
            log_manager: Optional LogManager for logging parse errors
            raw_response_saver: Optional RawResponseSaver to save responses before parsing

        Returns:
            List of prediction results
        """

        # Determine chunk size (based on original instances, not duplicated)
        chunk_size = (
            checkpoint_chunk_size
            if checkpoint_chunk_size
            else len(instances)
        )
        total_chunks = (len(instances) + chunk_size - 1) // chunk_size

        # Process in chunks
        results = []
        print(f"Processing {len(instances)} instances in {total_chunks} chunks...")
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(instances))
            chunk_instances = instances[start_idx:end_idx]

            # Duplicate this chunk for n_samples (INSIDE the loop!)
            duplicated_chunk = [instance for _ in range(self.n_samples) for instance in chunk_instances]

            # Create conversations for duplicated chunk
            conversations = create_grading_conversation(
                duplicated_chunk,
                prompt_type="chain_of_thought",
                include_rubric=self.include_rubric,
                include_answer_key=self.include_answer_key,
            )

            # Generate responses for all n_samples of this chunk
            responses = self.inference_engine.generate(
                conversations, seed=self.seed
            )

            # CRITICAL: Save raw responses IMMEDIATELY before parsing
            # This ensures responses are preserved even if parsing crashes
            if raw_response_saver is not None:
                instance_ids = [inst.id for inst in chunk_instances]
                raw_response_saver.save_responses(instance_ids, responses, chunk_idx=chunk_idx)

            # Aggregate results for this chunk
            # Use original chunk size (before duplication) for stride
            n_original_instances = len(chunk_instances)
            result_scs = []
            for i in range(n_original_instances):
                instance = chunk_instances[i]
                # Create lambda that captures instance_id and logger
                def make_extract_fn(inst_id, lgr):
                    return lambda r: parse_json_response(
                        r,
                        fallback_dict={"verdict": "unknown"},
                        logger=lgr,
                        instance_id=inst_id,
                    )

                result_sc = self.sampler.sample_and_aggregate(
                    responses[i::n_original_instances],
                    extract_answer_fn=make_extract_fn(instance.id, log_manager),
                    renormalize=renormalize,
                )
                result_scs.append(result_sc)

            # Create prediction results for this chunk
            results_iter = zip(chunk_instances, result_scs)
            if show_progress:
                try:
                    from tqdm import tqdm

                    results_iter = tqdm(
                        results_iter,
                        total=n_original_instances,
                        desc=f"Processing {self.get_name()} (chunk {chunk_idx+1}/{total_chunks})",
                    )
                except ImportError:
                    pass

            chunk_results = []
            for instance, result_sc in results_iter:
                majority_answer = result_sc["majority_answer"]
                raw_confidence = result_sc["confidence"]

                # Handle ties: if confidence is exactly 0.5, randomly choose prediction
                is_tie = result_sc.get("is_tie", False)
                if is_tie or raw_confidence == 0.5:
                    # Use instance-specific seed for reproducibility
                    import random
                    instance_seed = self.seed + hash(instance.id) % 1000000
                    random.seed(instance_seed)
                    prediction = random.choice([0, 1])
                else:
                    prediction = 1 if majority_answer == "yes" else 0

                result = self._create_prediction_result(
                    instance,
                    prediction,
                    raw_confidence,
                    metadata={
                        "n_samples": self.n_samples,
                        "vote_counts": result_sc["vote_counts"],
                        "method": "self_consistency",
                        "parsed_answer": majority_answer,
                        "is_tie": is_tie,
                        **{
                            f"raw_response_{i}": result_sc["all_responses"][i]
                            for i in range(len(result_sc["all_responses"]))
                        },
                    },
                )
                results.append(result)
                chunk_results.append(result)

            # Call checkpoint callback after each chunk with only the chunk's results
            if checkpoint_callback:
                checkpoint_callback(chunk_results, chunk_idx, total_chunks)


        return results
