"""
Perturbation-based inference for robustness testing.

This module applies perturbations to grading instances and runs inference
on both original and perturbed versions to measure output stability.
"""

from typing import List, Dict, Any, Callable
from grading_calibration.domain.models import GradingInstance
from .prompts.perturbations import (
    PerturbationStrategy,
    create_perturbation_set,
    get_default_perturbation_strategies,
)
from .inference_llm import BatchedLLMInference


class PerturbationInference:
    """
    Runs inference on original and perturbed instances to measure stability.

    Key idea: Robust predictions should be stable under semantics-preserving
    perturbations. If small changes flip the prediction, the model is uncertain.
    """

    def __init__(
        self,
        inference_engine: BatchedLLMInference,
        perturbation_strategies: List[PerturbationStrategy] = None,
        seed: int = 42,
    ):
        """
        Initialize perturbation inference.

        Args:
            inference_engine: Base inference engine
            perturbation_strategies: List of perturbation strategies to apply
            seed: Random seed for perturbations
        """
        self.engine = inference_engine
        self.strategies = (
            perturbation_strategies or get_default_perturbation_strategies(seed)
        )
        self.seed = seed

    def generate_with_perturbations(
        self,
        instance: GradingInstance,
        conversation_fn: Callable[[GradingInstance], List[Dict]],
        n_perturbations: int = 5,
        include_original: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate predictions for original and perturbed instances.

        Args:
            instance: Original grading instance
            conversation_fn: Function to convert instance to conversation
            n_perturbations: Number of perturbations to generate
            include_original: Whether to include original instance

        Returns:
            Dictionary with:
                - original_response: Response for original instance
                - perturbed_responses: List of responses for perturbations
                - perturbations: List of perturbed instances
                - output_stability: Proportion of perturbed outputs matching original
        """
        # Generate perturbations
        perturbed_instances = create_perturbation_set(
            instance, n_perturbations=n_perturbations, seed=self.seed
        )

        # Create conversations
        all_instances = (
            [instance] + perturbed_instances
            if include_original
            else perturbed_instances
        )
        conversations = [conversation_fn(inst) for inst in all_instances]

        # Run inference
        responses = self.engine.generate(conversations, temperature=0.0)

        # Separate original and perturbed responses
        if include_original:
            original_response = responses[0]
            perturbed_responses = responses[1:]
        else:
            original_response = None
            perturbed_responses = responses

        # Compute output stability
        if original_response is not None:
            # Normalize responses for comparison
            orig_normalized = self._normalize_response(original_response)
            stability_count = sum(
                1
                for resp in perturbed_responses
                if self._normalize_response(resp) == orig_normalized
            )
            output_stability = (
                stability_count / len(perturbed_responses)
                if perturbed_responses
                else 0.0
            )
        else:
            output_stability = None

        return {
            "original_response": original_response,
            "perturbed_responses": perturbed_responses,
            "perturbations": perturbed_instances,
            "output_stability": output_stability,
            "n_perturbations": len(perturbed_instances),
        }

    def generate_with_perturbations_batch(
        self,
        instances: List[GradingInstance],
        conversation_fn: Callable[[GradingInstance], List[Dict]],
        n_perturbations: int = 5,
        include_original: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate with perturbations for a batch of instances.

        Args:
            instances: List of grading instances
            conversation_fn: Function to convert instance to conversation
            n_perturbations: Number of perturbations per instance
            include_original: Whether to include original instances

        Returns:
            List of result dictionaries
        """
        results = []

        for instance in instances:
            result = self.generate_with_perturbations(
                instance, conversation_fn, n_perturbations, include_original
            )
            results.append(result)

        return results

    def compute_representation_stability(
        self,
        instance: GradingInstance,
        conversation_fn: Callable[[GradingInstance], List[Dict]],
        representation_extractor: Callable,
        n_perturbations: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute stability in model representations (embeddings).

        Args:
            instance: Original grading instance
            conversation_fn: Function to convert instance to conversation
            representation_extractor: Function to extract representations
            n_perturbations: Number of perturbations

        Returns:
            Dictionary with representation stability metrics
        """
        # Generate perturbations
        perturbed_instances = create_perturbation_set(
            instance, n_perturbations=n_perturbations, seed=self.seed
        )

        # Get representations
        all_instances = [instance] + perturbed_instances
        representations = [
            representation_extractor(conversation_fn(inst)) for inst in all_instances
        ]

        # Compute pairwise similarities with original
        import numpy as np

        original_repr = representations[0]
        perturbed_reprs = representations[1:]

        similarities = []
        for pert_repr in perturbed_reprs:
            # Cosine similarity
            sim = np.dot(original_repr, pert_repr) / (
                np.linalg.norm(original_repr) * np.linalg.norm(pert_repr)
            )
            similarities.append(sim)

        return {
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "similarities": similarities,
            "n_perturbations": n_perturbations,
        }

    @staticmethod
    def _normalize_response(response: str) -> str:
        """
        Normalize response for comparison.

        Args:
            response: Raw response string

        Returns:
            Normalized response (lowercase, stripped)
        """
        return response.strip().lower()

    @staticmethod
    def extract_binary_answer(response: str) -> str:
        """
        Extract binary answer (yes/no, 0/1, true/false) from response.

        Args:
            response: Response text

        Returns:
            Extracted answer
        """
        response_lower = response.strip().lower()

        # Check for yes/no
        if "yes" in response_lower:
            return "yes"
        elif "no" in response_lower:
            return "no"

        # Check for 1/0
        if "1" in response_lower:
            return "1"
        elif "0" in response_lower:
            return "0"

        # Check for true/false
        if "true" in response_lower:
            return "true"
        elif "false" in response_lower:
            return "false"

        return response_lower


def compute_output_stability(
    instances: List[GradingInstance],
    inference_engine: BatchedLLMInference,
    conversation_fn: Callable[[GradingInstance], List[Dict]],
    n_perturbations: int = 5,
    extract_answer_fn: Callable[[str], str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to compute output stability for a batch.

    Args:
        instances: List of grading instances
        inference_engine: LLM inference engine
        conversation_fn: Function to create conversation from instance
        n_perturbations: Number of perturbations per instance
        extract_answer_fn: Optional function to extract answer from response

    Returns:
        List of results with stability metrics
    """
    pert_inference = PerturbationInference(inference_engine)

    results = []

    for instance in instances:
        result = pert_inference.generate_with_perturbations(
            instance, conversation_fn, n_perturbations
        )

        # Extract answers if function provided
        if extract_answer_fn:
            original_answer = extract_answer_fn(result["original_response"])
            perturbed_answers = [
                extract_answer_fn(resp) for resp in result["perturbed_responses"]
            ]

            # Recompute stability based on extracted answers
            stability = sum(
                1 for ans in perturbed_answers if ans == original_answer
            ) / len(perturbed_answers)

            result["original_answer"] = original_answer
            result["perturbed_answers"] = perturbed_answers
            result["output_stability"] = stability

        results.append(result)

    return results


if __name__ == "__main__":
    # Example usage
    from grading_calibration.domain.models import GradingInstance
    from .prompts import create_grading_conversation
    from .inference_llm import BatchedLLMInference

    # Create test instance
    test_instance = GradingInstance(
        id="test_1",
        question="What is photosynthesis?",
        student_answer="Plants use sunlight to make food.",
        rubric_item="Mentions that plants convert light energy",
        ground_truth_label=1,
    )

    # Create inference engine
    engine = BatchedLLMInference(model_name="gpt-4o-mini", backend="gpt")

    # Create conversation function
    def conv_fn(inst):
        return create_grading_conversation(
            inst, prompt_type="binary", include_rubric=True
        )

    # Test perturbation inference
    print("=" * 80)
    print("PERTURBATION INFERENCE TEST")
    print("=" * 80)

    pert_inference = PerturbationInference(engine)

    result = pert_inference.generate_with_perturbations(
        instance=test_instance, conversation_fn=conv_fn, n_perturbations=5
    )

    print(f"\nOriginal response: {result['original_response']}")
    print(f"\nPerturbed responses:")
    for i, resp in enumerate(result["perturbed_responses"], 1):
        print(f"  {i}. {resp}")

    print(f"\nOutput stability: {result['output_stability']:.2%}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
