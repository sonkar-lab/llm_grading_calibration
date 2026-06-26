"""
Self-consistency sampling for LLMs.

This module implements self-consistency prompting where we sample multiple
responses with temperature and use agreement as a measure of confidence.
"""

from typing import List, Dict, Any
from collections import Counter
import random
from .inference_llm import BatchedLLMInference


class SelfConsistencySampler:
    """
    Generates multiple responses for the same prompt and measures agreement.

    Key idea: If the model is confident, it should produce consistent answers
    even with stochastic sampling. Disagreement indicates uncertainty.
    """

    def __init__(self, inference_engine: BatchedLLMInference):
        """
        Initialize self-consistency sampler.

        Args:
            inference_engine: Base inference engine
        """
        self.engine = inference_engine

    def compute_self_consistency(
        self,
        responses: List[str],
        extract_answer_fn: callable = None,
        renormalize: bool = False,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Compute self-consistency metrics from multiple responses.

        Args:
            responses: List of response strings
            extract_answer_fn: Optional function to extract answer from response
                               (e.g., extract "yes"/"no" from longer text)
                               If None, uses the full response text
            renormalize: Whether to renormalize confidence to [0.0, 1.0]
            seed: Random seed for tie-breaking when confidence is exactly 0.5

        Returns:
            Dictionary with:
                - majority_answer: Most common answer
                - confidence: Proportion agreeing with majority
                - vote_counts: Count of each unique answer
                - all_responses: Original responses
                - is_tie: Whether this was a tie (confidence = 0.5)
        """

        # Extract answers
        if extract_answer_fn is not None:
            try:
                answers = [extract_answer_fn(r)["verdict"] for r in responses]
            except Exception as e:
                print(f"Error extracting answers with provided function: {e}")
                for r in responses:
                    try:
                        ans = extract_answer_fn(r)
                        answer = ans["verdict"]
                    except Exception as e_inner:
                        print(f"ans is: {ans}")
                        print(f"Error extracting answer from response: {r}\nError: {e_inner}")
            # for i, ans in enumerate(answers):
            #     print(f"Extracted answer {i+1}: {ans}\n\n")
        else:
            # Use full responses
            answers = [r.strip().lower() for r in responses]

            # Count votes
        answers = [extract_answer_fn(r)["verdict"] for r in responses]
        vote_counts = Counter(answers)
        total_votes = len(answers)

        # Find majority answer
        # for ans, count in vote_counts.items():
        #     print(f"Answer: {ans} | Count: {count}\n")
        # input("Press Enter to continue...")
        majority_answer, majority_count = vote_counts.most_common(1)[0]

        # Confidence is proportion agreeing with majority
        confidence = majority_count / total_votes

        # Check if this is a tie (exactly 50/50 split)
        is_tie = (confidence == 0.5)

        # For ties, we need to randomly select the majority answer for consistency
        if is_tie:
            # Get the two tied answers
            tied_answers = [ans for ans, count in vote_counts.most_common(2)]
            if len(tied_answers) == 2:
                # Use seed to deterministically pick one
                random.seed(seed)
                majority_answer = random.choice(tied_answers)

        if renormalize:
            # Renormalize confidence to [0.0, 1.0]
            confidence = 2 * confidence - 1.0

        return {
            "majority_answer": majority_answer,
            "confidence": confidence,
            "majority_count": majority_count,
            "total_samples": total_votes,
            "vote_counts": dict(vote_counts),
            "all_responses": responses,
            "is_tie": is_tie,
        }

    def sample_and_aggregate(
        self,
        responses: list[str],
        extract_answer_fn: callable = None,
        renormalize: bool = False,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Convenience method: sample multiple times and compute self-consistency.

        Args:
            responses: List of response strings
            extract_answer_fn: Function to extract answer from response
            renormalize: Whether to renormalize confidence to [0.0, 1.0]
            seed: Random seed for tie-breaking

        Returns:
            Self-consistency result dictionary

        """
        return self.compute_self_consistency(responses, extract_answer_fn, renormalize, seed)


def create_self_consistency_sampler(
    model_name: str, backend: str = "gpt", **inference_kwargs
) -> SelfConsistencySampler:
    """
    Convenience function to create self-consistency sampler.

    Args:
        model_name: Model name
        backend: "gpt", "hf", or "vllm"
        **inference_kwargs: Additional arguments for BatchedLLMInference

    Returns:
        SelfConsistencySampler instance
    """
    engine = BatchedLLMInference(
        model_name=model_name, backend=backend, **inference_kwargs
    )

    return SelfConsistencySampler(engine)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="gpt")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("=" * 80)
    print("SELF-CONSISTENCY SAMPLING TEST")
    print("=" * 80)

    # Create sampler
    sampler = create_self_consistency_sampler(
        model_name=args.model, backend=args.backend
    )

    # Test conversation
    test_conversation = [
        {
            "role": "system",
            "content": "You are a grading assistant. Answer only 'yes' or 'no'.",
        },
        {
            "role": "user",
            "content": (
                "Question: What is photosynthesis?\n"
                "Student answer: Plants use sunlight to make food.\n"
                "Rubric item: Mentions that plants convert sunlight to energy.\n\n"
                "Does the student answer satisfy this rubric item? Answer 'yes' or 'no'."
            ),
        },
    ]

    print(
        f"\nSampling {args.n_samples} responses with temperature {args.temperature}..."
    )
    responses = sampler.engine.generate(
        [test_conversation] * args.n_samples, temperature=args.temperature
    )
    result = sampler.sample_and_aggregate(responses, extract_answer_fn=lambda r: {"verdict": r.strip().lower()})

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"Majority answer: {result['majority_answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Vote counts: {result['vote_counts']}")
    print(f"\nAll responses:")
    for i, response in enumerate(result["all_responses"], 1):
        print(f"  {i}. {response[:100]}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
