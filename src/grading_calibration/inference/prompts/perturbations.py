"""
Perturbation strategies for robustness-based confidence estimation.

This module defines various ways to perturb grading instances while
preserving their semantic meaning and ground truth labels.
"""

from typing import List, Callable
import random
from grading_calibration.domain.models import GradingInstance


class PerturbationStrategy:
    """Base class for perturbation strategies."""

    def __init__(self, seed: int = 42):
        """
        Initialize perturbation strategy.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def apply(self, instance: GradingInstance) -> GradingInstance:
        """
        Apply perturbation to an instance.

        Args:
            instance: Original grading instance

        Returns:
            Perturbed instance (ground truth preserved)
        """
        raise NotImplementedError


class RubricParaphraseStrategy(PerturbationStrategy):
    """
    Paraphrase the rubric item while preserving meaning.

    This tests whether the model's decision is stable to different
    phrasings of the same rubric criterion.
    """

    # Simple paraphrase templates
    PARAPHRASE_TEMPLATES = [
        lambda item: f"The answer should {item.lower()}",
        lambda item: f"Response must {item.lower()}",
        lambda item: f"Student should {item.lower()}",
        lambda item: f"The response needs to {item.lower()}",
        lambda item: f"Answer includes: {item.lower()}",
    ]

    def apply(self, instance: GradingInstance) -> GradingInstance:
        """Paraphrase the rubric item."""
        if not instance.has_rubric():
            return instance  # No rubric to paraphrase

        # Randomly select a paraphrase template
        template = random.choice(self.PARAPHRASE_TEMPLATES)
        perturbed_rubric = template(instance.rubric_item)

        # Create new instance with perturbed rubric
        return GradingInstance(
            id=f"{instance.id}_paraphrase",
            question=instance.question,
            student_answer=instance.student_answer,
            rubric_item=perturbed_rubric,
            answer_key=instance.answer_key,
            ground_truth_label=instance.ground_truth_label,
            metadata={**instance.metadata, "perturbation": "rubric_paraphrase"},
        )


class SynonymReplacementStrategy(PerturbationStrategy):
    """
    Replace words in rubric item with synonyms.

    Simple synonym dictionary for common grading vocabulary.
    """

    SYNONYM_DICT = {
        "mentions": ["states", "describes", "indicates", "specifies"],
        "explains": ["describes", "clarifies", "elaborates", "details"],
        "includes": ["contains", "has", "incorporates", "provides"],
        "identifies": ["names", "specifies", "points out", "recognizes"],
        "defines": ["explains", "describes", "characterizes", "specifies"],
        "correct": ["accurate", "right", "proper", "appropriate"],
        "incorrect": ["wrong", "inaccurate", "erroneous", "mistaken"],
    }

    def apply(self, instance: GradingInstance) -> GradingInstance:
        """Replace words with synonyms in rubric item."""
        if not instance.has_rubric():
            return instance

        perturbed_rubric = instance.rubric_item

        # Try to replace one word with a synonym
        for word, synonyms in self.SYNONYM_DICT.items():
            if word in perturbed_rubric.lower():
                synonym = random.choice(synonyms)
                # Case-preserving replacement
                if word.capitalize() in perturbed_rubric:
                    perturbed_rubric = perturbed_rubric.replace(
                        word.capitalize(), synonym.capitalize(), 1
                    )
                else:
                    perturbed_rubric = perturbed_rubric.replace(word, synonym, 1)
                break  # Only replace one word

        return GradingInstance(
            id=f"{instance.id}_synonym",
            question=instance.question,
            student_answer=instance.student_answer,
            rubric_item=perturbed_rubric,
            answer_key=instance.answer_key,
            ground_truth_label=instance.ground_truth_label,
            metadata={**instance.metadata, "perturbation": "synonym_replacement"},
        )


class QuestionRephraseStrategy(PerturbationStrategy):
    """
    Rephrase the question while preserving its meaning.

    This tests robustness to different question formulations.
    """

    REPHRASE_TEMPLATES = [
        lambda q: f"Question: {q}",
        lambda q: f"The question asks: {q}",
        lambda q: f"Students were asked: {q}",
        lambda q: q,  # Keep original
    ]

    def apply(self, instance: GradingInstance) -> GradingInstance:
        """Rephrase the question."""
        template = random.choice(self.REPHRASE_TEMPLATES)
        perturbed_question = template(instance.question)

        return GradingInstance(
            id=f"{instance.id}_qrephrase",
            question=perturbed_question,
            student_answer=instance.student_answer,
            rubric_item=instance.rubric_item,
            answer_key=instance.answer_key,
            ground_truth_label=instance.ground_truth_label,
            metadata={**instance.metadata, "perturbation": "question_rephrase"},
        )


class OrderPermutationStrategy(PerturbationStrategy):
    """
    Change the order in which information is presented.

    This tests whether the model is sensitive to presentation order.
    """

    def apply(self, instance: GradingInstance) -> GradingInstance:
        """Permute the order of information."""
        # For now, this is a placeholder
        # Could swap question/answer order, etc.
        return instance


class IdentityStrategy(PerturbationStrategy):
    """
    Identity perturbation (no change).

    Useful as a control in perturbation experiments.
    """

    def apply(self, instance: GradingInstance) -> GradingInstance:
        """Return instance unchanged."""
        return GradingInstance(
            id=f"{instance.id}_identity",
            question=instance.question,
            student_answer=instance.student_answer,
            rubric_item=instance.rubric_item,
            answer_key=instance.answer_key,
            ground_truth_label=instance.ground_truth_label,
            metadata={**instance.metadata, "perturbation": "identity"},
        )


# Perturbation set generators


def generate_perturbations(
    instance: GradingInstance,
    strategies: List[PerturbationStrategy],
    n_per_strategy: int = 1,
) -> List[GradingInstance]:
    """
    Generate multiple perturbations of an instance.

    Args:
        instance: Original grading instance
        strategies: List of perturbation strategies to apply
        n_per_strategy: Number of perturbations per strategy

    Returns:
        List of perturbed instances
    """
    perturbations = []

    for strategy in strategies:
        for _ in range(n_per_strategy):
            perturbed = strategy.apply(instance)
            perturbations.append(perturbed)

    return perturbations


def get_default_perturbation_strategies(seed: int = 42) -> List[PerturbationStrategy]:
    """
    Get default set of perturbation strategies.

    Args:
        seed: Random seed

    Returns:
        List of perturbation strategy instances
    """
    return [
        IdentityStrategy(seed),
        RubricParaphraseStrategy(seed),
        SynonymReplacementStrategy(seed),
        QuestionRephraseStrategy(seed),
    ]


def create_perturbation_set(
    instance: GradingInstance, n_perturbations: int = 5, seed: int = 42
) -> List[GradingInstance]:
    """
    Convenience function to create a set of perturbations.

    Args:
        instance: Original grading instance
        n_perturbations: Total number of perturbations to generate
        seed: Random seed

    Returns:
        List of perturbed instances (including original via identity)
    """
    strategies = get_default_perturbation_strategies(seed)

    # Calculate how many per strategy
    n_per_strategy = max(1, n_perturbations // len(strategies))

    return generate_perturbations(instance, strategies, n_per_strategy)[
        :n_perturbations
    ]


# Advanced perturbation using LLMs


class LLMParaphraseStrategy(PerturbationStrategy):
    """
    Use an LLM to paraphrase rubric items.

    This produces more natural paraphrases than template-based approaches.
    NOTE: Requires an LLM inference engine.
    """

    def __init__(self, inference_engine, seed: int = 42):
        """
        Initialize LLM-based paraphrase strategy.

        Args:
            inference_engine: BatchedLLMInference instance
            seed: Random seed
        """
        super().__init__(seed)
        self.engine = inference_engine

    def apply(self, instance: GradingInstance) -> GradingInstance:
        """Paraphrase using LLM."""
        if not instance.has_rubric():
            return instance

        # Create paraphrase prompt
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant that paraphrases text while preserving meaning.",
            },
            {
                "role": "user",
                "content": f"Paraphrase the following rubric item, keeping the same meaning:\n\n{instance.rubric_item}\n\nParaphrase:",
            },
        ]

        # Generate paraphrase
        responses = self.engine.generate([conversation], temperature=0.7)
        paraphrased = responses[0].strip()

        return GradingInstance(
            id=f"{instance.id}_llm_paraphrase",
            question=instance.question,
            student_answer=instance.student_answer,
            rubric_item=paraphrased,
            answer_key=instance.answer_key,
            ground_truth_label=instance.ground_truth_label,
            metadata={**instance.metadata, "perturbation": "llm_paraphrase"},
        )


if __name__ == "__main__":
    # Example usage
    from grading_calibration.domain.models import GradingInstance

    # Create test instance
    test_instance = GradingInstance(
        id="test_1",
        question="What is photosynthesis?",
        student_answer="Plants use sunlight to make food.",
        rubric_item="Mentions that plants convert light energy",
        ground_truth_label=1,
        metadata={"question_id": 1},
    )

    print("Original instance:")
    print(f"  Question: {test_instance.question}")
    print(f"  Answer: {test_instance.student_answer}")
    print(f"  Rubric: {test_instance.rubric_item}")

    # Generate perturbations
    print("\nGenerating perturbations...")
    perturbations = create_perturbation_set(test_instance, n_perturbations=5)

    print(f"\nGenerated {len(perturbations)} perturbations:")
    for i, pert in enumerate(perturbations, 1):
        print(f"\n{i}. {pert.metadata.get('perturbation', 'unknown')}:")
        print(f"   Question: {pert.question}")
        print(f"   Rubric: {pert.rubric_item}")
