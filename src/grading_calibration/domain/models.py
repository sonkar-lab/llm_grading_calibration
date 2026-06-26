"""
Standard data format for grading calibration project.

This module defines the core data structures used throughout the project
to ensure consistency across all components (datasets, inference, confidence
estimation, calibration, evaluation).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class SplitType(Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass
class GradingInstance:
    """
    Represents a single grading instance for binary rubric item evaluation.

    This is the standard format used across all modules in the project.
    Every dataset adapter must convert its raw data into this format.

    Attributes:
        id: Unique identifier for this instance (e.g., "Q1_student42_item3")
        question: The question posed to the student
        student_answer: The student's response text
        rubric_item: The specific rubric criterion to evaluate
        answer_key: Optional correct answer/solution (for reference)
        ground_truth_label: True label (1 if student answer satisfies rubric item, 0 otherwise)
        metadata: Additional information (e.g., question_id, student_id, dataset_name)

    Example:
        >>> instance = GradingInstance(
        ...     id="Q1_S42_I1",
        ...     question="Explain the photoelectric effect.",
        ...     student_answer="Light causes electrons to be emitted from metal surfaces.",
        ...     rubric_item="Mentions that light is composed of photons",
        ...     answer_key="The photoelectric effect shows that light behaves as particles...",
        ...     ground_truth_label=0,  # Student didn't mention photons
        ...     metadata={"question_id": 1, "student_id": 42}
        ... )
    """

    id: str
    question: str
    student_answer: str
    rubric_item: str
    ground_truth_label: int  # 0 or 1
    answer_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the instance after initialization."""
        if self.ground_truth_label not in [0, 1]:
            raise ValueError(
                f"ground_truth_label must be 0 or 1, got {self.ground_truth_label}"
            )

        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def has_rubric(self) -> bool:
        """Check if this instance has a rubric item."""
        return bool(self.rubric_item and self.rubric_item.strip())

    def has_answer_key(self) -> bool:
        """Check if this instance has an answer key."""
        return bool(self.answer_key and self.answer_key.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "student_answer": self.student_answer,
            "rubric_item": self.rubric_item,
            "answer_key": self.answer_key,
            "ground_truth_label": self.ground_truth_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradingInstance":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            student_answer=data["student_answer"],
            rubric_item=data["rubric_item"],
            ground_truth_label=data["ground_truth_label"],
            answer_key=data.get("answer_key"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PredictionResult:
    """
    Represents the output of a confidence estimation method.

    Attributes:
        instance_id: ID of the grading instance
        prediction: Predicted label (0 or 1)
        raw_confidence: Confidence score before calibration [0, 1]
        calibrated_confidence: Confidence score after calibration [0, 1] (optional)
        ground_truth: True label (for evaluation)
        metadata: Method-specific information (e.g., token probs, sample counts)
    """

    instance_id: str
    prediction: int  # 0 or 1
    raw_confidence: float  # [0, 1]
    ground_truth: int  # 0 or 1
    calibrated_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the prediction result."""
        if self.prediction not in [0, 1]:
            raise ValueError(f"prediction must be 0 or 1, got {self.prediction}")

        if not 0 <= self.raw_confidence <= 1:
            raise ValueError(
                f"raw_confidence must be in [0, 1], got {self.raw_confidence}"
            )

        if (
            self.calibrated_confidence is not None
            and not 0 <= self.calibrated_confidence <= 1
        ):
            raise ValueError(
                f"calibrated_confidence must be in [0, 1], got {self.calibrated_confidence}"
            )

        if self.ground_truth not in [0, 1]:
            raise ValueError(f"ground_truth must be 0 or 1, got {self.ground_truth}")

    def is_correct(self) -> bool:
        """Check if prediction matches ground truth."""
        return self.prediction == self.ground_truth

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_id": self.instance_id,
            "prediction": self.prediction,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


@dataclass
class DatasetSplit:
    """
    Represents a train/valid/test split of grading instances.

    Attributes:
        train: Training instances
        valid: Validation instances (for calibration)
        test: Test instances (for final evaluation)
        metadata: Information about the split (e.g., random seed, proportions)
    """

    train: List[GradingInstance]
    valid: List[GradingInstance]
    test: List[GradingInstance]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_split(self, split_type: SplitType) -> List[GradingInstance]:
        """Get instances for a specific split."""
        if split_type == SplitType.TRAIN:
            return self.train
        elif split_type == SplitType.VALID:
            return self.valid
        elif split_type == SplitType.TEST:
            return self.test
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    def summary(self) -> Dict[str, int]:
        """Get summary statistics of the split."""
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
            "total": len(self.train) + len(self.valid) + len(self.test),
        }


# Utility functions for working with grading instances


def filter_instances(
    instances: List[GradingInstance],
    has_rubric: Optional[bool] = None,
    has_answer_key: Optional[bool] = None,
) -> List[GradingInstance]:
    """
    Filter instances based on criteria (for ablation studies).

    Args:
        instances: List of grading instances
        has_rubric: If True, keep only instances with rubric; if False, only without
        has_answer_key: If True, keep only instances with answer key; if False, only without

    Returns:
        Filtered list of instances
    """
    filtered = instances

    if has_rubric is not None:
        filtered = [inst for inst in filtered if inst.has_rubric() == has_rubric]

    if has_answer_key is not None:
        filtered = [
            inst for inst in filtered if inst.has_answer_key() == has_answer_key
        ]

    return filtered


def create_ablation_variants(
    instance: GradingInstance,
    remove_rubric: bool = False,
    remove_answer_key: bool = False,
) -> GradingInstance:
    """
    Create ablation variants of an instance by removing components.

    Args:
        instance: Original grading instance
        remove_rubric: If True, set rubric_item to empty string
        remove_answer_key: If True, set answer_key to None

    Returns:
        Modified instance for ablation study
    """
    return GradingInstance(
        id=instance.id,
        question=instance.question,
        student_answer=instance.student_answer,
        rubric_item="" if remove_rubric else instance.rubric_item,
        answer_key=None if remove_answer_key else instance.answer_key,
        ground_truth_label=instance.ground_truth_label,
        metadata={
            **instance.metadata,
            "ablation_remove_rubric": remove_rubric,
            "ablation_remove_answer_key": remove_answer_key,
        },
    )


def validate_instances(instances: List[GradingInstance]) -> None:
    """
    Validate a list of grading instances.

    Raises:
        ValueError: If instances are invalid (duplicate IDs, invalid labels, etc.)
    """
    if not instances:
        raise ValueError("Instance list is empty")

    # Check for duplicate IDs
    ids = [inst.id for inst in instances]
    if len(ids) != len(set(ids)):
        duplicates = [id for id in ids if ids.count(id) > 1]
        raise ValueError(f"Duplicate instance IDs found: {set(duplicates)}")

    # Validate each instance
    for inst in instances:
        if inst.ground_truth_label not in [0, 1]:
            raise ValueError(
                f"Invalid ground truth label for {inst.id}: {inst.ground_truth_label}"
            )


def compute_label_distribution(instances: List[GradingInstance]) -> Dict[str, Any]:
    """
    Compute label distribution statistics.

    Returns:
        Dictionary with counts and proportions of positive/negative labels
    """
    total = len(instances)
    if total == 0:
        return {"total": 0, "positive": 0, "negative": 0, "positive_rate": 0.0}

    positive = sum(1 for inst in instances if inst.ground_truth_label == 1)
    negative = total - positive

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": positive / total,
        "negative_rate": negative / total,
    }
