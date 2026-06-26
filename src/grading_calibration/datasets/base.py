"""
Base dataset adapter interface.

All dataset-specific adapters should inherit from BaseDatasetAdapter
and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import random

from grading_calibration.domain.models import GradingInstance, DatasetSplit


class BaseDatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    All dataset adapters must:
    1. Load raw data from files
    2. Convert to standard GradingInstance format
    3. Create train/valid/test splits
    4. Support ablation variants (optional rubric/answer key)
    """

    def __init__(
        self,
        data_dir: str,
        seed: int = 42,
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
    ):
        """
        Initialize dataset adapter.

        Args:
            data_dir: Directory containing raw dataset files
            seed: Random seed for reproducibility
            train_size: Proportion of data for training
            valid_size: Proportion of data for validation
            test_size: Proportion of data for testing
        """
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

        # Validate split proportions
        if not abs(train_size + valid_size + test_size - 1.0) < 1e-6:
            raise ValueError(
                f"Split proportions must sum to 1.0, got {train_size + valid_size + test_size}"
            )

        # Set random seed
        random.seed(seed)

    @abstractmethod
    def load_raw_data(self) -> Any:
        """
        Load raw data from files.

        Returns:
            Raw data in dataset-specific format

        Raises:
            FileNotFoundError: If data files not found
        """
        pass

    @abstractmethod
    def to_grading_instances(self, raw_data: Any) -> List[GradingInstance]:
        """
        Convert raw data to standardized GradingInstance format.

        Args:
            raw_data: Raw data loaded by load_raw_data()

        Returns:
            List of GradingInstance objects
        """
        pass

    def create_splits(
        self, instances: List[GradingInstance], stratify_by_question: bool = False
    ) -> DatasetSplit:
        """
        Create train/valid/test splits.

        Args:
            instances: List of grading instances
            stratify_by_question: If True, ensure each question appears in all splits

        Returns:
            DatasetSplit object with train/valid/test instances
        """
        # Shuffle instances
        shuffled = instances.copy()
        random.shuffle(shuffled)

        # Calculate split sizes
        total = len(shuffled)
        train_end = int(total * self.train_size)
        valid_end = train_end + int(total * self.valid_size)

        # Split data
        train = shuffled[:train_end]
        valid = shuffled[train_end:valid_end]
        test = shuffled[valid_end:]

        return DatasetSplit(
            train=train,
            valid=valid,
            test=test,
            metadata={
                "seed": self.seed,
                "train_size": self.train_size,
                "valid_size": self.valid_size,
                "test_size": self.test_size,
                "stratify_by_question": stratify_by_question,
            },
        )

    def create_unseen_question_split(
        self, instances: List[GradingInstance], test_question_id: int
    ) -> DatasetSplit:
        """
        Create splits where one question is completely held out for testing.

        This is for evaluating generalization to unseen questions.

        Args:
            instances: List of grading instances
            test_question_id: ID of question to use only for testing

        Returns:
            DatasetSplit with test containing only test_question_id instances
        """
        # Separate instances by question
        test_instances = []
        other_instances = []

        for inst in instances:
            question_id = inst.metadata.get("question_id")
            if question_id == test_question_id:
                test_instances.append(inst)
            else:
                other_instances.append(inst)

        # Shuffle other instances and split into train/valid
        random.shuffle(other_instances)

        # Calculate split for non-test instances
        # We want train_size and valid_size to be relative to non-test data
        total_non_test = len(other_instances)
        train_ratio = self.train_size / (self.train_size + self.valid_size)
        train_end = int(total_non_test * train_ratio)

        train = other_instances[:train_end]
        valid = other_instances[train_end:]

        return DatasetSplit(
            train=train,
            valid=valid,
            test=test_instances,
            metadata={
                "seed": self.seed,
                "split_type": "unseen_question",
                "test_question_id": test_question_id,
                "train_size": len(train),
                "valid_size": len(valid),
                "test_size": len(test_instances),
            },
        )

    def load_and_split(self, unseen_question_id: Optional[int] = None) -> DatasetSplit:
        """
        Convenience method to load data and create splits in one call.

        Args:
            unseen_question_id: If provided, create unseen question split

        Returns:
            DatasetSplit object
        """
        # Load raw data
        raw_data = self.load_raw_data()

        # Convert to grading instances
        instances = self.to_grading_instances(raw_data)

        # Create splits
        if unseen_question_id is not None:
            return self.create_unseen_question_split(instances, unseen_question_id)
        else:
            return self.create_splits(instances)

    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Get the name of this dataset.

        Returns:
            Dataset name (e.g., "ricechem", "asap")
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this dataset.

        Returns:
            Dictionary with dataset information
        """
        return {
            "name": self.get_dataset_name(),
            "data_dir": str(self.data_dir),
            "seed": self.seed,
            "train_size": self.train_size,
            "valid_size": self.valid_size,
            "test_size": self.test_size,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.get_dataset_name()}, "
            f"data_dir={self.data_dir}, "
            f"splits={self.train_size}/{self.valid_size}/{self.test_size})"
        )
