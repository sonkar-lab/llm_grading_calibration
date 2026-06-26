"""
RiceChem dataset adapter.

Adapts the RiceChem automated long answer grading dataset to the
standardized GradingInstance format.

Dataset structure:
- Student Answers Q1-4.csv: Contains student responses
- Graded Rubric Q1-4.csv: Contains rubric item evaluations (TRUE/FALSE)
"""

import csv
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .base import BaseDatasetAdapter
from grading_calibration.domain.models import GradingInstance


class RiceChemAdapter(BaseDatasetAdapter):
    """
    Dataset adapter for RiceChem automated long answer grading dataset.

    Expected directory structure:
        data_dir/
            Student Answers Q1.csv
            Student Answers Q2.csv
            Student Answers Q3.csv
            Student Answers Q4.csv
            Graded Rubric Q1.csv
            Graded Rubric Q2.csv
            Graded Rubric Q3.csv
            Graded Rubric Q4.csv
    """

    # Invalid rubric items to skip (from original implementation)
    INVALID_ITEMS = {
        "score",
        "adjustment",
        "comments",
        "blank",
        "incorrect",
        "core charge calculation error",
        "incorrect statement included",
        "correct response",
        "incorrect/blank response",
        "incorrect/missing answer",
        "incorrect/misleading statement",
    }

    def __init__(
        self,
        data_dir: str,
        seed: int = 42,
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        num_questions: int = 4,
    ):
        """
        Initialize RiceChem adapter.

        Args:
            data_dir: Directory containing RiceChem CSV files
            seed: Random seed for reproducibility
            train_size: Proportion of data for training
            valid_size: Proportion of data for validation
            test_size: Proportion of data for testing
            num_questions: Number of questions in the dataset (default: 4)
        """
        super().__init__(data_dir, seed, train_size, valid_size, test_size)
        self.num_questions = num_questions

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return "ricechem"

    def load_raw_data(
        self,
    ) -> Dict[int, Tuple[Dict[str, str], List[Tuple[str, List[str], List[str]]]]]:
        """
        Load raw data from CSV files.

        Returns:
            Dictionary mapping question_id to (id2ans, rubric_data, question_text)
            where:
                - id2ans: maps student ID to answer text
                - rubric_data: list of (student_id, rubric_items, values) tuples
                - question_text: text of the question
        """
        raw_data = {}
        rubric_idx = {"idx": 1}
        for q_id in range(1, self.num_questions + 1):
            # Load student answers
            answer_file = self.data_dir / f"Student Answers Q{q_id}.csv"
            id2ans = self._load_answers(answer_file)

            # Load rubric evaluations
            rubric_file = self.data_dir / f"Graded Rubric Q{q_id}.csv"
            rubric_data = self._load_rubric_evaluations(rubric_file)

            rubric_item_2rubric_id = self._rubric_item_to_rubric_id(
                rubric_data, rubric_idx
            )

            # Get question text from answer file
            question_text = self._extract_question_text(answer_file)

            raw_data[q_id] = {
                "id2ans": id2ans,
                "rubric_data": rubric_data,
                "question_text": question_text,
                "rubric_item_2rubric_id": rubric_item_2rubric_id,
            }

        return raw_data

    def _rubric_item_to_rubric_id(
        self,
        rubric_data: List[Tuple[str, List[str], List[str]]],
        rubric_idx: dict = {"idx": 1},
    ) -> Dict[str, str]:
        """
        Create mapping from rubric items to rubric IDs.
        """
        rubric_item_2rubric_id = {}

        for _, rubric_items, _ in rubric_data:
            for idx, item in enumerate(rubric_items):
                if item not in rubric_item_2rubric_id:
                    rubric_item_2rubric_id[item] = f"R{rubric_idx['idx']}"
                    rubric_idx["idx"] += 1

        return rubric_item_2rubric_id

    def _load_answers(self, answer_file: Path) -> Dict[str, str]:
        """
        Load student answers from CSV file.

        Args:
            answer_file: Path to student answers CSV

        Returns:
            Dictionary mapping student ID to answer text
        """
        id2ans = {}

        with open(answer_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 2:
                    continue

                student_id = row[0].strip()
                answer = row[1].strip()

                # Skip blank answers
                if not answer:
                    continue

                id2ans[student_id] = answer

        return id2ans

    def _extract_question_text(self, answer_file: Path) -> str:
        """
        Extract question text from answer file header.

        The question text is in the header of the answer CSV.

        Args:
            answer_file: Path to student answers CSV

        Returns:
            Question text
        """
        with open(answer_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            if len(header) < 2:
                return f"Question from {answer_file.name}"

            # Question is in the second column header
            # Format: "Question: <text>"
            question_field = header[1]
            if ":" in question_field:
                question = question_field[question_field.find(":") + 1 :].strip()
            else:
                question = question_field.strip()

            return question

    def _load_rubric_evaluations(
        self, rubric_file: Path
    ) -> List[Tuple[str, List[str], List[str]]]:
        """
        Load rubric evaluations from CSV file.

        Args:
            rubric_file: Path to graded rubric CSV

        Returns:
            List of (student_id, rubric_items, values) tuples
            where values are "TRUE" or "FALSE"
        """
        evaluations = []

        with open(rubric_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            # Get rubric items from header
            header = next(reader)[1:]  # Skip student ID column

            # Filter out invalid rubric items
            valid_indices = []
            valid_items = []
            for i, item in enumerate(header):
                if item.lower().strip() not in self.INVALID_ITEMS:
                    valid_indices.append(i)
                    valid_items.append(item.strip())

            # Read student evaluations
            for row in reader:
                if len(row) < 2:
                    continue

                student_id = row[0].strip()
                values = row[1:]

                # Extract only valid rubric item values
                valid_values = [values[i] for i in valid_indices if i < len(values)]

                evaluations.append((student_id, valid_items, valid_values))

        return evaluations

    def to_grading_instances(
        self, raw_data: Dict[int, Dict[str, Any]]
    ) -> List[GradingInstance]:
        """
        Convert raw RiceChem data to GradingInstance format.

        Args:
            raw_data: Raw data from load_raw_data()

        Returns:
            List of GradingInstance objects
        """
        instances = []

        for q_id, data in raw_data.items():
            id2ans = data["id2ans"]
            rubric_data = data["rubric_data"]
            question_text = data["question_text"]
            rubric_item_2rubric_id = data["rubric_item_2rubric_id"]

            # Process each student's rubric evaluations
            for student_id, rubric_items, values in rubric_data:
                # Skip if no answer for this student
                if student_id not in id2ans:
                    continue

                student_answer = id2ans[student_id]

                # Create one instance per rubric item
                for rubric_item, value in zip(rubric_items, values):
                    # Skip empty rubric items
                    if not rubric_item.strip():
                        continue

                    # Convert TRUE/FALSE to 1/0
                    label = 1 if value == "TRUE" else 0

                    # Get rubric ID for this rubric item
                    rubric_id = rubric_item_2rubric_id.get(rubric_item, "unknown")

                    # Create unique instance ID using question, student, and rubric IDs
                    instance_id = f"Q{q_id}_S{student_id}_{rubric_id}"

                    # Create grading instance
                    instance = GradingInstance(
                        id=instance_id,
                        question=question_text,
                        student_answer=student_answer,
                        rubric_item=rubric_item,
                        ground_truth_label=label,
                        answer_key=None,  # RiceChem doesn't provide answer keys in CSV
                        metadata={
                            "question_id": q_id,
                            "student_id": student_id,
                            "dataset": "ricechem",
                            "rubric_id": rubric_id,
                        },
                    )

                    instances.append(instance)

        return instances

    def load_with_questions_metadata(
        self,
    ) -> Tuple[List[GradingInstance], Dict[int, Dict]]:
        """
        Load data and also return question metadata.

        Returns:
            Tuple of (instances, question_metadata)
            where question_metadata maps question_id to question info
        """
        raw_data = self.load_raw_data()
        instances = self.to_grading_instances(raw_data)

        # Extract question metadata
        question_metadata = {}
        for q_id, data in raw_data.items():
            # Get unique rubric items for this question
            rubric_items = set()
            for _, items, _ in data["rubric_data"]:
                rubric_items.update(items)

            question_metadata[q_id] = {
                "question_id": q_id,
                "question_text": data["question_text"],
                "rubric_items": sorted(list(rubric_items)),
            }

        return instances, question_metadata


# Convenience function for quick loading
def load_ricechem(
    data_dir: str,
    seed: int = 42,
    train_size: float = 0.6,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    unseen_question_id: int = None,
):
    """
    Convenience function to load RiceChem dataset.

    Args:
        data_dir: Directory containing RiceChem CSV files
        seed: Random seed
        train_size: Training set proportion
        valid_size: Validation set proportion
        test_size: Test set proportion
        unseen_question_id: If provided, use this question only for testing

    Returns:
        DatasetSplit object with train/valid/test instances
    """
    adapter = RiceChemAdapter(
        data_dir=data_dir,
        seed=seed,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
    )

    return adapter.load_and_split(unseen_question_id=unseen_question_id)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing RiceChem CSV files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--unseen_question",
        type=int,
        default=None,
        help="Question ID to use only for testing (1-4)",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading RiceChem dataset from {args.data_dir}...")
    split = load_ricechem(
        data_dir=args.data_dir,
        seed=args.seed,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        unseen_question_id=args.unseen_question,
    )

    # Print summary
    print("\nDataset Summary:")
    print(f"  Train: {len(split.train)} instances")
    print(f"  Valid: {len(split.valid)} instances")
    print(f"  Test: {len(split.test)} instances")
    print(f"  Total: {len(split.train) + len(split.valid) + len(split.test)} instances")

    # Show example instance
    if split.train:
        print("\nExample training instance:")
        example = split.train[0]
        print(f"  ID: {example.id}")
        print(f"  Question: {example.question[:100]}...")
        print(f"  Student Answer: {example.student_answer[:100]}...")
        print(f"  Rubric Item: {example.rubric_item}")
        print(f"  Ground Truth: {example.ground_truth_label}")
        print(f"  Metadata: {example.metadata}")
