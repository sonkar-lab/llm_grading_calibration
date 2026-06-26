import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grading_calibration.datasets import load_scientsbank, load_beetle


def _cached_path(dataset_name: str) -> Path:
    if dataset_name == "scientsbank":
        return PROJECT_ROOT / "dataset" / "SciEntsBank" / "scientsbank.csv"
    if dataset_name == "beetle":
        return PROJECT_ROOT / "dataset" / "Beetle" / "beetle.csv"
    return PROJECT_ROOT / "experiments" / "datasets" / f"{dataset_name}.csv"


def _skip_if_missing(csv_path: Path, dataset_name: str) -> None:
    if not csv_path.exists():
        pytest.skip(
            f"Cached {dataset_name} dataset not found at {csv_path}. "
            f"Run load_{dataset_name} to create it."
        )


def _basic_instance_checks(instance, dataset_name: str) -> None:
    assert instance.ground_truth_label in (0, 1)
    assert instance.question
    assert instance.student_answer
    assert instance.rubric_item
    assert instance.metadata.get("dataset_name") == dataset_name
    assert instance.metadata.get("question_id")
    assert instance.metadata.get("student_id")
    assert instance.metadata.get("rubric_id")


def test_scientsbank_real_dataset():
    csv_path = _cached_path("scientsbank")
    _skip_if_missing(csv_path, "scientsbank")
    splits = load_scientsbank(seed=0, save_csv=True, save_path=str(csv_path))
    assert len(splits.train) > 0
    assert len(splits.valid) == 0
    assert len(splits.test) == 0
    _basic_instance_checks(splits.train[0], "scientsbank")
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert df["ground_truth_label"].isin([0, 1]).all()
    assert df["question"].notna().all()
    assert df["student_answer"].notna().all()
    assert df["reference_answer"].notna().all()
    assert df["ground_truth_label"].mean() > 0.0
    assert df["ground_truth_label"].mean() < 1.0


def test_beetle_real_dataset():
    csv_path = _cached_path("beetle")
    _skip_if_missing(csv_path, "beetle")
    splits = load_beetle(seed=0, save_csv=True, save_path=str(csv_path))
    assert len(splits.train) > 0
    assert len(splits.valid) == 0
    assert len(splits.test) == 0
    _basic_instance_checks(splits.train[0], "beetle")
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert df["ground_truth_label"].isin([0, 1]).all()
    assert df["question"].notna().all()
    assert df["student_answer"].notna().all()
    assert df["reference_answer"].notna().all()
    assert df["ground_truth_label"].mean() > 0.0
    assert df["ground_truth_label"].mean() < 1.0
