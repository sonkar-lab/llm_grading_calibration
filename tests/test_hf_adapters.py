import sys
from pathlib import Path

import pytest
import pandas as pd

pytest.importorskip("datasets")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grading_calibration.datasets.hf_base_adapter import HuggingFaceBaseAdapter
from grading_calibration.datasets.scientsbank_adapter import SciEntsBankAdapter
from grading_calibration.datasets.beetle_adapter import BeetleAdapter


class DummyDataset:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def align_labels_with_mapping(self, mapping, column="label"):
        df = self._df.copy()
        df[column] = df[column].map(mapping)
        return DummyDataset(df)

    def cast_column(self, column, classlabel):
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self._df


class DummyAdapter(HuggingFaceBaseAdapter):
    DATASET_NAME = "dummy"

    def get_dataset_name(self) -> str:
        return "dummy"

    def parse_id(self, raw_id: str):
        return {
            "raw_id": raw_id,
            "module_key": "M",
            "question_key": "Q",
            "student_key": "S",
            "attempt_key": "",
            "parse_ok": True,
        }

    def expected_dot_count(self) -> int:
        return 1


def test_normalize_columns_scientsbank():
    df = pd.DataFrame(
        {
            "ID": ["A"],
            "question_text": ["Q"],
            "reference": ["R"],
            "student_response": ["S"],
            "gold_label": ["correct"],
        }
    )
    adapter = SciEntsBankAdapter()
    norm = adapter._normalize_columns(df)
    assert set(["id", "question", "reference_answer", "student_answer", "label"]).issubset(
        norm.columns
    )


def test_normalize_columns_beetle():
    df = pd.DataFrame(
        {
            "instance_id": ["A"],
            "prompt": ["Q"],
            "correct_answer": ["R"],
            "response": ["S"],
            "label": ["correct"],
        }
    )
    adapter = BeetleAdapter()
    norm = adapter._normalize_columns(df)
    assert set(["id", "question", "reference_answer", "student_answer", "label"]).issubset(
        norm.columns
    )


def test_label_mapping_success(monkeypatch):
    import grading_calibration.datasets.hf_base_adapter as hf_base

    df = pd.DataFrame(
        {
            "id": ["A", "B"],
            "question": ["Q1", "Q2"],
            "reference_answer": ["R1", "R2"],
            "student_answer": ["S1", "S2"],
            "label": ["correct", "irrelevant"],
        }
    )
    monkeypatch.setattr(hf_base, "load_dataset", lambda name: DummyDataset(df))
    adapter = DummyAdapter()
    out = adapter.load_raw_data()
    assert out["ground_truth_label"].tolist() == [1, 0]


def test_label_mapping_unmapped(monkeypatch):
    import grading_calibration.datasets.hf_base_adapter as hf_base

    df = pd.DataFrame(
        {
            "id": ["A"],
            "question": ["Q1"],
            "reference_answer": ["R1"],
            "student_answer": ["S1"],
            "label": ["unknown_label"],
        }
    )
    monkeypatch.setattr(hf_base, "load_dataset", lambda name: DummyDataset(df))
    adapter = DummyAdapter()
    with pytest.raises(ValueError, match="Unmapped labels found with counts"):
        adapter.load_raw_data()


def test_question_id_determinism():
    adapter = DummyAdapter()
    df1 = pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "module_key": ["M", "M", "M"],
            "question_key": ["Q1", "Q1", "Q2"],
            "question": ["Text1", "Text1", "Text2"],
        }
    )
    df2 = df1.sample(frac=1.0, random_state=7).reset_index(drop=True)
    out1 = adapter._assign_question_ids(df1)
    out2 = adapter._assign_question_ids(df2)
    map1 = dict(zip(out1["id"], out1["question_id"]))
    map2 = dict(zip(out2["id"], out2["question_id"]))
    assert map1 == map2


def test_student_id_usable_for_re_flag():
    adapter = DummyAdapter()
    df = pd.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "student_key": ["s1", "s2", "s3", "s4"],
        }
    )
    out = adapter._assign_student_ids(df)
    assert out["student_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert out["student_id_usable_for_re"].iloc[0] == False


def test_student_id_fallback_on_missing():
    adapter = DummyAdapter()
    df = pd.DataFrame(
        {
            "id": ["A", "B", "C", "D", "E"],
            "student_key": ["s1", "", None, "s2", ""],
        }
    )
    out = adapter._assign_student_ids(df)
    assert out["student_id"].tolist() == ["A", "B", "C", "D", "E"]
    assert out["student_id_usable_for_re"].iloc[0] == False
