"""
SciEntsBank dataset adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .hf_base_adapter import HuggingFaceBaseAdapter
from grading_calibration.domain.models import DatasetSplit


class SciEntsBankAdapter(HuggingFaceBaseAdapter):
    DATASET_NAME = "nkazi/SciEntsBank"

    def get_dataset_name(self) -> str:
        return "scientsbank"

    def _normalize_columns(self, df):
        df = df.copy()

        def pick(candidates, field_name):
            matches = [c for c in candidates if c in df.columns]
            if len(matches) > 1:
                print(f"Warning: multiple candidates for {field_name}: {matches}. Using {matches[0]}.")
            return matches[0] if matches else None

        col_map = {}
        id_col = pick(["id", "ID", "instance_id"], "id")
        question_col = pick(["question", "question_text", "prompt"], "question")
        ref_col = pick(["reference_answer", "reference", "reference_answer_text", "target_answer", "correct_answer"], "reference_answer")
        student_col = pick(["student_answer", "student_response", "response", "answer"], "student_answer")
        label_col = pick(["label", "gold_label", "correctness"], "label")
        if id_col and id_col != "id":
            col_map[id_col] = "id"
        if question_col and question_col != "question":
            col_map[question_col] = "question"
        if ref_col and ref_col != "reference_answer":
            col_map[ref_col] = "reference_answer"
        if student_col and student_col != "student_answer":
            col_map[student_col] = "student_answer"
        if label_col and label_col != "label":
            col_map[label_col] = "label"
        if col_map:
            print(f"Column normalization applied: {col_map}")
            df = df.rename(columns=col_map)
        return df

    def expected_dot_count(self) -> int:
        return 4

    def parse_id(self, raw_id: str) -> Dict[str, Any]:
        parts = raw_id.split(".")
        out: Dict[str, Any] = {
            "raw_id": raw_id,
            "dot_count": len(parts),
            "module_key": "",
            "question_key": "",
            "student_key": "",
            "attempt_key": "",
            "parse_ok": False,
        }
        if len(parts) == 4:
            out.update(
                {
                    "module_key": parts[0],
                    "question_key": parts[1],
                    "student_key": parts[2],
                    "attempt_key": parts[3],
                    "parse_ok": True,
                }
            )
        return out

    def _dataset_specific_diagnostics(self, df):
        if "attempt_key" in df.columns:
            nunique = int(df["attempt_key"].nunique(dropna=True))
            print(f"attempt_key nunique: {nunique}")
            if nunique <= 1:
                print("Note: attempt_key appears constant or near-constant.")
        if {"module_key", "question_key", "student_key"}.issubset(df.columns):
            grouped = df.groupby(["module_key", "question_key", "student_key"]).size()
            multi = int((grouped > 1).sum())
            print(f"Groups with multiple attempts (module,question,student): {multi}")


def load_scientsbank(
    seed: int = 42,
    train_size: float = 1.0,
    valid_size: float = 0.0,
    test_size: float = 0.0,
    save_csv: bool = True,
    save_path: Optional[str] = None,
) -> DatasetSplit:
    if not (abs(train_size - 1.0) < 1e-6 and abs(valid_size) < 1e-6 and abs(test_size) < 1e-6):
        print("Note: SciEntsBank uses a single split (all data in train).")
    adapter = SciEntsBankAdapter(
        data_dir="",
        seed=seed,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
    )
    if save_csv:
        if save_path is None:
            # Compute path relative to this file: dataset_adapter/ -> project_root/dataset/SciEntsBank/
            project_root = Path(__file__).parent.parent
            save_path = str(project_root / "dataset" / "SciEntsBank" / "scientsbank.csv")
    return adapter.load_and_split(save_path=save_path)
