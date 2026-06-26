"""
Shared adapter utilities for HuggingFace datasets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel

from .base import BaseDatasetAdapter
from grading_calibration.domain.models import GradingInstance, DatasetSplit


class HuggingFaceBaseAdapter(BaseDatasetAdapter):
    """
    Base adapter for HuggingFace datasets with common label mapping and ID handling.
    """

    DATASET_NAME: str = ""
    LABEL_MAPPING: Dict[str, int] = {
        "correct": 1,
        "contradictory": 0,
        "partially_correct_incomplete": 0,
        "irrelevant": 0,
        "non_domain": 0,
    }

    REQUIRED_COLUMNS = {"id", "question", "reference_answer", "student_answer", "label"}

    def __init__(
        self,
        data_dir: str = "",
        seed: int = 42,
        train_size: float = 1.0,
        valid_size: float = 0.0,
        test_size: float = 0.0,
    ):
        super().__init__(data_dir, seed, train_size, valid_size, test_size)

    def load_raw_data(self) -> pd.DataFrame:
        dataset = load_dataset(self.DATASET_NAME)
        if isinstance(dataset, DatasetDict):
            splits = [dataset[k] for k in dataset.keys()]
            dataset = concatenate_datasets(splits)
        
        try:
            dataset = dataset.align_labels_with_mapping(self.LABEL_MAPPING, "label")
            dataset = dataset.cast_column("label", ClassLabel(names=["incorrect", "correct"]))
        except Exception as e:
            print(f"Warning: label alignment via datasets failed ({e}); falling back to pandas mapping.")
        df = dataset.to_pandas()
        df = self._normalize_columns(df)
        self._validate_required_columns(df)

        if pd.api.types.is_numeric_dtype(df["label"]):
            df["label_raw"] = df["label"].astype(str)
            uniques = set(df["label"].dropna().unique().tolist())
            if df["label"].isna().any() or not uniques.issubset({0, 1}):
                bad = df["label_raw"].value_counts(dropna=False).to_dict()
                raise ValueError(f"Unmapped labels found with counts: {bad}")
            df["ground_truth_label"] = df["label"].astype(int)
        else:
            df["label_raw"] = df["label"].astype(str).str.strip()
            df["ground_truth_label"] = df["label_raw"].str.lower().map(self.LABEL_MAPPING)
            if df["ground_truth_label"].isna().any():
                bad = df[df["ground_truth_label"].isna()]["label_raw"].value_counts().to_dict()
                raise ValueError(f"Unmapped labels found with counts: {bad}")

        parsed = df["id"].apply(lambda x: self.parse_id(str(x)))
        parsed_df = pd.DataFrame(parsed.tolist())
        df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

        self.run_validation_diagnostics(df)
        df = self._assign_question_ids(df)
        df = self._assign_student_ids(df)
        df["instance_id"] = df["id"].astype(str)
        df["rubric_id"] = df["question_id"]
        df["rubric_item"] = df["reference_answer"].fillna("").astype(str)
        df["answer_key"] = df["reference_answer"].fillna("").astype(str)

        self._final_validation_checks(df)
        return df

    def to_grading_instances(self, raw_data: pd.DataFrame) -> List[GradingInstance]:
        instances: List[GradingInstance] = []
        for _, row in raw_data.iterrows():
            metadata = {
                "dataset_name": self.get_dataset_name(),
                "question_id": row.get("question_id", ""),
                "student_id": row.get("student_id", ""),
                "rubric_id": row.get("rubric_id", ""),
                "student_id_usable_for_re": bool(row.get("student_id_usable_for_re", False)),
                "module_key": row.get("module_key", ""),
                "question_key": row.get("question_key", ""),
                "student_key": row.get("student_key", ""),
                "session_key": row.get("session_key", ""),
                "attempt_key": row.get("attempt_key", ""),
                "student_session_key": row.get("student_session_key", ""),
                "raw_id": row.get("id", ""),
                "label_raw": row.get("label_raw", ""),
            }
            instances.append(
                GradingInstance(
                    id=str(row["instance_id"]),
                    question=str(row["question"]),
                    student_answer=str(row["student_answer"]),
                    rubric_item=str(row["rubric_item"]),
                    answer_key=str(row.get("answer_key", "")),
                    ground_truth_label=int(row["ground_truth_label"]),
                    metadata=metadata,
                )
            )
        return instances

    def load_and_split(
        self,
        unseen_question_id: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> DatasetSplit:
        if unseen_question_id is not None:
            print("Warning: unseen_question_id ignored for HF datasets (no split).")
        raw_data = None
        if save_path:
            out_path = Path(save_path)
            if out_path.exists():
                print(f"Loading cached dataset from {out_path}")
                raw_data = self._load_cached_csv(out_path)
            else:
                raw_data = self.load_raw_data()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                raw_data.to_csv(out_path, index=False)
                print(f"Wrote canonical dataset to {out_path}")
        else:
            raw_data = self.load_raw_data()
        instances = self.to_grading_instances(raw_data)
        return DatasetSplit(
            train=instances,
            valid=[],
            test=[],
            metadata={
                "seed": self.seed,
                "train_size": 1.0,
                "valid_size": 0.0,
                "test_size": 0.0,
                "split_type": "all_train",
            },
        )

    def parse_id(self, raw_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def expected_dot_count(self) -> int:
        raise NotImplementedError

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

    def _assign_question_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        base_by_index: Dict[Any, str] = {}
        for idx, row in df.iterrows():
            base_by_index[idx] = "|".join(
                [
                    self.get_dataset_name(),
                    str(row.get("module_key", "")),
                    str(row.get("question_key", "")),
                    str(row.get("question", "")),
                ]
            )
        digests = [hashlib.sha256(base_by_index[idx].encode("utf-8")).hexdigest() for idx in df.index]
        df = df.copy()
        df["question_id"] = digests

        # Extremely unlikely, but handle collisions deterministically if they occur.
        dup = df.groupby("question_id")["question_id"].transform("size") > 1
        if dup.any():
            collisions: Dict[str, int] = {}
            for qid, idxs in df[dup].groupby("question_id").groups.items():
                bases = {base_by_index[i] for i in idxs}
                if len(bases) > 1:
                    collisions[qid] = len(bases)
                    for i in idxs:
                        base = base_by_index[i]
                        df.at[i, "question_id"] = hashlib.sha256(
                            (base + "|collision").encode("utf-8")
                        ).hexdigest()
            if collisions:
                print(f"Warning: question_id hash collisions detected: {collisions}")
        return df

    def _assign_student_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        candidate = self._candidate_student_id(df)
        df = df.copy()
        df["student_id"] = candidate
        stable, stats = self.validate_student_id_stability(df["student_id"])
        usable_for_re = bool(stats.get("usable_for_re", False))
        df["student_id_usable_for_re"] = usable_for_re
        if not stable:
            print(f"Warning: missing/invalid student_id, falling back to instance_id. Stats: {stats}")
            df["student_id"] = df["id"].astype(str)
            df["student_id_usable_for_re"] = False
        elif not usable_for_re:
            print(f"Note: student_id has limited repeats; recommend disabling student RE. Stats: {stats}")
        return df

    def _candidate_student_id(self, df: pd.DataFrame) -> pd.Series:
        if "student_key" in df.columns:
            return df["student_key"].astype(str)
        return df["id"].astype(str)

    def validate_student_id_stability(self, student_ids: pd.Series) -> Tuple[bool, Dict[str, float]]:
        sid = student_ids.astype(str)
        missing = sid.isna() | (sid == "nan") | (sid == "")
        missing_rate = float(missing.mean())
        sid = sid[~missing]
        counts = sid.value_counts()
        avg = float(counts.mean()) if len(counts) else 0.0
        med = float(counts.median()) if len(counts) else 0.0
        pct_single = float((counts == 1).mean()) if len(counts) else 1.0
        stable = missing_rate <= 0.20 and len(counts) > 0
        usable_for_re = avg >= 2 and med >= 2 and pct_single <= 0.50
        return stable, {
            "missing_rate": missing_rate,
            "avg_answers_per_student": avg,
            "median_answers_per_student": med,
            "pct_single_answer": pct_single,
            "usable_for_re": usable_for_re,
        }

    def run_validation_diagnostics(self, df: pd.DataFrame) -> None:
        print("=" * 60)
        print(f"VALIDATION DIAGNOSTICS: {self.get_dataset_name()}")
        print("=" * 60)
        dot_counts = df["id"].astype(str).str.count("\\.") + 1
        counts = dot_counts.value_counts().sort_index()
        print(f"Dot-count distribution: {counts.to_dict()}")
        mismatch = dot_counts != self.expected_dot_count()
        print(f"Mismatched dot counts: {int(mismatch.sum())}")
        if mismatch.any():
            samples = df.loc[mismatch, "id"].astype(str).head(5).tolist()
            print(f"Example mismatched IDs: {samples}")
        for key in ["module_key", "question_key", "student_key", "session_key", "attempt_key"]:
            if key in df.columns:
                print(f"{key} nunique: {int(df[key].nunique(dropna=True))}")
        if "student_key" in df.columns:
            stable, stats = self.validate_student_id_stability(df["student_key"].astype(str))
            print(
                "Student ID stable: "
                f"{stable} (avg={stats['avg_answers_per_student']:.2f}, "
                f"median={stats['median_answers_per_student']:.2f}, "
                f"single={stats['pct_single_answer']:.2f}, "
                f"usable_for_re={stats['usable_for_re']})"
            )
        self._dataset_specific_diagnostics(df)

    def _dataset_specific_diagnostics(self, df: pd.DataFrame) -> None:
        return None

    def _final_validation_checks(self, df: pd.DataFrame) -> None:
        if not set(df["ground_truth_label"].unique()).issubset({0, 1}):
            raise ValueError("Labels must be binary 0/1.")
        if df["instance_id"].duplicated().any():
            raise ValueError("Duplicate instance_id detected.")
        required = ["question", "student_answer", "reference_answer"]
        for col in required:
            if df[col].isna().any():
                raise ValueError(f"Missing required field: {col}")

    def _load_cached_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {
            "instance_id",
            "question",
            "student_answer",
            "rubric_item",
            "answer_key",
            "ground_truth_label",
            "question_id",
            "student_id",
            "rubric_id",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Cached dataset missing required columns: {sorted(missing)}")
        if not set(df["ground_truth_label"].unique()).issubset({0, 1}):
            raise ValueError("Cached dataset labels must be binary 0/1.")
        return df
