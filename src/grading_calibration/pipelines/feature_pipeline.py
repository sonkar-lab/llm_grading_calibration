"""Deterministic feature assembly helpers for Step 3.

This module intentionally stops at prediction loading and FeatureExtractor-based
feature construction. Downstream inference and reporting stay in the CLI layer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from analysis import FeatureExtractor
from grading_calibration.domain.models import GradingInstance, PredictionResult


def get_response_columns(columns: list[str]) -> list[str]:
    """Return sorted ``llm_response_<n>`` columns."""
    prefix = "llm_response_"
    response_cols = [
        c for c in columns if c.startswith(prefix) and c[len(prefix):].isdigit()
    ]
    return sorted(response_cols)


def build_instances_predictions_from_df(
    df: pd.DataFrame,
    response_cols: list[str],
    *,
    include_responses: bool,
    include_llm_response: bool,
    include_answer_key: bool = True,
    include_calibrated_confidence: bool = True,
    show_progress: bool = True,
) -> tuple[list[GradingInstance], list[PredictionResult]]:
    """Reconstruct domain objects from a predictions DataFrame."""
    instances: list[GradingInstance] = []
    predictions: list[PredictionResult] = []

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Loading instances", unit="row")
    for _, row in iterator:
        instance_kwargs = {}
        if include_answer_key:
            instance_kwargs["answer_key"] = row.get("answer_key", None)
        instance = GradingInstance(
            id=row["instance_id"],
            question=row.get("question", ""),
            student_answer=row.get("student_answer", ""),
            rubric_item=row.get("rubric_item", ""),
            ground_truth_label=int(row["ground_truth_label"]),
            metadata={
                "question_id": row.get("question_id", ""),
                "rubric_id": row.get("rubric_id", ""),
                "student_id": row.get("student_id", ""),
            },
            **instance_kwargs,
        )
        instances.append(instance)

        calibrated_confidence = None
        if (
            include_calibrated_confidence
            and "calibrated_confidence" in df.columns
            and pd.notna(row.get("calibrated_confidence"))
        ):
            calibrated_confidence = float(row["calibrated_confidence"])

        metadata = {
            "split": row.get("split", ""),
            "question_id": row.get("question_id", ""),
            "student_id": row.get("student_id", ""),
        }

        if include_responses:
            llm_response_value = row.get("llm_response", "") if include_llm_response else ""
            llm_responses_list = []
            for col in response_cols:
                resp = row.get(col, "")
                if resp and pd.notna(resp):
                    llm_responses_list.append(str(resp))
            if not llm_response_value and llm_responses_list:
                llm_response_value = llm_responses_list[0]
            metadata["llm_response"] = "" if llm_response_value is None else str(llm_response_value)
            metadata["llm_responses"] = llm_responses_list

        if "vote_counts" in df.columns and pd.notna(row.get("vote_counts")):
            try:
                vote_counts_str = str(row["vote_counts"])
                metadata["vote_counts"] = ast.literal_eval(vote_counts_str)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse vote_counts for {row['instance_id']}: {e}")

        prediction_kwargs = {}
        if include_calibrated_confidence:
            prediction_kwargs["calibrated_confidence"] = calibrated_confidence
        pred = PredictionResult(
            instance_id=row["instance_id"],
            prediction=int(row["prediction"]),
            raw_confidence=float(row["raw_confidence"]),
            ground_truth=int(row["ground_truth_label"]),
            metadata=metadata,
            **prediction_kwargs,
        )
        predictions.append(pred)

    return instances, predictions


def load_predictions_csv(
    csv_path: str | Path,
    smoke_n: int | None = None,
    *,
    include_answer_key: bool = False,
    include_calibrated_confidence: bool = False,
    show_progress: bool = True,
) -> tuple[list[GradingInstance], list[PredictionResult], pd.DataFrame]:
    """Load predictions CSV and reconstruct Step 3 domain objects."""
    df = pd.read_csv(csv_path)
    if smoke_n is not None:
        df = df.head(smoke_n).copy()
        print(f"Smoke test: limiting to first {len(df)} rows")
    print(f"Loaded {len(df)} rows from {csv_path}")

    response_cols = get_response_columns(df.columns.tolist())
    instances, predictions = build_instances_predictions_from_df(
        df,
        response_cols,
        include_responses=True,
        include_llm_response="llm_response" in df.columns,
        include_answer_key=include_answer_key,
        include_calibrated_confidence=include_calibrated_confidence,
        show_progress=show_progress,
    )
    return instances, predictions, df


def extract_features_dataframe(
    instances: list[GradingInstance],
    predictions: list[PredictionResult],
    *,
    extractor: Optional[FeatureExtractor] = None,
) -> pd.DataFrame:
    """Build the deterministic feature DataFrame for instances/predictions."""
    feature_extractor = extractor or FeatureExtractor()
    return feature_extractor.extract_batch(instances, predictions)
