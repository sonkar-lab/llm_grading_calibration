from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grading_calibration.pipelines.feature_pipeline import (
    build_instances_predictions_from_df,
    extract_features_dataframe,
    get_response_columns,
    load_predictions_csv,
)


FIXTURES = Path(__file__).parent / "fixtures"


def assert_frames_match_golden(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert actual.columns.tolist() == expected.columns.tolist()
    assert actual.index.tolist() == expected.index.tolist()
    assert {c: str(t) for c, t in actual.dtypes.items()} == {
        c: str(t) for c, t in expected.dtypes.items()
    }
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=True,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def test_feature_pipeline_matches_step3_golden_output() -> None:
    fixture_csv = FIXTURES / "step3_feature_pipeline_predictions.csv"
    golden_csv = FIXTURES / "step3_feature_pipeline_golden.csv"
    golden_dtypes = FIXTURES / "step3_feature_pipeline_golden_dtypes.json"
    golden_objects = FIXTURES / "step3_feature_pipeline_golden_objects.json"

    fixture_df = pd.read_csv(fixture_csv)
    response_cols = get_response_columns(fixture_df.columns.tolist())
    instances, predictions = build_instances_predictions_from_df(
        fixture_df,
        response_cols,
        include_responses=True,
        include_llm_response=True,
        show_progress=False,
    )
    actual = extract_features_dataframe(instances, predictions)
    expected = pd.read_csv(golden_csv, index_col=0)
    expected_dtypes = json.loads(golden_dtypes.read_text())
    expected_objects = json.loads(golden_objects.read_text())

    assert {c: str(t) for c, t in expected.dtypes.items()} == expected_dtypes
    assert {
        "instances": [instance.to_dict() for instance in instances],
        "predictions": [prediction.to_dict() for prediction in predictions],
    } == expected_objects
    assert_frames_match_golden(actual, expected)


def test_load_predictions_csv_legacy_defaults_preserve_step3_behavior() -> None:
    fixture_csv = FIXTURES / "step3_feature_pipeline_predictions.csv"
    instances, predictions, df = load_predictions_csv(
        fixture_csv,
        show_progress=False,
    )

    assert len(df) == 3
    assert instances[0].answer_key is None
    assert predictions[0].calibrated_confidence is None
    assert predictions[1].metadata["vote_counts"] == {"yes": 1, "no": 1}
    assert predictions[2].metadata["vote_counts"] == {"yes": 0, "no": 0}
