from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import FeatureExtractor
from experiments.run_exp import step5_commonality_analysis as step5
from grading_calibration.pipelines.commonality_pipeline import run_commonality_comparisons


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PREDICTIONS_CSV = (
    FIXTURE_DIR
    / "phase4b_predictions"
    / "self_reported"
    / "predictions_gpt_oss_20b_self_reported_20251128-045340.csv"
)


def _args() -> Namespace:
    return Namespace(
        adaptive_perm=True,
        adaptive_perm_band=(0.5, 2.0),
        adaptive_perm_max=20,
        alpha=0.05,
        bootstrap_method="percentile",
        exact_power=False,
        high_threshold=0.9,
        low_threshold=0.8,
        n_boot=20,
        n_perm=20,
        power_perm=500,
        power_sim=200,
    )


def _features_df() -> pd.DataFrame:
    instances, predictions, _ = step5.load_data_from_csv(PREDICTIONS_CSV)
    return FeatureExtractor().extract_batch(instances, predictions)


def _assert_frame_matches(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, float]:
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)

    deltas: dict[str, float] = {}
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]) and pd.api.types.is_numeric_dtype(actual[col]):
            actual_values = actual[col].to_numpy(dtype=float)
            expected_values = expected[col].to_numpy(dtype=float)
            assert np.allclose(actual_values, expected_values, atol=1e-8, rtol=0.0, equal_nan=True), col
            finite = np.isfinite(actual_values) & np.isfinite(expected_values)
            deltas[col] = float(np.max(np.abs(actual_values[finite] - expected_values[finite]))) if finite.any() else 0.0
        else:
            actual_values = actual[col].fillna("").astype(str).tolist()
            expected_values = expected[col].fillna("").astype(str).tolist()
            assert actual_values == expected_values, col
    return deltas


def test_commonality_pipeline_matches_phase4d_golden():
    features_df = _features_df()
    args = _args()
    rng = np.random.default_rng(4242)
    student_ids = features_df["student_id"].to_numpy(dtype=str)
    question_ids = features_df["question_id"].to_numpy(dtype=str)
    rubric_ids = features_df["rubric_id"].to_numpy(dtype=str)

    conf_df, corr_df, group_info = run_commonality_comparisons(
        features_df=features_df,
        args=args,
        rng=rng,
        compare_groups=step5.compare_groups,
        run_cluster_robust=False,
        student_ids=student_ids,
        question_ids=question_ids,
        rubric_ids=rubric_ids,
    )

    assert int(group_info["n_high"]) == 11
    assert int(group_info["n_low"]) == 45
    _assert_frame_matches(conf_df, pd.read_csv(FIXTURE_DIR / "phase4d_step5_confidence_golden.csv"))
    _assert_frame_matches(corr_df, pd.read_csv(FIXTURE_DIR / "phase4d_step5_correctness_golden.csv"))
