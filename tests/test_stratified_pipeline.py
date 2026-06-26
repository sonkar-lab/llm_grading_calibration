from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_exp import step4_stratified_analysis as step4
from grading_calibration.pipelines.stratified_pipeline import (
    run_metric_pairwise_tests,
    run_stratified_group_metrics,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PREDICTIONS_CSV = (
    FIXTURE_DIR
    / "phase4b_predictions"
    / "self_reported"
    / "predictions_gpt_oss_20b_self_reported_20251128-045340.csv"
)


def _args() -> Namespace:
    return Namespace(
        alpha=0.05,
        compute_ci=True,
        metric_test_include_questions=False,
        metric_test_n_bootstrap=20,
        metric_test_pairwise_scope="topk_within_question",
        metric_test_pairwise_top_k=1,
        n_bins=5,
        n_bootstrap=20,
        skip_question_tests=True,
        stratify_by="rubric_id",
    )


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


def test_stratified_pipeline_matches_phase4d_golden():
    _, predictions, _ = step4.load_data_from_csv(PREDICTIONS_CSV)
    args = _args()
    rng = np.random.default_rng(4242)

    results_by_stratum, results_df = run_stratified_group_metrics(
        predictions=predictions,
        features_df=None,
        args=args,
        rng=rng,
        stratify_by_category=step4.stratify_by_category,
        stratify_by_feature_bins=step4.stratify_by_feature_bins,
        compute_metrics_for_group=step4.compute_metrics_for_group,
        compute_metrics_for_group_with_ci=step4.compute_metrics_for_group_with_ci,
    )
    metric_df, q_metric_df = run_metric_pairwise_tests(
        predictions=predictions,
        args=args,
        rng=rng,
        run_metric_pairwise_tests_within_question=step4._run_metric_pairwise_tests_within_question,
        run_metric_pairwise_tests_across_questions=step4._run_metric_pairwise_tests_across_questions,
    )

    assert list(results_by_stratum) == ["rubric_id"]
    assert q_metric_df.empty
    _assert_frame_matches(results_df, pd.read_csv(FIXTURE_DIR / "phase4d_step4_stratified_golden.csv"))
    _assert_frame_matches(metric_df, pd.read_csv(FIXTURE_DIR / "phase4d_step4_metric_pairwise_golden.csv"))
