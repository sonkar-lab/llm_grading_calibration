from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from grading_calibration.metrics.cross_comparison import load_all_predictions
from experiments.run_exp import step6_cross_comparison as step6
from grading_calibration.pipelines.cross_comparison_pipeline import run_cross_comparisons


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PREDICTIONS_DIR = FIXTURE_DIR / "phase4e_predictions"


def _args() -> Namespace:
    return Namespace(
        alpha=0.05,
        best_top_k=5,
        binning_strategies=["equal_width"],
        ci_method="percentile",
        gate_posthoc=False,
        interaction_method="artc",
        max_jackknife=1000,
        methods=["self_reported", "self_consistency"],
        metrics=["is_correct", "raw_confidence", "ece", "brier_mean"],
        models=["gptoss20b", "llama8b"],
        n_bins=10,
        n_boot=50,
        n_perm=50,
        seed=4242,
        skip_interaction_tests=False,
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


def test_cross_comparison_pipeline_matches_phase4e_golden():
    args = _args()
    all_preds = load_all_predictions(
        args.models,
        args.methods,
        PREDICTIONS_DIR,
        prefix="predictions",
        dataset="ricechem",
        flat_layout=False,
        filter_by_dataset=True,
    )
    rng = np.random.default_rng(args.seed)

    outputs = run_cross_comparisons(
        all_preds=all_preds,
        args=args,
        rng=rng,
        icc_flag=False,
        mixed_model_flag=False,
        compute_metrics_summary=step6._compute_metrics_summary,
        omnibus_friedman=step6._omnibus_friedman,
        align_by_instance=step6._align_by_instance,
        paired_compare=step6._paired_compare,
        summarize_comparisons=step6._summarize_comparisons,
        rank_summary_best=step6._rank_summary_best,
        select_best=step6._select_best,
    )

    _assert_frame_matches(outputs["metrics_summary"], pd.read_csv(FIXTURE_DIR / "phase4e_metrics_summary_golden.csv"))
    _assert_frame_matches(outputs["model_df"], pd.read_csv(FIXTURE_DIR / "phase4e_model_comparisons_golden.csv"))
    _assert_frame_matches(outputs["method_df"], pd.read_csv(FIXTURE_DIR / "phase4e_method_comparisons_golden.csv"))
    _assert_frame_matches(outputs["interaction_df"], pd.read_csv(FIXTURE_DIR / "phase4e_interaction_tests_golden.csv"))
