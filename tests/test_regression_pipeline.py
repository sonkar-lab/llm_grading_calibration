from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from grading_calibration.metrics.calibration_bins import assign_bin_calibration_errors
from grading_calibration.pipelines.feature_pipeline import (
    extract_features_dataframe,
    load_predictions_csv,
)
from grading_calibration.pipelines.regression_pipeline import (
    run_calibration_regression,
    run_mixed_model_feature_effects,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PREDICTIONS_CSV = (
    FIXTURE_DIR
    / "phase4b_predictions"
    / "self_reported"
    / "predictions_gpt_oss_20b_self_reported_20251128-045340.csv"
)


def _step3_args() -> Namespace:
    return Namespace(
        alpha=0.05,
        calibration_regression_bootstrap=0,
        calibration_regression_ci_method="percentile",
        calibration_regression_features=["answer_length"],
        calibration_regression_glmer_nAGQ=1,
        calibration_regression_jackknife_max=200,
        calibration_regression_max_features=None,
        calibration_regression_maxfun=100000,
        calibration_regression_mode="linear",
        calibration_regression_n_grid=50,
        calibration_regression_optimizer="bobyqa",
        calibration_regression_spline_df=4,
        effects_backend="lme4",
        effects_gee_mode="simple",
        effects_include_student_vc_glmm=False,
        effects_lme4_maxfun=100000,
        effects_max_features=1,
        effects_no_gee=True,
        effects_optimizer="powell",
        effects_question_constant_separate_block=False,
        effects_standardize_outcome=False,
        effects_transform_bounded="none",
        effects_use_ranks=False,
        force_progress=False,
        include_bin_calibration_error=False,
        question_col="question_id",
        rubric_col="rubric_id",
        seed=4242,
        student_col="student_id",
    )


def _features_df() -> pd.DataFrame:
    instances, predictions, _ = load_predictions_csv(PREDICTIONS_CSV, show_progress=False)
    features_df = extract_features_dataframe(instances, predictions)
    features_df["calibration_error"] = assign_bin_calibration_errors(
        predictions,
        n_bins=10,
        use_calibrated=False,
    )
    return features_df


def _assert_numeric_columns_match(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str],
    *,
    atol: float,
) -> dict[str, float]:
    deltas = {}
    for col in columns:
        actual_values = actual[col].to_numpy(dtype=float)
        expected_values = expected[col].to_numpy(dtype=float)
        assert np.allclose(actual_values, expected_values, atol=atol, rtol=0.0, equal_nan=True), col
        finite = np.isfinite(actual_values) & np.isfinite(expected_values)
        deltas[col] = float(np.max(np.abs(actual_values[finite] - expected_values[finite]))) if finite.any() else 0.0
    return deltas


def _step3_args_config2() -> Namespace:
    """Config-2: flips on GEE, nonlinear+bootstrap, and multi-feature paths.

    Exercises the pipeline branches the primary gate leaves untouched:
    GEE result-column collection, the nonlinear/bootstrap calibration branch,
    and multi-feature FDR masking. Golden captured from pre-4b step3 (529fb5a^).
    """
    args = _step3_args()
    args.effects_no_gee = False
    args.effects_max_features = 2
    args.calibration_regression_features = ["answer_length", "response_length"]
    args.calibration_regression_mode = "nonlinear"
    args.calibration_regression_bootstrap = 50
    return args


_CONFIG2_FEATURES = ["answer_length", "response_length"]


def test_mixed_model_feature_effects_config2_match_golden():
    """GEE on + 2 features: guards GEE columns and multi-feature collection."""
    features_df = _features_df()
    args = _step3_args_config2()
    actual, _, qc_df, _ = run_mixed_model_feature_effects(
        features_df=features_df,
        corr_feature_cols=_CONFIG2_FEATURES,
        conf_feature_cols=_CONFIG2_FEATURES,
        cal_feature_cols=_CONFIG2_FEATURES,
        question_constant=[],
        args=args,
    )
    expected = pd.read_csv(FIXTURE_DIR / "phase4b_config2_mixed_model_effects_golden.csv")
    assert actual is not None and qc_df is None
    key = ["feature", "target_col"]
    actual = actual.sort_values(key).reset_index(drop=True)
    expected = expected.sort_values(key).reset_index(drop=True)
    assert actual[["feature", "target_col"]].equals(expected[["feature", "target_col"]])
    _assert_numeric_columns_match(
        actual,
        expected,
        ["beta", "se", "z", "p_value", "ci_low", "ci_high",
         "gee_beta", "gee_se", "gee_p_value"],
        atol=1e-8,
    )


def test_calibration_regression_config2_match_golden():
    """Nonlinear + bootstrap + 2 features: guards nonlinear branch and multi-feature FDR."""
    features_df = _features_df()
    args = _step3_args_config2()
    actual, curve_rows = run_calibration_regression(
        features_df=features_df,
        conf_feature_cols=_CONFIG2_FEATURES,
        confidence_leakage_features={
            "raw_confidence", "calibrated_confidence", "confidence_delta",
            "is_high_confidence", "is_medium_confidence", "is_low_confidence",
        },
        label_leakage_features={"prediction", "ground_truth", "is_correct"},
        args=args,
    )
    expected = pd.read_csv(FIXTURE_DIR / "phase4b_config2_calibration_regression_golden.csv")
    assert actual is not None
    assert (actual["mode"] == "nonlinear").all()
    # nonlinear mode produced curve rows (the bootstrap/curve branch ran)
    assert len(curve_rows) > 0
    actual = actual.sort_values("feature").reset_index(drop=True)
    expected = expected.sort_values("feature").reset_index(drop=True)
    assert actual["feature"].tolist() == expected["feature"].tolist()
    _assert_numeric_columns_match(
        actual,
        expected,
        ["p_interaction", "p_interaction_adj"],
        atol=1e-8,
    )


def test_mixed_model_feature_effects_match_phase4b_golden():
    features_df = _features_df()
    args = _step3_args()
    actual, loo_details, qc_df, qc_loo_details = run_mixed_model_feature_effects(
        features_df=features_df,
        corr_feature_cols=["answer_length"],
        conf_feature_cols=["answer_length"],
        cal_feature_cols=["answer_length"],
        question_constant=[],
        args=args,
    )
    expected = pd.read_csv(FIXTURE_DIR / "phase4b_mixed_model_effects_golden.csv")

    assert loo_details == {}
    assert qc_df is None
    assert qc_loo_details == {}
    assert actual is not None
    assert actual[["feature", "target_col", "outcome_type"]].reset_index(drop=True).equals(
        expected[["feature", "target_col", "outcome_type"]].reset_index(drop=True)
    )
    _assert_numeric_columns_match(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        ["beta", "se", "z", "p_value", "ci_low", "ci_high"],
        atol=1e-8,
    )


def test_calibration_regression_matches_phase4b_golden():
    features_df = _features_df()
    args = _step3_args()
    actual, curve_rows = run_calibration_regression(
        features_df=features_df,
        conf_feature_cols=["answer_length"],
        confidence_leakage_features={
            "raw_confidence",
            "calibrated_confidence",
            "confidence_delta",
            "is_high_confidence",
            "is_medium_confidence",
            "is_low_confidence",
        },
        label_leakage_features={"prediction", "ground_truth", "is_correct"},
        args=args,
    )
    expected = pd.read_csv(FIXTURE_DIR / "phase4b_calibration_regression_golden.csv")

    assert curve_rows == []
    assert actual is not None
    assert actual[["feature", "mode"]].reset_index(drop=True).equals(
        expected[["feature", "mode"]].reset_index(drop=True)
    )
    _assert_numeric_columns_match(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        [
            "beta_main",
            "se_main",
            "p_main",
            "beta_interaction",
            "se_interaction",
            "p_interaction",
            "p_interaction_adj",
        ],
        atol=1e-8,
    )
