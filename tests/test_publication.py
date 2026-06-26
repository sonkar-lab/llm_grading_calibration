from pathlib import Path

import pandas as pd

from grading_calibration.pipelines.publication import (
    CALIBRATION_RATE_COL_OVERRIDES,
    DEFAULT_RESULTS_ROOT,
    SCOPE_OUTPUTS,
    calibration_interaction_summary,
    feature_family_significance_summary,
    published_anchor_values,
    write_publication_outputs,
)

RESULTS_FINAL_ROOT = Path("experiments/RESULTS_FINAL")
RESULTS_CANONICAL_ROOT = Path("experiments/RESULTS_CANONICAL")

PUBLISHED_TABLE4_ANCHORS = {
    ("ricechem", "self_reported"): (0.246, 0.134, 0.396, 43.9),
    ("ricechem", "self_consistency"): (0.047, 0.036, 0.068, 27.1),
    ("ricechem", "token_probability"): (0.075, 0.056, 0.092, 12.4),
    ("scientsbank", "self_reported"): (0.162, 0.056, 0.320, 35.3),
    ("scientsbank", "self_consistency"): (0.075, 0.044, 0.108, 35.3),
    ("scientsbank", "token_probability"): (0.059, 0.043, 0.109, 14.9),
    ("beetle", "self_reported"): (0.155, 0.091, 0.250, 23.9),
    ("beetle", "self_consistency"): (0.068, 0.058, 0.083, 15.8),
    ("beetle", "token_probability"): (0.076, 0.061, 0.077, 9.9),
}

PUBLISHED_FEATURE_RATE_ANCHORS = {
    "response": 73.3,
    "question": 12.4,
    "rubric": 0.0,
}

METHODS = ["self_reported", "self_consistency", "token_probability"]


def _parse_calibration_file(path: Path) -> tuple[str, str]:
    dataset = path.parts[path.parts.index("step3") + 1]
    stem = path.stem.removeprefix("calibration_regression_")
    for method in METHODS:
        if stem.endswith(f"_{method}"):
            return dataset, method
    raise AssertionError(f"cannot parse method from {path}")


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _manual_canonical_table4(results_root: Path, significance_col: str) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    frames = []
    for path in sorted((results_root / "step3").glob("*/tables/calibration_regression_*.csv")):
        dataset, method = _parse_calibration_file(path)
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "method", method)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    observed = {}
    for (dataset, method), subset in data.groupby(["dataset", "method"], sort=False):
        mask = _bool_series(subset[significance_col])
        betas = subset.loc[mask, "beta_interaction"].abs().dropna()
        observed[(dataset, method)] = (
            float(mask.mean() * 100),
            float(betas.median()),
            float(betas.quantile(0.25)),
            float(betas.quantile(0.75)),
        )
    return observed


def _summary_values(summary: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    return {
        (row.dataset, row.method): (
            float(row.pct_sig),
            float(row.median_abs_beta),
            float(row.q1_abs_beta),
            float(row.q3_abs_beta),
        )
        for row in summary.itertuples(index=False)
    }


def test_calibration_interaction_summary_reproduces_published_table4_anchors():
    summary = calibration_interaction_summary(RESULTS_FINAL_ROOT, scope="mixed")
    observed = {
        (row.dataset, row.method): (
            round(row.median_abs_beta, 3),
            round(row.q1_abs_beta, 3),
            round(row.q3_abs_beta, 3),
            round(row.pct_sig, 1),
        )
        for row in summary.itertuples(index=False)
    }

    assert observed == PUBLISHED_TABLE4_ANCHORS


def test_feature_family_summary_reproduces_published_feature_rate_anchors():
    summary = feature_family_significance_summary(RESULTS_FINAL_ROOT)
    observed = {
        row.feature_family: round(row.pct_sig, 1)
        for row in summary.itertuples(index=False)
    }

    assert observed["response"] == PUBLISHED_FEATURE_RATE_ANCHORS["response"]
    assert observed["question"] == PUBLISHED_FEATURE_RATE_ANCHORS["question"]
    assert observed["rubric"] == PUBLISHED_FEATURE_RATE_ANCHORS["rubric"]


def test_published_anchor_values_contains_minimum_phase5a_anchors():
    anchors = published_anchor_values(RESULTS_FINAL_ROOT)

    assert anchors["table4_pct_sig.ricechem.token_probability"] == PUBLISHED_TABLE4_ANCHORS[("ricechem", "token_probability")][3]
    assert anchors["table4_pct_sig.scientsbank.token_probability"] == PUBLISHED_TABLE4_ANCHORS[("scientsbank", "token_probability")][3]
    assert anchors["table4_pct_sig.beetle.token_probability"] == PUBLISHED_TABLE4_ANCHORS[("beetle", "token_probability")][3]
    assert anchors["feature_rate.response"] == PUBLISHED_FEATURE_RATE_ANCHORS["response"]


def test_canonical_within_table_matches_raw_results_canonical_without_beetle_override():
    summary = calibration_interaction_summary(RESULTS_CANONICAL_ROOT, scope="within")
    manual = _manual_canonical_table4(RESULTS_CANONICAL_ROOT, "significant_global")

    pd.testing.assert_frame_equal(
        pd.DataFrame.from_dict(_summary_values(summary), orient="index").sort_index(),
        pd.DataFrame.from_dict(manual, orient="index").sort_index(),
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )
    assert (summary["rate_significance_column"] == "significant_global").all()
    assert (summary["beta_significance_column"] == "significant_global").all()
    beetle_token = summary[(summary["dataset"] == "beetle") & (summary["method"] == "token_probability")].iloc[0]
    assert beetle_token.rate_significance_column == "significant_global"
    assert ("beetle", "token_probability") in CALIBRATION_RATE_COL_OVERRIDES


def test_canonical_across_table_matches_raw_results_canonical_without_beetle_override():
    summary = calibration_interaction_summary(RESULTS_CANONICAL_ROOT, scope="across")
    manual = _manual_canonical_table4(RESULTS_CANONICAL_ROOT, "significant_global_across_datasets")

    pd.testing.assert_frame_equal(
        pd.DataFrame.from_dict(_summary_values(summary), orient="index").sort_index(),
        pd.DataFrame.from_dict(manual, orient="index").sort_index(),
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )
    assert (summary["rate_significance_column"] == "significant_global_across_datasets").all()
    assert (summary["beta_significance_column"] == "significant_global_across_datasets").all()


def test_canonical_within_and_across_scopes_differ():
    within = calibration_interaction_summary(RESULTS_CANONICAL_ROOT, scope="within")
    across = calibration_interaction_summary(RESULTS_CANONICAL_ROOT, scope="across")
    merged = within.merge(across, on=["dataset", "method"], suffixes=("_within", "_across"))

    assert (merged["pct_sig_within"].round(12) != merged["pct_sig_across"].round(12)).any()


def test_default_results_root_is_canonical_and_writer_emits_all_scope_variants(tmp_path):
    assert DEFAULT_RESULTS_ROOT == RESULTS_CANONICAL_ROOT
    paths = write_publication_outputs(output_dir=tmp_path)

    expected_keys = {
        "calibration_within_csv",
        "calibration_within_tex",
        "calibration_across_csv",
        "calibration_across_tex",
        "calibration_mixed_csv",
        "calibration_mixed_tex",
        "feature_csv",
        "feature_tex",
    }
    assert set(paths) == expected_keys
    for path in paths.values():
        assert path.exists()
        assert path.parent == tmp_path

    assert set(SCOPE_OUTPUTS.values()).issubset({path.stem for path in paths.values()})
    assert "paper candidate, within-dataset FDR" in paths["calibration_within_tex"].read_text(encoding="utf-8")
    assert "paper candidate, across-dataset FDR" in paths["calibration_across_tex"].read_text(encoding="utf-8")
    assert "DIAGNOSTIC: reproduces published mixed scope; not for paper" in paths["calibration_mixed_tex"].read_text(encoding="utf-8")
    assert "RESULTS_CANONICAL" in paths["feature_tex"].read_text(encoding="utf-8")


def test_publication_writer_preserves_results_final_mixed_audit_path(tmp_path):
    paths = write_publication_outputs(RESULTS_FINAL_ROOT, tmp_path)
    calibration = pd.read_csv(paths["calibration_mixed_csv"])
    features = pd.read_csv(paths["feature_csv"])

    beetle_token_probability = calibration.loc[
        (calibration["dataset"] == "beetle")
        & (calibration["method"] == "token_probability"),
        "pct_sig",
    ].iloc[0]
    response_rate = features.loc[features["feature_family"] == "response", "pct_sig"].iloc[0]

    assert round(beetle_token_probability, 1) == PUBLISHED_TABLE4_ANCHORS[("beetle", "token_probability")][3]
    assert round(response_rate, 1) == PUBLISHED_FEATURE_RATE_ANCHORS["response"]
