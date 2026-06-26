"""Generate publication-derived tables from parameterized results artifacts.

Phase 5a reproduced the published RESULTS_FINAL numbers, including known
significance-flag contamination. Phase 6c defaults to corrected
RESULTS_CANONICAL while keeping RESULTS_FINAL + mixed-scope generation available
for publication audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

Table4Scope = Literal["within", "across", "mixed"]

DEFAULT_RESULTS_ROOT = Path("experiments/RESULTS_CANONICAL")
DEFAULT_OUTPUT_DIR = Path("generated_tables/phase5a")

DATASET_ORDER = ["ricechem", "scientsbank", "beetle"]
METHOD_ORDER = ["self_reported", "self_consistency", "token_probability"]
DATASET_LABELS = {
    "ricechem": "RiceChem",
    "scientsbank": "SciEntsBank",
    "beetle": "Beetle",
}
METHOD_LABELS = {
    "self_reported": "self-reported",
    "self_consistency": "self-consistency",
    "token_probability": "token-probability",
}
FEATURE_FAMILY_LABELS = {
    "answer_constant": "answer",
    "question_constant": "question",
    "row_varying": "response",
    "rubric_constant": "rubric",
}

CALIBRATION_BETA_SIGNIFICANCE_COL = "significant_global_across_datasets"
CALIBRATION_RATE_COL_DEFAULT = "significant_global"
CALIBRATION_RATE_COL_OVERRIDES = {
    ("beetle", "token_probability"): "significant_global_across_datasets",
}
MIXED_MODEL_FEATURE_RATE_COL = "significant_across_datasets"
SCOPE_COLUMN_MAP: dict[Table4Scope, tuple[str, str]] = {
    "within": ("significant_global", "significant_global"),
    "across": ("significant_global_across_datasets", "significant_global_across_datasets"),
    "mixed": (CALIBRATION_RATE_COL_DEFAULT, CALIBRATION_BETA_SIGNIFICANCE_COL),
}
SCOPE_OUTPUTS: dict[Table4Scope, str] = {
    "within": "calibration_regression_interactions_within",
    "across": "calibration_regression_interactions_across",
    "mixed": "calibration_regression_interactions_mixed_diagnostic",
}


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _source_label(results_root: Path | str) -> str:
    root = Path(results_root)
    return root.name or str(root)


def _parse_step3_filename(path: Path, prefix: str) -> tuple[str, str, str]:
    try:
        dataset = path.parts[path.parts.index("step3") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot infer dataset from step3 path: {path}") from exc

    stem = path.stem
    expected_prefix = f"{prefix}_"
    if not stem.startswith(expected_prefix):
        raise ValueError(f"Expected {path.name} to start with {expected_prefix!r}")

    model_and_method = stem.removeprefix(expected_prefix)
    for method in METHOD_ORDER:
        suffix = f"_{method}"
        if model_and_method.endswith(suffix):
            return dataset, model_and_method[: -len(suffix)], method
    raise ValueError(f"Cannot infer method from filename: {path.name}")


def _read_step3_tables(results_root: Path, prefix: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pattern = f"*/tables/{prefix}_*.csv"
    for path in sorted((results_root / "step3").glob(pattern)):
        dataset, model, method = _parse_step3_filename(path, prefix)
        frame = pd.read_csv(path)
        if "method" in frame.columns:
            frame = frame.rename(columns={"method": "analysis_method"})
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "model", model)
        frame.insert(2, "method", method)
        frame.insert(3, "source_file", str(path))
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No step3 {prefix!r} CSVs found under {results_root}")
    return pd.concat(rows, ignore_index=True)


def load_calibration_regressions(results_root: Path | str = DEFAULT_RESULTS_ROOT) -> pd.DataFrame:
    """Load step3 calibration-regression CSVs with dataset/model/method columns."""
    return _read_step3_tables(Path(results_root), "calibration_regression")


def load_mixed_model_effects(results_root: Path | str = DEFAULT_RESULTS_ROOT) -> pd.DataFrame:
    """Load step3 mixed-model-effect CSVs with dataset/model/method columns."""
    return _read_step3_tables(Path(results_root), "mixed_model_effects")


def _scope_columns(scope: Table4Scope, dataset: str, method: str) -> tuple[str, str]:
    if scope not in SCOPE_COLUMN_MAP:
        raise ValueError(f"Unknown Table-4 scope {scope!r}; expected one of {sorted(SCOPE_COLUMN_MAP)}")
    rate_col, beta_col = SCOPE_COLUMN_MAP[scope]
    if scope == "mixed":
        rate_col = CALIBRATION_RATE_COL_OVERRIDES.get((dataset, method), rate_col)
    return rate_col, beta_col


def calibration_interaction_summary(
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    scope: Table4Scope = "mixed",
) -> pd.DataFrame:
    """Summarize Table-4 calibration interactions for the requested FDR scope."""
    data = load_calibration_regressions(results_root)
    rows: list[dict[str, object]] = []

    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            subset = data[(data["dataset"] == dataset) & (data["method"] == method)].copy()
            if subset.empty:
                continue

            rate_col, beta_col = _scope_columns(scope, dataset, method)
            rate_mask = _bool_series(subset[rate_col])
            beta_mask = _bool_series(subset[beta_col])
            significant_betas = subset.loc[beta_mask, "beta_interaction"].abs().dropna()

            rows.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n_tests": int(len(subset)),
                    "n_significant_rate": int(rate_mask.sum()),
                    "pct_sig": float(rate_mask.mean() * 100),
                    "rate_significance_column": rate_col,
                    "beta_significance_column": beta_col,
                    "median_abs_beta": float(significant_betas.median()),
                    "q1_abs_beta": float(significant_betas.quantile(0.25)),
                    "q3_abs_beta": float(significant_betas.quantile(0.75)),
                    "n_significant_beta": int(len(significant_betas)),
                }
            )

    return pd.DataFrame(rows)


def feature_family_significance_summary(
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
) -> pd.DataFrame:
    """Summarize correctness feature-family rates using the published across-dataset column."""
    data = load_mixed_model_effects(results_root)
    correctness = data[data["target_col"] == "is_correct"].copy()
    correctness["feature_family"] = correctness["feature_type"].map(FEATURE_FAMILY_LABELS)
    correctness = correctness.dropna(subset=["feature_family"])
    correctness["is_significant"] = _bool_series(correctness[MIXED_MODEL_FEATURE_RATE_COL])

    grouped = (
        correctness.groupby("feature_family", sort=False)["is_significant"]
        .agg(n_significant="sum", n_tests="count", significance_rate="mean")
        .reset_index()
    )
    grouped["pct_sig"] = grouped["significance_rate"] * 100
    grouped["significance_column"] = MIXED_MODEL_FEATURE_RATE_COL

    order = {"response": 0, "answer": 1, "question": 2, "rubric": 3}
    grouped["_order"] = grouped["feature_family"].map(order).fillna(99)
    return grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def published_anchor_values(results_root: Path | str = Path("experiments/RESULTS_FINAL")) -> dict[str, float]:
    """Return the minimum published anchors from RESULTS_FINAL + mixed scope."""
    table4 = calibration_interaction_summary(results_root, scope="mixed")
    feature_rates = feature_family_significance_summary(results_root)

    anchors: dict[str, float] = {}
    for _, row in table4.iterrows():
        key = f"table4_pct_sig.{row['dataset']}.{row['method']}"
        anchors[key] = round(float(row["pct_sig"]), 1)
    for _, row in feature_rates.iterrows():
        key = f"feature_rate.{row['feature_family']}"
        anchors[key] = round(float(row["pct_sig"]), 1)
    return anchors


def _table4_header(results_root: Path | str, scope: Table4Scope) -> list[str]:
    root_label = _source_label(results_root)
    if scope == "mixed":
        return [
            "% Auto-generated by grading_calibration.pipelines.publication.",
            f"% Source root: {root_label}.",
            "% DIAGNOSTIC: reproduces published mixed scope; not for paper.",
        ]
    return [
        "% Auto-generated by grading_calibration.pipelines.publication.",
        f"% Source root: {root_label}; corrected (RESULTS_CANONICAL); paper candidate, {scope}-dataset FDR.",
    ]


def _format_table4_latex(summary: pd.DataFrame, *, results_root: Path | str, scope: Table4Scope) -> str:
    lines = [
        *_table4_header(results_root, scope),
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Dataset & Method & med$|\beta|$ & IQR [Q1--Q3] & \% sig \\",
        r"\hline",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['dataset_label']} & {row['method_label']} & "
            f"{row['median_abs_beta']:.3f} & "
            f"{row['q1_abs_beta']:.3f}--{row['q3_abs_beta']:.3f} & "
            f"{row['pct_sig']:.1f} \\\\"  # noqa: ISC003
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def _format_feature_rates_latex(summary: pd.DataFrame, *, results_root: Path | str) -> str:
    root_label = _source_label(results_root)
    lines = [
        "% Auto-generated by grading_calibration.pipelines.publication.",
        f"% Source root: {root_label}; feature-family rates use significant_across_datasets.",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Feature family & Tests & \% significant \\",
        r"\hline",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['feature_family']} & {int(row['n_tests'])} & {row['pct_sig']:.1f} \\\\"  # noqa: ISC003
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_publication_outputs(
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write generated publication tables to a separate output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for scope, stem in SCOPE_OUTPUTS.items():
        summary = calibration_interaction_summary(results_root, scope=scope)
        csv_path = output_path / f"{stem}.csv"
        tex_path = output_path / f"{stem}.tex"
        summary.to_csv(csv_path, index=False)
        tex_path.write_text(_format_table4_latex(summary, results_root=results_root, scope=scope), encoding="utf-8")
        paths[f"calibration_{scope}_csv"] = csv_path
        paths[f"calibration_{scope}_tex"] = tex_path

    feature_rates = feature_family_significance_summary(results_root)
    feature_csv = output_path / "feature_family_significance_rates.csv"
    feature_tex = output_path / "feature_family_significance_rates.tex"
    feature_rates.to_csv(feature_csv, index=False)
    feature_tex.write_text(_format_feature_rates_latex(feature_rates, results_root=results_root), encoding="utf-8")
    paths["feature_csv"] = feature_csv
    paths["feature_tex"] = feature_tex
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication tables from corrected RESULTS_CANONICAL artifacts."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing step3 results CSVs (default: experiments/RESULTS_CANONICAL).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated .csv/.tex outputs (default: generated_tables/phase5a).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    paths = write_publication_outputs(args.results_root, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
