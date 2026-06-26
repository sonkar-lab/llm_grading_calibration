"""
Step 4: Stratified Analysis by Rubric, Question, and Feature Bins

This script:
1. Loads predictions CSV and reconstructs data objects
2. Computes per-group metrics (optionally with student-block bootstrap CIs)
3. Runs rubric-within-question mixed-model LRT tests for row-additive outcomes (primary)
4. Runs student-block bootstrap tests for non-row-additive metrics (ECE/AUC) (secondary)
5. Optionally computes prospective power curves + Bayesian assurance for LRT tests

Usage:
    python step4_stratified_analysis.py \
        --model qwen4b \
        --method self_reported \
        --stratify-by all \
        --compute-ci --n-bootstrap 1000 \
        --test-group-differences \
        --test-metric-differences
"""

import sys
import os
import re
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
import warnings
from datetime import datetime

from grading_calibration.config import apply_config_to_parser, load_config
from grading_calibration.domain.models import PredictionResult, GradingInstance
from utils import find_csv_folder_from_root
from utils.output_paths import add_output_args, build_output_manager
from utils.feature_cache import (
    compute_features_cache_key,
    build_features_cache_path,
    find_compatible_feature_caches,
    write_features_cache_metadata,
)
from analysis import FeatureExtractor, MechanisticAnalyzer
from grading_calibration.metrics import (
    expected_calibration_error,
    brier_score,
    auc_roc_correctness,
    evaluate_predictions,
)
from grading_calibration.metrics.calibration_bins import assign_bin_calibration_errors
from utils.statistics import (
    benjamini_hochberg,
    compute_icc_mixed_model,
    fit_lme4_lrt_from_formula,
    fit_lme4_glmer_fixef_vcov,
    compute_power_curve_lme4_lrt_student_re,
    compute_bayesian_assurance_lme4_lrt_empirical_student_re,
    _select_question_effects,
)
from grading_calibration.inference.model_aliases import MODEL_ALIASES as model_nicknames
from grading_calibration.pipelines.stratified_pipeline import (
    run_metric_pairwise_tests as _pipeline_run_metric_pairwise_tests,
    run_stratified_group_metrics as _pipeline_run_stratified_group_metrics,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Step 4: Stratified analysis by rubric, question, or feature bins"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model used for predictions",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="self_reported",
        choices=["self_reported", "self_consistency", "token_probability"],
        help="Confidence estimation method used",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ricechem",
        choices=["ricechem", "scientsbank", "beetle"],
        help="Dataset name for selecting prediction CSVs (default: ricechem).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the predictions CSV to load (0 = most recent)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="predictions",
        help="Prefix of the predictions CSV files",
    )
    parser.add_argument(
        "--warnings-log",
        type=str,
        default=None,
        help="Optional path for warnings log output (default: step4/logs/warnings.log).",
    )
    parser.add_argument(
        "--lme4-trace",
        action="store_true",
        help="Enable lme4 optimizer trace output (iterations).",
    )
    parser.add_argument(
        "--lme4-trace-file",
        type=str,
        default=None,
        help="File to append lme4 optimizer trace output.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Override predictions directory (legacy layout). "
             "Use this to load old predictions while still saving organized outputs.",
    )
    parser.add_argument(
        "--enable-perplexity",
        action="store_true",
        help="Compute perplexity features using a fixed reference LM.",
    )
    parser.add_argument(
        "--perplexity-model",
        type=str,
        default="gpt2",
        help="Reference LM for perplexity (default: gpt2).",
    )
    parser.add_argument(
        "--perplexity-device",
        type=str,
        default="cuda:0",
        help="Device for perplexity computation (default: cuda:0).",
    )
    parser.add_argument(
        "--perplexity-batch-size",
        type=int,
        default=32,
        help="Batch size for perplexity computation.",
    )
    parser.add_argument(
        "--perplexity-max-length",
        type=int,
        default=512,
        help="Max token length for perplexity computation.",
    )
    parser.add_argument(
        "--perplexity-min-tokens",
        type=int,
        default=3,
        help="Minimum token count to compute perplexity (else NaN).",
    )
    parser.add_argument(
        "--perplexity-debug",
        action="store_true",
        help="Log raw texts passed to the perplexity scorer (debug only).",
    )
    parser.add_argument(
        "--features-cache-dir",
        type=str,
        default=str(project_root / "experiments" / "features_cache"),
        help="Directory for cached features (default: experiments/features_cache).",
    )
    parser.add_argument(
        "--features-cache-mode",
        type=str,
        default="read_write",
        choices=["read_write", "read_only", "refresh"],
        help="Cache mode for features (default: read_write).",
    )
    parser.add_argument(
        "--features-cache-version",
        type=str,
        default="v1",
        help="Feature cache version tag (default: v1).",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=None,
        help="Limit to first N rows for a smoke test run.",
    )
    parser.add_argument(
        "--force-progress",
        action="store_true",
        help="Force display of progress bars even in non-interactive environments.",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="rubric_id",
        choices=[
            "rubric_id",
            "question_id",
            "all",
            "answer_length",
            "response_length",
            "question_length",
            "rubric_length",
        ],
        help="Feature to stratify by (use 'all' to compute both rubric_id and question_id tables).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of bins for continuous feature stratification",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save stratified results to CSV",
    )
    parser.add_argument(
        "--compute-ci",
        action="store_true",
        help="Compute student-block bootstrap confidence intervals for group metrics",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of student-block bootstrap draws per group (default: 1000)",
    )
    parser.add_argument(
        "--ci-method",
        type=str,
        default="percentile",
        choices=["percentile"],
        help="Bootstrap CI method (default: percentile).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CIs/tests (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--print-structure",
        action="store_true",
        help="Print hierarchical structure diagnostics (rubric nesting, rubrics per question).",
    )
    parser.add_argument(
        "--include-bin-calibration-error",
        action="store_true",
        help="Include bin-assigned calibration_error as a row-level outcome (exploratory; not recommended).",
    )
    parser.add_argument(
        "--print-calibration-slopes",
        action="store_true",
        help="Print rubric-specific calibration slopes per question (from calibration regression).",
    )

    # Step-4 extensions: group difference tests (rubric/question)
    parser.add_argument(
        "--test-group-differences",
        action="store_true",
        help="Run hypothesis tests for group differences (rubrics/questions).",
    )
    parser.add_argument(
        "--pairwise-scope",
        type=str,
        default="none",
        choices=["none", "all_within_question", "topk_within_question"],
        help="Which pairwise rubric tests to run (default: none).",
    )
    parser.add_argument(
        "--pairwise-top-k",
        type=int,
        default=10,
        help="If --pairwise-scope=topk_within_question, run top-K pairs per (question,outcome).",
    )
    parser.add_argument(
        "--run-question-tests",
        dest="skip_question_tests",
        action="store_false",
        help="Run question-level group tests (EXPLORATORY; questions are only 4).",
    )
    parser.add_argument(
        "--skip-question-tests",
        dest="skip_question_tests",
        action="store_true",
        help="Skip question-level group tests (default; questions are only 4).",
    )
    parser.set_defaults(skip_question_tests=True)

    # Step-4 extensions: metric-level (non-row-additive) group difference tests
    parser.add_argument(
        "--test-metric-differences",
        action="store_true",
        help="Run secondary metric-level pairwise tests (ECE/AUC) via student-block bootstrap within-question.",
    )
    parser.add_argument(
        "--metric-test-pairwise-scope",
        type=str,
        default="topk_within_question",
        choices=["none", "all_within_question", "topk_within_question"],
        help="Which rubric pairs to test for metric differences (ECE/AUC) (default: topk_within_question).",
    )
    parser.add_argument(
        "--metric-test-pairwise-top-k",
        type=int,
        default=10,
        help="If --metric-test-pairwise-scope=topk_within_question, run top-K rubric pairs per (question,metric) (default: 10).",
    )
    parser.add_argument(
        "--metric-test-n-bootstrap",
        type=int,
        default=500,
        help="Student-block bootstrap draws per metric pairwise test (default: 500).",
    )
    parser.add_argument(
        "--metric-test-include-questions",
        action="store_true",
        help="Also test question-pair metric differences (ECE/AUC) via paired student-block bootstrap (exploratory; questions are fixed, n=4).",
    )

    # Step-4 extensions: prospective power + Bayesian assurance for mixed-model LRT tests
    parser.add_argument(
        "--power-method",
        type=str,
        default="none",
        choices=["none", "mixed_model_lrt"],
        help="Prospective power method to run for primary LRT tests (default: none).",
    )
    parser.add_argument(
        "--power-scope",
        type=str,
        default="omnibus",
        choices=["none", "omnibus", "pairwise", "all"],
        help="Which LRT tests to compute prospective power for (default: omnibus).",
    )
    parser.add_argument(
        "--power-n-students-target",
        type=int,
        default=None,
        help="Target number of students for prospective power (default: current n_students).",
    )
    parser.add_argument(
        "--power-n-sim",
        type=int,
        default=200,
        help="Monte Carlo simulations per |beta| point for prospective power (default: 200).",
    )
    parser.add_argument(
        "--power-beta-grid-abs",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2],
        help="Absolute beta values for the power curve (two-sided LRT; default: 0.01 0.02 0.05 0.1 0.2).",
    )
    parser.add_argument(
        "--power-seed-students",
        type=int,
        default=123,
        help="Seed for student resampling in power simulations (default: 123).",
    )
    parser.add_argument(
        "--power-seed-outcomes",
        type=int,
        default=456,
        help="Seed for outcome simulation in power simulations (default: 456).",
    )

    parser.add_argument(
        "--bayesian-assurance",
        action="store_true",
        help="Compute Bayesian assurance for LRT tests using an empirical prior over beta.",
    )
    parser.add_argument(
        "--assurance-mode",
        type=str,
        default="fast",
        choices=["fast", "faithful"],
        help="Assurance computation mode: fast (interpolate curve) or faithful (nested MC) (default: fast).",
    )
    parser.add_argument(
        "--assurance-scope",
        type=str,
        default="omnibus",
        choices=["none", "omnibus", "pairwise", "all"],
        help="Which LRT tests to compute Bayesian assurance for (default: omnibus).",
    )
    parser.add_argument(
        "--assurance-n-prior",
        type=int,
        default=200,
        help="Student-block bootstrap draws for empirical beta prior (default: 200).",
    )
    parser.add_argument(
        "--assurance-n-sim-inner",
        type=int,
        default=100,
        help="Simulations per prior draw in faithful assurance mode (default: 100).",
    )
    parser.add_argument(
        "--assurance-winsorize-pct",
        type=float,
        default=None,
        help="Optional winsorization percent for empirical prior beta samples (e.g., 2.5).",
    )
    parser.add_argument(
        "--assurance-seed-prior",
        type=int,
        default=789,
        help="Seed for empirical prior bootstrap (default: 789).",
    )
    parser.add_argument(
        "--plot-power-curves",
        action="store_true",
        help="Save power curve plots to the analysis directory (requires matplotlib).",
    )

    # lme4 controls (shared by LRT/power/assurance)
    parser.add_argument(
        "--lme4-glmer-nAGQ",
        type=int,
        default=1,
        help="glmer nAGQ setting for binary models (default: 1; Laplace approximation).",
    )
    parser.add_argument(
        "--lme4-optimizer",
        type=str,
        default="bobyqa",
        help="lme4 optimizer name (default: bobyqa).",
    )
    parser.add_argument(
        "--lme4-maxfun",
        type=int,
        default=100000,
        help="lme4 optimizer maxfun (default: 100000).",
    )

    # Step-4 extensions: rubric/question-level feature associations
    parser.add_argument(
        "--associate-group-features",
        action="store_true",
        help="Compute rubric-level aggregate features and test association with group metrics (Spearman + permutation/bootstrap).",
    )
    parser.add_argument(
        "--association-scope",
        type=str,
        default="rubric",
        choices=["rubric", "question", "both"],
        help="Which group level to analyze for feature associations (default: rubric).",
    )
    parser.add_argument(
        "--association-n-perm",
        type=int,
        default=2000,
        help="Permutation count for Spearman association tests (default: 2000).",
    )
    parser.add_argument(
        "--association-n-boot",
        type=int,
        default=2000,
        help="Bootstrap draws for Spearman CI (default: 2000).",
    )
    parser.add_argument(
        "--association-bootstrap-method",
        type=str,
        default="percentile",
        choices=["percentile", "bca"],
        help="Bootstrap CI method for Spearman association (default: percentile).",
    )

    # Step-4 extensions: flag "problematic" groups based on metric thresholds
    parser.add_argument(
        "--problematic-ece-threshold",
        type=float,
        default=None,
        help="If set, flag groups whose ECE exceeds this threshold (uses CI if available).",
    )
    parser.add_argument(
        "--problematic-accuracy-threshold",
        type=float,
        default=None,
        help="If set, flag groups whose accuracy is below this threshold (uses CI if available).",
    )
    parser.add_argument(
        "--problematic-max-groups",
        type=int,
        default=25,
        help="Maximum number of problematic groups to print (default: 25).",
    )

    # Optional: global multiple-testing correction across all Step-4 tests
    parser.add_argument(
        "--global-fdr",
        action="store_true",
        help="Also compute a global BH correction across all Step-4 hypothesis tests (LRT + metric bootstrap).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file with argparse destination overrides.",
    )

    add_output_args(parser)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        config = load_config(config_args.config, parser)
        apply_config_to_parser(parser, config)
    args = parser.parse_args(argv)
    if hasattr(args, "config"):
        delattr(args, "config")
    return args

def _format_sig(p_adj: float, alpha: float) -> str:
    if not np.isfinite(p_adj):
        return ""
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < alpha:
        return "*"
    return ""


def _apply_global_fdr_across_tests(
    test_frames: list[tuple[str, pd.DataFrame]],
    *,
    alpha: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Apply a single Benjamini-Hochberg correction across multiple Step-4 test families.

    Returns:
      - updated_frames: mapping from family name -> df with added columns:
          p_adjusted_global, sig_global
      - global_df: long-form table with one row per test (for saving/printing)
    """
    pieces: list[pd.DataFrame] = []
    updated: dict[str, pd.DataFrame] = {}

    for family, df in test_frames:
        if df is None or len(df) == 0 or "p_value" not in df.columns:
            continue
        local = df.copy()
        local["_global_family"] = str(family)
        local["_global_row"] = np.arange(len(local), dtype=int)
        pieces.append(local)
        updated[family] = local

    if not pieces:
        return updated, pd.DataFrame()

    global_df = pd.concat(pieces, ignore_index=True)
    pvals = global_df["p_value"].to_numpy(dtype=float)
    mask = np.isfinite(pvals)

    p_adj_global = np.full(len(global_df), np.nan, dtype=float)
    sig_flags = np.full(len(global_df), False, dtype=bool)
    if mask.any():
        p_adj, sig = benjamini_hochberg(pvals[mask], alpha=alpha)
        p_adj_global[mask] = p_adj
        sig_flags[mask] = sig

    global_df["p_adjusted_global"] = p_adj_global
    global_df["sig_global"] = [(_format_sig(p, alpha) if np.isfinite(p) else "") for p in p_adj_global]

    for family, df in list(updated.items()):
        g = global_df[global_df["_global_family"] == str(family)].sort_values("_global_row")
        if len(g) != len(df):
            # Should never happen, but avoid silent misalignment.
            raise RuntimeError(f"Global FDR alignment failed for family={family}: {len(g)} != {len(df)}")
        df["p_adjusted_global"] = g["p_adjusted_global"].to_numpy(dtype=float)
        df["sig_global"] = g["sig_global"].astype(str).to_numpy()
        updated[family] = df.drop(columns=["_global_family", "_global_row"], errors="ignore")

    return updated, global_df.drop(columns=["_global_row"], errors="ignore")


def _safe_slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    s = s.strip("._-")
    return s if s else "NA"


def _save_table(
    df: pd.DataFrame,
    out_path: Path,
    *,
    output_manager=None,
    artifact: str | None = None,
    model: str | None = None,
    method: str | None = None,
    legacy_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if output_manager and artifact:
        output_manager.register(
            step="step4",
            artifact=artifact,
            path=out_path,
            model=model,
            method=method,
        )
    if output_manager and output_manager.config.save_legacy and legacy_path is not None:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(legacy_path, index=False)


def _plot_power_curves(
    power_df: pd.DataFrame,
    analysis_dir: Path,
    *,
    prefix: str,
) -> int:
    if power_df is None or len(power_df) == 0:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Warning: matplotlib not available, skipping power curve plots: {e}")
        return 0

    beta_col = None
    if "beta_target_abs" in power_df.columns:
        beta_col = "beta_target_abs"
    elif "beta_target" in power_df.columns:
        beta_col = "beta_target"
    if beta_col is None:
        print("Warning: power curve table missing beta_target column; skipping plots.")
        return 0

    group_cols = [c for c in ["question_id", "outcome", "test", "coef_name", "rubric_a", "rubric_b"] if c in power_df.columns]
    analysis_dir.mkdir(parents=True, exist_ok=True)

    n_plots = 0
    groups = power_df.groupby(group_cols, dropna=False) if group_cols else [("all", power_df)]
    for key, df_g in groups:
        df_g = df_g.sort_values(beta_col)
        x = df_g[beta_col].to_numpy(dtype=float)
        y = df_g["power"].to_numpy(dtype=float)
        if x.size == 0 or y.size == 0:
            continue

        label_parts = []
        if "question_id" in df_g.columns:
            label_parts.append(f"q{df_g['question_id'].iloc[0]}")
        if "outcome" in df_g.columns:
            label_parts.append(str(df_g["outcome"].iloc[0]))
        if "test" in df_g.columns:
            label_parts.append(str(df_g["test"].iloc[0]))
        if "coef_name" in df_g.columns and str(df_g["coef_name"].iloc[0]) not in {"", "nan", "None"}:
            label_parts.append(str(df_g["coef_name"].iloc[0]))
        if "rubric_a" in df_g.columns and str(df_g["rubric_a"].iloc[0]) not in {"", "nan", "None"}:
            label_parts.append(str(df_g["rubric_a"].iloc[0]))
        if "rubric_b" in df_g.columns and str(df_g["rubric_b"].iloc[0]) not in {"", "nan", "None"}:
            label_parts.append(str(df_g["rubric_b"].iloc[0]))

        title = "Power curve"
        if label_parts:
            title += ": " + ", ".join(label_parts)

        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o", linewidth=1.5)
        if {"power_ci_low", "power_ci_high"}.issubset(df_g.columns):
            y_lo = df_g["power_ci_low"].to_numpy(dtype=float)
            y_hi = df_g["power_ci_high"].to_numpy(dtype=float)
            plt.fill_between(x, y_lo, y_hi, alpha=0.2)
        plt.ylim(0.0, 1.0)
        plt.xlabel("|beta|")
        plt.ylabel("Power")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        slug = _safe_slug("__".join(label_parts)) if label_parts else "all"
        out_path = analysis_dir / f"{prefix}_{slug}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        n_plots += 1

    return n_plots


def _select_topk_pairs_by_mean_diff(
    df_q: pd.DataFrame,
    outcome_col: str,
    rubrics: list[str],
    top_k: int,
) -> list[tuple[str, str]]:
    """
    Heuristic for selecting top-K rubric pairs within a question: |mean(y_rA)-mean(y_rB)|.

    This is only a *computational* shortcut to limit pairwise tests; inference still uses LRT.
    """
    means = (
        df_q.groupby("rubric_id")[outcome_col]
        .mean()
        .reindex(rubrics)
        .dropna()
        .to_dict()
    )
    pairs: list[tuple[str, str, float]] = []
    r_list = list(means.keys())
    for i in range(len(r_list)):
        for j in range(i + 1, len(r_list)):
            a, b = r_list[i], r_list[j]
            pairs.append((a, b, abs(means[a] - means[b])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pairs[: max(int(top_k), 0)]]


def _rubric_mean_range_effect_size(
    df_qo: pd.DataFrame,
    *,
    outcome_col: str,
    rubrics: list[str],
    alpha: float,
    compute_ci: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """
    Effect size summary for omnibus rubric heterogeneity:

      range = max_r mean(y|rubric=r) - min_r mean(y|rubric=r)   (within a question)

    CI is computed via student-block bootstrap (resample students with replacement).
    """
    df_qo = df_qo.dropna(subset=["student_id", "rubric_id", outcome_col]).copy()
    if len(df_qo) == 0:
        return {
            "rubric_mean_range": np.nan,
            "rubric_mean_range_ci_low": np.nan,
            "rubric_mean_range_ci_high": np.nan,
            "rubric_min": "",
            "rubric_max": "",
            "mean_min": np.nan,
            "mean_max": np.nan,
        }

    wide = (
        df_qo.pivot(index="student_id", columns="rubric_id", values=outcome_col)
        .reindex(columns=rubrics)
    )
    y = wide.to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        means_obs = np.nanmean(y, axis=0)
    finite = np.isfinite(means_obs)
    if finite.sum() < 2:
        return {
            "rubric_mean_range": np.nan,
            "rubric_mean_range_ci_low": np.nan,
            "rubric_mean_range_ci_high": np.nan,
            "rubric_min": "",
            "rubric_max": "",
            "mean_min": np.nan,
            "mean_max": np.nan,
        }

    idx_min = int(np.nanargmin(means_obs))
    idx_max = int(np.nanargmax(means_obs))
    mean_min = float(means_obs[idx_min])
    mean_max = float(means_obs[idx_max])
    r_min = str(rubrics[idx_min]) if idx_min < len(rubrics) else ""
    r_max = str(rubrics[idx_max]) if idx_max < len(rubrics) else ""
    range_obs = float(mean_max - mean_min)

    ci_low = np.nan
    ci_high = np.nan
    if compute_ci and int(n_bootstrap) > 0:
        n = y.shape[0]
        boot = np.full(int(n_bootstrap), np.nan, dtype=float)
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            with np.errstate(all="ignore"):
                m = np.nanmean(y[idx, :], axis=0)
            fin = np.isfinite(m)
            if fin.sum() < 2:
                continue
            boot[b] = float(np.nanmax(m[fin]) - np.nanmin(m[fin]))

        if np.isfinite(boot).any():
            q_lo, q_hi = np.nanquantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
            ci_low = float(q_lo)
            ci_high = float(q_hi)

    return {
        "rubric_mean_range": range_obs,
        "rubric_mean_range_ci_low": ci_low,
        "rubric_mean_range_ci_high": ci_high,
        "rubric_min": r_min,
        "rubric_max": r_max,
        "mean_min": mean_min,
        "mean_max": mean_max,
    }


def _match_interaction_coef_name(
    coef_names: list[str],
    rubric: str,
) -> str | None:
    rubric_str = str(rubric)
    for name in coef_names:
        if ("logit_conf_c:" in name or ":logit_conf_c" in name) and name.endswith(rubric_str):
            return name
    for name in coef_names:
        if ("logit_conf_c:" in name or ":logit_conf_c" in name) and rubric_str in name:
            return name
    return None


def _calibration_slopes_by_rubric(
    *,
    coef_names: list[str],
    coef_values: np.ndarray,
    rubrics: list[str],
) -> dict[str, float]:
    """
    Compute rubric-specific calibration slopes from fixed effects.

    slope_r = beta_logit + beta_logit:rubric_r (baseline rubric has no interaction term).
    """
    if not coef_names or coef_values.size == 0:
        return {}

    name_to_idx = {n: i for i, n in enumerate(coef_names)}
    idx_logit = name_to_idx.get("logit_conf_c")
    if idx_logit is None:
        return {}

    rubrics_sorted = [str(r) for r in rubrics]
    slopes: dict[str, float] = {}
    for i, r in enumerate(rubrics_sorted):
        if i == 0:
            slopes[r] = float(coef_values[idx_logit])
            continue
        coef_name = _match_interaction_coef_name(coef_names, r)
        if coef_name is None or coef_name not in name_to_idx:
            slopes[r] = float(coef_values[idx_logit])
            continue
        slopes[r] = float(coef_values[idx_logit] + coef_values[name_to_idx[coef_name]])
    return {k: v for k, v in slopes.items() if np.isfinite(v)}


def _calibration_slope_range_effect_size(
    *,
    coef_names: list[str],
    coef_values: np.ndarray,
    vcov: np.ndarray,
    rubrics: list[str],
    alpha: float,
    compute_ci: bool,
    n_draws: int,
    rng: np.random.Generator,
) -> dict:
    """
    Effect size for calibration regression: slope range across rubrics.

    Slopes are computed from fixed effects in the full model:
      slope_r = beta_logit + beta_logit:rubric_r (baseline rubric has no interaction term).
    """
    if not coef_names or coef_values.size == 0:
        return {
            "slope_min": np.nan,
            "slope_max": np.nan,
            "slope_range": np.nan,
            "slope_range_ci_low": np.nan,
            "slope_range_ci_high": np.nan,
            "rubric_min": "",
            "rubric_max": "",
        }

    rubrics_sorted = [str(r) for r in rubrics]
    slopes = _calibration_slopes_by_rubric(
        coef_names=coef_names,
        coef_values=coef_values,
        rubrics=rubrics_sorted,
    )
    finite_slopes = {k: v for k, v in slopes.items() if np.isfinite(v)}
    if len(finite_slopes) < 2:
        return {
            "slope_min": np.nan,
            "slope_max": np.nan,
            "slope_range": np.nan,
            "slope_range_ci_low": np.nan,
            "slope_range_ci_high": np.nan,
            "rubric_min": "",
            "rubric_max": "",
        }

    rubric_min = min(finite_slopes, key=finite_slopes.get)
    rubric_max = max(finite_slopes, key=finite_slopes.get)
    slope_min = float(finite_slopes[rubric_min])
    slope_max = float(finite_slopes[rubric_max])
    slope_range = float(slope_max - slope_min)

    ci_low = np.nan
    ci_high = np.nan
    if compute_ci and int(n_draws) > 0 and vcov.size:
        name_to_idx = {n: i for i, n in enumerate(coef_names)}
        idx_logit = name_to_idx.get("logit_conf_c")
        if idx_logit is None:
            return {
                "slope_min": slope_min,
                "slope_max": slope_max,
                "slope_range": slope_range,
                "slope_range_ci_low": np.nan,
                "slope_range_ci_high": np.nan,
                "rubric_min": rubric_min,
                "rubric_max": rubric_max,
            }
        n_draws = int(n_draws)
        try:
            draws = rng.multivariate_normal(mean=coef_values, cov=vcov, size=n_draws)
        except Exception:
            try:
                diag = np.diag(np.diag(vcov))
                draws = rng.multivariate_normal(mean=coef_values, cov=diag, size=n_draws)
            except Exception:
                draws = None

        if draws is not None:
            ranges = np.full(n_draws, np.nan, dtype=float)
            for i, b in enumerate(draws):
                if not np.isfinite(b[idx_logit]):
                    continue
                s_vals = []
                for j, r in enumerate(rubrics_sorted):
                    if j == 0:
                        s_vals.append(float(b[idx_logit]))
                        continue
                    coef_name = _match_interaction_coef_name(coef_names, r)
                    idx = name_to_idx.get(coef_name) if coef_name else None
                    if idx is None or not np.isfinite(b[idx]):
                        s_vals.append(float(b[idx_logit]))
                    else:
                        s_vals.append(float(b[idx_logit] + b[idx]))
                s_vals = [v for v in s_vals if np.isfinite(v)]
                if len(s_vals) >= 2:
                    ranges[i] = float(max(s_vals) - min(s_vals))

            if np.isfinite(ranges).any():
                q_lo, q_hi = np.nanquantile(ranges, [alpha / 2.0, 1.0 - alpha / 2.0])
                ci_low = float(q_lo)
                ci_high = float(q_hi)

    return {
        "slope_min": slope_min,
        "slope_max": slope_max,
        "slope_range": slope_range,
        "slope_range_ci_low": ci_low,
        "slope_range_ci_high": ci_high,
        "rubric_min": rubric_min,
        "rubric_max": rubric_max,
    }


def _interpret_rho(rho: float) -> str:
    if not np.isfinite(rho):
        return ""
    ar = abs(float(rho))
    if ar < 0.1:
        return "negligible"
    if ar < 0.3:
        return "weak"
    if ar < 0.5:
        return "moderate"
    if ar < 0.7:
        return "strong"
    return "very strong"


def _spearman_within_strata(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    strata_col: str,
    n_perm: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    """
    Stratified Spearman-style association test controlling for a categorical confounder.

    Implementation:
    - Compute within-stratum ranks for x and y, then Pearson correlation of those ranks.
    - Permutation p-value: shuffle x ranks *within each stratum*.
    - Bootstrap CI: sample rows with replacement *within each stratum*.

    This is used for rubric-level associations where rubrics are nested in questions
    (so question_id is a confounder we must control for).
    """
    df2 = df[[strata_col, x_col, y_col]].copy()
    df2 = df2.dropna(subset=[strata_col, x_col, y_col])
    if len(df2) < 6:
        return {"r": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": int(len(df2))}

    # Within-stratum ranks (Spearman within strata).
    df2["_x_rank"] = df2.groupby(strata_col)[x_col].rank(method="average")
    df2["_y_rank"] = df2.groupby(strata_col)[y_col].rank(method="average")

    x = df2["_x_rank"].to_numpy(dtype=float)
    y = df2["_y_rank"].to_numpy(dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {"r": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": int(len(df2))}

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a2 = np.asarray(a, dtype=float)
        b2 = np.asarray(b, dtype=float)
        if a2.size < 3 or np.std(a2) == 0 or np.std(b2) == 0:
            return float("nan")
        return float(np.corrcoef(a2, b2)[0, 1])

    r_obs = _corr(x, y)

    # Precompute indices per stratum for stratified resampling.
    strata_vals = df2[strata_col].astype(str).to_numpy()
    idx_by_s: dict[str, np.ndarray] = {}
    for s in np.unique(strata_vals):
        idx_by_s[str(s)] = np.flatnonzero(strata_vals == s)

    # Stratified permutation test.
    perm_stats: list[float] = []
    for _ in range(int(n_perm)):
        x_perm = x.copy()
        for idx in idx_by_s.values():
            if len(idx) >= 2:
                x_perm[idx] = x_perm[rng.permutation(idx)]
        perm_stats.append(_corr(x_perm, y))
    perm_arr = np.asarray(perm_stats, dtype=float)
    perm_arr = perm_arr[np.isfinite(perm_arr)]
    if perm_arr.size == 0 or not np.isfinite(r_obs):
        p_value = np.nan
    else:
        p_value = float((np.sum(np.abs(perm_arr) >= abs(r_obs)) + 1.0) / (perm_arr.size + 1.0))

    # Stratified bootstrap CI.
    boot_stats: list[float] = []
    for _ in range(int(n_boot)):
        boot_idx: list[int] = []
        for idx in idx_by_s.values():
            if len(idx) >= 1:
                boot_idx.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
        boot_idx_arr = np.asarray(boot_idx, dtype=int)
        boot_stats.append(_corr(x[boot_idx_arr], y[boot_idx_arr]))
    boot_arr = np.asarray(boot_stats, dtype=float)
    boot_arr = boot_arr[np.isfinite(boot_arr)]
    if boot_arr.size == 0:
        ci_low = np.nan
        ci_high = np.nan
    else:
        ci_low = float(np.percentile(boot_arr, 100 * (alpha / 2)))
        ci_high = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

    return {"r": float(r_obs), "p_value": p_value, "ci_low": ci_low, "ci_high": ci_high, "n": int(len(df2))}


def verify_hierarchical_structure(df: pd.DataFrame) -> dict:
    """
    Verify and summarize the RiceChem hierarchical structure.

    Critical: rubric_id must be nested in question_id.
    """
    required = {"student_id", "question_id", "rubric_id"}
    if not required.issubset(df.columns):
        return {"ok": False, "missing_cols": sorted(required - set(df.columns))}

    rubric_question_map = df.groupby("rubric_id")["question_id"].nunique()
    if not (rubric_question_map == 1).all():
        bad = rubric_question_map[rubric_question_map > 1].to_dict()
        raise ValueError(f"Rubric nesting violated: rubric_id spans multiple question_id values: {bad}")

    question_rubrics = (
        df.groupby("question_id")["rubric_id"]
        .apply(lambda s: sorted({str(x) for x in s.dropna()}))
        .to_dict()
    )

    return {
        "ok": True,
        "n_rows": int(len(df)),
        "n_students": int(df["student_id"].nunique()),
        "n_questions": int(df["question_id"].nunique()),
        "n_rubrics": int(df["rubric_id"].nunique()),
        "question_rubrics": {str(k): v for k, v in question_rubrics.items()},
    }


def print_hierarchical_diagnostics(structure: dict) -> None:
    print("\n" + "=" * 80)
    print("HIERARCHICAL STRUCTURE VERIFICATION")
    print("=" * 80)
    if not structure.get("ok", False):
        missing = structure.get("missing_cols", [])
        print(f"Skipped: missing required columns: {missing}")
        return

    print(f"Total rows:  {structure['n_rows']:,}")
    print(f"Students:    {structure['n_students']:,}")
    print(f"Questions:   {structure['n_questions']:,} (fixed set)")
    print(f"Rubrics:     {structure['n_rubrics']:,} (nested within questions)")
    print("\nRubrics per question:")
    for q, rubs in sorted(structure["question_rubrics"].items(), key=lambda kv: kv[0]):
        print(f"  Question {q}: {len(rubs)} rubrics ({', '.join(rubs)})")
    print("\nNesting check: ✓ each rubric_id maps to exactly one question_id")


def _compute_group_metrics_df(
    predictions: list[PredictionResult],
    group_col: str,
    *,
    compute_ci: bool,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    groups = stratify_by_category(predictions, group_col)
    results: list[dict] = []
    for group_name, group_preds in groups.items():
        if compute_ci:
            metrics = compute_metrics_for_group_with_ci(
                group_preds, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
            )
        else:
            metrics = compute_metrics_for_group(group_preds)
        metrics["group"] = str(group_name)
        results.append(metrics)
    out = pd.DataFrame(results)
    return out


def _compute_group_features_df(
    features_df: pd.DataFrame,
    group_col: str,
    *,
    keep_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate features to the group level by mean (for row/answer varying) or first (for constants).
    """
    df = features_df.copy()
    if group_col not in df.columns:
        raise ValueError(f"features_df missing group column {group_col!r}")
    df[group_col] = df[group_col].astype(str)
    cols = [c for c in keep_cols if c in df.columns and c != group_col]
    if not cols:
        return df[[group_col]].drop_duplicates().assign(_no_features=True)
    g = df.groupby(group_col, dropna=False)
    agg = g[cols].mean(numeric_only=True)
    agg = agg.reset_index()
    return agg


def _make_row_level_outcomes(
    df: pd.DataFrame,
    predictions: list[PredictionResult],
    *,
    n_bins: int = 10,
    include_calibration_error: bool = False,
) -> pd.DataFrame:
    """
    Construct row-additive outcomes for mixed-model inference.

    Returns a copy of df with added columns:
      - is_correct (0/1)
      - brier (squared error on probability scale)
      - calibration_error (row-level proxy from bin-assigned calibration errors; optional)
    """
    out = df.copy()
    if "prediction" not in out.columns:
        out["prediction"] = out["prediction"].astype(int)
    if "ground_truth_label" in out.columns and "ground_truth" not in out.columns:
        out["ground_truth"] = out["ground_truth_label"]
    out["is_correct"] = (out["prediction"].astype(int) == out["ground_truth"].astype(int)).astype(int)
    out["raw_confidence"] = pd.to_numeric(out["raw_confidence"], errors="coerce")
    out["brier"] = (out["raw_confidence"] - out["is_correct"].astype(float)) ** 2

    if include_calibration_error:
        # Row-level calibration proxy: assign each row to a confidence bin and attach that bin's error.
        # This yields a row-additive outcome usable in LMM/LRT (exploratory only).
        cal_err = assign_bin_calibration_errors(predictions, n_bins=n_bins, use_calibrated=False)
        if len(cal_err) != len(out):
            raise ValueError(
                f"Calibration error assignment length mismatch: got {len(cal_err)} for df rows {len(out)}"
            )
        out["calibration_error"] = np.asarray(cal_err, dtype=float)
    return out


def _bootstrap_metric_diff_student_block(
    preds_a: list[PredictionResult],
    preds_b: list[PredictionResult],
    *,
    metric_key: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    """
    Student-block bootstrap test for a metric difference between two rubric groups.

    Metric keys must be produced by `compute_metrics_for_group()` (e.g., 'ece', 'auc_roc').
    Returns dict with diff_obs, CI, and a two-sided bootstrap p-value.
    """
    m_a = compute_metrics_for_group(preds_a)
    m_b = compute_metrics_for_group(preds_b)
    diff_obs = float(m_a.get(metric_key, np.nan) - m_b.get(metric_key, np.nan))

    preds_by_student_a: dict[str, list[PredictionResult]] = {}
    preds_by_student_b: dict[str, list[PredictionResult]] = {}
    for p in preds_a:
        sid = str(p.metadata.get("student_id", ""))
        preds_by_student_a.setdefault(sid, []).append(p)
    for p in preds_b:
        sid = str(p.metadata.get("student_id", ""))
        preds_by_student_b.setdefault(sid, []).append(p)

    students = sorted(set(preds_by_student_a).intersection(preds_by_student_b))
    n_students = len(students)
    if n_students < 2:
        return {
            "diff_obs": diff_obs,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n_students": n_students,
        }

    boot_diffs: list[float] = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(students, size=n_students, replace=True)
        boot_a: list[PredictionResult] = []
        boot_b: list[PredictionResult] = []
        for sid in sampled:
            boot_a.extend(preds_by_student_a[sid])
            boot_b.extend(preds_by_student_b[sid])
        ma = compute_metrics_for_group(boot_a).get(metric_key, np.nan)
        mb = compute_metrics_for_group(boot_b).get(metric_key, np.nan)
        d = float(ma - mb) if np.isfinite(ma) and np.isfinite(mb) else np.nan
        if np.isfinite(d):
            boot_diffs.append(d)

    vals = np.asarray(boot_diffs, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            "diff_obs": diff_obs,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n_students": n_students,
        }

    lo = float(np.percentile(vals, 100 * (alpha / 2)))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))

    # Two-sided bootstrap p-value for H0: diff == 0.
    # Use +1 smoothing to avoid returning 0.
    n = len(vals)
    p_lo = (int((vals <= 0).sum()) + 1) / (n + 1)
    p_hi = (int((vals >= 0).sum()) + 1) / (n + 1)
    p_value = float(min(1.0, 2.0 * min(p_lo, p_hi)))

    return {
        "diff_obs": diff_obs,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "n_students": n_students,
    }


def _run_metric_pairwise_tests_across_questions(
    predictions: list[PredictionResult],
    *,
    alpha: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Exploratory: question-pair metric differences (ECE/AUC) via paired student-block bootstrap.

    Notes:
    - Questions are a fixed set (n=4). This quantifies uncertainty over students, not generalization to new questions.
    - Uses the same student resample for both questions (paired bootstrap).
    """
    by_q: dict[str, list[PredictionResult]] = {}
    for p in predictions:
        q = str(p.metadata.get("question_id", ""))
        if q == "":
            continue
        by_q.setdefault(q, []).append(p)

    q_levels = sorted(by_q.keys())
    if len(q_levels) < 2:
        return pd.DataFrame()

    metric_keys = ["ece", "auc_roc"]
    rows: list[dict] = []
    for metric_key in metric_keys:
        for i in range(len(q_levels)):
            for j in range(i + 1, len(q_levels)):
                qa, qb = q_levels[i], q_levels[j]
                res = _bootstrap_metric_diff_student_block(
                    by_q[qa],
                    by_q[qb],
                    metric_key=metric_key,
                    n_bootstrap=int(n_bootstrap),
                    alpha=alpha,
                    rng=rng,
                )
                rows.append(
                    {
                        "scope": "across_questions",
                        "question_id": "ALL",
                        "metric": metric_key,
                        "group_a": qa,
                        "group_b": qb,
                        "diff_obs": float(res["diff_obs"]),
                        "ci_low": float(res["ci_low"]),
                        "ci_high": float(res["ci_high"]),
                        "p_value": float(res["p_value"]),
                        "n_students": int(res["n_students"]),
                        "n_bootstrap": int(n_bootstrap),
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    pvals = out["p_value"].to_numpy(dtype=float)
    p_adj, significant = benjamini_hochberg(pvals, alpha=alpha)
    out["p_adjusted"] = p_adj
    out["significant"] = significant
    out["sig"] = out["p_adjusted"].apply(lambda p: _format_sig(p, alpha))
    out["p_adjustment_within_family"] = "BH_within_question_metric_family"
    out["p_adjusted_primary"] = out["p_adjusted"]
    out["sig_primary"] = out["sig"]
    out["p_adjustment_primary"] = out["p_adjustment_within_family"]
    return out


def _flag_problematic_groups(
    results_df: pd.DataFrame,
    *,
    ece_threshold: Optional[float],
    accuracy_threshold: Optional[float],
) -> pd.DataFrame:
    """
    Flag groups that look problematic based on threshold rules.

    If CIs are present, uses conservative rules:
      - high_ece if ece_ci_low > threshold
      - low_accuracy if accuracy_ci_high < threshold
    Otherwise uses point estimates.
    """
    if ece_threshold is None and accuracy_threshold is None:
        return pd.DataFrame()

    df = results_df.copy()
    df["group"] = df["group"].astype(str)

    high_ece = np.zeros(len(df), dtype=bool)
    low_acc = np.zeros(len(df), dtype=bool)

    if ece_threshold is not None:
        if "ece_ci_low" in df.columns:
            high_ece = pd.to_numeric(df["ece_ci_low"], errors="coerce").to_numpy() > float(ece_threshold)
        else:
            high_ece = pd.to_numeric(df["ece"], errors="coerce").to_numpy() > float(ece_threshold)

    if accuracy_threshold is not None:
        if "accuracy_ci_high" in df.columns:
            low_acc = pd.to_numeric(df["accuracy_ci_high"], errors="coerce").to_numpy() < float(accuracy_threshold)
        else:
            low_acc = pd.to_numeric(df["accuracy"], errors="coerce").to_numpy() < float(accuracy_threshold)

    df["high_ece"] = high_ece
    df["low_accuracy"] = low_acc
    out = df[df["high_ece"] | df["low_accuracy"]].copy()
    if len(out) == 0:
        return out
    if "ece" in out.columns:
        out = out.sort_values(["high_ece", "ece"], ascending=[False, False])
    return out


def _run_metric_pairwise_tests_within_question(
    predictions: list[PredictionResult],
    *,
    alpha: float,
    pairwise_scope: str,
    pairwise_top_k: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Secondary Step-4 inference for non-row-additive metrics (ECE/AUC) via student-block bootstrap.

    Rubric comparisons are always restricted to within-question pairs due to rubric nesting.
    """
    # Build mapping (question -> rubric -> predictions)
    by_qr: dict[str, dict[str, list[PredictionResult]]] = {}
    for p in predictions:
        q = str(p.metadata.get("question_id", ""))
        r = str(p.metadata.get("rubric_id", ""))
        if q == "" or r == "":
            continue
        by_qr.setdefault(q, {}).setdefault(r, []).append(p)

    metric_keys = ["ece", "auc_roc"]

    # Build the task list first so we can show a single progress bar.
    tasks: list[tuple[str, str, str, str]] = []  # (question_id, metric_key, rubric_a, rubric_b)
    for q, by_r in sorted(by_qr.items(), key=lambda x: x[0]):
        rubrics = sorted(by_r.keys())
        if len(rubrics) < 2:
            continue

        # Pre-compute per-rubric metric point estimates for pair selection.
        rubric_point: dict[str, dict[str, float]] = {}
        for r in rubrics:
            m = compute_metrics_for_group(by_r[r])
            rubric_point[r] = {mk: float(m.get(mk, np.nan)) for mk in metric_keys}

        for metric_key in metric_keys:
            if pairwise_scope == "none":
                continue
            if pairwise_scope == "all_within_question":
                for i in range(len(rubrics)):
                    for j in range(i + 1, len(rubrics)):
                        tasks.append((q, metric_key, rubrics[i], rubrics[j]))
            elif pairwise_scope == "topk_within_question":
                diffs: list[tuple[str, str, float]] = []
                for i in range(len(rubrics)):
                    for j in range(i + 1, len(rubrics)):
                        a, b = rubrics[i], rubrics[j]
                        va = rubric_point[a].get(metric_key, np.nan)
                        vb = rubric_point[b].get(metric_key, np.nan)
                        if np.isfinite(va) and np.isfinite(vb):
                            diffs.append((a, b, abs(va - vb)))
                diffs.sort(key=lambda x: x[2], reverse=True)
                for a, b, _ in diffs[: max(int(pairwise_top_k), 0)]:
                    tasks.append((q, metric_key, a, b))
            else:
                raise ValueError(f"Unknown metric pairwise_scope={pairwise_scope!r}")

    rows: list[dict] = []
    iterator = tasks
    if len(tasks) > 1:
        iterator = tqdm(tasks, desc="Metric tests (ECE/AUC)", unit="test", dynamic_ncols=True)

    for q, metric_key, ra, rb in iterator:
        by_r = by_qr.get(q, {})
        preds_a = by_r.get(ra, [])
        preds_b = by_r.get(rb, [])
        if not preds_a or not preds_b:
            continue

        res = _bootstrap_metric_diff_student_block(
            preds_a,
            preds_b,
            metric_key=metric_key,
            n_bootstrap=int(n_bootstrap),
            alpha=alpha,
            rng=rng,
        )
        rows.append(
            {
                "scope": "within_question",
                "question_id": q,
                "metric": metric_key,
                "rubric_a": ra,
                "rubric_b": rb,
                "diff_obs": float(res["diff_obs"]),
                "ci_low": float(res["ci_low"]),
                "ci_high": float(res["ci_high"]),
                "p_value": float(res["p_value"]),
                "n_students": int(res["n_students"]),
                "n_bootstrap": int(n_bootstrap),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    pvals = out["p_value"].to_numpy(dtype=float)
    p_adj, significant = benjamini_hochberg(pvals, alpha=alpha)
    out["p_adjusted"] = p_adj
    out["significant"] = significant
    out["sig"] = out["p_adjusted"].apply(lambda p: _format_sig(p, alpha))
    out["p_adjustment_within_family"] = "BH_within_rubric_metric_family"
    out["p_adjusted_primary"] = out["p_adjusted"]
    out["sig_primary"] = out["sig"]
    out["p_adjustment_primary"] = out["p_adjustment_within_family"]
    return out


def _run_step4_metric_tests(
    predictions: list[PredictionResult],
    *,
    alpha: float,
    pairwise_scope: str,
    pairwise_top_k: int,
    include_question_tests: bool,
    skip_question_tests: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
    no_save: bool,
    model_name: str,
    csv_timestamp: str,
    output_manager,
    step4_tables_dir: Path,
    legacy_analysis_dir: Path,
    file_suffix: str,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Secondary Step-4 inference for non-row-additive metrics (ECE/AUC).

    - Rubric comparisons are always within-question (nesting-safe).
    - Optional question-pair tests use paired student-block bootstrap (exploratory; n=4 questions).
    """
    metric_df = pd.DataFrame()
    q_metric_df = pd.DataFrame()

    print("\n" + "=" * 80)
    print("SECONDARY METRIC TESTS: ECE/AUC DIFFERENCES (STUDENT-BLOCK BOOTSTRAP)")
    print("=" * 80)
    print("Rubric tests are restricted to within-question comparisons due to rubric nesting.")

    try:
        metric_df = _run_metric_pairwise_tests_within_question(
            predictions,
            alpha=alpha,
            pairwise_scope=pairwise_scope,
            pairwise_top_k=pairwise_top_k,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
    except Exception as e:
        print(f"Warning: failed to run rubric metric-level tests: {e}")
        metric_df = pd.DataFrame()

    if len(metric_df) == 0:
        print("No rubric metric-level results produced (insufficient data or all metrics undefined).")
    else:
        show_cols = [
            "question_id",
            "metric",
            "rubric_a",
            "rubric_b",
            "diff_obs",
            "ci_low",
            "ci_high",
            "p_value",
            "p_adjusted_primary",
            "sig_primary",
            "n_students",
        ]
        show_cols = [c for c in show_cols if c in metric_df.columns]
        print("\nTop rubric metric differences (by adjusted p-value):")
        print(metric_df.sort_values(["p_adjusted_primary", "p_value"]).head(50)[show_cols].to_string(index=False))

        if not no_save:
            out_path = step4_tables_dir / f"step4_metric_pairwise_tests_{model_name}_{method}{file_suffix}.csv"
            legacy_path = legacy_analysis_dir / f"step4_metric_pairwise_tests_{model_name}_{csv_timestamp}.csv"
            _save_table(
                metric_df,
                out_path,
                output_manager=output_manager,
                artifact="metric_pairwise_tests",
                model=model_name,
                method=method,
                legacy_path=legacy_path,
            )
            print(f"\nMetric-level test results saved to: {out_path}")

    if include_question_tests and not skip_question_tests:
        print("\n" + "=" * 80)
        print("SECONDARY METRIC TESTS (EXPLORATORY): ECE/AUC DIFFERENCES ACROSS QUESTIONS (PAIRED STUDENT BOOTSTRAP)")
        print("=" * 80)
        try:
            q_metric_df = _run_metric_pairwise_tests_across_questions(
                predictions,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
        except Exception as e:
            print(f"Warning: failed to run question metric tests: {e}")
            q_metric_df = pd.DataFrame()

        if len(q_metric_df) == 0:
            print("No question metric results produced (insufficient data or all metrics undefined).")
        else:
            show_cols = [
                "metric",
                "group_a",
                "group_b",
                "diff_obs",
                "ci_low",
                "ci_high",
                "p_value",
                "p_adjusted_primary",
                "sig_primary",
                "n_students",
            ]
            show_cols = [c for c in show_cols if c in q_metric_df.columns]
            print("\nQuestion metric differences (by adjusted p-value):")
            print(q_metric_df.sort_values(["p_adjusted_primary", "p_value"]).head(50)[show_cols].to_string(index=False))

            if not no_save:
                out_path = step4_tables_dir / f"step4_question_metric_pairwise_tests_{model_name}_{method}{file_suffix}.csv"
                legacy_path = legacy_analysis_dir / f"step4_question_metric_pairwise_tests_{model_name}_{csv_timestamp}.csv"
                _save_table(
                    q_metric_df,
                    out_path,
                    output_manager=output_manager,
                    artifact="question_metric_pairwise_tests",
                    model=model_name,
                    method=method,
                    legacy_path=legacy_path,
                )
                print(f"\nQuestion metric test results saved to: {out_path}")

    return metric_df, q_metric_df


def _run_rubric_lrt_tests_within_question(
    df_out: pd.DataFrame,
    *,
    alpha: float,
    pairwise_scope: str,
    pairwise_top_k: int,
    compute_effect_ci: bool,
    effect_n_bootstrap: int,
    rng: np.random.Generator,
    include_calibration_error: bool,
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> pd.DataFrame:
    """
    Primary Step-4 inference: within-question rubric heterogeneity tests via mixed-model LRT.

    Omnibus (per question, per outcome):
      y ~ 1 + (1|student_id)  vs  y ~ factor(rubric_id) + (1|student_id)

    Pairwise (within question, optional):
      subset to two rubrics, test y ~ I(rubric_id==rB) + (1|student_id)
    """
    required = {"student_id", "question_id", "rubric_id"}
    if not required.issubset(df_out.columns):
        raise ValueError(f"df_out missing required columns: {sorted(required - set(df_out.columns))}")

    outcomes = [
        ("raw_confidence", False, "gaussian"),
        ("brier", False, "gaussian"),
        ("is_correct", True, "binomial"),
    ]
    if include_calibration_error and "calibration_error" in df_out.columns:
        outcomes.insert(2, ("calibration_error", False, "gaussian"))

    rows: list[dict] = []
    for q, df_q in df_out.groupby("question_id", sort=True):
        df_q = df_q.dropna(subset=["student_id", "rubric_id"])
        rubrics = sorted(df_q["rubric_id"].astype(str).unique().tolist())
        if len(rubrics) < 2:
            continue

        for outcome_col, is_glmm, family in outcomes:
            df_qo = df_q.dropna(subset=[outcome_col])
            if df_qo["student_id"].nunique() < 2:
                continue

            df_fit = df_qo[["student_id", "rubric_id", outcome_col]].copy()
            df_fit = df_fit.rename(columns={outcome_col: "y"})
            # Ensure R treats IDs as categorical (factors), not numeric covariates.
            df_fit["student_id"] = df_fit["student_id"].astype(str)
            df_fit["rubric_id"] = df_fit["rubric_id"].astype(str)

            formula_null = "y ~ 1 + (1|student_id)"
            formula_full = "y ~ factor(rubric_id) + (1|student_id)"

            es = _rubric_mean_range_effect_size(
                df_qo,
                outcome_col=outcome_col,
                rubrics=rubrics,
                alpha=alpha,
                compute_ci=bool(compute_effect_ci),
                n_bootstrap=int(effect_n_bootstrap),
                rng=rng,
            )

            # Omnibus rubric heterogeneity (within-question)
            try:
                res = fit_lme4_lrt_from_formula(
                    df_fit,
                    formula_null=formula_null,
                    formula_full=formula_full,
                    is_glmm=is_glmm,
                    family=family,
                    coef_name=None,
                    glmer_nAGQ=glmer_nAGQ,
                    optimizer=optimizer,
                    maxfun=maxfun,
                )
            except Exception as e:
                res = {
                    "p_value_lrt": np.nan,
                    "chisq": np.nan,
                    "df_diff": np.nan,
                    "beta": np.nan,
                    "se": np.nan,
                    "converged": False,
                    "singular": False,
                    "backend": "lme4_failed",
                    "warnings": [f"lme4 LRT failed: {e}"],
                }

            rows.append(
                {
                    "scope": "within_question",
                    "question_id": q,
                    "outcome": outcome_col,
                    "test": "omnibus_rubric",
                    "rubric_a": "",
                    "rubric_b": "",
                    "n_rows": int(len(df_fit)),
                    "n_students": int(df_fit["student_id"].nunique()),
                    "n_rubrics": int(len(rubrics)),
                    "family": family,
                    "formula_null": formula_null,
                    "formula_full": formula_full,
                    "odds_ratio": np.nan,
                    "beta": float(res.get("beta", np.nan)),
                    "se": float(res.get("se", np.nan)),
                    "chisq": float(res.get("chisq", np.nan)),
                    "backend": str(res.get("backend", "")),
                    "df_diff": float(res.get("df_diff", np.nan))
                    if np.isfinite(float(res.get("df_diff", np.nan)))
                    else float(len(rubrics) - 1),
                    "p_value": float(res.get("p_value_lrt", np.nan)),
                    "converged": bool(res.get("converged", False)),
                    "singular": bool(res.get("singular", False)),
                    "warnings": "; ".join([str(w) for w in res.get("warnings", []) if str(w)]),
                    **es,
                }
            )

            # Pairwise rubric comparisons (optional)
            if pairwise_scope == "none":
                continue

            pairs: list[tuple[str, str]]
            if pairwise_scope == "all_within_question":
                pairs = []
                for i in range(len(rubrics)):
                    for j in range(i + 1, len(rubrics)):
                        pairs.append((rubrics[i], rubrics[j]))
            elif pairwise_scope == "topk_within_question":
                pairs = _select_topk_pairs_by_mean_diff(df_qo, outcome_col, rubrics, pairwise_top_k)
            else:
                raise ValueError(f"Unknown pairwise_scope={pairwise_scope!r}")

            for ra, rb in pairs:
                df_pair = df_qo[df_qo["rubric_id"].astype(str).isin([ra, rb])].copy()
                df_pair = df_pair.dropna(subset=[outcome_col])
                if df_pair["student_id"].nunique() < 2:
                    continue
                df_pair["rubric_is_b"] = (df_pair["rubric_id"].astype(str) == rb).astype(int)
                df_pair_fit = df_pair[["student_id", "rubric_is_b", outcome_col]].rename(columns={outcome_col: "y"})

                formula_null_pair = "y ~ 1 + (1|student_id)"
                formula_full_pair = "y ~ rubric_is_b + (1|student_id)"
                try:
                    res_p = fit_lme4_lrt_from_formula(
                        df_pair_fit,
                        formula_null=formula_null_pair,
                        formula_full=formula_full_pair,
                        is_glmm=is_glmm,
                        family=family,
                        coef_name="rubric_is_b",
                        glmer_nAGQ=glmer_nAGQ,
                        optimizer=optimizer,
                        maxfun=maxfun,
                    )
                except Exception as e:
                    res_p = {
                        "p_value_lrt": np.nan,
                        "chisq": np.nan,
                        "df_diff": np.nan,
                        "beta": np.nan,
                        "se": np.nan,
                        "converged": False,
                        "singular": False,
                        "backend": "lme4_failed",
                        "warnings": [f"lme4 LRT failed: {e}"],
                    }

                beta = float(res_p.get("beta", np.nan))
                se = float(res_p.get("se", np.nan))
                or_val = float(np.exp(beta)) if is_glmm and np.isfinite(beta) else np.nan

                rows.append(
                    {
                        "scope": "within_question",
                        "question_id": q,
                        "outcome": outcome_col,
                        "test": "pairwise_rubric",
                        "rubric_a": ra,
                        "rubric_b": rb,
                        "n_rows": int(len(df_pair_fit)),
                        "n_students": int(df_pair_fit["student_id"].nunique()),
                        "n_rubrics": 2,
                        "family": family,
                        "formula_null": formula_null_pair,
                        "formula_full": formula_full_pair,
                        "beta": beta,
                        "se": se,
                        "odds_ratio": or_val,
                        "chisq": float(res_p.get("chisq", np.nan)),
                        "df_diff": float(res_p.get("df_diff", np.nan))
                        if np.isfinite(float(res_p.get("df_diff", np.nan)))
                        else 1.0,
                        "p_value": float(res_p.get("p_value_lrt", np.nan)),
                        "converged": bool(res_p.get("converged", False)),
                        "singular": bool(res_p.get("singular", False)),
                        "warnings": "; ".join([str(w) for w in res_p.get("warnings", []) if str(w)]),
                        "rubric_mean_range": np.nan,
                        "rubric_mean_range_ci_low": np.nan,
                        "rubric_mean_range_ci_high": np.nan,
                        "rubric_min": "",
                        "rubric_max": "",
                        "mean_min": np.nan,
                        "mean_max": np.nan,
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # Global BH correction within the LRT family.
    pvals = out["p_value"].to_numpy(dtype=float)
    p_adj, significant = benjamini_hochberg(pvals, alpha=alpha)
    out["p_adjusted"] = p_adj
    out["significant"] = significant
    out["sig"] = out["p_adjusted"].apply(lambda p: _format_sig(p, alpha))
    out["p_adjustment_within_family"] = "BH_within_rubric_lrt_family"
    out["p_adjusted_primary"] = out["p_adjusted"]
    out["sig_primary"] = out["sig"]
    out["p_adjustment_primary"] = out["p_adjustment_within_family"]
    return out


def _run_calibration_regression_rubric_slopes_within_question(
    df_out: pd.DataFrame,
    *,
    alpha: float,
    compute_ci: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> pd.DataFrame:
    """
    Primary calibration tests: rubric differences in calibration intercept and slope within each question.

    LRTs:
      Intercept: is_correct ~ logit_conf_c + (1|student_id)
                 vs is_correct ~ logit_conf_c + factor(rubric_id) + (1|student_id)
      Slope:     is_correct ~ logit_conf_c + factor(rubric_id) + (1|student_id)
                 vs is_correct ~ logit_conf_c * factor(rubric_id) + (1|student_id)
    """
    required = {"student_id", "question_id", "rubric_id", "raw_confidence", "is_correct"}
    if not required.issubset(df_out.columns):
        raise ValueError(f"df_out missing required columns: {sorted(required - set(df_out.columns))}")

    rows: list[dict] = []
    for q, df_q in df_out.groupby("question_id", sort=True):
        df_q = df_q.dropna(subset=["student_id", "rubric_id", "raw_confidence", "is_correct"]).copy()
        rubrics = sorted(df_q["rubric_id"].astype(str).unique().tolist())
        if len(rubrics) < 2:
            continue

        conf = pd.to_numeric(df_q["raw_confidence"], errors="coerce").to_numpy(dtype=float)
        is_corr = pd.to_numeric(df_q["is_correct"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(conf) & np.isfinite(is_corr)
        if mask.sum() < 5:
            continue

        conf = np.clip(conf[mask], 0.001, 0.999)
        is_corr = is_corr[mask]
        student_ids = df_q["student_id"].to_numpy(dtype=str)[mask]
        rubric_ids = df_q["rubric_id"].to_numpy(dtype=str)[mask]

        logit_conf = np.log(conf / (1.0 - conf))
        logit_conf_mean = float(np.mean(logit_conf))
        logit_conf_c = logit_conf - logit_conf_mean

        df_fit = pd.DataFrame(
            {
                "y": is_corr.astype(int),
                "logit_conf_c": logit_conf_c,
                "student_id": student_ids.astype(str),
                "rubric_id": rubric_ids.astype(str),
            }
        )
        if df_fit["student_id"].nunique() < 2 or df_fit["rubric_id"].nunique() < 2:
            continue

        formula_base = "y ~ logit_conf_c + (1|student_id)"
        formula_null = "y ~ logit_conf_c + factor(rubric_id) + (1|student_id)"
        formula_full = "y ~ logit_conf_c * factor(rubric_id) + (1|student_id)"

        try:
            res_intercept = fit_lme4_lrt_from_formula(
                df_fit,
                formula_null=formula_base,
                formula_full=formula_null,
                is_glmm=True,
                family="binomial",
                coef_name=None,
                glmer_nAGQ=glmer_nAGQ,
                optimizer=optimizer,
                maxfun=maxfun,
            )
        except Exception as e:
            res_intercept = {
                "p_value_lrt": np.nan,
                "chisq": np.nan,
                "df_diff": np.nan,
                "beta": np.nan,
                "se": np.nan,
                "converged": False,
                "singular": False,
                "backend": "lme4_failed",
                "warnings": [f"lme4 LRT failed: {e}"],
            }

        try:
            res = fit_lme4_lrt_from_formula(
                df_fit,
                formula_null=formula_null,
                formula_full=formula_full,
                is_glmm=True,
                family="binomial",
                coef_name=None,
                glmer_nAGQ=glmer_nAGQ,
                optimizer=optimizer,
                maxfun=maxfun,
            )
        except Exception as e:
            res = {
                "p_value_lrt": np.nan,
                "chisq": np.nan,
                "df_diff": np.nan,
                "beta": np.nan,
                "se": np.nan,
                "converged": False,
                "singular": False,
                "backend": "lme4_failed",
                "warnings": [f"lme4 LRT failed: {e}"],
            }

        slope_info = {
            "slope_min": np.nan,
            "slope_max": np.nan,
            "slope_range": np.nan,
            "slope_range_ci_low": np.nan,
            "slope_range_ci_high": np.nan,
            "rubric_min": "",
            "rubric_max": "",
        }
        slopes_map: dict[str, float] = {}
        try:
            fe = fit_lme4_glmer_fixef_vcov(
                df_fit,
                formula_full=formula_full,
                glmer_nAGQ=glmer_nAGQ,
                optimizer=optimizer,
                maxfun=maxfun,
            )
            slopes_map = _calibration_slopes_by_rubric(
                coef_names=list(fe.get("coef_names", [])),
                coef_values=np.asarray(fe.get("coef_values", np.array([])), dtype=float),
                rubrics=rubrics,
            )
            n_draws = max(200, min(2000, int(n_bootstrap)))
            slope_info = _calibration_slope_range_effect_size(
                coef_names=list(fe.get("coef_names", [])),
                coef_values=np.asarray(fe.get("coef_values", np.array([])), dtype=float),
                vcov=np.asarray(fe.get("vcov", np.array([[]])), dtype=float),
                rubrics=rubrics,
                alpha=alpha,
                compute_ci=bool(compute_ci),
                n_draws=n_draws,
                rng=rng,
            )
        except Exception:
            pass

        rows.append(
            {
                "scope": "within_question",
                "question_id": q,
                "outcome": "calibration_intercept",
                "test": "omnibus_calibration_intercept",
                "n_rows": int(len(df_fit)),
                "n_students": int(df_fit["student_id"].nunique()),
                "n_rubrics": int(len(rubrics)),
                "formula_null": formula_base,
                "formula_full": formula_null,
                "chisq": float(res_intercept.get("chisq", np.nan)),
                "backend": str(res_intercept.get("backend", "")),
                "df_diff": float(res_intercept.get("df_diff", np.nan))
                if np.isfinite(float(res_intercept.get("df_diff", np.nan)))
                else float(len(rubrics) - 1),
                "p_value": float(res_intercept.get("p_value_lrt", np.nan)),
                "converged": bool(res_intercept.get("converged", False)),
                "singular": bool(res_intercept.get("singular", False)),
                "warnings": "; ".join([str(w) for w in res_intercept.get("warnings", []) if str(w)]),
                "logit_conf_mean": logit_conf_mean,
                "rubric_slopes": "",
                "slope_min": np.nan,
                "slope_max": np.nan,
                "slope_range": np.nan,
                "slope_range_ci_low": np.nan,
                "slope_range_ci_high": np.nan,
                "rubric_min": "",
                "rubric_max": "",
            }
        )

        rows.append(
            {
                "scope": "within_question",
                "question_id": q,
                "outcome": "calibration_slope",
                "test": "omnibus_calibration_slope",
                "n_rows": int(len(df_fit)),
                "n_students": int(df_fit["student_id"].nunique()),
                "n_rubrics": int(len(rubrics)),
                "formula_null": formula_null,
                "formula_full": formula_full,
                "chisq": float(res.get("chisq", np.nan)),
                "backend": str(res.get("backend", "")),
                "df_diff": float(res.get("df_diff", np.nan))
                if np.isfinite(float(res.get("df_diff", np.nan)))
                else float(len(rubrics) - 1),
                "p_value": float(res.get("p_value_lrt", np.nan)),
                "converged": bool(res.get("converged", False)),
                "singular": bool(res.get("singular", False)),
                "warnings": "; ".join([str(w) for w in res.get("warnings", []) if str(w)]),
                "logit_conf_mean": logit_conf_mean,
                "rubric_slopes": ", ".join([f"{k}={v:.4f}" for k, v in slopes_map.items()]),
                **slope_info,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    pvals = out["p_value"].to_numpy(dtype=float)
    p_adj, significant = benjamini_hochberg(pvals, alpha=alpha)
    out["p_adjusted"] = p_adj
    out["significant"] = significant
    out["sig"] = out["p_adjusted"].apply(lambda p: _format_sig(p, alpha))
    out["p_adjustment_within_family"] = "BH_within_calibration_regression_family"
    out["p_adjusted_primary"] = out["p_adjusted"]
    out["sig_primary"] = out["sig"]
    out["p_adjustment_primary"] = out["p_adjustment_within_family"]
    return out


def _run_question_lrt_tests_across_questions(
    df_out: pd.DataFrame,
    *,
    alpha: float,
    include_calibration_error: bool,
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> pd.DataFrame:
    """
    Exploratory question-level group tests (only 4 questions).

    Omnibus (per outcome):
      If #questions is small -> fixed effects:
        y ~ 1 + (1|student_id)  vs  y ~ factor(question_id) + (1|student_id)
      If #questions is large -> random intercept:
        y ~ 1 + (1|student_id)  vs  y ~ 1 + (1|question_id) + (1|student_id)

    Pairwise (all pairs, per outcome):
      y ~ question_is_b + (1|student_id)
    """
    required = {"student_id", "question_id"}
    if not required.issubset(df_out.columns):
        raise ValueError(f"df_out missing required columns: {sorted(required - set(df_out.columns))}")

    outcomes = [
        ("raw_confidence", False, "gaussian"),
        ("brier", False, "gaussian"),
        ("is_correct", True, "binomial"),
    ]
    if include_calibration_error and "calibration_error" in df_out.columns:
        outcomes.insert(2, ("calibration_error", False, "gaussian"))

    q_levels = sorted(df_out["question_id"].astype(str).dropna().unique().tolist())
    if len(q_levels) < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for outcome_col, is_glmm, family in outcomes:
        df_qo = df_out.dropna(subset=["student_id", "question_id", outcome_col]).copy()
        if df_qo["student_id"].nunique() < 2 or df_qo["question_id"].nunique() < 2:
            continue

        df_fit = df_qo[["student_id", "question_id", outcome_col]].rename(columns={outcome_col: "y"})
        # Ensure R treats IDs as categorical (factors), not numeric covariates.
        df_fit["student_id"] = df_fit["student_id"].astype(str)
        df_fit["question_id"] = df_fit["question_id"].astype(str)

        formula_null = "y ~ 1 + (1|student_id)"
        include_question_fe, include_question_re, _, _ = _select_question_effects(
            df_fit["question_id"].to_numpy(),
            feature_type="question",
        )
        if include_question_re:
            formula_full = "y ~ 1 + (1|question_id) + (1|student_id)"
            df_diff_fallback = 1.0
        else:
            formula_full = "y ~ factor(question_id) + (1|student_id)"
            df_diff_fallback = float(len(q_levels) - 1)
        try:
            res = fit_lme4_lrt_from_formula(
                df_fit,
                formula_null=formula_null,
                formula_full=formula_full,
                is_glmm=is_glmm,
                family=family,
                coef_name=None,
                glmer_nAGQ=glmer_nAGQ,
                optimizer=optimizer,
                maxfun=maxfun,
            )
        except Exception as e:
            res = {
                "p_value_lrt": np.nan,
                "chisq": np.nan,
                "df_diff": np.nan,
                "beta": np.nan,
                "se": np.nan,
                "converged": False,
                "singular": False,
                "backend": "lme4_failed",
                "warnings": [f"lme4 LRT failed: {e}"],
            }

        rows.append(
                {
                    "scope": "across_questions",
                    "question_id": "ALL",
                    "outcome": outcome_col,
                    "test": "omnibus_question",
                    "group_a": "",
                    "group_b": "",
                    "n_rows": int(len(df_fit)),
                    "n_students": int(df_fit["student_id"].nunique()),
                    "n_questions": int(df_fit["question_id"].nunique()),
                    "family": family,
                    "formula_null": formula_null,
                    "formula_full": formula_full,
                    "odds_ratio": np.nan,
                    "beta": float(res.get("beta", np.nan)),
                    "se": float(res.get("se", np.nan)),
                    "chisq": float(res.get("chisq", np.nan)),
                    "backend": str(res.get("backend", "")),
                    "df_diff": float(res.get("df_diff", np.nan))
                    if np.isfinite(float(res.get("df_diff", np.nan)))
                    else df_diff_fallback,
                    "p_value": float(res.get("p_value_lrt", np.nan)),
                    "converged": bool(res.get("converged", False)),
                    "singular": bool(res.get("singular", False)),
                    "warnings": "; ".join([str(w) for w in res.get("warnings", []) if str(w)]),
                }
            )

        # Pairwise tests
        for i in range(len(q_levels)):
            for j in range(i + 1, len(q_levels)):
                qa, qb = q_levels[i], q_levels[j]
                df_pair = df_fit[df_fit["question_id"].astype(str).isin([qa, qb])].copy()
                if df_pair["student_id"].nunique() < 2:
                    continue
                df_pair["question_is_b"] = (df_pair["question_id"].astype(str) == qb).astype(int)
                df_pair_fit = df_pair[["student_id", "question_is_b", "y"]]
                formula_null_pair = "y ~ 1 + (1|student_id)"
                formula_full_pair = "y ~ question_is_b + (1|student_id)"
                try:
                    res_p = fit_lme4_lrt_from_formula(
                        df_pair_fit,
                        formula_null=formula_null_pair,
                        formula_full=formula_full_pair,
                        is_glmm=is_glmm,
                        family=family,
                        coef_name="question_is_b",
                        glmer_nAGQ=glmer_nAGQ,
                        optimizer=optimizer,
                        maxfun=maxfun,
                    )
                except Exception as e:
                    res_p = {
                        "p_value_lrt": np.nan,
                        "chisq": np.nan,
                        "df_diff": np.nan,
                        "beta": np.nan,
                        "se": np.nan,
                        "converged": False,
                        "singular": False,
                        "backend": "lme4_failed",
                        "warnings": [f"lme4 LRT failed: {e}"],
                    }
                rows.append(
                    {
                        "scope": "across_questions",
                        "question_id": "ALL",
                        "outcome": outcome_col,
                        "test": "pairwise_question",
                        "group_a": qa,
                        "group_b": qb,
                        "n_rows": int(len(df_pair_fit)),
                        "n_students": int(df_pair_fit["student_id"].nunique()),
                        "n_questions": 2,
                        "family": family,
                        "formula_null": formula_null_pair,
                        "formula_full": formula_full_pair,
                        "beta": float(res_p.get("beta", np.nan)),
                        "se": float(res_p.get("se", np.nan)),
                        "odds_ratio": float(np.exp(float(res_p.get("beta", np.nan))))
                        if is_glmm and np.isfinite(float(res_p.get("beta", np.nan)))
                        else np.nan,
                        "chisq": float(res_p.get("chisq", np.nan)),
                        "backend": str(res_p.get("backend", "")),
                        "df_diff": 1.0,
                        "p_value": float(res_p.get("p_value_lrt", np.nan)),
                        "converged": bool(res_p.get("converged", False)),
                        "singular": bool(res_p.get("singular", False)),
                        "warnings": "; ".join([str(w) for w in res_p.get("warnings", []) if str(w)]),
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    pvals = out["p_value"].to_numpy(dtype=float)
    p_adj, significant = benjamini_hochberg(pvals, alpha=alpha)
    out["p_adjusted"] = p_adj
    out["significant"] = significant
    out["sig"] = out["p_adjusted"].apply(lambda p: _format_sig(p, alpha))
    out["p_adjustment_within_family"] = "BH_within_question_lrt_family"
    out["p_adjusted_primary"] = out["p_adjusted"]
    out["sig_primary"] = out["sig"]
    out["p_adjustment_primary"] = out["p_adjustment_within_family"]
    return out


def _should_run_scope(scope: str, test_name: str) -> bool:
    if scope == "none":
        return False
    if scope == "all":
        return True
    if scope == "omnibus":
        return test_name.startswith("omnibus_")
    if scope == "pairwise":
        return test_name.startswith("pairwise_")
    raise ValueError(f"Unknown scope={scope!r}")


def _stable_int_seed(key: str, base: int) -> int:
    import zlib

    return int(base) + (zlib.crc32(key.encode("utf-8")) % 1000000)


def _install_warnings_logger(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        timestamp = datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"[{timestamp}] {category.__name__}: {message} ({Path(filename).name}:{lineno})\n"
            )

    warnings.simplefilter("default")
    warnings.showwarning = _showwarning


def load_path_by_index(
    idx: int,
    folder: Path | str,
    prefix: str,
    dataset: str = "ricechem",
    *,
    filter_by_dataset: bool = True,
) -> Path:
    """Load CSV path by index (0 = most recent), filtered by dataset tag."""
    folder = Path(folder)
    paths = sorted(folder.glob(f"{prefix}_*.csv"))
    if filter_by_dataset:
        dataset = dataset.lower()
        other_datasets = {"ricechem", "scientsbank", "beetle"} - {dataset}
        if dataset == "ricechem":
            paths = [p for p in paths if not any(f"_{d}_" in p.name for d in other_datasets)]
        else:
            paths = [p for p in paths if f"_{dataset}_" in p.name]
    if not paths:
        raise FileNotFoundError(f"No predictions CSVs found for dataset={dataset} in {folder}")
    new_idx = len(paths) - 1 - idx
    return paths[new_idx]


def load_data_from_csv(csv_path: str, smoke_n: int | None = None) -> tuple:
    """Load predictions CSV and reconstruct GradingInstance + PredictionResult objects."""
    df = pd.read_csv(csv_path)
    if smoke_n is not None:
        df = df.head(smoke_n).copy()
        print(f"Smoke test: limiting to first {len(df)} rows")
    
    instances = []
    predictions = []

    for _, row in df.iterrows():
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
            }
        )
        instances.append(instance)

        llm_response_value = row.get("llm_response", "")
        llm_responses_list = []
        response_cols = [c for c in df.columns if c.startswith("llm_response_") and c[13:].isdigit()]
        if response_cols:
            for col in sorted(response_cols):
                resp = row.get(col, "")
                if resp and pd.notna(resp):
                    llm_responses_list.append(str(resp))
            if not llm_response_value and llm_responses_list:
                llm_response_value = llm_responses_list[0]

        pred = PredictionResult(
            instance_id=row["instance_id"],
            prediction=int(row["prediction"]),
            raw_confidence=float(row["raw_confidence"]),
            ground_truth=int(row["ground_truth_label"]),
            metadata={
                "split": row.get("split", ""),
                "question_id": row.get("question_id", ""),
                "student_id": row.get("student_id", ""),
                "rubric_id": row.get("rubric_id", ""),
                "llm_response": llm_response_value,
                "llm_responses": llm_responses_list,
            },
        )
        predictions.append(pred)

    return instances, predictions, df


def compute_metrics_for_group(preds: list) -> dict:
    """Compute evaluation metrics for a group of predictions."""
    if len(preds) < 2:
        return {
            'n_instances': len(preds),
            'n_students': 0,
            'accuracy': np.nan,
            'ece': np.nan,
            'brier_score': np.nan,
            'auc_roc': np.nan,
            'mean_confidence': np.nan,
        }
    
    metrics = evaluate_predictions(preds, use_calibrated=False)
    return {
        'n_instances': metrics['n_predictions'],
        'n_students': len({str(p.metadata.get("student_id", "")) for p in preds if str(p.metadata.get("student_id", "")) != ""}),
        'accuracy': metrics['accuracy'],
        'ece': metrics['ece'],
        'brier_score': metrics['brier_score'],
        'auc_roc': metrics['auc_roc'],
        'mean_confidence': metrics.get('mean_confidence', np.mean([p.raw_confidence for p in preds])),
    }


def compute_metrics_for_group_with_ci(
    preds: list[PredictionResult],
    *,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    """
    Compute group metrics + student-block bootstrap percentile CIs.

    Resamples students with replacement within the group, carrying their full
    within-student structure (all predictions for that student within the group).
    """
    base = compute_metrics_for_group(preds)
    metric_names = ["accuracy", "ece", "brier_score", "auc_roc", "mean_confidence"]

    # Handle degenerate groups
    if base["n_instances"] < 2 or base["n_students"] < 2:
        for name in metric_names:
            base[f"{name}_ci_low"] = np.nan
            base[f"{name}_ci_high"] = np.nan
        return base

    # Block bootstrap by student_id within the group
    preds_by_student: dict[str, list[PredictionResult]] = {}
    for p in preds:
        sid = str(p.metadata.get("student_id", ""))
        if sid == "":
            continue
        preds_by_student.setdefault(sid, []).append(p)

    student_ids = list(preds_by_student.keys())
    n_students = len(student_ids)
    if n_students < 2:
        for name in metric_names:
            base[f"{name}_ci_low"] = np.nan
            base[f"{name}_ci_high"] = np.nan
        return base

    boot = {m: [] for m in metric_names}
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(student_ids, size=n_students, replace=True)
        boot_preds: list[PredictionResult] = []
        for sid in sampled:
            boot_preds.extend(preds_by_student[sid])

        m = compute_metrics_for_group(boot_preds)
        for name in metric_names:
            boot[name].append(m[name])

    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)
    for name in metric_names:
        vals = np.asarray(boot[name], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            base[f"{name}_ci_low"] = np.nan
            base[f"{name}_ci_high"] = np.nan
        else:
            base[f"{name}_ci_low"] = float(np.percentile(vals, lo))
            base[f"{name}_ci_high"] = float(np.percentile(vals, hi))

    return base


def stratify_by_category(predictions: list, category: str) -> dict:
    """
    Stratify predictions by a categorical feature (rubric_id or question_id).
    
    Returns:
        Dict mapping category values to lists of predictions
    """
    groups = {}
    for pred in predictions:
        key = pred.metadata.get(category, 'unknown')
        if key not in groups:
            groups[key] = []
        groups[key].append(pred)
    return groups


def stratify_by_feature_bins(predictions: list, features_df: pd.DataFrame, 
                             feature: str, n_bins: int) -> dict:
    """
    Stratify predictions by binning a continuous feature.
    
    Returns:
        Dict mapping bin labels to lists of predictions
    """
    # Get feature values aligned with predictions
    feature_values = features_df[feature].values
    
    # Create bins using quantiles for balanced bin sizes
    try:
        bins = pd.qcut(feature_values, q=n_bins, labels=False, duplicates='drop')
        bin_edges = pd.qcut(feature_values, q=n_bins, retbins=True, duplicates='drop')[1]
    except ValueError:
        # Fall back to equal-width bins if quantiles fail
        bins = pd.cut(feature_values, bins=n_bins, labels=False)
        bin_edges = pd.cut(feature_values, bins=n_bins, retbins=True)[1]
    
    # Group predictions by bin
    groups = {}
    for i, pred in enumerate(predictions):
        bin_idx = bins[i]
        if pd.isna(bin_idx):
            continue
        bin_idx = int(bin_idx)
        
        # Create meaningful bin label
        low = bin_edges[bin_idx]
        high = bin_edges[bin_idx + 1] if bin_idx + 1 < len(bin_edges) else bin_edges[-1]
        label = f"{feature}_{low:.0f}-{high:.0f}"
        
        if label not in groups:
            groups[label] = []
        groups[label].append(pred)
    
    return groups


def print_stratified_table(results_df: pd.DataFrame, title: str):
    """Print a formatted stratified results table."""
    print(f"\n{title}")
    if "accuracy_ci_low" in results_df.columns:
        print("-" * 140)
        print(
            f"{'Group':<30} {'N':>6} {'Stud':>6} "
            f"{'Acc [CI]':>20} {'ECE [CI]':>24} {'Brier [CI]':>24} {'AUC [CI]':>20} {'Conf [CI]':>20}"
        )
        print("-" * 140)

        def fmt_ci(row: pd.Series, name: str, fmt: str) -> str:
            v = row.get(name, np.nan)
            lo = row.get(f"{name}_ci_low", np.nan)
            hi = row.get(f"{name}_ci_high", np.nan)
            if not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi)):
                return "nan"
            return f"{v:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"

        for _, row in results_df.iterrows():
            print(
                f"{str(row['group']):<30} {int(row['n_instances']):>6} {int(row.get('n_students', 0)):>6} "
                f"{fmt_ci(row, 'accuracy', '.3f'):>20} {fmt_ci(row, 'ece', '.4f'):>24} "
                f"{fmt_ci(row, 'brier_score', '.4f'):>24} {fmt_ci(row, 'auc_roc', '.3f'):>20} "
                f"{fmt_ci(row, 'mean_confidence', '.3f'):>20}"
            )
        return

    print("-" * 120)
    print(f"{'Group':<30} {'N':>6} {'Acc':>8} {'ECE':>8} {'Brier':>10} {'AUC':>8} {'Conf':>8}")
    print("-" * 120)
    
    for _, row in results_df.iterrows():
        print(f"{str(row['group']):<30} {row['n_instances']:>6} "
              f"{row['accuracy']:>8.3f} {row['ece']:>8.4f} "
              f"{row['brier_score']:>8.4f} {row['auc_roc']:>8.3f} "
              f"{row['mean_confidence']:>8.3f}")


def main():
    args = parse_args()
    output_manager = None
    if args.output_root:
        if not args.run_id:
            raise ValueError("--run-id is required when using --output-root")
        output_manager = build_output_manager(
            output_root=args.output_root,
            dataset=args.dataset,
            run_id=args.run_id,
            save_legacy=args.save_legacy,
            write_manifest=args.write_manifest,
        )
    args.model = model_nicknames.get(args.model.lower(), args.model.lower())
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    rng = np.random.default_rng(args.seed)

    # rpy2/R needs a writable temp dir; avoid relying on system /tmp or $HOME
    # when running under sandboxed/workspace-only environments.
    tmp_dir = project_root.parent / "cache" / "tmp_r"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(tmp_dir))
        os.environ.setdefault("TMP", str(tmp_dir))
        os.environ.setdefault("TEMP", str(tmp_dir))
    except OSError:
        # Best-effort only; if the workspace is not writable, keep existing env vars.
        pass

    print("=" * 80)
    print("STEP 4: STRATIFIED ANALYSIS")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)

    # Do not rely on CWD: resolve relative to the repo's `grading_calibration` root.
    if args.predictions_dir:
        directory = Path(args.predictions_dir) / Path(args.method)
        filter_by_dataset = True
    elif output_manager:
        directory = output_manager.step_dir("step1", "tables")
        filter_by_dataset = False
    else:
        directory = project_root / "experiments" / "predictions_csvs" / args.method
        filter_by_dataset = True
    csv_path = load_path_by_index(
        args.index,
        directory,
        f"{args.prefix}_{model_name}_{args.method}",
        args.dataset,
        filter_by_dataset=filter_by_dataset,
    )
    
    instances, predictions, df = load_data_from_csv(csv_path, smoke_n=args.smoke_n)
    csv_timestamp = args.run_id if output_manager else csv_path.stem.split("_")[-1]

    print(f"Loaded {len(instances)} instances from {csv_path}")

    legacy_analysis_dir = project_root / "experiments" / "analysis" / args.method
    step4_tables_dir = output_manager.step_dir("step4", "tables") if output_manager else legacy_analysis_dir
    step4_plots_dir = output_manager.step_dir("step4", "plots") if output_manager else legacy_analysis_dir
    step4_logs_dir = output_manager.step_dir("step4", "logs") if output_manager else legacy_analysis_dir
    file_suffix = "" if output_manager else f"_{csv_timestamp}"

    warnings_log_path = Path(args.warnings_log) if args.warnings_log else (step4_logs_dir / "warnings.log")
    _install_warnings_logger(warnings_log_path)

    if args.lme4_trace:
        trace_path = args.lme4_trace_file
        if not trace_path:
            trace_path = str(step4_logs_dir / "lme4_optimizer_trace.log")
        os.environ["GC_LME4_TRACE"] = "1"
        os.environ["GC_LME4_TRACE_FILE"] = trace_path
        print(f"LME4 optimizer trace: {trace_path}")

    # ========================================================================
    # Integrity Checks (nesting + duplicates)
    # ========================================================================
    structure = verify_hierarchical_structure(df)
    if args.print_structure:
        print_hierarchical_diagnostics(structure)
    if structure.get("ok", False):
        if df.duplicated(["student_id", "question_id", "rubric_id"]).any():
            raise ValueError("Found duplicate (student_id, question_id, rubric_id) rows in predictions CSV.")

    # ========================================================================
    # Row-level outcomes + clustering diagnostics (Step-4 gold-standard)
    # ========================================================================
    df_out = _make_row_level_outcomes(
        df,
        predictions,
        n_bins=10,
        include_calibration_error=bool(args.include_bin_calibration_error),
    )

    # ICC diagnostics: descriptive only, but helps quantify non-independence.
    # Note: question_id has only 4 levels; interpret question components cautiously.
    if {"student_id", "question_id", "rubric_id"}.issubset(df_out.columns):
        print("\n" + "=" * 80)
        print("CLUSTERING DIAGNOSTICS (ICC; DESCRIPTIVE)")
        print("=" * 80)
        try:
            icc_conf = compute_icc_mixed_model(
                df_out["raw_confidence"].to_numpy(dtype=float),
                df_out["student_id"].to_numpy(dtype=str),
                df_out["question_id"].to_numpy(dtype=str),
                rubric_ids=df_out["rubric_id"].to_numpy(dtype=str),
                verbose=False,
            )
            icc_corr = compute_icc_mixed_model(
                df_out["is_correct"].to_numpy(dtype=float),
                df_out["student_id"].to_numpy(dtype=str),
                df_out["question_id"].to_numpy(dtype=str),
                rubric_ids=df_out["rubric_id"].to_numpy(dtype=str),
                verbose=False,
            )
            icc_brier = compute_icc_mixed_model(
                df_out["brier"].to_numpy(dtype=float),
                df_out["student_id"].to_numpy(dtype=str),
                df_out["question_id"].to_numpy(dtype=str),
                rubric_ids=df_out["rubric_id"].to_numpy(dtype=str),
                verbose=False,
            )
            print(
                "ICC summary (student / answer / rubric; question is fixed effect in Step-3-aligned ICC):\n"
                f"  raw_confidence   : {icc_conf.get('icc_student', np.nan):.4f} / {icc_conf.get('icc_answer', np.nan):.4f} / {icc_conf.get('icc_rubric', np.nan):.4f}\n"
                f"  is_correct       : {icc_corr.get('icc_student', np.nan):.4f} / {icc_corr.get('icc_answer', np.nan):.4f} / {icc_corr.get('icc_rubric', np.nan):.4f}\n"
                f"  brier            : {icc_brier.get('icc_student', np.nan):.4f} / {icc_brier.get('icc_answer', np.nan):.4f} / {icc_brier.get('icc_rubric', np.nan):.4f}\n"
            )
            if "calibration_error" in df_out.columns:
                icc_cal = compute_icc_mixed_model(
                    df_out["calibration_error"].to_numpy(dtype=float),
                    df_out["student_id"].to_numpy(dtype=str),
                    df_out["question_id"].to_numpy(dtype=str),
                    rubric_ids=df_out["rubric_id"].to_numpy(dtype=str),
                    verbose=False,
                )
                print(
                    f"  calibration_error: {icc_cal.get('icc_student', np.nan):.4f} / "
                    f"{icc_cal.get('icc_answer', np.nan):.4f} / {icc_cal.get('icc_rubric', np.nan):.4f}\n"
                )
        except Exception as e:
            print(f"Warning: ICC diagnostics failed: {e}")

    # ========================================================================
    # Extract Features (continuous stratification and/or group-feature associations)
    # ========================================================================
    continuous_features = ["answer_length", "response_length", "question_length", "rubric_length"]
    needs_features = args.stratify_by in continuous_features or args.associate_group_features

    features_df: pd.DataFrame | None = None
    if needs_features:
        print("\nExtracting features...")
        cache_path = None
        cache_meta = None
        if args.features_cache_dir:
            cache_key, cache_meta = compute_features_cache_key(
                predictions_path=Path(csv_path),
                smoke_n=args.smoke_n,
                enable_perplexity=args.enable_perplexity,
                perplexity_model=args.perplexity_model,
                perplexity_max_length=args.perplexity_max_length,
                perplexity_min_tokens=args.perplexity_min_tokens,
                feature_version=args.features_cache_version,
            )
            cache_path = build_features_cache_path(
                args.features_cache_dir,
                dataset=args.dataset,
                method=args.method,
                model=model_name,
                feature_version=args.features_cache_version,
                cache_key=cache_key,
            )
            if cache_path and args.features_cache_mode != "refresh" and cache_path.exists():
                features_df = pd.read_csv(cache_path)
                if "instance_id" in features_df.columns:
                    features_df.set_index("instance_id", inplace=True, drop=False)
                    ordered_ids = [p.instance_id for p in predictions]
                    features_df = features_df.reindex(ordered_ids)
                required_cols = []
                if args.enable_perplexity:
                    required_cols.extend(["answer_perplexity", "response_perplexity"])
                missing_cols = [c for c in required_cols if c not in features_df.columns]
                if missing_cols:
                    if args.features_cache_mode == "read_only":
                        raise RuntimeError(
                            f"Features cache missing required columns: {missing_cols}. "
                            "Use --features-cache-mode refresh to rebuild."
                        )
                    features_df = None
                else:
                    print(f"Loaded features cache: {cache_path}")
                    if output_manager:
                        output_manager.register(
                            step="step4",
                            artifact="features_cache",
                            path=cache_path,
                            model=model_name,
                            method=args.method,
                            extra={"cache_version": args.features_cache_version},
                        )

            if features_df is None and cache_meta and args.features_cache_mode != "refresh":
                candidate_paths = find_compatible_feature_caches(
                    args.features_cache_dir,
                    dataset=args.dataset,
                    method=args.method,
                    model=model_name,
                    feature_version=args.features_cache_version,
                    target_meta=cache_meta,
                    ignore_keys={"smoke_n"},
                )
                partial_df = None
                for path in candidate_paths:
                    tmp = pd.read_csv(path)
                    if "instance_id" not in tmp.columns:
                        continue
                    tmp.set_index("instance_id", inplace=True, drop=False)
                    partial_df = tmp if partial_df is None else partial_df.combine_first(tmp)
                if partial_df is not None:
                    ordered_ids = [p.instance_id for p in predictions]
                    partial_df = partial_df.reindex(ordered_ids)
                    required_cols = []
                    if args.enable_perplexity:
                        required_cols.extend(["answer_perplexity", "response_perplexity"])
                    missing_cols = [c for c in required_cols if c not in partial_df.columns]
                    if missing_cols:
                        if args.features_cache_mode == "read_only":
                            raise RuntimeError(
                                f"Features cache missing required columns: {missing_cols}. "
                                "Use --features-cache-mode refresh to rebuild."
                            )
                    else:
                        features_df = partial_df
                        print(f"Loaded partial features cache from {len(candidate_paths)} file(s).")

        if features_df is None:
            extractor = FeatureExtractor(
                enable_perplexity=args.enable_perplexity,
                perplexity_model=args.perplexity_model,
                perplexity_device=args.perplexity_device,
                perplexity_batch_size=args.perplexity_batch_size,
                perplexity_max_length=args.perplexity_max_length,
                perplexity_min_tokens=args.perplexity_min_tokens,
                perplexity_debug=args.perplexity_debug,
            )
            features_df = extractor.extract_batch(instances, predictions)
            if cache_path and args.features_cache_mode in ("read_write", "refresh"):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_df = features_df.copy()
                if cache_df.index.name == "instance_id" and "instance_id" not in cache_df.columns:
                    cache_df = cache_df.reset_index()
                cache_df.to_csv(cache_path, index=False)
                print(f"Saved features cache: {cache_path}")
                if cache_meta:
                    write_features_cache_metadata(cache_path, cache_meta)
                if output_manager:
                    output_manager.register(
                        step="step4",
                        artifact="features_cache",
                        path=cache_path,
                        model=model_name,
                        method=args.method,
                        extra={"cache_version": args.features_cache_version},
                    )
        else:
            ordered_ids = [p.instance_id for p in predictions]
            present_ids = set(features_df.index.dropna())
            missing_ids = [iid for iid in ordered_ids if iid not in present_ids]
            if missing_ids:
                if args.features_cache_mode == "read_only":
                    raise RuntimeError(
                        f"Features cache missing {len(missing_ids)} instances; "
                        "use --features-cache-mode read_write or refresh."
                    )
                extractor = FeatureExtractor(
                    enable_perplexity=args.enable_perplexity,
                    perplexity_model=args.perplexity_model,
                    perplexity_device=args.perplexity_device,
                    perplexity_batch_size=args.perplexity_batch_size,
                    perplexity_max_length=args.perplexity_max_length,
                    perplexity_min_tokens=args.perplexity_min_tokens,
                    perplexity_debug=args.perplexity_debug,
                )
                inst_map = {inst.id: inst for inst in instances}
                pred_map = {pred.instance_id: pred for pred in predictions}
                missing_instances = [inst_map[iid] for iid in missing_ids if iid in inst_map]
                missing_predictions = [pred_map[iid] for iid in missing_ids if iid in pred_map]
                new_df = extractor.extract_batch(missing_instances, missing_predictions)
                if "instance_id" in new_df.columns:
                    new_df.set_index("instance_id", inplace=True, drop=False)
                features_df = features_df.combine_first(new_df)
                features_df = features_df.reindex(ordered_ids)
                if cache_path and args.features_cache_mode in ("read_write", "refresh"):
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_df = features_df.copy()
                    if cache_df.index.name == "instance_id" and "instance_id" not in cache_df.columns:
                        cache_df = cache_df.reset_index()
                    cache_df.to_csv(cache_path, index=False)
                    print(f"Saved features cache: {cache_path}")
                    if cache_meta:
                        write_features_cache_metadata(cache_path, cache_meta)

    # ========================================================================
    # Compute Metrics per Group
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"STRATIFIED BY: {args.stratify_by.upper()}")
    print("=" * 80)

    results_by_stratum, results_df = _pipeline_run_stratified_group_metrics(
        predictions=predictions,
        features_df=features_df,
        args=args,
        rng=rng,
        stratify_by_category=stratify_by_category,
        stratify_by_feature_bins=stratify_by_feature_bins,
        compute_metrics_for_group=compute_metrics_for_group,
        compute_metrics_for_group_with_ci=compute_metrics_for_group_with_ci,
    )

    if args.stratify_by == "all":
        for col, df_col in results_by_stratum.items():
            print_stratified_table(df_col, f"Metrics by {col}")
    else:
        print_stratified_table(results_df, f"Metrics by {args.stratify_by}")

    # ========================================================================
    # Problematic groups (descriptive flags; CI-aware if available)
    # ========================================================================
    if int(args.problematic_max_groups) > 0:
        ece_thr = 0.15 if args.problematic_ece_threshold is None else float(args.problematic_ece_threshold)
        acc_thr = 0.70 if args.problematic_accuracy_threshold is None else float(args.problematic_accuracy_threshold)

        for label, df_stratum in results_by_stratum.items():
            print("\n" + "=" * 80)
            print(f"PROBLEMATIC GROUPS (DESCRIPTIVE FLAGS; CI-AWARE IF AVAILABLE) — {label}")
            print("=" * 80)
            print(f"Thresholds: ECE > {ece_thr} and/or Accuracy < {acc_thr}")

            flagged = _flag_problematic_groups(
                df_stratum,
                ece_threshold=ece_thr,
                accuracy_threshold=acc_thr,
            )
            if len(flagged) == 0:
                print("No groups flagged as problematic under current thresholds.")
            else:
                show = flagged.head(int(args.problematic_max_groups))
                cols = [
                    "group",
                    "n_instances",
                    "n_students",
                    "accuracy",
                    "ece",
                    "brier_score",
                    "auc_roc",
                    "mean_confidence",
                    "high_ece",
                    "low_accuracy",
                ]
                cols = [c for c in cols if c in show.columns]
                print(show[cols].to_string(index=False))
                if len(flagged) > len(show):
                    print(f"... ({len(flagged) - len(show)} more flagged groups not shown)")

    # ========================================================================
    # Hypothesis tests for rubric differences (primary: mixed-model LRT)
    # ========================================================================
    if args.test_group_differences:
        print("\n" + "=" * 80)
        print("HYPOTHESIS TESTS (PRIMARY): MIXED-MODEL LRT WITHIN QUESTION")
        print("=" * 80)
        print("H0: y ~ 1 + (1|student_id)")
        print("H1: y ~ factor(rubric_id) + (1|student_id)   (rubric tests restricted to within-question)")
        print(
            "Effect size (omnibus): `rubric_mean_range` = max_r mean(y|rubric=r) − min_r mean(y|rubric=r) within question; "
            "CI via student-block bootstrap when `--compute-ci` is set."
        )
        if args.include_bin_calibration_error:
            print("Bin-assigned calibration_error tests are enabled (exploratory).")
        else:
            print("Bin-assigned calibration_error tests are disabled (use --include-bin-calibration-error to enable).")
        if not args.global_fdr:
            print(
                "Multiple testing: `p_adjusted` is BH-FDR within the LRT family. "
                "For publication-style correction across Step 4 tests, use `--global-fdr`."
            )
        else:
            print("Multiple testing: primary is global BH-FDR (`--global-fdr` enabled); see GLOBAL MULTIPLE TESTING CORRECTION block.")
        try:
            lrt_df = _run_rubric_lrt_tests_within_question(
                df_out,
                alpha=args.alpha,
                pairwise_scope=args.pairwise_scope,
                pairwise_top_k=args.pairwise_top_k,
                compute_effect_ci=bool(args.compute_ci),
                effect_n_bootstrap=int(args.n_bootstrap),
                rng=rng,
                include_calibration_error=bool(args.include_bin_calibration_error),
                glmer_nAGQ=args.lme4_glmer_nAGQ,
                optimizer=args.lme4_optimizer,
                maxfun=args.lme4_maxfun,
            )
        except Exception as e:
            print(f"Error: failed to run mixed-model LRT tests: {e}")
            lrt_df = pd.DataFrame()

        cal_reg_df = pd.DataFrame()
        if len(lrt_df) == 0:
            print("No LRT results produced (insufficient data or missing lme4 backend).")
        else:
            # Print a concise summary for omnibus tests.
            omnibus = lrt_df[lrt_df["test"] == "omnibus_rubric"].copy()
            if len(omnibus):
                print("\nOmnibus rubric heterogeneity (per question, per outcome):")
                cols = [
                    "question_id",
                    "outcome",
                    "n_students",
                    "n_rubrics",
                    "rubric_mean_range",
                    "rubric_mean_range_ci_low",
                    "rubric_mean_range_ci_high",
                    "chisq",
                    "df_diff",
                    "p_value",
                    "p_adjusted_primary",
                    "sig_primary",
                ]
                cols = [c for c in cols if c in omnibus.columns]
                print(omnibus[cols].sort_values(["outcome", "question_id"]).to_string(index=False))

            pairwise = lrt_df[lrt_df["test"] == "pairwise_rubric"].copy()
            if len(pairwise):
                print("\nPairwise rubric contrasts (within-question; first 50 rows):")
                cols = [
                    "question_id",
                    "outcome",
                    "rubric_a",
                    "rubric_b",
                    "beta",
                    "se",
                    "odds_ratio",
                    "p_value",
                    "p_adjusted_primary",
                    "sig_primary",
                ]
                cols = [c for c in cols if c in pairwise.columns]
                print(
                    pairwise[cols]
                    .sort_values(["outcome", "question_id", "p_adjusted_primary"])
                    .head(50)
                    .to_string(index=False)
                )

            if not args.no_save:
                out_path = step4_tables_dir / f"step4_lrt_rubric_tests_{model_name}_{args.method}{file_suffix}.csv"
                legacy_path = legacy_analysis_dir / f"step4_lrt_rubric_tests_{model_name}_{csv_timestamp}.csv"
                _save_table(
                    lrt_df,
                    out_path,
                    output_manager=output_manager,
                    artifact="lrt_rubric_tests",
                    model=model_name,
                    method=args.method,
                    legacy_path=legacy_path,
                )
                print(f"\nLRT test results saved to: {out_path}")

            # Calibration regression: rubric slope heterogeneity (primary calibration test)
            print("\n" + "=" * 80)
            print("CALIBRATION REGRESSION (PRIMARY): RUBRIC INTERCEPT + SLOPE HETEROGENEITY")
            print("=" * 80)
            print("Testing: is_correct ~ logit_conf_c + factor(rubric_id) + logit_conf_c:factor(rubric_id) + (1|student_id)")
            try:
                cal_reg_df = _run_calibration_regression_rubric_slopes_within_question(
                    df_out,
                    alpha=args.alpha,
                    compute_ci=bool(args.compute_ci),
                    n_bootstrap=int(args.n_bootstrap),
                    rng=rng,
                    glmer_nAGQ=args.lme4_glmer_nAGQ,
                    optimizer=args.lme4_optimizer,
                    maxfun=args.lme4_maxfun,
                )
            except Exception as e:
                print(f"Warning: calibration regression failed: {e}")
                cal_reg_df = pd.DataFrame()

            if len(cal_reg_df) == 0:
                print("No calibration regression results produced (insufficient data or missing lme4 backend).")
            else:
                cols = [
                    "question_id",
                    "test",
                    "n_students",
                    "n_rubrics",
                    "slope_range",
                    "slope_range_ci_low",
                    "slope_range_ci_high",
                    "rubric_min",
                    "rubric_max",
                    "chisq",
                    "df_diff",
                    "p_value",
                    "p_adjusted_primary",
                    "sig_primary",
                ]
                cols = [c for c in cols if c in cal_reg_df.columns]
                print("\nCalibration regression heterogeneity (per question):")
                print(cal_reg_df[cols].sort_values(["test", "p_adjusted_primary", "p_value"]).to_string(index=False))

                if args.print_calibration_slopes:
                    print("\nRubric-specific calibration slopes (by question):")
                    for _, row in cal_reg_df.query("test == 'omnibus_calibration_slope'").sort_values("question_id").iterrows():
                        slopes_str = str(row.get("rubric_slopes", "")).strip()
                        if not slopes_str:
                            continue
                        print(f"Question {row.get('question_id')}: {slopes_str}")

                if not args.no_save:
                    out_path = step4_tables_dir / f"step4_calibration_regression_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_calibration_regression_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        cal_reg_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="calibration_regression",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )
                    print(f"\nCalibration regression results saved to: {out_path}")

            # Question-level tests (n=4; exploratory)
            q_lrt_df = pd.DataFrame()
            if not args.skip_question_tests:
                print("\n" + "=" * 80)
                print("HYPOTHESIS TESTS (EXPLORATORY): QUESTION DIFFERENCES (ONLY 4 QUESTIONS)")
                print("=" * 80)
                try:
                    q_lrt_df = _run_question_lrt_tests_across_questions(
                        df_out,
                        alpha=args.alpha,
                        include_calibration_error=bool(args.include_bin_calibration_error),
                        glmer_nAGQ=args.lme4_glmer_nAGQ,
                        optimizer=args.lme4_optimizer,
                        maxfun=args.lme4_maxfun,
                    )
                except Exception as e:
                    print(f"Warning: failed to run question-level tests: {e}")
                    q_lrt_df = pd.DataFrame()

                if len(q_lrt_df):
                    cols = [
                        "outcome",
                        "test",
                        "group_a",
                        "group_b",
                        "chisq",
                        "df_diff",
                        "p_value",
                        "p_adjusted_primary",
                        "sig_primary",
                    ]
                    cols = [c for c in cols if c in q_lrt_df.columns]
                    print(q_lrt_df[cols].sort_values(["outcome", "test", "p_adjusted_primary"]).head(50).to_string(index=False))
                    if not args.no_save:
                        out_path = step4_tables_dir / f"step4_lrt_question_tests_{model_name}_{args.method}{file_suffix}.csv"
                        legacy_path = legacy_analysis_dir / f"step4_lrt_question_tests_{model_name}_{csv_timestamp}.csv"
                        _save_table(
                            q_lrt_df,
                            out_path,
                            output_manager=output_manager,
                            artifact="lrt_question_tests",
                            model=model_name,
                            method=args.method,
                            legacy_path=legacy_path,
                        )
                        print(f"\nQuestion test results saved to: {out_path}")

            # --------------------------------------------------------------------
            # Prospective power + Bayesian assurance (primary LRT tests only)
            # --------------------------------------------------------------------
            power_df = pd.DataFrame()
            assurance_df = pd.DataFrame()

            run_power = args.power_method == "mixed_model_lrt" and args.power_scope != "none"
            run_assurance = bool(args.bayesian_assurance) and args.assurance_scope != "none"
            if (run_power or run_assurance) and len(lrt_df):
                print("\n" + "=" * 80)
                print("PROSPECTIVE POWER / BAYESIAN ASSURANCE (PRIMARY LRT TESTS)")
                print("=" * 80)
                print(
                    "Effect size axis: |beta| on the model scale (two-sided LRT). "
                    "For omnibus (multi-df) tests, power is computed for a single contrast only (conservative)."
                )

                power_curve_cache: dict[str, pd.DataFrame] = {}

                lrt_iter = lrt_df.iterrows()
                if len(lrt_df) > 1:
                    lrt_iter = tqdm(
                        lrt_df.iterrows(),
                        total=len(lrt_df),
                        desc="LRT tests (power/assurance)",
                        unit="test",
                        dynamic_ncols=True,
                    )

                for _, row in lrt_iter:
                    test_name = str(row.get("test", ""))
                    if not _should_run_scope(args.power_scope, test_name) and not _should_run_scope(
                        args.assurance_scope, test_name
                    ):
                        continue

                    q = str(row.get("question_id", ""))
                    outcome_col = str(row.get("outcome", ""))
                    is_glmm = outcome_col == "is_correct"
                    family = "binomial" if is_glmm else "gaussian"

                    # Build analytic df for this test.
                    df_q = df_out[df_out["question_id"].astype(str) == q].copy()
                    df_q = df_q.dropna(subset=["student_id", "rubric_id", outcome_col])
                    df_fit = df_q[["student_id", "rubric_id", outcome_col]].rename(columns={outcome_col: "y"})
                    # Ensure R treats IDs as categorical (factors), not numeric covariates.
                    df_fit["student_id"] = df_fit["student_id"].astype(str)
                    df_fit["rubric_id"] = df_fit["rubric_id"].astype(str)

                    formula_null = "y ~ 1 + (1|student_id)"
                    coef_name = None

                    if test_name == "pairwise_rubric":
                        ra = str(row.get("rubric_a", ""))
                        rb = str(row.get("rubric_b", ""))
                        df_pair = df_fit[df_fit["rubric_id"].astype(str).isin([ra, rb])].copy()
                        if df_pair["student_id"].nunique() < 2:
                            continue
                        df_pair["rubric_is_b"] = (df_pair["rubric_id"].astype(str) == rb).astype(int)
                        df_fit_test = df_pair[["student_id", "rubric_is_b", "y"]]
                        formula_full = "y ~ rubric_is_b + (1|student_id)"
                        coef_name = "rubric_is_b"
                    else:
                        # omnibus_rubric
                        rubrics = sorted(df_fit["rubric_id"].astype(str).unique().tolist())
                        if len(rubrics) < 2:
                            continue
                        baseline = rubrics[0]
                        means = df_fit.groupby(df_fit["rubric_id"].astype(str))["y"].mean().to_dict()
                        diffs = [(r, abs(float(means.get(r, np.nan) - means.get(baseline, np.nan)))) for r in rubrics[1:]]
                        diffs = [(r, d) for r, d in diffs if np.isfinite(d)]
                        if not diffs:
                            continue
                        rb = max(diffs, key=lambda x: x[1])[0]
                        df_fit_test = df_fit.copy()
                        formula_full = "y ~ factor(rubric_id) + (1|student_id)"
                        coef_name = f"factor(rubric_id){rb}"

                    test_key = f"step4|{test_name}|q={q}|outcome={outcome_col}|coef={coef_name}"
                    seed_students = _stable_int_seed(test_key, args.power_seed_students)
                    seed_outcomes = _stable_int_seed(test_key, args.power_seed_outcomes)

                    if run_power and _should_run_scope(args.power_scope, test_name):
                        try:
                            pc = compute_power_curve_lme4_lrt_student_re(
                                df_fit_test,
                                outcome_col="y",
                                formula_null=formula_null,
                                formula_full=formula_full,
                                coef_name=str(coef_name),
                                beta_targets_abs=[float(b) for b in args.power_beta_grid_abs],
                                n_students_target=args.power_n_students_target,
                                alpha=args.alpha,
                                n_sim=int(args.power_n_sim),
                                seed_students=int(seed_students),
                                seed_outcomes=int(seed_outcomes),
                                is_glmm=bool(is_glmm),
                                family=family,
                                glmer_nAGQ=int(args.lme4_glmer_nAGQ),
                                optimizer=str(args.lme4_optimizer),
                                maxfun=int(args.lme4_maxfun),
                                log_progress=True,
                            )
                            power_curve_cache[test_key] = pc
                            pc.insert(0, "question_id", q)
                            pc.insert(1, "outcome", outcome_col)
                            pc.insert(2, "test", test_name)
                            pc.insert(3, "coef_name", str(coef_name))
                            pc.insert(4, "rubric_a", ra if test_name == "pairwise_rubric" else "")
                            pc.insert(5, "rubric_b", rb if test_name == "pairwise_rubric" else "")
                            power_df = pd.concat([power_df, pc], ignore_index=True)
                        except Exception as e:
                            print(f"Warning: power curve failed for {test_key}: {e}")

                    if run_assurance and _should_run_scope(args.assurance_scope, test_name):
                        try:
                            pc_cached = power_curve_cache.get(test_key)
                            ar = compute_bayesian_assurance_lme4_lrt_empirical_student_re(
                                df_fit_test,
                                outcome_col="y",
                                formula_null=formula_null,
                                formula_full=formula_full,
                                coef_name=str(coef_name),
                                n_students_target=int(
                                    args.power_n_students_target
                                    if args.power_n_students_target is not None
                                    else df_fit_test["student_id"].nunique()
                                ),
                                alpha=args.alpha,
                                assurance_mode=args.assurance_mode,
                                beta_grid_abs=[float(b) for b in args.power_beta_grid_abs],
                                power_curve=pc_cached if args.assurance_mode == "fast" else None,
                                n_power_sim=int(args.power_n_sim),
                                seed_students=int(seed_students),
                                seed_outcomes=int(seed_outcomes),
                                n_prior=int(args.assurance_n_prior),
                                seed_prior=int(_stable_int_seed(test_key, args.assurance_seed_prior)),
                                n_power_sim_inner=int(args.assurance_n_sim_inner),
                                is_glmm=bool(is_glmm),
                                family=family,
                                glmer_nAGQ=int(args.lme4_glmer_nAGQ),
                                optimizer=str(args.lme4_optimizer),
                                maxfun=int(args.lme4_maxfun),
                                winsorize_pct=args.assurance_winsorize_pct,
                                log_progress=True,
                            )
                            assurance_df = pd.concat(
                                [
                                    assurance_df,
                                    pd.DataFrame(
                                        [
                                            {
                                                "question_id": q,
                                                "outcome": outcome_col,
                                                "test": test_name,
                                                "coef_name": str(coef_name),
                                                "rubric_a": ra if test_name == "pairwise_rubric" else "",
                                                "rubric_b": rb if test_name == "pairwise_rubric" else "",
                                                "assurance": float(ar["assurance"]),
                                                "assurance_ci_low": float(ar["assurance_ci"][0]),
                                                "assurance_ci_high": float(ar["assurance_ci"][1]),
                                                "assurance_mode": str(ar["mode"]),
                                                "n_students_target": int(ar["n_students_target"]),
                                                "prior_mean": float(ar["prior_summary"]["mean"]),
                                                "prior_sd": float(ar["prior_summary"]["sd"]),
                                                "prior_q05": float(ar["prior_summary"]["q05"]),
                                                "prior_q50": float(ar["prior_summary"]["q50"]),
                                                "prior_q95": float(ar["prior_summary"]["q95"]),
                                                "prior_n_attempted": int(ar["prior_n_attempted"]),
                                                "prior_n_failed": int(ar["prior_n_failed"]),
                                                "prior_n_singular": int(ar["prior_n_singular"]),
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                        except Exception as e:
                            print(f"Warning: assurance failed for {test_key}: {e}")

                if len(power_df):
                    # Drop accidental duplicates if the same test is computed multiple times.
                    power_df = power_df.drop_duplicates(
                        subset=["question_id", "outcome", "test", "coef_name", "rubric_a", "rubric_b", "beta_target_abs"],
                        keep="last",
                    )
                    n_tests = int(
                        power_df[["question_id", "outcome", "test", "coef_name", "rubric_a", "rubric_b"]]
                        .drop_duplicates()
                        .shape[0]
                    )
                    print(f"\nComputed prospective power curves for {n_tests} tests (|beta| grid size={power_df['beta_target_abs'].nunique()}).")
                    print("Power is reported with Wilson 95% CI over Monte Carlo simulations.")
                    if not args.no_save:
                        out_path = step4_tables_dir / f"step4_power_curves_{model_name}_{args.method}{file_suffix}.csv"
                        legacy_path = legacy_analysis_dir / f"step4_power_curves_{model_name}_{csv_timestamp}.csv"
                        _save_table(
                            power_df,
                            out_path,
                            output_manager=output_manager,
                            artifact="power_curves",
                            model=model_name,
                            method=args.method,
                            legacy_path=legacy_path,
                        )
                        print(f"\nPower curves saved to: {out_path}")
                    if args.plot_power_curves:
                        n_plots = _plot_power_curves(
                            power_df,
                            step4_plots_dir,
                            prefix=f"step4_power_curve_{model_name}{file_suffix}",
                        )
                        if n_plots:
                            print(f"\nSaved {n_plots} power curve plot(s) to: {step4_plots_dir}")

                if len(assurance_df):
                    print(f"\nComputed Bayesian assurance for {len(assurance_df)} tests (mode={args.assurance_mode}).")
                    show = assurance_df.sort_values(["assurance", "prior_sd"], ascending=[True, False]).head(10)
                    print("\nLowest-assurance tests (top 10):")
                    print(
                        show[
                            [
                                "question_id",
                                "outcome",
                                "test",
                                "rubric_a",
                                "rubric_b",
                                "assurance",
                                "assurance_ci_low",
                                "assurance_ci_high",
                                "prior_mean",
                                "prior_sd",
                            ]
                        ].to_string(index=False)
                    )
                    if not args.no_save:
                        out_path = step4_tables_dir / f"step4_bayesian_assurance_{model_name}_{args.method}{file_suffix}.csv"
                        legacy_path = legacy_analysis_dir / f"step4_bayesian_assurance_{model_name}_{csv_timestamp}.csv"
                        _save_table(
                            assurance_df,
                            out_path,
                            output_manager=output_manager,
                            artifact="bayesian_assurance",
                            model=model_name,
                            method=args.method,
                            legacy_path=legacy_path,
                        )
                        print(f"\nAssurance results saved to: {out_path}")

            # Question-level power/assurance (exploratory; only 4 questions)
            if (run_power or run_assurance) and ("q_lrt_df" in locals()) and isinstance(q_lrt_df, pd.DataFrame) and len(q_lrt_df):
                print("\n" + "=" * 80)
                print("PROSPECTIVE POWER / BAYESIAN ASSURANCE (QUESTION TESTS; EXPLORATORY)")
                print("=" * 80)

                q_power_df = pd.DataFrame()
                q_assurance_df = pd.DataFrame()
                q_power_curve_cache: dict[str, pd.DataFrame] = {}

                q_iter = q_lrt_df.iterrows()
                if len(q_lrt_df) > 1:
                    q_iter = tqdm(
                        q_lrt_df.iterrows(),
                        total=len(q_lrt_df),
                        desc="Question tests (power/assurance)",
                        unit="test",
                        dynamic_ncols=True,
                    )

                for _, row in q_iter:
                    test_name = str(row.get("test", ""))
                    if not _should_run_scope(args.power_scope, test_name) and not _should_run_scope(
                        args.assurance_scope, test_name
                    ):
                        continue

                    outcome_col = str(row.get("outcome", ""))
                    is_glmm = outcome_col == "is_correct"
                    family = "binomial" if is_glmm else "gaussian"

                    df_qo = df_out.dropna(subset=["student_id", "question_id", outcome_col]).copy()
                    df_fit = df_qo[["student_id", "question_id", outcome_col]].rename(columns={outcome_col: "y"})
                    # Ensure R treats IDs as categorical (factors), not numeric covariates.
                    df_fit["student_id"] = df_fit["student_id"].astype(str)
                    df_fit["question_id"] = df_fit["question_id"].astype(str)
                    formula_null = "y ~ 1 + (1|student_id)"

                    coef_name = None
                    if test_name == "pairwise_question":
                        qa = str(row.get("group_a", ""))
                        qb = str(row.get("group_b", ""))
                        df_pair = df_fit[df_fit["question_id"].astype(str).isin([qa, qb])].copy()
                        if df_pair["student_id"].nunique() < 2:
                            continue
                        df_pair["question_is_b"] = (df_pair["question_id"].astype(str) == qb).astype(int)
                        df_fit_test = df_pair[["student_id", "question_is_b", "y"]]
                        formula_full = "y ~ question_is_b + (1|student_id)"
                        coef_name = "question_is_b"
                        group_a = qa
                        group_b = qb
                    else:
                        q_levels = sorted(df_fit["question_id"].astype(str).unique().tolist())
                        if len(q_levels) < 2:
                            continue
                        baseline = q_levels[0]
                        means = df_fit.groupby(df_fit["question_id"].astype(str))["y"].mean().to_dict()
                        diffs = [(qq, abs(float(means.get(qq, np.nan) - means.get(baseline, np.nan)))) for qq in q_levels[1:]]
                        diffs = [(qq, d) for qq, d in diffs if np.isfinite(d)]
                        if not diffs:
                            continue
                        qb = max(diffs, key=lambda x: x[1])[0]
                        df_fit_test = df_fit.copy()
                        formula_full = "y ~ factor(question_id) + (1|student_id)"
                        coef_name = f"factor(question_id){qb}"
                        group_a = ""
                        group_b = ""

                    test_key = f"step4|question|{test_name}|outcome={outcome_col}|coef={coef_name}|a={group_a}|b={group_b}"
                    seed_students = _stable_int_seed(test_key, args.power_seed_students)
                    seed_outcomes = _stable_int_seed(test_key, args.power_seed_outcomes)

                    if run_power and _should_run_scope(args.power_scope, test_name):
                        try:
                            pc = compute_power_curve_lme4_lrt_student_re(
                                df_fit_test,
                                outcome_col="y",
                                formula_null=formula_null,
                                formula_full=formula_full,
                                coef_name=str(coef_name),
                                beta_targets_abs=[float(b) for b in args.power_beta_grid_abs],
                                n_students_target=args.power_n_students_target,
                                alpha=args.alpha,
                                n_sim=int(args.power_n_sim),
                                seed_students=int(seed_students),
                                seed_outcomes=int(seed_outcomes),
                                is_glmm=bool(is_glmm),
                                family=family,
                                glmer_nAGQ=int(args.lme4_glmer_nAGQ),
                                optimizer=str(args.lme4_optimizer),
                                maxfun=int(args.lme4_maxfun),
                                log_progress=True,
                            )
                            q_power_curve_cache[test_key] = pc
                            pc.insert(0, "question_id", "ALL")
                            pc.insert(1, "outcome", outcome_col)
                            pc.insert(2, "test", test_name)
                            pc.insert(3, "coef_name", str(coef_name))
                            pc.insert(4, "group_a", group_a)
                            pc.insert(5, "group_b", group_b)
                            q_power_df = pd.concat([q_power_df, pc], ignore_index=True)
                        except Exception as e:
                            print(f"Warning: question power curve failed for {test_key}: {e}")

                    if run_assurance and _should_run_scope(args.assurance_scope, test_name):
                        try:
                            pc_cached = q_power_curve_cache.get(test_key)
                            ar = compute_bayesian_assurance_lme4_lrt_empirical_student_re(
                                df_fit_test,
                                outcome_col="y",
                                formula_null=formula_null,
                                formula_full=formula_full,
                                coef_name=str(coef_name),
                                n_students_target=int(
                                    args.power_n_students_target
                                    if args.power_n_students_target is not None
                                    else df_fit_test["student_id"].nunique()
                                ),
                                alpha=args.alpha,
                                assurance_mode=args.assurance_mode,
                                beta_grid_abs=[float(b) for b in args.power_beta_grid_abs],
                                power_curve=pc_cached if args.assurance_mode == "fast" else None,
                                n_power_sim=int(args.power_n_sim),
                                seed_students=int(seed_students),
                                seed_outcomes=int(seed_outcomes),
                                n_prior=int(args.assurance_n_prior),
                                seed_prior=int(_stable_int_seed(test_key, args.assurance_seed_prior)),
                                n_power_sim_inner=int(args.assurance_n_sim_inner),
                                is_glmm=bool(is_glmm),
                                family=family,
                                glmer_nAGQ=int(args.lme4_glmer_nAGQ),
                                optimizer=str(args.lme4_optimizer),
                                maxfun=int(args.lme4_maxfun),
                                winsorize_pct=args.assurance_winsorize_pct,
                                log_progress=True,
                            )
                            q_assurance_df = pd.concat(
                                [
                                    q_assurance_df,
                                    pd.DataFrame(
                                        [
                                            {
                                                "question_id": "ALL",
                                                "outcome": outcome_col,
                                                "test": test_name,
                                                "coef_name": str(coef_name),
                                                "group_a": group_a,
                                                "group_b": group_b,
                                                "assurance": float(ar["assurance"]),
                                                "assurance_ci_low": float(ar["assurance_ci"][0]),
                                                "assurance_ci_high": float(ar["assurance_ci"][1]),
                                                "assurance_mode": str(ar["mode"]),
                                                "n_students_target": int(ar["n_students_target"]),
                                                "prior_mean": float(ar["prior_summary"]["mean"]),
                                                "prior_sd": float(ar["prior_summary"]["sd"]),
                                                "prior_q05": float(ar["prior_summary"]["q05"]),
                                                "prior_q50": float(ar["prior_summary"]["q50"]),
                                                "prior_q95": float(ar["prior_summary"]["q95"]),
                                                "prior_n_attempted": int(ar["prior_n_attempted"]),
                                                "prior_n_failed": int(ar["prior_n_failed"]),
                                                "prior_n_singular": int(ar["prior_n_singular"]),
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                        except Exception as e:
                            print(f"Warning: question assurance failed for {test_key}: {e}")

                if len(q_power_df) and not args.no_save:
                    out_path = step4_tables_dir / f"step4_question_power_curves_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_question_power_curves_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        q_power_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="question_power_curves",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )
                    print(f"\nQuestion power curves saved to: {out_path}")
                if len(q_power_df) and args.plot_power_curves:
                    n_plots = _plot_power_curves(
                        q_power_df,
                        step4_plots_dir,
                        prefix=f"step4_question_power_curve_{model_name}{file_suffix}",
                    )
                    if n_plots:
                        print(f"\nSaved {n_plots} question power curve plot(s) to: {step4_plots_dir}")

                if len(q_assurance_df) and not args.no_save:
                    out_path = step4_tables_dir / f"step4_question_bayesian_assurance_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_question_bayesian_assurance_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        q_assurance_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="question_bayesian_assurance",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )
                    print(f"\nQuestion assurance results saved to: {out_path}")

    # ========================================================================
    # Secondary hypothesis tests for non-row-additive metrics (ECE/AUC)
    # ========================================================================
    metric_df = pd.DataFrame()
    q_metric_df = pd.DataFrame()
    if args.test_metric_differences:
        metric_df, q_metric_df = _pipeline_run_metric_pairwise_tests(
            predictions=predictions,
            args=args,
            rng=rng,
            run_metric_pairwise_tests_within_question=_run_metric_pairwise_tests_within_question,
            run_metric_pairwise_tests_across_questions=_run_metric_pairwise_tests_across_questions,
        )
        if len(metric_df) and not args.no_save:
            out_path = step4_tables_dir / f"step4_metric_pairwise_tests_{model_name}_{args.method}{file_suffix}.csv"
            legacy_path = legacy_analysis_dir / f"step4_metric_pairwise_tests_{model_name}_{csv_timestamp}.csv"
            _save_table(
                metric_df,
                out_path,
                output_manager=output_manager,
                artifact="metric_pairwise_tests",
                model=model_name,
                method=args.method,
                legacy_path=legacy_path,
            )
            print(f"\nMetric-level test results saved to: {out_path}")
        if len(q_metric_df) and not args.no_save:
            out_path = step4_tables_dir / f"step4_question_metric_pairwise_tests_{model_name}_{args.method}{file_suffix}.csv"
            legacy_path = legacy_analysis_dir / f"step4_question_metric_pairwise_tests_{model_name}_{csv_timestamp}.csv"
            _save_table(
                q_metric_df,
                out_path,
                output_manager=output_manager,
                artifact="question_metric_pairwise_tests",
                model=model_name,
                method=args.method,
                legacy_path=legacy_path,
            )
            print(f"\nQuestion metric test results saved to: {out_path}")

    # ========================================================================
    # Global multiple testing correction (optional; across Step-4 hypothesis tests)
    # ========================================================================
    if args.global_fdr:
        print("\n" + "=" * 80)
        print("GLOBAL MULTIPLE TESTING CORRECTION (BH-FDR ACROSS STEP-4 TESTS)")
        print("=" * 80)

        test_frames: list[tuple[str, pd.DataFrame]] = []
        if "lrt_df" in locals() and isinstance(lrt_df, pd.DataFrame) and len(lrt_df):
            test_frames.append(("rubric_lrt", lrt_df))
        if "cal_reg_df" in locals() and isinstance(cal_reg_df, pd.DataFrame) and len(cal_reg_df):
            test_frames.append(("rubric_calibration_regression", cal_reg_df))
        if "metric_df" in locals() and isinstance(metric_df, pd.DataFrame) and len(metric_df):
            test_frames.append(("rubric_metric", metric_df))
        if "q_lrt_df" in locals() and isinstance(q_lrt_df, pd.DataFrame) and len(q_lrt_df):
            test_frames.append(("question_lrt", q_lrt_df))
        if "q_metric_df" in locals() and isinstance(q_metric_df, pd.DataFrame) and len(q_metric_df):
            test_frames.append(("question_metric", q_metric_df))

        updated_frames, global_df = _apply_global_fdr_across_tests(test_frames, alpha=args.alpha)
        if len(global_df) == 0:
            print("No hypothesis-test tables found (nothing to adjust).")
        else:
            finite_mask = np.isfinite(global_df["p_value"].to_numpy(dtype=float))
            n_tests = int(finite_mask.sum())
            n_sig = int((global_df.loc[finite_mask, "p_adjusted_global"] < args.alpha).sum())
            print(f"Global BH-FDR across {n_tests} tests: {n_sig} significant at α={args.alpha}")

            by_family = (
                global_df.assign(_is_sig=(global_df["p_adjusted_global"] < args.alpha))
                .groupby("_global_family", dropna=False)["_is_sig"]
                .agg(["count", "sum"])
                .reset_index()
                .rename(columns={"count": "n_tests", "sum": "n_sig"})
            )
            print("\nBy family:")
            print(by_family.to_string(index=False))

            show_cols = [
                "_global_family",
                "test",
                "metric",
                "question_id",
                "outcome",
                "rubric_a",
                "rubric_b",
                "group_a",
                "group_b",
                "p_value",
                "p_adjusted_global",
                "sig_global",
            ]
            print("\nMost significant tests (global-adjusted):")
            print(
                global_df.sort_values(["p_adjusted_global", "p_value"], na_position="last")
                .head(50)[[c for c in show_cols if c in global_df.columns]]
                .to_string(index=False)
            )

            # Mark the publication-style "primary" adjusted p-values for each family table.
            for key, df in list(updated_frames.items()):
                if df is None or len(df) == 0:
                    continue
                if "p_adjusted_global" in df.columns:
                    df["p_adjusted_primary"] = df["p_adjusted_global"]
                    df["sig_primary"] = df.get("sig_global", "")
                    df["p_adjustment_primary"] = "BH_global_step4"
                    updated_frames[key] = df

            if "rubric_lrt" in updated_frames and len(updated_frames["rubric_lrt"]):
                omnibus = updated_frames["rubric_lrt"].query("test == 'omnibus_rubric'").copy()
                if len(omnibus):
                    cols = [
                        "question_id",
                        "outcome",
                        "chisq",
                        "df_diff",
                        "p_value",
                        "p_adjusted_primary",
                        "sig_primary",
                    ]
                    cols = [c for c in cols if c in omnibus.columns]
                    print("\nOmnibus rubric LRTs (global-adjusted; primary):")
                    print(omnibus[cols].sort_values(["p_adjusted_primary", "p_value"]).to_string(index=False))

            # Overwrite per-family CSVs to include the global columns (if previously saved).
            if not args.no_save:
                if "rubric_lrt" in updated_frames and len(updated_frames["rubric_lrt"]):
                    lrt_df = updated_frames["rubric_lrt"]
                    out_path = step4_tables_dir / f"step4_lrt_rubric_tests_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_lrt_rubric_tests_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        lrt_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="lrt_rubric_tests",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )
                if "rubric_calibration_regression" in updated_frames and len(updated_frames["rubric_calibration_regression"]):
                    cal_reg_df = updated_frames["rubric_calibration_regression"]
                    out_path = step4_tables_dir / f"step4_calibration_regression_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_calibration_regression_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        cal_reg_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="calibration_regression",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )
                if "rubric_metric" in updated_frames and len(updated_frames["rubric_metric"]):
                    metric_df = updated_frames["rubric_metric"]
                    out_path = step4_tables_dir / f"step4_metric_pairwise_tests_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_metric_pairwise_tests_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        metric_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="metric_pairwise_tests",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )
                if "question_lrt" in updated_frames and len(updated_frames["question_lrt"]):
                    q_lrt_df = updated_frames["question_lrt"]
                    out_path = step4_tables_dir / f"step4_lrt_question_tests_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_lrt_question_tests_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        q_lrt_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="lrt_question_tests",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )
                if "question_metric" in updated_frames and len(updated_frames["question_metric"]):
                    q_metric_df = updated_frames["question_metric"]
                    out_path = step4_tables_dir / f"step4_question_metric_pairwise_tests_{model_name}_{args.method}{file_suffix}.csv"
                    legacy_path = legacy_analysis_dir / f"step4_question_metric_pairwise_tests_{model_name}_{csv_timestamp}.csv"
                    _save_table(
                        q_metric_df,
                        out_path,
                        output_manager=output_manager,
                        artifact="question_metric_pairwise_tests",
                        model=model_name,
                        method=args.method,
                        legacy_path=legacy_path,
                    )

                out_path = step4_tables_dir / f"step4_global_fdr_tests_{model_name}_{args.method}{file_suffix}.csv"
                legacy_path = legacy_analysis_dir / f"step4_global_fdr_tests_{model_name}_{csv_timestamp}.csv"
                _save_table(
                    global_df,
                    out_path,
                    output_manager=output_manager,
                    artifact="global_fdr_tests",
                    model=model_name,
                    method=args.method,
                    legacy_path=legacy_path,
                )
                print(f"\nGlobal-FDR table saved to: {out_path}")

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for label, df_stratum in results_by_stratum.items():
        print(f"\n[{label}] Total groups: {len(df_stratum)}")
        if "ece" in df_stratum.columns:
            print(f"[{label}] Mean ECE across groups: {df_stratum['ece'].mean():.4f}")
            print(f"[{label}] Std ECE across groups: {df_stratum['ece'].std():.4f}")
            print(f"[{label}] ECE range: [{df_stratum['ece'].min():.4f}, {df_stratum['ece'].max():.4f}]")
        if "accuracy" in df_stratum.columns:
            print(f"[{label}] Mean Accuracy across groups: {df_stratum['accuracy'].mean():.4f}")

    # ========================================================================
    # Save Results
    # ========================================================================
    if not args.no_save:
        if args.stratify_by == "all":
            for label, df_stratum in results_by_stratum.items():
                results_path = step4_tables_dir / f"stratified_{label}_{model_name}_{args.method}{file_suffix}.csv"
                legacy_path = legacy_analysis_dir / f"stratified_{label}_{model_name}_{csv_timestamp}.csv"
                _save_table(
                    df_stratum,
                    results_path,
                    output_manager=output_manager,
                    artifact=f"stratified_{label}",
                    model=model_name,
                    method=args.method,
                    legacy_path=legacy_path,
                )
                print(f"\nResults saved to: {results_path}")
        else:
            results_path = step4_tables_dir / f"stratified_{args.stratify_by}_{model_name}_{args.method}{file_suffix}.csv"
            legacy_path = legacy_analysis_dir / f"stratified_{args.stratify_by}_{model_name}_{csv_timestamp}.csv"
            _save_table(
                results_df,
                results_path,
                output_manager=output_manager,
                artifact=f"stratified_{args.stratify_by}",
                model=model_name,
                method=args.method,
                legacy_path=legacy_path,
            )
            print(f"\nResults saved to: {results_path}")

    # ========================================================================
    # Rubric/question feature associations with group metrics (secondary)
    # ========================================================================
    if args.associate_group_features:
        if features_df is None:
            raise RuntimeError("Internal error: expected features_df for --associate-group-features.")

        print("\n" + "=" * 80)
        print("GROUP-LEVEL FEATURE ASSOCIATIONS (SECONDARY)")
        print("=" * 80)
        print(
            "Method: rubric-level associations use stratified Spearman (within-question ranks + stratified permutation/bootstrap). "
            "Question-level associations are descriptive only (n=4)."
        )

        assoc_rows: list[dict] = []
        metric_cols = ["accuracy", "ece", "brier_score", "auc_roc", "mean_confidence"]

        # Candidate feature columns: prefer group-text features (rubric_/question_) plus
        # a small set of aggregated answer/response descriptors.
        rubric_feature_cols = [
            c for c in features_df.columns
            if c.startswith("rubric_") and pd.api.types.is_numeric_dtype(features_df[c])
        ]
        question_feature_cols = [
            c for c in features_df.columns
            if c.startswith("question_") and pd.api.types.is_numeric_dtype(features_df[c])
        ]
        extra_mean_cols = [
            c for c in ["answer_length", "response_length", "answer_readability", "response_readability"]
            if c in features_df.columns and pd.api.types.is_numeric_dtype(features_df[c])
        ]

        if args.association_scope in {"rubric", "both"}:
            rubric_to_question = (
                df_out[["rubric_id", "question_id"]]
                .dropna()
                .drop_duplicates()
                .assign(rubric_id=lambda d: d["rubric_id"].astype(str), question_id=lambda d: d["question_id"].astype(str))
            )
            rubric_metrics = _compute_group_metrics_df(
                predictions,
                "rubric_id",
                compute_ci=False,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            ).rename(columns={"group": "rubric_id"})

            rubric_feats = _compute_group_features_df(
                features_df,
                "rubric_id",
                keep_cols=rubric_feature_cols + extra_mean_cols,
            )

            rubric_merged = rubric_metrics.merge(rubric_feats, on="rubric_id", how="left").merge(
                rubric_to_question, on="rubric_id", how="left"
            )

            n_rubrics = rubric_merged["rubric_id"].nunique()
            print(f"\nRubric-level associations: n_rubrics={n_rubrics}")

            for feat in [c for c in rubric_feats.columns if c != "rubric_id"]:
                for metric in metric_cols:
                    df_xy = rubric_merged[["question_id", feat, metric]].copy()
                    res = _spearman_within_strata(
                        df_xy,
                        x_col=feat,
                        y_col=metric,
                        strata_col="question_id",
                        n_perm=int(args.association_n_perm),
                        n_boot=int(args.association_n_boot),
                        alpha=args.alpha,
                        rng=rng,
                    )
                    rho = float(res.get("r", np.nan))
                    n_used = int(res.get("n", 0))
                    if n_used < 10 or not np.isfinite(rho):
                        continue
                    assoc_rows.append(
                        {
                            "group_level": "rubric",
                            "feature": feat,
                            "metric": metric,
                            "n_groups": n_used,
                            "rho": rho,
                            "rho_squared": float(rho * rho) if np.isfinite(rho) else np.nan,
                            "ci_low": float(res.get("ci_low", np.nan)),
                            "ci_high": float(res.get("ci_high", np.nan)),
                            "p_value": float(res.get("p_value", np.nan)),
                            "interpretation": _interpret_rho(rho),
                        }
                    )

        if args.association_scope in {"question", "both"}:
            # Only 4 questions: descriptive only by default.
            q_metrics = _compute_group_metrics_df(
                predictions,
                "question_id",
                compute_ci=False,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            ).rename(columns={"group": "question_id"})

            q_feats = _compute_group_features_df(
                features_df,
                "question_id",
                keep_cols=question_feature_cols + extra_mean_cols,
            )
            q_merged = q_metrics.merge(q_feats, on="question_id", how="left")
            n_q = q_merged["question_id"].nunique()
            print(f"\nQuestion-level associations: n_questions={n_q} (EXPLORATORY; too small for reliable inference)")

            for feat in [c for c in q_feats.columns if c != "question_id"]:
                x = q_merged[feat].to_numpy(dtype=float)
                for metric in metric_cols:
                    y = q_merged[metric].to_numpy(dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 4:
                        continue
                    # With n=4, report rho but skip p-values/CI by default.
                    from scipy.stats import spearmanr
                    rho = float(spearmanr(x[mask], y[mask]).correlation)
                    assoc_rows.append(
                        {
                            "group_level": "question",
                            "feature": feat,
                            "metric": metric,
                            "n_groups": int(mask.sum()),
                            "rho": rho,
                            "rho_squared": float(rho * rho) if np.isfinite(rho) else np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "p_value": np.nan,
                            "interpretation": "exploratory_n=4",
                        }
                    )

        assoc_df = pd.DataFrame(assoc_rows)
        if len(assoc_df) == 0:
            print("\nNo association results produced (no numeric group-level features found).")
        else:
            # Global BH correction across rubric-level tests only (question-level are exploratory NaN).
            mask_p = assoc_df["p_value"].notna() & np.isfinite(assoc_df["p_value"])
            p_adj = np.full(len(assoc_df), np.nan, dtype=float)
            sig = np.full(len(assoc_df), False, dtype=bool)
            if mask_p.any():
                p_adj_vals, sig_vals = benjamini_hochberg(assoc_df.loc[mask_p, "p_value"].to_numpy(float), alpha=args.alpha)
                p_adj[mask_p.to_numpy()] = p_adj_vals
                sig[mask_p.to_numpy()] = sig_vals
            assoc_df["p_adjusted"] = p_adj
            assoc_df["significant"] = sig
            assoc_df["sig"] = assoc_df["p_adjusted"].apply(lambda p: _format_sig(p, args.alpha))

            print("\nTop rubric-feature associations (by adjusted p-value):")
            show = assoc_df.sort_values(["p_adjusted", "p_value"], na_position="last").head(50)
            print(show[["group_level", "feature", "metric", "rho", "ci_low", "ci_high", "p_value", "p_adjusted", "sig", "interpretation"]].to_string(index=False))

            if not args.no_save:
                out_path = step4_tables_dir / f"step4_group_feature_associations_{model_name}_{args.method}{file_suffix}.csv"
                legacy_path = legacy_analysis_dir / f"step4_group_feature_associations_{model_name}_{csv_timestamp}.csv"
                _save_table(
                    assoc_df,
                    out_path,
                    output_manager=output_manager,
                    artifact="group_feature_associations",
                    model=model_name,
                    method=args.method,
                    legacy_path=legacy_path,
                )
                print(f"\nAssociation results saved to: {out_path}")

    if output_manager:
        manifest_path = output_manager.write_manifest()
        if manifest_path:
            print(f"\nManifest saved to: {manifest_path}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
