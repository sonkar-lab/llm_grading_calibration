"""
Step 3: Feature Analysis for Calibration Research

This script:
1. Loads predictions CSV and reconstructs GradingInstance + PredictionResult objects
2. Extracts comprehensive features for all text fields (question, answer, rubric, LLM response)
3. Computes correlations between features and confidence/accuracy/calibration error
4. Applies FDR correction for multiple testing and reports statistical power
5. Identifies which features most strongly predict confidence and correctness
6. Saves extracted features and correlation analysis to CSV

Usage:
    python step3_feature_analysis.py \
        --model qwen4b \
        --method self_reported \
        --save-features \
        --alpha 0.05 \
        --plot
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import atexit
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from tqdm import tqdm
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
from utils.statistics import (
    benjamini_hochberg,
    interpret_correlation,
    compute_spearman_with_perm_and_bootstrap,
    estimate_power_spearman_simulation,
    compute_power_curve,
    compute_power_curve_mixed_model_lrt,
    estimate_power_spearman_simulation_fresh_perms,
    compare_power_estimation_methods,
    compute_bayesian_assurance_bootstrap,
    compute_bayesian_assurance_posterior,
    compute_bayesian_assurance_cluster_robust,
    compute_bayesian_assurance_mixed_model_empirical,
    compute_icc_mixed_model,
    estimate_question_level_effect_mixed_model_bootstrap,
    estimate_feature_effect,
    estimate_feature_effect_nonlinear,
    loo_question_sensitivity,
    analyze_feature_moderated_calibration,
)
from analysis import FeatureExtractor, FeatureAnalyzer
from grading_calibration.pipelines.feature_pipeline import (
    build_instances_predictions_from_df as _pipeline_build_instances_predictions_from_df,
    extract_features_dataframe as _pipeline_extract_features_dataframe,
    get_response_columns as _pipeline_get_response_columns,
    load_predictions_csv as _pipeline_load_predictions_csv,
)
from grading_calibration.pipelines.regression_pipeline import (
    run_calibration_regression as _pipeline_run_calibration_regression,
    run_mixed_model_feature_effects as _pipeline_run_mixed_model_feature_effects,
)
from grading_calibration.metrics import expected_calibration_error
from grading_calibration.metrics.calibration_bins import assign_bin_calibration_errors
from utils.debug import inspect_dataframe, print_dataframe, FullDataFrameDisplay
from typing import Optional
from grading_calibration.inference.model_aliases import MODEL_ALIASES as model_nicknames


def build_parser():
    parser = argparse.ArgumentParser(
        description="Step 3: Feature analysis for calibration research"
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
        help="Dataset name (default: ricechem).",
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
        "--features-stream-chunk-size",
        type=int,
        default=None,
        help=(
            "Enable streaming feature extraction by reading the predictions CSV in chunks. "
            "Recommended: 2000. Default: None (load whole CSV)."
        ),
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=None,
        help="Limit to first N rows for a smoke test run.",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Save extracted features to CSV",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Number of top correlated features to display",
    )
    parser.add_argument(
        "--no-save-correlations",
        action="store_true",
        help="Do not save correlation results to CSV",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help="Number of permutations for Spearman permutation test (default: 1000)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of bootstrap samples for Spearman CI (default: 1000)",
    )
    parser.add_argument(
        "--n-power-sim",
        type=int,
        default=500,
        help="Number of simulations for Spearman power estimation (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutation/bootstrap/power simulations (default: 42)",
    )
    parser.add_argument(
        "--warnings-log",
        type=str,
        default=None,
        help="Write Python warnings to this file instead of stderr.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate correlation heatmaps and plots",
    )
    parser.add_argument(
        "--force-progress",
        action="store_true",
        help="Force tqdm progress bars even when stdout is not a TTY.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable optional diagnostics (vote vs calibration mapping, ICC checks)",
    )
    parser.add_argument(
        "--no-cluster-robust",
        action="store_true",
        help="Skip cluster-robust correlation tables (mixed-model effects still run).",
    )
    parser.add_argument(
        "--no-correlation-tables",
        action="store_true",
        help="Skip correlation tables and related summary outputs (mixed-model effects still run).",
    )
    parser.add_argument(
        "--effects-method",
        type=str,
        choices=["spearman_permutation", "mixed_model", "all"],
        default="spearman_permutation",
        help="Effect inference method: "
             "'spearman_permutation' (permutation Spearman; ignores clustering), "
             "'mixed_model' (Step-3 crossed mixed model + cluster-robust robustness table), "
             "or 'all' (report Spearman + mixed model; mixed model treated as primary when present).",
    )
    parser.add_argument(
        "--power-method",
        choices=["spearman_perm", "cluster_robust", "mixed_model_lrt", "all"],
        default="spearman_perm",
        help="Choose power calculation method for power curves: "
             "'spearman_perm' (permutation Spearman), "
             "'cluster_robust' (multiway cluster-robust regression), "
             "'mixed_model_lrt' (crossed mixed model LRT; requires rpy2/lme4), "
             "or 'all' (run spearman_perm + cluster_robust + mixed_model_lrt; only runs tests that were requested by --effects-method).",
    )
    parser.add_argument(
        "--student-col",
        default="student_id",
        help="Column name for student clustering IDs.",
    )
    parser.add_argument(
        "--question-col",
        default="question_id",
        dest="question_col",
        help="Column name for question clustering IDs.",
    )
    parser.add_argument(
        "--rubric-col",
        default="rubric_id",
        help="Column name for rubric clustering IDs (optional third level).",
    )
    parser.add_argument(
        "--power-curve",
        action="store_true",
        help="Compute and display prospective power curves for range of effect sizes",
    )
    parser.add_argument(
        "--validate-power-method",
        action="store_true",
        help="Validate that vectorized power estimation (with permutation reuse) "
             "produces similar results to fresh-permutation method",
    )
    parser.add_argument(
        "--power-curve-r-values",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
        help="Comma-separated list of effect values for power curve. "
             "For spearman_perm/cluster_robust this is r values; "
             "for mixed_model_lrt this is |beta| values unless --power-curve-beta-values is provided.",
    )
    parser.add_argument(
        "--power-curve-beta-values",
        type=str,
        default=None,
        help="Comma-separated list of |beta| values for mixed_model_lrt power curves "
             "(overrides --power-curve-r-values when provided).",
    )
    parser.add_argument(
        "--power-curve-n-students-target",
        type=int,
        default=None,
        help="Target number of students for mixed_model_lrt prospective power (default: current n_students).",
    )
    parser.add_argument(
        "--plot-power-curve",
        action="store_true",
        help="Generate and save power curve plot with smooth curve",
    )
    parser.add_argument(
        "--power-curve-n-points",
        type=int,
        default=50,
        help="Number of r values for smooth power curve plot (default: 50)",
    )
    parser.add_argument(
        "--adaptive-perm",
        dest="adaptive_perm",
        action="store_true",
        default=True,
        help="Enable adaptive permutation refinement near alpha (default: enabled)",
    )
    parser.add_argument(
        "--no-adaptive-perm",
        dest="adaptive_perm",
        action="store_false",
        help="Disable adaptive permutation refinement",
    )
    parser.add_argument(
        "--adaptive-base-n-perm",
        type=int,
        default=2000,
        help="Base number of permutations when adaptive-perm is enabled (default: 2000)",
    )
    parser.add_argument(
        "--adaptive-max-n-perm",
        type=int,
        default=20000,
        help="Maximum permutations for adaptive refinement (default: 20000)",
    )
    parser.add_argument(
        "--adaptive-band",
        type=str,
        default="0.5,2.0",
        help="Adaptive band multiples of alpha to trigger refinement (default: 0.5,2.0)",
    )
    parser.add_argument(
        "--bootstrap-method",
        type=str,
        choices=['percentile', 'bca'],
        default='bca',
        help="Bootstrap CI method: 'percentile' (faster) or 'bca' (better coverage, default)",
    )
    parser.add_argument(
        "--use-fresh-perms-power",
        action="store_true",
        help="Use fresh-permutation method for post-hoc power estimation (slower but theoretically pure)",
    )
    parser.add_argument(
        "--bayesian-assurance",
        action="store_true",
        help="Compute Bayesian assurance (expected power under empirical prior)",
    )
    parser.add_argument(
        "--assurance-method",
        type=str,
        choices=['bootstrap', 'posterior', 'both'],
        default='both',
        help="Method for constructing empirical prior (default: both)",
    )
    parser.add_argument(
        "--assurance-mode",
        choices=["fast", "faithful"],
        default="fast",
        help="Assurance mode for mixed_model_lrt: "
             "'fast' (power curve on |beta| + interpolation over empirical prior draws) or "
             "'faithful' (nested simulation per prior draw; expensive).",
    )
    parser.add_argument(
        "--assurance-beta-grid",
        type=str,
        default=None,
        help="Comma-separated |beta| grid for mixed_model_lrt assurance (optional; default derives from prior).",
    )
    parser.add_argument(
        "--n-power-sim-inner",
        type=int,
        default=100,
        help="Inner simulations per prior draw for mixed_model_lrt assurance_mode='faithful' (default: 100).",
    )
    parser.add_argument(
        "--assurance-winsorize-pct",
        type=float,
        default=None,
        help="Optional winsorization percentile (e.g., 2.5) for empirical beta prior draws in mixed_model_lrt assurance.",
    )
    parser.add_argument(
        "--assurance-n-target",
        type=int,
        default=500,
        help="Target sample size for assurance calculation (default: 500)",
    )
    parser.add_argument(
        "--n-bootstrap-assurance",
        type=int,
        default=500,
        help="Bootstrap samples for empirical prior (default: 500)",
    )
    parser.add_argument(
        "--n-posterior-samples",
        type=int,
        default=1000,
        help="Posterior samples for Bayesian method (default: 1000)",
    )
    parser.add_argument(
        "--n-power-sim-per-r",
        type=int,
        default=200,
        help="Power simulations per r value in assurance (default: 200)",
    )
    parser.add_argument(
        "--assurance-all-features",
        action="store_true",
        help="Compute assurance for all features (not just first) - uses reduced defaults for speed",
    )
    parser.add_argument(
        "--assurance-top-k",
        type=int,
        default=None,
        help="Only compute assurance for top k features by |r| (default: None = all)",
    )
    parser.add_argument(
        "--assurance-for-significant-only",
        action="store_true",
        help="Only compute assurance for features with p_adjusted < alpha",
    )
    parser.add_argument(
        "--icc-method",
        type=str,
        choices=['anova', 'mixed_model', 'both'],
        default='anova',
        help="ICC computation method: 'anova' (current ANOVA-style), "
             "'mixed_model' (cross-classified random effects), or 'both'"
    )

    # --------------------------------------------------------------------
    # Step-3 mixed-model feature-effect screening config (used when effects-method includes mixed_model)
    # --------------------------------------------------------------------
    parser.add_argument(
        "--effects-max-features",
        type=int,
        default=None,
        help="Optional cap on number of features to analyze in mixed-model effects screening "
             "(for quick smoke runs). Only used when effects-method includes mixed_model.",
    )
    parser.add_argument(
        "--effects-include-prefixes",
        type=str,
        default="",
        help="Comma-separated list of feature prefixes to include (e.g., 'question_,answer_'). "
             "If set, only features with these prefixes are kept.",
    )
    parser.add_argument(
        "--effects-include-features",
        type=str,
        default="",
        help="Comma-separated list of exact feature names to include. "
             "Applied after prefix filtering; only listed features are kept.",
    )
    parser.add_argument(
        "--effects-include-features-file",
        type=str,
        default="",
        help="Path to a file with one feature name per line to include. "
             "Blank lines and lines starting with '#' are ignored.",
    )
    parser.add_argument(
        "--effects-exclude-prefixes",
        type=str,
        default="",
        help="Comma-separated list of feature prefixes to exclude from analysis.",
    )
    parser.add_argument(
        "--effects-question-constant-separate-block",
        action="store_true",
        help="Run question-constant features in a separate mixed-model/GLM block with no question effect, "
             "and exclude them from the primary mixed-model tables.",
    )
    parser.add_argument(
        "--effects-gee-mode",
        choices=["simple", "rubric_fe"],
        default="simple",
        help="GEE sensitivity mode for mixed-model effects: 'simple' uses C(question_id); "
             "'rubric_fe' uses C(rubric_id).",
    )
    parser.add_argument(
        "--effects-transform-bounded",
        choices=["none", "logit_clip"],
        default="none",
        help="Optional transform for bounded [0,1] continuous outcomes in mixed-model effects.",
    )
    parser.add_argument(
        "--effects-include-student-vc-glmm",
        action="store_true",
        help="Include a student variance component in the binary VB GLMM (slower, models cross-answer correlation).",
    )
    parser.add_argument(
        "--effects-use-ranks",
        action="store_true",
        help="Rank-transform feature and continuous outcomes before mixed-model estimation (robust/Spearman-style).",
    )
    parser.add_argument(
        "--effects-standardize-outcome",
        action="store_true",
        help="Z-score continuous outcomes before fitting (default keeps outcomes in original units).",
    )
    parser.add_argument(
        "--effects-optimizer",
        type=str,
        choices=["powell", "lbfgs", "bfgs", "cg", "nm"],
        default="powell",
        help="Optimizer for MixedLM fitting. 'powell' (default, robust) or 'lbfgs'/'bfgs' (faster, gradient-based).",
    )
    parser.add_argument(
        "--effects-lme4-maxfun",
        type=int,
        default=100000,
        help="Max function evaluations for lme4 mixed-model fits (default: 100000).",
    )
    parser.add_argument(
        "--effects-backend",
        choices=["statsmodels", "lme4"],
        default="lme4",
        help="Mixed-model backend: 'lme4' (R via rpy2) or 'statsmodels' (Python).",
    )
    parser.add_argument(
        "--effects-no-gee",
        action="store_true",
        help="Skip GEE sensitivity check (faster, primary model only).",
    )

    # --------------------------------------------------------------------
    # Step-3 non-linear feature effects (crossed mixed models)
    # --------------------------------------------------------------------
    parser.add_argument(
        "--effects-nonlinear",
        action="store_true",
        help="Run non-linear mixed-model feature effects (spline/binned/poly).",
    )
    parser.add_argument(
        "--effects-nonlinear-only",
        action="store_true",
        help="Run non-linear effects but skip linear mixed-model feature screening.",
    )
    parser.add_argument(
        "--effects-nonlinear-method",
        choices=["spline", "binned", "poly", "all"],
        default="spline",
        help="Nonlinear basis for f(feature): spline, binned, poly, or all.",
    )
    parser.add_argument(
        "--effects-nonlinear-spline-df",
        type=int,
        default=4,
        help="Spline degrees of freedom for nonlinear effects (default: 4).",
    )
    parser.add_argument(
        "--effects-nonlinear-bins",
        type=int,
        default=5,
        help="Number of bins for binned nonlinear effects (default: 5).",
    )
    parser.add_argument(
        "--effects-nonlinear-poly-degree",
        type=int,
        default=2,
        help="Polynomial degree for nonlinear effects (default: 2).",
    )
    parser.add_argument(
        "--effects-nonlinear-n-grid",
        type=int,
        default=50,
        help="Number of points for nonlinear effect curves (default: 50).",
    )
    parser.add_argument(
        "--effects-nonlinear-bootstrap",
        type=int,
        default=100,
        help="Bootstrap samples for nonlinear effect CIs (default: 100; set 0 to disable).",
    )
    parser.add_argument(
        "--effects-nonlinear-ci-method",
        choices=["percentile", "bca"],
        default="percentile",
        help="CI method for nonlinear effects (default: percentile; bca is slower but more accurate).",
    )
    parser.add_argument(
        "--effects-nonlinear-ci-type",
        choices=["pointwise", "simultaneous"],
        default="pointwise",
        help="CI type for nonlinear curves (default: pointwise; simultaneous uses max-deviation band).",
    )
    parser.add_argument(
        "--effects-nonlinear-jackknife-max",
        type=int,
        default=200,
        help="Max jackknife units for BCa (default: 200; reduce to speed up).",
    )
    parser.add_argument(
        "--effects-nonlinear-bootstrap-cluster",
        choices=["student", "row"],
        default="student",
        help="Bootstrap unit for nonlinear effects (default: student).",
    )
    parser.add_argument(
        "--effects-nonlinear-max-features",
        type=int,
        default=None,
        help="Optional cap on number of features for nonlinear effects (per target).",
    )
    parser.add_argument(
        "--effects-nonlinear-skip-question-features",
        action="store_true",
        help="Exclude question_* features from nonlinear effects.",
    )
    parser.add_argument(
        "--effects-nonlinear-plot-max",
        type=int,
        default=10,
        help="Max number of nonlinear curves to plot (default: 10).",
    )

    # --------------------------------------------------------------------
    # Calibration regression analysis flags (replaces bin-assigned calibration_error)
    # --------------------------------------------------------------------
    parser.add_argument(
        "--no-calibration-regression",
        action="store_true",
        help="Skip calibration regression analysis (not recommended). "
             "Calibration regression tests whether features moderate the confidence-accuracy relationship.",
    )
    parser.add_argument(
        "--calibration-regression-only",
        action="store_true",
        help="Run calibration regression only (skip mixed-model effects, correlations, power/assurance, and plots).",
    )
    parser.add_argument(
        "--calibration-regression-features",
        type=str,
        nargs="+",
        default=None,
        help="Specific features to test in calibration regression (default: all numeric features). "
             "Example: --calibration-regression-features answer_length response_length",
    )
    parser.add_argument(
        "--calibration-regression-mode",
        choices=["linear", "nonlinear"],
        default="linear",
        help="Calibration regression mode (default: linear).",
    )
    parser.add_argument(
        "--calibration-regression-spline-df",
        type=int,
        default=4,
        help="Spline df for nonlinear calibration regression (default: 4).",
    )
    parser.add_argument(
        "--calibration-regression-bootstrap",
        type=int,
        default=0,
        help="Bootstrap samples for nonlinear calibration regression AME CI (default: 0).",
    )
    parser.add_argument(
        "--calibration-regression-ci-method",
        choices=["percentile", "bca"],
        default="percentile",
        help="CI method for nonlinear calibration regression (default: percentile).",
    )
    parser.add_argument(
        "--calibration-regression-jackknife-max",
        type=int,
        default=200,
        help="Max jackknife units for BCa in nonlinear calibration regression (default: 200).",
    )
    parser.add_argument(
        "--calibration-regression-n-grid",
        type=int,
        default=50,
        help="Grid size for nonlinear calibration regression curves (default: 50).",
    )
    parser.add_argument(
        "--calibration-regression-max-features",
        type=int,
        default=None,
        help="Optional cap on number of features for calibration regression.",
    )
    parser.add_argument(
        "--include-bin-calibration-error",
        action="store_true",
        help="Include bin-assigned calibration_error correlations (exploratory, not recommended). "
             "This adds calibration_error as a third target alongside confidence and correctness.",
    )
    parser.add_argument(
        "--calibration-regression-glmer-nAGQ",
        type=int,
        default=1,
        help="Number of adaptive Gauss-Hermite quadrature points for calibration regression GLMM (default: 1 = Laplace).",
    )
    parser.add_argument(
        "--calibration-regression-optimizer",
        type=str,
        default="bobyqa",
        help="Optimizer for calibration regression glmer (default: bobyqa).",
    )
    parser.add_argument(
        "--calibration-regression-maxfun",
        type=int,
        default=100000,
        help="Max function evaluations for calibration regression optimizer (default: 100000).",
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
    # Keep the runtime Namespace focused on analysis options; --config only
    # controls parser defaults and is not consumed downstream.
    if hasattr(args, "config"):
        delattr(args, "config")
    return args


def load_path_by_index(
    idx: int,
    folder: Path | str,
    prefix: str,
    exclude_substrings: Optional[list[str]] = None,
) -> Path:
    """Load CSV path by index (0 = most recent)."""
    folder = Path(folder)
    print(f"Loading CSV from folder: {folder} with prefix: {prefix}")
    paths = sorted(folder.glob(f"{prefix}_*.csv"))
    if exclude_substrings:
        filtered = []
        for path in paths:
            stem = path.stem
            if any(token in stem for token in exclude_substrings):
                continue
            filtered.append(path)
        paths = filtered
    print(f"Found {len(paths)} matching CSV files.")
    new_idx = len(paths) - 1 - idx
    return paths[new_idx]


def load_data_from_csv(csv_path: str | Path, smoke_n: int | None = None) -> tuple:
    """
    Load predictions CSV and reconstruct GradingInstance + PredictionResult objects.

    Returns:
        Tuple of (instances, predictions, df)
    """
    return _pipeline_load_predictions_csv(
        csv_path,
        smoke_n=smoke_n,
        include_answer_key=False,
        include_calibrated_confidence=False,
        show_progress=True,
    )


def _get_response_columns(columns: list[str]) -> list[str]:
    return _pipeline_get_response_columns(columns)


def _build_instances_predictions_from_df(
    df: pd.DataFrame,
    response_cols: list[str],
    *,
    include_responses: bool,
    include_llm_response: bool,
    show_progress: bool = True,
) -> tuple[list[GradingInstance], list[PredictionResult]]:
    return _pipeline_build_instances_predictions_from_df(
        df,
        response_cols,
        include_responses=include_responses,
        include_llm_response=include_llm_response,
        include_answer_key=True,
        include_calibrated_confidence=True,
        show_progress=show_progress,
    )

def _stream_instances_predictions_minimal(
    csv_path: Path,
    *,
    chunk_size: int,
    smoke_n: int | None,
    usecols: list[str],
    response_cols: list[str],
    include_llm_response: bool,
) -> tuple[list[GradingInstance], list[PredictionResult]]:
    instances: list[GradingInstance] = []
    predictions: list[PredictionResult] = []
    processed = 0

    progress = tqdm(total=smoke_n, desc="Loading instances (streaming)", unit="row")
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        if smoke_n is not None:
            remaining = smoke_n - processed
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        print(f"Processing chunk with {len(chunk)} rows")
        chunk_instances, chunk_predictions = _build_instances_predictions_from_df(
            chunk,
            response_cols,
            include_responses=False,
            include_llm_response=include_llm_response,
            show_progress=False,
        )
        instances.extend(chunk_instances)
        predictions.extend(chunk_predictions)
        processed += len(chunk)
        progress.update(len(chunk))
        del chunk_instances, chunk_predictions, chunk
        gc.collect()
    progress.close()
    return instances, predictions


def _stream_extract_features(
    csv_path: Path,
    *,
    extractor: FeatureExtractor,
    chunk_size: int,
    smoke_n: int | None,
    usecols: list[str],
    response_cols: list[str],
    include_llm_response: bool,
    cache_path: Optional[Path],
    cache_meta: Optional[dict],
    cache_mode: str,
    output_manager,
    model_name: str,
    method: str,
    cache_version: str,
) -> tuple[list[GradingInstance], list[PredictionResult], pd.DataFrame]:
    instances: list[GradingInstance] = []
    predictions: list[PredictionResult] = []
    features_chunks: list[pd.DataFrame] = []
    processed = 0

    temp_path = None
    if cache_path and cache_mode in ("read_write", "refresh"):
        temp_path = Path(str(cache_path) + ".tmp")
        if temp_path.exists():
            temp_path.unlink()

    progress = tqdm(total=smoke_n, desc="Streaming features", unit="row")
    try:
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            if smoke_n is not None:
                remaining = smoke_n - processed
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            chunk_instances, chunk_predictions = _build_instances_predictions_from_df(
                chunk,
                response_cols,
                include_responses=True,
                include_llm_response=include_llm_response,
                show_progress=False,
            )
            chunk_df = _pipeline_extract_features_dataframe(
                chunk_instances, chunk_predictions, extractor=extractor
            )
            if chunk_df.index.name == "instance_id" and "instance_id" not in chunk_df.columns:
                chunk_df = chunk_df.reset_index()
            if temp_path is not None:
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                chunk_df.to_csv(
                    temp_path,
                    index=False,
                    mode="a",
                    header=not temp_path.exists(),
                )
            else:
                features_chunks.append(chunk_df)

            for pred in chunk_predictions:
                pred.metadata.pop("llm_response", None)
                pred.metadata.pop("llm_responses", None)
            instances.extend(chunk_instances)
            predictions.extend(chunk_predictions)
            processed += len(chunk)
            progress.update(len(chunk))
            del chunk_df, chunk_instances, chunk_predictions, chunk
            gc.collect()
    finally:
        progress.close()

    if temp_path is not None:
        temp_path.replace(cache_path)
        print(f"Saved features cache: {cache_path}")
        if cache_meta:
            write_features_cache_metadata(cache_path, cache_meta)
        if output_manager:
            output_manager.register(
                step="step3",
                artifact="features_cache",
                path=cache_path,
                model=model_name,
                method=method,
                extra={"cache_version": cache_version},
            )
        features_df = pd.read_csv(cache_path)
    else:
        features_df = pd.concat(features_chunks, ignore_index=True) if features_chunks else pd.DataFrame()

    if "instance_id" in features_df.columns:
        features_df.set_index("instance_id", inplace=True, drop=False)
        ordered_ids = [p.instance_id for p in predictions]
        features_df = features_df.reindex(ordered_ids)
    return instances, predictions, features_df


def _stream_extract_features_for_ids(
    csv_path: Path,
    *,
    extractor: FeatureExtractor,
    chunk_size: int,
    smoke_n: int | None,
    usecols: list[str],
    response_cols: list[str],
    include_llm_response: bool,
    target_ids: set[str],
) -> pd.DataFrame:
    features_chunks: list[pd.DataFrame] = []
    processed = 0
    progress = tqdm(total=smoke_n, desc="Streaming missing features", unit="row")
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        if smoke_n is not None:
            remaining = smoke_n - processed
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        if "instance_id" not in chunk.columns:
            continue
        sub = chunk[chunk["instance_id"].isin(target_ids)]
        if sub.empty:
            processed += len(chunk)
            progress.update(len(chunk))
            continue
        chunk_instances, chunk_predictions = _build_instances_predictions_from_df(
            sub,
            response_cols,
            include_responses=True,
            include_llm_response=include_llm_response,
            show_progress=False,
        )
        chunk_df = _pipeline_extract_features_dataframe(
                chunk_instances, chunk_predictions, extractor=extractor
            )
        if chunk_df.index.name == "instance_id" and "instance_id" not in chunk_df.columns:
            chunk_df = chunk_df.reset_index()
        features_chunks.append(chunk_df)
        processed += len(chunk)
        progress.update(len(chunk))
        del chunk_instances, chunk_predictions, chunk_df, sub
        gc.collect()
    progress.close()
    if not features_chunks:
        return pd.DataFrame()
    features_df = pd.concat(features_chunks, ignore_index=True)
    if "instance_id" in features_df.columns:
        features_df.set_index("instance_id", inplace=True, drop=False)
    return features_df


# REMOVED: compute_calibration_error_per_instance()
#
# Per-instance calibration error |confidence - correctness| is methodologically flawed.
# It mechanically correlates with confidence because:
#   - High confidence wrong prediction → large error (0.9)
#   - Low confidence wrong prediction → small error (0.1)
# Features correlating with confidence automatically correlate with per-instance error,
# leading to spurious discoveries.
#
# Instead, use assign_bin_calibration_errors() from grading_calibration.metrics.calibration_bins,
# which assigns each prediction its bin's calibration error (group-level property).
# This measures true calibration quality without mechanical correlation.


def print_correlation_table(
    correlation_results: pd.DataFrame, 
    title: str, 
    top_k: int = 15,
    alpha: float = 0.05
):
    """
    Print a formatted correlation table with p-values, power, and FDR correction.

    Args:
        correlation_results: DataFrame with columns: feature, r, r_squared, p_value, p_adjusted, power, significant
        title: Table title
        top_k: Number of top features to show
        alpha: Significance level
    """
    print(f"\n{title}")
    print("-" * 135)
    print(f"{'Feature':<35} {'r':>8} {'r²':>8} {'CI_low':>10} {'CI_high':>10} {'p-perm':>10} {'p-adj':>10} {'Power':>10} {'Interp':>12} {'Sig':>5}")
    print("-" * 135)

    for _, row in correlation_results.head(top_k).iterrows():
        feature = row['feature']
        r = row['r']
        r_squared = row['r_squared']
        ci_low = row.get('ci_low', np.nan)
        ci_high = row.get('ci_high', np.nan)
        p_val = row['p_value']
        p_adj = row['p_adjusted']
        power = row['power']
        interp = interpret_correlation(r)
        sig = "***" if row['significant'] else ""

        print(f"{feature:<35} {r:>8.6f} {r_squared:>8.6f} {ci_low:>10.6f} {ci_high:>10.6f} {p_val:>10.6f} {p_adj:>10.6f} {power:>10.2f} {interp:>12} {sig:>5}")

    # Summary
    n_sig = correlation_results['significant'].sum()
    n_total = len(correlation_results)
    print("-" * 135)
    print(f"Significant after FDR correction (α={alpha}): {n_sig}/{n_total}")


    # Power warnings
    low_power = correlation_results[correlation_results['power'] < 0.8]
    if len(low_power) > 0:
        print(f"⚠ {len(low_power)} features have power < 0.8 (may miss true effects)")

    # Effect size interpretation note
    print(f"\nNote: r² shows % variance explained. With large n={len(correlation_results)}, even small")
    print(f"      correlations are significant. Focus on r² > 0.01 (>1% variance) for practical importance.")


def print_cluster_robust_table(
    results: pd.DataFrame,
    title: str,
    top_k: int = 15,
    alpha: float = 0.05
):
    """
    Print cluster-robust regression results in formatted table.

    Args:
        results: DataFrame from compute_correlations_cluster_robust()
        title: Section title
        top_k: Number of top features to show
        alpha: Significance threshold
    """
    def _print_table(df: pd.DataFrame, heading: str, beta_col: str = 'beta', se_col: str = 'se',
                    p_col: str = 'p_adjusted', ci_low_col: str = 'ci_low', ci_high_col: str = 'ci_high'):
        print(f"\n{heading}")
        print("-" * 175)
        print(f"{'Feature':<35} {'Beta':>8} {'SE':>8} {'CI_low':>10} {'CI_high':>10} "
              f"{'Strength':>12} {'z':>8} {'p-adj':>10} {'n_clust(S)':>12} {'n_clust(Q)':>12} {'n_clust(R)':>12} {'Sig':>5}")
        print("-" * 175)

        df_print = df.copy()
        # Drop rows that would show NaNs in the table (keep them in saved CSVs).
        if beta_col in df_print.columns:
            df_print = df_print[np.isfinite(df_print[beta_col].to_numpy(dtype=float, copy=False))]
        if p_col in df_print.columns:
            df_print = df_print[np.isfinite(df_print[p_col].to_numpy(dtype=float, copy=False))]

        for _, row in df_print.head(top_k).iterrows():
            feature = row['feature']
            beta = row.get(beta_col, np.nan)
            se = row.get(se_col, np.nan)
            ci_low = row.get(ci_low_col, np.nan)
            ci_high = row.get(ci_high_col, np.nan)
            z = row.get('z', np.nan)
            p_val = row.get(p_col, np.nan)
            n_clust_s = row.get('n_clusters_student', np.nan)
            n_clust_q = row.get('n_clusters_question', np.nan)
            n_clust_r = row.get('n_clusters_rubric', np.nan)
            # Significance is based on adjusted p-values (BH-FDR within target).
            sig = "***" if bool(row.get('significant', False)) else ""

            strength = ""
            if np.isfinite(beta):
                strength = "extreme" if abs(beta) > 1.0 else interpret_correlation(float(beta))

            print(f"{feature:<35} {beta:>8.4f} {se:>8.4f} {ci_low:>10.4f} {ci_high:>10.4f} "
                  f"{strength:>12} {z:>8.4f} {p_val:>10.6f} {n_clust_s:>12,} {n_clust_q:>12,} {n_clust_r:>12,} {sig:>5}")

    # Primary table
    print(f"\n{title}")
    _print_table(results, "Primary effects", p_col='p_adjusted')

    # If robustness columns exist, print a secondary table
    if {'beta_cluster_robust', 'se_cluster_robust', 'p_value_cluster_robust',
        'ci_low_cluster_robust', 'ci_high_cluster_robust'}.issubset(results.columns):
        robust_df = results.copy()
        robust_df['beta'] = robust_df['beta_cluster_robust']
        robust_df['se'] = robust_df['se_cluster_robust']
        robust_df['p_value'] = robust_df['p_value_cluster_robust']
        robust_df['ci_low'] = robust_df['ci_low_cluster_robust']
        robust_df['ci_high'] = robust_df['ci_high_cluster_robust']
        # Compute BH-FDR on robustness p-values for honest stars / p-adj display.
        p = robust_df['p_value'].to_numpy(dtype=float)
        p_filled = np.where(np.isfinite(p), p, 1.0)
        p_adj, sig = benjamini_hochberg(p_filled, alpha)
        robust_df['p_adjusted'] = p_adj
        robust_df['significant'] = sig
        robust_df.loc[~np.isfinite(p), 'significant'] = False
        print("\nRobustness (cluster-robust) effects")
        _print_table(robust_df, "Robustness (cluster-robust) effects", p_col='p_adjusted')

    # Summary
    if 'significant_raw' in results.columns:
        n_significant_raw = int(results['significant_raw'].sum())
    elif 'p_value' in results.columns:
        p = results['p_value'].to_numpy(dtype=float)
        n_significant_raw = int((np.isfinite(p) & (p < alpha)).sum())
    else:
        n_significant_raw = 0

    n_significant = int(results['significant'].sum()) if 'significant' in results.columns else 0
    n_total = len(results)
    print("-" * 140)
    print(f"Significant features (before FDR correction): {n_significant_raw} / {n_total}")
    if 'p_adjusted' in results.columns:
        print(f"Significant features (after FDR correction):  {n_significant} / {n_total}")
    print(f"Note: Beta coefficients from rank-standardized regression (interpretable as correlation-like)")
    print(f"      Standard errors are multiway cluster-robust (student × question × rubric when provided)")


def compute_correlations_with_stats(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    alpha: float = 0.05,
    n_perm: int = 1000,
    n_boot: int = 1000,
    n_power_sim: int = 500,
    compute_power: bool = True,
    use_fresh_perms: bool = False,
    rng: Optional[np.random.Generator] = None,
    adaptive_perm: bool = False,
    adaptive_base_n_perm: int = 2000,
    adaptive_max_n_perm: int = 20000,
    adaptive_band: tuple = (0.5, 2.0),
    bootstrap_method: str = "bca",
) -> pd.DataFrame:
    """
    Compute Spearman correlations with permutation p-values, bootstrap CIs,
    FDR correction, and optional simulation-based power.

    Args:
        features_df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        alpha: Significance level
        n_perm: Number of permutations for p-value
        n_boot: Number of bootstrap samples for CI
        n_power_sim: Number of simulations for power estimation
        compute_power: If False, skip power simulation and set power=NaN
        use_fresh_perms: Use fresh-permutation method for power (slower but theoretically pure)
        rng: Optional numpy Generator for reproducibility

    Returns:
        DataFrame with correlation statistics
    """
    rng = rng or np.random.default_rng()
    results = []

    for feature in tqdm(feature_cols, desc=f"Computing {target_col} correlations", unit="feature"):
        try:
            x = features_df[feature].values
            y = features_df[target_col].values
            
            # Remove NaN pairs
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            n_valid = len(x_clean)
            
            if n_valid < 4:
                results.append({
                    'feature': feature,
                    'r': np.nan,
                    'p_value': np.nan,
                    'ci_low': np.nan,
                    'ci_high': np.nan,
                    'n': n_valid,
                    'power': np.nan,
                    'required_n': np.nan,  # analytic, not used (kept for compatibility)
                })
                continue
            
            # Spearman with permutation p-value and bootstrap CI
            print(f"Computing correlation for feature: {feature} (n={n_valid})")
            res = compute_spearman_with_perm_and_bootstrap(
                x_clean,
                y_clean,
                n_perm=n_perm,
                n_boot=n_boot,
                alpha=alpha,
                random_state=rng,
                log_progress=True,
                adaptive_perm=adaptive_perm,
                base_n_perm=adaptive_base_n_perm,
                max_n_perm=adaptive_max_n_perm,
                adaptive_band=adaptive_band,
                bootstrap_method=bootstrap_method,
            )
            r_val = res['rho']
            p_val = res['p_perm']
            ci_low = res['ci_low']
            ci_high = res['ci_high']

            power_val = np.nan
            if compute_power:
                # Simulation-based power (Monte Carlo)
                if use_fresh_perms:
                    print(f"Estimating power for feature: {feature} (fresh-permutation method)")
                    power_val = estimate_power_spearman_simulation_fresh_perms(
                        x_clean,
                        y_clean,
                        n_target=n_valid,
                        alpha=alpha,
                        n_power_sim=n_power_sim,
                        n_perm=n_perm,
                        random_state=rng,
                        log_progress=True,
                        adaptive_perm=adaptive_perm,
                        base_n_perm=adaptive_base_n_perm,
                        max_n_perm=adaptive_max_n_perm,
                        adaptive_band=adaptive_band,
                    )
                else:
                    print(f"Estimating power for feature: {feature} (vectorized method)")
                    power_val = estimate_power_spearman_simulation(
                        x_clean,
                        y_clean,
                        n_target=n_valid,
                        alpha=alpha,
                        n_power_sim=n_power_sim,
                        n_perm=n_perm,
                        random_state=rng,
                        log_progress=True,
                    )

            # Effect size (variance explained)
            r_squared = r_val ** 2

            results.append({
                'feature': feature,
                'r': r_val,
                'r_squared': r_squared,
                'p_value': p_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n': n_valid,
                'power': power_val,
                'required_n': np.nan,  # analytic Fisher z not used by default
            })
            
        except Exception as e:
            # Avoid spamming stdout with per-feature warnings; keep details in the output table.
            results.append({
                'feature': feature,
                'r': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'n': 0,
                'power': np.nan,
                'required_n': np.nan,
                'error': str(e),
            })
    
    df = pd.DataFrame(results)

    # FDR correction
    valid_p = df['p_value'].dropna()
    if len(valid_p) > 0:
        p_values = np.array(df['p_value'].values, dtype=float)
        # Replace NaN with 1.0 for FDR calculation
        p_values_filled = np.where(np.isnan(p_values), 1.0, p_values)
        p_adjusted, significant = benjamini_hochberg(p_values_filled, alpha)
        df['p_adjusted'] = p_adjusted
        df['significant'] = significant
        # Mark NaN p-values as not significant
        df.loc[df['p_value'].isna(), 'significant'] = False
    else:
        df['p_adjusted'] = np.nan
        df['significant'] = False
    
    # Sort by absolute correlation
    df['abs_r'] = df['r'].abs()
    df = df.sort_values('abs_r', ascending=False).drop('abs_r', axis=1)
    
    return df


def compute_correlations_cluster_robust(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    student_ids: np.ndarray,
    question_ids: Optional[np.ndarray] = None,
    rubric_ids: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    use_ranks: bool = True,
    standardize: bool = True,
    small_cluster_primary: bool = False,
    mixed_model_backend: str = "statsmodels",
    mixed_model_optimizer: str = "powell",
    mixed_model_transform_bounded: str = "none",
    mixed_model_include_student_vc_glmm: bool = False,
) -> pd.DataFrame:
    """
    Compute feature-target correlations with cluster-robust inference.

    Uses linear regression with two-way cluster-robust SEs (student × question).

    Args:
        features_df: DataFrame with all features
        feature_cols: List of feature column names
        target_col: Target variable column name
        student_ids: Student cluster IDs (length = len(features_df))
        question_ids: Question cluster IDs (if None, no question clustering)
        alpha: Significance level
        use_ranks: Use rank transformation (Spearman-style)
        standardize: Standardize variables

    Returns:
        DataFrame with columns:
        ['feature', 'beta', 'se', 'z', 'p_value', 'ci_low', 'ci_high',
         'n_obs', 'n_clusters_student', 'n_clusters_question', 'significant']
    """
    from utils.statistics import (
        estimate_feature_effect_cluster_robust,
        estimate_feature_effect,
    )

    results = []

    for feature in tqdm(feature_cols, desc=f"Cluster-robust correlation with {target_col}"):
        # Extract data
        x = features_df[feature].values
        y = features_df[target_col].values

        # Check for valid data
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 4:
            # Insufficient data
            results.append({
                'feature': feature,
                'beta': np.nan,
                'se': np.nan,
                'z': np.nan,
                'p_value': 1.0,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'n_obs': mask.sum(),
                'n_clusters_student': 0,
                'n_clusters_question': 0,
                'n_clusters_rubric': 0,
                'significant': False,
            })
            continue

        # Filter
        x_clean = x[mask]
        y_clean = y[mask]
        student_ids_clean = student_ids[mask]
        question_ids_clean = question_ids[mask] if question_ids is not None else None
        rubric_ids_clean = rubric_ids[mask] if rubric_ids is not None else None

        # Estimate with cluster-robust SEs
        try:
            result_robust = estimate_feature_effect_cluster_robust(
                feature=x_clean,
                outcome=y_clean,
                student_ids=student_ids_clean,
                question_ids=question_ids_clean,
                rubric_ids=rubric_ids_clean,
                use_ranks=use_ranks,
                standardize=standardize,
                alpha=alpha,
            )
            result_robust['feature'] = feature
            result_robust['significant'] = result_robust['p_value'] < alpha

            if small_cluster_primary:
                try:
                    if question_ids_clean is None or rubric_ids_clean is None:
                        raise ValueError("Mixed-model estimation requires question_ids and rubric_ids.")

                    outcome_type = "binary" if target_col == "is_correct" else "continuous"
                    mixed_res = estimate_feature_effect(
                        feature=x_clean,
                        outcome=y_clean,
                        student_ids=student_ids_clean,
                        question_ids=question_ids_clean,
                        rubric_ids=rubric_ids_clean,
                        outcome_type=outcome_type,
                        backend=mixed_model_backend,
                        run_gee_sensitivity=False,
                        use_ranks=use_ranks,
                        standardize_feature=standardize,
                        standardize_outcome=(standardize and outcome_type == "continuous"),
                        transform_bounded=mixed_model_transform_bounded,
                        include_student_vc_glmm=mixed_model_include_student_vc_glmm,
                        alpha=alpha,
                        optimizer=mixed_model_optimizer,
                        maxfun=args.effects_lme4_maxfun,
                    )
                    row = {
                        'feature': feature,
                        'beta': mixed_res.beta,
                        'se': mixed_res.se,
                        'z': mixed_res.z,
                        'p_value': mixed_res.p_value,
                        'ci_low': mixed_res.ci_low,
                        'ci_high': mixed_res.ci_high,
                        'n_obs': mixed_res.n_obs,
                        'n_clusters_student': result_robust['n_clusters_student'],
                        'n_clusters_question': result_robust['n_clusters_question'],
                        'n_clusters_rubric': result_robust['n_clusters_rubric'],
                        'significant': mixed_res.p_value < alpha,
                        'random_effects_used': ','.join(mixed_res.random_effects) if mixed_res.random_effects else '',
                        'beta_cluster_robust': result_robust['beta'],
                        'se_cluster_robust': result_robust['se'],
                        'p_value_cluster_robust': result_robust['p_value'],
                        'ci_low_cluster_robust': result_robust['ci_low'],
                        'ci_high_cluster_robust': result_robust['ci_high'],
                    }
                    results.append(row)
                except Exception as e:
                    result_robust['beta_cluster_robust'] = result_robust['beta']
                    result_robust['se_cluster_robust'] = result_robust['se']
                    result_robust['p_value_cluster_robust'] = result_robust['p_value']
                    result_robust['ci_low_cluster_robust'] = result_robust['ci_low']
                    result_robust['ci_high_cluster_robust'] = result_robust['ci_high']
                    result_robust['mixed_model_error'] = str(e)
                    results.append(result_robust)
            else:
                results.append(result_robust)

        except Exception as e:
            results.append({
                'feature': feature,
                'beta': np.nan,
                'se': np.nan,
                'z': np.nan,
                'p_value': 1.0,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'n_obs': len(x_clean),
                'n_clusters_student': 0,
                'n_clusters_question': 0,
                'n_clusters_rubric': 0,
                'significant': False,
                'beta_cluster_robust': np.nan if small_cluster_primary else None,
                'se_cluster_robust': np.nan if small_cluster_primary else None,
                'p_value_cluster_robust': np.nan if small_cluster_primary else None,
                'ci_low_cluster_robust': np.nan if small_cluster_primary else None,
                'ci_high_cluster_robust': np.nan if small_cluster_primary else None,
                'error': str(e),
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by |beta| descending
    df = df.sort_values(by='beta', key=lambda x: abs(x), ascending=False)

    # --------------------------------------------------------------------
    # Compatibility layer: downstream expects Spearman-style column names.
    # For rank-standardized regressions, beta is correlation-like, so we map:
    #   r := beta, n := n_obs
    # and add per-target BH-FDR columns (p_adjusted, significant).
    # --------------------------------------------------------------------
    if 'beta' in df.columns and 'r' not in df.columns:
        df['r'] = df['beta']
    if 'r' in df.columns and 'r_squared' not in df.columns:
        df['r_squared'] = df['r'] ** 2
    if 'n_obs' in df.columns and 'n' not in df.columns:
        df['n'] = df['n_obs']
    if 'power' not in df.columns:
        df['power'] = np.nan
    if 'required_n' not in df.columns:
        df['required_n'] = np.nan

    if 'p_value' in df.columns:
        p = df['p_value'].to_numpy(dtype=float)
        p_filled = np.where(np.isfinite(p), p, 1.0)
        p_adjusted, significant = benjamini_hochberg(p_filled, alpha)
        df['p_adjusted'] = p_adjusted
        df['significant_raw'] = np.isfinite(p) & (p < alpha)
        df['significant'] = significant
        df.loc[~np.isfinite(p), 'significant'] = False

    return df


def display_power_curve(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    r_values: list,
    alpha: float,
    n_power_sim: int,
    n_perm: int,
    rng: np.random.Generator,
    plot: bool = False,
    n_plot_points: int = 50,
    model_name: str = "model",
    method: str = "method",
    power_method: str = "spearman_perm",
    student_ids: Optional[np.ndarray] = None,
    question_ids: Optional[np.ndarray] = None,
    data_generator: str = "copula",
    rubric_ids: Optional[np.ndarray] = None,
    n_students_target: Optional[int] = None,
    mixed_model_include_student_vc_glmm: bool = False,
    mixed_model_use_ranks: bool = False,
    mixed_model_standardize_outcome: bool = False,
    mixed_model_transform_bounded: str = "none",
    output_dir: Optional[Path] = None,
) -> None:
    """
    Compute and display prospective power curve.

    Shows what effect sizes can be detected with current sample size.

    Args:
        features_df: DataFrame with features
        feature_cols: List of feature columns
        target_col: Target column name
        r_values: List of r values for table display
        alpha: Significance level
        n_power_sim: Number of power simulations
        n_perm: Number of permutations
        rng: Random number generator
        plot: Whether to generate and save plot
        n_plot_points: Number of points for smooth curve in plot
        model_name: Model name for plot filename
        method: Method name for plot filename
    """
    print("\n" + "="*100)
    print("PROSPECTIVE POWER ANALYSIS")
    print("="*100)

    # Use first feature as representative (assume similar n across features)
    x_sample = features_df[feature_cols[0]].values
    y_sample = features_df[target_col].values
    if power_method in ["cluster_robust", "mixed_model_lrt", "all"]:
        if student_ids is None:
            raise ValueError("student_ids are required for cluster-robust power curves")
        if question_ids is None:
            raise ValueError("question_ids are required for cluster-robust power curves")
        if power_method == "mixed_model_lrt" and rubric_ids is None:
            raise ValueError("rubric_ids are required for mixed_model_lrt power curves")

    # Remove NaNs
    mask = np.isfinite(x_sample) & np.isfinite(y_sample)
    x_clean = x_sample[mask]
    y_clean = y_sample[mask]
    student_clean = student_ids[mask] if student_ids is not None else None
    item_clean = question_ids[mask] if question_ids is not None else None
    rubric_clean = rubric_ids[mask] if rubric_ids is not None else None
    n_rows = len(x_clean)
    n_students = int(len(np.unique(student_clean))) if student_clean is not None else 0
    n = n_rows
    if power_method == "mixed_model_lrt":
        print(f"\nAnalytic rows: n_rows = {n_rows:,}")
        print(f"Students:      n_students = {n_students:,}")
        if n_students_target is not None:
            print(f"Target students for prospective power: {n_students_target:,}")
        print(f"Computing LRT power for {len(r_values)} |beta| values...")
        print(f"(Using {n_power_sim} simulations per |beta|)\n")

        power_curve_df = compute_power_curve_mixed_model_lrt(
            x_clean,
            y_clean,
            student_ids=student_clean,
            question_ids=item_clean,
            rubric_ids=rubric_clean,
            beta_targets=r_values,
            n_students_target=n_students_target,
            alpha=alpha,
            n_sim=n_power_sim,
            seed_students=int(rng.integers(1, 2**31 - 1)),
            seed_outcomes=int(rng.integers(1, 2**31 - 1)),
            include_student_vc_glmm=mixed_model_include_student_vc_glmm,
            use_ranks=mixed_model_use_ranks,
            standardize_feature=True,
            standardize_outcome=mixed_model_standardize_outcome,
            transform_bounded=mixed_model_transform_bounded,
            log_progress=True,
        )

        # For display compatibility, use 'beta_target' as the x-axis column (|beta|).
        # Avoid creating duplicate 'beta_target' columns (mixed_model_lrt already returns
        # both beta_target and beta_target_abs).
        power_curve_df = (
            power_curve_df.drop(columns=["beta_target"], errors="ignore")
            .rename(columns={"beta_target_abs": "beta_target"})
            .copy()
        )

        if "interpretation" not in power_curve_df.columns:
            def _interpret_power(p: float) -> str:
                if not np.isfinite(p):
                    return ""
                if p >= 0.8:
                    return "High"
                if p >= 0.5:
                    return "Moderate"
                if p >= 0.2:
                    return "Low"
                return "Very Low"

            power_curve_df["interpretation"] = power_curve_df["power"].apply(_interpret_power)
    else:
        n = n_rows
        print(f"\nSample size: n = {n:,}")
        print(f"Computing power for {len(r_values)} effect sizes...")
        print(f"(Using {n_power_sim} simulations per effect size)\n")

        # Compute power curve
        power_curve_df = compute_power_curve(
            x_clean,
            y_clean,
            r_values=r_values,
            n_target=n,
            alpha=alpha,
            n_power_sim=n_power_sim,
            n_perm=n_perm,
            random_state=rng,
            log_progress=True,
            power_method=power_method,
            student_ids=student_clean,
            question_ids=item_clean,
            data_generator=data_generator,
            rubric_ids=rubric_clean,
        )

    # Display results
    print("\n" + "-"*90)
    has_method = 'method' in power_curve_df.columns
    has_r_achieved = 'r_achieved' in power_curve_df.columns
    has_beta_used = 'beta_used' in power_curve_df.columns

    effect_col = 'r_target'
    if 'beta_target' in power_curve_df.columns:
        beta_block = power_curve_df['beta_target']
        # Handle potential duplicate columns defensively.
        try:
            has_any_beta = pd.notna(beta_block).to_numpy().any()
        except Exception:
            has_any_beta = False
        if has_any_beta:
            effect_col = 'beta_target'
    header_cols = [f"Effect Size ({effect_col})"]
    if has_method:
        header_cols.append("Method")
    if has_r_achieved:
        header_cols.append("r_achieved")
    if has_beta_used:
        header_cols.append("beta_used")
    header_cols += ["Power", "Interpretation", "Indicator"]

    print("  ".join(f"{h:<16}" for h in header_cols))
    print("-"*90)

    for _, row in power_curve_df.iterrows():
        effect_target = row.get(effect_col, np.nan)
        power = row.get('power', np.nan)
        interp = row.get('interpretation', '')
        method_name = row.get('method', '') if has_method else None
        r_achieved = row.get('r_achieved', np.nan) if has_r_achieved else None
        beta_used = row.get('beta_used', np.nan) if has_beta_used else None

        # Visual indicator
        if power >= 0.8:
            indicator = "✓ Adequate"
        elif power >= 0.5:
            indicator = "⚠ Moderate"
        else:
            indicator = "✗ Inadequate"

        # Avoid misleading duplicate-looking rows from rounding (e.g., 0.005 and 0.010
        # both printing as 0.01). Use higher precision for display.
        if effect_col == "beta_target":
            parts = [f"{effect_target:>8.4f}"]
        else:
            parts = [f"{effect_target:>7.3f}"]
        if has_method:
            parts.append(f"{str(method_name):<12}")
        if has_r_achieved:
            parts.append(f"{r_achieved:>8.3f}" if np.isfinite(r_achieved) else f"{'':>8}")
        if has_beta_used:
            parts.append(f"{beta_used:>8.3f}" if np.isfinite(beta_used) else f"{'':>8}")
        parts.append(f"{power:>6.3f}")
        parts.append(f"{str(interp):<14}")
        parts.append(indicator)
        print("  ".join(parts))

    print("-"*90)
    print("\nInterpretation:")
    print("  - Power ≥ 0.80: Adequate power (standard threshold)")
    print("  - Power 0.50-0.79: Moderate power (may miss some true effects)")
    print("  - Power < 0.50: Inadequate power (likely to miss effects)")

    # Generate and save plot if requested
    if plot:
        print(f"\nGenerating power curve plot with {n_plot_points} points...")

        # Generate many effect values for smooth curve
        if effect_col == 'beta_target':
            beta_min = float(np.nanmin(power_curve_df['beta_target']))
            beta_max = float(np.nanmax(power_curve_df['beta_target']))
            if not np.isfinite(beta_min) or not np.isfinite(beta_max) or beta_max <= beta_min:
                beta_min, beta_max = 0.01, 0.60
            r_values_smooth = np.linspace(beta_min, beta_max, n_plot_points).tolist()
        else:
            r_values_smooth = np.linspace(0.01, 0.60, n_plot_points).tolist()

        # Compute power for smooth curve
        if power_method == "mixed_model_lrt":
            power_curve_smooth = compute_power_curve_mixed_model_lrt(
                x_clean,
                y_clean,
                student_ids=student_clean,
                question_ids=item_clean,
                rubric_ids=rubric_clean,
                beta_targets=r_values_smooth,
                n_students_target=n_students_target,
                alpha=alpha,
                n_sim=n_power_sim,
                seed_students=int(rng.integers(1, 2**31 - 1)),
                seed_outcomes=int(rng.integers(1, 2**31 - 1)),
                include_student_vc_glmm=mixed_model_include_student_vc_glmm,
                use_ranks=mixed_model_use_ranks,
                standardize_feature=True,
                standardize_outcome=mixed_model_standardize_outcome,
                transform_bounded=mixed_model_transform_bounded,
                log_progress=True,
            )
            power_curve_smooth = (
                power_curve_smooth.drop(columns=["beta_target"], errors="ignore")
                .rename(columns={"beta_target_abs": "beta_target"})
                .copy()
            )
        else:
            power_curve_smooth = compute_power_curve(
                x_clean,
                y_clean,
                r_values=r_values_smooth,
                n_target=n,
                alpha=alpha,
                n_power_sim=n_power_sim,
                n_perm=n_perm,
                random_state=rng,
                log_progress=True,
                power_method=power_method,
                student_ids=student_clean,
                question_ids=item_clean,
                data_generator=data_generator,
                rubric_ids=rubric_clean,
            )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot smooth curve(s)
        effect_col_smooth = 'beta_target' if 'beta_target' in power_curve_smooth.columns and power_curve_smooth['beta_target'].notna().any() else 'r_target'
        if 'method' in power_curve_smooth.columns:
            for m, df_m in power_curve_smooth.groupby('method'):
                ax.plot(df_m[effect_col_smooth], df_m['power'], linewidth=2, label=f'Power curve ({m})')
        else:
            ax.plot(power_curve_smooth[effect_col_smooth], power_curve_smooth['power'],
                    'b-', linewidth=2, label='Power curve')

        # Add horizontal lines for power thresholds
        ax.axhline(y=0.80, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Adequate power (0.80)')
        ax.axhline(y=0.50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate power (0.50)')

        # Formatting
        ax.set_xlabel(f'Effect Size ({effect_col_smooth})', fontsize=12)
        ax.set_ylabel('Statistical Power', fontsize=12)
        ax.set_title(f'Prospective Power Analysis (n={n:,}, α={alpha})\n{model_name} - {method}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        # Use a generic x-range for either r_target or beta_target
        x_min = float(np.nanmin(power_curve_smooth[effect_col_smooth]))
        x_max = float(np.nanmax(power_curve_smooth[effect_col_smooth]))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
            pad = 0.05 * (x_max - x_min)
            ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(0, 1.05)

        # Save plot
        plot_dir = output_dir or (project_root / "experiments" / "analysis" / method)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"power_curve_{model_name}_{method}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Power curve plot saved to: {plot_path}")

    # Find minimum r with adequate power
    if has_method:
        for m, df_m in power_curve_df.groupby('method'):
            adequate_rows = df_m[df_m['power'] >= 0.8]
            if len(adequate_rows) > 0:
                min_eff = adequate_rows[effect_col].min()
                min_eff_fmt = f"{min_eff:.4f}" if effect_col == "beta_target" else f"{min_eff:.2f}"
                print(f"\nWith n={n:,} ({m}), you can reliably detect {effect_col} ≥ {min_eff_fmt}")
            else:
                print(f"\nWith n={n:,} ({m}), inadequate power for all tested effect sizes")
    else:
        adequate_rows = power_curve_df[power_curve_df['power'] >= 0.8]
        if len(adequate_rows) > 0:
            min_eff = adequate_rows[effect_col].min()
            min_eff_fmt = f"{min_eff:.4f}" if effect_col == "beta_target" else f"{min_eff:.2f}"
            print(f"\nWith n={n:,}, you can reliably detect {effect_col} ≥ {min_eff_fmt}")
        else:
            print(f"\nWith n={n:,}, inadequate power for all tested effect sizes")

    print("="*100)


def validate_power_estimation(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    n_power_sim: int,
    n_perm: int,
    rng: np.random.Generator,
) -> None:
    """
    Validate that vectorized power estimation matches fresh-permutation version.
    """
    print("\n" + "="*100)
    print("VALIDATING POWER ESTIMATION METHOD")
    print("="*100)
    print("\nComparing vectorized (permutation reuse) vs fresh-permutation methods...")
    print(f"Test parameters: n_power_sim={n_power_sim}, n_perm={n_perm}\n")

    # Test on first 3 features (representative sample)
    test_features = feature_cols
    
    results = []
    for feature in tqdm(test_features, desc="Validating power methods", unit="feature"):
        x = features_df[feature].values
        y = features_df[target_col].values

        # Remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        print(f"Testing feature: {feature} (n={len(x_clean):,})")

        # Compare methods
        comparison = compare_power_estimation_methods(
            x_clean,
            y_clean,
            n_target=len(x_clean),
            n_power_sim=n_power_sim,
            n_perm=n_perm,
            random_state=rng,
        )

        results.append({
            'feature': feature,
            **comparison
        })

        # Display individual result
        print(f"  Power (vectorized):    {comparison['power_vectorized']:.4f}")
        print(f"  Power (fresh perms):   {comparison['power_fresh_perms']:.4f}")
        print(f"  Absolute difference:   {comparison['absolute_diff']:.4f}")

        if np.isfinite(comparison['relative_diff_pct']):
            print(f"  Relative difference:   {comparison['relative_diff_pct']:.2f}%")
        else:
            print(f"  Relative difference:   N/A (power near zero)")

        if comparison['negligible']:
            print(f"  ✓ Difference is negligible (<2%)\n")
        else:
            print(f"  ⚠ Difference may be non-negligible (≥2%)\n")

    # Summary
    print("-"*100)
    print("SUMMARY")
    print("-"*100)

    avg_diff = np.mean([r['absolute_diff'] for r in results])
    max_diff = np.max([r['absolute_diff'] for r in results])
    all_negligible = all(r['negligible'] for r in results)

    print(f"Average absolute difference: {avg_diff:.4f}")
    print(f"Maximum absolute difference: {max_diff:.4f}")

    if all_negligible:
        print("\n✓ VALIDATION PASSED: Permutation reuse introduces negligible bias (<2%)")
        print("  → Vectorized method is suitable for production use")
    else:
        print("\n⚠ VALIDATION WARNING: Some differences ≥2%")
        print("  → Consider using fresh-permutation method for critical analyses")
        print("  → Or increase n_perm to reduce sensitivity to permutation reuse")

    print("="*100)


def display_bayesian_assurance(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    n_target: int,
    method: str,
    n_bootstrap: int,
    n_posterior_samples: int,
    n_power_sim_per_r: int,
    n_perm: int,
    alpha: float,
    rng: np.random.Generator,
    power_method: str = "spearman_perm",
    student_ids: Optional[np.ndarray] = None,
    question_ids: Optional[np.ndarray] = None,
    rubric_ids: Optional[np.ndarray] = None,
    assurance_mode: str = "fast",
    assurance_beta_grid: Optional[list] = None,
    n_power_sim_inner: int = 100,
    assurance_winsorize_pct: Optional[float] = None,
    mixed_model_include_student_vc_glmm: bool = False,
    mixed_model_use_ranks: bool = False,
    mixed_model_standardize_outcome: bool = False,
    mixed_model_transform_bounded: str = "none",
) -> None:
    """
    Compute and display Bayesian assurance with empirical priors.
    """
    print("\n" + "="*100)
    print("BAYESIAN ASSURANCE ANALYSIS (Empirical Priors)")
    print("="*100)

    # Use first feature as representative
    x_sample = features_df[feature_cols[0]].values
    y_sample = features_df[target_col].values

    # Remove NaNs
    mask = np.isfinite(x_sample) & np.isfinite(y_sample)
    x_clean = x_sample[mask]
    y_clean = y_sample[mask]
    student_clean = student_ids[mask] if student_ids is not None else None
    item_clean = question_ids[mask] if question_ids is not None else None
    rubric_clean = rubric_ids[mask] if rubric_ids is not None else None

    if power_method == "mixed_model_lrt":
        if student_clean is None or item_clean is None or rubric_clean is None:
            raise ValueError("student_ids, question_ids, and rubric_ids are required for mixed_model_lrt assurance")

        print("\n" + "-"*100)
        print("MIXED MODEL LRT ASSURANCE (Empirical prior over beta)")
        print("-"*100)

        if method != "both":
            print(f"Note: --assurance-method={method} is ignored for mixed_model_lrt (empirical prior is always bootstrap-based).")

        seed_students = int(rng.integers(1, 2**31 - 1))
        seed_outcomes = int(rng.integers(1, 2**31 - 1))
        seed_prior = int(rng.integers(1, 2**31 - 1))

        result_mm = compute_bayesian_assurance_mixed_model_empirical(
            x_clean,
            y_clean,
            student_ids=student_clean,
            question_ids=item_clean,
            rubric_ids=rubric_clean,
            n_students_target=n_target,
            alpha=alpha,
            assurance_mode=assurance_mode,
            beta_grid=assurance_beta_grid,
            n_power_sim=n_power_sim_per_r,
            seed_students=seed_students,
            seed_outcomes=seed_outcomes,
            n_prior=n_bootstrap,
            seed_prior=seed_prior,
            n_power_sim_inner=n_power_sim_inner,
            include_student_vc_glmm=mixed_model_include_student_vc_glmm,
            use_ranks=mixed_model_use_ranks,
            standardize_feature=True,
            standardize_outcome=mixed_model_standardize_outcome,
            transform_bounded=mixed_model_transform_bounded,
            drop_singular_prior=True,
            winsorize_pct=assurance_winsorize_pct,
        )

        print(f"\nRESULTS:")
        print(f"  Assurance (expected power): {result_mm['assurance']:.4f}")
        print(f"  95% Credible Interval:      [{result_mm['assurance_ci'][0]:.4f}, {result_mm['assurance_ci'][1]:.4f}]")
        print(f"  Standard Deviation:          {result_mm['assurance_sd']:.4f}")
        print(f"  Standard Error:              {result_mm['assurance_se']:.4f}")
        print(f"  Mode:                        {result_mm['mode']}")
        print(f"  Prior (beta) mean ± sd:      {result_mm['prior_summary']['mean']:.4f} ± {result_mm['prior_summary']['sd']:.4f}")
        print(f"  Prior draws:                 {result_mm['prior_n_attempted']} attempted, "
              f"{result_mm['prior_n_failed']} failed, {result_mm['prior_n_singular']} singular")

        print("\n" + "="*100)
        print("INTERPRETATION")
        print("="*100)
        print("Assurance = Expected LRT power under an empirical prior over beta,")
        print(f"              for a new study with n_students={n_target}.")
        print("fast mode: builds a power curve over |beta| and interpolates at prior draws.")
        print("faithful mode: nested simulation per prior draw (expensive).")
        print("="*100)
        return

    if power_method in ['spearman_perm', 'all'] and method in ['bootstrap', 'both']:
        print("\n" + "-"*100)
        print("METHOD 1: BOOTSTRAP EMPIRICAL PRIOR")
        print("-"*100)

        result_boot = compute_bayesian_assurance_bootstrap(
            x_clean, y_clean,
            n_target=n_target,
            n_bootstrap=n_bootstrap,
            n_power_sim_per_r=n_power_sim_per_r,
            n_perm=n_perm,
            alpha=alpha,
            random_state=rng,
            log_progress=True,
        )

        print(f"\nRESULTS:")
        print(f"  Assurance (expected power): {result_boot['assurance']:.4f}")
        print(f"  95% Credible Interval:      [{result_boot['assurance_ci'][0]:.4f}, {result_boot['assurance_ci'][1]:.4f}]")
        print(f"  Standard Deviation:          {result_boot['assurance_sd']:.4f}")

        # Interpretation
        if result_boot['assurance'] >= 0.8:
            interp = "✓ High (≥80% expected power)"
        elif result_boot['assurance'] >= 0.6:
            interp = "⚠ Moderate (60-80% expected power)"
        else:
            interp = "✗ Low (<60% expected power)"

        print(f"  Interpretation:              {interp}")

    if power_method in ['spearman_perm', 'all'] and method in ['posterior', 'both']:
        print("\n" + "-"*100)
        print("METHOD 2: BAYESIAN POSTERIOR (Fisher z)")
        print("-"*100)

        result_post = compute_bayesian_assurance_posterior(
            x_clean, y_clean,
            n_target=n_target,
            n_posterior_samples=n_posterior_samples,
            n_power_sim_per_r=n_power_sim_per_r,
            n_perm=n_perm,
            alpha=alpha,
            random_state=rng,
            log_progress=True,
        )

        print(f"\nRESULTS:")
        print(f"  Assurance (expected power): {result_post['assurance']:.4f}")
        print(f"  95% Credible Interval:      [{result_post['assurance_ci'][0]:.4f}, {result_post['assurance_ci'][1]:.4f}]")
        print(f"  Standard Deviation:          {result_post['assurance_sd']:.4f}")

        # Interpretation
        if result_post['assurance'] >= 0.8:
            interp = "✓ High (≥80% expected power)"
        elif result_post['assurance'] >= 0.6:
            interp = "⚠ Moderate (60-80% expected power)"
        else:
            interp = "✗ Low (<60% expected power)"

        print(f"  Interpretation:              {interp}")

    # Compare methods
    if power_method in ['spearman_perm', 'all'] and method == 'both':
        print("\n" + "-"*100)
        print("COMPARISON")
        print("-"*100)

        diff = abs(result_boot['assurance'] - result_post['assurance'])
        print(f"  Bootstrap assurance:  {result_boot['assurance']:.4f}")
        print(f"  Posterior assurance:  {result_post['assurance']:.4f}")
        print(f"  Absolute difference:  {diff:.4f}")

        if diff < 0.05:
            print(f"  ✓ Methods agree (difference < 5%)")
        else:
            print(f"  ⚠ Methods differ (difference ≥ 5%) - investigate!")

    if power_method in ['cluster_robust', 'all']:
        if student_clean is None or item_clean is None:
            raise ValueError("student_ids and question_ids are required for cluster-robust assurance")

        print("\n" + "-"*100)
        print("CLUSTER-ROBUST ASSURANCE")
        print("-"*100)

        result_cluster = compute_bayesian_assurance_cluster_robust(
            x_clean,
            y_clean,
            student_ids=student_clean,
            question_ids=item_clean,
            rubric_ids=rubric_clean,
            n_target=n_target,
            prior_spec={'n_power_sim_per_effect': n_power_sim_per_r},
            alpha=alpha,
            decision_rule="p_value",
            n_sim=n_bootstrap,
            random_state=rng,
            log_progress=True,
        )

        print(f"\nRESULTS:")
        print(f"  Assurance (expected power): {result_cluster['assurance']:.4f}")
        print(f"  95% Credible Interval:      [{result_cluster['assurance_ci'][0]:.4f}, {result_cluster['assurance_ci'][1]:.4f}]")
        print(f"  Standard Deviation:          {result_cluster['assurance_sd']:.4f}")

        if result_cluster['assurance'] >= 0.8:
            interp = "✓ High (≥80% expected power)"
        elif result_cluster['assurance'] >= 0.6:
            interp = "⚠ Moderate (60-80% expected power)"
        else:
            interp = "✗ Low (<60% expected power)"
        print(f"  Interpretation:              {interp}")

    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    print("Assurance = Expected power under empirical prior constructed from current data")
    print("This answers: 'If I believe my current estimate r, what's expected power for")
    print(f"              a new study with n={n_target}?'")
    print()
    print("Advantage over fixed-r power curves:")
    print("  - Accounts for uncertainty in the effect size estimate")
    print("  - Data-driven (no arbitrary prior specification)")
    print("  - Conservative (automatically penalizes for estimation uncertainty)")
    print("="*100)


def compute_assurance_for_all_features(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    n_target: int,
    method: str,
    n_bootstrap: int,
    n_posterior_samples: int,
    n_power_sim_per_r: int,
    n_perm: int,
    alpha: float,
    rng: np.random.Generator,
    top_k: Optional[int] = None,
    significant_only: bool = False,
    correlation_results: Optional[pd.DataFrame] = None,
    power_method: str = "spearman_perm",
    student_ids: Optional[np.ndarray] = None,
    question_ids: Optional[np.ndarray] = None,
    rubric_ids: Optional[np.ndarray] = None,
    assurance_mode: str = "fast",
    assurance_beta_grid: Optional[list] = None,
    n_power_sim_inner: int = 100,
    assurance_winsorize_pct: Optional[float] = None,
    mixed_model_include_student_vc_glmm: bool = False,
    mixed_model_use_ranks: bool = False,
    mixed_model_standardize_outcome: bool = False,
    mixed_model_transform_bounded: str = "none",
) -> pd.DataFrame:
    """
    Compute Bayesian assurance for all feature-target pairs.

    Args:
        features_df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Target column name
        n_target: Target sample size for assurance calculation
        method: 'bootstrap' | 'posterior' | 'both'
        n_bootstrap: Bootstrap samples for empirical prior
        n_posterior_samples: Posterior samples for Bayesian method
        n_power_sim_per_r: Power simulations per r value
        n_perm: Permutations per test
        alpha: Significance level
        rng: Random number generator
        top_k: Only compute for top k features by |r|
        significant_only: Only compute for significant features
        correlation_results: DataFrame with correlation results for filtering

    Returns:
        DataFrame with assurance statistics for each feature
    """
    # Filter features if requested
    filtered_feature_cols = feature_cols.copy()

    if significant_only and correlation_results is not None:
        # Only features with p_adjusted < alpha
        significant_features = correlation_results[correlation_results['significant']]['feature'].tolist()
        filtered_feature_cols = [f for f in filtered_feature_cols if f in significant_features]
        print(f"Computing assurance for {len(filtered_feature_cols)} significant features (p_adjusted < {alpha})")

    elif top_k is not None and correlation_results is not None:
        # Only top k features by absolute correlation
        top_features = correlation_results.head(top_k)['feature'].tolist()
        filtered_feature_cols = [f for f in filtered_feature_cols if f in top_features]
        print(f"Computing assurance for top {len(filtered_feature_cols)} features by |r|")

    else:
        print(f"Computing assurance for all {len(filtered_feature_cols)} features")

    # Compute assurance for each feature
    results = []

    corr_lookup = None
    if correlation_results is not None and 'feature' in correlation_results.columns:
        corr_lookup = correlation_results.set_index('feature', drop=False)

    for feature in tqdm(filtered_feature_cols, desc=f"Computing assurance for {target_col}", unit="feature"):
        try:
            x = features_df[feature].values
            y = features_df[target_col].values

            # Remove NaN pairs
            mask = np.isfinite(x) & np.isfinite(y)
            x_clean = x[mask]
            y_clean = y[mask]
            student_clean = student_ids[mask] if student_ids is not None else None
            question_clean = question_ids[mask] if question_ids is not None else None
            rubric_clean = rubric_ids[mask] if rubric_ids is not None else None

            if len(x_clean) < 3:
                # Insufficient data
                continue

            # Observed effect size (should match the effect metric used in correlation tables).
            # - Spearman: rho
            # - mixed_model_lrt: beta (rank-standardized slope used by Step-3 screening)
            effect_observed = np.nan
            effect_type = ""
            if corr_lookup is not None and feature in corr_lookup.index:
                row = corr_lookup.loc[feature]
                if power_method == "mixed_model_lrt":
                    beta_val = row.get('beta', np.nan)
                    if not np.isfinite(beta_val):
                        beta_val = row.get('r', np.nan)
                    effect_observed = float(beta_val) if np.isfinite(beta_val) else np.nan
                    effect_type = "beta"
                else:
                    r_val = row.get('r', np.nan)
                    if not np.isfinite(r_val):
                        r_val = row.get('beta', np.nan)
                    effect_observed = float(r_val) if np.isfinite(r_val) else np.nan
                    effect_type = "rho" if power_method == "spearman_perm" else "effect"
            else:
                r_fallback, _ = stats.spearmanr(x_clean, y_clean)
                effect_observed = float(r_fallback)
                effect_type = "rho"

            # Initialize result dict
            feature_result = {
                'feature': feature,
                'effect_observed': effect_observed,
                'effect_type': effect_type,
                'r_observed': float(effect_observed) if effect_type == "rho" and np.isfinite(effect_observed) else np.nan,
                'n_current': len(x_clean),
                'n_target': n_target,
            }

            if power_method == "mixed_model_lrt":
                if student_clean is None or question_clean is None or rubric_clean is None:
                    raise ValueError("student_ids, question_ids, and rubric_ids are required for mixed_model_lrt assurance")

                seed_students = int(rng.integers(1, 2**31 - 1))
                seed_outcomes = int(rng.integers(1, 2**31 - 1))
                seed_prior = int(rng.integers(1, 2**31 - 1))

                res = compute_bayesian_assurance_mixed_model_empirical(
                    x_clean,
                    y_clean,
                    student_ids=student_clean,
                    question_ids=question_clean,
                    rubric_ids=rubric_clean,
                    n_students_target=n_target,
                    alpha=alpha,
                    assurance_mode=assurance_mode,
                    beta_grid=assurance_beta_grid,
                    n_power_sim=n_power_sim_per_r,
                    seed_students=seed_students,
                    seed_outcomes=seed_outcomes,
                    n_prior=n_bootstrap,
                    seed_prior=seed_prior,
                    n_power_sim_inner=n_power_sim_inner,
                    include_student_vc_glmm=mixed_model_include_student_vc_glmm,
                    use_ranks=mixed_model_use_ranks,
                    standardize_feature=True,
                    standardize_outcome=mixed_model_standardize_outcome,
                    transform_bounded=mixed_model_transform_bounded,
                    drop_singular_prior=True,
                    winsorize_pct=assurance_winsorize_pct,
                )

                feature_result.update({
                    'assurance_mixed_model_lrt': float(res['assurance']),
                    'assurance_ci_low': float(res['assurance_ci'][0]),
                    'assurance_ci_high': float(res['assurance_ci'][1]),
                    'assurance_sd': float(res['assurance_sd']),
                    'assurance_se': float(res['assurance_se']),
                    'assurance_mode': str(res['mode']),
                    'prior_beta_mean': float(res['prior_summary']['mean']),
                    'prior_beta_sd': float(res['prior_summary']['sd']),
                    'prior_n_attempted': int(res['prior_n_attempted']),
                    'prior_n_failed': int(res['prior_n_failed']),
                    'prior_n_singular': int(res['prior_n_singular']),
                })

                results.append(feature_result)
                continue

            # Compute bootstrap assurance
            if method in ['bootstrap', 'both']:
                result_boot = compute_bayesian_assurance_bootstrap(
                    x_clean, y_clean,
                    n_target=n_target,
                    n_bootstrap=n_bootstrap,
                    n_power_sim_per_r=n_power_sim_per_r,
                    n_perm=n_perm,
                    alpha=alpha,
                    random_state=rng,
                    log_progress=False,  # Suppress inner progress bars
                )

                feature_result['assurance_bootstrap'] = float(result_boot['assurance'])
                feature_result['assurance_bootstrap_ci_low'] = float(result_boot['assurance_ci'][0])
                feature_result['assurance_bootstrap_ci_high'] = float(result_boot['assurance_ci'][1])
                feature_result['assurance_bootstrap_sd'] = float(result_boot['assurance_sd'])

            # Compute posterior assurance
            if method in ['posterior', 'both']:
                result_post = compute_bayesian_assurance_posterior(
                    x_clean, y_clean,
                    n_target=n_target,
                    n_posterior_samples=n_posterior_samples,
                    n_power_sim_per_r=n_power_sim_per_r,
                    n_perm=n_perm,
                    alpha=alpha,
                    random_state=rng,
                    log_progress=False,  # Suppress inner progress bars
                )

                feature_result['assurance_posterior'] = float(result_post['assurance'])
                feature_result['assurance_posterior_ci_low'] = float(result_post['assurance_ci'][0])
                feature_result['assurance_posterior_ci_high'] = float(result_post['assurance_ci'][1])
                feature_result['assurance_posterior_sd'] = float(result_post['assurance_sd'])

            # If both methods, compute agreement
            if method == 'both':
                diff = abs(result_boot['assurance'] - result_post['assurance'])
                feature_result['assurance_diff'] = float(diff)
                feature_result['methods_agree'] = bool(diff < 0.05)

            results.append(feature_result)

        except Exception as e:
            print(f"Warning: Error computing assurance for {feature}: {e}")
            continue

    # Convert to DataFrame
    assurance_df = pd.DataFrame(results)

    if len(assurance_df) == 0:
        print("Warning: No assurance results computed")
        return assurance_df

    # Sort by assurance (use bootstrap if both, otherwise use whichever method was run)
    if 'assurance_bootstrap' in assurance_df.columns:
        assurance_df = assurance_df.sort_values('assurance_bootstrap', ascending=False)
    elif 'assurance_posterior' in assurance_df.columns:
        assurance_df = assurance_df.sort_values('assurance_posterior', ascending=False)

    return assurance_df


def display_assurance_summary(
    assurance_df: pd.DataFrame,
    method: str,
    top_k: int = 10,
) -> None:
    """
    Display formatted summary table of Bayesian assurance results.

    Args:
        assurance_df: DataFrame with assurance results from compute_assurance_for_all_features()
        method: 'bootstrap' | 'posterior' | 'both'
        top_k: Number of top features to display (default: 10)
    """
    if len(assurance_df) == 0:
        print("No assurance results to display")
        return

    # Display table header
    print("\n" + "="*135)
    print("TOP FEATURES BY BAYESIAN ASSURANCE")
    print("="*135)

    # Mixed-model assurance table (empirical prior + LRT power)
    if 'assurance_mixed_model_lrt' in assurance_df.columns:
        assurance_col = 'assurance_mixed_model_lrt'

        def interpret_assurance(assurance_val):
            if np.isnan(assurance_val):
                return "N/A", "?"
            elif assurance_val >= 0.8:
                return "High", "✓"
            elif assurance_val >= 0.6:
                return "Moderate", "⚠"
            else:
                return "Low", "✗"

        df = assurance_df.copy()
        df = df.sort_values(assurance_col, ascending=False)
        df = df.head(top_k)
        df['interpretation'], df['marker'] = zip(*df[assurance_col].apply(interpret_assurance))

        header = (
            f"{'Feature':<35} {'beta_obs':>9} {'n_curr':>7} {'n_stu_t':>9} "
            f"{'Assur':>7} {'CI_low':>9} {'CI_hi':>8} "
            f"{'Prior_mu':>9} {'Prior_sd':>9} {'Mode':>8} {'Interp':<10} {'':>2}"
        )
        print(header)
        print("-"*135)

        for _, row in df.iterrows():
            print(
                f"{row['feature']:<35} "
                f"{row.get('effect_observed', row.get('r_observed', np.nan)):>9.3f} "
                f"{int(row.get('n_current', 0)):>7d} "
                f"{int(row.get('n_target', 0)):>9d} "
                f"{row.get('assurance_mixed_model_lrt', np.nan):>7.3f} "
                f"{row.get('assurance_ci_low', np.nan):>9.3f} "
                f"{row.get('assurance_ci_high', np.nan):>8.3f} "
                f"{row.get('prior_beta_mean', np.nan):>9.3f} "
                f"{row.get('prior_beta_sd', np.nan):>9.3f} "
                f"{str(row.get('assurance_mode', '')):>8} "
                f"{str(row.get('interpretation', '')):<10} "
                f"{row.get('marker', ''):>2}"
            )
        print("-"*135)
        return

    # Determine which assurance column to use for interpretation
    if method == 'both':
        assurance_col = 'assurance_bootstrap'  # Prefer bootstrap for 'both'
    elif method == 'bootstrap':
        assurance_col = 'assurance_bootstrap'
    else:
        assurance_col = 'assurance_posterior'

    # Add interpretation column
    def interpret_assurance(assurance_val):
        if np.isnan(assurance_val):
            return "N/A", "?"
        elif assurance_val >= 0.8:
            return "High", "✓"
        elif assurance_val >= 0.6:
            return "Moderate", "⚠"
        else:
            return "Low", "✗"

    interpretations = []
    markers = []
    for _, row in assurance_df.iterrows():
        interp, marker = interpret_assurance(row[assurance_col])
        interpretations.append(interp)
        markers.append(marker)

    assurance_df = assurance_df.copy()
    assurance_df['interpretation'] = interpretations
    assurance_df['marker'] = markers

    # Print table header
    if method == 'both':
        header = (
            f"{'Feature':<35} {'effect':>9} {'type':>5} {'n_curr':>7} {'n_targ':>7} "
            f"{'Assur_B':>8} {'CI_low_B':>9} {'CI_hi_B':>8} "
            f"{'Assur_P':>8} {'CI_low_P':>9} {'CI_hi_P':>8} "
            f"{'Agree':>6} {'Interp':<10} {'':>2}"
        )
    elif method == 'bootstrap':
        header = (
            f"{'Feature':<35} {'effect':>9} {'type':>5} {'n_curr':>7} {'n_targ':>7} "
            f"{'Assur':>8} {'CI_low':>8} {'CI_high':>8} {'SD':>7} "
            f"{'Interp':<10} {'':>2}"
        )
    else:  # posterior
        header = (
            f"{'Feature':<35} {'effect':>9} {'type':>5} {'n_curr':>7} {'n_targ':>7} "
            f"{'Assur':>8} {'CI_low':>8} {'CI_high':>8} {'SD':>7} "
            f"{'Interp':<10} {'':>2}"
        )

    print(header)
    print("-"*135)

    # Print top k rows
    display_df = assurance_df.head(top_k)

    for _, row in display_df.iterrows():
        effect_obs = row.get('effect_observed', row.get('r_observed', np.nan))
        effect_type = row.get('effect_type', '')
        if method == 'both':
            # Check method agreement
            diff = abs(row['assurance_bootstrap'] - row['assurance_posterior'])
            agree = "✓" if diff < 0.05 else "✗"

            line = (
                f"{row['feature']:<35} "
                f"{effect_obs:>9.4f} "
                f"{str(effect_type):>5} "
                f"{int(row['n_current']):>7,} "
                f"{int(row['n_target']):>7,} "
                f"{row['assurance_bootstrap']:>8.4f} "
                f"{row['assurance_bootstrap_ci_low']:>9.4f} "
                f"{row['assurance_bootstrap_ci_high']:>8.4f} "
                f"{row['assurance_posterior']:>8.4f} "
                f"{row['assurance_posterior_ci_low']:>9.4f} "
                f"{row['assurance_posterior_ci_high']:>8.4f} "
                f"{agree:>6} "
                f"{row['interpretation']:<10} "
                f"{row['marker']:>2}"
            )
        elif method == 'bootstrap':
            line = (
                f"{row['feature']:<35} "
                f"{effect_obs:>9.4f} "
                f"{str(effect_type):>5} "
                f"{int(row['n_current']):>7,} "
                f"{int(row['n_target']):>7,} "
                f"{row['assurance_bootstrap']:>8.4f} "
                f"{row['assurance_bootstrap_ci_low']:>8.4f} "
                f"{row['assurance_bootstrap_ci_high']:>8.4f} "
                f"{row['assurance_bootstrap_sd']:>7.4f} "
                f"{row['interpretation']:<10} "
                f"{row['marker']:>2}"
            )
        else:  # posterior
            line = (
                f"{row['feature']:<35} "
                f"{effect_obs:>9.4f} "
                f"{str(effect_type):>5} "
                f"{int(row['n_current']):>7,} "
                f"{int(row['n_target']):>7,} "
                f"{row['assurance_posterior']:>8.4f} "
                f"{row['assurance_posterior_ci_low']:>8.4f} "
                f"{row['assurance_posterior_ci_high']:>8.4f} "
                f"{row['assurance_posterior_sd']:>7.4f} "
                f"{row['interpretation']:<10} "
                f"{row['marker']:>2}"
            )

        print(line)

    # Show distribution summary
    print("-"*135)
    print("\nDISTRIBUTION SUMMARY:")

    high_count = sum(assurance_df['interpretation'] == 'High')
    moderate_count = sum(assurance_df['interpretation'] == 'Moderate')
    low_count = sum(assurance_df['interpretation'] == 'Low')
    total = len(assurance_df)

    print(f"  ✓ High assurance (≥0.8):       {high_count:>4} / {total} ({100*high_count/total:.1f}%)")
    print(f"  ⚠ Moderate assurance (0.6-0.8): {moderate_count:>4} / {total} ({100*moderate_count/total:.1f}%)")
    print(f"  ✗ Low assurance (<0.6):         {low_count:>4} / {total} ({100*low_count/total:.1f}%)")

    # Method agreement summary (if both)
    if method == 'both':
        print("\nMETHOD AGREEMENT:")
        agree_count = sum(assurance_df['methods_agree'])
        print(f"  Features where methods agree (<5% diff): {agree_count} / {total} ({100*agree_count/total:.1f}%)")

        if agree_count < total * 0.8:
            print("  ⚠ Warning: Methods disagree for >20% of features")
            print("    → Fisher z assumptions may not hold for all features")
            print("    → Trust bootstrap method for features with disagreement")

    # Footer notes
    print("\n" + "="*135)
    print("INTERPRETATION GUIDE:")
    print("  ✓ High (≥0.8):     Adequate expected power for future study at n_target")
    print("  ⚠ Moderate (0.6-0.8): Moderate expected power, consider larger n_target")
    print("  ✗ Low (<0.6):      Inadequate expected power, likely underpowered")
    print()
    print("NOTE: Assurance accounts for uncertainty in the effect size estimate (effect_obs).")
    print("      For mixed_model_lrt, effect_obs is beta (rank-standardized slope).")
    print("      For spearman_perm, effect_obs is rho (Spearman correlation).")
    print("      Assurance is expected power averaged over plausible true effect sizes.")
    if method == 'both':
        print("      Agree column: ✓ = methods differ by <5% (Fisher z assumptions hold)")
        print("                    ✗ = methods differ by ≥5% (trust bootstrap method)")
    print("="*135)


def main():
    args = parse_args()
    if args.calibration_regression_only:
        if args.no_calibration_regression:
            raise ValueError("--calibration-regression-only cannot be combined with --no-calibration-regression.")
        args.no_correlation_tables = True
        args.plot = False
        args.power_curve = False
        args.validate_power_method = False
        args.bayesian_assurance = False
        args.assurance_top_k = None
        args.assurance_for_significant_only = False
        args.assurance_all_features = False
        print("⚠ Calibration-regression-only mode: skipping correlations, mixed-model screening, power/assurance, and plots.")
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
        if not args.warnings_log:
            args.warnings_log = str(output_manager.step_dir("step3", "logs") / "warnings.log")
    if args.question_col is None:
        args.question_col = "question_id"
    args.model = model_nicknames.get(args.model.lower(), args.model.lower())
    if args.top_k == -1:
        args.top_k = 10000000  # effectively no limit
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    rng = np.random.default_rng(args.seed)
    if args.warnings_log:
        warnings_log_path = Path(args.warnings_log).expanduser()
        warnings_log_path.parent.mkdir(parents=True, exist_ok=True)
        warnings_log_fp = open(warnings_log_path, "a", encoding="utf-8")

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            text = warnings.formatwarning(message, category, filename, lineno, line)
            warnings_log_fp.write(text)
            warnings_log_fp.flush()

        warnings.showwarning = _showwarning
        atexit.register(warnings_log_fp.close)
        print(f"Logging Python warnings to {warnings_log_path}")

    # Parse power curve effect values.
    # - Spearman / cluster_robust: r values via --power-curve-r-values
    # - mixed_model_lrt: |beta| values via --power-curve-beta-values (defaults to r list if not provided)
    power_curve_r_values = None
    power_curve_beta_values = None
    if args.power_curve:
        power_curve_r_values = [float(r.strip()) for r in args.power_curve_r_values.split(',')]
        if args.power_curve_beta_values:
            power_curve_beta_values = [float(v.strip()) for v in args.power_curve_beta_values.split(',')]
        else:
            power_curve_beta_values = list(power_curve_r_values)
    adaptive_band = tuple(float(x.strip()) for x in args.adaptive_band.split(','))
    assurance_beta_grid = None
    if args.bayesian_assurance and args.assurance_beta_grid:
        assurance_beta_grid = [float(v.strip()) for v in args.assurance_beta_grid.split(',')]

    # Auto-reduce defaults if computing assurance for all features
    # This prevents 2-4 hour runtimes with original defaults
    if args.bayesian_assurance and (args.assurance_all_features or args.assurance_top_k is not None or args.assurance_for_significant_only):
        print("\n" + "="*80)
        print("AUTO-REDUCING DEFAULTS FOR ALL-FEATURES ASSURANCE MODE")
        print("="*80)

        # Check if user hasn't explicitly overridden defaults
        # (compare with default values from argparse)
        reduced = []

        if args.n_bootstrap_assurance == 500:
            args.n_bootstrap_assurance = 100
            reduced.append(f"n_bootstrap_assurance: 500 → {args.n_bootstrap_assurance}")

        if args.n_posterior_samples == 1000:
            args.n_posterior_samples = 200
            reduced.append(f"n_posterior_samples: 1000 → {args.n_posterior_samples}")

        if args.n_power_sim_per_r == 200:
            args.n_power_sim_per_r = 50
            reduced.append(f"n_power_sim_per_r: 200 → {args.n_power_sim_per_r}")

        if reduced:
            print("\n⚠ Auto-reducing computational defaults to avoid 2-4 hour runtime:")
            for msg in reduced:
                print(f"  • {msg}")
            print("\n💡 Tip: You can override with explicit values for full precision:")
            print("   --n-bootstrap-assurance 500 --n-posterior-samples 1000 --n-power-sim-per-r 200")
            print("   (Warning: This will take 2-4 hours for all features)")
            print("\nEstimated runtime with reduced defaults: 30-60 minutes")
        else:
            print("\n✓ Using user-specified defaults (no auto-reduction)")

        print("="*80)

    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)

    legacy_dir = find_csv_folder_from_root("grading_calibration") / Path(args.method)
    dataset = args.dataset.lower()
    use_organized = bool(output_manager) and not args.predictions_dir
    if args.predictions_dir:
        directory = Path(args.predictions_dir) / Path(args.method)
    else:
        directory = legacy_dir
    if use_organized:
        directory = output_manager.step_dir("step1", "tables")
        csv_prefix = f"{args.prefix}_{model_name}_{args.method}"
        exclude_tokens = None
    else:
        if dataset == "ricechem":
            csv_prefix = f"{args.prefix}_{model_name}_{args.method}"
        else:
            csv_prefix = f"{args.prefix}_{model_name}_{args.method}_{dataset}"
        exclude_tokens = ["_beetle_", "_scientsbank_"] if dataset == "ricechem" else None
    csv_path = load_path_by_index(args.index, directory, csv_prefix, exclude_substrings=exclude_tokens)
    csv_timestamp = args.run_id if output_manager else csv_path.stem.split("_")[-1]

    legacy_analysis_dir = project_root / "experiments" / "analysis" / args.method
    step3_tables_dir = output_manager.step_dir("step3", "tables") if output_manager else legacy_analysis_dir
    step3_plots_dir = output_manager.step_dir("step3", "plots") if output_manager else None
    step3_intermediate_dir = (
        output_manager.step_dir("step3", "intermediate") if output_manager else legacy_analysis_dir
    )
    file_suffix = "" if output_manager else f"_{csv_timestamp}"
    method_tag = f"_{args.method}"

    # ========================================================================
    # Extract Features
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING FEATURES")
    print("=" * 80)

    extractor = FeatureExtractor(
        enable_perplexity=args.enable_perplexity,
        perplexity_model=args.perplexity_model,
        perplexity_device=args.perplexity_device,
        perplexity_batch_size=args.perplexity_batch_size,
        perplexity_max_length=args.perplexity_max_length,
        perplexity_min_tokens=args.perplexity_min_tokens,
        perplexity_debug=args.perplexity_debug,
        perplexity_stream_responses=bool(args.features_stream_chunk_size),
    )

    
    features_df: pd.DataFrame | None = None
    cache_path = None
    cache_meta = None
    stream_chunk_size = args.features_stream_chunk_size

    print("Streaming feature extraction:" if stream_chunk_size else "Non-streaming feature extraction:")
    if stream_chunk_size:
        if stream_chunk_size <= 0:
            raise ValueError("--features-stream-chunk-size must be > 0")
        header = pd.read_csv(csv_path, nrows=0)
        columns = header.columns.tolist()
        response_cols = _get_response_columns(columns)
        include_llm_response = "llm_response" in columns
        base_cols = [
            "instance_id",
            "question",
            "student_answer",
            "rubric_item",
            "answer_key",
            "ground_truth_label",
            "question_id",
            "rubric_id",
            "student_id",
            "prediction",
            "raw_confidence",
            "calibrated_confidence",
            "split",
            "vote_counts",
        ]
        usecols_min = [c for c in base_cols if c in columns]
        usecols_full = list(dict.fromkeys(
            usecols_min + (["llm_response"] if include_llm_response else []) + response_cols
        ))

        required_cols = []
        if args.enable_perplexity:
            required_cols.extend(["answer_perplexity", "response_perplexity"])
        print(f"Was features cache directory given: {args.features_cache_dir is not None}")
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
            if cache_meta is not None:
                cache_meta = dict(cache_meta)
                cache_meta["streaming"] = True
                cache_meta["stream_chunk_size"] = int(stream_chunk_size)

            if cache_path and args.features_cache_mode != "refresh" and cache_path.exists():
                features_df = pd.read_csv(cache_path)
                if "instance_id" in features_df.columns:
                    features_df.set_index("instance_id", inplace=True, drop=False)
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
                            step="step3",
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
            if args.features_cache_mode == "read_only":
                raise RuntimeError(
                    "Features cache missing and --features-cache-mode read_only is set."
                )
            print("Extracting features from stream...")
            instances, predictions, features_df = _stream_extract_features(
                Path(csv_path),
                extractor=extractor,
                chunk_size=stream_chunk_size,
                smoke_n=args.smoke_n,
                usecols=usecols_full,
                response_cols=response_cols,
                include_llm_response=include_llm_response,
                cache_path=cache_path,
                cache_meta=cache_meta,
                cache_mode=args.features_cache_mode,
                output_manager=output_manager,
                model_name=model_name,
                method=args.method,
                cache_version=args.features_cache_version,
            )
        else:
            print("Loading instances and predictions from stream (minimal)...")
            instances, predictions = _stream_instances_predictions_minimal(
                Path(csv_path),
                chunk_size=stream_chunk_size,
                smoke_n=args.smoke_n,
                usecols=usecols_min,
                response_cols=response_cols,
                include_llm_response=include_llm_response,
            )
            print(f"Reindexing features to match predictions order...")
            ordered_ids = [p.instance_id for p in predictions]
            if "instance_id" in features_df.columns:
                features_df.set_index("instance_id", inplace=True, drop=False)
            features_df = features_df.reindex(ordered_ids)
            present_ids = set(features_df.index.dropna())
            missing_ids = [iid for iid in ordered_ids if iid not in present_ids]
            print(f"Checking for missing features for {len(missing_ids)} instances...")
            if missing_ids:
                if args.features_cache_mode == "read_only":
                    raise RuntimeError(
                        f"Features cache missing {len(missing_ids)} instances; "
                        "use --features-cache-mode read_write or refresh."
                    )
                missing_df = _stream_extract_features_for_ids(
                    Path(csv_path),
                    extractor=extractor,
                    chunk_size=stream_chunk_size,
                    smoke_n=args.smoke_n,
                    usecols=usecols_full,
                    response_cols=response_cols,
                    include_llm_response=include_llm_response,
                    target_ids=set(missing_ids),
                )
                if not missing_df.empty:
                    if "instance_id" in missing_df.columns:
                        missing_df.set_index("instance_id", inplace=True, drop=False)
                    features_df = features_df.combine_first(missing_df).reindex(ordered_ids)
                    if cache_path and args.features_cache_mode in ("read_write", "refresh"):
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_df = features_df.copy()
                        if cache_df.index.name == "instance_id" and "instance_id" not in cache_df.columns:
                            cache_df = cache_df.reset_index()
                        cache_df.to_csv(cache_path, index=False)
                        print(f"Saved features cache: {cache_path}")
                        if cache_meta:
                            write_features_cache_metadata(cache_path, cache_meta)
            print(f"Extracted features for {len(features_df)} instances.")
    else:
        instances, predictions, df = load_data_from_csv(csv_path, smoke_n=args.smoke_n)
        print(f"Loaded data from CSV: {csv_path}")

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
                            step="step3",
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

    print(f"\nLoaded {len(instances)} instances")
    print(f"Timestamp: {csv_timestamp}")

    if features_df is None:
        features_df = _pipeline_extract_features_dataframe(instances, predictions, extractor=extractor)
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
                    step="step3",
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
                    perplexity_stream_responses=bool(args.features_stream_chunk_size),
                )
            inst_map = {inst.id: inst for inst in instances}
            pred_map = {pred.instance_id: pred for pred in predictions}
            missing_instances = [inst_map[iid] for iid in missing_ids if iid in inst_map]
            missing_predictions = [pred_map[iid] for iid in missing_ids if iid in pred_map]
            new_df = _pipeline_extract_features_dataframe(
                missing_instances, missing_predictions, extractor=extractor
            )
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

    print(f"\nExtracted {len(features_df.columns)} features for {len(features_df)} instances")
    print(f"\nFeature categories:")
    
    # Group features by prefix
    prefixes = ['answer_', 'question_', 'rubric_', 'response_', 'confidence', 'prediction', 'vote_']
    for prefix in prefixes:
        matching = [c for c in features_df.columns if c.startswith(prefix)]
        if matching:
            print(f"  {prefix}: {len(matching)} features")

    # Add calibration error as a target column (bin-based, not per-instance)
    # This measures calibration quality at the group level, eliminating mechanical
    # correlation with confidence that would contaminate feature analysis
    features_df['calibration_error'] = assign_bin_calibration_errors(
        predictions, n_bins=10, use_calibrated=False
    )

    # ========================================================================
    # Dataset integrity checks (nested rubrics; unique triples)
    # ========================================================================
    required_id_cols = [args.student_col, args.question_col, args.rubric_col]
    missing_id_cols = [c for c in required_id_cols if c not in features_df.columns]
    if missing_id_cols:
        raise KeyError(f"Missing required ID columns in extracted features: {missing_id_cols}")

    rubric_map = features_df[[args.rubric_col, args.question_col]].dropna()
    if not rubric_map.empty:
        rubric_nested_ok = (rubric_map.groupby(args.rubric_col)[args.question_col].nunique() == 1).all()
        assert rubric_nested_ok, (
            "Rubric nesting violated: some rubric_id maps to multiple question_id values. "
            "Fix IDs upstream before treating rubric_id as nested within question_id."
        )

    triple_df = features_df[[args.student_col, args.question_col, args.rubric_col]].dropna()
    if not triple_df.empty:
        assert not triple_df.duplicated(required_id_cols).any(), (
            "Duplicate (student_id, question_id, rubric_id) rows detected. "
            "If duplicates are identical they may be dropped; otherwise aggregate explicitly."
        )


    # Save features if requested
    if args.save_features:
        features_path = step3_tables_dir / f"features_{model_name}{method_tag}{file_suffix}.csv"
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_path)
        print(f"\nFeatures saved to: {features_path}")
        if output_manager:
            output_manager.register(
                step="step3",
                artifact="features",
                path=features_path,
                model=model_name,
                method=args.method,
            )
            if output_manager.config.save_legacy:
                legacy_path = legacy_analysis_dir / f"features_{model_name}_{csv_timestamp}.csv"
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                features_df.to_csv(legacy_path, index=False)

    # ========================================================================
    # DIAGNOSTIC: Check vote_agreement vs confidence relationship
    # ========================================================================
    if args.diagnostics and args.method == 'self_consistency' and 'vote_agreement' in features_df.columns:
        print("\n" + "=" * 80)
        print("DIAGNOSTIC: vote_agreement vs raw_confidence")
        print("=" * 80)

        # Check if they're identical
        diff = np.abs(features_df['raw_confidence'] - features_df['vote_agreement'])
        print(f"\nIdentity Check:")
        print(f"  Max difference: {diff.max():.15f}")
        print(f"  Mean difference: {diff.mean():.15f}")
        print(f"  Median difference: {diff.median():.15f}")
        print(f"  Are they identical (np.allclose)? {np.allclose(features_df['raw_confidence'], features_df['vote_agreement'])}")

        # Show first few examples
        print(f"\nFirst 10 examples:")
        print(f"  {'raw_confidence':<20} {'vote_agreement':<20} {'difference':<20}")
        for i in range(min(10, len(features_df))):
            conf = features_df['raw_confidence'].iloc[i]
            vote = features_df['vote_agreement'].iloc[i]
            diff_val = conf - vote
            print(f"  {conf:<20.15f} {vote:<20.15f} {diff_val:<20.15e}")

        # Check Pearson vs Spearman correlation
        from scipy.stats import pearsonr, spearmanr
        r_pearson, p_pearson = pearsonr(features_df['raw_confidence'], features_df['vote_agreement'])
        r_spearman, p_spearman = spearmanr(features_df['raw_confidence'], features_df['vote_agreement'])

        print(f"\nCorrelation Analysis:")
        print(f"  Pearson correlation:  r = {r_pearson:.15f}, p = {p_pearson:.2e}")
        print(f"  Spearman correlation: r = {r_spearman:.15f}, p = {p_spearman:.2e}")
        print(f"\nInterpretation:")
        if r_pearson > 0.9999:
            print(f"  ✓ Pearson ≈ 1.0: vote_agreement and confidence are identical (or nearly so)")
        else:
            print(f"  ⚠ Pearson = {r_pearson:.4f}: They are NOT identical - investigating outliers...")

        # Find cases where they differ
        diff_threshold = 0.01  # 1% difference
        different_indices = diff[diff > diff_threshold].index

        if len(different_indices) > 0:
            print(f"\nOutlier Cases (difference > {diff_threshold}):")
            print(f"  Found {len(different_indices)} cases where confidence ≠ vote_agreement")
            print(f"\n  First 5 outliers:")
            print(f"  {'instance_id':<50} {'raw_conf':<10} {'vote_agr':<10} {'diff':<10} {'yes_p':<10} {'no_p':<10} {'vote_counts':<25}")

            for i, idx in enumerate(different_indices[:5]):
                # Get data from features_df
                conf = features_df.loc[idx, 'raw_confidence']
                vote = features_df.loc[idx, 'vote_agreement']
                diff_val = conf - vote
                yes_p = features_df.loc[idx, 'yes_prob']
                no_p = features_df.loc[idx, 'no_prob']

                # Get vote_counts from predictions metadata
                pred = next((p for p in predictions if p.instance_id == idx), None)
                if pred and 'vote_counts' in pred.metadata:
                    vote_counts = pred.metadata['vote_counts']
                    vote_str = str(vote_counts)
                else:
                    vote_str = "N/A"

                inst_id_str = str(idx)[:47]
                print(f"  {inst_id_str:<50} {conf:<10.6f} {vote:<10.6f} {diff_val:<10.6f} {yes_p:<10.6f} {no_p:<10.6f} {vote_str:<25}")

        print("=" * 80)

        # ========================================================================
        # DIAGNOSTIC: Calibration Error Variance
        # ========================================================================
        if args.diagnostics and 'calibration_error' in features_df.columns:
                print("\n" + "=" * 80)
                print("DIAGNOSTIC: Calibration Error Variance Analysis")
                print("=" * 80)

                cal_error = features_df['calibration_error']

                print(f"\nCalibration Error Statistics:")
                print(f"  Count:    {cal_error.count():>10,}")
                print(f"  Mean:     {cal_error.mean():>10.6f}")
                print(f"  Std Dev:  {cal_error.std():>10.6f}")
                print(f"  Variance: {cal_error.var():>10.6f}")
                print(f"  Min:      {cal_error.min():>10.6f}")
                print(f"  Max:      {cal_error.max():>10.6f}")
                print(f"  Median:   {cal_error.median():>10.6f}")

                # Count unique values
                unique_vals = cal_error.nunique()
                print(f"\n  Unique calibration error values: {unique_vals}")

                if unique_vals <= 20:
                    print(f"\n  Value counts (all unique values):")
                    value_counts = cal_error.value_counts().sort_index()
                    for val, count in value_counts.items():
                        pct = count / len(cal_error) * 100
                        print(f"    {val:>10.6f}: {count:>6,} instances ({pct:>5.1f}%)")
                else:
                    print(f"\n  Top 10 most common values:")
                    value_counts = cal_error.value_counts().head(10)
                    for val, count in value_counts.items():
                        pct = count / len(cal_error) * 100
                        print(f"    {val:>10.6f}: {count:>6,} instances ({pct:>5.1f}%)")

                # Interpretation
                print(f"\nInterpretation:")
                if cal_error.var() < 0.0001:
                    print(f"  ⚠ VERY LOW VARIANCE: Calibration error is nearly constant!")
                    print(f"    This explains perfect correlation (r=1.0) with other features.")
                    print(f"    When target has no variance, correlations become undefined/1.0.")
                elif cal_error.var() < 0.001:
                    print(f"  ⚠ LOW VARIANCE: Calibration error has limited variation.")
                    print(f"    This may cause inflated correlations.")
                elif unique_vals <= 10:
                    print(f"  ⚠ FEW UNIQUE VALUES: Only {unique_vals} distinct calibration errors.")
                    print(f"    This is expected for bin-based calibration (one value per bin).")
                    print(f"    Number of bins used: {unique_vals}")
                else:
                    print(f"  ✓ NORMAL VARIANCE: Calibration error varies across instances.")

                # Compare with vote features if available
                if args.method == 'self_consistency' and 'vote_agreement' in features_df.columns:
                    print(f"\nComparison with vote features:")
                    vote_agreement_unique = features_df['vote_agreement'].nunique()
                    print(f"  vote_agreement unique values: {vote_agreement_unique}")
                    print(f"  calibration_error unique values: {unique_vals}")

                    if unique_vals == vote_agreement_unique:
                        print(f"  ⚠ Same number of unique values - possible 1-to-1 mapping!")
                        print(f"    This could cause perfect correlation if each vote_agreement")
                        print(f"    value maps to exactly one calibration_error value.")

                        # Check the actual mapping
                        print(f"\n  Joint Distribution (vote_agreement vs calibration_error):")
                        crosstab = pd.crosstab(features_df['vote_agreement'],
                                            features_df['calibration_error'],
                                            margins=True)
                        print(crosstab)


                        # Check if it's a perfect 1-to-1 mapping
                        # For each vote_agreement value, count how many cal_error values it maps to
                        mapping_counts = features_df.groupby('vote_agreement')['calibration_error'].nunique()
                        print(f"\n  Mapping analysis (for each vote_agreement, how many cal_errors?):")
                        for vote_val, num_cal_errors in mapping_counts.items():
                            print(f"    vote_agreement={vote_val:.6f} → {num_cal_errors} distinct cal_error value(s)")

                        if (mapping_counts == 1).all():
                            print(f"\n  ✗ PERFECT 1-TO-1 MAPPING: Each vote_agreement maps to exactly ONE cal_error")
                            print(f"    This WILL cause perfect correlation (r=1.0 or r=-1.0)")
                        else:
                            print(f"\n  ✓ OVERLAPPING MAPPING: Some vote_agreement values map to multiple cal_errors")
                            print(f"    This prevents perfect correlation (allows normal r values)")

                print("=" * 80)


    # ========================================================================
    # Hierarchical Structure Diagnostics
    # ========================================================================
    print("\n" + "=" * 80)
    print("HIERARCHICAL STRUCTURE DIAGNOSTICS")
    print("=" * 80)
    print("\nChecking for clustering in data (violations of independence assumption)...")

    # Helper function to compute ICC
    def compute_icc_simple(values: np.ndarray, clusters: np.ndarray) -> float:
        """
        Compute intraclass correlation coefficient (ICC) using one-way ANOVA.

        ICC measures the proportion of variance due to clustering.
        ICC > 0.05 indicates substantial clustering that may violate independence.

        Args:
            values: Observed values (e.g., confidence scores)
            clusters: Cluster IDs (e.g., student_id, question_id)

        Returns:
            ICC value (0 to 1)
        """
        clusters = np.array(clusters)
        values = np.array(values)

        # Between-cluster variance
        unique_clusters = np.unique(clusters)
        cluster_means = {}
        for c in unique_clusters:
            cluster_means[c] = values[clusters == c].mean()

        grand_mean = values.mean()

        n_clusters = len(cluster_means)
        n_total = len(values)
        avg_cluster_size = n_total / n_clusters

        # Sum of squares
        SS_between = 0.0
        for c in cluster_means:
            n_c = len(values[clusters == c])
            SS_between += n_c * (cluster_means[c] - grand_mean)**2
        MS_between = SS_between / (n_clusters - 1)

        SS_within = 0.0
        for c in cluster_means:
            cluster_vals = values[clusters == c]
            SS_within += np.sum((cluster_vals - cluster_means[c])**2)
        MS_within = SS_within / (n_total - n_clusters)

        # ICC formula
        icc = (MS_between - MS_within) / (MS_between + (avg_cluster_size - 1) * MS_within)
        return max(0, icc)  # ICC can't be negative

    # Extract cluster IDs from metadata
    student_ids = np.array([inst.metadata.get('student_id', '') for inst in instances])
    question_ids = np.array([inst.metadata.get('question_id', '') for inst in instances])
    rubric_ids_diag = np.array([inst.metadata.get('rubric_id', '') for inst in instances])

    # Compute ICC for confidence
    confidences = features_df['raw_confidence'].values

    icc_student = compute_icc_simple(confidences, student_ids)
    icc_question = compute_icc_simple(confidences, question_ids)
    icc_rubric = compute_icc_simple(confidences, rubric_ids_diag) if len(np.unique(rubric_ids_diag)) > 1 else np.nan

    print(f"\nIntraclass Correlation (ICC) for raw_confidence:")
    if args.icc_method in ['anova', 'both']:
        print(f"\n  ANOVA-Style ICC (Current Method):")
        print(f"    By student_id:  {icc_student:.4f}")
        print(f"    By question_id: {icc_question:.4f}")
        if not np.isnan(icc_rubric):
            print(f"    By rubric_id:   {icc_rubric:.4f}")

    # NEW: Add Step-3-aligned mixed-model ICC if requested
    if args.icc_method in ['mixed_model', 'both']:
        print(f"\n  Mixed-Model ICC (Step-3-aligned; question FE):")
        try:

            # Extract rubric_ids if available
            rubric_ids = np.array([inst.metadata.get('rubric_id', '') for inst in instances])
            has_rubric = len(np.unique(rubric_ids)) > 1

            mm_results = compute_icc_mixed_model(
                values=confidences,
                student_ids=student_ids,
                question_ids=question_ids,
                rubric_ids=rubric_ids if has_rubric else None,
                verbose=True  # Print variance components
            )

            # Store for later use (Step-3-aligned: student + answer + rubric; question is FE)
            icc_student_mm = mm_results.get('icc_student', np.nan)
            icc_answer_mm = mm_results.get('icc_answer', np.nan)
            icc_rubric_mm = mm_results.get('icc_rubric', np.nan)

            print(f"\n  Comparison:")
            print(f"    ANOVA vs Mixed-Model ICC (student):  {icc_student:.4f} vs {icc_student_mm:.4f}")
            print(f"    Mixed-Model ICC (answer_id):         {icc_answer_mm:.4f}")
            if not np.isnan(icc_rubric):
                print(f"    ANOVA vs Mixed-Model ICC (rubric):   {icc_rubric:.4f} vs {icc_rubric_mm:.4f}")
            else:
                print(f"    Mixed-Model ICC (rubric):            {icc_rubric_mm:.4f}")
            print("    Note: Mixed-model ICC does not include a question random effect (question is fixed effects).")

            # Decide which ICCs to use for warnings
            icc_student_final = icc_student_mm if np.isfinite(icc_student_mm) else icc_student
            icc_question_final = icc_question
            icc_answer_final = icc_answer_mm
            icc_rubric_final = icc_rubric_mm if np.isfinite(icc_rubric_mm) else icc_rubric

        except ImportError as e:
            print(f"\n  ⚠ Mixed-model ICC not available: {e}")
            print(f"    Install statsmodels: pip install statsmodels")
            icc_student_final = icc_student
            icc_question_final = icc_question
            icc_answer_final = np.nan
            icc_rubric_final = icc_rubric
        except Exception as e:
            print(f"\n  ⚠ Mixed-model ICC failed: {e}")
            print(f"    Falling back to ANOVA-style ICC")
            icc_student_final = icc_student
            icc_question_final = icc_question
            icc_answer_final = np.nan
            icc_rubric_final = icc_rubric
    else:
        icc_student_final = icc_student
        icc_question_final = icc_question
        icc_answer_final = np.nan
        icc_rubric_final = icc_rubric

    print()  # Blank line before interpretation

    # Interpret ICC values (use final values) and align messaging with the chosen effects method.
    run_mixed_model_suite = (
        args.effects_method in {"mixed_model", "all"}
        and not args.effects_nonlinear_only
    ) or args.calibration_regression_only
    run_cluster_robust = run_mixed_model_suite and not args.no_cluster_robust and not args.no_correlation_tables
    icc_answer_flag = np.isfinite(icc_answer_final) and icc_answer_final > 0.05
    icc_rubric_flag = np.isfinite(icc_rubric_final) and icc_rubric_final > 0.05
    clustering_detected = (
        icc_student_final > 0.05
        or icc_question_final > 0.05
        or icc_answer_flag
        or icc_rubric_flag
    )

    if clustering_detected and not run_mixed_model_suite:
        print("⚠ WARNING: Substantial clustering detected (ICC > 0.05)")
        print("  effects-method excludes mixed_model, so inference will ignore clustering.")
        print("  Standard permutation tests may have inflated Type I error.")
        print("  Implications:")
        print("    - P-values may be too small (anti-conservative)")
        print("    - False positive rate may exceed nominal α")
        print("    - Results should be interpreted cautiously")
        print("\n  Consider re-running with:")
        print("    --effects-method mixed_model   (Step-3 mixed model + robustness check)")
        print("    --effects-method all           (report Spearman + mixed model; Spearman exploratory)")
    elif clustering_detected and run_cluster_robust:
        print("✓ Clustering detected (ICC > 0.05)")
        print("  effects-method includes mixed_model, so Step-3 mixed model will be used.")
        print("  Cluster-robust results will be reported as a robustness check.")
        print("  Spearman permutation results (if requested) are exploratory (ignore clustering).")
    elif clustering_detected and run_mixed_model_suite:
        print("✓ Clustering detected (ICC > 0.05)")
        print("  effects-method includes mixed_model, so Step-3 mixed model will be used.")
        print("  Cluster-robust correlations are disabled (--no-cluster-robust).")
        print("  Spearman permutation results (if requested) are exploratory (ignore clustering).")
    else:
        print("✓ Clustering is minimal (ICC < 0.05)")
        print("  Independence assumption is reasonably satisfied for screening.")

    # Additional cluster diagnostics
    n_unique_students = len(np.unique(student_ids))
    n_unique_questions = len(np.unique(question_ids))
    avg_instances_per_student = len(instances) / n_unique_students if n_unique_students > 0 else 0
    avg_instances_per_question = len(instances) / n_unique_questions if n_unique_questions > 0 else 0

    print(f"\nCluster Statistics:")
    print(f"  Total instances: {len(instances):,}")
    print(f"  Unique students: {n_unique_students:,} (avg {avg_instances_per_student:.1f} instances/student)")
    print(f"  Unique questions: {n_unique_questions:,} (avg {avg_instances_per_question:.1f} instances/question)")

    print("=" * 80)


    # ========================================================================
    # Correlation Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    if args.no_correlation_tables:
        print("⚠ Skipping correlation tables (--no-correlation-tables).")
    if args.effects_method == "spearman_permutation":
        print(
            f"Using Spearman (permutation) with permutation p-values (n_perm={args.n_perm}), "
            f"bootstrap CIs (n_boot={args.n_boot}), simulation power (n_power_sim={args.n_power_sim}), "
            f"and FDR correction (α={args.alpha})."
        )
    elif args.effects_method == "mixed_model":
        print(
            "Using Step-3 crossed mixed models for primary effects "
            "(with cluster-robust regression reported as a robustness check)."
        )
    else:
        print(
            "Using Spearman-permutation and Step-3 crossed mixed models "
            "(mixed model treated as primary when present)."
        )
    print("=" * 80)

    # Get numerical features only
    numerical_cols = extractor.get_numerical_features(features_df)

    # Exclude target variables from feature list
    target_vars = ['raw_confidence', 'calibrated_confidence', 'is_correct',
                   'calibration_error', 'prediction', 'ground_truth']
    feature_cols = [c for c in numerical_cols if c not in target_vars]

    # Filter out method-specific invalid features
    if args.method == 'self_consistency':
        # Only keep aggregated multi-response features; drop any single-response "response_*"
        # features (which would otherwise come from an arbitrary sampled response).
        keep_response_suffixes = ('_mean', '_std', '_cv', '_pooled', '_total')
        single_response_cols = [
            c for c in feature_cols
            if c.startswith('response_')
            and c != 'response_n_samples'
            and not c.endswith(keep_response_suffixes)
        ]
        if single_response_cols:
            feature_cols = [c for c in feature_cols if c not in single_response_cols]
            print(f"\n⚠ NOTE: Excluding single-response LLM response features for self_consistency")
            print(f"  Reason: Avoid analyzing an arbitrary sampled response; use aggregated response_*_mean/std/cv/pooled instead.")
            print(f"  Excluded: {', '.join(sorted(single_response_cols)[:12])}{' ...' if len(single_response_cols) > 12 else ''}")

    def _parse_prefix_list(raw: str) -> list[str]:
        return [p.strip() for p in raw.split(",") if p.strip()]

    def _parse_feature_list(raw: str, path: str) -> list[str]:
        features = [p.strip() for p in raw.split(",") if p.strip()]
        if path:
            path_obj = Path(path).expanduser()
            if not path_obj.exists():
                raise FileNotFoundError(f"Feature list file not found: {path_obj}")
            with open(path_obj, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    features.append(line)
        return features

    include_prefixes = _parse_prefix_list(args.effects_include_prefixes)
    exclude_prefixes = _parse_prefix_list(args.effects_exclude_prefixes)
    if include_prefixes:
        feature_cols = [c for c in feature_cols if any(c.startswith(p) for p in include_prefixes)]
        print(f"\n⚠ NOTE: Keeping only features with prefixes: {', '.join(include_prefixes)}")
    if exclude_prefixes:
        feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in exclude_prefixes)]
        print(f"\n⚠ NOTE: Excluding features with prefixes: {', '.join(exclude_prefixes)}")
    include_features = _parse_feature_list(args.effects_include_features, args.effects_include_features_file)
    if include_features:
        include_set = set(include_features)
        missing = [f for f in include_features if f not in feature_cols]
        feature_cols = [c for c in feature_cols if c in include_set]
        print(f"\n⚠ NOTE: Keeping only {len(feature_cols)} explicitly listed features.")
        if missing:
            print(f"  ⚠ {len(missing)} requested features not found in feature columns.")

    def _question_constant_features(cols: list[str], question_ids: np.ndarray, df: pd.DataFrame) -> list[str]:
        const = []
        q_series = pd.Series(question_ids, name="question_id")
        for c in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            tmp = pd.DataFrame({"q": q_series, "v": s}).dropna()
            if tmp.empty:
                continue
            nunique_by_q = tmp.groupby("q")["v"].nunique(dropna=True)
            if len(nunique_by_q) > 1 and (nunique_by_q <= 1).all():
                const.append(c)
        return const

    question_ids_array = np.array(question_ids)
    question_constant = _question_constant_features(feature_cols, question_ids_array, features_df)

    # Prepare target-specific feature lists
    vote_features = ['vote_agreement', 'vote_entropy', 'vote_margin', 'yes_prob', 'no_prob', 'n_votes']
    conf_feature_cols = [c for c in feature_cols if c not in vote_features]
    # Exclude leakage-prone confidence-derived features from confidence targets.
    _confidence_leakage_features = {
        "raw_confidence",
        "calibrated_confidence",
        "confidence_delta",
        "is_high_confidence",
        "is_medium_confidence",
        "is_low_confidence",
    }
    conf_feature_cols = [c for c in conf_feature_cols if c not in _confidence_leakage_features]
    cal_feature_cols = [c for c in feature_cols if c not in vote_features]
    # Correctness feature list: exclude tautological derived labels and confidence-derived leakage.
    # These can act like perfect predictors of is_correct and produce absurd coefficients.
    _tautological_correctness_features = {"is_false_positive", "is_false_negative"}
    _label_leakage_features = {"prediction", "ground_truth", "is_correct"}
    _correctness_leakage_features = (
        _tautological_correctness_features | _confidence_leakage_features | _label_leakage_features
    )
    corr_feature_cols = [c for c in feature_cols if c not in _correctness_leakage_features]

    # ========================================================================
    # Mixed-model feature effect screening (crossed REs: answer_id + rubric_id)
    # ========================================================================
    if (
        args.effects_method in {"mixed_model", "all"}
        and not args.effects_nonlinear_only
        and not args.calibration_regression_only
    ):
        effects_df, loo_details, qc_df, qc_loo_details = _pipeline_run_mixed_model_feature_effects(
            features_df=features_df,
            corr_feature_cols=corr_feature_cols,
            conf_feature_cols=conf_feature_cols,
            cal_feature_cols=cal_feature_cols,
            question_constant=question_constant,
            args=args,
        )

        if effects_df is not None:
            effects_path = step3_tables_dir / f"mixed_model_effects_{model_name}{method_tag}{file_suffix}.csv"
            effects_path.parent.mkdir(parents=True, exist_ok=True)
            effects_df.to_csv(effects_path, index=False)
            print(f"\nMixed-model effects saved to: {effects_path}")

            if loo_details:
                loo_path = step3_tables_dir / f"mixed_model_effects_loo_{model_name}{method_tag}{file_suffix}.json"
                with open(loo_path, "w", encoding="utf-8") as f:
                    json.dump(loo_details, f, indent=2)
                print(f"LOO-Q sensitivity details saved to: {loo_path}")
            if output_manager:
                output_manager.register(
                    step="step3",
                    artifact="mixed_model_effects",
                    path=effects_path,
                    model=model_name,
                    method=args.method,
                )
                if loo_details:
                    output_manager.register(
                        step="step3",
                        artifact="mixed_model_effects_loo",
                        path=loo_path,
                        model=model_name,
                        method=args.method,
                    )
                if output_manager.config.save_legacy:
                    legacy_effects_path = legacy_analysis_dir / f"mixed_model_effects_{model_name}_{csv_timestamp}.csv"
                    legacy_effects_path.parent.mkdir(parents=True, exist_ok=True)
                    effects_df.to_csv(legacy_effects_path, index=False)
                    if loo_details:
                        legacy_loo_path = legacy_analysis_dir / f"mixed_model_effects_loo_{model_name}_{csv_timestamp}.json"
                        with open(legacy_loo_path, "w", encoding="utf-8") as f:
                            json.dump(loo_details, f, indent=2)
        else:
            print("\n⚠ No primary mixed-model features to report (after filtering).")

        if args.effects_question_constant_separate_block:
            if qc_df is not None:
                qc_path = step3_tables_dir / f"mixed_model_effects_question_level_{model_name}{method_tag}{file_suffix}.csv"
                qc_path.parent.mkdir(parents=True, exist_ok=True)
                qc_df.to_csv(qc_path, index=False)
                print(f"\nQuestion-level mixed-model effects saved to: {qc_path}")
                if qc_loo_details:
                    qc_loo_path = step3_tables_dir / f"mixed_model_effects_question_level_loo_{model_name}{method_tag}{file_suffix}.json"
                    with open(qc_loo_path, "w", encoding="utf-8") as f:
                        json.dump(qc_loo_details, f, indent=2)
                    print(f"Question-level LOO-Q details saved to: {qc_loo_path}")
                if output_manager:
                    output_manager.register(
                        step="step3",
                        artifact="mixed_model_effects_question_level",
                        path=qc_path,
                        model=model_name,
                        method=args.method,
                    )
                    if qc_loo_details:
                        output_manager.register(
                            step="step3",
                            artifact="mixed_model_effects_question_level_loo",
                            path=qc_loo_path,
                            model=model_name,
                            method=args.method,
                        )
            elif set(question_constant):
                print("\n⚠ No question-constant features to report in separate block.")

    # ========================================================================
    # Non-linear feature effects (crossed mixed models)
    # ========================================================================
    if args.effects_nonlinear:
        if args.effects_method not in {"mixed_model", "all"} and not args.effects_nonlinear_only:
            print("\n" + "-" * 80)
            print("⚠ Non-linear effects requested but effects-method excludes mixed_model.")
            print("  Skipping nonlinear fits. Re-run with --effects-method mixed_model or --effects-method all.")
            print("-" * 80)
        else:
            print("\n" + "=" * 80)
            print("NON-LINEAR FEATURE EFFECTS (CROSSED MIXED MODELS)")
            print("=" * 80)
            print(
                "Effect sizes:\n"
                "  • Delta_10_90: predicted change from 10th to 90th percentile of feature\n"
                "  • AME: average marginal effect (approx. average slope)\n"
                "Model comparison:\n"
                "  • Linear vs Nonlinear (AIC/BIC + LRT where valid)\n"
            )

            nl_methods = (
                ["spline", "binned", "poly"]
                if args.effects_nonlinear_method == "all"
                else [args.effects_nonlinear_method]
            )
            nl_max = args.effects_nonlinear_max_features
            if nl_max is None:
                nl_max = args.effects_max_features

            targets_for_nonlinear = [
                ("raw_confidence", "Confidence", "continuous", conf_feature_cols),
                ("is_correct", "Correctness", "binary", corr_feature_cols),
            ]
            if args.include_bin_calibration_error:
                targets_for_nonlinear.insert(1, ("calibration_error", "Calibration Error", "continuous", cal_feature_cols))
                print("⚠ Including bin-assigned calibration_error in nonlinear effects (exploratory).")
            if args.effects_nonlinear_skip_question_features:
                filtered_targets = []
                for target_col, target_label, out_type, cols in targets_for_nonlinear:
                    filtered = [c for c in cols if not c.startswith("question_")]
                    removed = len(cols) - len(filtered)
                    if removed:
                        print(f"Note: skipping {removed} question_* features for nonlinear {target_col}.")
                    filtered_targets.append((target_col, target_label, out_type, filtered))
                targets_for_nonlinear = filtered_targets

            sid = features_df[args.student_col].to_numpy()
            qid = features_df[args.question_col].to_numpy()
            rid = features_df[args.rubric_col].to_numpy()

            nl_rows: list[dict] = []
            curves_rows: list[pd.DataFrame] = []

            for method in nl_methods:
                for target_col, target_label, out_type, cols in targets_for_nonlinear:
                    if target_col not in features_df.columns:
                        print(f"Warning: missing target column {target_col}; skipping.")
                        continue
                    if not cols:
                        print(f"Warning: no features left for nonlinear {target_col}; skipping.")
                        continue
                    y = features_df[target_col].to_numpy()

                    if nl_max is not None and nl_max > 0:
                        cols = cols[:nl_max]
                        print(f"⚠ Limiting nonlinear effects to first {nl_max} features for {target_col} ({method}).")

                    feature_arrays = [(col, features_df[col].to_numpy()) for col in cols]
                    pbar = tqdm(
                        feature_arrays,
                        total=len(feature_arrays),
                        desc=f"Nonlinear effects [{target_col}] ({method})",
                        unit="feature",
                        dynamic_ncols=True,
                        leave=True,
                        disable=(not args.force_progress and not sys.stdout.isatty()),
                    )
                    for feature_col, x in pbar:
                        pbar.set_postfix_str(feature_col)
                        try:
                            res, curve_df = estimate_feature_effect_nonlinear(
                                feature=x,
                                outcome=y,
                                student_ids=sid,
                                question_ids=qid,
                                rubric_ids=rid,
                                outcome_type=out_type,
                                backend=args.effects_backend,
                                method=method,
                                spline_df=args.effects_nonlinear_spline_df,
                                poly_degree=args.effects_nonlinear_poly_degree,
                                n_bins=args.effects_nonlinear_bins,
                                n_grid=args.effects_nonlinear_n_grid,
                                standardize_feature=True,
                                standardize_outcome=args.effects_standardize_outcome,
                                transform_bounded=args.effects_transform_bounded,
                                include_student_vc_glmm=args.effects_include_student_vc_glmm,
                                alpha=args.alpha,
                                optimizer=args.effects_optimizer,
                                maxfun=args.effects_lme4_maxfun,
                                bootstrap_n=args.effects_nonlinear_bootstrap,
                                bootstrap_cluster=args.effects_nonlinear_bootstrap_cluster,
                                ci_method=args.effects_nonlinear_ci_method,
                                ci_type=args.effects_nonlinear_ci_type,
                                jackknife_max=args.effects_nonlinear_jackknife_max,
                                log_progress=args.effects_nonlinear_bootstrap > 0,
                                progress_desc=f"Bootstrap [{target_col}] {feature_col}",
                                seed=int(rng.integers(1, 2**31 - 1)),
                            )
                            row = res.__dict__.copy()
                            row.update(
                                {
                                    "feature": feature_col,
                                    "target": target_label,
                                    "target_col": target_col,
                                    "outcome_type": out_type,
                                    "spline_df": args.effects_nonlinear_spline_df if method == "spline" else np.nan,
                                    "poly_degree": args.effects_nonlinear_poly_degree if method == "poly" else np.nan,
                                    "n_bins_requested": args.effects_nonlinear_bins if method == "binned" else np.nan,
                                    "n_bins_used": len(curve_df) if method == "binned" else np.nan,
                                    "feature_mean": float(np.nanmean(x)),
                                    "feature_sd": float(np.nanstd(x)),
                                    "bootstrap_n": args.effects_nonlinear_bootstrap,
                                    "bootstrap_cluster": args.effects_nonlinear_bootstrap_cluster,
                                    "ci_method": args.effects_nonlinear_ci_method,
                                    "ci_type": args.effects_nonlinear_ci_type,
                                    "jackknife_n": res.jackknife_n,
                                    "n_grid": args.effects_nonlinear_n_grid,
                                }
                            )
                            if res.warnings:
                                row["warnings"] = "; ".join(res.warnings)
                            nl_rows.append(row)

                            curve_df = curve_df.copy()
                            curve_df["feature"] = feature_col
                            curve_df["target"] = target_label
                            curve_df["target_col"] = target_col
                            curve_df["method"] = method
                            curves_rows.append(curve_df)
                        except KeyboardInterrupt:
                            raise
                        except BaseException as e:
                            nl_rows.append(
                                {
                                    "feature": feature_col,
                                    "target": target_label,
                                    "target_col": target_col,
                                    "outcome_type": out_type,
                                    "method": method,
                                    "error": str(e),
                                }
                            )

            if nl_rows:
                nl_df = pd.DataFrame(nl_rows)
                if "lrt_p_value" not in nl_df.columns:
                    nl_df["lrt_p_value"] = np.nan
                nl_df["lrt_p_adj"] = np.nan
                nl_df["lrt_significant"] = False
                for tgt in nl_df["target_col"].dropna().unique():
                    for m in nl_df["method"].dropna().unique():
                        mask = (
                            (nl_df["target_col"] == tgt)
                            & (nl_df["method"] == m)
                            & np.isfinite(nl_df["lrt_p_value"])
                        )
                        if not mask.any():
                            continue
                        p = nl_df.loc[mask, "lrt_p_value"].to_numpy(dtype=float)
                        p_adj, sig = benjamini_hochberg(p, alpha=args.alpha)
                        nl_df.loc[mask, "lrt_p_adj"] = p_adj
                        nl_df.loc[mask, "lrt_significant"] = sig

                nl_path = step3_tables_dir / f"nonlinear_effects_{model_name}{method_tag}{file_suffix}.csv"
                nl_path.parent.mkdir(parents=True, exist_ok=True)
                nl_df.to_csv(nl_path, index=False)
                print(f"\nNonlinear effects saved to: {nl_path}")

                if curves_rows:
                    curves_df = pd.concat(curves_rows, ignore_index=True)
                    curves_path = step3_intermediate_dir / f"nonlinear_effects_curves_{model_name}{method_tag}{file_suffix}.csv"
                    curves_df.to_csv(curves_path, index=False)
                    print(f"Nonlinear effect curves saved to: {curves_path}")
                else:
                    curves_df = None
                    curves_path = None

                if output_manager:
                    output_manager.register(
                        step="step3",
                        artifact="nonlinear_effects",
                        path=nl_path,
                        model=model_name,
                        method=args.method,
                    )
                    if curves_path:
                        output_manager.register(
                            step="step3",
                            artifact="nonlinear_effects_curves",
                            path=curves_path,
                            model=model_name,
                            method=args.method,
                        )
                    if output_manager.config.save_legacy:
                        legacy_nl_path = legacy_analysis_dir / f"nonlinear_effects_{model_name}_{csv_timestamp}.csv"
                        legacy_nl_path.parent.mkdir(parents=True, exist_ok=True)
                        nl_df.to_csv(legacy_nl_path, index=False)
                        if curves_path:
                            legacy_curves_path = legacy_analysis_dir / f"nonlinear_effects_curves_{model_name}_{csv_timestamp}.csv"
                            curves_df.to_csv(legacy_curves_path, index=False)

                print("\n" + "-" * 120)
                print(
                    f"{'Feature':<30} {'Target':<14} {'Method':<8} "
                    f"{'Delta_10_90':>12} {'CI Low':>10} {'CI High':>10} "
                    f"{'AME':>10} {'LRT p':>10}"
                )
                print("-" * 120)
                for _, row in nl_df.dropna(subset=["feature", "target"]).iterrows():
                    delta = row.get("delta_10_90", np.nan)
                    ci_low = row.get("delta_ci_low", np.nan)
                    ci_high = row.get("delta_ci_high", np.nan)
                    ame = row.get("ame", np.nan)
                    lrt_p = row.get("lrt_p_value", np.nan)
                    print(
                        f"{str(row['feature'])[:30]:<30} {str(row['target'])[:14]:<14} {str(row.get('method','')):<8} "
                        f"{delta:>12.4f} {ci_low:>10.4f} {ci_high:>10.4f} {ame:>10.4f} {lrt_p:>10.6f}"
                    )
                print("-" * 120)

                if args.plot and curves_rows:
                    import re

                    def _safe_name(name: str) -> str:
                        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

                    plot_candidates = nl_df.copy()
                    plot_candidates["score"] = plot_candidates["lrt_p_value"]
                    if plot_candidates["score"].isna().all():
                        plot_candidates["score"] = -plot_candidates["delta_10_90"].abs()
                    plot_candidates = plot_candidates.sort_values("score").head(args.effects_nonlinear_plot_max)

                    plots_dir = step3_plots_dir or (project_root / "plots")
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    for _, row in plot_candidates.iterrows():
                        f_name = row["feature"]
                        t_col = row["target_col"]
                        m_name = row["method"]
                        if curves_df is None:
                            continue
                        sub = curves_df[
                            (curves_df["feature"] == f_name)
                            & (curves_df["target_col"] == t_col)
                            & (curves_df["method"] == m_name)
                        ].copy()
                        if sub.empty:
                            continue
                        sub = sub.sort_values("feature_value")

                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(sub["feature_value"], sub["pred_mean"], color="tab:blue")
                        if sub["pred_ci_low"].notna().any():
                            ax.fill_between(
                                sub["feature_value"],
                                sub["pred_ci_low"],
                                sub["pred_ci_high"],
                                color="tab:blue",
                                alpha=0.2,
                            )
                        ax.set_xlabel(f_name)
                        ax.set_ylabel(t_col)
                        ax.set_title(f"Nonlinear effect: {f_name} -> {t_col} ({m_name})")
                        ax.grid(True, alpha=0.2)

                        plot_path = plots_dir / (
                            f"nonlinear_curve_{args.method}_{_safe_name(f_name)}_{_safe_name(t_col)}_{m_name}{file_suffix}.png"
                        )
                        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
                        plt.close(fig)
                        if output_manager:
                            output_manager.register(
                                step="step3",
                                artifact="nonlinear_curve_plot",
                                path=plot_path,
                                model=model_name,
                                method=args.method,
                                extra={"feature": str(f_name), "target_col": str(t_col), "basis": str(m_name)},
                            )
            else:
                print("\n⚠ No nonlinear effects computed.")

    # Features constant within question cause perfect collinearity with question fixed effects in
    # cluster-robust OLS (outcome ~ feature + C(question_id)), leading to exploding/unidentifiable betas.
    # Automatically exclude those from cluster-robust tables; question-level effects should be analyzed
    # via the dedicated question-level mixed-model + bootstrap section.
    if question_constant:
        preview = ", ".join(question_constant[:10])
        suffix = "" if len(question_constant) <= 10 else f" (+{len(question_constant) - 10} more)"
        print(f"\n⚠ NOTE: Excluding {len(question_constant)} question-constant features from cluster-robust tables "
              f"(collinear with question fixed effects). Examples: {preview}{suffix}")

    conf_feature_cols_cluster = [c for c in conf_feature_cols if c not in question_constant]
    cal_feature_cols_cluster = [c for c in cal_feature_cols if c not in question_constant]
    corr_feature_cols_cluster = [c for c in corr_feature_cols if c not in question_constant]

    if args.method == 'self_consistency':
        removed_votes = [v for v in vote_features if v in feature_cols]
        if removed_votes:
            print(f"\n⚠ NOTE: Removing vote features from confidence/calibration-error correlations:")
            print(f"  Removed for confidence/calibration-error: {', '.join(removed_votes)}")
            print(f"  Still used for correctness correlations.")

    # Create analyzer
    analyzer = FeatureAnalyzer(predictions, instances)

    # Extract cluster IDs once (for cluster-robust/power/assurance)
    student_ids_array = None
    question_ids_array = None

    if args.student_col in features_df.columns:
        student_ids_array = features_df[args.student_col].values
    else:
        student_ids_array = np.array([inst.metadata.get(args.student_col, '') for inst in instances])

    if args.question_col in features_df.columns:
        question_ids_array = features_df[args.question_col].values
    else:
        question_ids_array = np.array([inst.metadata.get(args.question_col, '') for inst in instances])
    if args.rubric_col:
        if args.rubric_col in features_df.columns:
            rubric_ids_array = features_df[args.rubric_col].values
        else:
            rubric_ids_array = np.array([inst.metadata.get(args.rubric_col, '') for inst in instances])
    else:
        rubric_ids_array = None

    run_spearman_suite = (
        args.effects_method in {"spearman_permutation", "all"}
        and not args.no_correlation_tables
        and not args.calibration_regression_only
    )
    run_mixed_model_suite = (
        args.effects_method in {"mixed_model", "all"}
        and not args.effects_nonlinear_only
        and not args.calibration_regression_only
    )
    run_cluster_robust = run_mixed_model_suite and not args.no_cluster_robust and not args.no_correlation_tables
    run_nonlinear_suite = args.effects_nonlinear and not args.calibration_regression_only
    if (
        run_mixed_model_suite
        and not run_cluster_robust
        and not run_spearman_suite
        and not args.no_correlation_tables
    ):
        run_spearman_suite = True
        print("⚠ Cluster-robust correlations disabled; running Spearman permutation tables for summary output.")
    # Only require explicit cluster IDs when we will actually run mixed-model / cluster-robust analyses.
    # Power/assurance methods that depend on clustering are skipped when effects-method excludes mixed_model.
    needs_cluster_ids = run_mixed_model_suite or run_nonlinear_suite

    if needs_cluster_ids and student_ids_array is None:
        raise ValueError("student IDs are required for mixed-model/cluster-robust methods; specify via --student-col")
    if needs_cluster_ids and question_ids_array is None:
        raise ValueError("question IDs are required for mixed-model/cluster-robust methods; specify via --question-col")

    n_clusters_student = len(np.unique(student_ids_array)) if student_ids_array is not None else 0
    n_clusters_question = len(np.unique(question_ids_array)) if question_ids_array is not None else 0
    n_clusters_rubric = len(np.unique(rubric_ids_array)) if rubric_ids_array is not None else 0

    if needs_cluster_ids and (n_clusters_student < 2 or n_clusters_question < 2):
        raise ValueError(
            f"Cluster methods requested but insufficient clusters "
            f"(students={n_clusters_student}, questions={n_clusters_question}). "
            f"Provide valid --student-col/--question-col with at least 2 unique values."
        )

    # ========================================================================
    # 1. Features correlated with raw confidence
    # ========================================================================
    print("\n" + "-" * 80)

    conf_corr_df_perm = None
    conf_corr_df_cluster = None

    if run_spearman_suite:
        if run_mixed_model_suite:
            print("Method: Spearman with Permutation Tests (EXPLORATORY - ignores clustering)")
        else:
            print("Method: Spearman with Permutation Tests (PRIMARY)")
        conf_corr_df_perm = compute_correlations_with_stats(
            features_df,
            conf_feature_cols,
            'raw_confidence',
            alpha=args.alpha,
            n_perm=args.n_perm,
            n_boot=args.n_boot,
            n_power_sim=args.n_power_sim,
            compute_power=(args.power_method in {"spearman_perm", "all"}),
            use_fresh_perms=args.use_fresh_perms_power,
            rng=rng,
            adaptive_perm=args.adaptive_perm,
            adaptive_base_n_perm=args.adaptive_base_n_perm,
            adaptive_max_n_perm=args.adaptive_max_n_perm,
            adaptive_band=adaptive_band,
            bootstrap_method=args.bootstrap_method,
        )
        title = "Features Correlated with CONFIDENCE (Spearman Permutation)"
        if run_mixed_model_suite:
            title += " - EXPLORATORY"
        else:
            title += " - PRIMARY"
        print_correlation_table(conf_corr_df_perm, title, args.top_k, args.alpha)

    if run_cluster_robust:
        print("\nMethod: Mixed Model + Cluster-Robust (PRIMARY - accounts for clustering)")
        conf_corr_df_cluster = compute_correlations_cluster_robust(
            features_df,
            conf_feature_cols_cluster,
            'raw_confidence',
            student_ids=student_ids_array,
            question_ids=question_ids_array,
            rubric_ids=rubric_ids_array,
            alpha=args.alpha,
            use_ranks=True,
            standardize=True,
            small_cluster_primary=True,
            mixed_model_backend=args.effects_backend,
            mixed_model_optimizer=args.effects_optimizer,
            mixed_model_transform_bounded=args.effects_transform_bounded,
            mixed_model_include_student_vc_glmm=args.effects_include_student_vc_glmm,
        )
        title = "Features Correlated with CONFIDENCE (Mixed Model / Cluster-Robust)"
        title += " - PRIMARY"
        # Show which random effects were used
        if 'random_effects_used' in conf_corr_df_cluster.columns:
            print(f"  Random effects: {conf_corr_df_cluster.iloc[0]['random_effects_used']}")
        print_cluster_robust_table(conf_corr_df_cluster, title, args.top_k, args.alpha)

    # Decide which to use for downstream
    if run_cluster_robust:
        conf_corr_df = conf_corr_df_cluster
    else:
        conf_corr_df = conf_corr_df_perm

    # ========================================================================
    # 2. Features correlated with correctness
    # ========================================================================
    corr_corr_df_perm = None
    corr_corr_df_cluster = None

    if run_spearman_suite:
        if run_mixed_model_suite:
            print("Method: Spearman with Permutation Tests (EXPLORATORY - ignores clustering)")
        else:
            print("Method: Spearman with Permutation Tests (PRIMARY)")
        corr_corr_df_perm = compute_correlations_with_stats(
            features_df,
            corr_feature_cols,
            'is_correct',
            alpha=args.alpha,
            n_perm=args.n_perm,
            n_boot=args.n_boot,
            n_power_sim=args.n_power_sim,
            compute_power=(args.power_method in {"spearman_perm", "all"}),
            use_fresh_perms=args.use_fresh_perms_power,
            rng=rng,
            adaptive_perm=args.adaptive_perm,
            adaptive_base_n_perm=args.adaptive_base_n_perm,
            adaptive_max_n_perm=args.adaptive_max_n_perm,
            adaptive_band=adaptive_band,
            bootstrap_method=args.bootstrap_method,
        )
        title = "Features Correlated with CORRECTNESS (Spearman Permutation)"
        if run_mixed_model_suite:
            title += " - EXPLORATORY"
        else:
            title += " - PRIMARY"
        print_correlation_table(corr_corr_df_perm, title, args.top_k, args.alpha)

    if run_cluster_robust:
        print("\nMethod: Mixed Model + Cluster-Robust (PRIMARY - accounts for clustering)")
        corr_corr_df_cluster = compute_correlations_cluster_robust(
            features_df,
            corr_feature_cols_cluster,
            'is_correct',
            student_ids=student_ids_array,
            question_ids=question_ids_array,
            rubric_ids=rubric_ids_array,
            alpha=args.alpha,
            use_ranks=True,
            standardize=True,
            small_cluster_primary=True,
            mixed_model_backend=args.effects_backend,
            mixed_model_optimizer=args.effects_optimizer,
            mixed_model_transform_bounded=args.effects_transform_bounded,
            mixed_model_include_student_vc_glmm=args.effects_include_student_vc_glmm,
        )
        title = "Features Correlated with CORRECTNESS (Mixed Model / Cluster-Robust)"
        title += " - PRIMARY"
        if 'random_effects_used' in corr_corr_df_cluster.columns:
            print(f"  Random effects: {corr_corr_df_cluster.iloc[0]['random_effects_used']}")
        print_cluster_robust_table(corr_corr_df_cluster, title, args.top_k, args.alpha)

    # Decide which to use for downstream
    if run_cluster_robust:
        corr_corr_df = corr_corr_df_cluster
    else:
        corr_corr_df = corr_corr_df_perm

    # ========================================================================
    # 3. Features correlated with calibration error (OPTIONAL - exploratory only)
    # ========================================================================
    # Bin-assigned calibration_error has piecewise-constant values that make
    # LMM/correlation analysis poorly suited. Use calibration regression instead.
    # This section only runs if --include-bin-calibration-error is specified.
    cal_corr_df_perm = None
    cal_corr_df_cluster = None
    cal_corr_df = None

    if args.include_bin_calibration_error:
        print("\n" + "=" * 100)
        print("Features Correlated with CALIBRATION ERROR (EXPLORATORY - bin-assigned)")
        print("=" * 100)
        print("⚠ WARNING: Bin-assigned calibration_error has piecewise-constant values.")
        print("  This makes LMM/correlation likelihoods poorly matched. Results are exploratory only.")
        print("  For proper calibration analysis, see CALIBRATION REGRESSION section below.\n")

        if run_spearman_suite:
            if run_mixed_model_suite:
                print("Method: Spearman with Permutation Tests (EXPLORATORY - ignores clustering)")
            else:
                print("Method: Spearman with Permutation Tests (PRIMARY)")
            cal_corr_df_perm = compute_correlations_with_stats(
                features_df,
                cal_feature_cols,
                'calibration_error',
                alpha=args.alpha,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                n_power_sim=args.n_power_sim,
                compute_power=(args.power_method in {"spearman_perm", "all"}),
                use_fresh_perms=args.use_fresh_perms_power,
                rng=rng,
                adaptive_perm=args.adaptive_perm,
                adaptive_base_n_perm=args.adaptive_base_n_perm,
                adaptive_max_n_perm=args.adaptive_max_n_perm,
                adaptive_band=adaptive_band,
                bootstrap_method=args.bootstrap_method,
            )
            title = "Features Correlated with CALIBRATION ERROR (Spearman Permutation)"
            if run_mixed_model_suite:
                title += " - EXPLORATORY"
            else:
                title += " - PRIMARY"
            print_correlation_table(cal_corr_df_perm, title, args.top_k, args.alpha)

        if run_cluster_robust:
            print("\nMethod: Mixed Model + Cluster-Robust (PRIMARY - accounts for clustering)")
            cal_corr_df_cluster = compute_correlations_cluster_robust(
                features_df,
                cal_feature_cols_cluster,
                'calibration_error',
                student_ids=student_ids_array,
                question_ids=question_ids_array,
                rubric_ids=rubric_ids_array,
                alpha=args.alpha,
                use_ranks=True,
                standardize=True,
                small_cluster_primary=True,
                mixed_model_backend=args.effects_backend,
                mixed_model_optimizer=args.effects_optimizer,
                mixed_model_transform_bounded=args.effects_transform_bounded,
                mixed_model_include_student_vc_glmm=args.effects_include_student_vc_glmm,
            )
            title = "Features Correlated with CALIBRATION ERROR (Mixed Model / Cluster-Robust)"
            title += " - PRIMARY"
            if 'random_effects_used' in cal_corr_df_cluster.columns:
                print(f"  Random effects: {cal_corr_df_cluster.iloc[0]['random_effects_used']}")
            print_cluster_robust_table(cal_corr_df_cluster, title, args.top_k, args.alpha)

        # Decide which to use for downstream
        if run_cluster_robust:
            cal_corr_df = cal_corr_df_cluster
        else:
            cal_corr_df = cal_corr_df_perm
    else:
        print("\n" + "-" * 100)
        print("ℹ Skipping bin-assigned calibration_error correlations (not recommended).")
        print("  Use --include-bin-calibration-error to enable. See CALIBRATION REGRESSION section for proper analysis.")
        print("-" * 100)

    # ========================================================================
    # 4. CALIBRATION REGRESSION: Feature-moderated calibration (PRIMARY)
    # ========================================================================
    # This is the proper way to test whether features moderate calibration quality.
    # Model: is_correct ~ logit(conf) + feature + logit(conf):feature + (1|student_id) + (1|rubric_id)
    # The interaction term (logit(conf):feature) tests whether the feature moderates
    # the confidence-accuracy relationship (i.e., calibration quality).
    cal_reg_results = []
    cal_reg_df = None

    if not args.no_calibration_regression:
        cal_reg_mode = args.calibration_regression_mode
        print("\n" + "=" * 100)
        print("CALIBRATION REGRESSION: FEATURE-MODERATED CALIBRATION (PRIMARY)")
        print("=" * 100)
        if cal_reg_mode == "linear":
            print("Mode: linear")
            print("Testing: is_correct ~ logit(conf) + feature + logit(conf):feature + (1|student_id) [+ (1|rubric_id)]")
            print("The interaction term tests whether the feature moderates calibration quality.\n")
            print("Interpretation:")
            print("  • Positive beta_interaction → feature associated with BETTER calibration (steeper slope)")
            print("  • Negative beta_interaction → feature associated with WORSE calibration (flatter slope)")
            print("  • p_interaction (LRT) tests whether the moderation effect is statistically significant\n")
        else:
            print("Mode: nonlinear")
            print("Testing: is_correct ~ bs(logit(conf), df=k) * feature + (1|student_id) [+ (1|rubric_id)]")
            print("LRT tests whether the calibration curve shape varies with the feature.\n")

        cal_reg_df, cal_reg_curve_rows = _pipeline_run_calibration_regression(
            features_df=features_df,
            conf_feature_cols=conf_feature_cols,
            confidence_leakage_features=_confidence_leakage_features,
            label_leakage_features=_label_leakage_features,
            args=args,
        )

        if cal_reg_df is not None:
            # Print table
            if cal_reg_mode == "linear":
                print("\n" + "-" * 120)
                print(f"{'Feature':<35} {'beta_main':>10} {'se_main':>10} {'p_main':>10} {'beta_int':>10} {'se_int':>10} {'p_int':>10} {'p_adj':>10} {'sig':>5}")
                print("-" * 120)
                for _, row in cal_reg_df.iterrows():
                    sig_str = "***" if row.get('significant_interaction', False) and row.get('p_interaction_adj', 1) < 0.001 else \
                              "**" if row.get('significant_interaction', False) and row.get('p_interaction_adj', 1) < 0.01 else \
                              "*" if row.get('significant_interaction', False) else ""
                    beta_main = row.get('beta_main', np.nan)
                    se_main = row.get('se_main', np.nan)
                    p_main = row.get('p_main', np.nan)
                    beta_int = row.get('beta_interaction', np.nan)
                    se_int = row.get('se_interaction', np.nan)
                    p_int = row.get('p_interaction', np.nan)
                    p_adj_val = row.get('p_interaction_adj', np.nan)

                    print(f"{row['feature']:<35} {beta_main:>10.4f} {se_main:>10.4f} {p_main:>10.6f} {beta_int:>10.4f} {se_int:>10.4f} {p_int:>10.6f} {p_adj_val:>10.6f} {sig_str:>5}")
                print("-" * 120)
            else:
                print("\n" + "-" * 120)
                print(f"{'Feature':<35} {'LRT p':>10} {'p_adj':>10} {'AME':>10} {'CI Low':>10} {'CI High':>10} {'sig':>5}")
                print("-" * 120)
                for _, row in cal_reg_df.iterrows():
                    sig_str = "***" if row.get('significant_interaction', False) and row.get('p_interaction_adj', 1) < 0.001 else \
                              "**" if row.get('significant_interaction', False) and row.get('p_interaction_adj', 1) < 0.01 else \
                              "*" if row.get('significant_interaction', False) else ""
                    p_int = row.get('p_interaction', np.nan)
                    p_adj_val = row.get('p_interaction_adj', np.nan)
                    ame = row.get('ame', np.nan)
                    ame_ci_low = row.get('ame_ci_low', np.nan)
                    ame_ci_high = row.get('ame_ci_high', np.nan)
                    print(f"{row['feature']:<35} {p_int:>10.6f} {p_adj_val:>10.6f} {ame:>10.4f} {ame_ci_low:>10.4f} {ame_ci_high:>10.4f} {sig_str:>5}")
                print("-" * 120)

            # Summary statistics
            n_sig = cal_reg_df['significant_interaction'].sum() if 'significant_interaction' in cal_reg_df.columns else 0
            n_converged = cal_reg_df['converged'].sum() if 'converged' in cal_reg_df.columns else len(cal_reg_df)
            print(f"\nSummary:")
            print(f"  Features tested: {len(cal_reg_df)}")
            print(f"  Converged: {n_converged}")
            print(f"  Significant moderation effects (FDR-adjusted): {n_sig}")

            if n_sig > 0:
                print(f"\nTop features moderating calibration:")
                sig_df = cal_reg_df[cal_reg_df['significant_interaction'] == True].head(5)
                for _, row in sig_df.iterrows():
                    if cal_reg_mode == "linear":
                        beta_int = row.get('beta_interaction', np.nan)
                        interp = "better calibration" if beta_int > 0 else "worse calibration"
                        print(f"  • {row['feature']}: beta_int={beta_int:.4f} → {interp}")
                    else:
                        ame = row.get('ame', np.nan)
                        interp = "higher predicted accuracy at high feature" if ame > 0 else "lower predicted accuracy at high feature"
                        print(f"  • {row['feature']}: AME={ame:.4f} → {interp}")

            # Save calibration regression results
            if not args.no_save_correlations:
                cal_reg_path = step3_tables_dir / f"calibration_regression_{model_name}{method_tag}{file_suffix}.csv"
                cal_reg_path.parent.mkdir(parents=True, exist_ok=True)
                cal_reg_df.to_csv(cal_reg_path, index=False)
                print(f"\nCalibration regression results saved to: {cal_reg_path}")
                if output_manager:
                    output_manager.register(
                        step="step3",
                        artifact="calibration_regression",
                        path=cal_reg_path,
                        model=model_name,
                        method=args.method,
                    )
                    if output_manager.config.save_legacy:
                        legacy_cal_reg_path = legacy_analysis_dir / f"calibration_regression_{model_name}_{csv_timestamp}.csv"
                        legacy_cal_reg_path.parent.mkdir(parents=True, exist_ok=True)
                        cal_reg_df.to_csv(legacy_cal_reg_path, index=False)
                if cal_reg_curve_rows:
                    cal_curve_df = pd.concat(cal_reg_curve_rows, ignore_index=True)
                    cal_curve_path = step3_intermediate_dir / f"calibration_regression_curves_{model_name}{method_tag}{file_suffix}.csv"
                    cal_curve_df.to_csv(cal_curve_path, index=False)
                    print(f"Calibration regression curves saved to: {cal_curve_path}")
                    if output_manager:
                        output_manager.register(
                            step="step3",
                            artifact="calibration_regression_curves",
                            path=cal_curve_path,
                            model=model_name,
                            method=args.method,
                        )
                        if output_manager.config.save_legacy:
                            legacy_curve_path = legacy_analysis_dir / f"calibration_regression_curves_{model_name}_{csv_timestamp}.csv"
                            legacy_curve_path.parent.mkdir(parents=True, exist_ok=True)
                            cal_curve_df.to_csv(legacy_curve_path, index=False)

                    if args.plot and cal_reg_df is not None and not cal_curve_df.empty:
                        import re

                        def _safe_name(name: str) -> str:
                            return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

                        plot_limit = args.effects_nonlinear_plot_max or 10
                        plot_candidates = cal_reg_df.copy()
                        plot_candidates["score"] = plot_candidates["p_interaction"]
                        plot_candidates = plot_candidates.sort_values("score").head(plot_limit)

                        plots_dir = step3_plots_dir or (project_root / "plots")
                        plots_dir.mkdir(parents=True, exist_ok=True)

                        for _, row in plot_candidates.iterrows():
                            f_name = row["feature"]
                            sub = cal_curve_df[cal_curve_df["feature"] == f_name].copy()
                            if sub.empty:
                                continue
                            sub = sub.sort_values("confidence")

                            q25_raw = sub["feature_q25_raw"].iloc[0] if "feature_q25_raw" in sub.columns else np.nan
                            q75_raw = sub["feature_q75_raw"].iloc[0] if "feature_q75_raw" in sub.columns else np.nan

                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.plot(sub["confidence"], sub["pred_q25"], color="tab:blue", label=f"Feature 25th ({q25_raw:.3f})")
                            ax.plot(sub["confidence"], sub["pred_q75"], color="tab:orange", label=f"Feature 75th ({q75_raw:.3f})")
                            ax.set_xlabel("confidence")
                            ax.set_ylabel("P(correct)")
                            ax.set_title(f"Calibration regression: {f_name} → P(correct)")
                            ax.set_ylim(0.0, 1.0)
                            ax.grid(True, alpha=0.2)
                            ax.legend(loc="best", fontsize=8)

                            plot_path = plots_dir / (
                                f"calibration_regression_curve_{args.method}_{_safe_name(f_name)}_{model_name}{file_suffix}.png"
                            )
                            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
                            plt.close(fig)
                            if output_manager:
                                output_manager.register(
                                    step="step3",
                                    artifact="calibration_regression_curve_plot",
                                    path=plot_path,
                                    model=model_name,
                                    method=args.method,
                                    extra={"feature": str(f_name), "mode": str(cal_reg_mode)},
                                )
        else:
            print("⚠ No calibration regression results computed.")
    else:
        print("\n" + "-" * 100)
        print("ℹ Skipping calibration regression analysis (--no-calibration-regression specified).")
        print("  Re-run without --no-calibration-regression for proper feature-calibration analysis.")
        print("-" * 100)

    # ========================================================================
    # Question-level effect analysis for ALL question-constant features
    # (uses random intercepts instead of fixed effects to avoid collinearity)
    # ========================================================================
    if args.effects_question_constant_separate_block:
        if question_constant:
            print("\n⚠ NOTE: Question-constant features analyzed in separate block (no question effect).")
    elif question_constant and run_mixed_model_suite:
        print("\n" + "=" * 80)
        print("QUESTION-LEVEL EFFECTS (Mixed Model + Question Cluster Bootstrap)")
        print("=" * 80)
        print(f"\nAnalyzing {len(question_constant)} question-constant features that were excluded from cluster-robust tables.")
        print("These features are constant within each question, so we use random effects (not fixed effects) for question.")
        
        # Default: only confidence and correctness
        # Calibration_error is optional (only if --include-bin-calibration-error)
        targets = [
            ('raw_confidence', "Confidence"),
            ('is_correct', "Correctness"),
        ]
        if args.include_bin_calibration_error:
            targets.append(('calibration_error', "Calibration Error"))
        
        question_level_results = []
        
        for q_feature in question_constant:
            if q_feature not in features_df.columns:
                continue
            # Skip if feature has insufficient variance
            if features_df[q_feature].nunique(dropna=True) < 2:
                print(f"\n⚠ Skipping {q_feature}: insufficient variance (< 2 unique values)")
                continue
                
            print(f"\n--- {q_feature} ---")
            for target_col, label in targets:
                try:
                    res_q = estimate_question_level_effect_mixed_model_bootstrap(
                        feature=features_df[q_feature].values,
                        outcome=features_df[target_col].values,
                        student_ids=student_ids_array,
                        question_ids=question_ids_array,
                        rubric_ids=rubric_ids_array,
                        use_ranks=True,
                        standardize=True,
                        alpha=args.alpha,
                        n_boot=200,
                        random_state=rng,
                    )
                    print(f"\n  {q_feature} → {label}:")
                    print(f"    Beta (mixed model): {res_q['beta']:.4f}")
                    print(f"    SE (analytic):      {res_q['se']:.4f}")
                    print(f"    CI (analytic):      [{res_q['ci_low']:.4f}, {res_q['ci_high']:.4f}]")
                    print(f"    p-value (analytic): {res_q['p_value']:.4f}")
                    print(f"    SE (q-cluster boot): {res_q['se_boot']:.4f}")
                    print(f"    CI (q-cluster boot): [{res_q['ci_low_boot']:.4f}, {res_q['ci_high_boot']:.4f}]")
                    print(f"    p-value (boot):      {res_q['p_boot']:.4f}")
                    if res_q['n_clusters_question'] <= 5:
                        print("    ⚠ Few question clusters; rely on bootstrap CI/p for inference.")
                    
                    question_level_results.append({
                        'feature': q_feature,
                        'target': label,
                        'beta': res_q['beta'],
                        'se': res_q['se'],
                        'p_value': res_q['p_value'],
                        'ci_low': res_q['ci_low'],
                        'ci_high': res_q['ci_high'],
                        'se_boot': res_q['se_boot'],
                        'p_boot': res_q['p_boot'],
                        'ci_low_boot': res_q['ci_low_boot'],
                        'ci_high_boot': res_q['ci_high_boot'],
                        'n_clusters_question': res_q['n_clusters_question'],
                    })
                except Exception as e:
                    print(f"\n  Warning: {q_feature} → {label} failed: {e}")
        
        # Print summary table of question-level results
        if question_level_results:
            print("\n" + "-" * 80)
            print("QUESTION-LEVEL EFFECTS SUMMARY TABLE")
            print("-" * 80)
            q_results_df = pd.DataFrame(question_level_results)
            # Use bootstrap p-values for significance (more reliable with few clusters)
            q_results_df['significant'] = q_results_df['p_boot'] < args.alpha
            print(f"\n{'Feature':<30} {'Target':<18} {'Beta':>8} {'p_boot':>10} {'Sig':>5}")
            print("-" * 75)
            for _, row in q_results_df.iterrows():
                sig_mark = "***" if row['p_boot'] < 0.001 else ("**" if row['p_boot'] < 0.01 else ("*" if row['p_boot'] < 0.05 else ""))
                print(f"{row['feature']:<30} {row['target']:<18} {row['beta']:>8.4f} {row['p_boot']:>10.4f} {sig_mark:>5}")
    elif question_constant and not run_mixed_model_suite:
        print(
            "\n⚠ Question-constant features detected, but effects-method excludes mixed_model.\n"
            "  Skipping question-level mixed-model analysis.\n"
            "  Re-run with --effects-method mixed_model or --effects-method all."
        )
    else:
        print("\n⚠ No question-constant features found for question-level analysis.")

    # ========================================================================
    # Global FDR Correction Across All Tests
    # ========================================================================
    print("\n" + "=" * 80)
    print("GLOBAL MULTIPLE TESTING CORRECTION")
    print("=" * 80)
    print("\nApplying FDR correction globally across ALL hypothesis tests...")

    # Collect all p-values from PRIMARY analyses only (avoid exploratory p-values when clustering is detected)
    all_p_values = []
    all_features = []
    all_targets = []
    all_effect_values = []
    all_effect_types = []

    # Build correlation set list dynamically (cal_corr_df may be None if not --include-bin-calibration-error)
    primary_corr_sets = [
        ('confidence', conf_corr_df),
        ('correctness', corr_corr_df),
    ]
    if cal_corr_df is not None:
        primary_corr_sets.append(('calibration_error', cal_corr_df))

    for target_name, corr_df in primary_corr_sets:
        if corr_df is None:
            continue
        for _, row in corr_df.iterrows():
            all_p_values.append(row['p_value'])
            all_features.append(row['feature'])
            all_targets.append(target_name)
            beta_val = row.get('beta', np.nan)
            r_val = row.get('r', np.nan)
            if np.isfinite(beta_val):
                all_effect_values.append(float(beta_val))
                all_effect_types.append("beta")
            elif np.isfinite(r_val):
                all_effect_values.append(float(r_val))
                all_effect_types.append("rho")
            else:
                all_effect_values.append(np.nan)
                all_effect_types.append("")

    # Add calibration regression interaction tests to the global pool (PRIMARY)
    if cal_reg_df is not None and not cal_reg_df.empty:
        for _, row in cal_reg_df.iterrows():
            all_p_values.append(row.get('p_interaction', np.nan))
            all_features.append(row.get('feature', ''))
            all_targets.append('calibration_regression')
            beta_int = row.get('beta_interaction', np.nan)
            ame_val = row.get('ame', np.nan)
            if np.isfinite(beta_int):
                all_effect_values.append(float(beta_int))
                all_effect_types.append("beta_int")
            elif np.isfinite(ame_val):
                all_effect_values.append(float(ame_val))
                all_effect_types.append("ame")
            else:
                all_effect_values.append(np.nan)
                all_effect_types.append("")

    all_p_values = np.array(all_p_values)

    # Apply global FDR correction
    # Replace NaN p-values with 1.0 for BH procedure
    p_values_filled = np.where(np.isnan(all_p_values), 1.0, all_p_values)
    p_adj_global, sig_global = benjamini_hochberg(p_values_filled, alpha=args.alpha)

    # Build target list dynamically
    targets_tested = ["confidence", "correctness"]
    features_tested_str = f"{len(conf_feature_cols)} (confidence), {len(corr_feature_cols)} (correctness)"
    if cal_corr_df is not None:
        targets_tested.append("calibration_error")
        features_tested_str += f", {len(cal_feature_cols)} (calibration_error)"
    if cal_reg_df is not None and not cal_reg_df.empty:
        targets_tested.append("calibration_regression")
        features_tested_str += f", {len(cal_reg_df)} (calibration_regression)"

    print(f"\nTesting {len(all_p_values)} hypotheses globally:")
    print(f"  Features tested: {features_tested_str}")
    print(f"  Targets: {len(targets_tested)} ({', '.join(targets_tested)})")
    print(f"  Total tests: {len(all_p_values)}")

    # Count significant results
    n_sig_per_target = 0
    if conf_corr_df is not None:
        n_sig_per_target += conf_corr_df['significant'].sum()
    if corr_corr_df is not None:
        n_sig_per_target += corr_corr_df['significant'].sum()
    if cal_corr_df is not None:
        n_sig_per_target += cal_corr_df['significant'].sum()
    if cal_reg_df is not None and 'significant_interaction' in cal_reg_df.columns:
        n_sig_per_target += cal_reg_df['significant_interaction'].sum()
    n_sig_global = sig_global.sum()

    print(f"\nSignificance at α={args.alpha}:")
    print(f"  Per-target FDR:  {n_sig_per_target} significant")
    print(f"  Global FDR:      {n_sig_global} significant")

    if n_sig_global < n_sig_per_target:
        print(f"\n⚠ WARNING: {n_sig_per_target - n_sig_global} findings lost with global FDR correction")
        print("  Interpretation:")
        print("    - Per-target FDR was too liberal for overall FDR control")
        print("    - Some findings may be false positives across targets")
        print("  Recommendation:")
        print("    - Report global FDR results in publication")
        print("    - Per-target FDR results are exploratory only")
    elif n_sig_global == n_sig_per_target:
        print(f"\n✓ Global and per-target FDR yield same results")
        print("  Interpretation:")
        print("    - Significant features remain significant after global correction")
        print("    - Results are robust to multiple testing across targets")
    else:
        print(f"\n✓ Global FDR is more powerful (unexpected but possible)")

    # Create global results dataframe
    global_results = pd.DataFrame({
        'feature': all_features,
        'target': all_targets,
        'effect': all_effect_values,
        'effect_type': all_effect_types,
        'p_value': all_p_values,
        'p_adjusted_global': p_adj_global,
        'significant_global': sig_global
    })

    # Merge back into original dataframes
    for target_name, corr_df in [
        ("confidence", conf_corr_df),
        ("correctness", corr_corr_df),
        ("calibration_error", cal_corr_df),
    ]:
        if corr_df is None:
            continue
        global_subset = global_results[global_results['target'] == target_name].copy()
        # Ensure alignment by resetting index
        corr_df['p_adjusted_global'] = global_subset['p_adjusted_global'].values
        corr_df['significant_global'] = global_subset['significant_global'].values

    if cal_reg_df is not None and not cal_reg_df.empty:
        global_subset = global_results[global_results['target'] == 'calibration_regression'].copy()
        cal_reg_df['p_adjusted_global'] = global_subset['p_adjusted_global'].values
        cal_reg_df['significant_global'] = global_subset['significant_global'].values
        if not args.no_save_correlations:
            cal_reg_path = step3_tables_dir / f"calibration_regression_{model_name}{method_tag}{file_suffix}.csv"
            cal_reg_df.to_csv(cal_reg_path, index=False)
            if output_manager and output_manager.config.save_legacy:
                legacy_cal_reg_path = legacy_analysis_dir / f"calibration_regression_{model_name}_{csv_timestamp}.csv"
                cal_reg_df.to_csv(legacy_cal_reg_path, index=False)

    # Display full global table
    print(f"\nGlobal FDR Table (all tests):")
    print("-" * 120)
    print(f"{'Feature':<30} {'Target':<20} {'effect':>10} {'type':>5} {'p-value':>12} {'p-adj (global)':>15} {'Sig':>5}")
    print("-" * 120)
    global_sorted = global_results.sort_values('p_adjusted_global')
    for _, row in global_sorted.iterrows():
        sig_mark = "***" if row['significant_global'] and row['p_adjusted_global'] < 0.001 else \
                   "**" if row['significant_global'] and row['p_adjusted_global'] < 0.01 else \
                   "*" if row['significant_global'] and row['p_adjusted_global'] < 0.05 else ""
        print(f"{row['feature']:<30} {row['target']:<20} {row['effect']:>10.4f} {str(row.get('effect_type','')):>5} "
              f"{row['p_value']:>12.6f} {row['p_adjusted_global']:>15.6f} {sig_mark:>5}")
    print("-" * 120)

    # Display top global findings
    print(f"\nFeatures by Global Significance (across all targets):")
    print("-" * 120)
    print(f"{'Feature':<30} {'Target':<20} {'effect':>10} {'type':>5} {'p-value':>12} {'p-adj (global)':>15} {'Sig':>5}")
    print("-" * 120)

    # Sort by p_adjusted_global
    top_global = global_results[global_results['significant_global']].sort_values('p_adjusted_global')
    if len(top_global) == 0:
        print("  No features significant after global FDR correction.")
    else:
        for _, row in top_global.iterrows():
            print(f"{row['feature']:<30} {row['target']:<20} {row['effect']:>10.4f} {str(row.get('effect_type','')):>5} "
                  f"{row['p_value']:>12.6f} {row['p_adjusted_global']:>15.6f} {'***':>5}")

    print("-" * 120)

    # Summary by target
    print(f"\nSignificance Summary by Target (Global FDR):")
    for target_name in ['confidence', 'correctness', 'calibration_error', 'calibration_regression']:
        target_sig = global_results[
            (global_results['target'] == target_name) &
            (global_results['significant_global'])
        ]
        print(f"  {target_name:<20}: {len(target_sig):>3} / "
              f"{len(global_results[global_results['target'] == target_name]):>3} significant")

    print("=" * 80)

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nText Length Statistics:")
    print(f"  Answer length (mean ± std): {features_df['answer_length'].mean():.1f} ± {features_df['answer_length'].std():.1f} words")
    print(f"  Question length (mean ± std): {features_df['question_length'].mean():.1f} ± {features_df['question_length'].std():.1f} words")
    print(f"  Rubric length (mean ± std): {features_df['rubric_length'].mean():.1f} ± {features_df['rubric_length'].std():.1f} words")
    # Response length: use aggregated features for self-consistency, single-response otherwise
    if 'response_length_mean' in features_df.columns and args.method == 'self_consistency':
        print(f"  Response length (mean of means ± std): {features_df['response_length_mean'].mean():.1f} ± {features_df['response_length_mean'].std():.1f} words")
        print(f"  Response length variability (mean CV): {features_df['response_length_cv'].mean():.3f}")
    elif 'response_length' in features_df.columns:
        print(f"  Response length (mean ± std): {features_df['response_length'].mean():.1f} ± {features_df['response_length'].std():.1f} words")

    if 'response_uncertainty_count' in features_df.columns and args.method != 'self_consistency':
        print(f"\nLinguistic Markers (LLM Response):")
        print(f"  Uncertainty markers (mean): {features_df['response_uncertainty_count'].mean():.2f}")
        print(f"  Confidence markers (mean): {features_df['response_confidence_count'].mean():.2f}")
    elif args.method == 'self_consistency' and 'response_uncertainty_ratio_mean' in features_df.columns:
        print(f"\nLinguistic Markers (Aggregated across {int(features_df['response_n_samples'].iloc[0])} samples):")
        print(f"  Uncertainty ratio (mean of per-response rates): {features_df['response_uncertainty_ratio_mean'].mean():.3f}%")
        print(f"  Uncertainty ratio variability (mean std): {features_df['response_uncertainty_ratio_std'].mean():.3f}%")
        print(f"  Confidence ratio (mean of per-response rates): {features_df['response_confidence_ratio_mean'].mean():.3f}%")
        print(f"  Confidence ratio variability (mean std): {features_df['response_confidence_ratio_std'].mean():.3f}%")

    if args.method == 'self_consistency' and 'vote_agreement' in features_df.columns:
        print(f"\nVote Statistics (Self-Consistency):")
        print(f"  Vote agreement (mean): {features_df['vote_agreement'].mean():.4f}")
        print(f"  Vote entropy (mean): {features_df['vote_entropy'].mean():.4f}")
        print(f"  Vote margin (mean): {features_df['vote_margin'].mean():.4f}")

    print(f"\nCalibration Statistics:")
    print(f"  Mean confidence: {features_df['raw_confidence'].mean():.4f}")
    print(f"  Mean accuracy: {features_df['is_correct'].mean():.4f}")
    print(f"  Mean calibration error (bin-based): {features_df['calibration_error'].mean():.4f}")
    print(f"    NOTE: This is the average of bin-level errors, consistent with ECE computation")

    if args.diagnostics:
        # Validation: bin-based error should have lower correlation with confidence
        # than per-instance error would have (which is typically ~0.5-0.7)
        from scipy.stats import spearmanr
        r_with_conf, p_with_conf = spearmanr(
            features_df['raw_confidence'],
            features_df['calibration_error']
        )
        print(f"\nCalibration Error Validation:")
        print(f"  Correlation with confidence: {r_with_conf:.4f} (p={p_with_conf:.4f})")
        print(f"  Expected: low correlation (<0.3) since using bin-based error")
        print(f"  Per-instance error would show ~0.5-0.7 (mechanical correlation)")

    # ========================================================================
    # Power Curve Analysis (if requested)
    # ========================================================================
    if args.power_curve:
        # Power should be computed only for tests that were actually run.
        if args.power_method == "all":
            power_methods_to_run = ["spearman_perm", "cluster_robust", "mixed_model_lrt"]
        else:
            power_methods_to_run = [args.power_method]

        for pm in power_methods_to_run:
            if pm == "spearman_perm" and not run_spearman_suite:
                print("\n⚠ Skipping Spearman power curve: --effects-method excludes spearman_permutation.")
                continue
            if pm in {"cluster_robust", "mixed_model_lrt"} and not run_mixed_model_suite:
                print(f"\n⚠ Skipping {pm} power curve: --effects-method excludes mixed_model.")
                continue

            power_data_generator = "mixed_model" if pm in {"cluster_robust", "mixed_model_lrt"} else "copula"
            values = power_curve_beta_values if pm == "mixed_model_lrt" else power_curve_r_values

            display_power_curve(
                features_df,
                conf_feature_cols,  # Use features that correlate with confidence
                target_col='raw_confidence',
                r_values=values,
                alpha=args.alpha,
                n_power_sim=args.n_power_sim,
                n_perm=args.n_perm,
                rng=rng,
                plot=args.plot_power_curve,
                n_plot_points=args.power_curve_n_points,
                model_name=model_name,
                method=args.method,
                power_method=pm,
                student_ids=student_ids_array,
                question_ids=question_ids_array,
                data_generator=power_data_generator,
                rubric_ids=rubric_ids_array,
                n_students_target=args.power_curve_n_students_target,
                mixed_model_include_student_vc_glmm=args.effects_include_student_vc_glmm,
                mixed_model_use_ranks=args.effects_use_ranks,
                mixed_model_standardize_outcome=args.effects_standardize_outcome,
                mixed_model_transform_bounded=args.effects_transform_bounded,
                output_dir=step3_plots_dir,
            )

    # ========================================================================
    # Validate Power Estimation Method (if requested)
    # ========================================================================
    if args.validate_power_method:
        # Use fewer simulations for validation (faster)
        validate_power_estimation(
            features_df,
            conf_feature_cols,  # Use features that correlate with confidence
            target_col='raw_confidence',
            n_power_sim=100,  # Quick validation with fewer sims
            n_perm=args.n_perm,
            rng=rng,
        )

    # ========================================================================
    # Bayesian Assurance Analysis (if requested)
    # ========================================================================
    if args.bayesian_assurance:
        if args.assurance_all_features or args.assurance_top_k is not None or args.assurance_for_significant_only:
            # NEW: All-features mode
            print("\n" + "="*120)
            print("BAYESIAN ASSURANCE FOR ALL FEATURES")
            print("="*120)

            # Define targets and their feature lists
            targets_and_features = [
                ('raw_confidence', conf_feature_cols, conf_corr_df),
                ('is_correct', corr_feature_cols, corr_corr_df),
                ('calibration_error', cal_feature_cols, cal_corr_df),
            ]

            if args.power_method == "all":
                assurance_power_methods = ["spearman_perm", "mixed_model_lrt"]
                print("\n⚠ Note: all-features assurance does not currently implement per-feature cluster_robust assurance; running spearman_perm + mixed_model_lrt.")
            else:
                assurance_power_methods = [args.power_method]

            # Compute for each target and each requested assurance power method.
            for target_col, feature_cols_for_target, corr_df in targets_and_features:
                for pm in assurance_power_methods:
                    if pm == "spearman_perm" and not run_spearman_suite:
                        print(f"\n⚠ Skipping Spearman assurance for {target_col}: --effects-method excludes spearman_permutation.")
                        continue
                    if pm == "mixed_model_lrt" and not run_mixed_model_suite:
                        print(f"\n⚠ Skipping mixed_model_lrt assurance for {target_col}: --effects-method excludes mixed_model.")
                        continue
                    if pm == "cluster_robust":
                        print(f"\n⚠ Skipping cluster_robust assurance for {target_col}: not implemented for all-features mode.")
                        continue

                    print(f"\n{'='*120}")
                    print(f"TARGET: {target_col} | assurance power_method={pm}")
                    print(f"{'='*120}")

                    assurance_df = compute_assurance_for_all_features(
                        features_df,
                        feature_cols_for_target,
                        target_col,
                        n_target=args.assurance_n_target,
                        method=args.assurance_method,
                        n_bootstrap=args.n_bootstrap_assurance,
                        n_posterior_samples=args.n_posterior_samples,
                        n_power_sim_per_r=args.n_power_sim_per_r,
                        n_perm=args.n_perm,
                        alpha=args.alpha,
                        rng=rng,
                        top_k=args.assurance_top_k,
                        significant_only=args.assurance_for_significant_only,
                        correlation_results=corr_df,
                        power_method=pm,
                        student_ids=student_ids_array,
                        question_ids=question_ids_array,
                        rubric_ids=rubric_ids_array,
                        assurance_mode=args.assurance_mode,
                        assurance_beta_grid=assurance_beta_grid,
                        n_power_sim_inner=args.n_power_sim_inner,
                        assurance_winsorize_pct=args.assurance_winsorize_pct,
                        mixed_model_include_student_vc_glmm=args.effects_include_student_vc_glmm,
                        mixed_model_use_ranks=args.effects_use_ranks,
                        mixed_model_standardize_outcome=args.effects_standardize_outcome,
                        mixed_model_transform_bounded=args.effects_transform_bounded,
                    )

                    display_assurance_summary(assurance_df, method=args.assurance_method, top_k=10)

                    if not args.no_save_correlations:
                        suffix = pm
                        assurance_path = step3_tables_dir / f"assurance_{target_col}_{suffix}_{model_name}{method_tag}{file_suffix}.csv"
                        assurance_path.parent.mkdir(parents=True, exist_ok=True)
                        assurance_df.to_csv(assurance_path, index=False)
                        print(f"\n✓ Saved assurance results to: {assurance_path}")
                        if output_manager:
                            output_manager.register(
                                step="step3",
                                artifact="assurance",
                                path=assurance_path,
                                model=model_name,
                                method=args.method,
                                extra={"target_col": str(target_col), "power_method": str(pm)},
                            )
                            if output_manager.config.save_legacy:
                                legacy_assurance_path = legacy_analysis_dir / f"assurance_{target_col}_{suffix}_{model_name}_{csv_timestamp}.csv"
                                legacy_assurance_path.parent.mkdir(parents=True, exist_ok=True)
                                assurance_df.to_csv(legacy_assurance_path, index=False)

        else:
            # ORIGINAL: Single-feature mode (backward compatibility)
            if args.power_method == "all":
                assurance_methods_to_run = ["spearman_perm", "cluster_robust", "mixed_model_lrt"]
            else:
                assurance_methods_to_run = [args.power_method]

            for pm in assurance_methods_to_run:
                if pm == "spearman_perm" and not run_spearman_suite:
                    print("\n⚠ Skipping Spearman assurance: --effects-method excludes spearman_permutation.")
                    continue
                if pm in {"cluster_robust", "mixed_model_lrt"} and not run_mixed_model_suite:
                    print(f"\n⚠ Skipping {pm} assurance: --effects-method excludes mixed_model.")
                    continue

                display_bayesian_assurance(
                    features_df,
                    conf_feature_cols,
                    target_col='raw_confidence',
                    n_target=args.assurance_n_target,
                    method=args.assurance_method,
                    n_bootstrap=args.n_bootstrap_assurance,
                    n_posterior_samples=args.n_posterior_samples,
                    n_power_sim_per_r=args.n_power_sim_per_r,
                    n_perm=args.n_perm,
                    alpha=args.alpha,
                    rng=rng,
                    power_method=pm,
                    student_ids=student_ids_array,
                    question_ids=question_ids_array,
                    rubric_ids=rubric_ids_array,
                    assurance_mode=args.assurance_mode,
                    assurance_beta_grid=assurance_beta_grid,
                    n_power_sim_inner=args.n_power_sim_inner,
                    assurance_winsorize_pct=args.assurance_winsorize_pct,
                    mixed_model_include_student_vc_glmm=args.effects_include_student_vc_glmm,
                    mixed_model_use_ranks=args.effects_use_ranks,
                    mixed_model_standardize_outcome=args.effects_standardize_outcome,
                    mixed_model_transform_bounded=args.effects_transform_bounded,
                )

    # ========================================================================
    # Save Correlation Results
    # ========================================================================
    if (
        not args.no_save_correlations
        and not args.no_correlation_tables
        and not args.calibration_regression_only
    ):
        step3_tables_dir.mkdir(parents=True, exist_ok=True)

        if conf_corr_df is None or corr_corr_df is None:
            print("\n⚠ Correlation tables were skipped; no correlation CSV will be saved.")
            print("  Re-run without --no-correlation-tables to export correlation results.")
            return

        # Combine all correlations into one comprehensive DataFrame (union of features)
        all_features = set(conf_corr_df['feature']) | set(corr_corr_df['feature'])
        if cal_corr_df is not None:
            all_features |= set(cal_corr_df['feature'])
        combined_df = pd.DataFrame({'feature': sorted(all_features)})

        # Add confidence correlations
        conf_cols = conf_corr_df.set_index('feature')[['r', 'p_value', 'p_adjusted', 'power', 'significant', 'n', 'ci_low', 'ci_high']]
        conf_cols.columns = ['conf_r', 'conf_p', 'conf_p_adj', 'conf_power', 'conf_sig', 'n', 'conf_ci_low', 'conf_ci_high']
        combined_df = combined_df.merge(conf_cols, left_on='feature', right_index=True, how='left')

        # Add correctness correlations
        corr_cols = corr_corr_df.set_index('feature')[['r', 'p_value', 'p_adjusted', 'power', 'significant', 'n', 'ci_low', 'ci_high']]
        corr_cols.columns = ['corr_r', 'corr_p', 'corr_p_adj', 'corr_power', 'corr_sig', 'corr_n', 'corr_ci_low', 'corr_ci_high']
        combined_df = combined_df.merge(corr_cols, left_on='feature', right_index=True, how='left')

        # Add calibration error correlations (only if --include-bin-calibration-error)
        if cal_corr_df is not None:
            cal_cols = cal_corr_df.set_index('feature')[['r', 'p_value', 'p_adjusted', 'power', 'significant', 'n', 'ci_low', 'ci_high']]
            cal_cols.columns = ['cal_r', 'cal_p', 'cal_p_adj', 'cal_power', 'cal_sig', 'cal_n', 'cal_ci_low', 'cal_ci_high']
            combined_df = combined_df.merge(cal_cols, left_on='feature', right_index=True, how='left')

        # Sort by absolute confidence correlation (NaNs last)
        combined_df['abs_conf_r'] = combined_df['conf_r'].abs()
        combined_df = combined_df.sort_values('abs_conf_r', ascending=False, na_position='last').drop('abs_conf_r', axis=1)

        corr_path = step3_tables_dir / f"correlations_{model_name}{method_tag}{file_suffix}.csv"
        combined_df.to_csv(corr_path, index=False)
        print(f"\nCorrelations saved to: {corr_path}")
        if output_manager:
            output_manager.register(
                step="step3",
                artifact="correlations",
                path=corr_path,
                model=model_name,
                method=args.method,
            )
            if output_manager.config.save_legacy:
                legacy_corr_path = legacy_analysis_dir / f"correlations_{model_name}_{csv_timestamp}.csv"
                legacy_corr_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df.to_csv(legacy_corr_path, index=False)
        
        # Print power analysis summary
        print("\n" + "-" * 80)
        print("POWER ANALYSIS SUMMARY")
        print("-" * 80)
        power_targets = [('confidence', conf_corr_df), ('correctness', corr_corr_df)]
        if cal_corr_df is not None:
            power_targets.append(('calibration error', cal_corr_df))
        for target, df_target in power_targets:
            if df_target is None:
                continue
            print(f"\n{target.upper()}:")
            sig_features = df_target[df_target['significant']] if 'significant' in df_target.columns else df_target.iloc[0:0]
            print(f"  Significant features: {len(sig_features)}")

            power_series = df_target['power'] if 'power' in df_target.columns else pd.Series([], dtype=float)
            has_power = bool(np.isfinite(power_series.to_numpy(dtype=float)).any()) if len(power_series) else False
            if not has_power:
                print("  Power: not computed for this target/test (enable via --power-method spearman_perm or --power-method all).")
                continue

            underpowered = df_target[(~df_target['significant']) & (df_target['power'] < 0.8)]
            print(f"  Underpowered tests (may miss effects): {len(underpowered)}")
            if len(underpowered) > 0:
                for _, row in underpowered.iterrows():
                    print(f"    - {row['feature']}: r={row['r']:.3f}, power={row['power']:.2f}, need n≈{row['required_n']}")

    # ========================================================================
    # Generate Plots (if requested)
    # ========================================================================
    if args.plot:
        try:
            from grading_calibration.metrics.plots import (
                plot_feature_correlation_heatmap,
                plot_mixed_model_effects_forest,
                plot_mixed_model_effects_heatmap,
            )
            
            print("\n" + "-" * 80)
            print("GENERATING PLOTS")
            print("-" * 80)
            
            # Create correlation heatmap for top features
            top_features = conf_corr_df.head(20)['feature'].tolist()
            plot_suffix = f"{method_tag}{file_suffix}"

            plot_feature_correlation_heatmap(
                features_df,
                target_columns=['raw_confidence', 'is_correct', 'calibration_error'],
                feature_columns=top_features,
                name=f"correlation_heatmap_{model_name}{plot_suffix}",
                method_name=args.method,
                output_dir=step3_plots_dir,
            )
            print(f"Correlation heatmap generated for {args.method}")
            if output_manager:
                output_manager.register(
                    step="step3",
                    artifact="plot_correlation_heatmap",
                    path=step3_plots_dir / f"correlation_heatmap_{model_name}{plot_suffix}.png",
                    model=model_name,
                    method=args.method,
                )

            # Mixed-model effect plots (if available): these come from effects-method including mixed_model.
            effects_for_plot = None
            if "effects_df" in locals() and isinstance(locals().get("effects_df"), pd.DataFrame):
                if not effects_df.empty:
                    effects_for_plot = effects_df
            if effects_for_plot is None:
                analysis_dir = step3_tables_dir if output_manager else (project_root / "experiments" / "analysis" / args.method)
                candidates = sorted(analysis_dir.glob(f"mixed_model_effects_{model_name}_*.csv"))
                if candidates:
                    effects_for_plot = pd.read_csv(candidates[-1])

            if effects_for_plot is not None and not effects_for_plot.empty:
                plot_mixed_model_effects_heatmap(
                    effects_for_plot,
                    name=f"mixed_model_effects_heatmap_{model_name}{plot_suffix}",
                    method_name=args.method,
                    output_dir=step3_plots_dir,
                )
                plot_mixed_model_effects_forest(
                    effects_for_plot,
                    name_prefix=f"mixed_model_effects_forest_{model_name}{plot_suffix}",
                    method_name=args.method,
                    output_dir=step3_plots_dir,
                )
                print("Mixed-model effect plots generated")
                if output_manager:
                    output_manager.register(
                        step="step3",
                        artifact="plot_mixed_model_effects_heatmap",
                        path=step3_plots_dir / f"mixed_model_effects_heatmap_{model_name}{plot_suffix}.png",
                        model=model_name,
                        method=args.method,
                    )
            else:
                print(
                    "Note: No mixed-model effects found to plot. "
                    "Run with --effects-method mixed_model or --effects-method all to generate model-based plots."
                )
            
        except ImportError as e:
            print(f"Warning: Could not generate plots (missing dependency): {e}")
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")

    if output_manager:
        manifest_path = output_manager.write_manifest()
        if manifest_path:
            print(f"Manifest written to: {manifest_path}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
