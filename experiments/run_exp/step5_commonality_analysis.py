"""
Step 5: Commonality Analysis - Patterns in High/Low Confidence Groups

This script:
1. Loads predictions CSV and reconstructs data objects
2. Identifies high-confidence vs low-confidence prediction groups
3. Compares correct vs incorrect predictions
4. Finds distinguishing features using statistical tests (effect size, p-value)
5. Extracts common word patterns in each group
6. Generates human-readable analysis summary

Usage:
    python step5_commonality_analysis.py \
        --model qwen4b \
        --method self_reported \
        --high-threshold 0.8 \
        --low-threshold 0.5
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from scipy import stats

from grading_calibration.config import apply_config_to_parser, load_config
from grading_calibration.domain.models import PredictionResult, GradingInstance
from utils import find_csv_folder_from_root
from utils.statistics import (
    ttest_power,
    benjamini_hochberg,
    compute_icc_mixed_model,
    estimate_feature_effect_cluster_robust,
)
from utils.output_paths import add_output_args, build_output_manager
from utils.feature_cache import (
    compute_features_cache_key,
    build_features_cache_path,
    find_compatible_feature_caches,
    write_features_cache_metadata,
)
from analysis import FeatureExtractor, CommonalityAnalyzer
from grading_calibration.inference.model_aliases import MODEL_ALIASES as model_nicknames
from grading_calibration.pipelines.commonality_pipeline import (
    run_commonality_comparisons as _pipeline_run_commonality_comparisons,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Step 5: Commonality analysis for high/low confidence groups"
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
        "--high-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for high confidence group",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for low confidence group",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top distinguishing features to display",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=2000,
        help="Number of permutations for Monte Carlo permutation tests",
    )
    parser.add_argument(
        "--adaptive-perm",
        dest="adaptive_perm",
        action="store_true",
        help="Enable adaptive permutation refinement (default)",
    )
    parser.add_argument(
        "--no-adaptive-perm",
        dest="adaptive_perm",
        action="store_false",
        help="Disable adaptive permutation refinement",
    )
    parser.set_defaults(adaptive_perm=True)
    parser.add_argument(
        "--adaptive-perm-max",
        type=int,
        default=20000,
        help="Max permutations if adaptive refinement triggers",
    )
    parser.add_argument(
        "--adaptive-perm-band",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("LOW", "HIGH"),
        help="Refine if p is within [LOW*alpha, HIGH*alpha] (default: 0.5 2.0)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Number of bootstrap draws for effect size CIs",
    )
    parser.add_argument(
        "--bootstrap-method",
        choices=["bca", "percentile"],
        default="bca",
        help="Bootstrap CI method (default: bca)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for tests and CIs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for permutation/bootstrap procedures",
    )
    parser.add_argument(
        "--exact-power",
        action="store_true",
        help="Estimate power via bootstrap+permutation simulation (slow)",
    )
    parser.add_argument(
        "--power-sim",
        type=int,
        default=200,
        help="Simulations for exact power estimation (default: 200)",
    )
    parser.add_argument(
        "--power-perm",
        type=int,
        default=500,
        help="Permutations per simulation for exact power (default: 500)",
    )
    parser.add_argument(
        "--cluster-robust-sensitivity",
        dest="cluster_robust_sensitivity",
        action="store_true",
        help="Force cluster-robust sensitivity checks",
    )
    parser.add_argument(
        "--no-cluster-robust-sensitivity",
        dest="cluster_robust_sensitivity",
        action="store_false",
        help="Disable cluster-robust sensitivity checks",
    )
    parser.set_defaults(cluster_robust_sensitivity=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save analysis results to CSV",
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

def load_path_by_index(
    idx: int,
    folder: Path | str,
    prefix: str,
    dataset: str = "ricechem",
    *,
    filter_by_dataset: bool = True,
    allow_exact: bool = False,
) -> Path:
    """Load CSV path by index (0 = most recent), filtered by dataset tag."""
    folder = Path(folder)
    paths: list[Path] = []
    if allow_exact:
        exact_path = folder / f"{prefix}.csv"
        if exact_path.exists():
            paths.append(exact_path)
    paths.extend(folder.glob(f"{prefix}_*.csv"))
    paths = sorted(set(paths))
    dataset = dataset.lower()
    if filter_by_dataset:
        other_datasets = {"ricechem", "scientsbank", "beetle"} - {dataset}
        if dataset == "ricechem":
            paths = [p for p in paths if not any(f"_{d}_" in p.name for d in other_datasets)]
        else:
            paths = [p for p in paths if f"_{dataset}_" in p.name]
    if not paths:
        raise FileNotFoundError(f"No predictions CSVs found for dataset={dataset} in {folder}")
    new_idx = len(paths) - 1 - idx
    return paths[new_idx]


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
            step="step5",
            artifact=artifact,
            path=out_path,
            model=model,
            method=method,
        )
    if output_manager and output_manager.config.save_legacy and legacy_path is not None:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(legacy_path, index=False)


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


def two_sample_permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    alpha: float,
    adaptive_perm: bool,
    adaptive_perm_max: int,
    adaptive_band: tuple[float, float],
) -> float:
    """
    Monte Carlo permutation test for difference in means (two-sided).

    Returns a Monte Carlo p-value; this is not an exact test unless all
    permutations are enumerated.
    """
    def _perm_p_value(n_draws: int) -> float:
        if n_draws <= 0:
            return np.nan
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        if n1 == 0 or len(group2) == 0:
            return np.nan
        obs_diff = float(np.mean(group1) - np.mean(group2))
        perm_diffs = np.empty(n_draws, dtype=float)
        for i in range(n_draws):
            perm = rng.permutation(combined)
            perm_diffs[i] = float(np.mean(perm[:n1]) - np.mean(perm[n1:]))
        p_val = (np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1) / (n_draws + 1)
        return float(p_val)

    p_value = _perm_p_value(n_perm)
    if adaptive_perm and np.isfinite(p_value):
        band_low, band_high = adaptive_band
        low = band_low * alpha
        high = band_high * alpha
        if low <= p_value <= high and adaptive_perm_max > n_perm:
            p_value = _perm_p_value(adaptive_perm_max)
    return float(p_value)


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Robust effect size: Cliff's delta (rank-biserial correlation).
    Range: [-1, 1], where positive values indicate group1 > group2.
    """
    n1 = len(group1)
    n2 = len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    delta = (2.0 * float(u_stat)) / (float(n1) * float(n2)) - 1.0
    return float(delta)


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude using common thresholds."""
    if not np.isfinite(delta):
        return "unknown"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _compute_bca_ci_two_sample(
    boot_stats: np.ndarray,
    observed_stat: float,
    x: np.ndarray,
    y: np.ndarray,
    stat_func,
    alpha: float,
) -> tuple[float, float]:
    """
    BCa CI for two-sample statistics using leave-one-out over both samples.
    """
    n_boot = len(boot_stats)
    n1 = len(x)
    n2 = len(y)
    if n_boot == 0 or n1 == 0 or n2 == 0:
        return np.nan, np.nan

    p0 = np.sum(boot_stats < observed_stat) / n_boot
    if p0 == 0:
        p0 = 0.5 / n_boot
    elif p0 == 1:
        p0 = 1 - 0.5 / n_boot
    z0 = stats.norm.ppf(p0)

    jack_stats = np.empty(n1 + n2, dtype=float)
    for i in range(n1):
        jack_stats[i] = stat_func(np.delete(x, i), y)
    for j in range(n2):
        jack_stats[n1 + j] = stat_func(x, np.delete(y, j))

    jack_mean = float(np.mean(jack_stats))
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = 0.0 if denom == 0 else float(num / denom)

    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
    p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))
    p_low = float(np.clip(p_low, 0, 1))
    p_high = float(np.clip(p_high, 0, 1))

    ci_low = float(np.percentile(boot_stats, 100 * p_low))
    ci_high = float(np.percentile(boot_stats, 100 * p_high))
    return ci_low, ci_high


def bootstrap_cliffs_delta_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    method: str,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Compute Cliff's delta with bootstrap CI (BCa or percentile).
    """
    delta_obs = cliffs_delta(group1, group2)
    if n_boot <= 0 or not np.isfinite(delta_obs):
        return delta_obs, np.nan, np.nan

    boot_stats = np.empty(n_boot, dtype=float)
    n1 = len(group1)
    n2 = len(group2)
    for i in range(n_boot):
        boot_g1 = rng.choice(group1, size=n1, replace=True)
        boot_g2 = rng.choice(group2, size=n2, replace=True)
        boot_stats[i] = cliffs_delta(boot_g1, boot_g2)

    if method == "bca":
        ci_low, ci_high = _compute_bca_ci_two_sample(
            boot_stats, delta_obs, group1, group2, cliffs_delta, alpha
        )
    else:
        ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return delta_obs, ci_low, ci_high


def _approx_cohens_d_from_delta(delta: float) -> float:
    """
    Approximate Cohen's d from a rank-biserial effect size.
    Used only for rough power estimates.
    """
    if not np.isfinite(delta) or abs(delta) >= 1:
        return np.nan
    denom = np.sqrt(1 - delta * delta)
    if denom <= 0:
        return np.nan
    return float(2 * delta / denom)


def estimate_power_permutation(
    group1: np.ndarray,
    group2: np.ndarray,
    *,
    n_sim: int,
    n_perm: int,
    alpha: float,
    rng: np.random.Generator,
    adaptive_perm: bool,
    adaptive_perm_max: int,
    adaptive_band: tuple[float, float],
) -> float:
    """
    Simulation-based power estimate using bootstrap resampling + permutation tests.
    This is a Monte Carlo approximation and can be slow.
    """
    if n_sim <= 0 or n_perm <= 0:
        return np.nan
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    n_sig = 0
    for _ in range(n_sim):
        boot_g1 = rng.choice(group1, size=n1, replace=True)
        boot_g2 = rng.choice(group2, size=n2, replace=True)
        p_val = two_sample_permutation_test(
            boot_g1,
            boot_g2,
            n_perm=n_perm,
            rng=rng,
            alpha=alpha,
            adaptive_perm=adaptive_perm,
            adaptive_perm_max=adaptive_perm_max,
            adaptive_band=adaptive_band,
        )
        if np.isfinite(p_val) and p_val < alpha:
            n_sig += 1
    return float(n_sig) / float(n_sim)


def compare_groups(features_df: pd.DataFrame, group1_mask: np.ndarray, 
                   group2_mask: np.ndarray, group1_name: str, group2_name: str,
                   *, n_perm: int, n_boot: int, alpha: float,
                   bootstrap_method: str, rng: np.random.Generator,
                   adaptive_perm: bool, adaptive_perm_max: int,
                   adaptive_band: tuple[float, float],
                   exact_power: bool, power_sim: int, power_perm: int,
                   run_cluster_robust: bool,
                   student_ids: np.ndarray | None,
                   question_ids: np.ndarray | None,
                   rubric_ids: np.ndarray | None) -> pd.DataFrame:
    """
    Compare two groups across all numerical features with Monte Carlo permutation tests,
    robust effect sizes (Cliff's delta), bootstrap CIs, and FDR correction.
    
    Returns DataFrame with feature, mean_group1, mean_group2, effect_size, effect_size_ci,
    p_value, p_adjusted, power, effect_interpretation.
    """
    results = []
    group1_mask = np.asarray(group1_mask, dtype=bool)
    group2_mask = np.asarray(group2_mask, dtype=bool)
    
    # Get numerical columns
    numerical_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target/identity columns
    exclude_cols = ['raw_confidence', 'calibrated_confidence', 'confidence_delta',
                    'is_correct', 'prediction', 'ground_truth', 'is_high_confidence',
                    'is_medium_confidence', 'is_low_confidence', 'confidence_bin',
                    'is_false_positive', 'is_false_negative']
    feature_cols = [c for c in numerical_cols if c not in exclude_cols]

    cluster_mask = group1_mask | group2_mask
    cluster_group = None
    cluster_students = None
    cluster_questions = None
    cluster_rubrics = None
    if run_cluster_robust and student_ids is not None:
        cluster_group = group1_mask[cluster_mask].astype(int)
        cluster_students = student_ids[cluster_mask]
        cluster_questions = question_ids[cluster_mask] if question_ids is not None else None
        cluster_rubrics = rubric_ids[cluster_mask] if rubric_ids is not None else None

    for feature in feature_cols:
        group1_vals = features_df.loc[group1_mask, feature].dropna().values.astype(float)
        group2_vals = features_df.loc[group2_mask, feature].dropna().values.astype(float)
        
        n1, n2 = len(group1_vals), len(group2_vals)
        
        if n1 < 2 or n2 < 2:
            continue
        
        # Robust effect size (Cliff's delta) with bootstrap CI
        effect_size, ci_low, ci_high = bootstrap_cliffs_delta_ci(
            group1_vals,
            group2_vals,
            n_boot=n_boot,
            alpha=alpha,
            method=bootstrap_method,
            rng=rng,
        )

        # Monte Carlo permutation p-value (difference in means)
        p_value = two_sample_permutation_test(
            group1_vals,
            group2_vals,
            n_perm=n_perm,
            rng=rng,
            alpha=alpha,
            adaptive_perm=adaptive_perm,
            adaptive_perm_max=adaptive_perm_max,
            adaptive_band=adaptive_band,
        )

        # Approximate power via Cohen's d conversion (heuristic)
        approx_d = _approx_cohens_d_from_delta(effect_size)
        power = ttest_power(abs(approx_d), n1, n2, alpha) if np.isfinite(approx_d) else np.nan
        power_exact = np.nan
        if exact_power:
            power_exact = estimate_power_permutation(
                group1_vals,
                group2_vals,
                n_sim=power_sim,
                n_perm=power_perm,
                alpha=alpha,
                rng=rng,
                adaptive_perm=adaptive_perm,
                adaptive_perm_max=adaptive_perm_max,
                adaptive_band=adaptive_band,
            )

        cluster_beta = np.nan
        cluster_se = np.nan
        cluster_p = np.nan
        cluster_ci_low = np.nan
        cluster_ci_high = np.nan
        cluster_n_obs = np.nan
        if run_cluster_robust and cluster_group is not None:
            try:
                feature_vals = (
                    features_df.loc[cluster_mask, feature]
                    .to_numpy(dtype=float)
                )
                res_cr = estimate_feature_effect_cluster_robust(
                    feature=cluster_group,
                    outcome=feature_vals,
                    student_ids=cluster_students,
                    question_ids=cluster_questions,
                    rubric_ids=cluster_rubrics,
                    use_ranks=True,
                    standardize=True,
                    alpha=alpha,
                )
                cluster_beta = float(res_cr.get("beta", np.nan))
                cluster_se = float(res_cr.get("se", np.nan))
                cluster_p = float(res_cr.get("p_value", np.nan))
                cluster_ci_low = float(res_cr.get("ci_low", np.nan))
                cluster_ci_high = float(res_cr.get("ci_high", np.nan))
                cluster_n_obs = float(res_cr.get("n_obs", np.nan))
            except Exception:
                pass
        
        results.append({
            'feature': feature,
            f'mean_{group1_name}': group1_vals.mean(),
            f'mean_{group2_name}': group2_vals.mean(),
            'effect_size': effect_size,
            'effect_size_ci_low': ci_low,
            'effect_size_ci_high': ci_high,
            'effect_interpretation': interpret_cliffs_delta(effect_size),
            'abs_effect_size': abs(effect_size),
            'p_value': p_value,
            'power': power,
            'power_exact': power_exact,
            'n1': n1,
            'n2': n2,
            'cluster_beta': cluster_beta,
            'cluster_se': cluster_se,
            'cluster_p_value': cluster_p,
            'cluster_ci_low': cluster_ci_low,
            'cluster_ci_high': cluster_ci_high,
            'cluster_n_obs': cluster_n_obs,
        })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Apply FDR correction
        p_adjusted, significant = benjamini_hochberg(results_df['p_value'].values, alpha)
        results_df['p_adjusted'] = p_adjusted
        results_df['significant'] = significant
        results_df = results_df.sort_values('abs_effect_size', ascending=False)
    
    return results_df


def print_comparison_table(comparison_df: pd.DataFrame, title: str, 
                           group1_name: str, group2_name: str, top_k: int = 10):
    """Print a formatted comparison table with effect size interpretation."""
    print(f"\n{title}")
    print("-" * 140)
    print(
        f"{'Feature':<30} {group1_name:>10} {group2_name:>10} "
        f"{'Delta':>8} {'CI_low':>8} {'CI_high':>8} "
        f"{'Interp':>10} {'p-adj':>10} {'Power':>7} {'Sig':>5}"
    )
    print("-" * 140)
    
    for _, row in comparison_df.head(top_k).iterrows():
        effect_str = f"{row['effect_size']:+.3f}"
        ci_low = row.get("effect_size_ci_low", np.nan)
        ci_high = row.get("effect_size_ci_high", np.nan)
        ci_low_str = f"{ci_low:+.3f}" if np.isfinite(ci_low) else "nan"
        ci_high_str = f"{ci_high:+.3f}" if np.isfinite(ci_high) else "nan"
        p_adj_val = row.get("p_adjusted_global", row.get("p_adjusted", row["p_value"]))
        if not np.isfinite(p_adj_val):
            p_adj_str = "nan"
        else:
            p_adj_str = f"{p_adj_val:.4f}" if p_adj_val >= 0.0001 else "<.0001"
        interp = row.get("effect_interpretation", interpret_cliffs_delta(row["effect_size"]))
        power_val = row.get("power_exact", np.nan)
        if not np.isfinite(power_val):
            power_val = row.get("power", np.nan)
        power_str = f"{power_val:.2f}" if np.isfinite(power_val) else "nan"
        sig = "***" if row.get("significant_global", row.get("significant", False)) else ""
        
        print(f"{row['feature']:<30} "
              f"{row[f'mean_{group1_name}']:>10.3f} "
              f"{row[f'mean_{group2_name}']:>10.3f} "
              f"{effect_str:>8} "
              f"{ci_low_str:>8} "
              f"{ci_high_str:>8} "
              f"{interp:>10} "
              f"{p_adj_str:>10} "
              f"{power_str:>7} "
              f"{sig:>5}")

    if "power" in comparison_df.columns:
        print("Note: Power values are approximate (delta->d conversion) unless --exact-power is set.")
    
    # Summary
    if 'significant_global' in comparison_df.columns:
        n_sig = comparison_df['significant_global'].sum()
    elif 'significant' in comparison_df.columns:
        n_sig = comparison_df['significant'].sum()
    else:
        n_sig = 0
    if n_sig is not None:
        n_total = len(comparison_df)
        print("-" * 140)
        print(f"Significant after FDR correction: {n_sig}/{n_total}")
        
        # Highlight practically significant findings (significant AND >= medium effect)
        sig_col = "significant_global" if "significant_global" in comparison_df.columns else "significant"
        practical = comparison_df[
            comparison_df.get(sig_col, False) & (comparison_df['abs_effect_size'] >= 0.33)
        ]
        if len(practical) > 0:
            print(f"Practically significant (|delta|>=0.33 and FDR sig): {len(practical)} features")


def print_cluster_robust_discrepancies(
    comparison_df: pd.DataFrame,
    label: str,
    alpha: float,
    top_k: int = 10,
) -> None:
    """Print discrepancies between permutation and cluster-robust sensitivity checks."""
    if comparison_df.empty or "cluster_p_value" not in comparison_df.columns:
        return
    if "significant_global" in comparison_df.columns:
        perm_sig = comparison_df["significant_global"]
    elif "significant" in comparison_df.columns:
        perm_sig = comparison_df["significant"]
    else:
        return
    cluster_sig = comparison_df["cluster_p_value"] < alpha
    mismatch = comparison_df[perm_sig != cluster_sig]
    print(f"\nCluster-robust sensitivity ({label}): {len(mismatch)} discrepancies at alpha={alpha}")
    if len(mismatch) == 0:
        return
    cols = [
        "feature",
        "p_value",
        "p_adjusted_global",
        "p_adjusted",
        "cluster_p_value",
        "effect_size",
    ]
    show_cols = [c for c in cols if c in mismatch.columns]
    print(mismatch.sort_values("cluster_p_value").head(top_k)[show_cols].to_string(index=False))


def extract_common_words(texts: list, top_k: int = 20) -> list:
    """Extract most common words from a list of texts."""
    from collections import Counter
    
    # Common stop words to exclude
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
                  'about', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'between', 'under', 'again', 'further', 'then', 'once',
                  'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                  'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
                  'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                  'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
                  'why', 'how', 'all', 'each', 'every', 'any', 'some', 'no', 'yes'}
    
    word_counts = Counter()
    for text in texts:
        if not text:
            continue
        words = text.lower().split()
        words = [w.strip('.,!?;:()[]{}"\'-') for w in words]
        words = [w for w in words if w and len(w) > 2 and w not in stop_words]
        word_counts.update(words)
    
    return word_counts.most_common(top_k)


def main():
    args = parse_args()
    args.model = model_nicknames.get(args.model.lower(), args.model.lower())
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    rng = np.random.default_rng(args.seed)
    output_manager = None
    if args.output_root:
        if not args.run_id:
            raise ValueError("--run-id is required when using --output-root.")
        output_manager = build_output_manager(
            output_root=args.output_root,
            dataset=args.dataset,
            run_id=args.run_id,
            save_legacy=args.save_legacy,
            write_manifest=args.write_manifest,
        )
    if args.low_threshold >= 0.2:
        print("Warning: --low-threshold >= 0.2; proceeding, but 0.2 is the recommended upper bound for low confidence.")
    if args.low_threshold >= args.high_threshold:
        raise ValueError("--low-threshold must be lower than --high-threshold.")

    print("=" * 80)
    print("STEP 5: COMMONALITY ANALYSIS")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)

    if args.predictions_dir:
        directory = Path(args.predictions_dir) / Path(args.method)
        filter_by_dataset = True
        allow_exact = False
    elif output_manager:
        directory = output_manager.step_dir("step1", "tables")
        filter_by_dataset = False
        allow_exact = True
    else:
        directory = find_csv_folder_from_root("grading_calibration") / Path(args.method)
        filter_by_dataset = True
        allow_exact = False
    csv_path = load_path_by_index(
        args.index,
        directory,
        f"{args.prefix}_{model_name}_{args.method}",
        args.dataset,
        filter_by_dataset=filter_by_dataset,
        allow_exact=allow_exact,
    )
    
    instances, predictions, df = load_data_from_csv(csv_path, smoke_n=args.smoke_n)
    csv_timestamp = output_manager.config.run_id if output_manager else csv_path.stem.split("_")[-1]
    file_suffix = "" if output_manager else f"_{csv_timestamp}"
    legacy_analysis_dir = project_root / "experiments" / "analysis" / args.method
    step5_tables_dir = output_manager.step_dir("step5", "tables") if output_manager else legacy_analysis_dir

    print(f"Loaded {len(instances)} instances from {csv_path}")

    # ========================================================================
    # Extract Features
    # ========================================================================
    print("\nExtracting features...")
    features_df: pd.DataFrame | None = None
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
                        step="step5",
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
                    step="step5",
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

    # ICC diagnostics (descriptive)
    print("\n" + "=" * 80)
    print("CLUSTERING DIAGNOSTICS (ICC; DESCRIPTIVE)")
    print("=" * 80)
    icc_flag = False
    if {"student_id", "question_id", "rubric_id"}.issubset(features_df.columns):
        try:
            icc_result = compute_icc_mixed_model(
                values=features_df["raw_confidence"].to_numpy(dtype=float),
                student_ids=features_df["student_id"].to_numpy(dtype=str),
                question_ids=features_df["question_id"].to_numpy(dtype=str),
                rubric_ids=features_df["rubric_id"].to_numpy(dtype=str),
            )
            icc_student = icc_result.get("icc_student", np.nan)
            icc_question = icc_result.get("icc_question", np.nan)
            print(f"ICC by student:   {icc_student:.4f}")
            print(f"ICC by question:  {icc_question:.4f}")
            if (np.isfinite(icc_student) and icc_student > 0.05) or (
                np.isfinite(icc_question) and icc_question > 0.05
            ):
                print("WARNING: Substantial clustering detected (ICC > 0.05).")
                print("Permutation tests may have inflated Type I error.")
                icc_flag = True
        except Exception as exc:
            print(f"Warning: ICC computation failed: {exc}")
    else:
        print("Skipped: missing student_id/question_id/rubric_id in features_df.")

    if icc_flag:
        if args.cluster_robust_sensitivity is False:
            print("WARNING: ICC > 0.05; cluster-robust sensitivity is mandatory. Ignoring --no-cluster-robust-sensitivity.")
        run_cluster_robust = True
    else:
        if args.cluster_robust_sensitivity is None:
            run_cluster_robust = False
        else:
            run_cluster_robust = bool(args.cluster_robust_sensitivity)
    if run_cluster_robust:
        print("Cluster-robust sensitivity checks: ENABLED")

    # Create instance lookup for text access
    instance_dict = {inst.id: inst for inst in instances}
    pred_dict = {pred.instance_id: pred for pred in predictions}

    student_ids = (
        features_df["student_id"].to_numpy(dtype=str)
        if "student_id" in features_df.columns
        else None
    )
    question_ids = (
        features_df["question_id"].to_numpy(dtype=str)
        if "question_id" in features_df.columns
        else None
    )
    rubric_ids = (
        features_df["rubric_id"].to_numpy(dtype=str)
        if "rubric_id" in features_df.columns
        else None
    )

    # ========================================================================
    # High/low confidence and correct/incorrect comparisons
    # ========================================================================
    conf_comparison, correctness_comparison, group_info = _pipeline_run_commonality_comparisons(
        features_df=features_df,
        args=args,
        rng=rng,
        compare_groups=compare_groups,
        run_cluster_robust=run_cluster_robust,
        student_ids=student_ids,
        question_ids=question_ids,
        rubric_ids=rubric_ids,
    )
    high_conf_mask = group_info["high_conf_mask"]
    low_conf_mask = group_info["low_conf_mask"]
    correct_mask = group_info["correct_mask"]
    incorrect_mask = group_info["incorrect_mask"]
    n_high = group_info["n_high"]
    n_low = group_info["n_low"]
    high_acc = group_info["high_acc"]

    # ========================================================================
    # Report comparisons with global-adjusted p-values (if available)
    # ========================================================================
    if not conf_comparison.empty:
        print_comparison_table(
            conf_comparison,
            "Distinguishing Features (High vs Low Confidence)",
            "high",
            "low",
            args.top_k,
        )
        if run_cluster_robust:
            print_cluster_robust_discrepancies(
                conf_comparison, "high_vs_low", args.alpha, top_k=args.top_k
            )
    if not correctness_comparison.empty:
        print_comparison_table(
            correctness_comparison,
            "Distinguishing Features (Correct vs Incorrect)",
            "correct",
            "incorrect",
            args.top_k,
        )
        if run_cluster_robust:
            print_cluster_robust_discrepancies(
                correctness_comparison, "correct_vs_incorrect", args.alpha, top_k=args.top_k
            )

    # ========================================================================
    # Common Words Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMMON WORDS IN EACH GROUP")
    print("=" * 80)

    # Get instance IDs for each group
    high_conf_ids = features_df.index[high_conf_mask].tolist()
    low_conf_ids = features_df.index[low_conf_mask].tolist()
    correct_ids = features_df.index[correct_mask].tolist()
    incorrect_ids = features_df.index[incorrect_mask].tolist()

    # Extract student answers for each group
    high_conf_answers = [instance_dict[id].student_answer for id in high_conf_ids if id in instance_dict]
    low_conf_answers = [instance_dict[id].student_answer for id in low_conf_ids if id in instance_dict]
    
    print("\nMost common words in HIGH confidence answers:")
    high_words = extract_common_words(high_conf_answers, top_k=15)
    for word, count in high_words[:10]:
        print(f"  {word}: {count}")

    print("\nMost common words in LOW confidence answers:")
    low_words = extract_common_words(low_conf_answers, top_k=15)
    for word, count in low_words[:10]:
        print(f"  {word}: {count}")

    # ========================================================================
    # Summary and Recommendations
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    print("\nKey Findings:")
    
    # Analyze high/low confidence comparison
    if not conf_comparison.empty:
        top_features = conf_comparison.head(3)
        print("\n1. High vs Low Confidence distinguishing features:")
        for _, row in top_features.iterrows():
            direction = "higher" if row['effect_size'] > 0 else "lower"
            print(f"   - {row['feature']}: High confidence has {direction} values (delta={row['effect_size']:.2f})")

    # Analyze correct/incorrect comparison
    if not correctness_comparison.empty:
        top_features = correctness_comparison.head(3)
        print("\n2. Correct vs Incorrect distinguishing features:")
        for _, row in top_features.iterrows():
            direction = "higher" if row['effect_size'] > 0 else "lower"
            print(f"   - {row['feature']}: Correct predictions have {direction} values (delta={row['effect_size']:.2f})")

    # Calibration insight
    if n_high > 5 and n_low > 5:
        expected_high = args.high_threshold
        if high_acc < expected_high - 0.1:
            print(f"\n⚠ OVERCONFIDENCE DETECTED: High confidence accuracy ({high_acc:.3f}) is below threshold ({expected_high})")
        elif high_acc > expected_high:
            print(f"\n✓ High confidence is well-calibrated or slightly underconfident")

    # ========================================================================
    # Save Results
    # ========================================================================
    if not args.no_save:
        if not conf_comparison.empty:
            conf_prefix = (
                f"commonality_confidence_{model_name}_{args.method}"
                if output_manager
                else f"commonality_confidence_{model_name}"
            )
            conf_path = step5_tables_dir / f"{conf_prefix}{file_suffix}.csv"
            legacy_path = legacy_analysis_dir / f"commonality_confidence_{model_name}_{csv_timestamp}.csv"
            _save_table(
                conf_comparison,
                conf_path,
                output_manager=output_manager,
                artifact="commonality_confidence",
                model=model_name,
                method=args.method,
                legacy_path=legacy_path,
            )
            print(f"\nConfidence comparison saved to: {conf_path}")

        if not correctness_comparison.empty:
            corr_prefix = (
                f"commonality_correctness_{model_name}_{args.method}"
                if output_manager
                else f"commonality_correctness_{model_name}"
            )
            corr_path = step5_tables_dir / f"{corr_prefix}{file_suffix}.csv"
            legacy_path = legacy_analysis_dir / f"commonality_correctness_{model_name}_{csv_timestamp}.csv"
            _save_table(
                correctness_comparison,
                corr_path,
                output_manager=output_manager,
                artifact="commonality_correctness",
                model=model_name,
                method=args.method,
                legacy_path=legacy_path,
            )
            print(f"Correctness comparison saved to: {corr_path}")

    if output_manager:
        manifest_path = output_manager.write_manifest()
        if manifest_path:
            print(f"\nManifest saved to: {manifest_path}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
