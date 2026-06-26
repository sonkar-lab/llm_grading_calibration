"""
Step 6: Cross-model and cross-method statistical comparisons.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
print(f"Adding project root to sys.path: {project_root}")
sys.path.insert(0, str(project_root))

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from grading_calibration.config import apply_config_to_parser, load_config
from utils import find_csv_folder_from_root
from utils.output_paths import add_output_args, build_output_manager
from grading_calibration.metrics.cross_comparison import (
    MODEL_NICKNAMES,
    load_all_predictions,
    compute_ece_mce,
    compute_auc,
    compute_brier_mean,
    compute_accuracy,
    accuracy_at_coverage,
    coverage_at_risk,
    paired_cliffs_delta,
    dz_effect,
    by_fdr,
    paired_sign_flip_permutation,
    paired_group_metric_permutation,
    bootstrap_differences,
    bootstrap_group_metric_diff,
    bca_ci_paired_diffs,
    bca_ci_group_diff,
    art_interaction_test,
    compute_i2,
    practical_threshold,
    warn_sample_size,
    missing_action,
    sensitivity_bounds,
)
from grading_calibration.metrics.comparison_plots import (
    plot_metrics_heatmap,
    plot_forest_plot,
    plot_cd_diagram,
    plot_reliability_diagram,
)
from grading_calibration.pipelines.cross_comparison_pipeline import (
    run_cross_comparisons as _pipeline_run_cross_comparisons,
)
from utils.statistics import (
    compute_icc_mixed_model,
    fit_lme4_lrt_from_formula,
    fit_calibration_curve_glmm,
    _extract_coefficient_from_glmm,
    _compute_bca_ci_from_jackknife,
)


PER_INSTANCE_METRICS = {"is_correct", "squared_error", "raw_confidence"}
GROUP_METRICS = {
    "ece",
    "mce",
    "auc_roc",
    "brier_mean",
    "cov_at_5pct_risk",
    "cov_at_10pct_risk",
    "acc_at_50",
    "acc_at_80",
}
LOWER_BETTER = {"ece", "mce", "brier_mean", "squared_error"}


def _save_table(
    df: pd.DataFrame,
    out_path: Path,
    *,
    output_manager=None,
    artifact: str | None = None,
    legacy_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if output_manager and artifact:
        output_manager.register(step="step6", artifact=artifact, path=out_path)
    if output_manager and output_manager.config.save_legacy and legacy_path is not None:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(legacy_path, index=False)


def _save_text(
    content: str,
    out_path: Path,
    *,
    output_manager=None,
    artifact: str | None = None,
    legacy_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    if output_manager and artifact:
        output_manager.register(step="step6", artifact=artifact, path=out_path)
    if output_manager and output_manager.config.save_legacy and legacy_path is not None:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 6: Cross-model/method comparison"
    )
    parser.add_argument("--models", nargs="+", default=["gptoss20b", "qwen4b", "llama8b"])
    parser.add_argument("--methods", nargs="+", default=["self_reported", "self_consistency", "token_probability", "perturbation"])
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["is_correct", "squared_error", "raw_confidence", "ece", "mce", "auc_roc", "brier_mean", "cov_at_5pct_risk", "cov_at_10pct_risk", "acc_at_50", "acc_at_80"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ricechem",
        choices=["ricechem", "scientsbank", "beetle"],
        help="Dataset name for selecting prediction CSVs (default: ricechem).",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Override predictions directory (legacy layout). "
             "Use this to load old predictions while still saving organized outputs.",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=None,
        help="Limit to first N instances for a smoke test run.",
    )
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--ci-method",
        choices=["bca", "percentile"],
        default="bca",
        help="Bootstrap CI method for paired diffs (default: bca).",
    )
    parser.add_argument(
        "--max-jackknife",
        type=int,
        default=1000,
        help="Max sample size for leave-one-out BCa; fallback to percentile above this.",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--binning-strategies",
        nargs="+",
        default=["equal_width"],
        choices=["equal_width", "equal_frequency"],
    )
    parser.add_argument(
        "--interaction-method",
        choices=["artc", "art"],
        default="artc",
        help="Interaction test method (default: artc).",
    )
    parser.add_argument(
        "--skip-interaction-tests",
        action="store_true",
        help="Skip interaction tests (including calibration regression interactions).",
    )
    parser.add_argument("--plot-all", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--print-results", action="store_true", help="Print result tables to stdout.")
    parser.add_argument("--print-summary", action="store_true", help="Alias for --print-results.")
    parser.add_argument("--print-max-rows", type=int, default=20, help="Max rows to print per table.")
    parser.add_argument("--best-top-k", type=int, default=5, help="Top-K per metric/binning for best summaries.")
    parser.add_argument("--gate-posthoc", action="store_true")
    parser.add_argument("--run-confirmatory", action="store_true")
    parser.add_argument("--force-progress", action="store_true", help="Force progress bars even when output is redirected.")
    parser.add_argument(
        "--confirmatory-summary-scope",
        choices=["none", "best", "full"],
        default="none",
        help="Run confirmatory re-tests for summary-selected pairs (best) or full family.",
    )
    parser.add_argument("--confirmatory-perm", type=int, default=10000)
    parser.add_argument("--confirmatory-alpha", type=float, default=0.017)
    parser.add_argument(
        "--calibration-regression",
        action="store_true",
        help="Run calibration regression diagnostics (optional).",
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
        help="Bootstrap samples for nonlinear calibration regression summaries (default: 0).",
    )
    parser.add_argument(
        "--calibration-regression-ci-method",
        choices=["percentile", "bca"],
        default="percentile",
        help="CI method for nonlinear calibration regression summaries (default: percentile).",
    )
    parser.add_argument(
        "--calibration-regression-jackknife-max",
        type=int,
        default=200,
        help="Max jackknife units for BCa in calibration regression (default: 200).",
    )
    parser.add_argument(
        "--calibration-regression-fallback",
        choices=["linear", "skip"],
        default="linear",
        help="Fallback when nonlinear calibration regression fails (default: linear).",
    )
    parser.add_argument(
        "--calibration-regression-max-models",
        type=int,
        default=None,
        help="Optional cap on number of models for calibration regression.",
    )
    parser.add_argument(
        "--calibration-regression-max-methods",
        type=int,
        default=None,
        help="Optional cap on number of methods for calibration regression.",
    )
    parser.add_argument(
        "--calibration-regression-min-n",
        type=int,
        default=30,
        help="Minimum observations per model-method cell for calibration regression (default: 30).",
    )
    parser.add_argument(
        "--calibration-regression-n-grid",
        type=int,
        default=50,
        help="Grid size for calibration regression curves (default: 50).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file with argparse destination overrides.",
    )

    add_output_args(parser)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
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
    if args.print_summary:
        args.print_results = True
    return args


def _ensure_instance_id_col(df: pd.DataFrame) -> pd.DataFrame:
    if "instance_id" in df.columns:
        return df
    if df.index.name == "instance_id":
        return df.reset_index()
    if "index" in df.columns:
        return df.rename(columns={"index": "instance_id"})
    raise KeyError(f"instance_id not found; columns={list(df.columns)} index_name={df.index.name}")


def _align_by_instance(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_a = _ensure_instance_id_col(df_a)
    df_b = _ensure_instance_id_col(df_b)
    a = df_a.set_index("instance_id")
    b = df_b.set_index("instance_id")
    common = a.index.intersection(b.index)
    a = a.loc[common].sort_index()
    b = b.loc[common].sort_index()
    return a, b


def _align_multiple(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    indexed = [_ensure_instance_id_col(df).set_index("instance_id") for df in dfs]
    common = indexed[0].index
    for df in indexed[1:]:
        common = common.intersection(df.index)
    return [df.loc[common].sort_index() for df in indexed]


def _metric_value(metric: str, conf: np.ndarray, labels: np.ndarray, *, n_bins: int, binning: str) -> float:
    if metric == "ece":
        return compute_ece_mce(conf, labels, n_bins=n_bins, strategy=binning)[0]
    if metric == "mce":
        return compute_ece_mce(conf, labels, n_bins=n_bins, strategy=binning)[1]
    if metric == "auc_roc":
        return compute_auc(conf, labels)
    if metric == "brier_mean":
        return compute_brier_mean(conf, labels)
    if metric == "cov_at_5pct_risk":
        return coverage_at_risk(conf, labels, 0.05)
    if metric == "cov_at_10pct_risk":
        return coverage_at_risk(conf, labels, 0.10)
    if metric == "acc_at_50":
        return accuracy_at_coverage(conf, labels, 0.50)
    if metric == "acc_at_80":
        return accuracy_at_coverage(conf, labels, 0.80)
    raise ValueError(f"Unknown metric: {metric}")


def _tost_from_bootstrap(
    boot_diffs: np.ndarray,
    delta: float,
    alpha: float,
) -> Tuple[float, float, float]:
    if len(boot_diffs) == 0 or not np.isfinite(delta):
        return np.nan, np.nan, np.nan
    p_lower = float(np.mean(boot_diffs <= -delta))
    p_upper = float(np.mean(boot_diffs >= delta))
    equivalent = float(p_lower < alpha and p_upper < alpha)
    return p_lower, p_upper, equivalent


def _tost_ttest(
    diffs: np.ndarray,
    delta: float,
    alpha: float,
) -> Tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2 or not np.isfinite(delta):
        return np.nan, np.nan, np.nan
    sd = np.std(diffs, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan, np.nan
    se = sd / np.sqrt(len(diffs))
    mean = float(np.mean(diffs))
    t_lower = (mean + delta) / se
    t_upper = (mean - delta) / se
    p_lower = float(1 - stats.t.cdf(t_lower, df=len(diffs) - 1))
    p_upper = float(stats.t.cdf(t_upper, df=len(diffs) - 1))
    equivalent = float(p_lower < alpha and p_upper < alpha)
    return p_lower, p_upper, equivalent


def _mixed_model_sensitivity(
    diffs: np.ndarray,
    student_ids: np.ndarray,
) -> Dict[str, object]:
    diffs = np.asarray(diffs, dtype=float)
    student_ids = np.asarray(student_ids, dtype=str)
    if len(diffs) < 3 or len(np.unique(student_ids)) < 2:
        return {
            "mixed_model_status": "skip",
            "mixed_model_coef": np.nan,
            "mixed_model_se": np.nan,
            "mixed_model_p": np.nan,
        }
    try:
        import statsmodels.api as sm
        exog = np.ones((len(diffs), 1))
        model = sm.MixedLM(diffs, exog, groups=student_ids)
        res = model.fit(reml=False, method="lbfgs", disp=False)
        coef = float(res.params[0]) if len(res.params) else np.nan
        se = float(res.bse[0]) if len(res.bse) else np.nan
        pval = float(res.pvalues[0]) if len(res.pvalues) else np.nan
        status = "ok" if res.converged else "not_converged"
        return {
            "mixed_model_status": status,
            "mixed_model_coef": coef,
            "mixed_model_se": se,
            "mixed_model_p": pval,
        }
    except Exception:
        return {
            "mixed_model_status": "fail",
            "mixed_model_coef": np.nan,
            "mixed_model_se": np.nan,
            "mixed_model_p": np.nan,
        }


def _metric_by_question(
    df: pd.DataFrame,
    metric: str,
    n_bins: int,
    binning: str,
) -> Dict[str, float]:
    if "question_id" not in df.columns:
        return {}
    out: Dict[str, float] = {}
    for q, sub in df.groupby("question_id"):
        conf = sub["raw_confidence"].to_numpy(dtype=float)
        labels = sub["is_correct"].to_numpy(dtype=float)
        if metric in PER_INSTANCE_METRICS:
            vals = sub[metric].to_numpy(dtype=float)
            out[str(q)] = float(np.nanmean(vals))
        else:
            out[str(q)] = _metric_value(metric, conf, labels, n_bins=n_bins, binning=binning)
    return out


def _mean_ranks_from_questions(
    group_to_values: Dict[str, Dict[str, float]],
    *,
    lower_better: bool,
) -> Tuple[Dict[str, float], int]:
    if not group_to_values:
        return {}, 0
    group_keys = list(group_to_values.keys())
    questions = None
    for g in group_keys:
        qset = set(group_to_values[g].keys())
        questions = qset if questions is None else questions.intersection(qset)
    if not questions:
        return {}, 0
    questions = sorted(questions)
    ranks_sum = {g: 0.0 for g in group_keys}
    used = 0
    for q in questions:
        vals = []
        for g in group_keys:
            v = group_to_values[g].get(q, np.nan)
            if not np.isfinite(v):
                vals = []
                break
            vals.append(v)
        if not vals:
            continue
        vals = np.array(vals, dtype=float)
        if lower_better:
            ranks = stats.rankdata(vals, method="average")
        else:
            ranks = stats.rankdata(-vals, method="average")
        for g, r in zip(group_keys, ranks):
            ranks_sum[g] += float(r)
        used += 1
    if used == 0:
        return {}, 0
    mean_ranks = {g: ranks_sum[g] / used for g in group_keys}
    return mean_ranks, used


def _print_table(name: str, df: pd.DataFrame, max_rows: int) -> None:
    print(f"\n{name}: {len(df)} rows")
    if df.empty:
        print("(empty)")
        return
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... truncated to {max_rows} rows")
    else:
        print(df.to_string(index=False))


def _effect_label(diff: float, moderate: float, large: float) -> str:
    if not np.isfinite(diff) or not np.isfinite(moderate) or not np.isfinite(large):
        return "na"
    ad = abs(diff)
    if ad < moderate:
        return "small"
    if ad < large:
        return "medium"
    return "large"


def _summarize_comparisons(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    max_possible_groups: int,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    df = df.copy()
    if "binning_strategy" not in df.columns:
        df["binning_strategy"] = ""
    group_cols = group_cols + ["metric", "binning_strategy"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        sub = sub.copy()
        diff = sub["diff"].to_numpy(dtype=float)
        ci_low = sub["diff_ci_low"].to_numpy(dtype=float)
        ci_high = sub["diff_ci_high"].to_numpy(dtype=float)
        valid = np.isfinite(diff) & np.isfinite(ci_low) & np.isfinite(ci_high)
        if valid.sum() == 0:
            continue
        diff = diff[valid]
        ci_low = ci_low[valid]
        ci_high = ci_high[valid]
        metric = sub["metric"].iloc[0]
        lower_better = metric in LOWER_BETTER
        ties = (ci_low <= 0) & (ci_high >= 0)
        if lower_better:
            better = (diff < 0) & (~ties)
            worse = (diff > 0) & (~ties)
        else:
            better = (diff > 0) & (~ties)
            worse = (diff < 0) & (~ties)
        n_groups = int(valid.sum())
        n_better = int(np.sum(better))
        n_worse = int(np.sum(worse))
        n_tie = int(np.sum(ties))
        concordance = float(max(n_better, n_worse) / n_groups) if n_groups > 0 else np.nan

        if "p_adjusted_by" in sub.columns:
            pvals = sub["p_adjusted_by"].to_numpy(dtype=float)[valid]
        else:
            pvals = sub.get("p_value", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)[valid]
        sig = np.isfinite(pvals) & (pvals < alpha)
        frac_significant = float(np.mean(sig)) if len(sig) else np.nan

        q1 = float(np.percentile(diff, 25))
        q3 = float(np.percentile(diff, 75))
        iqr = q3 - q1
        median_diff = float(np.median(diff))
        mean_diff = float(np.mean(diff))
        diff_min = float(np.min(diff))
        diff_max = float(np.max(diff))

        moderate, large = practical_threshold(metric)
        effect_strength = np.nan
        if np.isfinite(moderate) and moderate > 0 and np.isfinite(median_diff):
            effect_strength = min(abs(median_diff) / moderate, 2.0)
        label = _effect_label(median_diff, moderate, large)
        all_better = bool(n_better == n_groups and n_groups > 0)

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "better_direction": "lower_is_better" if lower_better else "higher_is_better",
            "n_groups": n_groups,
            "max_possible_groups": int(max_possible_groups),
            "n_missing": int(max_possible_groups - n_groups),
            "n_better": n_better,
            "n_worse": n_worse,
            "n_tie": n_tie,
            "concordance": concordance,
            "frac_significant": frac_significant,
            "median_diff": median_diff,
            "mean_diff": mean_diff,
            "iqr": iqr,
            "diff_range_low": diff_min,
            "diff_range_high": diff_max,
            "all_better": all_better,
            "effect_size_label": label,
            "effect_size_strength": effect_strength,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _rank_summary_best(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    df = summary_df.copy()
    if "binning_strategy" not in df.columns:
        df["binning_strategy"] = ""
    ranks = []
    for _, sub in df.groupby(["metric", "binning_strategy"], dropna=False):
        sub = sub.sort_values(
            ["concordance", "frac_significant", "effect_size_strength", "n_missing", "n_tie"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        sub["best_rank"] = np.arange(1, len(sub) + 1)
        ranks.append(sub)
    return pd.concat(ranks, ignore_index=True)


def _select_best(summary_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    if "binning_strategy" not in summary_df.columns:
        summary_df = summary_df.copy()
        summary_df["binning_strategy"] = ""
    return (
        summary_df.sort_values("best_rank")
        .groupby(["metric", "binning_strategy"], dropna=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def _filter_pairs_by_summary(
    pair_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    keys: List[str],
) -> pd.DataFrame:
    if pair_df.empty or summary_df.empty:
        return pair_df.iloc[0:0].copy()
    summary_keys = set(
        tuple(row[k] for k in keys) for _, row in summary_df.iterrows()
    )
    return pair_df[
        pair_df.apply(lambda r: tuple(r[k] for k in keys) in summary_keys, axis=1)
    ].copy()


def _compute_metrics_summary(
    all_preds: Dict[Tuple[str, str], pd.DataFrame],
    metrics: List[str],
    n_bins: int,
    binning_strategies: List[str],
) -> pd.DataFrame:
    rows = []
    for (model, method), df in all_preds.items():
        conf = df["raw_confidence"].to_numpy(dtype=float)
        labels = df["is_correct"].to_numpy(dtype=float)
        if "is_correct" in metrics:
            rows.append({"model": model, "method": method, "metric": "is_correct", "value": compute_accuracy(labels), "n": len(labels)})
        if "squared_error" in metrics:
            rows.append({"model": model, "method": method, "metric": "squared_error", "value": float(np.mean(df["squared_error"].to_numpy(dtype=float))), "n": len(labels)})
        if "raw_confidence" in metrics:
            rows.append({"model": model, "method": method, "metric": "raw_confidence", "value": float(np.mean(conf)), "n": len(labels)})
        for metric in [m for m in metrics if m in GROUP_METRICS]:
            for binning in (binning_strategies if metric in {"ece", "mce"} else [None]):
                val = _metric_value(metric, conf, labels, n_bins=n_bins, binning=binning or "equal_width")
                row = {"model": model, "method": method, "metric": metric, "value": val, "n": len(labels)}
                if binning is not None:
                    row["binning_strategy"] = binning
                rows.append(row)
    return pd.DataFrame(rows)


def _paired_compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    n_perm: int,
    n_boot: int,
    alpha: float,
    ci_method: str,
    max_jackknife: int,
    n_bins: int,
    binning: str,
    block_ids: pd.Series | None,
    mixed_model_flag: bool,
    student_ids: pd.Series | None,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "metric": metric,
        "metric_type": "per_instance" if metric in PER_INSTANCE_METRICS else "group",
        "binning_strategy": binning if metric in {"ece", "mce"} else "",
    }
    a, b = _align_by_instance(df_a, df_b)
    if student_ids is None and "student_id" in a.columns:
        student_ids = a["student_id"]
    total = len(a)
    if total == 0:
        out["n_pairs"] = 0
        out["status"] = "no_overlap"
        return out

    if metric in PER_INSTANCE_METRICS:
        vals_a = a[metric].to_numpy(dtype=float)
        vals_b = b[metric].to_numpy(dtype=float)
        mask = np.isfinite(vals_a) & np.isfinite(vals_b)
        vals_a = vals_a[mask]
        vals_b = vals_b[mask]
        if block_ids is not None:
            block_ids = np.asarray(block_ids)[mask]
        if student_ids is not None:
            student_ids = np.asarray(student_ids)[mask]
        n_pairs = len(vals_a)
        missing_rate = 1 - (n_pairs / total)
        action = missing_action(missing_rate)
        out.update({"n_pairs": n_pairs, "missing_rate": missing_rate, "missing_action": action})
        if warn_sample_size(n_pairs) == "skip" or action == "exclude":
            out["status"] = "excluded"
            return out
        diffs = vals_a - vals_b
        out["diff"] = float(np.mean(diffs))
        out["diff_ci_low"], out["diff_ci_high"] = np.nan, np.nan
        boot_stats = bootstrap_differences(vals_a, vals_b, n_boot, rng, block_ids=block_ids)
        if len(boot_stats) > 0:
            if ci_method == "bca":
                out["diff_ci_low"], out["diff_ci_high"] = bca_ci_paired_diffs(
                    diffs, boot_stats, alpha, block_ids=block_ids
                )
            else:
                out["diff_ci_low"] = float(np.percentile(boot_stats, 100 * alpha / 2))
                out["diff_ci_high"] = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        out["effect_delta"] = paired_cliffs_delta(diffs)
        out["effect_dz"] = dz_effect(diffs)
        out["p_value"] = paired_sign_flip_permutation(diffs, n_perm, rng, block_ids=block_ids)
        out["status"] = warn_sample_size(n_pairs)
        out["n_missing"] = int(total - n_pairs)
        if action == "sensitivity":
            best, worst = sensitivity_bounds(diffs, int(total - n_pairs))
            out["diff_best_case"] = best
            out["diff_worst_case"] = worst
        if mixed_model_flag and student_ids is not None:
            out.update(_mixed_model_sensitivity(diffs, student_ids))
    else:
        conf_a = a["raw_confidence"].to_numpy(dtype=float)
        conf_b = b["raw_confidence"].to_numpy(dtype=float)
        labels = a["is_correct"].to_numpy(dtype=float)
        mask = np.isfinite(conf_a) & np.isfinite(conf_b) & np.isfinite(labels)
        conf_a = conf_a[mask]
        conf_b = conf_b[mask]
        labels = labels[mask]
        if block_ids is not None:
            block_ids = np.asarray(block_ids)[mask]
        n_pairs = len(labels)
        missing_rate = 1 - (n_pairs / total)
        action = missing_action(missing_rate)
        out.update({"n_pairs": n_pairs, "missing_rate": missing_rate, "missing_action": action})
        if warn_sample_size(n_pairs) == "skip" or action == "exclude":
            out["status"] = "excluded"
            return out
        metric_func = lambda c, y: _metric_value(metric, c, y, n_bins=n_bins, binning=binning)
        val_a = metric_func(conf_a, labels)
        val_b = metric_func(conf_b, labels)
        out["mean_a"] = float(val_a)
        out["mean_b"] = float(val_b)
        out["diff"] = float(val_a - val_b)
        out["relative_diff"] = float((val_a - val_b) / val_b) if np.isfinite(val_b) and val_b != 0 else np.nan
        boot_diffs = bootstrap_group_metric_diff(conf_a, conf_b, labels, metric_func, n_boot, rng, block_ids=block_ids)
        if len(boot_diffs):
            if ci_method == "bca":
                out["diff_ci_low"], out["diff_ci_high"] = bca_ci_group_diff(
                    conf_a,
                    conf_b,
                    labels,
                    metric_func,
                    boot_diffs,
                    alpha,
                    block_ids=block_ids,
                    max_jackknife=max_jackknife,
                )
            else:
                out["diff_ci_low"] = float(np.percentile(boot_diffs, 100 * alpha / 2))
                out["diff_ci_high"] = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
        out["p_value"] = paired_group_metric_permutation(conf_a, conf_b, labels, metric_func, n_perm, rng, block_ids=block_ids)
        out["status"] = warn_sample_size(n_pairs)
        out["n_missing"] = int(total - n_pairs)
    moderate, large = practical_threshold(metric)
    out["practical_moderate"] = moderate
    out["practical_large"] = large
    if np.isfinite(moderate) and np.isfinite(out.get("diff_ci_low", np.nan)) and np.isfinite(out.get("diff_ci_high", np.nan)):
        out["equivalence_moderate"] = bool(out["diff_ci_low"] > -moderate and out["diff_ci_high"] < moderate)
    else:
        out["equivalence_moderate"] = np.nan
    out["tost_p_lower"] = np.nan
    out["tost_p_upper"] = np.nan
    out["tost_equivalent"] = np.nan
    if np.isfinite(moderate):
        if metric in PER_INSTANCE_METRICS:
            if "boot_stats" in locals() and len(boot_stats) > 0:
                p_lower, p_upper, eq = _tost_from_bootstrap(boot_stats, moderate, alpha)
                out["tost_p_lower"] = p_lower
                out["tost_p_upper"] = p_upper
                out["tost_equivalent"] = eq
            else:
                p_lower, p_upper, eq = _tost_ttest(diffs, moderate, alpha)
                out["tost_p_lower"] = p_lower
                out["tost_p_upper"] = p_upper
                out["tost_equivalent"] = eq
        else:
            if "boot_diffs" in locals() and len(boot_diffs) > 0:
                p_lower, p_upper, eq = _tost_from_bootstrap(boot_diffs, moderate, alpha)
                out["tost_p_lower"] = p_lower
                out["tost_p_upper"] = p_upper
                out["tost_equivalent"] = eq
    return out


def _omnibus_friedman(per_instance_data: Dict[str, np.ndarray]) -> Tuple[float, float, int]:
    values = [v for v in per_instance_data.values() if len(v) > 0]
    if len(values) < 2:
        return np.nan, np.nan, 0
    min_len = min(len(v) for v in values)
    values = [v[:min_len] for v in values]
    stat, p = stats.friedmanchisquare(*values)
    return float(stat), float(p), int(min_len)


def main() -> None:
    args = parse_args()
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

    project_root = Path(__file__).resolve().parent.parent.parent
    if args.predictions_dir:
        predictions_dir = Path(args.predictions_dir)
        legacy_layout = True
    elif output_manager:
        predictions_dir = output_manager.step_dir("step1", "tables")
        legacy_layout = False
    else:
        predictions_dir = find_csv_folder_from_root("grading_calibration")
        legacy_layout = True
    layout_label = "legacy" if legacy_layout else "organized"
    
    all_preds = load_all_predictions(
        args.models,
        args.methods,
        predictions_dir,
        prefix="predictions",
        dataset=args.dataset,
        flat_layout=not legacy_layout,
        filter_by_dataset=legacy_layout,
    )
    if not all_preds:
        raise RuntimeError("No predictions found for given models/methods.")

    if args.smoke_n is not None:
        ref_df = _ensure_instance_id_col(next(iter(all_preds.values())))
        ref_ids = ref_df["instance_id"].head(args.smoke_n).tolist()
        new_preds = {}
        for key, df in all_preds.items():
            df = _ensure_instance_id_col(df)
            df = df[df["instance_id"].isin(ref_ids)].head(args.smoke_n).copy()
            new_preds[key] = df
        all_preds = new_preds
        print(f"Smoke test: limiting to first {len(ref_ids)} instances by instance_id")

    # ICC diagnostics on first available combination
    icc_flag = False
    mixed_model_flag = False
    icc_student = np.nan
    first_df = next(iter(all_preds.values()))
    if {"student_id", "question_id", "rubric_id"}.issubset(first_df.columns):
        try:
            icc = compute_icc_mixed_model(
                values=first_df["raw_confidence"].to_numpy(dtype=float),
                student_ids=first_df["student_id"].to_numpy(dtype=str),
                question_ids=first_df["question_id"].to_numpy(dtype=str),
                rubric_ids=first_df["rubric_id"].to_numpy(dtype=str),
            )
            icc_student = icc.get("icc_student", np.nan)
            if np.isfinite(icc_student) and icc_student > 0.05:
                icc_flag = True
            if np.isfinite(icc_student) and icc_student > 0.1:
                mixed_model_flag = True
        except Exception:
            pass

    print("=" * 80)
    print("STEP 6: CROSS-MODEL/METHOD COMPARISON")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Methods: {args.methods}")
    print(f"Metrics: {args.metrics}")
    print(f"Student-block sensitivity: {'ON' if icc_flag else 'OFF'}")
    print(f"Mixed-model sensitivity: {'ON' if mixed_model_flag else 'OFF'}")
    print(f"CI method: {args.ci_method}")
    if args.skip_interaction_tests:
        print("Interaction tests: SKIPPED")
    else:
        print(f"Interaction method: {args.interaction_method}")
    print(f"Best ranking top_k: {args.best_top_k}")

    comparison_outputs = _pipeline_run_cross_comparisons(
        all_preds=all_preds,
        args=args,
        rng=rng,
        icc_flag=icc_flag,
        mixed_model_flag=mixed_model_flag,
        compute_metrics_summary=_compute_metrics_summary,
        omnibus_friedman=_omnibus_friedman,
        align_by_instance=_align_by_instance,
        paired_compare=_paired_compare,
        summarize_comparisons=_summarize_comparisons,
        rank_summary_best=_rank_summary_best,
        select_best=_select_best,
    )
    metrics_summary = comparison_outputs["metrics_summary"]
    omnibus_df = comparison_outputs["omnibus_df"]
    model_df = comparison_outputs["model_df"]
    method_df = comparison_outputs["method_df"]
    method_summary_df = comparison_outputs["method_summary_df"]
    model_summary_df = comparison_outputs["model_summary_df"]
    best_method_summary = comparison_outputs["best_method_summary"]
    best_model_summary = comparison_outputs["best_model_summary"]
    interaction_df = comparison_outputs["interaction_df"]
    hetero_df = comparison_outputs["hetero_df"]

    # Calibration regression diagnostics (optional)
    cal_reg_interaction_df = pd.DataFrame()
    cal_reg_summary_df = pd.DataFrame()
    cal_reg_curve_df = pd.DataFrame()
    if args.calibration_regression:
        print("Running calibration regression diagnostics...", flush=True)
        cal_models = args.models
        cal_methods = args.methods
        if args.calibration_regression_max_models:
            cal_models = cal_models[: args.calibration_regression_max_models]
        if args.calibration_regression_max_methods:
            cal_methods = cal_methods[: args.calibration_regression_max_methods]

        cal_rows = []
        progress_disable = (not args.force_progress and not sys.stderr.isatty())
        cell_pairs = [(model, method) for model in cal_models for method in cal_methods]
        skipped_cells = []
        for model, method in tqdm(cell_pairs, desc="Calibration regression cells", unit="cell", disable=progress_disable):
                df = all_preds.get((model, method))
                if df is None:
                    continue
                if "raw_confidence" not in df.columns or "is_correct" not in df.columns:
                    continue
                if len(df) < args.calibration_regression_min_n:
                    skipped_cells.append((model, method, len(df)))
                    continue
                df_sub = df.copy()
                df_sub["model"] = model
                df_sub["method"] = method
                cal_rows.append(df_sub)

        if not cal_rows:
            print("⚠ No model-method cells met calibration regression requirements.", flush=True)
        else:
            cal_df = pd.concat(cal_rows, ignore_index=True)
            if skipped_cells:
                print(f"Skipped {len(skipped_cells)} model-method cells with n < {args.calibration_regression_min_n}.", flush=True)
            cal_mask = np.isfinite(cal_df["raw_confidence"].to_numpy(dtype=float)) & np.isfinite(
                cal_df["is_correct"].to_numpy(dtype=float)
            )
            if "student_id" in cal_df.columns:
                cal_mask &= cal_df["student_id"].notna()
            if "rubric_id" in cal_df.columns:
                cal_mask &= cal_df["rubric_id"].notna()
            cal_df = cal_df.loc[cal_mask].copy()
            conf_all = np.clip(cal_df["raw_confidence"].to_numpy(dtype=float), 0.001, 0.999)
            logit_all = np.log(conf_all / (1 - conf_all))
            logit_mean = float(np.mean(logit_all))
            cal_df["logit_conf_c"] = logit_all - logit_mean
            cal_df["y"] = cal_df["is_correct"].astype(int)

            use_student_re = (
                "student_id" in cal_df.columns
                and cal_df["student_id"].nunique() >= 2
                and cal_df["student_id"].nunique() < len(cal_df)
            )
            use_rubric_re = (
                "rubric_id" in cal_df.columns
                and cal_df["rubric_id"].nunique() >= 2
                and cal_df["rubric_id"].nunique() < len(cal_df)
            )
            re_terms = []
            if use_student_re:
                re_terms.append("(1|student_id)")
            if use_rubric_re:
                re_terms.append("(1|rubric_id)")
            re_part = f" + {' + '.join(re_terms)}" if re_terms else ""

            def _interaction_row(
                test: str,
                mode: str,
                res: Optional[dict],
                expected_df: Optional[float] = None,
                **extra: Any,
            ) -> dict:
                row = {"test": test, "mode": mode}
                if res is not None:
                    row.update({
                        "chisq": res.get("chisq", np.nan),
                        "df_diff": res.get("df_diff", np.nan),
                        "p_value": res.get("p_value_lrt", np.nan),
                        "converged": res.get("converged", False),
                        "singular": res.get("singular", False),
                        "full_df_resid": res.get("full_df_resid", np.nan),
                        "overdispersion_ratio": res.get("overdispersion_ratio", np.nan),
                    })
                    if expected_df is not None and np.isfinite(expected_df):
                        df_diff = row.get("df_diff", np.nan)
                        row["df_expected"] = float(expected_df)
                        row["df_mismatch"] = bool(
                            np.isfinite(df_diff) and abs(float(df_diff) - float(expected_df)) > 0.5
                        )
                row.update(extra)
                return row

            interaction_rows = []
            if args.skip_interaction_tests:
                print("Skipping calibration regression interaction tests (--skip-interaction-tests).", flush=True)
            else:
                if args.calibration_regression_mode == "linear":
                    if cal_df["model"].nunique() >= 2:
                        expected_df = float(cal_df["model"].nunique() - 1)
                        res = fit_lme4_lrt_from_formula(
                            cal_df,
                            formula_null=f"y ~ logit_conf_c + model + method{re_part}",
                            formula_full=f"y ~ logit_conf_c * model + method{re_part}",
                            is_glmm=True,
                            family="binomial",
                            coef_name=None,
                        )
                        interaction_rows.append(_interaction_row(
                            "logit_conf:model",
                            "linear",
                            res,
                            expected_df=expected_df,
                        ))
                    if cal_df["method"].nunique() >= 2:
                        expected_df = float(cal_df["method"].nunique() - 1)
                        res = fit_lme4_lrt_from_formula(
                            cal_df,
                            formula_null=f"y ~ logit_conf_c + method + model{re_part}",
                            formula_full=f"y ~ logit_conf_c * method + model{re_part}",
                            is_glmm=True,
                            family="binomial",
                            coef_name=None,
                        )
                        interaction_rows.append(_interaction_row(
                            "logit_conf:method",
                            "linear",
                            res,
                            expected_df=expected_df,
                        ))
                    # Three-way (optional) only if both interactions show signal
                    p_model = next((r["p_value"] for r in interaction_rows if r["test"] == "logit_conf:model"), np.nan)
                    p_method = next((r["p_value"] for r in interaction_rows if r["test"] == "logit_conf:method"), np.nan)
                    if np.isfinite(p_model) and np.isfinite(p_method) and p_model < args.alpha and p_method < args.alpha:
                        expected_df = float((cal_df["model"].nunique() - 1) * (cal_df["method"].nunique() - 1))
                        res = fit_lme4_lrt_from_formula(
                            cal_df,
                            formula_null=f"y ~ logit_conf_c * model + logit_conf_c * method + model * method{re_part}",
                            formula_full=f"y ~ logit_conf_c * model * method{re_part}",
                            is_glmm=True,
                            family="binomial",
                            coef_name=None,
                        )
                        interaction_rows.append(_interaction_row(
                            "logit_conf:model:method",
                            "linear",
                            res,
                            expected_df=expected_df,
                        ))
                else:
                    max_df = int(min(6, max(3, np.floor(len(cal_df) / 10))))
                    spline_df = min(args.calibration_regression_spline_df, max_df)
                    basis = f"bs(logit_conf_c, df={spline_df}, degree=3, intercept=FALSE)"
                    if cal_df["model"].nunique() >= 2:
                        try:
                            expected_df = float(spline_df * (cal_df["model"].nunique() - 1))
                            res = fit_lme4_lrt_from_formula(
                                cal_df,
                                formula_null=f"y ~ {basis} + model + method{re_part}",
                                formula_full=f"y ~ {basis} * model + method{re_part}",
                                is_glmm=True,
                                family="binomial",
                                coef_name=None,
                            )
                            interaction_rows.append(_interaction_row(
                                "bs(logit_conf):model",
                                "nonlinear",
                                res,
                                expected_df=expected_df,
                                spline_df=spline_df,
                            ))
                        except Exception as e:
                            if args.calibration_regression_fallback == "linear":
                                expected_df = float(cal_df["model"].nunique() - 1)
                                res = fit_lme4_lrt_from_formula(
                                    cal_df,
                                    formula_null=f"y ~ logit_conf_c + model + method{re_part}",
                                    formula_full=f"y ~ logit_conf_c * model + method{re_part}",
                                    is_glmm=True,
                                    family="binomial",
                                    coef_name=None,
                                )
                                interaction_rows.append(_interaction_row(
                                    "logit_conf:model",
                                    "linear_fallback",
                                    res,
                                    expected_df=expected_df,
                                    error=str(e),
                                ))
                            else:
                                interaction_rows.append(_interaction_row(
                                    "bs(logit_conf):model",
                                    "nonlinear",
                                    None,
                                    p_value=np.nan,
                                    error=str(e),
                                ))
                    if cal_df["method"].nunique() >= 2:
                        try:
                            expected_df = float(spline_df * (cal_df["method"].nunique() - 1))
                            res = fit_lme4_lrt_from_formula(
                                cal_df,
                                formula_null=f"y ~ {basis} + method + model{re_part}",
                                formula_full=f"y ~ {basis} * method + model{re_part}",
                                is_glmm=True,
                                family="binomial",
                                coef_name=None,
                            )
                            interaction_rows.append(_interaction_row(
                                "bs(logit_conf):method",
                                "nonlinear",
                                res,
                                expected_df=expected_df,
                                spline_df=spline_df,
                            ))
                        except Exception as e:
                            if args.calibration_regression_fallback == "linear":
                                expected_df = float(cal_df["method"].nunique() - 1)
                                res = fit_lme4_lrt_from_formula(
                                    cal_df,
                                    formula_null=f"y ~ logit_conf_c + method + model{re_part}",
                                    formula_full=f"y ~ logit_conf_c * method + model{re_part}",
                                    is_glmm=True,
                                    family="binomial",
                                    coef_name=None,
                                )
                                interaction_rows.append(_interaction_row(
                                    "logit_conf:method",
                                    "linear_fallback",
                                    res,
                                    expected_df=expected_df,
                                    error=str(e),
                                ))
                            else:
                                interaction_rows.append(_interaction_row(
                                    "bs(logit_conf):method",
                                    "nonlinear",
                                    None,
                                    p_value=np.nan,
                                    error=str(e),
                                ))
                    p_model = next((r["p_value"] for r in interaction_rows if r["test"] == "bs(logit_conf):model"), np.nan)
                    p_method = next((r["p_value"] for r in interaction_rows if r["test"] == "bs(logit_conf):method"), np.nan)
                    if np.isfinite(p_model) and np.isfinite(p_method) and p_model < args.alpha and p_method < args.alpha:
                        try:
                            expected_df = float(
                                spline_df * (cal_df["model"].nunique() - 1) * (cal_df["method"].nunique() - 1)
                            )
                            res = fit_lme4_lrt_from_formula(
                                cal_df,
                                formula_null=f"y ~ {basis} * model + {basis} * method + model * method{re_part}",
                                formula_full=f"y ~ {basis} * model * method{re_part}",
                                is_glmm=True,
                                family="binomial",
                                coef_name=None,
                            )
                            interaction_rows.append(_interaction_row(
                                "bs(logit_conf):model:method",
                                "nonlinear",
                                res,
                                expected_df=expected_df,
                                spline_df=spline_df,
                            ))
                        except Exception as e:
                            if args.calibration_regression_fallback == "linear":
                                expected_df = float((cal_df["model"].nunique() - 1) * (cal_df["method"].nunique() - 1))
                                res = fit_lme4_lrt_from_formula(
                                    cal_df,
                                    formula_null=f"y ~ logit_conf_c * model + logit_conf_c * method + model * method{re_part}",
                                    formula_full=f"y ~ logit_conf_c * model * method{re_part}",
                                    is_glmm=True,
                                    family="binomial",
                                    coef_name=None,
                                )
                                interaction_rows.append(_interaction_row(
                                    "logit_conf:model:method",
                                    "linear_fallback",
                                    res,
                                    expected_df=expected_df,
                                    error=str(e),
                                ))
                            else:
                                interaction_rows.append(_interaction_row(
                                    "bs(logit_conf):model:method",
                                    "nonlinear",
                                    None,
                                    p_value=np.nan,
                                    error=str(e),
                                ))

            cal_reg_interaction_df = pd.DataFrame(interaction_rows)
            if len(cal_reg_interaction_df) and "p_value" in cal_reg_interaction_df.columns:
                p_adj, sig = by_fdr(cal_reg_interaction_df["p_value"].to_numpy(dtype=float), args.alpha)
                cal_reg_interaction_df["p_adjusted_by"] = p_adj
                cal_reg_interaction_df["significant_by"] = sig

            # Descriptive per model-method summaries
            summary_rows = []
            curve_rows = []
            conf_grid = np.linspace(0.01, 0.99, args.calibration_regression_n_grid)

            def _linear_cell_summary(df_cell: pd.DataFrame) -> dict:
                use_student_re_cell = (
                    "student_id" in df_cell.columns
                    and df_cell["student_id"].nunique() >= 2
                    and df_cell["student_id"].nunique() < len(df_cell)
                )
                use_rubric_re_cell = (
                    "rubric_id" in df_cell.columns
                    and df_cell["rubric_id"].nunique() >= 2
                    and df_cell["rubric_id"].nunique() < len(df_cell)
                )
                cols = ["y", "logit_conf_c"]
                if use_student_re_cell:
                    cols.append("student_id")
                if use_rubric_re_cell:
                    cols.append("rubric_id")
                df_fit = df_cell[cols].copy()
                re_terms_cell = []
                if use_student_re_cell:
                    re_terms_cell.append("(1|student_id)")
                if use_rubric_re_cell:
                    re_terms_cell.append("(1|rubric_id)")
                re_part_cell = f" + {' + '.join(re_terms_cell)}" if re_terms_cell else ""
                formula = f"y ~ logit_conf_c{re_part_cell}"
                beta, se, p = _extract_coefficient_from_glmm(
                    df_fit,
                    formula,
                    "logit_conf_c",
                    glmer_nAGQ=1,
                    optimizer="bobyqa",
                    maxfun=100000,
                )
                return {
                    "slope": beta,
                    "slope_se": se,
                    "slope_p": p,
                }

            for model, method in tqdm(
                cell_pairs, desc="Calibration regression summaries", unit="cell", disable=progress_disable
            ):
                    df_cell = cal_df[(cal_df["model"] == model) & (cal_df["method"] == method)]
                    n_obs = len(df_cell)
                    if n_obs < args.calibration_regression_min_n:
                        continue
                    cell_warnings: List[str] = []
                    cell_row = {
                        "model": model,
                        "method": method,
                        "n_obs": int(n_obs),
                        "mode": args.calibration_regression_mode,
                    }
                    if args.calibration_regression_mode == "linear":
                        cell_row.update(_linear_cell_summary(df_cell))
                    else:
                        curve_res = fit_calibration_curve_glmm(
                            df_cell["y"].to_numpy(dtype=float),
                            df_cell["raw_confidence"].to_numpy(dtype=float),
                            df_cell["student_id"].to_numpy(dtype=str) if "student_id" in df_cell.columns else np.arange(len(df_cell)),
                            df_cell["rubric_id"].to_numpy(dtype=str) if "rubric_id" in df_cell.columns else None,
                            spline_df=args.calibration_regression_spline_df,
                            n_grid=args.calibration_regression_n_grid,
                            logit_conf_mean=logit_mean,
                            conf_grid=conf_grid,
                        )
                        curve = curve_res.get("curve", None)
                        preds = None
                        if curve:
                            preds = curve.get("pred")
                        if preds is None:
                            warnings = curve_res.get("warnings", [])
                            if args.calibration_regression_fallback == "linear":
                                cell_row.update(_linear_cell_summary(df_cell))
                                cell_row["mode"] = "linear_fallback"
                                if warnings:
                                    warnings = warnings + ["fallback=linear"]
                                else:
                                    warnings = ["fallback=linear"]
                                cell_row["warnings"] = "; ".join(warnings)
                            else:
                                cell_row["warnings"] = "; ".join(warnings) if warnings else "curve_fit_failed"
                            summary_rows.append(cell_row)
                            continue
                        if preds is not None:
                            idx_low = int(np.argmin(np.abs(curve["conf_grid"] - 0.1)))
                            idx_high = int(np.argmin(np.abs(curve["conf_grid"] - 0.9)))
                            delta_10_90 = float(preds[idx_high] - preds[idx_low])
                            cell_row["delta_10_90"] = delta_10_90
                            if args.calibration_regression_bootstrap > 0:
                                boot_deltas = []
                                rng_local = np.random.default_rng(args.seed)
                                if "student_id" in df_cell.columns and df_cell["student_id"].nunique() >= 2:
                                    units = df_cell["student_id"].to_numpy(dtype=str)
                                    if args.calibration_regression_ci_method == "bca" and len(np.unique(units)) < 30:
                                        cell_warnings.append("BCa may be unstable with < 30 students; consider percentile CI.")
                                    uniq_units = np.unique(units)
                                    for _ in range(args.calibration_regression_bootstrap):
                                        sampled = rng_local.choice(uniq_units, size=len(uniq_units), replace=True)
                                        idx_list = [np.where(units == u)[0] for u in sampled]
                                        idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
                                        if idx.size < 10:
                                            boot_deltas.append(np.nan)
                                            continue
                                        df_boot = df_cell.iloc[idx].copy()
                                        try:
                                            boot_curve = fit_calibration_curve_glmm(
                                                df_boot["y"].to_numpy(dtype=float),
                                                df_boot["raw_confidence"].to_numpy(dtype=float),
                                                df_boot["student_id"].to_numpy(dtype=str),
                                                df_boot["rubric_id"].to_numpy(dtype=str) if "rubric_id" in df_boot.columns else None,
                                                spline_df=args.calibration_regression_spline_df,
                                                n_grid=args.calibration_regression_n_grid,
                                                logit_conf_mean=logit_mean,
                                                conf_grid=conf_grid,
                                            )
                                            boot_preds = boot_curve.get("curve", {}).get("pred")
                                            if boot_preds is None:
                                                boot_deltas.append(np.nan)
                                            else:
                                                b_low = int(np.argmin(np.abs(conf_grid - 0.1)))
                                                b_high = int(np.argmin(np.abs(conf_grid - 0.9)))
                                                boot_deltas.append(float(boot_preds[b_high] - boot_preds[b_low]))
                                        except Exception:
                                            boot_deltas.append(np.nan)
                                else:
                                    if "rubric_id" in df_cell.columns and df_cell["rubric_id"].nunique() >= 2:
                                        cell_warnings.append("Bootstrap uses IID resampling; rubric clustering not preserved.")
                                    n_boot = args.calibration_regression_bootstrap
                                    for _ in range(n_boot):
                                        idx = rng_local.integers(0, n_obs, size=n_obs)
                                        df_boot = df_cell.iloc[idx].copy()
                                        try:
                                            boot_curve = fit_calibration_curve_glmm(
                                                df_boot["y"].to_numpy(dtype=float),
                                                df_boot["raw_confidence"].to_numpy(dtype=float),
                                                np.arange(len(df_boot)),
                                                df_boot["rubric_id"].to_numpy(dtype=str) if "rubric_id" in df_boot.columns else None,
                                                spline_df=args.calibration_regression_spline_df,
                                                n_grid=args.calibration_regression_n_grid,
                                                logit_conf_mean=logit_mean,
                                                conf_grid=conf_grid,
                                            )
                                            boot_preds = boot_curve.get("curve", {}).get("pred")
                                            if boot_preds is None:
                                                boot_deltas.append(np.nan)
                                            else:
                                                b_low = int(np.argmin(np.abs(conf_grid - 0.1)))
                                                b_high = int(np.argmin(np.abs(conf_grid - 0.9)))
                                                boot_deltas.append(float(boot_preds[b_high] - boot_preds[b_low]))
                                        except Exception:
                                            boot_deltas.append(np.nan)
                                boot_arr = np.asarray(boot_deltas, dtype=float)
                                boot_arr = boot_arr[np.isfinite(boot_arr)]
                                if boot_arr.size < 10:
                                    cell_warnings.append(
                                        f"Bootstrap valid draws {boot_arr.size}/{args.calibration_regression_bootstrap}; CI not computed."
                                    )
                                if boot_arr.size >= 10:
                                    if args.calibration_regression_ci_method == "bca":
                                        jack_stats = []
                                        if "student_id" in df_cell.columns and df_cell["student_id"].nunique() >= 2:
                                            units = np.unique(df_cell["student_id"].to_numpy(dtype=str))
                                            if args.calibration_regression_jackknife_max > 0 and len(units) > args.calibration_regression_jackknife_max:
                                                units = rng_local.choice(
                                                    units,
                                                    size=args.calibration_regression_jackknife_max,
                                                    replace=False,
                                                )
                                            for u in units:
                                                idx_j = np.where(df_cell["student_id"].to_numpy(dtype=str) != u)[0]
                                                if idx_j.size < 10:
                                                    continue
                                                df_j = df_cell.iloc[idx_j].copy()
                                                try:
                                                    jack_curve = fit_calibration_curve_glmm(
                                                        df_j["y"].to_numpy(dtype=float),
                                                        df_j["raw_confidence"].to_numpy(dtype=float),
                                                        df_j["student_id"].to_numpy(dtype=str),
                                                        df_j["rubric_id"].to_numpy(dtype=str) if "rubric_id" in df_j.columns else None,
                                                        spline_df=args.calibration_regression_spline_df,
                                                        n_grid=args.calibration_regression_n_grid,
                                                        logit_conf_mean=logit_mean,
                                                        conf_grid=conf_grid,
                                                    )
                                                    jack_preds = jack_curve.get("curve", {}).get("pred")
                                                    if jack_preds is None:
                                                        continue
                                                    j_low = int(np.argmin(np.abs(conf_grid - 0.1)))
                                                    j_high = int(np.argmin(np.abs(conf_grid - 0.9)))
                                                    jack_stats.append(float(jack_preds[j_high] - jack_preds[j_low]))
                                                except Exception:
                                                    continue
                                        if jack_stats:
                                            ci_low, ci_high = _compute_bca_ci_from_jackknife(
                                                boot_arr, delta_10_90, np.asarray(jack_stats, dtype=float), alpha=args.alpha
                                            )
                                        else:
                                            ci_low, ci_high = np.percentile(
                                                boot_arr, [100 * args.alpha / 2, 100 * (1 - args.alpha / 2)]
                                            )
                                    else:
                                        ci_low, ci_high = np.percentile(
                                            boot_arr, [100 * args.alpha / 2, 100 * (1 - args.alpha / 2)]
                                        )
                                    cell_row["delta_10_90_ci_low"] = float(ci_low)
                                    cell_row["delta_10_90_ci_high"] = float(ci_high)
                            curve_df = pd.DataFrame({
                                "model": model,
                                "method": method,
                                "confidence": curve["conf_grid"],
                                "pred": preds,
                            })
                            curve_rows.append(curve_df)
                        cell_warnings.extend(curve_res.get("warnings", []))
                        if cell_warnings:
                            cell_row["warnings"] = "; ".join(cell_warnings)
                    summary_rows.append(cell_row)

            cal_reg_summary_df = pd.DataFrame(summary_rows)
            if curve_rows:
                cal_reg_curve_df = pd.concat(curve_rows, ignore_index=True)

    # Confirmatory tests
    confirm_rows = []
    confirmatory_df = pd.DataFrame()
    if args.run_confirmatory:
        print("Running confirmatory tests...", flush=True)
        # H1: self_consistency vs self_reported on ECE
        for model in tqdm(args.models, desc="Confirmatory H1/H2"):
            df_a = all_preds.get((model, "self_consistency"))
            df_b = all_preds.get((model, "self_reported"))
            if df_a is None or df_b is None:
                continue
            a, b = _align_by_instance(df_a, df_b)
            conf_a = a["raw_confidence"].to_numpy(dtype=float)
            conf_b = b["raw_confidence"].to_numpy(dtype=float)
            labels = a["is_correct"].to_numpy(dtype=float)
            mask = np.isfinite(conf_a) & np.isfinite(conf_b) & np.isfinite(labels)
            conf_a = conf_a[mask]
            conf_b = conf_b[mask]
            labels = labels[mask]
            if len(labels) < 10:
                continue
            p = paired_group_metric_permutation(
                conf_a,
                conf_b,
                labels,
                lambda c, y: compute_ece_mce(c, y, n_bins=args.n_bins, strategy="equal_width")[0],
                args.confirmatory_perm,
                rng,
                None,
            )
            confirm_rows.append({
                "hypothesis": "H1_ece_self_consistency",
                "model": model,
                "p_value": p,
                "n_pairs": int(len(labels)),
                "seed": args.seed,
            })
        # H2: token_probability vs self_reported on AUC
        for model in tqdm(args.models, desc="Confirmatory H2"):
            df_a = all_preds.get((model, "token_probability"))
            df_b = all_preds.get((model, "self_reported"))
            if df_a is None or df_b is None:
                continue
            a, b = _align_by_instance(df_a, df_b)
            conf_a = a["raw_confidence"].to_numpy(dtype=float)
            conf_b = b["raw_confidence"].to_numpy(dtype=float)
            labels = a["is_correct"].to_numpy(dtype=float)
            mask = np.isfinite(conf_a) & np.isfinite(conf_b) & np.isfinite(labels)
            conf_a = conf_a[mask]
            conf_b = conf_b[mask]
            labels = labels[mask]
            if len(labels) < 10:
                continue
            p = paired_group_metric_permutation(conf_a, conf_b, labels, compute_auc, args.confirmatory_perm, rng, None)
            confirm_rows.append({
                "hypothesis": "H2_auc_token_probability",
                "model": model,
                "p_value": p,
                "n_pairs": int(len(labels)),
                "seed": args.seed,
            })
        # H3: methods differ on coverage at 5% risk (omnibus)
        for model in tqdm(args.models, desc="Confirmatory H3"):
            dfs = [all_preds.get((model, method)) for method in args.methods]
            if any(df is None for df in dfs):
                continue
            aligned = _align_multiple(dfs)
            conf_matrix = np.vstack([df["raw_confidence"].to_numpy(dtype=float) for df in aligned])
            labels = aligned[0]["is_correct"].to_numpy(dtype=float)
            mask = np.isfinite(labels)
            for i in range(conf_matrix.shape[0]):
                mask &= np.isfinite(conf_matrix[i])
            conf_matrix = conf_matrix[:, mask]
            labels = labels[mask]
            n_pairs = int(len(labels))
            if n_pairs < 10:
                continue
            observed_cov = [coverage_at_risk(conf_matrix[i], labels, 0.05) for i in range(conf_matrix.shape[0])]
            stat_obs = float(np.max(observed_cov) - np.min(observed_cov))
            perms = np.array(list(itertools.permutations(range(conf_matrix.shape[0]))), dtype=int)
            if len(perms) == 0:
                continue
            perm_stats = np.empty(args.confirmatory_perm, dtype=float)
            n_methods = conf_matrix.shape[0]
            n_instances = conf_matrix.shape[1]
            for i in range(args.confirmatory_perm):
                perm_idx = rng.integers(0, len(perms), size=n_instances)
                permuted = np.empty_like(conf_matrix)
                perm_rows = perms[perm_idx]
                for m in range(n_methods):
                    permuted[m] = conf_matrix[perm_rows[:, m], np.arange(n_instances)]
                covs = [coverage_at_risk(permuted[m], labels, 0.05) for m in range(n_methods)]
                perm_stats[i] = np.max(covs) - np.min(covs)
            p = (np.sum(perm_stats >= stat_obs) + 1) / (args.confirmatory_perm + 1)
            confirm_rows.append({
                "hypothesis": "H3_cov_at_5pct_risk_methods",
                "model": model,
                "p_value": float(p),
                "statistic": stat_obs,
                "n_pairs": n_pairs,
                "seed": args.seed,
            })
    confirmatory_df = pd.DataFrame(confirm_rows)
    if len(confirmatory_df):
        confirmatory_df["significant"] = confirmatory_df["p_value"] < args.confirmatory_alpha

    # Confirmatory re-tests from summary (optional)
    confirmatory_summary_df = pd.DataFrame()
    if args.confirmatory_summary_scope != "none":
        print("Running confirmatory re-tests from summary tables...", flush=True)
        if args.confirmatory_summary_scope == "best":
            model_pairs = _filter_pairs_by_summary(
                model_df,
                best_model_summary,
                keys=["model_a", "model_b", "metric", "binning_strategy"],
            )
            method_pairs = _filter_pairs_by_summary(
                method_df,
                best_method_summary,
                keys=["method_a", "method_b", "metric", "binning_strategy"],
            )
        else:
            model_pairs = model_df.copy()
            method_pairs = method_df.copy()

        confirm_rows = []
        if len(model_pairs):
            print("Confirmatory re-tests (model pairs)...", flush=True)
            for _, row in tqdm(model_pairs.iterrows(), total=len(model_pairs), desc="Confirmatory model pairs"):
                method = row.get("method", "")
                ma = row.get("model_a", "")
                mb = row.get("model_b", "")
                metric = row.get("metric", "")
                binning = row.get("binning_strategy", "") or "equal_width"
                df_a = all_preds.get((ma, method))
                df_b = all_preds.get((mb, method))
                if df_a is None or df_b is None:
                    continue
                a, b = _align_by_instance(df_a, df_b)
                block_ids = None
                if icc_flag and "student_id" in a.columns:
                    block_ids = a["student_id"].to_numpy(dtype=str)
                res = _paired_compare(
                    a,
                    b,
                    metric,
                    rng,
                    args.confirmatory_perm,
                    0,
                    args.confirmatory_alpha,
                    args.ci_method,
                    args.max_jackknife,
                    args.n_bins,
                    binning,
                    block_ids,
                    mixed_model_flag,
                    None,
                )
                res.update({
                    "comparison_type": "model",
                    "method": method,
                    "model_a": ma,
                    "model_b": mb,
                    "seed": args.seed,
                    "confirmatory_scope": args.confirmatory_summary_scope,
                })
                confirm_rows.append(res)

        if len(method_pairs):
            print("Confirmatory re-tests (method pairs)...", flush=True)
            for _, row in tqdm(method_pairs.iterrows(), total=len(method_pairs), desc="Confirmatory method pairs"):
                model = row.get("model", "")
                ma = row.get("method_a", "")
                mb = row.get("method_b", "")
                metric = row.get("metric", "")
                binning = row.get("binning_strategy", "") or "equal_width"
                df_a = all_preds.get((model, ma))
                df_b = all_preds.get((model, mb))
                if df_a is None or df_b is None:
                    continue
                a, b = _align_by_instance(df_a, df_b)
                block_ids = None
                if icc_flag and "student_id" in a.columns:
                    block_ids = a["student_id"].to_numpy(dtype=str)
                res = _paired_compare(
                    a,
                    b,
                    metric,
                    rng,
                    args.confirmatory_perm,
                    0,
                    args.confirmatory_alpha,
                    args.ci_method,
                    args.max_jackknife,
                    args.n_bins,
                    binning,
                    block_ids,
                    mixed_model_flag,
                    None,
                )
                res.update({
                    "comparison_type": "method",
                    "model": model,
                    "method_a": ma,
                    "method_b": mb,
                    "seed": args.seed,
                    "confirmatory_scope": args.confirmatory_summary_scope,
                })
                confirm_rows.append(res)

        confirmatory_summary_df = pd.DataFrame(confirm_rows)
        if len(confirmatory_summary_df):
            confirmatory_summary_df["significant"] = confirmatory_summary_df["p_value"] < args.confirmatory_alpha

    if args.print_results:
        _print_table("metrics_summary", metrics_summary, args.print_max_rows)
        _print_table("model_comparisons", model_df, args.print_max_rows)
        _print_table("method_comparisons", method_df, args.print_max_rows)
        _print_table("method_summary", method_summary_df, args.print_max_rows)
        _print_table("model_summary", model_summary_df, args.print_max_rows)
        _print_table("best_method_summary", best_method_summary, args.print_max_rows)
        _print_table("best_model_summary", best_model_summary, args.print_max_rows)
        _print_table("calibration_regression_interactions", cal_reg_interaction_df, args.print_max_rows)
        _print_table("calibration_regression_summary", cal_reg_summary_df, args.print_max_rows)
        _print_table("confirmatory_summary", confirmatory_summary_df, args.print_max_rows)
        _print_table("omnibus_tests", omnibus_df, args.print_max_rows)
        _print_table("interaction_tests", interaction_df, args.print_max_rows)
        _print_table("heterogeneity", hetero_df, args.print_max_rows)
        _print_table("confirmatory_tests", confirmatory_df, args.print_max_rows)

    # Output
    csv_timestamp = output_manager.config.run_id if output_manager else time.strftime("%Y%m%d-%H%M%S")
    file_suffix = "" if output_manager else f"_{csv_timestamp}"
    legacy_analysis_dir = Path(project_root) / "experiments" / "analysis" / "cross_comparison"
    legacy_plots_dir = legacy_analysis_dir / "plots"
    step6_tables_dir = output_manager.step_dir("step6", "tables") if output_manager else legacy_analysis_dir
    step6_plots_dir = output_manager.step_dir("step6", "plots") if output_manager else legacy_plots_dir
    step6_reports_dir = output_manager.step_dir("step6", "reports") if output_manager else legacy_analysis_dir

    if not args.no_save:
        _save_table(
            metrics_summary,
            step6_tables_dir / f"metrics_summary{file_suffix}.csv",
            output_manager=output_manager,
            artifact="metrics_summary",
            legacy_path=legacy_analysis_dir / f"metrics_summary_{csv_timestamp}.csv",
        )
        _save_table(
            model_df,
            step6_tables_dir / f"model_comparisons{file_suffix}.csv",
            output_manager=output_manager,
            artifact="model_comparisons",
            legacy_path=legacy_analysis_dir / f"model_comparisons_{csv_timestamp}.csv",
        )
        _save_table(
            method_df,
            step6_tables_dir / f"method_comparisons{file_suffix}.csv",
            output_manager=output_manager,
            artifact="method_comparisons",
            legacy_path=legacy_analysis_dir / f"method_comparisons_{csv_timestamp}.csv",
        )
        _save_table(
            omnibus_df,
            step6_tables_dir / f"omnibus_tests{file_suffix}.csv",
            output_manager=output_manager,
            artifact="omnibus_tests",
            legacy_path=legacy_analysis_dir / f"omnibus_tests_{csv_timestamp}.csv",
        )
        _save_table(
            interaction_df,
            step6_tables_dir / f"interaction_tests{file_suffix}.csv",
            output_manager=output_manager,
            artifact="interaction_tests",
            legacy_path=legacy_analysis_dir / f"interaction_tests_{csv_timestamp}.csv",
        )
        _save_table(
            hetero_df,
            step6_tables_dir / f"heterogeneity{file_suffix}.csv",
            output_manager=output_manager,
            artifact="heterogeneity",
            legacy_path=legacy_analysis_dir / f"heterogeneity_{csv_timestamp}.csv",
        )
        if len(cal_reg_interaction_df):
            _save_table(
                cal_reg_interaction_df,
                step6_tables_dir / f"calibration_regression_interactions{file_suffix}.csv",
                output_manager=output_manager,
                artifact="calibration_regression_interactions",
                legacy_path=legacy_analysis_dir / f"calibration_regression_interactions_{csv_timestamp}.csv",
            )
        if len(cal_reg_summary_df):
            _save_table(
                cal_reg_summary_df,
                step6_tables_dir / f"calibration_regression_summary{file_suffix}.csv",
                output_manager=output_manager,
                artifact="calibration_regression_summary",
                legacy_path=legacy_analysis_dir / f"calibration_regression_summary_{csv_timestamp}.csv",
            )
        if len(cal_reg_curve_df):
            _save_table(
                cal_reg_curve_df,
                step6_tables_dir / f"calibration_regression_curves{file_suffix}.csv",
                output_manager=output_manager,
                artifact="calibration_regression_curves",
                legacy_path=legacy_analysis_dir / f"calibration_regression_curves_{csv_timestamp}.csv",
            )
        _save_table(
            method_summary_df,
            step6_tables_dir / f"method_summary{file_suffix}.csv",
            output_manager=output_manager,
            artifact="method_summary",
            legacy_path=legacy_analysis_dir / f"method_summary_{csv_timestamp}.csv",
        )
        _save_table(
            model_summary_df,
            step6_tables_dir / f"model_summary{file_suffix}.csv",
            output_manager=output_manager,
            artifact="model_summary",
            legacy_path=legacy_analysis_dir / f"model_summary_{csv_timestamp}.csv",
        )
        _save_table(
            best_method_summary,
            step6_tables_dir / f"best_method_summary{file_suffix}.csv",
            output_manager=output_manager,
            artifact="best_method_summary",
            legacy_path=legacy_analysis_dir / f"best_method_summary_{csv_timestamp}.csv",
        )
        _save_table(
            best_model_summary,
            step6_tables_dir / f"best_model_summary{file_suffix}.csv",
            output_manager=output_manager,
            artifact="best_model_summary",
            legacy_path=legacy_analysis_dir / f"best_model_summary_{csv_timestamp}.csv",
        )
        if len(confirmatory_df):
            _save_table(
                confirmatory_df,
                step6_tables_dir / f"confirmatory_tests{file_suffix}.csv",
                output_manager=output_manager,
                artifact="confirmatory_tests",
                legacy_path=legacy_analysis_dir / f"confirmatory_tests_{csv_timestamp}.csv",
            )
        if len(confirmatory_summary_df):
            _save_table(
                confirmatory_summary_df,
                step6_tables_dir / f"confirmatory_summary_{args.confirmatory_summary_scope}{file_suffix}.csv",
                output_manager=output_manager,
                artifact="confirmatory_summary",
                legacy_path=legacy_analysis_dir / f"confirmatory_summary_{args.confirmatory_summary_scope}_{csv_timestamp}.csv",
            )

        summary_lines = [
            "# Step 6 Cross-Comparison Summary\n\n",
            f"- Timestamp: {csv_timestamp}\n",
            f"- Models: {', '.join(args.models)}\n",
            f"- Methods: {', '.join(args.methods)}\n",
            f"- Metrics: {', '.join(args.metrics)}\n",
            f"- Permutations: {args.n_perm}\n",
            f"- Bootstraps: {args.n_boot}\n",
            f"- CI method: {args.ci_method}\n",
            f"- Interaction method: {'skipped' if args.skip_interaction_tests else args.interaction_method}\n",
            f"- Student-block sensitivity: {'ON' if icc_flag else 'OFF'}\n",
            f"- Mixed-model sensitivity: {'ON' if mixed_model_flag else 'OFF'}\n",
            f"- Model comparisons: {len(model_df)}\n",
            f"- Method comparisons: {len(method_df)}\n",
            f"- Omnibus tests: {len(omnibus_df)}\n",
            f"- Interaction tests: {len(interaction_df)}\n",
            f"- Heterogeneity summaries: {len(hetero_df)}\n",
            f"- Calibration regression interactions: {len(cal_reg_interaction_df)}\n",
            f"- Calibration regression summaries: {len(cal_reg_summary_df)}\n",
        ]
        if args.calibration_regression:
            summary_lines.append(
                f"- Calibration regression mode: {args.calibration_regression_mode}\n"
            )
        if len(confirmatory_df):
            summary_lines.append(f"- Confirmatory tests: {len(confirmatory_df)}\n")
        summary_path = step6_reports_dir / f"summary_report{file_suffix}.md"
        legacy_summary_path = legacy_analysis_dir / f"summary_report_{csv_timestamp}.md"
        _save_text(
            "".join(summary_lines),
            summary_path,
            output_manager=output_manager,
            artifact="summary_report",
            legacy_path=legacy_summary_path,
        )

    if args.plot_all:
        for metric in [m for m in args.metrics if m in GROUP_METRICS]:
            plot_metrics_heatmap(metrics_summary, metric, step6_plots_dir)
        if len(model_df):
            model_df = model_df.copy()
            if "model_a" in model_df.columns and "model_b" in model_df.columns:
                model_df["label"] = model_df["model_a"].astype(str) + " vs " + model_df["model_b"].astype(str)
            plot_forest_plot(model_df, metric="ece", output_path=step6_plots_dir, label_col="label")
        # Reliability diagram for baseline
        baseline = all_preds.get((args.models[0], args.methods[0]))
        if baseline is not None:
            conf = baseline["raw_confidence"].to_numpy(dtype=float)
            labels = baseline["is_correct"].to_numpy(dtype=float)
            from grading_calibration.metrics.cross_comparison import _bin_stats
            mean_conf, acc, bin_sizes, _ = _bin_stats(conf, labels, args.n_bins, "equal_width")
            plot_reliability_diagram({"mean_conf": mean_conf, "acc": acc}, step6_plots_dir, "Reliability Diagram", "reliability_diagram.png")
        # CD diagram (ECE) using question-level ranks when available
        if "ece" in args.metrics:
            for method in args.methods:
                group_vals = {}
                for model in args.models:
                    df = all_preds.get((model, method))
                    if df is None:
                        continue
                    group_vals[model] = _metric_by_question(df, "ece", args.n_bins, "equal_width")
                mean_ranks, n_q = _mean_ranks_from_questions(group_vals, lower_better=True)
                if mean_ranks and n_q >= 2:
                    k = len(mean_ranks)
                    cd = stats.studentized_range.ppf(1 - args.alpha, k, np.inf) * np.sqrt(k * (k + 1) / (6 * n_q))
                    plot_cd_diagram(mean_ranks, cd, step6_plots_dir, f"ECE CD (models | {method})", f"cd_models_{method}_ece.png")
            for model in args.models:
                group_vals = {}
                for method in args.methods:
                    df = all_preds.get((model, method))
                    if df is None:
                        continue
                    group_vals[method] = _metric_by_question(df, "ece", args.n_bins, "equal_width")
                mean_ranks, n_q = _mean_ranks_from_questions(group_vals, lower_better=True)
                if mean_ranks and n_q >= 2:
                    k = len(mean_ranks)
                    cd = stats.studentized_range.ppf(1 - args.alpha, k, np.inf) * np.sqrt(k * (k + 1) / (6 * n_q))
                    plot_cd_diagram(mean_ranks, cd, step6_plots_dir, f"ECE CD (methods | {model})", f"cd_methods_{model}_ece.png")

        # Calibration regression curves (if available)
        if args.calibration_regression and not cal_reg_curve_df.empty:
            for (model, method), sub in cal_reg_curve_df.groupby(["model", "method"]):
                if sub.empty:
                    continue
                sub = sub.sort_values("confidence")
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.plot(sub["confidence"], sub["pred"], color="tab:blue", label="Predicted P(correct)")
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Ideal calibration")
                ax.set_xlabel("confidence")
                ax.set_ylabel("P(correct)")
                ax.set_title(f"Calibration curve: {model} | {method}")
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.2)
                ax.legend(loc="best", fontsize=8)
                plot_path = step6_plots_dir / f"calibration_regression_curve_{model}_{method}{file_suffix}.png"
                plt.savefig(plot_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

        if output_manager and output_manager.config.write_manifest and step6_plots_dir.exists():
            for path in sorted(step6_plots_dir.glob("*.png")):
                output_manager.register(step="step6", artifact="plot", path=path)

    if output_manager:
        manifest_path = output_manager.write_manifest()
        if manifest_path:
            print(f"\nManifest saved to: {manifest_path}")

    print("Done.")


if __name__ == "__main__":
    main()
