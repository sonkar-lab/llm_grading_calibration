"""
Cross-model/method comparison utilities for Step 6.

Implements paired tests for per-instance metrics and method-swap permutation
for group-level metrics, with BCa bootstrap CIs, BY FDR by family,
student-block sensitivity, and interaction/heterogeneity helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from grading_calibration.domain.models import PredictionResult
from .calibration_bins import compute_calibration_bins
from grading_calibration.inference.model_aliases import MODEL_ALIASES as MODEL_NICKNAMES
KNOWN_DATASETS = ["ricechem", "scientsbank", "beetle"]


@dataclass
class MetricResult:
    metric: str
    value: float
    n: int
    extra: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_predictions_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ground_truth" not in df.columns and "ground_truth_label" in df.columns:
        df = df.rename(columns={"ground_truth_label": "ground_truth"})
    if "llm_response" not in df.columns:
        response_cols = [c for c in df.columns if c.startswith("llm_response_") and c[13:].isdigit()]
        if response_cols:
            ordered = sorted(response_cols)
            df["llm_response"] = df[ordered].bfill(axis=1).iloc[:, 0]
    required = ["instance_id", "prediction", "raw_confidence", "ground_truth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["ground_truth"] = pd.to_numeric(df["ground_truth"], errors="coerce")
    df["raw_confidence"] = pd.to_numeric(df["raw_confidence"], errors="coerce")
    df["is_correct"] = (df["prediction"] == df["ground_truth"]).astype(float)
    df["squared_error"] = (df["raw_confidence"] - df["is_correct"]) ** 2
    return df


def _matches_dataset(name: str, dataset: str) -> bool:
    dataset = dataset.lower()
    if dataset == "ricechem":
        return not any(f"_{d}_" in name for d in KNOWN_DATASETS if d != "ricechem")
    return f"_{dataset}_" in name


def find_latest_csv(
    pred_dir: Path,
    prefix: str,
    model_name: str,
    method: str,
    dataset: str,
    *,
    filter_by_dataset: bool = True,
    allow_exact: bool = False,
) -> Optional[Path]:
    patterns = [f"{prefix}_{model_name}_{method}_*.csv"]
    if allow_exact:
        patterns.insert(0, f"{prefix}_{model_name}_{method}.csv")
    paths: list[Path] = []
    
    for pattern in patterns:
        paths.extend(pred_dir.glob(pattern))
    paths = sorted(set(paths))

    if filter_by_dataset:
        paths = [p for p in paths if _matches_dataset(p.name, dataset)]
    return paths[-1] if paths else None


def load_all_predictions(
    models: List[str],
    methods: List[str],
    predictions_dir: Path,
    *,
    prefix: str = "predictions",
    dataset: str = "ricechem",
    flat_layout: bool = False,
    filter_by_dataset: bool = True,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    all_preds: Dict[Tuple[str, str], pd.DataFrame] = {}
    for method in methods:
        method_dir = predictions_dir if flat_layout else predictions_dir / method
        print(f"Looking for predictions in {method_dir} for method={method}")
        for model in models:
            model_name = MODEL_NICKNAMES.get(model.lower(), model.lower())
            csv_path = find_latest_csv(
                method_dir,
                prefix,
                model_name,
                method,
                dataset,
                filter_by_dataset=filter_by_dataset,
                allow_exact=flat_layout,
            )
            if csv_path is None:
                continue
            print(f"Loading predictions for model={model}, method={method} from {csv_path}")
            all_preds[(model, method)] = load_predictions_df(csv_path)
    return all_preds


# ---------------------------------------------------------------------------
# Metric computation (arrays)
# ---------------------------------------------------------------------------

def _bin_stats(conf: np.ndarray, labels: np.ndarray, n_bins: int, strategy: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "equal_width":
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == "equal_frequency":
        bin_boundaries = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning_strategy: {strategy}")
    mean_confidences = []
    accuracies = []
    bin_sizes = []
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (conf >= lower) & (conf <= upper)
        else:
            in_bin = (conf > lower) & (conf <= upper)
        if in_bin.sum() > 0:
            mean_confidences.append(conf[in_bin].mean())
            accuracies.append(labels[in_bin].mean())
            bin_sizes.append(in_bin.sum())
    return np.asarray(mean_confidences), np.asarray(accuracies), np.asarray(bin_sizes), bin_boundaries


def compute_ece_mce(conf: np.ndarray, labels: np.ndarray, *, n_bins: int = 10, strategy: str = "equal_width") -> Tuple[float, float]:
    mean_conf, acc, bin_sizes, _ = _bin_stats(conf, labels, n_bins, strategy)
    if len(bin_sizes) == 0:
        return np.nan, np.nan
    weights = bin_sizes / bin_sizes.sum()
    cal_errors = np.abs(mean_conf - acc)
    ece = float(np.sum(weights * cal_errors))
    mce = float(np.max(cal_errors))
    return ece, mce


def compute_auc(conf: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, conf))


def compute_brier_mean(conf: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((conf - labels) ** 2))


def compute_accuracy(labels: np.ndarray) -> float:
    return float(np.mean(labels))


def accuracy_at_coverage(conf: np.ndarray, labels: np.ndarray, coverage: float) -> float:
    n_selected = int(coverage * len(conf))
    if n_selected == 0:
        return np.nan
    idx = np.argsort(-conf)[:n_selected]
    return float(labels[idx].mean())


def coverage_at_risk(conf: np.ndarray, labels: np.ndarray, target_risk: float) -> float:
    if len(conf) == 0:
        return np.nan
    order = np.argsort(-conf)
    sorted_labels = labels[order]
    n = len(sorted_labels)
    coverages = np.linspace(0, 1, 1000)
    risks = []
    for cov in coverages:
        k = int(cov * n)
        if k == 0:
            risks.append(0.0)
        else:
            risk = 1 - sorted_labels[:k].mean()
            risks.append(risk)
    risks = np.asarray(risks)
    valid = coverages[risks <= target_risk]
    if len(valid) == 0:
        return 0.0
    return float(valid[-1])


# ---------------------------------------------------------------------------
# Effect sizes and inference
# ---------------------------------------------------------------------------

def paired_cliffs_delta(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs)
    n = len(diffs)
    if n == 0:
        return np.nan
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    return float((n_pos - n_neg) / n)


def dz_effect(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs)
    if len(diffs) < 2:
        return np.nan
    sd = np.std(diffs, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float(np.mean(diffs) / sd)


def by_fdr(pvals: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals, np.zeros(0, dtype=bool)
    mask = np.isfinite(pvals)
    if not np.any(mask):
        return np.full_like(pvals, np.nan), np.zeros_like(pvals, dtype=bool)
    m_eff = int(mask.sum())
    idx = np.where(mask)[0]
    order = np.argsort(pvals[mask])
    ranked = pvals[mask][order]
    q = np.sum(1.0 / np.arange(1, m_eff + 1))
    adj = ranked * m_eff * q / np.arange(1, m_eff + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    p_adj = np.full_like(pvals, np.nan)
    p_adj[idx[order]] = adj
    sig = np.zeros_like(pvals, dtype=bool)
    sig[mask] = p_adj[mask] < alpha
    return p_adj, sig


def paired_sign_flip_permutation(
    diffs: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    block_ids: Optional[np.ndarray] = None,
) -> float:
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return np.nan
    obs = float(np.mean(diffs))
    if block_ids is None:
        signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
        perm_means = (signs * diffs).mean(axis=1)
    else:
        block_ids = np.asarray(block_ids)
        uniq = np.unique(block_ids)
        block_map = {b: np.where(block_ids == b)[0] for b in uniq}
        perm_means = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm = diffs.copy()
            for b in uniq:
                sign = rng.choice([-1, 1])
                perm[block_map[b]] *= sign
            perm_means[i] = perm.mean()
    p_val = (np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1)
    return float(p_val)


def paired_group_metric_permutation(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    labels: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_perm: int,
    rng: np.random.Generator,
    block_ids: Optional[np.ndarray] = None,
) -> float:
    obs = metric_func(conf_a, labels) - metric_func(conf_b, labels)
    diffs = np.empty(n_perm, dtype=float)
    n = len(labels)
    if block_ids is None:
        for i in range(n_perm):
            mask = rng.choice([False, True], size=n)
            perm_a = np.where(mask, conf_b, conf_a)
            perm_b = np.where(mask, conf_a, conf_b)
            diffs[i] = metric_func(perm_a, labels) - metric_func(perm_b, labels)
    else:
        block_ids = np.asarray(block_ids)
        uniq = np.unique(block_ids)
        block_mask = {b: block_ids == b for b in uniq}
        for i in range(n_perm):
            perm_a = conf_a.copy()
            perm_b = conf_b.copy()
            for b in uniq:
                if rng.choice([False, True]):
                    m = block_mask[b]
                    perm_a[m], perm_b[m] = perm_b[m], perm_a[m]
            diffs[i] = metric_func(perm_a, labels) - metric_func(perm_b, labels)
    p_val = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    return float(p_val)


def bootstrap_ci(
    stats: np.ndarray,
    alpha: float = 0.05,
    method: str = "bca",
    observed: Optional[float] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    stat_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Tuple[float, float]:
    if len(stats) == 0:
        return np.nan, np.nan
    if method == "percentile" or observed is None or x is None or y is None or stat_func is None:
        return (float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2))))

    n_boot = len(stats)
    p0 = np.sum(stats < observed) / n_boot
    if p0 == 0:
        p0 = 0.5 / n_boot
    elif p0 == 1:
        p0 = 1 - 0.5 / n_boot
    z0 = stats_norm_ppf(p0)

    n = len(x)
    jack_stats = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack_stats[i] = stat_func(x[mask], y[mask])
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = 0.0 if denom == 0 else num / denom

    z_low = stats_norm_ppf(alpha / 2)
    z_high = stats_norm_ppf(1 - alpha / 2)
    p_low = stats_norm_cdf(z0 + (z0 + z_low) / (1 - a * (z0 + z_low)))
    p_high = stats_norm_cdf(z0 + (z0 + z_high) / (1 - a * (z0 + z_high)))
    p_low = float(np.clip(p_low, 0, 1))
    p_high = float(np.clip(p_high, 0, 1))
    return float(np.percentile(stats, 100 * p_low)), float(np.percentile(stats, 100 * p_high))


def stats_norm_ppf(x: float) -> float:
    return float(stats.norm.ppf(x))


def stats_norm_cdf(x: float) -> float:
    return float(stats.norm.cdf(x))


def bootstrap_differences(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    block_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    diffs = values_a - values_b
    n = len(diffs)
    if n == 0:
        return np.array([])
    if block_ids is None:
        idx = rng.choice(n, size=(n_boot, n), replace=True)
        return diffs[idx].mean(axis=1)
    uniq = np.unique(block_ids)
    block_map = {b: np.where(block_ids == b)[0] for b in uniq}
    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_blocks = rng.choice(uniq, size=len(uniq), replace=True)
        sampled_idx = np.concatenate([block_map[b] for b in sampled_blocks])
        boot_stats[i] = diffs[sampled_idx].mean()
    return boot_stats


def bootstrap_group_metric(
    conf: np.ndarray,
    labels: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    rng: np.random.Generator,
    block_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = len(labels)
    if n == 0:
        return np.array([])
    if block_ids is None:
        idx = rng.choice(n, size=(n_boot, n), replace=True)
        return np.array([metric_func(conf[i], labels[i]) for i in idx])
    uniq = np.unique(block_ids)
    block_map = {b: np.where(block_ids == b)[0] for b in uniq}
    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_blocks = rng.choice(uniq, size=len(uniq), replace=True)
        sampled_idx = np.concatenate([block_map[b] for b in sampled_blocks])
        boot_stats[i] = metric_func(conf[sampled_idx], labels[sampled_idx])
    return boot_stats


def bootstrap_group_metric_diff(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    labels: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    rng: np.random.Generator,
    block_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = len(labels)
    if n == 0:
        return np.array([])
    if block_ids is None:
        idx = rng.choice(n, size=(n_boot, n), replace=True)
        return np.array([metric_func(conf_a[i], labels[i]) - metric_func(conf_b[i], labels[i]) for i in idx])
    uniq = np.unique(block_ids)
    block_map = {b: np.where(block_ids == b)[0] for b in uniq}
    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_blocks = rng.choice(uniq, size=len(uniq), replace=True)
        sampled_idx = np.concatenate([block_map[b] for b in sampled_blocks])
        boot_stats[i] = metric_func(conf_a[sampled_idx], labels[sampled_idx]) - metric_func(conf_b[sampled_idx], labels[sampled_idx])
    return boot_stats


def _bca_interval(
    boot_stats: np.ndarray,
    observed: float,
    jack_stats: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    n_boot = len(boot_stats)
    if n_boot == 0 or len(jack_stats) == 0 or not np.isfinite(observed):
        return np.nan, np.nan
    p0 = np.sum(boot_stats < observed) / n_boot
    if p0 == 0:
        p0 = 0.5 / n_boot
    elif p0 == 1:
        p0 = 1 - 0.5 / n_boot
    z0 = stats_norm_ppf(p0)
    jack_mean = float(np.mean(jack_stats))
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = 0.0 if denom == 0 else float(num / denom)

    z_low = stats_norm_ppf(alpha / 2)
    z_high = stats_norm_ppf(1 - alpha / 2)
    p_low = stats_norm_cdf(z0 + (z0 + z_low) / (1 - a * (z0 + z_low)))
    p_high = stats_norm_cdf(z0 + (z0 + z_high) / (1 - a * (z0 + z_high)))
    p_low = float(np.clip(p_low, 0, 1))
    p_high = float(np.clip(p_high, 0, 1))
    return float(np.percentile(boot_stats, 100 * p_low)), float(np.percentile(boot_stats, 100 * p_high))


def _jackknife_mean(diffs: np.ndarray, block_ids: Optional[np.ndarray] = None) -> np.ndarray:
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n <= 1:
        return np.array([])
    if block_ids is None:
        total = float(np.sum(diffs))
        return (total - diffs) / (n - 1)
    block_ids = np.asarray(block_ids)
    uniq = np.unique(block_ids)
    jack = []
    total = float(np.sum(diffs))
    for b in uniq:
        mask = block_ids != b
        n_keep = int(mask.sum())
        if n_keep <= 1:
            continue
        jack.append((total - float(np.sum(diffs[~mask]))) / n_keep)
    return np.asarray(jack, dtype=float)


def bca_ci_paired_diffs(
    diffs: np.ndarray,
    boot_stats: np.ndarray,
    alpha: float,
    block_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if len(boot_stats) == 0 or len(diffs) == 0:
        return np.nan, np.nan
    observed = float(np.mean(diffs))
    jack_stats = _jackknife_mean(diffs, block_ids)
    if len(jack_stats) == 0:
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    return _bca_interval(boot_stats, observed, jack_stats, alpha)


def _jackknife_group_metric_diff(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    labels: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    block_ids: Optional[np.ndarray],
    max_jackknife: int,
) -> np.ndarray:
    n = len(labels)
    if n <= 1:
        return np.array([])
    if block_ids is None:
        if n > max_jackknife:
            return np.array([])
        jack_stats = np.empty(n, dtype=float)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jack_stats[i] = metric_func(conf_a[mask], labels[mask]) - metric_func(conf_b[mask], labels[mask])
        return jack_stats
    block_ids = np.asarray(block_ids)
    uniq = np.unique(block_ids)
    jack = []
    for b in uniq:
        mask = block_ids != b
        if mask.sum() <= 1:
            continue
        jack.append(metric_func(conf_a[mask], labels[mask]) - metric_func(conf_b[mask], labels[mask]))
    return np.asarray(jack, dtype=float)


def bca_ci_group_diff(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    labels: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    boot_stats: np.ndarray,
    alpha: float,
    block_ids: Optional[np.ndarray] = None,
    max_jackknife: int = 1000,
) -> Tuple[float, float]:
    if len(boot_stats) == 0 or len(labels) == 0:
        return np.nan, np.nan
    observed = float(metric_func(conf_a, labels) - metric_func(conf_b, labels))
    jack_stats = _jackknife_group_metric_diff(conf_a, conf_b, labels, metric_func, block_ids, max_jackknife)
    if len(jack_stats) == 0:
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    return _bca_interval(boot_stats, observed, jack_stats, alpha)


# ---------------------------------------------------------------------------
# Interaction and heterogeneity helpers
# ---------------------------------------------------------------------------

def art_interaction_test(
    values: np.ndarray,
    model_labels: np.ndarray,
    method_labels: np.ndarray,
    *,
    student_ids: Optional[np.ndarray] = None,
    method: str = "artc",
) -> Dict[str, Any]:
    """
    Interaction test for model × method.

    Default: ART-C (random intercept for student_id) via ARTool.
    Fallback: ART ANOVA approximation on aligned ranks.
    """
    warnings_list: List[str] = []
    if method == "artc":
        try:
            from utils.statistics import art_c_interaction_test

            res = art_c_interaction_test(values, model_labels, method_labels, student_ids)
            if np.isfinite(res.get("p_interaction", np.nan)):
                res["method_used"] = "artc"
                return res
            warnings_list.extend(res.get("warnings", []))
        except Exception as e:
            warnings_list.append(f"artc_failed: {e}")

    # ART ANOVA approximation
    try:
        import statsmodels.api as sm
        import pandas as pd
    except ImportError:
        return {"p_interaction": np.nan, "n": 0, "method_used": "art_approx", "warnings": warnings_list}

    df = pd.DataFrame(
        {
            "y": values,
            "model": model_labels,
            "method": method_labels,
        }
    ).dropna()
    if len(df) < 10:
        return {
            "p_interaction": np.nan,
            "n": int(len(df)),
            "method_used": "art_approx",
            "warnings": warnings_list + ["insufficient_rows"],
        }

    grand_mean = df["y"].mean()
    model_means = df.groupby("model")["y"].transform("mean")
    method_means = df.groupby("method")["y"].transform("mean")
    aligned = df["y"] - model_means - method_means + grand_mean
    df["aligned_rank"] = stats.rankdata(aligned.to_numpy())

    formula = "aligned_rank ~ C(model) * C(method)"
    model = sm.OLS.from_formula(formula, data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    p_int = np.nan
    if "C(model):C(method)" in anova.index:
        p_int = float(anova.loc["C(model):C(method)", "PR(>F)"])
    return {
        "p_interaction": p_int,
        "n": int(len(df)),
        "method_used": "art_approx",
        "warnings": warnings_list,
    }


def compute_i2(effect: np.ndarray, se: np.ndarray) -> float:
    effect = np.asarray(effect, dtype=float)
    se = np.asarray(se, dtype=float)
    mask = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    if mask.sum() < 2:
        return np.nan
    w = 1 / (se[mask] ** 2)
    mean_eff = np.sum(w * effect[mask]) / np.sum(w)
    q = np.sum(w * (effect[mask] - mean_eff) ** 2)
    df = mask.sum() - 1
    if q <= df:
        return 0.0
    return float((q - df) / q)


def practical_threshold(metric: str) -> Tuple[float, float]:
    if metric in {"ece", "mce"}:
        return 0.02, 0.05
    if metric in {"brier_mean", "brier"}:
        return 0.01, 0.03
    if metric in {"auc_roc"}:
        return 0.03, 0.05
    if metric.startswith("cov_at") or metric.startswith("acc_at"):
        return 0.05, 0.10
    return np.nan, np.nan


def warn_sample_size(n_pairs: int) -> str:
    if n_pairs < 10:
        return "skip"
    if n_pairs < 100:
        return "warn"
    return "ok"


def missing_action(missing_rate: float) -> str:
    if missing_rate > 0.20:
        return "exclude"
    if missing_rate > 0.05:
        return "sensitivity"
    return "complete_case"


def sensitivity_bounds(diffs: np.ndarray, n_missing: int) -> Tuple[float, float]:
    if n_missing <= 0 or len(diffs) == 0:
        return np.nan, np.nan
    d_min = float(np.min(diffs))
    d_max = float(np.max(diffs))
    mean_obs = float(np.mean(diffs))
    if mean_obs >= 0:
        best = (diffs.sum() + n_missing * d_max) / (len(diffs) + n_missing)
        worst = (diffs.sum() + n_missing * d_min) / (len(diffs) + n_missing)
    else:
        best = (diffs.sum() + n_missing * d_min) / (len(diffs) + n_missing)
        worst = (diffs.sum() + n_missing * d_max) / (len(diffs) + n_missing)
    return float(best), float(worst)
