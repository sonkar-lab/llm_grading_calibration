"""
Statistical utilities for calibration research.

This module provides:
- Power analysis for correlations and t-tests
- Multiple testing correction (FDR)
- Effect size interpretation
- Statistical assumption checking

These functions help ensure statistical rigor in feature analysis.
"""

import numpy as np
import pandas as pd
import warnings
import sys
import os
import json
import shutil
import textwrap
from pathlib import Path
from scipy import stats
from typing import Tuple, List, Optional, Dict, Any
from typing import Callable
import statsmodels.api as sm
from scipy.stats import rankdata, norm
from dataclasses import dataclass, field
from typing import Literal
from tqdm import tqdm
from grading_calibration.statistics.fdr import (
    benjamini_hochberg as _stable_benjamini_hochberg,
    is_na_sentinel,
    significant_at,
)

# =============================================================================
# Power Analysis
# =============================================================================

def correlation_power(r: float, n: int, alpha: float = 0.05) -> float:
    """
    Compute statistical power for detecting a correlation.
    
    Uses Fisher's z-transformation approach.
    
    Args:
        r: Observed or expected correlation coefficient
        n: Sample size
        alpha: Significance level (two-tailed)
        
    Returns:
        Statistical power (probability of detecting effect if real)
        
    Example:
        >>> power = correlation_power(r=0.3, n=50)
        >>> print(f"Power: {power:.2f}")  # Power: 0.57
    """
    # Handle edge cases
    if n < 4:
        return 0.0  # Insufficient sample size

    if abs(r) >= 0.9999999999:
        # Fisher z-transform undefined at r = ±1
        warnings.warn(
            f"Correlation r={r:.4f} too close to ±1 for power calculation. "
            "Fisher z-transformation is undefined. Returning NaN.",
            stacklevel=2
        )
        return np.nan
    
    # Fisher z-transform
    z_r = 0.5 * np.log((1 + abs(r)) / (1 - abs(r)))
    
    # Standard error of z
    se_z = 1 / np.sqrt(n - 3)
    
    # Critical z for alpha (two-tailed)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # Non-centrality parameter
    ncp = z_r / se_z
    
    # Power = P(reject H0 | H1 true)
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    
    return float(power)


def required_sample_size_correlation(
    r: float,
    power: float = 0.8,
    alpha: float = 0.05,
    max_n: int = 10_000_000
) -> int:
    """
    Compute sample size needed to detect a correlation with desired power.

    Uses Fisher's z-transformation and binary search.

    Args:
        r: Expected correlation coefficient to detect
        power: Desired statistical power (default 0.8)
        alpha: Significance level (default 0.05)
        max_n: Maximum sample size to search up to (default 10,000,000)

    Returns:
        Required sample size

    Raises:
        ValueError: If abs(r) >= 0.999 (Fisher transform undefined)
        ValueError: If required n exceeds max_n

    Example:
        >>> n = required_sample_size_correlation(r=0.3, power=0.8)
        >>> print(f"Need n={n} to detect r=0.3 with 80% power")  # n=84
    """
    # Validate inputs
    if not 0 < power < 1:
        raise ValueError(f"Power must be in (0, 1), got {power}")
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    # Handle extreme correlations
    if abs(r) >= 0.999:
        raise ValueError(
            f"Correlation r={r:.4f} too close to ±1. "
            "Fisher z-transformation is undefined. "
            "Check for data quality issues (duplicate rows, deterministic relationships)."
        )

    if abs(r) < 0.0001:
        raise ValueError(
            f"Correlation r={r:.4f} too weak for practical detection. "
            "Required sample size likely exceeds 10 million."
        )

    # Check if max_n would be sufficient before running full search
    if abs(r) < 0.001:  # Threshold where n > 10M typically needed
        test_power = correlation_power(r, max_n, alpha)
        if np.isnan(test_power) or test_power < power:
            raise ValueError(
                f"Correlation r={r:.4f} requires n > {max_n:,} (exceeds search max) for {power*100:.0f}% power. "
                f"At n={max_n:,}, achieved power is only {test_power:.3f}. "
                "This effect size may be impractical to detect even with synthetic data."
            )

    # Binary search for required n
    low, high = 4, max_n

    while low < high:
        mid = (low + high) // 2
        achieved_power = correlation_power(r, mid, alpha)

        # Handle NaN from correlation_power (e.g., if r too close to 1)
        if np.isnan(achieved_power):
            raise ValueError(
                f"Power calculation returned NaN for r={r:.4f}, n={mid}. "
                "Check correlation value."
            )

        if achieved_power >= power:
            high = mid
        else:
            low = mid + 1

    # Warn if we hit the upper bound (might be underestimate)
    if low >= max_n - 1:
        final_power = correlation_power(r, low, alpha)
        warnings.warn(
            f"Required sample size n={low:,} is at upper search bound. "
            f"True n might be larger. At n={low:,}, achieved power is {final_power:.3f}.",
            stacklevel=2
        )

    return low


def ttest_power(
    effect_size: float, 
    n1: int, 
    n2: Optional[int] = None, 
    alpha: float = 0.05
) -> float:
    """
    Compute statistical power for a two-sample t-test.
    
    Uses non-central t-distribution.
    
    Args:
        effect_size: Cohen's d effect size
        n1: Sample size of group 1
        n2: Sample size of group 2 (if None, assumes equal to n1)
        alpha: Significance level
        
    Returns:
        Statistical power
        
    Example:
        >>> power = ttest_power(effect_size=0.5, n1=30, n2=30)
        >>> print(f"Power: {power:.2f}")  # Power: 0.48
    """
    if n2 is None:
        n2 = n1
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
    
    # Critical t-value
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    
    # Power using non-central t-distribution
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    
    return float(power)


def required_sample_size_ttest(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0
) -> Tuple[int, int]:
    """
    Compute sample sizes needed for a two-sample t-test.
    
    Args:
        effect_size: Expected Cohen's d
        power: Desired power (default 0.8)
        alpha: Significance level (default 0.05)
        ratio: Ratio of n2/n1 (default 1.0 for equal groups)
        
    Returns:
        Tuple of (n1, n2)
    """
    # Binary search
    low, high = 4, 5000
    while low < high:
        mid = (low + high) // 2
        n2 = int(mid * ratio)
        achieved_power = ttest_power(effect_size, mid, n2, alpha)
        if achieved_power >= power:
            high = mid
        else:
            low = mid + 1
    
    return low, int(low * ratio)


# =============================================================================
# Multiple Testing Correction
# =============================================================================

def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
    *,
    inclusive: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.
    
    Args:
        p_values: Array of p-values
        alpha: False discovery rate threshold
        inclusive: Use <= when True; default False preserves legacy strict < behavior.
        
    Returns:
        Tuple of (adjusted_p_values, significant_mask)
        
    Example:
        >>> p_vals = np.array([0.001, 0.01, 0.03, 0.04, 0.05])
        >>> adj_p, sig = benjamini_hochberg(p_vals)
        >>> print(f"Significant after FDR: {sig.sum()}")
    """
    adjusted = np.asarray(_stable_benjamini_hochberg(list(p_values)), dtype=float)
    significant = np.asarray(
        [significant_at(value, alpha, inclusive=inclusive) for value in adjusted],
        dtype=bool,
    )
    return adjusted, significant


def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction (most conservative).
    
    Args:
        p_values: Array of p-values
        alpha: Family-wise error rate
        
    Returns:
        Tuple of (adjusted_p_values, significant_mask)
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    
    adjusted = np.minimum(p_values * n, 1.0)
    significant = adjusted < alpha
    
    return adjusted, significant


# =============================================================================
# Effect Size Interpretation
# =============================================================================

def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size using conventional thresholds.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def interpret_correlation(r: float) -> str:
    """
    Interpret correlation coefficient strength.
    
    Args:
        r: Correlation coefficient
        
    Returns:
        Interpretation string
    """
    r_abs = abs(r)
    if r_abs < 0.1:
        return "negligible"
    elif r_abs < 0.3:
        return "weak"
    elif r_abs < 0.5:
        return "moderate"
    elif r_abs < 0.7:
        return "strong"
    else:
        return "very strong"


def compute_icc_mixed_model(
    values: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute ICC-style variance decomposition aligned with the Step-3 LMM estimand.

    Step-3 continuous model treats questions as fixed effects and uses crossed
    random effects for student / answer / rubric:

        value ~ C(question_id) + (1|student_id) + (1|answer_id) + (1|rubric_id)

    where answer_id = (student_id, question_id).

    Because question_id is a fixed effect in Step-3, this function does not
    estimate a question random-effect ICC; for backwards compatibility it
    returns icc_question and var_question as NaN.

    Args:
        values: Outcome variable (e.g., raw_confidence)
        student_ids: Student cluster IDs
        question_ids: Question cluster IDs
        rubric_ids: Optional rubric cluster IDs
        verbose: Print model summary

    Returns:
        Dict with keys:
        - 'icc_student', 'icc_answer', 'icc_rubric'
        - 'icc_question' (NaN; question is FE)
        - 'var_student', 'var_answer', 'var_rubric', 'var_resid', 'total_var'
        - 'var_question' (NaN; question is FE)
        - 'n_obs', 'n_students', 'n_answers', 'n_questions', 'n_rubrics'
    """
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for mixed-model ICC. Install with: pip install statsmodels"
        ) from e

    values_numeric = np.asarray(values, dtype=np.float64)
    data = pd.DataFrame(
        {
            "value": values_numeric,
            "student_id": student_ids,
            "question_id": question_ids,
        }
    )
    if rubric_ids is not None:
        data["rubric_id"] = rubric_ids

    data = data.dropna()
    if len(data) < 10:
        raise ValueError(f"Insufficient data: only {len(data)} valid observations")

    answer_key = pd.MultiIndex.from_arrays([data["student_id"], data["question_id"]])
    data["answer_id"] = pd.factorize(answer_key)[0].astype(str)

    n_students = int(pd.Series(data["student_id"]).nunique(dropna=True))
    n_answers = int(pd.Series(data["answer_id"]).nunique(dropna=True))
    n_questions = int(pd.Series(data["question_id"]).nunique(dropna=True))
    n_rubrics = int(pd.Series(data["rubric_id"]).nunique(dropna=True)) if "rubric_id" in data else 0

    if n_students < 2 or n_answers < 2:
        raise ValueError(
            f"Degenerate clustering for ICC (students={n_students}, answers={n_answers}); need ≥2."
        )
    if n_questions < 2:
        raise ValueError(f"Degenerate question_id (n={n_questions}); need ≥2 for C(question_id).")

    vc_formula: Dict[str, str] = {"answer": "0 + C(answer_id)"}
    if "rubric_id" in data and n_rubrics >= 2:
        vc_formula["rubric"] = "0 + C(rubric_id)"
    vc_names = list(vc_formula.keys())

    md = MixedLM.from_formula(
        "value ~ C(question_id)",
        groups="student_id",
        vc_formula=vc_formula,
        data=data,
        re_formula="1",
    )
    mfit = md.fit(reml=True, method="powell")

    var_student = float(mfit.cov_re.iloc[0, 0]) if getattr(mfit, "cov_re", None) is not None else 0.0
    var_resid = float(mfit.scale)

    vcomp_raw = mfit.vcomp
    if isinstance(vcomp_raw, pd.Series):
        vcomp_series = vcomp_raw
    else:
        flat = np.asarray(vcomp_raw).flatten()
        vcomp_series = pd.Series(flat, index=vc_names[: len(flat)])

    var_answer = float(vcomp_series.get("answer", 0.0)) if len(vcomp_series) else 0.0
    var_rubric = float(vcomp_series.get("rubric", 0.0)) if "rubric" in vc_formula else 0.0

    total_var = var_student + var_answer + var_rubric + var_resid
    if not np.isfinite(total_var) or total_var <= 0:
        raise ValueError("Invalid total variance in Step-3 MixedLM ICC.")

    icc_student = var_student / total_var
    icc_answer = var_answer / total_var
    icc_rubric = var_rubric / total_var

    results = {
        "icc_student": float(icc_student),
        "icc_answer": float(icc_answer),
        "icc_rubric": float(icc_rubric),
        "icc_question": float("nan"),
        "var_student": float(var_student),
        "var_answer": float(var_answer),
        "var_rubric": float(var_rubric),
        "var_question": float("nan"),
        "var_resid": float(var_resid),
        "total_var": float(total_var),
        "n_obs": int(len(data)),
        "n_students": n_students,
        "n_answers": n_answers,
        "n_questions": n_questions,
        "n_rubrics": n_rubrics,
    }

    if verbose:
        print("\nMixed Model ICC Estimation (Step-3-aligned):")
        print("  Variance Components:")
        print(f"    Student:   {results['var_student']:.6f}")
        print(f"    Answer:    {results['var_answer']:.6f}")
        if n_rubrics >= 2:
            print(f"    Rubric:    {results['var_rubric']:.6f}")
        print(f"    Residual:  {results['var_resid']:.6f}")
        print(f"    Total:     {results['total_var']:.6f}")
        print("\n  ICCs:")
        print(f"    Student:   {results['icc_student']:.4f}")
        print(f"    Answer:    {results['icc_answer']:.4f}")
        if n_rubrics >= 2:
            print(f"    Rubric:    {results['icc_rubric']:.4f}")

    return results


def estimate_feature_effect_cluster_robust(
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: Optional[np.ndarray] = None,
    rubric_ids: Optional[np.ndarray] = None,
    use_ranks: bool = True,
    standardize: bool = True,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Estimate feature-outcome association with multiway cluster-robust SEs.

    **IMPORTANT**: This is a ROBUSTNESS CHECK, not the primary inferential method
    when substantial clustering is present. For primary inference with clustered data,
    use estimate_feature_effect() which accounts for the Step-3 crossed random effects
    structure via MixedLM / BayesMixedGLM backends.

    Cluster-robust SEs are appropriate when:
    - Number of clusters is large (≥30 in each dimension)
    - ICCs are modest (<0.05)
    - You want a sensitivity check alongside mixed models

    Model specification:
        OLS: outcome ~ feature + C(question_id)
        Two-way cluster-robust SEs: student × rubric

    Clustering dimensions:
    - student_id: Always clustered (required ≥2 unique students)
    - rubric_id: Clustered if provided and ≥2 unique values
    - question_id: NOT clustered (only ~4 questions, too few for cluster-robust)
              Instead, question_id enters as fixed effects C(question_id)

    Args:
        feature: Feature values (X)
        outcome: Outcome values (Y)
        student_ids: Student cluster IDs (required, ≥2 unique)
        question_ids: Question IDs for fixed effects (not clustered)
        rubric_ids: Rubric cluster IDs (if provided, needs ≥2 unique)
        use_ranks: If True, rank-transform X and Y (Spearman-style)
        standardize: If True, standardize X and Y after ranking
        alpha: Significance level for confidence intervals

    Returns:
        Dict with keys:
        - 'beta': Regression coefficient (slope)
        - 'se': Cluster-robust standard error
        - 'z': Test statistic (beta / se)
        - 'p_value': Two-sided p-value
        - 'ci_low': Lower bound of (1-alpha)% CI
        - 'ci_high': Upper bound of (1-alpha)% CI
        - 'n_obs': Number of observations
        - 'n_clusters_student': Number of student clusters
        - 'n_clusters_question': Number of question levels (fixed effects)
        - 'n_clusters_rubric': Number of rubric clusters (if clustered)
    """
    try:
        import statsmodels.api as sm
        from scipy.stats import norm, rankdata
    except ImportError:
        raise ImportError(
            "statsmodels and scipy required for cluster-robust regression. "
            "Install with: pip install statsmodels scipy"
        )

    # Create DataFrame
    import pandas as pd

    # Explicitly convert to float64 to handle boolean/object dtypes
    feature_numeric = np.asarray(feature, dtype=np.float64)
    outcome_numeric = np.asarray(outcome, dtype=np.float64)

    # Guard: if question fixed effects are included, a feature that is constant within each question
    # is perfectly collinear with the question dummy variables (and leads to unidentifiable/exploding betas).
    if question_ids is not None:
        tmp = pd.DataFrame({"q": question_ids, "v": feature_numeric}).dropna()
        if not tmp.empty:
            nunique_by_q = tmp.groupby("q")["v"].nunique(dropna=True)
            if len(nunique_by_q) > 1 and (nunique_by_q <= 1).all():
                raise ValueError(
                    "Feature is constant within each question_id; collinear with question fixed effects."
                )

    data = pd.DataFrame({
        'feature': feature_numeric,
        'outcome': outcome_numeric,
        'student_id': student_ids,
    })
    if question_ids is not None:
        data['question_id'] = question_ids
    else:
        # If no question_id provided, create dummy (no question clustering)
        data['question_id'] = np.arange(len(data))
    if rubric_ids is not None:
        data['rubric_id'] = rubric_ids
    else:
        data['rubric_id'] = np.arange(len(data))

    # Remove NaN values
    data = data.dropna()

    if len(data) < 10:
        raise ValueError(f"Insufficient data: only {len(data)} valid observations")

    # Cluster validation
    n_unique_student = len(np.unique(data['student_id']))
    n_unique_rubric = len(np.unique(data['rubric_id'])) if rubric_ids is not None else 0

    # Hard requirement: ≥2 student clusters
    if n_unique_student < 2:
        raise ValueError(
            f"Insufficient student clusters (n={n_unique_student}); "
            "cluster-robust SEs require ≥2 clusters."
        )

    # Soft requirement: ≥2 rubric clusters if rubric_ids provided
    use_rubric_clustering = False
    if rubric_ids is not None and n_unique_rubric < 2:
        warnings.warn(
            f"rubric clustering dropped: only {n_unique_rubric} rubric cluster(s); "
            "using student-only clustering for SEs.",
            stacklevel=2
        )
    elif rubric_ids is not None and n_unique_rubric >= 2:
        use_rubric_clustering = True

    # Rank transformation (Spearman-style)
    if use_ranks:
        data['feature'] = rankdata(data['feature'])
        data['outcome'] = rankdata(data['outcome'])

    # Standardization
    if standardize:
        f_std = float(data['feature'].std())
        y_std = float(data['outcome'].std())
        if not np.isfinite(f_std) or f_std <= 0:
            raise ValueError("Feature has zero/undefined variance after preprocessing.")
        if not np.isfinite(y_std) or y_std <= 0:
            raise ValueError("Outcome has zero/undefined variance after preprocessing.")
        data['feature'] = (data['feature'] - data['feature'].mean()) / f_std
        data['outcome'] = (data['outcome'] - data['outcome'].mean()) / y_std

    # Fixed effects: intercept + feature + C(question_id) (question) dummies
    X = sm.add_constant(data['feature'])
    X = pd.concat([X, pd.get_dummies(data['question_id'], prefix='question', drop_first=True)], axis=1)
    y = data['outcome']

    # Ensure numeric dtypes for statsmodels
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    model = sm.OLS(y, X)

    # Cluster-robust standard errors (student × rubric if use_rubric_clustering=True)
    student_clusters = data['student_id'].values
    question_clusters = data['question_id'].values
    rubric_clusters = data['rubric_id'].values

    fit_student = model.fit(cov_type='cluster', cov_kwds={'groups': student_clusters})
    cov_student = fit_student.cov_params()

    if use_rubric_clustering:
        # Two-way clustering: student × rubric
        cov_rubric = model.fit(cov_type='cluster', cov_kwds={'groups': rubric_clusters}).cov_params()
        interaction_sr = pd.Series(student_clusters.astype(str) + '_' + rubric_clusters.astype(str)).values
        cov_interaction = model.fit(cov_type='cluster', cov_kwds={'groups': interaction_sr}).cov_params()
        cov_multi = cov_student + cov_rubric - cov_interaction
    else:
        # Student-only clustering
        cov_multi = cov_student

    # Extract coefficient and SE for feature (index 1, not intercept)
    beta = fit_student.params["feature"]  # Same across all fits
    se = np.sqrt(cov_multi.iloc[1, 1])  # Diagonal element for feature

    # Test statistic and p-value
    z_stat = beta / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))  # Two-sided

    # Confidence interval
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = beta - z_crit * se
    ci_high = beta + z_crit * se

    # Cluster counts
    n_clusters_student = len(np.unique(student_clusters))
    n_clusters_question = len(np.unique(question_clusters))
    n_clusters_rubric = len(np.unique(rubric_clusters))

    return {
        'beta': float(beta),
        'se': float(se),
        'z': float(z_stat),
        'p_value': float(p_value),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n_obs': int(len(data)),
        'n_clusters_student': int(n_clusters_student),
        'n_clusters_question': int(n_clusters_question),
        'n_clusters_rubric': int(n_clusters_rubric),
    }


def estimate_feature_effect_mixed_model(
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: Optional[np.ndarray] = None,
    use_ranks: bool = True,
    standardize: bool = True,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Deprecated legacy estimator (question as random effect).

    This function was written for an older estimand that treated question_id as a
    random intercept. Step-3 screening now uses `estimate_feature_effect()` which
    matches the current dataset structure (answer_id and rubric_id random effects
    with question fixed effects).

    Use `estimate_feature_effect(...)` instead.
    """
    raise RuntimeError(
        "estimate_feature_effect_mixed_model() is deprecated; use estimate_feature_effect()."
    )


def estimate_question_level_effect_mixed_model_bootstrap(
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: Optional[np.ndarray] = None,
    use_ranks: bool = True,
    standardize: bool = True,
    alpha: float = 0.05,
    n_boot: int = 200,
    random_state: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Estimate a question-level effect with cross-classified random intercepts (student, question[, rubric])
    and cluster bootstrap over questions for small-cluster uncertainty.

    This is intentionally retained as a *special-case* estimator for question-constant features:
      - If a predictor is constant within question_id, it is collinear with C(question_id) fixed effects.
      - With only ~4 questions, uncertainty must be treated as coarse; we bootstrap over questions.

    This is a different estimand than the Step-3 primary mixed model (`estimate_feature_effect()`), and
    should not be used as a competing inference engine for general row-varying features.
    """
    rng = random_state or np.random.default_rng()
    try:
        import statsmodels.api as sm
        from scipy.stats import norm, rankdata
    except ImportError:
        raise ImportError("statsmodels and scipy are required for mixed-model estimation.")

    # Assemble data frame
    data = pd.DataFrame({
        'feature': np.asarray(feature, dtype=np.float64),
        'outcome': np.asarray(outcome, dtype=np.float64),
        'student_id': student_ids,
        'question_id': question_ids,
    }).dropna()
    if rubric_ids is not None:
        data['rubric_id'] = rubric_ids

    if len(data) < 10:
        raise ValueError(f"Insufficient data for mixed model: n={len(data)}")

    n_unique_q = len(np.unique(data['question_id']))
    if n_unique_q < 2:
        raise ValueError("Need at least 2 distinct questions for question-level effect.")

    if use_ranks:
        data['feature'] = rankdata(data['feature'])
        data['outcome'] = rankdata(data['outcome'])

    if standardize:
        data['feature'] = (data['feature'] - data['feature'].mean()) / data['feature'].std()
        data['outcome'] = (data['outcome'] - data['outcome'].mean()) / data['outcome'].std()

    vc_formula = {"question": "0 + C(question_id)"}
    random_effects_used = ["student", "question"]
    if rubric_ids is not None and len(np.unique(data['rubric_id'])) >= 2:
        vc_formula["rubric"] = "0 + C(rubric_id)"
        random_effects_used.append("rubric")

    # Fit mixed model without item fixed effects (question-length identifiable)
    fit = sm.MixedLM.from_formula(
        "outcome ~ feature",
        data=data,
        groups="student_id",
        re_formula="1",
        vc_formula=vc_formula,
    ).fit(reml=True, method="powell")

    beta = float(fit.params['feature'])
    se = float(fit.bse['feature'])
    z_stat = beta / se if se != 0 else np.nan
    p_value = 2 * (1 - norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = beta - z_crit * se if np.isfinite(se) else np.nan
    ci_high = beta + z_crit * se if np.isfinite(se) else np.nan

    # Cluster (question-level) bootstrap for small m
    unique_questions = np.unique(data['question_id'])
    boot_betas = []
    for _ in range(n_boot):
        sampled_q = rng.choice(unique_questions, size=len(unique_questions), replace=True)
        boot_idx_parts = [data.index[data['question_id'] == q].to_numpy() for q in sampled_q]
        if not boot_idx_parts:
            continue
        boot_idx = np.concatenate(boot_idx_parts)
        boot_data = data.loc[boot_idx].copy()
        try:
            boot_fit = sm.MixedLM.from_formula(
                "outcome ~ feature",
                data=boot_data,
                groups="student_id",
                re_formula="1",
                vc_formula=vc_formula,
            ).fit(reml=True, method="powell", disp=False)
            boot_betas.append(float(boot_fit.params.get('feature', np.nan)))
        except Exception:
            continue

    if boot_betas:
        se_boot = float(np.std(boot_betas))
        ci_low_boot, ci_high_boot = np.percentile(boot_betas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p_boot = 2 * min(
            np.mean(np.array(boot_betas) >= 0),
            np.mean(np.array(boot_betas) <= 0),
        )
    else:
        se_boot = np.nan
        ci_low_boot = np.nan
        ci_high_boot = np.nan
        p_boot = np.nan

    return {
        'beta': beta,
        'se': se,
        'z': z_stat,
        'p_value': p_value,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n_obs': int(len(data)),
        'n_clusters_student': int(len(np.unique(data['student_id']))),
        'n_clusters_question': int(n_unique_q),
        'n_clusters_rubric': int(len(np.unique(data['rubric_id'])) if rubric_ids is not None else 0),
        'random_effects_used': "+".join(random_effects_used),
        'se_boot': se_boot,
        'ci_low_boot': float(ci_low_boot) if np.isfinite(ci_low_boot) else np.nan,
        'ci_high_boot': float(ci_high_boot) if np.isfinite(ci_high_boot) else np.nan,
        'p_boot': float(p_boot) if np.isfinite(p_boot) else np.nan,
    }


def interpret_ece(ece: float) -> str:
    """
    Interpret Expected Calibration Error.
    
    Args:
        ece: ECE value
        
    Returns:
        Interpretation string
    """
    if ece < 0.05:
        return "excellent"
    elif ece < 0.10:
        return "good"
    elif ece < 0.15:
        return "acceptable"
    else:
        return "poor"


# =============================================================================
# Bootstrap Utilities
# =============================================================================

def _compute_bca_ci(
    boot_stats: np.ndarray,
    observed_stat: float,
    x: np.ndarray,
    y: np.ndarray,
    stat_func: Callable,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute BCa (bias-corrected and accelerated) bootstrap confidence interval.

    The BCa method adjusts for bias and skewness in the bootstrap distribution,
    providing better coverage than the percentile method, especially for skewed
    distributions or small samples.

    Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap", Chapter 14

    Args:
        boot_stats: Array of bootstrap statistics
        observed_stat: The observed statistic on original data
        x, y: Original data arrays for jackknife computation
        stat_func: Function to compute the statistic (e.g., Spearman correlation)
        alpha: Significance level

    Returns:
        Tuple of (lower, upper) confidence bounds
    """
    n_boot = len(boot_stats)
    n = len(x)

    # Bias correction (z0): proportion of bootstrap stats < observed
    p0 = np.sum(boot_stats < observed_stat) / n_boot
    if p0 == 0:
        p0 = 0.5 / n_boot  # Avoid z0 = -inf
    elif p0 == 1:
        p0 = 1 - 0.5 / n_boot  # Avoid z0 = inf
    z0 = stats.norm.ppf(p0)

    # Acceleration (a): jackknife estimate
    jack_stats = np.empty(n)
    for i in range(n):
        # Leave-one-out samples
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack_stats[i] = stat_func(x[mask], y[mask])

    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)

    if denom == 0:
        a = 0  # No acceleration if jackknife has no variance
    else:
        a = num / denom

    # Adjusted percentiles
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)

    # BCa formula
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
    p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

    # Ensure percentiles are in valid range
    p_low = np.clip(p_low, 0, 1)
    p_high = np.clip(p_high, 0, 1)

    # Get BCa confidence bounds
    ci_low = float(np.percentile(boot_stats, 100 * p_low))
    ci_high = float(np.percentile(boot_stats, 100 * p_high))

    return ci_low, ci_high


def _compute_bca_ci_from_jackknife(
    boot_stats: np.ndarray,
    observed_stat: float,
    jack_stats: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute BCa CI given bootstrap stats and jackknife stats (no stat_func).

    This is useful when the statistic is expensive to recompute from raw data.
    """
    boot_stats = np.asarray(boot_stats, dtype=float)
    jack_stats = np.asarray(jack_stats, dtype=float)
    boot_stats = boot_stats[np.isfinite(boot_stats)]
    jack_stats = jack_stats[np.isfinite(jack_stats)]
    n_boot = len(boot_stats)
    if n_boot == 0:
        return (np.nan, np.nan)
    if len(jack_stats) == 0:
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )

    # Bias correction (z0)
    p0 = np.sum(boot_stats < observed_stat) / n_boot
    if p0 == 0:
        p0 = 0.5 / n_boot
    elif p0 == 1:
        p0 = 1 - 0.5 / n_boot
    z0 = stats.norm.ppf(p0)
    if not np.isfinite(z0):
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )

    # Acceleration (a) from jackknife
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = 0.0 if denom == 0 else num / denom
    if not np.isfinite(a):
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )

    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
    p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))
    if not (np.isfinite(p_low) and np.isfinite(p_high)):
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    p_low = np.clip(p_low, 0, 1)
    p_high = np.clip(p_high, 0, 1)
    if p_low > p_high:
        p_low, p_high = p_high, p_low
    try:
        ci_low = float(np.percentile(boot_stats, 100 * p_low))
        ci_high = float(np.percentile(boot_stats, 100 * p_high))
    except ValueError:
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    return ci_low, ci_high


# =============================================================================
# Correlation with P-values and Power
# =============================================================================

def compute_spearman_with_perm_and_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 2000,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
    adaptive_perm: bool = False,
    base_n_perm: int = 2000,
    max_n_perm: int = 20000,
    adaptive_band: Tuple[float, float] = (0.5, 2.0),
    bootstrap_method: str = 'bca',
) -> Dict[str, Any]:
    """
    Compute Spearman rho with permutation p-value and bootstrap CI.

    Default method for feature-target correlations in analysis scripts.
    Fisher's z/analytic power should be considered approximate and secondary.
    Set log_progress=True to print lightweight progress during bootstrap.
    If adaptive_perm=True, refine permutation p-values near alpha by rerunning
    with more permutations.

    Args:
        x, y: Data arrays
        n_perm: Number of permutations for p-value (default: 2000)
        n_boot: Number of bootstrap samples for CI (default: 2000)
        alpha: Significance level (default: 0.05)
        random_state: Random number generator
        log_progress: Show progress bars (requires tqdm)
        adaptive_perm: Enable adaptive permutation refinement (default: False)
        base_n_perm: Initial number of permutations when adaptive (default: 2000)
        max_n_perm: Maximum permutations for refinement (default: 20000)
        adaptive_band: Multipliers of alpha to trigger refinement (default: 0.5, 2.0)
        bootstrap_method: 'percentile' or 'bca' (default: 'bca')
            - 'percentile': Simple percentile bootstrap (faster)
            - 'bca': Bias-corrected and accelerated (better coverage, slower)

    Returns:
        Dictionary with rho, p_perm, ci_low, ci_high, and metadata
    """
    rng = random_state or np.random.default_rng()

    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    result: Dict[str, Any] = {
        'rho': np.nan,
        'p_perm': np.nan,
        'ci_low': np.nan,
        'ci_high': np.nan,
        'n': n,
        'n_perm': n_perm,
        'n_boot': n_boot,
        'insufficient_data': False,
        'undefined_correlation': False,
    }

    if n < 3:
        result['insufficient_data'] = True
        return result

    rho_obs, _ = stats.spearmanr(x_clean, y_clean)
    if not np.isfinite(rho_obs):
        result['undefined_correlation'] = True
        return result

    result['rho'] = float(rho_obs)

    # Permutation p-value (two-sided)
    if n_perm and n_perm > 0:
        def _perm_test(n_perm_local: int) -> float:
            perm_rhos = np.empty(n_perm_local)
            perm_iterator = range(n_perm_local)
            if log_progress:
                try:
                    from tqdm import tqdm  # type: ignore
                    perm_iterator = tqdm(perm_iterator, desc="Permutations", leave=False)
                except ImportError:
                    pass
            for i in perm_iterator:
                perm_y = rng.permutation(y_clean)
                perm_rhos[i], _ = stats.spearmanr(x_clean, perm_y)
            return (np.sum(np.abs(perm_rhos) >= abs(rho_obs)) + 1) / (n_perm_local + 1)

        # Initial permutation test
        p_perm = _perm_test(base_n_perm if adaptive_perm else n_perm)

        # Adaptive refinement near alpha band
        if adaptive_perm and (adaptive_band[0] * alpha <= p_perm <= adaptive_band[1] * alpha) and base_n_perm < max_n_perm:
            p_perm = _perm_test(max_n_perm)
            result['n_perm'] = max_n_perm
        else:
            result['n_perm'] = base_n_perm if adaptive_perm else n_perm

        result['p_perm'] = float(p_perm)

    # Bootstrap CI
    if n_boot and n_boot > 0:
        boot_rhos = np.empty(n_boot)
        iterator = range(n_boot)
        if log_progress:
            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(iterator, total=n_boot, desc="Bootstrap", leave=False)
            except Exception:
                iterator = range(n_boot)
        for i in iterator:
            idx = rng.integers(0, n, size=n)
            boot_rhos[i], _ = stats.spearmanr(x_clean[idx], y_clean[idx])

        # Compute CI using specified method
        if bootstrap_method == 'bca':
            # BCa method: better coverage, especially for skewed distributions
            def spearman_stat(x_arr, y_arr):
                rho, _ = stats.spearmanr(x_arr, y_arr)
                return rho
            ci_low, ci_high = _compute_bca_ci(
                boot_rhos, rho_obs, x_clean, y_clean, spearman_stat, alpha
            )
        else:
            # Percentile method: faster, simpler
            ci_low, ci_high = np.percentile(boot_rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            ci_low, ci_high = float(ci_low), float(ci_high)

        result['ci_low'] = ci_low
        result['ci_high'] = ci_high
        result['bootstrap_method'] = bootstrap_method

    return result


# def estimate_power_spearman_simulation(
#     x: np.ndarray,
#     y: np.ndarray,
#     *,
#     n_target: int,
#     alpha: float = 0.05,
#     n_power_sim: int = 1000,
#     n_perm: int = 1000,
#     n_perm_power: Optional[int] = None,
#     random_state: Optional[np.random.Generator] = None,
#     log_progress: bool = False,
# ) -> float:
#     """
#     Monte Carlo power estimate using Spearman + permutation test.

#     Re-samples paired data with replacement to size n_target, computes
#     permutation p-value each iteration, and estimates power as the fraction
#     of significant runs. Analytic Fisher z power is optional/approximate.
#     """
#     rng = random_state or np.random.default_rng()
#     mask = ~(np.isnan(x) | np.isnan(y))
#     x_clean = x[mask]
#     y_clean = y[mask]
#     n_source = len(x_clean)
#     perm_for_power = n_perm_power if n_perm_power is not None else n_perm

#     if n_source < 3 or n_target < 3 or n_power_sim <= 0:
#         return np.nan

#     sig_count = 0
#     iterator = range(n_power_sim)
#     if log_progress:
#         try:
#             from tqdm import tqdm  # type: ignore
#             iterator = tqdm(iterator, total=n_power_sim, desc="Power sim", leave=False)
#         except Exception:
#             iterator = range(n_power_sim)
#     for i in iterator:
#         idx = rng.integers(0, n_source, size=n_target)
#         x_b = x_clean[idx]
#         y_b = y_clean[idx]
#         res = compute_spearman_with_perm_and_bootstrap(
#             x_b,
#             y_b,
#             n_perm=perm_for_power,
#             n_boot=0,
#             alpha=alpha,
#             random_state=rng,
#         )
#         p_val = res.get('p_perm', np.nan)
#         if (
#             not res.get('insufficient_data')
#             and not res.get('undefined_correlation')
#             and not np.isnan(p_val)
#             and p_val < alpha
#         ):
#             sig_count += 1

#     return sig_count / n_power_sim
from typing import Optional
import numpy as np
from scipy import stats


def estimate_power_spearman_simulation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_target: int,
    alpha: float = 0.05,
    n_power_sim: int = 1000,
    n_perm: int = 1000,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
) -> float:
    """
    Monte Carlo power estimate for a Spearman + permutation test at level alpha.

    This function still estimates the power of the SAME test as before:
      - test statistic: Spearman's rho
      - null distribution: permutation of y for each dataset
      - decision: reject if permutation p-value < alpha

    The speedup comes from:
      - doing the Spearman permutation test in a vectorized way
      - reusing a single set of random permutations across simulations

    No asymptotic shortcuts, no fewer permutations by default.
    """
    rng = random_state or np.random.default_rng()
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n_source = len(x_clean)

    if n_source < 3 or n_target < 3 or n_power_sim <= 0 or n_perm <= 0:
        return np.nan

    # ------------------------------------------------------------------
    # Pre-generate permutation indices for vectors of length n_target.
    # Reusing permutations across simulations is still a valid permutation
    # scheme for each dataset: each test uses n_perm random permutations
    # of {0, ..., n_target-1}.
    # ------------------------------------------------------------------
    perm_idx = np.vstack([rng.permutation(n_target) for _ in range(n_perm)])  # (n_perm, n_target)

    def spearman_perm_pvalue_vectorized(x_b: np.ndarray, y_b: np.ndarray) -> float:
        """
        Compute permutation p-value for Spearman's rho using vectorized ops.

        - Rank x_b and y_b once (Spearman = Pearson on ranks).
        - Center and scale once.
        - Apply the precomputed permutation indices to y_ranks.
        - Compute all permuted correlations in one go via matrix multiplication.
        """
        # Rank transform (Spearman)
        xr = stats.rankdata(x_b).astype(float)
        yr = stats.rankdata(y_b).astype(float)

        # Center
        xr -= xr.mean()
        yr -= yr.mean()

        # Denominator is fixed for all permutations
        denom = np.sqrt(np.sum(xr ** 2) * np.sum(yr ** 2))
        if denom == 0.0 or not np.isfinite(denom):
            return np.nan  # undefined correlation

        # Observed rho
        rho_obs = float((xr @ yr) / denom)

        # Apply precomputed permutations to y_ranks
        # yr_perm: shape (n_perm, n_target)
        yr_perm = yr[perm_idx]

        # Numerators for each permuted sample: dot(xr, yr_perm[i])
        # This is matrix-vector: (n_perm, n_target) @ (n_target,) -> (n_perm,)
        num_perm = yr_perm @ xr
        rho_perm = num_perm / denom  # (n_perm,)

        # Two-sided permutation p-value
        p_perm = (np.sum(np.abs(rho_perm) >= abs(rho_obs)) + 1) / (n_perm + 1)
        return p_perm

    # ------------------------------------------------------------------
    # Monte Carlo power: resample paired data, then run the SAME
    # permutation-based Spearman test using the vectorized helper.
    # ------------------------------------------------------------------
    sig_count = 0
    iterator = range(n_power_sim)
    if log_progress:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(iterator, total=n_power_sim, desc="Power sim", leave=False)
        except Exception:
            iterator = range(n_power_sim)

    for _ in iterator:
        idx = rng.integers(0, n_source, size=n_target)
        x_b = x_clean[idx]
        y_b = y_clean[idx]

        p_val = spearman_perm_pvalue_vectorized(x_b, y_b)
        if np.isfinite(p_val) and p_val < alpha:
            sig_count += 1

    return sig_count / n_power_sim


def correlation_with_stats(
    x: np.ndarray, 
    y: np.ndarray, 
    method: str = 'spearman',
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute correlation with p-value, power, and interpretation.
    
    Args:
        x: First variable
        y: Second variable
        method: 'spearman' or 'pearson'
        alpha: Significance level
        
    Returns:
        Dict with r, p_value, power, interpretation, required_n_80_power
    """
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)
    
    if n < 4:
        return {
            'r': np.nan,
            'p_value': np.nan,
            'power': 0.0,
            'interpretation': 'insufficient data',
            'required_n_80_power': np.nan,
            'n': n
        }
    
    # Compute correlation
    if method == 'spearman':
        r, p_value = stats.spearmanr(x_clean, y_clean)
    else:
        r, p_value = stats.pearsonr(x_clean, y_clean)
    
    # Compute power (analytic Fisher z is approximate/optional)
    power = correlation_power(r, n, alpha)
    
    # Required n for 80% power (analytic, approximate)
    req_n = required_sample_size_correlation(r, power=0.8, alpha=alpha)
    
    return {
        'r': r,
        'p_value': p_value,
        'power': power,
        'interpretation': interpret_correlation(r),
        'required_n_80_power': req_n,
        'n': n
    }


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two groups.
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((group1.mean() - group2.mean()) / pooled_std)


def ttest_with_power(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute t-test with effect size, power, and interpretation.
    
    Args:
        group1: First group values
        group2: Second group values
        alpha: Significance level
        
    Returns:
        Dict with statistic, p_value, cohens_d, power, interpretation
    """
    n1, n2 = len(group1), len(group2)
    
    if n1 < 2 or n2 < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'cohens_d': np.nan,
            'power': 0.0,
            'effect_interpretation': 'insufficient data',
            'n1': n1,
            'n2': n2
        }
    
    # Welch's t-test (doesn't assume equal variances)
    statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    # Cohen's d
    d = compute_cohens_d(group1, group2)
    
    # Power
    power = ttest_power(abs(d), n1, n2, alpha) if not np.isnan(d) else 0.0
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': d,
        'power': power,
        'effect_interpretation': interpret_cohens_d(d) if not np.isnan(d) else 'unknown',
        'n1': n1,
        'n2': n2
    }


# =============================================================================
# Fisher's z Appropriateness Assessment
# =============================================================================


# =============================================================================
# Power Curve Analysis (Prospective Power)
# =============================================================================

def _generate_correlated_data_copula(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    r_target: float,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with target correlation using Gaussian copula.

    Preserves marginal distributions while imposing specified correlation.

    Args:
        x_obs: Observed x values (for marginal distribution)
        y_obs: Observed y values (for marginal distribution)
        r_target: Target Spearman correlation
        n_samples: Number of samples to generate
        rng: Random number generator

    Returns:
        Tuple of (x_synthetic, y_synthetic) with correlation ≈ r_target
    """
    # Generate bivariate Gaussian with correlation r_target
    mean = [0, 0]
    cov = [[1, r_target], [r_target, 1]]
    z = rng.multivariate_normal(mean, cov, size=n_samples)

    # Convert to uniform via Gaussian CDF
    u = stats.norm.cdf(z)

    # Convert to original scale via quantile matching
    # This preserves marginal distributions of observed data
    x_synthetic = np.quantile(x_obs, u[:, 0])
    y_synthetic = np.quantile(y_obs, u[:, 1])

    return x_synthetic, y_synthetic


def generate_clustered_data_mixed_model(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    beta_target: float,
    rng: np.random.Generator,
    rubric_ids: Optional[np.ndarray] = None, 
    r_target: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float], float]:
    """
    Generate synthetic clustered data using cross-classified mixed model structure.

    Fits: y ~ x + C(question_id) + (1|student_id) + (1|rubric_id)
    Then simulates new data with same cluster structure and target effect size.

    Model structure (aligned with current Step-3 screening via `estimate_feature_effect()`):
    - Fixed effects: feature, C(question_id)
    - Random effects: student, rubric (if ≥2 rubrics)
    - Note: this generator does not include an answer_id random effect; it is an approximation for power sims.
    - Binary outcomes: Uses logistic link (expit)
    - Continuous outcomes: Uses linear link with Gaussian noise

    Calibration for binary outcomes:
    - Iteratively adjusts beta to achieve target correlation (max 5 iterations)
    - Converges when |r_achieved - r_target| < 0.02
    - Reports calibration_failed flag in power analysis if error > 0.05

    Args:
        x_obs: Observed feature values
        y_obs: Observed outcome values
        student_ids: Student cluster labels
        question_ids: Item cluster labels
        beta_target: Target fixed effect slope
        rng: Random number generator
        rubric_ids: Optional rubric cluster labels
        r_target: Optional target correlation for calibration (binary outcomes only)

    Returns:
        Tuple of (x_sim, y_sim, student_sim, item_sim, rubric_sim, r_achieved, beta_effect):
        - x_sim: Simulated feature values
        - y_sim: Simulated outcome values
        - student_sim: Student cluster labels
        - item_sim: Item cluster labels
        - rubric_sim: Rubric cluster labels
        - r_achieved: Achieved Spearman correlation (or None if not computed)
        - beta_effect: Final beta used in simulation

    Raises:
        ImportError: If statsmodels not installed (fallback to simple resampling)

    Warnings:
        - Degenerate clustering (<2 students or items) → fallback to resampling
        - Insufficient data (<10 observations) → fallback to resampling
        - Model fit failure → fallback to student-only random intercept
    """
    try:
        import statsmodels.api as sm
        from scipy.special import expit
    except ImportError:
        warnings.warn(
            "statsmodels and scipy are required for mixed-model data generation; "
            "falling back to simple resampling.",
            stacklevel=2,
        )
        idx = rng.integers(0, len(x_obs), size=len(x_obs))
        return (x_obs[idx], y_obs[idx], student_ids[idx], question_ids[idx],
                rubric_ids[idx] if rubric_ids is not None else np.zeros_like(student_ids[idx]),
                None, beta_target)

    data = pd.DataFrame({
        'x': x_obs,
        'y': y_obs,
        'student_id': student_ids,
        'question_id': question_ids,
    }).dropna()
    if rubric_ids is not None:
        data['rubric_id'] = rubric_ids

    if len(data) < 10:
        warnings.warn(
            f"Insufficient data for mixed-model generation (n={len(data)}); "
            "falling back to simple resampling.",
            stacklevel=2,
        )
        idx = rng.integers(0, len(x_obs), size=len(x_obs))
        return (x_obs[idx], y_obs[idx], student_ids[idx], question_ids[idx],
                rubric_ids[idx] if rubric_ids is not None else np.zeros_like(student_ids[idx]),
                None, beta_target)

    # Degenerate cluster guards (require ≥2 unique clusters)
    unique_students = pd.unique(data['student_id'])
    unique_items = pd.unique(data['question_id'])
    unique_rubrics = pd.unique(data['rubric_id']) if rubric_ids is not None else []

    if len(unique_students) < 2:
        warnings.warn(
            f"Degenerate clustering: {len(unique_students)} student(s); "
            "falling back to simple resampling.",
            stacklevel=2
        )
        idx = rng.integers(0, len(x_obs), size=len(x_obs))
        return (x_obs[idx], y_obs[idx], student_ids[idx], question_ids[idx],
                rubric_ids[idx] if rubric_ids is not None else np.zeros_like(student_ids[idx]),
                None, beta_target)

    if len(unique_items) < 2:
        warnings.warn(
            f"Degenerate clustering: {len(unique_items)} item(s); "
            "falling back to simple resampling.",
            stacklevel=2
        )
        idx = rng.integers(0, len(x_obs), size=len(x_obs))
        return (x_obs[idx], y_obs[idx], student_ids[idx], question_ids[idx],
                rubric_ids[idx] if rubric_ids is not None else np.zeros_like(student_ids[idx]),
                None, beta_target)

    # Random intercepts: student + rubric (if available), question fixed (C(question_id))
    vc_formula = {}
    if rubric_ids is not None and len(np.unique(rubric_ids)) >= 2:
        vc_formula["rubric"] = "0 + C(rubric_id)"
    try:
        fit = sm.MixedLM.from_formula(
            "y ~ x + C(question_id)",
            data=data,
            groups="student_id",
            re_formula="1",
            vc_formula=vc_formula if vc_formula else None,
        ).fit(reml=True, method="powell")
    except Exception as e:
        warnings.warn(
            f"Mixed-model fit failed ({e}); falling back to student-only random intercept.",
            stacklevel=2,
        )
        fit = sm.MixedLM.from_formula(
            "y ~ x + C(question_id)",
            data=data,
            groups="student_id",
            re_formula="1",
        ).fit(reml=True, method="powell")

    intercept = float(fit.params.get('Intercept', fit.params.iloc[0]))
    beta_param_name = [p for p in fit.params.index if p != 'Intercept'][0]
    beta_fit = float(fit.params[beta_param_name])
    sigma_student = np.sqrt(float(fit.cov_re.iloc[0, 0]))

    vcomp_raw = fit.vcomp
    if isinstance(vcomp_raw, pd.Series):
        var_rubric = float(vcomp_raw.get("rubric", np.nan)) if rubric_ids is not None else 0.0
    else:
        v_series = pd.Series(np.asarray(vcomp_raw).flatten())
        var_rubric = float(v_series.iloc[0]) if (rubric_ids is not None and len(v_series) > 0) else 0.0
    sigma_rubric = np.sqrt(var_rubric) if np.isfinite(var_rubric) else 0.0
    sigma_resid = np.sqrt(float(fit.scale)) if np.isfinite(fit.scale) else 0.0

    sample_seed = int(rng.integers(0, 2**32 - 1))
    x_sim = data['x'].sample(len(data), replace=True, random_state=sample_seed).to_numpy()
    student_sim = data['student_id'].to_numpy()
    item_sim = data['question_id'].to_numpy()
    rubric_sim = data['rubric_id'].to_numpy() if rubric_ids is not None else np.zeros(len(data))

    unique_students = pd.unique(student_sim)
    unique_items = pd.unique(item_sim)
    unique_rubrics = pd.unique(rubric_sim)
    if len(unique_students) < 2 or len(unique_items) < 2 or (rubric_ids is not None and len(unique_rubrics) < 2):
        warnings.warn(
            f"Mixed-model generation found degenerate clusters "
            f"(students={len(unique_students)}, questions={len(unique_items)}, rubrics={len(unique_rubrics)}); "
            "falling back to simple resampling.",
            stacklevel=2,
        )
        idx = rng.integers(0, len(x_obs), size=len(x_obs))
        return (
            x_obs[idx],
            y_obs[idx],
            student_ids[idx],
            question_ids[idx],
            rubric_ids[idx] if rubric_ids is not None else np.zeros_like(student_ids[idx]),
        )

    # Interpret beta_target as the slope on rank-standardized x (Spearman-style effect size).
    # This aligns the power-curve x-axis with the downstream rank-standardized regression beta.
    x_rank = stats.rankdata(x_sim).astype(float)
    x_rank -= x_rank.mean()
    x_rank_sd = float(np.std(x_rank))
    if np.isfinite(x_rank_sd) and x_rank_sd > 0:
        x_z = x_rank / x_rank_sd
    else:
        x_z = x_rank
    beta_effect = float(beta_target)

    u_student = {
        sid: rng.normal(0, sigma_student) if np.isfinite(sigma_student) else 0.0
        for sid in unique_students
    }
    question_effects = {}
    for pname, pval in fit.params.items():
        if pname.startswith("C(question_id)[T."):
            key = pname.split("[T.")[-1].strip("]")
            question_effects[key] = float(pval)
    v_item = {iid: question_effects.get(str(iid), 0.0) for iid in unique_items}
    w_rubric = {
        rid: rng.normal(0, sigma_rubric) if (rubric_ids is not None and np.isfinite(sigma_rubric)) else 0.0
        for rid in unique_rubrics
    }

    def _simulate(beta_val: float) -> Tuple[np.ndarray, np.ndarray]:
        mu_local = np.array([
            intercept + beta_val * x_val + u_student.get(sid, 0.0) + v_item.get(iid, 0.0) + w_rubric.get(rid, 0.0)
            for x_val, sid, iid, rid in zip(x_z, student_sim, item_sim, rubric_sim)
        ])
        if len(pd.unique(data['y'])) <= 2:
            p = expit(mu_local)
            y_sim_local = rng.binomial(1, p).astype(float)
        else:
            eps = rng.normal(0, sigma_resid, size=len(mu_local))
            y_sim_local = mu_local + eps
        return mu_local, y_sim_local

    mu, y_sim = _simulate(beta_effect)

    # Iterative calibration for binary outcomes to match target r
    r_achieved = None
    if r_target is not None and len(pd.unique(data['y'])) <= 2:
        # Iterative refinement for binary outcomes
        max_iter = 5
        convergence_threshold = 0.02

        for iteration in range(max_iter):
            r_current, _ = stats.spearmanr(x_sim, y_sim)

            if not np.isfinite(r_current) or abs(r_current) < 1e-6:
                break

            error = abs(r_current - r_target)
            if error < convergence_threshold:
                break

            # Adjust beta with tighter bounds
            scale = r_target / r_current
            scale = float(np.clip(scale, 0.5, 2.0))  # Tighter than before (-5, 5)
            beta_effect *= scale
            mu, y_sim = _simulate(beta_effect)

        r_achieved = r_current
    else:
        # For continuous outcomes, compute achieved correlation
        r_achieved, _ = stats.spearmanr(x_sim, y_sim)

    return x_sim, y_sim, student_sim, item_sim, rubric_sim, r_achieved, beta_effect


@dataclass
class FeatureEffectResult:
    """
    Standardized container for Step-3 feature effect estimation outputs.

    Notes:
      - For continuous outcomes (MixedLM): beta/se/p_value are frequentist.
      - For binary outcomes (BinomialBayesMixedGLM.fit_vb): beta is posterior mean,
        se is posterior SD, and p_value is a z-approximation for screening only.
    """
    beta: float
    se: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float

    odds_ratio: Optional[float] = None
    or_ci_low: Optional[float] = None
    or_ci_high: Optional[float] = None
    ame: Optional[float] = None

    n_obs: int = 0
    feature_type: str = ""
    outcome_type: str = ""
    model_spec: str = ""
    method: str = ""
    inference_type: str = ""
    fixed_effects: List[str] = field(default_factory=list)
    random_effects: List[str] = field(default_factory=list)
    include_student_vc_glmm: bool = False
    outcome_transformed: str = "none"
    converged: bool = True
    warnings: List[str] = field(default_factory=list)

    gee_mode: Optional[str] = None
    gee_beta: Optional[float] = None
    gee_se: Optional[float] = None
    gee_p_value: Optional[float] = None
    sign_consistent: Optional[bool] = None


@dataclass
class NonlinearFeatureEffectResult:
    """
    Standardized container for Step-3 non-linear feature effect estimation outputs.
    """
    delta_10_90: float
    delta_ci_low: float
    delta_ci_high: float
    ame: float
    ame_ci_low: float
    ame_ci_high: float
    aic_linear: float
    bic_linear: float
    aic_nonlinear: float
    bic_nonlinear: float
    lrt_stat: float
    lrt_df: float
    lrt_p_value: float
    n_obs: int
    feature_type: str
    outcome_type: str
    method: str
    basis: str
    model_spec_linear: str
    model_spec_nonlinear: str
    outcome_transformed: str = "none"
    converged: bool = True
    warnings: List[str] = field(default_factory=list)
    residual_skew: Optional[float] = None
    residual_kurtosis: Optional[float] = None
    singular_fit: Optional[bool] = None
    ci_method: str = "percentile"
    ci_type: str = "pointwise"
    jackknife_n: Optional[int] = None


def detect_feature_type(
    feature: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    answer_ids: np.ndarray,
    eps: float = 1e-8,
) -> str:
    """
    Detect feature "constancy type" for routing mixed models.

    Uses tolerance-based within-group range checks (max-min < eps) to avoid
    misclassifying near-constant floating features due to numerical noise.
    """
    df = pd.DataFrame(
        {
            "feature": np.asarray(feature, dtype=np.float64),
            "student_id": student_ids,
            "question_id": question_ids,
            "rubric_id": rubric_ids,
            "answer_id": answer_ids,
        }
    ).dropna(subset=["feature"])

    def is_constant_within(group_col: str) -> bool:
        grouped = df.groupby(group_col)["feature"]
        ranges = (grouped.max() - grouped.min()).dropna()
        return len(ranges) > 1 and (ranges < eps).all()

    if is_constant_within("question_id"):
        return "question_constant"
    if is_constant_within("rubric_id"):
        return "rubric_constant"
    if is_constant_within("student_id"):
        return "student_constant"
    if is_constant_within("answer_id"):
        return "answer_constant"
    return "row_varying"


def _should_include_answer_re(data: pd.DataFrame) -> bool:
    """
    Decide whether an answer_id random effect is estimable.

    We require at least two answer_id levels and at least one repeated observation
    per level; if answer_id is effectively unique per row, lme4/GLMMs will fail.
    """
    if "answer_id" not in data.columns:
        return False
    n_obs = int(len(data))
    if n_obs < 2:
        return False
    counts = data["answer_id"].value_counts(dropna=True)
    if counts.empty:
        return False
    n_levels = int(counts.size)
    max_count = int(counts.max())
    if n_levels < 2 or max_count < 2:
        return False
    if n_levels >= n_obs:
        return False
    return True


QUESTION_FE_MAX_LEVELS_DEFAULT = 20
_QUESTION_EFFECT_NOTICE_SEEN: set[tuple[str, int, int]] = set()


def _lme4_trace_settings(
    trace: Optional[bool] = None,
    trace_file: Optional[str] = None,
) -> tuple[bool, str]:
    if trace is None:
        env_val = os.getenv("GC_LME4_TRACE", "").strip().lower()
        trace = env_val in {"1", "true", "yes", "y", "on"}
    if trace_file is None:
        trace_file = os.getenv("GC_LME4_TRACE_FILE", "")
    return bool(trace), str(trace_file) if trace_file else ""


def _select_question_effects(
    question_ids: Optional[np.ndarray],
    feature_type: str,
    *,
    max_question_fe_levels: int = QUESTION_FE_MAX_LEVELS_DEFAULT,
    force_include_question_fe: Optional[bool] = None,
) -> Tuple[bool, bool, str, int]:
    n_questions = 0
    if question_ids is not None:
        n_questions = int(pd.Series(question_ids).nunique())
    if force_include_question_fe is not None:
        include_question_fe = bool(force_include_question_fe)
        include_question_re = False
        mode = "fixed" if include_question_fe else "none"
        return include_question_fe, include_question_re, mode, n_questions
    if n_questions < 2:
        return False, False, "none", n_questions
    if feature_type == "question_constant":
        return False, False, "none", n_questions
    if n_questions <= max_question_fe_levels:
        return True, False, "fixed", n_questions
    return False, True, "random", n_questions


def estimate_feature_effect(
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    outcome_type: Literal["continuous", "binary"] = "continuous",
    backend: Literal["statsmodels", "lme4"] = "lme4",
    run_gee_sensitivity: bool = True,
    gee_mode: Literal["simple", "rubric_fe"] = "simple",
    use_ranks: bool = False,
    standardize_feature: bool = True,
    standardize_outcome: bool = False,
    transform_bounded: Literal["none", "logit_clip"] = "none",
    include_student_vc_glmm: bool = False,
    alpha: float = 0.05,
    eps: float = 1e-8,
    force_feature_type: Optional[str] = None,
    force_include_question_fe: Optional[bool] = None,
    optimizer: str = "powell",
    maxfun: int = 100000,
) -> FeatureEffectResult:
    """
    Unified Step-3 effect estimator with crossed random effects:

      - Continuous: MixedLM (frequentist)
          outcome ~ feature_z + C(question_id) + (1|student_id) + (1|answer_id) + (1|rubric_id)
      - Binary: BinomialBayesMixedGLM (VB posterior approximation)
          logit P(Y=1) ~ feature_z + C(question_id) + (1|answer_id) + (1|rubric_id) [+ (1|student_id)]

    Key guardrails:
      - Constructs answer_id after dropna (feature×target analytic subset).
      - Prevents GEE rubric_fe when collinear (rubric_constant / question_constant).
      - Supports forced-spec LOO-Q by bypassing auto detection and question FE inclusion.
    """
    warnings_list: List[str] = []
    lme4_optimizer = optimizer
    lme4_maxfun = int(maxfun) if maxfun is not None else 100000
    if backend == "lme4":
        lme4_optimizer = _normalize_lme4_optimizer(optimizer)
        if lme4_optimizer != optimizer:
            warnings_list.append(
                f"Mapped optimizer '{optimizer}' to lme4 optimizer '{lme4_optimizer}'."
            )

    # Build analytic data frame and drop missing rows first (guardrail).
    data = pd.DataFrame(
        {
            "feature": np.asarray(feature, dtype=np.float64),
            "outcome": np.asarray(outcome, dtype=np.float64),
            "student_id": student_ids,
            "question_id": question_ids,
            "rubric_id": rubric_ids,
        }
    ).dropna()

    if len(data) < 30:
        raise ValueError(f"Insufficient data: n={len(data)}, need ≥30")

    # Construct answer_id after dropna (guardrail).
    answer_key = pd.MultiIndex.from_arrays([data["student_id"], data["question_id"]])
    data["answer_id"] = pd.factorize(answer_key)[0].astype(str)

    # Convert IDs to categorical for stable formula handling.
    data["student_id"] = pd.Categorical(data["student_id"])
    data["question_id"] = pd.Categorical(data["question_id"])
    data["rubric_id"] = pd.Categorical(data["rubric_id"])
    data["answer_id"] = pd.Categorical(data["answer_id"])

    include_answer_re = _should_include_answer_re(data)
    if not include_answer_re:
        n_levels = int(data["answer_id"].nunique())
        max_count = int(data["answer_id"].value_counts().max()) if n_levels else 0
        warnings_list.append(
            "Dropping answer_id random effect (answer_id has no repeats: "
            f"levels={n_levels}, max_count={max_count})."
        )

    feat_sd = float(data["feature"].std())
    if not np.isfinite(feat_sd) or feat_sd < eps:
        raise ValueError(f"Feature has near-zero variance (std={feat_sd:.2e} < eps={eps})")

    # Detect feature type (or override).
    if force_feature_type is not None:
        feature_type = force_feature_type
    else:
        feature_type = detect_feature_type(
            data["feature"].to_numpy(),
            data["student_id"].to_numpy(),
            data["question_id"].to_numpy(),
            data["rubric_id"].to_numpy(),
            data["answer_id"].to_numpy(),
            eps=eps,
        )

    include_question_fe, include_question_re, question_effect_mode, n_questions = _select_question_effects(
        data["question_id"].to_numpy(),
        feature_type,
        force_include_question_fe=force_include_question_fe,
    )
    if question_effect_mode != "none":
        notice_key = (question_effect_mode, n_questions, QUESTION_FE_MAX_LEVELS_DEFAULT)
        if notice_key not in _QUESTION_EFFECT_NOTICE_SEEN:
            _QUESTION_EFFECT_NOTICE_SEEN.add(notice_key)
            print(
                "[auto-question-effects] "
                f"mode={question_effect_mode} "
                f"n_questions={n_questions} "
                f"fe_max={QUESTION_FE_MAX_LEVELS_DEFAULT}",
                file=sys.stderr,
            )
        warnings_list.append(
            "Question effect mode: "
            f"{question_effect_mode} (n_questions={n_questions}, fe_max={QUESTION_FE_MAX_LEVELS_DEFAULT})."
        )

    if feature_type == "question_constant":
        warnings_list.append(
            "Question-constant feature (n≈4 questions). Inference is exploratory; require LOO-Q sensitivity."
        )

    # Guardrail: rubric_fe GEE is collinear with rubric-constant/question-constant features.
    if gee_mode == "rubric_fe" and feature_type in {"rubric_constant", "question_constant"}:
        gee_mode = "simple"
        warnings_list.append(
            f"Switched gee_mode to 'simple' (rubric_fe collinear with {feature_type} features)."
        )

    # Optional bounded outcome transform for continuous outcomes.
    outcome_transformed = "none"
    if outcome_type == "continuous":
        out_min = float(data["outcome"].min())
        out_max = float(data["outcome"].max())
        if np.isfinite(out_min) and np.isfinite(out_max) and out_min >= 0.0 and out_max <= 1.0:
            if transform_bounded == "logit_clip":
                clip_eps = 1e-6
                y = data["outcome"].clip(clip_eps, 1.0 - clip_eps)
                data["outcome"] = np.log(y / (1.0 - y))
                outcome_transformed = "logit_clip"
                warnings_list.append("Applied logit_clip transform to bounded [0,1] outcome.")
            else:
                warnings_list.append(
                    "Bounded outcome in [0,1]. Gaussian LMM OK for screening; "
                    "use transform_bounded='logit_clip' if residuals are problematic."
                )

    # Optional rank transforms.
    if use_ranks:
        data["feature"] = rankdata(data["feature"].to_numpy())
        if outcome_type == "continuous":
            data["outcome"] = rankdata(data["outcome"].to_numpy())

    # Standardize feature (default True) and optionally outcome.
    if standardize_feature:
        x_mean = float(data["feature"].mean())
        x_sd = float(data["feature"].std())
        if not np.isfinite(x_sd) or x_sd < eps:
            raise ValueError(f"Feature has near-zero variance after transforms (std={x_sd:.2e})")
        data["feature_z"] = (data["feature"] - x_mean) / x_sd
    else:
        data["feature_z"] = data["feature"]

    if outcome_type == "continuous" and standardize_outcome:
        y_mean = float(data["outcome"].mean())
        y_sd = float(data["outcome"].std())
        if not np.isfinite(y_sd) or y_sd < eps:
            raise ValueError(f"Outcome has near-zero variance (std={y_sd:.2e})")
        data["outcome_z"] = (data["outcome"] - y_mean) / y_sd
    else:
        data["outcome_z"] = data["outcome"]

    # Fit primary model.
    if outcome_type == "continuous":
        if backend == "lme4":
            try:
                result = _fit_lmm_continuous_r(
                    data,
                    include_question_fe=include_question_fe,
                    include_question_re=include_question_re,
                    alpha=alpha,
                    include_answer_re=include_answer_re,
                    optimizer=lme4_optimizer,
                    maxfun=lme4_maxfun,
                )
                if not (np.isfinite(result.beta) and np.isfinite(result.se)):
                    if not np.isfinite(result.beta):
                        print("DEBUG: lme4 beta is non-finite")
                    if not np.isfinite(result.se):
                        print("DEBUG: lme4 se is non-finite")
            except Exception as e:
                warnings_list.append(f"lme4 backend failed; no fallback: {e}")
                raise RuntimeError(f"lme4 backend failed (no fallback): {e}") from e
        else:
            result = _fit_lmm_continuous(
                data,
                include_question_fe=include_question_fe,
                include_question_re=include_question_re,
                alpha=alpha,
                optimizer=optimizer,
                include_answer_re=include_answer_re,
            )
    else:
        if backend == "lme4":
            try:
                result = _fit_glmm_binary_r(
                    data,
                    include_question_fe=include_question_fe,
                    include_question_re=include_question_re,
                    feat_std=float(x_sd) if standardize_feature else float(feat_sd),
                    alpha=alpha,
                    include_student_vc=include_student_vc_glmm,
                    include_answer_re=include_answer_re,
                    optimizer=lme4_optimizer,
                    maxfun=lme4_maxfun,
                )
                if not (np.isfinite(result.beta) and np.isfinite(result.se)):
                    raise ValueError("lme4 returned non-finite beta/se")
            except Exception as e:
                warnings_list.append(f"lme4 backend failed; no fallback: {e}")
                raise RuntimeError(f"lme4 backend failed (no fallback): {e}") from e
        else:
            result = _fit_glmm_binary(
                data,
                include_question_fe=include_question_fe,
                include_question_re=include_question_re,
                feat_std=float(x_sd) if standardize_feature else float(feat_sd),
                alpha=alpha,
                include_student_vc=include_student_vc_glmm,
                include_answer_re=include_answer_re,
            )
        result.include_student_vc_glmm = bool(include_student_vc_glmm)

    # Sensitivity via GEE (student-clustered).
    if run_gee_sensitivity:
        gee_res = _fit_gee_sensitivity(
            data=data,
            outcome_type=outcome_type,
            include_question_fe=include_question_fe,
            mode=gee_mode,
        )
        if gee_res is not None:
            result.gee_mode = gee_mode
            result.gee_beta = gee_res["beta"]
            result.gee_se = gee_res["se"]
            result.gee_p_value = gee_res["p_value"]
            if np.isfinite(result.beta) and np.isfinite(result.gee_beta):
                result.sign_consistent = np.sign(result.beta) == np.sign(result.gee_beta)
                if result.sign_consistent is False:
                    warnings_list.append(f"Primary and GEE({gee_mode}) estimates have opposite signs.")

    # Finalize metadata.
    fe_list = ["feature_z"] + (["C(question_id)"] if include_question_fe else [])
    if outcome_type == "continuous":
        re_list = ["(1|student_id)"]
        if include_question_re:
            re_list.append("(1|question_id)")
        if include_answer_re:
            re_list.append("(1|answer_id)")
        if int(pd.Series(data["rubric_id"]).nunique()) >= 2:
            re_list.append("(1|rubric_id)")
    else:
        re_list = []
        if include_question_re:
            re_list.append("(1|question_id)")
        if include_answer_re:
            re_list.append("(1|answer_id)")
        if int(pd.Series(data["rubric_id"]).nunique()) >= 2:
            re_list.append("(1|rubric_id)")
        if include_student_vc_glmm:
            re_list.append("(1|student_id)")

    result.feature_type = feature_type
    result.outcome_type = outcome_type
    result.outcome_transformed = outcome_transformed
    result.fixed_effects = fe_list
    result.random_effects = re_list
    result.model_spec = f"outcome ~ {' + '.join(fe_list)} + {' + '.join(re_list)}"
    result.n_obs = int(len(data))
    result.warnings.extend(warnings_list)

    return result


def _prepare_nonlinear_base_data(
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "feature": np.asarray(feature, dtype=np.float64),
            "outcome": np.asarray(outcome, dtype=np.float64),
            "student_id": student_ids,
            "question_id": question_ids,
            "rubric_id": rubric_ids,
        }
    ).dropna()

    if len(data) < 30:
        raise ValueError(f"Insufficient data: n={len(data)}, need ≥30")

    answer_key = pd.MultiIndex.from_arrays([data["student_id"], data["question_id"]])
    data["answer_id"] = pd.factorize(answer_key)[0].astype(str)

    # Keep IDs as strings to avoid categorical setitem errors during bootstrap relabeling.
    data["student_id"] = data["student_id"].astype(str)
    data["question_id"] = data["question_id"].astype(str)
    data["rubric_id"] = data["rubric_id"].astype(str)
    data["answer_id"] = data["answer_id"].astype(str)
    return data


def _fit_mixedlm_generic(
    data: pd.DataFrame,
    formula: str,
    optimizer: str,
    include_question_re: bool,
) -> tuple:
    include_answer_re = _should_include_answer_re(data)
    vc_formula: Dict[str, str] = {}
    random_effects = ["(1|student_id)"]
    if include_question_re:
        vc_formula["question"] = "0 + C(question_id)"
        random_effects.append("(1|question_id)")
    if include_answer_re:
        vc_formula["answer"] = "0 + C(answer_id)"
        random_effects.append("(1|answer_id)")

    n_rubric = int(pd.Series(data["rubric_id"]).nunique())
    if n_rubric >= 2:
        vc_formula["rubric"] = "0 + C(rubric_id)"
        random_effects.append("(1|rubric_id)")

    warnings_list: List[str] = []
    converged = True
    try:
        fit = sm.MixedLM.from_formula(
            formula,
            data=data,
            groups="student_id",
            re_formula="1",
            vc_formula=vc_formula if vc_formula else None,
        ).fit(reml=True, method=optimizer)
        if hasattr(fit, "converged") and not fit.converged:
            converged = False
            warnings_list.append(f"MixedLM did not converge (optimizer={optimizer}).")
    except Exception as e:
        if "answer" in vc_formula:
            warnings_list.append(f"MixedLM failed ({e}); retrying without answer VC.")
            vc_formula = {k: v for k, v in vc_formula.items() if k != "answer"}
        else:
            warnings_list.append(f"MixedLM failed ({e}); retrying without VC.")
            vc_formula = {}
        fit = sm.MixedLM.from_formula(
            formula,
            data=data,
            groups="student_id",
            re_formula="1",
            vc_formula=vc_formula if vc_formula else None,
        ).fit(reml=True, method=optimizer)

    singular_fit = None
    try:
        vcomp = np.asarray(getattr(fit, "vcomp", []), dtype=float)
        cov_re = np.asarray(getattr(fit, "cov_re", []), dtype=float)
        small_vc = (vcomp.size > 0 and np.nanmin(np.abs(vcomp)) < 1e-8)
        small_re = (cov_re.size > 0 and np.nanmin(np.abs(cov_re)) < 1e-8)
        singular_fit = bool(small_vc or small_re)
        if singular_fit:
            warnings_list.append("Potential singular fit: variance components near zero.")
    except Exception:
        singular_fit = None

    return fit, warnings_list, converged, random_effects, singular_fit


def _fit_glmm_vb_generic(
    data: pd.DataFrame,
    formula: str,
    include_student_vc: bool,
    include_question_re: bool,
) -> tuple:
    import statsmodels.genmod.bayes_mixed_glm as bmg

    include_answer_re = _should_include_answer_re(data)
    vc_formulas: Dict[str, str] = {"rubric": "0 + C(rubric_id)"}
    random_effects = ["(1|rubric_id)"]
    if include_question_re:
        vc_formulas["question"] = "0 + C(question_id)"
        random_effects.insert(0, "(1|question_id)")
    if include_answer_re:
        vc_formulas["answer"] = "0 + C(answer_id)"
        random_effects.insert(0, "(1|answer_id)")
    if include_student_vc:
        vc_formulas["student"] = "0 + C(student_id)"
        random_effects.append("(1|student_id)")

    warnings_list: List[str] = []
    converged = True
    try:
        model = bmg.BinomialBayesMixedGLM.from_formula(formula, vc_formulas=vc_formulas, data=data)
        fit = model.fit_vb()
    except Exception as e:
        warnings_list.append(f"BinomialBayesMixedGLM failed ({e}); retrying rubric-only VC.")
        converged = False
        model = bmg.BinomialBayesMixedGLM.from_formula(
            formula, vc_formulas={"rubric": "0 + C(rubric_id)"}, data=data
        )
        fit = model.fit_vb()

    singular_fit = None
    try:
        vcp_mean = np.asarray(getattr(fit, "vcp_mean", []), dtype=float)
        if vcp_mean.size > 0 and np.nanmin(np.abs(vcp_mean)) < 1e-8:
            singular_fit = True
            warnings_list.append("Potential singular fit: variance components near zero.")
        else:
            singular_fit = False
    except Exception:
        singular_fit = None

    return fit, warnings_list, converged, random_effects, singular_fit


def _predict_fixed_effects(
    fit: Any,
    new_data: pd.DataFrame,
    outcome_type: Literal["continuous", "binary"],
    fallback_formula: Optional[str] = None,
) -> np.ndarray:
    try:
        import patsy
        design_info = fit.model.data.design_info
        X = patsy.build_design_matrices([design_info], new_data)[0]
    except Exception:
        if fallback_formula is None:
            raise
        import patsy
        X = patsy.dmatrix(fallback_formula, new_data, return_type="dataframe")

    beta = fit.fe_params if hasattr(fit, "fe_params") else fit.fe_mean
    lin_pred = np.asarray(X) @ np.asarray(beta)
    if outcome_type == "binary":
        return 1.0 / (1.0 + np.exp(-lin_pred))
    return lin_pred


def estimate_feature_effect_nonlinear(
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    outcome_type: Literal["continuous", "binary"] = "continuous",
    backend: Literal["lme4", "statsmodels"] = "lme4",
    method: Literal["spline", "binned", "poly"] = "spline",
    spline_df: int = 4,
    poly_degree: int = 2,
    n_bins: int = 5,
    n_grid: int = 50,
    standardize_feature: bool = True,
    standardize_outcome: bool = False,
    transform_bounded: Literal["none", "logit_clip"] = "none",
    include_student_vc_glmm: bool = False,
    alpha: float = 0.05,
    optimizer: str = "powell",
    glmer_nAGQ: int = 1,
    maxfun: int = 100000,
    bootstrap_n: int = 0,
    bootstrap_cluster: Literal["student", "row"] = "student",
    ci_method: Literal["percentile", "bca"] = "percentile",
    ci_type: Literal["pointwise", "simultaneous"] = "pointwise",
    jackknife_max: Optional[int] = 200,
    seed: int = 123,
    eps: float = 1e-8,
    log_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> tuple[NonlinearFeatureEffectResult, pd.DataFrame]:
    warnings_list: List[str] = []
    rng = np.random.default_rng(seed)

    data = _prepare_nonlinear_base_data(feature, outcome, student_ids, question_ids, rubric_ids)

    include_answer_re = _should_include_answer_re(data)
    if not include_answer_re:
        n_levels = int(data["answer_id"].nunique())
        max_count = int(data["answer_id"].value_counts().max()) if n_levels else 0
        warnings_list.append(
            "Dropping answer_id random effect (answer_id has no repeats: "
            f"levels={n_levels}, max_count={max_count})."
        )

    feature_type = detect_feature_type(
        data["feature"].to_numpy(),
        data["student_id"].to_numpy(),
        data["question_id"].to_numpy(),
        data["rubric_id"].to_numpy(),
        data["answer_id"].to_numpy(),
        eps=eps,
    )
    include_question_fe, include_question_re, question_effect_mode, n_questions = _select_question_effects(
        data["question_id"].to_numpy(),
        feature_type,
    )
    if question_effect_mode != "none":
        notice_key = (question_effect_mode, n_questions, QUESTION_FE_MAX_LEVELS_DEFAULT)
        if notice_key not in _QUESTION_EFFECT_NOTICE_SEEN:
            _QUESTION_EFFECT_NOTICE_SEEN.add(notice_key)
            print(
                "[auto-question-effects] "
                f"mode={question_effect_mode} "
                f"n_questions={n_questions} "
                f"fe_max={QUESTION_FE_MAX_LEVELS_DEFAULT}",
                file=sys.stderr,
            )
        warnings_list.append(
            "Question effect mode: "
            f"{question_effect_mode} (n_questions={n_questions}, fe_max={QUESTION_FE_MAX_LEVELS_DEFAULT})."
        )
    if feature_type == "question_constant":
        warnings_list.append(
            "Question-constant feature (n≈4 questions). Nonlinear inference is exploratory."
        )

    feat_mean = float(data["feature"].mean())
    feat_sd = float(data["feature"].std())
    if not np.isfinite(feat_sd) or feat_sd < eps:
        raise ValueError(f"Feature has near-zero variance (std={feat_sd:.2e} < eps={eps})")

    if standardize_feature:
        data["feature_z"] = (data["feature"] - feat_mean) / feat_sd
    else:
        data["feature_z"] = data["feature"]

    outcome_transformed = "none"
    if outcome_type == "continuous":
        out_min = float(data["outcome"].min())
        out_max = float(data["outcome"].max())
        if np.isfinite(out_min) and np.isfinite(out_max) and out_min >= 0.0 and out_max <= 1.0:
            if transform_bounded == "logit_clip":
                clip_eps = 1e-6
                y = data["outcome"].clip(clip_eps, 1.0 - clip_eps)
                data["outcome"] = np.log(y / (1.0 - y))
                outcome_transformed = "logit_clip"
                warnings_list.append("Applied logit_clip transform to bounded [0,1] outcome.")
            else:
                warnings_list.append(
                    "Bounded outcome in [0,1]. Gaussian LMM OK for screening; "
                    "use transform_bounded='logit_clip' if residuals are problematic."
                )

    if outcome_type == "continuous" and standardize_outcome:
        y_mean = float(data["outcome"].mean())
        y_sd = float(data["outcome"].std())
        if not np.isfinite(y_sd) or y_sd < eps:
            raise ValueError(f"Outcome has near-zero variance (std={y_sd:.2e})")
        data["outcome_z"] = (data["outcome"] - y_mean) / y_sd
    else:
        y_mean = 0.0
        y_sd = 1.0
        data["outcome_z"] = data["outcome"]

    basis = ""
    bin_edges = None
    bin_categories = None
    n_bins_used = None

    basis_r = ""
    if method == "spline":
        max_df = int(min(6, max(3, np.floor(len(data) / 10))))
        if spline_df > max_df:
            warnings_list.append(f"Reducing spline df from {spline_df} to {max_df} to avoid overfitting.")
            spline_df = max_df
        basis = f"bs(feature_z, df={spline_df}, degree=3, intercept=FALSE)"
        basis_r = f"bs(feature_z, df={spline_df}, degree=3, intercept=FALSE)"
    elif method == "binned":
        try:
            data["feature_bin"], bin_edges = pd.qcut(
                data["feature_z"], q=n_bins, retbins=True, duplicates="drop"
            )
        except ValueError:
            data["feature_bin"], bin_edges = pd.cut(
                data["feature_z"], bins=n_bins, retbins=True, duplicates="drop"
            )
        data["feature_bin"] = pd.Categorical(data["feature_bin"])
        bin_categories = data["feature_bin"].cat.categories
        n_bins_used = len(bin_categories)
        if n_bins_used < 2:
            raise ValueError("Binning produced <2 bins; cannot fit binned nonlinear model.")
        if n_bins_used != n_bins:
            warnings_list.append(f"Binning used {n_bins_used} bins (requested {n_bins}).")
        basis = "C(feature_bin)"
        basis_r = "feature_bin"
    elif method == "poly":
        if poly_degree < 2:
            warnings_list.append("poly_degree < 2; falling back to linear term only.")
            poly_degree = 1
        terms = ["feature_z"]
        for deg in range(2, poly_degree + 1):
            terms.append(f"I(feature_z ** {deg})")
        basis = " + ".join(terms)
        r_terms = ["feature_z"]
        for deg in range(2, poly_degree + 1):
            r_terms.append(f"I(feature_z^{deg})")
        basis_r = " + ".join(r_terms)
    else:
        raise ValueError(f"Unknown nonlinear method: {method}")

    if include_question_fe:
        formula_lin = "outcome_z ~ feature_z + C(question_id)"
        formula_nl = f"outcome_z ~ {basis} + C(question_id)"
        fixed_linear = ["feature_z", "C(question_id)"]
    else:
        formula_lin = "outcome_z ~ feature_z"
        formula_nl = f"outcome_z ~ {basis}"
        fixed_linear = ["feature_z"]

    formula_lin_r = "outcome_z ~ feature_z"
    formula_nl_r = f"outcome_z ~ {basis_r}"
    if include_question_fe:
        formula_lin_r += " + question_id"
        formula_nl_r += " + question_id"
    if outcome_type == "continuous":
        re_terms = ["(1|student_id)"]
        if include_question_re:
            re_terms.append("(1|question_id)")
        if include_answer_re:
            re_terms.append("(1|answer_id)")
    else:
        re_terms = []
        if include_question_re:
            re_terms.append("(1|question_id)")
        if include_answer_re:
            re_terms.append("(1|answer_id)")
        if include_student_vc_glmm:
            re_terms.append("(1|student_id)")
    if data["rubric_id"].nunique() >= 2:
        re_terms.append("(1|rubric_id)")
    formula_lin_r = f"{formula_lin_r} + {' + '.join(re_terms)}"
    formula_nl_r = f"{formula_nl_r} + {' + '.join(re_terms)}"

    fit_lin = None
    fit_nl = None
    aic_linear = np.nan
    bic_linear = np.nan
    aic_nl = np.nan
    bic_nl = np.nan
    lrt_stat = np.nan
    lrt_df = np.nan
    lrt_p = np.nan
    singular_fit = None
    converged = True

    use_lme4 = backend == "lme4"
    if use_lme4:
        opt_lme4 = _normalize_lme4_optimizer(optimizer)
        if opt_lme4 != optimizer:
            warnings_list.append(f"Mapped optimizer '{optimizer}' to lme4 optimizer '{opt_lme4}'.")
            optimizer = opt_lme4
    if use_lme4:
        try:
            fit_pack = _fit_lme4_nonlinear_inprocess(
                data=data,
                formula_lin=formula_lin_r,
                formula_nl=formula_nl_r,
                include_question_fe=include_question_fe,
                include_student_vc_glmm=include_student_vc_glmm,
                include_rubric_re=(data["rubric_id"].nunique() >= 2),
                is_glmm=(outcome_type == "binary"),
                glmer_nAGQ=glmer_nAGQ,
                optimizer=optimizer,
                maxfun=maxfun,
            )
            fit_lin = fit_pack["fit_lin"]
            fit_nl = fit_pack["fit_full"]
            aic_linear = fit_pack["aic_lin"]
            bic_linear = fit_pack["bic_lin"]
            aic_nl = fit_pack["aic_full"]
            bic_nl = fit_pack["bic_full"]
            lrt_stat = fit_pack["lrt_stat"]
            lrt_df = fit_pack["lrt_df"]
            lrt_p = fit_pack["lrt_p"]
            singular_fit = fit_pack["singular"]
            converged = fit_pack["converged"]
            warnings_list.extend(fit_pack["warnings"])
        except Exception as e:
            warnings_list.append(f"lme4 backend failed; no fallback: {e}")
            raise RuntimeError(f"lme4 backend failed (no fallback): {e}") from e

    if not use_lme4:
        if outcome_type == "continuous":
            fit_lin, warn_lin, conv_lin, _, sing_lin = _fit_mixedlm_generic(
                data, formula_lin, optimizer, include_question_re
            )
            fit_nl, warn_nl, conv_nl, _, sing_nl = _fit_mixedlm_generic(
                data, formula_nl, optimizer, include_question_re
            )
            warnings_list.extend(warn_lin)
            warnings_list.extend(warn_nl)
            converged = bool(conv_lin and conv_nl)
            singular_fit = True if (sing_lin or sing_nl) else False if (sing_lin is not None and sing_nl is not None) else None

            aic_linear = float(getattr(fit_lin, "aic", np.nan))
            bic_linear = float(getattr(fit_lin, "bic", np.nan))
            aic_nl = float(getattr(fit_nl, "aic", np.nan))
            bic_nl = float(getattr(fit_nl, "bic", np.nan))

            lrt_stat = 2.0 * (float(getattr(fit_nl, "llf", np.nan)) - float(getattr(fit_lin, "llf", np.nan)))
            df_lin = float(getattr(fit_lin, "df_modelwc", len(fit_lin.params)))
            df_nl = float(getattr(fit_nl, "df_modelwc", len(fit_nl.params)))
            lrt_df = max(0.0, df_nl - df_lin)
            if np.isfinite(lrt_stat) and lrt_stat >= 0 and lrt_df > 0:
                lrt_p = float(stats.chi2.sf(lrt_stat, lrt_df))
            else:
                lrt_stat = np.nan
                lrt_df = np.nan
                lrt_p = np.nan
        else:
            fit_lin, warn_lin, conv_lin, _, sing_lin = _fit_glmm_vb_generic(
                data, formula_lin, include_student_vc_glmm, include_question_re
            )
            fit_nl, warn_nl, conv_nl, _, sing_nl = _fit_glmm_vb_generic(
                data, formula_nl, include_student_vc_glmm, include_question_re
            )
            warnings_list.extend(warn_lin)
            warnings_list.extend(warn_nl)
            warnings_list.append("VB posterior approximation (screening only).")
            converged = bool(conv_lin and conv_nl)
            singular_fit = True if (sing_lin or sing_nl) else False if (sing_lin is not None and sing_nl is not None) else None

    question_levels = pd.Index(pd.unique(data["question_id"]))
    question_weights = data["question_id"].value_counts(normalize=True).to_dict()

    qs = np.linspace(0.02, 0.98, max(5, n_grid))
    x_grid_raw = np.quantile(data["feature"].to_numpy(), qs)
    x_grid_z = (x_grid_raw - feat_mean) / feat_sd if standardize_feature else x_grid_raw

    if method == "binned":
        bin_centers_z = []
        bin_centers_raw = []
        for i in range(len(bin_edges) - 1):
            center = 0.5 * (bin_edges[i] + bin_edges[i + 1])
            bin_centers_z.append(center)
            bin_centers_raw.append(center * feat_sd + feat_mean if standardize_feature else center)
        x_grid_z = np.array(bin_centers_z)
        x_grid_raw = np.array(bin_centers_raw)

    def _make_pred_frame(x_vals_z: np.ndarray, q_levels: Any) -> pd.DataFrame:
        rows = []
        for idx, xz in enumerate(x_vals_z):
            if include_question_fe:
                for q in q_levels:
                    rows.append({"feature_z": xz, "question_id": q, "grid_idx": idx})
            else:
                rows.append({"feature_z": xz, "grid_idx": idx})
        pred_df = pd.DataFrame(rows)
        if method == "binned":
            pred_df["feature_bin"] = pd.cut(
                pred_df["feature_z"], bins=bin_edges, include_lowest=True
            )
            pred_df["feature_bin"] = pd.Categorical(pred_df["feature_bin"], categories=bin_categories)
        return pred_df

    pred_df = _make_pred_frame(x_grid_z, question_levels)
    if use_lme4:
        preds = _predict_lme4_fixed(
            fit_nl,
            pred_df,
            is_glmm=(outcome_type == "binary"),
        )
    else:
        preds = _predict_fixed_effects(
            fit_nl, pred_df, outcome_type, fallback_formula=formula_nl.replace("outcome_z ~ ", "")
        )
    pred_df["pred"] = preds

    if include_question_fe:
        pred_df["weight"] = pred_df["question_id"].map(question_weights).fillna(0.0)
        curve = pred_df.groupby("grid_idx").apply(
            lambda g: np.average(g["pred"], weights=g["weight"])
        ).to_numpy()
    else:
        curve = pred_df.groupby("grid_idx")["pred"].mean().to_numpy()

    if outcome_type == "continuous":
        if standardize_outcome:
            curve = curve * y_sd + y_mean
        if outcome_transformed == "logit_clip":
            curve = 1.0 / (1.0 + np.exp(-curve))

    def _interp_curve(x_raw: float) -> float:
        return float(np.interp(x_raw, x_grid_raw, curve))

    x10, x90 = np.quantile(data["feature"].to_numpy(), [0.1, 0.9])
    delta_10_90 = _interp_curve(x90) - _interp_curve(x10)

    if len(x_grid_raw) >= 3:
        grad = np.gradient(curve, x_grid_raw)
        ame = float(np.nanmean(grad))
    else:
        ame = float(delta_10_90 / (x90 - x10)) if x90 != x10 else np.nan

    delta_ci = (np.nan, np.nan)
    ame_ci = (np.nan, np.nan)
    curve_ci_low = np.full_like(curve, np.nan, dtype=float)
    curve_ci_high = np.full_like(curve, np.nan, dtype=float)
    jackknife_n = None

    if bootstrap_n and bootstrap_n > 0:
        boot_delta = []
        boot_ame = []
        boot_curves = []
        use_student_bootstrap = bootstrap_cluster == "student" and data["student_id"].nunique() > 1
        if bootstrap_cluster == "student" and not use_student_bootstrap:
            warnings_list.append("Student bootstrap requested but only 1 student; falling back to row bootstrap.")

        iterator = range(bootstrap_n)
        if log_progress:
            miniters = max(1, bootstrap_n // 20)
            iterator = tqdm(
                iterator,
                total=bootstrap_n,
                desc=progress_desc or "Nonlinear bootstrap",
                unit="boot",
                dynamic_ncols=True,
                leave=False,
                miniters=miniters,
                mininterval=2.0,
                disable=False,
            )
        for _ in iterator:
            if use_student_bootstrap:
                students = pd.unique(data["student_id"])
                sample_ids = rng.choice(students, size=len(students), replace=True)
                pieces = []
                for i, sid in enumerate(sample_ids):
                    block = data[data["student_id"] == sid].copy()
                    block["student_id"] = f"{sid}__boot{i}"
                    pieces.append(block)
                boot_df = pd.concat(pieces, ignore_index=True)
            else:
                boot_df = data.sample(
                    frac=1.0,
                    replace=True,
                    random_state=int(rng.integers(1, 2**31 - 1)),
                ).copy()
            boot_df["student_id"] = boot_df["student_id"].astype(str)
            boot_df["question_id"] = boot_df["question_id"].astype(str)
            boot_df["rubric_id"] = boot_df["rubric_id"].astype(str)
            boot_df["answer_id"] = pd.factorize(
                pd.MultiIndex.from_arrays([boot_df["student_id"], boot_df["question_id"]])
            )[0].astype(str)

            boot_feat_mean = float(boot_df["feature"].mean())
            boot_feat_sd = float(boot_df["feature"].std())
            if not np.isfinite(boot_feat_sd) or boot_feat_sd < eps:
                continue

            boot_df["feature_z"] = (boot_df["feature"] - boot_feat_mean) / boot_feat_sd if standardize_feature else boot_df["feature"]
            if outcome_type == "continuous":
                if transform_bounded == "logit_clip":
                    clip_eps = 1e-6
                    y = boot_df["outcome"].clip(clip_eps, 1.0 - clip_eps)
                    boot_df["outcome"] = np.log(y / (1.0 - y))
                if standardize_outcome:
                    boot_y_mean = float(boot_df["outcome"].mean())
                    boot_y_sd = float(boot_df["outcome"].std())
                    boot_df["outcome_z"] = (boot_df["outcome"] - boot_y_mean) / boot_y_sd
                else:
                    boot_y_mean = 0.0
                    boot_y_sd = 1.0
                    boot_df["outcome_z"] = boot_df["outcome"]
            else:
                boot_y_mean = 0.0
                boot_y_sd = 1.0
                boot_df["outcome_z"] = boot_df["outcome"]

            if method == "binned":
                try:
                    boot_df["feature_bin"], boot_edges = pd.qcut(
                        boot_df["feature_z"], q=n_bins, retbins=True, duplicates="drop"
                    )
                except ValueError:
                    boot_df["feature_bin"], boot_edges = pd.cut(
                        boot_df["feature_z"], bins=n_bins, retbins=True, duplicates="drop"
                    )
                boot_df["feature_bin"] = pd.Categorical(boot_df["feature_bin"])
                boot_bin_edges = boot_edges
                boot_bin_categories = boot_df["feature_bin"].cat.categories
            else:
                boot_bin_edges = None
                boot_bin_categories = None

            try:
                if use_lme4:
                    boot_fit_pack = _fit_lme4_nonlinear_inprocess(
                        data=boot_df,
                        formula_lin=formula_lin_r,
                        formula_nl=formula_nl_r,
                        include_question_fe=include_question_fe,
                        include_student_vc_glmm=include_student_vc_glmm,
                        include_rubric_re=(boot_df["rubric_id"].nunique() >= 2),
                        is_glmm=(outcome_type == "binary"),
                        glmer_nAGQ=glmer_nAGQ,
                        optimizer=optimizer,
                        maxfun=maxfun,
                    )
                    boot_fit = boot_fit_pack["fit_full"]
                else:
                    if outcome_type == "continuous":
                        boot_fit, _, _, _, _ = _fit_mixedlm_generic(
                            boot_df, formula_nl, optimizer, include_question_re
                        )
                    else:
                        boot_fit, _, _, _, _ = _fit_glmm_vb_generic(
                            boot_df, formula_nl, include_student_vc_glmm, include_question_re
                        )
            except Exception:
                continue

            boot_xz = (x_grid_raw - boot_feat_mean) / boot_feat_sd if standardize_feature else x_grid_raw
            if method == "binned":
                boot_xz = []
                boot_xraw = []
                for i in range(len(boot_bin_edges) - 1):
                    center = 0.5 * (boot_bin_edges[i] + boot_bin_edges[i + 1])
                    boot_xz.append(center)
                    boot_xraw.append(center * boot_feat_sd + boot_feat_mean if standardize_feature else center)
                boot_xz = np.array(boot_xz)
                boot_xraw = np.array(boot_xraw)
            else:
                boot_xraw = x_grid_raw

            boot_pred_df = _make_pred_frame(boot_xz, pd.Index(pd.unique(boot_df["question_id"])))
            if method == "binned":
                boot_pred_df["feature_bin"] = pd.cut(
                    boot_pred_df["feature_z"], bins=boot_bin_edges, include_lowest=True
                )
                boot_pred_df["feature_bin"] = pd.Categorical(
                    boot_pred_df["feature_bin"], categories=boot_bin_categories
                )

            try:
                if use_lme4:
                    boot_preds = _predict_lme4_fixed(
                        boot_fit,
                        boot_pred_df,
                        is_glmm=(outcome_type == "binary"),
                    )
                else:
                    boot_preds = _predict_fixed_effects(
                        boot_fit, boot_pred_df, outcome_type, fallback_formula=formula_nl.replace("outcome_z ~ ", "")
                    )
            except Exception:
                continue

            boot_pred_df["pred"] = boot_preds
            if include_question_fe:
                boot_weights = boot_df["question_id"].value_counts(normalize=True).to_dict()
                boot_pred_df["weight"] = boot_pred_df["question_id"].map(boot_weights).fillna(0.0)
                boot_curve = boot_pred_df.groupby("grid_idx").apply(
                    lambda g: np.average(g["pred"], weights=g["weight"])
                ).to_numpy()
            else:
                boot_curve = boot_pred_df.groupby("grid_idx")["pred"].mean().to_numpy()

            if outcome_type == "continuous":
                if standardize_outcome:
                    boot_curve = boot_curve * boot_y_sd + boot_y_mean
                if outcome_transformed == "logit_clip":
                    boot_curve = 1.0 / (1.0 + np.exp(-boot_curve))

            if method == "binned":
                if len(boot_curve) != len(curve):
                    continue
                boot_xraw = x_grid_raw

            boot_delta.append(float(np.interp(x90, boot_xraw, boot_curve) - np.interp(x10, boot_xraw, boot_curve)))
            if len(boot_xraw) >= 3:
                boot_grad = np.gradient(boot_curve, boot_xraw)
                boot_ame.append(float(np.nanmean(boot_grad)))
            else:
                boot_ame.append(float(boot_delta[-1] / (x90 - x10)) if x90 != x10 else np.nan)

            if len(boot_curve) == len(curve):
                boot_curves.append(boot_curve)

        jack_curves = []
        if ci_method == "bca":
            # Jackknife for BCa (can be expensive; capped by jackknife_max)
            if bootstrap_cluster == "student" and data["student_id"].nunique() > 1:
                unit_col = "student_id"
                units = pd.unique(data["student_id"])
            else:
                unit_col = None
                units = np.arange(len(data))
            if jackknife_max is not None and len(units) > jackknife_max:
                warnings_list.append(
                    f"BCa jackknife capped at {jackknife_max} units (requested {len(units)})."
                )
                units = rng.choice(units, size=jackknife_max, replace=False)
            for unit in units:
                if unit_col:
                    jack_df = data[data[unit_col] != unit].copy()
                else:
                    jack_df = data.drop(index=unit).copy()
                if len(jack_df) < 10:
                    continue
                try:
                    # Recompute ids to keep factor levels valid
                    jack_df["student_id"] = jack_df["student_id"].astype(str)
                    jack_df["question_id"] = jack_df["question_id"].astype(str)
                    jack_df["rubric_id"] = jack_df["rubric_id"].astype(str)
                    jack_df["answer_id"] = pd.factorize(
                        pd.MultiIndex.from_arrays([jack_df["student_id"], jack_df["question_id"]])
                    )[0].astype(str)

                    jack_feat_mean = float(jack_df["feature"].mean())
                    jack_feat_sd = float(jack_df["feature"].std())
                    if not np.isfinite(jack_feat_sd) or jack_feat_sd < eps:
                        continue
                    jack_df["feature_z"] = (
                        (jack_df["feature"] - jack_feat_mean) / jack_feat_sd
                        if standardize_feature
                        else jack_df["feature"]
                    )
                    if outcome_type == "continuous":
                        if transform_bounded == "logit_clip":
                            clip_eps = 1e-6
                            y = jack_df["outcome"].clip(clip_eps, 1.0 - clip_eps)
                            jack_df["outcome"] = np.log(y / (1.0 - y))
                        if standardize_outcome:
                            jack_y_mean = float(jack_df["outcome"].mean())
                            jack_y_sd = float(jack_df["outcome"].std())
                            jack_df["outcome_z"] = (jack_df["outcome"] - jack_y_mean) / jack_y_sd
                        else:
                            jack_df["outcome_z"] = jack_df["outcome"]
                    else:
                        jack_df["outcome_z"] = jack_df["outcome"]

                    if method == "binned":
                        try:
                            jack_df["feature_bin"], jack_edges = pd.qcut(
                                jack_df["feature_z"], q=n_bins, retbins=True, duplicates="drop"
                            )
                        except ValueError:
                            jack_df["feature_bin"], jack_edges = pd.cut(
                                jack_df["feature_z"], bins=n_bins, retbins=True, duplicates="drop"
                            )
                        jack_df["feature_bin"] = pd.Categorical(jack_df["feature_bin"])
                        jack_bin_edges = jack_edges
                        jack_bin_categories = jack_df["feature_bin"].cat.categories
                    else:
                        jack_bin_edges = None
                        jack_bin_categories = None

                    if use_lme4:
                        jack_fit_pack = _fit_lme4_nonlinear_inprocess(
                            data=jack_df,
                            formula_lin=formula_lin_r,
                            formula_nl=formula_nl_r,
                            include_question_fe=include_question_fe,
                            include_student_vc_glmm=include_student_vc_glmm,
                            include_rubric_re=(jack_df["rubric_id"].nunique() >= 2),
                            is_glmm=(outcome_type == "binary"),
                            glmer_nAGQ=glmer_nAGQ,
                            optimizer=optimizer,
                            maxfun=maxfun,
                        )
                        jack_fit = jack_fit_pack["fit_full"]
                    else:
                        if outcome_type == "continuous":
                            jack_fit, _, _, _, _ = _fit_mixedlm_generic(
                                jack_df, formula_nl, optimizer, include_question_re
                            )
                        else:
                            jack_fit, _, _, _, _ = _fit_glmm_vb_generic(
                                jack_df, formula_nl, include_student_vc_glmm, include_question_re
                            )

                    jack_xz = (
                        (x_grid_raw - jack_feat_mean) / jack_feat_sd
                        if standardize_feature
                        else x_grid_raw
                    )
                    if method == "binned":
                        jack_pred_df = _make_pred_frame(jack_xz, pd.Index(pd.unique(jack_df["question_id"])))
                        jack_pred_df["feature_bin"] = pd.cut(
                            jack_pred_df["feature_z"], bins=jack_bin_edges, include_lowest=True
                        )
                        jack_pred_df["feature_bin"] = pd.Categorical(
                            jack_pred_df["feature_bin"], categories=jack_bin_categories
                        )
                    else:
                        jack_pred_df = _make_pred_frame(jack_xz, pd.Index(pd.unique(jack_df["question_id"])))

                    if use_lme4:
                        jack_preds = _predict_lme4_fixed(
                            jack_fit,
                            jack_pred_df,
                            is_glmm=(outcome_type == "binary"),
                        )
                    else:
                        jack_preds = _predict_fixed_effects(
                            jack_fit,
                            jack_pred_df,
                            outcome_type,
                            fallback_formula=formula_nl.replace("outcome_z ~ ", ""),
                        )
                    jack_pred_df["pred"] = jack_preds
                    if include_question_fe:
                        jack_weights = jack_df["question_id"].value_counts(normalize=True).to_dict()
                        jack_pred_df["weight"] = jack_pred_df["question_id"].map(jack_weights).fillna(0.0)
                        jack_curve = jack_pred_df.groupby("grid_idx").apply(
                            lambda g: np.average(g["pred"], weights=g["weight"])
                        ).to_numpy()
                    else:
                        jack_curve = jack_pred_df.groupby("grid_idx")["pred"].mean().to_numpy()

                    if outcome_type == "continuous":
                        if standardize_outcome:
                            jack_curve = jack_curve * jack_y_sd + jack_y_mean
                        if outcome_transformed == "logit_clip":
                            jack_curve = 1.0 / (1.0 + np.exp(-jack_curve))

                    if len(jack_curve) == len(curve):
                        jack_curves.append(jack_curve)
                except Exception:
                    continue

            if jack_curves:
                jackknife_n = len(jack_curves)

        if boot_delta:
            if ci_method == "bca" and jack_curves:
                jack_delta = []
                for jc in jack_curves:
                    jack_delta.append(float(np.interp(x90, x_grid_raw, jc) - np.interp(x10, x_grid_raw, jc)))
                delta_ci = _compute_bca_ci_from_jackknife(
                    np.asarray(boot_delta), float(delta_10_90), np.asarray(jack_delta), alpha=alpha
                )
            else:
                if ci_method == "bca":
                    warnings_list.append("BCa requested but jackknife unavailable; using percentile CI for delta.")
                delta_ci = tuple(np.nanpercentile(boot_delta, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        if boot_ame:
            if ci_method == "bca" and jack_curves:
                jack_ame = []
                for jc in jack_curves:
                    if len(x_grid_raw) >= 3:
                        jack_grad = np.gradient(jc, x_grid_raw)
                        jack_ame.append(float(np.nanmean(jack_grad)))
                    else:
                        jack_ame.append(float((np.interp(x90, x_grid_raw, jc) - np.interp(x10, x_grid_raw, jc)) / (x90 - x10)))
                ame_ci = _compute_bca_ci_from_jackknife(
                    np.asarray(boot_ame), float(ame), np.asarray(jack_ame), alpha=alpha
                )
            else:
                if ci_method == "bca":
                    warnings_list.append("BCa requested but jackknife unavailable; using percentile CI for AME.")
                ame_ci = tuple(np.nanpercentile(boot_ame, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        if boot_curves:
            boot_mat = np.asarray(boot_curves)
            if ci_type == "simultaneous":
                if ci_method == "bca":
                    warnings_list.append("Simultaneous band uses percentile method; BCa not supported for simultaneous bands.")
                sup_dev = np.nanmax(np.abs(boot_mat - curve), axis=1)
                q = float(np.nanpercentile(sup_dev, 100 * (1 - alpha)))
                curve_ci_low = curve - q
                curve_ci_high = curve + q
            else:
                if ci_method == "bca" and jack_curves:
                    jack_mat = np.asarray(jack_curves)
                    ci_low = []
                    ci_high = []
                    for i in range(len(curve)):
                        ci_i = _compute_bca_ci_from_jackknife(
                            boot_mat[:, i], float(curve[i]), jack_mat[:, i], alpha=alpha
                        )
                        ci_low.append(ci_i[0])
                        ci_high.append(ci_i[1])
                    curve_ci_low = np.asarray(ci_low)
                    curve_ci_high = np.asarray(ci_high)
                else:
                    if ci_method == "bca":
                        warnings_list.append("BCa requested but jackknife unavailable; using percentile curve CI.")
                    curve_ci_low = np.nanpercentile(boot_mat, 100 * alpha / 2, axis=0)
                    curve_ci_high = np.nanpercentile(boot_mat, 100 * (1 - alpha / 2), axis=0)

    resid_skew = None
    resid_kurt = None
    if outcome_type == "continuous":
        try:
            if use_lme4:
                fitted = _predict_lme4_fixed(
                    fit_nl,
                    data.assign(grid_idx=0),
                    is_glmm=False,
                )
            else:
                fitted = _predict_fixed_effects(
                    fit_nl,
                    data.assign(grid_idx=0),
                    outcome_type,
                    fallback_formula=formula_nl.replace("outcome_z ~ ", ""),
                )
            resid = data["outcome_z"].to_numpy() - fitted
            resid_skew = float(stats.skew(resid, nan_policy="omit"))
            resid_kurt = float(stats.kurtosis(resid, nan_policy="omit"))
        except Exception:
            resid_skew = None
            resid_kurt = None

    question_re = " + (1|question_id)" if include_question_re else ""
    summary = NonlinearFeatureEffectResult(
        delta_10_90=float(delta_10_90),
        delta_ci_low=float(delta_ci[0]) if np.isfinite(delta_ci[0]) else np.nan,
        delta_ci_high=float(delta_ci[1]) if np.isfinite(delta_ci[1]) else np.nan,
        ame=float(ame),
        ame_ci_low=float(ame_ci[0]) if np.isfinite(ame_ci[0]) else np.nan,
        ame_ci_high=float(ame_ci[1]) if np.isfinite(ame_ci[1]) else np.nan,
        aic_linear=aic_linear,
        bic_linear=bic_linear,
        aic_nonlinear=aic_nl,
        bic_nonlinear=bic_nl,
        lrt_stat=float(lrt_stat) if np.isfinite(lrt_stat) else np.nan,
        lrt_df=float(lrt_df) if np.isfinite(lrt_df) else np.nan,
        lrt_p_value=float(lrt_p) if np.isfinite(lrt_p) else np.nan,
        n_obs=int(len(data)),
        feature_type=feature_type,
        outcome_type=outcome_type,
        method=method,
        basis=basis,
        model_spec_linear=f"outcome ~ {' + '.join(fixed_linear)}{question_re}",
        model_spec_nonlinear=f"outcome ~ {basis}"
        + (" + C(question_id)" if include_question_fe else "")
        + question_re,
        outcome_transformed=outcome_transformed,
        converged=converged,
        warnings=warnings_list,
        residual_skew=resid_skew,
        residual_kurtosis=resid_kurt,
        singular_fit=singular_fit,
        ci_method=ci_method,
        ci_type=ci_type,
        jackknife_n=jackknife_n,
    )

    curve_df = pd.DataFrame(
        {
            "feature_value": x_grid_raw,
            "feature_z": x_grid_z,
            "pred_mean": curve,
            "pred_ci_low": curve_ci_low,
            "pred_ci_high": curve_ci_high,
        }
    )

    if method == "binned":
        curve_df["bin_index"] = np.arange(len(curve_df))

    return summary, curve_df


def _fit_lmm_continuous(
    data: pd.DataFrame,
    include_question_fe: bool,
    include_question_re: bool,
    alpha: float = 0.05,
    optimizer: str = "powell",
    include_answer_re: bool = True,
) -> FeatureEffectResult:
    """Backend: frequentist MixedLM for continuous outcomes.
    
    Args:
        optimizer: Optimization method for MixedLM.fit(). Options:
            - 'powell' (default): Derivative-free, robust but slower
            - 'lbfgs': L-BFGS-B, fast gradient-based
            - 'bfgs': BFGS, gradient-based
            - 'cg': Conjugate gradient
            - 'nm': Nelder-Mead (derivative-free, faster than powell)
    """
    if include_question_fe:
        formula = "outcome_z ~ feature_z + C(question_id)"
        fixed_effects = ["feature_z", "C(question_id)"]
    else:
        formula = "outcome_z ~ feature_z"
        fixed_effects = ["feature_z"]

    vc_formula: Dict[str, str] = {}
    random_effects = ["(1|student_id)"]
    if include_question_re:
        vc_formula["question"] = "0 + C(question_id)"
        random_effects.append("(1|question_id)")
    if include_answer_re:
        vc_formula["answer"] = "0 + C(answer_id)"
        random_effects.append("(1|answer_id)")

    # Only include rubric VC if there is variability.
    n_rubric = int(pd.Series(data["rubric_id"]).nunique())
    if n_rubric >= 2:
        vc_formula["rubric"] = "0 + C(rubric_id)"
        random_effects.append("(1|rubric_id)")

    warnings_list: List[str] = []
    converged = True
    print(f"Fitting MixedLM with optimizer='{optimizer}'... in statsmodels")
    try:
        fit = sm.MixedLM.from_formula(
            formula,
            data=data,
            groups="student_id",
            re_formula="1",
            vc_formula=vc_formula if vc_formula else None,
        ).fit(reml=True, method=optimizer)
        if hasattr(fit, "converged") and not fit.converged:
            converged = False
            warnings_list.append(f"MixedLM did not converge (optimizer={optimizer}).")
    except Exception as e:
        if "answer" in vc_formula:
            warnings_list.append(f"MixedLM failed ({e}); retrying without answer VC.")
            vc_formula = {k: v for k, v in vc_formula.items() if k != "answer"}
        else:
            warnings_list.append(f"MixedLM failed ({e}); retrying without VC.")
            vc_formula = {}
        fit = sm.MixedLM.from_formula(
            formula,
            data=data,
            groups="student_id",
            re_formula="1",
            vc_formula=vc_formula if vc_formula else None,
        ).fit(reml=True, method=optimizer)

    beta = float(fit.params.get("feature_z", np.nan))
    se = float(fit.bse.get("feature_z", np.nan))
    z_stat = beta / se if np.isfinite(beta) and np.isfinite(se) and se > 0 else np.nan
    p_value = 2 * (1 - norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = beta - z_crit * se if np.isfinite(se) else np.nan
    ci_high = beta + z_crit * se if np.isfinite(se) else np.nan

    return FeatureEffectResult(
        beta=beta,
        se=se,
        z=z_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        method="MixedLM_REML",
        inference_type="frequentist",
        fixed_effects=fixed_effects,
        random_effects=random_effects,
        converged=converged,
        warnings=warnings_list,
    )


def _configure_rpy2_environment() -> None:
    """
    Best-effort environment setup so rpy2 can find and run R reliably.

    This is intentionally defensive because rpy2 initialization can fail if:
    - `R`/`Rscript` are not on PATH
    - `R_HOME` is not set and auto-detection fails
    - `TMPDIR` is not writable (rpy2 uses tempfiles during init)
    - shared libs are not found unless conda's `lib/` is on `LD_LIBRARY_PATH`
    """
    import os
    import sys
    from pathlib import Path
    import shutil

    # 1) Prefer a writable TMPDIR (rpy2 creates temp files during init).
    if not os.environ.get("TMPDIR"):
        candidates = []
        try:
            candidates.append(Path.home() / "tmp")
            candidates.append(Path.home() / ".cache" / "grading_calibration" / "tmp")
        except Exception:
            pass
        try:
            candidates.append(Path.cwd() / ".tmp")
        except Exception:
            pass
        for d in candidates:
            try:
                d.mkdir(parents=True, exist_ok=True)
                test_path = d / ".write_test"
                test_path.write_text("ok")
                test_path.unlink(missing_ok=True)
                os.environ["TMPDIR"] = str(d)
                break
            except Exception:
                continue

    # 2) If running under conda, prepend the env `bin/` to PATH and ensure R_HOME.
    try:
        prefix = Path(sys.executable).resolve().parents[1]
        bin_dir = prefix / "bin"
        r_home = prefix / "lib" / "R"
        if bin_dir.exists():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH','')}"
        if r_home.exists() and not os.environ.get("R_HOME"):
            os.environ["R_HOME"] = str(r_home)
        # Help R find its shared libs in some environments.
        lib_dir = prefix / "lib"
        if lib_dir.exists():
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if str(lib_dir) not in ld.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{ld}" if ld else str(lib_dir)
    except Exception:
        pass

    # 3) Final fallback: if R is not on PATH but conda-style R exists near python,
    # allow rpy2 to discover it via R_HOME; otherwise do nothing.
    if shutil.which("R") is None:
        # If R_HOME was set above, rpy2 can still work.
        return


def _try_import_lme4():
    """
    Best-effort import for the optional R/lme4 backend via rpy2.

    Returns:
        (ok, payload)
        - ok=True: payload is a dict with rpy2 objects and imported packages.
        - ok=False: payload is a string describing the import error.
    """
    try:
        _configure_rpy2_environment()
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as rconversion
        from rpy2.robjects import pandas2ri
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        lme4 = importr("lme4")
        stats_r = importr("stats")

        converter = default_converter + pandas2ri.converter

        return True, {
            "ro": ro,
            "rconversion": rconversion,
            "localconverter": localconverter,
            "converter": converter,
            "pandas2ri": pandas2ri,
            "lme4": lme4,
            "stats": stats_r,
        }
    except Exception as e:
        return False, str(e)


def _normalize_lme4_optimizer(optimizer: str) -> str:
    """
    Map statsmodels-style optimizer names to a safe lme4 optimizer.
    """
    if optimizer is None:
        return "bobyqa"
    opt = str(optimizer).lower()
    if opt in {"powell", "lbfgs", "bfgs", "cg", "nm", "nelder-mead", "nelder_mead"}:
        return "bobyqa"
    return optimizer


def _fit_lme4_nonlinear_inprocess(
    *,
    data: pd.DataFrame,
    formula_lin: str,
    formula_nl: str,
    include_question_fe: bool,
    include_student_vc_glmm: bool,
    include_rubric_re: bool,
    is_glmm: bool,
    glmer_nAGQ: int,
    optimizer: str,
    maxfun: int,
) -> Dict[str, Any]:
    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    df_r = data.copy()
    for col in df_r.columns:
        if pd.api.types.is_categorical_dtype(df_r[col]) or df_r[col].dtype == object:
            df_r[col] = df_r[col].astype(str)

    with localconverter(converter):
        ro.globalenv["df_nl_tmp"] = rconversion.py2rpy(df_r)

    if "gc_fit_nonlinear_lme4" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_fit_nonlinear_lme4 <- function(df, formula_lin, formula_full, is_glmm, glmer_nAGQ, optimizer, maxfun) {
                  suppressPackageStartupMessages(library(lme4))
                  suppressPackageStartupMessages(library(stats))
                  suppressPackageStartupMessages(library(splines))
                  suppressPackageStartupMessages(library(splines))

                  df$student_id <- factor(df$student_id)
                  df$question_id <- factor(df$question_id)
                  df$rubric_id <- factor(df$rubric_id)
                  df$answer_id <- factor(df$answer_id)
                  if ("feature_bin" %in% names(df)) df$feature_bin <- factor(df$feature_bin)

                  quiet_eval <- function(expr) {
                    suppressWarnings(withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    ))
                  }

                  if (is_glmm) {
                    ctrl <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))
                    fit_lin <- quiet_eval(glmer(as.formula(formula_lin), data=df, family=binomial(), nAGQ=glmer_nAGQ, control=ctrl))
                    fit_full <- quiet_eval(glmer(as.formula(formula_full), data=df, family=binomial(), nAGQ=glmer_nAGQ, control=ctrl))
                  } else {
                    ctrl <- lmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))
                    fit_lin <- quiet_eval(lmer(as.formula(formula_lin), data=df, REML=FALSE, control=ctrl))
                    fit_full <- quiet_eval(lmer(as.formula(formula_full), data=df, REML=FALSE, control=ctrl))
                  }

                  a <- quiet_eval(tryCatch(anova(fit_lin, fit_full, test="Chisq"), error=function(e) NULL))
                  lrt_stat <- NA_real_
                  lrt_df <- NA_real_
                  lrt_p <- NA_real_
                  if (!is.null(a)) {
                    if ("Chisq" %in% colnames(a)) lrt_stat <- a[nrow(a), "Chisq"]
                    if ("Df" %in% colnames(a)) lrt_df <- a[nrow(a), "Df"]
                    if ("Pr(>Chisq)" %in% colnames(a)) lrt_p <- a[nrow(a), "Pr(>Chisq)"]
                  }

                  singular <- FALSE
                  singular <- tryCatch(isSingular(fit_full, tol=1e-4), error=function(e) FALSE)

                  msgs <- character(0)
                  msgs <- tryCatch({
                    oi <- fit_full@optinfo
                    conv <- oi$conv$lme4$messages
                    if (!is.null(conv)) conv else character(0)
                  }, error=function(e) character(0))

                  list(
                    fit_lin=fit_lin,
                    fit_full=fit_full,
                    aic_lin=as.numeric(AIC(fit_lin)),
                    bic_lin=as.numeric(BIC(fit_lin)),
                    aic_full=as.numeric(AIC(fit_full)),
                    bic_full=as.numeric(BIC(fit_full)),
                    lrt_stat=lrt_stat,
                    lrt_df=lrt_df,
                    lrt_p=lrt_p,
                    singular=singular,
                    conv_messages=msgs
                  )
                }
                """
            )
        )

    res = ro.globalenv["gc_fit_nonlinear_lme4"](
        ro.globalenv["df_nl_tmp"],
        formula_lin,
        formula_nl,
        bool(is_glmm),
        int(glmer_nAGQ),
        str(_normalize_lme4_optimizer(optimizer)),
        int(maxfun),
    )

    warnings_list: List[str] = []
    try:
        msgs = list(res.rx2("conv_messages"))
        for m in msgs:
            ms = str(m)
            if ms:
                warnings_list.append(ms)
    except Exception:
        pass

    def _get_float(key: str) -> float:
        try:
            v = res.rx2(key)
            if v is None or not len(v):
                return float("nan")
            val = float(v[0])
            # R NA integers can come through as INT_MIN-style sentinels.
            if is_na_sentinel(val):
                return float("nan")
            return val
        except Exception:
            return float("nan")

    singular = False
    try:
        singular = bool(res.rx2("singular")[0])
    except Exception:
        singular = False

    lrt_stat = _get_float("lrt_stat")
    lrt_df = _get_float("lrt_df")
    lrt_p = _get_float("lrt_p")

    if np.isfinite(lrt_df) and lrt_df <= 0:
        lrt_df = float("nan")
        lrt_p = float("nan")
    if np.isfinite(lrt_stat) and lrt_stat < 0:
        lrt_stat = float("nan")
    if np.isfinite(lrt_p) and (lrt_p < 0 or lrt_p > 1):
        lrt_p = float("nan")

    converged = bool(np.isfinite(lrt_p))

    return {
        "fit_lin": res.rx2("fit_lin"),
        "fit_full": res.rx2("fit_full"),
        "aic_lin": _get_float("aic_lin"),
        "bic_lin": _get_float("bic_lin"),
        "aic_full": _get_float("aic_full"),
        "bic_full": _get_float("bic_full"),
        "lrt_stat": lrt_stat,
        "lrt_df": lrt_df,
        "lrt_p": lrt_p,
        "singular": singular,
        "converged": converged,
        "warnings": warnings_list,
    }


def _predict_lme4_fixed(fit: Any, newdata: pd.DataFrame, is_glmm: bool) -> np.ndarray:
    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    df_r = newdata.copy()
    for col in df_r.columns:
        if pd.api.types.is_categorical_dtype(df_r[col]) or df_r[col].dtype == object:
            df_r[col] = df_r[col].astype(str)

    with localconverter(converter):
        ro.globalenv["df_pred_tmp"] = rconversion.py2rpy(df_r)
    ro.globalenv["fit_pred_tmp"] = fit

    if "gc_predict_lme4_fixed" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_predict_lme4_fixed <- function(fit, newdata, is_glmm) {
                  suppressPackageStartupMessages(library(stats))
                  frame <- tryCatch(fit@frame, error=function(e) NULL)
                  if (!is.null(frame)) {
                    for (nm in names(frame)) {
                      if (nm %in% names(newdata) && is.factor(frame[[nm]])) {
                        newdata[[nm]] <- factor(newdata[[nm]], levels=levels(frame[[nm]]))
                      }
                    }
                  }
                  if (is_glmm) {
                    return(as.numeric(predict(fit, newdata=newdata, re.form=NA, type="response")))
                  }
                  return(as.numeric(predict(fit, newdata=newdata, re.form=NA)))
                }
                """
            )
        )

    pred = ro.globalenv["gc_predict_lme4_fixed"](ro.globalenv["fit_pred_tmp"], ro.globalenv["df_pred_tmp"], bool(is_glmm))
    return np.asarray(pred, dtype=float)


def _try_import_artool():
    """
    Best-effort import for the optional R/ARTool backend via rpy2.

    Returns:
        (ok, payload)
        - ok=True: payload is a dict with rpy2 objects and imported packages.
        - ok=False: payload is a string describing the import error.
    """
    try:
        _configure_rpy2_environment()
        import rpy2.robjects as ro
        import rpy2.robjects.conversion as rconversion
        from rpy2.robjects import pandas2ri
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        artool = importr("ARTool")
        lme4 = importr("lme4")
        stats_r = importr("stats")

        converter = default_converter + pandas2ri.converter

        return True, {
            "ro": ro,
            "rconversion": rconversion,
            "localconverter": localconverter,
            "converter": converter,
            "pandas2ri": pandas2ri,
            "artool": artool,
            "lme4": lme4,
            "stats": stats_r,
        }
    except Exception as e:
        return False, str(e)


def art_c_interaction_test(
    values: np.ndarray,
    model_labels: np.ndarray,
    method_labels: np.ndarray,
    student_ids: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    ART-C interaction test using R ARTool with a random intercept for student_id.

    Returns:
        dict with p_interaction, n, method, warnings (list)
    """
    warnings_list: List[str] = []
    values = np.asarray(values)
    if student_ids is None:
        return {
            "p_interaction": np.nan,
            "n": int(len(values)),
            "method": "artc",
            "warnings": ["missing student_ids"],
        }
    model_labels = np.asarray(model_labels)
    method_labels = np.asarray(method_labels)
    student_ids = np.asarray(student_ids)
    if not (len(values) == len(model_labels) == len(method_labels) == len(student_ids)):
        return {
            "p_interaction": np.nan,
            "n": int(len(values)),
            "method": "artc",
            "warnings": ["length_mismatch_inputs"],
        }

    df = pd.DataFrame(
        {
            "y": values,
            "model": model_labels,
            "method": method_labels,
            "student": student_ids,
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "model", "method", "student"])
    if len(df) < 10:
        return {
            "p_interaction": np.nan,
            "n": int(len(df)),
            "method": "artc",
            "warnings": ["insufficient_rows"],
        }

    ok, payload = _try_import_artool()
    if not ok:
        raise ImportError(f"rpy2/ARTool not available: {payload}")

    ro = payload["ro"]
    converter = payload["converter"]
    localconverter = payload["localconverter"]
    rconversion = payload["rconversion"]

    if "gc_artc_interaction" not in list(ro.globalenv.keys()):
        ro.r(
            """
            gc_artc_interaction <- function(df) {
              suppressPackageStartupMessages(library(ARTool))
              suppressPackageStartupMessages(library(lme4))
              df$model <- as.factor(df$model)
              df$method <- as.factor(df$method)
              df$student <- as.factor(df$student)
              res <- tryCatch({
                fit <- art(y ~ model * method + (1|student), data=df)
                aov <- anova(fit)
                p <- NA_real_
                if ("model:method" %in% rownames(aov)) {
                  p <- aov["model:method", "Pr(>F)"]
                }
                list(p_interaction=p)
              }, error=function(e) {
                list(p_interaction=NA_real_, error=as.character(e))
              })
              res
            }
            """
        )

    with localconverter(converter):
        r_df = ro.conversion.py2rpy(df)

    res = ro.globalenv["gc_artc_interaction"](r_df)
    p_val = np.nan
    try:
        p_val = float(res.rx2("p_interaction")[0])
    except Exception:
        warnings_list.append("failed_to_extract_p_value")
    try:
        if "error" in list(res.names):
            err = str(res.rx2("error")[0])
            if err and err != "NULL":
                warnings_list.append(f"artc_error: {err}")
    except Exception:
        pass

    return {
        "p_interaction": p_val,
        "n": int(len(df)),
        "method": "artc",
        "warnings": warnings_list,
    }


def _lme4_conv_messages(_ro, fit) -> List[str]:
    """
    Best-effort extraction of lme4 convergence messages.
    """
    msgs: List[str] = []
    try:
        optinfo = fit.do_slot("optinfo")
        conv = optinfo.rx2("conv")
        lme4_conv = conv.rx2("lme4")
        if lme4_conv is not None:
            m = lme4_conv.rx2("messages")
            if m is not None:
                # m can be NULL, a character vector, or a list.
                try:
                    for s in list(m):
                        s2 = str(s)
                        if s2 and s2 != "NULL":
                            msgs.append(s2)
                except Exception:
                    s2 = str(m)
                    if s2 and s2 != "NULL":
                        msgs.append(s2)
    except Exception:
        pass
    return msgs


def fit_lme4_lrt_from_formula(
    df: pd.DataFrame,
    *,
    formula_null: str,
    formula_full: str,
    is_glmm: bool,
    family: Literal["binomial", "gaussian"] = "gaussian",
    coef_name: Optional[str] = None,
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
    lme4_trace: Optional[bool] = None,
    lme4_trace_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fit two nested lme4 models and compute an LRT p-value.

    This is a small utility intended for Step-4-style group tests where the
    model spec is naturally expressed as formulas (e.g., rubric fixed effects
    within a question, student random intercept).

    Args:
        df: Analytic data frame passed to R. Any categorical columns should be
            string-like (we coerce object/category to strings).
        formula_null: R formula string for the null model.
        formula_full: R formula string for the full model.
        is_glmm: If True, use `lme4::glmer`; else `lme4::lmer` (ML).
        family: GLMM family (only used when is_glmm=True).
        coef_name: If provided, extract (beta, se) for this coefficient from the full model.
        glmer_nAGQ: glmer quadrature points (1 = Laplace, recommended for speed).
        optimizer/maxfun: glmer optimizer settings for stability.

    Returns:
        Dict with keys:
          - p_value_lrt, chisq, df_diff
          - beta, se (NaN if coef_name is None or extraction fails)
          - full_deviance, full_df_resid, overdispersion_ratio (GLMM diagnostics)
          - converged (best-effort), singular (isSingular)
          - warnings: list[str] (convergence messages + extraction issues)
    """
    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    df_r = df.copy()
    # rpy2 categorical conversion is most reliable when categories are strings.
    for col in df_r.columns:
        if pd.api.types.is_categorical_dtype(df_r[col]) or df_r[col].dtype == object:
            df_r[col] = df_r[col].astype(str)

    with localconverter(converter):
        ro.globalenv["df_lrt_tmp"] = rconversion.py2rpy(df_r)

    lme4_trace_enabled, lme4_trace_path = _lme4_trace_settings(lme4_trace, lme4_trace_file)
    if lme4_trace_enabled and lme4_trace_path:
        try:
            Path(lme4_trace_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # Define an R helper once per session to (a) silence warnings/messages and
    # (b) return a simple, JSON-like list structure back to Python.
    if "gc_lme4_lrt_fit" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_lme4_lrt_fit <- function(df, formula_null, formula_full,
                                           is_glmm, family_name,
                                           glmer_nAGQ, optimizer, maxfun,
                                           coef_name, trace, trace_file) {
                  suppressPackageStartupMessages(library(lme4))
                  suppressPackageStartupMessages(library(stats))
                  suppressPackageStartupMessages(library(splines))

                  quiet_eval <- function(expr) {
                    withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    )
                  }

                  trace_val <- if (isTRUE(trace)) 1 else 0
                  if (isTRUE(trace) && nzchar(trace_file)) {
                    sink(trace_file, append=TRUE, type="output")
                    sink(trace_file, append=TRUE, type="message")
                    on.exit({
                      sink(type="message")
                      sink(type="output")
                    }, add=TRUE)
                  }

                  if (is_glmm) {
                    fam <- if (family_name == "binomial") binomial(link="logit") else gaussian()
                    ctrl <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun, trace=trace_val))
                    fit_null <- quiet_eval(glmer(as.formula(formula_null), data=df, family=fam, nAGQ=glmer_nAGQ, control=ctrl))
                    fit_full <- quiet_eval(glmer(as.formula(formula_full), data=df, family=fam, nAGQ=glmer_nAGQ, control=ctrl))
                  } else {
                    ctrl <- lmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun, trace=trace_val))
                    fit_null <- quiet_eval(lmer(as.formula(formula_null), data=df, REML=FALSE, control=ctrl))
                    fit_full <- quiet_eval(lmer(as.formula(formula_full), data=df, REML=FALSE, control=ctrl))
                  }

                  # LRT
                  p_value <- NA_real_
                  chisq <- NA_real_
                  df_diff <- NA_real_
                  a <- if (is_glmm) {
                    quiet_eval(tryCatch(anova(fit_null, fit_full, test="Chisq"), error=function(e) NULL))
                  } else {
                    quiet_eval(tryCatch(anova(fit_null, fit_full), error=function(e) NULL))
                  }
                  if (!is.null(a)) {
                    # lme4 returns Pr(>Chisq) and Chisq in the last row of comparison.
                    if ("Pr(>Chisq)" %in% colnames(a)) p_value <- a[nrow(a), "Pr(>Chisq)"]
                    if ("Chisq" %in% colnames(a)) chisq <- a[nrow(a), "Chisq"]
                    if ("Df" %in% colnames(a)) df_diff <- a[nrow(a), "Df"] - a[nrow(a)-1, "Df"]
                  }

                  # singular + convergence messages
                  singular <- FALSE
                  singular <- tryCatch(isSingular(fit_full, tol=1e-4), error=function(e) FALSE)

                  msgs <- character(0)
                  msgs <- tryCatch({
                    oi <- fit_full@optinfo
                    conv <- oi$conv$lme4$messages
                    if (!is.null(conv)) conv else character(0)
                  }, error=function(e) character(0))

                  full_dev <- tryCatch(deviance(fit_full), error=function(e) NA_real_)
                  full_df_resid <- tryCatch(df.residual(fit_full), error=function(e) NA_real_)
                  overdispersion_ratio <- NA_real_
                  if (is.finite(full_dev) && is.finite(full_df_resid) && full_df_resid > 0) {
                    overdispersion_ratio <- full_dev / full_df_resid
                  }

                  # beta/se extraction (optional)
                  beta <- NA_real_
                  se <- NA_real_
                  if (!is.null(coef_name) && nzchar(coef_name)) {
                    cf <- tryCatch(summary(fit_full)$coefficients, error=function(e) NULL)
                    if (!is.null(cf) && coef_name %in% rownames(cf)) {
                      beta <- cf[coef_name, "Estimate"]
                      se <- cf[coef_name, "Std. Error"]
                    }
                  }

                  list(
                    p_value_lrt=p_value,
                    chisq=chisq,
                    df_diff=df_diff,
                    beta=beta,
                    se=se,
                    singular=singular,
                    conv_messages=msgs,
                    full_deviance=full_dev,
                    full_df_resid=full_df_resid,
                    overdispersion_ratio=overdispersion_ratio
                  )
                }
                """
            )
        )

    coef_name_r = coef_name if coef_name is not None else ""
    res = ro.globalenv["gc_lme4_lrt_fit"](
        ro.globalenv["df_lrt_tmp"],
        formula_null,
        formula_full,
        bool(is_glmm),
        str(family),
        int(glmer_nAGQ),
        str(optimizer),
        int(maxfun),
        str(coef_name_r),
        bool(lme4_trace_enabled),
        str(lme4_trace_path),
    )

    def _get_float(key: str) -> float:
        try:
            v = res.rx2(key)
            return float(v[0]) if v is not None and len(v) else float("nan")
        except Exception:
            return float("nan")

    warnings_list: List[str] = []
    try:
        msgs = list(res.rx2("conv_messages"))
        for m in msgs:
            ms = str(m)
            if ms:
                warnings_list.append(ms)
    except Exception:
        pass

    singular = False
    try:
        singular = bool(res.rx2("singular")[0])
    except Exception:
        singular = False

    # We don't have a perfect converged flag from lme4, but missing LRT p-values
    # are a strong indicator of fit failure for our purposes.
    p_value = _get_float("p_value_lrt")
    converged = bool(np.isfinite(p_value))

    out = {
        "p_value_lrt": p_value,
        "chisq": _get_float("chisq"),
        "df_diff": _get_float("df_diff"),
        "beta": _get_float("beta"),
        "se": _get_float("se"),
        "full_deviance": _get_float("full_deviance"),
        "full_df_resid": _get_float("full_df_resid"),
        "overdispersion_ratio": _get_float("overdispersion_ratio"),
        "converged": converged,
        "singular": singular,
        "backend": "lme4",
        "warnings": warnings_list,
    }
    return out


def fit_lme4_glmer_fixef_vcov(
    df: pd.DataFrame,
    *,
    formula_full: str,
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
    lme4_trace: Optional[bool] = None,
    lme4_trace_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fit an lme4::glmer model and return fixed effects + vcov.

    Intended for Step-4 calibration regression effect sizes where we need
    rubric-specific slopes from interaction terms.
    """
    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    df_r = df.copy()
    # rpy2 categorical conversion is most reliable when categories are strings.
    for col in df_r.columns:
        if pd.api.types.is_categorical_dtype(df_r[col]) or df_r[col].dtype == object:
            df_r[col] = df_r[col].astype(str)

    with localconverter(converter):
        ro.globalenv["df_glmer_fixef_tmp"] = rconversion.py2rpy(df_r)

    lme4_trace_enabled, lme4_trace_path = _lme4_trace_settings(lme4_trace, lme4_trace_file)
    if lme4_trace_enabled and lme4_trace_path:
        try:
            Path(lme4_trace_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    if "gc_lme4_glmer_fixef" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_lme4_glmer_fixef <- function(df, formula_full,
                                                glmer_nAGQ, optimizer, maxfun,
                                                trace, trace_file) {
                  suppressPackageStartupMessages(library(lme4))
                  suppressPackageStartupMessages(library(stats))

                  quiet_eval <- function(expr) {
                    withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    )
                  }

                  trace_val <- if (isTRUE(trace)) 1 else 0
                  if (isTRUE(trace) && nzchar(trace_file)) {
                    sink(trace_file, append=TRUE, type="output")
                    sink(trace_file, append=TRUE, type="message")
                    on.exit({
                      sink(type="message")
                      sink(type="output")
                    }, add=TRUE)
                  }

                  ctrl <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun, trace=trace_val))
                  fit_full <- quiet_eval(glmer(as.formula(formula_full), data=df,
                                               family=binomial(link="logit"),
                                               nAGQ=glmer_nAGQ, control=ctrl))

                  singular <- FALSE
                  singular <- tryCatch(isSingular(fit_full, tol=1e-4), error=function(e) FALSE)

                  msgs <- character(0)
                  msgs <- tryCatch({
                    oi <- fit_full@optinfo
                    conv <- oi$conv$lme4$messages
                    if (!is.null(conv)) conv else character(0)
                  }, error=function(e) character(0))

                  fe <- fixef(fit_full)
                  vc <- as.matrix(vcov(fit_full))
                  list(
                    coef_names=names(fe),
                    coef_values=unname(fe),
                    vcov=vc,
                    singular=singular,
                    conv_messages=msgs
                  )
                }
                """
            )
        )

    res = ro.globalenv["gc_lme4_glmer_fixef"](
        ro.globalenv["df_glmer_fixef_tmp"],
        formula_full,
        int(glmer_nAGQ),
        str(optimizer),
        int(maxfun),
        bool(lme4_trace_enabled),
        str(lme4_trace_path),
    )

    warnings_list: List[str] = []
    try:
        msgs = list(res.rx2("conv_messages"))
        for m in msgs:
            ms = str(m)
            if ms:
                warnings_list.append(ms)
    except Exception:
        pass

    singular = False
    try:
        singular = bool(res.rx2("singular")[0])
    except Exception:
        singular = False

    coef_names = []
    coef_values = np.array([], dtype=float)
    vcov = np.array([[]], dtype=float)
    try:
        coef_names = list(res.rx2("coef_names"))
        coef_values = np.asarray(res.rx2("coef_values"), dtype=float)
        vcov = np.asarray(res.rx2("vcov"), dtype=float)
    except Exception as e:
        warnings_list.append(f"Failed to extract fixef/vcov: {e}")

    converged = bool(np.isfinite(coef_values).all()) if len(coef_values) else False

    return {
        "coef_names": coef_names,
        "coef_values": coef_values,
        "vcov": vcov,
        "converged": converged,
        "singular": singular,
        "backend": "lme4",
        "warnings": warnings_list,
    }


def _fit_lmm_continuous_r_inprocess(
    data: pd.DataFrame,
    include_question_fe: bool,
    include_question_re: bool,
    alpha: float = 0.05,
    include_answer_re: bool = True,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> FeatureEffectResult:
    """
    Backend: lme4::lmer (via rpy2) with LRT p-value for feature_z.

    p-value definition:
      Likelihood ratio test comparing full vs null model (drop feature_z),
      both fit with REML=FALSE (ML) for valid LRT comparison.
    """
    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]
    lme4 = payload["lme4"]
    stats_r = payload["stats"]

    # Keep only columns needed for the R fit to avoid pandas2ri conversion surprises
    # from unused columns (e.g., object-typed "feature"/"outcome").
    needed_cols = ["outcome_z", "feature_z", "student_id", "question_id", "answer_id", "rubric_id"]
    df = data[[c for c in needed_cols if c in data.columns]].copy()
    df["outcome_z"] = pd.to_numeric(df["outcome_z"], errors="coerce")
    df["feature_z"] = pd.to_numeric(df["feature_z"], errors="coerce")

    # Ensure group columns are treated as categorical/factors on the R side.
    for c in ["student_id", "question_id", "rubric_id", "answer_id"]:
        if c in df.columns:
            # rpy2 can only convert pandas Categorical -> R factor when categories are strings.
            df[c] = pd.Categorical(df[c].astype(str))

    with localconverter(converter):
        ro.globalenv["df_step3"] = rconversion.py2rpy(df)

    fe = "feature_z"
    q_fe = " + factor(question_id)" if include_question_fe else ""
    re = " + (1|student_id)"
    if include_question_re:
        re += " + (1|question_id)"
    if include_answer_re:
        re += " + (1|answer_id)"
    if int(df["rubric_id"].nunique()) >= 2:
        re += " + (1|rubric_id)"

    full_formula = ro.Formula(f"outcome_z ~ {fe}{q_fe}{re}")
    null_formula = ro.Formula(f"outcome_z ~ 1{q_fe}{re}")

    warnings_list: List[str] = []
    converged = True

    # Fit models with ML (REML=FALSE) for LRT.
    lme4_trace_enabled, lme4_trace_path = _lme4_trace_settings()
    if lme4_trace_enabled and lme4_trace_path:
        try:
            Path(lme4_trace_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    opt_lme4 = _normalize_lme4_optimizer(optimizer)
    maxfun = int(maxfun) if maxfun is not None else 100000
    if maxfun < 1:
        maxfun = 1
    trace_val = int(bool(lme4_trace_enabled))
    ctrl = ro.r(
        f"lme4::lmerControl(optimizer='{opt_lme4}', optCtrl=list(maxfun={maxfun}, trace={trace_val}))"
    )
    lmer_fn = ro.r("lme4::lmer")
    trace_open = False
    if lme4_trace_enabled and lme4_trace_path:
        try:
            with localconverter(converter):
                ro.globalenv["gc_trace_file"] = rconversion.py2rpy(lme4_trace_path)
            ro.r("gc_trace_con <- file(gc_trace_file, open='a')")
            ro.r("sink(gc_trace_con, type='output'); sink(gc_trace_con, type='message')")
            trace_open = True
        except Exception:
            trace_open = False
    try:
        fit_full = lmer_fn(full_formula, data=ro.globalenv["df_step3"], REML=False, control=ctrl)
        fit_null = lmer_fn(null_formula, data=ro.globalenv["df_step3"], REML=False, control=ctrl)
    finally:
        if trace_open:
            try:
                ro.r("sink(type='message'); sink(type='output'); close(gc_trace_con)")
            except Exception:
                pass

    # Convergence/singularity diagnostics.
    try:
        is_sing = bool(ro.r("lme4::isSingular")(fit_full, tol=1e-4)[0])
        if is_sing:
            warnings_list.append("lme4 lmer singular fit (variance component ~0).")
    except Exception:
        pass
    warnings_list.extend(_lme4_conv_messages(ro, fit_full))

    # LRT p-value.
    p_value = np.nan
    try:
        # Use positional arguments - R's anova() doesn't accept keyword args for model objects
        a = stats_r.anova(fit_null, fit_full)
        with localconverter(converter):
            a_df = rconversion.rpy2py(a)
        # lme4 returns Pr(>Chisq) for model comparison.
        if "Pr(>Chisq)" in a_df.columns:
            p_value = float(a_df["Pr(>Chisq)"].iloc[-1])
    except Exception as e:
        warnings_list.append(f"lme4 LRT failed: {e}")
        converged = False

    # Extract beta and SE directly in R to avoid rpy2 conversion issues with vcov matrix
    beta = np.nan
    se = np.nan
    try:
        # Get beta from fixef()
        fixef_result = ro.r("lme4::fixef")(fit_full)
        fixef_names = list(fixef_result.names)
        fixef_values = list(fixef_result)
        
        # Find the feature_z coefficient - R may name it slightly differently
        beta_idx = None
        coef_name_used = None
        for i, name in enumerate(fixef_names):
            if name == "feature_z" or "feature_z" in name:
                beta_idx = i
                coef_name_used = name
                break
        
        if beta_idx is not None:
            beta = float(fixef_values[beta_idx])
        else:
            # Fallback: assume it's the second coefficient (after intercept)
            if len(fixef_values) >= 2:
                beta = float(fixef_values[1])
                coef_name_used = fixef_names[1] if len(fixef_names) >= 2 else "index_1"
                warnings_list.append(f"feature_z not found in fixef names {fixef_names}; using index 1")
            else:
                warnings_list.append(f"Could not find feature_z in fixef: {fixef_names}")
        
        # Get SE from summary(fit)$coefficients - compute in R to avoid numpy conversion issues
        if coef_name_used is not None:
            try:
                # summary(fit)$coefficients is a matrix with rownames=coefficient names, colnames=Estimate/Std. Error/t value
                summary_coef = ro.r("summary")(fit_full).rx2("coefficients")
                # Try to get SE by coefficient name
                se_r = summary_coef.rx(coef_name_used, "Std. Error")
                if se_r is not None and len(se_r) > 0:
                    se = float(se_r[0])
            except Exception as e_summary:
                # Fallback: try computing sqrt(vcov[name, name]) directly in R
                try:
                    ro.globalenv["_fit_tmp"] = fit_full
                    se_r = ro.r(f'sqrt(vcov(_fit_tmp)["{coef_name_used}", "{coef_name_used}"])')
                    if se_r is not None and len(se_r) > 0:
                        se = float(se_r[0])
                except Exception as e_vcov:
                    warnings_list.append(f"Failed to extract SE via summary ({e_summary}) and vcov ({e_vcov})")
    except Exception as e:
        warnings_list.append(f"Failed to extract lme4 beta/se: {e}")
        converged = False

    z_stat = beta / se if np.isfinite(beta) and np.isfinite(se) and se > 0 else np.nan
    # p_value is LRT; keep z only for reporting convenience.
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = beta - z_crit * se if np.isfinite(se) else np.nan
    ci_high = beta + z_crit * se if np.isfinite(se) else np.nan

    fixed_effects = ["feature_z"] + (["C(question_id)"] if include_question_fe else [])
    random_effects = ["(1|student_id)"]
    if include_question_re:
        random_effects.append("(1|question_id)")
    if include_answer_re:
        random_effects.append("(1|answer_id)")
    if int(df["rubric_id"].nunique()) >= 2:
        random_effects.append("(1|rubric_id)")

    return FeatureEffectResult(
        beta=float(beta) if np.isfinite(beta) else np.nan,
        se=float(se) if np.isfinite(se) else np.nan,
        z=float(z_stat) if np.isfinite(z_stat) else np.nan,
        p_value=float(p_value) if np.isfinite(p_value) else np.nan,
        ci_low=float(ci_low) if np.isfinite(ci_low) else np.nan,
        ci_high=float(ci_high) if np.isfinite(ci_high) else np.nan,
        method="lme4_lmer_LRT",
        inference_type="frequentist_lrt",
        fixed_effects=fixed_effects,
        random_effects=random_effects,
        converged=converged,
        warnings=warnings_list,
    )


def _run_lme4_subprocess_fit(
    *,
    statistics_py_path: str,
    fit_kind: str,
    data: pd.DataFrame,
    include_question_fe: bool,
    include_question_re: bool,
    alpha: float,
    include_student_vc: bool = False,
    include_answer_re: bool = True,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> FeatureEffectResult:
    import dataclasses
    import errno
    import os
    import pickle
    import subprocess
    import sys
    from typing import Any, Dict

    payload = {
        "fit_kind": fit_kind,
        "include_question_fe": include_question_fe,
        "include_question_re": include_question_re,
        "alpha": alpha,
        "include_student_vc": include_student_vc,
        "include_answer_re": include_answer_re,
        "data": data,
        "optimizer": optimizer,
        "maxfun": maxfun,
    }
    in_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    
    worker_code = r"""
import dataclasses
import importlib.util
import os
import pickle
import sys

def _load_statistics_module(path: str):
    spec = importlib.util.spec_from_file_location("_gc_statistics_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    stats_path = os.environ["GC_STATISTICS_PY"]
    payload = pickle.loads(sys.stdin.buffer.read())
    mod = _load_statistics_module(stats_path)

    fit_kind = payload["fit_kind"]
    data = payload["data"]
    include_question_fe = payload["include_question_fe"]
    include_question_re = payload["include_question_re"]
    alpha = payload["alpha"]
    include_student_vc = payload.get("include_student_vc", False)
    include_answer_re = payload.get("include_answer_re", True)
    optimizer = payload.get("optimizer", "bobyqa")
    maxfun = payload.get("maxfun", 100000)

    if fit_kind == "lmm":
        res = mod._fit_lmm_continuous_r_inprocess(
            data,
            include_question_fe,
            include_question_re,
            alpha,
            include_answer_re=include_answer_re,
            optimizer=optimizer,
            maxfun=maxfun,
        )
    elif fit_kind == "glmm":
        # feat_std is used only to scale AME; pass as 1.0 to avoid needing it here.
        res = mod._fit_glmm_binary_r_inprocess(
            data,
            include_question_fe,
            include_question_re,
            1.0,
            alpha,
            include_student_vc=include_student_vc,
            include_answer_re=include_answer_re,
            optimizer=optimizer,
            maxfun=maxfun,
        )
    else:
        raise ValueError(f"Unknown fit_kind: {fit_kind}")

    out = dataclasses.asdict(res)
    sys.stdout.buffer.write(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))

if __name__ == "__main__":
    main()
"""

    env = dict(os.environ)
    env["GC_STATISTICS_PY"] = statistics_py_path

    def _fallback_inprocess(reason: str) -> FeatureEffectResult:
        if fit_kind == "lmm":
            res = _fit_lmm_continuous_r_inprocess(
                data,
                include_question_fe,
                include_question_re,
                alpha,
                include_answer_re=include_answer_re,
                optimizer=optimizer,
                maxfun=maxfun,
            )
        elif fit_kind == "glmm":
            res = _fit_glmm_binary_r_inprocess(
                data,
                include_question_fe,
                include_question_re,
                1.0,
                alpha,
                include_student_vc=include_student_vc,
                include_answer_re=include_answer_re,
                optimizer=optimizer,
                maxfun=maxfun,
            )
        else:
            raise ValueError(f"Unknown fit_kind: {fit_kind}")
        res.warnings.append(
            f"lme4 subprocess failed; ran in-process lme4 instead: {reason}"
        )
        return res
    
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker_code],
            input=in_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
    except OSError as e:
        if e.errno == errno.E2BIG or "Argument list too long" in str(e):
            return _fallback_inprocess(str(e))
        raise

    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        if "Argument list too long" in stderr:
            return _fallback_inprocess(stderr.strip())
        raise RuntimeError(
            f"lme4 subprocess failed (rc={proc.returncode}). stderr:\n{stderr}"
        )
    if not proc.stdout:
        stderr = proc.stderr.decode(errors="replace")
        if "Argument list too long" in stderr:
            return _fallback_inprocess(stderr.strip())
        raise RuntimeError(
            "lme4 subprocess exited with rc=0 but produced no output. "
            f"stderr:\n{stderr}"
        )

    out_dict: Dict[str, Any] = pickle.loads(proc.stdout)
    res = FeatureEffectResult(**out_dict)
    if proc.stderr:
        res.warnings.append(f"lme4 subprocess stderr:\n{proc.stderr.decode(errors='replace')}")
    return res


def _fit_lmm_continuous_r(
    data: pd.DataFrame,
    include_question_fe: bool,
    include_question_re: bool,
    alpha: float = 0.05,
    include_answer_re: bool = True,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> FeatureEffectResult:
    from pathlib import Path

    
    res = _run_lme4_subprocess_fit(
        statistics_py_path=str(Path(__file__).resolve()),
        fit_kind="lmm",
        data=data,
        include_question_fe=include_question_fe,
        include_question_re=include_question_re,
        alpha=alpha,
        include_answer_re=include_answer_re,
        optimizer=optimizer,
        maxfun=maxfun,
    )
    res.method = "lme4_lmer_LRT_subprocess"

    return res


def _fit_glmm_binary(
    data: pd.DataFrame,
    include_question_fe: bool,
    include_question_re: bool,
    feat_std: float,
    alpha: float = 0.05,
    include_student_vc: bool = False,
    include_answer_re: bool = True,
) -> FeatureEffectResult:
    """Backend: BinomialBayesMixedGLM via variational Bayes (posterior approx)."""
    import statsmodels.genmod.bayes_mixed_glm as bmg

    if include_question_fe:
        formula = "outcome_z ~ feature_z + C(question_id)"
        fixed_effects = ["feature_z", "C(question_id)"]
    else:
        formula = "outcome_z ~ feature_z"
        fixed_effects = ["feature_z"]

    vc_formulas: Dict[str, str] = {"rubric": "0 + C(rubric_id)"}
    random_effects = ["(1|rubric_id)"]
    if include_question_re:
        vc_formulas["question"] = "0 + C(question_id)"
        random_effects.insert(0, "(1|question_id)")
    if include_answer_re:
        vc_formulas["answer"] = "0 + C(answer_id)"
        random_effects.insert(0, "(1|answer_id)")
    if include_student_vc:
        vc_formulas["student"] = "0 + C(student_id)"
        random_effects.append("(1|student_id)")

    warnings_list: List[str] = []
    converged = True
   
    try:
        model = bmg.BinomialBayesMixedGLM.from_formula(formula, vc_formulas=vc_formulas, data=data)
        fit = model.fit_vb()
        try:
            feat_idx = fit.model.exog_names.index("feature_z")
        except ValueError:
            feat_idx = [i for i, n in enumerate(fit.model.exog_names) if "feature_z" in n][0]
        beta = float(fit.fe_mean[feat_idx])
        se = float(fit.fe_sd[feat_idx])
    except Exception as e:
        warnings_list.append(f"BinomialBayesMixedGLM failed ({e}); retrying rubric-only VC.")
        converged = False
        model = bmg.BinomialBayesMixedGLM.from_formula(
            formula, vc_formulas={"rubric": "0 + C(rubric_id)"}, data=data
        )
        fit = model.fit_vb()
        feat_idx = fit.model.exog_names.index("feature_z")
        beta = float(fit.fe_mean[feat_idx])
        se = float(fit.fe_sd[feat_idx])

    z_stat = beta / se if np.isfinite(beta) and np.isfinite(se) and se > 0 else np.nan
    p_vb_approx = 2 * (1 - norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = beta - z_crit * se if np.isfinite(se) else np.nan
    ci_high = beta + z_crit * se if np.isfinite(se) else np.nan

    odds_ratio = float(np.exp(beta)) if np.isfinite(beta) else np.nan
    or_ci_low = float(np.exp(ci_low)) if np.isfinite(ci_low) else np.nan
    or_ci_high = float(np.exp(ci_high)) if np.isfinite(ci_high) else np.nan

    # Fixed-effects AME approximation (screening aid; ignores RE uncertainty).
    ame = np.nan
    try:
        linear_pred = fit.model.exog @ fit.fe_mean
        probs = 1 / (1 + np.exp(-linear_pred))
        ame_raw = float(np.mean(beta * probs * (1 - probs)))
        ame = ame_raw / float(feat_std) if np.isfinite(feat_std) and feat_std > 0 else ame_raw
    except Exception:
        pass

    warnings_list.append(
        "VB posterior approximation (screening only). Do not interpret p-values as frequentist."
    )

    return FeatureEffectResult(
        beta=beta,
        se=se,
        z=z_stat,
        p_value=p_vb_approx,
        ci_low=ci_low,
        ci_high=ci_high,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        ame=ame,
        method="BinomialBayesMixedGLM_VB",
        inference_type="vb_posterior_approx",
        fixed_effects=fixed_effects,
        random_effects=random_effects,
        converged=converged,
        warnings=warnings_list,
    )


def _fit_glmm_binary_r_inprocess(
    data: pd.DataFrame,
    include_question_fe: bool,
    include_question_re: bool,
    feat_std: float,
    alpha: float = 0.05,
    include_student_vc: bool = False,
    include_answer_re: bool = True,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> FeatureEffectResult:
    """
    Backend: lme4::glmer (via rpy2) with LRT p-value for feature_z.

    p-value definition:
      Likelihood ratio test comparing full vs null model (drop feature_z),
      using anova(null, full, test="Chisq").
    """
    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]
    lme4 = payload["lme4"]
    stats_r = payload["stats"]

    # Keep only columns needed for the R fit to avoid pandas2ri conversion surprises
    # from unused columns (e.g., object-typed "feature"/"outcome").
    needed_cols = ["outcome_z", "feature_z", "student_id", "question_id", "answer_id", "rubric_id"]
    df = data[[c for c in needed_cols if c in data.columns]].copy()
    df["feature_z"] = pd.to_numeric(df["feature_z"], errors="coerce")

    for c in ["student_id", "question_id", "rubric_id", "answer_id"]:
        if c in df.columns:
            # rpy2 can only convert pandas Categorical -> R factor when categories are strings.
            df[c] = pd.Categorical(df[c].astype(str))
    # Ensure binary outcome is 0/1 integer for R.
    df["outcome_z"] = pd.to_numeric(df["outcome_z"], errors="coerce").astype(int)

    with localconverter(converter):
        ro.globalenv["df_step3"] = rconversion.py2rpy(df)

    q_fe = " + factor(question_id)" if include_question_fe else ""
    re = ""
    if include_question_re:
        re += " + (1|question_id)"
    if include_answer_re:
        re += " + (1|answer_id)"
    re += " + (1|rubric_id)"
    if include_student_vc:
        re += " + (1|student_id)"

    full_formula = ro.Formula(f"outcome_z ~ feature_z{q_fe}{re}")
    null_formula = ro.Formula(f"outcome_z ~ 1{q_fe}{re}")

    warnings_list: List[str] = []
    converged = True

    lme4_trace_enabled, lme4_trace_path = _lme4_trace_settings()
    if lme4_trace_enabled and lme4_trace_path:
        try:
            Path(lme4_trace_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    opt_lme4 = _normalize_lme4_optimizer(optimizer)
    maxfun = int(maxfun) if maxfun is not None else 100000
    if maxfun < 1:
        maxfun = 1
    trace_val = int(bool(lme4_trace_enabled))
    ctrl = ro.r(
        f"lme4::glmerControl(optimizer='{opt_lme4}', optCtrl=list(maxfun={maxfun}, trace={trace_val}))"
    )
    glmer_fn = ro.r("lme4::glmer")
    trace_open = False
    if lme4_trace_enabled and lme4_trace_path:
        try:
            with localconverter(converter):
                ro.globalenv["gc_trace_file"] = rconversion.py2rpy(lme4_trace_path)
            ro.r("gc_trace_con <- file(gc_trace_file, open='a')")
            ro.r("sink(gc_trace_con, type='output'); sink(gc_trace_con, type='message')")
            trace_open = True
        except Exception:
            trace_open = False
    try:
        fit_full = glmer_fn(
            full_formula,
            data=ro.globalenv["df_step3"],
            family=stats_r.binomial(),
            control=ctrl,
        )
        fit_null = glmer_fn(
            null_formula,
            data=ro.globalenv["df_step3"],
            family=stats_r.binomial(),
            control=ctrl,
        )
    finally:
        if trace_open:
            try:
                ro.r("sink(type='message'); sink(type='output'); close(gc_trace_con)")
            except Exception:
                pass

    try:
        is_sing = bool(ro.r("lme4::isSingular")(fit_full, tol=1e-4)[0])
        if is_sing:
            warnings_list.append("lme4 glmer singular fit (variance component ~0).")
    except Exception:
        pass
    warnings_list.extend(_lme4_conv_messages(ro, fit_full))

    p_value = np.nan
    try:
        # Use positional model arguments; R's anova() does not accept keyword args for model objects.
        a = stats_r.anova(fit_null, fit_full, test="Chisq")
        with localconverter(converter):
            a_df = rconversion.rpy2py(a)
        if "Pr(>Chisq)" in a_df.columns:
            p_value = float(a_df["Pr(>Chisq)"].iloc[-1])
    except Exception as e:
        warnings_list.append(f"lme4 LRT failed: {e}")
        converged = False

    beta = np.nan
    se = np.nan
    try:
        fixef_result = ro.r("lme4::fixef")(fit_full)
        fixef_names = list(fixef_result.names)
        fixef_values = list(fixef_result)

        beta_idx = None
        coef_name_used = None
        for i, name in enumerate(fixef_names):
            if name == "feature_z" or "feature_z" in name:
                beta_idx = i
                coef_name_used = name
                break

        if beta_idx is not None:
            beta = float(fixef_values[beta_idx])
        else:
            if len(fixef_values) >= 2:
                beta = float(fixef_values[1])
                coef_name_used = fixef_names[1] if len(fixef_names) >= 2 else "index_1"
                warnings_list.append(f"feature_z not found in fixef names {fixef_names}; using index 1")
            else:
                warnings_list.append(f"Could not find feature_z in fixef: {fixef_names}")

        if coef_name_used is not None:
            try:
                summary_coef = ro.r("summary")(fit_full).rx2("coefficients")
                se_r = summary_coef.rx(coef_name_used, "Std. Error")
                if se_r is not None and len(se_r) > 0:
                    se = float(se_r[0])
            except Exception as e_summary:
                try:
                    ro.globalenv["_fit_tmp"] = fit_full
                    se_r = ro.r(f'sqrt(vcov(_fit_tmp)[\"{coef_name_used}\", \"{coef_name_used}\"])')
                    if se_r is not None and len(se_r) > 0:
                        se = float(se_r[0])
                except Exception as e_vcov:
                    warnings_list.append(f"Failed to extract SE via summary ({e_summary}) and vcov ({e_vcov})")
    except Exception as e:
        warnings_list.append(f"Failed to extract lme4 beta/se: {e}")
        converged = False

    z_stat = beta / se if np.isfinite(beta) and np.isfinite(se) and se > 0 else np.nan
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = beta - z_crit * se if np.isfinite(se) else np.nan
    ci_high = beta + z_crit * se if np.isfinite(se) else np.nan

    odds_ratio = float(np.exp(beta)) if np.isfinite(beta) else np.nan
    or_ci_low = float(np.exp(ci_low)) if np.isfinite(ci_low) else np.nan
    or_ci_high = float(np.exp(ci_high)) if np.isfinite(ci_high) else np.nan

    # Keep AME as a fixed-effects-only approximation for continuity with statsmodels backend.
    ame = np.nan
    try:
        # FE-only linear predictor: use lme4::getME(fit, "X")
        xmat = np.asarray(ro.r("lme4::getME")(fit_full, "X"))
        fe = np.asarray(ro.r("lme4::fixef")(fit_full), dtype=float)
        linear_pred = xmat @ fe
        probs = 1 / (1 + np.exp(-linear_pred))
        ame_raw = float(np.mean(beta * probs * (1 - probs)))
        ame = ame_raw / float(feat_std) if np.isfinite(feat_std) and feat_std > 0 else ame_raw
    except Exception:
        pass

    warnings_list.append("LRT p-value from lme4; Wald CI for beta (screening).")

    fixed_effects = ["feature_z"] + (["C(question_id)"] if include_question_fe else [])
    random_effects = ["(1|rubric_id)"]
    if include_question_re:
        random_effects.insert(0, "(1|question_id)")
    if include_answer_re:
        random_effects.insert(0, "(1|answer_id)")
    if include_student_vc:
        random_effects.append("(1|student_id)")

    return FeatureEffectResult(
        beta=float(beta) if np.isfinite(beta) else np.nan,
        se=float(se) if np.isfinite(se) else np.nan,
        z=float(z_stat) if np.isfinite(z_stat) else np.nan,
        p_value=float(p_value) if np.isfinite(p_value) else np.nan,
        ci_low=float(ci_low) if np.isfinite(ci_low) else np.nan,
        ci_high=float(ci_high) if np.isfinite(ci_high) else np.nan,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        ame=ame,
        method="lme4_glmer_LRT",
        inference_type="frequentist_lrt",
        fixed_effects=fixed_effects,
        random_effects=random_effects,
        converged=converged,
        warnings=warnings_list,
    )


def _fit_glmm_binary_r(
    data: pd.DataFrame,
    include_question_fe: bool,
    include_question_re: bool,
    feat_std: float,
    alpha: float = 0.05,
    include_student_vc: bool = False,
    include_answer_re: bool = True,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
) -> FeatureEffectResult:
    from pathlib import Path

    res = _run_lme4_subprocess_fit(
        statistics_py_path=str(Path(__file__).resolve()),
        fit_kind="glmm",
        data=data,
        include_question_fe=include_question_fe,
        include_question_re=include_question_re,
        alpha=alpha,
        include_student_vc=include_student_vc,
        include_answer_re=include_answer_re,
        optimizer=optimizer,
        maxfun=maxfun,
    )
    res.method = "lme4_glmer_LRT_subprocess"
    if np.isfinite(res.ame) and np.isfinite(feat_std) and feat_std > 0:
        res.ame = float(res.ame) / float(feat_std)
    return res


def _fit_gee_sensitivity(
    data: pd.DataFrame,
    outcome_type: str,
    include_question_fe: bool,
    mode: Literal["simple", "rubric_fe"] = "simple",
) -> Optional[Dict[str, float]]:
    """Sensitivity check via student-clustered GEE (marginal model)."""
    from statsmodels.genmod.families import Gaussian, Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    if mode == "rubric_fe":
        formula = "outcome_z ~ feature_z + C(rubric_id)"
    else:
        if include_question_fe:
            formula = "outcome_z ~ feature_z + C(question_id)"
        else:
            formula = "outcome_z ~ feature_z"

    family = Gaussian() if outcome_type == "continuous" else Binomial()
    try:
        model = sm.GEE.from_formula(
            formula,
            groups="student_id",
            data=data,
            family=family,
            cov_struct=Exchangeable(),
        )
        fit = model.fit()
        return {
            "beta": float(fit.params.get("feature_z", np.nan)),
            "se": float(fit.bse.get("feature_z", np.nan)),
            "p_value": float(fit.pvalues.get("feature_z", np.nan)),
        }
    except Exception:
        return None


def loo_question_sensitivity(
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    outcome_type: Literal["continuous", "binary"] = "continuous",
    include_student_vc_glmm: bool = False,
    transform_bounded: Literal["none", "logit_clip"] = "none",
    alpha: float = 0.05,
    optimizer: str = "powell",
    maxfun: int = 100000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Leave-one-question-out sensitivity for question-constant features (n≈4).

    Guardrail: forces feature_type='question_constant' and include_question_fe=False
    to keep the estimand fixed across folds (no re-detection).
    """
    unique_q = np.unique(question_ids)
    rows: List[Dict[str, Any]] = []

    for q_out in unique_q:
        mask = question_ids != q_out
        try:
            res = estimate_feature_effect(
                feature=feature[mask],
                outcome=outcome[mask],
                student_ids=student_ids[mask],
                question_ids=question_ids[mask],
                rubric_ids=rubric_ids[mask],
                outcome_type=outcome_type,
                run_gee_sensitivity=False,
                transform_bounded=transform_bounded,
                include_student_vc_glmm=include_student_vc_glmm,
                alpha=alpha,
                optimizer=optimizer,
                maxfun=maxfun,
                force_feature_type="question_constant",
                force_include_question_fe=False,
            )
            rows.append(
                {
                    "question_left_out": q_out,
                    "beta": res.beta,
                    "se": res.se,
                    "p_value": res.p_value,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "n_obs": res.n_obs,
                    "converged": res.converged,
                }
            )
        except Exception as e:
            rows.append({"question_left_out": q_out, "error": str(e)})

    df = pd.DataFrame(rows)
    betas = df["beta"].dropna() if "beta" in df.columns else pd.Series([], dtype=float)
    summary: Dict[str, Any] = {"n_successful_fits": int(len(betas))}
    if len(betas) >= 2:
        summary["sign_consistent"] = bool((betas > 0).all() or (betas < 0).all() or (betas == 0).all())
        summary["mean_beta"] = float(betas.mean())
        summary["beta_range"] = (float(betas.min()), float(betas.max()))
        mean_beta = float(betas.mean())
        summary["coefficient_of_variation"] = float(betas.std() / abs(mean_beta)) if mean_beta != 0 else np.inf
    else:
        summary["sign_consistent"] = None
        summary["mean_beta"] = np.nan
        summary["beta_range"] = (np.nan, np.nan)
        summary["coefficient_of_variation"] = np.nan

    return df, summary


def compute_power_curve(
    x: np.ndarray,
    y: np.ndarray,
    r_values: Optional[List[float]] = None,
    n_target: Optional[int] = None,
    alpha: float = 0.05,
    n_power_sim: int = 1000,
    n_perm: int = 1000,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
    power_method: str = "spearman_perm",
    student_ids: Optional[np.ndarray] = None,
    question_ids: Optional[np.ndarray] = None,
    rubric_ids: Optional[np.ndarray] = None,
    data_generator: str = "mixed_model",
) -> pd.DataFrame:
    """
    Compute statistical power for a range of effect sizes.

    This is prospective power analysis, but the "effect size" depends on the
    chosen data generator:
      - data_generator="copula": r_values are interpreted as target Spearman correlations.
      - data_generator="mixed_model": r_values are interpreted as target beta slopes on
        rank-standardized x (comparable to the downstream rank-standardized regression beta).

    More useful than post-hoc power conditional on observed r.

    Args:
        x: Feature values (used to estimate data distribution)
        y: Target values (used to estimate data distribution)
        r_values: List of hypothetical effect sizes
                 (default: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
        n_target: Sample size for power calculation (default: len(x))
        alpha: Significance level
        n_power_sim: Number of Monte Carlo simulations per r value
        n_perm: Number of permutations per permutation test
        random_state: Random number generator
        log_progress: Show progress bars
        power_method: {"spearman_perm", "cluster_robust", "both"}
        student_ids: Required for cluster_robust/both (cluster labels)
        question_ids: Optional item cluster labels for cluster_robust/both
        rubric_ids: Optional rubric cluster labels for cluster_robust/both
        data_generator: {"copula", "mixed_model"} (default: "mixed_model")
                       Use "mixed_model" to match primary model specification.
                       Use "copula" for simpler non-clustered generation.

    Returns:
        DataFrame with columns: r, power, interpretation, method

    Example:
        >>> x, y = load_features()
        >>> power_df = compute_power_curve(x, y, n_target=1000)
        >>> print(power_df)
           r  power interpretation
        0.05   0.12  Very Low
        0.10   0.54  Moderate
        0.15   0.87  High
        ...
    """
    rng = random_state or np.random.default_rng()

    valid_methods = {"spearman_perm", "cluster_robust", "both"}
    if power_method not in valid_methods:
        raise ValueError(f"power_method must be one of {valid_methods}, got {power_method}")
    valid_generators = {"copula", "mixed_model"}
    if data_generator not in valid_generators:
        raise ValueError(f"data_generator must be one of {valid_generators}, got {data_generator}")

    # Clean data
    if power_method in {"cluster_robust", "both"}:
        if student_ids is None:
            raise ValueError("student_ids are required for power_method='cluster_robust' or 'both'")
        data_cols: Dict[str, Any] = {
            'x': x,
            'y': y,
            'student_id': student_ids,
        }
        if question_ids is not None:
            data_cols['question_id'] = question_ids
        if rubric_ids is not None:
            data_cols['rubric_id'] = rubric_ids
        data_df = pd.DataFrame(data_cols).dropna()
        x_clean = data_df['x'].to_numpy()
        y_clean = data_df['y'].to_numpy()
        sid_clean = data_df['student_id'].to_numpy()
        iid_clean = data_df['question_id'].to_numpy() if question_ids is not None else None
        rid_clean = data_df['rubric_id'].to_numpy() if rubric_ids is not None else None
    else:
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        sid_clean = None
        iid_clean = None
        rid_clean = None

    if data_generator == "mixed_model":
        if sid_clean is None or iid_clean is None:
            raise ValueError("student_ids and question_ids are required for data_generator='mixed_model'")
        if len(np.unique(sid_clean)) < 2 or len(np.unique(iid_clean)) < 2:
            raise ValueError("data_generator='mixed_model' requires at least 2 unique student and item clusters")
        if rubric_ids is not None and (rid_clean is None or len(np.unique(rid_clean)) < 2):
            raise ValueError("data_generator='mixed_model' with rubric_ids requires at least 2 unique rubric clusters")
    else:
        # Warn if binary outcome with copula generator
        if len(np.unique(y_clean)) <= 2:
            warnings.warn(
                "Binary outcome detected; copula generator may misrepresent discrete structure. "
                "Consider data_generator='mixed_model' for cluster-aware simulation.",
                stacklevel=2,
            )

    if n_target is None:
        n_target = len(x_clean)

    if r_values is None:
        r_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    results = []

    iterator = r_values
    if log_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(r_values, desc="Power curve", leave=True)
        except ImportError:
            pass

    methods_to_run = []
    if power_method in {"spearman_perm", "both"}:
        methods_to_run.append("spearman_perm")
    if power_method in {"cluster_robust", "both"}:
        methods_to_run.append("cluster_robust")

    for effect_target in iterator:
        # Copula generator: interpret effect_target as r_target (Spearman correlation)
        if data_generator == "copula":
            synth_n = n_target if n_target is not None else len(x_clean)
            x_syn, y_syn = _generate_correlated_data_copula(
                x_clean, y_clean, float(effect_target), n_samples=synth_n, rng=rng
            )

            # VALIDATION: Check that synthetic data achieves target correlation
            r_synthetic, _ = stats.spearmanr(x_syn, y_syn)
            correlation_error = abs(r_synthetic - float(effect_target))

            if correlation_error > 0.05:
                warnings.warn(
                    f"Synthetic data correlation r={r_synthetic:.3f} differs from "
                    f"target {float(effect_target):.3f} (error={correlation_error:.3f}). "
                    f"This may occur with discrete or heavily tied data. "
                    f"Power estimates for this r value may be biased.",
                    stacklevel=2
                )

            # Optional: Check distributional match (verbose mode only)
            if log_progress and correlation_error > 0.05:
                ks_x, p_x = stats.ks_2samp(x_clean, x_syn)
                ks_y, p_y = stats.ks_2samp(y_clean, y_syn)

                print(f"\n  Distributional validation for r_target={float(effect_target):.3f}:")
                print(f"    Correlation: target={float(effect_target):.3f}, achieved={r_synthetic:.3f}, error={correlation_error:.3f}")
                print(f"    X marginal: KS={ks_x:.3f}, p={p_x:.3f}")
                print(f"    Y marginal: KS={ks_y:.3f}, p={p_y:.3f}")

                if p_x < 0.05 or p_y < 0.05:
                    print("    ⚠ Warning: Marginal distributions differ significantly from original")

        for method in methods_to_run:
            # Choose data generator
            if data_generator == "mixed_model":
                x_gen, y_gen, sid_gen, iid_gen, rid_gen, r_achieved, beta_used = generate_clustered_data_mixed_model(
                    x_clean,
                    y_clean,
                    sid_clean,
                    iid_clean if iid_clean is not None else np.zeros_like(sid_clean),
                    rubric_ids=rid_clean,
                    beta_target=float(effect_target),
                    rng=rng,
                    r_target=None,
                )

                calibration_failed = False
            else:
                x_gen, y_gen = x_syn, y_syn
                sid_gen = sid_clean
                iid_gen = iid_clean
                rid_gen = rid_clean
                r_achieved = None
                beta_used = np.nan
                calibration_failed = False

            if method == "spearman_perm":
                power = estimate_power_spearman_simulation(
                    x_gen,
                    y_gen,
                    n_target=n_target,
                    alpha=alpha,
                    n_power_sim=n_power_sim,
                    n_perm=n_perm,
                    random_state=rng,
                    log_progress=False,  # Don't show nested progress
                )
            else:
                # Cluster-robust path requires cluster labels
                if sid_gen is None:
                    raise ValueError("student_ids must be provided for cluster_robust power calculation")

                # print("Running cluster-robust power estimation...")
                # print(f"rubric_ids {rid_gen}")

                power = estimate_power_cluster_robust_simulation(
                    x_gen,
                    y_gen,
                    student_ids=sid_gen,
                    question_ids=iid_gen,
                    rubric_ids=rid_gen,
                    n_target=n_target,
                    alpha=alpha,
                    n_power_sim=n_power_sim,
                    random_state=rng,
                    log_progress=True,
                )

            # Interpret power
            if power >= 0.9:
                interpretation = "Very High"
            elif power >= 0.8:
                interpretation = "High"
            elif power >= 0.5:
                interpretation = "Moderate"
            elif power >= 0.2:
                interpretation = "Low"
            else:
                interpretation = "Very Low"

            results.append({
                'r_target': float(effect_target) if data_generator == "copula" else np.nan,
                'beta_target': float(effect_target) if data_generator == "mixed_model" else np.nan,
                'r_achieved': r_achieved if data_generator == "mixed_model" else np.nan,
                'beta_used': beta_used if data_generator == "mixed_model" else np.nan,
                'calibration_failed': calibration_failed,
                'power': power,
                'interpretation': interpretation,
                'method': method,
            })

    return pd.DataFrame(results)


def estimate_power_spearman_simulation_fresh_perms(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_target: int,
    alpha: float = 0.05,
    n_power_sim: int = 1000,
    n_perm: int = 1000,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
    adaptive_perm: bool = False,
    base_n_perm: int = 2000,
    max_n_perm: int = 20000,
    adaptive_band: Tuple[float, float] = (0.5, 2.0),
) -> float:
    """
    Monte Carlo power estimate using FRESH permutations per simulation.

    This is the theoretically pure version: each power simulation generates
    its own fresh random permutations. Slower than vectorized version but
    serves as gold standard for validation.

    Args:
        x: Feature values
        y: Target values
        n_target: Sample size for power calculation
        alpha: Significance level
        n_power_sim: Number of Monte Carlo simulations
        n_perm: Number of permutations per test
        random_state: Random number generator
        log_progress: Show progress bar

    Returns:
        Estimated statistical power (fraction of rejections)

    Example:
        >>> power = estimate_power_spearman_simulation_fresh_perms(x, y, n_target=100)
        >>> print(f"Power: {power:.3f}")
    """
    rng = random_state or np.random.default_rng()

    # Clean data
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n_source = len(x_clean)

    if n_source < 3 or n_target < 3 or n_power_sim <= 0:
        return np.nan

    sig_count = 0

    iterator = range(n_power_sim)
    if log_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_power_sim, desc="Power (fresh perms)", leave=False)
        except ImportError:
            pass

    for _ in iterator:
        # Resample data
        idx = rng.integers(0, n_source, size=n_target)
        x_b = x_clean[idx]
        y_b = y_clean[idx]

        # Compute permutation p-value with FRESH permutations
        res = compute_spearman_with_perm_and_bootstrap(
            x_b,
            y_b,
            n_perm=n_perm,
            n_boot=0,  # Skip bootstrap (only need p-value)
            alpha=alpha,
            random_state=rng,
            log_progress=False,
            adaptive_perm=adaptive_perm,
            base_n_perm=base_n_perm,
            max_n_perm=max_n_perm,
            adaptive_band=adaptive_band,
        )

        # Check if significant
        p_val = res.get('p_perm', np.nan)
        if (
            not res.get('insufficient_data')
            and not res.get('undefined_correlation')
            and np.isfinite(p_val)
            and p_val < alpha
        ):
            sig_count += 1

    return sig_count / n_power_sim


def estimate_power_cluster_robust_simulation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    student_ids: np.ndarray,
    question_ids: Optional[np.ndarray] = None,
    rubric_ids: Optional[np.ndarray] = None,
    n_target: int,
    alpha: float = 0.05,
    n_power_sim: int = 1000,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
) -> float:
    """
    Monte Carlo power estimate using the cluster-robust correlation method.

    - Resamples paired data with replacement to size n_target.
    - Resamples cluster labels (student/item) from the observed structure.
    - Computes p-values via estimate_feature_effect_cluster_robust.
    - Power = fraction of runs with p_value < alpha.
    """
    rng = random_state or np.random.default_rng()

    # Clean data in sync
    data_cols: Dict[str, Any] = {
        'x': x,
        'y': y,
        'student_id': student_ids,
    }
    if question_ids is not None:
        data_cols['question_id'] = question_ids
    if rubric_ids is not None:
        data_cols['rubric_id'] = rubric_ids
    data_df = pd.DataFrame(data_cols).dropna()

    x_clean = data_df['x'].to_numpy()
    y_clean = data_df['y'].to_numpy()
    sid_clean = data_df['student_id'].to_numpy()
    iid_clean = data_df['question_id'].to_numpy() if question_ids is not None else None
    rid_clean = data_df['rubric_id'].to_numpy() if rubric_ids is not None else None
    n_source = len(x_clean)

    if n_source < 3 or n_target < 3 or n_power_sim <= 0:
        return np.nan

    sig_count = 0

    iterator = range(n_power_sim)
    if log_progress:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(iterator, total=n_power_sim, desc="Power (cluster-robust)", leave=False)
        except Exception:
            iterator = range(n_power_sim)

    for _ in iterator:
        idx = rng.integers(0, n_source, size=n_target)
        x_b = x_clean[idx]
        y_b = y_clean[idx]
        sid_b = sid_clean[idx]
        iid_b = iid_clean[idx] if iid_clean is not None else None
        rid_b = rid_clean[idx] if rid_clean is not None else None

        res = estimate_feature_effect_cluster_robust(
            feature=x_b,
            outcome=y_b,
            student_ids=sid_b,
            question_ids=iid_b,
            rubric_ids=rid_b,
            use_ranks=True,
            standardize=True,
            alpha=alpha,
        )

        p_val = res.get("p_value", np.nan)
        if np.isfinite(p_val) and p_val < alpha:
            sig_count += 1

    return sig_count / n_power_sim


def compare_power_estimation_methods(
    x: np.ndarray,
    y: np.ndarray,
    n_target: Optional[int] = None,
    n_power_sim: int = 100,
    n_perm: int = 1000,
    random_state: Optional[np.random.Generator] = None,
    adaptive_perm: bool = False,
    base_n_perm: int = 2000,
    max_n_perm: int = 20000,
    adaptive_band: Tuple[float, float] = (0.5, 2.0),
) -> Dict[str, Any]:
    """
    Compare power estimates from fresh vs reused permutations.

    Runs both methods on same data with same random seed and reports:
        - Power from each method
        - Absolute difference
        - Whether difference is negligible (<2%)

    Args:
        x: Feature values
        y: Target values
        n_target: Sample size (default: len(x))
        n_power_sim: Number of simulations (use fewer for quick validation)
        n_perm: Number of permutations per test
        random_state: Random number generator

    Returns:
        Dict with power_vectorized, power_fresh_perms, absolute_diff,
        relative_diff_pct, negligible (bool), n_power_sim, n_perm

    Example:
        >>> comparison = compare_power_estimation_methods(x, y, n_power_sim=100)
        >>> if comparison['negligible']:
        >>>     print("Permutation reuse is acceptable")
    """
    rng = random_state or np.random.default_rng()

    # Clean data
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if n_target is None:
        n_target = len(x_clean)

    # Method 1: Vectorized (reused permutations)
    power_vectorized = estimate_power_spearman_simulation(
        x_clean,
        y_clean,
        n_target=n_target,
        n_power_sim=n_power_sim,
        n_perm=n_perm,
        random_state=rng,
        log_progress=False,
    )

    # Method 2: Fresh permutations
    # Create fresh RNG with same seed to ensure fair comparison
    rng_fresh = np.random.default_rng(rng.integers(0, 2**32))
    power_fresh = estimate_power_spearman_simulation_fresh_perms(
        x_clean,
        y_clean,
        n_target=n_target,
        n_power_sim=n_power_sim,
        n_perm=n_perm,
        random_state=rng_fresh,
        log_progress=False,
        adaptive_perm=adaptive_perm,
        base_n_perm=base_n_perm,
        max_n_perm=max_n_perm,
        adaptive_band=adaptive_band,
    )

    # Compute differences
    abs_diff = abs(power_vectorized - power_fresh)

    # Relative difference (handle edge case where power is near 0)
    if power_fresh > 0.01:
        rel_diff_pct = (abs_diff / power_fresh) * 100
    else:
        rel_diff_pct = np.nan

    # Threshold for "negligible": 2% absolute difference
    negligible = abs_diff < 0.02

    return {
        'power_vectorized': float(power_vectorized),
        'power_fresh_perms': float(power_fresh),
        'absolute_diff': float(abs_diff),
        'relative_diff_pct': float(rel_diff_pct) if np.isfinite(rel_diff_pct) else np.nan,
        'negligible': bool(negligible),
        'n_power_sim': n_power_sim,
        'n_perm': n_perm,
    }


def _assurance_power_from_effect_samples(
    effect_samples: np.ndarray,
    x_template: np.ndarray,
    y_template: np.ndarray,
    *,
    power_fn: Callable[[np.ndarray, np.ndarray], float],
    n_target: int,
    rng: np.random.Generator,
    log_progress: bool = False,
) -> np.ndarray:
    """
    Helper to compute power for a set of effect samples using provided power_fn.
    """
    power_per_effect = []
    iterator = effect_samples
    if log_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(effect_samples, desc="Power for each effect", leave=False)
        except ImportError:
            pass

    for eff in iterator:
        x_syn, y_syn = _generate_correlated_data_copula(
            x_template, y_template, eff, n_samples=len(x_template) * 2, rng=rng
        )
        power_per_effect.append(power_fn(x_syn, y_syn))

    return np.array(power_per_effect)


def compute_bayesian_assurance_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    n_target: int,
    n_bootstrap: int = 1000,
    n_power_sim_per_r: int = 200,
    n_perm: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
) -> Dict[str, Any]:
    """
    Compute Bayesian assurance using bootstrap empirical prior.

    Instead of subjectively specifying a prior, bootstrap the observed
    data to get an empirical distribution of plausible effect sizes,
    then compute expected power under this distribution.

    Args:
        x, y: Observed data (current study)
        n_target: Sample size for future study
        n_bootstrap: Number of bootstrap samples
        n_power_sim_per_r: Power simulations per bootstrapped r
        n_perm: Permutations per power test
        alpha: Significance level
        random_state: RNG
        log_progress: Show progress

    Returns:
        {
            'assurance': float,              # Expected power
            'assurance_ci': (low, high),     # 95% CI via percentiles
            'assurance_sd': float,           # Standard deviation
            'bootstrap_rs': np.ndarray,      # Bootstrapped correlations
            'power_per_r': np.ndarray,       # Power for each bootstrap r
            'n_current': int,                # Current sample size
            'n_target': int,                 # Target sample size
            'r_observed': float,             # Observed correlation
        }

    Example:
        >>> # Current study: n=8392, r=0.15
        >>> result = compute_bayesian_assurance_bootstrap(x, y, n_target=500)
        >>> print(f"Assurance for n=500: {result['assurance']:.3f}")
        >>> # "Expected power = 0.68 (95% CI: [0.62, 0.74])"
    """
    rng = random_state or np.random.default_rng()

    # Clean data
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n_current = len(x_clean)

    # Observed correlation
    r_observed, _ = stats.spearmanr(x_clean, y_clean)

    print(f"Empirical Bayesian Assurance (Bootstrap Method)")
    print(f"Current sample: n = {n_current:,}, r_obs = {r_observed:.4f}")
    print(f"Target sample: n = {n_target:,}")
    print(f"Bootstrapping {n_bootstrap} samples to construct empirical prior...\n")

    # STEP 1: Bootstrap to get empirical distribution of r
    bootstrap_rs = []
    iterator = range(n_bootstrap)
    if log_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Bootstrap samples", leave=False)
        except ImportError:
            pass

    for _ in iterator:
        idx = rng.integers(0, n_current, size=n_current)
        r_boot, _ = stats.spearmanr(x_clean[idx], y_clean[idx])
        bootstrap_rs.append(r_boot)

    bootstrap_rs = np.array(bootstrap_rs)

    print(f"Bootstrap distribution of r:")
    print(f"  Mean: {np.mean(bootstrap_rs):.4f}")
    print(f"  Std:  {np.std(bootstrap_rs):.4f}")
    print(f"  95% CI: [{np.percentile(bootstrap_rs, 2.5):.4f}, {np.percentile(bootstrap_rs, 97.5):.4f}]")
    print()

    # STEP 2: Compute power for each bootstrapped r
    print(f"Computing power for each bootstrapped r (n={n_power_sim_per_r} sims per r)...")

    power_fn = lambda xs, ys: estimate_power_spearman_simulation(  # noqa: E731
        xs,
        ys,
        n_target=n_target,
        alpha=alpha,
        n_power_sim=n_power_sim_per_r,
        n_perm=n_perm,
        random_state=rng,
        log_progress=False,
    )

    power_per_r = _assurance_power_from_effect_samples(
        bootstrap_rs,
        x_clean,
        y_clean,
        power_fn=power_fn,
        n_target=n_target,
        rng=rng,
        log_progress=log_progress,
    )

    # STEP 3: Assurance = E[Power(r)] under empirical distribution
    assurance = np.mean(power_per_r)
    assurance_ci = tuple(np.percentile(power_per_r, [2.5, 97.5]))
    assurance_sd = np.std(power_per_r)

    return {
        'assurance': float(assurance),
        'assurance_ci': assurance_ci,
        'assurance_sd': float(assurance_sd),
        'bootstrap_rs': bootstrap_rs,
        'power_per_r': power_per_r,
        'n_current': n_current,
        'n_target': n_target,
        'r_observed': float(r_observed),
    }


def compute_bayesian_assurance_posterior(
    x: np.ndarray,
    y: np.ndarray,
    n_target: int,
    n_posterior_samples: int = 1000,
    n_power_sim_per_r: int = 200,
    n_perm: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
) -> Dict[str, Any]:
    """
    Compute Bayesian assurance using posterior from Fisher z-transform.

    Uses Fisher's asymptotic result: z(r) ~ Normal(z(ρ), 1/√(n-3))
    to compute posterior over true correlation ρ, then estimates
    expected power under this posterior.

    Faster than bootstrap but relies on asymptotic normality.

    Args:
        x, y: Observed data
        n_target: Target sample size
        n_posterior_samples: Samples from posterior
        n_power_sim_per_r: Power sims per sampled r
        n_perm: Permutations per test
        alpha: Significance level
        random_state: RNG
        log_progress: Show progress

    Returns:
        Same structure as compute_bayesian_assurance_bootstrap but
        with 'posterior_rs' instead of 'bootstrap_rs'

    Example:
        >>> # Current study: n=8392, r=0.15
        >>> result = compute_bayesian_assurance_posterior(x, y, n_target=500)
        >>> print(f"Assurance for n=500: {result['assurance']:.3f}")
    """
    rng = random_state or np.random.default_rng()

    # Clean data
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n_current = len(x_clean)

    # Observed correlation
    r_observed, _ = stats.spearmanr(x_clean, y_clean)

    print(f"Empirical Bayesian Assurance (Posterior Method)")
    print(f"Current sample: n = {n_current:,}, r_obs = {r_observed:.4f}")
    print(f"Target sample: n = {n_target:,}")
    print(f"Sampling {n_posterior_samples} from posterior via Fisher z...\n")

    # STEP 1: Fisher z-transform to get posterior
    z_observed = np.arctanh(np.clip(r_observed, -0.9999, 0.9999))
    se_z = 1 / np.sqrt(n_current - 3)

    # Sample from posterior: z ~ Normal(z_obs, se_z)
    posterior_zs = rng.normal(z_observed, se_z, size=n_posterior_samples)

    # Transform back to correlation
    posterior_rs = np.tanh(posterior_zs)
    posterior_rs = np.clip(posterior_rs, -0.999, 0.999)

    print(f"Posterior distribution of ρ:")
    print(f"  Mean: {np.mean(posterior_rs):.4f}")
    print(f"  Std:  {np.std(posterior_rs):.4f}")
    print(f"  95% CI: [{np.percentile(posterior_rs, 2.5):.4f}, {np.percentile(posterior_rs, 97.5):.4f}]")
    print()

    # STEP 2: Compute power for each sampled r
    print(f"Computing power for each sampled r (n={n_power_sim_per_r} sims per r)...")

    power_fn = lambda xs, ys: estimate_power_spearman_simulation(  # noqa: E731
        xs,
        ys,
        n_target=n_target,
        alpha=alpha,
        n_power_sim=n_power_sim_per_r,
        n_perm=n_perm,
        random_state=rng,
        log_progress=False,
    )

    power_per_r = _assurance_power_from_effect_samples(
        posterior_rs,
        x_clean,
        y_clean,
        power_fn=power_fn,
        n_target=n_target,
        rng=rng,
        log_progress=log_progress,
    )

    # STEP 3: Assurance
    assurance = np.mean(power_per_r)
    assurance_ci = tuple(np.percentile(power_per_r, [2.5, 97.5]))
    assurance_sd = np.std(power_per_r)

    return {
        'assurance': float(assurance),
        'assurance_ci': assurance_ci,
        'assurance_sd': float(assurance_sd),
        'posterior_rs': posterior_rs,
        'power_per_r': power_per_r,
        'n_current': n_current,
        'n_target': n_target,
        'r_observed': float(r_observed),
    }


def compute_bayesian_assurance_cluster_robust(
    x: np.ndarray,
    y: np.ndarray,
    *,
    student_ids: np.ndarray,
    question_ids: Optional[np.ndarray] = None,
    rubric_ids: Optional[np.ndarray] = None,
    n_target: int,
    prior_spec: Dict[str, Any],
    alpha: float = 0.05,
    decision_rule: str = "p_value",
    n_sim: int = 2000,
    random_state: Optional[np.random.Generator] = None,
    log_progress: bool = False,
) -> Dict[str, Any]:
    """
    Compute Bayesian assurance for the cluster-robust correlation-like slope.

    Reuses the Bayesian assurance machinery by plugging in:
      - estimate_feature_effect_cluster_robust for effect estimation
      - estimate_power_cluster_robust_simulation for the design power

    Args:
        x, y: Observed data
        student_ids: Student cluster labels
        question_ids: Optional item cluster labels
        n_target: Target sample size for future study
        prior_spec: Dict describing prior; if it contains 'effect_samples',
                    those are used directly; otherwise bootstrap draws (n_sim)
                    from the observed data are used as an empirical prior.
        alpha: Significance level
        decision_rule: Currently only "p_value" is supported
        n_sim: Number of prior samples to draw when bootstrapping
        random_state: RNG
        log_progress: Show progress during bootstrapping/power
    """
    if prior_spec is None:
        raise ValueError("prior_spec is required for Bayesian assurance.")
    if decision_rule != "p_value":
        raise ValueError(f"Only decision_rule='p_value' is supported, got {decision_rule}")

    rng = random_state or np.random.default_rng()

    data_cols: Dict[str, Any] = {
        'x': x,
        'y': y,
        'student_id': student_ids,
    }
    if question_ids is not None:
        data_cols['question_id'] = question_ids
    if rubric_ids is not None:
        data_cols['rubric_id'] = rubric_ids

    data_df = pd.DataFrame(data_cols).dropna()
    x_clean = data_df['x'].to_numpy()
    y_clean = data_df['y'].to_numpy()
    sid_clean = data_df['student_id'].to_numpy()
    iid_clean = data_df['question_id'].to_numpy() if question_ids is not None else None
    rid_clean = data_df['rubric_id'].to_numpy() if rubric_ids is not None else None
    n_current = len(x_clean)

    if n_current < 3:
        raise ValueError(f"Insufficient data after cleaning for assurance: n={n_current}")

    observed = estimate_feature_effect_cluster_robust(
        feature=x_clean,
        outcome=y_clean,
        student_ids=sid_clean,
        question_ids=iid_clean,
        use_ranks=True,
        standardize=True,
        alpha=alpha,
    )
    beta_obs = observed.get("beta", np.nan)

    effect_samples = prior_spec.get("effect_samples")
    if effect_samples is None:
        print(f"Empirical Bayesian Assurance (Cluster-robust, bootstrap prior)")
        print(f"Current sample: n = {n_current:,}, beta_obs = {beta_obs:.4f}")
        print(f"Target sample: n = {n_target:,}")
        print(f"Bootstrapping {n_sim} samples to construct empirical prior...\n")

        samples = []
        iterator = range(n_sim)
        if log_progress:
            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(iterator, desc="Bootstrap beta", leave=False)
            except Exception:
                iterator = range(n_sim)

        # Build mappings once for multiway cluster bootstrap
        student_to_indices: Dict[Any, np.ndarray] = {
            sid: np.where(sid_clean == sid)[0] for sid in np.unique(sid_clean)
        }
        rubric_to_indices: Dict[Any, np.ndarray] = {}
        if rid_clean is not None:
            rubric_to_indices = {rid: np.where(rid_clean == rid)[0] for rid in np.unique(rid_clean)}
        item_to_indices: Dict[Any, np.ndarray] = {}
        if iid_clean is not None:
            item_to_indices = {iid: np.where(iid_clean == iid)[0] for iid in np.unique(iid_clean)}

        for _ in iterator:
            sampled_idx_parts: List[np.ndarray] = []

            # Sample students
            sampled_students = rng.choice(
                list(student_to_indices.keys()),
                size=len(student_to_indices),
                replace=True,
            )
            sampled_idx_parts.extend([student_to_indices[s] for s in sampled_students if len(student_to_indices[s]) > 0])

            # Sample rubric clusters if available
            if rubric_to_indices:
                sampled_rubrics = rng.choice(
                    list(rubric_to_indices.keys()),
                    size=len(rubric_to_indices),
                    replace=True,
                )
                sampled_idx_parts.extend([rubric_to_indices[r] for r in sampled_rubrics if len(rubric_to_indices[r]) > 0])

            # Sample item clusters if available
            if item_to_indices:
                sampled_items = rng.choice(
                    list(item_to_indices.keys()),
                    size=len(item_to_indices),
                    replace=True,
                )
                sampled_idx_parts.extend([item_to_indices[i] for i in sampled_items if len(item_to_indices[i]) > 0])

            if not sampled_idx_parts:
                continue

            sampled_idx = np.concatenate(sampled_idx_parts)

            # Enforce fixed length to preserve target n_current
            if len(sampled_idx) >= n_current:
                idx = sampled_idx[:n_current]
            else:
                extra = rng.choice(sampled_idx, size=n_current - len(sampled_idx), replace=True)
                idx = np.concatenate([sampled_idx, extra])

            res = estimate_feature_effect_cluster_robust(
                feature=x_clean[idx],
                outcome=y_clean[idx],
                student_ids=sid_clean[idx],
                question_ids=iid_clean[idx] if iid_clean is not None else None,
                rubric_ids=rid_clean[idx] if rid_clean is not None else None,
                use_ranks=True,
                standardize=True,
                alpha=alpha,
            )
            samples.append(res.get("beta", np.nan))

        effect_samples = np.array(samples)
    else:
        effect_samples = np.asarray(effect_samples)

    effect_samples = effect_samples[np.isfinite(effect_samples)]
    if len(effect_samples) == 0:
        raise ValueError("No finite effect samples available for assurance calculation.")

    print(f"Effect prior:")
    print(f"  Mean: {np.mean(effect_samples):.4f}")
    print(f"  Std:  {np.std(effect_samples):.4f}")
    print(f"  95% CI: [{np.percentile(effect_samples, 2.5):.4f}, {np.percentile(effect_samples, 97.5):.4f}]\n")

    power_fn = lambda xs, ys: estimate_power_cluster_robust_simulation(  # noqa: E731
        xs,
        ys,
        student_ids=sid_clean,
        question_ids=iid_clean,
        rubric_ids=rid_clean,
        n_target=n_target,
        alpha=alpha,
        n_power_sim=prior_spec.get("n_power_sim_per_effect", 200),
        random_state=rng,
        log_progress=False,
    )

    power_per_effect = _assurance_power_from_effect_samples(
        effect_samples,
        x_clean,
        y_clean,
        power_fn=power_fn,
        n_target=n_target,
        rng=rng,
        log_progress=log_progress,
    )

    assurance = float(np.mean(power_per_effect))
    assurance_ci = tuple(np.percentile(power_per_effect, [2.5, 97.5]))
    assurance_sd = float(np.std(power_per_effect))

    return {
        'assurance': assurance,
        'assurance_ci': assurance_ci,
        'assurance_sd': assurance_sd,
        'effect_samples': effect_samples,
        'power_per_effect': power_per_effect,
        'n_current': n_current,
        'n_target': n_target,
        'beta_observed': float(beta_obs),
        'method': 'cluster_robust',
    }


def compute_power_curve_mixed_model_lrt(
    feature: np.ndarray,
    outcome: np.ndarray,
    *,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    beta_targets: List[float],
    n_students_target: Optional[int] = None,
    alpha: float = 0.05,
    n_sim: int = 200,
    seed_students: int = 123,
    seed_outcomes: int = 456,
    include_student_vc_glmm: bool = False,
    use_ranks: bool = False,
    standardize_feature: bool = True,
    standardize_outcome: bool = False,
    transform_bounded: Literal["none", "logit_clip"] = "none",
    eps: float = 1e-8,
    log_progress: bool = False,
) -> pd.DataFrame:
    """
    Prospective power curve for the Step-3 mixed model using an LRT for H0: beta=0.

    This is a two-sided detection problem, so the curve is parameterized by |beta_target|.

    Notes:
    - Students are resampled as blocks (with relabeling) to preserve within-student dependence
      while treating duplicated draws as independent subjects.
    - For LMM LRT validity, both models are fit with ML (REML=FALSE).
    - Requires the optional rpy2/lme4 backend.
    """
    prep = _prepare_mixed_model_power_inputs(
        feature=feature,
        outcome=outcome,
        student_ids=student_ids,
        question_ids=question_ids,
        rubric_ids=rubric_ids,
        use_ranks=use_ranks,
        standardize_feature=standardize_feature,
        standardize_outcome=standardize_outcome,
        transform_bounded=transform_bounded,
        eps=eps,
    )
    df = prep["df"]
    include_question_fe = bool(prep["include_question_fe"])
    include_question_re = bool(prep["include_question_re"])
    feature_type = str(prep["feature_type"])
    is_binary = bool(prep["is_binary"])
    include_answer_re = _should_include_answer_re(df)
    if not include_answer_re:
        n_levels = int(df["answer_id"].nunique())
        max_count = int(df["answer_id"].value_counts().max()) if n_levels else 0
        warnings.warn(
            "Dropping answer_id random effect in power curve "
            f"(answer_id has no repeats: n_levels={n_levels}, max_count={max_count}).",
            RuntimeWarning,
        )
    n_students_current = int(prep["n_students_current"])
    n_students_target_val = int(n_students_target) if n_students_target is not None else n_students_current
    if n_students_target_val < 2:
        raise ValueError(f"n_students_target must be >=2, got {n_students_target_val}")

    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]
    stats_r = payload["stats"]

    # Convert df to R once; run simulation loop in R for performance.
    needed_cols = ["outcome_z", "feature_z", "student_id", "question_id", "answer_id", "rubric_id"]
    df_r = df[needed_cols].copy()
    for c in ["student_id", "question_id", "rubric_id", "answer_id"]:
        df_r[c] = pd.Categorical(df_r[c].astype(str))

    with localconverter(converter):
        ro.globalenv["df_power"] = rconversion.py2rpy(df_r)

    # Define the R implementation once per session.
    if "gc_power_curve_mixed_model_lrt" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
	                gc_power_curve_mixed_model_lrt <- function(df, beta_targets_abs, n_students_target, alpha,
	                                                          n_sim, seed_students, seed_outcomes,
	                                                          include_question_fe, include_question_re,
	                                                          is_glmm, include_student_vc,
	                                                          include_answer_re) {
	                  suppressPackageStartupMessages(library(lme4))
	                  suppressPackageStartupMessages(library(stats))

	                  # Silence lme4 warnings like:
	                  # "boundary (singular) fit: see help('isSingular')"
	                  # We track singular fits explicitly via isSingular() and return counts.
	                  quiet_eval <- function(expr) {
	                    withCallingHandlers(
	                      expr,
	                      warning=function(w) invokeRestart("muffleWarning"),
	                      message=function(m) invokeRestart("muffleMessage")
	                    )
	                  }

	                  df$student_id <- factor(df$student_id)
	                  df$question_id <- factor(df$question_id)
	                  df$rubric_id <- factor(df$rubric_id)
	                  df$answer_id <- factor(df$answer_id)

                  q_fe <- if (include_question_fe) " + factor(question_id)" else ""
                  if (!is_glmm) {
	                    re_terms <- c("(1|student_id)")
	                    if (include_question_re) re_terms <- c(re_terms, "(1|question_id)")
	                    if (include_answer_re) re_terms <- c(re_terms, "(1|answer_id)")
	                    if (nlevels(df$rubric_id) >= 2) re_terms <- c(re_terms, "(1|rubric_id)")
	                    re <- paste0(" + ", paste(re_terms, collapse=" + "))
	                    full_formula <- as.formula(paste0("outcome_z ~ feature_z", q_fe, re))
	                    null_formula <- as.formula(paste0("outcome_z ~ 1", q_fe, re))
	                    ctrl <- lmerControl(optimizer="bobyqa")
	                    fit_full_template <- quiet_eval(lmer(full_formula, data=df, REML=FALSE, control=ctrl))
	                    fit_null_nuis <- quiet_eval(lmer(null_formula, data=df, REML=FALSE, control=ctrl))
	                    theta0 <- getME(fit_null_nuis, "theta")
	                    sigma0 <- sigma(fit_null_nuis)
	                    beta_null <- fixef(fit_null_nuis)
	                  } else {
	                    re_terms <- c()
	                    if (include_question_re) re_terms <- c(re_terms, "(1|question_id)")
	                    if (include_answer_re) re_terms <- c(re_terms, "(1|answer_id)")
	                    if (nlevels(df$rubric_id) >= 2) re_terms <- c(re_terms, "(1|rubric_id)")
	                    if (include_student_vc) re_terms <- c(re_terms, "(1|student_id)")
	                    re <- if (length(re_terms) > 0) paste0(" + ", paste(re_terms, collapse=" + ")) else ""
	                    full_formula <- as.formula(paste0("outcome_z ~ feature_z", q_fe, re))
	                    null_formula <- as.formula(paste0("outcome_z ~ 1", q_fe, re))
	                    ctrl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5))
	                    fit_full_template <- quiet_eval(glmer(full_formula, data=df, family=binomial(link="logit"),
	                                               control=ctrl, nAGQ=1))
	                    fit_null_nuis <- quiet_eval(glmer(null_formula, data=df, family=binomial(link="logit"),
	                                           control=ctrl, nAGQ=1))
	                    theta0 <- getME(fit_null_nuis, "theta")
	                    beta_null <- fixef(fit_null_nuis)
	                  }

                  students <- levels(df$student_id)
                  n_students_current <- length(students)

                  set.seed(seed_students)
                  draws <- replicate(n_sim, sample(students, size=n_students_target, replace=TRUE), simplify=FALSE)

                  build_sample_df <- function(draw_vec) {
                    parts <- vector("list", length(draw_vec))
                    for (i in seq_along(draw_vec)) {
                      sid <- draw_vec[[i]]
                      block <- df[df$student_id == sid, , drop=FALSE]
                      tag <- paste0("__rep", i)
                      block$student_id <- factor(paste0(as.character(block$student_id), tag))
                      block$answer_id <- factor(paste0(as.character(block$answer_id), tag))
                      parts[[i]] <- block
                    }
                    out <- do.call(rbind, parts)
                    out$question_id <- factor(out$question_id, levels=levels(df$question_id))
                    out$rubric_id <- factor(out$rubric_id, levels=levels(df$rubric_id))
                    out
                  }

                  coef_names <- names(fixef(fit_full_template))
                  beta_vec_base <- fixef(fit_full_template)
                  beta_vec_base[] <- 0
                  for (nm in names(beta_null)) {
                    if (nm %in% coef_names) beta_vec_base[[nm]] <- beta_null[[nm]]
                  }

                  out_rows <- vector("list", length(beta_targets_abs))
                  for (j in seq_along(beta_targets_abs)) {
                    beta_abs <- beta_targets_abs[[j]]
                    beta_vec <- beta_vec_base
                    if ("feature_z" %in% coef_names) {
                      beta_vec[["feature_z"]] <- beta_abs
                    } else {
                      # Defensive: if feature_z got renamed, set the first matching term.
                      hit <- grep("feature_z", coef_names, fixed=TRUE)
                      if (length(hit) >= 1) beta_vec[[coef_names[[hit[[1]]]]]] <- beta_abs
                    }

                    set.seed(seed_outcomes + j)
                    pvals <- rep(NA_real_, n_sim)
                    n_failed <- 0L
                    n_singular <- 0L

                    for (b in seq_len(n_sim)) {
                      sim_df <- build_sample_df(draws[[b]])
                      if (!is_glmm) {
	                        ysim <- tryCatch(
	                          simulate(fit_full_template, nsim=1, newdata=sim_df,
	                                   newparams=list(theta=theta0, beta=beta_vec, sigma=sigma0),
	                                   re.form=NULL, allow.new.levels=TRUE)[[1]],
	                          error=function(e) NA_real_
	                        )
	                        sim_df$outcome_z <- as.numeric(ysim)
	                        fit_full <- quiet_eval(tryCatch(lmer(full_formula, data=sim_df, REML=FALSE, control=ctrl),
	                                             error=function(e) NULL))
	                        fit_null <- quiet_eval(tryCatch(lmer(null_formula, data=sim_df, REML=FALSE, control=ctrl),
	                                             error=function(e) NULL))
	                        if (is.null(fit_full) || is.null(fit_null)) {
	                          n_failed <- n_failed + 1L
	                          next
	                        }
                        if (isSingular(fit_full, tol=1e-4)) n_singular <- n_singular + 1L
                        a <- quiet_eval(tryCatch(anova(fit_null, fit_full), error=function(e) NULL))
                        if (!is.null(a) && "Pr(>Chisq)" %in% colnames(a)) {
                          pvals[[b]] <- a[["Pr(>Chisq)"]][nrow(a)]
                        } else {
                          n_failed <- n_failed + 1L
                        }
                      } else {
	                        ysim <- tryCatch(
	                          simulate(fit_full_template, nsim=1, newdata=sim_df,
	                                   newparams=list(theta=theta0, beta=beta_vec),
	                                   re.form=NULL, allow.new.levels=TRUE)[[1]],
	                          error=function(e) NA_real_
	                        )
	                        sim_df$outcome_z <- as.integer(ysim)
	                        fit_full <- quiet_eval(tryCatch(glmer(full_formula, data=sim_df, family=binomial(link="logit"),
	                                                   control=ctrl, nAGQ=1),
	                                             error=function(e) NULL))
	                        fit_null <- quiet_eval(tryCatch(glmer(null_formula, data=sim_df, family=binomial(link="logit"),
	                                                   control=ctrl, nAGQ=1),
	                                             error=function(e) NULL))
	                        if (is.null(fit_full) || is.null(fit_null)) {
	                          n_failed <- n_failed + 1L
	                          next
	                        }
                        if (isSingular(fit_full, tol=1e-4)) n_singular <- n_singular + 1L
                        a <- quiet_eval(tryCatch(anova(fit_null, fit_full), error=function(e) NULL))
                        if (!is.null(a) && "Pr(>Chisq)" %in% colnames(a)) {
                          pvals[[b]] <- a[["Pr(>Chisq)"]][nrow(a)]
                        } else {
                          n_failed <- n_failed + 1L
                        }
                      }
                    }

                    ok <- is.finite(pvals)
                    n_success <- sum(ok)
                    power <- if (n_success > 0) mean(pvals[ok] < alpha) else NA_real_
                    out_rows[[j]] <- data.frame(
                      beta_target=beta_abs,
                      power=power,
                      n_success=n_success,
                      n_failed=n_failed,
                      n_singular=n_singular,
                      stringsAsFactors=FALSE
                    )
                  }

                  do.call(rbind, out_rows)
                }
                """
            )
        )

    beta_abs = [abs(float(b)) for b in beta_targets]

    # Optional progress reporting: loop in Python over |beta| values while keeping the
    # Monte Carlo loop in R. We preserve the original per-beta seeding behavior:
    # R uses set.seed(seed_outcomes + j), where j is 1-indexed within beta_targets_abs.
    # When calling with a single beta, j==1, so pass seed_outcomes + (idx) so that
    # seed_outcomes_single + 1 == seed_outcomes + (idx + 1).
    if log_progress and len(beta_abs) > 1:
        import sys

        rows = []
        iterator = tqdm(
            enumerate(beta_abs),
            total=len(beta_abs),
            desc="Power curve (mixed_model_lrt)",
            unit="|beta|",
            dynamic_ncols=True,
            leave=True,
            disable=not log_progress,
            file=sys.stderr,
        )
        for idx, beta_abs_one in iterator:
            ro.globalenv["beta_targets_abs"] = ro.FloatVector([float(beta_abs_one)])
            res_r_one = ro.globalenv["gc_power_curve_mixed_model_lrt"](
                ro.globalenv["df_power"],
                ro.globalenv["beta_targets_abs"],
                int(n_students_target_val),
                float(alpha),
                int(n_sim),
                int(seed_students),
                int(seed_outcomes + idx),
                bool(include_question_fe),
                bool(include_question_re),
                bool(is_binary),
                bool(include_student_vc_glmm),
                bool(include_answer_re),
            )
            with localconverter(converter):
                rows.append(rconversion.rpy2py(res_r_one))
        res_df = pd.concat(rows, ignore_index=True)
    else:
        ro.globalenv["beta_targets_abs"] = ro.FloatVector(beta_abs)
        res_r = ro.globalenv["gc_power_curve_mixed_model_lrt"](
            ro.globalenv["df_power"],
            ro.globalenv["beta_targets_abs"],
            int(n_students_target_val),
            float(alpha),
            int(n_sim),
            int(seed_students),
            int(seed_outcomes),
            bool(include_question_fe),
            bool(include_question_re),
            bool(is_binary),
            bool(include_student_vc_glmm),
            bool(include_answer_re),
        )
        with localconverter(converter):
            res_df = rconversion.rpy2py(res_r)

    res_df["beta_target"] = res_df["beta_target"].astype(float)
    res_df["beta_target_abs"] = res_df["beta_target"].abs()
    res_df["method"] = "mixed_model_lrt"
    res_df["alpha"] = float(alpha)
    res_df["n_students_current"] = n_students_current
    res_df["n_students_target"] = n_students_target_val
    res_df["include_question_fe"] = bool(include_question_fe)
    res_df["include_question_re"] = bool(include_question_re)
    res_df["feature_type"] = feature_type
    res_df["is_binary"] = bool(is_binary)
    res_df["include_student_vc_glmm"] = bool(include_student_vc_glmm)
    res_df["include_answer_re"] = bool(include_answer_re)
    return res_df.sort_values("beta_target_abs").reset_index(drop=True)


def _prepare_mixed_model_power_inputs(
    *,
    feature: np.ndarray,
    outcome: np.ndarray,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    use_ranks: bool,
    standardize_feature: bool,
    standardize_outcome: bool,
    transform_bounded: Literal["none", "logit_clip"],
    eps: float,
) -> Dict[str, Any]:
    df = pd.DataFrame(
        {
            "feature": np.asarray(feature, dtype=float),
            "outcome": np.asarray(outcome, dtype=float),
            "student_id": student_ids,
            "question_id": question_ids,
            "rubric_id": rubric_ids,
        }
    ).dropna()
    if len(df) < 30:
        raise ValueError(f"Insufficient data for mixed-model power/assurance: n={len(df)}")

    # Build answer_id after dropna on the analytic subset (guardrail).
    answer_keys = pd.MultiIndex.from_arrays(
        [df["student_id"].astype(str).to_numpy(), df["question_id"].astype(str).to_numpy()]
    )
    df["answer_id"] = pd.factorize(answer_keys)[0].astype(str)
    for c in ["student_id", "question_id", "rubric_id", "answer_id"]:
        df[c] = pd.Categorical(df[c].astype(str))

    feat_std = float(df["feature"].std())
    if not np.isfinite(feat_std) or feat_std < eps:
        raise ValueError(f"Feature has near-zero variance (std={feat_std:.2e})")

    feature_type = detect_feature_type(
        df["feature"].to_numpy(),
        df["student_id"].to_numpy(),
        df["question_id"].to_numpy(),
        df["rubric_id"].to_numpy(),
        df["answer_id"].to_numpy(),
        eps=eps,
    )
    include_question_fe, include_question_re, question_effect_mode, n_questions = _select_question_effects(
        df["question_id"].to_numpy(),
        feature_type,
    )

    if use_ranks:
        df["feature"] = rankdata(df["feature"].to_numpy())
        df["outcome"] = rankdata(df["outcome"].to_numpy())

    if transform_bounded != "none":
        out_min = float(np.nanmin(df["outcome"].to_numpy(dtype=float)))
        out_max = float(np.nanmax(df["outcome"].to_numpy(dtype=float)))
        if out_min >= 0 and out_max <= 1 and transform_bounded == "logit_clip":
            clip_eps = 1e-6
            y = np.clip(df["outcome"].to_numpy(dtype=float), clip_eps, 1 - clip_eps)
            df["outcome"] = np.log(y / (1 - y))

    if standardize_feature:
        df["feature_z"] = (df["feature"] - df["feature"].mean()) / float(df["feature"].std())
    else:
        df["feature_z"] = df["feature"]

    if standardize_outcome:
        out_std = float(df["outcome"].std())
        if not np.isfinite(out_std) or out_std < eps:
            raise ValueError(f"Outcome has near-zero variance (std={out_std:.2e})")
        df["outcome_z"] = (df["outcome"] - df["outcome"].mean()) / out_std
    else:
        df["outcome_z"] = df["outcome"]

    y_unique = pd.unique(df["outcome_z"])
    y_unique = np.array([v for v in y_unique if np.isfinite(v)], dtype=float)
    is_binary = set(np.unique(y_unique).tolist()).issubset({0.0, 1.0}) and len(np.unique(y_unique)) <= 2
    if is_binary:
        df["outcome_z"] = pd.to_numeric(df["outcome_z"], errors="coerce").astype(int)

    return {
        "df": df,
        "include_question_fe": include_question_fe,
        "include_question_re": include_question_re,
        "question_effect_mode": question_effect_mode,
        "n_questions": n_questions,
        "feature_type": feature_type,
        "is_binary": is_binary,
        "n_students_current": int(df["student_id"].nunique()),
    }


def compute_empirical_prior_mixed_model_beta(
    feature: np.ndarray,
    outcome: np.ndarray,
    *,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    n_boot: int = 200,
    seed: int = 123,
    include_student_vc_glmm: bool = False,
    use_ranks: bool = False,
    standardize_feature: bool = True,
    standardize_outcome: bool = False,
    transform_bounded: Literal["none", "logit_clip"] = "none",
    drop_singular: bool = True,
    winsorize_pct: Optional[float] = None,
    eps: float = 1e-8,
    log_progress: bool = False,
) -> Dict[str, Any]:
    """
    Empirical prior for beta via student-block bootstrap under the Step-3 mixed model.

    Returns beta samples on the same scale as the Step-3 estimand (based on feature_z/outcome_z).
    """
    if n_boot < 10:
        raise ValueError("n_boot must be >= 10 for a useful empirical prior")

    prep = _prepare_mixed_model_power_inputs(
        feature=feature,
        outcome=outcome,
        student_ids=student_ids,
        question_ids=question_ids,
        rubric_ids=rubric_ids,
        use_ranks=use_ranks,
        standardize_feature=standardize_feature,
        standardize_outcome=standardize_outcome,
        transform_bounded=transform_bounded,
        eps=eps,
    )
    df = prep["df"]
    include_question_fe = bool(prep["include_question_fe"])
    include_question_re = bool(prep["include_question_re"])
    feature_type = str(prep["feature_type"])
    is_binary = bool(prep["is_binary"])

    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    needed_cols = ["outcome_z", "feature_z", "student_id", "question_id", "answer_id", "rubric_id"]
    df_r = df[needed_cols].copy()
    for c in ["student_id", "question_id", "rubric_id", "answer_id"]:
        df_r[c] = pd.Categorical(df_r[c].astype(str))
    with localconverter(converter):
        ro.globalenv["df_prior"] = rconversion.py2rpy(df_r)

    if "gc_empirical_prior_beta_mixed_model" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
	                gc_empirical_prior_beta_mixed_model <- function(df, n_boot, seed,
	                                                               include_question_fe, include_question_re,
	                                                               is_glmm, include_student_vc) {
	                  suppressPackageStartupMessages(library(lme4))
	                  suppressPackageStartupMessages(library(stats))

	                  # Silence lme4 warnings like:
	                  # "boundary (singular) fit: see help('isSingular')"
	                  # We track singular fits explicitly via isSingular() and return them.
	                  quiet_eval <- function(expr) {
	                    withCallingHandlers(
	                      expr,
	                      warning=function(w) invokeRestart("muffleWarning"),
	                      message=function(m) invokeRestart("muffleMessage")
	                    )
	                  }

	                  df$student_id <- factor(df$student_id)
	                  df$question_id <- factor(df$question_id)
	                  df$rubric_id <- factor(df$rubric_id)
	                  df$answer_id <- factor(df$answer_id)

                  q_fe <- if (include_question_fe) " + factor(question_id)" else ""
                  if (!is_glmm) {
	                    re <- " + (1|student_id) + (1|answer_id)"
	                    if (include_question_re) re <- paste0(re, " + (1|question_id)")
	                    if (nlevels(df$rubric_id) >= 2) re <- paste0(re, " + (1|rubric_id)")
	                    full_formula <- as.formula(paste0("outcome_z ~ feature_z", q_fe, re))
	                    ctrl <- lmerControl(optimizer="bobyqa")
	                    fit_template <- quiet_eval(lmer(full_formula, data=df, REML=FALSE, control=ctrl))
	                  } else {
	                    re <- " + (1|answer_id) + (1|rubric_id)"
	                    if (include_question_re) re <- paste0(re, " + (1|question_id)")
	                    if (include_student_vc) re <- paste0(re, " + (1|student_id)")
	                    full_formula <- as.formula(paste0("outcome_z ~ feature_z", q_fe, re))
	                    ctrl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5))
	                    fit_template <- quiet_eval(glmer(full_formula, data=df, family=binomial(link="logit"),
	                                          control=ctrl, nAGQ=1))
	                  }

                  coef_names <- names(fixef(fit_template))
                  coef_name_used <- if ("feature_z" %in% coef_names) "feature_z" else {
                    hit <- grep("feature_z", coef_names, fixed=TRUE)
                    if (length(hit) >= 1) coef_names[[hit[[1]]]] else NA_character_
                  }

                  students <- levels(df$student_id)
                  n_students_current <- length(students)
                  set.seed(seed)

                  betas <- rep(NA_real_, n_boot)
                  singular <- rep(FALSE, n_boot)

                  build_sample_df <- function(draw_vec) {
                    parts <- vector("list", length(draw_vec))
                    for (i in seq_along(draw_vec)) {
                      sid <- draw_vec[[i]]
                      block <- df[df$student_id == sid, , drop=FALSE]
                      tag <- paste0("__rep", i)
                      block$student_id <- factor(paste0(as.character(block$student_id), tag))
                      block$answer_id <- factor(paste0(as.character(block$answer_id), tag))
                      parts[[i]] <- block
                    }
                    out <- do.call(rbind, parts)
                    out$question_id <- factor(out$question_id, levels=levels(df$question_id))
                    out$rubric_id <- factor(out$rubric_id, levels=levels(df$rubric_id))
                    out
                  }

                  for (k in seq_len(n_boot)) {
	                    draw <- sample(students, size=n_students_current, replace=TRUE)
	                    boot_df <- build_sample_df(draw)
	                    fit <- tryCatch(
	                      if (!is_glmm) {
	                        quiet_eval(lmer(full_formula, data=boot_df, REML=FALSE, control=ctrl))
	                      } else {
	                        quiet_eval(glmer(full_formula, data=boot_df, family=binomial(link="logit"),
	                              control=ctrl, nAGQ=1))
	                      },
	                      error=function(e) NULL
	                    )
                    if (is.null(fit) || is.na(coef_name_used)) next
                    singular[[k]] <- isSingular(fit, tol=1e-4)
                    betas[[k]] <- unname(fixef(fit)[[coef_name_used]])
                  }

                  data.frame(beta=betas, singular=singular, stringsAsFactors=FALSE)
                }
                """
            )
        )

    if log_progress and n_boot > 1:
        import sys

        # Chunked calls to provide progress without modifying the R loop; this
        # trades a small amount of overhead for observability.
        chunk_size = 25
        frames = []
        n_done = 0
        iterator = tqdm(
            range(0, n_boot, chunk_size),
            desc="Empirical prior (mixed_model beta)",
            unit="chunk",
            dynamic_ncols=True,
            leave=True,
            disable=not sys.stderr.isatty(),
            file=sys.stderr,
        )
        for start in iterator:
            chunk_n = int(min(chunk_size, n_boot - start))
            # Different seed per chunk for reproducibility and to avoid repeating draws.
            chunk_seed = int(seed + start)
            res_r = ro.globalenv["gc_empirical_prior_beta_mixed_model"](
                ro.globalenv["df_prior"],
                int(chunk_n),
                int(chunk_seed),
                bool(include_question_fe),
                bool(include_question_re),
                bool(is_binary),
                bool(include_student_vc_glmm),
            )
            with localconverter(converter):
                frames.append(rconversion.rpy2py(res_r))
            n_done += chunk_n
        prior_df = pd.concat(frames, ignore_index=True)
    else:
        res_r = ro.globalenv["gc_empirical_prior_beta_mixed_model"](
            ro.globalenv["df_prior"],
            int(n_boot),
            int(seed),
            bool(include_question_fe),
            bool(include_question_re),
            bool(is_binary),
            bool(include_student_vc_glmm),
        )
        with localconverter(converter):
            prior_df = rconversion.rpy2py(res_r)

    prior_df["beta"] = pd.to_numeric(prior_df["beta"], errors="coerce")
    prior_df["singular"] = prior_df["singular"].astype(bool)

    n_attempted = int(len(prior_df))
    n_failed = int((~np.isfinite(prior_df["beta"].to_numpy(dtype=float))).sum())
    n_singular = int(prior_df["singular"].sum())

    beta_vals = prior_df.loc[np.isfinite(prior_df["beta"]), "beta"].to_numpy(dtype=float)
    if drop_singular and n_singular > 0:
        beta_vals = prior_df.loc[np.isfinite(prior_df["beta"]) & (~prior_df["singular"]), "beta"].to_numpy(dtype=float)

    failure_rate = n_failed / max(n_attempted, 1)
    if failure_rate > 0.50:
        raise RuntimeError(
            f"Empirical prior bootstrap failure rate too high: {failure_rate:.1%} "
            f"({n_failed}/{n_attempted})."
        )
    if failure_rate > 0.30:
        warnings.warn(
            f"High empirical prior bootstrap failure rate: {failure_rate:.1%} ({n_failed}/{n_attempted}).",
            stacklevel=2,
        )

    if len(beta_vals) == 0:
        raise RuntimeError("No finite beta samples available for empirical prior.")

    if winsorize_pct is not None:
        p = float(winsorize_pct)
        if not (0 < p < 50):
            raise ValueError("winsorize_pct must be between 0 and 50")
        lo, hi = np.percentile(beta_vals, [p, 100 - p])
        beta_vals = np.clip(beta_vals, lo, hi)

    summary = {
        "mean": float(np.mean(beta_vals)),
        "sd": float(np.std(beta_vals)),
        "q05": float(np.percentile(beta_vals, 5)),
        "q25": float(np.percentile(beta_vals, 25)),
        "q50": float(np.percentile(beta_vals, 50)),
        "q75": float(np.percentile(beta_vals, 75)),
        "q95": float(np.percentile(beta_vals, 95)),
    }

    return {
        "beta_samples": beta_vals,
        "summary": summary,
        "n_attempted": n_attempted,
        "n_failed": n_failed,
        "n_singular": n_singular,
        "include_question_fe": include_question_fe,
        "include_question_re": include_question_re,
        "feature_type": feature_type,
        "is_binary": is_binary,
    }


def compute_bayesian_assurance_mixed_model_empirical(
    feature: np.ndarray,
    outcome: np.ndarray,
    *,
    student_ids: np.ndarray,
    question_ids: np.ndarray,
    rubric_ids: np.ndarray,
    n_students_target: int,
    alpha: float = 0.05,
    assurance_mode: Literal["fast", "faithful"] = "fast",
    beta_grid: Optional[List[float]] = None,
    n_power_sim: int = 200,
    seed_students: int = 123,
    seed_outcomes: int = 456,
    n_prior: int = 200,
    seed_prior: int = 789,
    n_power_sim_inner: int = 100,
    include_student_vc_glmm: bool = False,
    use_ranks: bool = False,
    standardize_feature: bool = True,
    standardize_outcome: bool = False,
    transform_bounded: Literal["none", "logit_clip"] = "none",
    drop_singular_prior: bool = True,
    winsorize_pct: Optional[float] = None,
    eps: float = 1e-8,
    log_progress: bool = True,
) -> Dict[str, Any]:
    """
    Bayesian assurance for the Step-3 mixed model using an empirical prior over beta.

    - fast: build an LRT power curve over |beta| and interpolate at |beta_samples|
    - faithful: for each |beta_sample|, estimate power by Monte Carlo (nested)
    """
    prior = compute_empirical_prior_mixed_model_beta(
        feature,
        outcome,
        student_ids=student_ids,
        question_ids=question_ids,
        rubric_ids=rubric_ids,
        n_boot=n_prior,
        seed=seed_prior,
        include_student_vc_glmm=include_student_vc_glmm,
        use_ranks=use_ranks,
        standardize_feature=standardize_feature,
        standardize_outcome=standardize_outcome,
        transform_bounded=transform_bounded,
        drop_singular=drop_singular_prior,
        winsorize_pct=winsorize_pct,
        eps=eps,
        log_progress=log_progress,
    )
    beta_samples = np.asarray(prior["beta_samples"], dtype=float)
    beta_abs = np.abs(beta_samples)

    if beta_grid is None:
        hi = float(np.percentile(beta_abs, 95))
        hi = hi * 1.25 if np.isfinite(hi) and hi > 0 else 0.5
        beta_grid = np.linspace(0.0, hi, 11).tolist()
        if beta_grid[0] == 0.0:
            beta_grid[0] = 0.0

    beta_grid_abs = sorted({abs(float(b)) for b in beta_grid})
    if len(beta_grid_abs) < 3:
        raise ValueError("beta_grid must have at least 3 distinct values (after abs())")

    power_curve = compute_power_curve_mixed_model_lrt(
        feature,
        outcome,
        student_ids=student_ids,
        question_ids=question_ids,
        rubric_ids=rubric_ids,
        beta_targets=beta_grid_abs,
        n_students_target=n_students_target,
        alpha=alpha,
        n_sim=n_power_sim,
        seed_students=seed_students,
        seed_outcomes=seed_outcomes,
        include_student_vc_glmm=include_student_vc_glmm,
        use_ranks=use_ranks,
        standardize_feature=standardize_feature,
        standardize_outcome=standardize_outcome,
        transform_bounded=transform_bounded,
        eps=eps,
        log_progress=log_progress,
    )

    # Enforce monotonicity in |beta| for interpolation (two-sided LRT).
    pc = power_curve.sort_values("beta_target_abs").copy()
    x = pc["beta_target_abs"].to_numpy(dtype=float)
    y = pc["power"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        raise RuntimeError("Power curve produced too few finite points for interpolation.")
    x = x[ok]
    y = y[ok]
    y_mono = np.maximum.accumulate(y)

    def power_hat(beta_abs_vals: np.ndarray) -> np.ndarray:
        b = np.asarray(beta_abs_vals, dtype=float)
        b = np.clip(b, x.min(), x.max())
        return np.interp(b, x, y_mono)

    if assurance_mode == "fast":
        power_at_prior = power_hat(beta_abs)
    elif assurance_mode == "faithful":
        # Nested Monte Carlo per prior draw (expensive, but faithful).
        # Use a reduced inner simulation count.
        power_list = []
        iterator = enumerate(beta_abs)
        if log_progress and len(beta_abs) > 1:
            import sys

            iterator = tqdm(
                iterator,
                total=len(beta_abs),
                desc="Assurance (mixed_model_lrt, faithful)",
                unit="prior-draw",
                dynamic_ncols=True,
                leave=True,
                disable=not log_progress,
                file=sys.stderr,
            )
        for k, b in iterator:
            row = compute_power_curve_mixed_model_lrt(
                feature,
                outcome,
                student_ids=student_ids,
                question_ids=question_ids,
                rubric_ids=rubric_ids,
                beta_targets=[float(b)],
                n_students_target=n_students_target,
                alpha=alpha,
                n_sim=n_power_sim_inner,
                seed_students=seed_students,
                seed_outcomes=seed_outcomes + 1000 + k,
                include_student_vc_glmm=include_student_vc_glmm,
                use_ranks=use_ranks,
                standardize_feature=standardize_feature,
                standardize_outcome=standardize_outcome,
                transform_bounded=transform_bounded,
                eps=eps,
                log_progress=False,
            )
            power_list.append(float(row["power"].iloc[0]))
        power_at_prior = np.asarray(power_list, dtype=float)
    else:
        raise ValueError("assurance_mode must be 'fast' or 'faithful'")

    assurance = float(np.nanmean(power_at_prior))
    assurance_ci = (float(np.nanpercentile(power_at_prior, 2.5)), float(np.nanpercentile(power_at_prior, 97.5)))
    assurance_sd = float(np.nanstd(power_at_prior))
    assurance_se = float(assurance_sd / np.sqrt(max(len(power_at_prior), 1)))

    return {
        "assurance": assurance,
        "assurance_ci": assurance_ci,
        "assurance_sd": assurance_sd,
        "assurance_se": assurance_se,
        "mode": assurance_mode,
        "alpha": float(alpha),
        "n_students_target": int(n_students_target),
        "power_curve": power_curve,
        "prior_summary": prior["summary"],
        "prior_n_attempted": prior["n_attempted"],
        "prior_n_failed": prior["n_failed"],
        "prior_n_singular": prior["n_singular"],
        "feature_type": prior["feature_type"],
        "include_question_fe": prior["include_question_fe"],
        "include_question_re": prior["include_question_re"],
        "is_binary": prior["is_binary"],
    }


def compute_power_curve_lme4_lrt_student_re(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    formula_null: str,
    formula_full: str,
    coef_name: str,
    beta_targets_abs: List[float],
    n_students_target: Optional[int] = None,
    alpha: float = 0.05,
    n_sim: int = 200,
    seed_students: int = 123,
    seed_outcomes: int = 456,
    is_glmm: bool = False,
    family: Literal["binomial", "gaussian"] = "gaussian",
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
    log_progress: bool = True,
    log_progress_sim: bool = False,
) -> pd.DataFrame:
    """
    Prospective power curve for an lme4 mixed-model LRT where the only random effect is (1|student_id).

    This is intended for Step 4 group comparisons: students are the design unit, and simulation uses
    student-block resampling (with relabeling) to preserve the within-student structure.

    The effect size axis is |beta_target| on the model scale:
      - LMM: additive change in outcome units
      - GLMM (logit): log-odds contrast (OR = exp(beta_target))
    log_progress_sim enables an inner per-simulation progress bar (useful in assurance runs).
    """
    if not beta_targets_abs:
        raise ValueError("beta_targets_abs must be non-empty")
    if n_sim < 10:
        raise ValueError("n_sim must be >= 10 for a useful power estimate")

    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    if "student_id" not in df.columns:
        raise ValueError("df must include 'student_id'")
    if outcome_col not in df.columns:
        raise ValueError(f"df missing outcome_col {outcome_col!r}")

    df_clean = df.copy()
    df_clean["y"] = pd.to_numeric(df_clean[outcome_col], errors="coerce")
    df_clean = df_clean.dropna(subset=["student_id", "y"])
    if len(df_clean) < 30:
        raise ValueError(f"Insufficient rows after cleaning for power curve: n={len(df_clean)}")

    # Ensure R-friendly categorical conversion (categories must be strings).
    df_r = df_clean.copy()
    df_r["student_id"] = pd.Categorical(df_r["student_id"].astype(str))
    for col in df_r.columns:
        if col == "y":
            continue
        if isinstance(df_r[col].dtype, pd.CategoricalDtype) or df_r[col].dtype == object:
            df_r[col] = pd.Categorical(df_r[col].astype(str))

    n_students_current = int(df_r["student_id"].nunique())
    n_students_target_val = int(n_students_target) if n_students_target is not None else n_students_current
    if n_students_target_val < 2:
        raise ValueError(f"n_students_target must be >=2, got {n_students_target_val}")

    with localconverter(converter):
        ro.globalenv["df_step4_power"] = rconversion.py2rpy(df_r)

    if "gc_power_curve_lrt_student_re" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_power_curve_lrt_student_re <- function(df, beta_targets_abs, n_students_target, alpha,
                                                          n_sim, seed_students, seed_outcomes,
                                                          formula_null, formula_full, coef_name,
                                                          is_glmm, family, glmer_nAGQ, optimizer, maxfun,
                                                          log_progress_sim) {
                  suppressPackageStartupMessages(library(lme4))
                  suppressPackageStartupMessages(library(stats))
                  suppressPackageStartupMessages(library(utils))

                  quiet_eval <- function(expr) {
                    suppressWarnings(withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    ))
                  }

                  df$student_id <- factor(df$student_id)
                  if (!("y" %in% names(df))) stop("df must contain y column")

                  ctrl <- lmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))
                  ctrl_glm <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))

                  if (!is_glmm) {
                    fit_null0 <- quiet_eval(tryCatch(lmer(formula_null, data=df, REML=FALSE, control=ctrl),
                                                     error=function(e) NULL))
                    fit_full_template <- quiet_eval(tryCatch(lmer(formula_full, data=df, REML=FALSE, control=ctrl),
                                                            error=function(e) NULL))
                  } else {
                    fam <- if (family == "binomial") binomial(link="logit") else gaussian()
                    fit_null0 <- quiet_eval(tryCatch(glmer(formula_null, data=df, family=fam, nAGQ=glmer_nAGQ, control=ctrl_glm),
                                                     error=function(e) NULL))
                    fit_full_template <- quiet_eval(tryCatch(glmer(formula_full, data=df, family=fam, nAGQ=glmer_nAGQ, control=ctrl_glm),
                                                            error=function(e) NULL))
                  }
                  if (is.null(fit_null0) || is.null(fit_full_template)) stop("Failed to fit initial null/full models")

                  theta0 <- getME(fit_null0, "theta")
                  sigma0 <- if (!is_glmm) sigma(fit_null0) else NA_real_
                  beta_null <- fixef(fit_null0)

                  students <- levels(df$student_id)

                  set.seed(seed_students)
                  draws <- replicate(n_sim, sample(students, size=n_students_target, replace=TRUE), simplify=FALSE)

                  build_sample_df <- function(draw_vec) {
                    parts <- vector("list", length(draw_vec))
                    for (i in seq_along(draw_vec)) {
                      sid <- draw_vec[[i]]
                      block <- df[df$student_id == sid, , drop=FALSE]
                      tag <- paste0("__rep", i)
                      block$student_id <- factor(paste0(as.character(block$student_id), tag))
                      parts[[i]] <- block
                    }
                    do.call(rbind, parts)
                  }

                  coef_names <- names(fixef(fit_full_template))
                  beta_vec_base <- fixef(fit_full_template)
                  beta_vec_base[] <- 0
                  for (nm in names(beta_null)) {
                    if (nm %in% coef_names) beta_vec_base[[nm]] <- beta_null[[nm]]
                  }
                  if (!(coef_name %in% coef_names)) stop(paste0("coef_name not found in full model: ", coef_name))

                  out_rows <- vector("list", length(beta_targets_abs))
                  for (j in seq_along(beta_targets_abs)) {
                    beta_abs <- beta_targets_abs[[j]]
                    beta_vec <- beta_vec_base
                    beta_vec[[coef_name]] <- beta_abs

                    set.seed(seed_outcomes + j)
                    pvals <- rep(NA_real_, n_sim)
                    n_failed <- 0L
                    n_singular <- 0L
                    n_reject <- 0L

                    pb <- NULL
                    if (isTRUE(log_progress_sim) && n_sim >= 5) {
                      pb <- txtProgressBar(min=0, max=n_sim, style=3, file=stderr())
                    }

                    for (b in seq_len(n_sim)) {
                      sim_df <- build_sample_df(draws[[b]])
                      if (!is_glmm) {
                        ysim <- tryCatch(
                          simulate(fit_full_template, nsim=1, newdata=sim_df,
                                   newparams=list(theta=theta0, beta=beta_vec, sigma=sigma0),
                                   re.form=NULL, allow.new.levels=TRUE)[[1]],
                          error=function(e) NA_real_
                        )
                        sim_df$y <- as.numeric(ysim)
                        fit_full <- quiet_eval(tryCatch(lmer(formula_full, data=sim_df, REML=FALSE, control=ctrl),
                                                       error=function(e) NULL))
                        fit_null <- quiet_eval(tryCatch(lmer(formula_null, data=sim_df, REML=FALSE, control=ctrl),
                                                       error=function(e) NULL))
                        if (is.null(fit_full) || is.null(fit_null)) {
                          n_failed <- n_failed + 1L
                          next
                        }
                        if (isSingular(fit_full, tol=1e-4)) n_singular <- n_singular + 1L
                        a <- quiet_eval(tryCatch(anova(fit_null, fit_full), error=function(e) NULL))
                        if (!is.null(a) && "Pr(>Chisq)" %in% colnames(a)) {
                          p <- a[["Pr(>Chisq)"]][nrow(a)]
                          pvals[[b]] <- p
                          if (is.finite(p) && p < alpha) n_reject <- n_reject + 1L
                        } else {
                          n_failed <- n_failed + 1L
                        }
                      } else {
                        ysim <- tryCatch(
                          simulate(fit_full_template, nsim=1, newdata=sim_df,
                                   newparams=list(theta=theta0, beta=beta_vec),
                                   re.form=NULL, allow.new.levels=TRUE)[[1]],
                          error=function(e) NA_real_
                        )
                        sim_df$y <- as.integer(ysim)
                        fam <- if (family == "binomial") binomial(link="logit") else gaussian()
                        fit_full <- quiet_eval(tryCatch(glmer(formula_full, data=sim_df, family=fam, nAGQ=glmer_nAGQ, control=ctrl_glm),
                                                       error=function(e) NULL))
                        fit_null <- quiet_eval(tryCatch(glmer(formula_null, data=sim_df, family=fam, nAGQ=glmer_nAGQ, control=ctrl_glm),
                                                       error=function(e) NULL))
                        if (is.null(fit_full) || is.null(fit_null)) {
                          n_failed <- n_failed + 1L
                          next
                        }
                        if (isSingular(fit_full, tol=1e-4)) n_singular <- n_singular + 1L
                        a <- quiet_eval(tryCatch(anova(fit_null, fit_full, test="Chisq"), error=function(e) NULL))
                        if (!is.null(a) && "Pr(>Chisq)" %in% colnames(a)) {
                          p <- a[["Pr(>Chisq)"]][nrow(a)]
                          pvals[[b]] <- p
                          if (is.finite(p) && p < alpha) n_reject <- n_reject + 1L
                        } else {
                          n_failed <- n_failed + 1L
                        }
                      }
                      if (!is.null(pb)) setTxtProgressBar(pb, b)
                    }
                    if (!is.null(pb)) close(pb)

                    ok <- is.finite(pvals)
                    n_success <- sum(ok)
                    power <- if (n_success > 0) mean(pvals[ok] < alpha) else NA_real_
                    out_rows[[j]] <- data.frame(
                      beta_target_abs=beta_abs,
                      power=power,
                      n_success=n_success,
                      n_reject=n_reject,
                      n_failed=n_failed,
                      n_singular=n_singular,
                      stringsAsFactors=FALSE
                    )
                  }

                  do.call(rbind, out_rows)
                }
                """
            )
        )

    beta_abs = [abs(float(b)) for b in beta_targets_abs]

    if log_progress and len(beta_abs) > 1:
        import sys

        rows: list[pd.DataFrame] = []
        iterator = tqdm(
            enumerate(beta_abs),
            total=len(beta_abs),
            desc="Power curve (LRT, student-RE)",
            unit="|beta|",
            dynamic_ncols=True,
            leave=True,
            disable=not log_progress,
            file=sys.stderr,
        )
        for idx, beta_abs_one in iterator:
            ro.globalenv["beta_targets_abs_one"] = ro.FloatVector([float(beta_abs_one)])
            res_r_one = ro.globalenv["gc_power_curve_lrt_student_re"](
                ro.globalenv["df_step4_power"],
                ro.globalenv["beta_targets_abs_one"],
                int(n_students_target_val),
                float(alpha),
                int(n_sim),
                int(seed_students),
                int(seed_outcomes + idx),
                str(formula_null),
                str(formula_full),
                str(coef_name),
                bool(is_glmm),
                str(family),
                int(glmer_nAGQ),
                str(optimizer),
                int(maxfun),
                bool(log_progress_sim),
            )
            with localconverter(converter):
                rows.append(rconversion.rpy2py(res_r_one))
        res_df = pd.concat(rows, ignore_index=True)
    else:
        ro.globalenv["beta_targets_abs_vec"] = ro.FloatVector(beta_abs)
        res_r = ro.globalenv["gc_power_curve_lrt_student_re"](
            ro.globalenv["df_step4_power"],
            ro.globalenv["beta_targets_abs_vec"],
            int(n_students_target_val),
            float(alpha),
            int(n_sim),
            int(seed_students),
            int(seed_outcomes),
            str(formula_null),
            str(formula_full),
            str(coef_name),
            bool(is_glmm),
            str(family),
            int(glmer_nAGQ),
            str(optimizer),
            int(maxfun),
            bool(log_progress_sim),
        )
        with localconverter(converter):
            res_df = rconversion.rpy2py(res_r)

    res_df["beta_target_abs"] = res_df["beta_target_abs"].astype(float)
    res_df["power"] = pd.to_numeric(res_df["power"], errors="coerce")
    res_df["n_success"] = pd.to_numeric(res_df["n_success"], errors="coerce").astype("Int64")
    res_df["n_reject"] = pd.to_numeric(res_df["n_reject"], errors="coerce").astype("Int64")
    res_df["n_failed"] = pd.to_numeric(res_df["n_failed"], errors="coerce").astype("Int64")
    res_df["n_singular"] = pd.to_numeric(res_df["n_singular"], errors="coerce").astype("Int64")

    def _wilson_ci(k: float, n: float, alpha_val: float) -> tuple[float, float]:
        if not (np.isfinite(k) and np.isfinite(n)) or n <= 0:
            return (np.nan, np.nan)
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha_val / 2))
        phat = float(k / n)
        denom = 1 + (z * z) / n
        center = (phat + (z * z) / (2 * n)) / denom
        half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
        return (max(0.0, center - half), min(1.0, center + half))

    ci_low = []
    ci_high = []
    for k, n in zip(res_df["n_reject"].to_numpy(dtype=float), res_df["n_success"].to_numpy(dtype=float)):
        lo, hi = _wilson_ci(k, n, alpha)
        ci_low.append(lo)
        ci_high.append(hi)
    res_df["power_ci_low"] = ci_low
    res_df["power_ci_high"] = ci_high

    res_df["alpha"] = float(alpha)
    res_df["n_students_current"] = n_students_current
    res_df["n_students_target"] = n_students_target_val
    res_df["method"] = "mixed_model_lrt"
    return res_df.sort_values("beta_target_abs").reset_index(drop=True)


def compute_empirical_prior_lme4_beta_student_re(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    formula_full: str,
    coef_name: str,
    n_boot: int = 200,
    seed: int = 123,
    is_glmm: bool = False,
    family: Literal["binomial", "gaussian"] = "gaussian",
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
    winsorize_pct: Optional[float] = None,
    log_progress: bool = True,
) -> Dict[str, Any]:
    """
    Empirical prior for a fixed-effect beta under an lme4 model with (1|student_id).

    Uses a student-block bootstrap (resample students with replacement, relabel duplicates as independent).
    """
    if n_boot < 10:
        raise ValueError("n_boot must be >= 10 for a useful empirical prior")

    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    if "student_id" not in df.columns:
        raise ValueError("df must include 'student_id'")
    if outcome_col not in df.columns:
        raise ValueError(f"df missing outcome_col {outcome_col!r}")

    df_clean = df.copy()
    df_clean["y"] = pd.to_numeric(df_clean[outcome_col], errors="coerce")
    df_clean = df_clean.dropna(subset=["student_id", "y"])
    if len(df_clean) < 30:
        raise ValueError(f"Insufficient rows after cleaning for empirical prior: n={len(df_clean)}")

    df_r = df_clean.copy()
    df_r["student_id"] = pd.Categorical(df_r["student_id"].astype(str))
    for col in df_r.columns:
        if col == "y":
            continue
        if isinstance(df_r[col].dtype, pd.CategoricalDtype) or df_r[col].dtype == object:
            df_r[col] = pd.Categorical(df_r[col].astype(str))

    with localconverter(converter):
        ro.globalenv["df_step4_prior"] = rconversion.py2rpy(df_r)

    if "gc_empirical_prior_beta_student_re" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_empirical_prior_beta_student_re <- function(df, n_boot, seed, formula_full, coef_name,
                                                               is_glmm, family, glmer_nAGQ, optimizer, maxfun) {
                  suppressPackageStartupMessages(library(lme4))
                  suppressPackageStartupMessages(library(stats))

                  quiet_eval <- function(expr) {
                    suppressWarnings(withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    ))
                  }

                  df$student_id <- factor(df$student_id)
                  if (!("y" %in% names(df))) stop("df must contain y column")

                  ctrl <- lmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))
                  ctrl_glm <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))

                  students <- levels(df$student_id)
                  n_students <- length(students)

                  set.seed(seed)
                  draws <- replicate(n_boot, sample(students, size=n_students, replace=TRUE), simplify=FALSE)

                  build_sample_df <- function(draw_vec) {
                    parts <- vector("list", length(draw_vec))
                    for (i in seq_along(draw_vec)) {
                      sid <- draw_vec[[i]]
                      block <- df[df$student_id == sid, , drop=FALSE]
                      tag <- paste0("__rep", i)
                      block$student_id <- factor(paste0(as.character(block$student_id), tag))
                      parts[[i]] <- block
                    }
                    do.call(rbind, parts)
                  }

                  betas <- rep(NA_real_, n_boot)
                  n_failed <- 0L
                  n_singular <- 0L

                  for (b in seq_len(n_boot)) {
                    boot_df <- build_sample_df(draws[[b]])
                    if (!is_glmm) {
                      fit <- quiet_eval(tryCatch(lmer(formula_full, data=boot_df, REML=FALSE, control=ctrl),
                                                 error=function(e) NULL))
                    } else {
                      fam <- if (family == "binomial") binomial(link="logit") else gaussian()
                      fit <- quiet_eval(tryCatch(glmer(formula_full, data=boot_df, family=fam, nAGQ=glmer_nAGQ, control=ctrl_glm),
                                                 error=function(e) NULL))
                    }
                    if (is.null(fit)) {
                      n_failed <- n_failed + 1L
                      next
                    }
                    if (isSingular(fit, tol=1e-4)) n_singular <- n_singular + 1L
                    fx <- fixef(fit)
                    if (!(coef_name %in% names(fx))) {
                      n_failed <- n_failed + 1L
                      next
                    }
                    betas[[b]] <- as.numeric(fx[[coef_name]])
                  }

                  list(
                    betas=betas,
                    n_attempted=n_boot,
                    n_failed=n_failed,
                    n_singular=n_singular
                  )
                }
                """
            )
        )

    res_r = ro.globalenv["gc_empirical_prior_beta_student_re"](
        ro.globalenv["df_step4_prior"],
        int(n_boot),
        int(seed),
        str(formula_full),
        str(coef_name),
        bool(is_glmm),
        str(family),
        int(glmer_nAGQ),
        str(optimizer),
        int(maxfun),
    )
    betas = np.asarray(res_r.rx2("betas"), dtype=float)
    n_attempted = int(res_r.rx2("n_attempted")[0])
    n_failed = int(res_r.rx2("n_failed")[0])
    n_singular = int(res_r.rx2("n_singular")[0])

    betas = betas[np.isfinite(betas)]
    if betas.size == 0:
        raise ValueError("No finite beta samples available for empirical prior.")

    if winsorize_pct is not None:
        pct = float(winsorize_pct)
        if not (0.0 < pct < 50.0):
            raise ValueError("winsorize_pct must be in (0,50)")
        lo = np.percentile(betas, pct)
        hi = np.percentile(betas, 100.0 - pct)
        betas = np.clip(betas, lo, hi)

    return {
        "beta_samples": betas,
        "n_attempted": n_attempted,
        "n_failed": n_failed,
        "n_singular": n_singular,
        "coef_name": coef_name,
        "formula_full": formula_full,
        "is_glmm": bool(is_glmm),
        "family": family,
    }


def compute_bayesian_assurance_lme4_lrt_empirical_student_re(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    formula_null: str,
    formula_full: str,
    coef_name: str,
    n_students_target: int,
    alpha: float = 0.05,
    assurance_mode: Literal["fast", "faithful"] = "fast",
    beta_grid_abs: Optional[List[float]] = None,
    power_curve: Optional[pd.DataFrame] = None,
    n_power_sim: int = 200,
    seed_students: int = 123,
    seed_outcomes: int = 456,
    n_prior: int = 200,
    seed_prior: int = 789,
    n_power_sim_inner: int = 100,
    is_glmm: bool = False,
    family: Literal["binomial", "gaussian"] = "gaussian",
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
    winsorize_pct: Optional[float] = None,
    log_progress: bool = True,
) -> Dict[str, Any]:
    """
    Bayesian assurance for an lme4 LRT with (1|student_id), using an empirical prior over beta.
    """
    prior = compute_empirical_prior_lme4_beta_student_re(
        df,
        outcome_col=outcome_col,
        formula_full=formula_full,
        coef_name=coef_name,
        n_boot=n_prior,
        seed=seed_prior,
        is_glmm=is_glmm,
        family=family,
        glmer_nAGQ=glmer_nAGQ,
        optimizer=optimizer,
        maxfun=maxfun,
        winsorize_pct=winsorize_pct,
        log_progress=log_progress,
    )
    beta_samples = np.asarray(prior["beta_samples"], dtype=float)
    beta_abs = np.abs(beta_samples)

    if beta_grid_abs is None:
        hi = float(np.percentile(beta_abs, 95))
        hi = hi * 1.25 if np.isfinite(hi) and hi > 0 else 0.5
        beta_grid_abs = np.linspace(0.0, hi, 11).tolist()

    beta_grid_abs = sorted({abs(float(b)) for b in beta_grid_abs})
    if len(beta_grid_abs) < 3:
        raise ValueError("beta_grid_abs must have at least 3 distinct values (after abs())")

    if power_curve is None:
        power_curve = compute_power_curve_lme4_lrt_student_re(
            df,
            outcome_col=outcome_col,
            formula_null=formula_null,
            formula_full=formula_full,
            coef_name=coef_name,
            beta_targets_abs=beta_grid_abs,
            n_students_target=n_students_target,
            alpha=alpha,
            n_sim=n_power_sim,
            seed_students=seed_students,
            seed_outcomes=seed_outcomes,
            is_glmm=is_glmm,
            family=family,
            glmer_nAGQ=glmer_nAGQ,
            optimizer=optimizer,
            maxfun=maxfun,
            log_progress=log_progress,
        )
    else:
        power_curve = power_curve.copy()

    pc = power_curve.sort_values("beta_target_abs").copy()
    x = pc["beta_target_abs"].to_numpy(dtype=float)
    y = pc["power"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        raise RuntimeError("Power curve produced too few finite points for interpolation.")
    x = x[ok]
    y = y[ok]
    y_mono = np.maximum.accumulate(y)

    def power_hat(beta_abs_vals: np.ndarray) -> np.ndarray:
        b = np.asarray(beta_abs_vals, dtype=float)
        b = np.clip(b, x.min(), x.max())
        return np.interp(b, x, y_mono)

    if assurance_mode == "fast":
        power_at_prior = power_hat(beta_abs)
    elif assurance_mode == "faithful":
        power_list: list[float] = []
        iterator = enumerate(beta_abs)
        if log_progress and len(beta_abs) > 1:
            import sys

            iterator = tqdm(
                iterator,
                total=len(beta_abs),
                desc="Assurance (LRT, student-RE, faithful)",
                unit="prior-draw",
                dynamic_ncols=True,
                leave=True,
                disable=not log_progress,
                file=sys.stderr,
            )
        for k, b in iterator:
            row = compute_power_curve_lme4_lrt_student_re(
                df,
                outcome_col=outcome_col,
                formula_null=formula_null,
                formula_full=formula_full,
                coef_name=coef_name,
                beta_targets_abs=[float(b)],
                n_students_target=n_students_target,
                alpha=alpha,
                n_sim=n_power_sim_inner,
                seed_students=seed_students,
                seed_outcomes=seed_outcomes + 1000 + k,
                is_glmm=is_glmm,
                family=family,
                glmer_nAGQ=glmer_nAGQ,
                optimizer=optimizer,
                maxfun=maxfun,
                log_progress=False,
                log_progress_sim=False,
            )
            power_list.append(float(row["power"].iloc[0]))
        power_at_prior = np.asarray(power_list, dtype=float)
    else:
        raise ValueError("assurance_mode must be 'fast' or 'faithful'")

    assurance = float(np.nanmean(power_at_prior))
    assurance_ci = (float(np.nanpercentile(power_at_prior, 2.5)), float(np.nanpercentile(power_at_prior, 97.5)))
    assurance_sd = float(np.nanstd(power_at_prior))
    assurance_se = float(assurance_sd / np.sqrt(max(len(power_at_prior), 1)))

    return {
        "assurance": assurance,
        "assurance_ci": assurance_ci,
        "assurance_sd": assurance_sd,
        "assurance_se": assurance_se,
        "mode": assurance_mode,
        "alpha": float(alpha),
        "n_students_target": int(n_students_target),
        "beta_grid_abs": beta_grid_abs,
        "n_power_sim": int(n_power_sim),
        "n_power_sim_inner": int(n_power_sim_inner),
        "power_curve": power_curve,
        "prior_summary": {
            "mean": float(np.mean(beta_samples)),
            "sd": float(np.std(beta_samples)),
            "q05": float(np.percentile(beta_samples, 5)),
            "q50": float(np.percentile(beta_samples, 50)),
            "q95": float(np.percentile(beta_samples, 95)),
        },
        "prior_n_attempted": int(prior["n_attempted"]),
        "prior_n_failed": int(prior["n_failed"]),
        "prior_n_singular": int(prior["n_singular"]),
        "coef_name": coef_name,
        "is_glmm": bool(is_glmm),
        "family": family,
    }


def _build_calibration_conf_grid(
    confidence: np.ndarray,
    *,
    n_grid: int = 50,
    clip_min: float = 0.001,
    clip_max: float = 0.999,
) -> np.ndarray:
    conf = np.asarray(confidence, dtype=float)
    conf = conf[np.isfinite(conf)]
    if conf.size == 0:
        return np.linspace(clip_min, clip_max, n_grid)
    conf = np.clip(conf, clip_min, clip_max)
    lo = float(np.quantile(conf, 0.05))
    hi = float(np.quantile(conf, 0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(conf))
        hi = float(np.max(conf))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = clip_min, clip_max
    if hi - lo < 1e-6:
        hi = min(clip_max, lo + 1e-3)
        lo = max(clip_min, hi - 1e-3)
    return np.linspace(lo, hi, n_grid)


def _fit_calibration_regression_nonlinear(
    df: pd.DataFrame,
    *,
    formula_reduced: str,
    formula_full: str,
    conf_grid_logit_c: np.ndarray,
    feature_vals: np.ndarray,
    glmer_nAGQ: int,
    optimizer: str,
    maxfun: int,
) -> Dict[str, Any]:
    ok, payload = _try_import_lme4()
    if not ok:
        raise ImportError(f"rpy2/lme4 not available: {payload}")

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    df_r = df.copy()
    for col in df_r.columns:
        if pd.api.types.is_categorical_dtype(df_r[col]) or df_r[col].dtype == object:
            df_r[col] = df_r[col].astype(str)

    with localconverter(converter):
        ro.globalenv["df_calib_tmp"] = rconversion.py2rpy(df_r)

    if "gc_calib_nonlinear_fit" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_calib_nonlinear_fit <- function(df, formula_reduced, formula_full,
                                                  conf_grid, feature_vals,
                                                  glmer_nAGQ, optimizer, maxfun) {
                  suppressPackageStartupMessages(library(lme4))
                  suppressPackageStartupMessages(library(stats))
                  suppressPackageStartupMessages(library(splines))

                  df$student_id <- factor(df$student_id)
                  if ("rubric_id" %in% names(df)) df$rubric_id <- factor(df$rubric_id)

                  quiet_eval <- function(expr) {
                    withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    )
                  }

                  ctrl <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))
                  fit_null <- quiet_eval(glmer(as.formula(formula_reduced), data=df,
                                               family=binomial(link="logit"),
                                               nAGQ=glmer_nAGQ, control=ctrl))
                  fit_full <- quiet_eval(glmer(as.formula(formula_full), data=df,
                                               family=binomial(link="logit"),
                                               nAGQ=glmer_nAGQ, control=ctrl))

                  a <- quiet_eval(tryCatch(anova(fit_null, fit_full, test="Chisq"), error=function(e) NULL))
                  lrt_stat <- NA_real_
                  lrt_df <- NA_real_
                  lrt_p <- NA_real_
                  if (!is.null(a)) {
                    if ("Chisq" %in% colnames(a)) lrt_stat <- a[nrow(a), "Chisq"]
                    if ("Df" %in% colnames(a)) lrt_df <- a[nrow(a), "Df"] - a[nrow(a)-1, "Df"]
                    if ("Pr(>Chisq)" %in% colnames(a)) lrt_p <- a[nrow(a), "Pr(>Chisq)"]
                  }

                  singular <- FALSE
                  singular <- tryCatch(isSingular(fit_full, tol=1e-4), error=function(e) FALSE)

                  msgs <- character(0)
                  msgs <- tryCatch({
                    oi <- fit_full@optinfo
                    conv <- oi$conv$lme4$messages
                    if (!is.null(conv)) conv else character(0)
                  }, error=function(e) character(0))

                  sid <- levels(df$student_id)[1]
                  rid <- if ("rubric_id" %in% names(df)) levels(df$rubric_id)[1] else NA

                  preds <- list()
                  for (fv in feature_vals) {
                    newdata <- data.frame(
                      logit_conf_c=conf_grid,
                      feature_z=rep(fv, length(conf_grid)),
                      student_id=rep(sid, length(conf_grid))
                    )
                    if ("rubric_id" %in% names(df)) {
                      newdata$rubric_id <- rep(rid, length(conf_grid))
                    }
                    p <- predict(fit_full, newdata=newdata, type="response",
                                 re.form=NA, allow.new.levels=TRUE)
                    preds[[length(preds) + 1]] <- as.numeric(p)
                  }

                  list(
                    lrt_stat=lrt_stat,
                    lrt_df=lrt_df,
                    lrt_p=lrt_p,
                    singular=singular,
                    conv_messages=msgs,
                    preds=preds
                  )
                }
                """
            )
        )

    res = ro.globalenv["gc_calib_nonlinear_fit"](
        ro.globalenv["df_calib_tmp"],
        str(formula_reduced),
        str(formula_full),
        ro.FloatVector(conf_grid_logit_c.tolist()),
        ro.FloatVector(np.asarray(feature_vals, dtype=float).tolist()),
        int(glmer_nAGQ),
        str(_normalize_lme4_optimizer(optimizer)),
        int(maxfun),
    )

    def _get_float(key: str) -> float:
        try:
            v = res.rx2(key)
            if v is None or not len(v):
                return float("nan")
            val = float(v[0])
            if is_na_sentinel(val):
                return float("nan")
            return val
        except Exception:
            return float("nan")

    warnings_list: List[str] = []
    try:
        msgs = list(res.rx2("conv_messages"))
        for m in msgs:
            ms = str(m)
            if ms:
                warnings_list.append(ms)
    except Exception:
        pass

    preds: List[np.ndarray] = []
    try:
        preds_r = res.rx2("preds")
        for i in range(len(preds_r)):
            preds.append(np.asarray(list(preds_r[i]), dtype=float))
    except Exception:
        preds = []

    return {
        "lrt_stat": _get_float("lrt_stat"),
        "lrt_df": _get_float("lrt_df"),
        "lrt_p": _get_float("lrt_p"),
        "singular": bool(res.rx2("singular")[0]) if "singular" in list(res.names) else False,
        "warnings": warnings_list,
        "preds": preds,
    }


def fit_calibration_curve_glmm(
    is_correct: np.ndarray,
    confidence: np.ndarray,
    student_ids: np.ndarray,
    rubric_ids: Optional[np.ndarray] = None,
    *,
    spline_df: int = 4,
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
    n_grid: int = 50,
    logit_conf_mean: Optional[float] = None,
    conf_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit a GLMM calibration curve: is_correct ~ bs(logit(conf), df=k) + RE.

    Returns curve predictions on a confidence grid (population-level, re.form=NA).
    """
    n = len(is_correct)
    if len(confidence) != n or len(student_ids) != n:
        raise ValueError("All input arrays must have the same length")

    valid = np.isfinite(is_correct.astype(float)) & np.isfinite(confidence.astype(float))
    if rubric_ids is not None:
        if len(rubric_ids) != n:
            raise ValueError("rubric_ids must have same length as other arrays")
        valid = valid & ~pd.isna(rubric_ids)

    if valid.sum() < 10:
        return {"curve": None, "warnings": ["Insufficient valid observations (<10)"]}

    y = np.asarray(is_correct, dtype=float)[valid]
    conf = np.asarray(confidence, dtype=float)[valid]
    students = np.asarray(student_ids, dtype=str)[valid]
    rubrics = np.asarray(rubric_ids, dtype=str)[valid] if rubric_ids is not None else None

    conf_clipped = np.clip(conf, 0.001, 0.999)
    logit_conf = np.log(conf_clipped / (1 - conf_clipped))
    logit_mean = float(np.mean(logit_conf)) if logit_conf_mean is None else float(logit_conf_mean)
    logit_conf_c = logit_conf - logit_mean

    df = pd.DataFrame({
        "y": y.astype(int),
        "logit_conf_c": logit_conf_c,
        "student_id": students,
    })
    use_student_re = len(np.unique(students)) >= 2 and len(np.unique(students)) < len(df)
    use_rubric_re = False
    if rubrics is not None and len(np.unique(rubrics)) >= 2 and len(np.unique(rubrics)) < len(df):
        df["rubric_id"] = rubrics
        use_rubric_re = True

    max_df = int(min(6, max(3, np.floor(len(df) / 10))))
    if spline_df > max_df:
        spline_df = max_df

    basis = f"bs(logit_conf_c, df={spline_df}, degree=3, intercept=FALSE)"
    re_terms = []
    if use_student_re:
        re_terms.append("(1|student_id)")
    if use_rubric_re:
        re_terms.append("(1|rubric_id)")
    re_part = " + ".join(re_terms)
    re_part = f" + {re_part}" if re_part else ""
    formula_full = f"y ~ {basis}{re_part}"

    if conf_grid is None:
        conf_grid = _build_calibration_conf_grid(conf, n_grid=n_grid)
    else:
        conf_grid = np.asarray(conf_grid, dtype=float)
        conf_grid = np.clip(conf_grid, 0.001, 0.999)
    logit_conf_grid = np.log(conf_grid / (1 - conf_grid)) - logit_mean

    ok, payload = _try_import_lme4()
    if not ok:
        return {"curve": None, "warnings": [f"rpy2/lme4 not available: {payload}"]}

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    df_r = df.copy()
    for col in df_r.columns:
        if pd.api.types.is_categorical_dtype(df_r[col]) or df_r[col].dtype == object:
            df_r[col] = df_r[col].astype(str)

    with localconverter(converter):
        ro.globalenv["df_calib_curve_tmp"] = rconversion.py2rpy(df_r)

    if "gc_calib_curve_fit" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_calib_curve_fit <- function(df, formula_full, conf_grid,
                                              glmer_nAGQ, optimizer, maxfun) {
                  suppressPackageStartupMessages(library(lme4))
                  suppressPackageStartupMessages(library(stats))
                  suppressPackageStartupMessages(library(splines))

                  df$student_id <- factor(df$student_id)
                  if ("rubric_id" %in% names(df)) df$rubric_id <- factor(df$rubric_id)

                  quiet_eval <- function(expr) {
                    withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    )
                  }

                  ctrl <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))
                  fit_full <- quiet_eval(glmer(as.formula(formula_full), data=df,
                                               family=binomial(link="logit"),
                                               nAGQ=glmer_nAGQ, control=ctrl))

                  singular <- FALSE
                  singular <- tryCatch(isSingular(fit_full, tol=1e-4), error=function(e) FALSE)

                  msgs <- character(0)
                  msgs <- tryCatch({
                    oi <- fit_full@optinfo
                    conv <- oi$conv$lme4$messages
                    if (!is.null(conv)) conv else character(0)
                  }, error=function(e) character(0))

                  sid <- levels(df$student_id)[1]
                  rid <- if ("rubric_id" %in% names(df)) levels(df$rubric_id)[1] else NA

                  newdata <- data.frame(
                    logit_conf_c=conf_grid,
                    student_id=rep(sid, length(conf_grid))
                  )
                  if ("rubric_id" %in% names(df)) {
                    newdata$rubric_id <- rep(rid, length(conf_grid))
                  }
                  preds <- predict(fit_full, newdata=newdata, type="response",
                                   re.form=NA, allow.new.levels=TRUE)
                  list(
                    preds=as.numeric(preds),
                    singular=singular,
                    conv_messages=msgs
                  )
                }
                """
            )
        )

    res = ro.globalenv["gc_calib_curve_fit"](
        ro.globalenv["df_calib_curve_tmp"],
        str(formula_full),
        ro.FloatVector(logit_conf_grid.tolist()),
        int(glmer_nAGQ),
        str(_normalize_lme4_optimizer(optimizer)),
        int(maxfun),
    )

    warnings_list: List[str] = []
    try:
        msgs = list(res.rx2("conv_messages"))
        for m in msgs:
            ms = str(m)
            if ms:
                warnings_list.append(ms)
    except Exception:
        pass

    preds = None
    try:
        preds = np.asarray(list(res.rx2("preds")), dtype=float)
    except Exception:
        preds = None

    return {
        "curve": {
            "conf_grid": conf_grid,
            "pred": preds,
        },
        "logit_conf_mean": logit_mean,
        "spline_df": spline_df,
        "singular": bool(res.rx2("singular")[0]) if "singular" in list(res.names) else False,
        "warnings": warnings_list,
    }


def analyze_feature_moderated_calibration(
    is_correct: np.ndarray,
    confidence: np.ndarray,
    feature: np.ndarray,
    student_ids: np.ndarray,
    rubric_ids: Optional[np.ndarray] = None,
    question_ids: Optional[np.ndarray] = None,
    *,
    alpha: float = 0.05,
    glmer_nAGQ: int = 1,
    optimizer: str = "bobyqa",
    maxfun: int = 100000,
    mode: Literal["linear", "nonlinear"] = "linear",
    spline_df: int = 4,
    bootstrap_n: int = 0,
    ci_method: Literal["percentile", "bca"] = "percentile",
    jackknife_max: int = 200,
    n_grid: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Test whether a feature moderates calibration via GLMM.

    Linear mode (default):
        Reduced: is_correct ~ logit_conf_c + feature_z + (1|student_id) [+ (1|rubric_id)]
        Full:    is_correct ~ logit_conf_c + feature_z + logit_conf_c:feature_z + (1|student_id) [+ (1|rubric_id)]

    Nonlinear mode (optional):
        Reduced: is_correct ~ bs(logit_conf_c, df=k) + feature_z + (1|student_id) [+ (1|rubric_id)]
        Full:    is_correct ~ bs(logit_conf_c, df=k) * feature_z + (1|student_id) [+ (1|rubric_id)]
        LRT tests the interaction spline block (df = k).

    Args:
        is_correct: Binary outcome array (0/1).
        confidence: Raw confidence values (logit-transformed, centered).
        feature: Feature values (standardized).
        student_ids: Student cluster IDs for random effects.
        rubric_ids: Optional rubric cluster IDs for crossed random effects.
        question_ids: Optional question IDs (currently unused, reserved for future).
        alpha: Significance level for inference.
        glmer_nAGQ: glmer quadrature points (1 = Laplace approximation).
        optimizer: lme4 optimizer name.
        maxfun: Maximum optimizer iterations.
        mode: "linear" or "nonlinear".
        spline_df: Degrees of freedom for bs() in nonlinear mode.
        bootstrap_n: Bootstrap samples for AME CI (nonlinear only).
        ci_method: "percentile" or "bca" for AME CI (nonlinear only).
        jackknife_max: Cap on jackknife units for BCa (nonlinear only).
        n_grid: Confidence grid points for curve summaries (nonlinear only).
        rng: Optional RNG for bootstrap.

    Returns:
        Dict with linear coefficients (when mode="linear") or LRT + curve summary
        (when mode="nonlinear"), plus metadata and warnings.
    """
    # Input validation
    n = len(is_correct)
    if len(confidence) != n or len(feature) != n or len(student_ids) != n:
        raise ValueError("All input arrays must have the same length")

    # Filter out NaN/inf values
    valid = (
        np.isfinite(is_correct.astype(float))
        & np.isfinite(confidence.astype(float))
        & np.isfinite(feature.astype(float))
    )
    if rubric_ids is not None:
        if len(rubric_ids) != n:
            raise ValueError("rubric_ids must have same length as other arrays")
        # rubric_ids can be string/object, just check for not-null
        valid = valid & ~pd.isna(rubric_ids)

    if valid.sum() < 10:
        return _empty_calibration_regression_result(
            "Insufficient valid observations (<10)"
        )

    is_correct = np.asarray(is_correct, dtype=float)[valid]
    confidence = np.asarray(confidence, dtype=float)[valid]
    feature = np.asarray(feature, dtype=float)[valid]
    student_ids = np.asarray(student_ids, dtype=str)[valid]
    if rubric_ids is not None:
        rubric_ids = np.asarray(rubric_ids, dtype=str)[valid]

    n_rows = len(is_correct)
    n_students = len(np.unique(student_ids))
    n_rubrics = len(np.unique(rubric_ids)) if rubric_ids is not None else 0

    # Standardize feature (z-score)
    feature_mean = float(np.mean(feature))
    feature_sd = float(np.std(feature))
    if feature_sd < 1e-10:
        return _empty_calibration_regression_result(
            "Feature has zero variance"
        )
    feature_z = (feature - feature_mean) / feature_sd

    # Logit-transform and center confidence
    conf_clipped = np.clip(confidence, 0.001, 0.999)
    logit_conf = np.log(conf_clipped / (1 - conf_clipped))
    logit_conf_mean = float(np.mean(logit_conf))
    logit_conf_c = logit_conf - logit_conf_mean

    # Build data frame for R
    df = pd.DataFrame({
        "y": is_correct.astype(int),
        "logit_conf_c": logit_conf_c,
        "feature_z": feature_z,
        "student_id": student_ids,
    })
    if rubric_ids is not None and n_rubrics >= 2:
        df["rubric_id"] = rubric_ids
        use_rubric_re = True
    else:
        use_rubric_re = False

    # Define random effects
    re_part = "(1|student_id)"
    if use_rubric_re:
        re_part += " + (1|rubric_id)"

    if mode == "linear":
        formula_reduced = f"y ~ logit_conf_c + feature_z + {re_part}"
        formula_full = f"y ~ logit_conf_c + feature_z + logit_conf_c:feature_z + {re_part}"

        # Fit models using lme4
        try:
            lrt_result = fit_lme4_lrt_from_formula(
                df,
                formula_null=formula_reduced,
                formula_full=formula_full,
                is_glmm=True,
                family="binomial",
                coef_name="logit_conf_c:feature_z",
                glmer_nAGQ=glmer_nAGQ,
                optimizer=optimizer,
                maxfun=maxfun,
            )
        except Exception as e:
            return _empty_calibration_regression_result(f"lme4 fitting failed: {e}")

        # Extract interaction coefficient (primary result)
        beta_interaction = float(lrt_result.get("beta", np.nan))
        se_interaction = float(lrt_result.get("se", np.nan))
        p_interaction = float(lrt_result.get("p_value_lrt", np.nan))
        chisq = float(lrt_result.get("chisq", np.nan))
        df_diff = float(lrt_result.get("df_diff", np.nan))

        # Also extract feature main effect from the full model
        # We need to fit the full model separately to get Wald p-value for feature_z
        beta_main, se_main, p_main = _extract_coefficient_from_glmm(
            df, formula_full, "feature_z", glmer_nAGQ, optimizer, maxfun
        )

        return {
            "mode": "linear",
            "beta_main": beta_main,
            "se_main": se_main,
            "p_main": p_main,
            "beta_interaction": beta_interaction,
            "se_interaction": se_interaction,
            "p_interaction": p_interaction,
            "chisq": chisq,
            "df_diff": df_diff,
            "converged": bool(lrt_result.get("converged", False)),
            "singular": bool(lrt_result.get("singular", False)),
            "warnings": lrt_result.get("warnings", []),
            "n_rows": n_rows,
            "n_students": n_students,
            "n_rubrics": n_rubrics,
            "feature_mean": feature_mean,
            "feature_sd": feature_sd,
            "logit_conf_mean": logit_conf_mean,
            "ame": np.nan,
            "ame_ci_low": np.nan,
            "ame_ci_high": np.nan,
            "curve": None,
        }

    if mode != "nonlinear":
        return _empty_calibration_regression_result(f"Unknown mode: {mode}")

    warnings_list: List[str] = []
    max_df = int(min(6, max(3, np.floor(n_rows / 10))))
    if spline_df > max_df:
        warnings_list.append(f"Reducing spline df from {spline_df} to {max_df} to avoid overfitting.")
        spline_df = max_df

    basis = f"bs(logit_conf_c, df={spline_df}, degree=3, intercept=FALSE)"
    formula_reduced = f"y ~ {basis} + feature_z + {re_part}"
    formula_full = f"y ~ {basis} * feature_z + {re_part}"

    # Grid + feature quantiles for curve summaries
    conf_grid = _build_calibration_conf_grid(confidence, n_grid=n_grid)
    logit_conf_grid = np.log(conf_grid / (1 - conf_grid)) - logit_conf_mean
    feat_q25 = float(np.quantile(feature_z, 0.25))
    feat_q75 = float(np.quantile(feature_z, 0.75))

    try:
        fit_pack = _fit_calibration_regression_nonlinear(
            df,
            formula_reduced=formula_reduced,
            formula_full=formula_full,
            conf_grid_logit_c=logit_conf_grid,
            feature_vals=np.array([feat_q25, feat_q75]),
            glmer_nAGQ=glmer_nAGQ,
            optimizer=optimizer,
            maxfun=maxfun,
        )
    except Exception as e:
        return _empty_calibration_regression_result(f"lme4 nonlinear fitting failed: {e}")

    warnings_list.extend(fit_pack.get("warnings", []))

    preds = fit_pack.get("preds", [])
    if len(preds) != 2:
        warnings_list.append("Nonlinear calibration regression predictions unavailable.")
        preds = [np.full_like(conf_grid, np.nan), np.full_like(conf_grid, np.nan)]

    curve_low = preds[0]
    curve_high = preds[1]
    curve_diff = curve_high - curve_low
    ame = float(np.nanmean(curve_diff)) if np.any(np.isfinite(curve_diff)) else np.nan

    ame_ci_low = np.nan
    ame_ci_high = np.nan

    # Optional bootstrap for AME CI
    if bootstrap_n and bootstrap_n > 0:
        rng_local = rng or np.random.default_rng()
        unique_students = np.unique(student_ids)
        if ci_method == "bca" and len(unique_students) < 30:
            warnings_list.append("BCa may be unstable with < 30 students; consider percentile CI.")
        boot_ame: List[float] = []
        iter_boot = range(bootstrap_n)
        try:
            from tqdm import tqdm  # type: ignore
            iter_boot = tqdm(iter_boot, desc="Calibration regression bootstrap", leave=False)
        except Exception:
            pass
        for _ in iter_boot:
            sampled_students = rng_local.choice(unique_students, size=len(unique_students), replace=True)
            idx_list = [np.where(student_ids == sid)[0] for sid in sampled_students]
            idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
            if idx.size < 10:
                boot_ame.append(np.nan)
                continue
            df_boot = df.iloc[idx].copy()
            try:
                boot_pack = _fit_calibration_regression_nonlinear(
                    df_boot,
                    formula_reduced=formula_reduced,
                    formula_full=formula_full,
                    conf_grid_logit_c=logit_conf_grid,
                    feature_vals=np.array([feat_q25, feat_q75]),
                    glmer_nAGQ=glmer_nAGQ,
                    optimizer=optimizer,
                    maxfun=maxfun,
                )
                boot_preds = boot_pack.get("preds", [])
                if len(boot_preds) != 2:
                    boot_ame.append(np.nan)
                else:
                    boot_diff = boot_preds[1] - boot_preds[0]
                    boot_ame.append(float(np.nanmean(boot_diff)))
            except Exception:
                boot_ame.append(np.nan)
                continue

        boot_ame_arr = np.asarray(boot_ame, dtype=float)
        boot_ame_arr = boot_ame_arr[np.isfinite(boot_ame_arr)]
        if boot_ame_arr.size < 10:
            warnings_list.append(
                f"Bootstrap valid draws {boot_ame_arr.size}/{bootstrap_n}; CI not computed."
            )
        if boot_ame_arr.size >= 10:
            if ci_method == "bca":
                if jackknife_max <= 0:
                    warnings_list.append("BCa requested but jackknife_max <= 0; using percentile CI.")
                    ame_ci_low, ame_ci_high = np.percentile(
                        boot_ame_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]
                    )
                else:
                    # Jackknife over students (capped)
                    jack_stats: List[float] = []
                    units = unique_students
                    if len(units) > jackknife_max:
                        rng_local = rng_local or np.random.default_rng()
                        units = rng_local.choice(units, size=jackknife_max, replace=False)
                        warnings_list.append(
                            f"BCa jackknife capped at {jackknife_max} students (requested {len(unique_students)})."
                        )
                    for sid in units:
                        idx_jack = np.where(student_ids != sid)[0]
                        if idx_jack.size < 10:
                            continue
                        df_jack = df.iloc[idx_jack].copy()
                        try:
                            jack_pack = _fit_calibration_regression_nonlinear(
                                df_jack,
                                formula_reduced=formula_reduced,
                                formula_full=formula_full,
                                conf_grid_logit_c=logit_conf_grid,
                                feature_vals=np.array([feat_q25, feat_q75]),
                                glmer_nAGQ=glmer_nAGQ,
                                optimizer=optimizer,
                                maxfun=maxfun,
                            )
                            jack_preds = jack_pack.get("preds", [])
                            if len(jack_preds) != 2:
                                continue
                            jack_diff = jack_preds[1] - jack_preds[0]
                            jack_stats.append(float(np.nanmean(jack_diff)))
                        except Exception:
                            continue
                    if jack_stats:
                        ame_ci_low, ame_ci_high = _compute_bca_ci_from_jackknife(
                            boot_ame_arr, ame, np.asarray(jack_stats, dtype=float), alpha=alpha
                        )
                    else:
                        warnings_list.append("BCa requested but jackknife unavailable; using percentile CI.")
                        ame_ci_low, ame_ci_high = np.percentile(
                            boot_ame_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]
                        )
            else:
                ame_ci_low, ame_ci_high = np.percentile(
                    boot_ame_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]
                )

    return {
        "mode": "nonlinear",
        "beta_main": np.nan,
        "se_main": np.nan,
        "p_main": np.nan,
        "beta_interaction": np.nan,
        "se_interaction": np.nan,
        "p_interaction": float(fit_pack.get("lrt_p", np.nan)),
        "chisq": float(fit_pack.get("lrt_stat", np.nan)),
        "df_diff": float(fit_pack.get("lrt_df", np.nan)),
        "converged": bool(np.isfinite(fit_pack.get("lrt_p", np.nan))),
        "singular": bool(fit_pack.get("singular", False)),
        "warnings": warnings_list,
        "n_rows": n_rows,
        "n_students": n_students,
        "n_rubrics": n_rubrics,
        "feature_mean": feature_mean,
        "feature_sd": feature_sd,
        "feature_q25": feat_q25,
        "feature_q75": feat_q75,
        "logit_conf_mean": logit_conf_mean,
        "spline_df": spline_df,
        "ame": ame,
        "ame_ci_low": float(ame_ci_low) if np.isfinite(ame_ci_low) else np.nan,
        "ame_ci_high": float(ame_ci_high) if np.isfinite(ame_ci_high) else np.nan,
        "curve": {
            "conf_grid": conf_grid,
            "curve_q25": curve_low,
            "curve_q75": curve_high,
            "curve_diff": curve_diff,
            "feature_q25_raw": feature_mean + feature_sd * feat_q25,
            "feature_q75_raw": feature_mean + feature_sd * feat_q75,
        },
    }


def _empty_calibration_regression_result(reason: str) -> Dict[str, Any]:
    """Return empty result dict when calibration regression cannot be computed."""
    return {
        "mode": None,
        "beta_main": np.nan,
        "se_main": np.nan,
        "p_main": np.nan,
        "beta_interaction": np.nan,
        "se_interaction": np.nan,
        "p_interaction": np.nan,
        "chisq": np.nan,
        "df_diff": np.nan,
        "converged": False,
        "singular": False,
        "warnings": [reason],
        "n_rows": 0,
        "n_students": 0,
        "n_rubrics": 0,
        "feature_mean": np.nan,
        "feature_sd": np.nan,
        "logit_conf_mean": np.nan,
        "spline_df": np.nan,
        "feature_q25": np.nan,
        "feature_q75": np.nan,
        "ame": np.nan,
        "ame_ci_low": np.nan,
        "ame_ci_high": np.nan,
        "curve": None,
    }


def _extract_coefficient_from_glmm(
    df: pd.DataFrame,
    formula: str,
    coef_name: str,
    glmer_nAGQ: int,
    optimizer: str,
    maxfun: int,
) -> Tuple[float, float, float]:
    """
    Fit a GLMM and extract Wald coefficient, SE, and p-value for a specific term.

    Returns:
        Tuple of (beta, se, p_wald) for the named coefficient.
    """
    ok, payload = _try_import_lme4()
    if not ok:
        return (np.nan, np.nan, np.nan)

    ro = payload["ro"]
    rconversion = payload["rconversion"]
    localconverter = payload["localconverter"]
    converter = payload["converter"]

    df_r = df.copy()
    for col in df_r.columns:
        if pd.api.types.is_categorical_dtype(df_r[col]) or df_r[col].dtype == object:
            df_r[col] = df_r[col].astype(str)

    with localconverter(converter):
        ro.globalenv["df_coef_tmp"] = rconversion.py2rpy(df_r)

    # Define R helper to extract coefficient
    if "gc_extract_coef" not in list(ro.globalenv.keys()):
        ro.r(
            textwrap.dedent(
                r"""
                gc_extract_coef <- function(df, formula_str, coef_name,
                                           glmer_nAGQ, optimizer, maxfun) {
                  suppressPackageStartupMessages(library(lme4))

                  quiet_eval <- function(expr) {
                    withCallingHandlers(
                      expr,
                      warning=function(w) invokeRestart("muffleWarning"),
                      message=function(m) invokeRestart("muffleMessage")
                    )
                  }

                  ctrl <- glmerControl(optimizer=optimizer, optCtrl=list(maxfun=maxfun))
                  fit <- quiet_eval(glmer(as.formula(formula_str), data=df,
                                         family=binomial(link="logit"),
                                         nAGQ=glmer_nAGQ, control=ctrl))

                  beta <- NA_real_
                  se <- NA_real_
                  z <- NA_real_
                  p_wald <- NA_real_

                  cf <- tryCatch(summary(fit)$coefficients, error=function(e) NULL)
                  if (!is.null(cf) && coef_name %in% rownames(cf)) {
                    beta <- cf[coef_name, "Estimate"]
                    se <- cf[coef_name, "Std. Error"]
                    z <- cf[coef_name, "z value"]
                    p_wald <- cf[coef_name, "Pr(>|z|)"]
                  }

                  list(beta=beta, se=se, z=z, p_wald=p_wald)
                }
                """
            )
        )

    try:
        res = ro.globalenv["gc_extract_coef"](
            ro.globalenv["df_coef_tmp"],
            formula,
            coef_name,
            int(glmer_nAGQ),
            str(optimizer),
            int(maxfun),
        )

        def _get_float(key: str) -> float:
            try:
                v = res.rx2(key)
                return float(v[0]) if v is not None and len(v) else float("nan")
            except Exception:
                return float("nan")

        return (_get_float("beta"), _get_float("se"), _get_float("p_wald"))
    except Exception:
        return (np.nan, np.nan, np.nan)
