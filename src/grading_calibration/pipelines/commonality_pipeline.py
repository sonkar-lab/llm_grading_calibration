"""Step 5 commonality-analysis orchestration helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from utils.statistics import benjamini_hochberg


def run_commonality_comparisons(
    *,
    features_df: pd.DataFrame,
    args,
    rng: np.random.Generator,
    compare_groups: Callable,
    run_cluster_robust: bool,
    student_ids,
    question_ids,
    rubric_ids,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run Step 5 comparison tables and global FDR without file I/O."""
    print("\n" + "=" * 80)
    print(f"HIGH CONFIDENCE (>{args.high_threshold}) vs LOW CONFIDENCE (<{args.low_threshold})")
    print("=" * 80)

    high_conf_mask = features_df["raw_confidence"] > args.high_threshold
    low_conf_mask = features_df["raw_confidence"] < args.low_threshold

    n_high = high_conf_mask.sum()
    n_low = low_conf_mask.sum()
    print(f"\nHigh confidence group: {n_high} instances ({n_high/len(features_df)*100:.1f}%)")
    print(f"Low confidence group: {n_low} instances ({n_low/len(features_df)*100:.1f}%)")

    high_acc = np.nan
    low_acc = np.nan
    if n_high > 5 and n_low > 5:
        high_acc = features_df.loc[high_conf_mask, "is_correct"].mean()
        low_acc = features_df.loc[low_conf_mask, "is_correct"].mean()
        print(f"\nAccuracy in high confidence: {high_acc:.3f}")
        print(f"Accuracy in low confidence: {low_acc:.3f}")

        conf_comparison = compare_groups(
            features_df,
            high_conf_mask,
            low_conf_mask,
            "high",
            "low",
            n_perm=args.n_perm,
            n_boot=args.n_boot,
            alpha=args.alpha,
            bootstrap_method=args.bootstrap_method,
            rng=rng,
            adaptive_perm=args.adaptive_perm,
            adaptive_perm_max=args.adaptive_perm_max,
            adaptive_band=tuple(args.adaptive_perm_band),
            exact_power=bool(args.exact_power),
            power_sim=int(args.power_sim),
            power_perm=int(args.power_perm),
            run_cluster_robust=run_cluster_robust,
            student_ids=student_ids,
            question_ids=question_ids,
            rubric_ids=rubric_ids,
        )
    else:
        print("\n⚠ Not enough instances in one or both groups for comparison")
        conf_comparison = pd.DataFrame()

    print("\n" + "=" * 80)
    print("CORRECT vs INCORRECT PREDICTIONS")
    print("=" * 80)

    correct_mask = features_df["is_correct"] == 1.0
    incorrect_mask = features_df["is_correct"] == 0.0

    n_correct = correct_mask.sum()
    n_incorrect = incorrect_mask.sum()
    print(f"\nCorrect predictions: {n_correct} ({n_correct/len(features_df)*100:.1f}%)")
    print(f"Incorrect predictions: {n_incorrect} ({n_incorrect/len(features_df)*100:.1f}%)")

    if n_correct > 5 and n_incorrect > 5:
        correct_conf = features_df.loc[correct_mask, "raw_confidence"].mean()
        incorrect_conf = features_df.loc[incorrect_mask, "raw_confidence"].mean()
        print(f"\nMean confidence for correct: {correct_conf:.3f}")
        print(f"Mean confidence for incorrect: {incorrect_conf:.3f}")

        correctness_comparison = compare_groups(
            features_df,
            correct_mask,
            incorrect_mask,
            "correct",
            "incorrect",
            n_perm=args.n_perm,
            n_boot=args.n_boot,
            alpha=args.alpha,
            bootstrap_method=args.bootstrap_method,
            rng=rng,
            adaptive_perm=args.adaptive_perm,
            adaptive_perm_max=args.adaptive_perm_max,
            adaptive_band=tuple(args.adaptive_perm_band),
            exact_power=bool(args.exact_power),
            power_sim=int(args.power_sim),
            power_perm=int(args.power_perm),
            run_cluster_robust=run_cluster_robust,
            student_ids=student_ids,
            question_ids=question_ids,
            rubric_ids=rubric_ids,
        )
    else:
        print("\n⚠ Not enough instances in one or both groups for comparison")
        correctness_comparison = pd.DataFrame()

    print("\n" + "=" * 80)
    print("GLOBAL MULTIPLE TESTING CORRECTION")
    print("=" * 80)

    all_p_values = []
    all_sources = []
    if not conf_comparison.empty:
        all_p_values.extend(conf_comparison["p_value"].tolist())
        all_sources.extend(["confidence"] * len(conf_comparison))
    if not correctness_comparison.empty:
        all_p_values.extend(correctness_comparison["p_value"].tolist())
        all_sources.extend(["correctness"] * len(correctness_comparison))

    if all_p_values:
        p_adj_global, sig_global = benjamini_hochberg(np.array(all_p_values), args.alpha)
        n_conf = len(conf_comparison) if not conf_comparison.empty else 0
        if not conf_comparison.empty:
            conf_comparison["p_adjusted_global"] = p_adj_global[:n_conf]
            conf_comparison["significant_global"] = sig_global[:n_conf]
        if not correctness_comparison.empty:
            correctness_comparison["p_adjusted_global"] = p_adj_global[n_conf:]
            correctness_comparison["significant_global"] = sig_global[n_conf:]
        n_sig_global = int(sig_global.sum())
        print(f"Total tests: {len(all_p_values)}")
        print(f"Significant (global FDR): {n_sig_global}")
    else:
        print("No tests available for global FDR correction.")

    group_info = {
        "high_conf_mask": high_conf_mask,
        "low_conf_mask": low_conf_mask,
        "correct_mask": correct_mask,
        "incorrect_mask": incorrect_mask,
        "n_high": n_high,
        "n_low": n_low,
        "high_acc": high_acc,
        "low_acc": low_acc,
    }
    return conf_comparison, correctness_comparison, group_info
