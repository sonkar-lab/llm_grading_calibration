"""Step 4 stratified-analysis orchestration helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm


def run_stratified_group_metrics(
    *,
    predictions: list,
    features_df: pd.DataFrame | None,
    args,
    rng: np.random.Generator,
    stratify_by_category: Callable,
    stratify_by_feature_bins: Callable,
    compute_metrics_for_group: Callable,
    compute_metrics_for_group_with_ci: Callable,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute Step 4 stratified group metric tables without file I/O."""

    def _compute_group_table(group_col: str) -> pd.DataFrame:
        groups_local = stratify_by_category(predictions, group_col)
        rows: list[dict] = []
        iterator_local = sorted(groups_local.items(), key=lambda x: len(x[1]), reverse=True)
        if args.compute_ci:
            iterator_local = tqdm(
                iterator_local, desc=f"Groups[{group_col}] (bootstrap CIs)", unit="group", dynamic_ncols=True
            )
        for group_name, group_preds in iterator_local:
            if args.compute_ci:
                metrics = compute_metrics_for_group_with_ci(
                    group_preds,
                    n_bootstrap=args.n_bootstrap,
                    alpha=args.alpha,
                    rng=rng,
                )
            else:
                metrics = compute_metrics_for_group(group_preds)
            metrics["group"] = group_name
            rows.append(metrics)
        out_df = pd.DataFrame(rows)
        return out_df.sort_values("ece", ascending=False)

    continuous_features = ["answer_length", "response_length", "question_length", "rubric_length"]
    results_by_stratum: dict[str, pd.DataFrame] = {}

    if args.stratify_by in continuous_features:
        assert features_df is not None
        groups = stratify_by_feature_bins(predictions, features_df, args.stratify_by, args.n_bins)
        rows: list[dict] = []
        iterator = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        if args.compute_ci:
            iterator = tqdm(iterator, desc="Groups (bootstrap CIs)", unit="group", dynamic_ncols=True)
        for group_name, group_preds in iterator:
            if args.compute_ci:
                metrics = compute_metrics_for_group_with_ci(
                    group_preds,
                    n_bootstrap=args.n_bootstrap,
                    alpha=args.alpha,
                    rng=rng,
                )
            else:
                metrics = compute_metrics_for_group(group_preds)
            metrics["group"] = group_name
            rows.append(metrics)
        results_df = pd.DataFrame(rows).sort_values("ece", ascending=False)
        results_by_stratum[args.stratify_by] = results_df
    elif args.stratify_by == "all":
        for col in ["rubric_id", "question_id"]:
            df_col = _compute_group_table(col)
            results_by_stratum[col] = df_col
        results_df = results_by_stratum["rubric_id"].copy()
    else:
        results_df = _compute_group_table(args.stratify_by)
        results_by_stratum[args.stratify_by] = results_df

    return results_by_stratum, results_df


def run_metric_pairwise_tests(
    *,
    predictions: list,
    args,
    rng: np.random.Generator,
    run_metric_pairwise_tests_within_question: Callable,
    run_metric_pairwise_tests_across_questions: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Step 4 metric pairwise tests without file I/O."""
    metric_df = pd.DataFrame()
    q_metric_df = pd.DataFrame()

    print("\n" + "=" * 80)
    print("SECONDARY METRIC TESTS: ECE/AUC DIFFERENCES (STUDENT-BLOCK BOOTSTRAP)")
    print("=" * 80)
    print("Rubric tests are restricted to within-question comparisons due to rubric nesting.")

    try:
        metric_df = run_metric_pairwise_tests_within_question(
            predictions,
            alpha=args.alpha,
            pairwise_scope=args.metric_test_pairwise_scope,
            pairwise_top_k=args.metric_test_pairwise_top_k,
            n_bootstrap=int(args.metric_test_n_bootstrap),
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

    if bool(args.metric_test_include_questions) and not bool(args.skip_question_tests):
        print("\n" + "=" * 80)
        print("SECONDARY METRIC TESTS (EXPLORATORY): ECE/AUC DIFFERENCES ACROSS QUESTIONS (PAIRED STUDENT BOOTSTRAP)")
        print("=" * 80)
        try:
            q_metric_df = run_metric_pairwise_tests_across_questions(
                predictions,
                alpha=args.alpha,
                n_bootstrap=int(args.metric_test_n_bootstrap),
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

    return metric_df, q_metric_df
