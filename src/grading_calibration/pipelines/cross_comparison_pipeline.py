"""Step 6 cross-comparison orchestration helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from grading_calibration.metrics.cross_comparison import art_interaction_test, by_fdr, compute_i2


PER_INSTANCE_METRICS = {"is_correct", "squared_error", "raw_confidence"}


def run_cross_comparisons(
    *,
    all_preds: dict[tuple[str, str], pd.DataFrame],
    args,
    rng: np.random.Generator,
    icc_flag: bool,
    mixed_model_flag: bool,
    compute_metrics_summary: Callable,
    omnibus_friedman: Callable,
    align_by_instance: Callable,
    paired_compare: Callable,
    summarize_comparisons: Callable,
    rank_summary_best: Callable,
    select_best: Callable,
) -> dict[str, pd.DataFrame]:
    """Run Step 6 core comparisons without file I/O or argparse."""
    metrics_summary = compute_metrics_summary(all_preds, args.metrics, args.n_bins, args.binning_strategies)

    omnibus_rows = []
    if args.gate_posthoc:
        for method in args.methods:
            for metric in [m for m in args.metrics if m in PER_INSTANCE_METRICS]:
                per_instance = {}
                for model in args.models:
                    df = all_preds.get((model, method))
                    if df is None:
                        continue
                    per_instance[model] = df[metric].to_numpy(dtype=float)
                stat, p, n = omnibus_friedman(per_instance)
                omnibus_rows.append({
                    "scope": "models_within_method",
                    "method": method,
                    "metric": metric,
                    "statistic": stat,
                    "p_value": p,
                    "n_subjects": n,
                })
        for model in args.models:
            for metric in [m for m in args.metrics if m in PER_INSTANCE_METRICS]:
                per_instance = {}
                for method in args.methods:
                    df = all_preds.get((model, method))
                    if df is None:
                        continue
                    per_instance[method] = df[metric].to_numpy(dtype=float)
                stat, p, n = omnibus_friedman(per_instance)
                omnibus_rows.append({
                    "scope": "methods_within_model",
                    "model": model,
                    "metric": metric,
                    "statistic": stat,
                    "p_value": p,
                    "n_subjects": n,
                })
    omnibus_df = pd.DataFrame(omnibus_rows)
    gate_models: set[tuple[str, str]] = set()
    gate_methods: set[tuple[str, str]] = set()
    if args.gate_posthoc and len(omnibus_df):
        for _, row in omnibus_df.iterrows():
            p = row.get("p_value", np.nan)
            metric = row.get("metric", "")
            if not np.isfinite(p) or p >= args.alpha:
                continue
            if row.get("scope") == "models_within_method":
                gate_models.add((str(row.get("method")), str(metric)))
            elif row.get("scope") == "methods_within_model":
                gate_methods.add((str(row.get("model")), str(metric)))

    model_rows = []
    model_tasks: list[tuple[str, str, str, str, str]] = []
    for method in args.methods:
        for i in range(len(args.models)):
            for j in range(i + 1, len(args.models)):
                ma = args.models[i]
                mb = args.models[j]
                if all_preds.get((ma, method)) is None or all_preds.get((mb, method)) is None:
                    continue
                for metric in args.metrics:
                    if args.gate_posthoc and metric in PER_INSTANCE_METRICS and (method, metric) not in gate_models:
                        continue
                    if metric in {"ece", "mce"}:
                        for binning in args.binning_strategies:
                            model_tasks.append((method, ma, mb, metric, binning))
                    else:
                        model_tasks.append((method, ma, mb, metric, "equal_width"))
    print("Comparing model pairs...", flush=True)
    for method, ma, mb, metric, binning in tqdm(model_tasks, desc="Model comparisons"):
        df_a = all_preds.get((ma, method))
        df_b = all_preds.get((mb, method))
        if df_a is None or df_b is None:
            continue
        a, b = align_by_instance(df_a, df_b)
        block_ids = None
        if icc_flag and "student_id" in a.columns:
            block_ids = a["student_id"].to_numpy(dtype=str)
        res = paired_compare(
            a,
            b,
            metric,
            rng,
            args.n_perm,
            args.n_boot,
            args.alpha,
            args.ci_method,
            args.max_jackknife,
            args.n_bins,
            binning,
            block_ids,
            mixed_model_flag,
            None,
        )
        res.update({"comparison_type": "model", "method": method, "model_a": ma, "model_b": mb, "seed": args.seed})
        model_rows.append(res)

    method_rows = []
    method_tasks: list[tuple[str, str, str, str, str]] = []
    for model in args.models:
        for i in range(len(args.methods)):
            for j in range(i + 1, len(args.methods)):
                ma = args.methods[i]
                mb = args.methods[j]
                if all_preds.get((model, ma)) is None or all_preds.get((model, mb)) is None:
                    continue
                for metric in args.metrics:
                    if args.gate_posthoc and metric in PER_INSTANCE_METRICS and (model, metric) not in gate_methods:
                        continue
                    if metric in {"ece", "mce"}:
                        for binning in args.binning_strategies:
                            method_tasks.append((model, ma, mb, metric, binning))
                    else:
                        method_tasks.append((model, ma, mb, metric, "equal_width"))
    print("Comparing method pairs...", flush=True)
    for model, ma, mb, metric, binning in tqdm(method_tasks, desc="Method comparisons"):
        df_a = all_preds.get((model, ma))
        df_b = all_preds.get((model, mb))
        if df_a is None or df_b is None:
            continue
        a, b = align_by_instance(df_a, df_b)
        block_ids = None
        if icc_flag and "student_id" in a.columns:
            block_ids = a["student_id"].to_numpy(dtype=str)
        res = paired_compare(
            a,
            b,
            metric,
            rng,
            args.n_perm,
            args.n_boot,
            args.alpha,
            args.ci_method,
            args.max_jackknife,
            args.n_bins,
            binning,
            block_ids,
            mixed_model_flag,
            None,
        )
        res.update({"comparison_type": "method", "model": model, "method_a": ma, "method_b": mb, "seed": args.seed})
        method_rows.append(res)

    model_df = pd.DataFrame(model_rows)
    method_df = pd.DataFrame(method_rows)

    if "p_value" in model_df.columns and len(model_df):
        p_adj, sig = by_fdr(model_df["p_value"].to_numpy(dtype=float), args.alpha)
        model_df["p_adjusted_by"] = p_adj
        model_df["significant_by"] = sig
    if "p_value" in method_df.columns and len(method_df):
        p_adj, sig = by_fdr(method_df["p_value"].to_numpy(dtype=float), args.alpha)
        method_df["p_adjusted_by"] = p_adj
        method_df["significant_by"] = sig

    method_summary_df = summarize_comparisons(
        method_df,
        group_cols=["method_a", "method_b"],
        max_possible_groups=len(args.models),
        alpha=args.alpha,
    )
    model_summary_df = summarize_comparisons(
        model_df,
        group_cols=["model_a", "model_b"],
        max_possible_groups=len(args.methods),
        alpha=args.alpha,
    )
    method_summary_df = rank_summary_best(method_summary_df)
    model_summary_df = rank_summary_best(model_summary_df)
    best_method_summary = select_best(method_summary_df, args.best_top_k)
    best_model_summary = select_best(model_summary_df, args.best_top_k)

    interaction_df = pd.DataFrame()
    if args.skip_interaction_tests:
        print("Skipping interaction tests (--skip-interaction-tests).", flush=True)
    else:
        interaction_rows = []
        interaction_metrics = [m for m in args.metrics if m in PER_INSTANCE_METRICS]
        print("Running interaction tests...", flush=True)
        for metric in tqdm(interaction_metrics, desc="Interaction tests"):
            values = []
            models = []
            methods = []
            students = []
            has_students = True
            for (model, method), df in all_preds.items():
                if metric not in df.columns:
                    continue
                v = df[metric].to_numpy(dtype=float)
                values.append(v)
                models.append(np.array([model] * len(v)))
                methods.append(np.array([method] * len(v)))
                if "student_id" in df.columns:
                    students.append(df["student_id"].to_numpy(dtype=str))
                else:
                    has_students = False
            if not values:
                continue
            vals = np.concatenate(values)
            mod = np.concatenate(models)
            met = np.concatenate(methods)
            stud = np.concatenate(students) if has_students and students else None
            res = art_interaction_test(vals, mod, met, student_ids=stud, method=args.interaction_method)
            interaction_rows.append({"metric": metric, "seed": args.seed, **res})
        interaction_df = pd.DataFrame(interaction_rows)

    hetero_rows = []
    print("Computing heterogeneity (I2)...", flush=True)
    for metric in tqdm(args.metrics, desc="Heterogeneity"):
        if len(model_df):
            df_m = model_df[model_df["metric"] == metric]
            if len(df_m) > 1 and "diff" in df_m.columns:
                se = (df_m["diff_ci_high"] - df_m["diff_ci_low"]) / (2 * 1.96)
                i2 = compute_i2(df_m["diff"].to_numpy(dtype=float), se.to_numpy(dtype=float))
                hetero_rows.append({"family": "model", "metric": metric, "i2": i2})
        if len(method_df):
            df_m = method_df[method_df["metric"] == metric]
            if len(df_m) > 1 and "diff" in df_m.columns:
                se = (df_m["diff_ci_high"] - df_m["diff_ci_low"]) / (2 * 1.96)
                i2 = compute_i2(df_m["diff"].to_numpy(dtype=float), se.to_numpy(dtype=float))
                hetero_rows.append({"family": "method", "metric": metric, "i2": i2})
    hetero_df = pd.DataFrame(hetero_rows)

    return {
        "metrics_summary": metrics_summary,
        "omnibus_df": omnibus_df,
        "model_df": model_df,
        "method_df": method_df,
        "method_summary_df": method_summary_df,
        "model_summary_df": model_summary_df,
        "best_method_summary": best_method_summary,
        "best_model_summary": best_model_summary,
        "interaction_df": interaction_df,
        "hetero_df": hetero_df,
    }
