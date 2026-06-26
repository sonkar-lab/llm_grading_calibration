"""Step 3 statistical orchestration helpers.

These functions keep the estimator calls and post-processing from
``step3_feature_analysis.py`` in one importable place. They intentionally do not
write files, parse CLI arguments, or plot.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.statistics import (
    analyze_feature_moderated_calibration,
    benjamini_hochberg,
    estimate_feature_effect,
    loo_question_sensitivity,
)


def _process_one_mixed_model_feature(
    feature_col: str,
    x: np.ndarray,
    y: np.ndarray,
    target_col: str,
    target_label: str,
    out_type: str,
    sid: np.ndarray,
    qid: np.ndarray,
    rid: np.ndarray,
    args,
    force_feature_type: Optional[str] = None,
    force_include_question_fe: Optional[bool] = None,
):
    """Fit the Step 3 mixed model for one feature."""
    try:
        res = estimate_feature_effect(
            feature=x,
            outcome=y,
            student_ids=sid,
            question_ids=qid,
            rubric_ids=rid,
            outcome_type=out_type,
            backend=args.effects_backend,
            run_gee_sensitivity=not args.effects_no_gee,
            gee_mode=args.effects_gee_mode,
            use_ranks=args.effects_use_ranks,
            standardize_feature=True,
            standardize_outcome=args.effects_standardize_outcome,
            transform_bounded=args.effects_transform_bounded,
            include_student_vc_glmm=args.effects_include_student_vc_glmm,
            alpha=args.alpha,
            optimizer=args.effects_optimizer,
            maxfun=args.effects_lme4_maxfun,
            force_feature_type=force_feature_type,
            force_include_question_fe=force_include_question_fe,
        )
        row = {
            "feature": feature_col,
            "target": target_label,
            "target_col": target_col,
            "outcome_type": out_type,
            "feature_type": res.feature_type,
            "beta": res.beta,
            "se": res.se,
            "z": res.z,
            "p_value": res.p_value,
            "ci_low": res.ci_low,
            "ci_high": res.ci_high,
            "n_obs": res.n_obs,
            "method": res.method,
            "inference_type": res.inference_type,
            "outcome_transformed": res.outcome_transformed,
            "converged": res.converged,
            "gee_mode": res.gee_mode,
            "gee_beta": res.gee_beta,
            "gee_se": res.gee_se,
            "gee_p_value": res.gee_p_value,
            "sign_consistent": res.sign_consistent,
        }

        if out_type == "binary":
            row["odds_ratio"] = res.odds_ratio
            row["or_ci_low"] = res.or_ci_low
            row["or_ci_high"] = res.or_ci_high
            row["ame_fe_approx"] = res.ame
            row["include_student_vc_glmm"] = res.include_student_vc_glmm

        if res.warnings:
            row["warnings"] = "; ".join(res.warnings)

        loo_info = None
        if res.feature_type == "question_constant":
            loo_df, loo_sum = loo_question_sensitivity(
                feature=x,
                outcome=y,
                student_ids=sid,
                question_ids=qid,
                rubric_ids=rid,
                outcome_type=out_type,
                include_student_vc_glmm=args.effects_include_student_vc_glmm,
                transform_bounded=args.effects_transform_bounded,
                alpha=args.alpha,
                optimizer=args.effects_optimizer,
                maxfun=args.effects_lme4_maxfun,
            )
            row["loo_sign_consistent"] = loo_sum.get("sign_consistent")
            row["loo_cv"] = loo_sum.get("coefficient_of_variation")
            beta_range = loo_sum.get("beta_range") or (np.nan, np.nan)
            row["loo_beta_min"] = beta_range[0]
            row["loo_beta_max"] = beta_range[1]
            row["loo_n_successful"] = loo_sum.get("n_successful_fits")
            loo_info = {
                "key": f"{feature_col}__{target_col}",
                "summary": loo_sum,
                "folds": loo_df.to_dict(orient="records"),
            }

        return row, loo_info
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        return {
            "feature": feature_col,
            "target": target_label,
            "target_col": target_col,
            "outcome_type": out_type,
            "error": str(e),
        }, None


def _apply_effects_fdr(effects_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    effects_df["p_adjusted"] = np.nan
    effects_df["significant_fdr"] = False
    effects_df["fdr_note"] = None
    for tgt in effects_df["target_col"].dropna().unique():
        for inf in ["frequentist", "vb_posterior_approx"]:
            mask = (
                (effects_df["target_col"] == tgt)
                & (effects_df["inference_type"] == inf)
                & np.isfinite(effects_df["p_value"])
            )
            if not mask.any():
                continue
            p = effects_df.loc[mask, "p_value"].to_numpy(dtype=float)
            p_adj, sig = benjamini_hochberg(p, alpha=alpha)
            effects_df.loc[mask, "p_adjusted"] = p_adj
            effects_df.loc[mask, "significant_fdr"] = sig
            effects_df.loc[mask, "fdr_note"] = (
                "frequentist_fdr" if inf == "frequentist" else "screening_heuristic_vb"
            )
    return effects_df


def run_mixed_model_feature_effects(
    *,
    features_df: pd.DataFrame,
    corr_feature_cols: list[str],
    conf_feature_cols: list[str],
    cal_feature_cols: list[str],
    question_constant: list[str],
    args,
) -> tuple[pd.DataFrame | None, dict, pd.DataFrame | None, dict]:
    """Run Step 3 mixed-model feature-effect screening without writing files."""
    print("\n" + "=" * 80)
    print("MIXED-MODEL FEATURE EFFECTS (SCREENING)")
    print("=" * 80)
    print(
        "Primary models:\n"
        "  Continuous: MixedLM with student groups + answer_id and rubric_id variance components\n"
        "  Binary: BinomialBayesMixedGLM (VB) with answer_id and rubric_id variance components\n"
        "Controls:\n"
        "  C(question_id) fixed effects unless feature is question-constant (collinear)\n"
        f"Backend:\n  {args.effects_backend}\n"
        f"Sensitivity:\n  GEE clustered by {args.student_col} (mode={args.effects_gee_mode})\n"
    )

    targets_for_effects = [
        ("is_correct", "Correctness", "binary", corr_feature_cols),
        ("raw_confidence", "Confidence", "continuous", conf_feature_cols),
    ]
    if args.include_bin_calibration_error:
        targets_for_effects.insert(1, ("calibration_error", "Calibration Error", "continuous", cal_feature_cols))
        print("⚠ Including bin-assigned calibration_error as target (exploratory, not recommended).")

    max_feats = args.effects_max_features
    if max_feats is not None and max_feats > 0:
        targets_for_effects = [
            (tcol, tlabel, ttype, cols[:max_feats]) for (tcol, tlabel, ttype, cols) in targets_for_effects
        ]
        print(f"⚠ Limiting mixed-model effects to first {max_feats} features per target.")

    targets_for_effects_full = list(targets_for_effects)
    question_constant_set = set(question_constant) if args.effects_question_constant_separate_block else set()
    if question_constant_set:
        targets_for_effects = [
            (tcol, tlabel, ttype, [c for c in cols if c not in question_constant_set])
            for (tcol, tlabel, ttype, cols) in targets_for_effects
        ]
        print(
            f"⚠ Moving {len(question_constant_set)} question-constant features to separate block "
            "(no question effect)."
        )

    sid = features_df[args.student_col].to_numpy()
    qid = features_df[args.question_col].to_numpy()
    rid = features_df[args.rubric_col].to_numpy()

    effects_rows: list[dict] = []
    loo_details: dict = {}

    for target_col, target_label, out_type, cols in targets_for_effects:
        if target_col not in features_df.columns:
            print(f"Warning: missing target column {target_col}; skipping.")
            continue
        y = features_df[target_col].to_numpy()

        feature_arrays = [(col, features_df[col].to_numpy()) for col in cols]

        results = []
        pbar = tqdm(
            feature_arrays,
            total=len(feature_arrays),
            desc=f"Mixed-model effects [{target_col}]",
            unit="feature",
            dynamic_ncols=True,
            leave=True,
            disable=(not args.force_progress and not sys.stdout.isatty()),
        )
        for feature_col, x in pbar:
            pbar.set_postfix_str(feature_col)
            results.append(
                _process_one_mixed_model_feature(
                    feature_col,
                    x,
                    y,
                    target_col,
                    target_label,
                    out_type,
                    sid,
                    qid,
                    rid,
                    args,
                )
            )

        for row, loo_info in results:
            effects_rows.append(row)
            if loo_info is not None:
                loo_details[loo_info["key"]] = {
                    "summary": loo_info["summary"],
                    "folds": loo_info["folds"],
                }

    effects_df = _apply_effects_fdr(pd.DataFrame(effects_rows), args.alpha) if effects_rows else None

    qc_df = None
    qc_loo_details: dict = {}
    if args.effects_question_constant_separate_block and question_constant_set:
        print("\n" + "-" * 80)
        print("QUESTION-LEVEL FEATURES (NO QUESTION EFFECT)")
        print("-" * 80)
        qc_rows: list[dict] = []
        for target_col, target_label, out_type, cols in targets_for_effects_full:
            qc_cols = [c for c in cols if c in question_constant_set]
            if not qc_cols:
                continue
            y = features_df[target_col].to_numpy()
            feature_arrays = [(col, features_df[col].to_numpy()) for col in qc_cols]
            pbar = tqdm(
                feature_arrays,
                total=len(feature_arrays),
                desc=f"Question-level effects [{target_col}]",
                unit="feature",
                dynamic_ncols=True,
                leave=True,
                disable=(not args.force_progress and not sys.stdout.isatty()),
            )
            for feature_col, x in pbar:
                pbar.set_postfix_str(feature_col)
                row, loo_info = _process_one_mixed_model_feature(
                    feature_col,
                    x,
                    y,
                    target_col,
                    target_label,
                    out_type,
                    sid,
                    qid,
                    rid,
                    args,
                    force_feature_type="question_constant",
                    force_include_question_fe=False,
                )
                qc_rows.append(row)
                if loo_info is not None:
                    qc_loo_details[loo_info["key"]] = {
                        "summary": loo_info["summary"],
                        "folds": loo_info["folds"],
                    }

        if qc_rows:
            qc_df = _apply_effects_fdr(pd.DataFrame(qc_rows), args.alpha)

    return effects_df, loo_details, qc_df, qc_loo_details


def run_calibration_regression(
    *,
    features_df: pd.DataFrame,
    conf_feature_cols: list[str],
    confidence_leakage_features: set[str],
    label_leakage_features: set[str],
    args,
) -> tuple[pd.DataFrame | None, list[pd.DataFrame]]:
    """Run Step 3 feature-moderated calibration regressions without writing files."""
    if args.calibration_regression_features:
        features_to_test = args.calibration_regression_features
        print(f"Testing {len(features_to_test)} specified features: {', '.join(features_to_test)}")
    else:
        features_to_test = conf_feature_cols
        print(f"Testing {len(features_to_test)} numeric features (all confidence-related features)")

    _cal_reg_leakage = confidence_leakage_features | label_leakage_features | {"confidence_delta"}
    leakage_hits = [f for f in features_to_test if f in _cal_reg_leakage]
    if leakage_hits:
        features_to_test = [f for f in features_to_test if f not in _cal_reg_leakage]
        print(f"⚠ Excluding leakage features from calibration regression: {', '.join(sorted(leakage_hits))}")
    if args.calibration_regression_max_features and len(features_to_test) > args.calibration_regression_max_features:
        features_to_test = features_to_test[: args.calibration_regression_max_features]
        print(f"Limiting to first {len(features_to_test)} features due to --calibration-regression-max-features")

    is_correct = features_df["is_correct"].values
    confidence = features_df["raw_confidence"].values
    student_ids = features_df[args.student_col].values

    rubric_ids = None
    if args.rubric_col in features_df.columns:
        rubric_ids_check = features_df[args.rubric_col].values
        if len(np.unique(rubric_ids_check)) >= 2:
            rubric_ids = rubric_ids_check

    question_ids = None
    if args.question_col in features_df.columns:
        question_ids = features_df[args.question_col].values

    cal_reg_mode = args.calibration_regression_mode
    cal_reg_results = []
    cal_reg_rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
    cal_reg_curve_rows = []
    cal_reg_iter = tqdm(
        features_to_test,
        desc="Calibration regression",
        unit="feature",
        dynamic_ncols=True,
        disable=(not args.force_progress and not sys.stderr.isatty()),
    )
    for feature_name in cal_reg_iter:
        if feature_name not in features_df.columns:
            print(f"⚠ Feature '{feature_name}' not found in DataFrame, skipping.")
            continue

        feature_values = features_df[feature_name].values

        if len(np.unique(feature_values[np.isfinite(feature_values)])) < 2:
            print(f"⚠ Skipping {feature_name}: insufficient variance")
            continue

        try:
            result = analyze_feature_moderated_calibration(
                is_correct=is_correct,
                confidence=confidence,
                feature=feature_values,
                student_ids=student_ids,
                rubric_ids=rubric_ids,
                question_ids=question_ids,
                alpha=args.alpha,
                glmer_nAGQ=args.calibration_regression_glmer_nAGQ,
                optimizer=args.calibration_regression_optimizer,
                maxfun=args.calibration_regression_maxfun,
                mode=cal_reg_mode,
                spline_df=args.calibration_regression_spline_df,
                bootstrap_n=args.calibration_regression_bootstrap,
                ci_method=args.calibration_regression_ci_method,
                jackknife_max=args.calibration_regression_jackknife_max,
                n_grid=args.calibration_regression_n_grid,
                rng=cal_reg_rng,
            )
            curve = result.pop("curve", None)
            if curve is not None:
                curve_df = pd.DataFrame({
                    "feature": feature_name,
                    "confidence": curve.get("conf_grid", []),
                    "pred_q25": curve.get("curve_q25", []),
                    "pred_q75": curve.get("curve_q75", []),
                    "pred_diff": curve.get("curve_diff", []),
                })
                curve_df["feature_q25"] = result.get("feature_q25", np.nan)
                curve_df["feature_q75"] = result.get("feature_q75", np.nan)
                curve_df["feature_q25_raw"] = curve.get("feature_q25_raw", np.nan)
                curve_df["feature_q75_raw"] = curve.get("feature_q75_raw", np.nan)
                curve_df["mode"] = cal_reg_mode
                cal_reg_curve_rows.append(curve_df)
            result["feature"] = feature_name
            cal_reg_results.append(result)
        except Exception as e:
            print(f"⚠ Error in calibration regression for {feature_name}: {e}")
            continue

    if cal_reg_results:
        p_interactions = np.array([r["p_interaction"] for r in cal_reg_results])
        p_interactions_filled = np.where(np.isnan(p_interactions), 1.0, p_interactions)
        p_adj, sig = benjamini_hochberg(p_interactions_filled, alpha=args.alpha)

        for i, result in enumerate(cal_reg_results):
            result["p_interaction_adj"] = p_adj[i]
            result["significant_interaction"] = sig[i]

        cal_reg_df = pd.DataFrame(cal_reg_results)
        cal_reg_df = cal_reg_df.sort_values("p_interaction", na_position="last")
        return cal_reg_df, cal_reg_curve_rows

    return None, cal_reg_curve_rows
