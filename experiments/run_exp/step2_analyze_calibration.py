"""
Step 2: Analyze Calibration from Predictions

This script:
1. Reads predictions CSV from Step 1
2. Checks if confidence is calibratable
3. Applies calibration (isotonic/temperature/platt)
4. Evaluates metrics (ECE, Brier, AUC-ROC, etc.)
5. Answers: "Is self-reported confidence calibrated?"

Usage:
    cd experiments/self_reported
    python step2_analyze_calibration.py \
        --predictions predictions_qwen.csv \
        --calibrator isotonic
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
print(f"Adding project root to sys.path: {project_root}")
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from scipy.stats import norm

from grading_calibration.domain.models import PredictionResult
from utils import find_csv_folder_from_root
from utils.output_paths import add_output_args, build_output_manager
from grading_calibration.calibration import (
    IsotonicCalibrator,
    TemperatureScalingCalibrator,
    PlattScalingCalibrator,
)
from grading_calibration.metrics import (
    check_calibratability,
    evaluate_predictions,
    expected_calibration_error,
    brier_score,
    auc_roc_correctness,
    plot_calibration_curve,
)
from grading_calibration.metrics.calibration_bins import _extract_confidence_correctness
from grading_calibration.inference.model_aliases import MODEL_ALIASES as model_nicknames

def load_path_by_index(
    idx: int,
    folder: Path | str,
    prefix: str,
    dataset: str = "ricechem",
    *,
    filter_by_dataset: bool = True,
) -> Path:
    folder = Path(folder)
    print(f"Loading CSV from folder: {folder} with prefix: {prefix}")
    paths = sorted(folder.glob(f"{prefix}_*.csv"))
    if filter_by_dataset:
        dataset = dataset.lower()
        other_datasets = {"ricechem", "scientsbank", "beetle"} - {dataset}
        if dataset == "ricechem":
            paths = [p for p in paths if not any(f"_{d}_" in p.name for d in other_datasets)]
        else:
            paths = [p for p in paths if f"_{dataset}_" in p.name]
    print(f"Found {len(paths)} matching CSV files.")
    if not paths:
        raise FileNotFoundError(f"No predictions CSVs found for dataset={dataset} in {folder}")
    new_idx = len(paths) - 1 - idx

    return paths[new_idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Analyze calibration from predictions CSV"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ricechem",
        choices=["ricechem", "scientsbank", "beetle"],
        help="Dataset name for selecting prediction CSVs (default: ricechem).",
    )
    parser.add_argument(
        "--predictions", type=str, default=None, help="CSV file from step 1"
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Override predictions directory (legacy layout). "
             "Use this to load old predictions while still saving organized outputs.",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default="none",
        choices=["isotonic", "temperature", "platt", "none"],
        help="Calibration method",
    )
    parser.add_argument(
        "--allow-calibration-fitting",
        action="store_true",
        help="Allow calibration fitting (disabled by default for this study).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "valid", "test"],
        help="Which split to analyze",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="self_reported",
        choices=["self_reported", "self_consistency", "token_probability"],
        help="Confidence estimation method used",
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
        "--smoke-n",
        type=int,
        default=None,
        help="Limit to first N rows for a smoke test run.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model used for predictions",
    )
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Do not save metrics to CSV",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Do not save calibration plots",
    )
    parser.add_argument(
        "--calibration-curve-only",
        action="store_true",
        help="Only generate calibration curve plots (skip metrics/CI/calibration).",
    )
    parser.add_argument(
        "--compute-ci",
        action="store_true",
        help="Compute bootstrap CIs for selected metrics using student-block resampling.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap draws for CIs (default: 200).",
    )
    parser.add_argument(
        "--ci-method",
        type=str,
        choices=["percentile", "bca"],
        default="percentile",
        help="Bootstrap CI method (default: percentile).",
    )
    parser.add_argument(
        "--bootstrap-block",
        type=str,
        choices=["student", "none"],
        default="student",
        help="Bootstrap block for CIs (default: student).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=123,
        help="Seed for bootstrap resampling (default: 123).",
    )
    add_output_args(parser)
    return parser.parse_args()


def load_predictions_from_csv(csv_path: str, split: str = "all", smoke_n: int | None = None) -> dict:
    """Load predictions from CSV and convert to PredictionResult objects."""
    df = pd.read_csv(csv_path)

    # Filter by split if requested
    if split != "all":
        df = df[df["split"] == split]

    if smoke_n is not None:
        df = df.head(smoke_n).copy()
        print(f"Smoke test: limiting to first {len(df)} rows")

    print(f"Loaded {len(df)} predictions from {csv_path}")
    if split != "all":
        print(f"Filtered to '{split}' split")

    # Convert to PredictionResult objects
    predictions = []
    for _, row in df.iterrows():
        pred = PredictionResult(
            instance_id=row["instance_id"],
            prediction=int(row["prediction"]),
            raw_confidence=float(row["raw_confidence"]),
            ground_truth=int(row["ground_truth_label"]),
            metadata={
                "split": row["split"],
                "question_id": row.get("question_id", ""),
                "student_id": row.get("student_id", ""),
            },
        )
        predictions.append(pred)

    # Separate by split
    train_preds = [p for p in predictions if p.metadata["split"] == "train"]
    valid_preds = [p for p in predictions if p.metadata["split"] == "valid"]
    test_preds = [p for p in predictions if p.metadata["split"] == "test"]

    return {
        "all": predictions,
        "train": train_preds,
        "valid": valid_preds,
        "test": test_preds,
        "df": df,
    }


def print_metrics(predictions, title="Metrics", use_calibrated=False):
    """Print evaluation metrics."""
    metrics = evaluate_predictions(predictions, use_calibrated=use_calibrated)
    metrics["mean_confidence"] = float(
        np.mean(_extract_confidence_correctness(predictions, use_calibrated)[0])
    )

    print(f"\n{title}")
    print("-" * 60)
    print(f"  N instances: {metrics['n_predictions']}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Spearman Corr: {metrics['spearman_corr']:.4f}")
    print(f"  Acc@50%: {metrics['acc_at_50']:.2%}")
    print(f"  Acc@80%: {metrics['acc_at_80']:.2%}")
    print(f"  Cov@5% risk: {metrics['cov_at_5pct_risk']:.2%}")
    print(f"  Cov@10% risk: {metrics['cov_at_10pct_risk']:.2%}")
    print(f"  Mean confidence: {metrics['mean_confidence']:.4f}")

    return metrics


def _bootstrap_metric_cis(
    predictions: list[PredictionResult],
    *,
    use_calibrated: bool,
    n_bootstrap: int,
    ci_method: str,
    block: str,
    seed: int,
) -> dict:
    if n_bootstrap <= 0 or len(predictions) == 0:
        return {}

    rng = np.random.default_rng(seed)
    metrics_to_ci = ["accuracy", "ece", "brier_score", "auc_roc", "mean_confidence"]
    samples = {key: [] for key in metrics_to_ci}

    def _compute_metrics(preds: list[PredictionResult]) -> dict:
        metrics = evaluate_predictions(preds, use_calibrated=use_calibrated)
        metrics["mean_confidence"] = float(
            np.mean(_extract_confidence_correctness(preds, use_calibrated)[0])
        )
        return metrics

    def _bca_interval(values: np.ndarray, theta_hat: float, jack_vals: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
        values = np.asarray(values, dtype=float)
        jack_vals = np.asarray(jack_vals, dtype=float)
        if values.size == 0 or jack_vals.size == 0:
            return np.nan, np.nan
        prop_less = np.mean(values < theta_hat)
        prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
        z0 = norm.ppf(prop_less)

        jack_mean = np.mean(jack_vals)
        diffs = jack_mean - jack_vals
        num = np.sum(diffs ** 3)
        den = 6.0 * (np.sum(diffs ** 2) ** 1.5)
        a = num / den if den > 0 else 0.0

        z_alpha_low = norm.ppf(alpha / 2)
        z_alpha_high = norm.ppf(1 - alpha / 2)

        def _adjust(z_alpha: float) -> float:
            denom = 1 - a * (z0 + z_alpha)
            if denom == 0:
                denom = np.sign(denom) * 1e-6
            return norm.cdf(z0 + (z0 + z_alpha) / denom)

        adj_low = _adjust(z_alpha_low)
        adj_high = _adjust(z_alpha_high)
        adj_low = min(max(adj_low, 0.0), 1.0)
        adj_high = min(max(adj_high, 0.0), 1.0)
        low = float(np.quantile(values, adj_low))
        high = float(np.quantile(values, adj_high))
        return low, high

    if block == "student":
        by_student = {}
        for pred in predictions:
            sid = pred.metadata.get("student_id", "")
            if not sid:
                sid = pred.instance_id
            by_student.setdefault(sid, []).append(pred)
        student_ids = list(by_student.keys())
        if len(student_ids) < 2:
            block = "none"
        else:
            for _ in range(int(n_bootstrap)):
                sampled = rng.choice(student_ids, size=len(student_ids), replace=True)
                boot_preds = []
                for sid in sampled:
                    boot_preds.extend(by_student[sid])
                metrics = evaluate_predictions(boot_preds, use_calibrated=use_calibrated)
                metrics["mean_confidence"] = float(
                    np.mean(_extract_confidence_correctness(boot_preds, use_calibrated)[0])
                )
                for key in metrics_to_ci:
                    samples[key].append(metrics[key])

    if block == "none":
        n = len(predictions)
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            boot_preds = [predictions[i] for i in idx]
            metrics = evaluate_predictions(boot_preds, use_calibrated=use_calibrated)
            metrics["mean_confidence"] = float(
                np.mean(_extract_confidence_correctness(boot_preds, use_calibrated)[0])
            )
            for key in metrics_to_ci:
                samples[key].append(metrics[key])

    alpha = 0.05
    out = {}
    theta_hat = _compute_metrics(predictions)

    if ci_method == "bca":
        if block != "student":
            raise ValueError("BCa CIs require --bootstrap-block student in Step2.")
        student_ids = [
            pred.metadata.get("student_id", "") or pred.instance_id
            for pred in predictions
        ]
        unique_students = list(dict.fromkeys(student_ids))
        if len(unique_students) < 2:
            print("⚠ BCa requires at least 2 student blocks; falling back to percentile CIs.")
            ci_method = "percentile"
        else:
            jackknife = {key: [] for key in metrics_to_ci}
            for sid in unique_students:
                jack_preds = [pred for pred, pid in zip(predictions, student_ids) if pid != sid]
                if not jack_preds:
                    continue
                metrics = _compute_metrics(jack_preds)
                for key in metrics_to_ci:
                    jackknife[key].append(metrics[key])

            for key, values in samples.items():
                arr = np.asarray(values, dtype=float)
                if arr.size == 0:
                    continue
                jack_vals = np.asarray(jackknife.get(key, []), dtype=float)
                if jack_vals.size == 0:
                    continue
                low, high = _bca_interval(arr, float(theta_hat[key]), jack_vals, alpha=alpha)
                out[f"{key}_ci_low"] = low
                out[f"{key}_ci_high"] = high

    if ci_method == "percentile":
        for key, values in samples.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            low = float(np.quantile(arr, alpha / 2))
            high = float(np.quantile(arr, 1 - alpha / 2))
            out[f"{key}_ci_low"] = low
            out[f"{key}_ci_high"] = high
    return out


def save_metrics_to_csv(
    metrics: dict,
    model_name: str,
    method: str,
    timestamp: str,
    calibrator: str = "none",
    is_calibrated: bool = False,
    output_dir: Path | None = None,
    filename_override: str | None = None,
) -> Path:
    """Save metrics to a CSV file in metrics/{method}/ folder."""
    # Create metrics directory
    metrics_dir = output_dir or (project_root / "experiments" / "metrics" / method)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    calibrated_suffix = "_calibrated" if is_calibrated else "_raw"
    calibrator_suffix = f"_{calibrator}" if is_calibrated and calibrator != "none" else ""
    filename = filename_override or f"metrics_{model_name}_{method}{calibrated_suffix}{calibrator_suffix}_{timestamp}.csv"
    filepath = metrics_dir / filename

    # Prepare data for CSV
    metrics_data = {
        "model_name": model_name,
        "method": method,
        "calibrator": calibrator if is_calibrated else "none",
        "is_calibrated": is_calibrated,
        "timestamp": timestamp,
        **metrics,
    }

    # Convert to DataFrame and save
    df = pd.DataFrame([metrics_data])
    df.to_csv(filepath, index=False)
    print(f"\nMetrics saved to: {filepath}")
    return filepath


def main():
    args = parse_args()
    args.model = model_nicknames.get(args.model.lower(), args.model.lower())
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
    if not args.allow_calibration_fitting:
        print("\n" + "=" * 80)
        print("NOTE: CALIBRATION FITTING DISABLED FOR THIS STUDY")
        print("=" * 80)
        print("Outputs are descriptive; calibration fitting is ignored unless --allow-calibration-fitting is set.")
        if args.calibrator != "none":
            print(f"Ignoring --calibrator={args.calibrator} (set --allow-calibration-fitting to enable).")
            args.calibrator = "none"
    print("=" * 80)
    print("STEP 2: CALIBRATION ANALYSIS")
    print("=" * 80)
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    # ========================================================================
    # Load Predictions
    # ========================================================================
    print("\n" + "-" * 80)
    print("Loading Predictions")
    print("-" * 80)
    if args.predictions:
        csv_path = Path(args.predictions)
    else:
        if args.predictions_dir:
            directory = Path(args.predictions_dir) / Path(args.method)
            filter_by_dataset = True
        elif output_manager:
            directory = output_manager.step_dir("step1", "tables")
            filter_by_dataset = False
        else:
            directory = find_csv_folder_from_root("grading_calibration") / Path(args.method)
            filter_by_dataset = True
        csv_path = load_path_by_index(
            args.index,
            directory,
            f"{args.prefix}_{model_name}_{args.method}",
            args.dataset,
            filter_by_dataset=filter_by_dataset,
        )
    data = load_predictions_from_csv(csv_path, args.split, smoke_n=args.smoke_n)
    if args.method == "self_consistency":
        assert (
            data["df"].filter(like="llm_response_").notnull().all().all()
        ), "LLM response columns are missing or contain null values."
    else:
        assert (
            data["df"]["llm_response"].notnull().all()
        ), "LLM response column is missing or contains null values."

    if args.split == "all":
        print(f"\nSplit distribution:")
        print(f"  Train: {len(data['train'])}")
        print(f"  Valid: {len(data['valid'])}")
        print(f"  Test: {len(data['test'])}")

    # Use validation set for calibration, test set for evaluation
    # If analyzing specific split, use that
    if args.split == "test":
        val_preds = data["valid"]
        test_preds = data["test"]
    elif args.split == "valid":
        val_preds = data["train"]  # Use train to calibrate, valid to test
        test_preds = data["valid"]
    elif args.split == "train":
        # Just analyze train set (no calibration)
        val_preds = []
        test_preds = data["train"]
    else:  # all
        val_preds = []
        test_preds = data["train"] + data["valid"] + data["test"]

    # ========================================================================
    # Pre-Calibration: Check if Calibratable
    # ========================================================================
    if args.calibration_curve_only:
        if args.calibrator != "none":
            print("Note: --calibrator ignored when using --calibration-curve-only.")
        if args.no_save_plots:
            print("Nothing to do: --no-save-plots set with --calibration-curve-only.")
            return
        plot_name = f"calibration_curve_raw_{model_name}_{args.method}"
        plots_dir = output_manager.step_dir("step2", "plots") if output_manager else None
        plot_info = plot_calibration_curve(
            test_preds,
            num_bins=10,
            use_calibrated=False,
            name=plot_name,
            model_name=model_name,
            method_name=args.method,
            output_dir=plots_dir,
        )
        if output_manager:
            output_manager.register(
                step="step2",
                artifact="plot_calibration_curve_raw",
                path=plot_info.get("plot_path", plots_dir / f"{plot_name}.png"),
                model=model_name,
                method=args.method,
            )
            output_manager.register(
                step="step2",
                artifact="plot_calibration_curve_raw_hist",
                path=plot_info.get("hist_path", plots_dir / f"{plot_name}_histogram.png"),
                model=model_name,
                method=args.method,
            )
        print("Calibration curve plots generated.")
        return
    print("\n" + "=" * 80)
    print("PRE-CALIBRATION DIAGNOSTICS")
    print("=" * 80)

    if val_preds:
        calibratability = check_calibratability(val_preds)

        print(f"\nCalibratability Check (Validation Set):")
        print(f"  Correlation: {calibratability['correlation']:.4f} ", end="")
        print(f"{'✓' if calibratability['correlation'] > 0.4 else '✗'}")

        print(f"  AUC-ROC: {calibratability['auc_roc']:.4f} ", end="")
        print(f"{'✓' if calibratability['auc_roc'] > 0.65 else '✗'}")

        print(f"  Monotonicity: {calibratability['monotonicity_score']:.4f} ", end="")
        print(f"{'✓' if calibratability['monotonicity_score'] > 0.6 else '✗'}")

        print(f"  Resolution (std): {calibratability['resolution_std']:.4f} ", end="")
        print(f"{'✓' if calibratability['resolution_std'] > 0.1 else '✗'}")

        print(
            f"\n  Overall: {'✓ CALIBRATABLE' if calibratability['is_calibratable'] else '✗ NOT CALIBRATABLE'}"
        )
        print(f"  Recommendation: {calibratability['recommendation']}")

    # ========================================================================
    # Raw Confidence Metrics
    # ========================================================================
    print("\n" + "=" * 80)
    print("RAW CONFIDENCE (Before Calibration)")
    print("=" * 80)

    if val_preds:
        val_metrics_raw = print_metrics(
            val_preds, "Validation Set (Raw)", use_calibrated=False
        )

    test_metrics_raw = print_metrics(test_preds, "Test Set (Raw)", use_calibrated=False)
    if args.compute_ci:
        ci_raw = _bootstrap_metric_cis(
            test_preds,
            use_calibrated=False,
            n_bootstrap=args.n_bootstrap,
            ci_method=args.ci_method,
            block=args.bootstrap_block,
            seed=args.bootstrap_seed,
        )
        test_metrics_raw.update(ci_raw)
        if ci_raw:
            print("\nBootstrap CIs (Raw, student-block):")
            for key in ("accuracy", "ece", "brier_score", "auc_roc", "mean_confidence"):
                low = ci_raw.get(f"{key}_ci_low")
                high = ci_raw.get(f"{key}_ci_high")
                if low is None or high is None:
                    continue
                print(f"  {key}: [{low:.4f}, {high:.4f}]")

    # Extract timestamp from csv path for saving metrics
    csv_timestamp = args.run_id if output_manager else csv_path.stem.split("_")[-1]

    # Save raw test metrics to CSV
    if not args.no_save_metrics:
        output_dir = output_manager.step_dir("step2", "tables") if output_manager else None
        filename_override = None
        if output_manager:
            filename_override = f"metrics_{model_name}_{args.method}_raw.csv"
        raw_metrics_path = save_metrics_to_csv(
            metrics=test_metrics_raw,
            model_name=model_name,
            method=args.method,
            timestamp=csv_timestamp,
            calibrator=args.calibrator,
            is_calibrated=False,
            output_dir=output_dir,
            filename_override=filename_override,
        )
        if output_manager:
            output_manager.register(
                step="step2",
                artifact="metrics_raw",
                path=raw_metrics_path,
                model=model_name,
                method=args.method,
            )
            if output_manager.config.save_legacy:
                save_metrics_to_csv(
                    metrics=test_metrics_raw,
                    model_name=model_name,
                    method=args.method,
                    timestamp=csv_timestamp,
                    calibrator=args.calibrator,
                    is_calibrated=False,
                    output_dir=None,
                    filename_override=None,
                )

    # ========================================================================
    # Apply Calibration
    # ========================================================================
    if args.calibrator != "none" and val_preds:
        print("\n" + "=" * 80)
        print(f"APPLYING CALIBRATION: {args.calibrator.upper()}")
        print("=" * 80)

        # Select calibrator
        if args.calibrator == "isotonic":
            calibrator = IsotonicCalibrator()
        elif args.calibrator == "temperature":
            calibrator = TemperatureScalingCalibrator()
        elif args.calibrator == "platt":
            calibrator = PlattScalingCalibrator()

        # Fit on validation set
        print(
            f"\nFitting {args.calibrator} calibrator on {len(val_preds)} validation instances..."
        )
        calibrator.fit(val_preds)

        if args.calibrator == "temperature":
            print(f"  Learned temperature: {calibrator.temperature:.4f}")

        # Apply to test set
        print(f"Applying calibration to {len(test_preds)} test instances...")
        test_preds_calibrated = calibrator.transform(test_preds)

        # ========================================================================
        # Calibrated Metrics
        # ========================================================================
        print("\n" + "=" * 80)
        print("CALIBRATED CONFIDENCE (After Calibration)")
        print("=" * 80)

        test_metrics_cal = print_metrics(
            test_preds_calibrated, "Test Set (Calibrated)", use_calibrated=True
        )
        if args.compute_ci:
            ci_cal = _bootstrap_metric_cis(
                test_preds_calibrated,
                use_calibrated=True,
                n_bootstrap=args.n_bootstrap,
                ci_method=args.ci_method,
                block=args.bootstrap_block,
                seed=args.bootstrap_seed,
            )
            test_metrics_cal.update(ci_cal)
            if ci_cal:
                print("\nBootstrap CIs (Calibrated, student-block):")
                for key in ("accuracy", "ece", "brier_score", "auc_roc", "mean_confidence"):
                    low = ci_cal.get(f"{key}_ci_low")
                    high = ci_cal.get(f"{key}_ci_high")
                    if low is None or high is None:
                        continue
                    print(f"  {key}: [{low:.4f}, {high:.4f}]")

        # Save calibrated test metrics to CSV
        if not args.no_save_metrics:
            output_dir = output_manager.step_dir("step2", "tables") if output_manager else None
            filename_override = None
            if output_manager:
                filename_override = (
                    f"metrics_{model_name}_{args.method}_calibrated_{args.calibrator}.csv"
                )
            cal_metrics_path = save_metrics_to_csv(
                metrics=test_metrics_cal,
                model_name=model_name,
                method=args.method,
                timestamp=csv_timestamp,
                calibrator=args.calibrator,
                is_calibrated=True,
                output_dir=output_dir,
                filename_override=filename_override,
            )
            if output_manager:
                output_manager.register(
                    step="step2",
                    artifact="metrics_calibrated",
                    path=cal_metrics_path,
                    model=model_name,
                    method=args.method,
                )
                if output_manager.config.save_legacy:
                    save_metrics_to_csv(
                        metrics=test_metrics_cal,
                        model_name=model_name,
                        method=args.method,
                        timestamp=csv_timestamp,
                        calibrator=args.calibrator,
                        is_calibrated=True,
                        output_dir=None,
                        filename_override=None,
                    )

        # ========================================================================
        # Comparison
        # ========================================================================
        print("\n" + "=" * 80)
        print("COMPARISON: Raw vs Calibrated")
        print("=" * 80)

        print(f"\n{'Metric':<25} {'Raw':<15} {'Calibrated':<15} {'Change':<15}")
        print("-" * 70)

        metrics_to_compare = [
            ("ECE", "ece", "lower"),
            ("Brier Score", "brier_score", "lower"),
            ("AUC-ROC", "auc_roc", "higher"),
            ("Spearman Correlation", "spearman_corr", "higher"),
            ("Acc@50%", "acc_at_50", "higher"),
            ("Acc@80%", "acc_at_80", "higher"),
        ]

        for name, key, direction in metrics_to_compare:
            raw_val = test_metrics_raw[key]
            cal_val = test_metrics_cal[key]
            change = cal_val - raw_val

            if direction == "lower":
                symbol = "✓" if change < 0 else "✗"
            else:
                symbol = "✓" if change > 0 else "✗"

            print(
                f"{name:<25} {raw_val:<15.4f} {cal_val:<15.4f} {change:+.4f} {symbol}"
            )

    # ========================================================================
    # Final Answer
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    ece_threshold = 0.1
    ece = test_metrics_raw["ece"]

    print(f"\nQuestion: Is {args.method} confidence well-calibrated?")
    print(f"\nRaw Confidence ECE: {ece:.4f}")

    if ece < 0.05:
        print("Answer: ✓ YES - Excellent calibration (ECE < 0.05)")
    elif ece < ece_threshold:
        print("Answer: ✓ GOOD - Acceptable calibration (ECE < 0.10)")
    elif ece < 0.20:
        print("Answer: ✗ POOR - Needs calibration (0.10 < ECE < 0.20)")
    else:
        print("Answer: ✗ VERY POOR - Strongly miscalibrated (ECE > 0.20)")

    print(f"\nIs it Calibratable?")
    if "calibratability" in test_metrics_raw:
        if test_metrics_raw["calibratability"]:
            print("  ✓ YES - Confidence is a good proxy for correctness")
        else:
            print("  ✗ NO - Confidence is a poor proxy for correctness")

    if not args.no_save_plots:
        plot_name = f"calibration_curve_raw_{model_name}_{args.method}"
        plots_dir = output_manager.step_dir("step2", "plots") if output_manager else None
        plot_info = plot_calibration_curve(
            test_preds,
            num_bins=10,
            use_calibrated=False,
            name=plot_name,
            model_name=model_name,
            method_name=args.method,
            output_dir=plots_dir,
        )
        if output_manager:
            output_manager.register(
                step="step2",
                artifact="plot_calibration_curve_raw",
                path=plot_info.get("plot_path", plots_dir / f"{plot_name}.png"),
                model=model_name,
                method=args.method,
            )
            output_manager.register(
                step="step2",
                artifact="plot_calibration_curve_raw_hist",
                path=plot_info.get("hist_path", plots_dir / f"{plot_name}_histogram.png"),
                model=model_name,
                method=args.method,
            )
    if args.calibrator != "none" and val_preds:
        ece_cal = test_metrics_cal["ece"]
        improvement = ece - ece_cal
        print(f"\nAfter {args.calibrator} calibration:")
        print(f"  Calibrated ECE: {ece_cal:.4f}")
        print(
            f"  Improvement: {improvement:.4f} ({improvement/ece*100:.1f}% reduction)"
        )

        if ece_cal < 0.10:
            print(f"  Result: ✓ Calibration successful!")
        else:
            print(f"  Result: ⚠ Still not well calibrated, try different method")
        if not args.no_save_plots:
            plot_name = f"calibration_curve_calibrated_{model_name}_{args.method}_{args.calibrator}"
            plots_dir = output_manager.step_dir("step2", "plots") if output_manager else None
            plot_info = plot_calibration_curve(
                test_preds_calibrated,
                num_bins=10,  # CHANGED: standardized to 10 bins
                use_calibrated=True,
                name=plot_name,
                model_name=model_name,
                method_name=args.method,
                output_dir=plots_dir,
            )
            if output_manager:
                output_manager.register(
                    step="step2",
                    artifact="plot_calibration_curve_calibrated",
                    path=plot_info.get("plot_path", plots_dir / f"{plot_name}.png"),
                    model=model_name,
                    method=args.method,
                )
                output_manager.register(
                    step="step2",
                    artifact="plot_calibration_curve_calibrated_hist",
                    path=plot_info.get("hist_path", plots_dir / f"{plot_name}_histogram.png"),
                    model=model_name,
                    method=args.method,
                )

    # Calibratability verdict
    if val_preds and "calibratability" in locals():
        print(
            f"\nCalibratable: {'✓ Yes' if calibratability['is_calibratable'] else '✗ No'}"
        )
        if not calibratability["is_calibratable"]:
            print("  → Self-reported confidence is a poor proxy for correctness")
            print("  → Consider using a different confidence estimation method")

    if output_manager:
        manifest_path = output_manager.write_manifest()
        if manifest_path:
            print(f"Manifest written to: {manifest_path}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
