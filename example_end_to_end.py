"""
End-to-end example: Complete confidence calibration pipeline.

This script demonstrates the full workflow:
1. Load dataset
2. Estimate confidence with multiple methods
3. Evaluate raw confidence scores
4. Apply calibration
5. Evaluate calibrated scores
6. Compare all methods
"""

import argparse
from pathlib import Path

# Data loading
from grading_calibration.datasets import load_ricechem

# LLM inference
from grading_calibration.inference.inference_llm import BatchedLLMInference

# Confidence methods
from grading_calibration.confidence import (
    TokenProbabilityConfidence,
    SelfReportedConfidence,
    SelfConsistencyConfidence,
    PerturbationConfidence,
)

# Calibration
from grading_calibration.calibration import IsotonicCalibrator

# Evaluation
from grading_calibration.metrics import evaluate_predictions, check_calibratability


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end confidence calibration example"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing RiceChem dataset",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Model arguments
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--backend", type=str, default="gpt", choices=["gpt", "hf", "vllm"]
    )

    # Confidence method
    parser.add_argument(
        "--method",
        type=str,
        default="token_probability",
        choices=[
            "token_probability",
            "self_reported",
            "self_consistency",
            "perturbation",
        ],
    )

    # Self-consistency parameters
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples for self-consistency",
    )

    # Perturbation parameters
    parser.add_argument(
        "--n_perturbations", type=int, default=5, help="Number of perturbations"
    )

    # Evaluation
    parser.add_argument(
        "--max_instances",
        type=int,
        default=100,
        help="Maximum instances to evaluate (for quick testing)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("END-TO-END CONFIDENCE CALIBRATION PIPELINE")
    print("=" * 80)

    # ========================================================================
    # Step 1: Load Dataset
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Loading Dataset")
    print("-" * 80)

    splits = load_ricechem(
        data_dir=args.data_dir,
        seed=args.seed,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )

    print(f"Train: {len(splits.train)} instances")
    print(f"Valid: {len(splits.valid)} instances")
    print(f"Test: {len(splits.test)} instances")

    # Limit for quick testing
    if args.max_instances:
        splits.valid = splits.valid[: args.max_instances]
        splits.test = splits.test[: args.max_instances]
        print(f"\nUsing {args.max_instances} instances for quick testing")

    # ========================================================================
    # Step 2: Initialize LLM
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Initializing LLM")
    print("-" * 80)

    engine = BatchedLLMInference(
        model_name=args.model, backend=args.backend, temperature=0.0
    )

    # ========================================================================
    # Step 3: Select Confidence Method
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Initializing Confidence Estimator")
    print("-" * 80)

    if args.method == "token_probability":
        estimator = TokenProbabilityConfidence(engine)
    elif args.method == "self_reported":
        estimator = SelfReportedConfidence(engine)
    elif args.method == "self_consistency":
        estimator = SelfConsistencyConfidence(
            engine, n_samples=args.n_samples, temperature=0.7
        )
    elif args.method == "perturbation":
        estimator = PerturbationConfidence(engine, n_perturbations=args.n_perturbations)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Using method: {estimator.get_name()}")

    # ========================================================================
    # Step 4: Estimate Confidence (Validation Set)
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Estimating Confidence on Validation Set")
    print("-" * 80)

    print(f"Processing {len(splits.valid)} validation instances...")
    val_predictions = estimator.estimate_batch(splits.valid, show_progress=True)

    # ========================================================================
    # Step 5: Pre-Calibration Verification
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Pre-Calibration Verification")
    print("-" * 80)

    calibratability = check_calibratability(val_predictions)

    print(f"Is calibratable: {calibratability['is_calibratable']}")
    print(f"Correlation: {calibratability['correlation']:.4f}")
    print(f"AUC-ROC: {calibratability['auc_roc']:.4f}")
    print(f"Monotonicity: {calibratability['monotonicity_score']:.4f}")
    print(f"Resolution: {calibratability['resolution_std']:.4f}")
    print(f"\nRecommendation: {calibratability['recommendation']}")

    if not calibratability["is_calibratable"]:
        print("\nWARNING: Confidence scores may not be calibratable!")
        print("Consider using a different confidence method.")

    # ========================================================================
    # Step 6: Evaluate Raw Confidence (Validation)
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 6: Evaluating Raw Confidence (Validation Set)")
    print("-" * 80)

    val_metrics_raw = evaluate_predictions(val_predictions, use_calibrated=False)

    print(f"Accuracy: {val_metrics_raw['accuracy']:.2%}")
    print(f"ECE: {val_metrics_raw['ece']:.4f}")
    print(f"Brier Score: {val_metrics_raw['brier_score']:.4f}")
    print(f"AUC-ROC: {val_metrics_raw['auc_roc']:.4f}")
    print(f"Spearman Corr: {val_metrics_raw['spearman_corr']:.4f}")
    print(f"Acc@50%: {val_metrics_raw['acc_at_50']:.2%}")
    print(f"Acc@80%: {val_metrics_raw['acc_at_80']:.2%}")
    print(f"Cov@5% risk: {val_metrics_raw['cov_at_5pct_risk']:.2%}")
    print(f"Cov@10% risk: {val_metrics_raw['cov_at_10pct_risk']:.2%}")

    # ========================================================================
    # Step 7: Apply Calibration
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 7: Applying Isotonic Regression Calibration")
    print("-" * 80)

    calibrator = IsotonicCalibrator()
    calibrator.fit(val_predictions)

    print("Calibrator fitted on validation set")

    # ========================================================================
    # Step 8: Estimate Confidence (Test Set)
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 8: Estimating Confidence on Test Set")
    print("-" * 80)

    print(f"Processing {len(splits.test)} test instances...")
    test_predictions = estimator.estimate_batch(splits.test, show_progress=True)

    # ========================================================================
    # Step 9: Calibrate Test Predictions
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 9: Calibrating Test Predictions")
    print("-" * 80)

    test_predictions_calibrated = calibrator.transform(test_predictions)

    print("Test predictions calibrated")

    # ========================================================================
    # Step 10: Evaluate Raw Confidence (Test)
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 10: Evaluating Raw Confidence (Test Set)")
    print("-" * 80)

    test_metrics_raw = evaluate_predictions(test_predictions, use_calibrated=False)

    print(f"Accuracy: {test_metrics_raw['accuracy']:.2%}")
    print(f"ECE: {test_metrics_raw['ece']:.4f}")
    print(f"Brier Score: {test_metrics_raw['brier_score']:.4f}")
    print(f"AUC-ROC: {test_metrics_raw['auc_roc']:.4f}")
    print(f"Spearman Corr: {test_metrics_raw['spearman_corr']:.4f}")

    # ========================================================================
    # Step 11: Evaluate Calibrated Confidence (Test)
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 11: Evaluating Calibrated Confidence (Test Set)")
    print("-" * 80)

    test_metrics_cal = evaluate_predictions(
        test_predictions_calibrated, use_calibrated=True
    )

    print(f"Accuracy: {test_metrics_cal['accuracy']:.2%}")
    print(f"ECE: {test_metrics_cal['ece']:.4f}")
    print(f"Brier Score: {test_metrics_cal['brier_score']:.4f}")
    print(f"AUC-ROC: {test_metrics_cal['auc_roc']:.4f}")
    print(f"Spearman Corr: {test_metrics_cal['spearman_corr']:.4f}")

    # ========================================================================
    # Step 12: Compare Raw vs Calibrated
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Raw vs Calibrated")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Raw':<15} {'Calibrated':<15} {'Improvement'}")
    print("-" * 65)

    metrics_to_compare = [
        ("ECE", "ece", "lower"),
        ("Brier Score", "brier_score", "lower"),
        ("AUC-ROC", "auc_roc", "higher"),
        ("Acc@50%", "acc_at_50", "higher"),
        ("Acc@80%", "acc_at_80", "higher"),
        ("Cov@5% risk", "cov_at_5pct_risk", "higher"),
        ("Cov@10% risk", "cov_at_10pct_risk", "higher"),
    ]

    for name, key, direction in metrics_to_compare:
        raw_val = test_metrics_raw[key]
        cal_val = test_metrics_cal[key]

        if direction == "lower":
            improvement = "✓" if cal_val < raw_val else "✗"
        else:
            improvement = "✓" if cal_val > raw_val else "✗"

        print(f"{name:<20} {raw_val:< 15.4f} {cal_val:<15.4f} {improvement}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
