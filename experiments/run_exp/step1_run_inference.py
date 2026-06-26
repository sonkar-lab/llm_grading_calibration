"""
Step 1: Run LLM Inference with Self-Reported Confidence

This script:
1. Loads dataset (ricechem/scientsbank/beetle)
2. Runs self-reported confidence prompts on LLM
3. Saves predictions to CSV for later analysis

Usage:
    cd experiments/self_reported
    python step1_run_inference.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --backend vllm \
        --data_dir ../../dataset/RiceChem_Dataset \
        --output predictions_qwen.csv \
        --max_instances 100
"""

import sys
from pathlib import Path

# Add project root to Python path
print(f"file is: {__file__}")
project_root = Path(__file__).resolve().parent.parent.parent
print(f"Adding project root to path: {project_root}")
sys.path.insert(0, str(project_root))
import pandas as pd
from datetime import datetime
import torch
from time import time
import shutil

from grading_calibration.inference.inference_llm import BatchedLLMInference
from experiments.sample_args import (
    initialize_inference_engine,
    parse_arguments as parse_args,
)
from grading_calibration.confidence import (
    SelfReportedConfidence,
    SelfConsistencyConfidence,
    TokenProbabilityConfidence,
)
from utils import set_seed, find_csv_folder_from_root
from utils.checkpoint_manager import CheckpointManager, LogManager, RawResponseSaver
from utils.output_paths import build_output_manager, default_run_id
import os
from grading_calibration.datasets import load_ricechem, load_scientsbank, load_beetle

confidence_method_type = (
    SelfReportedConfidence | SelfConsistencyConfidence | TokenProbabilityConfidence
)

print("CUDA_VISIBLE_DEVICES in Python:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Logical GPU {i}:", torch.cuda.get_device_name(i))


def get_confidence_method(
    method_name: str,
    inference_engine: BatchedLLMInference,
    seed: int,
    confidence_scale: float,
    n_samples: int,
) -> confidence_method_type:
    if method_name == "self_reported":
        return SelfReportedConfidence(
            inference_engine=inference_engine,
            seed=seed,
            confidence_scale=confidence_scale,
        )
    elif method_name == "self_consistency":
        return SelfConsistencyConfidence(
            inference_engine=inference_engine, seed=seed, n_samples=n_samples
        )
    elif method_name == "token_probability":
        return TokenProbabilityConfidence(inference_engine=inference_engine, seed=seed)
    else:
        raise ValueError(f"Unknown confidence method: {method_name}")


def main():
    start = time()
    args = parse_args()

    if args.smoke_n is not None:
        if args.max_instances is None:
            args.max_instances = args.smoke_n
        else:
            args.max_instances = min(args.max_instances, args.smoke_n)
        print(f"Smoke test: limiting to first {args.max_instances} instances")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup directories
    directory = find_csv_folder_from_root("grading_calibration") / Path(args.method)
    checkpoint_dir = directory / "checkpoints"

    output_manager = None
    run_id = None
    if args.output_root:
        if args.resume and not args.run_id:
            raise ValueError("--run-id is required when using --output-root with --resume")
        run_id = args.run_id or default_run_id()
        output_manager = build_output_manager(
            output_root=args.output_root,
            dataset=args.dataset,
            run_id=run_id,
            save_legacy=args.save_legacy,
            write_manifest=args.write_manifest,
        )
        checkpoint_dir = output_manager.step_dir("step1", "intermediate") / "checkpoints"

    print("=" * 80)
    print(f"STEP 1: LLM INFERENCE WITH {args.method.upper()} CONFIDENCE")
    print("=" * 80)

    # Setup checkpoint manager
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        method=args.method,
        dataset=args.dataset,
        timestamp=run_id,
    )

    # Check for resume
    completed_ids = set()
    if args.resume:
        timestamp, completed_ids = checkpoint_manager.find_latest_checkpoint()
        if timestamp:
            checkpoint_manager.timestamp = timestamp
            if run_id and timestamp != run_id:
                print(
                    f"Warning: checkpoint timestamp ({timestamp}) does not match --run-id "
                    f"({run_id}); continuing with checkpoint timestamp."
                )
        print(f"Using timestamp: {checkpoint_manager.timestamp}")
    else:
        print(f"Starting new run with timestamp: {checkpoint_manager.timestamp}")

    # Setup log manager for parse errors
    experiments_dir = find_csv_folder_from_root("grading_calibration").parent
    if output_manager:
        experiments_dir = output_manager.step_root("step1")
    log_manager = LogManager(
        base_dir=experiments_dir,
        model_name=model_name,
        method=args.method,
        timestamp=checkpoint_manager.timestamp,
    )
    print(f"Log file: {log_manager.get_log_path()}")

    # Setup raw response saver - saves responses BEFORE parsing to allow recovery
    raw_response_saver = RawResponseSaver(
        base_dir=experiments_dir,
        model_name=model_name,
        method=args.method,
        timestamp=checkpoint_manager.timestamp,
    )
    print(f"Raw responses file: {raw_response_saver.get_file_path()}")

    # ========================================================================
    # Load Dataset
    # ========================================================================
    print("\n" + "-" * 80)
    print(f"Loading dataset: {args.dataset}")
    print("-" * 80)

    if args.dataset == "ricechem":
        if not args.data_dir:
            raise ValueError("--data_dir is required for ricechem")
        splits = load_ricechem(
            data_dir=args.data_dir,
            seed=args.seed,
            train_size=args.train_size,
            valid_size=args.valid_size,
            test_size=args.test_size,
        )
    else:
        if args.data_dir:
            print("Note: --data_dir ignored for HuggingFace datasets.")
        if any(
            abs(val - target) > 1e-6
            for val, target in [
                (args.train_size, 1.0),
                (args.valid_size, 0.0),
                (args.test_size, 0.0),
            ]
        ):
            print("Note: HF datasets ignore split args; all instances placed in train.")
        if args.dataset == "scientsbank":
            splits = load_scientsbank(seed=args.seed)
        elif args.dataset == "beetle":
            splits = load_beetle(seed=args.seed)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Train: {len(splits.train)} instances")
    print(f"Valid: {len(splits.valid)} instances")
    print(f"Test: {len(splits.test)} instances")
    print(
        f"Total: {len(splits.train) + len(splits.valid) + len(splits.test)} instances"
    )

    # Combine all splits for inference
    all_instances = splits.train + splits.valid + splits.test

    # Track split membership
    split_labels = (
        ["train"] * len(splits.train)
        + ["valid"] * len(splits.valid)
        + ["test"] * len(splits.test)
    )

    instance_ids = [instance.id for instance in all_instances]
    print(f"\nHow many unique instance IDs? {len(set(instance_ids))}")
    assert len(set(instance_ids)) == len(all_instances), "Duplicate instance IDs found!"
    # Limit if requested
    if args.max_instances:
        all_instances = all_instances[: args.max_instances]
        split_labels = split_labels[: args.max_instances]
        print(f"\nLimited to {args.max_instances} instances for testing")

        
    # Filter out completed instances if resuming
    if completed_ids:
        filtered_instances = []
        filtered_labels = []
        for instance, label in zip(all_instances, split_labels):
            if instance.id not in completed_ids:
                filtered_instances.append(instance)
                filtered_labels.append(label)

        print(f"\nFiltered out {len(completed_ids)} completed instances")
        print(f"Remaining instances to process: {len(filtered_instances)}")
        all_instances = filtered_instances
        split_labels = filtered_labels

    

    # ========================================================================
    # Initialize LLM
    # ========================================================================
    print("\n" + "-" * 80)
    print("Initializing LLM")
    print("-" * 80)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")

    engine = initialize_inference_engine(args, args.backend)

    # ========================================================================
    # Initialize Confidence Estimator
    # ========================================================================
    print("\n" + "-" * 80)
    print("Initializing Self-Reported Confidence Estimator")
    print("-" * 80)

    estimator = get_confidence_method(
        method_name=args.method,
        inference_engine=engine,
        seed=args.seed,
        confidence_scale=args.confidence_scale,
        n_samples=args.n_samples,
    )

    print(
        "Prompt format: 'Grade this response and provide confidence (1-10 scale)'"
        if args.method == "self_reported"
        else (
            f"Self-consistency with {args.n_samples} samples"
            if args.method == "self_consistency"
            else "Token probability-based confidence, using (yes/no) tokens"
        )
    )

    # ========================================================================
    # Run Inference
    # ========================================================================
    print("\n" + "-" * 80)
    print(f"Running Inference on {len(all_instances)} Instances")
    print("-" * 80)
    print("This may take a while...")

    # Setup checkpoint callback if needed
    if args.checkpoint_chunk_size:
        save_checkpoint = checkpoint_manager.create_checkpoint_callback(
            all_instances=all_instances, split_labels=split_labels, args=args
        )
        estimator_kwargs = {
            "instances": all_instances,
            "show_progress": True,
            "checkpoint_callback": save_checkpoint,
            "checkpoint_chunk_size": args.checkpoint_chunk_size,
            "log_manager": log_manager,
            "raw_response_saver": raw_response_saver,
        }
    else:
        estimator_kwargs = {
            "instances": all_instances,
            "show_progress": True,
            "log_manager": log_manager,
            "raw_response_saver": raw_response_saver,
        }

    if args.method == "token_probability" and args.top_k_probability is not None:
        estimator_kwargs["top_k_log_probs"] = args.top_k_probability

    predictions = estimator.estimate(**estimator_kwargs)

    print(f"\nInference complete! Processed {len(predictions)} instances")

    # ========================================================================
    # Prepare Results
    # ========================================================================
    print("\n" + "-" * 80)
    print("Preparing Results")
    print("-" * 80)

    results = []
    for instance, pred, split in zip(all_instances, predictions, split_labels):
        result = {
            "instance_id": instance.id,
            "question": instance.question,
            "student_answer": instance.student_answer,
            "rubric_item": instance.rubric_item,
            "answer_key": instance.answer_key if instance.answer_key else "",
            "ground_truth_label": instance.ground_truth_label,
            "prediction": pred.prediction,
            "raw_confidence": pred.raw_confidence,
            "is_correct": pred.is_correct(),
            "split": split,
            "question_id": instance.metadata.get("question_id", ""),
            "student_id": instance.metadata.get("student_id", ""),
            "rubric_id": instance.metadata.get("rubric_id", ""),
        }

        if args.method == "self_consistency":
            # Add vote counts for self-consistency
            result["vote_counts"] = pred.metadata.get("vote_counts", {})
            for i in range(args.n_samples):
                result[f"llm_response_{i}"] = pred.metadata.get(f"raw_response_{i}", "")
        else:
            result["llm_response"] = pred.metadata.get("raw_response", "")

        if args.method == "token_probability":
            result["yes_prob"] = pred.metadata.get("yes_prob", None)
            result["no_prob"] = pred.metadata.get("no_prob", None)

        results.append(result)

    df = pd.DataFrame(results)

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "-" * 80)
    print("Saving Results")
    print("-" * 80)

    # Determine output filename
    if args.output is None:
        if output_manager:
            output_file = f"predictions_{model_name}_{args.method}.csv"
        else:
            output_file = (
                f"predictions_{model_name}_{args.method}_{args.dataset}_{checkpoint_manager.timestamp}.csv"
            )
    else:
        output_file = args.output

    output_path = directory / output_file
    if output_manager:
        output_path = output_manager.step_dir("step1", "tables") / output_file
    os.makedirs(directory, exist_ok=True)

    # If checkpointing was used, consolidate checkpoints
    if args.checkpoint_chunk_size:
        print(f"\nConsolidating checkpoints...")
        checkpoint_manager.consolidate_checkpoints(output_path)
    else:
        # Direct save (no checkpointing)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    if output_manager:
        output_manager.register(
            step="step1",
            artifact="predictions",
            path=output_path,
            model=model_name,
            method=args.method,
        )
        if output_manager.config.save_legacy:
            legacy_file = (
                f"predictions_{model_name}_{args.method}_{args.dataset}_{checkpoint_manager.timestamp}.csv"
                if args.output is None
                else output_file
            )
            legacy_path = directory / legacy_file
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            if args.checkpoint_chunk_size:
                shutil.copy2(output_path, legacy_path)
            else:
                df.to_csv(legacy_path, index=False)
            print(f"Legacy results saved to: {legacy_path}")
        manifest_path = output_manager.write_manifest()
        if manifest_path:
            print(f"Manifest written to: {manifest_path}")

    print(f"Total instances: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    end = time()
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    if len(df) == 0:
        print("No new instances were processed. Exiting.")
        return
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nRaw responses saved: {raw_response_saver.get_write_count()}")
    print(f"Raw responses file: {raw_response_saver.get_file_path()}")

    print(f"\nDataset: {len(df)} instances")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Valid: {len(df[df['split'] == 'valid'])}")
    print(f"  Test: {len(df[df['split'] == 'test'])}")

    print(f"\nOverall Accuracy: {df['is_correct'].mean():.2%}")

    print(f"\nConfidence Distribution:")
    print(f"  Mean: {df['raw_confidence'].mean():.3f}")
    print(f"  Std: {df['raw_confidence'].std():.3f}")
    print(f"  Min: {df['raw_confidence'].min():.3f}")
    print(f"  Max: {df['raw_confidence'].max():.3f}")

    print(f"\nPredictions:")
    print(
        f"  Positive (1): {(df['prediction'] == 1).sum()} ({(df['prediction'] == 1).mean():.1%})"
    )
    print(
        f"  Negative (0): {(df['prediction'] == 0).sum()} ({(df['prediction'] == 0).mean():.1%})"
    )

    print(f"\n{'='*80}")
    print("✓ Inference complete! Now run step 2:")
    print(f"  python step2_analyze_calibration.py --predictions {output_file}")
    print(f"Total time taken: {end - start:.2f} seconds")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
