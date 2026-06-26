"""
Validate and Fix Prediction CSVs for Non-Self-Consistency Methods
AND Filter Self-Consistency Samples

This script:
- For self_reported, token_probability, perturbation methods:
  1. Re-parsing all llm_response fields to ensure they're valid JSON
  2. Verifying prediction matches the parsed verdict
  3. Verifying confidence calculations are correct
  4. Fixing any inconsistencies and saving to {original_name}_fixed.csv

- For self_consistency method:
  1. (Default) Filters to keep only the first n samples per instance
     Saves filtered CSV to {original_name}_n{num_samples}.csv
  2. (With --validate_votes) Validates vote counts and recomputes if needed
     Saves fixed CSV to {original_name}_fixed.csv

Usage:
    # Method and model auto-detected from filename
    python validate_predictions_csv.py --csv predictions_llama8b_self_reported_20240101.csv
    python validate_predictions_csv.py --csv predictions_qwen4b_token_probability_20240101.csv
    python validate_predictions_csv.py --csv predictions_llama8b_self_consistency_20240101.csv --num_samples 10
    python validate_predictions_csv.py --csv predictions_llama8b_self_consistency_20240101.csv --validate_votes

    # Or specify method explicitly
    python validate_predictions_csv.py --csv path/to/predictions.csv --method self_reported
    python validate_predictions_csv.py --csv path/to/predictions.csv --method self_consistency --validate_votes
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import ast
import random
from collections import Counter
from typing import Dict, List, Tuple

from utils.helpers import parse_json_response
from experiments.run_exp.step2_analyze_calibration import load_path_by_index
from utils import find_csv_folder_from_root

PREVIEW_CHARS = 500


def parse_method_and_model_from_filename(csv_path: Path) -> Tuple[str, str]:
    """
    Parse method and model name from CSV filename.

    Expected format: predictions_{model_name}_{method}_{timestamp}.csv

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (method, model_name)
    """
    filename = csv_path.stem  # Remove .csv extension

    # Handle _fixed or _n{num} suffixes
    if filename.endswith('_fixed'):
        filename = filename[:-6]
    elif '_n' in filename:
        # Remove _n{num} suffix
        parts = filename.split('_n')
        if len(parts) > 1 and parts[-1].isdigit():
            filename = '_n'.join(parts[:-1])

    # Split by underscores
    for method in ["self_reported", "token_probability", "perturbation", "self_consistency"]:
        if f"_{method}_" in filename:
            parts = filename.split(f"_{method}_")
            model_name = parts[0].replace("predictions_", "")
            return method, model_name


    raise ValueError(
        f"Could not parse method and model_name from filename: {csv_path.name}\n"
        f"Expected format: predictions_{{model_name}}_{{method}}_{{timestamp}}.csv"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and fix predictions CSV files for non-self_consistency methods"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        choices=["self_reported", "token_probability", "perturbation", "self_consistency"],
        help="Confidence method name (auto-detected from CSV filename if not provided)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name used in the experiment (auto-detected from CSV filename if not provided)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of CSV file to load (if --csv not specified)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="predictions",
        help="Prefix for CSV filename (if --csv not specified)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="For self_consistency method: keep only the first n samples per instance (discards the rest)",
    )
    parser.add_argument(
        "--validate_votes",
        action="store_true",
        help="For self_consistency method: validate and fix vote counts instead of filtering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking in self_consistency (default: 42)",
    )
    parser.add_argument(
        "--delete_original",
        action="store_true",
        help="Delete the original CSV file after saving the fixed/filtered version (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for unknown verdicts (shows LLM responses that failed to parse)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Max characters to show when printing raw LLM responses (default: 500)",
    )
    return parser.parse_args()


def validate_and_fix_row_self_reported(
    row: pd.Series, row_idx: int
) -> Tuple[bool, int, float, bool, str]:
    """
    Validate and fix a row for self_reported method.

    For self_reported:
    - Prediction is based on verdict in llm_response (what the LLM said)
    - Confidence is what the LLM self-reported (1-10 scale, normalized to 0-1)
    - No tie case handling (LLM determines both prediction and confidence)

    Args:
        row: DataFrame row
        row_idx: Row index for error reporting

    Returns:
        Tuple of (is_fixed, new_prediction, new_confidence, new_is_correct, error_message)
        - is_fixed: True if row was valid or successfully fixed
        - new_prediction: Updated prediction (or original if no fix needed)
        - new_confidence: Updated confidence (or original if no fix needed)
        - new_is_correct: Updated is_correct (or original if no fix needed)
        - error_message: Error description if couldn't fix, empty string otherwise
    """
    # Parse llm_response
    try:
        response = str(row["llm_response"])
        parsed = parse_json_response(
            response,
            fallback_dict={"verdict": "unknown", "confidence": 5},
            preview_chars=PREVIEW_CHARS,
        )
        verdict = parsed.get("verdict", "unknown")
        confidence_raw = parsed.get("confidence", 5)

        # Normalize confidence to [0, 1] (assuming 1-10 scale)
        if isinstance(confidence_raw, (int, float)):
            new_confidence = max(min((float(confidence_raw) - 1) / 9, 1.0), 0.0)
        else:
            new_confidence = 0.5  # Default if parsing fails

        # Prediction determined by verdict
        if verdict == "yes":
            new_prediction = 1
        elif verdict == "no":
            new_prediction = 0
        else:
            # Unknown verdict - use current prediction
            new_prediction = int(row["prediction"])
            new_confidence = 0.5

        # Calculate is_correct
        new_is_correct = bool(new_prediction == row["ground_truth_label"])

        # Check if current values are correct
        current_prediction = int(row["prediction"])
        current_confidence = float(row["raw_confidence"])
        current_is_correct = bool(row["is_correct"])

        is_valid = (
            current_prediction == new_prediction
            and abs(current_confidence - new_confidence) < 0.01
            and current_is_correct == new_is_correct
        )

        if is_valid:
            return True, current_prediction, current_confidence, current_is_correct, ""
        else:
            return True, new_prediction, new_confidence, new_is_correct, ""

    except Exception as e:
        return (
            False,
            int(row["prediction"]),
            float(row["raw_confidence"]),
            bool(row["is_correct"]),
            f"Failed to parse response: {str(e)}"
        )


def validate_and_fix_row_token_probability(
    row: pd.Series, row_idx: int
) -> Tuple[bool, int, float, bool, str]:
    """
    Validate and fix a row for token_probability method.

    For token_probability:
    - Prediction is based on the verdict in llm_response (what the LLM actually said)
    - Confidence is the probability of the predicted token:
      - If verdict = "yes", confidence = yes_prob
      - If verdict = "no", confidence = no_prob
    - No tie case handling (prediction is always based on what LLM said)

    Args:
        row: DataFrame row
        row_idx: Row index for error reporting

    Returns:
        Tuple of (is_fixed, new_prediction, new_confidence, new_is_correct, error_message)
    """
    try:
        # Parse llm_response to get the actual verdict (what was predicted)
        response = str(row["llm_response"])
        parsed = parse_json_response(
            response,
            fallback_dict={"verdict": "unknown"},
            preview_chars=PREVIEW_CHARS,
        )
        verdict = parsed.get("verdict", "unknown")

        # Get token probabilities
        yes_prob = float(row.get("yes_prob", 0.0)) if pd.notna(row.get("yes_prob")) else None
        no_prob = float(row.get("no_prob", 0.0)) if pd.notna(row.get("no_prob")) else None

        # Determine prediction from verdict
        if verdict == "yes":
            new_prediction = 1
            # Confidence is the probability of "yes" token
            new_confidence = yes_prob if yes_prob is not None else 0.5
        elif verdict == "no":
            new_prediction = 0
            # Confidence is the probability of "no" token
            new_confidence = no_prob if no_prob is not None else 0.5
        else:
            # Unknown verdict - keep current prediction
            new_prediction = int(row["prediction"])
            new_confidence = float(row["raw_confidence"])

        # Calculate is_correct
        new_is_correct = bool(new_prediction == row["ground_truth_label"])

        # Check if current values are correct
        current_prediction = int(row["prediction"])
        current_confidence = float(row["raw_confidence"])
        current_is_correct = bool(row["is_correct"])

        
        is_valid = (
            current_prediction == new_prediction
            and abs(current_confidence - new_confidence) < 0.0000000001
            and current_is_correct == new_is_correct
        )

        if is_valid:
            return True, current_prediction, current_confidence, current_is_correct, ""
        else:
            return True, new_prediction, new_confidence, new_is_correct, ""

    except Exception as e:
        return (
            False,
            int(row["prediction"]),
            float(row["raw_confidence"]),
            bool(row["is_correct"]),
            f"Failed to validate: {str(e)}"
        )


def validate_and_fix_row_perturbation(
    row: pd.Series, row_idx: int
) -> Tuple[bool, int, float, bool, str]:
    """
    Validate and fix a row for perturbation method.

    For perturbation:
    - Prediction is based on verdict in llm_response
    - Confidence is computed from agreement across perturbations
    - No tie case handling (prediction based on what LLM said)

    Args:
        row: DataFrame row
        row_idx: Row index for error reporting

    Returns:
        Tuple of (is_fixed, new_prediction, new_confidence, new_is_correct, error_message)
    """
    try:
        # Parse llm_response
        response = str(row["llm_response"])
        parsed = parse_json_response(
            response,
            fallback_dict={"verdict": "unknown"},
            preview_chars=PREVIEW_CHARS,
        )
        verdict = parsed.get("verdict", "unknown")

        # Calculate prediction from verdict
        if verdict == "yes":
            new_prediction = 1
        elif verdict == "no":
            new_prediction = 0
        else:
            new_prediction = int(row["prediction"])

        # For perturbation, confidence is typically stored in raw_confidence
        # Just validate that prediction matches verdict
        new_confidence = float(row["raw_confidence"])

        # Calculate is_correct
        new_is_correct = bool(new_prediction == row["ground_truth_label"])

        # Check if current values are correct
        current_prediction = int(row["prediction"])
        current_is_correct = bool(row["is_correct"])

        is_valid = (
            current_prediction == new_prediction
            and current_is_correct == new_is_correct
        )

        if is_valid:
            return True, current_prediction, new_confidence, current_is_correct, ""
        else:
            return True, new_prediction, new_confidence, new_is_correct, ""

    except Exception as e:
        return (
            False,
            int(row["prediction"]),
            float(row["raw_confidence"]),
            bool(row["is_correct"]),
            f"Failed to parse response: {str(e)}"
        )


def validate_and_fix_row_self_consistency(
    row: pd.Series, llm_response_cols: List[str], row_idx: int, seed: int = 42, debug: bool = False
) -> Tuple[bool, Dict, float, int, bool, str]:
    """
    Validate and fix a row for self_consistency method.

    For self_consistency:
    - Validates that vote counts (yes + no) match the number of llm_response columns
    - Re-parses all responses if counts don't match
    - Recomputes vote_counts, raw_confidence, and prediction
    - Handles tie cases (confidence == 0.5) with random tie-breaking

    Args:
        row: DataFrame row
        llm_response_cols: List of llm_response column names
        row_idx: Row index for error reporting
        seed: Random seed for tie-breaking

    Returns:
        Tuple of (is_fixed, new_vote_counts, new_raw_confidence, new_prediction, new_is_correct, error_message)
        - is_fixed: True if row was valid or successfully fixed
        - new_vote_counts: Updated vote counts dict (or original if no fix needed)
        - new_raw_confidence: Updated confidence (or original if no fix needed)
        - new_prediction: Updated prediction (or original if no fix needed)
        - new_is_correct: Updated is_correct (or original if no fix needed)
        - error_message: Error description if couldn't fix, empty string otherwise
    """
    # Parse vote_counts from string
    try:
        vote_counts = ast.literal_eval(str(row["vote_counts"]))
        if not isinstance(vote_counts, dict):
            vote_counts = {}
    except (ValueError, SyntaxError):
        vote_counts = {}

    # Count non-null llm_response columns
    num_responses = sum(1 for col in llm_response_cols if pd.notna(row[col]))

    # Check if yes + no votes match num_responses
    yes_votes = vote_counts.get("yes", 0)
    no_votes = vote_counts.get("no", 0)
    yes_no_total = yes_votes + no_votes

    # Compute what confidence and prediction SHOULD be based on vote counts
    if yes_no_total > 0:
        expected_confidence = max(yes_votes, no_votes) / yes_no_total
        if expected_confidence > 0.5:
            # Clear majority
            expected_prediction = 1 if yes_votes > no_votes else 0
        elif expected_confidence == 0.5:
            # Tie - prediction can be either 0 or 1 (was randomly chosen)
            expected_prediction = int(row["prediction"])  # Keep whatever was chosen
        else:
            # This shouldn't happen in self_consistency
            expected_prediction = int(row["prediction"])
    else:
        expected_confidence = 0.5
        expected_prediction = int(row["prediction"])

    current_confidence = float(row["raw_confidence"])
    current_prediction = int(row["prediction"])

    # Check if everything is valid
    is_valid = (
        yes_no_total == num_responses  # Vote counts match total
        and abs(current_confidence - expected_confidence) < 0.0001  # Confidence is correct
        and (expected_confidence == 0.5 or current_prediction == expected_prediction)  # Prediction is correct (or tie)
    )

    if is_valid:
        # Valid - no fix needed
        return True, vote_counts, current_confidence, current_prediction, bool(row["is_correct"]), ""

    # Invalid - attempt to re-parse responses
    verdicts = []
    unknown_responses = []  # Track responses that failed to parse

    for col in llm_response_cols:
        response = row[col]
        if pd.notna(response):
            try:
                parsed = parse_json_response(
                    response,
                    fallback_dict={"verdict": "unknown"},
                    preview_chars=PREVIEW_CHARS,
                )
                verdict = parsed.get("verdict", "unknown")
                verdicts.append(verdict)

                # DEBUG: Track unknown verdicts
                if verdict == "unknown":
                    unknown_responses.append({
                        'column': col,
                        'response': response,
                        'parsed': parsed,
                        'instance_id': row['instance_id']
                    })
            except Exception as e:
                verdicts.append("unknown")
                unknown_responses.append({
                    'column': col,
                    'response': response,
                    'parsed': None,
                    'exception': str(e),
                    'instance_id': row['instance_id']
                })

    # Check if we got any verdicts
    if not verdicts:
        return (
            False,
            vote_counts,
            float(row["raw_confidence"]),
            int(row["prediction"]),
            bool(row["is_correct"]),
            f"No valid responses found (expected {num_responses})",
        )

    # Recompute vote counts and confidence
    new_vote_counts = dict(Counter(verdicts))
    majority_answer, majority_count = Counter(verdicts).most_common(1)[0]
    new_raw_confidence = majority_count / len(verdicts)

    # DEBUG: Print unknown verdicts if found and debug mode enabled
    if unknown_responses and debug:
        print(f"\n{'='*80}")
        print(f"WARNING: Found {len(unknown_responses)} 'unknown' verdicts for instance: {row['instance_id']}")
        print(f"Vote counts: {new_vote_counts}")
        print(f"{'='*80}")
        for i, unk in enumerate(unknown_responses, 1):
            print(f"\nUnknown #{i} (from {unk['column']}):")
            print(f"  Response text :")
            response_preview = unk['response'] if isinstance(unk['response'], str) else str(unk['response'])
            print(f"    {response_preview}...")
            print(f"  Parsed result: {unk.get('parsed', 'N/A')}")
            if 'exception' in unk:
                print(f"  Exception: {unk['exception']}")
        print(f"{'='*80}\n")

    # Check if there are any 'unknown' verdicts - if so, parsing failed and we couldn't fix
    if 'unknown' in new_vote_counts and new_vote_counts['unknown'] > 0:
        return (
            False,
            new_vote_counts,
            new_raw_confidence,
            int(row["prediction"]),
            bool(row["is_correct"]),
            f"Failed to parse {new_vote_counts['unknown']} out of {len(verdicts)} responses (verdict='unknown')"
        )

    # Determine prediction based on majority answer
    # If confidence is exactly 0.5 (tie), keep the original prediction (don't re-randomize)
    if new_raw_confidence == 0.5:
        new_prediction = int(row["prediction"])  # Keep original for ties
    else:
        new_prediction = 1 if majority_answer == "yes" else 0

    # Calculate is_correct
    new_is_correct = bool(new_prediction == row["ground_truth_label"])

    return True, new_vote_counts, new_raw_confidence, new_prediction, new_is_correct, ""


def filter_self_consistency_samples(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    Filter self-consistency CSV to keep only the first n samples per instance.

    Args:
        df: DataFrame with self-consistency predictions
        num_samples: Number of samples to keep per instance (must be > 0)

    Returns:
        Filtered DataFrame with only the first num_samples per instance
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    # Check for required columns
    if "instance_id" not in df.columns:
        raise ValueError("Input DataFrame must contain 'instance_id' column")

    # Get total samples per instance before filtering
    samples_per_instance = len([col for col in df.columns if col.startswith("llm_response_")])

    print(f"\nBefore filtering:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique instances: {df['instance_id'].nunique()}")
    print(f"  Samples per instance: {samples_per_instance}")

    if num_samples > samples_per_instance:
        print(f"\nWARNING: Requested {num_samples} samples, but some instances have fewer ({samples_per_instance})")
        print(f"         These instances will keep all their samples.")

    # Filter to keep only first num_samples per instance
    cols_to_drop = [col for col in df.columns if col.startswith("llm_response_") and int(col.split("_")[-1] ) >= num_samples]
    filtered_df = df.drop(columns=cols_to_drop)

    # Verify filtering
    samples_after = len([col for col in filtered_df.columns if col.startswith("llm_response_")])
    print(f"\nAfter filtering:")
    print(f"  Total rows: {len(filtered_df)}")
    print(f"  Unique instances: {filtered_df['instance_id'].nunique()}")
    print(f"  Samples per instance: {samples_after}")
    return filtered_df


def main():
    """Main validation and fixing logic."""
    args = parse_args()
    global PREVIEW_CHARS
    try:
        PREVIEW_CHARS = max(int(args.preview_chars), 0)
    except (TypeError, ValueError):
        PREVIEW_CHARS = 500

    # Auto-detect method and model_name from CSV filename if not provided
    if args.csv is not None:
        csv_path_temp = Path(args.csv)
        if not csv_path_temp.is_absolute():
            # If relative path, assume it's in the method subdirectory
            # Try to parse method from filename first
            try:
                parsed_method, parsed_model = parse_method_and_model_from_filename(csv_path_temp)
                if args.method is None:
                    args.method = parsed_method
                    print(f"Auto-detected method: {args.method}")
                if args.model_name is None:
                    args.model_name = parsed_model
                    print(f"Auto-detected model: {args.model_name}")
            except ValueError as e:
                if args.method is None:
                    print(f"Error: {e}")
                    print("Please provide --method explicitly")
                    sys.exit(1)
        else:
            # Absolute path - parse from filename
            try:
                parsed_method, parsed_model = parse_method_and_model_from_filename(csv_path_temp)
                if args.method is None:
                    args.method = parsed_method
                    print(f"Auto-detected method: {args.method}")
                if args.model_name is None:
                    args.model_name = parsed_model
                    print(f"Auto-detected model: {args.model_name}")
            except ValueError as e:
                if args.method is None:
                    print(f"Error: {e}")
                    print("Please provide --method explicitly")
                    sys.exit(1)
    else:
        # Using index-based loading - require method
        if args.method is None:
            print("Error: --method is required when not providing --csv")
            sys.exit(1)
        if args.model_name is None:
            args.model_name = "llama8b"  # Default

    # Load CSV
    directory = find_csv_folder_from_root("grading_calibration") / Path(args.method)
    if args.csv is None:
        # Load default CSV based on method
        csv_path = load_path_by_index(
            args.index,
            directory,
            f"{args.prefix}_{args.model_name}_{args.method}"
        )
    else:
        csv_path_input = Path(args.csv)
        if csv_path_input.is_absolute():
            csv_path = csv_path_input
        else:
            csv_path = directory / csv_path_input

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print("=" * 80)
    if args.method == 'self_consistency':
        mode = 'VOTE VALIDATION' if args.validate_votes else 'FILTERING'
        print(f"PREDICTIONS CSV {mode} - {args.method.upper()}")
    else:
        print(f"PREDICTIONS CSV VALIDATION - {args.method.upper()}")
    print("=" * 80)
    print(f"\nInput CSV: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Handle self_consistency method
    if args.method == "self_consistency":
        # Check which mode: filtering or vote validation
        if args.validate_votes:
            # Vote validation mode
            # Identify llm_response columns
            llm_response_cols = [col for col in df.columns if col.startswith("llm_response_")]
            print(f"Found {len(llm_response_cols)} llm_response columns")

            if not llm_response_cols:
                print("Error: No llm_response_* columns found in CSV")
                sys.exit(1)

            # Validate and fix each row
            print("\n" + "-" * 80)
            print("Validating vote counts...")
            print("-" * 80)

            invalid_rows = []
            fixed_rows = []
            unfixable_rows = []

            for idx, row in df.iterrows():
                is_fixed, new_vote_counts, new_raw_confidence, new_prediction, new_is_correct, error = validate_and_fix_row_self_consistency(
                    row, llm_response_cols, idx, args.seed, args.debug
                )

                # Check if row needed fixing by comparing returned values with original
                try:
                    original_vote_counts = ast.literal_eval(str(row["vote_counts"]))
                except (ValueError, SyntaxError):
                    original_vote_counts = {}

                current_prediction = int(row["prediction"])
                current_confidence = float(row["raw_confidence"])
                current_is_correct = bool(row["is_correct"])

                needs_fix = (
                    original_vote_counts != new_vote_counts
                    or abs(current_confidence - new_raw_confidence) > 0.0001
                    or current_prediction != new_prediction
                    or current_is_correct != new_is_correct
                )

                if needs_fix:
                    invalid_rows.append(idx)

                    if is_fixed:
                        # Update the DataFrame
                        df.at[idx, "vote_counts"] = str(new_vote_counts)
                        df.at[idx, "raw_confidence"] = new_raw_confidence
                        df.at[idx, "prediction"] = new_prediction
                        df.at[idx, "is_correct"] = new_is_correct
                        fixed_rows.append(idx)
                    else:
                        unfixable_rows.append((idx, error))

            # Print summary
            print("\n" + "=" * 80)
            print("VALIDATION SUMMARY")
            print("=" * 80)
            print(f"\nTotal rows: {len(df)}")
            print(f"Rows with invalid vote counts: {len(invalid_rows)}")
            print(f"Successfully fixed: {len(fixed_rows)}")
            print(f"Could not fix: {len(unfixable_rows)}")

            if unfixable_rows:
                print("\n" + "-" * 80)
                print("UNFIXABLE ROWS:")
                print("-" * 80)
                for row_idx, error in unfixable_rows[:10]:  # Show first 10
                    print(f"  Row {row_idx}: {error}")
                if len(unfixable_rows) > 10:
                    print(f"  ... and {len(unfixable_rows) - 10} more")

            # Save fixed CSV
            if len(fixed_rows) > 0:
                output_path = csv_path.parent / f"{csv_path.stem}_fixed.csv"
                df.to_csv(output_path, index=False)

                print(f"\n" + "=" * 80)
                print(f"Fixed CSV saved to: {output_path}")
                
                # Delete original and rename new file if requested
                if args.delete_original:
                    csv_path.unlink()
                    output_path.rename(csv_path)
                    print(f"Original CSV replaced: {csv_path}")
                
                print("=" * 80 + "\n")
            else:
                print("\nNo fixes were necessary. No new CSV created.\n")
            return  # Exit after vote validation

        else:
            # Filtering mode (default)

            if args.num_samples is None:
                print("\nError: --num_samples is required for self_consistency filtering")
                print("Usage: python validate_predictions_csv.py --csv path/to/predictions.csv --method self_consistency --num_samples 10")
                print("Or use --validate_votes to validate vote counts instead of filtering")
                sys.exit(1)

            # Filter samples
            filtered_df = filter_self_consistency_samples(df, args.num_samples)

            # Save filtered CSV
            output_path = csv_path.parent / f"{csv_path.stem}_n{args.num_samples}.csv"
            filtered_df.to_csv(output_path, index=False)

            print("\n" + "=" * 80)
            print(f"Filtered CSV saved to: {output_path}")
            
            # Delete original and rename new file if requested
            if args.delete_original:
                csv_path.unlink()
                output_path.rename(csv_path)
                print(f"Original CSV replaced: {csv_path}")
            
            print("=" * 80 + "\n")
            return  # Exit early for self_consistency filtering

    # Check for required columns
    required_cols = ["llm_response", "prediction", "raw_confidence", "is_correct", "ground_truth_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)

    # Select validation function based on method
    if args.method == "self_reported":
        validate_fn = validate_and_fix_row_self_reported
    elif args.method == "token_probability":
        validate_fn = validate_and_fix_row_token_probability
    else:  # perturbation
        validate_fn = validate_and_fix_row_perturbation

    # Validate and fix each row
    print("\n" + "-" * 80)
    print("Validating predictions...")
    print("-" * 80)

    invalid_rows = []
    fixed_rows = []
    unfixable_rows = []
    fixes_comparison = []  # Track before/after for comparison CSV

    for idx, row in df.iterrows():
        is_fixed, new_prediction, new_confidence, new_is_correct, error = validate_fn(row, idx)

        # Check if row needed fixing
        current_prediction = int(row["prediction"])
        current_confidence = float(row["raw_confidence"])
        current_is_correct = bool(row["is_correct"])

        needs_fix = (
            current_prediction != new_prediction
            or abs(current_confidence - new_confidence) > 0.00000001
            or current_is_correct != new_is_correct
        )

        if needs_fix:
            invalid_rows.append(idx)

            if is_fixed:
                # Track before/after for comparison
                fix_record = {
                    "row_index": idx,
                    "instance_id": row.get("instance_id", ""),
                    "before_prediction": current_prediction,
                    "after_prediction": new_prediction,
                    "before_confidence": current_confidence,
                    "after_confidence": new_confidence,
                    "before_is_correct": current_is_correct,
                    "after_is_correct": new_is_correct,
                    "ground_truth_label": row["ground_truth_label"],
                    "llm_response": row["llm_response"],  # Do not truncate for debug
                }
                fixes_comparison.append(fix_record)

                # Update the DataFrame
                df.at[idx, "prediction"] = new_prediction
                df.at[idx, "raw_confidence"] = new_confidence
                df.at[idx, "is_correct"] = new_is_correct
                fixed_rows.append(idx)
            else:
                unfixable_rows.append((idx, error))

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal rows: {len(df)}")
    print(f"Rows with issues: {len(invalid_rows)}")
    print(f"Successfully fixed: {len(fixed_rows)}")
    print(f"Could not fix: {len(unfixable_rows)}")

    if unfixable_rows:
        print("\n" + "-" * 80)
        print("UNFIXABLE ROWS:")
        print("-" * 80)
        for row_idx, error in unfixable_rows[:10]:  # Show first 10
            print(f"  Row {row_idx}: {error}")
        if len(unfixable_rows) > 10:
            print(f"  ... and {len(unfixable_rows) - 10} more")

    # Save fixed CSV
    print("\n" + "=" * 80   )
    print(len(fixed_rows))
    if len(fixed_rows) > 0:
        output_path = csv_path.parent / f"{csv_path.stem}_fixed.csv"
        df.to_csv(output_path, index=False)

        # Save comparison CSV showing before/after changes
        comparison_path = csv_path.parent / f"{csv_path.stem}_fixes_comparison.csv"
        comparison_df = pd.DataFrame(fixes_comparison)
        comparison_df.to_csv(comparison_path, index=False)

        print(f"\n" + "=" * 80)
        print(f"Fixed CSV saved to: {output_path}")
        print(f"Comparison CSV saved to: {comparison_path}")
        
        # Delete original and rename new file if requested
        if args.delete_original:
            csv_path.unlink()
            output_path.rename(csv_path)
            print(f"Original CSV replaced: {csv_path}")
        
        print("=" * 80 + "\n")
    else:
        print(f"\n" + "=" * 80)
        print("No fixes needed - all rows are valid!")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
