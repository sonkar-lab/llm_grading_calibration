#!/usr/bin/env python3
"""
Reparse Raw Responses Script

This script takes a raw responses JSONL file (saved by RawResponseSaver) and
attempts to reparse the responses using the current (potentially fixed) parsing logic.

Use this when:
1. Parsing crashed during inference due to a bug
2. You've fixed the parsing bug
3. You want to recover predictions without re-running expensive LLM inference

Usage:
    python reparse_raw_responses.py \
        --raw-responses raw_responses/self_consistency/raw_qwen80b_self_consistency_*.jsonl \
        --output reparsed_predictions.csv \
        --method self_consistency \
        --n-samples 5

The script will:
1. Load all raw responses from the JSONL file
2. Group responses by instance_id
3. Reparse each response using parse_json_response
4. For self_consistency: aggregate votes across samples
5. Output a CSV with predictions and confidence scores
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.helpers import parse_json_response
from utils.checkpoint_manager import RawResponseSaver


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reparse raw LLM responses to recover predictions"
    )
    parser.add_argument(
        "--raw-responses",
        type=str,
        required=True,
        help="Path to raw responses JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path for reparsed predictions",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["self_consistency", "self_reported", "token_probability"],
        help="Confidence method used",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples per instance (for self_consistency)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    return parser.parse_args()


def reparse_self_consistency(
    responses_by_instance: Dict[str, List[str]],
    n_samples: int,
    seed: int = 42,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Reparse self-consistency responses and aggregate votes.

    Args:
        responses_by_instance: Dict mapping instance_id -> list of responses
        n_samples: Expected number of samples per instance
        seed: Random seed for tie-breaking
        debug: Enable debug output

    Returns:
        List of result dicts with predictions and confidence
    """
    import random

    results = []

    for instance_id, responses in responses_by_instance.items():
        if debug:
            print(f"\nProcessing instance: {instance_id}")
            print(f"  Number of responses: {len(responses)}")

        # Parse each response
        verdicts = []
        raw_responses = []
        for i, response in enumerate(responses):
            parsed = parse_json_response(
                response,
                fallback_dict={"verdict": "unknown"},
                instance_id=f"{instance_id}_sample{i}",
            )
            verdict = parsed.get("verdict", "unknown")
            verdicts.append(verdict)
            raw_responses.append(response)

            if debug:
                print(f"  Sample {i}: verdict={verdict}")

        # Count votes
        vote_counts = Counter(verdicts)
        total_votes = len(verdicts)

        if debug:
            print(f"  Vote counts: {dict(vote_counts)}")

        # Find majority answer
        majority_answer, majority_count = vote_counts.most_common(1)[0]
        confidence = majority_count / total_votes

        # Handle ties
        is_tie = confidence == 0.5
        if is_tie:
            tied_answers = [ans for ans, count in vote_counts.most_common(2)]
            if len(tied_answers) == 2:
                instance_seed = seed + hash(instance_id) % 1000000
                random.seed(instance_seed)
                majority_answer = random.choice(tied_answers)

        # Determine prediction
        if majority_answer == "unknown":
            # If majority is unknown, randomly pick
            instance_seed = seed + hash(instance_id) % 1000000
            random.seed(instance_seed)
            prediction = random.choice([0, 1])
        else:
            prediction = 1 if majority_answer == "yes" else 0

        result = {
            "instance_id": instance_id,
            "prediction": prediction,
            "raw_confidence": confidence,
            "vote_counts": str(dict(vote_counts)),
            "majority_answer": majority_answer,
            "is_tie": is_tie,
            "n_responses": len(responses),
            "n_unknown": vote_counts.get("unknown", 0),
        }

        # Add individual responses
        for i, resp in enumerate(raw_responses):
            result[f"llm_response_{i}"] = resp

        results.append(result)

        if debug:
            print(f"  -> prediction={prediction}, confidence={confidence:.3f}")

    return results


def reparse_single_response(
    responses_by_instance: Dict[str, List[str]],
    seed: int = 42,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Reparse single-response methods (self_reported, token_probability).

    Args:
        responses_by_instance: Dict mapping instance_id -> list of responses (should be 1 each)
        seed: Random seed for tie-breaking
        debug: Enable debug output

    Returns:
        List of result dicts with predictions and confidence
    """
    import random

    results = []

    for instance_id, responses in responses_by_instance.items():
        if len(responses) == 0:
            print(f"Warning: No responses for instance {instance_id}")
            continue

        response = responses[0]  # Take first response

        if debug:
            print(f"\nProcessing instance: {instance_id}")
            print(f"  Response (first 200 chars): {response[:200]}")

        # Parse response
        parsed = parse_json_response(
            response,
            fallback_dict={"verdict": "unknown", "confidence": 0.5},
            instance_id=instance_id,
        )

        verdict = parsed.get("verdict", "unknown")
        confidence = parsed.get("confidence", 0.5)

        # Normalize confidence to [0, 1]
        if isinstance(confidence, str):
            try:
                confidence = float(confidence.replace("%", "")) / 100
            except:
                confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        # Handle unknown verdict
        if verdict == "unknown":
            instance_seed = seed + hash(instance_id) % 1000000
            random.seed(instance_seed)
            prediction = random.choice([0, 1])
            confidence = 0.5
        else:
            prediction = 1 if verdict == "yes" else 0

        result = {
            "instance_id": instance_id,
            "prediction": prediction,
            "raw_confidence": confidence,
            "verdict": verdict,
            "llm_response": response,
            "parse_success": verdict != "unknown",
        }

        results.append(result)

        if debug:
            print(f"  -> verdict={verdict}, prediction={prediction}, confidence={confidence:.3f}")

    return results


def main():
    args = parse_args()

    print("=" * 80)
    print("REPARSE RAW RESPONSES")
    print("=" * 80)

    # Load raw responses
    raw_path = Path(args.raw_responses)
    if not raw_path.exists():
        print(f"Error: Raw responses file not found: {raw_path}")
        sys.exit(1)

    print(f"\nLoading raw responses from: {raw_path}")
    records = RawResponseSaver.load_responses(raw_path)
    print(f"Loaded {len(records)} raw response records")

    # Group by instance
    responses_by_instance = RawResponseSaver.group_by_instance(records)
    print(f"Found {len(responses_by_instance)} unique instances")

    # Sample check
    sample_instance = list(responses_by_instance.keys())[0]
    sample_responses = responses_by_instance[sample_instance]
    print(f"\nSample instance: {sample_instance}")
    print(f"  Number of responses: {len(sample_responses)}")
    print(f"  First response (first 200 chars): {sample_responses[0][:200]}...")

    # Reparse based on method
    print(f"\nReparsing with method: {args.method}")

    if args.method == "self_consistency":
        results = reparse_self_consistency(
            responses_by_instance,
            n_samples=args.n_samples,
            seed=args.seed,
            debug=args.debug,
        )
    else:
        results = reparse_single_response(
            responses_by_instance,
            seed=args.seed,
            debug=args.debug,
        )

    # Convert to DataFrame and save
    import pandas as pd

    df = pd.DataFrame(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nReparsed {len(df)} instances")
    print(f"Output saved to: {output_path}")

    # Statistics
    print(f"\nPrediction distribution:")
    print(f"  Positive (1): {(df['prediction'] == 1).sum()} ({(df['prediction'] == 1).mean():.1%})")
    print(f"  Negative (0): {(df['prediction'] == 0).sum()} ({(df['prediction'] == 0).mean():.1%})")

    print(f"\nConfidence distribution:")
    print(f"  Mean: {df['raw_confidence'].mean():.3f}")
    print(f"  Std: {df['raw_confidence'].std():.3f}")
    print(f"  Min: {df['raw_confidence'].min():.3f}")
    print(f"  Max: {df['raw_confidence'].max():.3f}")

    if args.method == "self_consistency":
        print(f"\nUnknown verdicts:")
        print(f"  Instances with unknown: {(df['n_unknown'] > 0).sum()}")
        print(f"  Total unknown responses: {df['n_unknown'].sum()}")
    else:
        print(f"\nParse success rate: {df['parse_success'].mean():.1%}")

    print(f"\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
