#!/usr/bin/env python3
"""
Standalone script to validate that confidence values in CSV match LLM responses.

Usage:
    python validate_confidence_csv.py --csv path/to/predictions.csv --method self_reported
    python validate_confidence_csv.py --csv path/to/predictions.csv --method token_probability
    python validate_confidence_csv.py --csv path/to/predictions.csv --method self_consistency
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import re
import json
import ast
from collections import Counter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Validate CSV confidence values against LLM responses")
    parser.add_argument("--csv", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--method", type=str, required=True, 
                       choices=["self_reported", "token_probability", "self_consistency"],
                       help="Confidence estimation method")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to check")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Tolerance for confidence mismatch")
    parser.add_argument("--verbose", action="store_true", help="Show all mismatches in detail")
    return parser.parse_args()


def extract_last_json(response: str) -> dict | None:
    """
    Extract the LAST valid JSON object from a response.
    
    This mirrors the logic in utils/helpers.py _extract_json_snippet.
    Important for thinking models (e.g., Qwen3-thinking) that may output
    multiple JSON-like strings in their reasoning process.
    """
    # Find all brace positions
    brace_positions = [i for i, c in enumerate(response) if c == "{"]
    
    if not brace_positions:
        return None
    
    # Try from last brace working backwards
    decoder = json.JSONDecoder()
    for start in reversed(brace_positions):
        try:
            obj, _ = decoder.raw_decode(response[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    
    return None


def validate_self_reported(df: pd.DataFrame, tolerance: float, verbose: bool) -> dict:
    """Validate self_reported confidence: parse JSON and check (confidence - 1) / 9."""
    errors = []
    warnings = []
    n_checked = 0
    
    print(f"\nValidating self_reported confidence...")
    print(f"Expected: raw_confidence = (parsed_confidence - 1) / 9  (scale 1-10)")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking rows"):
        llm_response = row.get('llm_response', '')
        stored_conf = row.get('raw_confidence')
        
        if pd.isna(stored_conf):
            warnings.append(f"Row {idx}: Missing raw_confidence")
            continue
        
        if not llm_response or pd.isna(llm_response):
            warnings.append(f"Row {idx}: No llm_response")
            continue
        
        try:
            # Find LAST JSON in response (important for thinking models!)
            data = extract_last_json(str(llm_response))
            if data:
                parsed_conf_raw = float(data.get('confidence', 5))
                
                # Normalize: (raw - 1) / (scale - 1) where scale = 10
                parsed_conf = (parsed_conf_raw - 1) / 9.0
                parsed_conf = max(0.0, min(1.0, parsed_conf))
                
                diff = abs(stored_conf - parsed_conf)
                if diff > tolerance:
                    errors.append({
                        'row': idx,
                        'instance_id': row.get('instance_id', 'N/A'),
                        'stored_conf': stored_conf,
                        'parsed_conf': parsed_conf,
                        'parsed_raw': parsed_conf_raw,
                        'diff': diff,
                        'response': str(llm_response)[:300]
                    })
                n_checked += 1
            else:
                warnings.append(f"Row {idx}: No JSON found in response")
        except Exception as e:
            warnings.append(f"Row {idx}: Parse error - {e}")
    
    return {'errors': errors, 'warnings': warnings, 'n_checked': n_checked}


def validate_token_probability(df: pd.DataFrame, tolerance: float, verbose: bool) -> dict:
    """Validate token_probability confidence: check yes_prob/no_prob alignment."""
    errors = []
    warnings = []
    n_checked = 0
    
    print(f"\nValidating token_probability confidence...")
    print(f"Expected: raw_confidence = yes_prob if prediction=1 else no_prob")
    
    # Check required columns
    has_yes_prob = 'yes_prob' in df.columns
    has_no_prob = 'no_prob' in df.columns
    
    if not has_yes_prob or not has_no_prob:
        print(f"⚠ Missing columns: yes_prob={has_yes_prob}, no_prob={has_no_prob}")
        print(f"  Available columns: {list(df.columns)}")
        return {'errors': [], 'warnings': ['Missing yes_prob/no_prob columns'], 'n_checked': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking rows"):
        stored_conf = row.get('raw_confidence')
        yes_prob = row.get('yes_prob')
        no_prob = row.get('no_prob')
        prediction = row.get('prediction')
        
        if pd.isna(stored_conf) or pd.isna(yes_prob) or pd.isna(no_prob) or pd.isna(prediction):
            warnings.append(f"Row {idx}: Missing required values")
            continue
        
        # Expected confidence based on prediction
        expected_conf = yes_prob if int(prediction) == 1 else no_prob
        
        diff = abs(stored_conf - expected_conf)
        if diff > tolerance:
            errors.append({
                'row': idx,
                'instance_id': row.get('instance_id', 'N/A'),
                'stored_conf': stored_conf,
                'expected_conf': expected_conf,
                'yes_prob': yes_prob,
                'no_prob': no_prob,
                'prediction': prediction,
                'diff': diff
            })
        n_checked += 1
    
    return {'errors': errors, 'warnings': warnings, 'n_checked': n_checked}


def validate_self_consistency(df: pd.DataFrame, tolerance: float, verbose: bool) -> dict:
    """Validate self_consistency confidence: check vote_counts alignment."""
    errors = []
    warnings = []
    n_checked = 0
    
    print(f"\nValidating self_consistency confidence...")
    print(f"Expected: raw_confidence = majority_count / total_votes")
    
    # Check for vote_counts column
    has_vote_counts = 'vote_counts' in df.columns
    
    if not has_vote_counts:
        print(f"⚠ Missing vote_counts column")
        print(f"  Available columns: {list(df.columns)}")
        return {'errors': [], 'warnings': ['Missing vote_counts column'], 'n_checked': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking rows"):
        stored_conf = row.get('raw_confidence')
        vote_counts_str = row.get('vote_counts')
        
        if pd.isna(stored_conf):
            warnings.append(f"Row {idx}: Missing raw_confidence")
            continue
        
        if pd.isna(vote_counts_str) or not vote_counts_str:
            warnings.append(f"Row {idx}: Missing vote_counts")
            continue
        
        try:
            # Parse vote_counts (stored as string representation of dict)
            vote_counts = ast.literal_eval(str(vote_counts_str))
            
            if not isinstance(vote_counts, dict) or not vote_counts:
                warnings.append(f"Row {idx}: Empty or invalid vote_counts")
                continue
            
            total_votes = sum(vote_counts.values())
            majority_count = max(vote_counts.values())
            expected_conf = majority_count / total_votes
            
            diff = abs(stored_conf - expected_conf)
            if diff > tolerance:
                errors.append({
                    'row': idx,
                    'instance_id': row.get('instance_id', 'N/A'),
                    'stored_conf': stored_conf,
                    'expected_conf': expected_conf,
                    'vote_counts': vote_counts,
                    'total_votes': total_votes,
                    'majority_count': majority_count,
                    'diff': diff
                })
            n_checked += 1
            
        except Exception as e:
            warnings.append(f"Row {idx}: Parse error - {e}")
    
    return {'errors': errors, 'warnings': warnings, 'n_checked': n_checked}


def main():
    args = parse_args()
    
    print("=" * 80)
    print("CSV CONFIDENCE VALIDATION")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} rows")
    
    # Validate based on method
    if args.method == 'self_reported':
        result = validate_self_reported(df, args.tolerance, args.verbose)
    elif args.method == 'token_probability':
        result = validate_token_probability(df, args.tolerance, args.verbose)
    elif args.method == 'self_consistency':
        result = validate_self_consistency(df, args.tolerance, args.verbose)
    else:
        print(f"Unknown method: {args.method}")
        sys.exit(1)
    
    # Report results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    errors = result['errors']
    warnings = result['warnings']
    n_checked = result['n_checked']
    
    print(f"\nInstances checked: {n_checked}")
    print(f"Errors (mismatches): {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        error_rate = len(errors) / n_checked * 100 if n_checked > 0 else 0
        print(f"\n⚠ ERROR RATE: {error_rate:.2f}%")
        
        print(f"\n{'='*60}")
        print("MISMATCHES FOUND:")
        print(f"{'='*60}")
        
        for i, err in enumerate(errors[:20 if not args.verbose else len(errors)]):
            print(f"\n[{i+1}] Row {err['row']} | Instance: {err['instance_id']}")
            print(f"    Stored confidence:   {err['stored_conf']:.6f}")
            if 'expected_conf' in err:
                print(f"    Expected confidence: {err['expected_conf']:.6f}")
            if 'parsed_conf' in err:
                print(f"    Parsed from JSON:    {err['parsed_conf']:.6f} (raw: {err['parsed_raw']})")
            print(f"    Difference:          {err['diff']:.6f}")
            
            if 'vote_counts' in err:
                print(f"    Vote counts: {err['vote_counts']} (majority: {err['majority_count']}/{err['total_votes']})")
            if 'yes_prob' in err:
                print(f"    yes_prob={err['yes_prob']:.6f}, no_prob={err['no_prob']:.6f}, prediction={err['prediction']}")
            if 'response' in err and args.verbose:
                print(f"    Response: {err['response'][:200]}...")
        
        if len(errors) > 20 and not args.verbose:
            print(f"\n... and {len(errors) - 20} more errors (use --verbose to see all)")
    else:
        print(f"\n✓ ALL {n_checked} INSTANCES VALIDATED SUCCESSFULLY!")
        print(f"  Confidence values match LLM responses within tolerance={args.tolerance}")
    
    # Summary statistics
    if errors:
        diffs = [e['diff'] for e in errors]
        print(f"\nMismatch Statistics:")
        print(f"  Mean difference: {np.mean(diffs):.6f}")
        print(f"  Max difference:  {np.max(diffs):.6f}")
        print(f"  Min difference:  {np.min(diffs):.6f}")
    
    # Warnings summary
    if warnings:
        print(f"\nWarnings ({len(warnings)} total):")
        warning_counts = Counter(w.split(':')[1].strip() if ':' in w else w for w in warnings)
        for warn_type, count in warning_counts.most_common(5):
            print(f"  - {warn_type}: {count}")
    
    print("\n" + "=" * 80)
    
    # Exit code based on errors
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
