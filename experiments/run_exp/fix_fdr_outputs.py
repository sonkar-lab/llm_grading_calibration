#!/usr/bin/env python3
"""
Post-process Step-3 outputs to ensure BH-FDR adjusted p-values are present.

Scope (base):
- mixed_model_effects_*.csv: compute global BH within each target family
  (e.g., Confidence vs Correctness) pooled across all models/methods in the
  same tables/ directory.
- calibration_regression_*.csv: compute global BH on p_interaction across all
  models/methods in the same tables/ directory.

Optional expanded family (guarded):
- mixed_model_effects_*.csv: compute BH within each target family, pooled across
  ALL models/methods present in a dataset (small + big), written to
  p_adjusted_expanded/significant_expanded.
- calibration_regression_*.csv: same expansion, written to
  p_adjusted_global_expanded/significant_global_expanded.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from typing import Dict, List, Tuple

from grading_calibration.statistics.fdr import (
    benjamini_hochberg,
    benjamini_hochberg_excluding_nan,
    parse_pvalue,
    significant_at,
)


_KNOWN_METHODS = ("self_reported", "self_consistency", "token_probability")
_DEFAULT_EXPECTED_MODELS = [
    "llama_3.1_8b_instruct",
    "qwen3_4b_thinking_2507",
    "gpt_oss_20b",
    "llama_3.3_70b_instruct",
    "qwen3_next_80b_a3b_instruct",
    "qwen3_next_80b_a3b_thinking",
    "gpt_oss_120b",
]

_STEP5_PREFIXES = ("commonality_confidence_", "commonality_correctness_")
_STEP6_FILES = ("model_comparisons.csv", "method_comparisons.csv")

def _parse_pval(val: str, canonical_correction: bool = False) -> float | None:
    if canonical_correction:
        return parse_pvalue(val)
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        p = float(val)
    except ValueError:
        return None
    if not math.isfinite(p):
        return None
    # Legacy behavior: clamp all negatives, including R NA sentinels, to zero.
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0
    return p


def _bh_adjust(pvals: List[float], canonical_correction: bool = False) -> List[float]:
    if canonical_correction:
        return benjamini_hochberg_excluding_nan(pvals)
    return benjamini_hochberg(pvals)


def _is_significant(adj_p: float, alpha: float) -> bool:
    return significant_at(adj_p, alpha, inclusive=True)


def _format_p(val: float) -> str:
    # Keep a concise, stable representation without forcing fixed decimals.
    return format(val, ".12g")


def _load_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def _write_csv(path: str, header: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_col(header: List[str], col: str) -> None:
    if col not in header:
        header.append(col)


def _process_mixed_model_dir(
    tables_dir: str,
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted")
        _ensure_col(header, "significant_fdr")
        data.append((path, header, rows))

    # Collect p-values by target family.
    by_target: Dict[str, List[Tuple[float, int, int]]] = {}
    for fi, (_, _, rows) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_value", ""), canonical_correction)
            if p is None:
                continue
            target = (row.get("target") or "").strip() or "unknown"
            by_target.setdefault(target, []).append((p, fi, ri))

    updated_rows = 0
    for target, items in by_target.items():
        pvals = [p for p, _, _ in items]
        adj = _bh_adjust(pvals, canonical_correction)
        for (p, fi, ri), adj_p in zip(items, adj):
            rows = data[fi][2]
            row = rows[ri]
            prev = _parse_pval(row.get("p_adjusted", ""), canonical_correction)
            needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
            if needs_update:
                row["p_adjusted"] = _format_p(adj_p)
                row["significant_fdr"] = "True" if _is_significant(adj_p, alpha) else "False"
                updated_rows += 1

    if not dry_run:
        for path, header, rows in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def _extract_model_from_filename(path: str) -> str | None:
    name = os.path.basename(path)
    if not name.startswith("mixed_model_effects_") or not name.endswith(".csv"):
        return None
    stem = name[len("mixed_model_effects_") : -len(".csv")]
    for method in _KNOWN_METHODS:
        suffix = f"_{method}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def _extract_model_from_calib_filename(path: str) -> str | None:
    name = os.path.basename(path)
    if not name.startswith("calibration_regression_") or not name.endswith(".csv"):
        return None
    stem = name[len("calibration_regression_") : -len(".csv")]
    for method in _KNOWN_METHODS:
        suffix = f"_{method}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def _expanded_guard_ok(
    files: List[str],
    expected_models: List[str],
    extractor,
) -> Tuple[bool, List[str]]:
    present = set()
    for path in files:
        model = extractor(path)
        if model:
            present.add(model)
    missing = [m for m in expected_models if m not in present]
    return len(missing) == 0, missing


def _process_expanded_mixed_model_dir(
    tables_dir: str,
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    expected_models: List[str],
    require_all_models: bool,
    verbose: bool,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    ok, missing = _expanded_guard_ok(files, expected_models, _extract_model_from_filename)
    if require_all_models and not ok:
        if verbose:
            print(f"SKIP expanded FDR for {tables_dir}: missing models {missing}")
        return len(files), 0

    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted_expanded")
        _ensure_col(header, "significant_expanded")
        data.append((path, header, rows))

    # Pool across ALL targets combined within dataset (expanded family within target).
    by_target: Dict[str, List[Tuple[float, int, int]]] = {}
    for fi, (_, _, rows) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_value", ""), canonical_correction)
            if p is None:
                continue
            target = (row.get("target") or "").strip() or "unknown"
            by_target.setdefault(target, []).append((p, fi, ri))

    updated_rows = 0
    for target, items in by_target.items():
        pvals = [p for p, _, _ in items]
        adj = _bh_adjust(pvals, canonical_correction)
        for (p, fi, ri), adj_p in zip(items, adj):
            rows = data[fi][2]
            row = rows[ri]
            prev = _parse_pval(row.get("p_adjusted_expanded", ""), canonical_correction)
            needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
            if needs_update:
                row["p_adjusted_expanded"] = _format_p(adj_p)
                row["significant_expanded"] = "True" if _is_significant(adj_p, alpha) else "False"
                updated_rows += 1

    if not dry_run:
        for path, header, rows in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def _process_expanded_calibration_dir(
    tables_dir: str,
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    expected_models: List[str],
    require_all_models: bool,
    verbose: bool,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    ok, missing = _expanded_guard_ok(files, expected_models, _extract_model_from_calib_filename)
    if require_all_models and not ok:
        if verbose:
            print(f"SKIP expanded calibration FDR for {tables_dir}: missing models {missing}")
        return len(files), 0

    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted_global_expanded")
        _ensure_col(header, "significant_global_expanded")
        data.append((path, header, rows))

    items: List[Tuple[float, int, int]] = []
    for fi, (_, _, rows) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_interaction", ""), canonical_correction)
            if p is None:
                continue
            items.append((p, fi, ri))

    pvals = [p for p, _, _ in items]
    adj = _bh_adjust(pvals, canonical_correction)
    updated_rows = 0
    for (p, fi, ri), adj_p in zip(items, adj):
        rows = data[fi][2]
        row = rows[ri]
        prev = _parse_pval(row.get("p_adjusted_global_expanded", ""), canonical_correction)
        needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
        if needs_update:
            row["p_adjusted_global_expanded"] = _format_p(adj_p)
            row["significant_global_expanded"] = "True" if _is_significant(adj_p, alpha) else "False"
            updated_rows += 1

    if not dry_run:
        for path, header, rows in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def _process_calibration_dir(
    tables_dir: str,
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted_global")
        _ensure_col(header, "significant_global")
        data.append((path, header, rows))

    items: List[Tuple[float, int, int]] = []
    for fi, (_, _, rows) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_interaction", ""), canonical_correction)
            if p is None:
                continue
            items.append((p, fi, ri))

    pvals = [p for p, _, _ in items]
    adj = _bh_adjust(pvals, canonical_correction)
    updated_rows = 0
    for (p, fi, ri), adj_p in zip(items, adj):
        rows = data[fi][2]
        row = rows[ri]
        prev = _parse_pval(row.get("p_adjusted_global", ""), canonical_correction)
        needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
        if needs_update:
            row["p_adjusted_global"] = _format_p(adj_p)
            row["significant_global"] = "True" if _is_significant(adj_p, alpha) else "False"
            updated_rows += 1

    if not dry_run:
        for path, header, rows in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def _extract_step5_analysis_type(path: str) -> str:
    name = os.path.basename(path)
    for prefix in _STEP5_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_").replace("commonality_", "")
    return "unknown"


def _process_step5_within_dataset(
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted_dataset")
        _ensure_col(header, "significant_dataset")
        analysis_type = _extract_step5_analysis_type(path)
        data.append((path, header, rows, analysis_type))

    by_type: Dict[str, List[Tuple[float, int, int]]] = {}
    for fi, (_, _, rows, analysis_type) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_value", ""), canonical_correction)
            if p is None:
                continue
            by_type.setdefault(analysis_type, []).append((p, fi, ri))

    updated_rows = 0
    for analysis_type, items in by_type.items():
        pvals = [p for p, _, _ in items]
        adj = _bh_adjust(pvals, canonical_correction)
        for (p, fi, ri), adj_p in zip(items, adj):
            rows = data[fi][2]
            row = rows[ri]
            prev = _parse_pval(row.get("p_adjusted_dataset", ""), canonical_correction)
            needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
            if needs_update:
                row["p_adjusted_dataset"] = _format_p(adj_p)
                row["significant_dataset"] = "True" if _is_significant(adj_p, alpha) else "False"
                updated_rows += 1

    if not dry_run:
        for path, header, rows, _ in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def _process_step5_across_datasets(
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted_across_datasets")
        _ensure_col(header, "significant_across_datasets")
        analysis_type = _extract_step5_analysis_type(path)
        data.append((path, header, rows, analysis_type))

    by_type: Dict[str, List[Tuple[float, int, int]]] = {}
    for fi, (_, _, rows, analysis_type) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_value", ""), canonical_correction)
            if p is None:
                continue
            by_type.setdefault(analysis_type, []).append((p, fi, ri))

    updated_rows = 0
    for analysis_type, items in by_type.items():
        pvals = [p for p, _, _ in items]
        adj = _bh_adjust(pvals, canonical_correction)
        for (p, fi, ri), adj_p in zip(items, adj):
            rows = data[fi][2]
            row = rows[ri]
            prev = _parse_pval(row.get("p_adjusted_across_datasets", ""), canonical_correction)
            needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
            if needs_update:
                row["p_adjusted_across_datasets"] = _format_p(adj_p)
                row["significant_across_datasets"] = "True" if _is_significant(adj_p, alpha) else "False"
                updated_rows += 1

    if not dry_run:
        for path, header, rows, _ in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def _process_step6_within_dataset(
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted_by_metric")
        _ensure_col(header, "significant_by_metric")
        data.append((path, header, rows))

    updated_rows = 0
    for fi, (_, _, rows) in enumerate(data):
        by_metric: Dict[str, List[Tuple[float, int]]] = {}
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_value", ""), canonical_correction)
            if p is None:
                continue
            metric = (row.get("metric") or "").strip() or "unknown"
            by_metric.setdefault(metric, []).append((p, ri))

        for metric, items in by_metric.items():
            pvals = [p for p, _ in items]
            adj = _bh_adjust(pvals, canonical_correction)
            for (p, ri), adj_p in zip(items, adj):
                row = rows[ri]
                prev = _parse_pval(row.get("p_adjusted_by_metric", ""), canonical_correction)
                needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
                if needs_update:
                    row["p_adjusted_by_metric"] = _format_p(adj_p)
                    row["significant_by_metric"] = "True" if _is_significant(adj_p, alpha) else "False"
                    updated_rows += 1

    if not dry_run:
        for path, header, rows in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def _process_step6_across_datasets(
    files: List[str],
    alpha: float,
    dry_run: bool,
    tol: float,
    canonical_correction: bool = False,
) -> Tuple[int, int]:
    data = []
    for path in files:
        header, rows = _load_csv(path)
        _ensure_col(header, "p_adjusted_across_datasets_by_metric")
        _ensure_col(header, "significant_across_datasets_by_metric")
        data.append((path, header, rows))

    by_key: Dict[Tuple[str, str], List[Tuple[float, int, int]]] = {}
    for fi, (_, _, rows) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_value", ""), canonical_correction)
            if p is None:
                continue
            metric = (row.get("metric") or "").strip() or "unknown"
            comparison_type = (row.get("comparison_type") or "").strip() or "unknown"
            by_key.setdefault((comparison_type, metric), []).append((p, fi, ri))

    updated_rows = 0
    for key, items in by_key.items():
        pvals = [p for p, _, _ in items]
        adj = _bh_adjust(pvals, canonical_correction)
        for (p, fi, ri), adj_p in zip(items, adj):
            rows = data[fi][2]
            row = rows[ri]
            prev = _parse_pval(row.get("p_adjusted_across_datasets_by_metric", ""), canonical_correction)
            needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
            if needs_update:
                row["p_adjusted_across_datasets_by_metric"] = _format_p(adj_p)
                row["significant_across_datasets_by_metric"] = "True" if _is_significant(adj_p, alpha) else "False"
                updated_rows += 1

    if not dry_run:
        for path, header, rows in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix or add BH-FDR adjusted p-values in outputs.")
    parser.add_argument(
        "--root",
        default="grading_calibration/experiments/outputs_runs/outputs",
        help="Root directory to scan for outputs.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance for existing p-values.")
    parser.add_argument(
        "--expected-models",
        default=",".join(_DEFAULT_EXPECTED_MODELS),
        help="Comma-separated list of models required for expanded FDR.",
    )
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="Disable model-completeness guard for expanded FDR.",
    )
    parser.add_argument(
        "--no-expanded",
        action="store_true",
        help="Skip expanded-family FDR columns.",
    )
    parser.add_argument(
        "--steps",
        default="step3",
        help="Comma-separated list of steps to process (step3,step5,step6).",
    )
    parser.add_argument(
        "--across-datasets",
        action="store_true",
        help="Compute additional across-dataset FDR columns where supported.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument(
        "--canonical-correction",
        action="store_true",
        help="Opt into Phase-6 canonical correction semantics: sentinels become NaN and are excluded from FDR.",
    )
    args = parser.parse_args()

    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    expected_models = [m.strip() for m in args.expected_models.split(",") if m.strip()]
    require_all_models = not args.no_guard

    mixed_files: List[str] = []
    calib_files: List[str] = []
    step5_files: List[str] = []
    step6_files: List[str] = []

    if "step3" in steps:
        mixed_files = glob.glob(
            os.path.join(args.root, "**", "mixed_model_effects_*.csv"),
            recursive=True,
        )
        calib_files = glob.glob(
            os.path.join(args.root, "**", "calibration_regression_*.csv"),
            recursive=True,
        )

    if "step5" in steps:
        step5_files = [
            p
            for p in glob.glob(os.path.join(args.root, "**", "commonality_*.csv"), recursive=True)
            if os.path.basename(p).startswith(_STEP5_PREFIXES)
        ]

    if "step6" in steps:
        step6_files = [
            p
            for p in glob.glob(os.path.join(args.root, "**", "*.csv"), recursive=True)
            if os.path.basename(p) in _STEP6_FILES
        ]

    mixed_by_dir: Dict[str, List[str]] = {}
    for path in mixed_files:
        mixed_by_dir.setdefault(os.path.dirname(path), []).append(path)

    calib_by_dir: Dict[str, List[str]] = {}
    for path in calib_files:
        calib_by_dir.setdefault(os.path.dirname(path), []).append(path)

    total_files = 0
    total_updates = 0
    expanded_updates = 0

    if "step3" in steps:
        for tables_dir, files in sorted(mixed_by_dir.items()):
            n_files, n_updates = _process_mixed_model_dir(
                tables_dir, files, args.alpha, args.dry_run, args.tol, args.canonical_correction
            )
            total_files += n_files
            total_updates += n_updates
            if not args.no_expanded:
                _, n_expanded = _process_expanded_mixed_model_dir(
                    tables_dir,
                    files,
                    args.alpha,
                    args.dry_run,
                    args.tol,
                    expected_models,
                    require_all_models,
                    args.verbose,
                    args.canonical_correction,
                )
                expanded_updates += n_expanded

        if args.across_datasets:
            data = []
            for path in mixed_files:
                header, rows = _load_csv(path)
                _ensure_col(header, "p_adjusted_across_datasets")
                _ensure_col(header, "significant_across_datasets")
                data.append((path, header, rows))
            by_target: Dict[str, List[Tuple[float, int, int]]] = {}
            for fi, (_, _, rows) in enumerate(data):
                for ri, row in enumerate(rows):
                    p = _parse_pval(row.get("p_value", ""), args.canonical_correction)
                    if p is None:
                        continue
                    target = (row.get("target") or "").strip() or "unknown"
                    by_target.setdefault(target, []).append((p, fi, ri))
            updated_rows = 0
            for target, items in by_target.items():
                pvals = [p for p, _, _ in items]
                adj = _bh_adjust(pvals, args.canonical_correction)
                for (p, fi, ri), adj_p in zip(items, adj):
                    rows = data[fi][2]
                    row = rows[ri]
                    prev = _parse_pval(row.get("p_adjusted_across_datasets", ""), args.canonical_correction)
                    needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > args.tol
                    if needs_update:
                        row["p_adjusted_across_datasets"] = _format_p(adj_p)
                        row["significant_across_datasets"] = "True" if _is_significant(adj_p, args.alpha) else "False"
                        updated_rows += 1
            if not args.dry_run:
                for path, header, rows in data:
                    _write_csv(path, header, rows)
            total_updates += updated_rows
            total_files += len(data)

    if "step3" in steps:
        for tables_dir, files in sorted(calib_by_dir.items()):
            n_files, n_updates = _process_calibration_dir(
                tables_dir, files, args.alpha, args.dry_run, args.tol, args.canonical_correction
            )
            total_files += n_files
            total_updates += n_updates
            if not args.no_expanded:
                _, n_expanded = _process_expanded_calibration_dir(
                    tables_dir,
                    files,
                    args.alpha,
                    args.dry_run,
                    args.tol,
                    expected_models,
                    require_all_models,
                    args.verbose,
                    args.canonical_correction,
                )
                expanded_updates += n_expanded

        if args.across_datasets and calib_files:
            data = []
            for path in calib_files:
                header, rows = _load_csv(path)
                _ensure_col(header, "p_adjusted_global_across_datasets")
                _ensure_col(header, "significant_global_across_datasets")
                data.append((path, header, rows))
            items: List[Tuple[float, int, int]] = []
            for fi, (_, _, rows) in enumerate(data):
                for ri, row in enumerate(rows):
                    p = _parse_pval(row.get("p_interaction", ""), args.canonical_correction)
                    if p is None:
                        continue
                    items.append((p, fi, ri))
            pvals = [p for p, _, _ in items]
            adj = _bh_adjust(pvals, args.canonical_correction)
            updated_rows = 0
            for (p, fi, ri), adj_p in zip(items, adj):
                rows = data[fi][2]
                row = rows[ri]
                prev = _parse_pval(row.get("p_adjusted_global_across_datasets", ""), args.canonical_correction)
                needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > args.tol
                if needs_update:
                    row["p_adjusted_global_across_datasets"] = _format_p(adj_p)
                    row["significant_global_across_datasets"] = "True" if _is_significant(adj_p, args.alpha) else "False"
                    updated_rows += 1
            if not args.dry_run:
                for path, header, rows in data:
                    _write_csv(path, header, rows)
            total_updates += updated_rows
            total_files += len(data)

    if "step5" in steps and step5_files:
        step5_by_dir: Dict[str, List[str]] = {}
        for path in step5_files:
            step5_by_dir.setdefault(os.path.dirname(path), []).append(path)
        for tables_dir, files in sorted(step5_by_dir.items()):
            n_files, n_updates = _process_step5_within_dataset(
                files, args.alpha, args.dry_run, args.tol, args.canonical_correction
            )
            total_files += n_files
            total_updates += n_updates
        if args.across_datasets:
            n_files, n_updates = _process_step5_across_datasets(
                step5_files, args.alpha, args.dry_run, args.tol, args.canonical_correction
            )
            total_files += n_files
            total_updates += n_updates

    if "step6" in steps and step6_files:
        step6_by_dir: Dict[str, List[str]] = {}
        for path in step6_files:
            step6_by_dir.setdefault(os.path.dirname(path), []).append(path)
        for tables_dir, files in sorted(step6_by_dir.items()):
            n_files, n_updates = _process_step6_within_dataset(
                files, args.alpha, args.dry_run, args.tol, args.canonical_correction
            )
            total_files += n_files
            total_updates += n_updates
        if args.across_datasets:
            n_files, n_updates = _process_step6_across_datasets(
                step6_files, args.alpha, args.dry_run, args.tol, args.canonical_correction
            )
            total_files += n_files
            total_updates += n_updates

    mode = "DRY RUN" if args.dry_run else "WROTE"
    print(f"{mode}: {total_updates} rows updated across {total_files} files.")
    if not args.no_expanded:
        print(f"{mode}: {expanded_updates} expanded-FDR rows updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
