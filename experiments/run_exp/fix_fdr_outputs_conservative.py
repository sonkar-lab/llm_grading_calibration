#!/usr/bin/env python3
"""
Compute a more conservative FDR adjustment using Benjamini–Yekutieli (BY).

Scope:
- mixed_model_effects_*.csv: compute BY across ALL tests in a tables/ directory
  (all targets combined), pooled across models/methods.
- calibration_regression_*.csv: compute BY across ALL interaction tests in the
  same tables/ directory.

Writes new columns (does not overwrite existing BH fields):
- mixed models: p_adjusted_by, significant_by
- calibration regression: p_adjusted_by_global, significant_by_global
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from typing import Dict, List, Tuple

from grading_calibration.statistics.fdr import (
    benjamini_yekutieli,
    benjamini_yekutieli_excluding_nan,
    parse_pvalue,
    significant_at,
)


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
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0
    return p


def _harmonic(m: int) -> float:
    return sum(1.0 / i for i in range(1, m + 1))


def _by_adjust(pvals: List[float], canonical_correction: bool = False) -> List[float]:
    if canonical_correction:
        return benjamini_yekutieli_excluding_nan(pvals)
    return benjamini_yekutieli(pvals)


def _is_significant(adj_p: float, alpha: float) -> bool:
    return significant_at(adj_p, alpha, inclusive=True)


def _format_p(val: float) -> str:
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
        _ensure_col(header, "p_adjusted_by")
        _ensure_col(header, "significant_by")
        data.append((path, header, rows))

    items: List[Tuple[float, int, int]] = []
    for fi, (_, _, rows) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_value", ""), canonical_correction)
            if p is None:
                continue
            items.append((p, fi, ri))

    pvals = [p for p, _, _ in items]
    adj = _by_adjust(pvals, canonical_correction)
    updated_rows = 0

    for (p, fi, ri), adj_p in zip(items, adj):
        rows = data[fi][2]
        row = rows[ri]
        prev = _parse_pval(row.get("p_adjusted_by", ""), canonical_correction)
        needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
        if needs_update:
            row["p_adjusted_by"] = _format_p(adj_p)
            row["significant_by"] = "True" if _is_significant(adj_p, alpha) else "False"
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
        _ensure_col(header, "p_adjusted_by_global")
        _ensure_col(header, "significant_by_global")
        data.append((path, header, rows))

    items: List[Tuple[float, int, int]] = []
    for fi, (_, _, rows) in enumerate(data):
        for ri, row in enumerate(rows):
            p = _parse_pval(row.get("p_interaction", ""), canonical_correction)
            if p is None:
                continue
            items.append((p, fi, ri))

    pvals = [p for p, _, _ in items]
    adj = _by_adjust(pvals, canonical_correction)
    updated_rows = 0

    for (p, fi, ri), adj_p in zip(items, adj):
        rows = data[fi][2]
        row = rows[ri]
        prev = _parse_pval(row.get("p_adjusted_by_global", ""), canonical_correction)
        needs_update = prev is None or not math.isfinite(prev) or not math.isfinite(adj_p) or abs(prev - adj_p) > tol
        if needs_update:
            row["p_adjusted_by_global"] = _format_p(adj_p)
            row["significant_by_global"] = "True" if _is_significant(adj_p, alpha) else "False"
            updated_rows += 1

    if not dry_run:
        for path, header, rows in data:
            _write_csv(path, header, rows)

    return len(files), updated_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply conservative BY FDR adjustment.")
    parser.add_argument(
        "--root",
        default="grading_calibration/experiments/outputs_runs/outputs",
        help="Root directory to scan for outputs.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance for existing values.")
    parser.add_argument(
        "--canonical-correction",
        action="store_true",
        help="Opt into Phase-6 canonical correction semantics: sentinels become NaN and are excluded from FDR.",
    )
    args = parser.parse_args()

    mixed_files = glob.glob(
        os.path.join(args.root, "**", "mixed_model_effects_*.csv"),
        recursive=True,
    )
    calib_files = glob.glob(
        os.path.join(args.root, "**", "calibration_regression_*.csv"),
        recursive=True,
    )

    mixed_by_dir: Dict[str, List[str]] = {}
    for path in mixed_files:
        mixed_by_dir.setdefault(os.path.dirname(path), []).append(path)

    calib_by_dir: Dict[str, List[str]] = {}
    for path in calib_files:
        calib_by_dir.setdefault(os.path.dirname(path), []).append(path)

    total_files = 0
    total_updates = 0

    for tables_dir, files in sorted(mixed_by_dir.items()):
        n_files, n_updates = _process_mixed_model_dir(
            tables_dir, files, args.alpha, args.dry_run, args.tol, args.canonical_correction
        )
        total_files += n_files
        total_updates += n_updates

    for tables_dir, files in sorted(calib_by_dir.items()):
        n_files, n_updates = _process_calibration_dir(
            tables_dir, files, args.alpha, args.dry_run, args.tol, args.canonical_correction
        )
        total_files += n_files
        total_updates += n_updates

    mode = "DRY RUN" if args.dry_run else "WROTE"
    print(f"{mode}: {total_updates} rows updated across {total_files} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
