#!/usr/bin/env python3
"""Phase 6b helpers for building and auditing corrected RESULTS_CANONICAL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from grading_calibration.statistics.fdr import (
    benjamini_hochberg_excluding_nan,
    is_na_sentinel,
    parse_pvalue,
    significant_at,
)

RELEVANT_PREFIXES = ("p_adjusted", "significant_")
RELEVANT_EXACT = {"p_interaction", "p_interaction_adj"}
SIGNIFICANCE_TRUE = {"true", "1", "yes"}
SIGNIFICANCE_FALSE = {"false", "0", "no", "", "nan"}
DEFAULT_ALPHA = 0.05
ADJUST_TOL = 5e-12


@dataclass(frozen=True)
class CsvTable:
    relpath: str
    header: list[str]
    rows: list[dict[str, str]]


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def write_checksum_manifest(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        for path in iter_files(root):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rel = path.relative_to(root).as_posix()
            handle.write(f"{digest}  {rel}\n")


def verify_checksum_manifest(root: Path, manifest: Path) -> tuple[bool, list[str]]:
    expected: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        expected[rel] = digest
    actual = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in iter_files(root)}
    messages: list[str] = []
    for rel in sorted(set(expected) | set(actual)):
        if rel not in expected:
            messages.append(f"unexpected file: {rel}")
        elif rel not in actual:
            messages.append(f"missing file: {rel}")
        elif expected[rel] != actual[rel]:
            messages.append(f"checksum mismatch: {rel}")
    return not messages, messages


def read_csv_table(path: Path, root: Path) -> CsvTable:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return CsvTable(path.relative_to(root).as_posix(), reader.fieldnames or [], list(reader))


def write_csv_table(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def is_relevant_column(column: str) -> bool:
    return column in RELEVANT_EXACT or column.startswith(RELEVANT_PREFIXES)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in SIGNIFICANCE_TRUE


def is_false(value: object) -> bool:
    return str(value).strip().lower() in SIGNIFICANCE_FALSE


def is_nonfinite_text(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"", "nan", "na", "none", "null"}:
        return True
    try:
        return not math.isfinite(float(text))
    except ValueError:
        return False


_MISSING_TOKENS = {"", "nan", "na", "none", "null"}


def equivalent_cell(old: object, new: object) -> bool:
    old_text = str(old).strip()
    new_text = str(new).strip()
    if old_text == new_text:
        return True
    # Treat empty cells and NaN-like tokens as the same "missing" value so the
    # diff reports only substantive changes (e.g. an empty p_adjusted that the
    # correction rewrites as the literal "nan" is not a real change).
    if old_text.lower() in _MISSING_TOKENS and new_text.lower() in _MISSING_TOKENS:
        return True
    try:
        old_float = float(old_text)
        new_float = float(new_text)
    except ValueError:
        return False
    if math.isnan(old_float) and math.isnan(new_float):
        return True
    return old_float == new_float


def cells_equivalent(column: str, old: object, new: object) -> bool:
    """Column-aware change detection.

    Significance columns are compared by truthiness so that semantically equal
    "not significant" encodings (empty cell vs the literal "False") are not
    reported as changes. All other columns use the numeric/text equivalence in
    ``equivalent_cell`` (which already treats empty vs "nan" as equal).
    """
    if column.startswith("significant_"):
        return is_true(old) == is_true(new)
    return equivalent_cell(old, new)


def floats_match_observed(value: object, expected: float, *, tol: float = ADJUST_TOL) -> bool:
    if is_nonfinite_text(value):
        return not math.isfinite(expected)
    try:
        observed = float(str(value).strip())
    except ValueError:
        return False
    if not math.isfinite(observed) or not math.isfinite(expected):
        return False
    return abs(observed - expected) <= tol


def sanitize_canonical_sentinel_rows(root: Path) -> int:
    """Rewrite calibration sentinel p_* cells to nan and sentinel-row significant flags to False."""
    changed = 0
    for path in sorted((root / "step3").glob("*/tables/calibration_regression_*.csv")):
        table = read_csv_table(path, root)
        dirty = False
        for row in table.rows:
            row_has_sentinel = is_na_sentinel(row.get("p_interaction"))
            for column in table.header:
                if column.startswith("p_") and is_na_sentinel(row.get(column)):
                    row[column] = "nan"
                    dirty = True
                    changed += 1
            if row_has_sentinel:
                for column in table.header:
                    if column.startswith("significant_") and is_true(row.get(column)):
                        row[column] = "False"
                        dirty = True
                        changed += 1
        if dirty:
            write_csv_table(path, table.header, table.rows)
    return changed


def parse_calibration_filename(relpath: str) -> tuple[str, str, str] | None:
    path = Path(relpath)
    parts = path.parts
    if "step3" not in parts or not path.name.startswith("calibration_regression_"):
        return None
    idx = parts.index("step3")
    if idx + 1 >= len(parts):
        return None
    dataset = parts[idx + 1]
    stem = path.stem.removeprefix("calibration_regression_")
    for method in ("self_reported", "self_consistency", "token_probability"):
        suffix = f"_{method}"
        if stem.endswith(suffix):
            return dataset, stem[: -len(suffix)], method
    return None


def _tables(root: Path, pattern: str) -> list[CsvTable]:
    return [read_csv_table(path, root) for path in sorted((root / "step3").glob(pattern))]


def calibration_tables(root: Path) -> list[CsvTable]:
    return _tables(root, "*/tables/calibration_regression_*.csv")


def mixed_model_tables(root: Path) -> list[CsvTable]:
    return _tables(root, "*/tables/mixed_model_effects_*.csv")


def _record_family(
    items: list[tuple[CsvTable, int, float]],
    adjusted_column: str,
    significant_column: str,
    expected_adjusted: dict[tuple[str, int, str], float],
    expected_significant: dict[tuple[str, int, str], bool],
    *,
    alpha: float,
) -> None:
    adjusted = benjamini_hochberg_excluding_nan([pval for _, _, pval in items])
    for (table, idx, _), adj in zip(items, adjusted):
        expected_adjusted[(table.relpath, idx, adjusted_column)] = adj
        expected_significant[(table.relpath, idx, significant_column)] = significant_at(adj, alpha, inclusive=True)


def expected_fdr_outputs(root: Path, *, alpha: float = DEFAULT_ALPHA) -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int, str], bool]]:
    """Independently mirror Step-3 fix_fdr_outputs.py --no-expanded --across-datasets families."""
    expected_adjusted: dict[tuple[str, int, str], float] = {}
    expected_significant: dict[tuple[str, int, str], bool] = {}

    mixed = mixed_model_tables(root)
    mixed_by_dir_target: dict[tuple[str, str], list[tuple[CsvTable, int, float]]] = {}
    mixed_by_target: dict[str, list[tuple[CsvTable, int, float]]] = {}
    for table in mixed:
        dir_rel = str(Path(table.relpath).parent)
        for idx, row in enumerate(table.rows):
            target = (row.get("target") or "").strip() or "unknown"
            item = (table, idx, parse_pvalue(row.get("p_value")))
            mixed_by_dir_target.setdefault((dir_rel, target), []).append(item)
            mixed_by_target.setdefault(target, []).append(item)
    for items in mixed_by_dir_target.values():
        _record_family(items, "p_adjusted", "significant_fdr", expected_adjusted, expected_significant, alpha=alpha)
    for items in mixed_by_target.values():
        _record_family(
            items,
            "p_adjusted_across_datasets",
            "significant_across_datasets",
            expected_adjusted,
            expected_significant,
            alpha=alpha,
        )

    calibration = calibration_tables(root)
    calib_by_dir: dict[str, list[tuple[CsvTable, int, float]]] = {}
    calib_all: list[tuple[CsvTable, int, float]] = []
    for table in calibration:
        dir_rel = str(Path(table.relpath).parent)
        for idx, row in enumerate(table.rows):
            item = (table, idx, parse_pvalue(row.get("p_interaction")))
            calib_by_dir.setdefault(dir_rel, []).append(item)
            calib_all.append(item)
    for items in calib_by_dir.values():
        _record_family(items, "p_adjusted_global", "significant_global", expected_adjusted, expected_significant, alpha=alpha)
    if calib_all:
        _record_family(
            calib_all,
            "p_adjusted_global_across_datasets",
            "significant_global_across_datasets",
            expected_adjusted,
            expected_significant,
            alpha=alpha,
        )
    return expected_adjusted, expected_significant


def classify_change(
    *,
    column: str,
    old_value: str,
    new_value: str,
    old_row: dict[str, str],
    relpath: str,
    row_index: int,
    expected_adjusted: dict[tuple[str, int, str], float],
    expected_significant: dict[tuple[str, int, str], bool],
) -> str:
    sentinel_row = is_na_sentinel(old_row.get("p_interaction"))
    if sentinel_row:
        if column.startswith("p_") and is_nonfinite_text(new_value):
            return "sentinel_correction"
        if column.startswith("significant_") and is_true(old_value) and is_false(new_value):
            return "sentinel_correction"
    adjusted_key = (relpath, row_index, column)
    if adjusted_key in expected_adjusted and floats_match_observed(new_value, expected_adjusted[adjusted_key]):
        return "repool_recompute"
    significant_key = (relpath, row_index, column)
    if significant_key in expected_significant:
        expected = expected_significant[significant_key]
        if (expected and is_true(new_value)) or (not expected and is_false(new_value)):
            return "repool_recompute"
    return "unexpected"


def compare_results(final_root: Path, canonical_root: Path) -> tuple[list[dict[str, object]], Counter[str]]:
    expected_adjusted, expected_significant = expected_fdr_outputs(canonical_root)
    changes: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for old_path in sorted(final_root.rglob("*.csv")):
        rel = old_path.relative_to(final_root).as_posix()
        new_path = canonical_root / rel
        if not new_path.exists():
            changes.append({"category": "unexpected", "file": rel, "reason": "missing canonical file"})
            counts["unexpected"] += 1
            continue
        old = read_csv_table(old_path, final_root)
        new = read_csv_table(new_path, canonical_root)
        if old.header != new.header or len(old.rows) != len(new.rows):
            changes.append({"category": "unexpected", "file": rel, "reason": "shape/header changed"})
            counts["unexpected"] += 1
            continue
        for idx, (old_row, new_row) in enumerate(zip(old.rows, new.rows)):
            dataset, model, method = parse_calibration_filename(rel) or ("", "", "")
            feature = old_row.get("feature", str(idx))
            for column in old.header:
                if not is_relevant_column(column):
                    continue
                old_value = old_row.get(column, "")
                new_value = new_row.get(column, "")
                if cells_equivalent(column, old_value, new_value):
                    continue
                category = classify_change(
                    column=column,
                    old_value=old_value,
                    new_value=new_value,
                    old_row=old_row,
                    relpath=rel,
                    row_index=idx,
                    expected_adjusted=expected_adjusted,
                    expected_significant=expected_significant,
                )
                counts[category] += 1
                changes.append(
                    {
                        "category": category,
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "feature": feature,
                        "file": rel,
                        "row_index": idx,
                        "column": column,
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )
    return changes, counts


def contamination_inventory(final_root: Path, canonical_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cells: list[dict[str, object]] = []
    flips: list[dict[str, object]] = []
    for old_path in sorted((final_root / "step3").glob("*/tables/calibration_regression_*.csv")):
        rel = old_path.relative_to(final_root).as_posix()
        new_path = canonical_root / rel
        old = read_csv_table(old_path, final_root)
        new = read_csv_table(new_path, canonical_root) if new_path.exists() else None
        dataset, model, method = parse_calibration_filename(rel) or ("", "", "")
        for idx, old_row in enumerate(old.rows):
            feature = old_row.get("feature", str(idx))
            for column in old.header:
                if column.startswith("p_") and is_na_sentinel(old_row.get(column)):
                    cells.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "feature": feature,
                            "file": rel,
                            "row_index": idx,
                            "column": column,
                            "raw_value": old_row.get(column, ""),
                        }
                    )
            if new is not None and is_na_sentinel(old_row.get("p_interaction")):
                new_row = new.rows[idx]
                for column in old.header:
                    if column.startswith("significant_") and is_true(old_row.get(column)) and is_false(new_row.get(column)):
                        flips.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "method": method,
                                "feature": feature,
                                "file": rel,
                                "row_index": idx,
                                "column": column,
                                "old_value": old_row.get(column, ""),
                                "new_value": new_row.get(column, ""),
                            }
                        )
    return cells, flips


def validate_canonical(root: Path) -> tuple[int, int]:
    negative_p_values = 0
    nonfinite_significant = 0
    for path in sorted(root.rglob("*.csv")):
        table = read_csv_table(path, root)
        p_columns = [c for c in table.header if c.startswith("p_")]
        sig_columns = [c for c in table.header if c.startswith("significant_")]
        for row in table.rows:
            row_nonfinite = False
            for column in p_columns:
                text = str(row.get(column, "")).strip()
                if not text:
                    row_nonfinite = True
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    row_nonfinite = True
                elif value < 0:
                    negative_p_values += 1
            if row_nonfinite:
                for column in sig_columns:
                    if is_true(row.get(column)):
                        nonfinite_significant += 1
    return negative_p_values, nonfinite_significant


def write_dicts(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_inventory_reports(final_root: Path, canonical_root: Path, output_dir: Path) -> None:
    cells, flips = contamination_inventory(final_root, canonical_root)
    write_dicts(output_dir / "contaminated_cells.csv", cells)
    write_dicts(output_dir / "significance_flips.csv", flips)
    files = sorted({row["file"] for row in cells})
    sentinel_rows = {(row["file"], row["row_index"]) for row in cells if row["column"] == "p_interaction"}
    by_cell_col = Counter(str(row["column"]) for row in cells)
    by_flip_col = Counter(str(row["column"]) for row in flips)
    text = [
        "# Phase 6b Contamination Inventory",
        "",
        f"Contaminated files: {len(files)}",
        f"Raw sentinel-cell occurrences: {len(cells)}",
        f"Rows with sentinel `p_interaction`: {len(sentinel_rows)}",
        f"Significance flips `True -> False`: {len(flips)}",
        "",
        "Reconciliation: the current RESULTS_FINAL snapshot has 27 failed rows. Each failed row carries sentinels in three `p_*` columns (`p_interaction`, `p_interaction_adj`, `p_adjusted_global`) and three `significant_*` flags that flip from `True` to `False`. The earlier approximate raw-cell count was stale; these committed audit counts are the Phase 6b source of truth.",
        "",
        "## Sentinel Cell Columns",
    ]
    text.extend(f"- `{column}`: {count}" for column, count in sorted(by_cell_col.items()))
    text.append("")
    text.append("## Significance Flip Columns")
    text.extend(f"- `{column}`: {count}" for column, count in sorted(by_flip_col.items()))
    text.append("")
    text.append("## Files")
    text.extend(f"- `{file}`" for file in files)
    (output_dir / "contamination_inventory.md").write_text("\n".join(text) + "\n")


def write_diff_reports(final_root: Path, canonical_root: Path, output_dir: Path) -> int:
    changes, counts = compare_results(final_root, canonical_root)
    write_dicts(output_dir / "before_after_changed_cells.csv", changes)
    negative_p_values, nonfinite_significant = validate_canonical(canonical_root)
    text = [
        "# Phase 6b Before/After Diff Gate",
        "",
        f"Sentinel correction changes: {counts['sentinel_correction']}",
        f"Repool recompute changes: {counts['repool_recompute']}",
        f"Unexpected changes: {counts['unexpected']}",
        f"Negative values in canonical `p_*` columns: {negative_p_values}",
        f"Rows with non-finite p-values and `significant_* = True`: {nonfinite_significant}",
        "",
        "Allowed categories are exactly: `sentinel_correction` and `repool_recompute`.",
    ]
    (output_dir / "before_after_diff_summary.md").write_text("\n".join(text) + "\n")
    if counts["unexpected"] or negative_p_values or nonfinite_significant:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6b audit helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_checksum = sub.add_parser("checksum")
    p_checksum.add_argument("--root", type=Path, required=True)
    p_checksum.add_argument("--output", type=Path, required=True)
    p_verify = sub.add_parser("verify-checksum")
    p_verify.add_argument("--root", type=Path, required=True)
    p_verify.add_argument("--manifest", type=Path, required=True)
    p_inventory = sub.add_parser("inventory")
    p_inventory.add_argument("--final-root", type=Path, required=True)
    p_inventory.add_argument("--canonical-root", type=Path, required=True)
    p_inventory.add_argument("--output-dir", type=Path, required=True)
    p_diff = sub.add_parser("diff")
    p_diff.add_argument("--final-root", type=Path, required=True)
    p_diff.add_argument("--canonical-root", type=Path, required=True)
    p_diff.add_argument("--output-dir", type=Path, required=True)
    p_validate = sub.add_parser("validate-canonical")
    p_validate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "checksum":
        write_checksum_manifest(args.root, args.output)
        return 0
    if args.cmd == "verify-checksum":
        ok, messages = verify_checksum_manifest(args.root, args.manifest)
        if ok:
            print("RESULTS_FINAL checksum verification passed")
            return 0
        print("RESULTS_FINAL checksum verification failed")
        for message in messages:
            print(message)
        return 1
    if args.cmd == "inventory":
        write_inventory_reports(args.final_root, args.canonical_root, args.output_dir)
        return 0
    if args.cmd == "diff":
        return write_diff_reports(args.final_root, args.canonical_root, args.output_dir)
    if args.cmd == "validate-canonical":
        negative, nonfinite_sig = validate_canonical(args.root)
        print(f"negative_p_values={negative}")
        print(f"nonfinite_significant={nonfinite_sig}")
        return 0 if negative == 0 and nonfinite_sig == 0 else 1
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
