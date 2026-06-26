#!/usr/bin/env python3
"""Check correct vs incorrect label percentages for each dataset.

This script reads the canonical dataset CSVs (SciEntsBank, Beetle) and the
RiceChem rubric CSVs to compute the proportion of correct (label=1/TRUE)
vs incorrect (label=0/FALSE) instances.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Keep in sync with RiceChemAdapter.INVALID_ITEMS.
RICECHEM_INVALID_ITEMS = {
    "score",
    "adjustment",
    "comments",
    "blank",
    "incorrect",
    "core charge calculation error",
    "incorrect statement included",
    "correct response",
    "incorrect/blank response",
    "incorrect/missing answer",
    "incorrect/misleading statement",
}


def _parse_label(raw: str) -> Tuple[bool, int]:
    """Parse a raw label string into (ok, value)."""
    if raw is None:
        return False, 0
    val = str(raw).strip()
    if val == "":
        return False, 0
    if val.upper() == "TRUE":
        return True, 1
    if val.upper() == "FALSE":
        return True, 0
    try:
        num = int(float(val))
    except ValueError:
        return False, 0
    if num in (0, 1):
        return True, num
    return False, 0


def read_label_counts(csv_path: Path, label_cols: Iterable[str]) -> Dict[str, int]:
    """Read 0/1 labels from a CSV and count correct/incorrect."""
    counts = {"correct": 0, "incorrect": 0, "unknown": 0, "total": 0}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        label_col = None
        for col in label_cols:
            if col in reader.fieldnames:
                label_col = col
                break
        if label_col is None:
            raise ValueError(f"No label column found in {csv_path}: {label_cols}")
        for row in reader:
            ok, value = _parse_label(row.get(label_col, ""))
            if not ok:
                counts["unknown"] += 1
                continue
            if value == 1:
                counts["correct"] += 1
            else:
                counts["incorrect"] += 1
            counts["total"] += 1
    return counts


def read_ricechem_counts(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Count TRUE/FALSE rubric-item labels for RiceChem, matching adapter logic."""
    invalid_items = {s.lower().strip() for s in RICECHEM_INVALID_ITEMS}
    per_question: Dict[str, Dict[str, int]] = {}
    overall = {"correct": 0, "incorrect": 0, "unknown": 0, "total": 0}

    for q_id in range(1, 5):
        answer_file = data_dir / f"Student Answers Q{q_id}.csv"
        rubric_file = data_dir / f"Graded Rubric Q{q_id}.csv"
        if not answer_file.exists() or not rubric_file.exists():
            continue

        # Collect students with non-empty answers.
        id2ans = {}
        with answer_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                sid = row[0].strip()
                ans = row[1].strip()
                if ans:
                    id2ans[sid] = ans

        # Read rubric header and valid indices.
        with rubric_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rubric_items = header[1:]
            valid_indices = []
            for i, item in enumerate(rubric_items):
                if item.lower().strip() not in invalid_items and item.strip():
                    valid_indices.append(i)

            q_counts = {"correct": 0, "incorrect": 0, "unknown": 0, "total": 0}
            for row in reader:
                if len(row) < 2:
                    continue
                sid = row[0].strip()
                if sid not in id2ans:
                    continue
                values = row[1:]
                valid_values = [values[i] for i in valid_indices if i < len(values)]
                for v in valid_values:
                    ok, value = _parse_label(v)
                    if not ok:
                        # Match adapter behavior: anything not TRUE is treated as 0.
                        value = 0
                    if value == 1:
                        q_counts["correct"] += 1
                    else:
                        q_counts["incorrect"] += 1
                    q_counts["total"] += 1

            per_question[f"Q{q_id}"] = q_counts
            for k in ("correct", "incorrect", "unknown", "total"):
                overall[k] += q_counts[k]

    per_question["overall"] = overall
    return per_question


def format_counts(name: str, counts: Dict[str, int]) -> str:
    total = counts["total"]
    if total == 0:
        return f"{name}: total=0 (no labeled instances)"
    correct = counts["correct"]
    incorrect = counts["incorrect"]
    correct_pct = 100.0 * correct / total
    incorrect_pct = 100.0 * incorrect / total
    return (
        f"{name}: total={total} | correct={correct} ({correct_pct:.2f}%) | "
        f"incorrect={incorrect} ({incorrect_pct:.2f}%)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report correct vs incorrect label percentages by dataset."
    )
    parser.add_argument(
        "--scientsbank-csv",
        type=Path,
        default=Path("grading_calibration/dataset/SciEntsBank/scientsbank.csv"),
        help="Path to SciEntsBank CSV (default: dataset/SciEntsBank/scientsbank.csv)",
    )
    parser.add_argument(
        "--beetle-csv",
        type=Path,
        default=Path("grading_calibration/dataset/Beetle/beetle.csv"),
        help="Path to Beetle CSV (default: dataset/Beetle/beetle.csv)",
    )
    parser.add_argument(
        "--ricechem-dir",
        type=Path,
        default=Path("grading_calibration/dataset/RiceChem_Dataset"),
        help="Path to RiceChem directory (default: dataset/RiceChem_Dataset)",
    )
    parser.add_argument(
        "--no-per-question",
        action="store_true",
        help="Only print RiceChem overall counts (skip per-question breakdown).",
    )

    args = parser.parse_args()

    print("SciEntsBank:")
    sb_counts = read_label_counts(args.scientsbank_csv, ["ground_truth_label", "label"])
    print("  " + format_counts("scientsbank", sb_counts))
    if sb_counts.get("unknown", 0):
        print(f"  unknown labels: {sb_counts['unknown']}")

    print("\nBeetle:")
    beetle_counts = read_label_counts(args.beetle_csv, ["ground_truth_label", "label"])
    print("  " + format_counts("beetle", beetle_counts))
    if beetle_counts.get("unknown", 0):
        print(f"  unknown labels: {beetle_counts['unknown']}")

    print("\nRiceChem (rubric-item labels; only students with non-empty answers):")
    ricechem = read_ricechem_counts(args.ricechem_dir)
    if not args.no_per_question:
        for q in sorted(k for k in ricechem.keys() if k.startswith("Q")):
            print("  " + format_counts(q, ricechem[q]))
    print("  " + format_counts("overall", ricechem["overall"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
