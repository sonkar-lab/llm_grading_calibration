#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from grading_calibration.statistics.fdr import is_na_sentinel


DATASETS = {"ricechem", "beetle", "scientsbank"}
METHODS = ["self_reported", "self_consistency", "token_probability"]
DEFAULT_STEP3_ARGS = [
    "--effects-method",
    "mixed_model",
    "--effects-backend",
    "lme4",
    "--effects-lme4-maxfun",
    "50000",
    "--n-perm",
    "200",
    "--n-boot",
    "200",
    "--bootstrap-method",
    "percentile",
    "--calibration-regression-mode",
    "linear",
    "--calibration-regression-bootstrap",
    "0",
]


def _is_nan_value(value: str) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    lowered = text.lower()
    if lowered in {"nan", "na", "none", "null"}:
        return True
    try:
        num = float(text)
    except ValueError:
        return False
    if num != num:
        return True
    if is_na_sentinel(num):
        return True
    return False


def _parse_model_method(filename: str, prefix: str) -> tuple[str, str] | None:
    if not filename.startswith(prefix) or not filename.endswith(".csv"):
        return None
    base = filename[len(prefix):-4]
    for method in METHODS:
        suffix = f"_{method}"
        if base.endswith(suffix):
            model = base[:-len(suffix)]
            return model, method
    return None


def _detect_dataset(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    if "step3" in parts:
        idx = parts.index("step3")
        for offset in (1, 2, 3):
            if idx - offset >= 0 and parts[idx - offset] in DATASETS:
                return parts[idx - offset]
    for part in parts:
        if part in DATASETS:
            return part
    return None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return reader.fieldnames or [], rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_structural_error(row: dict[str, str]) -> bool:
    err = str(row.get("error", "")).lower()
    if "insufficient data" in err or "near-zero variance" in err:
        return True
    n_rows = row.get("n_rows")
    if n_rows is not None and str(n_rows).strip() in {"0", "0.0"}:
        return True
    return False


def _needs_rerun(
    row: dict[str, str],
    required_cols: list[str],
    skip_structural_errors: bool,
) -> bool:
    if skip_structural_errors and _is_structural_error(row):
        return False
    for col in required_cols:
        if col not in row:
            return True
        if _is_nan_value(row[col]):
            return True
    converged = row.get("converged")
    if converged is not None and str(converged).strip().lower() in {"false", "0"}:
        return True
    error = row.get("error", "")
    if str(error).strip():
        return True
    warnings = row.get("warnings", "")
    if "fitting failed" in str(warnings).lower():
        return True
    return False


def _row_key(row: dict[str, str], key_cols: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(col, "")).strip() for col in key_cols)


def _build_feature_file(path: Path, features: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for feature in sorted(set(features)):
            handle.write(f"{feature}\n")


def _run_step3(
    step3_path: Path,
    dataset: str,
    model: str,
    method: str,
    predictions_dir: Path,
    output_root: Path,
    run_id: str,
    features_file: Path,
    mode: str,
    extra_args: list[str],
    dry_run: bool,
    use_default_flags: bool,
) -> Path | None:
    if mode == "mixed_model" and "--calibration-regression-only" in extra_args:
        raise ValueError("Extra args include --calibration-regression-only for mixed_model rerun.")
    if mode == "calibration_regression" and "--no-calibration-regression" in extra_args:
        raise ValueError("Extra args include --no-calibration-regression for calibration_regression rerun.")
    cmd = [
        sys.executable,
        str(step3_path),
        "--dataset",
        dataset,
        "--model",
        model,
        "--method",
        method,
        "--predictions-dir",
        str(predictions_dir),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--effects-include-features-file",
        str(features_file),
    ]
    if use_default_flags:
        cmd.extend(DEFAULT_STEP3_ARGS)
    if mode == "mixed_model":
        cmd.append("--no-calibration-regression")
    elif mode == "calibration_regression":
        cmd.append("--calibration-regression-only")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    cmd.extend(extra_args)
    print("Rerun command:", " ".join(cmd))
    if dry_run:
        return None
    subprocess.run(cmd, check=True)
    return output_root / dataset / run_id / "step3" / "tables"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun Step-3 NaN fits and merge updated rows back into CSVs."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing Step-3 outputs (will scan for step3/tables/*.csv).",
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Predictions CSVs directory (passed to step3_feature_analysis.py).",
    )
    parser.add_argument(
        "--step3-path",
        default=str(Path(__file__).resolve().parent / "step3_feature_analysis.py"),
        help="Path to step3_feature_analysis.py (default: sibling of this script).",
    )
    parser.add_argument(
        "--tmp-root",
        default="",
        help="Temporary output root for reruns (default: auto temp dir).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned reruns but do not execute.",
    )
    parser.add_argument(
        "--modes",
        default="mixed_model,calibration_regression",
        help="Comma-separated list of modes to rerun (mixed_model,calibration_regression).",
    )
    parser.add_argument(
        "--skip-structural-errors",
        action="store_true",
        help="Skip reruns for structural errors (insufficient data or near-zero variance).",
    )
    parser.add_argument(
        "--no-default-flags",
        action="store_true",
        help="Do not include the default Step-3 flags from RESULTS_FINAL/step3/readme.txt.",
    )
    parser.add_argument(
        "--recompute-fdr",
        action="store_true",
        help="After merging, run fix_fdr_outputs.py on the root directory.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to step3_feature_analysis.py (prefix with --).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    step3_path = Path(args.step3_path).expanduser().resolve()
    if not root.exists():
        print(f"Error: root not found: {root}")
        return 1
    if not predictions_dir.exists():
        print(f"Error: predictions dir not found: {predictions_dir}")
        return 1
    if not step3_path.exists():
        print(f"Error: step3_feature_analysis.py not found: {step3_path}")
        return 1

    tmp_root = Path(args.tmp_root).expanduser().resolve() if args.tmp_root else None
    if tmp_root:
        tmp_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="rerun_step3_nan_"))

    csv_paths = list(root.rglob("tables/*.csv"))
    mixed_files = []
    calreg_files = []
    for path in csv_paths:
        name = path.name
        if name.startswith("mixed_model_effects_"):
            mixed_files.append(path)
        elif name.startswith("calibration_regression_"):
            calreg_files.append(path)

    if not mixed_files and not calreg_files:
        print(f"No step3 tables found under {root}")
        return 0

    total_updates = 0
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    for mode, files, required_cols in [
        ("mixed_model", mixed_files, ["beta", "se", "p_value"]),
        ("calibration_regression", calreg_files, ["beta_main", "se_main", "p_main", "beta_interaction", "se_interaction", "p_interaction"]),
    ]:
        if mode not in modes:
            continue
        for path in files:
            model_method = _parse_model_method(path.name, "mixed_model_effects_" if mode == "mixed_model" else "calibration_regression_")
            if not model_method:
                continue
            model, method = model_method
            dataset = _detect_dataset(path)
            if not dataset:
                print(f"Skipping (dataset not detected): {path}")
                continue

            fieldnames, rows = _read_csv(path)
            original_rows = [row.copy() for row in rows]
            features = []
            for row in rows:
                if _needs_rerun(row, required_cols, args.skip_structural_errors):
                    feat = row.get("feature", "").strip()
                    if feat:
                        features.append(feat)

            if not features:
                continue

            print(f"\nFound {len(set(features))} features to rerun for {mode}: {dataset} | {model} | {method}")
            features_file = tmp_root / f"features_{mode}_{dataset}_{model}_{method}.txt"
            _build_feature_file(features_file, features)
            run_id = f"rerun_{mode}_{dataset}_{model}_{method}_{int(time.time())}"
            out_tables = _run_step3(
                step3_path,
                dataset,
                model,
                method,
                predictions_dir,
                tmp_root,
                run_id,
                features_file,
                mode,
                args.extra_args,
                args.dry_run,
                not args.no_default_flags,
            )
            if args.dry_run:
                continue

            new_csv = out_tables / path.name
            if not new_csv.exists():
                print(f"Warning: rerun output not found: {new_csv}")
                continue

            new_fieldnames, new_rows = _read_csv(new_csv)
            if mode == "mixed_model":
                key_cols = [c for c in ("feature", "target_col", "outcome_type") if c in fieldnames]
            else:
                key_cols = [c for c in ("feature", "mode") if c in fieldnames]
            if not key_cols:
                print(f"Warning: no key columns for {path}; skipping merge.")
                continue

            new_map = {_row_key(row, key_cols): row for row in new_rows}
            updated = 0
            for row in rows:
                key = _row_key(row, key_cols)
                if key in new_map:
                    new_row = new_map[key]
                    for col in fieldnames:
                        if col in new_row:
                            row[col] = new_row[col]
                    updated += 1

            if updated:
                backup_path = path.with_suffix(path.suffix + f".bak_{int(time.time())}")
                _write_csv(backup_path, fieldnames, original_rows)
                _write_csv(path, fieldnames, rows)
                total_updates += updated
                print(f"Updated {updated} rows in {path} (backup: {backup_path.name})")

    if args.recompute_fdr and not args.dry_run:
        fix_script = Path(__file__).resolve().parent.parent / "run_exp" / "fix_fdr_outputs.py"
        if fix_script.exists():
            cmd = [sys.executable, str(fix_script)]
            print("\nRecomputing FDR across outputs...")
            subprocess.run(cmd, check=True, cwd=str(fix_script.parent))
        else:
            print(f"Warning: fix_fdr_outputs.py not found at {fix_script}")

    print(f"\nTotal rows updated: {total_updates}")
    if not args.dry_run and args.tmp_root == "":
        print(f"Temporary rerun outputs stored at: {tmp_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
