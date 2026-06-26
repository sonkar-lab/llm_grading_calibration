#!/usr/bin/env python3
"""
Cleanup legacy checkpoint files that were created without dataset tags.

Actions:
1) Move checkpoint files with batch > max_batch into an archive folder.
2) For remaining files, insert the dataset tag into the filename so Step 1 can resume.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})")
BATCH_RE = re.compile(r"_batch(\d+)\.csv$")


def _find_predictions_root() -> Path | None:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "grading_calibration" / "experiments" / "predictions_csvs"
        if candidate.exists():
            return candidate
    return None


def _insert_dataset_tag(filename: str, dataset: str) -> tuple[str, bool]:
    if f"_{dataset}_" in filename:
        return filename, False
    match = TIMESTAMP_RE.search(filename)
    if not match:
        return filename, False
    ts = match.group(1)
    new_name = filename.replace(f"_{ts}", f"_{dataset}_{ts}", 1)
    return new_name, True


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_dup{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup legacy checkpoint files.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to predictions_csvs (default: auto-detect from repo).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ricechem",
        help="Dataset tag to insert into checkpoint filenames (default: ricechem).",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Batch threshold for cleanup logic (default: 8).",
    )
    parser.add_argument(
        "--dataset-over",
        type=str,
        default=None,
        help="If set, rename checkpoints with batch index > max-batch to this dataset tag (no archive).",
    )
    parser.add_argument(
        "--reindex-over",
        action="store_true",
        help="Reindex batch numbers for files with batch index > max-batch (per prefix group).",
    )
    parser.add_argument(
        "--skip-below",
        action="store_true",
        help="Do not rename checkpoints with batch index <= max-batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying files.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else _find_predictions_root()
    if not root or not root.exists():
        raise SystemExit("Could not locate predictions_csvs. Use --root to set it explicitly.")

    dataset = args.dataset.lower()
    max_batch = args.max_batch
    dataset_over = args.dataset_over.lower() if args.dataset_over else None
    skip_below = args.skip_below
    reindex_over = args.reindex_over
    dry_run = args.dry_run

    moved = 0
    renamed = 0
    skipped = 0
    retagged_over = 0
    reindexed_over = 0

    reindex_groups: dict[str, list[tuple[int, Path]]] = {}
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        checkpoint_dir = method_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue

        archive_dir = checkpoint_dir / f"archived_over_batch{max_batch}"
        for path in sorted(checkpoint_dir.glob("checkpoint_*.csv")):
            match = BATCH_RE.search(path.name)
            if not match:
                skipped += 1
                continue
            batch_idx = int(match.group(1))

            if batch_idx > max_batch:
                current_path = path
                if dataset_over:
                    new_name, changed = _insert_dataset_tag(path.name, dataset_over)
                    if changed:
                        dest = _unique_path(path.with_name(new_name))
                        if dry_run:
                            print(f"[DRY] rename {path} -> {dest}")
                        else:
                            path.rename(dest)
                        retagged_over += 1
                        current_path = dest
                    else:
                        skipped += 1
                        current_path = path
                else:
                    if reindex_over:
                        current_path = path
                    else:
                        dest = _unique_path(archive_dir / path.name)
                        if dry_run:
                            print(f"[DRY] move {path} -> {dest}")
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            path.rename(dest)
                        moved += 1
                        current_path = dest
                if reindex_over:
                    prefix = current_path.name.split("_batch")[0]
                    reindex_groups.setdefault(prefix, []).append((batch_idx, current_path))
                continue

            new_name, changed = _insert_dataset_tag(path.name, dataset)
            if skip_below:
                skipped += 1
                continue
            if not changed:
                skipped += 1
                continue
            dest = _unique_path(path.with_name(new_name))
            if dry_run:
                print(f"[DRY] rename {path} -> {dest}")
            else:
                path.rename(dest)
            renamed += 1

    if reindex_over and reindex_groups:
        for prefix, items in reindex_groups.items():
            items_sorted = sorted(items, key=lambda x: x[0])
            temp_paths = []
            for new_idx, (_, path) in enumerate(items_sorted):
                target_name = f"{prefix}_batch{new_idx}.csv"
                target_path = path.with_name(target_name)
                if target_path == path:
                    continue
                temp_path = path.with_name(f"{target_name}.tmp")
                if dry_run:
                    print(f"[DRY] rename {path} -> {temp_path} -> {target_path}")
                else:
                    path.rename(temp_path)
                temp_paths.append((temp_path, target_path))
            for temp_path, target_path in temp_paths:
                final_path = _unique_path(target_path)
                if dry_run:
                    print(f"[DRY] finalize {temp_path} -> {final_path}")
                else:
                    temp_path.rename(final_path)
                reindexed_over += 1

    print(
        "Done. "
        f"moved={moved}, renamed={renamed}, retagged_over={retagged_over}, "
        f"reindexed_over={reindexed_over}, skipped={skipped}"
    )


if __name__ == "__main__":
    main()
