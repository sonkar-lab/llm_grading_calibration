#!/usr/bin/env python3
"""Build corrected RESULTS_CANONICAL from frozen RESULTS_FINAL for Phase 6b."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from phase6_canonical_tools import sanitize_canonical_sentinel_rows


def _resolve(path: Path) -> Path:
    return path.resolve(strict=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build experiments/RESULTS_CANONICAL from RESULTS_FINAL.")
    parser.add_argument("--source", type=Path, default=Path("experiments/RESULTS_FINAL"))
    parser.add_argument("--root", type=Path, default=Path("experiments/RESULTS_CANONICAL"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    source = _resolve(args.source)
    dest = _resolve(args.root)
    repo = _resolve(Path.cwd())
    results_final = _resolve(Path("experiments/RESULTS_FINAL"))

    if dest == results_final or str(dest).startswith(f"{results_final}/"):
        raise SystemExit(f"Refusing to write RESULTS_CANONICAL into RESULTS_FINAL: {dest}")
    default_dest = _resolve(Path("experiments/RESULTS_CANONICAL"))
    source_inside_dest = str(source).startswith(f"{dest}/")
    dest_inside_source = str(dest).startswith(f"{source}/") and dest != default_dest
    if source == dest or source_inside_dest or dest_inside_source:
        raise SystemExit(f"Unsafe source/destination combination: source={source}, dest={dest}")
    if not source.exists():
        raise SystemExit(f"Missing source RESULTS_FINAL: {source}")

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)

    cmd = [
        args.python,
        "experiments/run_exp/fix_fdr_outputs.py",
        "--root",
        str(dest),
        "--steps",
        "step3",
        "--canonical-correction",
        "--tol",
        "0",
        "--no-expanded",
        "--across-datasets",
    ]
    env = {**os.environ, "PYTHONPATH": "src:."}
    subprocess.run(cmd, cwd=repo, env=env, check=True)
    changed = sanitize_canonical_sentinel_rows(dest)
    print(f"Sanitized sentinel/significance cells: {changed}")
    print("Phase 6b build performed no revert/filter step; audit gate must classify all changes honestly.")
    print(f"Built RESULTS_CANONICAL at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
