from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run_fix_fdr(root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    return subprocess.run(
        [
            sys.executable,
            "experiments/run_exp/fix_fdr_outputs.py",
            "--root",
            str(root),
            "--steps",
            "step3",
            "--no-expanded",
            *extra_args,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def test_fix_fdr_outputs_default_off_byte_reproduces_results_final_subset(tmp_path: Path) -> None:
    source = Path("experiments/RESULTS_FINAL/step3/ricechem/tables")
    copied_tables = tmp_path / "step3" / "ricechem" / "tables"
    shutil.copytree(source, copied_tables)

    before = {path.name: path.read_bytes() for path in sorted(copied_tables.glob("*.csv"))}
    result = _run_fix_fdr(tmp_path, "--tol", "1e-8")
    after = {path.name: path.read_bytes() for path in sorted(copied_tables.glob("*.csv"))}

    assert result.returncode == 0, result.stderr
    assert "WROTE: 0 rows updated" in result.stdout
    assert after == before


def test_fix_fdr_outputs_canonical_correction_excludes_sentinel_rows(tmp_path: Path) -> None:
    tables = tmp_path / "step3" / "synthetic" / "tables"
    tables.mkdir(parents=True)
    path = tables / "calibration_regression_demo_self_reported.csv"
    fieldnames = [
        "feature",
        "p_interaction",
        "p_adjusted_global",
        "significant_global",
    ]
    rows = [
        {
            "feature": "sentinel_failed_fit",
            "p_interaction": "-2147483648",
            "p_adjusted_global": "0",
            "significant_global": "True",
        },
        {
            "feature": "finite_feature",
            "p_interaction": "0.01",
            "p_adjusted_global": "0.01",
            "significant_global": "True",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = _run_fix_fdr(tmp_path, "--canonical-correction", "--tol", "1e-12")

    assert result.returncode == 0, result.stderr
    corrected = pd.read_csv(path)
    sentinel = corrected.loc[corrected["feature"] == "sentinel_failed_fit"].iloc[0]
    finite = corrected.loc[corrected["feature"] == "finite_feature"].iloc[0]

    assert pd.isna(sentinel["p_adjusted_global"])
    assert sentinel["significant_global"] is False or str(sentinel["significant_global"]) == "False"
    assert finite["p_adjusted_global"] == 0.01
    assert finite["significant_global"] is True or str(finite["significant_global"]) == "True"
