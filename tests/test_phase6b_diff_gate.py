from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "run_exp"))

from phase6_canonical_tools import compare_results, validate_canonical, write_diff_reports

REPO = Path(__file__).resolve().parents[1]
CALIBRATION_HEADER = [
    "feature",
    "p_interaction",
    "p_interaction_adj",
    "p_adjusted_global",
    "p_adjusted_global_across_datasets",
    "significant_interaction",
    "significant_global",
    "significant_global_across_datasets",
]
MIXED_HEADER = [
    "feature",
    "target",
    "p_value",
    "p_adjusted",
    "significant_fdr",
    "p_adjusted_across_datasets",
    "significant_across_datasets",
]


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _make_results_final(root: Path) -> tuple[Path, Path]:
    tables = root / "step3" / "ricechem" / "tables"
    contaminated = tables / "calibration_regression_bad_self_reported.csv"
    uncontaminated = tables / "calibration_regression_good_self_reported.csv"
    _write_csv(
        contaminated,
        CALIBRATION_HEADER,
        [
            {
                "feature": "sentinel_feature",
                "p_interaction": "-2147483648",
                "p_interaction_adj": "-2147483648",
                "p_adjusted_global": "-53687091200",
                "p_adjusted_global_across_datasets": "0",
                "significant_interaction": "True",
                "significant_global": "True",
                "significant_global_across_datasets": "True",
            },
            {
                "feature": "finite_feature",
                "p_interaction": "0.01",
                "p_interaction_adj": "0.02",
                "p_adjusted_global": "0.03",
                "p_adjusted_global_across_datasets": "0.03",
                "significant_interaction": "True",
                "significant_global": "True",
                "significant_global_across_datasets": "True",
            },
        ],
    )
    _write_csv(
        uncontaminated,
        CALIBRATION_HEADER,
        [
            {
                "feature": "uncontaminated_feature",
                "p_interaction": "0.04",
                "p_interaction_adj": "0.04",
                "p_adjusted_global": "0.03",
                "p_adjusted_global_across_datasets": "0.03",
                "significant_interaction": "True",
                "significant_global": "True",
                "significant_global_across_datasets": "True",
            }
        ],
    )
    _write_csv(
        tables / "mixed_model_effects_good_self_reported.csv",
        MIXED_HEADER,
        [
            {
                "feature": "finite_mixed",
                "target": "is_correct",
                "p_value": "0.02",
                "p_adjusted": "0.03",
                "significant_fdr": "True",
                "p_adjusted_across_datasets": "0.03",
                "significant_across_datasets": "True",
            },
            {
                "feature": "missing_mixed",
                "target": "is_correct",
                "p_value": "",
                "p_adjusted": "",
                "significant_fdr": "",
                "p_adjusted_across_datasets": "",
                "significant_across_datasets": "",
            },
        ],
    )
    return contaminated, uncontaminated


def _run_build(source: Path, dest: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": "src:."}
    return subprocess.run(
        [
            sys.executable,
            "experiments/run_exp/build_results_canonical.py",
            "--source",
            str(source),
            "--root",
            str(dest),
            "--python",
            sys.executable,
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_phase6b_build_performs_no_revert_and_accepts_global_repool(tmp_path: Path) -> None:
    final = tmp_path / "RESULTS_FINAL"
    canonical = tmp_path / "RESULTS_CANONICAL"
    _, uncontaminated = _make_results_final(final)

    result = _run_build(final, canonical)
    changes, counts = compare_results(final, canonical)
    negative, nonfinite_significant = validate_canonical(canonical)

    rebuilt_uncontaminated = canonical / uncontaminated.relative_to(final)
    rebuilt_rows = list(csv.DictReader(rebuilt_uncontaminated.open()))
    assert rebuilt_rows[0]["p_adjusted_global_across_datasets"] == "0.04"
    assert rebuilt_rows[0]["p_adjusted_global_across_datasets"] != "0.03"
    assert "no revert/filter step" in result.stdout
    assert "restored_unexpected" not in result.stdout
    assert counts["unexpected"] == 0
    assert counts["sentinel_correction"] == 7
    assert counts["repool_recompute"] >= 6
    assert any(change["file"] == uncontaminated.relative_to(final).as_posix() for change in changes)
    assert negative == 0
    assert nonfinite_significant == 0


def test_phase6b_diff_gate_exits_nonzero_for_unverifiable_change(tmp_path: Path) -> None:
    final = tmp_path / "RESULTS_FINAL"
    canonical = tmp_path / "RESULTS_CANONICAL"
    _make_results_final(final)
    _run_build(final, canonical)

    target = canonical / "step3" / "ricechem" / "tables" / "calibration_regression_good_self_reported.csv"
    rows = list(csv.DictReader(target.open()))
    rows[0]["p_interaction"] = "0.02"
    _write_csv(target, CALIBRATION_HEADER, rows)

    exit_code = write_diff_reports(final, canonical, tmp_path / "audit")
    _, counts = compare_results(final, canonical)

    assert exit_code == 1
    assert counts["unexpected"] >= 1
