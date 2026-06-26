from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.test_publication import PUBLISHED_FEATURE_RATE_ANCHORS, PUBLISHED_TABLE4_ANCHORS

CANONICAL_ARTIFACTS = {
    "calibration_regression_interactions_within.csv",
    "calibration_regression_interactions_within.tex",
    "calibration_regression_interactions_across.csv",
    "calibration_regression_interactions_across.tex",
    "calibration_regression_interactions_mixed_diagnostic.csv",
    "calibration_regression_interactions_mixed_diagnostic.tex",
    "feature_family_significance_rates.csv",
    "feature_family_significance_rates.tex",
}


def _run_runner(output_dir: Path | str, *, results_root: str | None = "experiments/RESULTS_FINAL") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["PYTHONPATH"] = "src:."
    cmd = ["bash", "scripts/run_reproduction_pipeline.sh"]
    if results_root is not None:
        cmd.extend(["--results-root", results_root])
    cmd.extend(["--output-dir", str(output_dir)])
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _assert_artifacts_exist(output_dir: Path, stdout: str) -> None:
    for artifact in CANONICAL_ARTIFACTS:
        path = output_dir / artifact
        assert path.exists(), f"missing {path}\nstdout:\n{stdout}"
        assert str(path) in stdout


def test_reproduction_pipeline_runner_generates_published_phase5_artifacts(tmp_path):
    result = _run_runner(tmp_path, results_root="experiments/RESULTS_FINAL")

    assert result.returncode == 0, result.stderr
    _assert_artifacts_exist(tmp_path, result.stdout)

    calibration = pd.read_csv(tmp_path / "calibration_regression_interactions_mixed_diagnostic.csv")
    features = pd.read_csv(tmp_path / "feature_family_significance_rates.csv")

    beetle_token_probability = calibration.loc[
        (calibration["dataset"] == "beetle")
        & (calibration["method"] == "token_probability"),
        "pct_sig",
    ].iloc[0]
    response_rate = features.loc[features["feature_family"] == "response", "pct_sig"].iloc[0]

    assert round(beetle_token_probability, 1) == PUBLISHED_TABLE4_ANCHORS[
        ("beetle", "token_probability")
    ][3]
    assert round(response_rate, 1) == PUBLISHED_FEATURE_RATE_ANCHORS["response"]

    first_bytes = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    second_result = _run_runner(tmp_path, results_root="experiments/RESULTS_FINAL")
    assert second_result.returncode == 0, second_result.stderr
    second_bytes = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    assert second_bytes == first_bytes


def test_reproduction_pipeline_runner_default_reads_results_canonical(tmp_path):
    result = _run_runner(tmp_path, results_root=None)

    assert result.returncode == 0, result.stderr
    assert "experiments/RESULTS_CANONICAL" in result.stdout
    _assert_artifacts_exist(tmp_path, result.stdout)
    within = pd.read_csv(tmp_path / "calibration_regression_interactions_within.csv")
    across = pd.read_csv(tmp_path / "calibration_regression_interactions_across.csv")
    merged = within.merge(across, on=["dataset", "method"], suffixes=("_within", "_across"))
    assert (merged["pct_sig_within"].round(12) != merged["pct_sig_across"].round(12)).any()


@pytest.mark.parametrize(
    "protected_output",
    [
        "experiments/RESULTS_FINAL/should_not_write",
        "experiments/RESULTS_CANONICAL/should_not_write",
        "overleaf_paperv3/tables/should_not_write",
    ],
)
def test_reproduction_pipeline_runner_refuses_writing_into_protected_dir(protected_output):
    result = _run_runner(protected_output, results_root="experiments/RESULTS_FINAL")

    assert result.returncode != 0
    assert "refusing to write output under protected directory" in result.stderr
    assert not Path(protected_output).exists()
