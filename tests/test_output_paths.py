import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grading_calibration.metrics.cross_comparison import load_all_predictions, MODEL_NICKNAMES
from utils.output_paths import build_output_manager


def _write_min_predictions(path: Path) -> None:
    df = pd.DataFrame(
        {
            "instance_id": ["a1"],
            "prediction": [1],
            "raw_confidence": [0.75],
            "ground_truth": [1],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_output_manager_manifest(tmp_path: Path) -> None:
    manager = build_output_manager(
        output_root=str(tmp_path),
        dataset="ricechem",
        run_id="run123",
        save_legacy=False,
        write_manifest=True,
    )
    tables_dir = manager.step_dir("step3", "tables")
    out_path = tables_dir / "dummy.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("x\n1\n", encoding="utf-8")

    manager.register(step="step3", artifact="dummy", path=out_path)
    manifest_path = manager.write_manifest()
    assert manifest_path is not None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "ricechem"
    assert payload["run_id"] == "run123"
    assert payload["outputs"][0]["artifact"] == "dummy"
    assert payload["outputs"][0]["path"] == str(out_path)


def test_load_all_predictions_flat_layout(tmp_path: Path) -> None:
    model = "qwen4b"
    method = "self_reported"
    model_name = MODEL_NICKNAMES.get(model, model)
    pred_path = tmp_path / f"predictions_{model_name}_{method}.csv"
    _write_min_predictions(pred_path)

    all_preds = load_all_predictions(
        [model],
        [method],
        tmp_path,
        flat_layout=True,
        filter_by_dataset=False,
    )
    assert (model, method) in all_preds


def test_load_all_predictions_filters_dataset(tmp_path: Path) -> None:
    model = "qwen4b"
    method = "self_reported"
    model_name = MODEL_NICKNAMES.get(model, model)
    method_dir = tmp_path / method
    _write_min_predictions(method_dir / f"predictions_{model_name}_{method}_20240101.csv")
    _write_min_predictions(method_dir / f"predictions_{model_name}_{method}_beetle_20240102.csv")

    all_preds = load_all_predictions(
        [model],
        [method],
        tmp_path,
        dataset="ricechem",
        flat_layout=False,
        filter_by_dataset=True,
    )
    assert (model, method) in all_preds
