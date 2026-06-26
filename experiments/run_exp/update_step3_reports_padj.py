#!/usr/bin/env python3
"""
Update Step 3 Pass1/Pass2 reports with refreshed adjusted p-values.

Scopes:
 - step3_pass1_results_feature_analysis_small_models_all_datasets.md
 - step3_pass2_results_feature_analysis_small_models.md

Assumes fix_fdr_outputs.py has already populated p_adjusted/p_adjusted_global.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from typing import Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs_runs", "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs_runs", "run_files")

MODEL_DISPLAY_TO_SLUG = {
    "Llama-3.1-8B-Instruct": "llama_3.1_8b_instruct",
    "Qwen3-4B": "qwen3_4b_thinking_2507",
    "Qwen3-4B-Thinking-2507": "qwen3_4b_thinking_2507",
    "gpt-oss-20b": "gpt_oss_20b",
    "Llama-3.3-70B-Instruct": "llama_3.3_70b_instruct",
    "Qwen3-Next-80B-A3B-Instruct": "qwen3_next_80b_a3b_instruct",
    "Qwen3-Next-80B-A3B-Thinking": "qwen3_next_80b_a3b_thinking",
    "gpt-oss-120b": "gpt_oss_120b",
}

DATASET_NAME_TO_DIR = {
    "RiceChem": "RiceChem",
    "Scientsbank": "Scientsbank",
    "Beetle": "Beetle",
}


def _format_p(p: float) -> str:
    if p == 0:
        return "0"
    return format(p, ".3g")


def _read_csv(path: str):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_step3_pass1_maps():
    mixed = {}
    calib = {}
    base = os.path.join(OUTPUTS_DIR, "step3_pass1")
    for dataset, ddir in DATASET_NAME_TO_DIR.items():
        tables_dir = os.path.join(base, ddir, "step3", "tables")
        if not os.path.isdir(tables_dir):
            continue
        for path in glob.glob(os.path.join(tables_dir, "mixed_model_effects_*.csv")):
            fname = os.path.basename(path)
            if not fname.startswith("mixed_model_effects_"):
                continue
            stem = fname[len("mixed_model_effects_"):-4]
            method = None
            model = None
            for m in ("self_reported", "self_consistency", "token_probability"):
                if stem.endswith("_" + m):
                    method = m
                    model = stem[: -len("_" + m)]
                    break
            if not method:
                continue
            for row in _read_csv(path):
                feature = row.get("feature") or ""
                target = row.get("target") or ""
                try:
                    p_adj = float(row.get("p_adjusted", ""))
                except ValueError:
                    continue
                mixed[(dataset, model, method, target, feature)] = p_adj
        for path in glob.glob(os.path.join(tables_dir, "calibration_regression_*.csv")):
            fname = os.path.basename(path)
            if not fname.startswith("calibration_regression_"):
                continue
            stem = fname[len("calibration_regression_"):-4]
            method = None
            model = None
            for m in ("self_reported", "self_consistency", "token_probability"):
                if stem.endswith("_" + m):
                    method = m
                    model = stem[: -len("_" + m)]
                    break
            if not method:
                continue
            for row in _read_csv(path):
                feature = row.get("feature") or ""
                try:
                    p_adj = float(row.get("p_adjusted_global", ""))
                except ValueError:
                    continue
                calib[(dataset, model, method, feature)] = p_adj
    return mixed, calib


def _load_step3_pass2_maps():
    nonlinear = {}
    nonlinear_counts = {}
    calib = {}
    calib_counts = {}
    base = os.path.join(OUTPUTS_DIR, "step3_pass2")
    for dataset, ddir in DATASET_NAME_TO_DIR.items():
        tables_dir = os.path.join(base, ddir, "step3", "tables")
        if not os.path.isdir(tables_dir):
            continue
        for path in glob.glob(os.path.join(tables_dir, "nonlinear_effects_*.csv")):
            fname = os.path.basename(path)
            stem = fname[len("nonlinear_effects_"):-4]
            method = None
            model = None
            for m in ("self_reported", "self_consistency", "token_probability"):
                if stem.endswith("_" + m):
                    method = m
                    model = stem[: -len("_" + m)]
                    break
            if not method:
                continue
            counts = {}
            for row in _read_csv(path):
                feature = row.get("feature") or ""
                target = row.get("target") or ""
                try:
                    p_adj = float(row.get("lrt_p_adj", ""))
                except ValueError:
                    continue
                try:
                    delta = float(row.get("delta_10_90", ""))
                except ValueError:
                    delta = None
                nonlinear[(dataset, model, method, target, feature)] = (p_adj, delta)
                counts[target] = counts.get(target, 0) + (1 if p_adj <= 0.05 else 0)
            for target, n_sig in counts.items():
                nonlinear_counts[(dataset, model, method, target)] = n_sig
        for path in glob.glob(os.path.join(tables_dir, "calibration_regression_*.csv")):
            fname = os.path.basename(path)
            stem = fname[len("calibration_regression_"):-4]
            method = None
            model = None
            for m in ("self_reported", "self_consistency", "token_probability"):
                if stem.endswith("_" + m):
                    method = m
                    model = stem[: -len("_" + m)]
                    break
            if not method:
                continue
            n_sig = 0
            for row in _read_csv(path):
                feature = row.get("feature") or ""
                try:
                    p_adj = float(row.get("p_adjusted_global", ""))
                except ValueError:
                    continue
                try:
                    ame = float(row.get("ame", ""))
                except ValueError:
                    ame = None
                calib[(dataset, model, method, feature)] = (p_adj, ame)
                if p_adj <= 0.05:
                    n_sig += 1
            calib_counts[(dataset, model, method)] = n_sig
    return nonlinear, nonlinear_counts, calib, calib_counts


def _update_summary_cell(cell: str, dataset: str, model: str, method: str, target: str, mixed_map):
    pattern = re.compile(r"([A-Za-z0-9_]+) \(([^,]+),\s*([^)]+)\)")

    def repl(match):
        feature = match.group(1)
        beta = match.group(2)
        old = match.group(3).strip()
        ns_suffix = " (ns)" if old.endswith("(ns)") else ""
        old_num = old.replace("(ns)", "").strip()
        model_slug = MODEL_DISPLAY_TO_SLUG.get(model)
        if not model_slug:
            return match.group(0)
        p_adj = mixed_map.get((dataset, model_slug, method, target, feature))
        if p_adj is None:
            return match.group(0)
        new_p = _format_p(p_adj)
        if p_adj >= 0.05:
            return f"{feature} ({beta}, {new_p} (ns))"
        return f"{feature} ({beta}, {new_p})"

    return pattern.sub(repl, cell)


def _update_summary_cell_calib(cell: str, dataset: str, model: str, method: str, calib_map):
    pattern = re.compile(r"([A-Za-z0-9_]+) \(([^,]+),\s*([^)]+)\)")

    def repl(match):
        feature = match.group(1)
        beta = match.group(2)
        old = match.group(3).strip()
        ns_suffix = " (ns)" if old.endswith("(ns)") else ""
        model_slug = MODEL_DISPLAY_TO_SLUG.get(model)
        if not model_slug:
            return match.group(0)
        p_adj = calib_map.get((dataset, model_slug, method, feature))
        if p_adj is None:
            return match.group(0)
        new_p = _format_p(p_adj)
        if p_adj >= 0.05:
            return f"{feature} ({beta}, {new_p} (ns))"
        return f"{feature} ({beta}, {new_p})"

    return pattern.sub(repl, cell)


def update_step3_pass1_report(report_path: str) -> None:
    mixed_map, calib_map = _load_step3_pass1_maps()
    with open(report_path) as f:
        lines = f.read().splitlines()

    dataset = "RiceChem"
    method = None
    model = None
    target = None
    table_mode = None
    in_table = False
    header_cells = None
    summary_mode = None
    summary_header = None

    updated = []
    for line in lines:
        dataset_match = re.match(r"## Dataset Addendum: (.+)", line)
        if dataset_match:
            dataset = dataset_match.group(1).strip()
            if dataset not in DATASET_NAME_TO_DIR:
                dataset = "RiceChem"
        method_match = re.match(r"#+ Method: (.+)", line)
        if method_match:
            method = method_match.group(1).strip()
        model_match = re.match(r"### Model: (.+)", line)
        if model_match:
            model = model_match.group(1).strip()
        if "Mixed-Model Feature Effects on Confidence" in line:
            target = "Confidence"
            table_mode = "mixed"
        elif "Mixed-Model Feature Effects on Correctness" in line:
            target = "Correctness"
            table_mode = "mixed"
        elif "Calibration Regression" in line and "Top Interactions" in line:
            table_mode = "calib"
            target = None

        if line.startswith("| Model | Top Confidence Features"):
            summary_mode = "mixed"
            summary_header = [c.strip() for c in line.strip("|").split("|")]
            updated.append(line)
            continue
        if line.startswith("| Model | Top Interaction Features"):
            summary_mode = "calib"
            summary_header = [c.strip() for c in line.strip("|").split("|")]
            updated.append(line)
            continue

        if summary_mode and line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if summary_mode == "mixed":
                if len(cells) >= 3:
                    row_model = cells[0]
                    cells[1] = _update_summary_cell(cells[1], dataset, row_model, method, "Confidence", mixed_map)
                    cells[2] = _update_summary_cell(cells[2], dataset, row_model, method, "Correctness", mixed_map)
                updated.append("| " + " | ".join(cells) + " |")
                continue
            if summary_mode == "calib":
                if len(cells) >= 2:
                    row_model = cells[0]
                    cells[1] = _update_summary_cell_calib(cells[1], dataset, row_model, method, calib_map)
                updated.append("| " + " | ".join(cells) + " |")
                continue

        if summary_mode and (not line.startswith("|") or line.strip() == ""):
            summary_mode = None
            summary_header = None

        if line.startswith("| Feature |") and "p_adj" in line:
            in_table = True
            header_cells = [c.strip() for c in line.strip("|").split("|")]
            updated.append(line)
            continue

        if in_table:
            if not line.startswith("|") or line.startswith("|---"):
                if line.startswith("|---"):
                    updated.append(line)
                    continue
                in_table = False
                header_cells = None
            else:
                cells = [c.strip() for c in line.strip("|").split("|")]
                try:
                    p_idx = header_cells.index("p_adj")
                except ValueError:
                    p_idx = None
                sig_idx = None
                if "Sig" in header_cells:
                    sig_idx = header_cells.index("Sig")
                feature = cells[0] if cells else ""
                model_slug = MODEL_DISPLAY_TO_SLUG.get(model or "")
                if p_idx is not None and model_slug and method:
                    if table_mode == "mixed" and target:
                        p_adj = mixed_map.get((dataset, model_slug, method, target, feature))
                    elif table_mode == "calib":
                        p_adj = calib_map.get((dataset, model_slug, method, feature))
                    else:
                        p_adj = None
                    if p_adj is not None:
                        cells[p_idx] = _format_p(p_adj)
                        if sig_idx is not None:
                            if p_adj <= 0.001:
                                cells[sig_idx] = "***"
                            elif p_adj <= 0.01:
                                cells[sig_idx] = "**"
                            elif p_adj <= 0.05:
                                cells[sig_idx] = "*"
                            else:
                                cells[sig_idx] = ""
                updated.append("| " + " | ".join(cells) + " |")
                continue

        updated.append(line)

    with open(report_path, "w") as f:
        f.write("\n".join(updated) + "\n")


def update_step3_pass2_report(report_path: str) -> None:
    nonlinear_map, nonlinear_counts, calib_map, calib_counts = _load_step3_pass2_maps()
    with open(report_path) as f:
        lines = f.read().splitlines()

    dataset = None
    target = None
    in_table = False
    header_cells = None
    table_kind = None

    updated = []
    for line in lines:
        dataset_match = re.match(r"## Dataset: (.+)", line)
        if dataset_match:
            dataset = dataset_match.group(1).strip()
        target_match = re.match(r"#### Target: (.+)", line)
        if target_match:
            target = target_match.group(1).strip()

        if line.startswith("| Model | Method | n_sig") and "LRT" in line:
            table_kind = "nonlinear"
            in_table = True
            header_cells = [c.strip() for c in line.strip("|").split("|")]
            updated.append(line)
            continue
        if line.startswith("| Model | Method | n_sig") and "p_adj" in line and "AME" in line:
            table_kind = "calib"
            in_table = True
            header_cells = [c.strip() for c in line.strip("|").split("|")]
            updated.append(line)
            continue

        if in_table:
            if line.startswith("|---"):
                updated.append(line)
                continue
            if not line.startswith("|"):
                in_table = False
                header_cells = None
                table_kind = None
            else:
                cells = [c.strip() for c in line.strip("|").split("|")]
                model = cells[0]
                method = cells[1]
                model_slug = MODEL_DISPLAY_TO_SLUG.get(model, model)
                if table_kind == "nonlinear":
                    # columns: Model | Method | n_sig | Top feature | Delta_10_90 | LRT p_adj
                    n_sig_idx = header_cells.index("n_sig (LRT p_adj<0.05)") if "n_sig (LRT p_adj<0.05)" in header_cells else 2
                    delta_idx = header_cells.index("Delta_10_90") if "Delta_10_90" in header_cells else 4
                    padj_idx = header_cells.index("LRT p_adj") if "LRT p_adj" in header_cells else 5
                    feature = cells[3]
                    if dataset and target:
                        n_sig = nonlinear_counts.get((dataset, model_slug, method, target))
                        if n_sig is not None:
                            cells[n_sig_idx] = str(n_sig)
                        entry = nonlinear_map.get((dataset, model_slug, method, target, feature))
                        if entry is not None:
                            p_adj, delta = entry
                            if delta is not None:
                                cells[delta_idx] = str(delta)
                            cells[padj_idx] = _format_p(p_adj)
                elif table_kind == "calib":
                    # columns: Model | Method | n_sig (p_adj<0.05) | Top feature | AME | p_adj
                    n_sig_idx = header_cells.index("n_sig (p_adj<0.05)") if "n_sig (p_adj<0.05)" in header_cells else 2
                    ame_idx = header_cells.index("AME") if "AME" in header_cells else 4
                    padj_idx = header_cells.index("p_adj") if "p_adj" in header_cells else 5
                    feature = cells[3]
                    if dataset:
                        n_sig = calib_counts.get((dataset, model_slug, method))
                        if n_sig is not None:
                            cells[n_sig_idx] = str(n_sig)
                        entry = calib_map.get((dataset, model_slug, method, feature))
                        if entry is not None:
                            p_adj, ame = entry
                            if ame is not None:
                                cells[ame_idx] = str(ame)
                            cells[padj_idx] = _format_p(p_adj)
                updated.append("| " + " | ".join(cells) + " |")
                continue

        updated.append(line)

    with open(report_path, "w") as f:
        f.write("\n".join(updated) + "\n")


def main():
    pass1_report = os.path.join(
        REPORTS_DIR,
        "step3_pass1",
        "Report",
        "step3_pass1_results_feature_analysis_small_models_all_datasets.md",
    )
    pass2_report = os.path.join(
        REPORTS_DIR,
        "step3_pass2",
        "Report",
        "step3_pass2_results_feature_analysis_small_models.md",
    )
    if os.path.isfile(pass1_report):
        update_step3_pass1_report(pass1_report)
    if os.path.isfile(pass2_report):
        update_step3_pass2_report(pass2_report)


if __name__ == "__main__":
    main()
