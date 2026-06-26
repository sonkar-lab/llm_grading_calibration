#!/usr/bin/env python3
"""Build RESULTS_FINAL from RESULTS with step-level structure and Step3 merges."""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import shlex
from collections import defaultdict

from grading_calibration.inference.model_aliases import MODEL_ALIASES as ALIAS_MAP

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RESULTS"))
DEFAULT_OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RESULTS_FINAL"))

METHODS_DEFAULT = ["self_reported", "self_consistency", "token_probability"]
MODELS_DEFAULT_BIG = [
    "gpt_oss_120b",
    "qwen3_next_80b_a3b_instruct",
    "qwen3_next_80b_a3b_thinking",
    "llama_3.3_70b_instruct",
    "gpt_oss_20b",
    "llama_3.1_8b_instruct",
    "qwen3_4b_thinking_2507",
]
MODEL_TOKEN_PATTERN = re.compile(r"^(gpt|qwen|llama)", re.IGNORECASE)


def run_type(name):
    if "rubric_only" in name and "calreg" in name:
        return "rubric_only_calreg"
    if "rubric_only" in name:
        return "rubric_only"
    if "question_only" in name and "calreg" in name:
        return "question_only_calreg"
    if "question_only" in name:
        return "question_only"
    if "main_calreg" in name:
        return "main_calreg"
    if name.startswith("step3_pass1_"):
        return "main"
    return None


def extract_arg_list(tokens, flag):
    if flag not in tokens:
        return None
    idx = tokens.index(flag)
    vals = []
    for tok in tokens[idx + 1 :]:
        if tok.startswith("--"):
            break
        vals.append(tok)
    return vals or None


def parse_readme(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    low = text.lower()
    datasets = set()
    if "all datasets" in low:
        datasets.update(["ricechem", "beetle", "scientsbank"])
    for ds in ["ricechem", "beetle", "scientsbank"]:
        if re.search(rf"\b{ds}\b", low) and "all datasets" not in low:
            datasets.add(ds)

    tokens = []
    try:
        tokens = re.findall(r"[^\s]+", text)
    except Exception:
        tokens = []
    models = extract_arg_list(tokens, "--models")
    methods = extract_arg_list(tokens, "--methods")

    if not models:
        m = re.search(r"(?i)models?:\s*([^\n]+)", text)
        if m:
            models = m.group(1).strip().split()
        else:
            m2 = re.search(r"(?i)only\s+([\w\-\.]+)", text)
            if m2 and MODEL_TOKEN_PATTERN.search(m2.group(1)):
                models = [m2.group(1)]
            elif re.search(r"(?i)all models", text):
                models = MODELS_DEFAULT_BIG

    if not methods:
        if re.search(r"(?i)all methods", text):
            methods = METHODS_DEFAULT
        else:
            m3 = re.search(r"(?i)methods?:\s*([^\n]+)", text)
            if m3:
                methods = m3.group(1).strip().split()

    return text, datasets, models, methods


def normalize_models(models, step):
    if step == "step6":
        return models
    return [ALIAS_MAP.get(m, m) for m in models]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_tree(src, dst):
    if not os.path.isdir(src):
        return
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for d in dirs:
            ensure_dir(os.path.join(dst, rel, d))
        for f in files:
            src_f = os.path.join(root, f)
            dst_f = os.path.join(dst, rel, f)
            if os.path.exists(dst_f):
                # skip duplicates
                continue
            ensure_dir(os.path.dirname(dst_f))
            shutil.copy2(src_f, dst_f)


def list_runs():
    return sorted(
        d
        for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    )


def schema_diff_step3(out_root):
    headers = defaultdict(lambda: defaultdict(set))
    for run in list_runs():
        rtype = run_type(run)
        if not rtype:
            continue
        run_path = os.path.join(RESULTS_DIR, run)
        csvs = glob.glob(os.path.join(run_path, "**", "*.csv"), recursive=True)
        for path in csvs:
            base = os.path.basename(path)
            if base.startswith("mixed_model_effects_"):
                kind = "mixed_model"
            elif base.startswith("calibration_regression_"):
                kind = "calreg"
            else:
                continue
            try:
                with open(path, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader)
            except Exception:
                continue
            headers[rtype][kind].update(header)

    main_cols = {k: headers["main"].get(k, set()) for k in ["mixed_model", "calreg"]}

    out_path = os.path.join(out_root, "step3", "schema_diff.md")
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Step3 schema diff\n\n")
        for kind in ["mixed_model", "calreg"]:
            f.write(f"## {kind} schema diff\n")
            f.write("| run_type | col_count | missing_vs_main | extra_vs_main |\n")
            f.write("|---|---:|---|---|\n")
            for rtype in sorted(headers):
                if kind not in headers[rtype]:
                    continue
                cols = sorted(headers[rtype][kind])
                main = main_cols.get(kind, set())
                missing = sorted(main - set(cols))
                extra = sorted(set(cols) - main)
                f.write(
                    f"| {rtype} | {len(cols)} | "
                    f"{', '.join(missing) if missing else '-'} | "
                    f"{', '.join(extra) if extra else '-'} |\n"
                )
            f.write("\n")


def merge_step3_tables(out_root, provenance_dir):
    # Gather all mixed_model and calreg CSVs from step3 runs
    runs = [r for r in list_runs() if r.startswith("step3_pass1_")]
    sources = []
    for run in runs:
        run_path = os.path.join(RESULTS_DIR, run)
        for ds in ["ricechem", "beetle", "scientsbank"]:
            csvs = glob.glob(os.path.join(run_path, ds, "**", "*.csv"), recursive=True)
            for path in csvs:
                base = os.path.basename(path)
                if base.startswith("mixed_model_effects_"):
                    sources.append(("mixed_model", run, ds, path))
                elif base.startswith("calibration_regression_"):
                    sources.append(("calreg", run, ds, path))

    # Group by dataset/model/method
    grouped = defaultdict(list)
    for kind, run, ds, path in sources:
        base = os.path.basename(path)
        # parse model/method from filename
        # e.g., mixed_model_effects_gpt_oss_20b_self_reported.csv
        m = re.match(r"^(mixed_model_effects|calibration_regression)_(.+)_(self_reported|self_consistency|token_probability)\.csv$", base)
        if not m:
            continue
        model = m.group(2)
        method = m.group(3)
        grouped[(kind, ds, model, method)].append((run, path))

    for (kind, ds, model, method), files in grouped.items():
        rows = []
        all_cols = []
        col_set = set()
        for run, path in files:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                for c in cols:
                    if c not in col_set:
                        col_set.add(c)
                        all_cols.append(c)
                for row in reader:
                    row["_source_run"] = run
                    row["_source_path"] = path
                    row["_source_type"] = run_type(run) or "unknown"
                    rows.append(row)

        # align columns
        for row in rows:
            for c in all_cols:
                if c not in row:
                    row[c] = ""

        # dedupe by feature (keep most complete)
        if "feature" in all_cols:
            def non_null_count(r):
                return sum(1 for c in all_cols if r.get(c) not in ("", None))
            def priority(r):
                order = [
                    "main",
                    "main_calreg",
                    "question_only",
                    "question_only_calreg",
                    "rubric_only",
                    "rubric_only_calreg",
                ]
                try:
                    return order.index(r.get("_source_type"))
                except ValueError:
                    return len(order)
            dedup = {}
            for r in rows:
                key = r.get("feature")
                if key is None:
                    continue
                score = (non_null_count(r), -priority(r))
                if key not in dedup or score > dedup[key][0]:
                    dedup[key] = (score, r)
            rows = [v[1] for v in dedup.values()]

        # drop internal columns
        for r in rows:
            r.pop("_source_run", None)
            r.pop("_source_path", None)
            r.pop("_source_type", None)

        # write merged table
        out_dir = os.path.join(out_root, "step3", ds, "tables")
        ensure_dir(out_dir)
        if kind == "mixed_model":
            out_name = f"mixed_model_effects_{model}_{method}.csv"
        else:
            out_name = f"calibration_regression_{model}_{method}.csv"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            writer.writerows(rows)

        # write provenance
        prov = {
            "kind": kind,
            "dataset": ds,
            "model": model,
            "method": method,
            "sources": [
                {
                    "run": run,
                    "path": path,
                    "run_type": run_type(run) or "unknown",
                }
                for run, path in files
            ],
            "schema_alignment": {
                "added_columns": sorted(col_set),
            },
        }
        ensure_dir(provenance_dir)
        prov_path = os.path.join(provenance_dir, out_name + ".source.json")
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2)


def copy_step(run_prefix, out_root):
    runs = [r for r in list_runs() if r.startswith(run_prefix + "_")]
    for run in runs:
        run_path = os.path.join(RESULTS_DIR, run)
        for ds in ["ricechem", "beetle", "scientsbank"]:
            ds_path = os.path.join(run_path, ds)
            if not os.path.isdir(ds_path):
                continue
            # find step dir inside dataset (ds/run_id/stepX)
            step_dir = None
            candidates = [
                p
                for p in glob.glob(os.path.join(ds_path, "**", run_prefix), recursive=True)
                if os.path.isdir(p)
            ]
            # prefer a candidate that actually has tables/plots subdirs
            for candidate in candidates:
                if os.path.isdir(os.path.join(candidate, "tables")) or os.path.isdir(
                    os.path.join(candidate, "plots")
                ):
                    step_dir = candidate
                    break
            if not step_dir and candidates:
                step_dir = candidates[0]
            if not step_dir:
                continue
            target_base = os.path.join(out_root, run_prefix, ds)
            # copy tables/plots/logs
            copy_tree(os.path.join(step_dir, "tables"), os.path.join(target_base, "tables"))
            copy_tree(os.path.join(step_dir, "plots"), os.path.join(target_base, "plots"))
            # logs by run_id
            logs_dst = os.path.join(target_base, "logs", "by_run_id", run)
            ensure_dir(logs_dst)
            copy_tree(os.path.join(run_path, "logs"), logs_dst)


def write_readmes(out_root):
    # collect readmes per step
    steps = defaultdict(list)
    for run in list_runs():
        step = run.split("_")[0]
        readme = None
        for cand in [os.path.join(RESULTS_DIR, run, "readme.txt"), os.path.join(RESULTS_DIR, run, "logs", "readme.txt")]:
            if os.path.isfile(cand):
                readme = cand
                break
        if not readme:
            continue
        steps[step].append((run, readme))

    step_flags = {
        "step2": [
            "--split",
            "--compute-ci",
            "--n-bootstrap",
            "--ci-method",
            "--bootstrap-block",
        ],
        "step3": [
            "--effects-method",
            "--effects-backend",
            "--effects-lme4-maxfun",
            "--n-perm",
            "--n-boot",
            "--bootstrap-method",
            "--calibration-regression-mode",
            "--calibration-regression-bootstrap",
            "--effects-include-prefixes",
            "--effects-exclude-prefixes",
            "--effects-include-features",
            "--effects-include-features-file",
            "--effects-question-constant-separate-block",
        ],
        "step4": [
            "--stratify-by",
            "--compute-ci",
            "--n-bootstrap",
            "--ci-method",
        ],
        "step5": [
            "--n-perm",
            "--n-boot",
            "--bootstrap-method",
        ],
        "step6": [
            "--n-perm",
            "--n-boot",
            "--ci-method",
            "--skip-interaction-tests",
        ],
    }

    def extract_run_flags(run, step):
        logs_dir = os.path.join(RESULTS_DIR, run, "logs")
        if not os.path.isdir(logs_dir):
            return {}
        flags_of_interest = step_flags.get(step, [])
        found = defaultdict(set)
        for path in glob.glob(os.path.join(logs_dir, "*.log")):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if f"Running {step}" in line and "arguments:" in line:
                            try:
                                tokens = shlex.split(line)
                            except Exception:
                                tokens = line.split()
                            for i, tok in enumerate(tokens):
                                if tok in flags_of_interest:
                                    # capture value if present
                                    val = True
                                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                                        val = tokens[i + 1]
                                    found[tok].add(str(val))
            except Exception:
                continue
        # normalize to string values
        out = {}
        for k in flags_of_interest:
            if k in found:
                vals = sorted(found[k])
                out[k] = vals[0] if len(vals) == 1 else vals
        return out

    for step, entries in steps.items():
        out_path = os.path.join(out_root, step, "readme.txt")
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Step {step} combined results\n")
            f.write("\nSources:\n")
            for run, _ in entries:
                f.write(f"- {run}\n")
            f.write("\nNotes by run:\n")
            for run, path in entries:
                f.write(f"\n[{run}]\n")
                f.write(open(path, "r", encoding="utf-8", errors="ignore").read().strip())
                f.write("\n")
                flags = extract_run_flags(run, step)
                if flags:
                    f.write("Flags (paper-relevant):\n")
                    for k, v in flags.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write("Flags (paper-relevant): not found in logs\n")
                f.write("\n")


def mismatch_report(out_root):
    # record dataset/model/method mismatches per run vs readme
    report_path = os.path.join(out_root, "mismatch_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for run in list_runs():
            readme = None
            for cand in [os.path.join(RESULTS_DIR, run, "readme.txt"), os.path.join(RESULTS_DIR, run, "logs", "readme.txt")]:
                if os.path.isfile(cand):
                    readme = cand
                    break
            if not readme:
                continue
            text, datasets, models, methods = parse_readme(readme)
            if not datasets:
                datasets = {"ricechem", "beetle", "scientsbank"}
            if not models:
                models = MODELS_DEFAULT_BIG
            if not methods:
                methods = METHODS_DEFAULT
            step = run.split("_")[0]
            models = normalize_models(models, step)
            f.write(f"{run} ({step})\n")
            f.write(f"  datasets_expected: {sorted(datasets)}\n")
            f.write(f"  models_expected: {models}\n")
            f.write(f"  methods_expected: {methods}\n")
            # dataset dirs
            run_path = os.path.join(RESULTS_DIR, run)
            ds_dirs = [d for d in ["ricechem", "beetle", "scientsbank"] if os.path.isdir(os.path.join(run_path, d))]
            f.write(f"  datasets_present: {ds_dirs}\n\n")
            # model/method combo checks
            for ds in sorted(datasets):
                if ds not in ds_dirs:
                    f.write(f"  {ds}: missing dataset dir\n")
                    continue
                ds_path = os.path.join(run_path, ds)
                csvs = glob.glob(os.path.join(ds_path, "**", "*.csv"), recursive=True)
                found = set()
                if step == "step6":
                    ms = glob.glob(os.path.join(ds_path, "**", "metrics_summary.csv"), recursive=True)
                    if ms:
                        with open(ms[0], newline="") as ms_f:
                            reader = csv.DictReader(ms_f)
                            for row in reader:
                                m = row.get("model")
                                me = row.get("method")
                                if m in models and me in methods:
                                    found.add((m, me))
                else:
                    for path in csvs:
                        name = os.path.basename(path)
                        for m in models:
                            if m in name:
                                for me in methods:
                                    if me in name:
                                        found.add((m, me))
                missing = [(m, me) for m in models for me in methods if (m, me) not in found]
                if missing:
                    f.write(f"  {ds}: missing {len(missing)} model/method combos\n")
                    miss_by_model = {}
                    for m, me in missing:
                        miss_by_model.setdefault(m, []).append(me)
                    for m in sorted(miss_by_model):
                        miss = miss_by_model[m]
                        if len(miss) == len(methods):
                            f.write(f"    - {m}: all methods\n")
                        else:
                            f.write(f"    - {m}: {', '.join(sorted(miss))}\n")
                else:
                    f.write(f"  {ds}: OK\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_root)
    ensure_dir(out_root)

    # schema diff for step3
    schema_diff_step3(out_root)

    # step copy
    for step in ["step2", "step4", "step5", "step6"]:
        copy_step(step, out_root)

    # step3 merge
    provenance_dir = os.path.join(out_root, "step3", "provenance")
    merge_step3_tables(out_root, provenance_dir)

    # readmes
    write_readmes(out_root)

    # mismatch report
    mismatch_report(out_root)

    print(f"RESULTS_FINAL built at {out_root}")


if __name__ == "__main__":
    main()
