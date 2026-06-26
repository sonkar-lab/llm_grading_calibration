# LLM Grading Confidence Calibration

Code for estimating and calibrating confidence in LLM-based automated grading.

Rather than only trying to make the grader more accurate, this project asks a
complementary question: **can we predict when an LLM grader is likely to be
correct?** A reliable confidence estimate enables *selective automation* —
high-confidence cases are graded automatically while uncertain cases are routed
to a human.

The repository contains the full pipeline used in the paper: dataset adapters,
confidence-estimation methods, post-hoc calibrators, evaluation metrics, and the
statistical analysis that produces the publication tables.

> **Public release note.** This repository ships the **code** only. The grading
> datasets and the large-model inference outputs (raw prediction CSVs) are not
> bundled — they are large and produced by running the models yourself. Obtain
> the datasets from their original sources and run
> `experiments/run_exp/step1_run_inference.py` to regenerate predictions.

---

## What's in here

| Concern | Location | Summary |
|---|---|---|
| Dataset adapters | `grading_calibration.datasets` | Load RiceChem / SciEntsBank / Beetle into a standard format |
| LLM inference | `grading_calibration.inference` | Batched generation (vLLM / HF / OpenAI), token probabilities, self-consistency sampling, perturbations, prompts |
| Confidence methods | `grading_calibration.confidence` | Token probability, self-reported, self-consistency, perturbation, hybrid |
| Calibration | `grading_calibration.calibration` | Isotonic, temperature scaling, Platt scaling |
| Metrics | `grading_calibration.metrics` | ECE, Brier, AUC-ROC, risk-coverage, calibratability checks |
| Statistics | `grading_calibration.statistics` | FDR correction, bootstrap CIs, mixed models |
| Analysis pipelines | `grading_calibration.pipelines` | Feature, stratified, commonality, regression, cross-comparison analyses + publication tables |
| Data models | `grading_calibration.domain.models` | `GradingInstance`, `PredictionResult`, `DatasetSplit` |

Everything flows through two dataclasses — `GradingInstance` (input) and
`PredictionResult` (output) — so modules are interchangeable.

---

## Repository layout

```
grading_calibration/
├── src/grading_calibration/   # Installable package
│   ├── datasets/          # Dataset adapters (RiceChem, SciEntsBank, Beetle)
│   ├── inference/         # LLM invocation (vLLM / HF / OpenAI) + prompts/
│   ├── confidence/        # Confidence estimation methods
│   ├── calibration/       # Post-hoc calibration methods
│   ├── metrics/           # Calibration metrics and evaluation
│   ├── statistics/        # Statistical utilities (FDR, bootstrap, mixed models)
│   ├── pipelines/         # Analysis pipelines + publication tables
│   ├── features/          # Feature extraction
│   └── domain/, config/   # Data models and configuration
├── experiments/run_exp/   # Step scripts: step1 (inference) … step6 (analysis)
├── configs/               # Per-step default configs (YAML)
├── utils/                 # Shared utilities (parsing, statistics, caching)
├── analysis/              # Feature / commonality analysis helpers
├── tests/                 # Unit tests
└── scripts/               # Helper scripts
```

---

## Installation

Tested on Python 3.10.

```bash
conda create -n ricechem python=3.10 -y
conda activate ricechem

pip install -r requirements.txt
pip install -e .          # installs the grading_calibration package
```

**Optional — R for mixed-model LRTs / ART-C.** Some analyses call R (`lme4`,
`ARTool`) via `rpy2`. The lockfile is `renv.lock` at the repository root.

```bash
# Conda (recommended on WSL)
conda install -c conda-forge r-base=4.3 r-essentials
sudo apt-get install -y cmake
R -q -e 'install.packages("renv", repos="https://cloud.r-project.org"); renv::restore()'
```

---

## Quick start

The core loop: estimate confidence, then evaluate how well it predicts
correctness.

```python
from grading_calibration.datasets import load_ricechem
from grading_calibration.inference.inference_llm import BatchedLLMInference
from grading_calibration.confidence import SelfReportedConfidence
from grading_calibration.metrics import evaluate_predictions

# 1. Load data
splits = load_ricechem(data_dir="/path/to/RiceChem_Dataset",
                       train_size=0.6, valid_size=0.2, test_size=0.2, seed=42)

# 2. Estimate confidence
engine = BatchedLLMInference("Qwen/Qwen3-4B-Instruct-2507", backend="vllm", temperature=0.0)
estimator = SelfReportedConfidence(engine)
predictions = estimator.estimate_batch(splits.test)

# 3. Evaluate the confidence scores (calibration + ranking + selective prediction)
metrics = evaluate_predictions(predictions)
print(f"ECE={metrics['ece']:.4f}  Brier={metrics['brier_score']:.4f}  "
      f"AUC={metrics['auc_roc']:.4f}  cov@5%risk={metrics['cov_at_5pct_risk']:.3f}")
```

Swap `SelfReportedConfidence` for `TokenProbabilityConfidence`,
`SelfConsistencyConfidence`, `PerturbationConfidence`, or `HybridConfidence` —
the rest of the loop is unchanged.

### Optional: post-hoc calibration

Raw confidence scores can additionally be mapped to calibrated probabilities
(isotonic / temperature / Platt), fit on the validation split. This is optional
and not required for the paper's primary results, which are computed on raw
confidence:

```python
from grading_calibration.calibration import IsotonicCalibrator
from grading_calibration.metrics import check_calibratability

val_predictions = estimator.estimate_batch(splits.valid)

# Only calibrate a proxy that actually carries signal
diag = check_calibratability(val_predictions)
print("Calibratable:", diag["is_calibratable"], "| AUC-ROC:", round(diag["auc_roc"], 3))

calibrator = IsotonicCalibrator()
calibrator.fit(val_predictions)
calibrated = calibrator.transform(predictions)
metrics_cal = evaluate_predictions(calibrated, use_calibrated=True)
```

`check_calibratability()` guards against calibrating a meaningless proxy:
calibration cannot create information that does not exist, so if correlation /
AUC-ROC / monotonicity fail, the fix is a better confidence method, not a
calibrator.

---

## Reproducing the experiments

The paper is reproduced by the numbered step scripts in
`experiments/run_exp/`. Each is a standalone CLI; run them in order. Datasets are
**not bundled** — point `--data_dir` at your local copy.

```bash
cd experiments/run_exp

# Step 1 — inference (expensive; one model × method × dataset per call)
python step1_run_inference.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --method self_reported \
    --backend vllm \
    --data_dir /path/to/RiceChem_Dataset \
    --output predictions.csv

# Step 2 — calibration metrics (ECE, Brier, AUC, coverage)
python step2_analyze_calibration.py --model qwen4b --method self_reported

# Step 3 — feature analysis (correlations, calibration regression)
python step3_feature_analysis.py --model qwen4b --method self_reported

# Step 4 — stratified analysis (by rubric / question / feature bins)
python step4_stratified_analysis.py --model qwen4b --method self_reported

# Step 5 — commonality analysis (high vs low confidence, correct vs incorrect)
python step5_commonality_analysis.py --model qwen4b --method self_reported

# Step 6 — cross-model / cross-method comparison + publication tables
python step6_cross_comparison.py
```

To reproduce the full grid, sweep the models in the table below across the three
methods (`self_reported`, `self_consistency`, `token_probability`). Large models
use multi-GPU tensor parallelism; small models run on a single GPU. Per-step
defaults live in `configs/*.yaml`.

### Models and methods

| Short name | Full model path |
|---|---|
| `llama8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `llama70b` | `meta-llama/Llama-3.1-70B-Instruct` |
| `qwen4b` | `Qwen/Qwen3-4B-Instruct-2507` |
| `qwen80b` | `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| `gptoss20b` | `openai/gpt-oss-20b` |
| `gptoss120b` | `openai/gpt-oss-120b` |

| Method | Cost | Idea |
|---|---|---|
| `token_probability` | 1 call | P(yes) vs P(no) over verdict tokens |
| `self_reported` | 1 call | Ask the model for a confidence score |
| `self_consistency` | 10–30 calls | Agreement across temperature samples |

`step1_run_inference.py` writes predictions to
`experiments/predictions_csvs/<method>/`, named by model. Each CSV holds the
predictions and confidence scores consumed by steps 2–6.

---

## Design principles

- **Modularity** — each module has one responsibility and communicates only
  through the standard data format.
- **Extensibility** — add a dataset, confidence method, or calibrator by
  subclassing the relevant base class (`BaseDatasetAdapter`,
  `BaseConfidenceEstimator`, `BaseCalibrator`).
- **Verification before calibration** — check that a confidence proxy carries
  signal before trusting a calibrator.
- **Reproducibility** — fixed seeds, per-step config files, deterministic
  instance IDs.

---

## Citation

```bibtex
@misc{ferrer2026trustllmgraderscalibrating,
      title={When Can We Trust LLM Graders? Calibrating Confidence for Automated Assessment},
      author={Robinson Ferrer and Damla Turgut and Zhongzhou Chen and Shashank Sonkar},
      year={2026},
      eprint={2603.29559},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.29559},
}
```
