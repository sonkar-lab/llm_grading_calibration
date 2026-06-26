# Project Overview

This document explains the research framing, the mathematical formulation, the
methods implemented, and the architecture behind them. For installation and
commands, see `README.md`.

## Contents
1. [Motivation](#motivation)
2. [Research questions](#research-questions)
3. [Formalization](#formalization)
4. [Confidence methods](#confidence-methods)
5. [Calibration](#calibration)
6. [Architecture](#architecture)
7. [Design decisions](#design-decisions)
8. [References](#references)

---

## Motivation

LLMs can grade short student answers, but they make mistakes, and the cost of a
silent error in an educational setting is high. Rather than only improving
grading accuracy, this project targets a complementary problem:

> Can we predict *when* an LLM grader is likely to be correct?

A reliable, calibrated confidence estimate enables **selective automation**:
grade the high-confidence cases automatically and route uncertain cases to a
human. Knowing when to trust the model is as important as the grade itself —
it lets a deployment keep quality high while reducing human workload and avoiding
automation bias.

---

## Research questions

1. **Estimation** — which methods produce confidence scores that actually
   correlate with correctness?
2. **Calibration** — can raw scores be transformed into meaningful
   probabilities?
3. **Tradeoffs** — what is the cost/quality tradeoff across methods?

Different methods are expected to capture different uncertainty: token
probabilities are cheap but limited; self-consistency captures *aleatoric*
uncertainty (inherent ambiguity); perturbation stability captures *epistemic*
uncertainty (model knowledge gaps); hybrids combine signals.

---

## Formalization

The grading task is binary:

```
T: (Q, S, I, K) → {0, 1}
```

where **Q** is the question, **S** the student answer, **I** a rubric item, **K**
the answer key, and the output is 1 if the answer satisfies the rubric item.

Given an LLM grader `M` that approximates `T`, we want a confidence function

```
C: (Q, S, I, K) → [0, 1]    estimating    P(M(x) = T(x) | x)
```

with three desirable properties:

- **Calibration** — `E[1{M(x) = T(x)} | C(x) = p] ≈ p`. When the model reports
  80%, it is correct about 80% of the time.
- **Ranking** — higher confidence implies higher accuracy.
- **Selective-prediction utility** — at a coverage threshold, the predictor
  maximizes coverage while holding a target accuracy.

---

## Confidence methods

Five methods, two baselines and three proposed, all returning a
`PredictionResult` with `raw_confidence ∈ [0, 1]`.

**Token probability** (baseline) — use the model's output distribution directly:
`C = max(P(yes), P(no))` over the verdict tokens. One call; fast but typically
overconfident. `grading_calibration.confidence.TokenProbabilityConfidence`.

**Self-reported** (baseline) — ask the model to grade and report a 1–10
confidence, normalized to `[0, 1]`. One call; simple but often poorly calibrated.
`grading_calibration.confidence.SelfReportedConfidence`.

**Self-consistency** — sample `n` outputs at temperature `τ > 0`; confidence is
the majority proportion. Captures aleatoric uncertainty; 10–30 calls.
`grading_calibration.confidence.SelfConsistencyConfidence`.

**Perturbation stability** — apply ground-truth-preserving perturbations
(paraphrase rubric, synonym replacement, question rephrase, identity control) and
measure how often the verdict is stable. Captures epistemic uncertainty; 5–10
calls. `grading_calibration.confidence.PerturbationConfidence`.

**Hybrid** — combine signals from multiple methods (fixed or learned weights).
Most expensive; requires its component methods.
`grading_calibration.confidence.HybridConfidence`.

---

## Calibration

Raw scores are **not** probabilities. A self-consistency score of 0.8 means "8 of
10 samples agreed", not "80% likely correct" — the true accuracy could be higher
(underconfident) or lower (overconfident). Post-hoc calibration learns the
mapping from raw score to empirical accuracy, fit on the validation split and
applied to test.

- **Isotonic regression** (recommended) — monotonic, flexible, handles non-linear
  miscalibration; needs sufficient validation data.
  `grading_calibration.calibration.IsotonicCalibrator`.
- **Temperature scaling** — one parameter rescaling logits; simple, less flexible.
  `grading_calibration.calibration.TemperatureScalingCalibrator`.
- **Platt scaling** — logistic regression on confidence; between the two in
  flexibility. `grading_calibration.calibration.PlattScalingCalibrator`.

**Pre-calibration verification.** Calibration cannot create information that does
not exist. Before calibrating, `check_calibratability()` checks correlation,
discrimination (AUC-ROC), monotonicity, and resolution. If a proxy fails — e.g.
AUC-ROC ≈ 0.5 — calibration will only overfit noise, and the right fix is a
better confidence method. `grading_calibration.metrics.check_calibratability`.

---

## Architecture

Data flows through two dataclasses, so every stage is interchangeable:

```
GradingInstance ──▶ confidence method ──▶ PredictionResult ──▶ calibrator ──▶ metrics
   (input)                                    (output)
```

| Module | Role |
|---|---|
| `grading_calibration.domain.models` | `GradingInstance`, `PredictionResult`, `DatasetSplit` |
| `grading_calibration.datasets` | `BaseDatasetAdapter` + RiceChem / SciEntsBank / Beetle loaders |
| `grading_calibration.inference` | Batched generation, token probs, self-consistency, perturbation, prompts |
| `grading_calibration.confidence` | The five confidence estimators (`BaseConfidenceEstimator`) |
| `grading_calibration.calibration` | The three calibrators (`BaseCalibrator`) |
| `grading_calibration.metrics` | ECE, Brier, AUC-ROC, risk-coverage, calibratability |
| `grading_calibration.statistics` | FDR correction, bootstrap CIs, mixed models |
| `grading_calibration.pipelines` | Feature / stratified / commonality / regression / cross-comparison analyses |

The numbered scripts in `experiments/run_exp/` (step1 … step6) drive this
pipeline end to end and produce the publication tables.

### Metrics

- **Calibration** — ECE (mean |confidence − accuracy| over bins), Brier score.
- **Ranking** — AUC-ROC, Spearman correlation with correctness.
- **Selective prediction** — risk-coverage curves, accuracy@K%, coverage@risk.
- **Diagnostics** — pre-calibration verification (correlation, monotonicity, AUC).

---

## Design decisions

- **Standard data format** — all stages consume/emit `GradingInstance` /
  `PredictionResult`, eliminating conversion bugs and making components swappable
  (any confidence method works with any calibrator and the shared evaluation
  code).
- **Abstract base classes** — `BaseDatasetAdapter`, `BaseConfidenceEstimator`,
  `BaseCalibrator` enforce consistent interfaces and define the extension
  contract.
- **Verification before calibration** — a deliberate gate so a meaningless proxy
  cannot masquerade as a calibrated probability.
- **Prompts separated from methods** — prompts are research variables in their
  own module, so prompt ablations change one place.
- **Deterministic IDs and seeds** — instance IDs are semantic
  (`Q{q}_S{student}_{rubric}`) and order-independent, supporting reproducible
  splits and checkpoint resumption.

### Extending

Add a **dataset** by subclassing `BaseDatasetAdapter` (`load_raw_data`,
`to_grading_instances`, `get_dataset_name`). Add a **confidence method** by
subclassing `BaseConfidenceEstimator` (`estimate`, `get_name`) and returning a
`PredictionResult`. Add a **calibrator** by subclassing `BaseCalibrator` (`fit`,
`transform`). Each integrates with the existing pipeline without further changes.

---

## References

- Wang et al., 2022. *Self-Consistency Improves Chain of Thought Reasoning in
  Language Models.*
- Guo et al., 2017. *On Calibration of Modern Neural Networks.*
- Geifman & El-Yaniv, 2017. *Selective Prediction with Neural Networks.*
- Sonkar et al., 2024. *Automated Long Answer Grading with the RiceChem Dataset.*
