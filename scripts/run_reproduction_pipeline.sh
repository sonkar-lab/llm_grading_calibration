#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_reproduction_pipeline.sh [--results-root PATH] [--output-dir PATH]

Regenerate manuscript-facing artifacts that have generators today.
Defaults:
  --results-root experiments/RESULTS_CANONICAL
  --output-dir   generated_tables/phase5a

Set PYTHON to override the Python command. If unset, the runner uses `python`
when it has required packages, otherwise falls back to `conda run -n alag python`.
Example:
  PYTHON="conda run -n alag python" scripts/run_reproduction_pipeline.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

RESULTS_ROOT="experiments/RESULTS_CANONICAL"
OUTPUT_DIR="generated_tables/phase5a"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root)
      [[ $# -ge 2 ]] || { echo "error: --results-root requires a path" >&2; exit 2; }
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "error: --output-dir requires a path" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

resolve_path() {
  realpath -m "$1"
}

is_equal_or_nested() {
  local child="$1"
  local parent="$2"
  [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

RESULTS_ROOT_ABS="$(resolve_path "${RESULTS_ROOT}")"
OUTPUT_DIR_ABS="$(resolve_path "${OUTPUT_DIR}")"
EXPERIMENTS_ABS="$(resolve_path "experiments")"

FORBIDDEN_DIRS=(
  "${RESULTS_ROOT_ABS}"
  "${EXPERIMENTS_ABS}/RESULTS"
  "${EXPERIMENTS_ABS}/RESULTS_CANONICAL"
  "$(resolve_path "overleaf_paperv3/tables")"
)
while IFS= read -r protected_dir; do
  FORBIDDEN_DIRS+=("$(resolve_path "${protected_dir}")")
done < <(find "${EXPERIMENTS_ABS}" -maxdepth 1 -type d -name 'RESULTS_*' 2>/dev/null | sort)

for protected_dir in "${FORBIDDEN_DIRS[@]}"; do
  if is_equal_or_nested "${OUTPUT_DIR_ABS}" "${protected_dir}"; then
    echo "error: refusing to write output under protected directory: ${protected_dir}" >&2
    echo "       requested output-dir: ${OUTPUT_DIR_ABS}" >&2
    exit 1
  fi
done

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_COMMAND="${PYTHON}"
elif python -c "import pandas" >/dev/null 2>&1; then
  PYTHON_COMMAND="python"
elif command -v conda >/dev/null 2>&1; then
  PYTHON_COMMAND="conda run -n alag python"
else
  PYTHON_COMMAND="python"
fi
read -r -a PYTHON_CMD <<< "${PYTHON_COMMAND}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Results root: %s\n' "${RESULTS_ROOT_ABS}"
printf 'Output dir: %s\n' "${OUTPUT_DIR_ABS}"
printf 'Python: %s\n' "${PYTHON_COMMAND}"
printf '\nSteps:\n'
printf '  1. publication tables: grading_calibration.pipelines.publication\n\n'

"${PYTHON_CMD[@]}" -m grading_calibration.pipelines.publication \
  --results-root "${RESULTS_ROOT_ABS}" \
  --output-dir "${OUTPUT_DIR_ABS}"

EXPECTED_ARTIFACTS=(
  "calibration_regression_interactions_within.csv"
  "calibration_regression_interactions_within.tex"
  "calibration_regression_interactions_across.csv"
  "calibration_regression_interactions_across.tex"
  "calibration_regression_interactions_mixed_diagnostic.csv"
  "calibration_regression_interactions_mixed_diagnostic.tex"
  "feature_family_significance_rates.csv"
  "feature_family_significance_rates.tex"
)

printf '\nGenerated artifacts:\n'
for artifact in "${EXPECTED_ARTIFACTS[@]}"; do
  artifact_path="${OUTPUT_DIR_ABS}/${artifact}"
  if [[ ! -f "${artifact_path}" ]]; then
    echo "error: expected artifact was not generated: ${artifact_path}" >&2
    exit 1
  fi
  printf '  %s\n' "${artifact_path}"
done
