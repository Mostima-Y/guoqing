#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_v23_v36_protein_bootstrap.sh [options]

Analyze existing V23/V36 seed-42/43 test predictions at the validation-frozen
inference epsilon 0.0125. This launcher does not run model inference or use a GPU.

Options:
  --dry-run                    Audit inputs and predictions without writing output.
  --root PATH                  Project root.
  --python-bin PATH            Python interpreter used by the project.
  --evaluation-root PATH       Existing epsilon re-evaluation output root.
  --test-csv PATH              Locked protein-cold test CSV.
  --output-dir PATH            New analysis output directory.
  --reference-metrics PATH     Previous result used for macro regression checks.
  --bootstrap-replicates N     Protein bootstrap replicates (default: 2000).
  --bootstrap-seed N           Bootstrap RNG seed (default: 20260926).
  --protein-column NAME        Protein identity CSV column (default: Protein).
  -h, --help                   Show this help.
EOF
}

dry_run=0
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
root="${GLOBIS_ROOT:-/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev}"
python_bin="${GLOBIS_PYTHON:-/mnt/home/dachuang/conda_envs/DTIAM/bin/python}"
evaluation_root=""
test_csv="${GLOBIS_TEST_CSV:-/mnt/home/dachuang/dti_in_domain_test/splits/protein_cold_start/test.csv}"
output_dir=""
reference_metrics=""
bootstrap_replicates=2000
bootstrap_seed=20260926
protein_column=Protein

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --root) root="$2"; shift 2 ;;
    --python-bin) python_bin="$2"; shift 2 ;;
    --evaluation-root) evaluation_root="$2"; shift 2 ;;
    --test-csv) test_csv="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --reference-metrics) reference_metrics="$2"; shift 2 ;;
    --bootstrap-replicates) bootstrap_replicates="$2"; shift 2 ;;
    --bootstrap-seed) bootstrap_seed="$2"; shift 2 ;;
    --protein-column) protein_column="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

evaluation_root="${evaluation_root:-${root}/external_benchmark/test_evaluations/epsilon_reval_20260926}"
output_dir="${output_dir:-${root}/external_benchmark/audits/protein_bootstrap_20260927_fixed}"
reference_metrics="${reference_metrics:-${root}/external_benchmark/audits/protein_bootstrap_20260926/metrics.json}"
expected_metrics="${repo_root}/results/epsilon_reval_20260926/metrics.json"
analysis_script="${script_dir}/analyze_v23_v36_protein_bootstrap.py"
project_pythonpath="${root}/code"
if [[ -n "${PYTHONPATH:-}" ]]; then
  project_pythonpath="${project_pythonpath}:${PYTHONPATH}"
fi

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "required file missing: $1"
}

[[ -x "${python_bin}" ]] || fail "Python interpreter missing or not executable: ${python_bin}"
require_file "${analysis_script}"
require_file "${expected_metrics}"
require_file "${test_csv}"
require_file "${reference_metrics}"

for model in v23 v36; do
  for seed in 42 43; do
    require_file "${evaluation_root}/${model}/seed${seed}/epsilon_0p0125/completed.json"
  done
done

if [[ "${dry_run}" -eq 0 && -e "${output_dir}" ]]; then
  fail "refusing to overwrite existing output: ${output_dir}"
fi

command=(
  env "PYTHONPATH=${project_pythonpath}" "${python_bin}" -u "${analysis_script}"
  --evaluation-root "${evaluation_root}"
  --expected-metrics "${expected_metrics}"
  --test-csv "${test_csv}"
  --protein-column "${protein_column}"
  --output-dir "${output_dir}"
  --reference-metrics "${reference_metrics}"
  --bootstrap-replicates "${bootstrap_replicates}"
  --bootstrap-seed "${bootstrap_seed}"
)

if [[ "${dry_run}" -eq 1 ]]; then
  command+=(--dry-run)
  echo "DRY RUN: auditing locked CSV, completed metadata, checkpoints, and score arrays."
fi

cd "${root}"
"${command[@]}"
