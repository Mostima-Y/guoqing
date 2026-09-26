#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_v23_v36_epsilon_reval.sh [options]

Evaluation-only V23/V36 comparison for seeds 42/43 and inference epsilons
0.025/0.0125. Existing selected checkpoints are never modified.

Options:
  --dry-run                 Validate every input and print commands only.
  --root PATH               Project root.
  --python-bin PATH         Python interpreter used by the project.
  --output-root PATH        New evaluation output directory.
  --gpu42 DEVICE            CUDA_VISIBLE_DEVICES selector for seed 42.
  --gpu43 DEVICE            CUDA_VISIBLE_DEVICES selector for seed 43.
  --v23-checkpoint-42 PATH  Override the documented V23 seed-42 checkpoint.
  --v23-checkpoint-43 PATH  Override the documented V23 seed-43 checkpoint.
  --v36-checkpoint-42 PATH  Override the documented V36 seed-42 checkpoint.
  --v36-checkpoint-43 PATH  Override the documented V36 seed-43 checkpoint.
  -h, --help                Show this help.

Equivalent environment overrides: GLOBIS_ROOT, GLOBIS_PYTHON, OUTPUT_ROOT,
GPU42, GPU43, V23_CHECKPOINT_42, V23_CHECKPOINT_43, V36_CHECKPOINT_42,
V36_CHECKPOINT_43.
EOF
}

dry_run=0
root="${GLOBIS_ROOT:-/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev}"
python_bin="${GLOBIS_PYTHON:-/mnt/home/dachuang/conda_envs/DTIAM/bin/python}"
output_root_override="${OUTPUT_ROOT:-}"
gpu42="${GPU42:-0}"
gpu43="${GPU43:-1}"
v23_checkpoint_42="${V23_CHECKPOINT_42:-}"
v23_checkpoint_43="${V23_CHECKPOINT_43:-}"
v36_checkpoint_42="${V36_CHECKPOINT_42:-}"
v36_checkpoint_43="${V36_CHECKPOINT_43:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --root) root="$2"; shift 2 ;;
    --python-bin) python_bin="$2"; shift 2 ;;
    --output-root) output_root_override="$2"; shift 2 ;;
    --gpu42) gpu42="$2"; shift 2 ;;
    --gpu43) gpu43="$2"; shift 2 ;;
    --v23-checkpoint-42) v23_checkpoint_42="$2"; shift 2 ;;
    --v23-checkpoint-43) v23_checkpoint_43="$2"; shift 2 ;;
    --v36-checkpoint-42) v36_checkpoint_42="$2"; shift 2 ;;
    --v36-checkpoint-43) v36_checkpoint_43="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

output_root="${output_root_override:-${root}/external_benchmark/test_evaluations/epsilon_reval_20260926}"
indices="${root}/external_benchmark/screen_subsets/full_protein_cold_train_indices.npy"
segments="${root}/external_benchmark/protein_segments/full_protein_cold_esm2_t33_s16_v1"
test_segments="${root}/external_benchmark/protein_segments/protein_cold_test_esm2_t33_s16_v1"
tokens="${root}/external_benchmark/drug_substructures/full_protein_cold_bermol_r1_t96_v1"
base_root="${root}/external_benchmark/ablations/full_ln_confirmation_v1"
v23_root="${root}/external_benchmark/ablations/full_energy_normalized_v23_exploratory"
v36_root="${root}/external_benchmark/ablations/full_two_seed_mutual_qk_norm_v36"
test_csv="/mnt/home/dachuang/dti_in_domain_test/splits/protein_cold_start/test.csv"
bermol_model="${root}/models/BerMolModel_base.pkl"

v23_checkpoint_42="${v23_checkpoint_42:-${v23_root}/protein_cold_energy_normalized_v23_seed42_e64/members/member_00/best.pt}"
v23_checkpoint_43="${v23_checkpoint_43:-${v23_root}/protein_cold_energy_normalized_v23_seed43_e64/members/member_00/best.pt}"
v36_checkpoint_42="${v36_checkpoint_42:-${v36_root}/protein_cold_mutual_qk_norm_v36_seed42_e64/members/member_00/best.pt}"
v36_checkpoint_43="${v36_checkpoint_43:-${v36_root}/protein_cold_mutual_qk_norm_v36_seed43_e64/members/member_00/best.pt}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "required file missing: $1"
}

require_directory() {
  [[ -d "$1" ]] || fail "required directory missing: $1"
}

candidate_checkpoint() {
  case "$1:$2" in
    v23:42) printf '%s' "${v23_checkpoint_42}" ;;
    v23:43) printf '%s' "${v23_checkpoint_43}" ;;
    v36:42) printf '%s' "${v36_checkpoint_42}" ;;
    v36:43) printf '%s' "${v36_checkpoint_43}" ;;
    *) fail "unsupported model/seed: $1/$2" ;;
  esac
}

model_variant() {
  case "$1" in
    v23) printf '%s' "energy_v23" ;;
    v36) printf '%s' "mutual_qk_norm_v36" ;;
    *) fail "unsupported model: $1" ;;
  esac
}

epsilon_directory() {
  printf 'epsilon_%s' "${1/./p}"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

[[ ! -e "${output_root}" ]] || fail "output root already exists; refusing to overwrite: ${output_root}"
[[ -x "${python_bin}" ]] || fail "Python interpreter missing or not executable: ${python_bin}"
require_file "${root}/code/evaluate_atom_segment_test.py"
require_file "${root}/code/summarize_v23_v36_epsilon_reval.py"
require_file "${indices}"
require_file "${test_csv}"
require_file "${bermol_model}"
require_directory "${segments}"
require_directory "${test_segments}"
require_directory "${tokens}"
require_file "${segments}/metadata.json"
require_file "${test_segments}/metadata.json"
require_file "${tokens}/metadata.json"

for seed in 42 43; do
  base="${base_root}/protein_cold_bilinear_control_bn_ln_seed${seed}_e60/members/member_00"
  require_file "${base}/best.pt"
  require_file "${base}/protein_cold_test_scores.npy"
  require_file "$(candidate_checkpoint v23 "${seed}")"
  require_file "$(candidate_checkpoint v36 "${seed}")"
done

run_evaluation() {
  local model="$1" seed="$2" epsilon="$3" gpu="$4"
  local variant checkpoint base output log_file
  variant="$(model_variant "${model}")"
  checkpoint="$(candidate_checkpoint "${model}" "${seed}")"
  base="${base_root}/protein_cold_bilinear_control_bn_ln_seed${seed}_e60/members/member_00"
  output="${output_root}/${model}/seed${seed}/$(epsilon_directory "${epsilon}")"
  log_file="${output_root}/logs/${model}_seed${seed}_epsilon_${epsilon}.log"
  local command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${python_bin}" -u
    code/evaluate_atom_segment_test.py
    --scenario protein_cold
    --test-split protein_cold_test
    --test-csv "${test_csv}"
    --test-protein-segments-dir "${test_segments}"
    --protein-segments-dir "${segments}"
    --drug-tokens-dir "${tokens}"
    --train-indices "${indices}"
    --base-checkpoint "${base}/best.pt"
    --base-test-scores "${base}/protein_cold_test_scores.npy"
    --candidate-checkpoint "${checkpoint}"
    --bermol-model "${bermol_model}"
    --output-dir "${output}"
    --seed "${seed}"
    --trained-epsilon 0.025
    --inference-epsilon "${epsilon}"
    --residual-energy-weight 0.005
    --interaction-dim 64
    --model-variant "${variant}"
    --injection-mode feature
    --batch-size 512
    --device cuda:0
    --amp
    --preload-test
  )
  if [[ "${dry_run}" -eq 1 ]]; then
    print_command "${command[@]}"
    printf '    > %q 2>&1\n' "${log_file}"
  else
    "${command[@]}" >"${log_file}" 2>&1
  fi
}

run_seed() {
  local seed="$1" gpu="$2" model epsilon
  for model in v23 v36; do
    for epsilon in 0.025 0.0125; do
      run_evaluation "${model}" "${seed}" "${epsilon}" "${gpu}"
    done
  done
}

summary_command=(
  "${python_bin}" -u code/summarize_v23_v36_epsilon_reval.py
  --root "${output_root}"
  --test-split protein_cold_test
)

cd "${root}"
if [[ "${dry_run}" -eq 1 ]]; then
  echo "DRY RUN: inputs validated; no output directory was created."
  echo "Seed 42 commands (CUDA_VISIBLE_DEVICES=${gpu42}):"
  run_seed 42 "${gpu42}"
  echo "Seed 43 commands (CUDA_VISIBLE_DEVICES=${gpu43}):"
  run_seed 43 "${gpu43}"
  echo "Summary command (run only after all eight evaluations succeed):"
  print_command "${summary_command[@]}"
  exit 0
fi

mkdir -p "${output_root}/logs"
run_seed 42 "${gpu42}" & worker42="$!"
run_seed 43 "${gpu43}" & worker43="$!"

status=0
wait "${worker42}" || status=1
wait "${worker43}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "V23_V36_EPSILON_REVAL_FAILED; inspect ${output_root}/logs" >&2
  exit "${status}"
fi

"${summary_command[@]}" | tee "${output_root}/summary_stdout.log"
echo "V23_V36_EPSILON_REVAL_COMPLETE"
