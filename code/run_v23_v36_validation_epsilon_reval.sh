#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_v23_v36_validation_epsilon_reval.sh [options]

Evaluation-only validation comparison for V23/V36, seeds 42/43, and inference
epsilons 0.025/0.0125. The test split is never loaded.

Options:
  --dry-run                 Validate inputs and show job decisions/commands only.
  --root PATH               Project root.
  --python-bin PATH         Project Python interpreter.
  --output-root PATH        New validation evaluation output root.
  --gpu42 DEVICE            CUDA_VISIBLE_DEVICES selector for seed 42.
  --gpu43 DEVICE            CUDA_VISIBLE_DEVICES selector for seed 43.
  --seed all|42|43          Seeds to process (default: all).
  --v23-checkpoint-42 PATH  Override V23 seed-42 selected checkpoint.
  --v23-checkpoint-43 PATH  Override V23 seed-43 selected checkpoint.
  --v36-checkpoint-42 PATH  Override V36 seed-42 selected checkpoint.
  --v36-checkpoint-43 PATH  Override V36 seed-43 selected checkpoint.
  -h, --help                Show this help.
EOF
}

dry_run=0
root="${GLOBIS_ROOT:-/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev}"
python_bin="${GLOBIS_PYTHON:-/mnt/home/dachuang/conda_envs/DTIAM/bin/python}"
output_root_override="${OUTPUT_ROOT:-}"
gpu42="${GPU42:-0}"
gpu43="${GPU43:-1}"
selected_seed="all"
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
    --seed) selected_seed="$2"; shift 2 ;;
    --v23-checkpoint-42) v23_checkpoint_42="$2"; shift 2 ;;
    --v23-checkpoint-43) v23_checkpoint_43="$2"; shift 2 ;;
    --v36-checkpoint-42) v36_checkpoint_42="$2"; shift 2 ;;
    --v36-checkpoint-43) v36_checkpoint_43="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${selected_seed}" in
  all) selected_seeds=(42 43) ;;
  42) selected_seeds=(42) ;;
  43) selected_seeds=(43) ;;
  *) echo "Invalid --seed: ${selected_seed}" >&2; exit 2 ;;
esac

output_root="${output_root_override:-${root}/external_benchmark/validation_evaluations/epsilon_reval_20260926}"
indices="${root}/external_benchmark/screen_subsets/full_protein_cold_train_indices.npy"
segments="${root}/external_benchmark/protein_segments/full_protein_cold_esm2_t33_s16_v1"
tokens="${root}/external_benchmark/drug_substructures/full_protein_cold_bermol_r1_t96_v1"
base_root="${root}/external_benchmark/ablations/full_ln_confirmation_v1"
v23_root="${root}/external_benchmark/ablations/full_energy_normalized_v23_exploratory"
v36_root="${root}/external_benchmark/ablations/full_two_seed_mutual_qk_norm_v36"

v23_checkpoint_42="${v23_checkpoint_42:-${v23_root}/protein_cold_energy_normalized_v23_seed42_e64/members/member_00/best.pt}"
v23_checkpoint_43="${v23_checkpoint_43:-${v23_root}/protein_cold_energy_normalized_v23_seed43_e64/members/member_00/best.pt}"
v36_checkpoint_42="${v36_checkpoint_42:-${v36_root}/protein_cold_mutual_qk_norm_v36_seed42_e64/members/member_00/best.pt}"
v36_checkpoint_43="${v36_checkpoint_43:-${v36_root}/protein_cold_mutual_qk_norm_v36_seed43_e64/members/member_00/best.pt}"

fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "required file missing: $1"; }
require_directory() { [[ -d "$1" ]] || fail "required directory missing: $1"; }

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

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

[[ ! -e "${output_root}" || -d "${output_root}" ]] || fail "output root is not a directory: ${output_root}"
[[ -x "${python_bin}" ]] || fail "Python missing or not executable: ${python_bin}"
require_file "${root}/code/evaluate_atom_segment_validation.py"
require_file "${root}/code/summarize_v23_v36_validation_epsilon.py"
require_file "${indices}"
require_directory "${segments}"
require_directory "${tokens}"
require_file "${segments}/metadata.json"
require_file "${tokens}/metadata.json"

for seed in "${selected_seeds[@]}"; do
  base="${base_root}/protein_cold_bilinear_control_bn_ln_seed${seed}_e60/members/member_00/best.pt"
  require_file "${base}"
  require_file "$(candidate_checkpoint v23 "${seed}")"
  require_file "$(candidate_checkpoint v36 "${seed}")"
done

run_evaluation() {
  local model="$1" seed="$2" gpu="$3" variant checkpoint base output log_file
  variant="$(model_variant "${model}")"
  checkpoint="$(candidate_checkpoint "${model}" "${seed}")"
  base="${base_root}/protein_cold_bilinear_control_bn_ln_seed${seed}_e60/members/member_00/best.pt"
  output="${output_root}/${model}/seed${seed}"
  log_file="${output_root}/logs/${model}_seed${seed}.log"
  if [[ -f "${output}/completed.json" ]]; then
    echo "SKIP completed: ${output}"
    return 0
  fi
  if [[ -e "${output}" ]]; then
    echo "INCOMPLETE: ${output}" >&2
    return 1
  fi
  echo "RUN missing: ${output}"
  local command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${python_bin}" -u
    code/evaluate_atom_segment_validation.py
    --scenario protein_cold
    --protein-segments-dir "${segments}"
    --drug-tokens-dir "${tokens}"
    --train-indices "${indices}"
    --base-checkpoint "${base}"
    --candidate-checkpoint "${checkpoint}"
    --output-dir "${output}"
    --seed "${seed}"
    --trained-epsilon 0.025
    --inference-epsilons 0.025 0.0125
    --residual-energy-weight 0.005
    --interaction-dim 64
    --model-variant "${variant}"
    --batch-size 512
    --device cuda:0
    --amp
    --preload-validation
  )
  if [[ "${dry_run}" -eq 1 ]]; then
    print_command "${command[@]}"
    printf '    > %q 2>&1\n' "${log_file}"
  else
    "${command[@]}" >"${log_file}" 2>&1
  fi
}

run_seed() {
  local seed="$1" gpu="$2" model
  for model in v23 v36; do
    run_evaluation "${model}" "${seed}" "${gpu}" || return 1
  done
}

reject_incomplete_selected_jobs() {
  local seed model output
  for seed in "${selected_seeds[@]}"; do
    for model in v23 v36; do
      output="${output_root}/${model}/seed${seed}"
      if [[ -e "${output}" && ! -f "${output}/completed.json" ]]; then
        echo "INCOMPLETE: ${output}" >&2
        return 1
      fi
    done
  done
}

all_evaluations_completed() {
  local seed model
  for seed in 42 43; do
    for model in v23 v36; do
      [[ -f "${output_root}/${model}/seed${seed}/completed.json" ]] || return 1
    done
  done
}

summary_command=(
  "${python_bin}" -u code/summarize_v23_v36_validation_epsilon.py
  --root "${output_root}"
)

cd "${root}"
if [[ "${dry_run}" -eq 1 ]]; then
  echo "DRY RUN: validation inputs checked; no GPU command will run."
  for seed in "${selected_seeds[@]}"; do
    if [[ "${seed}" -eq 42 ]]; then
      run_seed 42 "${gpu42}"
    else
      run_seed 43 "${gpu43}"
    fi
  done
  echo "Summary command (after all four checkpoint evaluations complete):"
  print_command "${summary_command[@]}"
  exit 0
fi

reject_incomplete_selected_jobs
mkdir -p "${output_root}/logs"
status=0
if [[ "${selected_seed}" == "all" ]]; then
  run_seed 42 "${gpu42}" & worker42="$!"
  run_seed 43 "${gpu43}" & worker43="$!"
  wait "${worker42}" || status=1
  wait "${worker43}" || status=1
elif [[ "${selected_seed}" == "42" ]]; then
  run_seed 42 "${gpu42}" || status=1
else
  run_seed 43 "${gpu43}" || status=1
fi
[[ "${status}" -eq 0 ]] || fail "validation epsilon evaluation failed; inspect ${output_root}/logs"

if all_evaluations_completed; then
  if [[ -f "${output_root}/metrics.json" && -f "${output_root}/metrics.tsv" ]]; then
    echo "SKIP completed summary: ${output_root}"
  elif [[ -e "${output_root}/metrics.json" || -e "${output_root}/metrics.tsv" ]]; then
    fail "incomplete summary exists: ${output_root}"
  else
    "${summary_command[@]}" | tee "${output_root}/summary_stdout.log"
  fi
else
  echo "SUMMARY pending: all four checkpoint evaluations are not complete"
fi
echo "V23_V36_VALIDATION_EPSILON_REVAL_COMPLETE"
