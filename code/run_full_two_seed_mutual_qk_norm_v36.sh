#!/usr/bin/env bash
set -euo pipefail

root="/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev"
python_bin="/mnt/home/dachuang/conda_envs/DTIAM/bin/python"
indices="${root}/external_benchmark/screen_subsets/full_protein_cold_train_indices.npy"
segments="${root}/external_benchmark/protein_segments/full_protein_cold_esm2_t33_s16_v1"
test_segments="${root}/external_benchmark/protein_segments/protein_cold_test_esm2_t33_s16_v1"
tokens="${root}/external_benchmark/drug_substructures/full_protein_cold_bermol_r1_t96_v1"
base_root="${root}/external_benchmark/ablations/full_ln_confirmation_v1"
train_root="${root}/external_benchmark/ablations/full_two_seed_mutual_qk_norm_v36"
test_root="${root}/external_benchmark/test_evaluations/full_two_seed_mutual_qk_norm_v36"
queue="${root}/external_benchmark/queues/full_two_seed_mutual_qk_norm_v36"
gpu42="GPU-fbf54061-df7c-a9c8-6373-dd0888818877"
gpu43="GPU-78cf510e-f593-5743-950e-5a89e1d968b5"

cd "${root}"
mkdir -p "${queue}" "${train_root}" "${test_root}"
echo "$$" > "${queue}/supervisor.pid"
export PYTHONPATH=code
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

run_seed() {
  local seed="$1" gpu="$2"
  local run_dir="${train_root}/protein_cold_mutual_qk_norm_v36_seed${seed}_e64"
  local output="${test_root}/seed${seed}"
  local base="${base_root}/protein_cold_bilinear_control_bn_ln_seed${seed}_e60"

  if [[ ! -f "${run_dir}/completed.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
      code/external_benchmark_atom_segment_mutual_qk_norm_v36.py \
      --scenario protein_cold --run-dir "${run_dir}" \
      --protein-segments-dir "${segments}" --drug-tokens-dir "${tokens}" \
      --train-indices "${indices}" --base-checkpoint "${base}/members/member_00/best.pt" \
      --device cuda:0 --epochs 64 --patience 99 --val-every 2 \
      --selection-window 5 --batch-size 256 --predict-batch-size 512 \
      --learning-rate 0.0002 --weight-decay 0.001 --gradient-clip 1.0 \
      --interaction-dim 64 --injection-mode feature --epsilon 0.025 \
      --residual-energy-weight 0.005 --ema-decay 0 --seed "${seed}" \
      --amp --preload-matrices >"${queue}/seed${seed}_train.log" 2>&1 || return 1
  fi

  if [[ ! -f "${output}/completed.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
      code/evaluate_atom_segment_test.py --scenario protein_cold \
      --test-split protein_cold_test \
      --test-csv /mnt/home/dachuang/dti_in_domain_test/splits/protein_cold_start/test.csv \
      --test-protein-segments-dir "${test_segments}" \
      --protein-segments-dir "${segments}" --drug-tokens-dir "${tokens}" \
      --train-indices "${indices}" \
      --base-checkpoint "${base}/members/member_00/best.pt" \
      --base-test-scores "${base}/members/member_00/protein_cold_test_scores.npy" \
      --candidate-checkpoint "${run_dir}/members/member_00/best.pt" \
      --bermol-model "${root}/models/BerMolModel_base.pkl" \
      --output-dir "${output}" --seed "${seed}" \
      --trained-epsilon 0.025 --inference-epsilon 0.0125 \
      --residual-energy-weight 0.005 --interaction-dim 64 \
      --model-variant mutual_qk_norm_v36 --injection-mode feature \
      --batch-size 512 --device cuda:0 --amp --preload-test \
      >"${queue}/seed${seed}_test.log" 2>&1 || return 1
  fi
}

run_seed 42 "${gpu42}" & worker42="$!"
run_seed 43 "${gpu43}" & worker43="$!"

status=0
wait "${worker42}" || status=1
wait "${worker43}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo FULL_TWO_SEED_MUTUAL_QK_NORM_FAILED
  exit "${status}"
fi

"${python_bin}" -u code/ensemble_atom_segment_test_scores.py \
  --test-split protein_cold_test --output-dir "${test_root}/ensemble2" \
  "${test_root}/seed42/candidate_scores_eps00125.npy" \
  "${test_root}/seed43/candidate_scores_eps00125.npy" \
  >"${queue}/ensemble2.log" 2>&1

touch "${queue}/SUCCESS"
echo FULL_TWO_SEED_MUTUAL_QK_NORM_COMPLETE
