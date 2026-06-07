#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "sr" ]]; then
  echo "[info] Activating conda environment: sr"
  # shellcheck disable=SC1091
  source /home/cvlab/anaconda3/etc/profile.d/conda.sh
  conda activate sr
fi

echo "[info] Working directory: ${PROJECT_DIR}"
python -c "import torch; print('[info] torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"

MAX_STEPS="${MAX_STEPS:-2000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-24}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# Space-separated list. Override like:
# ENTROPY_WEIGHTS="0.01 0.03 0.05 0.07 0.1" bash ./run/run_experiments.sh
ENTROPY_WEIGHTS="${ENTROPY_WEIGHTS:-0.01 0.03 0.05 0.07 0.1}"

run_inference() {
  local model_dir="$1"
  echo "[info] Running inference and automatic WER for: ${model_dir}"
  MODEL_DIR="${model_dir}" python ./run/wav2vec_inference.py
}

train_baseline() {
  local model_dir="${SCRIPT_DIR}/models/wav2vec2_baseline"
  if [[ -f "${model_dir}/model.safetensors" && "${FORCE_RETRAIN_BASELINE:-0}" != "1" ]]; then
    echo "[info] Baseline model already exists, skipping training: ${model_dir}"
  else
    echo "[info] Training baseline model: ${model_dir}"
    MAX_STEPS="${MAX_STEPS}" \
    EVAL_STEPS="${EVAL_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    OUTPUT_DIR="${model_dir}" \
    python ./run/wav2vec_finetuning_without_ME.py
  fi
  run_inference "${model_dir}"
}

train_maxent() {
  local entropy_weight="$1"
  local entropy_tag="${entropy_weight/./p}"
  local model_dir="${SCRIPT_DIR}/models/wav2vec2_maxent_${entropy_tag}"

  if [[ -f "${model_dir}/model.safetensors" && "${FORCE_RETRAIN_MAXENT:-0}" != "1" ]]; then
    echo "[info] MaxEnt model already exists, skipping training: ${model_dir}"
  else
    echo "[info] Training MaxEnt model: ENTROPY_WEIGHT=${entropy_weight}, output=${model_dir}"
    MAX_STEPS="${MAX_STEPS}" \
    EVAL_STEPS="${EVAL_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    ENTROPY_WEIGHT="${entropy_weight}" \
    OUTPUT_DIR="${model_dir}" \
    python ./run/wav2vec_finetuning.py
  fi
  run_inference "${model_dir}"
}

echo "[info] Experiment settings:"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "  EVAL_STEPS=${EVAL_STEPS}"
echo "  SAVE_STEPS=${SAVE_STEPS}"
echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "  EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "  GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "  LEARNING_RATE=${LEARNING_RATE}"
echo "  ENTROPY_WEIGHTS=${ENTROPY_WEIGHTS}"

train_baseline

for entropy_weight in ${ENTROPY_WEIGHTS}; do
  train_maxent "${entropy_weight}"
done

echo "[info] All experiments finished."
echo "[info] Results are saved under: ${SCRIPT_DIR}/results"
