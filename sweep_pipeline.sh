#!/usr/bin/env bash
# Orchestrates the remaining sweeps AFTER the NIPS26 CIFAR job (PID below)
# finishes, so we never contend for GPU memory or kill someone else's run.
#
# Order:
#   1. Wait for the blocking GPU job to exit.
#   2. Label Smoothing sweep:  alpha = 0.01 0.03 0.05 0.07 0.15 0.20  (0.10 already done)
#   3. Maximum Entropy extend:  lambda = 0.15 0.20                    (0.01-0.10 already done)
# Each training run is immediately followed by inference + automatic WER.

cd /home/cvlab/Desktop/SR/courses/2026_spring/project

# Activate conda BEFORE any strict-mode flags — conda's cuda-nvcc activation
# script references unbound vars (NVCC_PREPEND_FLAGS) and dies under `set -u`.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sr

BLOCK_PID=201361
LOG=/tmp/sweep_pipeline.log

echo "[$(date '+%F %T')] waiting for blocking GPU job PID=$BLOCK_PID (NIPS26 CIFAR) to finish..." | tee -a "$LOG"
while kill -0 "$BLOCK_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%F %T')] blocking job finished. starting sweeps." | tee -a "$LOG"

run_one () {
    # $1 = script, $2 = ENV_NAME, $3 = value, $4 = model_dir basename
    local script="$1" envname="$2" val="$3" mdir="$4"
    echo "[$(date '+%F %T')] TRAIN $envname=$val -> models/$mdir" | tee -a "$LOG"
    env "$envname=$val" python "./run/$script" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] INFER models/$mdir" | tee -a "$LOG"
    MODEL_DIR="./run/models/$mdir" python ./run/wav2vec_inference.py >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] DONE  $envname=$val" | tee -a "$LOG"
}

# --- Label Smoothing sweep ---
run_one wav2vec_finetuning_label_smoothing.py LABEL_SMOOTHING 0.01 wav2vec2_label_smoothing_0p01
run_one wav2vec_finetuning_label_smoothing.py LABEL_SMOOTHING 0.03 wav2vec2_label_smoothing_0p03
run_one wav2vec_finetuning_label_smoothing.py LABEL_SMOOTHING 0.05 wav2vec2_label_smoothing_0p05
run_one wav2vec_finetuning_label_smoothing.py LABEL_SMOOTHING 0.07 wav2vec2_label_smoothing_0p07
run_one wav2vec_finetuning_label_smoothing.py LABEL_SMOOTHING 0.15 wav2vec2_label_smoothing_0p15
run_one wav2vec_finetuning_label_smoothing.py LABEL_SMOOTHING 0.2  wav2vec2_label_smoothing_0p2

# --- Maximum Entropy extend ---
run_one wav2vec_finetuning.py ENTROPY_WEIGHT 0.15 wav2vec2_maxent_0p15
run_one wav2vec_finetuning.py ENTROPY_WEIGHT 0.2  wav2vec2_maxent_0p2

echo "[$(date '+%F %T')] ALL SWEEPS COMPLETE" | tee -a "$LOG"
