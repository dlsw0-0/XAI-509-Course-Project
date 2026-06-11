# Wav2Vec2 CTC Fine-Tuning with Output-Distribution Regularization

**🌐 Language:** **English** | [한국어](README_ko.md)

XAI 509 Course Project — fine-tune `facebook/wav2vec2-base` with the CTC loss on
a 1-hour LibriSpeech subset, then compare two output-distribution regularizers
(Maximum Entropy and Label Smoothing) against the plain CTC baseline.

## ⬇️ Downloads

| Asset | Link |
| :---- | :--- |
| **Pretrained weights** (3 best models, ~1 GB) | [![Weights](https://img.shields.io/badge/Download-Weights%20(GitHub%20Release)-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/dlsw0-0/XAI-509-Course-Project/releases/download/v1.0/pretrained_weights.zip) |
| **Fine-tuning set** (Libri-light 1 h, train) | [![Train](https://img.shields.io/badge/Download-Train%20Set-34A853?logo=googledrive&logoColor=white&style=for-the-badge)](https://drive.google.com/file/d/153mEZhH_PvwbAgqvgrhKCY3BKT4mnu9/view?usp=drive_link) |
| **Test sets** (test-clean + test-other) | [![Test](https://img.shields.io/badge/Download-Test%20Sets-34A853?logo=googledrive&logoColor=white&style=for-the-badge)](https://drive.google.com/file/d/1OT4KazgFBdWIXYGizlNUUdDmmctPY8vN/view?usp=drive_link) |

The weights bundle holds the three reported models (`wav2vec2_baseline`,
`wav2vec2_maxent_0p15`, `wav2vec2_label_smoothing_0p1`), inference-only
(optimizer/checkpoint states stripped). **Unzip it inside `run/`** so the folders
land exactly where inference expects them:

```bash
# from the run/ directory
unzip pretrained_weights.zip          # creates run/models/wav2vec2_*
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1 python ./wav2vec_inference.py
```

Datasets: download the two Drive archives and unzip them under `dataset/`
(see §3 for the expected layout).

## 1. Results Summary

Best operating point of each method (full sweeps in §1.1):

| Method                              | test-clean WER | test-other WER | Δ vs baseline (other) |
| :---------------------------------- | :------------: | :------------: | :-------------------: |
| Baseline (CTC only)                 |     21.16 %    |     30.17 %    |          —            |
| Maximum Entropy (best, λ = 0.15)    |     19.88 %    |     28.63 %    |       −1.54 pp        |
| **Label Smoothing (best, α = 0.10)**|   **19.51 %**  |   **28.30 %**  |     **−1.87 pp**      |

Both regularizers improve over baseline. **Label Smoothing (α = 0.10) is the
overall best on both splits**, edging out the best MaxEnt by 0.37 pp (clean) /
0.33 pp (other). Gains are larger on the harder `test-other` split.

> **Note on tuning:** there is no separate validation set (only train /
> test-clean / test-other are provided, and train has just 286 utterances).
> Following the skeleton, test-clean was used as the eval set and α/λ were
> selected on test WER — so the reported best is an optimistic estimate. The
> project goal is to *compare* over-confidence regularizers, not to claim a
> leak-free WER.

### 1.1 Hyper-parameter Sweeps

**Label Smoothing — α sweep** (sharp optimum at 0.10):

|  α   | test-clean | test-other |
| :--: | :--------: | :--------: |
| 0.00 |   21.16    |   30.17    |
| 0.01 |   20.90    |   30.15    |
| 0.03 |   20.71    |   29.98    |
| 0.05 |   21.43    |   30.09    |
| 0.07 |   20.91    |   29.75    |
| **0.10** | **19.51** | **28.30** |
| 0.15 |   20.16    |   29.39    |
| 0.20 |   20.69    |   29.43    |

**Maximum Entropy — λ sweep** (broad optimum at 0.15–0.20):

|  λ   | test-clean | test-other |
| :--: | :--------: | :--------: |
| 0.00 |   21.16    |   30.17    |
| 0.01 |   20.60    |   29.67    |
| 0.03 |   20.79    |   29.65    |
| 0.05 |   21.21    |   30.46    |
| 0.07 |   20.93    |   29.80    |
| 0.10 |   19.93    |   28.80    |
| **0.15** | **19.88** |   28.63    |
| 0.20 |   20.09    | **28.60**  |

LS has a sharp minimum at α = 0.10 (sensitive to α); MaxEnt plateaus over
λ = 0.15–0.20 (robust, but a lower peak than LS).

## 2. Code Layout

```
run/
├── README.md / README_ko.md             # docs
├── .gitignore
│
├── sample_util.py                        # WebDataset loading + audio/text preprocessing
├── evaluate_wer.py                       # WER computation (CLI + importable)
│
├── wav2vec_finetuning_without_ME.py      # Baseline (CTC only)
├── wav2vec_finetuning.py                 # + Maximum Entropy regularization
├── wav2vec_finetuning_label_smoothing.py # + Label Smoothing regularization
├── wav2vec_inference.py                  # Inference + auto WER
├── run_experiments.sh                    # Sequential baseline + sweep runner
│
└── results/                              # WER text logs (committed)
    ├── wav2vec2_baseline/
    ├── wav2vec2_maxent_0p01 … 0p2/
    └── wav2vec2_label_smoothing_0p01 … 0p2/
```

`dataset/` and `models/` are **not** in git — download them from the buttons
above.

## 3. Expected Data / Model Layout

```
run/
├── dataset/
│   ├── train/        shard-000000.tar … shard-000004.tar   # Libri-light 1 h (286 utt)
│   ├── test-clean/   shard-*.tar                           # 2620 utt
│   └── test-other/   shard-*.tar                           # 2939 utt
└── models/                                            # from pretrained_weights.zip
    ├── wav2vec2_baseline/
    ├── wav2vec2_maxent_0p15/             # MaxEnt best
    └── wav2vec2_label_smoothing_0p1/     # Label Smoothing best (overall best)
```

The weights bundle ships only the three reported models; the full sweep WER
numbers are in `results/` (committed). Each shard packs `{key}.audio` (FLAC
bytes), `{key}.text` (transcript) and
`{key}.meta` (json). Audio is decoded explicitly via `soundfile` inside
`sample_util.preprocess_sample()`.

## 4. Environment

```bash
conda create --name sr python=3.10
conda activate sr
pip install torch torchaudio                    # match your CUDA version
pip install "transformers[torch]" datasets
pip install webdataset soundfile evaluate jiwer
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 5. How to Run

### Smoke test

```bash
MAX_STEPS=2 EVAL_STEPS=1 SAVE_STEPS=2 TRAIN_BATCH_SIZE=2 EVAL_BATCH_SIZE=2 \
python ./wav2vec_finetuning_label_smoothing.py
```

### Full training (single model)

```bash
# Baseline
python ./wav2vec_finetuning_without_ME.py

# Maximum Entropy   (override ENTROPY_WEIGHT, e.g. best λ = 0.15)
ENTROPY_WEIGHT=0.15 python ./wav2vec_finetuning.py

# Label Smoothing   (override LABEL_SMOOTHING, e.g. best α = 0.10)
LABEL_SMOOTHING=0.10 python ./wav2vec_finetuning_label_smoothing.py
```

Each run takes ~35 min on an RTX 3090 and writes a checkpoint to
`models/wav2vec2_<variant>/`.

### Inference + automatic WER

```bash
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1 \
python ./wav2vec_inference.py
```

Writes `results/<model_name>/test_{clean,other}_result.txt` (REF/HYP) and
`*_wer.txt` (jiwer summary).

### Manual WER

```bash
python ./evaluate_wer.py results/wav2vec2_label_smoothing_0p1/test_clean_result.txt
```

## 6. Regularization Math

Two output-distribution regularizers added on top of the standard CTC loss:

**Maximum Entropy** — `wav2vec_finetuning.py:MaxEntropyCTCTrainer`

```
Loss = CTC − λ · H(p)
H(p) = −Σᵥ pᵥ log pᵥ            # weighted by predicted probability
```

**Label Smoothing** — `wav2vec_finetuning_label_smoothing.py:LabelSmoothingCTCTrainer`

```
Loss = (1−α) · CTC + α · LS(p)
LS(p) = −(1/V) Σᵥ log pᵥ        # uniform weight across vocab
```

Both shrink confident predictions toward uniform; the weighting differs.
Closely related but distinct — see Pereyra et al. 2017, *Regularizing Neural
Networks by Penalizing Confident Output Distributions*.

## 7. Reproducing the Reported Numbers

```bash
# 1. Baseline
python ./wav2vec_finetuning_without_ME.py
MODEL_DIR=./models/wav2vec2_baseline             python ./wav2vec_inference.py

# 2. MaxEnt best λ=0.15
ENTROPY_WEIGHT=0.15 python ./wav2vec_finetuning.py
MODEL_DIR=./models/wav2vec2_maxent_0p15          python ./wav2vec_inference.py

# 3. Label Smoothing best α=0.10
LABEL_SMOOTHING=0.10 python ./wav2vec_finetuning_label_smoothing.py
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1  python ./wav2vec_inference.py
```

All runs share: 2000 training steps, lr = 1e-4, warmup = 500, batch = 16 ×
grad_accum 2, fp16, frozen CNN feature encoder.
