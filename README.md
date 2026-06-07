# Wav2Vec2 CTC Fine-Tuning with Output-Distribution Regularization

**🌐 Language:** **English** | [한국어](README_ko.md)

XAI 509 Course Project — fine-tune `facebook/wav2vec2-base` with the CTC loss on
a 1-hour LibriSpeech subset, then compare two output-distribution regularizers
against the plain CTC baseline.

## 1. Results Summary

| Method                          | test-clean WER | test-other WER | Δ vs baseline |
| :------------------------------ | :------------: | :------------: | :-----------: |
| Baseline (CTC only)             |     21.16 %    |     30.17 %    |        —      |
| Maximum Entropy (λ = 0.10)      |     19.93 %    |     28.80 %    |    −1.37 pp   |
| **Label Smoothing (α = 0.10)**  |   **19.51 %**  |   **28.30 %**  |  **−1.87 pp** |

Both regularizers improve over baseline. Label Smoothing is the most effective,
with the larger gain on the harder `test-other` split.

## 2. Repository Layout

```
run/
├── README.md                            # this file
├── README_ko.md                         # 한국어 버전
├── .gitignore
│
├── sample_util.py                       # WebDataset loading + audio/text preprocessing
├── evaluate_wer.py                      # WER computation (CLI + importable)
│
├── wav2vec_finetuning_without_ME.py     # Baseline (CTC only)
├── wav2vec_finetuning.py                # Maximum Entropy regularization
├── wav2vec_finetuning_label_smoothing.py # Label Smoothing regularization
├── wav2vec_inference.py                 # Inference + auto WER
├── run_experiments.sh                   # Sequential baseline + MaxEnt λ sweep
│
├── presentation/                        # Course presentation deliverables
│   ├── presentation.pptx                # PowerPoint slides
│   ├── presentation.pdf                 # rendered preview
│   ├── presentation.md                  # Marp source
│   └── make_presentation.py             # python-pptx builder script
│
├── results/                             # WER text logs (committed)
│   ├── wav2vec2_baseline/
│   ├── wav2vec2_maxent_0p01/ … 0p1/
│   └── wav2vec2_label_smoothing_0p1/
│
├── dataset/                             # WebDataset shards          (gitignored)
├── models/                              # Trained checkpoints        (gitignored)
└── references/                          # Lecture slides, project PDF (gitignored)
```

## 3. Environment

```bash
conda create --name sr python=3.10
conda activate sr
pip install torch torchaudio                    # match your CUDA version
pip install "transformers[torch]" datasets
pip install webdataset soundfile evaluate jiwer
pip install python-pptx                         # only for rebuilding the deck
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 4. Dataset Layout

The scripts expect WebDataset `.tar` shards at:

```
run/dataset/
  train/            shard-000000.tar … shard-000004.tar   # LibriLight 1h
  test-clean/       shard-*.tar                           # 2620 utt
  test-other/       shard-*.tar                           # 2939 utt
```

Each shard packs `{key}.audio` (FLAC bytes), `{key}.text` (transcript) and
`{key}.meta` (json) entries. Audio is decoded explicitly via `soundfile` inside
`sample_util.preprocess_sample()`.

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

# Maximum Entropy   (default ENTROPY_WEIGHT=0.01, override e.g. 0.10)
ENTROPY_WEIGHT=0.10 python ./wav2vec_finetuning.py

# Label Smoothing   (default LABEL_SMOOTHING=0.10)
LABEL_SMOOTHING=0.10 python ./wav2vec_finetuning_label_smoothing.py
```

Each run takes ~35 min on an RTX 3090 and writes a checkpoint to
`models/wav2vec2_<variant>/`.

### Sweep helper (baseline + MaxEnt λ ∈ {0.01, 0.03, 0.05} by default)

```bash
bash ./run_experiments.sh
```

Override the sweep:

```bash
ENTROPY_WEIGHTS="0.05 0.07 0.10" bash ./run_experiments.sh
```

### Inference + automatic WER

```bash
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1 \
python ./wav2vec_inference.py
```

This writes `results/<model_name>/test_{clean,other}_result.txt` (REF/HYP) and
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
MODEL_DIR=./models/wav2vec2_baseline                  python ./wav2vec_inference.py

# 2. MaxEnt λ=0.10
ENTROPY_WEIGHT=0.10 python ./wav2vec_finetuning.py
MODEL_DIR=./models/wav2vec2_maxent_0p1                python ./wav2vec_inference.py

# 3. Label Smoothing α=0.10
LABEL_SMOOTHING=0.10 python ./wav2vec_finetuning_label_smoothing.py
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1       python ./wav2vec_inference.py
```

All three runs share these settings: 2000 training steps, lr=1e-4, warmup=500,
batch=16 × grad_accum 2, fp16, frozen CNN feature encoder.

## 8. Presentation Deck

`presentation/presentation.pptx` is the 11-slide deck for the June 11 class
presentation. To rebuild after editing the Python script:

```bash
python presentation/make_presentation.py
```
