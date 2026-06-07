# Wav2Vec2 CTC Fine-Tuning + Output Regularization (한국어)

XAI 509 course project. `facebook/wav2vec2-base`를 LibriSpeech 1시간 set에서
CTC loss로 fine-tune하고, two output-distribution regularizer를 baseline과 비교합니다.

영문 자세한 설명은 `README.md` 참고.

## 1. 결과 요약

| 모델                              | test-clean WER | test-other WER | baseline 대비 |
| :-------------------------------- | :------------: | :------------: | :-----------: |
| Baseline (CTC only)               |    21.16 %     |    30.17 %     |        —      |
| Maximum Entropy (λ = 0.10)        |    19.93 %     |    28.80 %     |    −1.37 pp   |
| **Label Smoothing (α = 0.10)**    |  **19.51 %**   |  **28.30 %**   |  **−1.87 pp** |

두 regularizer 모두 baseline보다 우수, Label Smoothing이 가장 좋고 어려운
`test-other`에서 개선폭이 더 큼.

## 2. 폴더 구조

```
run/
├── README.md / README_ko.md
├── *.py                                 # baseline / MaxEnt / Label Smoothing / inference
├── run_experiments.sh
├── presentation/                        # 발표 자료 (pptx, pdf, 빌더 script)
├── results/                             # 텍스트 WER 결과 (commit됨)
├── dataset/                             # WebDataset shards            (gitignore)
├── models/                              # checkpoint                    (gitignore)
└── references/                          # 강의 자료 / project.pdf        (gitignore)
```

## 3. 환경

```bash
conda create --name sr python=3.10
conda activate sr
pip install torch torchaudio
pip install "transformers[torch]" datasets webdataset soundfile evaluate jiwer
pip install python-pptx          # 발표자료 재빌드용 (선택)
```

## 4. 실행 예시

```bash
# 1. Baseline
python ./wav2vec_finetuning_without_ME.py

# 2. Maximum Entropy
ENTROPY_WEIGHT=0.10 python ./wav2vec_finetuning.py

# 3. Label Smoothing
LABEL_SMOOTHING=0.10 python ./wav2vec_finetuning_label_smoothing.py

# 4. Inference + 자동 WER
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1 python ./wav2vec_inference.py
```

각 run은 RTX 3090에서 약 35분 소요.

## 5. 정규화 수식

**Maximum Entropy**

```
Loss = CTC − λ · H(p),   H(p) = −Σ p log p
```

**Label Smoothing**

```
Loss = (1−α) · CTC + α · LS(p),   LS(p) = −(1/V) Σ log p
```

둘 다 over-confident output을 누르는 정규화. weighting 방식이 다름
(MaxEnt는 p로 가중, LS는 uniform). Pereyra et al. 2017 참고.

## 6. Smoke Test

```bash
MAX_STEPS=2 EVAL_STEPS=1 SAVE_STEPS=2 TRAIN_BATCH_SIZE=2 EVAL_BATCH_SIZE=2 \
python ./wav2vec_finetuning_label_smoothing.py
```

## 7. 발표 자료

`presentation/presentation.pptx` — 발표 슬라이드. Python 빌더로 재생성 가능:

```bash
python presentation/make_presentation.py
```
