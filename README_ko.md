# Wav2Vec2 CTC Fine-Tuning + Output Regularization (한국어)

**🌐 언어:** [English](README.md) | **한국어**

XAI 509 course project. `facebook/wav2vec2-base`를 LibriSpeech 1시간 set에서
CTC loss로 fine-tune하고, two output-distribution regularizer(Maximum Entropy,
Label Smoothing)를 baseline과 비교합니다. 영문 상세 설명은 `README.md` 참고.

## ⬇️ 다운로드

| 항목 | 링크 |
| :--- | :--- |
| **Pretrained weights** (best 3개 모델, ~1 GB) | [![Weights](https://img.shields.io/badge/Download-Weights%20(GitHub%20Release)-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/dlsw0-0/XAI-509-Course-Project/releases/download/v1.0/pretrained_weights.zip) |
| **Fine-tuning set** (Libri-light 1시간, train) | [![Train](https://img.shields.io/badge/Download-Train%20Set-34A853?logo=googledrive&logoColor=white&style=for-the-badge)](https://drive.google.com/file/d/153mEZhH_PvwbAgqvgrhKCY3BKT4mnu9/view?usp=drive_link) |
| **Test sets** (test-clean + test-other) | [![Test](https://img.shields.io/badge/Download-Test%20Sets-34A853?logo=googledrive&logoColor=white&style=for-the-badge)](https://drive.google.com/file/d/1OT4KazgFBdWIXYGizlNUUdDmmctPY8vN/view?usp=drive_link) |

weights 묶음에는 보고된 3개 모델(`wav2vec2_baseline`, `wav2vec2_maxent_0p15`,
`wav2vec2_label_smoothing_0p1`)이 추론 전용(optimizer/checkpoint 제거)으로 들어있습니다.
**`run/` 안에서 압축을 풀면** 추론 코드가 기대하는 위치에 그대로 들어갑니다:

```bash
# run/ 디렉토리에서
unzip pretrained_weights.zip          # run/models/wav2vec2_* 생성
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1 python ./wav2vec_inference.py
```

데이터셋은 위 Drive 링크 2개를 받아 `dataset/` 아래에 풀면 됩니다 (§3 구조 참고).

## 1. 결과 요약

각 방법의 best operating point (전체 sweep은 §1.1):

| 모델                                | test-clean WER | test-other WER | baseline 대비(other) |
| :---------------------------------- | :------------: | :------------: | :------------------: |
| Baseline (CTC only)                 |    21.16 %     |    30.17 %     |          —           |
| Maximum Entropy (best, λ = 0.15)    |    19.88 %     |    28.63 %     |       −1.54 pp       |
| **Label Smoothing (best, α = 0.10)**|  **19.51 %**   |  **28.30 %**   |     **−1.87 pp**     |

두 regularizer 모두 baseline보다 우수. **Label Smoothing α=0.10이 두 셋 모두에서
전체 최고** (best MaxEnt 대비 clean −0.37 / other −0.33 pp). 어려운 `test-other`에서
개선폭이 더 큼.

> **튜닝 관련 한계:** 별도 validation set이 없고(제공 데이터는 train / test-clean /
> test-other 뿐, train은 286개 발화) skeleton 설정대로 test-clean을 eval로 사용해
> α/λ를 test WER 기준으로 골랐습니다. 따라서 보고된 best는 낙관적 추정이며, 본
> 프로젝트 목적은 over-confidence 완화 기법의 *비교*입니다.

### 1.1 하이퍼파라미터 Sweep

**Label Smoothing — α sweep** (0.10에서 날카로운 최적점):

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

**Maximum Entropy — λ sweep** (0.15–0.20에서 완만한 최적):

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

## 2. 코드 구조

```
run/
├── README.md / README_ko.md
├── .gitignore
├── sample_util.py                        # WebDataset 로딩 + 전처리
├── evaluate_wer.py                       # WER 계산
├── wav2vec_finetuning_without_ME.py      # Baseline
├── wav2vec_finetuning.py                 # + Maximum Entropy
├── wav2vec_finetuning_label_smoothing.py # + Label Smoothing
├── wav2vec_inference.py                  # 추론 + 자동 WER
├── run_experiments.sh                    # sweep 순차 실행
└── results/                              # WER 텍스트 결과 (commit됨)
```

`dataset/`, `models/`는 git에 없음 — 위 다운로드 버튼에서 받으세요.

## 3. 데이터 / 모델 배치

```
run/
├── dataset/
│   ├── train/        shard-000000~4.tar   # Libri-light 1시간 (286 utt)
│   ├── test-clean/   shard-*.tar          # 2620 utt
│   └── test-other/   shard-*.tar          # 2939 utt
└── models/
    ├── wav2vec2_baseline/
    ├── wav2vec2_maxent_0p01 … 0p2/
    └── wav2vec2_label_smoothing_0p01 … 0p2/
```

## 4. 환경

```bash
conda create --name sr python=3.10
conda activate sr
pip install torch torchaudio
pip install "transformers[torch]" datasets webdataset soundfile evaluate jiwer
```

## 5. 실행

```bash
# Baseline
python ./wav2vec_finetuning_without_ME.py

# Maximum Entropy (best λ=0.15)
ENTROPY_WEIGHT=0.15 python ./wav2vec_finetuning.py

# Label Smoothing (best α=0.10)
LABEL_SMOOTHING=0.10 python ./wav2vec_finetuning_label_smoothing.py

# 추론 + 자동 WER
MODEL_DIR=./models/wav2vec2_label_smoothing_0p1 python ./wav2vec_inference.py
```

각 run은 RTX 3090에서 약 35분 소요.

## 6. 정규화 수식

```
Maximum Entropy:  Loss = CTC − λ · H(p),   H(p) = −Σ p log p
Label Smoothing:  Loss = (1−α) · CTC + α · LS(p),   LS(p) = −(1/V) Σ log p
```

둘 다 over-confident output을 누르는 정규화 (MaxEnt는 p로 가중, LS는 uniform).
Pereyra et al. 2017 참고.
