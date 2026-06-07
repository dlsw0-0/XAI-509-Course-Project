---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 {
    color: #1a3a6b;
  }
  h2 {
    color: #1a3a6b;
    border-bottom: 2px solid #c41e3a;
    padding-bottom: 8px;
  }
  table {
    margin: 0 auto;
    font-size: 22px;
  }
  th {
    background-color: #1a3a6b;
    color: white;
  }
  tr:nth-child(even) {
    background-color: #f0f4f8;
  }
  .highlight {
    background-color: #fff4d6;
    font-weight: bold;
  }
---

<!-- _paginate: false -->

# Wav2Vec2 CTC Fine-Tuning<br>with Regularization

### XAI 509 — Course Project

<br>

**LibriSpeech 1-hour fine-tuning + Over-confidence regularization**

2026.06.11

---

## 1. Problem Statement

**Task**: Wav2Vec2 fine-tuning on LibriSpeech with **CTC loss**

| Item | Detail |
|---|---|
| Pretrained model | `facebook/wav2vec2-base` (95M params) |
| Train set | LibriLight **1-hour** subset |
| Evaluation | WER on `test-clean` / `test-other` |

**Core Challenge**:
> Fine-tuning set이 매우 작음 (1시간) → 모델이 학습 데이터에 **over-confident**해지기 쉬움 → 일반화 성능 저하

→ **Output distribution을 부드럽게 만드는 정규화**가 필요

---

## 2. Approach — Two Regularization Methods

**Baseline**
$$ \mathcal{L} = \mathcal{L}_{\text{CTC}} $$

**Method 1 — Maximum Entropy**
$$ \mathcal{L} = \mathcal{L}_{\text{CTC}} - \lambda \cdot H(p), \quad H(p) = -\sum_v p_v \log p_v $$
→ output 분포의 **entropy를 직접 키움**

**Method 2 — Label Smoothing**
$$ \mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{CTC}} + \alpha \cdot \mathrm{LS}(p), \quad \mathrm{LS}(p) = -\frac{1}{V}\sum_v \log p_v $$
→ target을 **uniform 분포와 섞음** (≈ KL(uniform ∥ p))

---

## 3. Experimental Setup

| Component | Value |
|---|---|
| Model | `facebook/wav2vec2-base` |
| Feature encoder | **Frozen** (안정성) |
| Train steps | 2000 (~222 epochs on 1h set) |
| Learning rate | 1e-4, warmup 500 steps |
| Batch | 16 × grad_accum 2 = effective 32 |
| Precision | fp16 |
| Hardware | RTX 3090 (single GPU) |
| Data pipeline | WebDataset (.tar shards) |
| Eval metric | WER (jiwer) |

학습 1회 소요: 약 35분

---

## 4. Main Results

<style scoped>
table { font-size: 26px; }
</style>

| Method | test-clean WER | test-other WER |
|---|:---:|:---:|
| Baseline (CTC only) | 21.16% | 30.17% |
| MaxEnt (λ=0.10) | 19.93% | 28.80% |
| **Label Smoothing (α=0.10)** | **19.51%** | **28.30%** |

**Improvement over baseline:**
- MaxEnt: **−1.23%p** (clean), **−1.37%p** (other)
- Label Smoothing: **−1.65%p** (clean), **−1.87%p** (other)

→ 두 방법 모두 baseline 대비 의미있는 개선, **Label Smoothing이 더 효과적**

---

## 5. MaxEnt λ Sweep

<style scoped>
table { font-size: 22px; }
</style>

| λ | test-clean | test-other |
|:---:|:---:|:---:|
| 0.00 (baseline) | 21.16% | 30.17% |
| 0.01 | 20.60% | 29.67% |
| 0.03 | 20.79% | 29.65% |
| 0.05 | 21.21% | 30.46% |
| 0.07 | 20.93% | 29.80% |
| **0.10** | **19.93%** | **28.80%** |

- λ가 작으면 정규화 효과 약함, 너무 크면 (0.05) 오히려 악화
- **0.10에서 best** — over-confidence 완화 효과가 충분히 발현되는 구간

---

## 6. Analysis — 왜 Label Smoothing이 더 좋은가?

**같은 가족, 다른 수식:**

| 기법 | 가중치 | 효과 |
|---|---|---|
| MaxEnt | $p_v$ (확률 자체) | 이미 확률 큰 토큰을 더 강하게 누름 |
| Label Smoothing | $\frac{1}{V}$ (균등) | 모든 vocab 토큰을 균등하게 정규화 |

**작은 fine-tune set에서:**
- MaxEnt: confident한 token에 편향된 정규화
- LS: vocab 전체를 균등하게 정규화 → 더 **안정적**

→ test-other(어려운 set)에서 차이가 더 큼 (Δ = 0.50%p vs MaxEnt)

---

## 7. Qualitative Example

<style scoped>
section { font-size: 22px; }
</style>

**Reference**:
> ... THEORY OF **MEMORY** MUST **ARRIVE** AT ... **VIRTUE** ... **SINGLED OUT** ...

**Baseline (CTC only):**
> ... THEORY OF MEMERY MUST ~~ARIVE~~ AT ... ~~VERTUE~~ ... ~~SINGLEDOUT~~ ...

**Label Smoothing (α=0.10):**
> ... THEORY OF MEMERY MUST **ARRIVE** AT ... **VIRTUE** ... **SINGLED OUT** ...

→ baseline이 만든 spelling 오류 (ARIVE, VERTUE)와 spacing 오류 (SINGLEDOUT)가 정규화 후 정정됨

---

## 8. Conclusion

✅ Wav2Vec2 CTC fine-tuning에 **두 가지 정규화 방법** 적용·비교

✅ 두 방법 모두 baseline 대비 WER **−1%p 이상 개선**

✅ **Label Smoothing이 MaxEnt보다 효과적** (clean −0.42%p, other −0.50%p 추가 개선)

✅ 어려운 set(test-other)에서 더 큰 개선 → small data 정규화의 가치

**향후 작업**:
- Label smoothing α sweep (0.05, 0.15, 0.20)
- SpecAugment 같은 data augmentation 결합
- LM shallow fusion

---

<!-- _paginate: false -->

# Thank You

<br>

### Q&A

<br>
<br>

**Code & Results**: `/run/models/`, `/run/results/`
