# 논문 요약: 소음 데이터를 이용한 딥러닝 기반의 차량 진단 기술 개발

## 논문 정보

- **제목**: Development of Deep Learning-based Vehicle Diagnosis Technology Using Noise Data
- **저자**: 노경진, 이동철, 진재민, 정인수 (현대자동차), 장준혁 (한양대학교)
- **학회**: 한국소음진동공학회논문집 Vol.33, No.3, pp.306-312
- **발행일**: 2023년
- **DOI**: https://doi.org/10.5050/KSNVE.2023.33.3.306

---

## 1. 연구 목적

1. **고장 진단 (Classification)**: 차량 부품이 정상인지, 특정 부품에 이상이 있는지 분류
2. **소음 레벨 추정 (Regression)**: 정상 상태에서 소음 레벨 지수(NLI)를 1~5 정수로 추정

---

## 2. 전체 시스템 흐름

```
데이터 입력
    ↓
전처리 (Peak Normalization → STFT → HPSS)
    ↓
특징 벡터 추출 (Full Spectrogram + Percussive Spectrogram)
    ↓
┌─────────────────────────────────────────┐
│         고장 진단 모델 (분류)              │
│  결과: 정상 / 오토텐셔너 / 캠샤프트 /       │
│        댐퍼풀리 / 타이밍체인 / 기타         │
└─────────────────────────────────────────┘
    ↓ (정상으로 판정된 경우만)
┌─────────────────────────────────────────┐
│       소음 레벨 지수 추정 모델 (회귀)       │
│         결과: NLI (1~5 정수)              │
└─────────────────────────────────────────┘
```

---

## 3. 특징 벡터 추출 방법

### 3.1 전처리 단계

```
1. Peak Normalization: 최대값이 1이 되도록 정규화
2. STFT: 시간→주파수 도메인 변환
3. Log-Scale Spectrogram: Y(l,k) = 20*log10|X(l,k)|
4. HPSS: Harmonic-Percussive Source Separation
```

### 3.2 STFT 파라미터 (중요!)

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Sampling Rate | 16 kHz | |
| Frame Length | 150 ms | 아이들 RPM 고려 |
| Hop Length | 75 ms | Frame의 50% |
| FFT Size | 4096 | |
| Window | Hanning | |
| 오디오 길이 | 5초 | |

### 3.3 HPSS (Harmonic-Percussive Source Separation)

- **목적**: 타음(percussive) 성분 분리로 타음 유사 소음 분류 성능 향상
- **출력**:
  - Full Spectrogram (원본)
  - Percussive Spectrogram (타음 성분)

---

## 4. 고장 진단 모델 (분류, Fig. 3)

### 4.1 모델 구조

```
[Full Spectrogram]          [Percussive Spectrogram]
        ↓                            ↓
┌───────────────┐            ┌───────────────┐
│ Convolution   │            │ Convolution   │
│ Block         │            │ Block         │
│ (Conv+BN+ReLU)│            │ (Conv+BN+ReLU)│
└───────────────┘            └───────────────┘
        ↓                            ↓
┌───────────────┐            ┌───────────────┐
│    CBAM       │            │    CBAM       │
│ (Attention)   │            │ (Attention)   │
└───────────────┘            └───────────────┘
        ↓                            ↓
┌───────────────┐            ┌───────────────┐
│  Max-Pooling  │            │  Max-Pooling  │
└───────────────┘            └───────────────┘
        ↓                            ↓
   (위 과정 반복)               (위 과정 반복)
        ↓                            ↓
    Reshape                      Reshape
        └──────────┬─────────────┘
                   ↓
            Concatenate
                   ↓
         Fully Connected Layer
                   ↓
         Fully Connected Layer
                   ↓
              Softmax
                   ↓
        [정상/부품1/부품2/.../기타]
```

### 4.2 핵심 특징

- **두 입력 분리 처리**: Full과 Percussive를 각각 독립적인 CNN에 입력
- **CBAM 적용**: 각 Conv Block 후에 CBAM으로 중요 영역 학습
- **출력**: Softmax → One-hot 형태로 Cross Entropy Loss

### 4.3 학습 설정

| 항목 | 값 |
|------|-----|
| Batch Size | 10 |
| Optimizer | Adam |
| Learning Rate | 0.001 (10 epoch마다 0.7배 감쇠) |
| Train:Test | 4:1 |

### 4.4 결과

- **정확도**: 96%
- **CBAM 효과**: 94% → 96% (+2%)

---

## 5. 소음 레벨 지수 추정 모델 (회귀, Fig. 6)

### 5.1 소음 레벨 지수 (NLI) 정의

```
NLI = α × Specific_Loudness + β × SAML + c
```

#### Specific Loudness (특정 주파수 대역 비율)
```
Specific_Loudness = (PN_Range1 / PN_Range2) × 100
```
- PN_Range: 특정 주파수 대역의 크기(loudness)

#### SAML (Sum of Audible Modulation Level)
```
SAML = 10 × log10 × Σ 10^((ΔL(dB) - ΔT.H(dB)) / 10)
```
- ΔL: 모듈레이션 레벨
- ΔT.H: 모듈레이션 임계값 레벨
- 모듈레이션 스펙트럼에서 임계값 초과 성분들의 합

### 5.2 모델 구조 (중요! - 분류 모델과 다름)

```
[Full Spectrogram + Percussive Spectrogram]
              ↓
        채널 축으로 결합 (Concatenate)
              ↓
┌─────────────────────────────────────┐
│       Convolution Block             │
│       (Conv + BN + ReLU)            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│            CBAM                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│         Max-Pooling                 │
└─────────────────────────────────────┘
              ↓
         (위 과정 4~5회 반복)
              ↓
           Reshape
              ↓
      Fully Connected Layer
              ↓
           Sigmoid × 5
              ↓
         [0~5 실수값]
              ↓
            올림
              ↓
         [1~5 정수값]
```

### 5.3 분류 모델과의 핵심 차이

| 항목 | 고장 진단 (분류) | 소음 레벨 추정 (회귀) |
|------|-----------------|---------------------|
| **입력 방식** | 두 스펙트로그램 **분리** 입력 | 두 스펙트로그램 **채널 결합** |
| **출력 함수** | Softmax | Sigmoid × 5 |
| **Loss** | Cross Entropy | MSE |
| **출력 형태** | 클래스 확률 | 0~5 실수 → 1~5 정수 |

### 5.4 채널 결합 방식을 선택한 이유

> "소음 레벨 지수 추정 모델은 정상 상태의 소음 레벨 지수만을 타겟으로 하는 모델이기 때문에
> 타음 성분에서 따로 특징을 추출하는 방법이 성능 향상을 가져오지 않는 것으로 분석된다."
> (논문 p.310)

### 5.5 학습 설정

| 항목 | 값 |
|------|-----|
| Batch Size | 10 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Train:Test | 4:1 (정상 데이터만) |

### 5.6 결과

- **정확도**: 86% (1~5 정수 분류 기준)
- **CBAM 효과**: 78% → 86% (+8%) ← **매우 중요!**

---

## 6. CBAM (Convolutional Block Attention Module)

### 6.1 구조

```
입력 Feature Map
        ↓
┌───────────────────────────┐
│   Channel Attention       │
│   (채널별 중요도 학습)      │
└───────────────────────────┘
        ↓
┌───────────────────────────┐
│   Spatial Attention       │
│   (공간별 중요도 학습)      │
└───────────────────────────┘
        ↓
출력 Feature Map
```

### 6.2 역할

- **Channel Attention**: 어떤 채널(특징)이 중요한지 학습
- **Spatial Attention**: 어떤 시간/주파수 구간이 중요한지 학습
- **효과**: 모델이 데이터 기반으로 **주요 소음 발생 구간과 주파수 대역을 자동 학습**

### 6.3 CBAM 성능 향상 효과

| 태스크 | CBAM 미사용 | CBAM 사용 | 향상 |
|--------|------------|----------|------|
| 고장 진단 (분류) | 94% | 96% | **+2%** |
| 소음 레벨 추정 (회귀) | 78% | 86% | **+8%** |

---

## 7. 실험 환경

### 7.1 데이터 수집

- **장비**: 태블릿 PC 내장 마이크
- **위치**: 차량 정면 30cm, 보닛 끝에서 25cm 높이
- **환경**: 외부 (외란 최소화)
- **상태**: 아이들 상태
- **차량 수**: 80대
- **엔진 종류**: 디젤

### 7.2 진단 대상 부품 (4개)

1. 오토텐셔너 (Auto-Tensioner)
2. 캠샤프트 (Camshaft)
3. 댐퍼 풀리 (Damper Pulley)
4. 타이밍 체인 (Timing Chain)

### 7.3 데이터 구성

- 정상 소음
- 4개 부품별 이상 소음 (고장 부품을 엔진에 조립하여 발생)
- 기타 부품 이상 소음

---

## 8. 핵심 인사이트 (우리 프로젝트 적용)

### 8.1 적용 가능한 개선점

| 개선점 | 현재 상태 | 논문 방식 | 예상 효과 |
|--------|----------|----------|----------|
| **CBAM 추가** | 미적용 | 적용 | **+8% (회귀에서)** |
| 입력 방식 | Dual (분리) | 채널 결합 | 회귀에 적합 |
| FFT Size | 2048 | 4096 | 주파수 해상도 향상 |
| Frame Length | - | 150ms | 아이들 RPM 최적화 |
| Sampling Rate | 22050 | 16000 | 불필요 주파수 제거 |

### 8.2 태스크 차이점

| 항목 | 논문 | 우리 프로젝트 |
|------|------|--------------|
| 회귀 출력 | NLI 1개 | 5개 점수 동시 |
| 데이터 | 80대 | 594개 |
| 엔진 종류 | 디젤 | 다양 |
| 정상/고장 | 둘 다 있음 | 정상만 |

### 8.3 우선 적용 권장

1. **Simple CNN + CBAM**: 최소 변경으로 +8% 기대
2. **채널 결합 방식**: 회귀 태스크에 최적화
3. **전처리 파라미터**: Frame 150ms, Hop 75ms, FFT 4096

---

## 9. 참고 문헌 (논문에서 인용)

- CBAM: Woo et al., ECCV 2018
- HPSS: Fitzgerald 2010, Driedger et al. 2014
- Adam Optimizer: Kingma & Ba 2014

---

## 10. 원문 Figure 설명

| Figure | 설명 |
|--------|------|
| Fig. 1 | 전체 시스템 순서도 |
| Fig. 2 | 특징 추출 과정 (Normalization → STFT → HPSS) |
| Fig. 3 | **고장 진단 모델 (분류)** - 두 입력 분리 |
| Fig. 4 | Specific Loudness 정의 |
| Fig. 5 | Modulation Analysis (SAML 계산) |
| Fig. 6 | **소음 레벨 추정 모델 (회귀)** - 채널 결합 |
| Fig. 7 | 데이터 수집 환경 |
| Fig. 8 | 진단 대상 부품 위치 |
| Fig. 9 | 분류 모델 Loss/Accuracy 곡선 |
| Fig. 10 | 분류 모델 Confusion Matrix |
| Fig. 11 | 회귀 모델 Loss/Accuracy 곡선 |

---

**문서 작성일**: 2026-01-11
**원본 파일**: `reference/딥러닝을_활용_한_자동차_엔진음_판단.pdf`
