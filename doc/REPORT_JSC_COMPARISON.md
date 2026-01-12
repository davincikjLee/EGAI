# EGAI 프로젝트 분석 보고서

## 개요

**작성일**: 2026-01-11
**목적**: JSC 원본 코드(`analysis_jsc`)와 현재 개선된 코드(`analysis`)의 차이점 분석 및 문의사항 답변

---

## 1. 프로젝트 구조 비교

### 1.1 폴더 구조 변경

| 구분 | analysis_jsc (원본) | analysis (현재) |
|------|-------------------|-----------------|
| **파일 수** | 14개 | 25개 |
| **모델 정의** | `fault_diagnosis_model.py` | `models.py` (6개 모델) |
| **학습 스크립트** | `train.py` 1개 | `train.py`, `train_cv.py` 등 5개 |
| **실험 관리** | 없음 | `experiment_logger.py`, `experiment_tracker.py` |
| **데이터 로더** | train.py 내장 | 분리된 `data_loader.py` |

### 1.2 주요 추가 파일

```
analysis/
├── data_loader.py          # 데이터 로딩/캐싱 분리 (NEW)
├── models.py               # 6개 모델 아키텍처 (확장)
├── train_cv.py             # 5-Fold Cross Validation (NEW)
├── experiment_logger.py    # 실험 자동 기록 (NEW)
├── experiment_tracker.py   # 실험 추적 시스템 (NEW)
├── analyze_accuracy.py     # 정확도 분석 도구 (NEW)
├── score_relationship.py   # 점수 상관관계 분석 (NEW)
└── experiments/            # 실험 결과 저장 폴더 (NEW)
```

**문제점1 해결**: 코드가 모듈별로 분리되어 구조가 명확해짐

---

## 2. 태스크 변경

### 2.1 분류 → 회귀로 전환

| 구분 | analysis_jsc | analysis |
|------|--------------|----------|
| **태스크** | 분류 (Classification) | 회귀 (Regression) |
| **타겟** | `overall_score` 1개 | 5개 개별 점수 |
| **출력 형태** | Softmax (클래스 확률) | Linear (연속 값) |
| **손실 함수** | CrossEntropyLoss | MSE (Mean Squared Error) |
| **평가 지표** | Accuracy | MAE, R² |

### 2.2 전환 이유

1. **세부 정보 보존**: `overall_score`는 5개 점수의 종합 → 개별 점수 예측 시 상세 진단 가능
2. **클래스 불균형 해소**: 원본은 4.2, 4.4점에 데이터 집중 → 분류 성능 저하
3. **활용성 향상**: "중음은 양호하지만 비규칙성이 높다" 같은 구체적 피드백 가능

### 2.3 예측 대상 5개 점수

| 컬럼명 | 의미 | 분포 특성 |
|--------|------|----------|
| `mid_freq_score` | 중음역대 품질 | 대부분 4점 |
| `low_high_freq` | 저/고주파 규칙성 | 3-4점 분포 |
| `audable_range_score` | 가청대역 점수 | 94%가 5점 (불균형) |
| `regularity` | 규칙성 | 4점 중심 |
| `irregularity` | 비규칙성 | 2-3점 분포 |

---

## 3. 모델 구조 비교

### 3.1 analysis_jsc 모델 (FaultDiagnosisModel)

```
[Full Spec] → Conv(64) → CBAM → Pool → Conv(128) → CBAM → Pool → Flatten ─┐
                                                                            ├→ Concat → FC(256) → FC(128) → Softmax
[Percussive] → Conv(64) → CBAM → Pool → Conv(128) → CBAM → Pool → Flatten ─┘
```

- **Conv 블록**: 2개 (각 브랜치)
- **특징 추출**: Flatten → 262,144차원 (매우 큼)
- **구현 방식**: Subclassed Model

### 3.2 analysis 모델 (build_dual_input_cbam)

```
[Full Spec] → Conv(32→64→128→256) → CBAM → GlobalAvgPool ─┐
                                                           ├→ Concat(512) → Dense → Linear(5)
[Percussive] → Conv(32→64→128→256) → CBAM → GlobalAvgPool ─┘
```

- **Conv 블록**: 4개 (각 브랜치)
- **특징 추출**: GlobalAveragePooling → 512차원 (효율적)
- **구현 방식**: Functional API

### 3.3 구조적 개선점

| 항목 | analysis_jsc | analysis | 개선 효과 |
|------|--------------|----------|----------|
| 특징 차원 | 262,144 | 512 | 메모리 99.8% 절감 |
| Conv 깊이 | 2층 | 4층 | 더 복잡한 패턴 학습 |
| Pooling | Flatten | GlobalAvgPool | 과적합 감소 |

---

## 4. 실험 결과 및 문제점2 대응

### 4.1 현재 데이터 상황

- **유효 샘플**: 594개 (7개 오디오 파일 손상)
- **데이터 특성**: 모두 **정상 엔진** (고장 데이터 없음)
- **라벨 분포**: 극심한 불균형 (audable_range_score 94%가 5점)

### 4.2 실험 결과 요약 (5-Fold Cross Validation)

| 실험 ID | 모델 | MAE | Baseline 대비 |
|---------|------|-----|--------------|
| **Baseline** | Simple CNN | **0.5635 ± 0.0948** | **- (최고)** |
| exp_002 | multiinput_physics (+14 물리량) | 0.7876 | +40% 악화 |
| exp_003 | multiinput_metadata (+14 메타데이터) | 0.8512 | +51% 악화 |
| exp_004 | multiinput_modulation (+20 모듈레이션) | 0.7184 | +27% 악화 |

### 4.3 컬럼별 예측 성능 (Baseline)

| 점수 | MAE | 해석 |
|------|-----|------|
| mid_freq_score | 0.51 | 평균 0.5점 오차 (양호) |
| audable_range_score | 0.26 | 가장 좋음 (94%가 5점이라 쉬움) |
| regularity | 0.56 | 보통 |
| low_high_freq | 0.66 | 다소 어려움 |
| **irregularity** | **0.83** | **가장 어려움** |

### 4.4 데이터 부족 문제 분석

> **문제점2**: 1000개 데이터 중 라벨링하면 100개 이하로 떨어지는 데이터셋 존재

**발견된 패턴**:
- 594개 샘플에서 **추가 특징을 넣을수록 성능 악화**
- 물리량 14개 추가 → +40% 악화
- 메타데이터 14개 추가 → +51% 악화
- 모듈레이션 20개 추가 → +27% 악화

**결론**: 데이터가 적은 상황에서는 **Simple CNN이 최적**. 복잡한 모델은 오히려 과적합 발생.

---

## 5. 문의점3 대응: 시각적 피드백 시스템

### 5.1 구현된 시각화 도구

1. **스펙트로그램 시각화** (`audio_preprocessing.py`)
   - Full Spectrogram + Percussive Spectrogram 비교 이미지 저장
   - 첫 샘플 자동 저장: `first_sample_spectrograms.png`

2. **학습 과정 시각화** (`train_cv.py`)
   - Loss/MAE 곡선 그래프
   - 폴드별 성능 비교

3. **실험 결과 리포트** (`experiment_logger.py`)
   - JSON 형태 자동 저장
   - 컬럼별 MAE, R² 기록
   - 실험 파라미터 전체 기록

### 5.2 실험 결과 저장 위치

```
analysis/experiments/
├── exp_001/
│   ├── config.json       # 실험 설정
│   ├── results.json      # 결과 요약
│   └── fold_results.json # 폴드별 상세 결과
├── exp_002/
├── exp_003/
└── exp_004/
```

### 5.3 스펙트로그램 "거의 초록색" 문제

> **문의점3**: 스펙트로그램이 큰 차이 없이 거의 초록색

**원인 분석**:
1. **정상 엔진만 수집**: 고장 엔진 데이터가 없어 스펙트로그램 패턴이 유사
2. **정규화 효과**: 0~1 정규화로 미세한 차이가 시각적으로 드러나지 않음
3. **주파수 제한**: 0~6000Hz로 제한하여 고주파 노이즈 제거됨

**실제 학습 가능 여부**:
- 시각적으로 유사해도 **수치적 차이는 존재**
- CNN은 미세한 패턴 차이도 학습 가능
- 단, **정상 vs 고장 데이터가 모두 필요**

---

## 6. 문의점4 대응: 엔진음 분석 기법 비교

### 6.1 현재 적용된 기법 (논문 기반)

| 기법 | 출처 | 구현 상태 |
|------|------|----------|
| STFT (Short-Time Fourier Transform) | 기본 | 적용됨 |
| HPSS (Harmonic-Percussive Source Separation) | 논문 | 적용됨 |
| CBAM (Convolutional Block Attention Module) | 논문 | 적용됨 |
| Modulation Spectrum | 현대차 논문 | 적용됨 (exp_004) |

### 6.2 추가 시도한 기법

| 기법 | 실험 ID | 결과 |
|------|---------|------|
| 물리량 특징 (ZCR, Spectral Centroid 등 14개) | exp_002 | 실패 (+40%) |
| 메타데이터 결합 (주행거리, 연식 등) | exp_003 | 실패 (+51%) |
| 모듈레이션 특징 (Spectral Flux, Onset 등 20개) | exp_004 | 실패 (+27%) |

### 6.3 왜 추가 기법이 실패했나?

1. **데이터 부족**: 594개 샘플로는 복잡한 모델 학습 불가
2. **정보 중복**: 스펙트로그램에 이미 물리량 정보가 포함됨
3. **태스크 불일치**: 논문은 "고장 진단"(정상 vs 고장), 현재는 "품질 점수 회귀"

### 6.4 권장 개선 방향

| 우선순위 | 방법 | 예상 효과 |
|----------|------|----------|
| **1순위** | 데이터 확대 (2000개 이상) | 가장 큰 효과 |
| **2순위** | 고장 엔진 데이터 수집 | 정상/고장 구분 학습 가능 |
| 3순위 | Sample Weighting | 클래스 불균형 완화 |
| 4순위 | Ordinal Regression | 정수 점수 예측 최적화 |
| 5순위 | 앙상블 (여러 Simple CNN) | 안정성 향상 |

---

## 7. 핵심 발견 및 결론

### 7.1 주요 발견

1. **Simple CNN이 현재 최적**: 데이터가 적을 때 복잡한 모델은 역효과
2. **irregularity 예측이 가장 어려움**: MAE 0.83 (다른 점수는 0.26~0.66)
3. **정상 엔진만으로는 한계**: 고장 데이터 없이는 진정한 진단 AI 불가

### 7.2 JSC 의뢰사항 해결 현황

| 문제/문의 | 상태 | 설명 |
|-----------|------|------|
| 문제점1: 구조 정리 | 완료 | 모듈별 분리, 실험 관리 시스템 구축 |
| 문제점2: 데이터 부족 | 확인됨 | Simple CNN 유지 권장, 데이터 확대 필요 |
| 문의점3: 시각적 피드백 | 구현됨 | 스펙트로그램, 학습 곡선, 실험 리포트 자동 저장 |
| 문의점4: 더 좋은 방법 | 시도함 | 3가지 추가 기법 모두 실패 → 데이터 확대가 우선 |

### 7.3 최종 권장사항

```
현재 상태: Simple CNN (MAE 0.56) 유지

단기 개선:
├── 1. 데이터 클리닝 (손상된 7개 파일 교체)
├── 2. Sample Weighting (audable_range_score 불균형 해소)
└── 3. 더 많은 데이터 수집 (최소 2000개 목표)

장기 개선:
├── 1. 고장 엔진 오디오 수집 (정상/고장 분류 모델 구축)
├── 2. Transfer Learning (YAMNet, VGGish 등 사전학습 모델)
└── 3. Ordinal Regression (1-5점 정수 예측 최적화)
```

---

## 부록: 실험 환경

- **GPU**: NVIDIA RTX 4060 (8GB)
- **Python**: 3.x
- **Framework**: TensorFlow/Keras
- **검증 방식**: 5-Fold Stratified Cross-Validation
- **Early Stopping**: patience=10

---

**작성자**: Claude (AI Assistant)
**검토 필요**: JSC
