# EGAI 모델 개선 실험 로그

## 실험 개요

- **목표**: 엔진 오디오 기반 품질 점수 예측 모델 성능 개선
- **기준 모델**: Simple CNN (STFT)
- **평가 지표**: MAE (Mean Absolute Error), R² (결정계수)
- **데이터**: 567개 유효 샘플 (Train: 453, Val: 57, Test: 57)
- **타겟**: 4개 점수 (mid_freq_score, irregularity, low_high_freq, regularity)

---

## 실험 결과 요약

| 실험 | MAE ↓ | R² ↑ | 결과 |
|------|-------|------|------|
| **기준: Simple CNN (STFT)** | **0.5387** | **0.0430** | **최고 성능** |
| 개선 1: Mel Spectrogram + fmax=4000Hz | 0.9097 | -2.7726 | 크게 악화 |
| 개선 2: 분류 문제 (3등급) | - | - | 클래스 불균형 |
| 개선 3: 데이터 증강 2x | 0.5493 | 0.0289 | 소폭 악화 |
| 개선 4: Spectral Features MLP | 0.5564 | -0.1338 | 악화 |

---

## 상세 실험 내용

### 기준 모델: Simple CNN (STFT)

**설정**:
- 입력: STFT 스펙트로그램 (128x128x1)
- fmax: 6000Hz
- 모델: 4-layer CNN (431K 파라미터)
- Epochs: 100, Batch: 16

**결과**:
```
전체 MAE: 0.5387
전체 R²:  0.0430

컬럼별 결과:
  mid_freq_score  - MAE: 0.3889, R²: 0.2553
  irregularity    - MAE: 0.7434, R²: -0.0252
  low_high_freq   - MAE: 0.4797, R²: -0.0660
  regularity      - MAE: 0.5427, R²: 0.0079
```

**분석**: R²가 0에 가까운 이유는 점수 분포가 좁고(대부분 3~5점), 데이터가 부족하기 때문.

---

### 개선 1: Mel Spectrogram + fmax=4000Hz

**가설**: Mel 스케일이 인간 청각 특성을 반영하여 더 좋은 특징 추출 가능

**설정**:
- 입력: Mel Spectrogram (128 mel bands)
- fmax: 4000Hz (엔진음 주파수 범위)
- n_mels: 128

**결과**:
```
전체 MAE: 0.9097
전체 R²:  -2.7726

컬럼별 결과:
  mid_freq_score  - MAE: 1.3274, R²: -6.1563
  irregularity    - MAE: 0.7125, R²: -0.4896
  low_high_freq   - MAE: 1.0271, R²: -4.3425
  regularity      - MAE: 0.5719, R²: -0.1020
```

**분석**:
- Mel Spectrogram이 음성 인식에는 적합하지만 엔진음에는 부적합
- 엔진음은 하모닉 구조가 음성과 다름
- fmax 4000Hz가 너무 낮아 중요 정보 손실 가능

**결론**: 기각 - STFT가 엔진음에 더 적합

---

### 개선 2: 분류 문제로 변환

**가설**: 연속 회귀보다 등급 분류가 더 쉬울 수 있음

**설정**:
- 4개 점수 평균 → 3등급 (Poor/Normal/Good)
- 구간: Poor(1-2.33), Normal(2.33-3.67), Good(3.67-5)
- 모델: CNN + Softmax (431K 파라미터)

**클래스 분포**:
```
Poor:   Train=0,   Val=0,  Test=0
Normal: Train=96,  Val=12, Test=13
Good:   Train=353, Val=45, Test=43
```

**결과**:
```
정확도: 76.8%

혼동 행렬:
          Poor  Normal  Good
Poor        0       0     0
Normal      0       0    13
Good        0       0    43
```

**분석**:
- 심각한 클래스 불균형 (Poor 클래스 0개)
- 모델이 모든 샘플을 "Good"으로 예측
- 76.8% 정확도는 단순히 다수 클래스 예측 결과

**결론**: 기각 - 데이터 불균형 문제로 의미 없음

---

### 개선 3: 데이터 증강 (2x)

**가설**: 데이터 증강으로 과적합 방지 및 일반화 성능 향상

**증강 기법**:
- Time Shift (시간 이동)
- Add Noise (백색 잡음 추가)
- Volume Change (볼륨 변경)
- 적용 확률: 70%

**제외한 기법**:
- Pitch Shift: 엔진 RPM 정보 왜곡 우려
- Time Stretch: 주파수 특성 변화 우려

**설정**:
- 원본 453개 + 증강 906개 = 총 1,359개
- 동일 모델 구조

**결과**:
```
전체 MAE: 0.5493
전체 R²:  0.0289

컬럼별 결과:
  mid_freq_score  - MAE: 0.4294, R²: 0.1559
  irregularity    - MAE: 0.7803, R²: -0.0517
  low_high_freq   - MAE: 0.4738, R²: -0.1157
  regularity      - MAE: 0.5137, R²: 0.1273
```

**분석**:
- MAE 소폭 악화 (0.5387 → 0.5493)
- 증강이 노이즈를 추가하여 오히려 학습 방해
- 원본 데이터의 패턴이 이미 충분히 학습됨

**결론**: 기각 - 효과 없음 (오히려 소폭 악화)

---

### 개선 4: Spectral Features (MLP)

**가설**: CNN 대신 수작업 특징(hand-crafted features)이 더 효과적일 수 있음

**추출 특징 (52개)**:
1. Spectral Centroid (주파수 무게중심) - mean, std
2. Spectral Rolloff (에너지 분포) - mean, std
3. Spectral Flatness (노이즈 비율) - mean, std
4. Spectral Bandwidth (주파수 대역폭) - mean, std
5. Zero Crossing Rate (신호 변화율) - mean, std
6. RMS Energy (에너지) - mean, std
7. Spectral Contrast (7 bands) - mean, std (14개)
8. MFCC (13 coefficients) - mean, std (26개)

**모델**:
- MLP: 256 → 128 → 64 → 32 → 4
- Dropout: 0.4, 0.3, 0.2
- StandardScaler 정규화

**결과**:
```
전체 MAE: 0.5564
전체 R²:  -0.1338

컬럼별 결과:
  mid_freq_score  - MAE: 0.4453, R²: 0.0133
  irregularity    - MAE: 0.7565, R²: -0.1596
  low_high_freq   - MAE: 0.4409, R²: -0.1322
  regularity      - MAE: 0.5831, R²: -0.2569
```

**분석**:
- CNN이 자동으로 더 좋은 특징을 학습
- 수작업 특징이 엔진 품질과 직접적 연관성 부족
- 시간 정보(시간에 따른 변화)가 손실됨

**결론**: 기각 - CNN보다 성능 낮음

---

## 근본적 문제 분석

### 1. 데이터 부족
- 567개 샘플은 딥러닝에 매우 적음
- 권장: 최소 2,000~5,000개

### 2. 점수 분포 편향
```
overall_score 분포:
  1점 (Poor):    ~0%
  2점:           ~0%
  3점 (Normal): ~17%
  4점:          ~40%
  5점 (Good):   ~43%
```
- 대부분 "좋음" 등급에 집중
- 모델이 "나쁨"을 학습할 데이터 없음

### 3. 레이블 품질 불확실
- 4개 점수가 실제로 오디오와 연관되는지 불확실
- 평가 기준의 일관성 검증 필요

### 4. 오디오 파일 품질
- 5개 MP3 파일 손상
- 일부 파일 로드 실패

---

## 생성된 파일

### 학습 스크립트
- `analysis/train.py`: 기본 학습 (Simple CNN, Dual+CBAM, ResNet50)
- `analysis/train_classifier.py`: 분류 모델 학습
- `analysis/train_augmented.py`: 데이터 증강 학습
- `analysis/train_spectral.py`: Spectral Features 학습

### 모델 파일
- `analysis/models/simple_final.keras`: 기준 모델 (최고 성능)
- `analysis/models/mel_fmax4000/simple_final.keras`: Mel Spectrogram 모델
- `analysis/models/classifier_3class/classifier_final.keras`: 분류 모델
- `analysis/models/augmented_x2/simple_final.keras`: 증강 모델
- `analysis/models/spectral/spectral_final.keras`: Spectral Features 모델

### 수정된 파일
- `analysis/audio_preprocessing.py`: Mel Spectrogram 지원 추가
- `analysis/data_loader.py`: use_mel, fmax 파라미터 추가

---

## 권장 사항

### 단기 (현재 가능)
1. **현재 모델 유지**: Simple CNN + STFT (MAE=0.54)
2. **손상 파일 정리**: 5개 손상 MP3 제거/교체
3. **하이퍼파라미터 튜닝**: 학습률, 배치 크기 등 미세 조정

### 중기 (데이터 확보 후)
1. **더 많은 데이터 수집**: 최소 2,000개 이상
2. **클래스 균형 확보**: Poor/Normal/Good 비율 조정
3. **레이블 검증**: 전문가 검수

### 장기 (연구 방향)
1. **Transfer Learning**: 대규모 오디오 데이터셋으로 사전학습
2. **Self-Supervised Learning**: 레이블 없이 특징 학습
3. **Multi-Task Learning**: 관련 태스크 동시 학습

---

## 결론

**4가지 개선 시도 모두 기준 모델(Simple CNN + STFT)을 능가하지 못했습니다.**

현재 성능(MAE=0.54)은 1~5점 척도에서 약 ±0.5점 오차입니다. 이는 실용적 사용에는 부족하며, 근본적인 해결을 위해서는 **더 많은 데이터**와 **클래스 균형**이 필요합니다.

---

## 중요 발견: 누락된 점수 파라미터

### 문제 발견

점수 관계 분석 중 `overall_score`가 4개 점수의 평균과 일치하지 않는 것을 발견했습니다.

**원래 5개 점수 파라미터**:
1. `mid_freq_score` (point1) - 중주파 대역 ✅ 수집됨
2. `low_high_freq` (point2) - 저/고주파 대역 ✅ 수집됨
3. `audable_range_score` (point3) - 가청 대역 ❌ **누락됨**
4. `regularity` (point4) - 규칙성 ✅ 수집됨
5. `irregularity` (point5) - 불규칙성 ✅ 수집됨

**공식**: `overall_score = (5개 점수 합계) / 5` (단순 평균)

### 분석 결과

누락된 `audable_range_score`를 역산한 결과:
- 추정값 평균: ~4.95 (거의 모든 샘플이 5점)
- 이로 인해 기존 4개 점수로는 overall_score 예측에 offset이 발생

### 해결 방안

1. **crawler_config.json 수정**: `audable_range_score` 셀렉터 추가 ✅ 완료
2. **데이터 재수집**: 새로운 크롤링 실행 필요
3. **모델 재학습**: 5개 점수로 학습 파이프라인 업데이트

### 추가된 설정 (crawler_config.json)

```json
"audable_range_score": {
  "type": "xpath",
  "selector": "//div[@id='experienceCont3']//p[@class='point point3']//span[@data-ref='enginePoint3']",
  "extract_method": "text",
  "clean_regex": "[^0-9.]"
}
```

### 크롤링 테스트 결과 (2026-01-10)

테스트 차량: `HIG251121021317`

| 점수 | 웹 크롤링 값 | CSV 기존 값 |
|------|-------------|------------|
| mid_freq_score (point1) | 3.0 | 3.0 |
| low_high_freq (point2) | 4.0 | 4.0 |
| **audable_range_score (point3)** | **5.0** | **누락** |
| regularity (point4) | 3.0 | 3.0 |
| irregularity (point5) | 3.0 | 3.0 |
| **overall_score** | **3.6** | **3.6** |

**공식 검증:**
- 5개 점수 합계: 3.0 + 4.0 + 5.0 + 3.0 + 3.0 = **18.00**
- 평균 (합/5): 18.00 / 5 = **3.60**
- 실제 overall_score: **3.60**
- 결과: **정확히 일치!**

---

## overall_score 계산 공식

### 최종 공식

```
overall_score = (mid_freq_score + low_high_freq + audable_range_score + regularity + irregularity) / 5
```

**즉, 5개 개별 점수의 단순 평균 (가중치 모두 동일: 1/5 = 0.2)**

### 가중치 정보

| 점수 | 가중치 | 비율 |
|------|--------|------|
| mid_freq_score (중주파 대역) | 1/5 | 20% |
| low_high_freq (저/고주파 대역) | 1/5 | 20% |
| audable_range_score (가청 대역) | 1/5 | 20% |
| regularity (규칙성) | 1/5 | 20% |
| irregularity (불규칙성) | 1/5 | 20% |
| **합계** | **5/5** | **100%** |

### audable_range_score 특성

- 대부분의 샘플에서 **5점** (만점)
- 추정 평균: ~4.95점
- 분산이 매우 작음 → 모델 학습에 기여도 낮음
- 실질적으로 overall_score에 **+1점 상수** 역할

### 기존 4개 점수만으로의 예측 문제

기존 CSV에는 `audable_range_score`가 누락되어 있어:
- 4개 점수 합계: 13.00 (예시)
- 누락된 점수 역산: 3.6 × 5 - 13.0 = **5.0**
- 이로 인해 선형 회귀 시 절편(intercept)에 offset 발생

---

## 실험 환경

- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **Python**: 3.x
- **TensorFlow/Keras**: 최신 버전
- **librosa**: 오디오 처리
- **실험 일자**: 2026-01-10
