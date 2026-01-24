# EGAI 모델 문제점 식별 보고서

**작성일**: 2026-01-14 (Phase 3 결과 추가: 2026-01-24)
**분석 기반**: Multi-head CBAM 모델 (MAE 0.4578), VAE 이상탐지 모델

---

## 0. 이상 데이터 부재 (Critical) ⭐ NEW

### 현상

```text
Phase 3 테스트 결과 (2026-01-24):
- 제네시스 (이상 차량) → 정상으로 분류 ❌
- Overall 점수: 4.35 (정상 범위)
- VAE Score: 5.7 (정상 범위)
```

### 원인

```text
학습 데이터 문제:
├─ 인증중고차 웹사이트 = 100% 정상 차량
├─ 모델이 "정상 중 순위"만 학습
├─ 이상 패턴을 본 적 없음
└─ 구조적으로 이상 탐지 불가
```

### 영향

- **Regression**: 이상 차량도 4.0+ 예측 (정상 범위)
- **VAE**: 녹음 환경 차이만 탐지 (도메인 갭)
- **서비스**: OK/NG 판단 불가

### 해결 방안

| 방안 | 필요 사항 | 효과 |
|------|-----------|------|
| 이상 데이터 수집 | 50~100개 | 근본 해결 |
| OK/NG 이진 분류기 추가 | 이상 데이터 필요 | 2단계 판정 가능 |
| 노이즈 필터링 | ✅ 구현 완료 | 배경 소음 60% 감소 (근본 해결 X) |

**상세**: [doc/research/SAMPLE_TEST_REPORT.md](research/SAMPLE_TEST_REPORT.md)

---

## 1. 출력 범위 제한 없음 (Critical)

### 현상
```python
# 현재 코드
outputs = layers.Dense(num_outputs, activation="linear")
```
- 예측값이 1~5점 범위를 벗어남
- irregularity 2점 → 6점 예측 (범위 초과)

### 영향
- 실제 사용 불가능한 예측값 생성
- MAE 계산 시 과대 오차

### 해결 방안
```python
# 방안 1: Sigmoid 스케일링
x = layers.Dense(num_outputs, activation="sigmoid")(x)
outputs = x * 4 + 1  # 0~1 → 1~5

# 방안 2: 추론 시 클리핑
y_pred = np.clip(model.predict(X), 1.0, 5.0)
```

---

## 2. 끝점(Edge) 오차 문제 (High)

### 현상
| 지표 | 끝점 점수 | MAE | 샘플 수 |
|------|----------|-----|---------|
| irregularity | 2점 | **4.02** | 9개 |
| regularity | 2점 | **2.07** | 25개 |
| mid_freq_score | 2점 | **1.91** | 1개 |
| audible_range_score | 4점 | **1.60** | 30개 |

### 원인
1. **데이터 불균형**: 끝점 데이터가 극소수
2. **평균 회귀**: 모델이 다수 클래스(중간/높은 점수)로 편향
3. **경계 효과**: 끝점은 한 방향으로만 오차 발생

### 해결 방안
- Class-weighted Loss
- Oversampling (SMOTE)
- Focal Loss

---

## 3. 데이터 불균형 (High)

### 현상
```
irregularity 분포:
  2점:    9개 ( 1.6%)  ← 극소수
  3점:   80개 (14.6%)
  4점:  118개 (21.5%)
  5점:  341개 (62.2%)  ← 대다수

audible_range_score 분포:
  4점:   30개 ( 5.5%)
  5점:  518개 (94.5%)  ← 거의 전부
```

### 영향
- 소수 클래스 예측 실패
- 모델이 다수 클래스로 편향

### 해결 방안
```python
# Class weights 계산
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
```

---

## 4. 평균 회귀 현상 (Medium)

### 현상
- 극단값(1점, 2점, 5점) → 평균(3~4점) 방향으로 예측
- 특히 소수 클래스에서 심각

### 예시
```
실제 irregularity 2점 → 예측 6점 (평균 4.4 + overshoot)
실제 regularity 2점 → 예측 4점 (평균 3.5 방향)
```

### 원인
- MAE/MSE 손실 함수가 평균 예측을 선호
- 소수 클래스의 gradient가 다수 클래스에 묻힘

---

## 5. 지표별 난이도 차이 (Medium)

### 분석
| 지표 | 데이터 범위 | 분포 특성 | 예측 난이도 |
|------|------------|----------|------------|
| audible_range_score | 4~5 (1점) | 95% 5점 | 쉬움 |
| low_high_freq | 3~5 (2점) | 50/50 분포 | 중간 |
| mid_freq_score | 2~5 (3점) | 고른 분포 | 중간 |
| regularity | 2~5 (3점) | 고른 분포 | 어려움 |
| irregularity | 1~5 (4점) | 60% 5점 | 어려움 |

### 문제
- 단일 손실 함수로 다양한 난이도 지표 동시 최적화
- 쉬운 지표가 전체 손실 지배

---

## 6. STD 해석 주의 (Low)

### 현상
- Multi-head MAE STD: 0.018 (낮아 보임)
- 실제 Fold별 MAE: 0.43, 0.44, 0.47, 0.47, 0.48
- Fold 3 (0.43)이 이상치로 평균을 낮춤

### 주의점
- STD가 낮다고 안정적인 것 아님
- Fold간 일관성을 별도 확인 필요

---

## 7. 요약: 우선순위별 해결 과제

| 우선순위 | 문제 | 영향도 | 해결 난이도 |
|---------|------|--------|------------|
| **1** | 출력 범위 제한 없음 | Critical | 쉬움 |
| **2** | 끝점 오차 | High | 중간 |
| **3** | 데이터 불균형 | High | 중간 |
| **4** | 평균 회귀 | Medium | 어려움 |
| **5** | 지표별 난이도 차이 | Medium | 어려움 |

---

## 8. 즉시 적용 가능한 개선

### 8.1 출력 범위 제한
```python
# models.py 수정
freq_output = layers.Dense(3, activation="sigmoid", name="freq_output")(freq_x)
freq_output = freq_output * 4 + 1  # 1~5 범위

reg_output = layers.Dense(2, activation="sigmoid", name="reg_output")(reg_x)
reg_output = reg_output * 4 + 1  # 1~5 범위
```

### 8.2 Class-weighted Loss
```python
# training.py 수정
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights, ...)
```

### 8.3 예측 후 클리핑
```python
# inference 시
y_pred = model.predict(X)
y_pred = np.clip(y_pred, 1.0, 5.0)
```

---

## 9. 추가 분석 필요 항목

1. [ ] 각 지표별 독립 모델 vs 통합 모델 비교
2. [ ] Focal Loss 적용 효과 측정
3. [ ] 데이터 증강(오디오 augmentation) 효과
4. [ ] 앙상블 모델 (Baseline + Multi-head) 검토
