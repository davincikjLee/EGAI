# EGAI 모델 개선 실험 Phase 2

## 실험 개요

- **목표**: 엔진 오디오 품질 점수 예측 모델 성능 개선
- **기준 모델**: Simple CNN (STFT) - 5-Fold CV MAE: 0.5635 ± 0.0948
- **평가 방식**: 5-Fold Stratified Cross-Validation
- **데이터**: 594개 유효 샘플, 5개 타겟 점수
- **실험 일자**: 2026-01-11

---

## 기준선 (Baseline)

### 현재 최고 성능 (5-Fold CV)
```
전체 MAE: 0.5635 ± 0.0948
전체 R²:  -0.8613 ± 0.6532

컬럼별 결과:
  mid_freq_score       - MAE: 0.5093 ± 0.0580
  low_high_freq        - MAE: 0.6608 ± 0.1697
  audable_range_score  - MAE: 0.2592 ± 0.0729
  regularity           - MAE: 0.5567 ± 0.0436
  irregularity         - MAE: 0.8316 ± 0.2319
```

---

## 실험 5-2: 물리량 특징 추가 (multiinput_physics)

### 가설
오디오에서 추출한 물리량(ZCR, Spectral Centroid 등)이 엔진 상태를 더 직접적으로 반영

### 추가한 특징 (14개)
1. **Zero Crossing Rate (ZCR)**: mean, std - 고주파 노이즈 지표
2. **Spectral Centroid**: mean, std - 주파수 중심
3. **Spectral Flatness**: mean, std - 노이즈 비율 (0=순음, 1=노이즈)
4. **RMS Energy**: mean, std - 전체 에너지
5. **Harmonic Ratio**: 하모닉 에너지 비율 - 규칙성 지표
6. **Percussive Ratio**: 퍼커시브 에너지 비율
7. **Spectral Bandwidth**: mean, std - 주파수 대역폭
8. **Spectral Rolloff**: mean, std - 에너지 집중 주파수

### 결과 (exp_002)
```
전체 MAE: 0.7876 ± 0.2462 (Baseline 대비 +40% 악화)
전체 R²:  -6.2904 ± 6.2305 (Baseline 대비 악화)
평균 에포크: 34.8

컬럼별 결과:
  mid_freq_score       - MAE: 0.6922 ± 0.1683, R²: -1.5403 ± 1.0309
  low_high_freq        - MAE: 0.7982 ± 0.3207, R²: -2.7288 ± 2.4359
  audable_range_score  - MAE: 0.8752 ± 0.4207, R²: -25.5924 ± 27.2500
  regularity           - MAE: 0.6545 ± 0.1325, R²: -0.8606 ± 0.6932
  irregularity         - MAE: 0.9179 ± 0.2308, R²: -0.7301 ± 0.7019
```

### 분석
- **실패 원인**: 14개의 추가 특징이 594개의 적은 샘플에서 오버피팅 유발
- **모델 복잡도 증가**: Audio CNN (512차원) + Physics (64차원) = 576차원
- **특징 간 간섭**: 스펙트로그램과 물리량 특징의 정보 중복

---

## 실험 5-1: 메타데이터 결합 모델 (multiinput_metadata)

### 가설
차량 메타데이터(주행거리, 연식, 배기량 등)가 엔진 상태와 연관되어 예측 성능 향상

### 사용한 메타데이터 (14개)
**수치형 (4개)**: year, current_mileage_km, displacement_cc, seating_capacity
**범주형 (4개 → 원-핫)**: vehicle_type, fuel_type, drivetrain, transmission_type

### 모델 구조
```
[오디오 Full Spec] → CNN → CBAM → 256차원 ─┐
[오디오 Percussive] → CNN → CBAM → 256차원 ─┼→ Concat (512) ─┐
                                              │               ├→ Concat (544) → Dense → 5개 출력
[메타데이터 14개] → Dense(64) → Dense(32) → 32차원 ──────────────┘
```

### 결과 (exp_003)
```
전체 MAE: 0.8512 ± 0.1502 (Baseline 대비 +51% 악화)
전체 R²:  -5.2146 ± 2.1563 (Baseline 대비 악화)
평균 에포크: 37.0

컬럼별 결과:
  mid_freq_score       - MAE: 0.7293 ± 0.1086, R²: -1.8141 ± 0.5958
  low_high_freq        - MAE: 0.8491 ± 0.1794, R²: -2.9161 ± 1.2421
  audable_range_score  - MAE: 0.9438 ± 0.2694, R²: -19.1408 ± 8.5100
  regularity           - MAE: 0.7008 ± 0.1320, R²: -1.0594 ± 0.4504
  irregularity         - MAE: 1.0332 ± 0.1813, R²: -1.1427 ± 0.8077
```

### 분석
- **실패 원인**: 메타데이터 추가로 모델 복잡도 증가
- **Dual Input + Metadata**: 세 개의 입력 브랜치 → 파라미터 수 증가
- **메타데이터-오디오 상관관계**: 주행거리/연식과 엔진 소리의 관계가 명확하지 않음

---

## 실험 5-4: 모듈레이션 특징 추가 (multiinput_modulation)

### 가설
논문 "소음 데이터를 이용한 딥러닝 기반의 차량 진단 기술 개발" (현대자동차, 2023)의 모듈레이션 스펙트럼 기법으로 regularity/irregularity 예측 성능 향상

### 참조 논문 핵심 내용
- **Modulation Spectrum**: RMS envelope의 FFT로 소리 변동성 측정
- **SAML (Sum of Audible Modulation Level)**: 가청 대역 모듈레이션 레벨
- 논문에서 86% 노이즈 레벨 추정 정확도 달성

### 추가한 특징 (20개)
1. **Modulation Spectrum (8개)**: mod_mean, mod_std, mod_max, mod_peak_freq, mod_peak_ratio, mod_low_energy, mod_high_energy, mod_low_high_ratio
2. **Spectral Flux (3개)**: spectral_flux_mean, spectral_flux_std, spectral_flux_max
3. **Onset Detection (6개)**: onset_mean, onset_std, onset_max, onset_count, onset_rate
4. **Autocorrelation (3개)**: autocorr_peak_lag, autocorr_peak_value, autocorr_decay

### 모델 구조
```
[오디오 Full Spec] → CNN → CBAM → 256차원 ─┐
[오디오 Percussive] → CNN → CBAM → 256차원 ─┼→ Concat (512) ─┐
                                              │               ├→ Concat (576) → Dense → 5개 출력
[Modulation 20개] → Dense(64) → Dense(64) → 64차원 ───────────┘
```

### 결과 (exp_004)
```
전체 MAE: 0.7184 ± 0.2241 (Baseline 대비 +27% 악화)
전체 R²:  -4.7085 ± 5.2654 (Baseline 대비 악화)
평균 에포크: 40.0

컬럼별 결과:
  mid_freq_score       - MAE: 0.5690 ± 0.1673, R²: -0.7513 ± 0.7046
  low_high_freq        - MAE: 0.7157 ± 0.2801, R²: -2.2239 ± 2.0310
  audable_range_score  - MAE: 0.7004 ± 0.3998, R²: -17.9135 ± 23.7312
  regularity           - MAE: 0.6355 ± 0.1347, R²: -0.7418 ± 0.4706
  irregularity         - MAE: 0.9715 ± 0.1929, R²: -1.9119 ± 1.3920
```

### 분석
- **실패 원인**: 20개의 추가 특징이 594개의 적은 샘플에서 오버피팅 유발
- **regularity 개선 실패**: MAE 0.5567 → 0.6355 (+14% 악화)
- **irregularity 개선 실패**: MAE 0.8316 → 0.9715 (+17% 악화)
- **논문과의 차이점**: 논문은 더 많은 데이터와 다른 태스크(노이즈 레벨 분류) 사용

---

## 실험 5-3: 데이터 증강 (SpecAugment + Mixup)

### 가설
최신 연구에서 효과 입증된 증강 기법으로 일반화 성능 향상

### 상태
미실행 - Phase 2B에서 진행 예정

---

## 결과 비교표

| 실험 | MAE (mean±std) | R² (mean±std) | Baseline 대비 |
|------|----------------|---------------|--------------|
| **Baseline (Simple CNN)** | **0.5635 ± 0.0948** | **-0.8613 ± 0.6532** | **- (최고)** |
| 5-2: 물리량 추가 | 0.7876 ± 0.2462 | -6.2904 ± 6.2305 | +40% 악화 |
| 5-1: 메타데이터 결합 | 0.8512 ± 0.1502 | -5.2146 ± 2.1563 | +51% 악화 |
| 5-4: 모듈레이션 특징 | 0.7184 ± 0.2241 | -4.7085 ± 5.2654 | +27% 악화 |
| 5-3: 데이터 증강 | - | - | 미실행 |

---

## 주요 발견 및 교훈

### 1. 추가 특징의 역효과
- 594개의 적은 샘플에서 특징 추가는 오히려 성능 저하
- 모델 복잡도 증가로 오버피팅 심화
- **Simple CNN이 현재 데이터셋에서 최적**

### 2. R² 음수의 의미
- 모든 실험에서 R²가 음수 → 평균 예측보다 나쁨
- 원인: 타겟 분포 불균형 (audable_range_score 94%가 5점)
- 연속 회귀로 정수값(1-5) 예측 → 불일치

### 3. 데이터 품질 이슈
- 7개 오디오 파일 손상 (MP4.mp3 형식 문제)
- 일부 MPEG 헤더 파싱 실패

---

## 향후 개선 방향

### 단기 (Quick Wins)
1. **데이터 클리닝**: 손상된 오디오 파일 제외/교체
2. **Sample Weighting**: 클래스 불균형 해결
3. **Regularization 강화**: Dropout 증가, L2 정규화

### 중기
1. **Ordinal Regression**: 정수 점수에 적합한 손실 함수 (Coral Loss)
2. **앙상블**: 여러 Simple CNN 모델 결합
3. **Transfer Learning**: 사전학습된 오디오 모델 활용 (YAMNet, VGGish)

### 장기
1. **데이터 수집 확대**: 594개 → 2000개 이상
2. **도메인 전문가 피드백**: 점수 기준 재검토

---

## 최종 결론

**Simple CNN (Baseline)이 현재 최적 모델**

Phase 2 실험에서 추가 특징(물리량, 메타데이터) 도입은 594개의 적은 데이터에서 오히려 성능 저하를 유발했습니다.

**권장 사항**:
1. 데이터 확대 전까지 Simple CNN 유지
2. Sample Weighting으로 클래스 불균형 해결
3. Ordinal Regression으로 정수 점수 예측 최적화

---

## 실험 환경
- GPU: NVIDIA RTX 4060 (8GB)
- Python: 3.x
- TensorFlow/Keras
- 실험 일자: 2026-01-11

## 실험 로그

| 실험 ID | 모델 | MAE | R² | 상태 |
|---------|------|-----|-----|------|
| exp_001 | test_logger | - | - | 테스트 |
| exp_002 | multiinput_physics | 0.7876 | -6.2904 | 완료 |
| exp_003 | multiinput_metadata | 0.8512 | -5.2146 | 완료 |
| exp_004 | multiinput_modulation | 0.7184 | -4.7085 | 완료 |

실험 상세 결과: `analysis/experiments/` 디렉토리
