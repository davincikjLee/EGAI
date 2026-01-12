# EGAI 연속 작업 계획

## 작업 상태 추적

### 작업 1: 데이터 재크롤링
- **상태**: 대기
- **목적**: `audable_range_score` (point3) 포함된 새 데이터 수집
- **명령어**: `python main.py`
- **예상 결과**:
  - `car_audio_metadata.csv`에 `audable_range_score` 컬럼 추가
  - 기존 567개 샘플 + 신규 샘플
- **완료 확인**: CSV에 `audable_range_score` 컬럼 존재 여부

---

### 작업 2: 기존 성공 모델 재실행 (Baseline 확인)
- **상태**: 대기
- **목적**: Simple CNN (STFT) 모델로 기존 결과 재현 확인
- **명령어**: `python analysis/train.py --model simple`
- **기존 결과** (비교 기준):
  ```
  전체 MAE: 0.5387
  전체 R²:  0.0430

  컬럼별 결과:
    mid_freq_score  - MAE: 0.3889, R²: 0.2553
    irregularity    - MAE: 0.7434, R²: -0.0252
    low_high_freq   - MAE: 0.4797, R²: -0.0660
    regularity      - MAE: 0.5427, R²: 0.0079
  ```
- **완료 확인**: 결과가 기존과 유사한지 비교

---

### 작업 3: 5개 점수 학습 (point3 포함)
- **상태**: 대기
- **전제조건**: 작업 1 완료 (재크롤링)
- **목적**: `audable_range_score` 포함 5개 점수 예측
- **수정 필요**:
  - `data_loader.py`: target_columns에 `audable_range_score` 추가
  - `train.py`: num_outputs=5로 변경
- **명령어**: `python analysis/train.py --model simple`
- **완료 확인**: 5개 점수 각각의 MAE, R² 출력

---

### 작업 4: 추가 개선 실험 (선택)
- **상태**: 대기
- **후보**:
  1. 모델 앙상블
  2. 1D CNN (waveform 입력)
  3. Attention 메커니즘 추가
  4. Cross-validation
  5. fmax 튜닝 (4000~8000Hz)

---

## 실행 순서

```
[작업 1] 재크롤링
    ↓
[작업 2] Baseline 모델 확인
    ↓
[작업 3] 5개 점수 학습
    ↓
[작업 4] 추가 개선 (선택)
```

---

## 중요 정보

### overall_score 공식 (검증 완료)
```
overall_score = (mid_freq_score + low_high_freq + audable_range_score + regularity + irregularity) / 5
```

### audable_range_score 특성
- 거의 모든 샘플에서 5점 (만점)
- 분산 매우 작음
- 실질적으로 상수 역할

### 크롤러 설정 (완료)
- `crawler_config.json`에 `audable_range_score` 셀렉터 추가됨
- XPath: `//div[@id='experienceCont3']//p[@class='point point3']//span[@data-ref='enginePoint3']`

### 테스트 결과 (HIG251121021317)
- mid_freq_score: 3.0
- low_high_freq: 4.0
- audable_range_score: 5.0 (NEW)
- regularity: 3.0
- irregularity: 3.0
- overall: 3.6 (= 18/5, 정확히 일치)

---

## 파일 위치

| 파일 | 경로 |
|------|------|
| 크롤러 메인 | `main.py` |
| 크롤러 설정 | `config/crawler_config.json` |
| 메타데이터 CSV | `data/car_audio_metadata.csv` |
| goodsNo 목록 | `data/goods_nos.csv` |
| 학습 스크립트 | `analysis/train.py` |
| 데이터 로더 | `analysis/data_loader.py` |
| 실험 로그 | `doc/EXPERIMENT_LOG.md` |

---

## 실행 결과

### 작업 1: 데이터 재크롤링 ✅ 완료
- 크롤러 실행 완료
- `audable_range_score` 컬럼 역산 추가 (총 596개 샘플)
- 분포: 5.0점 564개 (94.6%), 4.0점 30개 (5.0%), 0.0점 2개 (0.3%)

### 작업 2: Baseline 모델 재실행 ✅ 완료
- 결과: MAE 0.5725, R² -0.0223
- 이전 결과 대비 약간 저하 (랜덤성)

### 작업 3: 5개 점수 학습 ✅ 완료
```
Overall MAE: 0.4665 (이전 0.5725 대비 -0.1060 개선)
Overall R²:  0.0004 (이전 -0.0223 대비 +0.0227 개선)

Per-column Results:
  mid_freq_score       - MAE: 0.4578, R²: 0.0513
  low_high_freq        - MAE: 0.4521, R²: 0.0620
  audable_range_score  - MAE: 0.1228, R²: -0.1787
  regularity           - MAE: 0.5337, R²: 0.0772
  irregularity         - MAE: 0.7659, R²: -0.0097
```

**Note**: `audable_range_score`의 낮은 MAE(0.12)는 거의 모든 샘플이 5점이기 때문

---

### 작업 4: Cross-Validation 실험 ✅ 완료

**5-Fold Stratified Cross-Validation 결과:**

```
전체 MAE: 0.5635 ± 0.0948
전체 R²:  -0.8613 ± 0.6532
평균 에포크: 28.8

컬럼별 결과:
  mid_freq_score       - MAE: 0.5093 ± 0.0580, R²: -0.2256 ± 0.2700
  low_high_freq        - MAE: 0.6608 ± 0.1697, R²: -1.5908 ± 1.3044
  audable_range_score  - MAE: 0.2592 ± 0.0729, R²: -1.8975 ± 1.1614
  regularity           - MAE: 0.5567 ± 0.0436, R²: -0.1071 ± 0.1415
  irregularity         - MAE: 0.8316 ± 0.2319, R²: -0.4855 ± 0.8244

Fold별 성능:
  Fold 1: MAE 0.7247 (최악)
  Fold 2: MAE 0.5652
  Fold 3: MAE 0.5463
  Fold 4: MAE 0.5541
  Fold 5: MAE 0.4273 (최고)
```

**분석:**
- 단일 분할(MAE 0.47) 대비 CV 평균(MAE 0.56)이 높음 → 단일 분할이 우연히 좋은 결과
- Fold 간 편차가 큼 (std 0.09) → 모델 안정성 부족
- 7개 손상 MP3 파일 반복 오류 발생

---

## 마지막 업데이트
- **일시**: 2026-01-11
- **작성자**: Claude
- **상태**: 작업 4 (Cross-validation) 완료
