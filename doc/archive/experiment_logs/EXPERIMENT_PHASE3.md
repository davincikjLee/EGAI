# EGAI 실험 Phase 3 결과 보고서

## 실험 개요

**목적**: 논문 기반 모델 개선 (CBAM Attention)
**기간**: 2026-01-12
**Baseline MAE**: 0.5635 ± 0.0948 (Simple CNN, 5-Fold CV)

## 실험 결과 요약

| 실험 | 모델 | 데이터 | MAE | STD | Baseline 대비 | 결과 |
|------|------|--------|-----|-----|--------------|------|
| **exp_005** | **Simple+CBAM** | **전체 (594)** | **0.4819** | **0.0225** | **-14.5%** | **최고** |
| exp_006 | Channel Concat+CBAM | 전체 (594) | 0.4886 | 0.0515 | -13.3% | 양호 |
| exp_007 | Simple+CBAM | 가솔린 (555) | 0.4884 | 0.0301 | -13.3% | 양호 |

## 상세 결과

### exp_005: Simple CNN + CBAM (최고 성능)

**구조**: Simple CNN 4블록 + CBAM Attention
**파라미터**: 455,853개

| 컬럼 | MAE | R² |
|------|-----|-----|
| low_high_freq | 0.4564 | 0.0169 |
| mid_freq_score | 0.4811 | -0.2193 |
| regularity | 0.5505 | -0.0252 |
| irregularity | 0.7127 | -0.1253 |
| audable_range_score | 0.2089 | -0.7597 |

**소요 시간**: 34.4분

### exp_006: Channel Concatenation + CBAM

**구조**: Full+Percussive 채널 결합 + CBAM (논문 Fig.6 방식)

| 컬럼 | MAE | R² |
|------|-----|-----|
| low_high_freq | 0.4503 | -0.0061 |
| mid_freq_score | 0.4768 | -0.0024 |
| regularity | 0.5965 | -0.3365 |
| irregularity | 0.7233 | -0.0645 |
| audable_range_score | 0.1964 | -0.7761 |

**소요 시간**: 47.2분

### exp_007: 가솔린 전용 Simple+CBAM

**데이터**: 가솔린 555개 (디젤 39개 제외)

| 컬럼 | MAE | R² |
|------|-----|-----|
| low_high_freq | 0.4930 | -0.1234 |
| mid_freq_score | 0.5033 | -0.0098 |
| regularity | 0.5455 | -0.0900 |
| irregularity | 0.7185 | -0.1200 |
| audable_range_score | 0.1818 | -0.5793 |

**소요 시간**: 30.3분

## 핵심 발견

### 1. CBAM이 효과적
- Simple CNN + CBAM: **14.5% 개선** (0.5635 → 0.4819)
- 논문에서 회귀 태스크 +8% 향상과 일치하는 결과

### 2. 채널 결합은 미미한 효과
- 논문 Fig.6 방식 (채널 결합) 적용했으나 Simple+CBAM 대비 효과 없음
- 복잡도 증가 대비 성능 향상 없음

### 3. 연료 타입 분리 효과 없음
- 가솔린 전용 모델이 전체 데이터 모델보다 약간 낮은 성능
- 디젤 39개로는 별도 학습 불가

### 4. 예측 난이도 순위 (MAE 기준)
1. audable_range_score: 0.21 (쉬움 - 94%가 5점)
2. low_high_freq: 0.46
3. mid_freq_score: 0.48
4. regularity: 0.55
5. irregularity: 0.71 (어려움)

## 기술적 개선

### 캐싱 시스템 활성화
- `--cache` 옵션으로 스펙트로그램 캐싱
- 첫 실행 후 전처리 시간 대폭 단축 (15분 → 수 초)

### 연료 타입 필터링 추가
- `--fuel_type` 옵션으로 가솔린/디젤 분리 학습 가능
- data_loader.py의 `load_metadata()` 함수 확장

### MLOps 피드백 추적 시스템
- 에이전트 제안 추적 및 성공률 계산
- 신뢰도 기반 우선순위 결정

## 권장 사항

### 최적 모델 설정
```bash
uv run python analysis/train_cv.py \
    --model simple_cbam \
    --folds 5 \
    --epochs 100 \
    --cache \
    --exp_name production
```

### 추가 개선 방향
1. **데이터 증강**: Mixup, SpecAugment 적용
2. **앙상블**: Simple+CBAM과 Channel Concat 앙상블
3. **하이퍼파라미터 튜닝**: Learning Rate, Batch Size 최적화
4. **고장 데이터 수집**: 현재 정상 데이터만 있어 진정한 진단 불가

## 결론

논문 기반 CBAM Attention 적용으로 **14.5% 성능 향상** 달성.
Simple CNN + CBAM이 현재 최적 모델이며, 추가 복잡도(채널 결합)는 효과가 없었음.

**최종 MAE: 0.4819 ± 0.0225** (5-Fold CV)
