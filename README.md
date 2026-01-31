# EGAI - Engine Grading AI

현대자동차 인증중고차 웹사이트에서 엔진 오디오를 수집하고, 딥러닝 기반으로 엔진 품질을 진단하는 AI 프로젝트입니다.

## 프로젝트 목표

> **"내가 사용할 엔진의 상태를 확인할 수 있는 AI 모델 개발"**

현대차 품질 평가 시스템을 **리버스 엔지니어링**하여, 누구나 스마트폰으로 엔진 상태를 확인할 수 있는 모델을 구축합니다.

## 주요 성과 (2026-01-24)

| 항목 | 값 |
|------|-----|
| **최적 모델** | 4-Channel CBAM + Multi-head |
| **MAE** | **0.3463** (±0.009) |
| **개선율** | +27% (이전 대비) |
| **데이터** | 2,478개 엔진 오디오 |

### 5개 품질 점수 예측 성능

| 점수 | MAE | 난이도 |
|------|-----|--------|
| audible_range | 0.04 | 쉬움 |
| low_high_freq | 0.32 | 보통 |
| mid_freq_score | 0.36 | 보통 |
| regularity | 0.42 | 어려움 |
| irregularity | 0.60 | 어려움 |

## 빠른 시작

### 설치

```bash
pip install -r analysis/requirements.txt
```

### 엔진 품질 예측 (의뢰자용)

```bash
# 단일 파일 분석
py -3.12 scripts/predict.py 엔진소리.mp3
py -3.12 scripts/predict.py 엔진소리.m4a

# 폴더 내 모든 파일 분석
py -3.12 scripts/predict.py sample/
```

> M4A 지원을 위해 `imageio-ffmpeg` 패키지가 자동으로 FFmpeg를 제공합니다.

**출력 예시**:
```
==================================================
  [OK] 분석 결과: 양호
==================================================
  종합 점수: 4.24
  저/고주파 균형: 4.15
  중주파 품질: 4.32
  가청범위 상태: 4.89
  규칙성: 4.01
  불규칙성 없음: 3.82
  VAE 점수: 152.3 (낮을수록 정상)
==================================================
```

**판정 기준**:

- `[OK]` 양호: 종합 4.0 이상 + VAE < 200
- `[--]` 보통: 종합 3.5 이상
- `[NG]` 점검권장: 종합 3.5 미만

### 모델 학습

```bash
# 회귀 모델 학습 (5-Fold CV)
py -3.12 scripts/train.py --model 4channel_cbam --folds 5 --epochs 50

# VAE 이상탐지 모델 학습
py -3.12 scripts/train_anomaly.py --epochs 50 --beta 0.5
```

## 프로젝트 구조

```
EGAI/
├── egai/                    # 핵심 모듈 (Clean Architecture)
│   ├── application/         # 유스케이스 (train, predict)
│   ├── domain/              # 도메인 엔티티, 서비스
│   ├── infrastructure/      # 모델 정의 (CBAM, VAE)
│   ├── preprocessing/       # 오디오 전처리
│   ├── pipelines/           # MLOps 파이프라인
│   └── data/                # 데이터 로더, 크롤러
├── scripts/                 # 실행 스크립트
│   ├── train.py             # 회귀 모델 학습
│   ├── train_anomaly.py     # VAE 이상탐지 학습
│   └── predict.py           # 예측 (의뢰자용)
├── experiments/             # 학습된 모델 저장
├── config/                  # 설정 파일
├── doc/                     # 문서
├── data/                    # 데이터 (git 제외)
└── main.py                  # 크롤러 진입점
```

## 기술 스택

- **딥러닝**: TensorFlow/Keras
- **모델**: CBAM Attention, VAE
- **오디오 처리**: librosa (STFT, HPSS)
- **데이터**: pandas, numpy
- **크롤링**: Selenium, requests

## MLOps 파이프라인

1. **TrainingPipeline**: 5개 품질 점수 예측 (회귀)
2. **AnomalyTrainingPipeline**: OK/NG 판단 (VAE 이상탐지)

## 핵심 발견

### 성공한 것

- **CBAM Attention**: 성능 향상에 핵심 기여
- **4채널 스펙트로그램**: Full + Percussive + Diff + Variance
- **HPSS 분리**: 충격음 성분 분리로 정확도 향상

### 현재 한계

- 학습 데이터가 100% 정상 차량
- 이상 차량 데이터 수집 후 OK/NG 분류 모델 개발 예정

## 문서

- [CLAUDE.md](CLAUDE.md) - 프로젝트 개요 및 현황
- [doc/README.md](doc/README.md) - 문서 색인
- [doc/IDENTIFIED_ISSUES.md](doc/IDENTIFIED_ISSUES.md) - 식별된 이슈

## 라이선스

연구/교육 목적 프로젝트
