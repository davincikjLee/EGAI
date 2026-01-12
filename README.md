# EGAI - Engine Grading AI

현대자동차 인증중고차 웹사이트에서 엔진 오디오를 수집하고, 딥러닝 기반으로 엔진 품질을 진단하는 AI 프로젝트입니다.

## 프로젝트 목표

> **"내가 사용할 엔진의 상태를 확인할 수 있는 AI 모델 개발"**

현대차 품질 평가 시스템을 **리버스 엔지니어링**하여, 누구나 스마트폰으로 엔진 상태를 확인할 수 있는 모델을 구축합니다.

## 주요 성과

| 항목 | 값 |
|------|-----|
| **최적 모델** | Simple CNN + CBAM |
| **MAE** | 0.4819 (±0.5점 오차, 1~5점 척도) |
| **개선율** | +14.5% (Baseline 대비) |
| **데이터** | 594개 엔진 오디오 |

### 5개 품질 점수 예측 성능

| 점수 | MAE | 난이도 |
|------|-----|--------|
| audable_range | 0.21 | 쉬움 |
| low_high_freq | 0.46 | 보통 |
| mid_freq_score | 0.48 | 보통 |
| regularity | 0.55 | 어려움 |
| irregularity | 0.71 | 어려움 |

## 프로젝트 구조

```
EGAI/
├── analysis/               # AI 분석 모듈
│   ├── models.py           # 모델 정의 (Simple CNN, CBAM)
│   ├── audio_preprocessing.py  # 전처리 (STFT + HPSS)
│   ├── data_loader.py      # 데이터 로더
│   ├── train_cv.py         # 학습 (5-Fold CV)
│   └── inference.py        # 추론
├── src/                    # 크롤러 모듈
│   ├── main_crawler.py     # 메인 크롤러
│   └── audio_downloader.py # MP3 다운로드
├── config/                 # 설정 파일
├── doc/                    # 문서
└── main.py                 # 진입점
```

## 빠른 시작

### 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip
pip install -r requirements.txt
```

### 모델 학습

```bash
# 최적 모델 학습 (Simple CNN + CBAM, 5-Fold CV)
uv run python analysis/train_cv.py \
    --model simple_cbam \
    --folds 5 \
    --epochs 100 \
    --cache \
    --exp_name production
```

### 추론

```bash
uv run python analysis/inference.py \
    --audio_path "path/to/engine.mp3" \
    --model_path "models/simple_cbam.h5"
```

## 기술 스택

- **딥러닝**: TensorFlow/Keras
- **오디오 처리**: librosa (STFT, HPSS)
- **데이터**: pandas, numpy
- **크롤링**: Selenium, requests

## 핵심 발견

### 성공한 것

- **CBAM Attention**: +14.5% 성능 향상
- **STFT + HPSS**: Mel Spectrogram 대비 우수
- **GlobalAveragePooling**: 과적합 방지

### 실패한 것 (교훈)

- **물리량 특징 추가**: 과적합으로 -40% 악화
- **모듈레이션 특징**: 벡터 결합 방식의 한계
- **데이터 증강**: 엔진음에 효과 없음

## 문서

- [REVERSE_ENGINEERING_RESULTS.md](doc/REVERSE_ENGINEERING_RESULTS.md) - 리버스 엔지니어링 최종 결과
- [EXPERIMENT_PHASE3.md](doc/EXPERIMENT_PHASE3.md) - Phase 3 실험 결과
- [EXPERT_PANEL_DISCUSSION_V2.md](doc/EXPERT_PANEL_DISCUSSION_V2.md) - 전문가 패널 토론
- [KEY_FINDINGS.md](doc/KEY_FINDINGS.md) - 핵심 발견 사항

## 라이선스

연구/교육 목적 프로젝트
