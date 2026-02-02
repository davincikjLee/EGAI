# EGAI 프로젝트

## 프로젝트 개요

현대자동차 인증 중고차 웹사이트에서 차량 정보와 엔진 오디오를 수집하고, 딥러닝 기반 엔진 고장 진단 AI를 개발하는 프로젝트입니다.

---

## 현재 진행 상황 (2026-02-02)

### 3개 모델 시스템 완성 (2026-02-02)

| 모델 | 목적 | 성능 | 상태 |
|------|------|------|------|
| **Regression** | 품질 점수 예측 (5개) | MAE 0.35 | ✅ 완료 |
| **Binary** | 가솔린/디젤 분류 | F1 0.68, AUC 0.95 | ✅ 완료 |
| **VAE** | 이상 탐지 | Score 154.66 | ✅ 완료 |

### 통합 XAI 시각화 완성 (2026-02-02)

3개 모델의 분석 결과를 한 이미지로 시각화하는 기능 구현:
- Regression/Binary: Grad-CAM (주목 영역)
- VAE: Reconstruction Error (이상 영역)
- 상세: [고객 기술 보고서](docs/CUSTOMER_TECHNICAL_REPORT_20260202.md)

### NG 데이터 요구량 분석

| 목적 | 최소 | 권장 | 이상적 |
|------|------|------|--------|
| VAE 임계값 검증 | 100개 | 200개 | 500개 |
| OK/NG 분류 학습 | 200개 | 400개 | 800개 |

**현재 상태**: NG 데이터 없음 → 의뢰자 수집 요청 중

### 데이터 현황

- **전체 데이터**: 2,605개 (가솔린 2,402 + 디젤 197)
- **유효 데이터**: 2,599개 (손상된 오디오 제외)
- **저장 위치**: `data/vehicle_assets/`, `data/car_audio_metadata.csv`

### 데이터 품질 관리 (2026-01-23)

- `goods_nos.csv`에 `exclude_reason` 컬럼 추가
- 문제 데이터 7건 마킹 (엔진 오디오 섹션 없는 차량)
- 크롤러에 자동 제외 로직 적용

---

## MLOps 파이프라인 (2개)

### 1. TrainingPipeline (회귀 모델)

**목적**: 엔진 품질 5개 점수 예측

**파일**: `egai/pipelines/training.py`

**스크립트**: `scripts/train.py` (또는 `analysis/train_cv.py`)

```bash
# 실행 방법
py -3.12 scripts/train.py --model 4channel_cbam --folds 5 --epochs 50
```

**모델**: Multi-head CBAM (4채널 입력)
- 입력: 128x128x4 스펙트로그램
- 출력: 5개 품질 점수
- **성능: MAE 0.3463 (±0.009)** ← 2026-01-24 재학습

**개별 점수 MAE** (2026-01-24):

| 점수 | MAE |
| ---- | ---- |
| low_high_freq | 0.3174 |
| mid_freq_score | 0.3577 |
| audible_range_score | 0.0361 |
| regularity | 0.4201 |
| irregularity | 0.6000 |

**모델 저장 위치**: `experiments/best_model_4channel_cbam_1.keras`

---

### 2. AnomalyTrainingPipeline (이상탐지 VAE)

**목적**: OK/NG 판단 (정상 vs 이상)

**파일**: `egai/pipelines/anomaly_training.py`

**스크립트**: `scripts/train_anomaly.py`

```bash
# 실행 방법
py -3.12 scripts/train_anomaly.py --epochs 50 --beta 0.5 --min-score 4.0
```

**모델**: VAE (Variational Autoencoder)
- 입력: 128x128x4 스펙트로그램
- 잠재 공간: 128차원
- 이상 점수: 재구성 오차 + β×KL Divergence

**최근 학습 결과** (2026-01-24):
- 데이터: 2,478개 → ~1,900개 (4점 이상 필터링)
- Final Loss: 155.12
- Reconstruction: 151.76
- KL Divergence: 6.70
- VAE Score: 154.66 ± 25.45
- 저장 위치: `experiments/20260124_013112_530a70/`

---

## 추론 시간 벤치마크

| 단계 | 시간 | 비고 |
|------|------|------|
| 오디오 처리 (MP3→스펙트로그램) | 785ms | librosa (병목) |
| VAE 추론 | 29ms | 이상탐지 |
| Regression 추론 | 43ms | 품질예측 |
| MC Dropout (30샘플) | 1,719ms | 불확실성 |
| **전체 파이프라인** | **843ms** | 종합 |

---

## 주요 구성 요소

### 모델 (`egai/infrastructure/models.py`)

| 모델 | 용도 | 입력 |
|------|------|------|
| `simple_cbam` | 기본 회귀 | (128,128,1) |
| `4channel_cbam` | 4채널 회귀 | (128,128,4) |
| `multihead_cbam` | 멀티헤드 회귀 | (128,128,4) |
| `vae` | 이상탐지 | (128,128,4) |
| `autoencoder` | 기본 AE | (128,128,4) |

### 서비스 (`egai/domain/services.py`)

- `ScoreInterpreter`: 품질 점수 해석
- `AnomalyDetector`: 이상탐지 앙상블 (VAE + MC Dropout + 점수 불일치)

### 평가 스크립트

- `scripts/evaluate_vae.py`: VAE 학습 품질 평가
- `scripts/benchmark_inference.py`: 추론 시간 측정

---

## Phase 3 결과: 샘플 테스트 (2026-01-24)

### 테스트 결과

| 차종 | 파일 수 | 평균 Overall | 의도된 라벨 | 실제 판정 |
|------|---------|-------------|-------------|-----------|
| 디올뉴팰리세이드 | 4개 | 4.24 | 정상 | ✅ 정상 |
| **제네시스** | 4개 | **4.35** | **이상** | **❌ 정상** |
| 코나 | 5개 | 4.16 | 정상 | ✅ 정상 |

### 핵심 문제

```
⚠️ 이상 차량(제네시스)이 정상으로 분류됨

원인:
├─ 학습 데이터 2,000개가 모두 정상 차량
├─ 모델이 "정상 중 순위 매기기"만 학습
└─ 이상 데이터 없이는 이상 탐지 불가
```

### 노이즈 필터링

- **구현 완료**: `scripts/noise_filter_test.py`
- **효과**: 배경 소음 평균 60% 감소
- **결과**: 필터링 후에도 제네시스 여전히 정상 범위 (4.0+)

### 다음 단계 (의뢰자 협의 필요)

| 항목 | 필요 사항 |
|------|-----------|
| 이상 데이터 | **50~100개 수집 필요** |
| 제네시스 상태 | 이상 유형 확인 필요 (노킹, 진동 등) |
| OK/NG 모델 | 이상 데이터 확보 후 개발 가능 |

### 현재 상태: 의뢰자 피드백 대기 중 (2026-01-24 14:10)

**상세 결과**: [doc/research/SAMPLE_TEST_REPORT.md](doc/research/SAMPLE_TEST_REPORT.md)

---

## 프로젝트 구조

```
EGAI/
├── main.py                     # 크롤러 진입점
├── config/                     # 설정 파일
│   ├── crawler_config.json     # 크롤러 설정 (셀렉터, URL 패턴)
│   └── default_config.json     # 기본 설정
├── src/                        # 데이터 수집 모듈
│   ├── main_crawler.py         # 메인 크롤러 (오케스트레이터)
│   ├── web_scraper.py          # Selenium/requests 웹 스크래핑
│   ├── page_parser.py          # HTML 파싱 (XPath/CSS)
│   ├── audio_downloader.py     # MP3 다운로드
│   ├── data_manager.py         # CSV/파일 관리
│   └── config_loader.py        # 설정 로더
├── analysis/                   # AI 분석 모듈
│   ├── audio_preprocessing.py  # 오디오 → 스펙트로그램 변환
│   ├── AudioAugment.py         # 데이터 증강
│   ├── conv_blocks.py          # CNN 블록
│   ├── attention_modules.py    # CBAM Attention
│   ├── inference.py            # 모델 추론
│   ├── TrainModelTest.py       # 모델 테스트
│   ├── check_data.py           # 데이터 검증
│   └── data prep.py            # 데이터 시각화
├── data/                       # 수집 데이터 저장
└── doc/                        # 상세 문서
```

## 핵심 기능

### 1. 데이터 수집 (src/)
- **대상**: https://certified.hyundai.com
- **수집 항목**: 차량 메타데이터 35개 컬럼 + 엔진 오디오(MP3)
- **기술**: Selenium + lxml/BeautifulSoup

### 2. AI 분석 (analysis/)
- **입력**: 엔진 오디오 → 스펙트로그램 (Full + Percussive)
- **모델**: 이중 입력 CNN + CBAM Attention
- **출력**: 엔진 상태 분류

## 주요 명령어

### 엔진 품질 예측 (의뢰자용)

```bash
# 단일 파일 분석
py -3.12 scripts/predict.py 엔진소리.mp3
py -3.12 scripts/predict.py 엔진소리.m4a

# 폴더 내 모든 파일 분석
py -3.12 scripts/predict.py sample/

# Grad-CAM 시각화 포함 (XAI)
py -3.12 scripts/predict.py 엔진소리.m4a --visualize
```

> M4A 지원을 위해 `imageio-ffmpeg` 패키지가 자동으로 FFmpeg를 제공합니다.

### Grad-CAM 시각화 (설명가능한 AI)

CNN 모델이 스펙트로그램의 어떤 영역을 보고 판단했는지 시각화합니다.

```bash
# 단일 파일 시각화
py -3.12 scripts/visualize_gradcam.py sample/제네시스_엔진음.m4a

# 여러 파일 비교 시각화
py -3.12 scripts/visualize_gradcam.py sample/ --compare
```

**출력**: `{파일명}.gradcam.png` 이미지 생성

**구성**:

- 원본 스펙트로그램 (입력)
- Grad-CAM 히트맵 (모델 주목 영역)
- 오버레이 (빨간색 = 중요, 파란색 = 덜 중요)

### 통합 XAI 시각화 (3개 모델)

3개 모델(Regression, Binary Classification, VAE)의 분석 결과를 하나의 이미지로 시각화합니다.

```bash
# 단일 파일 통합 시각화
py -3.12 scripts/visualize_unified.py sample/엔진소리.m4a

# 폴더 내 모든 파일
py -3.12 scripts/visualize_unified.py sample/

# 보고서 생성 없이
py -3.12 scripts/visualize_unified.py sample/엔진소리.m4a --no-report
```

**출력**: `{파일명}.unified_gradcam.png` + `{파일명}.unified_gradcam.md`

**구성** (각 행별 3개 패널):
- **Regression**: 품질 점수 예측 시 주목하는 영역 (Grad-CAM)
- **Binary**: 가솔린/디젤 분류 시 주목하는 영역 (Grad-CAM)
- **VAE**: 원본 → 재구성 → 오차 맵 (Reconstruction Error)

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

**참고**: 스마트폰 녹음(M4A)은 학습 데이터(웹사이트 MP3)와 녹음 환경이 달라 VAE 점수가 높게 나올 수 있습니다.

### 기타 명령어

```bash
# 크롤러 실행
python main.py

# 모델 학습
py -3.12 scripts/train.py --model 4channel_cbam --folds 5 --epochs 50

# VAE 학습
py -3.12 scripts/train_anomaly.py --epochs 50 --beta 0.5
```

## 기술 스택

- **웹 크롤링**: Selenium, requests, lxml, BeautifulSoup
- **오디오 처리**: librosa (STFT, HPSS)
- **딥러닝**: TensorFlow/Keras
- **데이터**: pandas, numpy

## 설정 파일

`config/crawler_config.json`에서 크롤링 설정 변경 가능:
- `crawler_settings`: 요청 딜레이, 타임아웃, Selenium 옵션
- `urls`: 대상 URL 패턴
- `data_selectors`: XPath/CSS 셀렉터 정의

## 데이터 흐름

```
웹사이트 → 크롤링 → CSV/MP3 저장 → 스펙트로그램 변환 → 모델 학습 → 엔진 진단
```

## 문서

### 고객용 문서

- [docs/CUSTOMER_TECHNICAL_REPORT_20260202.md](docs/CUSTOMER_TECHNICAL_REPORT_20260202.md): **고객 기술 보고서** (모델 구조, NG 데이터 요구량)

### 기술 문서

- `doc/research/SAMPLE_TEST_REPORT.md`: 샘플 테스트 결과
- `doc/CRAWLER_MODULE.md`: 크롤러 모듈 상세
- `doc/PROJECT_RESPONSE_REPORT.md`: 요구사항 대응 보고서

### 실험 보고서

- `docs/04-report/features/gasoline-diesel-classification.report.md`: 가솔린/디젤 분류 실험

---

## AIL/HIL 설정

> 참조: Obsidian `[[60_Tools/AIL_HIL_Framework]]`

| 설정 | 값 | 비고 |
|------|-----|------|
| **bkit 적용** | 부분 | API/웹 인터페이스만 |
| **AIL 등급** | L1 (Supervised) | 외부 계약, 신뢰 구축 필요 |
| **연속 정확** | 0 | 초기화 |

### 영역별 도구

| 영역 | 도구 | bkit |
|------|------|------|
| 모델 학습 | MLflow, TensorBoard | ❌ |
| 데이터 파이프라인 | 크롤러, librosa | ❌ |
| API 서버 | FastAPI (예정) | ✅ /dynamic |
| 웹 대시보드 | Next.js (예정) | ✅ /dynamic |

### HIL 필수 트리거

- [ ] Gap 분석 < 80%
- [ ] 모델 성능 변경 (MAE, 정확도)
- [ ] 의뢰자 피드백 반영
- [ ] 배포/프로덕션 변경

### AIL 성과 기록

| 날짜 | Check 결과 | HIL 개입 | 비고 |
|------|-----------|---------|------|
| 2026-02-02 | 3개 모델 완성 | - | Binary 추가, 통합 XAI |

---

*최종 수정: 2026-02-02 (3개 모델 시스템 + 통합 XAI 완성)*
