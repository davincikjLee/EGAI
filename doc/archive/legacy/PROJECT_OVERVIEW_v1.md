# EGAI (Engine/car audio AI) 프로젝트 개요

## 프로젝트 목적

현대자동차 인증 중고차(Certified Hyundai) 웹사이트에서 차량 정보와 엔진 오디오 데이터를 수집하고, 이를 기반으로 **딥러닝 기반 엔진 고장 진단 AI 모델**을 학습하는 시스템입니다.

## 프로젝트 구성

### 1. 데이터 수집 파이프라인 (`src/`)

웹 크롤링을 통해 차량 메타데이터와 엔진 오디오(MP3) 파일을 자동 수집합니다.

### 2. AI 분석 모듈 (`analysis/`)

수집된 오디오 데이터를 전처리하고, 딥러닝 모델을 학습하여 엔진 상태를 진단합니다.

## 기술 스택

| 분야 | 기술 |
|------|------|
| 웹 크롤링 | Selenium, requests, lxml, BeautifulSoup |
| 오디오 처리 | librosa (STFT, HPSS) |
| 딥러닝 | TensorFlow/Keras |
| 데이터 관리 | pandas, CSV |

## 주요 워크플로우

```
[현대 인증중고차 웹사이트]
         │
         ▼
[웹 크롤링 (Selenium)]
    ├── 차량 목록 수집 (goodsNo)
    ├── 상세 정보 추출 (메타데이터)
    └── 엔진 오디오(MP3) 다운로드
         │
         ▼
[데이터 전처리]
    ├── 오디오 → 스펙트로그램 변환
    ├── HPSS (하모닉/타악음 분리)
    └── 데이터 증강 (Augmentation)
         │
         ▼
[딥러닝 모델 학습]
    ├── CNN + CBAM (Attention)
    └── 이중 입력 (Full + Percussive)
         │
         ▼
[엔진 고장 진단]
```

## 디렉토리 구조

```
EGAI/
├── main.py                  # 메인 진입점
├── config/
│   ├── crawler_config.json  # 크롤러 설정
│   └── default_config.json  # 기본 설정
├── src/                     # 데이터 수집 모듈
│   ├── main_crawler.py      # 메인 크롤러 오케스트레이터
│   ├── web_scraper.py       # HTTP/Selenium 요청 처리
│   ├── page_parser.py       # HTML 파싱 (XPath/CSS)
│   ├── audio_downloader.py  # MP3 다운로드
│   ├── data_manager.py      # CSV/파일 관리
│   └── config_loader.py     # 설정 로드
├── analysis/                # AI 분석 모듈
│   ├── audio_preprocessing.py  # 오디오 → 스펙트로그램
│   ├── AudioAugment.py         # 데이터 증강
│   ├── conv_blocks.py          # CNN 블록
│   ├── attention_modules.py    # CBAM Attention
│   ├── inference.py            # 모델 추론
│   ├── TrainModelTest.py       # 모델 테스트
│   ├── check_data.py           # 데이터 검증
│   └── data prep.py            # 데이터 탐색/시각화
├── data/                    # 수집된 데이터 저장
│   ├── goods_nos.csv        # 수집된 차량 ID 목록
│   ├── car_audio_metadata.csv  # 차량 메타데이터
│   ├── vehicle_assets/      # MP3 파일 저장
│   └── debug_html/          # 디버깅용 HTML
└── doc/                     # 문서
```
