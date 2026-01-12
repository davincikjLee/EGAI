# EGAI 프로젝트

## 프로젝트 개요

현대자동차 인증 중고차 웹사이트에서 차량 정보와 엔진 오디오를 수집하고, 딥러닝 기반 엔진 고장 진단 AI를 개발하는 프로젝트입니다.

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

```bash
# 크롤러 실행
python main.py

# 데이터 확인
python analysis/check_data.py

# 모델 테스트
python analysis/TrainModelTest.py
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

- `doc/PROJECT_OVERVIEW.md`: 프로젝트 전체 개요
- `doc/CRAWLER_MODULE.md`: 크롤러 모듈 상세
- `doc/ANALYSIS_MODULE.md`: AI 분석 모듈 상세
