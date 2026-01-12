# 크롤러 모듈 문서 (src/)

## 개요

현대자동차 인증 중고차(https://certified.hyundai.com) 웹사이트에서 차량 정보와 엔진 오디오를 수집하는 자동화된 크롤러입니다.

## 모듈 구성

### 1. main_crawler.py

**역할**: 전체 크롤링 프로세스를 오케스트레이션하는 메인 컨트롤러

**주요 기능**:
- 연료 필터 적용 (가솔린/디젤)
- "더보기" 버튼 반복 클릭을 통한 전체 목록 로드
- 상세 페이지 순회 및 데이터 추출
- MP3 파일 다운로드 및 메타데이터 저장

**크롤링 플로우**:
```
1. 리스트 페이지 로드
2. 연료 필터 클릭 (가솔린, 디젤)
3. 총 차량 대수 확인
4. "더보기" 버튼 반복 클릭
5. goodsNo 목록 수집
6. 각 상세 페이지 순회:
   - 메타데이터 추출
   - 엔진 오디오 URL 추출 및 다운로드
   - CSV에 저장
```

### 2. web_scraper.py

**역할**: HTTP 요청 및 Selenium 웹드라이버 관리

**주요 클래스**: `WebScraper`

| 메서드 | 설명 |
|--------|------|
| `__init__()` | 설정 초기화, Selenium 드라이버 생성 |
| `_init_selenium_driver()` | ChromeDriver 자동 다운로드/설정 |
| `get_html(url, scroll_limit)` | 페이지 HTML 가져오기 |
| `close()` | 드라이버 종료 |

**특징**:
- `webdriver_manager`를 통한 ChromeDriver 자동 관리
- 재시도 로직 (configurable)
- 동적 페이지 로딩 지원

### 3. page_parser.py

**역할**: HTML 파싱 및 데이터 추출

**주요 클래스**: `PageParser`

**지원하는 추출 방식**:
| extract_method | 설명 |
|----------------|------|
| `text` | 요소의 텍스트 추출 |
| `attribute` | 특정 속성값 추출 |
| `exists` | 요소 존재 여부 (boolean) |
| `count` | 매칭되는 요소 개수 |
| `count_gt_zero` | 요소 내 숫자가 0보다 큰지 |
| `list_key_value` | 리스트에서 키-값 쌍 추출 |

**추출 데이터 예시**:
- 차량명, 연식, 주행거리, 연료타입
- 점검 상태 (오일 교체, 에어컨 필터 등)
- 보증 잔여 기간
- 엔진 오디오 점수 (overall, mid_freq, regularity 등)

### 4. audio_downloader.py

**역할**: MP3 오디오 파일 다운로드

**주요 클래스**: `AudioDownloader`

| 메서드 | 설명 |
|--------|------|
| `download_audio_file(url, path)` | MP3 다운로드 및 저장 |

**특징**:
- 스트리밍 다운로드 (메모리 효율)
- 재시도 로직
- 디렉토리 자동 생성

### 5. data_manager.py

**역할**: 데이터 저장 및 상태 관리

**주요 클래스**: `DataManager`

| 메서드 | 설명 |
|--------|------|
| `load_goods_nos_with_status()` | 수집된 goodsNo 및 상태 로드 |
| `save_goods_nos_with_status()` | goodsNo 상태 저장 |
| `save_metadata_to_csv()` | 메타데이터 CSV 저장 |
| `save_debug_html()` | 디버깅용 HTML 저장 |
| `create_vehicle_asset_dir()` | 차량별 MP3 저장 폴더 생성 |

**저장 파일**:
- `data/goods_nos.csv`: goodsNo, data_collected, mp3_downloaded
- `data/car_audio_metadata.csv`: 전체 메타데이터 (35개 컬럼)

### 6. config_loader.py

**역할**: JSON 설정 파일 로드 및 유효성 검사

**주요 클래스**: `ConfigLoader`

| 메서드 | 설명 |
|--------|------|
| `load_config()` | 설정 파일 로드 |
| `get(key_path, expected_type)` | 점 표기법으로 설정값 접근 |
| `_validate_config()` | 설정 유효성 검사 |

## 설정 파일 (crawler_config.json)

### crawler_settings
```json
{
  "user_agent": "Mozilla/5.0 ...",
  "request_delay_sec": 2,
  "timeout_sec": 30,
  "max_retries": 3,
  "use_selenium": true,
  "selenium_headless": false
}
```

### urls
```json
{
  "base_url": "https://certified.hyundai.com",
  "list_page_pattern": "/p/search/vehicle",
  "detail_page_pattern": "/p/goods/goodsDetail.do?goodsNo={goods_no}"
}
```

### data_selectors
XPath/CSS 셀렉터를 통한 데이터 추출 정의
- 차량 기본 정보
- 점검 상태
- 엔진 오디오 점수

## 사용법

```bash
python main.py
```

## 수집되는 메타데이터 컬럼

| 컬럼명 | 설명 |
|--------|------|
| audio_file_path | MP3 파일 경로 |
| goodsNo | 차량 고유 ID |
| vehicle_name | 차량명 |
| first_registration_date | 최초 등록일 |
| year | 연식 |
| current_mileage_km | 주행거리 |
| fuel_type | 연료 타입 |
| overall_score | 엔진 종합 점수 |
| mid_freq_score | 중주파 점수 |
| low_high_freq | 저/고주파 점수 |
| regularity | 규칙성 점수 |
| irregularity | 불규칙성 점수 |
