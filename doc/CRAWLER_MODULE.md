# 크롤러 모듈 문서 (src/)

## 개요

현대자동차 인증 중고차(https://certified.hyundai.com) 웹사이트에서 차량 정보와 엔진 오디오를 수집하는 자동화된 크롤러입니다.

## 실행 방법

```bash
python main.py
# 또는
py -3.12 main.py
```

## 모듈 구성

```
src/
├── main_crawler.py      # 전체 오케스트레이션 (453줄)
├── web_scraper.py       # Selenium 웹 자동화 (149줄)
├── page_parser.py       # HTML 파싱 (328줄)
├── audio_downloader.py  # MP3 다운로드 (75줄)
├── data_manager.py      # CSV/파일 관리 (135줄)
└── config_loader.py     # 설정 로더 (336줄)
```

---

## 크롤링 프로세스 (5단계)

### [단계 1/5] 페이지 로드 및 연료 필터 선택

```python
# 1. 리스트 페이지 로드
self.scraper.driver.get("https://certified.hyundai.com/p/search/vehicle")

# 2. 가솔린/디젤 필터 클릭 (JavaScript Executor)
self.scraper.driver.execute_script("document.getElementById('가솔린ID').click();")
self.scraper.driver.execute_script("document.getElementById('디젤ID').click();")

# 3. 필터 적용 확인 (WebDriverWait)
WebDriverWait(driver, timeout).until(
    EC.presence_of_element_located((By.XPATH, '필터 셀렉터'))
)
```

### [단계 2/5] 총 차량 대수 수집

```python
total_count_selector = {'type': 'xpath', 'selector': "//em[@id='totalVehicleCnt']"}
total_cars = self.parser.get_total_count(html_content, total_count_selector)
```

### [단계 3/5] "더보기" 버튼 반복 클릭

```python
# 1. 초기 아이템 개수 확인
initial_count = len(driver.find_elements(By.XPATH, "//ul[@id='productList']/li"))

# 2. "더보기" 클릭하여 증가량 측정
more_button.click()
items_per_load = new_count - initial_count  # 예: 20개씩 증가

# 3. 필요한 클릭 횟수 계산 및 반복
clicks_needed = (total_cars - initial_count) // items_per_load
for i in range(clicks_needed):
    more_button = WebDriverWait(...).until(EC.element_to_be_clickable(...))
    more_button.click()
```

### [단계 4/5] goodsNo 수집

```python
# XPath로 모든 상품 링크에서 goodsNo 추출
goods_no_selector = {
    'type': 'xpath',
    'selector': "//ul[@id='productList']/li[@class='type02']/a",
    'extract_attribute': 'href'
}
found_goods_nos = self.parser.parse_list_page_goods_nos(html, goods_no_selector)
# 결과: {'HIG251121021317', 'HDN251218022433', ...}

# 새로 발견된 것만 goods_nos.csv에 저장
newly_discovered = found_goods_nos - existing_goods_nos_set
self.data_manager.save_goods_nos_with_status(goods_nos_df)
```

### [단계 5/5] 상세 페이지 크롤링

```python
for goods_no in unprocessed_goods_nos:
    # 1. 상세 페이지 URL
    detail_url = f"{base_url}/p/goods/goodsDetail.do?goodsNo={goods_no}"

    # 2. HTML 가져오기
    detail_html = self.scraper.get_html(detail_url)

    # 3. 35개 필드 데이터 추출
    extracted_data = self.parser.parse_detail_page(detail_html)

    # 4. MP3 다운로드
    audio_url = extracted_data.get('audio_url_on_page')
    self.audio_downloader.download_audio_file(audio_url, audio_file_path)

    # 5. 메타데이터 CSV 저장
    self.data_manager.save_metadata_to_csv(extracted_data)

    # 6. 상태 업데이트 (매번 저장 - 중단/재개 지원)
    self.data_manager.update_goods_no_status(goods_nos_df, goods_no, 'data_collected', True)
```

---

## 모듈 상세

### 1. main_crawler.py

**역할**: 전체 크롤링 프로세스 오케스트레이션

**주요 메서드**:
| 메서드 | 설명 |
|--------|------|
| `run()` | 전체 크롤링 실행 |
| `_get_list_page_url()` | 리스트 페이지 URL 생성 |

### 2. web_scraper.py

**역할**: HTTP 요청 및 Selenium 웹드라이버 관리

**주요 메서드**:
| 메서드 | 설명 |
|--------|------|
| `__init__()` | 설정 초기화, Selenium 드라이버 생성 |
| `_init_selenium_driver()` | ChromeDriver 자동 다운로드/설정 |
| `get_html(url)` | 페이지 HTML 가져오기 |
| `close()` | 드라이버 종료 |

**특징**:
- `webdriver_manager`를 통한 ChromeDriver 자동 관리
- 재시도 로직 (max_retries 설정 가능)
- 동적 페이지 로딩 지원

### 3. page_parser.py

**역할**: HTML 파싱 및 데이터 추출

**지원하는 추출 방식**:
| extract_method | 설명 |
|----------------|------|
| `text` | 요소의 텍스트 추출 |
| `attribute` | 특정 속성값 추출 |
| `exists` | 요소 존재 여부 (boolean) |
| `count` | 매칭되는 요소 개수 |
| `list_key_value` | 리스트에서 키-값 쌍 추출 |

### 4. audio_downloader.py

**역할**: MP3 오디오 파일 다운로드

**주요 메서드**:
| 메서드 | 설명 |
|--------|------|
| `download_audio_file(url, path)` | MP3 다운로드 및 저장 |

**특징**:
- 스트리밍 다운로드 (메모리 효율)
- 재시도 로직
- 디렉토리 자동 생성

### 5. data_manager.py

**역할**: 데이터 저장 및 상태 관리

**주요 메서드**:
| 메서드 | 설명 |
|--------|------|
| `load_goods_nos_with_status()` | 수집된 goodsNo 및 상태 로드 |
| `save_goods_nos_with_status()` | goodsNo 상태 저장 |
| `update_goods_no_status()` | 개별 항목 상태 업데이트 |
| `save_metadata_to_csv()` | 메타데이터 CSV 저장 |
| `save_debug_html()` | 디버깅용 HTML 저장 |
| `create_vehicle_asset_dir()` | 차량별 MP3 저장 폴더 생성 |

### 6. config_loader.py

**역할**: JSON 설정 파일 로드 및 유효성 검사

**주요 메서드**:
| 메서드 | 설명 |
|--------|------|
| `load_config()` | 설정 파일 로드 |
| `get(key_path, expected_type)` | 점 표기법으로 설정값 접근 |

---

## 설정 파일 (config/crawler_config.json)

### crawler_settings

```json
{
  "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
  "request_delay_sec": 2,
  "timeout_sec": 30,
  "max_retries": 3,
  "retry_delay_sec": 5,
  "use_selenium": true,
  "selenium_driver_path": null,
  "use_auto_driver_download": true,
  "selenium_headless": false,
  "scroll_load_limit": 5
}
```

### urls

```json
{
  "base_url": "https://certified.hyundai.com",
  "list_page_pattern": "/p/search/vehicle",
  "detail_page_pattern": "/p/goods/goodsDetail.do?goodsNo={goods_no}",
  "next_page_selector": {
    "type": "xpath",
    "selector": "//button[@id='btnSeeMore']"
  },
  "total_count_selector": {
    "type": "xpath",
    "selector": "//em[@id='totalVehicleCnt']"
  }
}
```

---

## 저장 파일

### goods_nos.csv

수집 상태 추적 (중단/재개 지원)

```csv
goodsNo,data_collected,mp3_downloaded
HIG251121021317,True,True
HDN251218022433,True,True
HGN251125021481,True,False
```

### car_audio_metadata.csv

전체 메타데이터 (36개 컬럼)

| 컬럼명 | 설명 |
|--------|------|
| audio_file_path | MP3 파일 상대 경로 |
| goodsNo | 차량 고유 ID |
| vehicle_name | 차량명 |
| first_registration_date | 최초 등록일 |
| year | 연식 |
| current_mileage_km | 주행거리 |
| fuel_type | 연료 타입 (가솔린/디젤) |
| overall_score | 엔진 종합 점수 |
| mid_freq_score | 중주파 점수 |
| low_high_freq | 저/고주파 점수 |
| regularity | 규칙성 점수 |
| irregularity | 불규칙성 점수 |
| audible_range_score | 가청대역 점수 |

### 디렉토리 구조

```
data/
├── car_audio_metadata.csv      # 메타데이터
├── goods_nos.csv               # 수집 상태
├── vehicle_assets/             # MP3 저장소
│   ├── HIG251121021317/
│   │   └── audio.mp3
│   ├── HDN251218022433/
│   │   └── audio.mp3
│   └── ...
└── debug_html/                 # 디버깅용 HTML
```

---

## 주요 특징

| 기능 | 구현 방식 |
|------|----------|
| **동적 로딩** | Selenium + "더보기" 버튼 반복 클릭 |
| **필터 적용** | JavaScript Executor로 체크박스 클릭 |
| **중단/재개** | `goods_nos.csv`에 상태 저장, 미완료만 처리 |
| **에러 핸들링** | try/except + 재시도 로직 (max_retries=3) |
| **진행 상황 보존** | 매 항목마다 CSV 저장 |

---

## 문제 해결

| 증상 | 원인 | 해결 |
|-----|-----|-----|
| ChromeDriver 다운로드 실패 | 네트워크 문제 | `use_auto_driver_download: false` + 수동 경로 설정 |
| 셀렉터 매칭 실패 | 웹사이트 구조 변경 | config 셀렉터 업데이트 + debug_html 확인 |
| MP3 다운로드 실패 | 링크 만료 | `max_retries` 증가 또는 수동 확인 |
| 필터 적용 안 됨 | JS 로딩 지연 | `request_delay_sec` 증가 |

---

## 현재 데이터 현황 (2026-01-15)

| 항목 | 수량 |
|------|------|
| 총 레코드 | 2,239개 |
| 가솔린 | 2,070개 |
| 디젤 | 169개 |
| MP3 파일 | 2,237개 |

---

*마지막 업데이트: 2026-01-23*
