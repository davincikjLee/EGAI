import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# ChromeDriver 자동 다운로드 및 관리를 위한 라이브러리
from webdriver_manager.chrome import ChromeDriverManager


class WebScraper:
    """
    HTTP 요청을 보내고 응답을 처리하는 클래스입니다.
    네트워크 관련 에러 핸들링 및 재시도 로직을 포함하며, Selenium을 통한 웹 자동화 기능도 제공합니다.
    """

    def __init__(self, user_agent, request_delay, timeout, max_retries, retry_delay,
                 use_selenium=False, selenium_driver_path=None, use_auto_driver_download=False,
                 selenium_headless=True):

        # --- 디버그 시작: 전달받은 파라미터 값과 타입을 확인 ---
        print("\n--- WebScraper __init__ 디버그 시작 ---")
        print(f"DEBUG(WebScraper.__init__): user_agent={user_agent} ({type(user_agent)})")
        print(f"DEBUG(WebScraper.__init__): request_delay={request_delay} ({type(request_delay)})")
        print(f"DEBUG(WebScraper.__init__): timeout={timeout} ({type(timeout)})")
        print(f"DEBUG(WebScraper.__init__): max_retries={max_retries} ({type(max_retries)})")
        print(f"DEBUG(WebScraper.__init__): retry_delay={retry_delay} ({type(retry_delay)})")
        print(f"DEBUG(WebScraper.__init__): use_selenium={use_selenium} ({type(use_selenium)})")
        print(f"DEBUG(WebScraper.__init__): selenium_driver_path={selenium_driver_path} ({type(selenium_driver_path)})")
        print(f"DEBUG(WebScraper.__init__): use_auto_driver_download={use_auto_driver_download} ({type(use_auto_driver_download)})")
        print(f"DEBUG(WebScraper.__init__): selenium_headless={selenium_headless} ({type(selenium_headless)})")
        print("--- WebScraper __init__ 디버그 끝 ---")
        # --- 디버그 끝 ---

        # 파라미터들을 self. 속성으로 명확하게 할당
        self.headers = {'User-Agent': user_agent}
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.use_selenium = use_selenium
        self.use_auto_driver_download = use_auto_driver_download
        self.driver = None  # Selenium WebDriver 인스턴스 초기화

        # max_retries가 유효한지 최종 검사 (ConfigLoader에서 처리하지만 방어적으로)
        if not isinstance(self.max_retries, int) or self.max_retries < 1:
            raise ValueError(f"WebScraper 초기화 오류: max_retries는 1 이상의 정수여야 합니다. 현재 값: {self.max_retries}")

        if self.use_selenium:
            self._init_selenium_driver(selenium_driver_path, selenium_headless)

    def _init_selenium_driver(self, driver_path, headless):
        """Selenium WebDriver를 초기화합니다."""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument(f"user-agent={self.headers['User-Agent']}")
            if headless:
                options.add_argument("--headless")  # UI 없이 백그라운드 실행
                options.add_argument("--disable-gpu")  # Headless 모드에서 GPU 사용 안 함
            options.add_argument("--no-sandbox")  # Docker 등 리눅스 환경에서 필요
            options.add_argument("--disable-dev-shm-usage")  # Docker 등 환경에서 /dev/shm 문제 해결

            if self.use_auto_driver_download:
                print("ChromeDriver 자동 다운로드 및 설치 중...")
                # ChromeDriverManager().install()은 드라이버 경로를 반환합니다.
                service = webdriver.ChromeService(executable_path=ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            elif driver_path:
                print(f"지정된 ChromeDriver 경로 사용: {driver_path}")
                service = webdriver.ChromeService(executable_path=driver_path)
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                # driver_path가 None이고 자동 다운로드도 false이면 시스템 PATH에서 찾음
                print("시스템 PATH에서 ChromeDriver를 찾습니다.")
                self.driver = webdriver.Chrome(options=options)

            self.driver.set_page_load_timeout(self.timeout)  # 페이지 로드 타임아웃
            print("Selenium WebDriver 초기화 완료.")
        except WebDriverException as e:
            print(f"오류: Selenium WebDriver 초기화 실패. 드라이버 경로/설치 또는 Chrome 버전 확인: {e}")
            self.use_selenium = False  # Selenium 사용 불가로 설정
            self.driver = None

    def get_html(self, url, scroll_limit=0, click_selector_info=None):
        """
        주어진 URL에서 HTML 내용을 가져옵니다. Selenium 사용 시 동적 로딩 콘텐츠도 처리합니다.

        Args:
            url (str): HTML을 가져올 웹 페이지 URL.
            scroll_limit (int): '더보기' 버튼을 누르거나 스크롤할 최대 횟수 (Selenium 사용 시). 0이면 스크롤 안 함.
            click_selector_info (dict): '더보기' 버튼의 셀렉터 정보 (Selenium 사용 시).

        Returns:
            str: 성공적으로 가져온 HTML 내용. 실패 시 None.
        """
        for attempt in range(self.max_retries):
            try:
                print(f"  요청 중: {url} (시도: {attempt + 1}/{self.max_retries})")
                time.sleep(self.request_delay)

                if self.use_selenium and self.driver:
                    self.driver.get(url)
                    # 페이지 로딩 대기 (필요 시 명시적 대기 조건 추가)
                    WebDriverWait(self.driver, self.timeout).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )

                    # '더보기' 버튼 클릭을 통한 동적 로딩
                    if scroll_limit > 0 and click_selector_info:
                        click_by = getattr(By, click_selector_info['type'].upper())
                        for i in range(scroll_limit):
                            try:
                                # '더보기' 버튼이 나타날 때까지 대기
                                more_button = WebDriverWait(self.driver, self.timeout).until(
                                    EC.element_to_be_clickable((click_by, click_selector_info['selector']))
                                )
                                print(f"  '더보기' 버튼 클릭 (시도 {i + 1}/{scroll_limit})...")
                                more_button.click()
                                time.sleep(self.request_delay * 2)  # 클릭 후 데이터 로딩 대기
                                # 추가 데이터가 로드될 때까지 기다리는 명시적 조건 추가 가능
                            except (TimeoutException, NoSuchElementException):
                                print(f"  '더보기' 버튼을 더 이상 찾을 수 없거나 클릭할 수 없습니다. 스크롤 종료.")
                                break  # 버튼 없으면 종료
                            except WebDriverException as e:
                                print(f"  '더보기' 버튼 클릭 중 오류 발생: {e}. 스크롤 종료.")
                                break
                    return self.driver.page_source  # Selenium이 렌더링한 최종 HTML 반환

                else:  # requests 라이브러리 사용 (정적 HTML)
                    response = requests.get(url, headers=self.headers, timeout=self.timeout)
                    response.raise_for_status()
                    return response.text

            except (requests.exceptions.RequestException, WebDriverException, TimeoutException) as e:
                print(f"  요청 실패 (시도 {attempt + 1}/{self.max_retries}) for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"  최대 재시도 횟수 도달. {url} 가져오기 실패.")
                    return None
        return None

    def close(self):
        """Selenium WebDriver를 종료합니다."""
        if self.driver:
            print("Selenium WebDriver 종료.")
            self.driver.quit()