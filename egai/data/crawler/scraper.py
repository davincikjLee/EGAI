"""
Web Scraper - HTTP/Selenium 요청 처리

기능:
    - requests 기반 정적 크롤링
    - Selenium 기반 동적 크롤링
    - 재시도 로직
"""

import time
import requests
from typing import Optional, Dict

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class WebScraper:
    """
    HTTP 요청 및 Selenium 웹 자동화 클래스
    """

    def __init__(
        self,
        user_agent: str,
        request_delay: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        use_selenium: bool = False,
        selenium_headless: bool = True,
        use_auto_driver_download: bool = True,
    ):
        self.headers = {"User-Agent": user_agent}
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_selenium = use_selenium
        self.driver = None

        if self.use_selenium and SELENIUM_AVAILABLE:
            self._init_selenium(selenium_headless, use_auto_driver_download)

    def _init_selenium(self, headless: bool, auto_download: bool):
        """Selenium WebDriver 초기화"""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument(f"user-agent={self.headers['User-Agent']}")
            if headless:
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            if auto_download:
                service = webdriver.ChromeService(
                    executable_path=ChromeDriverManager().install()
                )
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)

            self.driver.set_page_load_timeout(self.timeout)
            print("[Scraper] Selenium WebDriver 초기화 완료")
        except Exception as e:
            print(f"[Scraper] Selenium 초기화 실패: {e}")
            self.use_selenium = False
            self.driver = None

    def get_html(
        self,
        url: str,
        scroll_limit: int = 0,
        click_selector: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        URL에서 HTML 가져오기

        Args:
            url: 대상 URL
            scroll_limit: 더보기 클릭 횟수
            click_selector: 클릭할 요소 셀렉터 정보

        Returns:
            HTML 문자열 또는 None
        """
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)

                if self.use_selenium and self.driver:
                    return self._get_with_selenium(url, scroll_limit, click_selector)
                else:
                    return self._get_with_requests(url)

            except Exception as e:
                print(f"[Scraper] 요청 실패 ({attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def _get_with_requests(self, url: str) -> Optional[str]:
        """requests로 HTML 가져오기"""
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _get_with_selenium(
        self,
        url: str,
        scroll_limit: int,
        click_selector: Optional[Dict],
    ) -> Optional[str]:
        """Selenium으로 동적 HTML 가져오기"""
        self.driver.get(url)
        WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # 더보기 클릭
        if scroll_limit > 0 and click_selector:
            by_type = getattr(By, click_selector["type"].upper())
            for i in range(scroll_limit):
                try:
                    button = WebDriverWait(self.driver, self.timeout).until(
                        EC.element_to_be_clickable((by_type, click_selector["selector"]))
                    )
                    button.click()
                    time.sleep(self.request_delay * 2)
                except (TimeoutException, NoSuchElementException):
                    break

        return self.driver.page_source

    def close(self):
        """WebDriver 종료"""
        if self.driver:
            self.driver.quit()
            print("[Scraper] WebDriver 종료")
