"""
Main Crawler - 크롤링 오케스트레이션

전체 크롤링 파이프라인:
    1. 리스트 페이지 로드
    2. 필터 적용 (가솔린/디젤)
    3. 더보기 클릭
    4. goodsNo 수집
    5. 상세 페이지 크롤링
    6. MP3 다운로드
"""

import os
import time
from pathlib import Path
from typing import Optional, Set

from egai.data.crawler.scraper import WebScraper
from egai.data.crawler.parser import PageParser
from egai.data.crawler.downloader import AudioDownloader
from egai.data.manager import DataManager


class MainCrawler:
    """
    크롤링 파이프라인 오케스트레이터
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 크롤러 설정 딕셔너리
        """
        self.config = config
        crawler_settings = config.get("crawler_settings", {})

        # 컴포넌트 초기화
        self.scraper = WebScraper(
            user_agent=crawler_settings.get("user_agent", "EGAI-Crawler/1.0"),
            request_delay=crawler_settings.get("request_delay_sec", 1.0),
            timeout=crawler_settings.get("timeout_sec", 30),
            max_retries=crawler_settings.get("max_retries", 3),
            retry_delay=crawler_settings.get("retry_delay_sec", 2.0),
            use_selenium=crawler_settings.get("use_selenium", True),
            selenium_headless=crawler_settings.get("selenium_headless", True),
            use_auto_driver_download=crawler_settings.get("use_auto_driver_download", True),
        )

        self.parser = PageParser(config.get("data_selectors", {}))

        self.downloader = AudioDownloader(
            user_agent=crawler_settings.get("user_agent", "EGAI-Crawler/1.0"),
            request_delay=crawler_settings.get("request_delay_sec", 1.0),
            timeout=crawler_settings.get("timeout_sec", 30),
            max_retries=crawler_settings.get("max_retries", 3),
            retry_delay=crawler_settings.get("retry_delay_sec", 2.0),
        )

        self.data_manager = DataManager()

    def run(self, max_items: Optional[int] = None):
        """
        크롤링 실행

        Args:
            max_items: 최대 수집 개수 (None이면 전체)
        """
        print("=" * 60)
        print("EGAI 크롤러 시작")
        print("=" * 60)

        try:
            # 1. 리스트 페이지에서 goodsNo 수집
            goods_nos = self._collect_goods_nos()

            if not goods_nos:
                print("[Crawler] 수집된 goodsNo가 없습니다.")
                return

            print(f"[Crawler] 총 {len(goods_nos)}개 goodsNo 발견")

            # 2. 기존 데이터와 비교
            existing = self.data_manager.get_existing_goods_nos()
            new_goods_nos = goods_nos - existing

            if not new_goods_nos:
                print("[Crawler] 새로운 goodsNo가 없습니다.")
                return

            print(f"[Crawler] 신규 {len(new_goods_nos)}개 goodsNo 처리 예정")

            # 3. 상세 페이지 크롤링
            processed = 0
            for goods_no in new_goods_nos:
                if max_items and processed >= max_items:
                    break

                self._process_detail_page(goods_no)
                processed += 1

            print(f"[Crawler] {processed}개 처리 완료")

        finally:
            self.scraper.close()
            print("[Crawler] 크롤러 종료")

    def _collect_goods_nos(self) -> Set[str]:
        """리스트 페이지에서 goodsNo 수집"""
        urls = self.config.get("urls", {})
        base_url = urls.get("base_url", "")
        list_pattern = urls.get("list_page_pattern", "")
        list_url = f"{base_url}{list_pattern}"

        print(f"[Crawler] 리스트 페이지 로드: {list_url}")

        html_content = self.scraper.get_html(
            list_url,
            scroll_limit=self.config.get("crawler_settings", {}).get("scroll_load_limit", 5),
            click_selector=urls.get("next_page_selector"),
        )

        if not html_content:
            return set()

        goods_no_selector = urls.get("goods_no_selector", {})
        return self.parser.parse_goods_nos(html_content, goods_no_selector)

    def _process_detail_page(self, goods_no: str):
        """상세 페이지 처리"""
        urls = self.config.get("urls", {})
        base_url = urls.get("base_url", "")
        detail_pattern = urls.get("detail_page_pattern", "")
        detail_url = f"{base_url}{detail_pattern.format(goods_no=goods_no)}"

        print(f"[Crawler] 상세 페이지: {goods_no}")

        html_content = self.scraper.get_html(detail_url)
        if not html_content:
            return

        # 데이터 추출
        data = self.parser.parse_detail_page(html_content)
        data["goodsNo"] = goods_no

        # MP3 다운로드
        audio_url = data.get("audio_url_on_page")
        if audio_url:
            filename = self.downloader.get_filename_from_url(audio_url)
            save_dir = self.data_manager.get_vehicle_asset_dir(goods_no)
            save_path = os.path.join(save_dir, filename)

            if self.downloader.download(audio_url, save_path):
                data["audio_file_path"] = os.path.join("vehicle_assets", goods_no, filename)

        # 메타데이터 저장
        self.data_manager.save_metadata(data)


def run_crawler(config_path: str = "config/crawler_config.json"):
    """CLI 진입점"""
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    crawler = MainCrawler(config)
    crawler.run()


if __name__ == "__main__":
    run_crawler()
