# src/main_crawler.py

from src.config_loader import ConfigLoader
from src.web_scraper import WebScraper
from src.page_parser import PageParser
from src.data_manager import DataManager
from src.audio_downloader import AudioDownloader


import time
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from typing import Set, List, Dict, Union, Any, Optional  # Added Optional for clarity in type hints


class MainCrawler:
    """
    크롤링 프로세스의 전체 흐름을 제어하고 각 모듈을 오케스트레이션하는 메인 크롤러 클래스입니다.
    """

    def __init__(self):
        self.config_loader = ConfigLoader()  # ConfigLoader 인스턴스 생성
        self.config = self.config_loader.load_config()  # config 로드 (딕셔너리)

        # WebScraper 초기화
        self.scraper = WebScraper(
            user_agent=self.config_loader.get('crawler_settings.user_agent', expected_type=str),
            request_delay=self.config_loader.get('crawler_settings.request_delay_sec', expected_type=(int, float)),
            timeout=self.config_loader.get('crawler_settings.timeout_sec', expected_type=int),
            max_retries=self.config_loader.get('crawler_settings.max_retries', expected_type=int),
            retry_delay=self.config_loader.get('crawler_settings.retry_delay_sec', expected_type=(int, float)),
            use_selenium=self.config_loader.get('crawler_settings.use_selenium', expected_type=bool),
            selenium_driver_path=self.config_loader.get('crawler_settings.selenium_driver_path',
                                                        expected_type=(str, type(None)), default=None),
            use_auto_driver_download=self.config_loader.get('crawler_settings.use_auto_driver_download',
                                                            expected_type=bool),
            selenium_headless=self.config_loader.get('crawler_settings.selenium_headless', expected_type=bool)
        )

        # PageParser 초기화
        self.parser = PageParser(self.config_loader.get('data_selectors', expected_type=dict))
        self.data_manager = DataManager()  # DataManager 인스턴스 생성

        # AudioDownloader 초기화 (WebScraper와 동일한 설정 사용)
        self.audio_downloader = AudioDownloader(
            request_delay=self.config_loader.get('crawler_settings.request_delay_sec', expected_type=(int, float)),
            timeout=self.config_loader.get('crawler_settings.timeout_sec', expected_type=int),
            max_retries=self.config_loader.get('crawler_settings.max_retries', expected_type=int),
            retry_delay=self.config_loader.get('crawler_settings.retry_delay_sec', expected_type=(int, float)),
            user_agent=self.config_loader.get('crawler_settings.user_agent', expected_type=str)
        )

    def _get_list_page_url(self) -> str:
        """리스트 페이지 URL을 생성합니다."""
        base_url = self.config_loader.get('urls.base_url', expected_type=str)
        list_pattern = self.config_loader.get('urls.list_page_pattern', expected_type=str)
        return f"{base_url}{list_pattern}"

    def run(self):
        """
        크롤링 프로세스의 전체 흐름을 제어하고 각 모듈을 오케스트레이션하는 메인 크롤러 클래스입니다.
        고객 요구사항에 맞춰 연료 필터 선택, 총 대수 수집, '더보기' 클릭, goodsNo 수집 순서로 진행합니다.
        이후 수집된 goodsNo를 바탕으로 상세 페이지를 크롤링하고 데이터를 저장합니다.
        """
        print("--- 크롤러 시작: 리스트 페이지 조회 ---")

        all_found_goods_nos_set = set()  # 리스트 페이지에서 발견된 모든 goodsNo를 저장할 집합
        list_url = self._get_list_page_url()

        # 기존에 수집된 goodsNo (상태 포함) 로드
        goods_nos_df = self.data_manager.load_goods_nos_with_status()
        existing_goods_nos_set = set(goods_nos_df['goodsNo'].tolist())
        print(f"  [정보] 기존에 수집된 goodsNo {len(existing_goods_nos_set)}개 로드 완료.")

        print(f"\n[단계 1/5] 현재 페이지 로드 및 연료 필터 선택 시도: {list_url}")

        if self.scraper.use_selenium and self.scraper.driver:
            try:
                self.scraper.driver.get(list_url)
                WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                print("  [성공] 페이지 초기 로드 완료.")

                fuel_filter_gasoline_selector_info = self.config_loader.get('data_selectors.fuel_type_filter_gasoline',
                                                                            expected_type=dict)
                fuel_filter_diesel_selector_info = self.config_loader.get('data_selectors.fuel_type_filter_diesel',
                                                                          expected_type=dict)

                # --- '가솔린' 필터 클릭 (JavaScript Executor 사용) ---
                gasoline_input_id = fuel_filter_gasoline_selector_info.get('id_value')
                if not gasoline_input_id:
                    print(f"  [오류] '가솔린' 필터의 'id_value'가 config에 누락되었습니다. 클릭 불가.")
                    self.scraper.close()
                    return

                print(f"  '가솔린' 필터 클릭 시도 (JS Executor 사용, ID: {gasoline_input_id})...")
                try:
                    self.scraper.driver.execute_script(f"document.getElementById('{gasoline_input_id}').click();")
                    print("  [성공] '가솔린' 필터 체크박스 JS 클릭 완료.")
                    time.sleep(self.scraper.request_delay)
                except Exception as e:
                    print(f"  [오류] '가솔린' 필터 JS 클릭 실패: {e}. 다음 필터로 진행.")
                    self.scraper.close()
                    return

                # --- '디젤' 필터 클릭 (JavaScript Executor 사용) ---
                diesel_input_id = fuel_filter_diesel_selector_info.get('id_value')
                if not diesel_input_id:
                    print(f"  [오류] '디젤' 필터의 'id_value'가 config에 누락되었습니다. 클릭 불가.")
                    self.scraper.close()
                    return

                print(f"  '디젤' 필터 클릭 시도 (JS Executor 사용, ID: {diesel_input_id})...")
                try:
                    self.scraper.driver.execute_script(f"document.getElementById('{diesel_input_id}').click();")
                    print("  [성공] '디젤' 필터 체크박스 JS 클릭 완료.")
                    time.sleep(self.scraper.request_delay)
                except Exception as e:
                    print(f"  [오류] '디젤' 필터 JS 클릭 실패: {e}. 다음 단계로 진행.")
                    self.scraper.close()
                    return

                # 필터 적용 후 데이터 로딩 대기 및 확인
                print("  필터 적용 후 데이터 업데이트 대기 중 (확인 중)...")

                filter_word_gasoline_selector_info = self.config_loader.get(
                    'data_selectors.applied_filter_gasoline_word', expected_type=dict)
                filter_word_diesel_selector_info = self.config_loader.get('data_selectors.applied_filter_diesel_word',
                                                                          expected_type=dict)

                try:
                    WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                        EC.presence_of_element_located((getattr(By, filter_word_gasoline_selector_info['type'].upper()),
                                                        filter_word_gasoline_selector_info['selector']))
                    )
                    print("  [성공] '가솔린' 필터 적용 확인 완료.")
                except TimeoutException:
                    print(f"  [경고] '가솔린' 필터 적용을 확인하지 못했습니다. (셀렉터: {filter_word_gasoline_selector_info['selector']})")
                    print("  필터 적용 셀렉터가 정확한지, 필터 적용 후 HTML이 즉시 업데이트되는지 확인 필요.")
                    self.scraper.close()
                    return

                try:
                    WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                        EC.presence_of_element_located((getattr(By, filter_word_diesel_selector_info['type'].upper()),
                                                        filter_word_diesel_selector_info['selector']))
                    )
                    print("  [성공] '디젤' 필터 적용 확인 완료.")
                except TimeoutException:
                    print(f"  [경고] '디젤' 필터 적용을 확인하지 못했습니다. (셀렉터: {filter_word_diesel_selector_info['selector']})")
                    print("  필터 적용 셀렉터가 정확한지, 필터 적용 후 HTML이 즉시 업데이트되는지 확인 필요.")
                    self.scraper.close()
                    return

                # 두 필터가 모두 적용된 것을 확인한 후, 총 대수 요소가 업데이트될 때까지 대기
                print("  두 필터 모두 적용 확인. 총 차량 대수 업데이트 대기...")
                total_count_selector = self.config_loader.get('urls.total_count_selector', expected_type=dict)
                try:
                    WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                        EC.visibility_of_element_located(
                            (getattr(By, total_count_selector['type'].upper()), total_count_selector['selector']))
                    )
                    print("  [성공] 총 차량 대수 요소가 나타남을 확인.")
                    time.sleep(self.scraper.request_delay)
                except TimeoutException:
                    print(f"  [경고] 총 차량 대수 요소를 찾거나 업데이트를 확인하지 못했습니다. (셀렉터: {total_count_selector['selector']})")
                    self.scraper.close()
                    return

                # 필터링된 페이지의 HTML 콘텐츠 가져오기
                html_content_after_filter = self.scraper.driver.page_source
                print("  [성공] 필터링된 페이지의 HTML 콘텐츠 가져오기 완료.")

                # 2. 총 대수 정보 수집
                print("\n[단계 2/5] 총 차량 대수 정보 수집 시도...")
                total_cars = self.parser.get_total_count(html_content_after_filter, total_count_selector)
                if total_cars is not None:
                    print(f"  [성공] 총 차량 대수: {total_cars} 대")
                else:
                    print(f"  [경고] 총 차량 대수를 찾을 수 없습니다. 셀렉터가 유효한지 확인하세요: {total_count_selector['selector']}")
                    self.scraper.close()
                    return

                # 3. '더보기' 버튼 클릭을 통한 동적 로딩
                print("\n[단계 3/5] '더보기' 버튼 클릭 시도 (동적 로딩)...")

                # 더보기 버튼 및 아이템 체크 셀렉터 정보 가져오기
                more_button_selector_info = self.config_loader.get('urls.next_page_selector', expected_type=dict)
                click_by_type = getattr(By, more_button_selector_info['type'].upper())

                item_check_selector_for_count_info = more_button_selector_info.get('item_check_selector')
                if not item_check_selector_for_count_info:
                    print(f"  [오류] 'next_page_selector'에 'item_check_selector'가 누락되었습니다. 리스트 아이템 개수 확인 불가.")
                    self.scraper.close()
                    return

                item_check_selector_for_count_type = getattr(By, item_check_selector_for_count_info[
                    'type'].upper()) if isinstance(item_check_selector_for_count_info,
                                                   dict) and 'type' in item_check_selector_for_count_info else click_by_type
                item_check_selector_for_count_value = item_check_selector_for_count_info['selector'] if isinstance(
                    item_check_selector_for_count_info,
                    dict) and 'selector' in item_check_selector_for_count_info else item_check_selector_for_count_info

                # 초기 리스트 아이템 개수 확인
                initial_item_count = len(self.scraper.driver.find_elements(item_check_selector_for_count_type,
                                                                           item_check_selector_for_count_value))
                print(f"  초기 리스트 아이템 개수: {initial_item_count}개")

                if initial_item_count == 0:
                    print("  [경고] 초기 리스트 아이템 개수가 0개입니다. 필터링 결과가 없거나, 리스트 로딩에 문제가 있습니다. '더보기' 클릭을 건너뜁니다.")
                    final_html_content = self.scraper.driver.page_source

                    self.data_manager.save_debug_html("empty_initial_list_page", final_html_content,
                                                      filename_suffix="empty_initial_list_page")
                    print("  디버깅을 위해 'data/debug_html/debug_empty_initial_list_page.html'에 최종 HTML을 저장했습니다.")

                # '더보기' 1회 클릭 후 증가하는 아이템 개수 측정 (initial_item_count가 0이 아닐 때만)
                items_per_load = 0
                if initial_item_count > 0:
                    try:
                        first_more_button = WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                            EC.element_to_be_clickable((click_by_type, more_button_selector_info['selector']))
                        )
                        print(f"  '더보기' 버튼 (최초 1회) 클릭하여 증가량 측정 시도...")
                        first_more_button.click()

                        WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                            lambda driver: len(driver.find_elements(item_check_selector_for_count_type,
                                                                    item_check_selector_for_count_value)) > initial_item_count
                        )
                        current_item_after_first_click = len(
                            self.scraper.driver.find_elements(item_check_selector_for_count_type,
                                                              item_check_selector_for_count_value))
                        items_per_load = current_item_after_first_click - initial_item_count
                        print(f"  [측정 성공] '더보기' 1회 클릭 시 {items_per_load}개 아이템 증가 확인.")

                    except (TimeoutException, NoSuchElementException) as e:
                        print(f"  [경고] '더보기' 버튼 (최초 1회)을 찾을 수 없거나 클릭할 수 없습니다. 또는 아이템 증가 없음. 오류: {e}")
                        print("  모든 아이템이 이미 로드되었거나 '더보기' 버튼이 없습니다. 현재 로드된 아이템만 수집합니다.")
                        items_per_load = 0
                    except Exception as e:
                        print(f"  [오류] '더보기' 버튼 측정 클릭 중 치명적 오류 발생: {e}. 현재 로드된 아이템만 수집합니다.")
                        items_per_load = 0

                # 총 클릭 횟수 계산
                clicks_needed = 0
                if total_cars is not None and items_per_load > 0 and total_cars > initial_item_count:
                    clicks_needed = (
                                                total_cars - initial_item_count - items_per_load + items_per_load) // items_per_load  # ceiling division
                    clicks_needed = max(0, clicks_needed)  # 음수가 되지 않도록
                    print(
                        f"  총 {total_cars}대 중 {initial_item_count + items_per_load}대가 로드됨. 추가로 {clicks_needed}번 더 클릭 필요.")
                elif total_cars is not None and total_cars <= initial_item_count:
                    print(f"  총 {total_cars}대 중 초기 {initial_item_count}대가 이미 로드됨. '더보기' 클릭 불필요.")
                else:
                    clicks_needed = self.config_loader.get('crawler_settings.scroll_load_limit', expected_type=int,
                                                           default=5)
                    print(f"  총 대수 또는 증가량 파악 불가. config의 scroll_load_limit({clicks_needed}회)만큼 클릭 시도.")

                # 실제 '더보기' 버튼 반복 클릭 (initial_item_count가 0이 아닐 때만)
                if initial_item_count > 0:
                    current_item_count = initial_item_count + items_per_load if items_per_load > 0 else initial_item_count
                    for i in range(clicks_needed):
                        try:
                            more_button = WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                                EC.element_to_be_clickable((click_by_type, more_button_selector_info['selector']))
                            )
                            print(f"  '더보기' 버튼 클릭 (시도 {i + 1}/{clicks_needed})...")
                            more_button.click()

                            WebDriverWait(self.scraper.driver, self.scraper.timeout).until(
                                lambda driver: len(driver.find_elements(item_check_selector_for_count_type,
                                                                        item_check_selector_for_count_value)) > current_item_count
                            )

                            new_item_count = len(self.scraper.driver.find_elements(item_check_selector_for_count_type,
                                                                                   item_check_selector_for_count_value))
                            print(f"  현재 리스트 아이템 개수: {new_item_count}개")

                            if new_item_count <= current_item_count:
                                print(f"  [정보] 리스트 아이템 개수가 증가하지 않았습니다. 더 이상 로드할 내용이 없거나 오류입니다. 스크롤 종료.")
                                break

                            current_item_count = new_item_count

                        except (TimeoutException, NoSuchElementException) as e:
                            print(f"  [정보] '더보기' 버튼을 더 이상 찾을 수 없거나 클릭할 수 없습니다. 스크롤 종료. 오류: {e}")
                            break
                        except Exception as e:
                            print(f"  [오류] '더보기' 버튼 클릭 또는 아이템 개수 확인 중 치명적 오류 발생: {e}. 스크롤 종료.")
                            break

                # '더보기' 클릭 루프가 완료되었거나 건너뛴 경우 최종 HTML 가져오기
                final_html_content = self.scraper.driver.page_source
                print("  [성공] 모든 '더보기' 클릭 후 최종 HTML 콘텐츠 가져오기 완료.")

                # 4. 가솔린/디젤 제품 번호 정보 수집
                print("\n[단계 4/5] 최종 HTML에서 goodsNo 정보 수집 시도...")
                goods_no_selector = self.config_loader.get('urls.goods_no_selector', expected_type=dict)
                found_goods_nos_on_page = self.parser.parse_list_page_goods_nos(final_html_content, goods_no_selector)

                if found_goods_nos_on_page:
                    print(f"  [성공] 최종적으로 {len(found_goods_nos_on_page)}개의 goodsNo 발견.")
                    # 새로 발견된 goodsNo만 추가 (기존 목록과 비교)
                    newly_discovered_goods_nos = found_goods_nos_on_page - existing_goods_nos_set
                    all_found_goods_nos_set.update(found_goods_nos_on_page)  # 전체 목록 업데이트

                    # goods_nos.csv에 저장 (중복 방지 및 상태 초기화)
                    if newly_discovered_goods_nos:
                        # 기존 DataFrame에 새로 발견된 goodsNo 추가 (data_collected=False, mp3_downloaded=False)
                        new_goods_nos_list = [{'goodsNo': gn, 'data_collected': False, 'mp3_downloaded': False} for gn
                                              in newly_discovered_goods_nos]
                        goods_nos_df = self.data_manager.add_new_goods_nos_to_df(goods_nos_df, new_goods_nos_list)
                        self.data_manager.save_goods_nos_with_status(goods_nos_df)  # 전체 DataFrame 저장
                        print(f"  [성공] 새로 발견된 goodsNo {len(newly_discovered_goods_nos)}개를 포함하여 goods_nos.csv에 저장 완료.")
                    else:
                        print("  [정보] 새로 발견된 goodsNo가 없습니다. goods_nos.csv 업데이트 건너뜁니다.")

                else:
                    print(f"  [경고] 최종 페이지에서 goodsNo를 찾을 수 없습니다. 셀렉터 오류일 수 있습니다.")
                    self.data_manager.save_debug_html("final_list_page", final_html_content,
                                                      filename_suffix="final_list_page")  # 디버그 저장
                    print("  디버깅을 위해 'data/debug_html/debug_final_list_page.html'에 최종 HTML을 저장했습니다.")

            except Exception as e:  # Selenium 관련 최상위 오류 처리
                print(f"  [치명적 오류] Selenium 크롤링 과정에서 예상치 못한 오류 발생: {e}")
                self.scraper.close()
                return

        else:  # Selenium 비활성화 시 로직 (정적 크롤링)
            print("  [정보] Selenium이 비활성화되어 동적 필터링 및 '더보기' 기능을 건너뛰고 정적 크롤링을 시도합니다.")
            html_content = self.scraper.get_html(list_url)
            if html_content:
                total_count_selector = self.config_loader.get('urls.total_count_selector', expected_type=dict)
                total_cars = self.parser.get_total_count(html_content, total_count_selector)
                if total_cars is not None:
                    print(f"  [정적] 웹사이트 총 차량 대수: {total_cars} 대")
                goods_no_selector = self.config_loader.get('urls.goods_no_selector', expected_type=dict)
                found_goods_nos_on_page = self.parser.parse_list_page_goods_nos(html_content, goods_no_selector)
                if found_goods_nos_on_page:
                    print(f"  [정적] {len(found_goods_nos_on_page)}개의 goodsNo 발견.")
                    all_found_goods_nos_set.update(found_goods_nos_on_page)
                    # 정적 크롤링 시에도 goods_nos_df 업데이트 및 저장
                    new_goods_nos_list = [{'goodsNo': gn, 'data_collected': False, 'mp3_downloaded': False} for gn in
                                          found_goods_nos_on_page if gn not in existing_goods_nos_set]
                    if new_goods_nos_list:
                        goods_nos_df = self.data_manager.add_new_goods_nos_to_df(goods_nos_df, new_goods_nos_list)
                        self.data_manager.save_goods_nos_with_status(goods_nos_df)
                        print(f"  [정적] 새로 발견된 goodsNo {len(new_goods_nos_list)}개를 goods_nos.csv에 저장 완료.")
                else:
                    print(f"  [경고] 정적 페이지에서 goodsNo를 찾을 수 없습니다. 셀렉터 오류일 수 있습니다.")
            else:
                print(f"  [오류] {list_url} 페이지 HTML을 가져오는 데 실패했습니다. 리스트 페이지 크롤링을 중단합니다.")

        print(f"\n--- 리스트 페이지 크롤링 완료 ---")
        print(f"총 발견된 고유 goodsNo 개수: {len(all_found_goods_nos_set)}")
        if len(all_found_goods_nos_set) > 0:
            print("발견된 goodsNo (일부):", list(all_found_goods_nos_set)[:10])
        else:
            print("발견된 goodsNo가 없습니다.")

        # --- 상세 페이지 크롤링 루프 시작 ---
        print("\n--- 상세 페이지 크롤링 및 데이터 수집 시작 ---")

        # goods_nos.csv에서 처리되지 않은 goodsNo만 가져오기
        goods_nos_to_process_df = self.data_manager.load_goods_nos_with_status()
        unprocessed_goods_nos = goods_nos_to_process_df[
            (goods_nos_to_process_df['data_collected'] == False) |
            (goods_nos_to_process_df['mp3_downloaded'] == False)
            ]['goodsNo'].tolist()

        if not unprocessed_goods_nos:
            print("  [정보] 처리할 새로운 goodsNo가 없습니다. 상세 페이지 크롤링을 건너뜁니다.")
            self.scraper.close()  # Selenium 드라이버 종료
            print("Crawler finished.")
            return

        print(f"  총 {len(unprocessed_goods_nos)}개의 goodsNo에 대해 상세 페이지 크롤링을 진행합니다.")

        detail_page_pattern = self.config_loader.get('urls.detail_page_pattern', expected_type=str)
        base_url = self.config_loader.get('urls.base_url', expected_type=str)

        for i, goods_no in enumerate(unprocessed_goods_nos):
            print(f"\n  [진행 {i + 1}/{len(unprocessed_goods_nos)}] goodsNo: {goods_no} 상세 데이터 수집 중...")
            detail_url = f"{base_url}{detail_page_pattern.format(goods_no=goods_no)}"

            try:
                # 상세 페이지 HTML 가져오기 (Selenium 사용)
                detail_html_content = self.scraper.get_html(detail_url)

                if detail_html_content:
                    # 1. 상세 페이지 데이터 추출
                    extracted_data = self.parser.parse_detail_page(detail_html_content)
                    extracted_data['goodsNo'] = goods_no  # goodsNo 추가

                    # 2. MP3 파일 다운로드
                    audio_url = extracted_data.get('audio_url_on_page')
                    audio_file_path = None  # 초기화
                    if audio_url:
                        assets_dir = self.data_manager.create_vehicle_asset_dir(goods_no)  # 폴더 생성 및 경로 반환

                        # URL에서 파일명 추출 (쿼리스트링 제거 및 확장자 확인)
                        audio_filename = os.path.basename(audio_url.split('?')[0])
                        if not audio_filename.lower().endswith(('.mp3', '.wav', '.ogg')):  # 확장자가 없으면 mp3 추가
                            audio_filename += ".mp3"

                        audio_file_path_full = os.path.join(assets_dir, audio_filename)

                        # CSV에 저장될 상대 경로
                        audio_file_path_relative = os.path.join('vehicle_assets', goods_no, audio_filename)
                        extracted_data['audio_file_path'] = audio_file_path_relative  # 메타데이터에 파일 경로 추가

                        print(f"    MP3 파일 다운로드 시도: {audio_url} -> {audio_file_path_full}")
                        if self.audio_downloader.download_audio_file(audio_url,
                                                                     audio_file_path_full):  # AudioDownloader 사용
                            print(f"    [성공] MP3 파일 다운로드 완료: {audio_file_path_full}")
                            # goods_nos.csv의 mp3_downloaded 상태 업데이트
                            goods_nos_df = self.data_manager.update_goods_no_status(goods_nos_df, goods_no,
                                                                                    'mp3_downloaded', True)
                        else:
                            print(f"    [오류] MP3 파일 다운로드 실패: {audio_url}")
                            extracted_data['audio_file_path'] = None  # 실패 시 경로 제거
                    else:
                        print("    [정보] 오디오 URL을 찾을 수 없습니다. MP3 다운로드 건너뜁니다.")
                        extracted_data['audio_file_path'] = None  # 오디오 URL 없으면 경로도 없음

                    # 3. 메타데이터 CSV에 저장
                    self.data_manager.save_metadata_to_csv(extracted_data)
                    print(f"    [성공] goodsNo {goods_no}의 메타데이터 car_audio_metadata.csv에 저장 완료.")

                    # 4. goods_nos.csv에 데이터 수집 완료 상태 변경
                    goods_nos_df = self.data_manager.update_goods_no_status(goods_nos_df, goods_no, 'data_collected',
                                                                            True)

                else:
                    print(f"  [오류] goodsNo {goods_no}의 상세 페이지 HTML을 가져오는 데 실패했습니다.")
                    # 실패 시에도 상태 업데이트 (재시도 방지 또는 로그)
                    # goods_nos_df = self.data_manager.update_goods_no_status(goods_nos_df, goods_no, 'data_collected', False) # 실패했으니 False 유지
                    # goods_nos_df = self.data_manager.update_goods_no_status(goods_nos_df, goods_no, 'mp3_downloaded', False) # 실패했으니 False 유지

            except Exception as e:
                print(f"  [치명적 오류] goodsNo {goods_no} 상세 페이지 처리 중 오류 발생: {e}")

            # 각 상세 페이지 처리 후 goods_nos_df를 저장하여 진행 상황을 보존
            self.data_manager.save_goods_nos_with_status(goods_nos_df)

        print("\n--- 상세 페이지 크롤링 및 데이터 수집 완료 ---")

        self.scraper.close()  # Selenium 드라이버 종료
        print("Crawler finished.")