from bs4 import BeautifulSoup
from typing import Optional, Union, Type, Tuple, Any  # Type 추가
import re
from lxml import html
from lxml.cssselect import CSSSelector


# ConfigLoader와 WebScraper는 PageParser.py가 직접 실행될 때만 필요하므로,
# if __name__ == "__main__": 블록 안으로 임포트 위치를 옮깁니다.


class PageParser:
    """
    HTML 콘텐츠를 분석하여 config에 정의된 셀렉터들을 기반으로 데이터를 추출하는 클래스입니다.
    BeautifulSoup과 lxml을 모두 사용하여 CSS Selector와 XPath를 지원합니다.
    """

    def __init__(self, data_selectors_config: dict):  # config_loader_instance 인자 제거
        self.selectors = data_selectors_config

    def _get_lxml_tree(self, html_content: str):
        return html.fromstring(html_content)

    def _find_element_lxml(self, tree: html.HtmlElement, selector_type: str, selector_value: str) -> Optional[
        html.HtmlElement]:
        elements = []
        if selector_type == "xpath":
            elements = tree.xpath(selector_value)
        elif selector_type == "css":
            css_selector_obj = CSSSelector(selector_value)
            elements = css_selector_obj(tree)

        if elements and isinstance(elements[0], html.HtmlElement):
            return elements[0]
        return None

    def parse_list_page_goods_nos(self, html_content: str, goods_no_selector_config: dict) -> set:
        tree = self._get_lxml_tree(html_content)
        goods_nos = set()

        selector_type = goods_no_selector_config.get("type")
        selector_value = goods_no_selector_config.get("selector")
        extract_attribute = goods_no_selector_config.get("extract_attribute")

        if not selector_type or not selector_value:
            print(f"오류: goods_no_selector 설정에 'type' 또는 'selector'가 누락되었습니다.")
            return goods_nos

        elements = []
        if selector_type == "xpath":
            elements = tree.xpath(selector_value)
        elif selector_type == "css":
            css_selector_obj = CSSSelector(selector_value)
            elements = css_selector_obj(tree)
        else:
            print(f"오류: 지원되지 않는 셀렉터 타입: {selector_type}")
            return goods_nos

        if not elements:
            print(f"  [파서] 셀렉터 '{selector_value}'로 아무 요소도 찾을 수 없습니다. (HTML 내용 확인 필요)")
            return goods_nos

        for element in elements:
            goods_no_info = None
            if extract_attribute:
                goods_no_info = element.get(extract_attribute)

            if goods_no_info:
                match = re.search(r"common\.link\.goodsDeatil\('([^']+)'\)", goods_no_info)
                if match:
                    goods_nos.add(match.group(1))
                else:
                    print(f"경고: goodsNo를 '{goods_no_info}'에서 추출할 수 없습니다. 정규식 확인 필요. (전체 href: {goods_no_info})")
            else:
                print(f"경고: 요소에 'extract_attribute'('{extract_attribute}') 속성이 없거나 비어 있습니다. (요소: {element.tag})")
        return goods_nos

    def get_total_count(self, html_content: str, selector_info: dict) -> Optional[int]:
        tree = self._get_lxml_tree(html_content)

        selector_type = selector_info.get("type")
        selector_value = selector_info.get("selector")

        element = self._find_element_lxml(tree, selector_type, selector_value)
        if element is not None:
            try:
                text = element.text_content().strip()
                numbers = re.findall(r'\d+', text)
                if numbers:
                    return int(numbers[0])
                else:
                    print(f"  [파서] '{text}'에서 숫자를 찾을 수 없습니다.")
            except (ValueError, TypeError) as e:
                print(f"  [파서] 총 대수 텍스트 파싱 중 오류 ({selector_value}): {e}")
        else:
            print(f"  [파서] 총 대수 셀렉터 '{selector_value}'로 요소를 찾을 수 없습니다.")
        return None

    def parse_detail_page(self, html_content: str) -> dict:
        tree = self._get_lxml_tree(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        extracted_data = {}

        for key, selector_info in self.selectors.items():
            # 필터 관련 셀렉터는 상세 페이지 디버그에서 제외 (None 나오는 원인 중 하나)
            if key.startswith("fuel_type_filter_") or key.startswith("applied_filter_"):
                extracted_data[key] = None
                continue

            selector_type = selector_info.get("type")
            selector_value = selector_info.get("selector")
            extract_method = selector_info.get("extract_method")
            extract_attribute = selector_info.get("extract_attribute")
            clean_regex = selector_info.get("clean_regex")
            invert_boolean = selector_info.get("invert_boolean", False)
            is_iframe = selector_info.get("is_iframe", False)

            if is_iframe:
                print(f"경고: iframe 감지됨 ({key}). iframe 내 요소는 Selenium/Playwright가 필요합니다.")
                extracted_data[key] = None
                continue

            # --- 'list_key_value' 추출 방식 처리 (기본 정보 리스트) ---
            if extract_method == "list_key_value":
                if selector_type == "xpath":
                    list_elements = tree.xpath(selector_value)
                elif selector_type == "css":
                    css_selector_obj = CSSSelector(selector_value)
                    list_elements = css_selector_obj(tree)
                else:
                    print(f"경고: 'list_key_value'를 위한 지원되지 않는 셀렉터 타입: {selector_type} for {key}.")
                    continue

                if list_elements:
                    for li_element in list_elements:
                        try:
                            title_span_elements = li_element.xpath("./span[@class='tit']")
                            value_span_elements = li_element.xpath("./span[@class='txt']")

                            title = title_span_elements[0].text_content().strip() if title_span_elements else None
                            value = value_span_elements[0].text_content().strip() if value_span_elements else None

                            if title and value is not None:
                                # 키 매핑 및 클리닝 (ConfigManager의 컬럼명과 일치하도록)
                                mapped_key = None
                                if title == "최초등록":
                                    mapped_key = "first_registration_date"
                                elif title == "주행거리":
                                    mapped_key = "current_mileage_km"
                                elif title == "연료":
                                    mapped_key = "fuel_type"
                                elif title == "배기량":
                                    mapped_key = "displacement_cc"
                                elif title == "외관컬러":
                                    mapped_key = "exterior_color"
                                elif title == "내장컬러":
                                    mapped_key = "interior_color"
                                elif title == "차종":
                                    mapped_key = "vehicle_type"
                                elif title == "승차인원":
                                    mapped_key = "seating_capacity"
                                elif title == "구동방식":
                                    mapped_key = "drivetrain"
                                elif title == "차량번호":
                                    mapped_key = "vehicle_number"
                                elif title == "연식":
                                    mapped_key = "year"
                                elif title == "변속기":
                                    mapped_key = "transmission_type"
                                # '압류', '저당', '내차피해', '소유자 변경' 등은 base_01 리스트에 포함되지 않으므로 여기서 매핑하지 않음.
                                # 이들은 아래 개별 셀렉터로 처리됩니다. (경고 제거)

                                if mapped_key:
                                    if mapped_key in ["current_mileage_km", "displacement_cc", "seating_capacity",
                                                      "year"]:
                                        extracted_data[mapped_key] = re.sub(r'[^0-9]', '', value)
                                    else:
                                        extracted_data[mapped_key] = value
                                # else:
                                # print(f"경고: 기본 정보 리스트의 알 수 없는 항목: {title}: {value}") # 이 경고는 이제 안 나와야 함
                            # else:
                            # print(f"경고: 기본 정보 리스트 항목의 제목 또는 값이 누락되었습니다. (li: {li_element.text_content().strip()[:50]})")

                        except Exception as e:
                            print(f"경고: 기본 정보 리스트 파싱 중 오류 발생: {e} (요소: {li_element.text_content().strip()[:50]})")
                continue

                # --- 기존 개별 셀렉터 처리 (list_key_value에 포함되지 않는 셀렉터만) ---
            elements_found_by_selector = []
            if selector_type == "xpath":
                elements_found_by_selector = tree.xpath(selector_value)
            elif selector_type == "css":
                css_selector_obj = CSSSelector(selector_value)
                elements_found_by_selector = css_selector_obj(tree)
            else:
                print(f"경고: 지원되지 않는 셀렉터 타입 '{selector_type}' for {key}.")

            element_or_value = elements_found_by_selector[0] if elements_found_by_selector else None

            value = None

            if element_or_value is not None:
                if extract_method == "text":
                    if isinstance(element_or_value, html.HtmlElement):
                        value = element_or_value.text_content().strip()
                    else:
                        value = str(element_or_value).strip()

                    if clean_regex:
                        value = re.sub(clean_regex, '', value).strip()
                elif extract_method == "attribute" and extract_attribute:
                    if isinstance(element_or_value, html.HtmlElement):
                        value = element_or_value.get(extract_attribute)
                    else:  # 이미 속성 값 (문자열)이 추출된 경우
                        value = str(element_or_value).strip() if isinstance(element_or_value, (str,
                                                                                               html.etree._ElementUnicodeResult)) else None
                        # print(f"경고: '{key}' ({selector_value}) 셀렉터는 속성 추출을 요구하지만, 반환된 요소는 HTML Element가 아닙니다. 값: '{value}'") # 이 경고 제거
                elif extract_method == "exists":
                    value = True
                elif extract_method == "count":
                    value = len(elements_found_by_selector)
                elif extract_method == "count_gt_zero":
                    text_val = ""
                    if isinstance(element_or_value, html.HtmlElement):
                        text_val = element_or_value.text_content().strip()
                    else:
                        text_val = str(element_or_value).strip()
                    try:
                        num_val = int(re.sub(r'[^0-9]', '', text_val))
                        value = num_val > 0
                    except ValueError:
                        value = False
                else:
                    value = None
            else:  # 요소를 찾지 못했을 때
                if extract_method == "exists":
                    value = False
                elif extract_method == "count" or extract_method == "count_gt_zero":
                    value = 0 if extract_method == "count" else False
                else:
                    value = None

            if extract_method in ["exists", "count_gt_zero"] and invert_boolean:
                value = not value

            extracted_data[key] = value

        return extracted_data


# --- page_parser.py를 직접 실행하기 위한 디버그 블록 ---
if __name__ == "__main__":
    # ConfigLoader와 WebScraper 초기화 (PageParser 테스트를 위해 필요)
    # 실제 프로젝트 루트 디렉토리를 기준으로 config 파일을 찾습니다.
    from src.config_loader import ConfigLoader  # <-- 이 줄을 추가합니다.
    from src.web_scraper import WebScraper  # <-- 이 줄을 추가합니다.
    import time  # <-- 이 줄을 추가합니다.
    from selenium.webdriver.common.by import By  # <-- 이 줄을 추가합니다.
    from selenium.webdriver.support.ui import WebDriverWait  # <-- 이 줄을 추가합니다.
    from selenium.webdriver.support import expected_conditions as EC  # <-- 이 줄을 추가합니다.
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, \
        WebDriverException  # <-- 이 줄을 추가합니다.
    import os  # <-- 이 줄을 추가합니다. (경로 관련 함수 사용을 위해)
    from src.data_manager import DataManager  # <-- 이 줄을 추가합니다. (DataManager 임포트)

    # ConfigLoader와 WebScraper 초기화
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.normpath(os.path.join(current_script_dir, '..'))

    config_path = os.path.join(project_root_dir, 'config', 'crawler_config.json')
    default_config_path = os.path.join(project_root_dir, 'config', 'default_config.json')

    config_loader = ConfigLoader(config_path=config_path, default_config_path=default_config_path)
    config_data = config_loader.load_config()  # config 로드

    scraper = WebScraper(
        user_agent=config_loader.get('crawler_settings.user_agent', expected_type=str),
        request_delay=config_loader.get('crawler_settings.request_delay_sec', expected_type=(int, float)),
        timeout=config_loader.get('crawler_settings.timeout_sec', expected_type=int),
        max_retries=config_loader.get('crawler_settings.max_retries', expected_type=int),
        retry_delay=config_loader.get('crawler_settings.retry_delay_sec', expected_type=(int, float)),
        use_selenium=config_loader.get('crawler_settings.use_selenium', expected_type=bool),
        selenium_driver_path=config_loader.get('crawler_settings.selenium_driver_path', expected_type=(str, type(None)),
                                               default=None),
        use_auto_driver_download=config_loader.get('crawler_settings.use_auto_driver_download', expected_type=bool),
        selenium_headless=config_loader.get('crawler_settings.selenium_headless', expected_type=bool)
    )

    data_manager = DataManager()  # DataManager 초기화

    # PageParser 초기화 (data_selectors만 전달)
    parser = PageParser(config_loader.get('data_selectors', expected_type=dict))  # config_loader 인스턴스 인자 제거

    # --- 상세 페이지 1회 디버그 실행 ---
    debug_goods_no_1 = "GJJ250412013935"  # debug_detail_page_GJJ250412013935.html
    # debug_goods_no_1 = "HGN250428014419" # debug_detail_page_HGN250428014419.html

    try:
        detail_url_1 = config_loader.get('urls.base_url', expected_type=str) + \
                       config_loader.get('urls.detail_page_pattern', expected_type=str).format(
                           goods_no=debug_goods_no_1)
        print(f"\n--- 상세 페이지 1회 디버그 시작: goodsNo {debug_goods_no_1} ---")
        print(f"  디버그 대상 URL: {detail_url_1}")
        html_content_1 = scraper.get_html(detail_url_1, scroll_limit=0)
        if html_content_1:
            print("  상세 페이지 HTML 가져오기 완료. 데이터 추출 시도...")
            extracted_data_1 = parser.parse_detail_page(html_content_1)

            print("\n--- 추출된 상세 데이터 ---")
            if extracted_data_1:
                for key, value in extracted_data_1.items():
                    print(f"  {key}: {value} (타입: {type(value)})")
            else:
                print("  추출된 데이터가 없습니다. data_selectors를 확인하세요.")

            # 디버깅을 위해 HTML 파일 저장 (DataManager 사용)
            data_manager.save_debug_html(debug_goods_no_1, html_content_1, filename_suffix="detail_page")

        else:
            print(f"  상세 페이지 ({detail_url_1}) HTML을 가져오는 데 실패했습니다.")
        print("\n--- 상세 페이지 1회 디버그 종료 ---")

    except Exception as e:
        print(f"\n[치명적 오류] PageParser 디버그 중 예상치 못한 오류 발생: {e}")
    finally:
        if scraper.driver:
            print("디버그 완료 후 Selenium WebDriver를 종료합니다.")
            scraper.close()
    print("프로그램 종료.")