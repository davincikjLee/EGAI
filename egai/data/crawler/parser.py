"""
Page Parser - HTML 파싱

기능:
    - lxml/BeautifulSoup 기반 파싱
    - XPath/CSS 셀렉터 지원
    - 상세 페이지 데이터 추출
"""

import re
from typing import Optional, Set, Dict, Any
from bs4 import BeautifulSoup
from lxml import html
from lxml.cssselect import CSSSelector


class PageParser:
    """
    HTML 파싱 및 데이터 추출
    """

    def __init__(self, selectors: Dict[str, Any]):
        """
        Args:
            selectors: data_selectors 설정 딕셔너리
        """
        self.selectors = selectors

    def _get_tree(self, html_content: str):
        """lxml 트리 생성"""
        return html.fromstring(html_content)

    def _find_element(
        self,
        tree,
        selector_type: str,
        selector_value: str,
    ):
        """단일 요소 찾기"""
        elements = []
        if selector_type == "xpath":
            elements = tree.xpath(selector_value)
        elif selector_type == "css":
            elements = CSSSelector(selector_value)(tree)

        if elements and isinstance(elements[0], html.HtmlElement):
            return elements[0]
        return None

    def parse_goods_nos(
        self,
        html_content: str,
        selector_config: Dict,
    ) -> Set[str]:
        """
        리스트 페이지에서 goodsNo 추출

        Returns:
            goodsNo 집합
        """
        tree = self._get_tree(html_content)
        goods_nos = set()

        selector_type = selector_config.get("type")
        selector_value = selector_config.get("selector")
        extract_attr = selector_config.get("extract_attribute")

        if not selector_type or not selector_value:
            return goods_nos

        elements = []
        if selector_type == "xpath":
            elements = tree.xpath(selector_value)
        elif selector_type == "css":
            elements = CSSSelector(selector_value)(tree)

        for element in elements:
            if extract_attr:
                attr_value = element.get(extract_attr)
                if attr_value:
                    match = re.search(r"common\.link\.goodsDeatil\('([^']+)'\)", attr_value)
                    if match:
                        goods_nos.add(match.group(1))

        return goods_nos

    def get_total_count(
        self,
        html_content: str,
        selector_config: Dict,
    ) -> Optional[int]:
        """총 차량 대수 추출"""
        tree = self._get_tree(html_content)
        element = self._find_element(
            tree,
            selector_config.get("type"),
            selector_config.get("selector"),
        )

        if element is not None:
            text = element.text_content().strip()
            numbers = re.findall(r"\d+", text)
            if numbers:
                return int(numbers[0])
        return None

    def parse_detail_page(self, html_content: str) -> Dict[str, Any]:
        """
        상세 페이지 데이터 추출

        Returns:
            추출된 데이터 딕셔너리
        """
        tree = self._get_tree(html_content)
        extracted = {}

        for key, config in self.selectors.items():
            # 필터 관련 셀렉터 스킵
            if key.startswith("fuel_type_filter_") or key.startswith("applied_filter_"):
                continue

            selector_type = config.get("type")
            selector_value = config.get("selector")
            extract_method = config.get("extract_method")
            extract_attr = config.get("extract_attribute")
            clean_regex = config.get("clean_regex")

            # list_key_value 처리 (기본 정보 리스트)
            if extract_method == "list_key_value":
                self._parse_key_value_list(tree, selector_type, selector_value, extracted)
                continue

            # 일반 셀렉터 처리
            elements = []
            if selector_type == "xpath":
                elements = tree.xpath(selector_value)
            elif selector_type == "css":
                elements = CSSSelector(selector_value)(tree)

            element = elements[0] if elements else None
            value = self._extract_value(element, extract_method, extract_attr, clean_regex)
            extracted[key] = value

        return extracted

    def _parse_key_value_list(
        self,
        tree,
        selector_type: str,
        selector_value: str,
        extracted: Dict,
    ):
        """기본 정보 리스트 파싱"""
        elements = []
        if selector_type == "xpath":
            elements = tree.xpath(selector_value)
        elif selector_type == "css":
            elements = CSSSelector(selector_value)(tree)

        key_mapping = {
            "최초등록": "first_registration_date",
            "주행거리": "current_mileage_km",
            "연료": "fuel_type",
            "배기량": "displacement_cc",
            "외관컬러": "exterior_color",
            "내장컬러": "interior_color",
            "차종": "vehicle_type",
            "승차인원": "seating_capacity",
            "구동방식": "drivetrain",
            "차량번호": "vehicle_number",
            "연식": "year",
            "변속기": "transmission_type",
        }

        numeric_fields = {"current_mileage_km", "displacement_cc", "seating_capacity", "year"}

        for li in elements:
            try:
                title_els = li.xpath("./span[@class='tit']")
                value_els = li.xpath("./span[@class='txt']")

                title = title_els[0].text_content().strip() if title_els else None
                value = value_els[0].text_content().strip() if value_els else None

                if title and value and title in key_mapping:
                    mapped_key = key_mapping[title]
                    if mapped_key in numeric_fields:
                        extracted[mapped_key] = re.sub(r"[^0-9]", "", value)
                    else:
                        extracted[mapped_key] = value
            except Exception:
                continue

    def _extract_value(
        self,
        element,
        method: str,
        attr: Optional[str],
        clean_regex: Optional[str],
    ) -> Any:
        """값 추출"""
        if element is None:
            if method == "exists":
                return False
            return None

        value = None

        if method == "text":
            if isinstance(element, html.HtmlElement):
                value = element.text_content().strip()
            else:
                value = str(element).strip()

            if clean_regex:
                value = re.sub(clean_regex, "", value).strip()

        elif method == "attribute" and attr:
            if isinstance(element, html.HtmlElement):
                value = element.get(attr)

        elif method == "exists":
            value = True

        return value
