import json
import os
import copy
from typing import Type, Union, Tuple, Any


class ConfigLoader:
    def __init__(self, config_path='config/crawler_config.json', default_config_path='config/default_config.json'):
        self.config_path = config_path
        self.default_config_path = default_config_path
        self._config = None

    def load_config(self):
        if not os.path.exists(self.config_path):
            print(f"경고: 설정 파일 '{self.config_path}'이(가) 없습니다. 기본 설정으로 새 파일을 생성합니다.")
            self._create_config_from_default()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            self._validate_config()
            print(f"'{self.config_path}' 설정 파일을 성공적으로 로드했습니다.")
            return copy.deepcopy(self._config)
        except (json.decoder.JSONDecodeError, ValueError) as e:
            print(f"경고: '{self.config_path}' 설정 파일이 유효하지 않습니다. ({e}) 기본 설정으로 새 파일을 생성합니다.")
            self._create_config_from_default()

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            self._validate_config()
            print(f"기본 설정으로 '{self.config_path}' 파일을 재생성하고 로드했습니다.")
            return copy.deepcopy(self._config)

    def get(self, key_path: str, expected_type: Union[Type, Tuple[Type, ...]] = None, default: Any = None) -> Any:
        """
        점(.)으로 구분된 키 경로를 사용하여 설정 값을 가져옵니다.
        키를 찾을 수 없거나 타입 캐스팅 실패 시, default 값이 주어지면 default를 반환합니다.
        default 값이 없고 유효하지 않으면 ValueError를 발생시킵니다.
        """
        if self._config is None:
            raise RuntimeError("오류: 설정 파일이 아직 로드되지 않았습니다. load_config()를 먼저 호출하세요.")

        keys = key_path.split('.')
        current_data_node = self._config

        try:
            for i, key in enumerate(keys):
                if not isinstance(current_data_node, dict):
                    raise ValueError(
                        f"경로 '{key_path}' 탐색 중 오류: '{'.'.join(keys[:i])}'의 값 '{current_data_node}'이(가) 딕셔너리가 아닙니다. (타입: {type(current_data_node)})")

                if key not in current_data_node:
                    raise ValueError(f"필수 키 '{key}'이(가) '{'.'.join(keys[:i])}' 경로에 누락되었습니다. (전체 경로: {key_path})")

                current_data_node = current_data_node[key]

            final_value = current_data_node

            if final_value is None:
                if expected_type is not None and isinstance(expected_type, tuple) and type(None) in expected_type:
                    return None
                else:
                    raise ValueError(f"경로 '{key_path}'의 값이 'null'(None)입니다. 'null' 값은 허용되지 않습니다.")

            if expected_type is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(final_value, expected_type):
                        raise TypeError(
                            f"'{key_path}' 값 '{final_value}' ({type(final_value)})이(가) 예상 타입 {expected_type} 중 하나가 아닙니다.")
                else:
                    if not isinstance(final_value, expected_type):
                        if expected_type is int and isinstance(final_value, float):
                            final_value = int(final_value)
                        elif expected_type is bool and isinstance(final_value, str):
                            if final_value.lower() == 'true':
                                final_value = True
                            elif final_value.lower() == 'false':
                                final_value = False
                            else:
                                raise TypeError("Boolean 문자열 변환 실패")
                        else:
                            raise TypeError(
                                f"'{key_path}' 값 '{final_value}' ({type(final_value)})의 형식이 올바르지 않습니다. 예상: {expected_type}, 실제: {type(final_value)}")

                    if not isinstance(final_value, expected_type):
                        final_value = expected_type(final_value)

            return final_value

        except (ValueError, TypeError) as e:
            if default is not None:
                print(f"경고(ConfigLoader.get): 설정 경로 '{key_path}' 접근/캐스팅 중 오류 발생: {e}. 기본값 '{default}' 반환.")
                return default
            raise ValueError(f"오류(ConfigLoader.get): 설정 경로 '{key_path}' 접근/캐스팅 중 치명적인 오류 발생: {e}")

    def _validate_config(self):
        """
        로드된 설정의 필수 키 존재 여부 및 기본 구조를 검사합니다.
        누락된 필수 키가 있거나 형식이 올바르지 않거나 값이 None (JSON null)일 경우 ValueError를 발생시킵니다.
        """

        def _check_value(config_dict: dict, key_path_full: str, expected_type: Union[Type, Tuple[Type, ...]]):
            keys = key_path_full.split('.')
            current_value = config_dict

            for i, key in enumerate(keys):
                if not isinstance(current_value, dict):
                    raise ValueError(
                        f"유효성 검사 오류: 경로 '{'.'.join(keys[:i])}'의 '{key}'는 딕셔너리가 아닙니다. 실제 타입: {type(current_value)} (경로: {key_path_full})")

                if key not in current_value:
                    raise ValueError(
                        f"유효성 검사 오류: 필수 키 '{key}'이(가) '{'.'.join(keys[:i])}' 경로에 누락되었습니다. (전체 경로: {key_path_full})")

                current_value = current_value[key]

            if current_value is None:
                if isinstance(expected_type, tuple) and type(None) in expected_type:
                    return None
                else:
                    raise ValueError(
                        f"유효성 검사 오류: '{key_path_full}' 키의 값이 'null'(None)입니다. 유효한 값을 지정해야 합니다. (예상: {expected_type})")

            if isinstance(expected_type, tuple):
                if not isinstance(current_value, expected_type):
                    raise ValueError(
                        f"유효성 검사 오류: '{key_path_full}' 값 '{current_value}' ({type(current_value)})의 형식이 올바르지 않습니다. 예상: {expected_type}")
                return current_value
            else:
                try:
                    if expected_type is bool:
                        if isinstance(current_value, str):
                            if current_value.lower() == 'true': return True
                            if current_value.lower() == 'false': return False
                            raise ValueError(
                                f"유효성 검사 오류: '{key_path_full}' 값 '{current_value}'을(를) boolean으로 변환할 수 없습니다.")
                        return bool(current_value)
                    elif expected_type is int and isinstance(current_value, float):
                        return int(current_value)
                    else:
                        if not isinstance(current_value, expected_type):
                            raise ValueError(
                                f"유효성 검사 오류: '{key_path_full}' 값 '{current_value}' ({type(current_value)})의 형식이 올바르지 않습니다. 예상: {expected_type}")
                        return expected_type(current_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"유효성 검사 오류: '{key_path_full}' 값 '{current_value}' ({type(current_value)})을(를) "
                                     f"예상 타입 {expected_type}으로 캐스팅 실패: {e}")

        self._config['crawler_settings'] = _check_value(self._config, "crawler_settings", dict)
        required_crawler_settings = {
            "user_agent": str, "request_delay_sec": (int, float), "timeout_sec": int,
            "max_retries": int, "retry_delay_sec": (int, float), "proxy_enabled": bool, "proxy_list": list,
            "use_selenium": bool,
            "selenium_driver_path": (str, type(None)),
            "use_auto_driver_download": bool,
            "selenium_headless": bool, "scroll_load_limit": int
        }
        for key, expected_type in required_crawler_settings.items():
            _check_value(self._config, f"crawler_settings.{key}", expected_type)

        self._config['urls'] = _check_value(self._config, "urls", dict)
        required_url_patterns = [
            "base_url", "list_page_pattern", "detail_page_pattern", "vr_page_pattern"
        ]
        for key in required_url_patterns:
            pattern_val = _check_value(self._config, f"urls.{key}", str)
            # 'list_page_pattern'에 대한 플레이스홀더 경고를 제거합니다.
            # 이 사이트는 동적 로딩이므로 URL 패턴에 플레이스홀더가 없습니다.
            if key in ["detail_page_pattern", "vr_page_pattern"] and ('{' not in pattern_val or '}' not in pattern_val):
                print(f"경고: 'urls.{key}' 패턴에 플레이스홀더 '{{...}}'가 없습니다. 동적 URL 생성에 문제가 있을 수 있습니다.")
            elif key == "list_page_pattern" and ('{' in pattern_val or '}' in pattern_val):
                print(f"경고: 'urls.{key}' 패턴에 플레이스홀더 '{{...}}'가 있습니다. 이 페이지는 동적 로딩이므로 URL에 페이지 번호가 포함되지 않을 수 있습니다.")

        required_url_selectors = [
            "next_page_selector", "total_count_selector", "goods_no_selector"
        ]
        for key in required_url_selectors:
            selector_config = _check_value(self._config, f"urls.{key}", dict)
            _check_value(selector_config, "type", str)
            _check_value(selector_config, "selector", str)
            # item_check_selector는 next_page_selector에 선택적으로 포함되므로 여기서 필수로 검사하지 않음

        # item_check_selector가 next_page_selector 안에 있다면 그 값을 검사
        next_page_sel = self._config['urls'].get('next_page_selector')
        if isinstance(next_page_sel, dict) and 'item_check_selector' in next_page_sel:
            item_check_sel_info = next_page_sel.get('item_check_selector')
            if isinstance(item_check_sel_info, dict):
                _check_value(item_check_sel_info, "type", str)
                _check_value(item_check_sel_info, "selector", str)
            elif isinstance(item_check_sel_info, str):  # string도 허용
                pass  # string이면 그냥 통과
            else:
                raise ValueError(f"유효성 검사 오류: 'urls.next_page_selector.item_check_selector'의 형식이 올바르지 않습니다.")

                # 3. data_selectors 검사 (업데이트)
                self._config['data_selectors'] = _check_value(self._config, "data_selectors", dict)

                # 각 데이터 셀렉터 항목의 필수 필드 검사
                for key, sel_info in self._config['data_selectors'].items():
                    # sel_info 자체가 dict 타입이어야 함
                    _check_value(self._config['data_selectors'], key,
                                 dict)  # 예를 들어 config['data_selectors']['vehicle_name']

                    # 모든 셀렉터는 'type', 'selector', 'extract_method'를 가져야 함
                    _check_value(sel_info, "type", str)
                    _check_value(sel_info, "selector", str)
                    _check_value(sel_info, "extract_method", str)  # extract_method는 일단 string으로 존재해야 함

                    # 'list_key_value'는 특정 형태의 추출 방식이므로, 해당 셀렉터에만 적용
                    if key == "base_info_list":
                        if sel_info.get("extract_method") != "list_key_value":
                            raise ValueError(
                                f"유효성 검사 오류: 'data_selectors.base_info_list'의 'extract_method'는 'list_key_value'여야 합니다.")
                        # base_info_list는 id_value가 필요 없으므로 추가 검사하지 않음 (pass)
                    # 'fuel_type_filter_gasoline', 'fuel_type_filter_diesel' 등 id_value를 사용하는 셀렉터
                    elif key in ["fuel_type_filter_gasoline", "fuel_type_filter_diesel"]:
                        if "id_value" not in sel_info:
                            raise ValueError(
                                f"유효성 검사 오류: 'data_selectors.{key}' 셀렉터에 'id_value'가 누락되었습니다. JavaScript 클릭에 필요합니다.")
                        _check_value(sel_info, "id_value", str)  # id_value가 문자열인지 확인
                    # 그 외 'extract_attribute'가 필요한 경우 검사 추가
                    elif sel_info.get("extract_method") == "attribute":
                        if "extract_attribute" not in sel_info:
                            raise ValueError(
                                f"유효성 검사 오류: 'data_selectors.{key}' 셀렉터에 'extract_method'가 'attribute'이지만 'extract_attribute'가 누락되었습니다.")
                        _check_value(sel_info, "extract_attribute", str)
                    # 'clean_regex'나 'invert_boolean'은 선택적이므로 필수로 검사하지 않음

                print("설정 파일 유효성 검사를 통과했습니다.")

    def _create_config_from_default(self):
        """
        기본 설정 파일 (default_config.json)을 읽어와서
        사용자 설정 파일 (crawler_config.json)을 생성합니다.
        """
        config_dir = os.path.dirname(self.config_path)
        os.makedirs(config_dir, exist_ok=True)  # config 폴더가 없으면 생성

        if not os.path.exists(self.default_config_path):
            raise FileNotFoundError(
                f"오류: 기본 설정 파일 '{self.default_config_path}'이(가) 없습니다. "
                "프로젝트 구조를 확인하거나 수동으로 생성해주세요."
            )

        try:
            with open(self.default_config_path, 'r', encoding='utf-8') as f:
                default_content = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise ValueError(f"오류: 기본 설정 파일 '{self.default_config_path}'의 JSON 형식이 유효하지 않습니다: {e}")

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_content, f, indent=2, ensure_ascii=False)
        print(f"'{self.default_config_path}'에서 기본 설정을 가져와 '{self.config_path}' 파일을 생성했습니다. "
              "실제 웹사이트에 맞춰 내용을 수정해주세요.")


# 직접 실행 테스트
if __name__ == "__main__":
    # 현재 스크립트(config_loader.py)의 디렉토리 경로
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트 디렉토리 (src의 부모 디렉토리)
    project_root_dir = os.path.normpath(os.path.join(current_script_dir, '..'))

    # 실제 ConfigLoader에서 사용하는 경로와 동일하게 설정
    test_config_path = os.path.join(project_root_dir, 'config', 'crawler_config.json')  # 테스트 시 crawler_config.json 사용
    test_default_path = os.path.join(project_root_dir, 'config', 'default_config.json')  # 실제 default_config.json 사용

    # --- 테스트 시작 ---

    # 테스트 전에 기존 config 파일들 정리 (깨끗한 테스트 환경을 위해)
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    # default_config.json은 미리 수동으로 생성되어 있고 올바른 내용임을 전제합니다.
    # 만약 default_config.json이 없으면 첫 테스트에서 FileNotFoundError가 발생할 것입니다.

    print("\n--- 테스트 1: 사용자 설정 파일이 없을 때 default에서 생성되는지 테스트 ---")
    loader1 = ConfigLoader(test_config_path, test_default_path)
    try:
        config1 = loader1.load_config()
        print(f"로드된 base_url: {loader1.get('urls.base_url')}")
    except Exception as e:
        print(f"오류 발생: {e}")

    # Test 1 후 생성된 crawler_config.json을 읽어서 유효하지 않은 내용으로 변경
    if os.path.exists(test_config_path):
        with open(test_config_path, 'w', encoding='utf-8') as f:
            f.write("{invalid json content")  # 유효하지 않은 JSON 내용으로 덮어쓰기

    print("\n--- 테스트 2: 유효하지 않은 JSON 형식일 때 default에서 재생성되는지 테스트 ---")
    loader2 = ConfigLoader(test_config_path, test_default_path)
    try:
        config2 = loader2.load_config()
        print(f"로드된 request_delay_sec: {loader2.get('crawler_settings.request_delay_sec')}")
    except Exception as e:
        print(f"오류 발생: {e}")

    # Test 2 후 생성된 crawler_config.json을 읽어서 필수 키가 누락된 내용으로 변경
    if os.path.exists(test_config_path):
        try:
            with open(test_default_path, 'r', encoding='utf-8') as f:
                temp_default_content = json.load(f)
            temp_default_content['urls'].pop('list_page_pattern')  # 필수 키 제거
            with open(test_config_path, 'w', encoding='utf-8') as f:
                json.dump(temp_default_content, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"테스트 설정 파일 수정 실패: {e}")

    print("\n--- 테스트 3: 필수 키 누락될 때 (list_page_pattern 제거) ---")
    loader3 = ConfigLoader(test_config_path, test_default_path)
    try:
        config3 = loader3.load_config()
        print(f"로드된 list_page_pattern: {loader3.get('urls.list_page_pattern')}")
    except Exception as e:
        print(f"오류 발생: {e}")

    # 테스트 4: default_config.json 자체가 없을 때 오류 테스트
    print("\n--- 테스트 4: 기본 설정 파일이 없을 때 ---")
    # default_config.json을 임시로 삭제
    if os.path.exists(test_default_path):
        os.rename(test_default_path, test_default_path + '_bak')  # 파일 이름을 변경하여 임시 삭제 효과
    if os.path.exists(test_config_path):
        os.remove(test_config_path)

    loader4 = ConfigLoader(test_config_path, test_default_path)
    try:
        config4 = loader4.load_config()
    except Exception as e:
        print(f"예상된 오류 발생: {e}")
    finally:
        # 테스트 후 default_config.json 복원
        if os.path.exists(test_default_path + '_bak'):
            os.rename(test_default_path + '_bak', test_default_path)

    # 테스트 파일 최종 정리
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    print("\n테스트 완료 및 테스트 파일 정리.")