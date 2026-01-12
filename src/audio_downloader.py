# src/audio_downloader.py

import os
import requests
import time
from typing import Optional, Union, Tuple, Any


# WebScraper에서 사용하는 설정들을 AudioDownloader도 사용할 수 있도록 ConfigLoader 임포트
# 또는 AudioDownloader의 __init__에서 필요한 설정들을 직접 파라미터로 받을 수도 있음.
# 여기서는 필요한 설정들을 파라미터로 받도록 구현합니다.

class AudioDownloader:
    """
    오디오 파일을 다운로드하고 저장하는 클래스입니다.
    네트워크 관련 에러 핸들링 및 재시도 로직을 포함합니다.
    """

    def __init__(self, request_delay: Union[int, float], timeout: int, max_retries: int, retry_delay: Union[int, float],
                 user_agent: str):
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {'User-Agent': user_agent}

    def download_audio_file(self, audio_url: str, save_path: str) -> bool:
        """
        주어진 URL에서 오디오 파일(MP3)을 다운로드하여 지정된 경로에 저장합니다.
        """
        if not audio_url:
            print("    [경고] 다운로드할 오디오 URL이 유효하지 않습니다.")
            return False

        for attempt in range(self.max_retries):
            try:
                print(f"    오디오 파일 다운로드 시도: {audio_url} (시도: {attempt + 1}/{self.max_retries})")
                time.sleep(self.request_delay)  # 요청 간 지연

                response = requests.get(audio_url, headers=self.headers, timeout=self.timeout, stream=True)
                response.raise_for_status()  # HTTP 오류 발생 시 예외 throw (4xx, 5xx)

                # 파일 저장 경로의 디렉토리가 없으면 생성 (DataManager가 주로 하지만, 여기서도 방어적으로)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True  # 다운로드 성공

            except requests.exceptions.RequestException as e:
                print(f"    오디오 파일 다운로드 실패 (시도 {attempt + 1}/{self.max_retries}) for {audio_url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)  # 재시도 대기
                else:
                    print(f"    최대 재시도 횟수 도달. {audio_url} 다운로드 최종 실패.")
                    return False  # 다운로드 실패
            except Exception as e:
                print(f"    오디오 파일 저장 중 예상치 못한 오류 발생: {e}")
                return False
        return False  # 이 부분에 도달해서는 안 되지만, 명시적으로 False 반환

    def extract_audio_url_from_vr_page(self, html_content: str, selector_info: dict) -> Optional[str]:
        """
        VR 페이지 (상세 페이지 내) HTML에서 오디오 URL을 추출합니다.
        PageParser의 역할과 일부 중복될 수 있으나, AudioDownloader가 오디오 URL 추출도 담당하도록 정의.
        """
        # 이 메서드는 config.json의 audio_url_on_page 셀렉터에 해당
        # PageParser에서 이미 해당 URL을 추출하도록 했으므로, 이 메서드는 현재는 직접 사용하지 않을 수 있음.
        # (extracted_data['audio_url_on_page']에 이미 URL이 있을 것이므로)
        # 하지만, 향후 AudioDownloader가 독립적으로 오디오 URL을 파싱해야 할 경우를 대비하여 남겨둠.

        # PageParser에서 이미 추출된 'audio_url_on_page' 값을 MainCrawler에서 전달받아 사용하는 것이 더 효율적입니다.
        # 이 메서드는 현재 크롤링 플로우에서 직접 호출되지 않습니다.
        print("경고: AudioDownloader.extract_audio_url_from_vr_page는 현재 사용되지 않습니다. PageParser에서 URL이 추출됩니다.")
        return None