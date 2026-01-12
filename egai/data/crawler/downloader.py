"""
Audio Downloader - MP3 파일 다운로드

기능:
    - HTTP 스트리밍 다운로드
    - 재시도 로직
"""

import os
import time
import requests
from typing import Optional


class AudioDownloader:
    """
    오디오 파일 다운로더
    """

    def __init__(
        self,
        user_agent: str,
        request_delay: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.headers = {"User-Agent": user_agent}
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def download(
        self,
        url: str,
        save_path: str,
    ) -> bool:
        """
        오디오 파일 다운로드

        Args:
            url: 다운로드 URL
            save_path: 저장 경로

        Returns:
            성공 여부
        """
        if not url:
            return False

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)

                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    stream=True,
                )
                response.raise_for_status()

                # 디렉토리 생성
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 파일 저장
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return True

            except Exception as e:
                print(f"[Downloader] 다운로드 실패 ({attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return False

    def get_filename_from_url(self, url: str) -> str:
        """URL에서 파일명 추출"""
        filename = os.path.basename(url.split("?")[0])
        if not filename.lower().endswith((".mp3", ".wav", ".ogg")):
            filename += ".mp3"
        return filename
