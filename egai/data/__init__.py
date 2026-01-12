"""
Data Module - 데이터 수집 및 관리 (MLOps Stage 1)

크롤링 → 저장 → 로딩 파이프라인:
    1. crawler/: 웹 크롤러 (Selenium 기반)
    2. manager.py: 데이터 저장/관리
    3. loader.py: 학습용 데이터 로더
"""

from egai.data.manager import DataManager
from egai.data.loader import AudioDataLoader

__all__ = [
    "DataManager",
    "AudioDataLoader",
]
