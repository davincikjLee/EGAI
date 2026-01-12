"""
Preprocessing Module - 전처리 (MLOps Stage 2)

오디오 → 스펙트로그램 변환:
    - STFT + HPSS
    - 주파수 범위 제한 (0~6kHz)
    - dB 변환 및 정규화
"""

from egai.preprocessing.audio import AudioPreprocessor

__all__ = [
    "AudioPreprocessor",
]
