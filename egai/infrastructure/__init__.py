"""
Infrastructure Layer - 외부 의존성

TensorFlow, librosa 등 외부 라이브러리 래핑
- preprocessing: 오디오 전처리 (librosa)
- models: 딥러닝 모델 (TensorFlow/Keras)
- data_loader: 데이터 로딩 (pandas, numpy)
"""

from egai.infrastructure.preprocessing import AudioPreprocessor
from egai.infrastructure.models import ModelFactory

__all__ = [
    "AudioPreprocessor",
    "ModelFactory",
]
