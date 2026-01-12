"""
EGAI - Engine Grading AI

Clean Architecture 기반 엔진 품질 진단 AI 패키지

구조:
    egai/
    ├── domain/          # 비즈니스 로직 (프레임워크 독립)
    │   ├── entities.py  # 엔티티 정의
    │   └── services.py  # 도메인 서비스
    ├── infrastructure/  # 외부 의존성 (TensorFlow, librosa)
    │   ├── preprocessing.py   # 오디오 전처리
    │   ├── models.py          # 딥러닝 모델
    │   └── data_loader.py     # 데이터 로딩
    └── application/     # 유즈케이스
        ├── train.py     # 학습
        ├── predict.py   # 추론
        └── evaluate.py  # 평가
"""

__version__ = "0.2.0"
__author__ = "EGAI Team"

from egai.domain.entities import AudioSample, PredictionResult
from egai.application.predict import EngineGrader

__all__ = [
    "AudioSample",
    "PredictionResult",
    "EngineGrader",
]
