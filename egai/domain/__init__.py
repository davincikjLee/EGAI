"""
Domain Layer - 비즈니스 로직

프레임워크 독립적인 핵심 비즈니스 규칙 정의
- 엔티티: 핵심 데이터 구조
- 서비스: 도메인 규칙 및 계산
"""

from egai.domain.entities import AudioSample, PredictionResult, QualityScore
from egai.domain.services import ScoreInterpreter

__all__ = [
    "AudioSample",
    "PredictionResult",
    "QualityScore",
    "ScoreInterpreter",
]
