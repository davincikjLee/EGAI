"""
Domain Entities - 핵심 데이터 구조

프레임워크 독립적인 순수 Python 클래스
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np


@dataclass
class QualityScore:
    """
    엔진 품질 점수 (1~5점 척도)

    현대자동차 품질 평가 기준 리버스 엔지니어링 결과:
    - audable_range: 가청 범위 점수 (MAE 0.21, 쉬움)
    - low_high_freq: 저/고주파 균형 (MAE 0.46, 보통)
    - mid_freq_score: 중주파 품질 (MAE 0.48, 보통)
    - regularity: 규칙성 (MAE 0.55, 어려움)
    - irregularity: 불규칙성 (MAE 0.71, 매우 어려움)
    """
    audable_range: float
    low_high_freq: float
    mid_freq_score: float
    regularity: float
    irregularity: float

    @property
    def overall(self) -> float:
        """전체 점수 (5개 점수의 평균)"""
        return (
            self.audable_range +
            self.low_high_freq +
            self.mid_freq_score +
            self.regularity +
            self.irregularity
        ) / 5

    def to_dict(self) -> Dict[str, float]:
        """딕셔너리 변환"""
        return {
            "audable_range": self.audable_range,
            "low_high_freq": self.low_high_freq,
            "mid_freq_score": self.mid_freq_score,
            "regularity": self.regularity,
            "irregularity": self.irregularity,
            "overall": self.overall,
        }

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "QualityScore":
        """NumPy 배열에서 생성 (모델 출력 순서)"""
        return cls(
            low_high_freq=float(arr[0]),
            mid_freq_score=float(arr[1]),
            audable_range=float(arr[2]),
            regularity=float(arr[3]),
            irregularity=float(arr[4]),
        )


@dataclass
class AudioSample:
    """
    오디오 샘플 엔티티

    엔진 오디오 파일과 관련 메타데이터를 캡슐화
    """
    file_path: Path
    sample_rate: int = 22050
    duration: Optional[float] = None

    # 메타데이터 (선택적)
    vehicle_id: Optional[str] = None
    year: Optional[int] = None
    fuel_type: Optional[str] = None
    mileage_km: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

    @property
    def exists(self) -> bool:
        """파일 존재 여부"""
        return self.file_path.exists()

    @property
    def format(self) -> str:
        """파일 형식 (mp3, wav 등)"""
        return self.file_path.suffix.lower().lstrip(".")


@dataclass
class PredictionResult:
    """
    예측 결과 엔티티

    모델 예측 결과와 신뢰도 정보
    """
    scores: QualityScore
    confidence: float = 1.0
    model_version: str = "simple_cbam_v1"

    # 해석 정보
    interpretation: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "scores": self.scores.to_dict(),
            "overall_score": self.scores.overall,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "interpretation": self.interpretation,
            "recommendations": self.recommendations,
        }


@dataclass
class TrainingConfig:
    """
    학습 설정 엔티티
    """
    model_type: str = "simple_cbam"
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    n_folds: int = 5
    target_shape: tuple = (128, 128)

    # 전처리 설정
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    fmax: int = 6000
    use_mel: bool = False

    # 캐싱
    use_cache: bool = True
    cache_dir: Optional[Path] = None
