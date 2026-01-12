"""
Domain Services - 비즈니스 로직

프레임워크 독립적인 도메인 규칙 및 해석
"""

from typing import List, Tuple
from egai.domain.entities import QualityScore, PredictionResult


class ScoreInterpreter:
    """
    품질 점수 해석 서비스

    현대자동차 리버스 엔지니어링 기반 점수 해석
    """

    # 점수 구간별 해석
    SCORE_LEVELS = {
        (4.5, 5.0): ("우수", "엔진 상태가 매우 양호합니다"),
        (4.0, 4.5): ("양호", "엔진 상태가 좋습니다"),
        (3.5, 4.0): ("보통", "일반적인 수준입니다"),
        (3.0, 3.5): ("주의", "점검을 권장합니다"),
        (0.0, 3.0): ("경고", "정밀 점검이 필요합니다"),
    }

    # 각 점수별 의미
    SCORE_MEANINGS = {
        "audable_range": "가청 범위 품질 (전반적 소음 수준)",
        "low_high_freq": "저/고주파 균형 (엔진 밸런스)",
        "mid_freq_score": "중주파 품질 (기계 마모 상태)",
        "regularity": "규칙성 (엔진 안정성)",
        "irregularity": "불규칙성 (이상 소음 발생)",
    }

    def interpret(self, result: PredictionResult) -> PredictionResult:
        """
        예측 결과에 해석 추가

        Args:
            result: 원본 예측 결과

        Returns:
            해석이 추가된 예측 결과
        """
        scores = result.scores
        overall = scores.overall

        # 전체 등급 결정
        level, description = self._get_level(overall)
        interpretation = f"[{level}] {description} (종합: {overall:.1f}점)"

        # 권장 사항 생성
        recommendations = self._generate_recommendations(scores)

        return PredictionResult(
            scores=scores,
            confidence=result.confidence,
            model_version=result.model_version,
            interpretation=interpretation,
            recommendations=recommendations,
        )

    def _get_level(self, score: float) -> Tuple[str, str]:
        """점수 구간별 등급 반환"""
        for (low, high), (level, desc) in self.SCORE_LEVELS.items():
            if low <= score < high:
                return level, desc
        return "경고", "정밀 점검이 필요합니다"

    def _generate_recommendations(self, scores: QualityScore) -> List[str]:
        """점수 기반 권장 사항 생성"""
        recommendations = []

        # 규칙성 낮음 (MAE가 높아 예측이 어려운 항목)
        if scores.regularity < 3.5:
            recommendations.append("엔진 회전 불균형 점검 필요")

        # 불규칙성 높음
        if scores.irregularity < 3.5:
            recommendations.append("이상 소음 원인 확인 권장")

        # 저/고주파 균형 이상
        if scores.low_high_freq < 3.5:
            recommendations.append("엔진 마운트 및 베어링 점검")

        # 중주파 품질 저하
        if scores.mid_freq_score < 3.5:
            recommendations.append("밸브 트레인 및 피스톤 상태 확인")

        # 가청 범위 이상
        if scores.audable_range < 4.0:
            recommendations.append("전반적인 소음 수준 점검")

        if not recommendations:
            recommendations.append("현재 엔진 상태 양호, 정기 점검 유지")

        return recommendations


class ModelAccuracyInfo:
    """
    모델 정확도 정보 서비스

    리버스 엔지니어링 결과 기반 예측 신뢰도 제공
    """

    # 각 점수별 MAE (5-Fold CV 결과)
    MAE_BY_SCORE = {
        "audable_range": 0.21,    # 쉬움
        "low_high_freq": 0.46,    # 보통
        "mid_freq_score": 0.48,   # 보통
        "regularity": 0.55,       # 어려움
        "irregularity": 0.71,     # 매우 어려움
    }

    # 전체 MAE
    OVERALL_MAE = 0.4819
    OVERALL_STD = 0.0225

    @classmethod
    def get_accuracy_info(cls) -> dict:
        """정확도 정보 반환"""
        return {
            "model": "Simple CNN + CBAM",
            "overall_mae": cls.OVERALL_MAE,
            "overall_std": cls.OVERALL_STD,
            "per_score_mae": cls.MAE_BY_SCORE,
            "baseline_improvement": "+14.5%",
            "training_samples": 594,
            "validation_method": "5-Fold Stratified CV",
        }

    @classmethod
    def get_confidence(cls, score_name: str) -> float:
        """점수별 예측 신뢰도 (0~1)"""
        mae = cls.MAE_BY_SCORE.get(score_name, 0.5)
        # MAE를 신뢰도로 변환 (낮을수록 높은 신뢰도)
        # MAE 0 → 1.0, MAE 1 → 0.0
        return max(0.0, min(1.0, 1.0 - mae))
