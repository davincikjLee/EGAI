"""
Domain Services - 비즈니스 로직

프레임워크 독립적인 도메인 규칙 및 해석
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
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


@dataclass
class AnomalyResult:
    """
    이상 탐지 결과

    Attributes:
        anomaly_score: 종합 이상 점수 (높을수록 이상)
        vae_score: VAE 재구성 오차 + KL Divergence
        mc_uncertainty: MC Dropout 불확실성
        score_inconsistency: 5개 점수 간 불일치 정도
        status: 상태 ("정상", "주의", "점검 권장")
        confidence: 판정 신뢰도 (0~1)
        threshold: 사용된 임계값
    """
    anomaly_score: float
    vae_score: float
    mc_uncertainty: float
    score_inconsistency: float
    status: str
    confidence: float
    threshold: Optional[float] = None


class AnomalyDetector:
    """
    이상 탐지 서비스

    3가지 방법의 앙상블:
    1. VAE 재구성 오차 + KL Divergence
    2. MC Dropout 예측 불확실성
    3. 5개 점수 간 불일치 감지

    최종 점수 = 0.5 * VAE + 0.3 * MC_Dropout + 0.2 * 불일치
    """

    # 앙상블 가중치 (전문가 패널 합의)
    WEIGHT_VAE = 0.5
    WEIGHT_MC_DROPOUT = 0.3
    WEIGHT_INCONSISTENCY = 0.2

    # 기본 임계값 (새 차 데이터로 조정 예정)
    DEFAULT_THRESHOLD_NORMAL = 0.3      # 정상 기준
    DEFAULT_THRESHOLD_CAUTION = 0.6     # 주의 기준
    DEFAULT_THRESHOLD_CHECK = 0.8       # 점검 권장 기준

    def __init__(
        self,
        vae_model=None,
        regression_model=None,
        threshold_normal: float = None,
        threshold_caution: float = None,
        threshold_check: float = None,
    ):
        """
        초기화

        Args:
            vae_model: 학습된 VAE 모델
            regression_model: 학습된 회귀 모델 (Multi-head CBAM)
            threshold_*: 판정 임계값 (None이면 기본값 사용)
        """
        self.vae_model = vae_model
        self.regression_model = regression_model
        self.threshold_normal = threshold_normal or self.DEFAULT_THRESHOLD_NORMAL
        self.threshold_caution = threshold_caution or self.DEFAULT_THRESHOLD_CAUTION
        self.threshold_check = threshold_check or self.DEFAULT_THRESHOLD_CHECK

        # 정규화를 위한 통계값 (학습 데이터에서 계산)
        self.vae_mean = None
        self.vae_std = None
        self.mc_mean = None
        self.mc_std = None

    def detect(
        self,
        spectrogram: np.ndarray,
        n_mc_samples: int = 30,
    ) -> AnomalyResult:
        """
        이상 탐지 수행

        Args:
            spectrogram: 4채널 스펙트로그램 (H, W, 4) 또는 (1, H, W, 4)
            n_mc_samples: MC Dropout 샘플 수

        Returns:
            AnomalyResult: 이상 탐지 결과
        """
        # 배치 차원 추가
        if spectrogram.ndim == 3:
            spectrogram = np.expand_dims(spectrogram, axis=0)

        # 1. VAE 이상 점수
        vae_score = self._compute_vae_score(spectrogram)

        # 2. MC Dropout 불확실성
        mc_uncertainty, predictions = self._compute_mc_uncertainty(
            spectrogram, n_mc_samples
        )

        # 3. 점수 불일치
        score_inconsistency = self._compute_score_inconsistency(predictions)

        # 정규화 (0~1 범위로)
        vae_normalized = self._normalize(vae_score, self.vae_mean, self.vae_std)
        mc_normalized = self._normalize(mc_uncertainty, self.mc_mean, self.mc_std)
        inconsistency_normalized = score_inconsistency  # 이미 0~1

        # 앙상블 점수
        anomaly_score = (
            self.WEIGHT_VAE * vae_normalized
            + self.WEIGHT_MC_DROPOUT * mc_normalized
            + self.WEIGHT_INCONSISTENCY * inconsistency_normalized
        )

        # 상태 판정
        status = self._determine_status(anomaly_score)

        # 신뢰도 계산
        confidence = self._compute_confidence(anomaly_score)

        return AnomalyResult(
            anomaly_score=float(anomaly_score),
            vae_score=float(vae_score),
            mc_uncertainty=float(mc_uncertainty),
            score_inconsistency=float(score_inconsistency),
            status=status,
            confidence=float(confidence),
            threshold=self.threshold_caution,
        )

    def _compute_vae_score(self, spectrogram: np.ndarray) -> float:
        """VAE 이상 점수 계산"""
        if self.vae_model is None:
            return 0.0

        score = self.vae_model.compute_anomaly_score(spectrogram)
        return float(score.numpy()[0])

    def _compute_mc_uncertainty(
        self,
        spectrogram: np.ndarray,
        n_samples: int = 30,
    ) -> Tuple[float, np.ndarray]:
        """
        MC Dropout 불확실성 계산

        정상: 예측 분산 낮음
        이상: 예측 분산 높음 (모델이 "혼란스러움")

        Args:
            spectrogram: 입력 데이터
            n_samples: Monte Carlo 샘플 수

        Returns:
            uncertainty: 평균 표준편차
            mean_predictions: 평균 예측값 (5개 점수)
        """
        if self.regression_model is None:
            return 0.0, np.zeros(5)

        predictions = []
        for _ in range(n_samples):
            # training=True로 Dropout 활성화
            pred = self.regression_model(spectrogram, training=True)
            predictions.append(pred.numpy())

        predictions = np.array(predictions)  # (n_samples, batch, 5)

        # 평균 예측
        mean_pred = np.mean(predictions, axis=0)[0]  # (5,)

        # 표준편차 (불확실성)
        std_pred = np.std(predictions, axis=0)[0]  # (5,)

        # 전체 불확실성 = 5개 점수의 평균 표준편차
        uncertainty = float(np.mean(std_pred))

        return uncertainty, mean_pred

    def _compute_score_inconsistency(self, predictions: np.ndarray) -> float:
        """
        점수 불일치 계산

        정상: 5개 점수 비슷 (대부분 4-5점)
        이상: 점수 간 불일치 (일부만 낮음)

        예: [4.5, 4.2, 4.8, 4.3, 2.1] → 불일치 높음 → 의심

        Args:
            predictions: 5개 점수 예측값

        Returns:
            inconsistency: 불일치 정도 (0~1, 높을수록 불일치)
        """
        if len(predictions) == 0:
            return 0.0

        mean_score = np.mean(predictions)
        if mean_score == 0:
            return 0.0

        # 변동계수 (CV) = std / mean
        cv = np.std(predictions) / mean_score

        # CV를 0~1로 정규화 (CV가 0.5 이상이면 매우 불일치)
        inconsistency = min(1.0, cv * 2)

        return float(inconsistency)

    def _normalize(
        self,
        value: float,
        mean: Optional[float],
        std: Optional[float],
    ) -> float:
        """Z-score 정규화 후 0~1 변환"""
        if mean is None or std is None or std == 0:
            return min(1.0, max(0.0, value))

        z_score = (value - mean) / std
        # Z-score를 0~1로 변환 (sigmoid-like)
        normalized = 1 / (1 + np.exp(-z_score))
        return float(normalized)

    def _determine_status(self, score: float) -> str:
        """점수 기반 상태 판정"""
        if score < self.threshold_normal:
            return "정상"
        elif score < self.threshold_caution:
            return "주의"
        else:
            return "점검 권장"

    def _compute_confidence(self, score: float) -> float:
        """판정 신뢰도 계산"""
        # 경계에서 멀수록 높은 신뢰도
        distances = [
            abs(score - self.threshold_normal),
            abs(score - self.threshold_caution),
        ]
        min_distance = min(distances)

        # 거리가 0.1 이상이면 높은 신뢰도
        confidence = min(1.0, min_distance / 0.1)
        return confidence

    def set_thresholds_from_new_car_data(
        self,
        new_car_scores: np.ndarray,
        margin_factor: float = 1.5,
    ) -> dict:
        """
        새 차 데이터 기반 임계값 설정

        새 차 = 확실한 정상의 기준점
        threshold = max(mean + 3*std, max * margin)

        Args:
            new_car_scores: 새 차 이상 점수 배열
            margin_factor: 최대값 대비 마진

        Returns:
            설정된 임계값 정보
        """
        mean_score = np.mean(new_car_scores)
        std_score = np.std(new_car_scores)
        max_score = np.max(new_car_scores)

        # 정상 임계값: mean + 2*std (약 95% 커버)
        self.threshold_normal = float(mean_score + 2 * std_score)

        # 주의 임계값: mean + 3*std (약 99.7% 커버)
        threshold_3sigma = mean_score + 3 * std_score
        threshold_margin = max_score * margin_factor
        self.threshold_caution = float(max(threshold_3sigma, threshold_margin))

        # 점검 권장: 주의 임계값 * 1.5
        self.threshold_check = float(self.threshold_caution * 1.5)

        return {
            "new_car_mean": float(mean_score),
            "new_car_std": float(std_score),
            "new_car_max": float(max_score),
            "threshold_normal": self.threshold_normal,
            "threshold_caution": self.threshold_caution,
            "threshold_check": self.threshold_check,
        }

    def set_normalization_stats(
        self,
        vae_scores: np.ndarray,
        mc_uncertainties: np.ndarray,
    ):
        """
        정규화 통계값 설정 (학습 데이터 기반)

        Args:
            vae_scores: 학습 데이터의 VAE 점수 배열
            mc_uncertainties: 학습 데이터의 MC 불확실성 배열
        """
        self.vae_mean = float(np.mean(vae_scores))
        self.vae_std = float(np.std(vae_scores))
        self.mc_mean = float(np.mean(mc_uncertainties))
        self.mc_std = float(np.std(mc_uncertainties))
