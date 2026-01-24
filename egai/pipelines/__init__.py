"""
Pipelines Module - MLOps 파이프라인 오케스트레이션

2개의 학습 파이프라인:
    1. TrainingPipeline: 회귀 모델 학습 (5개 품질 점수 예측)
    2. AnomalyTrainingPipeline: 이상탐지 VAE 학습 (OK/NG 판단)

공통 모듈:
    - ExperimentTracker: 실험 추적 및 기록
"""

from egai.pipelines.experiment import ExperimentTracker
from egai.pipelines.training import TrainingPipeline, run_experiment
from egai.pipelines.anomaly_training import AnomalyTrainingPipeline, run_anomaly_training

__all__ = [
    # 실험 추적
    "ExperimentTracker",
    # 파이프라인 1: 회귀 모델
    "TrainingPipeline",
    "run_experiment",
    # 파이프라인 2: 이상탐지 VAE
    "AnomalyTrainingPipeline",
    "run_anomaly_training",
]
