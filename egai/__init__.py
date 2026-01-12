"""
EGAI - Engine Grading AI

MLOps 파이프라인 기반 엔진 품질 진단 AI 패키지

Pipeline Stages:
    1. collect  - 데이터 수집 (크롤링)
    2. process  - 전처리 (스펙트로그램)
    3. train    - 모델 학습
    4. evaluate - 평가
    5. serve    - 추론 서빙

Structure:
    egai/
    ├── data/            # 데이터 수집 및 관리 (MLOps Stage 1)
    │   ├── crawler/     # 웹 크롤러
    │   ├── loader.py    # 데이터 로더
    │   └── manager.py   # 데이터 관리
    ├── preprocessing/   # 전처리 (MLOps Stage 2)
    ├── models/          # 모델 정의
    ├── training/        # 학습 (MLOps Stage 3)
    ├── evaluation/      # 평가 (MLOps Stage 4)
    ├── serving/         # 추론 (MLOps Stage 5)
    └── pipelines/       # 파이프라인 오케스트레이션
"""

__version__ = "0.3.0"
__author__ = "EGAI Team"
