"""
이상탐지 모델 학습 스크립트

사용법:
    python scripts/train_anomaly.py
    python scripts/train_anomaly.py --beta 0.3 --epochs 50
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from egai.pipelines.anomaly_training import AnomalyTrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="이상탐지 VAE 모델 학습")

    parser.add_argument(
        "--epochs", type=int, default=100,
        help="학습 에포크 수 (기본: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="배치 크기 (기본: 32)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0005,
        help="학습률 (기본: 0.0005)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="VAE β 가중치 (기본: 0.5)"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128,
        help="잠재 공간 차원 (기본: 128)"
    )
    parser.add_argument(
        "--min-score", type=float, default=4.0,
        help="정상 데이터 최소 점수 (기본: 4.0)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="데이터 디렉토리 (기본: data)"
    )
    parser.add_argument(
        "--experiments-dir", type=str, default="experiments",
        help="실험 저장 디렉토리 (기본: experiments)"
    )
    parser.add_argument(
        "--regression-model", type=str, default=None,
        help="기존 회귀 모델 경로 (MC Dropout용)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="실험 이름"
    )

    args = parser.parse_args()

    # 설정 (fuel_type=None으로 전체 데이터 사용)
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "latent_dim": args.latent_dim,
        "min_score": args.min_score,
        "target_shape": (128, 128),
        "fmax": 6000,
        "use_cache": True,
        "fuel_type": None,  # 전체 데이터 사용 (가솔린 + 디젤)
        "validation_split": 0.2,
    }

    print("=" * 60)
    print("EGAI 이상탐지 VAE 모델 학습")
    print("=" * 60)
    print(f"\n[설정]")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  β (KL Weight): {args.beta}")
    print(f"  Latent Dim: {args.latent_dim}")
    print(f"  Min Score: {args.min_score}")
    print(f"  Data Dir: {args.data_dir}")

    # 파이프라인 실행
    pipeline = AnomalyTrainingPipeline(
        config=config,
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
    )

    results = pipeline.run(
        experiment_name=args.name,
        regression_model_path=args.regression_model,
        tags=["anomaly", "vae", f"beta_{args.beta}"],
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
