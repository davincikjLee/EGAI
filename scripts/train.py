"""
회귀 모델 학습 스크립트

사용법:
    python scripts/train.py
    python scripts/train.py --model 4channel_cbam --folds 5 --epochs 50
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from egai.pipelines.training import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="회귀 모델 학습")

    parser.add_argument(
        "--model", type=str, default="4channel_cbam",
        choices=["simple_cbam", "4channel_cbam", "multihead_cbam"],
        help="모델 타입 (기본: 4channel_cbam)"
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="K-Fold 수 (기본: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="학습 에포크 수 (기본: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="배치 크기 (기본: 32)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001,
        help="학습률 (기본: 0.001)"
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
        "--fuel-type", type=str, default=None,
        choices=["gasoline", "diesel", None],
        help="연료 타입 필터 (기본: None - 전체)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="캐시 사용 안 함"
    )

    args = parser.parse_args()

    # 설정 구성
    config = {
        "model_type": args.model,
        "n_folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "target_shape": (128, 128),
        "fmax": 6000,
        "use_cache": not args.no_cache,
        "fuel_type": args.fuel_type,
    }

    print("=" * 60)
    print("회귀 모델 학습 시작")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"Folds: {args.folds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Fuel Type: {args.fuel_type or 'All'}")
    print("=" * 60)

    # 파이프라인 실행
    pipeline = TrainingPipeline(
        config=config,
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
    )

    results = pipeline.run(
        experiment_name=f"{args.model}_{args.folds}fold",
        tags=[args.model, f"{args.folds}-fold", "regression"],
    )

    # 최고 모델 저장
    model_path = pipeline.save_best_model()

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"전체 MAE: {results['overall_MAE_mean']:.4f} ± {results['overall_MAE_std']:.4f}")
    print(f"최고 모델: {model_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
