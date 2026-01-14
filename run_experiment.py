#!/usr/bin/env python
"""
EGAI 실험 실행 스크립트

사용법:
    python run_experiment.py --model simple_cbam --folds 5 --epochs 50
    python run_experiment.py --model simple --name baseline_test
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from egai.pipelines import TrainingPipeline, run_experiment


def main():
    parser = argparse.ArgumentParser(description="EGAI 실험 실행")

    # 모델 설정
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cbam",
        choices=["simple", "simple_cbam", "channel_concat", "4channel_cbam", "multihead_cbam"],
        help="모델 타입 (default: simple_cbam)",
    )

    # 학습 설정
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="CV Fold 수 (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="에폭 수 (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="학습률 (default: 0.001)",
    )

    # 데이터 설정
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="데이터 디렉토리 (default: data)",
    )
    parser.add_argument(
        "--fuel",
        type=str,
        default="가솔린",
        help="연료 타입 필터 (default: 가솔린)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="캐싱 비활성화",
    )

    # 전처리 설정
    parser.add_argument(
        "--fmax",
        type=int,
        default=6000,
        help="최대 주파수 (default: 6000)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=128,
        help="스펙트로그램 크기 (default: 128)",
    )

    # 실험 설정
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="실험 이름",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="태그 리스트",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments",
        help="실험 저장 디렉토리 (default: experiments)",
    )

    # 기타
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="최고 모델 저장",
    )

    args = parser.parse_args()

    # 설정 생성
    config = {
        "model_type": args.model,
        "n_folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "target_shape": (args.target_size, args.target_size),
        "fmax": args.fmax,
        "use_cache": not args.no_cache,
        "fuel_type": args.fuel,
    }

    print("=" * 60)
    print("EGAI 실험 실행")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"Folds: {args.folds}")
    print(f"Epochs: {args.epochs}")
    print(f"데이터: {args.data_dir}")
    print("=" * 60)

    # 파이프라인 실행
    pipeline = TrainingPipeline(
        config=config,
        data_dir=args.data_dir,
        experiments_dir=args.exp_dir,
    )

    results = pipeline.run(
        experiment_name=args.name,
        tags=args.tags,
    )

    # 모델 저장
    if args.save_model:
        pipeline.save_best_model()

    print("\n" + "=" * 60)
    print("실험 완료")
    print("=" * 60)
    print(f"최종 MAE: {results['overall_MAE_mean']:.4f} ± {results['overall_MAE_std']:.4f}")

    return results


if __name__ == "__main__":
    main()
