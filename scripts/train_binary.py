"""
가솔린/디젤 이진 분류 학습 스크립트

사용법:
    python scripts/train_binary.py
    python scripts/train_binary.py --folds 5 --epochs 50
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from egai.pipelines.binary_classification import BinaryClassificationPipeline


def main():
    parser = argparse.ArgumentParser(description="가솔린/디젤 이진 분류 학습")

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
        "--name", type=str, default=None,
        help="실험 이름"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="캐시 사용 안 함"
    )

    args = parser.parse_args()

    # 설정
    config = {
        "n_folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "target_shape": (128, 128),
        "fmax": 6000,
        "use_cache": not args.no_cache,
    }

    print("=" * 60)
    print("EGAI 가솔린/디젤 이진 분류 학습")
    print("=" * 60)
    print(f"\n[설정]")
    print(f"  Folds: {args.folds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Data Dir: {args.data_dir}")
    print(f"  Use Cache: {not args.no_cache}")

    print(f"\n[목표 메트릭]")
    print(f"  F1-Score: > 0.70")
    print(f"  ROC-AUC: > 0.85")
    print(f"  PR-AUC: > 0.60")

    # 파이프라인 실행
    pipeline = BinaryClassificationPipeline(
        config=config,
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
    )

    results = pipeline.run(
        experiment_name=args.name,
        tags=["binary", "gasoline_diesel", f"{args.folds}fold"],
    )

    # 최고 모델 저장
    model_path = pipeline.save_best_model()

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"\n[최종 결과]")
    print(f"  F1-Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"  ROC-AUC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
    print(f"  PR-AUC: {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}")
    print(f"\n  모델 저장: {model_path}")

    return results


if __name__ == "__main__":
    main()
