"""
Training Use Case - 모델 학습

K-Fold Cross-Validation 기반 학습 파이프라인
"""

from pathlib import Path
from typing import Optional, List, Dict
import json
import numpy as np

from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score

from egai.domain.entities import TrainingConfig
from egai.infrastructure.models import ModelFactory


class Trainer:
    """
    모델 학습기

    5-Fold Stratified CV로 안정적인 평가
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        초기화

        Args:
            config: 학습 설정 (None이면 기본값)
        """
        self.config = config or TrainingConfig()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Cross-Validation 학습

        Args:
            X: 입력 데이터 (N, H, W, C)
            y: 타겟 (N, 5)
            save_dir: 모델 저장 디렉토리

        Returns:
            CV 결과 딕셔너리
        """
        config = self.config

        # 층화 추출용 레이블 (전체 점수 구간화)
        overall_scores = y.mean(axis=1)
        bins = np.linspace(overall_scores.min(), overall_scores.max() + 0.01, 6)
        stratify_labels = np.digitize(overall_scores, bins)

        kf = StratifiedKFold(
            n_splits=config.n_folds, shuffle=True, random_state=42
        )

        fold_results = []

        print(f"\n{'='*60}")
        print(f"{config.model_type.upper()} {config.n_folds}-Fold CV")
        print(f"{'='*60}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, stratify_labels)):
            print(f"\n--- Fold {fold + 1} ---")

            # 데이터 분할
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 모델 생성
            model = ModelFactory.create(
                config.model_type,
                input_shape=X.shape[1:],
                num_outputs=y.shape[1],
            )

            model.compile(
                optimizer=keras.optimizers.Adam(config.learning_rate),
                loss="mse",
                metrics=["mae"],
            )

            # 콜백
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    restore_best_weights=True,
                    verbose=0,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ]

            # 학습
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            # 평가
            y_pred = model.predict(X_val, verbose=0)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            actual_epochs = len(history.history["loss"])
            print(f"  Epochs: {actual_epochs}, MAE: {mae:.4f}, R²: {r2:.4f}")

            fold_results.append({
                "fold": fold + 1,
                "mae": mae,
                "r2": r2,
                "epochs": actual_epochs,
            })

            # 메모리 정리
            keras.backend.clear_session()

        # 결과 집계
        cv_results = self._aggregate_results(fold_results)
        self._print_results(cv_results)

        # 저장
        if save_dir:
            self._save_results(cv_results, save_dir)

        return cv_results

    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """결과 집계"""
        maes = [r["mae"] for r in fold_results]
        r2s = [r["r2"] for r in fold_results]
        epochs = [r["epochs"] for r in fold_results]

        return {
            "n_folds": len(fold_results),
            "mae_mean": np.mean(maes),
            "mae_std": np.std(maes),
            "r2_mean": np.mean(r2s),
            "r2_std": np.std(r2s),
            "avg_epochs": np.mean(epochs),
            "fold_results": fold_results,
        }

    def _print_results(self, results: Dict):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"Cross-Validation 결과 ({results['n_folds']}-Fold)")
        print(f"{'='*60}")
        print(f"MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"R²:  {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        print(f"평균 에포크: {results['avg_epochs']:.1f}")

    def _save_results(self, results: Dict, save_dir: str):
        """결과 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "cv_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n결과 저장: {save_path / 'cv_results.json'}")


def main():
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser(description="EGAI 모델 학습")
    parser.add_argument(
        "--data_dir", "-d", default="data", help="데이터 디렉토리"
    )
    parser.add_argument(
        "--model", "-m", default="simple_cbam",
        choices=["simple_cbam", "simple", "channel_concat"],
        help="모델 타입",
    )
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--folds", "-f", type=int, default=5)
    parser.add_argument("--save_dir", "-s", default="models")

    args = parser.parse_args()

    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_folds=args.folds,
    )

    print("데이터 로드는 아직 구현되지 않았습니다.")
    print("analysis_legacy/train_cv.py를 참조하세요.")


if __name__ == "__main__":
    main()
