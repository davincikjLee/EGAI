"""
Training Pipeline - MLOps 학습 파이프라인

전체 학습 과정 오케스트레이션:
    1. 데이터 로딩
    2. 전처리
    3. 모델 생성
    4. 학습 (Cross-Validation)
    5. 평가
    6. 실험 기록
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from egai.data.loader import AudioDataLoader
from egai.infrastructure.models import ModelFactory
from egai.pipelines.experiment import ExperimentTracker


class TrainingPipeline:
    """
    MLOps 학습 파이프라인

    사용법:
        pipeline = TrainingPipeline(config)
        results = pipeline.run()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: str = "data",
        experiments_dir: str = "experiments",
    ):
        """
        Args:
            config: 학습 설정
            data_dir: 데이터 디렉토리
            experiments_dir: 실험 저장 디렉토리
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.experiments_dir = Path(experiments_dir)

        # 기본 설정
        self.model_type = config.get("model_type", "simple_cbam")
        self.n_folds = config.get("n_folds", 5)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.target_shape = config.get("target_shape", (128, 128))
        self.fmax = config.get("fmax", 6000)
        self.use_cache = config.get("use_cache", True)
        self.fuel_type = config.get("fuel_type", None)  # None이면 전체 데이터 사용

        # 컴포넌트 초기화
        self.loader: Optional[AudioDataLoader] = None
        self.tracker: Optional[ExperimentTracker] = None

        # 결과 저장
        self.fold_results: List[Dict] = []
        self.best_model: Optional[tf.keras.Model] = None
        self.best_fold: int = -1
        self.best_mae: float = float("inf")

    def run(
        self,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행

        Args:
            experiment_name: 실험 이름
            tags: 태그 리스트

        Returns:
            최종 결과
        """
        # 실험 이름 생성
        if experiment_name is None:
            experiment_name = f"{self.model_type}_{self.n_folds}fold"

        # 실험 추적기 초기화
        self.tracker = ExperimentTracker(str(self.experiments_dir))
        exp_id = self.tracker.start_experiment(
            name=experiment_name,
            config=self.config,
            tags=tags or [self.model_type, f"{self.n_folds}-fold"],
        )

        try:
            # 1. 데이터 로딩
            print("\n" + "=" * 50)
            print("[Pipeline] 1. 데이터 로딩")
            print("=" * 50)
            self._load_data()

            # 2. Cross-Validation 실행
            print("\n" + "=" * 50)
            print(f"[Pipeline] 2. {self.n_folds}-Fold Cross-Validation")
            print("=" * 50)
            self._run_cv()

            # 3. 결과 집계
            print("\n" + "=" * 50)
            print("[Pipeline] 3. 결과 집계")
            print("=" * 50)
            final_results = self._aggregate_results()

            # 4. 실험 완료
            self.tracker.finish_experiment(final_results, status="completed")

            return final_results

        except Exception as e:
            print(f"[Pipeline] 오류 발생: {e}")
            if self.tracker:
                self.tracker.finish_experiment(
                    {"error": str(e)},
                    status="failed",
                )
            raise

    def _load_data(self):
        """데이터 로딩"""
        cache_dir = str(self.data_dir / "cache") if self.use_cache else None

        self.loader = AudioDataLoader(
            data_dir=str(self.data_dir),
            target_shape=self.target_shape,
            cache_dir=cache_dir,
            fmax=self.fmax,
        )

        self.loader.load_metadata(fuel_type=self.fuel_type)

        # 통계 출력
        stats = self.loader.get_stats()
        print(f"[Pipeline] 유효 데이터: {stats['valid']}개")

    def _run_cv(self):
        """Cross-Validation 실행"""
        indices = np.array(self.loader.valid_indices)
        scores = self.loader.metadata.iloc[indices]["overall_score"].values

        # Stratified 분할
        bins = np.linspace(scores.min(), scores.max() + 0.01, 6)
        stratify_labels = np.digitize(scores, bins)

        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=42,
        )

        cv_splits = []

        for fold, (train_val_idx, test_idx) in enumerate(
            skf.split(indices, stratify_labels)
        ):
            print(f"\n{'='*40}")
            print(f"[Fold {fold + 1}/{self.n_folds}]")
            print(f"{'='*40}")

            train_indices = indices[train_val_idx].tolist()
            test_indices = indices[test_idx].tolist()

            # 분할 정보 저장
            cv_splits.append({
                "fold": fold,
                "train": train_indices,
                "test": test_indices,
            })

            # Fold 학습
            fold_result = self._train_fold(
                fold=fold,
                train_indices=train_indices,
                test_indices=test_indices,
            )

            self.fold_results.append(fold_result)
            self.tracker.log_fold_result(fold, fold_result)

            # 최고 모델 추적
            if fold_result["MAE"] < self.best_mae:
                self.best_mae = fold_result["MAE"]
                self.best_fold = fold

        # CV 분할 저장
        self.tracker.save_data_splits(
            train_idx=[],
            val_idx=[],
            test_idx=[],
            cv_splits=cv_splits,
        )

    def _train_fold(
        self,
        fold: int,
        train_indices: List[int],
        test_indices: List[int],
    ) -> Dict[str, float]:
        """
        단일 Fold 학습

        Returns:
            평가 메트릭
        """
        # 데이터 준비
        print(f"[Fold {fold + 1}] 데이터 준비...")
        X_train, y_train = self.loader.prepare_dataset(
            train_indices,
            model_type=self.model_type,
            use_cache=self.use_cache,
            verbose=False,
        )
        X_test, y_test = self.loader.prepare_dataset(
            test_indices,
            model_type=self.model_type,
            use_cache=self.use_cache,
            verbose=False,
        )

        print(f"[Fold {fold + 1}] Train: {len(y_train)}, Test: {len(y_test)}")

        # 모델 생성
        if self.model_type in ["4channel_cbam", "multihead_cbam"]:
            input_shape = (*self.target_shape, 4)
        else:
            input_shape = (*self.target_shape, 1)

        model = ModelFactory.create(
            model_type=self.model_type,
            input_shape=input_shape,
            num_outputs=5,
        )

        # 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mae",
            metrics=["mae"],
        )

        # 콜백
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mae",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        # 학습
        print(f"[Fold {fold + 1}] 학습 시작...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # 평가
        y_pred = model.predict(X_test, verbose=0)
        mae_per_col = np.abs(y_test - y_pred).mean(axis=0)
        overall_mae = mae_per_col.mean()

        # 컬럼별 MAE
        target_cols = AudioDataLoader.TARGET_COLUMNS
        col_mae = {
            col: float(mae_per_col[i])
            for i, col in enumerate(target_cols)
        }

        result = {
            "MAE": float(overall_mae),
            "best_epoch": len(history.history["loss"]),
            **col_mae,
        }

        print(f"[Fold {fold + 1}] MAE: {overall_mae:.4f}")

        # 모델 저장 (최고 성능)
        if overall_mae < self.best_mae:
            self.best_model = model

        return result

    def _aggregate_results(self) -> Dict[str, Any]:
        """결과 집계"""
        target_cols = AudioDataLoader.TARGET_COLUMNS

        # MAE 통계
        mae_values = [r["MAE"] for r in self.fold_results]
        overall_mae_mean = np.mean(mae_values)
        overall_mae_std = np.std(mae_values)

        # 컬럼별 통계
        col_stats = {}
        for col in target_cols:
            col_values = [r[col] for r in self.fold_results]
            col_stats[f"{col}_mean"] = float(np.mean(col_values))
            col_stats[f"{col}_std"] = float(np.std(col_values))

        results = {
            "overall_MAE_mean": float(overall_mae_mean),
            "overall_MAE_std": float(overall_mae_std),
            "n_folds": self.n_folds,
            "best_fold": self.best_fold,
            "best_MAE": float(self.best_mae),
            **col_stats,
        }

        # 결과 출력
        print(f"\n[결과 요약]")
        print(f"전체 MAE: {overall_mae_mean:.4f} ± {overall_mae_std:.4f}")
        print(f"최고 Fold: {self.best_fold + 1} (MAE: {self.best_mae:.4f})")
        print(f"\n[컬럼별 MAE]")
        for col in target_cols:
            print(f"  {col}: {col_stats[f'{col}_mean']:.4f}")

        return results

    def save_best_model(self, path: Optional[str] = None) -> str:
        """
        최고 모델 저장

        Args:
            path: 저장 경로 (None이면 자동 생성)

        Returns:
            저장 경로
        """
        if self.best_model is None:
            raise ValueError("학습된 모델이 없습니다")

        if path is None:
            path = str(
                self.experiments_dir
                / f"best_model_{self.model_type}_{self.best_fold}.keras"
            )

        self.best_model.save(path)
        print(f"[Pipeline] 모델 저장: {path}")

        return path


def run_experiment(
    config: Dict[str, Any],
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    실험 실행 헬퍼 함수

    Args:
        config: 학습 설정
        name: 실험 이름
        tags: 태그

    Returns:
        실험 결과
    """
    pipeline = TrainingPipeline(config)
    return pipeline.run(experiment_name=name, tags=tags)
