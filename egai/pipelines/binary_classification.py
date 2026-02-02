"""
Binary Classification Pipeline - 가솔린/디젤 이진 분류

가솔린=OK(0), 디젤=NG(1)로 모사하여 OK/NG 분류 실험
- 클래스 가중치로 데이터 불균형 처리
- F1, ROC-AUC, PR-AUC 평가
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, confusion_matrix
)
import tensorflow as tf

from egai.data.loader import AudioDataLoader
from egai.infrastructure.models import ModelFactory
from egai.pipelines.experiment import ExperimentTracker


class BinaryClassificationPipeline:
    """
    가솔린/디젤 이진 분류 파이프라인

    사용법:
        pipeline = BinaryClassificationPipeline(config)
        results = pipeline.run()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: str = "data",
        experiments_dir: str = "experiments",
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.experiments_dir = Path(experiments_dir)

        # 기본 설정
        self.n_folds = config.get("n_folds", 5)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.target_shape = config.get("target_shape", (128, 128))
        self.fmax = config.get("fmax", 6000)
        self.use_cache = config.get("use_cache", True)

        # 컴포넌트 초기화
        self.loader: Optional[AudioDataLoader] = None
        self.tracker: Optional[ExperimentTracker] = None

        # 결과 저장
        self.fold_results: List[Dict] = []
        self.best_model: Optional[tf.keras.Model] = None
        self.best_fold: int = -1
        self.best_f1: float = 0.0

        # 클래스 가중치 (나중에 계산)
        self.class_weights: Dict[int, float] = {}

    def run(
        self,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        if experiment_name is None:
            experiment_name = f"binary_cbam_{self.n_folds}fold"

        self.tracker = ExperimentTracker(str(self.experiments_dir))
        exp_id = self.tracker.start_experiment(
            name=experiment_name,
            config=self.config,
            tags=tags or ["binary_classification", "gasoline_diesel"],
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
        """가솔린 + 디젤 데이터 로딩"""
        cache_dir = str(self.data_dir / "cache") if self.use_cache else None

        self.loader = AudioDataLoader(
            data_dir=str(self.data_dir),
            target_shape=self.target_shape,
            cache_dir=cache_dir,
            fmax=self.fmax,
        )

        # 전체 데이터 로드 (가솔린 + 디젤)
        self.loader.load_metadata(fuel_type=None)

        # 연료 타입 라벨 생성 (가솔린=0, 디젤=1)
        self.fuel_labels = []
        valid_indices = []

        for idx in self.loader.valid_indices:
            fuel_type = self.loader.metadata.iloc[idx]["fuel_type"]
            if fuel_type == "가솔린":
                self.fuel_labels.append(0)
                valid_indices.append(idx)
            elif fuel_type == "디젤":
                self.fuel_labels.append(1)
                valid_indices.append(idx)

        self.valid_indices = np.array(valid_indices)
        self.fuel_labels = np.array(self.fuel_labels)

        # 클래스 분포
        n_gasoline = np.sum(self.fuel_labels == 0)
        n_diesel = np.sum(self.fuel_labels == 1)

        print(f"[Pipeline] 가솔린: {n_gasoline}개")
        print(f"[Pipeline] 디젤: {n_diesel}개")
        print(f"[Pipeline] 불균형 비율: {n_gasoline / n_diesel:.1f}:1")

        # 클래스 가중치 계산
        total = n_gasoline + n_diesel
        self.class_weights = {
            0: total / (2 * n_gasoline),
            1: total / (2 * n_diesel),
        }
        print(f"[Pipeline] 클래스 가중치: {self.class_weights}")

    def _run_cv(self):
        """Cross-Validation 실행"""
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=42,
        )

        for fold, (train_idx, test_idx) in enumerate(
            skf.split(self.valid_indices, self.fuel_labels)
        ):
            print(f"\n{'='*40}")
            print(f"[Fold {fold + 1}/{self.n_folds}]")
            print(f"{'='*40}")

            train_data_indices = self.valid_indices[train_idx].tolist()
            test_data_indices = self.valid_indices[test_idx].tolist()
            train_labels = self.fuel_labels[train_idx]
            test_labels = self.fuel_labels[test_idx]

            # Fold 학습
            fold_result = self._train_fold(
                fold=fold,
                train_indices=train_data_indices,
                test_indices=test_data_indices,
                train_labels=train_labels,
                test_labels=test_labels,
            )

            self.fold_results.append(fold_result)
            self.tracker.log_fold_result(fold, fold_result)

            # 최고 모델 추적 (F1 기준)
            if fold_result["f1"] > self.best_f1:
                self.best_f1 = fold_result["f1"]
                self.best_fold = fold

    def _train_fold(
        self,
        fold: int,
        train_indices: List[int],
        test_indices: List[int],
        train_labels: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """단일 Fold 학습"""
        # 데이터 준비
        print(f"[Fold {fold + 1}] 데이터 준비...")
        X_train, _ = self.loader.prepare_dataset(
            train_indices,
            model_type="4channel_cbam",
            use_cache=self.use_cache,
            verbose=False,
        )
        X_test, _ = self.loader.prepare_dataset(
            test_indices,
            model_type="4channel_cbam",
            use_cache=self.use_cache,
            verbose=False,
        )

        y_train = train_labels.astype(np.float32)
        y_test = test_labels.astype(np.float32)

        # 클래스 분포 출력
        n_train_gas = np.sum(y_train == 0)
        n_train_diesel = np.sum(y_train == 1)
        n_test_gas = np.sum(y_test == 0)
        n_test_diesel = np.sum(y_test == 1)

        print(f"[Fold {fold + 1}] Train: {n_train_gas} gasoline, {n_train_diesel} diesel")
        print(f"[Fold {fold + 1}] Test: {n_test_gas} gasoline, {n_test_diesel} diesel")

        # 모델 생성
        input_shape = (*self.target_shape, 4)
        model = ModelFactory.create(
            model_type="binary_cbam",
            input_shape=input_shape,
        )

        # 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        # 콜백
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                mode="max",
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
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=0,
        )

        # 예측
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 평가 메트릭
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        accuracy = np.mean(y_pred == y_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        result = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "best_epoch": len(history.history["loss"]),
        }

        print(f"[Fold {fold + 1}] Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")

        # 최고 모델 저장
        if f1 > self.best_f1:
            self.best_model = model

        return result

    def _aggregate_results(self) -> Dict[str, Any]:
        """결과 집계"""
        metrics = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]

        results = {
            "n_folds": self.n_folds,
            "best_fold": self.best_fold,
            "best_f1": float(self.best_f1),
        }

        # 메트릭별 통계
        for metric in metrics:
            values = [r[metric] for r in self.fold_results]
            results[f"{metric}_mean"] = float(np.mean(values))
            results[f"{metric}_std"] = float(np.std(values))

        # Confusion Matrix 합계
        results["total_true_negatives"] = sum(r["true_negatives"] for r in self.fold_results)
        results["total_false_positives"] = sum(r["false_positives"] for r in self.fold_results)
        results["total_false_negatives"] = sum(r["false_negatives"] for r in self.fold_results)
        results["total_true_positives"] = sum(r["true_positives"] for r in self.fold_results)

        # 결과 출력
        print(f"\n[결과 요약]")
        print(f"Best Fold: {self.best_fold + 1} (F1: {self.best_f1:.4f})")
        print(f"\n[메트릭 평균 ± 표준편차]")
        for metric in metrics:
            mean = results[f"{metric}_mean"]
            std = results[f"{metric}_std"]
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")

        print(f"\n[Confusion Matrix (전체)]")
        print(f"  TN: {results['total_true_negatives']} | FP: {results['total_false_positives']}")
        print(f"  FN: {results['total_false_negatives']} | TP: {results['total_true_positives']}")

        # 목표 달성 여부 확인
        print(f"\n[목표 달성 확인]")
        targets = {
            "F1 > 0.70": results["f1_mean"] > 0.70,
            "ROC-AUC > 0.85": results["roc_auc_mean"] > 0.85,
            "PR-AUC > 0.60": results["pr_auc_mean"] > 0.60,
        }
        for target, achieved in targets.items():
            status = "PASS" if achieved else "FAIL"
            print(f"  [{status}] {target}")

        return results

    def save_best_model(self, path: Optional[str] = None) -> str:
        """최고 모델 저장"""
        if self.best_model is None:
            raise ValueError("학습된 모델이 없습니다")

        if path is None:
            path = str(
                self.experiments_dir
                / f"best_binary_model_fold{self.best_fold}.keras"
            )

        self.best_model.save(path)
        print(f"[Pipeline] 모델 저장: {path}")

        return path
