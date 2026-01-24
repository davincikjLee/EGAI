"""
Anomaly Detection Training Pipeline - 이상탐지 학습 파이프라인

VAE 기반 이상탐지 모델 학습:
    1. 정상 데이터만 필터링 (4~5점 데이터)
    2. VAE 학습
    3. 정규화 통계값 계산
    4. 임계값 설정 (새 차 데이터 기반)
    5. 모델 및 설정 저장
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import json
import tensorflow as tf

from egai.data.loader import AudioDataLoader
from egai.infrastructure.models import ModelFactory
from egai.domain.services import AnomalyDetector, AnomalyResult
from egai.pipelines.experiment import ExperimentTracker


class AnomalyTrainingPipeline:
    """
    이상탐지 학습 파이프라인

    정상 데이터(4~5점)만으로 VAE를 학습하고,
    MC Dropout 불확실성과 앙상블하여 이상 탐지

    사용법:
        pipeline = AnomalyTrainingPipeline(config)
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
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # 기본 설정
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.0005)
        self.target_shape = config.get("target_shape", (128, 128))
        self.fmax = config.get("fmax", 6000)
        self.use_cache = config.get("use_cache", True)
        self.fuel_type = config.get("fuel_type", None)  # None이면 전체 데이터 사용

        # VAE 설정
        self.latent_dim = config.get("latent_dim", 128)
        self.beta = config.get("beta", 0.5)  # KL 가중치

        # 정상 데이터 필터링 기준
        self.min_score = config.get("min_score", 4.0)  # 4점 이상만 사용

        # 검증 분할 비율
        self.validation_split = config.get("validation_split", 0.2)

        # 컴포넌트
        self.loader: Optional[AudioDataLoader] = None
        self.tracker: Optional[ExperimentTracker] = None
        self.vae_model = None
        self.regression_model = None
        self.detector: Optional[AnomalyDetector] = None

        # 결과 저장
        self.training_history = None
        self.normalization_stats = {}

    def run(
        self,
        experiment_name: Optional[str] = None,
        regression_model_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행

        Args:
            experiment_name: 실험 이름
            regression_model_path: 기존 회귀 모델 경로 (MC Dropout용)
            tags: 태그 리스트

        Returns:
            최종 결과
        """
        if experiment_name is None:
            experiment_name = f"anomaly_vae_beta{self.beta}"

        # 실험 추적기 초기화
        self.tracker = ExperimentTracker(str(self.experiments_dir))
        exp_id = self.tracker.start_experiment(
            name=experiment_name,
            config=self.config,
            tags=tags or ["anomaly", "vae", f"beta_{self.beta}"],
        )

        try:
            # 1. 데이터 로딩
            print("\n" + "=" * 50)
            print("[Pipeline] 1. 데이터 로딩")
            print("=" * 50)
            self._load_data()

            # 2. 정상 데이터 필터링
            print("\n" + "=" * 50)
            print("[Pipeline] 2. 정상 데이터 필터링")
            print("=" * 50)
            train_indices, val_indices = self._filter_normal_data()

            # 3. VAE 학습
            print("\n" + "=" * 50)
            print("[Pipeline] 3. VAE 학습")
            print("=" * 50)
            self._train_vae(train_indices, val_indices)

            # 4. 회귀 모델 로드 (선택적)
            if regression_model_path:
                print("\n" + "=" * 50)
                print("[Pipeline] 4. 회귀 모델 로드")
                print("=" * 50)
                self._load_regression_model(regression_model_path)

            # 5. 정규화 통계값 계산
            print("\n" + "=" * 50)
            print("[Pipeline] 5. 정규화 통계값 계산")
            print("=" * 50)
            self._compute_normalization_stats(train_indices)

            # 6. 결과 집계
            print("\n" + "=" * 50)
            print("[Pipeline] 6. 결과 집계")
            print("=" * 50)
            final_results = self._aggregate_results()

            # 7. 모델 저장
            print("\n" + "=" * 50)
            print("[Pipeline] 7. 모델 저장")
            print("=" * 50)
            self._save_models(exp_id)

            # 실험 완료
            self.tracker.finish_experiment(final_results, status="completed")

            return final_results

        except Exception as e:
            print(f"[Pipeline] 오류 발생: {e}")
            import traceback
            traceback.print_exc()
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

        stats = self.loader.get_stats()
        print(f"[Pipeline] 전체 유효 데이터: {stats['valid']}개")

    def _filter_normal_data(self) -> Tuple[List[int], List[int]]:
        """
        정상 데이터 필터링 (overall_score >= min_score)

        Returns:
            train_indices, val_indices
        """
        indices = np.array(self.loader.valid_indices)
        scores = self.loader.metadata.iloc[indices]["overall_score"].values

        # 정상 데이터만 필터링
        normal_mask = scores >= self.min_score
        normal_indices = indices[normal_mask].tolist()

        print(f"[Pipeline] 정상 데이터 (>= {self.min_score}점): {len(normal_indices)}개")
        print(f"[Pipeline] 제외된 데이터: {len(indices) - len(normal_indices)}개")

        # Train/Val 분할
        np.random.seed(42)
        np.random.shuffle(normal_indices)

        split_idx = int(len(normal_indices) * (1 - self.validation_split))
        train_indices = normal_indices[:split_idx]
        val_indices = normal_indices[split_idx:]

        print(f"[Pipeline] Train: {len(train_indices)}개, Val: {len(val_indices)}개")

        return train_indices, val_indices

    def _train_vae(
        self,
        train_indices: List[int],
        val_indices: List[int],
    ):
        """VAE 학습"""
        # 데이터 준비 (4채널)
        print("[Pipeline] 데이터 준비 중...")
        X_train, _ = self.loader.prepare_dataset(
            train_indices,
            model_type="4channel_cbam",
            use_cache=self.use_cache,
            verbose=True,
        )
        X_val, _ = self.loader.prepare_dataset(
            val_indices,
            model_type="4channel_cbam",
            use_cache=self.use_cache,
            verbose=False,
        )

        print(f"[Pipeline] X_train shape: {X_train.shape}")
        print(f"[Pipeline] X_val shape: {X_val.shape}")

        # VAE 모델 생성
        input_shape = (*self.target_shape, 4)
        self.vae_model = ModelFactory.create(
            model_type="vae",
            input_shape=input_shape,
        )

        # β 설정
        self.vae_model.beta = self.beta

        # 컴파일
        self.vae_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        )

        # 콜백
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_reconstruction_loss",
                patience=15,
                restore_best_weights=True,
                mode="min",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_reconstruction_loss",
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                mode="min",
                verbose=1,
            ),
        ]

        # 학습
        print(f"[Pipeline] VAE 학습 시작 (β={self.beta})...")
        history = self.vae_model.fit(
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val,),
            callbacks=callbacks,
            verbose=1,
        )

        self.training_history = history.history

        # 최종 손실 출력
        final_loss = history.history["loss"][-1]
        final_recon = history.history["reconstruction_loss"][-1]
        final_kl = history.history["kl_loss"][-1]
        print(f"\n[Pipeline] 최종 Loss: {final_loss:.4f}")
        print(f"[Pipeline] Reconstruction: {final_recon:.4f}, KL: {final_kl:.4f}")

    def _load_regression_model(self, model_path: str):
        """기존 회귀 모델 로드"""
        from egai.infrastructure.models import ModelFactory
        self.regression_model = ModelFactory.load(model_path)
        print(f"[Pipeline] 회귀 모델 로드: {model_path}")

    def _compute_normalization_stats(self, indices: List[int]):
        """
        정규화 통계값 계산

        학습 데이터에서 VAE 점수와 MC 불확실성의 평균/표준편차 계산
        """
        print("[Pipeline] 정규화 통계값 계산 중...")

        # 데이터 준비
        X, _ = self.loader.prepare_dataset(
            indices[:100],  # 샘플링 (속도 위해)
            model_type="4channel_cbam",
            use_cache=self.use_cache,
            verbose=False,
        )

        # VAE 점수 계산
        vae_scores = []
        for i in range(len(X)):
            score = self.vae_model.compute_anomaly_score(X[i:i+1])
            vae_scores.append(float(score.numpy()[0]))

        vae_scores = np.array(vae_scores)

        # MC Dropout 불확실성 계산 (회귀 모델 있는 경우)
        mc_uncertainties = np.zeros(len(X))
        if self.regression_model is not None:
            for i in range(len(X)):
                preds = []
                for _ in range(30):
                    pred = self.regression_model(X[i:i+1], training=True)
                    preds.append(pred.numpy())
                mc_uncertainties[i] = np.mean(np.std(preds, axis=0))

        # 통계값 저장
        self.normalization_stats = {
            "vae_mean": float(np.mean(vae_scores)),
            "vae_std": float(np.std(vae_scores)),
            "mc_mean": float(np.mean(mc_uncertainties)),
            "mc_std": float(np.std(mc_uncertainties)),
        }

        print(f"[Pipeline] VAE 점수: {self.normalization_stats['vae_mean']:.4f} ± {self.normalization_stats['vae_std']:.4f}")

        # AnomalyDetector 초기화
        self.detector = AnomalyDetector(
            vae_model=self.vae_model,
            regression_model=self.regression_model,
        )
        self.detector.set_normalization_stats(vae_scores, mc_uncertainties)

    def _aggregate_results(self) -> Dict[str, Any]:
        """결과 집계"""
        results = {
            "model_type": "VAE",
            "latent_dim": self.latent_dim,
            "beta": self.beta,
            "epochs_trained": len(self.training_history["loss"]),
            "final_loss": float(self.training_history["loss"][-1]),
            "final_reconstruction_loss": float(self.training_history["reconstruction_loss"][-1]),
            "final_kl_loss": float(self.training_history["kl_loss"][-1]),
            **self.normalization_stats,
        }

        print("\n[결과 요약]")
        print(f"학습 Epoch: {results['epochs_trained']}")
        print(f"최종 Loss: {results['final_loss']:.4f}")
        print(f"Reconstruction: {results['final_reconstruction_loss']:.4f}")
        print(f"KL Divergence: {results['final_kl_loss']:.4f}")

        return results

    def _save_models(self, exp_id: str):
        """모델 및 설정 저장"""
        save_dir = self.experiments_dir / exp_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # VAE 모델 저장 (encoder, decoder 분리)
        encoder_path = save_dir / "vae_encoder.keras"
        decoder_path = save_dir / "vae_decoder.keras"
        self.vae_model.encoder.save(encoder_path)
        self.vae_model.decoder.save(decoder_path)
        print(f"[Pipeline] VAE Encoder 저장: {encoder_path}")
        print(f"[Pipeline] VAE Decoder 저장: {decoder_path}")

        # 설정 저장
        config_path = save_dir / "anomaly_config.json"
        config_data = {
            "beta": self.beta,
            "latent_dim": self.latent_dim,
            "target_shape": self.target_shape,
            "fmax": self.fmax,
            "min_score": self.min_score,
            **self.normalization_stats,
            "thresholds": {
                "normal": self.detector.threshold_normal if self.detector else 0.3,
                "caution": self.detector.threshold_caution if self.detector else 0.6,
                "check": self.detector.threshold_check if self.detector else 0.8,
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"[Pipeline] 설정 저장: {config_path}")

    def validate_with_new_car_data(
        self,
        new_car_audio_paths: List[str],
    ) -> Dict[str, Any]:
        """
        새 차 데이터로 검증 및 임계값 설정

        Args:
            new_car_audio_paths: 새 차 오디오 파일 경로 리스트

        Returns:
            검증 결과 및 임계값 정보
        """
        if self.detector is None:
            raise ValueError("먼저 run()을 실행하세요")

        print(f"\n[검증] 새 차 데이터 {len(new_car_audio_paths)}개 검증 중...")

        # 새 차 데이터 전처리
        from egai.preprocessing.audio import AudioPreprocessor
        preprocessor = AudioPreprocessor(
            target_shape=self.target_shape,
            fmax=self.fmax,
        )

        anomaly_scores = []
        for path in new_car_audio_paths:
            spec = preprocessor.process_4channel(path)
            result = self.detector.detect(spec)
            anomaly_scores.append(result.anomaly_score)

        anomaly_scores = np.array(anomaly_scores)

        # 임계값 설정
        threshold_info = self.detector.set_thresholds_from_new_car_data(
            anomaly_scores
        )

        # 정상 판정율 계산
        normal_rate = np.mean(anomaly_scores < self.detector.threshold_caution)

        results = {
            "n_new_cars": len(new_car_audio_paths),
            "normal_rate": float(normal_rate),
            "target_rate": 0.99,
            "pass": normal_rate >= 0.99,
            **threshold_info,
        }

        print(f"\n[검증 결과]")
        print(f"새 차 정상 판정율: {normal_rate:.1%} (목표: 99%+)")
        print(f"Pass: {'Yes' if results['pass'] else 'No'}")
        print(f"\n[임계값 설정]")
        print(f"정상: < {threshold_info['threshold_normal']:.4f}")
        print(f"주의: < {threshold_info['threshold_caution']:.4f}")
        print(f"점검 권장: >= {threshold_info['threshold_check']:.4f}")

        return results


def run_anomaly_training(
    config: Dict[str, Any],
    name: Optional[str] = None,
    regression_model_path: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    이상탐지 학습 실행 헬퍼 함수

    Args:
        config: 학습 설정
        name: 실험 이름
        regression_model_path: 회귀 모델 경로
        tags: 태그

    Returns:
        실험 결과
    """
    pipeline = AnomalyTrainingPipeline(config)
    return pipeline.run(
        experiment_name=name,
        regression_model_path=regression_model_path,
        tags=tags,
    )
