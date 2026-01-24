"""
추론 시간 벤치마크

측정 항목:
1. 오디오 로딩 + 스펙트로그램 변환
2. VAE 추론
3. Regression 모델 추론 (있는 경우)
4. 전체 파이프라인
"""

import sys
from pathlib import Path
import time
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from egai.preprocessing.audio import AudioPreprocessor
from egai.data.loader import AudioDataLoader
from egai.infrastructure.models import Sampling


def benchmark_audio_processing(audio_path: str, n_runs: int = 10):
    """오디오 처리 시간 측정"""
    preprocessor = AudioPreprocessor(
        fmax=6000,
        target_shape=(128, 128)
    )

    # Warmup
    preprocessor.process(audio_path)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        full_spec, perc_spec = preprocessor.process(audio_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_vae_inference(encoder, decoder, spectrogram: np.ndarray, n_runs: int = 50):
    """VAE 추론 시간 측정"""
    if spectrogram.ndim == 3:
        spectrogram = np.expand_dims(spectrogram, axis=0)

    # Warmup
    z_mean, z_log_var, z = encoder(spectrogram)
    reconstruction = decoder(z)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        z_mean, z_log_var, z = encoder(spectrogram)
        reconstruction = decoder(z)
        # 이상 점수 계산
        recon_error = tf.reduce_sum(tf.keras.losses.mse(spectrogram, reconstruction), axis=(1, 2))
        kl_div = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        anomaly_score = recon_error + 0.5 * kl_div
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_regression_inference(model, spectrogram: np.ndarray, n_runs: int = 50):
    """Regression 모델 추론 시간 측정"""
    if spectrogram.ndim == 3:
        spectrogram = np.expand_dims(spectrogram, axis=0)

    # Warmup
    _ = model(spectrogram, training=False)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        pred = model(spectrogram, training=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_mc_dropout(model, spectrogram: np.ndarray, n_samples: int = 30, n_runs: int = 10):
    """MC Dropout 추론 시간 측정"""
    if spectrogram.ndim == 3:
        spectrogram = np.expand_dims(spectrogram, axis=0)

    # Warmup
    for _ in range(5):
        _ = model(spectrogram, training=True)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predictions = []
        for _ in range(n_samples):
            pred = model(spectrogram, training=True)
            predictions.append(pred.numpy())
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "n_mc_samples": n_samples,
    }


def main():
    print("=" * 60)
    print("EGAI 추론 시간 벤치마크")
    print("=" * 60)

    # 1. 테스트 오디오 찾기
    print("\n[1] 테스트 오디오 준비...")
    loader = AudioDataLoader(
        data_dir="data",
        target_shape=(128, 128),
        cache_dir="data/cache",
        fmax=6000
    )
    loader.load_metadata(fuel_type=None)

    # 유효한 오디오 파일 찾기
    test_idx = loader.valid_indices[0]
    audio_path = str(loader.get_audio_path(test_idx))
    print(f"  테스트 파일: {audio_path}")

    results = {}

    # 2. 오디오 처리 벤치마크
    print("\n[2] 오디오 처리 (로딩 + 스펙트로그램)...")
    audio_result = benchmark_audio_processing(audio_path, n_runs=5)
    results["audio_processing"] = audio_result
    print(f"  평균: {audio_result['mean']*1000:.1f}ms ± {audio_result['std']*1000:.1f}ms")

    # 3. 4채널 데이터 준비
    print("\n[3] 4채널 스펙트로그램 준비...")
    X, y = loader.prepare_dataset([test_idx], model_type="4channel_cbam", use_cache=True, verbose=False)
    print(f"  Shape: {X.shape}")

    # 4. VAE 벤치마크
    print("\n[4] VAE 추론...")
    exp_dir = Path("experiments/20260115_004055_530a70")
    if (exp_dir / "vae_encoder.keras").exists():
        encoder = tf.keras.models.load_model(
            exp_dir / "vae_encoder.keras",
            custom_objects={"Sampling": Sampling}
        )
        decoder = tf.keras.models.load_model(exp_dir / "vae_decoder.keras")

        vae_result = benchmark_vae_inference(encoder, decoder, X[0], n_runs=50)
        results["vae_inference"] = vae_result
        print(f"  평균: {vae_result['mean']*1000:.2f}ms ± {vae_result['std']*1000:.2f}ms")
    else:
        print("  VAE 모델 없음")

    # 5. Regression 모델 벤치마크 (있는 경우)
    print("\n[5] Regression 모델 추론...")
    from egai.infrastructure.models import ModelFactory, CBAM

    # 모델 생성 (또는 로드)
    regression_model = ModelFactory.create(
        model_type="4channel_cbam",
        input_shape=(128, 128, 4),
        num_outputs=5
    )

    reg_result = benchmark_regression_inference(regression_model, X[0], n_runs=50)
    results["regression_inference"] = reg_result
    print(f"  평균: {reg_result['mean']*1000:.2f}ms ± {reg_result['std']*1000:.2f}ms")

    # 6. MC Dropout 벤치마크
    print("\n[6] MC Dropout (30 샘플)...")
    mc_result = benchmark_mc_dropout(regression_model, X[0], n_samples=30, n_runs=5)
    results["mc_dropout_30"] = mc_result
    print(f"  평균: {mc_result['mean']*1000:.1f}ms ± {mc_result['std']*1000:.1f}ms")

    # 7. 전체 파이프라인
    print("\n[7] 전체 파이프라인 (오디오 → 결과)...")

    def full_pipeline(audio_path):
        # 오디오 처리
        preprocessor = AudioPreprocessor(fmax=6000, target_shape=(128, 128))
        full_spec, perc_spec = preprocessor.process(audio_path)

        # 4채널 준비
        from scipy.ndimage import uniform_filter
        full = full_spec[:, :, 0]
        perc = perc_spec[:, :, 0]
        diff = np.abs(np.diff(full, axis=1))
        diff = np.pad(diff, ((0, 0), (0, 1)), mode='edge')
        if diff.max() > 0:
            diff = diff / diff.max()
        local_mean = uniform_filter(full, size=(1, 8))
        local_sq_mean = uniform_filter(full**2, size=(1, 8))
        variance = np.maximum(local_sq_mean - local_mean**2, 0)
        if variance.max() > 0:
            variance = variance / variance.max()

        X_4ch = np.stack([full, perc, diff, variance], axis=-1)
        X_4ch = np.expand_dims(X_4ch, axis=0).astype(np.float32)

        # VAE 추론
        z_mean, z_log_var, z = encoder(X_4ch)
        reconstruction = decoder(z)
        recon_error = tf.reduce_sum(tf.keras.losses.mse(X_4ch, reconstruction), axis=(1, 2))
        kl_div = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        vae_score = (recon_error + 0.5 * kl_div).numpy()[0]

        # Regression 추론
        pred = regression_model(X_4ch, training=False).numpy()[0]

        return vae_score, pred

    # Warmup
    full_pipeline(audio_path)

    times = []
    for _ in range(5):
        start = time.perf_counter()
        vae_score, pred = full_pipeline(audio_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    full_result = {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }
    results["full_pipeline"] = full_result
    print(f"  평균: {full_result['mean']*1000:.1f}ms ± {full_result['std']*1000:.1f}ms")

    # 8. 요약
    print("\n" + "=" * 60)
    print("벤치마크 요약")
    print("=" * 60)

    print("\n┌─────────────────────────────┬────────────┬────────────┐")
    print("│ 단계                         │ 평균 (ms)  │ 비고       │")
    print("├─────────────────────────────┼────────────┼────────────┤")
    print(f"│ 오디오 처리 (MP3→스펙트로그램)│ {results['audio_processing']['mean']*1000:>8.1f}   │ librosa    │")
    if "vae_inference" in results:
        print(f"│ VAE 추론                    │ {results['vae_inference']['mean']*1000:>8.2f}   │ 이상탐지   │")
    print(f"│ Regression 추론             │ {results['regression_inference']['mean']*1000:>8.2f}   │ 품질예측   │")
    print(f"│ MC Dropout (30샘플)         │ {results['mc_dropout_30']['mean']*1000:>8.1f}   │ 불확실성   │")
    print(f"│ 전체 파이프라인              │ {results['full_pipeline']['mean']*1000:>8.1f}   │ 종합       │")
    print("└─────────────────────────────┴────────────┴────────────┘")

    # 처리 가능 속도
    fps = 1.0 / results['full_pipeline']['mean']
    print(f"\n예상 처리 속도: {fps:.1f} 파일/초")
    print(f"1000개 파일 처리: {1000 * results['full_pipeline']['mean'] / 60:.1f}분")

    return results


if __name__ == "__main__":
    main()
