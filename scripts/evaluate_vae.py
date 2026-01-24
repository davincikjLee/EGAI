"""
VAE 학습 품질 평가 스크립트

평가 항목:
1. 재구성 품질 시각화
2. 잠재 공간 분포 분석
3. 이상 점수 분포
4. 점수별 이상도 상관관계
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from egai.data.loader import AudioDataLoader
from egai.infrastructure.models import Sampling


def load_vae_model(experiment_dir: str):
    """VAE 모델 로드"""
    exp_path = Path(experiment_dir)

    # Encoder, Decoder 로드
    encoder = tf.keras.models.load_model(
        exp_path / "vae_encoder.keras",
        custom_objects={"Sampling": Sampling}
    )
    decoder = tf.keras.models.load_model(
        exp_path / "vae_decoder.keras"
    )

    # Config 로드
    with open(exp_path / "anomaly_config.json") as f:
        config = json.load(f)

    return encoder, decoder, config


def compute_anomaly_scores(encoder, decoder, X, beta=0.5):
    """이상 점수 계산"""
    z_mean, z_log_var, z = encoder(X)
    reconstruction = decoder(z)

    # Reconstruction error (per sample)
    recon_error = tf.reduce_sum(
        tf.keras.losses.mse(X, reconstruction),
        axis=(1, 2)
    ).numpy()

    # KL divergence (per sample)
    kl_div = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1
    ).numpy()

    # Total anomaly score
    anomaly_score = recon_error + beta * kl_div

    return anomaly_score, recon_error, kl_div, z.numpy(), reconstruction.numpy()


def evaluate_vae(experiment_dir: str, data_dir: str = "data", n_samples: int = 100):
    """VAE 평가 메인 함수"""
    print("=" * 60)
    print("VAE 학습 품질 평가")
    print("=" * 60)

    # 1. 모델 로드
    print("\n[1] 모델 로드...")
    encoder, decoder, config = load_vae_model(experiment_dir)
    beta = config.get("beta", 0.5)
    print(f"  - Beta: {beta}")
    print(f"  - Latent dim: {config.get('latent_dim', 128)}")
    print(f"  - VAE mean: {config.get('vae_mean', 0):.2f}")
    print(f"  - VAE std: {config.get('vae_std', 0):.2f}")

    # 2. 데이터 로드
    print("\n[2] 데이터 로드...")
    loader = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        cache_dir=f"{data_dir}/cache",
        fmax=6000
    )
    loader.load_metadata(fuel_type=None)

    # 정상 데이터 필터링 (min_score 이상)
    min_score = config.get("min_score", 4.0)
    valid_indices = [
        idx for idx in loader.valid_indices
        if loader.metadata.iloc[idx]["overall_score"] >= min_score
    ]

    # 샘플링
    if len(valid_indices) > n_samples:
        sample_indices = np.random.choice(valid_indices, n_samples, replace=False)
    else:
        sample_indices = valid_indices

    print(f"  - 전체 유효 데이터: {len(valid_indices)}")
    print(f"  - 평가 샘플: {len(sample_indices)}")

    # 3. 4채널 데이터 준비
    print("\n[3] 스펙트로그램 준비...")
    X, y = loader.prepare_dataset(
        list(sample_indices),
        model_type="4channel_cbam",
        use_cache=True,
        verbose=False
    )
    print(f"  - Shape: {X.shape}")

    # 4. 이상 점수 계산
    print("\n[4] 이상 점수 계산...")
    anomaly_scores, recon_errors, kl_divs, latent_z, reconstructions = compute_anomaly_scores(
        encoder, decoder, X, beta
    )

    # 5. 결과 분석
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)

    print("\n[이상 점수 통계]")
    print(f"  Mean: {np.mean(anomaly_scores):.2f}")
    print(f"  Std:  {np.std(anomaly_scores):.2f}")
    print(f"  Min:  {np.min(anomaly_scores):.2f}")
    print(f"  Max:  {np.max(anomaly_scores):.2f}")
    print(f"  Median: {np.median(anomaly_scores):.2f}")

    print("\n[재구성 오차 통계]")
    print(f"  Mean: {np.mean(recon_errors):.2f}")
    print(f"  Std:  {np.std(recon_errors):.2f}")

    print("\n[KL Divergence 통계]")
    print(f"  Mean: {np.mean(kl_divs):.2f}")
    print(f"  Std:  {np.std(kl_divs):.2f}")

    # 6. 잠재 공간 분석
    print("\n[잠재 공간 분석]")
    z_mean = np.mean(latent_z, axis=0)
    z_std = np.std(latent_z, axis=0)
    print(f"  Z mean range: [{z_mean.min():.2f}, {z_mean.max():.2f}]")
    print(f"  Z std range:  [{z_std.min():.2f}, {z_std.max():.2f}]")

    # 잠재 공간이 N(0,1)에 가까운지 확인
    overall_z_mean = np.mean(np.abs(z_mean))
    overall_z_std = np.mean(z_std)
    print(f"  Average |z_mean|: {overall_z_mean:.4f} (이상적: 0에 가까움)")
    print(f"  Average z_std: {overall_z_std:.4f} (이상적: 1에 가까움)")

    # 7. 재구성 품질 (SSIM-like)
    print("\n[재구성 품질]")
    # 픽셀별 오차
    pixel_errors = np.mean(np.abs(X - reconstructions), axis=(1, 2, 3))
    print(f"  Mean Absolute Error: {np.mean(pixel_errors):.4f}")
    print(f"  (0에 가까울수록 좋음, 입력 범위 0~1)")

    # 상관계수
    correlations = []
    for i in range(min(10, len(X))):
        corr = np.corrcoef(X[i].flatten(), reconstructions[i].flatten())[0, 1]
        correlations.append(corr)
    print(f"  Mean Correlation: {np.mean(correlations):.4f}")
    print(f"  (1에 가까울수록 좋음)")

    # 8. 점수별 이상도 분석
    print("\n[점수별 이상도 상관관계]")
    overall_scores = y.mean(axis=1)  # 5개 점수의 평균
    correlation = np.corrcoef(overall_scores, anomaly_scores)[0, 1]
    print(f"  품질 점수 vs 이상 점수 상관계수: {correlation:.4f}")
    print(f"  (음수: 낮은 품질 = 높은 이상도, 예상대로)")

    # 9. 임계값 분석
    print("\n[임계값 분석]")
    vae_mean = config.get("vae_mean", np.mean(anomaly_scores))
    vae_std = config.get("vae_std", np.std(anomaly_scores))

    threshold_1sigma = vae_mean + 1 * vae_std
    threshold_2sigma = vae_mean + 2 * vae_std
    threshold_3sigma = vae_mean + 3 * vae_std

    pct_above_1sigma = np.mean(anomaly_scores > threshold_1sigma) * 100
    pct_above_2sigma = np.mean(anomaly_scores > threshold_2sigma) * 100
    pct_above_3sigma = np.mean(anomaly_scores > threshold_3sigma) * 100

    print(f"  μ + 1σ ({threshold_1sigma:.1f}): {pct_above_1sigma:.1f}% 초과 (예상: 16%)")
    print(f"  μ + 2σ ({threshold_2sigma:.1f}): {pct_above_2sigma:.1f}% 초과 (예상: 2.5%)")
    print(f"  μ + 3σ ({threshold_3sigma:.1f}): {pct_above_3sigma:.1f}% 초과 (예상: 0.15%)")

    # 10. 시각화 저장
    print("\n[시각화 생성 중...]")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1) 이상 점수 분포
    ax = axes[0, 0]
    ax.hist(anomaly_scores, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(vae_mean, color='g', linestyle='--', label=f'Mean: {vae_mean:.1f}')
    ax.axvline(threshold_2sigma, color='orange', linestyle='--', label=f'μ+2σ: {threshold_2sigma:.1f}')
    ax.axvline(threshold_3sigma, color='r', linestyle='--', label=f'μ+3σ: {threshold_3sigma:.1f}')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Anomaly Score Distribution')
    ax.legend()

    # 2) 재구성 오차 vs KL Divergence
    ax = axes[0, 1]
    ax.scatter(recon_errors, kl_divs, alpha=0.5, c=overall_scores, cmap='RdYlGn')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Recon Error vs KL (color=quality)')
    plt.colorbar(ax.collections[0], ax=ax, label='Quality Score')

    # 3) 품질 점수 vs 이상 점수
    ax = axes[0, 2]
    ax.scatter(overall_scores, anomaly_scores, alpha=0.5)
    z = np.polyfit(overall_scores, anomaly_scores, 1)
    p = np.poly1d(z)
    ax.plot(overall_scores, p(overall_scores), "r--", alpha=0.8)
    ax.set_xlabel('Quality Score (mean of 5)')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'Quality vs Anomaly (r={correlation:.3f})')

    # 4) 잠재 공간 2D (PCA)
    ax = axes[1, 0]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latent_z)
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=anomaly_scores, cmap='viridis', alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Latent Space (PCA, color=anomaly)')
    plt.colorbar(scatter, ax=ax)

    # 5) 재구성 예시 (가장 정상적인 샘플)
    ax = axes[1, 1]
    best_idx = np.argmin(anomaly_scores)
    # 첫 번째 채널만 표시
    combined = np.hstack([X[best_idx, :, :, 0], reconstructions[best_idx, :, :, 0]])
    ax.imshow(combined, aspect='auto', origin='lower', cmap='magma')
    ax.axvline(128, color='white', linestyle='--')
    ax.set_title(f'Best Reconstruction (score={anomaly_scores[best_idx]:.1f})')
    ax.set_xlabel('Original | Reconstructed')

    # 6) 재구성 예시 (가장 이상한 샘플)
    ax = axes[1, 2]
    worst_idx = np.argmax(anomaly_scores)
    combined = np.hstack([X[worst_idx, :, :, 0], reconstructions[worst_idx, :, :, 0]])
    ax.imshow(combined, aspect='auto', origin='lower', cmap='magma')
    ax.axvline(128, color='white', linestyle='--')
    ax.set_title(f'Worst Reconstruction (score={anomaly_scores[worst_idx]:.1f})')
    ax.set_xlabel('Original | Reconstructed')

    plt.tight_layout()

    # 저장
    output_path = Path(experiment_dir) / "evaluation_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n시각화 저장: {output_path}")

    # 11. 평가 요약
    print("\n" + "=" * 60)
    print("학습 품질 평가 요약")
    print("=" * 60)

    # 점수 계산
    quality_metrics = {
        "reconstruction": min(1.0, 1.0 - np.mean(pixel_errors) * 5),  # 0~1
        "latent_regularity": min(1.0, 1.0 / (1.0 + overall_z_mean)),  # z_mean이 0에 가까우면 1
        "correlation": np.mean(correlations),  # 재구성 상관계수
        "score_consistency": max(0, -correlation),  # 품질과 이상도 반비례하면 좋음
    }

    overall_quality = np.mean(list(quality_metrics.values()))

    print(f"\n  재구성 품질:     {quality_metrics['reconstruction']:.2%}")
    print(f"  잠재공간 정규성: {quality_metrics['latent_regularity']:.2%}")
    print(f"  재구성 상관계수: {quality_metrics['correlation']:.2%}")
    print(f"  점수 일관성:     {quality_metrics['score_consistency']:.2%}")
    print(f"\n  종합 품질 점수: {overall_quality:.2%}")

    if overall_quality >= 0.7:
        print("\n  ✓ 학습이 잘 되었습니다!")
    elif overall_quality >= 0.5:
        print("\n  △ 보통 수준입니다. 하이퍼파라미터 튜닝 권장")
    else:
        print("\n  ✗ 추가 학습이 필요합니다.")

    return {
        "anomaly_scores": anomaly_scores,
        "quality_metrics": quality_metrics,
        "overall_quality": overall_quality,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAE 학습 품질 평가")
    parser.add_argument(
        "--experiment", type=str,
        default="experiments/20260115_004055_530a70",
        help="실험 디렉토리"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="데이터 디렉토리"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="평가할 샘플 수"
    )

    args = parser.parse_args()

    evaluate_vae(
        experiment_dir=args.experiment,
        data_dir=args.data_dir,
        n_samples=args.n_samples
    )
