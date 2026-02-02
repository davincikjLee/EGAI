"""
VAE 가솔린/디젤 분류 평가 스크립트

가솔린만으로 학습된 VAE 모델로 디젤 데이터가 OOD로 탐지되는지 평가
- 가솔린 vs 디젤 VAE 점수 분포 비교
- Welch's t-test
- ROC-AUC (디젤을 양성 클래스로)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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

    return anomaly_score, recon_error, kl_div


def evaluate_vae_diesel(experiment_dir: str, data_dir: str = "data"):
    """VAE 가솔린/디젤 분류 평가"""
    print("=" * 60)
    print("VAE 가솔린/디젤 분류 평가 (OOD Detection)")
    print("=" * 60)

    # 1. 모델 로드
    print("\n[1] 모델 로드...")
    encoder, decoder, config = load_vae_model(experiment_dir)
    beta = config.get("beta", 0.5)
    print(f"  - Beta: {beta}")
    print(f"  - Latent dim: {config.get('latent_dim', 128)}")
    print(f"  - 학습 연료 타입: {config.get('fuel_type', '전체')}")

    # 2. 가솔린 데이터 로드
    print("\n[2] 가솔린 데이터 로드...")
    loader_gasoline = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        cache_dir=f"{data_dir}/cache",
        fmax=6000
    )
    loader_gasoline.load_metadata(fuel_type="gasoline")

    # 가솔린 샘플 (최대 500개)
    gasoline_indices = loader_gasoline.valid_indices[:500]
    X_gasoline, _ = loader_gasoline.prepare_dataset(
        gasoline_indices,
        model_type="4channel_cbam",
        use_cache=True,
        verbose=False
    )
    print(f"  - 가솔린 샘플: {len(X_gasoline)}개")

    # 3. 디젤 데이터 로드
    print("\n[3] 디젤 데이터 로드...")
    loader_diesel = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        cache_dir=f"{data_dir}/cache",
        fmax=6000
    )
    loader_diesel.load_metadata(fuel_type="diesel")

    # 디젤 샘플 (전체 사용)
    diesel_indices = loader_diesel.valid_indices
    X_diesel, _ = loader_diesel.prepare_dataset(
        diesel_indices,
        model_type="4channel_cbam",
        use_cache=True,
        verbose=False
    )
    print(f"  - 디젤 샘플: {len(X_diesel)}개")

    # 4. 이상 점수 계산
    print("\n[4] 이상 점수 계산...")
    scores_gasoline, recon_gasoline, kl_gasoline = compute_anomaly_scores(
        encoder, decoder, X_gasoline, beta
    )
    scores_diesel, recon_diesel, kl_diesel = compute_anomaly_scores(
        encoder, decoder, X_diesel, beta
    )

    # 5. 통계 분석
    print("\n" + "=" * 60)
    print("분석 결과")
    print("=" * 60)

    print("\n[가솔린 VAE 점수 통계]")
    print(f"  Mean: {np.mean(scores_gasoline):.2f}")
    print(f"  Std:  {np.std(scores_gasoline):.2f}")
    print(f"  Min:  {np.min(scores_gasoline):.2f}")
    print(f"  Max:  {np.max(scores_gasoline):.2f}")

    print("\n[디젤 VAE 점수 통계]")
    print(f"  Mean: {np.mean(scores_diesel):.2f}")
    print(f"  Std:  {np.std(scores_diesel):.2f}")
    print(f"  Min:  {np.min(scores_diesel):.2f}")
    print(f"  Max:  {np.max(scores_diesel):.2f}")

    # 6. Welch's t-test
    print("\n[Welch's t-test]")
    t_stat, p_value = stats.ttest_ind(scores_diesel, scores_gasoline, equal_var=False)
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")

    if p_value < 0.001:
        print("  -> 통계적으로 유의미한 차이 (p < 0.001) [PASS]")
    elif p_value < 0.05:
        print("  -> 통계적으로 유의미한 차이 (p < 0.05)")
    else:
        print("  -> 유의미한 차이 없음")

    # 7. ROC-AUC 계산 (디젤 = 양성)
    print("\n[ROC-AUC 분석]")
    y_true = np.concatenate([
        np.zeros(len(scores_gasoline)),  # 가솔린 = 0
        np.ones(len(scores_diesel))       # 디젤 = 1
    ])
    y_scores = np.concatenate([scores_gasoline, scores_diesel])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # 최적 임계값 (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    print(f"  최적 임계값 (Youden's J): {best_threshold:.2f}")
    print(f"  - TPR (Recall): {tpr[best_idx]:.4f}")
    print(f"  - FPR: {fpr[best_idx]:.4f}")

    # 8. PR-AUC
    print("\n[PR-AUC 분석]")
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    print(f"  PR-AUC: {pr_auc:.4f}")

    # 9. 분포 분리 분석
    print("\n[분포 분리 분석]")
    gasoline_95pct = np.percentile(scores_gasoline, 95)
    diesel_above = np.mean(scores_diesel > gasoline_95pct) * 100
    print(f"  가솔린 95 percentile: {gasoline_95pct:.2f}")
    print(f"  디젤 중 이 값 초과 비율: {diesel_above:.1f}%")

    # Cohen's d (효과 크기)
    pooled_std = np.sqrt((np.std(scores_gasoline)**2 + np.std(scores_diesel)**2) / 2)
    cohens_d = (np.mean(scores_diesel) - np.mean(scores_gasoline)) / pooled_std
    print(f"  Cohen's d (효과 크기): {cohens_d:.4f}")

    if abs(cohens_d) >= 0.8:
        effect_size = "큰 효과"
    elif abs(cohens_d) >= 0.5:
        effect_size = "중간 효과"
    elif abs(cohens_d) >= 0.2:
        effect_size = "작은 효과"
    else:
        effect_size = "효과 없음"
    print(f"  → {effect_size}")

    # 10. 시각화
    print("\n[시각화 생성 중...]")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) VAE 점수 분포 히스토그램
    ax = axes[0, 0]
    ax.hist(scores_gasoline, bins=50, alpha=0.7, label=f'Gasoline (n={len(scores_gasoline)})', color='blue')
    ax.hist(scores_diesel, bins=30, alpha=0.7, label=f'Diesel (n={len(scores_diesel)})', color='red')
    ax.axvline(best_threshold, color='green', linestyle='--', label=f'Threshold: {best_threshold:.1f}')
    ax.set_xlabel('VAE Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title('VAE Score Distribution: Gasoline vs Diesel')
    ax.legend()

    # 2) ROC 곡선
    ax = axes[0, 1]
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5, label='Best Threshold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Diesel = Positive)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) PR 곡선
    ax = axes[1, 0]
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    ax.axhline(len(scores_diesel) / (len(scores_gasoline) + len(scores_diesel)),
               color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Box Plot
    ax = axes[1, 1]
    bp = ax.boxplot([scores_gasoline, scores_diesel], labels=['Gasoline', 'Diesel'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('salmon')
    ax.set_ylabel('VAE Anomaly Score')
    ax.set_title(f'Score Comparison (p={p_value:.2e})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 저장
    output_path = Path(experiment_dir) / "gasoline_diesel_evaluation.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n시각화 저장: {output_path}")

    # 11. 결과 요약
    print("\n" + "=" * 60)
    print("평가 요약")
    print("=" * 60)

    success_criteria = {
        "통계적 유의성 (p < 0.001)": p_value < 0.001,
        "ROC-AUC > 0.70": roc_auc > 0.70,
        "Cohen's d > 0.5 (중간 효과)": abs(cohens_d) > 0.5,
        "디젤 95% > 가솔린 95%": np.mean(scores_diesel > gasoline_95pct) > 0.5,
    }

    passed = sum(success_criteria.values())
    total = len(success_criteria)

    print(f"\n성공 기준 ({passed}/{total}):")
    for criterion, result in success_criteria.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  [{status}] {criterion}")

    if passed >= 3:
        print("\n→ 실험 성공: VAE가 가솔린과 디젤을 구분할 수 있습니다!")
        print("  디젤을 'NG'로 취급하는 OK/NG 실험이 유효합니다.")
    elif passed >= 2:
        print("\n→ 부분적 성공: 일부 구분 가능하지만 개선 필요")
    else:
        print("\n→ 실험 실패: VAE로는 가솔린/디젤 구분이 어렵습니다.")
        print("  다른 접근법 (이진 분류)을 시도해 보세요.")

    # 결과 저장
    results = {
        "gasoline_mean": float(np.mean(scores_gasoline)),
        "gasoline_std": float(np.std(scores_gasoline)),
        "diesel_mean": float(np.mean(scores_diesel)),
        "diesel_std": float(np.std(scores_diesel)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "cohens_d": float(cohens_d),
        "best_threshold": float(best_threshold),
        "passed_criteria": int(passed),
        "total_criteria": int(total),
    }

    results_path = Path(experiment_dir) / "gasoline_diesel_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAE 가솔린/디젤 분류 평가")
    parser.add_argument(
        "--experiment", type=str, required=True,
        help="가솔린으로 학습된 VAE 실험 디렉토리"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="데이터 디렉토리"
    )

    args = parser.parse_args()

    evaluate_vae_diesel(
        experiment_dir=args.experiment,
        data_dir=args.data_dir
    )
