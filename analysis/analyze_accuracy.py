"""
정확도 분석 스크립트
역할: 연속 회귀 예측에서 정수 라벨에 대한 정확도 지표 계산

분석 지표:
- Within-0.5: |예측 - 실제| <= 0.5 인 비율
- Within-1: |예측 - 실제| <= 1 인 비율 (±1점 이내)
- Exact Match: 반올림 후 정확히 일치하는 비율
- MAE: 평균 절대 오차
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from data_loader import AudioDataLoader
from models import get_model


def calculate_accuracy_metrics(y_true, y_pred):
    """
    정수 라벨에 대한 정확도 지표 계산

    매개변수:
        y_true: 실제값 (n_samples, n_outputs) - 정수값
        y_pred: 예측값 (n_samples, n_outputs) - 연속값

    반환값:
        딕셔너리: 각종 정확도 지표
    """
    abs_error = np.abs(y_pred - y_true)

    metrics = {
        'within_0.5': np.mean(abs_error <= 0.5) * 100,  # ±0.5 이내 (%)
        'within_1.0': np.mean(abs_error <= 1.0) * 100,  # ±1.0 이내 (%)
        'within_1.5': np.mean(abs_error <= 1.5) * 100,  # ±1.5 이내 (%)
        'exact_match': np.mean(np.round(y_pred) == y_true) * 100,  # 반올림 후 정확히 맞음 (%)
        'mae': np.mean(abs_error),
        'r2': r2_score(y_true.flatten(), y_pred.flatten()),
    }

    # 오차 분포
    metrics['error_distribution'] = {
        '0.0-0.5': np.mean(abs_error <= 0.5) * 100,
        '0.5-1.0': np.mean((abs_error > 0.5) & (abs_error <= 1.0)) * 100,
        '1.0-1.5': np.mean((abs_error > 1.0) & (abs_error <= 1.5)) * 100,
        '1.5-2.0': np.mean((abs_error > 1.5) & (abs_error <= 2.0)) * 100,
        '2.0+': np.mean(abs_error > 2.0) * 100,
    }

    return metrics


def calculate_column_accuracy(y_true, y_pred, column_names):
    """
    컬럼별 정확도 계산
    """
    column_metrics = {}

    for i, col in enumerate(column_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        abs_error = np.abs(col_pred - col_true)

        column_metrics[col] = {
            'within_0.5': np.mean(abs_error <= 0.5) * 100,
            'within_1.0': np.mean(abs_error <= 1.0) * 100,
            'exact_match': np.mean(np.round(col_pred) == col_true) * 100,
            'mae': np.mean(abs_error),
            'unique_true': np.unique(col_true).tolist(),
            'pred_range': (float(col_pred.min()), float(col_pred.max())),
        }

    return column_metrics


def run_accuracy_analysis(n_folds=5, epochs=50, batch_size=16):
    """
    베이스라인 모델(Simple CNN)로 정확도 분석 실행
    """
    print("=" * 70)
    print("정확도 분석 (Simple CNN Baseline)")
    print("=" * 70)

    # 데이터 로더 초기화
    data_dir = PROJECT_ROOT / "data"
    cache_dir = PROJECT_ROOT / "analysis" / "models" / "cv" / "cache"

    loader = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        cache_dir=cache_dir
    )
    loader.load_metadata()

    print(f"\n총 샘플 수: {len(loader.valid_indices)}")
    print(f"타겟 컬럼: {loader.target_columns}")

    # CV 설정
    indices = np.array(loader.valid_indices)
    overall_scores = loader.metadata.iloc[indices]['overall_score'].values
    bins = np.linspace(overall_scores.min(), overall_scores.max() + 0.01, 6)
    stratify_labels = np.digitize(overall_scores, bins)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 모든 fold의 예측/실제값 수집
    all_y_true = []
    all_y_pred = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, stratify_labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        train_indices = indices[train_idx].tolist()
        val_indices = indices[val_idx].tolist()

        # 데이터 준비
        X_train, y_train = loader.prepare_dataset(train_indices, model_type='simple', verbose=False)
        X_val, y_val = loader.prepare_dataset(val_indices, model_type='simple', verbose=False)

        # 모델 생성 및 학습
        model = get_model('simple', num_outputs=len(loader.target_columns))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
                ),
            ],
            verbose=0
        )

        # 예측
        y_pred = model.predict(X_val, verbose=0)

        # 수집
        all_y_true.append(y_val)
        all_y_pred.append(y_pred)

        # Fold별 정확도 계산
        fold_metric = calculate_accuracy_metrics(y_val, y_pred)
        fold_metrics.append(fold_metric)
        print(f"  Within-0.5: {fold_metric['within_0.5']:.1f}%, "
              f"Within-1: {fold_metric['within_1.0']:.1f}%, "
              f"Exact: {fold_metric['exact_match']:.1f}%, "
              f"MAE: {fold_metric['mae']:.4f}")

        # 메모리 정리
        keras.backend.clear_session()
        del model

    # 전체 결과 집계
    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)

    print("\n" + "=" * 70)
    print("전체 정확도 분석 결과")
    print("=" * 70)

    # 전체 정확도
    overall_metrics = calculate_accuracy_metrics(all_y_true, all_y_pred)

    print(f"\n[전체 지표] (n={len(all_y_true)}개 예측)")
    print(f"  ±0.5점 이내 정확도: {overall_metrics['within_0.5']:.1f}%")
    print(f"  ±1.0점 이내 정확도: {overall_metrics['within_1.0']:.1f}%")
    print(f"  ±1.5점 이내 정확도: {overall_metrics['within_1.5']:.1f}%")
    print(f"  반올림 Exact Match: {overall_metrics['exact_match']:.1f}%")
    print(f"  MAE: {overall_metrics['mae']:.4f}")
    print(f"  R²: {overall_metrics['r2']:.4f}")

    print(f"\n[오차 분포]")
    for range_name, pct in overall_metrics['error_distribution'].items():
        count = int(pct / 100 * len(all_y_true) * len(loader.target_columns))
        bar = "#" * int(pct / 2)
        print(f"  {range_name:8s}: {pct:5.1f}% ({count:4d}) {bar}")

    # 컬럼별 정확도
    column_metrics = calculate_column_accuracy(all_y_true, all_y_pred, loader.target_columns)

    print(f"\n[컬럼별 정확도]")
    print(f"{'컬럼':25s} | {'±0.5':>6s} | {'±1.0':>6s} | {'Exact':>6s} | {'MAE':>6s} | 실제값")
    print("-" * 85)

    for col, m in column_metrics.items():
        unique_str = ','.join(map(str, [int(x) for x in m['unique_true']]))
        print(f"{col:25s} | {m['within_0.5']:5.1f}% | {m['within_1.0']:5.1f}% | "
              f"{m['exact_match']:5.1f}% | {m['mae']:.4f} | {unique_str}")

    # Fold별 집계 (mean ± std)
    print(f"\n[Fold별 통계]")
    within_05 = [m['within_0.5'] for m in fold_metrics]
    within_10 = [m['within_1.0'] for m in fold_metrics]
    exact = [m['exact_match'] for m in fold_metrics]
    maes = [m['mae'] for m in fold_metrics]

    print(f"  ±0.5점 이내: {np.mean(within_05):.1f}% ± {np.std(within_05):.1f}%")
    print(f"  ±1.0점 이내: {np.mean(within_10):.1f}% ± {np.std(within_10):.1f}%")
    print(f"  Exact Match: {np.mean(exact):.1f}% ± {np.std(exact):.1f}%")
    print(f"  MAE:         {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # 예측값 분포 분석
    print(f"\n[예측값 vs 실제값 분포]")
    print(f"  실제값 범위: {all_y_true.min():.0f} ~ {all_y_true.max():.0f} (정수)")
    print(f"  예측값 범위: {all_y_pred.min():.2f} ~ {all_y_pred.max():.2f} (연속)")
    print(f"  예측값 평균: {all_y_pred.mean():.2f}, 표준편차: {all_y_pred.std():.2f}")

    # 반올림 예측의 혼동 분석
    print(f"\n[반올림 예측 분석]")
    rounded_pred = np.round(all_y_pred)
    for score in range(1, 6):
        true_mask = all_y_true == score
        if true_mask.sum() > 0:
            pred_at_score = rounded_pred[true_mask]
            correct = (pred_at_score == score).sum()
            total = true_mask.sum()
            print(f"  실제 {score}: {total:4d} 중 {correct:4d} 정확 ({correct/total*100:.1f}%)")

    return {
        'overall': overall_metrics,
        'column': column_metrics,
        'fold_metrics': fold_metrics,
        'y_true': all_y_true,
        'y_pred': all_y_pred
    }


if __name__ == "__main__":
    results = run_accuracy_analysis(n_folds=5, epochs=50, batch_size=16)
