"""
K-Fold Cross-Validation 학습 스크립트
역할: 적은 데이터에서 더 안정적인 모델 평가
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU 사용 가능: {len(gpus)}개")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from data_loader import AudioDataLoader
from models import get_model
from experiment_logger import ExperimentLogger, load_config


def create_cv_callbacks(fold, save_dir):
    """
    Cross-validation용 콜백 생성
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
    ]

    return callbacks


def train_fold(model_type, data_loader, train_idx, val_idx, fold_num,
               batch_size=16, epochs=100, save_dir="models", learning_rate=1e-3):
    """
    단일 Fold 학습

    반환값:
        val_results: 검증 결과 딕셔너리
    """
    print(f"\n--- Fold {fold_num + 1} ---")

    # 데이터 준비
    X_train, y_train = data_loader.prepare_dataset(train_idx, model_type=model_type, verbose=False)
    X_val, y_val = data_loader.prepare_dataset(val_idx, model_type=model_type, verbose=False)

    # 모델 생성 (매 Fold마다 새로 초기화)
    # metadata_dim은 multiinput_metadata 모델에서 필요
    # physics_dim은 multiinput_physics 모델에서 필요
    # modulation_dim은 multiinput_modulation 모델에서 필요
    metadata_dim = getattr(data_loader, 'metadata_dim', None)
    physics_dim = getattr(data_loader, 'physics_dim', None)
    modulation_dim = getattr(data_loader, 'modulation_dim', None)
    model = get_model(
        model_type,
        num_outputs=len(data_loader.target_columns),
        metadata_dim=metadata_dim,
        physics_dim=physics_dim,
        modulation_dim=modulation_dim
    )

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    # 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=create_cv_callbacks(fold_num, save_dir),
        verbose=0
    )

    # 예측
    y_pred = model.predict(X_val, verbose=0)

    # 평가
    overall_mae = mean_absolute_error(y_val, y_pred)
    overall_r2 = r2_score(y_val, y_pred)

    # 컬럼별 평가
    column_metrics = {}
    for i, col in enumerate(data_loader.target_columns):
        mae = mean_absolute_error(y_val[:, i], y_pred[:, i])
        r2 = r2_score(y_val[:, i], y_pred[:, i])
        column_metrics[col] = {'MAE': mae, 'R2': r2}

    # 학습 에포크 수
    actual_epochs = len(history.history['loss'])

    print(f"  Epochs: {actual_epochs}, Val MAE: {overall_mae:.4f}, Val R²: {overall_r2:.4f}")

    # 메모리 정리
    keras.backend.clear_session()
    del model

    return {
        'overall_MAE': overall_mae,
        'overall_R2': overall_r2,
        'column_metrics': column_metrics,
        'epochs': actual_epochs
    }


def run_cross_validation(model_type, data_loader, n_folds=5, batch_size=16,
                         epochs=100, save_dir="models", stratify=True,
                         logger=None, learning_rate=1e-3):
    """
    K-Fold Cross-Validation 실행

    매개변수:
        model_type: 'simple', 'dual', 'resnet', 'multiinput_metadata'
        data_loader: AudioDataLoader 인스턴스
        n_folds: Fold 수
        batch_size: 배치 크기
        epochs: 최대 에포크 수
        save_dir: 저장 디렉토리
        stratify: 층화 추출 사용 여부
        logger: ExperimentLogger 인스턴스 (선택)
        learning_rate: 학습률

    반환값:
        cv_results: 전체 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} 모델 {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")

    indices = np.array(data_loader.valid_indices)
    random_state = 42

    # 층화 추출용 레이블 생성
    if stratify:
        overall_scores = data_loader.metadata.iloc[indices]['overall_score'].values
        bins = np.linspace(overall_scores.min(), overall_scores.max() + 0.01, 6)
        stratify_labels = np.digitize(overall_scores, bins)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(indices, stratify_labels))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(indices))

    # CV 분할 정보 저장 (재현성)
    cv_splits = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        cv_splits.append({
            "train": indices[train_idx].tolist(),
            "val": indices[val_idx].tolist()
        })

    # 로거에 CV 분할 저장
    if logger:
        # 첫 번째 fold의 train/val을 기본값으로 사용
        logger.save_data_splits(
            train_idx=cv_splits[0]["train"],
            val_idx=cv_splits[0]["val"],
            cv_splits=cv_splits,
            random_state=random_state
        )

    # 각 Fold 결과 저장
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_indices = indices[train_idx].tolist()
        val_indices = indices[val_idx].tolist()

        result = train_fold(
            model_type=model_type,
            data_loader=data_loader,
            train_idx=train_indices,
            val_idx=val_indices,
            fold_num=fold,
            batch_size=batch_size,
            epochs=epochs,
            save_dir=save_dir,
            learning_rate=learning_rate
        )
        fold_results.append(result)

        # 로거에 fold 결과 저장
        if logger:
            logger.log_fold_result(fold, {
                "MAE": float(result['overall_MAE']),
                "R2": float(result['overall_R2']),
                "epochs": result['epochs']
            })

    # 결과 집계
    cv_results = aggregate_cv_results(fold_results, data_loader.target_columns)
    print_cv_results(cv_results, model_type)

    return cv_results


def aggregate_cv_results(fold_results, target_columns):
    """
    Cross-Validation 결과 집계

    반환값:
        집계된 결과 딕셔너리 (mean, std 포함)
    """
    overall_maes = [r['overall_MAE'] for r in fold_results]
    overall_r2s = [r['overall_R2'] for r in fold_results]
    epochs_list = [r['epochs'] for r in fold_results]

    # 컬럼별 집계
    column_metrics = {}
    for col in target_columns:
        col_maes = [r['column_metrics'][col]['MAE'] for r in fold_results]
        col_r2s = [r['column_metrics'][col]['R2'] for r in fold_results]
        column_metrics[col] = {
            'MAE_mean': np.mean(col_maes),
            'MAE_std': np.std(col_maes),
            'R2_mean': np.mean(col_r2s),
            'R2_std': np.std(col_r2s)
        }

    return {
        'n_folds': len(fold_results),
        'overall_MAE_mean': np.mean(overall_maes),
        'overall_MAE_std': np.std(overall_maes),
        'overall_R2_mean': np.mean(overall_r2s),
        'overall_R2_std': np.std(overall_r2s),
        'avg_epochs': np.mean(epochs_list),
        'column_metrics': column_metrics,
        'fold_results': fold_results
    }


def print_cv_results(cv_results, model_name):
    """Cross-Validation 결과 출력"""
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} Cross-Validation 결과 ({cv_results['n_folds']}-Fold)")
    print(f"{'='*60}")
    print(f"전체 MAE: {cv_results['overall_MAE_mean']:.4f} ± {cv_results['overall_MAE_std']:.4f}")
    print(f"전체 R²:  {cv_results['overall_R2_mean']:.4f} ± {cv_results['overall_R2_std']:.4f}")
    print(f"평균 에포크: {cv_results['avg_epochs']:.1f}")

    print(f"\n컬럼별 결과:")
    for col, metrics in cv_results['column_metrics'].items():
        print(f"  {col:25s} - MAE: {metrics['MAE_mean']:.4f} ± {metrics['MAE_std']:.4f}, "
              f"R²: {metrics['R2_mean']:.4f} ± {metrics['R2_std']:.4f}")


def save_cv_results(all_results, save_dir):
    """결과 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # JSON 직렬화 가능하게 변환
    serializable_results = {}
    for model_type, results in all_results.items():
        serializable_results[model_type] = {
            'n_folds': results['n_folds'],
            'overall_MAE_mean': float(results['overall_MAE_mean']),
            'overall_MAE_std': float(results['overall_MAE_std']),
            'overall_R2_mean': float(results['overall_R2_mean']),
            'overall_R2_std': float(results['overall_R2_std']),
            'avg_epochs': float(results['avg_epochs']),
            'column_metrics': {
                col: {k: float(v) for k, v in m.items()}
                for col, m in results['column_metrics'].items()
            }
        }

    save_path = save_dir / "cv_results.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\nCV 결과 저장: {save_path}")


def compare_cv_results(all_results):
    """모델 비교 출력"""
    print(f"\n{'='*70}")
    print("모델 비교 (Cross-Validation)")
    print(f"{'='*70}")

    print(f"\n{'모델':15s} | {'MAE (mean±std)':>20s} | {'R² (mean±std)':>20s}")
    print("-" * 60)

    for model_type, results in all_results.items():
        mae_str = f"{results['overall_MAE_mean']:.4f} ± {results['overall_MAE_std']:.4f}"
        r2_str = f"{results['overall_R2_mean']:.4f} ± {results['overall_R2_std']:.4f}"
        print(f"{model_type:15s} | {mae_str:>20s} | {r2_str:>20s}")

    # 최고 모델
    best_model = min(all_results.items(), key=lambda x: x[1]['overall_MAE_mean'])
    print(f"\n최고 성능 모델: {best_model[0]} (MAE: {best_model[1]['overall_MAE_mean']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='EGAI K-Fold Cross-Validation')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'simple_cbam', 'channel_concat', 'dual', 'resnet', 'multiinput_metadata', 'multiinput_physics', 'multiinput_modulation', 'all'],
                        help='학습할 모델 타입')
    parser.add_argument('--folds', type=int, default=5,
                        help='K-Fold의 K 값 (기본: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='최대 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='학습률 (기본: 0.001)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='데이터 디렉토리')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='결과 저장 디렉토리')
    parser.add_argument('--cache', action='store_true',
                        help='스펙트로그램 캐싱 사용')
    parser.add_argument('--fuel_type', type=str, default=None,
                        choices=['가솔린', '디젤'],
                        help='연료 타입 필터 (가솔린/디젤)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='실험 이름 (지정 시 실험 추적 활성화)')

    args = parser.parse_args()

    # 경로 설정
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "analysis" / "models" / "cv"
    cache_dir = save_dir / "cache" if args.cache else None
    experiments_dir = PROJECT_ROOT / "analysis" / "experiments"

    print("=" * 60)
    print("EGAI K-Fold Cross-Validation")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"K-Fold: {args.folds}")
    print(f"최대 에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습률: {args.lr}")
    print(f"데이터 경로: {data_dir}")
    print(f"저장 경로: {save_dir}")
    print(f"캐싱: {'활성화' if cache_dir else '비활성화'}")
    print(f"연료 타입: {args.fuel_type if args.fuel_type else '전체'}")
    print(f"실험 추적: {'활성화' if args.exp_name else '비활성화'}")

    # 데이터 로더 초기화
    loader = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        cache_dir=cache_dir
    )

    # 메타데이터 로드
    loader.load_metadata(fuel_type=args.fuel_type)

    # 학습할 모델 목록
    if args.model == 'all':
        model_types = ['simple', 'dual', 'resnet']
    else:
        model_types = [args.model]

    # 결과 저장
    all_results = {}

    # 각 모델 CV 실행
    for model_type in model_types:
        try:
            # 실험 로거 설정
            logger = None
            if args.exp_name:
                logger = ExperimentLogger(experiments_dir)
                exp_name = args.exp_name if len(model_types) == 1 else f"{args.exp_name}_{model_type}"

                # 실험 설정 구성
                config = {
                    "seed": 42,
                    "data": {
                        "n_samples": len(loader.valid_indices),
                        "target_columns": loader.target_columns,
                    },
                    "preprocessing": {
                        "target_shape": [128, 128],
                        "use_mel": False,
                        "fmax": 6000,
                    },
                    "model": {
                        "type": model_type,
                        "metadata_dim": getattr(loader, 'metadata_dim', None),
                        "physics_dim": getattr(loader, 'physics_dim', None),
                        "modulation_dim": getattr(loader, 'modulation_dim', None),
                    },
                    "training": {
                        "n_folds": args.folds,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.lr,
                    },
                }
                logger.start_experiment(exp_name, config)

            results = run_cross_validation(
                model_type=model_type,
                data_loader=loader,
                n_folds=args.folds,
                batch_size=args.batch_size,
                epochs=args.epochs,
                save_dir=save_dir,
                logger=logger,
                learning_rate=args.lr
            )
            all_results[model_type] = results

            # 최종 결과 저장
            if logger:
                logger.finish_experiment(final_metrics=results)

        except Exception as e:
            print(f"\n[오류] {model_type} CV 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 결과 비교 및 저장
    if len(all_results) >= 1:
        if len(all_results) >= 2:
            compare_cv_results(all_results)
        save_cv_results(all_results, save_dir)

    print("\n" + "=" * 60)
    print("Cross-Validation 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
