"""
메인 학습 스크립트
역할: 3가지 모델 학습 및 비교 실험
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
from models import get_model, unfreeze_resnet


def create_callbacks(model_name, save_dir):
    """
    학습 콜백 생성

    매개변수:
        model_name: 모델 이름
        save_dir: 저장 디렉토리

    반환값:
        콜백 리스트
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Early Stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        # Model Checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / f"{model_name}_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Learning Rate Reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),

        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=str(save_dir / "logs" / model_name),
            histogram_freq=1
        )
    ]

    return callbacks


def evaluate_model(model, X_test, y_test, target_columns):
    """
    모델 평가

    매개변수:
        model: 학습된 모델
        X_test: 테스트 입력
        y_test: 테스트 타겟
        target_columns: 타겟 컬럼 이름들

    반환값:
        평가 결과 딕셔너리
    """
    # 예측
    y_pred = model.predict(X_test, verbose=0)

    # 전체 평가
    overall_mae = mean_absolute_error(y_test, y_pred)
    overall_r2 = r2_score(y_test, y_pred)

    # 컬럼별 평가
    column_metrics = {}
    for i, col in enumerate(target_columns):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        column_metrics[col] = {'MAE': mae, 'R2': r2}

    results = {
        'overall_MAE': overall_mae,
        'overall_R2': overall_r2,
        'column_metrics': column_metrics
    }

    return results


def print_results(results, model_name):
    """평가 결과 출력"""
    print(f"\n{'='*60}")
    print(f"{model_name} 평가 결과")
    print(f"{'='*60}")
    print(f"전체 MAE: {results['overall_MAE']:.4f}")
    print(f"전체 R²:  {results['overall_R2']:.4f}")
    print(f"\n컬럼별 결과:")
    for col, metrics in results['column_metrics'].items():
        print(f"  {col:20s} - MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")


def train_model(model_type, data_loader, train_idx, val_idx, test_idx,
                batch_size=16, epochs=100, save_dir="models"):
    """
    단일 모델 학습

    매개변수:
        model_type: 'simple', 'dual', 'resnet'
        data_loader: AudioDataLoader 인스턴스
        train_idx, val_idx, test_idx: 데이터 인덱스
        batch_size: 배치 크기
        epochs: 에포크 수
        save_dir: 저장 디렉토리

    반환값:
        history, results
    """
    print(f"\n{'#'*60}")
    print(f"# {model_type.upper()} 모델 학습 시작")
    print(f"{'#'*60}")

    # 데이터 준비
    print("\n데이터 준비 중...")
    print("  Train 데이터 로드...")
    X_train, y_train = data_loader.prepare_dataset(train_idx, model_type=model_type)
    print("  Val 데이터 로드...")
    X_val, y_val = data_loader.prepare_dataset(val_idx, model_type=model_type)
    print("  Test 데이터 로드...")
    X_test, y_test = data_loader.prepare_dataset(test_idx, model_type=model_type)

    if isinstance(X_train, list):
        print(f"\n입력 형태: [{X_train[0].shape}, {X_train[1].shape}]")
    else:
        print(f"\n입력 형태: {X_train.shape}")
    print(f"타겟 형태: {y_train.shape}")

    # 모델 생성
    print("\n모델 생성...")
    model = get_model(model_type, num_outputs=len(data_loader.target_columns))
    print(f"총 파라미터: {model.count_params():,}")

    # 컴파일
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # 콜백
    callbacks = create_callbacks(model_type, save_dir)

    # 학습
    print(f"\n학습 시작 (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tuning (ResNet만)
    if model_type == 'resnet':
        print("\nResNet Fine-tuning 시작...")
        unfreeze_resnet(model, unfreeze_layers=30)

        # 낮은 학습률로 재컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='mse',
            metrics=['mae']
        )

        # 추가 학습
        history_ft = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            callbacks=create_callbacks(f"{model_type}_finetune", save_dir),
            verbose=1
        )

        # History 병합
        for key in history.history:
            history.history[key].extend(history_ft.history[key])

    # 평가
    results = evaluate_model(model, X_test, y_test, data_loader.target_columns)
    print_results(results, model_type.upper())

    # 최종 모델 저장
    model.save(str(Path(save_dir) / f"{model_type}_final.keras"))
    print(f"\n모델 저장: {save_dir}/{model_type}_final.keras")

    return history, results


def compare_models(all_results, save_dir):
    """
    모델 비교 결과 출력 및 저장

    매개변수:
        all_results: {model_type: results} 딕셔너리
        save_dir: 저장 디렉토리
    """
    print(f"\n{'='*70}")
    print("모델 비교 결과")
    print(f"{'='*70}")

    # 테이블 헤더
    print(f"\n{'모델':15s} | {'전체 MAE':>10s} | {'전체 R²':>10s}")
    print("-" * 45)

    for model_type, results in all_results.items():
        print(f"{model_type:15s} | {results['overall_MAE']:>10.4f} | {results['overall_R2']:>10.4f}")

    # 최고 모델
    best_model = min(all_results.items(), key=lambda x: x[1]['overall_MAE'])
    print(f"\n최고 성능 모델: {best_model[0]} (MAE: {best_model[1]['overall_MAE']:.4f})")

    # 컬럼별 상세 비교
    print(f"\n{'='*70}")
    print("컬럼별 MAE 비교")
    print(f"{'='*70}")

    # 첫 번째 모델에서 컬럼 목록 가져오기
    columns = list(next(iter(all_results.values()))['column_metrics'].keys())

    # 헤더
    header = f"{'컬럼':20s}"
    for model_type in all_results.keys():
        header += f" | {model_type:>12s}"
    print(header)
    print("-" * (25 + 15 * len(all_results)))

    # 각 컬럼별 결과
    for col in columns:
        row = f"{col:20s}"
        for model_type, results in all_results.items():
            mae = results['column_metrics'][col]['MAE']
            row += f" | {mae:>12.4f}"
        print(row)

    # JSON으로 저장
    save_path = Path(save_dir) / "comparison_results.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        # numpy 타입을 일반 Python 타입으로 변환
        serializable_results = {}
        for model_type, results in all_results.items():
            serializable_results[model_type] = {
                'overall_MAE': float(results['overall_MAE']),
                'overall_R2': float(results['overall_R2']),
                'column_metrics': {
                    col: {'MAE': float(m['MAE']), 'R2': float(m['R2'])}
                    for col, m in results['column_metrics'].items()
                }
            }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='EGAI 모델 학습')
    parser.add_argument('--model', type=str, default='all',
                        choices=['simple', 'dual', 'resnet', 'all'],
                        help='학습할 모델 타입')
    parser.add_argument('--epochs', type=int, default=100,
                        help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='데이터 디렉토리')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='모델 저장 디렉토리')
    parser.add_argument('--cache', action='store_true',
                        help='스펙트로그램 캐싱 사용')
    parser.add_argument('--use_mel', action='store_true',
                        help='Mel Spectrogram 사용 (기본: STFT)')
    parser.add_argument('--fmax', type=int, default=6000,
                        help='최대 주파수 Hz (기본: 6000)')

    args = parser.parse_args()

    # 경로 설정
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "analysis" / "models"
    cache_dir = save_dir / "cache" if args.cache else None

    # Mel 사용시 저장 경로 구분
    if args.use_mel:
        save_dir = save_dir / f"mel_fmax{args.fmax}"

    print("=" * 60)
    print("EGAI 모델 학습")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"스펙트로그램: {'Mel' if args.use_mel else 'STFT'}")
    print(f"최대 주파수: {args.fmax}Hz")
    print(f"데이터 경로: {data_dir}")
    print(f"저장 경로: {save_dir}")
    print(f"캐싱: {'활성화' if cache_dir else '비활성화'}")

    # 데이터 로더 초기화
    loader = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        cache_dir=cache_dir,
        use_mel=args.use_mel,
        fmax=args.fmax
    )

    # 메타데이터 로드
    loader.load_metadata()

    # 데이터 분할
    train_idx, val_idx, test_idx = loader.get_splits()

    # 학습할 모델 목록
    if args.model == 'all':
        model_types = ['simple', 'dual', 'resnet']
    else:
        model_types = [args.model]

    # 결과 저장
    all_results = {}
    all_histories = {}

    # 각 모델 학습
    for model_type in model_types:
        try:
            history, results = train_model(
                model_type=model_type,
                data_loader=loader,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                batch_size=args.batch_size,
                epochs=args.epochs,
                save_dir=save_dir
            )
            all_results[model_type] = results
            all_histories[model_type] = history.history

        except Exception as e:
            print(f"\n[오류] {model_type} 모델 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 모델 비교 (2개 이상 학습된 경우)
    if len(all_results) >= 2:
        compare_models(all_results, save_dir)

    # 학습 히스토리 저장
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        # numpy 배열을 리스트로 변환
        serializable_histories = {}
        for model_type, history in all_histories.items():
            serializable_histories[model_type] = {
                k: [float(v) for v in vals] for k, vals in history.items()
            }
        json.dump(serializable_histories, f, indent=2)
    print(f"\n학습 히스토리 저장: {history_path}")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
