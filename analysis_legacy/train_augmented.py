"""
데이터 증강을 적용한 학습 스크립트
역할: AudioAugmentor를 사용하여 학습 데이터를 증강하고 모델 학습
"""

import os
import sys
import json
import argparse
from pathlib import Path

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
from audio_preprocessing import AudioPreprocessor
from AudioAugment import AudioAugmentor
from models import get_model
import cv2


class AugmentedDataLoader(AudioDataLoader):
    """데이터 증강을 지원하는 데이터 로더"""

    def __init__(self, *args, num_augmentations=2, **kwargs):
        """
        초기화

        매개변수:
            num_augmentations: 각 샘플당 생성할 증강 개수
        """
        super().__init__(*args, **kwargs)
        self.num_augmentations = num_augmentations
        self.augmentor = AudioAugmentor(sample_rate=22050)

    def prepare_augmented_dataset(self, indices, model_type='simple', verbose=True):
        """
        증강된 데이터셋 준비

        매개변수:
            indices: 데이터 인덱스 리스트
            model_type: 'simple', 'dual', 'resnet'
            verbose: 진행 상황 출력

        반환값:
            X: 입력 데이터 (증강 포함)
            y: 타겟 데이터 (증강된 샘플에 대해 동일한 타겟)
        """
        X_full = []
        X_percussive = []
        y = []

        total = len(indices) * (1 + self.num_augmentations)
        processed = 0

        for i, idx in enumerate(indices):
            try:
                # 원본 오디오 로드
                audio_path = self.get_audio_path(idx)
                audio = self.preprocessor.load_audio(str(audio_path))
                targets = self.get_targets(idx)

                # 원본 스펙트로그램 처리
                full_spec, percussive_spec = self.preprocessor.preprocess_from_array(audio)
                full_spec = self.resize_spectrogram(full_spec)
                percussive_spec = self.resize_spectrogram(percussive_spec)

                X_full.append(full_spec)
                X_percussive.append(percussive_spec)
                y.append(targets)
                processed += 1

                # 증강 버전 생성
                for aug_idx in range(self.num_augmentations):
                    aug_audio = self.augmentor.apply_random_augmentations(
                        audio,
                        apply_time_shift=True,
                        apply_pitch_shift=False,  # 피치 변환은 엔진음에 영향
                        apply_time_stretch=False,  # 시간 신축도 영향
                        apply_noise=True,
                        apply_volume=True,
                        probability=0.7
                    )

                    aug_full, aug_perc = self.preprocessor.preprocess_from_array(aug_audio)
                    aug_full = self.resize_spectrogram(aug_full)
                    aug_perc = self.resize_spectrogram(aug_perc)

                    X_full.append(aug_full)
                    X_percussive.append(aug_perc)
                    y.append(targets)  # 동일한 타겟
                    processed += 1

                if verbose and processed % 100 == 0:
                    print(f"  처리 중: {processed}/{total}")

            except Exception as e:
                if verbose:
                    print(f"  [경고] 인덱스 {idx} 처리 실패: {e}")
                continue

        X_full = np.array(X_full, dtype=np.float32)
        X_percussive = np.array(X_percussive, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # 채널 차원 추가
        X_full = X_full[..., np.newaxis]
        X_percussive = X_percussive[..., np.newaxis]

        if model_type == 'simple':
            return X_full, y
        elif model_type == 'dual':
            return [X_full, X_percussive], y
        elif model_type == 'resnet':
            X_rgb = np.concatenate([X_full, X_percussive, X_full], axis=-1)
            X_resized = np.array([cv2.resize(img, (224, 224)) for img in X_rgb])
            return X_resized, y
        else:
            raise ValueError(f"지원하지 않는 model_type: {model_type}")


def create_callbacks(model_name, save_dir):
    """학습 콜백 생성"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / f"{model_name}_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    return callbacks


def evaluate_model(model, X_test, y_test, target_columns):
    """모델 평가"""
    y_pred = model.predict(X_test, verbose=0)

    overall_mae = mean_absolute_error(y_test, y_pred)
    overall_r2 = r2_score(y_test, y_pred)

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


def main():
    parser = argparse.ArgumentParser(description='EGAI 증강 학습')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'dual', 'resnet'],
                        help='학습할 모델 타입')
    parser.add_argument('--num_aug', type=int, default=2,
                        help='샘플당 증강 개수')
    parser.add_argument('--epochs', type=int, default=100,
                        help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='데이터 디렉토리')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='모델 저장 디렉토리')

    args = parser.parse_args()

    # 경로 설정
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "analysis" / "models" / f"augmented_x{args.num_aug}"

    print("=" * 60)
    print("EGAI 증강 학습")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"증강 배수: {args.num_aug}x (원본 + 증강{args.num_aug}개)")
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"데이터 경로: {data_dir}")
    print(f"저장 경로: {save_dir}")

    # 데이터 로더 초기화
    loader = AugmentedDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        num_augmentations=args.num_aug
    )

    # 메타데이터 로드
    loader.load_metadata()

    # 데이터 분할
    train_idx, val_idx, test_idx = loader.get_splits()

    # 데이터 준비
    print(f"\n{'#'*60}")
    print(f"# 데이터 준비 (증강 적용)")
    print(f"{'#'*60}")

    print(f"\nTrain 데이터 준비 (원본 {len(train_idx)} + 증강 {len(train_idx)*args.num_aug})...")
    X_train, y_train = loader.prepare_augmented_dataset(train_idx, model_type=args.model)

    print(f"\nVal 데이터 준비 (증강 없음)...")
    X_val, y_val = loader.prepare_dataset(val_idx, model_type=args.model)

    print(f"\nTest 데이터 준비 (증강 없음)...")
    X_test, y_test = loader.prepare_dataset(test_idx, model_type=args.model)

    if isinstance(X_train, list):
        print(f"\n입력 형태: [{X_train[0].shape}, {X_train[1].shape}]")
    else:
        print(f"\n입력 형태: {X_train.shape}")
    print(f"타겟 형태: {y_train.shape}")
    print(f"증강 후 Train 샘플: {len(y_train)}개")

    # 모델 생성
    print(f"\n{'#'*60}")
    print(f"# {args.model.upper()} 모델 학습")
    print(f"{'#'*60}")

    print("\n모델 생성...")
    model = get_model(args.model, num_outputs=len(loader.target_columns))
    print(f"총 파라미터: {model.count_params():,}")

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # 콜백
    callbacks = create_callbacks(args.model, save_dir)

    # 학습
    print(f"\n학습 시작 (epochs={args.epochs}, batch_size={args.batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 평가
    results = evaluate_model(model, X_test, y_test, loader.target_columns)
    print_results(results, f"{args.model.upper()} (증강 {args.num_aug}x)")

    # 모델 저장
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / f"{args.model}_final.keras"))
    print(f"\n모델 저장: {save_dir}/{args.model}_final.keras")

    # 결과 저장
    results_path = save_dir / "augmented_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        serializable_results = {
            'model': args.model,
            'num_augmentations': args.num_aug,
            'overall_MAE': float(results['overall_MAE']),
            'overall_R2': float(results['overall_R2']),
            'column_metrics': {
                col: {'MAE': float(m['MAE']), 'R2': float(m['R2'])}
                for col, m in results['column_metrics'].items()
            }
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {results_path}")

    print("\n" + "=" * 60)
    print("증강 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
