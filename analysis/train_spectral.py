"""
Spectral Features를 사용한 학습 스크립트
역할: 엔진 진단에 적합한 스펙트럼 특징 추출 및 학습
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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


class SpectralFeatureExtractor:
    """엔진 진단을 위한 스펙트럼 특징 추출기"""

    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_features(self, audio):
        """
        오디오에서 스펙트럼 특징 추출

        반환 특징:
        1. Spectral Centroid (주파수 무게중심)
        2. Spectral Rolloff (에너지 분포)
        3. Spectral Flatness (노이즈 vs 톤 비율)
        4. Spectral Bandwidth (주파수 대역폭)
        5. Zero Crossing Rate (신호 변화율)
        6. RMS Energy (에너지)
        7. Spectral Contrast (대역별 에너지 차이)
        """
        features = {}

        # 1. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['centroid_mean'] = np.mean(centroid)
        features['centroid_std'] = np.std(centroid)

        # 2. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)

        # 3. Spectral Flatness (노이즈 비율 - 엔진 이상 감지에 중요)
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)

        # 4. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['bandwidth_mean'] = np.mean(bandwidth)
        features['bandwidth_std'] = np.std(bandwidth)

        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # 6. RMS Energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        # 7. Spectral Contrast (7개 대역)
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        for i in range(contrast.shape[0]):
            features[f'contrast_{i}_mean'] = np.mean(contrast[i])
            features[f'contrast_{i}_std'] = np.std(contrast[i])

        # 8. MFCC (13개 계수) - 추가
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])

        return features

    def extract_batch(self, audio_list, verbose=True):
        """배치로 특징 추출"""
        all_features = []

        for i, audio in enumerate(audio_list):
            if verbose and (i + 1) % 50 == 0:
                print(f"  특징 추출 중: {i+1}/{len(audio_list)}")

            try:
                features = self.extract_features(audio)
                all_features.append(features)
            except Exception as e:
                print(f"  [경고] 인덱스 {i} 특징 추출 실패: {e}")
                # 빈 특징으로 대체
                all_features.append({k: 0.0 for k in self.get_feature_names()})

        # DataFrame 형태로 변환
        feature_names = list(all_features[0].keys())
        X = np.array([[f[name] for name in feature_names] for f in all_features])

        return X, feature_names

    def get_feature_names(self):
        """특징 이름 목록 반환"""
        names = [
            'centroid_mean', 'centroid_std',
            'rolloff_mean', 'rolloff_std',
            'flatness_mean', 'flatness_std',
            'bandwidth_mean', 'bandwidth_std',
            'zcr_mean', 'zcr_std',
            'rms_mean', 'rms_std'
        ]
        # Spectral Contrast (7 bands)
        for i in range(7):
            names.extend([f'contrast_{i}_mean', f'contrast_{i}_std'])
        # MFCC (13 coefficients)
        for i in range(13):
            names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])

        return names


def build_spectral_model(num_features, num_outputs=4):
    """Spectral Features를 위한 MLP 모델"""
    inputs = layers.Input(shape=(num_features,), name='input')

    # Dense layers
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)

    # Output
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='SpectralMLP')
    return model


def create_callbacks(model_name, save_dir):
    """학습 콜백 생성"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
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
            patience=7,
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
    parser = argparse.ArgumentParser(description='EGAI Spectral Features 학습')
    parser.add_argument('--epochs', type=int, default=200,
                        help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='데이터 디렉토리')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='모델 저장 디렉토리')

    args = parser.parse_args()

    # 경로 설정
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "analysis" / "models" / "spectral"

    print("=" * 60)
    print("EGAI Spectral Features 학습")
    print("=" * 60)
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"데이터 경로: {data_dir}")
    print(f"저장 경로: {save_dir}")

    # 데이터 로더 초기화
    loader = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128)
    )

    # 메타데이터 로드
    loader.load_metadata()

    # 데이터 분할
    train_idx, val_idx, test_idx = loader.get_splits()

    # 특징 추출기 초기화
    extractor = SpectralFeatureExtractor()

    # 오디오 로드 및 특징 추출
    print(f"\n{'#'*60}")
    print(f"# 오디오 로드 및 특징 추출")
    print(f"{'#'*60}")

    def load_audio_batch(indices, verbose=True):
        """인덱스 리스트에서 오디오 로드"""
        audio_list = []
        targets_list = []
        valid_indices = []

        for i, idx in enumerate(indices):
            try:
                audio_path = loader.get_audio_path(idx)
                audio = loader.preprocessor.load_audio(str(audio_path))
                targets = loader.get_targets(idx)

                audio_list.append(audio)
                targets_list.append(targets)
                valid_indices.append(idx)

                if verbose and (i + 1) % 50 == 0:
                    print(f"  로드 중: {i+1}/{len(indices)}")
            except Exception as e:
                if verbose:
                    print(f"  [경고] 인덱스 {idx} 로드 실패: {e}")
                continue

        return audio_list, np.array(targets_list), valid_indices

    print("\nTrain 오디오 로드...")
    train_audio, y_train, train_valid = load_audio_batch(train_idx)
    print("\nVal 오디오 로드...")
    val_audio, y_val, val_valid = load_audio_batch(val_idx)
    print("\nTest 오디오 로드...")
    test_audio, y_test, test_valid = load_audio_batch(test_idx)

    print(f"\n유효 샘플: Train={len(train_audio)}, Val={len(val_audio)}, Test={len(test_audio)}")

    # 특징 추출
    print("\nTrain 특징 추출...")
    X_train, feature_names = extractor.extract_batch(train_audio)
    print("\nVal 특징 추출...")
    X_val, _ = extractor.extract_batch(val_audio)
    print("\nTest 특징 추출...")
    X_test, _ = extractor.extract_batch(test_audio)

    print(f"\n특징 수: {len(feature_names)}")
    print(f"특징 이름: {feature_names[:10]}...")  # 처음 10개만

    # 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\n입력 형태: {X_train.shape}")
    print(f"타겟 형태: {y_train.shape}")

    # 모델 생성
    print(f"\n{'#'*60}")
    print(f"# Spectral MLP 모델 학습")
    print(f"{'#'*60}")

    print("\n모델 생성...")
    model = build_spectral_model(num_features=X_train.shape[1], num_outputs=4)
    model.summary()
    print(f"총 파라미터: {model.count_params():,}")

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # 콜백
    callbacks = create_callbacks('spectral', save_dir)

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
    print_results(results, "Spectral MLP")

    # 모델 저장
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "spectral_final.keras"))
    print(f"\n모델 저장: {save_dir}/spectral_final.keras")

    # Scaler 저장
    import pickle
    with open(save_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler 저장: {save_dir}/scaler.pkl")

    # 결과 저장
    results_path = save_dir / "spectral_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        serializable_results = {
            'num_features': len(feature_names),
            'feature_names': feature_names,
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
    print("Spectral Features 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
