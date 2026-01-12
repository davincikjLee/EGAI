"""
데이터 로더 모듈
역할: CSV 메타데이터와 MP3 파일을 로드하여 학습용 데이터셋 생성
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from audio_preprocessing import AudioPreprocessor
import cv2


class AudioDataLoader:
    """
    오디오 데이터 로더
    - CSV 메타데이터 로드
    - MP3 → 스펙트로그램 변환
    - Train/Val/Test 분할
    """

    def __init__(self,
                 data_dir="data",
                 target_columns=None,
                 target_shape=(128, 128),
                 cache_dir=None,
                 use_mel=False,
                 fmax=6000,
                 n_mels=128):
        """
        초기화

        매개변수:
            data_dir: 데이터 디렉토리 경로
            target_columns: 예측할 컬럼 리스트
            target_shape: 스펙트로그램 크기 (height, width)
            cache_dir: 캐시 디렉토리 (None이면 캐싱 안함)
            use_mel: True면 Mel Spectrogram 사용
            fmax: 최대 주파수 (Hz)
            n_mels: Mel 필터뱅크 개수
        """
        self.data_dir = Path(data_dir)
        self.target_columns = target_columns or [
            'mid_freq_score', 'low_high_freq', 'audable_range_score', 'regularity', 'irregularity'
        ]
        self.target_shape = target_shape
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_mel = use_mel
        self.fmax = fmax

        # 전처리기 초기화
        self.preprocessor = AudioPreprocessor(
            fmax=fmax,
            use_mel=use_mel,
            n_mels=n_mels
        )

        # 메타데이터
        self.metadata = None
        self.valid_indices = None

        # 메타데이터 특징 (모델 입력용)
        self.metadata_features = None
        self.metadata_scaler = None
        self.metadata_encoders = {}

        # 사용할 메타데이터 컬럼 정의
        self.numeric_meta_cols = ['year', 'current_mileage_km', 'displacement_cc', 'seating_capacity']
        self.categorical_meta_cols = ['vehicle_type', 'fuel_type', 'drivetrain', 'transmission_type']

        # 물리량 특징 캐시
        self.physics_features = None
        self.physics_scaler = None
        self.physics_dim = len(self.preprocessor.get_physics_feature_names())

        # 모듈레이션 특징 캐시
        self.modulation_features = None
        self.modulation_scaler = None
        self.modulation_dim = len(self.preprocessor.get_modulation_feature_names())

    def load_metadata(self, csv_path=None, fuel_type=None):
        """
        CSV 메타데이터 로드

        매개변수:
            csv_path: CSV 파일 경로 (None이면 기본 경로)
            fuel_type: 연료 타입 필터 ('가솔린', '디젤', None=전체)

        반환값:
            DataFrame
        """
        if csv_path is None:
            csv_path = self.data_dir / "car_audio_metadata.csv"

        print(f"메타데이터 로드: {csv_path}")
        self.metadata = pd.read_csv(csv_path)

        # 유효한 데이터 필터링 (overall_score > 0)
        valid_mask = self.metadata['overall_score'] > 0

        # 연료 타입 필터링
        if fuel_type:
            valid_mask = valid_mask & (self.metadata['fuel_type'] == fuel_type)
            print(f"연료 타입 필터: {fuel_type}")

        # MP3 파일 존재 여부 확인
        valid_files = []
        for idx, row in self.metadata.iterrows():
            if not valid_mask[idx]:
                valid_files.append(False)
                continue

            audio_path = self.data_dir / row['audio_file_path']
            valid_files.append(audio_path.exists())

        self.metadata['valid_file'] = valid_files
        valid_mask = valid_mask & self.metadata['valid_file']

        self.valid_indices = self.metadata[valid_mask].index.tolist()

        print(f"전체 데이터: {len(self.metadata)}개")
        print(f"유효 데이터: {len(self.valid_indices)}개")
        print(f"타겟 컬럼: {self.target_columns}")

        # 메타데이터 특징 준비
        self._prepare_metadata_features()

        return self.metadata

    def _prepare_metadata_features(self):
        """
        모델 입력용 메타데이터 특징 준비
        - 수치형: StandardScaler 정규화
        - 범주형: LabelEncoder → 원-핫 인코딩
        """
        if self.metadata is None:
            return

        # 수치형 컬럼 처리
        numeric_data = self.metadata[self.numeric_meta_cols].copy()
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # 정규화
        self.metadata_scaler = StandardScaler()
        numeric_scaled = self.metadata_scaler.fit_transform(numeric_data)

        # 범주형 컬럼 처리
        categorical_encoded = []
        for col in self.categorical_meta_cols:
            le = LabelEncoder()
            # 결측치를 'unknown'으로 채우기
            col_data = self.metadata[col].fillna('unknown').astype(str)
            encoded = le.fit_transform(col_data)
            self.metadata_encoders[col] = le

            # 원-핫 인코딩
            n_classes = len(le.classes_)
            one_hot = np.zeros((len(encoded), n_classes))
            one_hot[np.arange(len(encoded)), encoded] = 1
            categorical_encoded.append(one_hot)

        # 결합
        if categorical_encoded:
            categorical_array = np.concatenate(categorical_encoded, axis=1)
            self.metadata_features = np.concatenate([numeric_scaled, categorical_array], axis=1)
        else:
            self.metadata_features = numeric_scaled

        self.metadata_dim = self.metadata_features.shape[1]
        print(f"메타데이터 특징 차원: {self.metadata_dim}")

    def get_metadata_features(self, idx):
        """인덱스로 메타데이터 특징 반환"""
        if self.metadata_features is None:
            raise ValueError("메타데이터 특징이 준비되지 않았습니다")
        return self.metadata_features[idx].astype(np.float32)

    def load_physics_features(self, idx, use_cache=True):
        """
        인덱스로 물리량 특징 로드

        매개변수:
            idx: 데이터 인덱스
            use_cache: 캐시 사용 여부

        반환값:
            physics_features: 물리량 특징 벡터 (14차원)
        """
        # 캐시 확인
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{idx}_physics.npy"
            if cache_path.exists():
                return np.load(cache_path).astype(np.float32)

        # 오디오 로드 및 물리량 특징 추출
        audio_path = self.get_audio_path(idx)
        audio = self.preprocessor.load_audio(str(audio_path))
        features_dict = self.preprocessor.extract_physics_features(audio)

        # Dict → Array 변환 (순서 보장)
        feature_names = self.preprocessor.get_physics_feature_names()
        features = np.array([features_dict[name] for name in feature_names], dtype=np.float32)

        # 캐시 저장
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)

        return features

    def prepare_physics_features(self, indices, use_cache=True, verbose=True):
        """
        전체 인덱스에 대해 물리량 특징 준비 및 정규화

        매개변수:
            indices: 데이터 인덱스 리스트
            use_cache: 캐시 사용 여부
            verbose: 진행 상황 출력

        반환값:
            physics_features: 정규화된 물리량 특징 배열 (N, 14)
        """
        features_list = []

        for i, idx in enumerate(indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"  물리량 특징 추출: {i + 1}/{len(indices)}")

            try:
                features = self.load_physics_features(idx, use_cache)
                features_list.append(features)
            except Exception as e:
                print(f"  [경고] 인덱스 {idx} 물리량 추출 실패: {e}")
                # 실패 시 0으로 채우기
                features_list.append(np.zeros(self.physics_dim, dtype=np.float32))

        physics_array = np.array(features_list, dtype=np.float32)

        # 정규화 (학습 데이터 기준)
        if self.physics_scaler is None:
            self.physics_scaler = StandardScaler()
            physics_normalized = self.physics_scaler.fit_transform(physics_array)
        else:
            physics_normalized = self.physics_scaler.transform(physics_array)

        return physics_normalized.astype(np.float32)

    def load_modulation_features(self, idx, use_cache=True):
        """
        인덱스로 모듈레이션 특징 로드

        매개변수:
            idx: 데이터 인덱스
            use_cache: 캐시 사용 여부

        반환값:
            modulation_features: 모듈레이션 특징 벡터 (20차원)
        """
        # 캐시 확인
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{idx}_modulation.npy"
            if cache_path.exists():
                return np.load(cache_path).astype(np.float32)

        # 오디오 로드 및 모듈레이션 특징 추출
        audio_path = self.get_audio_path(idx)
        audio = self.preprocessor.load_audio(str(audio_path))
        features_dict = self.preprocessor.extract_modulation_features(audio)

        # Dict → Array 변환 (순서 보장)
        feature_names = self.preprocessor.get_modulation_feature_names()
        features = np.array([features_dict[name] for name in feature_names], dtype=np.float32)

        # 캐시 저장
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)

        return features

    def prepare_modulation_features(self, indices, use_cache=True, verbose=True):
        """
        전체 인덱스에 대해 모듈레이션 특징 준비 및 정규화

        매개변수:
            indices: 데이터 인덱스 리스트
            use_cache: 캐시 사용 여부
            verbose: 진행 상황 출력

        반환값:
            modulation_features: 정규화된 모듈레이션 특징 배열 (N, 20)
        """
        features_list = []

        for i, idx in enumerate(indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"  모듈레이션 특징 추출: {i + 1}/{len(indices)}")

            try:
                features = self.load_modulation_features(idx, use_cache)
                features_list.append(features)
            except Exception as e:
                print(f"  [경고] 인덱스 {idx} 모듈레이션 추출 실패: {e}")
                # 실패 시 0으로 채우기
                features_list.append(np.zeros(self.modulation_dim, dtype=np.float32))

        modulation_array = np.array(features_list, dtype=np.float32)

        # 정규화 (학습 데이터 기준)
        if self.modulation_scaler is None:
            self.modulation_scaler = StandardScaler()
            modulation_normalized = self.modulation_scaler.fit_transform(modulation_array)
        else:
            modulation_normalized = self.modulation_scaler.transform(modulation_array)

        return modulation_normalized.astype(np.float32)

    def get_audio_path(self, idx):
        """인덱스로 오디오 파일 경로 반환"""
        row = self.metadata.iloc[idx]
        return self.data_dir / row['audio_file_path']

    def get_targets(self, idx):
        """인덱스로 타겟 값들 반환"""
        row = self.metadata.iloc[idx]
        return np.array([row[col] for col in self.target_columns], dtype=np.float32)

    def resize_spectrogram(self, spec):
        """스펙트로그램 크기 조정"""
        if spec.shape[:2] != self.target_shape:
            spec = cv2.resize(spec, self.target_shape[::-1])  # cv2는 (width, height) 순서
        return spec

    def load_spectrogram(self, idx, use_cache=True):
        """
        인덱스로 스펙트로그램 로드

        매개변수:
            idx: 데이터 인덱스
            use_cache: 캐시 사용 여부

        반환값:
            full_spec, percussive_spec: 스펙트로그램 (target_shape)
        """
        audio_path = self.get_audio_path(idx)

        # 캐시 확인
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{idx}_spec.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                return data['full'], data['percussive']

        # 스펙트로그램 생성
        full_spec, percussive_spec = self.preprocessor.preprocess(str(audio_path))

        # 크기 조정
        full_spec = self.resize_spectrogram(full_spec)
        percussive_spec = self.resize_spectrogram(percussive_spec)

        # 캐시 저장
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, full=full_spec, percussive=percussive_spec)

        return full_spec, percussive_spec

    def prepare_dataset(self, indices, model_type='simple', use_cache=True, verbose=True):
        """
        인덱스 리스트로 데이터셋 준비

        매개변수:
            indices: 데이터 인덱스 리스트
            model_type: 'simple', 'dual', 'resnet', 'multiinput_metadata', 'multiinput_physics', 'multiinput_modulation'
            use_cache: 캐시 사용 여부
            verbose: 진행 상황 출력

        반환값:
            X: 입력 데이터 (model_type에 따라 다름)
            y: 타겟 데이터
        """
        X_full = []
        X_percussive = []
        X_metadata = []
        y = []
        valid_indices = []  # 성공적으로 로드된 인덱스 추적

        for i, idx in enumerate(indices):
            if verbose and (i + 1) % 50 == 0:
                print(f"  처리 중: {i + 1}/{len(indices)}")

            try:
                full_spec, percussive_spec = self.load_spectrogram(idx, use_cache)
                targets = self.get_targets(idx)

                X_full.append(full_spec)
                X_percussive.append(percussive_spec)
                y.append(targets)
                valid_indices.append(idx)

                # 메타데이터 (multiinput_metadata 모델용)
                if model_type == 'multiinput_metadata':
                    meta_features = self.get_metadata_features(idx)
                    X_metadata.append(meta_features)

            except Exception as e:
                print(f"  [경고] 인덱스 {idx} 처리 실패: {e}")
                continue

        X_full = np.array(X_full, dtype=np.float32)
        X_percussive = np.array(X_percussive, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # 채널 차원 추가
        X_full = X_full[..., np.newaxis]  # (N, H, W, 1)
        X_percussive = X_percussive[..., np.newaxis]

        if model_type == 'simple':
            return X_full, y
        elif model_type == 'simple_cbam':
            # Simple CBAM: Full 스펙트로그램만 사용 (simple과 동일)
            return X_full, y
        elif model_type == 'dual':
            return [X_full, X_percussive], y
        elif model_type == 'channel_concat':
            # 채널 결합: Full + Percussive (논문 Fig.6 방식)
            return [X_full, X_percussive], y
        elif model_type == 'multiinput_metadata':
            X_metadata = np.array(X_metadata, dtype=np.float32)
            return [X_full, X_percussive, X_metadata], y
        elif model_type == 'multiinput_physics':
            # 물리량 특징 준비
            if verbose:
                print("  물리량 특징 추출 중...")
            X_physics = self.prepare_physics_features(valid_indices, use_cache, verbose)
            return [X_full, X_percussive, X_physics], y
        elif model_type == 'multiinput_modulation':
            # 모듈레이션 특징 준비 (regularity/irregularity 개선용)
            if verbose:
                print("  모듈레이션 특징 추출 중...")
            X_modulation = self.prepare_modulation_features(valid_indices, use_cache, verbose)
            return [X_full, X_percussive, X_modulation], y
        elif model_type == 'resnet':
            # ResNet용: 3채널로 변환 (224x224)
            X_rgb = np.concatenate([X_full, X_percussive, X_full], axis=-1)
            # 크기 조정
            X_resized = np.array([
                cv2.resize(img, (224, 224)) for img in X_rgb
            ])
            return X_resized, y
        else:
            raise ValueError(f"지원하지 않는 model_type: {model_type}")

    def get_splits(self, test_size=0.1, val_size=0.1, random_state=42, stratify_bins=5):
        """
        Train/Val/Test 분할

        매개변수:
            test_size: 테스트 비율
            val_size: 검증 비율
            random_state: 랜덤 시드
            stratify_bins: 층화 추출용 구간 수

        반환값:
            train_indices, val_indices, test_indices
        """
        if self.valid_indices is None:
            raise ValueError("먼저 load_metadata()를 호출하세요")

        # overall_score 기반 층화 추출
        overall_scores = self.metadata.iloc[self.valid_indices]['overall_score'].values

        # 점수를 구간으로 나누기 (층화 추출용)
        bins = np.linspace(overall_scores.min(), overall_scores.max() + 0.01, stratify_bins + 1)
        stratify_labels = np.digitize(overall_scores, bins)

        indices = np.array(self.valid_indices)

        # Train+Val / Test 분할
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )

        # Train / Val 분할
        train_val_scores = self.metadata.iloc[train_val_idx]['overall_score'].values
        train_val_stratify = np.digitize(train_val_scores, bins)

        val_ratio = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_stratify
        )

        print(f"\n데이터 분할:")
        print(f"  Train: {len(train_idx)}개 ({len(train_idx)/len(indices)*100:.1f}%)")
        print(f"  Val:   {len(val_idx)}개 ({len(val_idx)/len(indices)*100:.1f}%)")
        print(f"  Test:  {len(test_idx)}개 ({len(test_idx)/len(indices)*100:.1f}%)")

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def get_target_stats(self):
        """타겟 컬럼들의 통계 정보 반환"""
        if self.metadata is None:
            raise ValueError("먼저 load_metadata()를 호출하세요")

        valid_data = self.metadata.iloc[self.valid_indices]

        stats = {}
        for col in self.target_columns:
            stats[col] = {
                'min': valid_data[col].min(),
                'max': valid_data[col].max(),
                'mean': valid_data[col].mean(),
                'std': valid_data[col].std()
            }

        return stats


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("AudioDataLoader 테스트")
    print("=" * 60)

    # 프로젝트 루트 기준 경로
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # 데이터 로더 초기화
    loader = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128)
    )

    # 메타데이터 로드
    loader.load_metadata()

    # 통계 출력
    print("\n타겟 통계:")
    stats = loader.get_target_stats()
    for col, stat in stats.items():
        print(f"  {col}: min={stat['min']:.2f}, max={stat['max']:.2f}, "
              f"mean={stat['mean']:.2f}, std={stat['std']:.2f}")

    # 데이터 분할
    train_idx, val_idx, test_idx = loader.get_splits()

    # 샘플 로드 테스트
    print("\n샘플 로드 테스트:")
    if len(train_idx) > 0:
        idx = train_idx[0]
        full_spec, perc_spec = loader.load_spectrogram(idx, use_cache=False)
        targets = loader.get_targets(idx)

        print(f"  Full Spectrogram: {full_spec.shape}")
        print(f"  Percussive Spectrogram: {perc_spec.shape}")
        print(f"  Targets: {targets}")

    print("\n테스트 완료!")
