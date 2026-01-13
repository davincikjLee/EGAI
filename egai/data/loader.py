"""
Audio Data Loader - 학습용 데이터 로더

기능:
    - CSV 메타데이터 로드
    - MP3 → 스펙트로그램 변환
    - 캐싱 지원
    - Train/Val/Test 분할
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from egai.preprocessing.audio import AudioPreprocessor


class AudioDataLoader:
    """
    학습용 오디오 데이터 로더
    """

    # 타겟 컬럼 정의
    TARGET_COLUMNS = [
        "low_high_freq",
        "mid_freq_score",
        "audible_range_score",
        "regularity",
        "irregularity",
    ]

    def __init__(
        self,
        data_dir: str = "data",
        target_shape: Tuple[int, int] = (128, 128),
        cache_dir: Optional[str] = None,
        fmax: int = 6000,
    ):
        """
        Args:
            data_dir: 데이터 디렉토리
            target_shape: 스펙트로그램 크기 (H, W)
            cache_dir: 캐시 디렉토리 (None이면 캐싱 안함)
            fmax: 최대 주파수
        """
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fmax = fmax

        # 전처리기
        self.preprocessor = AudioPreprocessor(
            fmax=fmax,
            target_shape=target_shape,
        )

        # 메타데이터
        self.metadata: Optional[pd.DataFrame] = None
        self.valid_indices: List[int] = []

    def load_metadata(
        self,
        csv_path: Optional[str] = None,
        fuel_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        메타데이터 로드

        Args:
            csv_path: CSV 경로 (None이면 기본)
            fuel_type: 연료 타입 필터

        Returns:
            메타데이터 DataFrame
        """
        if csv_path is None:
            csv_path = self.data_dir / "car_audio_metadata.csv"

        print(f"[Loader] 메타데이터 로드: {csv_path}")
        self.metadata = pd.read_csv(csv_path)

        # 유효 데이터 필터링
        valid_mask = self.metadata["overall_score"] > 0

        if fuel_type:
            valid_mask = valid_mask & (self.metadata["fuel_type"] == fuel_type)
            print(f"[Loader] 연료 타입 필터: {fuel_type}")

        # 오디오 파일 존재 확인
        valid_files = []
        for idx, row in self.metadata.iterrows():
            if not valid_mask.iloc[idx]:
                valid_files.append(False)
                continue
            audio_path = self.data_dir / str(row["audio_file_path"])
            valid_files.append(audio_path.exists())

        self.metadata["valid_file"] = valid_files
        valid_mask = valid_mask & self.metadata["valid_file"]

        self.valid_indices = self.metadata[valid_mask].index.tolist()

        print(f"[Loader] 전체: {len(self.metadata)}, 유효: {len(self.valid_indices)}")

        return self.metadata

    def get_audio_path(self, idx: int) -> Path:
        """인덱스로 오디오 경로 반환"""
        row = self.metadata.iloc[idx]
        return self.data_dir / str(row["audio_file_path"])

    def get_targets(self, idx: int) -> np.ndarray:
        """인덱스로 타겟 반환"""
        row = self.metadata.iloc[idx]
        return np.array(
            [row[col] for col in self.TARGET_COLUMNS],
            dtype=np.float32,
        )

    def load_spectrogram(
        self,
        idx: int,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        스펙트로그램 로드

        Args:
            idx: 데이터 인덱스
            use_cache: 캐시 사용 여부

        Returns:
            (full_spec, percussive_spec)
        """
        # 캐시 확인
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{idx}_spec.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                return data["full"], data["percussive"]

        # 스펙트로그램 생성
        audio_path = self.get_audio_path(idx)
        full_spec, perc_spec = self.preprocessor.process(str(audio_path))

        # 채널 차원 제거 (저장용)
        full_2d = full_spec[..., 0]
        perc_2d = perc_spec[..., 0]

        # 캐시 저장
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, full=full_2d, percussive=perc_2d)

        return full_2d, perc_2d

    def prepare_dataset(
        self,
        indices: List[int],
        model_type: str = "simple_cbam",
        use_cache: bool = True,
        verbose: bool = True,
    ) -> Tuple[Any, np.ndarray]:
        """
        데이터셋 준비

        Args:
            indices: 데이터 인덱스 리스트
            model_type: 모델 타입
            use_cache: 캐시 사용
            verbose: 진행 출력

        Returns:
            (X, y)
        """
        X_full = []
        X_perc = []
        y = []

        for i, idx in enumerate(indices):
            if verbose and (i + 1) % 50 == 0:
                print(f"  처리 중: {i + 1}/{len(indices)}")

            try:
                full_spec, perc_spec = self.load_spectrogram(idx, use_cache)
                targets = self.get_targets(idx)

                X_full.append(full_spec)
                X_perc.append(perc_spec)
                y.append(targets)
            except Exception as e:
                print(f"  [경고] 인덱스 {idx} 실패: {e}")
                continue

        X_full = np.array(X_full, dtype=np.float32)[..., np.newaxis]
        X_perc = np.array(X_perc, dtype=np.float32)[..., np.newaxis]
        y = np.array(y, dtype=np.float32)

        # 모델 타입에 따라 반환
        if model_type in ["simple", "simple_cbam"]:
            return X_full, y
        elif model_type in ["dual", "channel_concat"]:
            return [X_full, X_perc], y
        else:
            return X_full, y

    def get_splits(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        데이터 분할

        Returns:
            (train_indices, val_indices, test_indices)
        """
        if not self.valid_indices:
            raise ValueError("먼저 load_metadata()를 호출하세요")

        # 층화 추출용 레이블
        scores = self.metadata.iloc[self.valid_indices]["overall_score"].values
        bins = np.linspace(scores.min(), scores.max() + 0.01, 6)
        stratify_labels = np.digitize(scores, bins)

        indices = np.array(self.valid_indices)

        # Train+Val / Test 분할
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )

        # Train / Val 분할
        train_val_scores = self.metadata.iloc[train_val_idx]["overall_score"].values
        train_val_bins = np.digitize(train_val_scores, bins)

        val_ratio = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_bins,
        )

        print(f"[Loader] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def get_stats(self) -> Dict[str, Any]:
        """데이터 통계 반환"""
        if self.metadata is None:
            return {}

        valid_df = self.metadata.iloc[self.valid_indices]

        stats = {
            "total": len(self.metadata),
            "valid": len(self.valid_indices),
            "target_stats": {},
        }

        for col in self.TARGET_COLUMNS:
            stats["target_stats"][col] = {
                "mean": valid_df[col].mean(),
                "std": valid_df[col].std(),
                "min": valid_df[col].min(),
                "max": valid_df[col].max(),
            }

        return stats
