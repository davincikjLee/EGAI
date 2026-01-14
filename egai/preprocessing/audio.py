"""
Audio Preprocessing - 오디오 전처리

librosa 기반 오디오 신호 처리
- STFT 스펙트로그램 생성
- HPSS (Harmonic-Percussive Source Separation)
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import librosa


class AudioPreprocessor:
    """
    오디오 전처리기

    엔진 오디오를 스펙트로그램으로 변환
    최적 설정 (리버스 엔지니어링 결과):
    - STFT (Mel 아님)
    - fmax=6000Hz
    - HPSS 분리 사용
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmax: int = 6000,
        target_shape: Tuple[int, int] = (128, 128),
    ):
        """
        초기화

        Args:
            sample_rate: 샘플링 레이트 (Hz)
            n_fft: FFT 윈도우 크기
            hop_length: STFT 홉 길이
            fmax: 최대 주파수 (Hz)
            target_shape: 출력 스펙트로그램 크기 (H, W)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.target_shape = target_shape

        # 주파수 bin 계산
        self.freq_resolution = sample_rate / n_fft
        self.max_freq_bin = int(fmax / self.freq_resolution)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        오디오 파일 로드

        Args:
            audio_path: 오디오 파일 경로 (WAV, MP3 지원)

        Returns:
            오디오 배열
        """
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        return audio

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """오디오 정규화 (-1 ~ 1)"""
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        return audio / max_val

    def compute_spectrogram(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        스펙트로그램 계산 (Full + Percussive)

        Args:
            audio: 오디오 배열

        Returns:
            full_spec: 전체 스펙트로그램
            percussive_spec: 타악음 스펙트로그램
        """
        # 정규화
        audio = self.normalize(audio)

        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

        # HPSS 분리
        _, stft_percussive = librosa.decompose.hpss(stft, kernel_size=31, margin=2.0)

        # 주파수 범위 제한 (0 ~ fmax)
        stft_limited = stft[:self.max_freq_bin + 1, :]
        percussive_limited = stft_percussive[:self.max_freq_bin + 1, :]

        # dB 변환 및 정규화
        full_spec = self._to_db_normalized(np.abs(stft_limited))
        percussive_spec = self._to_db_normalized(np.abs(percussive_limited))

        # 크기 조정
        full_spec = self._resize(full_spec)
        percussive_spec = self._resize(percussive_spec)

        return full_spec, percussive_spec

    def _to_db_normalized(self, magnitude: np.ndarray) -> np.ndarray:
        """dB 변환 후 0~1 정규화"""
        db = librosa.amplitude_to_db(magnitude, ref=np.max)
        min_val, max_val = db.min(), db.max()
        return (db - min_val) / (max_val - min_val + 1e-8)

    def _resize(self, spec: np.ndarray) -> np.ndarray:
        """스펙트로그램 크기 조정"""
        if spec.shape == self.target_shape:
            return spec

        import cv2
        return cv2.resize(spec, self.target_shape[::-1])  # cv2는 (W, H) 순서

    def process(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        전체 전처리 파이프라인

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            full_spec: 전체 스펙트로그램 (H, W, 1)
            percussive_spec: 타악음 스펙트로그램 (H, W, 1)
        """
        audio = self.load_audio(audio_path)
        full_spec, percussive_spec = self.compute_spectrogram(audio)

        # 채널 차원 추가
        full_spec = full_spec[..., np.newaxis]
        percussive_spec = percussive_spec[..., np.newaxis]

        return full_spec, percussive_spec

    def process_batch(
        self, audio_paths: list, show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 전처리

        Args:
            audio_paths: 오디오 파일 경로 리스트
            show_progress: 진행 상황 출력

        Returns:
            full_specs: (N, H, W, 1)
            percussive_specs: (N, H, W, 1)
        """
        full_specs = []
        percussive_specs = []

        for i, path in enumerate(audio_paths):
            if show_progress and (i + 1) % 50 == 0:
                print(f"  처리 중: {i + 1}/{len(audio_paths)}")

            full_spec, percussive_spec = self.process(path)
            full_specs.append(full_spec)
            percussive_specs.append(percussive_spec)

        return np.array(full_specs), np.array(percussive_specs)

    def compute_diff_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """
        Difference Spectrogram - 시간 변동 시각화

        연속 프레임 간 차이를 계산하여 급격한 변화(노킹, 미스파이어)를 강조

        Args:
            spec: 스펙트로그램 (H, W)

        Returns:
            diff_spec: 차분 스펙트로그램 (H, W)
        """
        # 시간축(axis=1) 차분
        diff = np.diff(spec, axis=1)
        diff = np.abs(diff)

        # 마지막 열 패딩 (원본 크기 유지)
        diff = np.pad(diff, ((0, 0), (0, 1)), mode='edge')

        # 정규화
        if diff.max() > 0:
            diff = diff / diff.max()

        return diff

    def compute_variance_map(self, spec: np.ndarray, window: int = 8) -> np.ndarray:
        """
        Variance Map - 불안정 영역 강조

        이동 윈도우 분산을 계산하여 시간에 따라 불안정한 영역을 시각화

        Args:
            spec: 스펙트로그램 (H, W)
            window: 이동 윈도우 크기

        Returns:
            variance_map: 분산 맵 (H, W)
        """
        from scipy.ndimage import uniform_filter

        # 이동 평균
        local_mean = uniform_filter(spec, size=(1, window))
        # 이동 제곱 평균
        local_sq_mean = uniform_filter(spec**2, size=(1, window))
        # 분산 = E[X²] - E[X]²
        variance = local_sq_mean - local_mean**2
        variance = np.maximum(variance, 0)  # 수치 오류 방지

        # 정규화
        if variance.max() > 0:
            variance = variance / variance.max()

        return variance

    def compute_4channel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        4채널 스펙트로그램 생성

        채널 구성:
            1. Full Spectrogram - 주파수 특성
            2. Percussive Spectrogram - 충격음 성분
            3. Difference Spectrogram - 시간 변동 (급격한 변화)
            4. Variance Map - 불안정 영역

        Args:
            audio: 오디오 배열

        Returns:
            4채널 스펙트로그램 (H, W, 4)
        """
        # 기존 Full + Percussive
        full_spec, percussive_spec = self.compute_spectrogram(audio)

        # 신규: Difference + Variance
        diff_spec = self.compute_diff_spectrogram(full_spec)
        variance_map = self.compute_variance_map(full_spec)

        # 크기 조정
        diff_spec = self._resize(diff_spec)
        variance_map = self._resize(variance_map)

        # 4채널 스택
        return np.stack([full_spec, percussive_spec, diff_spec, variance_map], axis=-1)

    def process_4channel(self, audio_path: str) -> np.ndarray:
        """
        4채널 전처리 파이프라인

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            4채널 스펙트로그램 (H, W, 4)
        """
        audio = self.load_audio(audio_path)
        return self.compute_4channel_spectrogram(audio)
