#!/usr/bin/env python
"""
노이즈 필터링 테스트

스마트폰 녹음 파일에 노이즈 필터링을 적용하고 전/후 비교

Usage:
    py -3.12 scripts/noise_filter_test.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import noisereduce as nr

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EngineNoiseFilter:
    """엔진 오디오 노이즈 필터링"""

    def __init__(self, sr: int = 22050):
        self.sr = sr
        # 엔진 주파수 범위 (아이들링 ~ 고RPM 고조파)
        self.low_freq = 30      # 저역 컷오프 (바람 소리 제거)
        self.high_freq = 6000   # 고역 컷오프 (고주파 노이즈 제거)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """오디오 로드"""
        y, _ = librosa.load(audio_path, sr=self.sr)
        return y

    def bandpass_filter(self, y: np.ndarray) -> np.ndarray:
        """
        밴드패스 필터 - 엔진 주파수 대역만 통과
        """
        nyquist = self.sr / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist

        # Butterworth 필터 (4차)
        b, a = butter(4, [low, high], btype='band')
        y_filtered = filtfilt(b, a, y)

        return y_filtered

    def spectral_gating(self, y: np.ndarray, prop_decrease: float = 0.8) -> np.ndarray:
        """
        Spectral Gating - 배경 소음 제거

        Args:
            y: 오디오 신호
            prop_decrease: 노이즈 감소 비율 (0~1)
        """
        # 첫 0.5초를 노이즈 프로파일로 사용 (또는 전체에서 추정)
        y_reduced = nr.reduce_noise(
            y=y,
            sr=self.sr,
            prop_decrease=prop_decrease,
            stationary=False,  # 비정상 노이즈도 처리
            n_fft=2048,
            hop_length=512
        )
        return y_reduced

    def apply_all_filters(self, y: np.ndarray) -> dict:
        """
        모든 필터 적용 및 결과 반환

        Returns:
            {
                'original': 원본,
                'bandpass': 밴드패스만,
                'spectral': 스펙트럴 게이팅만,
                'combined': 모두 적용
            }
        """
        results = {
            'original': y
        }

        # 1. 밴드패스 필터
        y_bandpass = self.bandpass_filter(y)
        results['bandpass'] = y_bandpass

        # 2. 스펙트럴 게이팅 (원본에 적용)
        y_spectral = self.spectral_gating(y)
        results['spectral'] = y_spectral

        # 3. 결합: 밴드패스 → 스펙트럴 게이팅
        y_combined = self.spectral_gating(y_bandpass, prop_decrease=0.6)
        results['combined'] = y_combined

        return results

    def compute_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """멜 스펙트로그램 계산"""
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=128,
            n_fft=2048, hop_length=512, fmax=6000
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db


def plot_comparison(audio_path: str, output_dir: Path):
    """전/후 스펙트로그램 비교 플롯"""

    filter_engine = EngineNoiseFilter()

    # 오디오 로드
    y = filter_engine.load_audio(audio_path)

    # 필터 적용
    results = filter_engine.apply_all_filters(y)

    # 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    titles = ['Original', 'Bandpass (30-6000Hz)', 'Spectral Gating', 'Combined']
    keys = ['original', 'bandpass', 'spectral', 'combined']

    for ax, title, key in zip(axes.flat, titles, keys):
        S_db = filter_engine.compute_spectrogram(results[key])
        librosa.display.specshow(
            S_db, sr=filter_engine.sr, hop_length=512,
            x_axis='time', y_axis='mel', ax=ax, fmax=6000
        )
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

    plt.suptitle(f'Noise Filtering: {Path(audio_path).name}', fontsize=14)
    plt.tight_layout()

    # 저장
    output_path = output_dir / f"{Path(audio_path).stem}_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def analyze_noise_levels(audio_path: str) -> dict:
    """노이즈 레벨 분석"""

    filter_engine = EngineNoiseFilter()
    y = filter_engine.load_audio(audio_path)
    results = filter_engine.apply_all_filters(y)

    analysis = {}
    for key, signal in results.items():
        # RMS 에너지
        rms = np.sqrt(np.mean(signal**2))
        # 피크
        peak = np.max(np.abs(signal))
        # SNR 추정 (신호 대 잔차)
        if key != 'original':
            residual = results['original'][:len(signal)] - signal
            snr = 10 * np.log10(np.mean(signal**2) / (np.mean(residual**2) + 1e-10))
        else:
            snr = 0

        analysis[key] = {
            'rms': float(rms),
            'peak': float(peak),
            'snr_db': float(snr) if key != 'original' else None
        }

    return analysis


def main():
    print("=" * 60)
    print("노이즈 필터링 테스트")
    print("=" * 60)
    print()

    # 샘플 디렉토리
    sample_dir = project_root / "sample"
    output_dir = project_root / "results" / "noise_filter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # M4A 파일 목록
    audio_files = sorted(sample_dir.glob("*.m4a"))
    print(f"테스트 파일: {len(audio_files)}개\n")

    all_results = []

    for audio_path in audio_files:
        print(f"처리 중: {audio_path.name}")

        try:
            # 스펙트로그램 비교 이미지 생성
            img_path = plot_comparison(str(audio_path), output_dir)
            print(f"  → 이미지 저장: {img_path.name}")

            # 노이즈 레벨 분석
            analysis = analyze_noise_levels(str(audio_path))

            # 결과 요약
            orig_rms = analysis['original']['rms']
            comb_rms = analysis['combined']['rms']
            reduction = (1 - comb_rms / orig_rms) * 100

            print(f"  → RMS: {orig_rms:.4f} → {comb_rms:.4f} ({reduction:.1f}% 감소)")

            all_results.append({
                'filename': audio_path.name,
                'analysis': analysis,
                'rms_reduction_pct': reduction
            })

        except Exception as e:
            print(f"  → 에러: {e}")
            all_results.append({
                'filename': audio_path.name,
                'error': str(e)
            })

    # 결과 요약
    print()
    print("=" * 60)
    print("결과 요약")
    print("=" * 60)

    valid = [r for r in all_results if 'error' not in r]
    if valid:
        reductions = [r['rms_reduction_pct'] for r in valid]
        print(f"평균 RMS 감소: {np.mean(reductions):.1f}%")
        print(f"최소 감소: {np.min(reductions):.1f}%")
        print(f"최대 감소: {np.max(reductions):.1f}%")

    print(f"\n결과 이미지: {output_dir}")

    # JSON 저장
    import json
    result_path = output_dir / "noise_analysis.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"분석 결과: {result_path}")


if __name__ == "__main__":
    main()
