"""
오디오 전처리 모듈 (메모리 기반 증강 지원)
역할: 오디오 파일 또는 배열을 스펙트로그램으로 변환
- STFT 또는 Mel Spectrogram 지원
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt


class AudioPreprocessor:
    """
    오디오 신호를 스펙트로그램으로 변환하는 클래스
    - 파일 경로 또는 NumPy 배열 입력 지원
    - STFT 또는 Mel Spectrogram 선택 가능
    """

    def __init__(self,
                 sample_rate=22050,
                 n_fft=2048,
                 hop_length=512,
                 fmax=6000,
                 use_mel=False,
                 n_mels=128):
        """
        초기화 함수

        매개변수:
            sample_rate: 샘플링 레이트 (Hz)
            n_fft: FFT 윈도우 크기
            hop_length: STFT 홉 길이
            fmax: 최대 주파수 (Hz)
            use_mel: True면 Mel Spectrogram 사용
            n_mels: Mel 필터뱅크 개수 (use_mel=True일 때)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.use_mel = use_mel
        self.n_mels = n_mels

        # 주파수 bin 계산 (STFT용)
        self.freq_resolution = sample_rate / n_fft
        self.max_freq_bin = int(fmax / self.freq_resolution)

        # 로그 출력 플래그 (중복 방지)
        self._freq_logged = False

    def _log_freq_info(self):
        """주파수 설정 정보 출력 (한 번만)"""
        if not self._freq_logged:
            if self.use_mel:
                print(f"[MEL] n_mels={self.n_mels}, fmax={self.fmax}Hz")
            else:
                print(f"[STFT] freq_res={self.freq_resolution:.2f}Hz/bin, "
                      f"fmax={self.fmax}Hz, bins=0~{self.max_freq_bin}")
            self._freq_logged = True

    def load_audio(self, audio_path):
        """
        오디오 파일 로드 (WAV, MP3 모두 지원)

        매개변수:
            audio_path: 오디오 파일 경로

        반환값:
            audio: 오디오 NumPy 배열
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            raise RuntimeError(f"오디오 로드 실패 ({audio_path}): {e}")

    def normalize_audio(self, audio):
        """
        오디오 신호 정규화 (-1 ~ 1)

        매개변수:
            audio: 오디오 배열

        반환값:
            normalized_audio: 정규화된 오디오
        """
        max_value = np.max(np.abs(audio))

        if max_value == 0:
            return audio

        normalized_audio = audio / max_value
        return normalized_audio

    def compute_stft(self, audio):
        """
        STFT 계산

        매개변수:
            audio: 오디오 배열

        반환값:
            stft_matrix: STFT 행렬 (복소수)
        """
        stft_matrix = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return stft_matrix

    def limit_frequency_range(self, stft_matrix):
        """
        STFT 결과에서 0~fmax Hz 범위만 추출

        매개변수:
            stft_matrix: 전체 STFT 결과 (n_fft/2+1, time_frames)

        반환값:
            limited_stft: 주파수 제한된 STFT (max_freq_bin+1, time_frames)
        """
        limited_stft = stft_matrix[:self.max_freq_bin + 1, :]
        return limited_stft

    def to_db_and_normalize(self, magnitude):
        """
        크기 스펙트럼을 dB 변환 후 0~1 정규화

        매개변수:
            magnitude: 크기 스펙트럼 (실수)

        반환값:
            normalized: 정규화된 스펙트로그램 (0~1)
        """
        # dB 변환
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

        # 0~1 정규화
        min_val = magnitude_db.min()
        max_val = magnitude_db.max()
        normalized = (magnitude_db - min_val) / (max_val - min_val + 1e-8)

        return normalized

    def compute_mel_spectrogram(self, audio):
        """
        Mel Spectrogram 계산

        매개변수:
            audio: 오디오 배열

        반환값:
            mel_spec: Mel Spectrogram (0~1 정규화)
        """
        # Mel Spectrogram 계산
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax
        )

        # dB 변환 및 정규화
        mel_spec_normalized = self.to_db_and_normalize(mel_spec)

        return mel_spec_normalized

    def compute_log_spectrogram(self, stft_matrix):
        """
        로그 스펙트로그램 계산 (주파수 범위 제한 적용)

        매개변수:
            stft_matrix: STFT 행렬

        반환값:
            log_spectrogram: 로그 스펙트로그램 (0~1 정규화)
        """
        # 주파수 범위 제한
        limited_stft = self.limit_frequency_range(stft_matrix)

        # 크기 스펙트럼
        magnitude = np.abs(limited_stft)

        # dB 변환 및 정규화
        log_spectrogram = self.to_db_and_normalize(magnitude)

        return log_spectrogram

    def separate_harmonic_percussive(self, stft_matrix, audio=None):
        """
        HPSS 분리 (주파수 범위 제한 또는 Mel 적용)

        매개변수:
            stft_matrix: STFT 행렬
            audio: 오디오 배열 (Mel 모드에서 필요)

        반환값:
            harmonic_spec: Harmonic 스펙트로그램 (0~1)
            percussive_spec: Percussive 스펙트로그램 (0~1)
        """
        # HPSS 분리
        stft_harmonic, stft_percussive = librosa.decompose.hpss(
            stft_matrix,
            kernel_size=31,
            margin=2.0
        )

        if self.use_mel:
            # Mel Spectrogram 모드: HPSS 결과를 Mel 스케일로 변환
            # Harmonic
            harmonic_audio = librosa.istft(stft_harmonic, hop_length=self.hop_length)
            harmonic_mel = librosa.feature.melspectrogram(
                y=harmonic_audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmax=self.fmax
            )
            harmonic_spec = self.to_db_and_normalize(harmonic_mel)

            # Percussive
            percussive_audio = librosa.istft(stft_percussive, hop_length=self.hop_length)
            percussive_mel = librosa.feature.melspectrogram(
                y=percussive_audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmax=self.fmax
            )
            percussive_spec = self.to_db_and_normalize(percussive_mel)
        else:
            # STFT 모드: 주파수 범위 제한
            limited_harmonic = self.limit_frequency_range(stft_harmonic)
            limited_percussive = self.limit_frequency_range(stft_percussive)

            # 크기 스펙트럼
            harmonic_magnitude = np.abs(limited_harmonic)
            percussive_magnitude = np.abs(limited_percussive)

            # dB 변환 및 정규화
            harmonic_spec = self.to_db_and_normalize(harmonic_magnitude)
            percussive_spec = self.to_db_and_normalize(percussive_magnitude)

        return harmonic_spec, percussive_spec

    def preprocess(self, audio_path, save_first_sample=False):
        """
        전체 전처리 파이프라인 (파일 경로 입력)

        매개변수:
            audio_path: 오디오 파일 경로 (WAV 또는 MP3)
            save_first_sample: True면 첫 샘플 스펙트로그램 저장

        반환값:
            full_spec: 전체 스펙트로그램 (0~fmax Hz)
            percussive_spec: 타악음 스펙트로그램 (0~fmax Hz)
        """
        # 주파수 정보 출력 (첫 호출 시만)
        self._log_freq_info()

        # 1. 오디오 로드
        audio = self.load_audio(audio_path)

        # 2. 배열 기반 전처리 호출
        return self.preprocess_from_array(
            audio,
            sr=self.sample_rate,
            save_first_sample=save_first_sample
        )

    def preprocess_from_array(self, audio_array, sr=None, save_first_sample=False):
        """
        전체 전처리 파이프라인 (오디오 배열 입력) - 메모리 기반 증강용

        매개변수:
            audio_array: 오디오 NumPy 배열
            sr: 샘플링 레이트 (None이면 self.sample_rate 사용)
            save_first_sample: True면 첫 샘플 스펙트로그램 저장

        반환값:
            full_spec: 전체 스펙트로그램 (STFT 또는 Mel)
            percussive_spec: 타악음 스펙트로그램 (STFT 또는 Mel)
        """
        # 주파수 정보 출력 (첫 호출 시만)
        self._log_freq_info()

        if sr is None:
            sr = self.sample_rate

        # 1. 정규화
        audio = self.normalize_audio(audio_array)

        # 2. STFT 계산 (HPSS에 필요)
        stft_matrix = self.compute_stft(audio)

        # 3. 전체 스펙트로그램
        if self.use_mel:
            full_spectrogram = self.compute_mel_spectrogram(audio)
        else:
            full_spectrogram = self.compute_log_spectrogram(stft_matrix)

        # 4. HPSS 분리
        harmonic_spec, percussive_spec = self.separate_harmonic_percussive(stft_matrix, audio)

        # 5. 시각화 저장 (첫 샘플만)
        if save_first_sample:
            self._save_spectrograms(full_spectrogram, percussive_spec)

        return full_spectrogram, percussive_spec

    def extract_physics_features(self, audio_array, sr=None):
        """
        오디오에서 물리량 기반 특징 추출

        매개변수:
            audio_array: 오디오 NumPy 배열
            sr: 샘플링 레이트 (None이면 self.sample_rate 사용)

        반환값:
            features: Dict 형태의 물리량 특징
                - zcr: Zero Crossing Rate (고주파 노이즈 지표)
                - zcr_std: ZCR 표준편차
                - spectral_centroid: 스펙트럴 센트로이드 (주파수 중심)
                - spectral_centroid_std: 센트로이드 표준편차
                - spectral_flatness: 스펙트럴 평탄도 (노이즈 비율)
                - spectral_flatness_std: 평탄도 표준편차
                - rms: RMS 에너지 (전체 에너지)
                - rms_std: RMS 표준편차
                - harmonic_ratio: 하모닉 에너지 비율 (규칙성)
                - percussive_ratio: 퍼커시브 에너지 비율
        """
        if sr is None:
            sr = self.sample_rate

        # 정규화
        audio = self.normalize_audio(audio_array)

        features = {}

        # 1. Zero Crossing Rate (고주파 노이즈 지표)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # 2. Spectral Centroid (주파수 중심)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(centroid))
        features['spectral_centroid_std'] = float(np.std(centroid))

        # 3. Spectral Flatness (노이즈 비율, 0=순음, 1=노이즈)
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features['spectral_flatness'] = float(np.mean(flatness))
        features['spectral_flatness_std'] = float(np.std(flatness))

        # 4. RMS Energy (전체 에너지)
        rms = librosa.feature.rms(y=audio)[0]
        features['rms'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # 5. Harmonic-to-Percussive Ratio (규칙성 지표)
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        harmonic, percussive = librosa.decompose.hpss(stft)

        harmonic_energy = np.sum(np.abs(harmonic) ** 2)
        percussive_energy = np.sum(np.abs(percussive) ** 2)
        total_energy = harmonic_energy + percussive_energy + 1e-10

        features['harmonic_ratio'] = float(harmonic_energy / total_energy)
        features['percussive_ratio'] = float(percussive_energy / total_energy)

        # 6. Spectral Bandwidth (주파수 대역폭)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth'] = float(np.mean(bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(bandwidth))

        # 7. Spectral Rolloff (에너지 집중 주파수)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(rolloff))
        features['spectral_rolloff_std'] = float(np.std(rolloff))

        return features

    def get_physics_feature_names(self):
        """
        물리량 특징 이름 목록 반환 (모델 입력 차원 계산용)

        반환값:
            feature_names: 특징 이름 리스트
        """
        return [
            'zcr', 'zcr_std',
            'spectral_centroid', 'spectral_centroid_std',
            'spectral_flatness', 'spectral_flatness_std',
            'rms', 'rms_std',
            'harmonic_ratio', 'percussive_ratio',
            'spectral_bandwidth', 'spectral_bandwidth_std',
            'spectral_rolloff', 'spectral_rolloff_std'
        ]

    def extract_modulation_features(self, audio_array, sr=None):
        """
        모듈레이션 스펙트럼 기반 특징 추출 (regularity/irregularity 예측용)

        논문 참조: "소음 데이터를 이용한 딥러닝 기반의 차량 진단 기술 개발"
        - Modulation Spectrum: 음의 변동성 측정
        - SAML (Sum of Audible Modulation Level): 임계값 초과 성분 합

        매개변수:
            audio_array: 오디오 NumPy 배열
            sr: 샘플링 레이트 (None이면 self.sample_rate 사용)

        반환값:
            features: Dict 형태의 모듈레이션 특징
        """
        if sr is None:
            sr = self.sample_rate

        # 정규화
        audio = self.normalize_audio(audio_array)

        features = {}

        # 논문 설정: frame_length=150ms, hop_length=75ms
        frame_length = int(0.150 * sr)  # 150ms
        hop_length_mod = int(0.075 * sr)  # 75ms

        # 1. 프레임별 RMS 에너지 (envelope)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length_mod
        )[0]

        # 2. Envelope의 FFT → Modulation Spectrum
        if len(rms) > 1:
            # DC 성분 제거
            rms_centered = rms - np.mean(rms)

            # Modulation Spectrum 계산
            mod_spectrum = np.abs(np.fft.rfft(rms_centered))
            mod_freqs = np.fft.rfftfreq(len(rms_centered), d=hop_length_mod / sr)

            # Modulation 특징 추출
            features['mod_mean'] = float(np.mean(mod_spectrum))
            features['mod_std'] = float(np.std(mod_spectrum))
            features['mod_max'] = float(np.max(mod_spectrum))

            # Peak frequency (주기성 → regularity)
            if len(mod_spectrum) > 1:
                peak_idx = np.argmax(mod_spectrum[1:]) + 1  # DC 제외
                features['mod_peak_freq'] = float(mod_freqs[peak_idx])
                features['mod_peak_ratio'] = float(mod_spectrum[peak_idx] / (np.mean(mod_spectrum) + 1e-8))
            else:
                features['mod_peak_freq'] = 0.0
                features['mod_peak_ratio'] = 1.0

            # 저주파/고주파 모듈레이션 에너지 (10Hz 기준)
            low_mask = mod_freqs < 10
            high_mask = mod_freqs >= 10
            features['mod_low_energy'] = float(np.sum(mod_spectrum[low_mask]))
            features['mod_high_energy'] = float(np.sum(mod_spectrum[high_mask]))
            features['mod_low_high_ratio'] = float(
                features['mod_low_energy'] / (features['mod_high_energy'] + 1e-8)
            )
        else:
            # 오디오가 너무 짧은 경우
            features['mod_mean'] = 0.0
            features['mod_std'] = 0.0
            features['mod_max'] = 0.0
            features['mod_peak_freq'] = 0.0
            features['mod_peak_ratio'] = 1.0
            features['mod_low_energy'] = 0.0
            features['mod_high_energy'] = 0.0
            features['mod_low_high_ratio'] = 1.0

        # 3. 시간적 변동성 특징 (irregularity 관련)
        # Spectral Flux: 연속 프레임 간 스펙트럼 변화
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_mag = np.abs(stft)

        if spec_mag.shape[1] > 1:
            spectral_flux = np.sqrt(np.sum(np.diff(spec_mag, axis=1) ** 2, axis=0))
            features['spectral_flux_mean'] = float(np.mean(spectral_flux))
            features['spectral_flux_std'] = float(np.std(spectral_flux))
            features['spectral_flux_max'] = float(np.max(spectral_flux))
        else:
            features['spectral_flux_mean'] = 0.0
            features['spectral_flux_std'] = 0.0
            features['spectral_flux_max'] = 0.0

        # 4. Onset Strength (충격음/불규칙 이벤트 감지)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_mean'] = float(np.mean(onset_env))
        features['onset_std'] = float(np.std(onset_env))
        features['onset_max'] = float(np.max(onset_env))

        # Onset 개수 (급격한 변화 횟수)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames')
        features['onset_count'] = float(len(onset_frames))
        features['onset_rate'] = float(len(onset_frames) / (len(audio) / sr))  # per second

        # 5. Autocorrelation 기반 주기성 (regularity)
        autocorr = np.correlate(rms, rms, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # 양의 lag만
        autocorr = autocorr / (autocorr[0] + 1e-8)  # 정규화

        if len(autocorr) > 1:
            # 첫 번째 피크 찾기 (lag=0 제외)
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                    peaks.append((i, autocorr[i]))

            if peaks:
                first_peak = max(peaks, key=lambda x: x[1])
                features['autocorr_peak_lag'] = float(first_peak[0])
                features['autocorr_peak_value'] = float(first_peak[1])
            else:
                features['autocorr_peak_lag'] = 0.0
                features['autocorr_peak_value'] = 0.0

            # 자기상관 감쇠율 (규칙적일수록 천천히 감쇠)
            decay_point = np.argmax(autocorr < 0.5) if np.any(autocorr < 0.5) else len(autocorr)
            features['autocorr_decay'] = float(decay_point / len(autocorr))
        else:
            features['autocorr_peak_lag'] = 0.0
            features['autocorr_peak_value'] = 0.0
            features['autocorr_decay'] = 0.0

        return features

    def get_modulation_feature_names(self):
        """
        모듈레이션 특징 이름 목록 반환

        반환값:
            feature_names: 특징 이름 리스트 (20개)
        """
        return [
            # Modulation Spectrum (9개)
            'mod_mean', 'mod_std', 'mod_max',
            'mod_peak_freq', 'mod_peak_ratio',
            'mod_low_energy', 'mod_high_energy', 'mod_low_high_ratio',
            # Spectral Flux (3개)
            'spectral_flux_mean', 'spectral_flux_std', 'spectral_flux_max',
            # Onset (5개)
            'onset_mean', 'onset_std', 'onset_max', 'onset_count', 'onset_rate',
            # Autocorrelation (3개)
            'autocorr_peak_lag', 'autocorr_peak_value', 'autocorr_decay'
        ]

    def _save_spectrograms(self, full_spec, percussive_spec):
        """
        스펙트로그램을 이미지 파일로 저장

        매개변수:
            full_spec: 전체 스펙트로그램
            percussive_spec: 타악음 스펙트로그램
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ========== 1. 전체 스펙트로그램 ==========
        img1 = axes[0].imshow(
            full_spec,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        axes[0].set_title(f'Full Spectrogram (0~{self.fmax}Hz)', fontsize=12)
        axes[0].set_xlabel('Time Frame', fontsize=10)
        axes[0].set_ylabel('Frequency Bin', fontsize=10)
        fig.colorbar(img1, ax=axes[0], label='Normalized Amplitude')

        # ========== 2. 타악음 스펙트로그램 ==========
        img2 = axes[1].imshow(
            percussive_spec,
            aspect='auto',
            origin='lower',
            cmap='magma',
            interpolation='nearest'
        )
        axes[1].set_title(f'Percussive Spectrogram (0~{self.fmax}Hz)', fontsize=12)
        axes[1].set_xlabel('Time Frame', fontsize=10)
        axes[1].set_ylabel('Frequency Bin', fontsize=10)
        fig.colorbar(img2, ax=axes[1], label='Normalized Amplitude')

        plt.tight_layout()
        plt.savefig('first_sample_spectrograms.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 첫 번째 샘플 스펙트로그램 저장: first_sample_spectrograms.png")


# ========== 테스트용 코드 ==========
if __name__ == "__main__":
    """
    AudioPreprocessor 테스트
    """
    print("=" * 60)
    print("🎵 AudioPreprocessor 테스트")
    print("=" * 60)

    # 전처리 객체 생성
    preprocessor = AudioPreprocessor()

    # ========== 테스트 1: 더미 오디오 배열 처리 ==========
    print("\n📊 테스트 1: 더미 오디오 배열 처리")
    print("-" * 60)

    # 더미 오디오 생성 (3초, 440Hz 사인파)
    duration = 3.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)

    print(f"✅ 더미 오디오 생성")
    print(f"   - 샘플링 레이트: {sr} Hz")
    print(f"   - 지속시간: {duration}초")
    print(f"   - 샘플 수: {len(audio)}")

    # 배열 기반 전처리
    print(f"\n🔄 배열 기반 전처리 수행...")
    full_spec, perc_spec = preprocessor.preprocess_from_array(
        audio,
        sr=sr,
        save_first_sample=True
    )

    print(f"\n✅ 전처리 완료!")
    print(f"   - Full Spectrogram: {full_spec.shape}")
    print(f"   - Percussive Spectrogram: {perc_spec.shape}")
    print(f"   - 값 범위: {full_spec.min():.4f} ~ {full_spec.max():.4f}")

    # ========== 테스트 2: 증강 시뮬레이션 ==========
    print("\n" + "=" * 60)
    print("📊 테스트 2: 증강 시뮬레이션")
    print("-" * 60)

    # 증강 시뮬레이션 (노이즈 추가)
    noise = np.random.randn(len(audio)) * 0.01
    augmented_audio = audio + noise

    print("🔄 증강된 오디오 전처리 수행...")
    aug_full_spec, aug_perc_spec = preprocessor.preprocess_from_array(
        augmented_audio,
        sr=sr,
        save_first_sample=False
    )

    print(f"\n✅ 증강 전처리 완료!")
    print(f"   - Augmented Full Spectrogram: {aug_full_spec.shape}")
    print(f"   - Augmented Percussive Spectrogram: {aug_perc_spec.shape}")

    # 차이 분석
    diff = np.abs(full_spec - aug_full_spec).mean()
    print(f"\n📊 원본 vs 증강 차이:")
    print(f"   - 평균 절대 차이: {diff:.4f}")

    print("\n" + "=" * 60)
    print("🎉 모든 테스트 완료!")
    print("=" * 60)

    print("\n생성된 파일:")
    print("   ✅ first_sample_spectrograms.png")
