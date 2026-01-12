import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def analyze_mp3_peaks(file_path):
    if not os.path.exists(file_path):
        print("파일이 존재하지 않습니다.")
        return

    print(f"[MP3 분석: {file_path}]")
    # 1. 오디오 로드
    y, sr = librosa.load(file_path)

    # 2. 오디오 신호의 히스토그램 plot
    times = np.arange(len(y)) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(times, y, color='skyblue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform (Time vs Amplitude)")
    plt.tight_layout()
    plt.show()

    # 3. Waveform peak normalization (최대값이 1)
    peak = np.max(np.abs(y))
    if peak == 0:
        y_norm = y
    else:
        y_norm = y / peak
    print(f"Peak-normalized waveform max value: {np.max(np.abs(y_norm))}")

    # 4. Waveform plot (Peak Normalized)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y_norm, sr=sr, color='purple')
    plt.title('Peak-normalized Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (normalized)')
    plt.tight_layout()
    plt.show()

    # 5. STFT 후 로그 스케일 스펙트로그램 (dB)
    D = librosa.stft(y_norm, n_fft=2048, hop_length=512)
    D_mag = np.abs(D)
    D_db = librosa.amplitude_to_db(D_mag, ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.title("Log-Scaled Spectrogram (dB) of STFT")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()

    # 6. HPSS(하모닉/타음 분리)
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    H_db = librosa.amplitude_to_db(np.abs(D_harmonic), ref=np.max)
    P_db = librosa.amplitude_to_db(np.abs(D_percussive), ref=np.max)

    # 7. (그래프) 하모닉 성분 스펙트로그램
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(H_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.title("Harmonic Spectrogram (Log Scaled dB)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()

    # 8. (그래프) 타음(퍼커시브) 성분 스펙트로그램
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(P_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.title("Percussive Spectrogram (Log Scaled dB)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()

    # 9. 딥러닝 특성 벡터 예시 (하모닉/타음 각각 2D 배열!)
    # 필요에 따라 H_db, P_db를 flatten 하거나 pooling/축약 가능
    print(f"Harmonic spectrogram shape for ML: {H_db.shape}")
    print(f"Percussive spectrogram shape for ML: {P_db.shape}")
    # 예시: model_input = np.concatenate([H_db.flatten(), P_db.flatten()])
    input("그래프 창을 닫으려면 엔터(Enter)를 누르세요.")


# === 메인 부분 ===
if __name__ == "__main__":
    mp3_file_path = r"D:\123.mp3"  # 고정된 경로
    analyze_mp3_peaks(mp3_file_path)

"""
    # 5. STFT 및 Peak Normalization
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    D_mag = np.abs(D)
    stft_peak = np.max(D_mag)
    if stft_peak == 0:
        D_norm = D_mag
    else:
        D_norm = D_mag / stft_peak
    print(f"Peak-normalized STFT max value: {np.max(D_norm)}")

    # 6. STFT magnitude plot (Peak Normalized)
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(D_norm, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.title('Peak-normalized STFT Spectrogram')
    plt.colorbar(format='%+2.2f')
    plt.tight_layout()
    plt.show()
"""

"""
    # 6. 특정 프레임의 STFT 결과 x축=Frequency, y축=Amplitude로 plot
    frame_idx = D_mag.shape[1] // 2  # 중간 프레임 (전체 프레임 중간)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    amplitudes = D_mag[:, frame_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, amplitudes, color='darkgreen')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'STFT Spectrum at Frame {frame_idx} (Peak normalized waveform)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""