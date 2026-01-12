"""
🎵 오디오 데이터 증강 모듈 (AudioAugment.py)
역할: 오디오 데이터에 다양한 증강 기법 적용
"""

import numpy as np
import librosa
from typing import Tuple


class AudioAugmentor:
    """
    오디오 증강 클래스
    - 다양한 증강 기법으로 학습 데이터 다양성 증가
    """

    def __init__(self, sample_rate=22050):
        """
        초기화 함수

        매개변수:
            sample_rate: 샘플링 레이트 (기본값: 22050 Hz)
        """
        self.sample_rate = sample_rate  # 샘플링 레이트

    def time_shift(self, audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """
        시간 이동 (Time Shift)
        - 오디오를 좌우로 무작위 이동

        매개변수:
            audio: 입력 오디오 신호
            shift_max: 최대 이동 비율 (0~1, 0.2 = 20%)

        반환값:
            shifted_audio: 이동된 오디오
        """
        # 이동할 샘플 수 계산 (무작위)
        shift_amount = int(np.random.uniform(-shift_max, shift_max) * len(audio))

        # 오디오 이동 (roll: 순환 이동)
        shifted_audio = np.roll(audio, shift_amount)

        return shifted_audio

    def pitch_shift(self, audio: np.ndarray, n_steps: int = None) -> np.ndarray:
        """
        피치 이동 (Pitch Shift)
        - 음높이 변경 (속도 변화 없이)

        매개변수:
            audio: 입력 오디오
            n_steps: 이동할 반음 수 (None이면 -2~2 무작위)

        반환값:
            pitched_audio: 피치 이동된 오디오
        """
        # n_steps가 없으면 무작위 선택
        if n_steps is None:
            n_steps = np.random.randint(-2, 3)  # -2, -1, 0, 1, 2

        # librosa의 pitch_shift 사용
        pitched_audio = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=n_steps
        )

        return pitched_audio

    def time_stretch(self, audio: np.ndarray, rate: float = None) -> np.ndarray:
        """
        시간 신축 (Time Stretch)
        - 속도 변경 (피치 변화 없이)

        매개변수:
            audio: 입력 오디오
            rate: 속도 배율 (None이면 0.8~1.2 무작위)
                  1.0 = 원본
                  1.2 = 1.2배 빠르게
                  0.8 = 0.8배 느리게

        반환값:
            stretched_audio: 시간 신축된 오디오
        """
        # rate가 없으면 무작위 선택
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)

        # librosa의 time_stretch 사용
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate)

        return stretched_audio

    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        잡음 추가 (Add Noise)
        - 백색 잡음 (White Noise) 추가

        매개변수:
            audio: 입력 오디오
            noise_factor: 잡음 강도 (0~1, 0.005 = 0.5%)

        반환값:
            noisy_audio: 잡음이 추가된 오디오
        """
        # 백색 잡음 생성 (평균 0, 표준편차 1의 정규분포)
        noise = np.random.randn(len(audio))

        # 잡음 강도 조절
        noise = noise * noise_factor

        # 원본 오디오에 잡음 추가
        noisy_audio = audio + noise

        # 클리핑 방지 (-1 ~ 1 범위로 제한)
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        return noisy_audio

    def change_volume(self, audio: np.ndarray, gain_db: float = None) -> np.ndarray:
        """
        볼륨 변경 (Change Volume)
        - 오디오 볼륨 증가/감소

        매개변수:
            audio: 입력 오디오
            gain_db: 볼륨 변화량 (dB, None이면 -6~6 무작위)

        반환값:
            adjusted_audio: 볼륨 조정된 오디오
        """
        # gain_db가 없으면 무작위 선택
        if gain_db is None:
            gain_db = np.random.uniform(-6, 6)

        # dB를 선형 배율로 변환
        gain_linear = 10 ** (gain_db / 20.0)

        # 볼륨 조정
        adjusted_audio = audio * gain_linear

        # 클리핑 방지
        adjusted_audio = np.clip(adjusted_audio, -1.0, 1.0)

        return adjusted_audio

    def apply_random_augmentations(
            self,
            audio: np.ndarray,
            apply_time_shift: bool = True,
            apply_pitch_shift: bool = True,
            apply_time_stretch: bool = True,
            apply_noise: bool = True,
            apply_volume: bool = True,
            probability: float = 0.5
    ) -> np.ndarray:
        """
        무작위 증강 조합 적용
        - 여러 증강 기법을 확률적으로 적용

        매개변수:
            audio: 입력 오디오
            apply_*: 각 증강 기법 활성화 여부
            probability: 각 증강 기법 적용 확률 (0~1)

        반환값:
            augmented_audio: 증강된 오디오
        """
        augmented_audio = audio.copy()  # 원본 보존

        # 1. 시간 이동 (50% 확률)
        if apply_time_shift and np.random.random() < probability:
            augmented_audio = self.time_shift(augmented_audio)

        # 2. 피치 이동 (50% 확률)
        if apply_pitch_shift and np.random.random() < probability:
            augmented_audio = self.pitch_shift(augmented_audio)

        # 3. 시간 신축 (50% 확률)
        if apply_time_stretch and np.random.random() < probability:
            augmented_audio = self.time_stretch(augmented_audio)

        # 4. 잡음 추가 (50% 확률)
        if apply_noise and np.random.random() < probability:
            augmented_audio = self.add_noise(augmented_audio)

        # 5. 볼륨 변경 (50% 확률)
        if apply_volume and np.random.random() < probability:
            augmented_audio = self.change_volume(augmented_audio)

        return augmented_audio

    def augment_batch(
            self,
            audio_list: list,
            num_augmentations: int = 1,
            **kwargs
    ) -> Tuple[list, list]:
        """
        배치 증강 (여러 오디오 동시 처리)

        매개변수:
            audio_list: 오디오 리스트
            num_augmentations: 각 오디오당 생성할 증강 개수
            **kwargs: apply_random_augmentations에 전달할 인자

        반환값:
            augmented_list: 증강된 오디오 리스트
            labels_list: 대응하는 레이블 리스트 (원본과 동일)
        """
        augmented_list = []
        labels_list = []

        for idx, audio in enumerate(audio_list):
            # 원본 추가
            augmented_list.append(audio)
            labels_list.append(idx)

            # 증강 버전 추가
            for _ in range(num_augmentations):
                aug_audio = self.apply_random_augmentations(audio, **kwargs)
                augmented_list.append(aug_audio)
                labels_list.append(idx)  # 같은 레이블

        return augmented_list, labels_list


# ========== 사용 예시 (테스트용) ==========
if __name__ == "__main__":
    """
    AudioAugmentor 테스트 코드
    """
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("🎵 AudioAugmentor 테스트")
    print("=" * 60)

    # 증강기 생성
    augmentor = AudioAugmentor(sample_rate=22050)

    # 테스트용 더미 오디오 생성 (1초, 440Hz 사인파)
    duration = 1.0
    t = np.linspace(0, duration, int(augmentor.sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # A4 음 (라)

    print(f"\n✅ 원본 오디오 생성 완료")
    print(f"   - 길이: {len(audio)} 샘플")
    print(f"   - 지속시간: {len(audio) / augmentor.sample_rate:.2f}초")

    # 각 증강 기법 테스트
    print(f"\n📊 증강 기법 테스트:")

    # 1. 시간 이동
    shifted = augmentor.time_shift(audio)
    print(f"   ✅ 시간 이동: {len(shifted)} 샘플")

    # 2. 피치 이동
    pitched = augmentor.pitch_shift(audio, n_steps=2)
    print(f"   ✅ 피치 이동 (+2 반음): {len(pitched)} 샘플")

    # 3. 시간 신축
    stretched = augmentor.time_stretch(audio, rate=1.2)
    print(f"   ✅ 시간 신축 (1.2배 빠르게): {len(stretched)} 샘플")

    # 4. 잡음 추가
    noisy = augmentor.add_noise(audio)
    print(f"   ✅ 잡음 추가: {len(noisy)} 샘플")

    # 5. 볼륨 변경
    louder = augmentor.change_volume(audio, gain_db=3)
    print(f"   ✅ 볼륨 증가 (+3dB): {len(louder)} 샘플")

    # 6. 무작위 조합
    augmented = augmentor.apply_random_augmentations(audio, probability=0.5)
    print(f"   ✅ 무작위 증강: {len(augmented)} 샘플")

    # 시각화
    print(f"\n📊 증강 결과 시각화 중...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 원본
    axes[0, 0].plot(t[:1000], audio[:1000])
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # 시간 이동
    axes[0, 1].plot(t[:1000], shifted[:1000])
    axes[0, 1].set_title('Time Shifted', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 피치 이동
    axes[1, 0].plot(t[:1000], pitched[:1000])
    axes[1, 0].set_title('Pitch Shifted (+2 semitones)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)

    # 시간 신축
    axes[1, 1].plot(stretched[:1000])
    axes[1, 1].set_title('Time Stretched (1.2x)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # 잡음 추가
    axes[2, 0].plot(t[:1000], noisy[:1000])
    axes[2, 0].set_title('With Noise', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].grid(True, alpha=0.3)

    # 무작위 증강
    axes[2, 1].plot(augmented[:1000])
    axes[2, 1].set_title('Random Augmentations', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Samples')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('audio_augmentation_examples.png', dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 저장: audio_augmentation_examples.png")
    plt.show()

    print("\n" + "=" * 60)
    print("🎉 AudioAugmentor 테스트 완료!")
    print("=" * 60)
