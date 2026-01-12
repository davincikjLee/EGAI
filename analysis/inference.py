"""
🔮 모델 추론 스크립트
역할: 학습된 모델로 새로운 음성 데이터 진단
"""

import numpy as np
import tensorflow as tf
from audio_preprocessing import AudioPreprocessor
from fault_diagnosis_model import FaultDiagnosisModel


class FaultDiagnosisInference:
    """
    고장 진단 추론 클래스
    """

    def __init__(self, model_path, class_names, target_shape=(128, 128)):
        """
        초기화

        매개변수:
            model_path: 저장된 모델 가중치 경로
            class_names: 클래스 이름 리스트 (예: ['정상', '고장', '기타'])
            target_shape: 스펙트로그램 크기
        """
        self.class_names = class_names
        self.target_shape = target_shape
        self.preprocessor = AudioPreprocessor()

        # 모델 로드
        self.model = FaultDiagnosisModel(num_classes=len(class_names))

        # 더미 입력으로 모델 빌드
        dummy_full = tf.random.normal((1, *target_shape, 1))
        dummy_perc = tf.random.normal((1, *target_shape, 1))
        _ = self.model([dummy_full, dummy_perc])

        # 가중치 로드
        self.model.load_weights(model_path)
        print(f"✅ 모델 로드 완료: {model_path}")


    def predict(self, audio_path):
        """
        음성 파일로 고장 진단

        매개변수:
            audio_path: 음성 파일 경로

        반환값:
            predicted_class: 예측된 클래스 이름
            probabilities: 각 클래스별 확률
        """
        print(f"\n🔍 진단 시작: {audio_path}")

        # 1. 전처리
        full_spec, perc_spec = self.preprocessor.preprocess(audio_path)

        # 2. 크기 조정
        full_spec = self._resize(full_spec)
        perc_spec = self._resize(perc_spec)

        # 3. 배치 차원 추가
        full_spec = full_spec[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
        perc_spec = perc_spec[np.newaxis, ..., np.newaxis]

        # 4. 예측
        probabilities = self.model([full_spec, perc_spec], training=False)
        probabilities = probabilities.numpy()[0]  # (num_classes,)

        # 5. 결과 해석
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx]

        # 6. 결과 출력
        print("\n📊 진단 결과:")
        print("-" * 40)
        for i, class_name in enumerate(self.class_names):
            bar = "█" * int(probabilities[i] * 50)
            print(f"{class_name:10s}: {probabilities[i]*100:5.2f}% {bar}")
        print("-" * 40)
        print(f"✅ 최종 진단: {predicted_class} (신뢰도: {confidence*100:.2f}%)")

        return predicted_class, probabilities


    def _resize(self, spec):
        """스펙트로그램 크기 조정"""
        spec = tf.image.resize(
            spec[..., np.newaxis],
            self.target_shape,
            method='bilinear'
        )
        return spec.numpy().squeeze()


# ========== 사용 예제 ==========
if __name__ == "__main__":
    # 추론 객체 생성
    inference = FaultDiagnosisInference(
        model_path='fault_diagnosis_model.h5',
        class_names=['정상', '베어링 고장', '기타 소음']
    )

    # 새로운 음성 파일 진단
    test_audio = "test_machine_sound.wav"
    predicted_class, probabilities = inference.predict(test_audio)
