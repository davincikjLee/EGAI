"""
🔲 합성곱 블록 모듈
역할: 이미지(스펙트로그램)에서 패턴을 추출하는 기본 단위
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ========== 8단계: 합성곱 블록 ==========
class ConvBlock(layers.Layer):
    """
    합성곱 블록: Conv2D + BatchNormalization + ReLU

    🤔 각 레이어의 역할은?

    1. Conv2D (합성곱):
       - 이미지에서 패턴 찾기 (예: 가장자리, 질감)
       - 비유: 돋보기로 이미지의 작은 부분씩 살펴보기

    2. BatchNormalization (배치 정규화):
       - 학습을 안정적으로 만들기
       - 비유: 학생들 시험 점수를 표준화 (평균 0, 분산 1)

    3. ReLU (활성화 함수):
       - 비선형성 추가 (복잡한 패턴 학습 가능)
       - 비유: 음수는 0으로, 양수는 그대로 (f(x) = max(0, x))
    """

    def __init__(self,
                 filters,  # 필터(채널) 개수
                 kernel_size=3,  # 필터 크기 (3x3)
                 strides=1,  # 이동 간격
                 padding='same',  # 패딩 방식
                 **kwargs):
        """
        초기화

        매개변수:
            filters: 출력 채널 수 (예: 64개 필터 → 64개 특징 맵)
            kernel_size: 필터 크기 (3 = 3x3 필터)
            strides: 필터 이동 간격
            padding: 'same' = 출력 크기 유지, 'valid' = 패딩 없음
        """
        super(ConvBlock, self).__init__(**kwargs)

        # 1. 합성곱 레이어
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,  # BatchNorm이 있어서 bias 불필요
            name=f'conv_{filters}'
        )

        # 2. 배치 정규화
        self.bn = layers.BatchNormalization(
            name=f'bn_{filters}'
        )

        # 3. ReLU 활성화 함수
        self.relu = layers.ReLU(
            name=f'relu_{filters}'
        )

    def call(self, x, training=None):
        """
        순전파

        매개변수:
            x: 입력 텐서 (batch, height, width, channels)
            training: 학습 모드인지 여부

        반환값:
            출력 텐서 (batch, new_height, new_width, filters)
        """
        # Conv2D 적용
        x = self.conv(x)
        # 예: (32, 128, 128, 3) → (32, 128, 128, 64)

        # BatchNormalization 적용
        x = self.bn(x, training=training)

        # ReLU 적용
        x = self.relu(x)

        return x


# ========== 시각화: 합성곱이 어떻게 작동하는가? ==========
"""
입력 이미지 (간단한 예시):
[
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

3x3 필터:
[
  [1, 0, -1],
  [1, 0, -1],
  [1, 0, -1]
]

합성곱 결과 (가장자리 검출):
중앙값 = (1×1 + 2×0 + 3×-1) + 
         (4×1 + 5×0 + 6×-1) + 
         (7×1 + 8×0 + 9×-1)
       = (1 + 0 - 3) + (4 + 0 - 6) + (7 + 0 - 9)
       = -2 - 2 - 2 = -6

→ ReLU 적용: max(0, -6) = 0
"""
