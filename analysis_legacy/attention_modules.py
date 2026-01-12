"""
🎯 Attention 모듈
역할: 중요한 특징에 집중하도록 모델을 훈련
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ========== 10~11단계: CBAM ==========
class CBAM(layers.Layer):
    """
    CBAM (Convolutional Block Attention Module)

    🤔 CBAM이란?
    - 딥러닝 모델이 "어디를 봐야 하는지" 학습
    - 채널 어텐션: "어떤 특징이 중요한가?" (예: 저주파 vs 고주파)
    - 공간 어텐션: "스펙트로그램의 어느 위치가 중요한가?"

    비유:
    - 의사가 X-ray 사진에서 "이상한 부분"에 집중하는 것처럼
    - AI도 스펙트로그램에서 "고장 신호"에 집중
    """

    def __init__(self, reduction_ratio=8, **kwargs):
        """
        초기화

        매개변수:
            reduction_ratio: 채널 축소 비율 (메모리 절약)
        """
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        """
        레이어 구조 생성
        """
        # 입력 채널 수
        channels = input_shape[-1]

        # ========== 채널 어텐션 ==========
        # MLP (Multi-Layer Perceptron)로 채널 중요도 학습
        self.channel_fc1 = layers.Dense(
            channels // self.reduction_ratio,
            activation='relu',
            name='channel_fc1'
        )
        self.channel_fc2 = layers.Dense(
            channels,
            name='channel_fc2'
        )

        # ========== 공간 어텐션 ==========
        # 합성곱으로 공간 중요도 학습
        self.spatial_conv = layers.Conv2D(
            filters=1,  # 출력: 중요도 맵 (1채널)
            kernel_size=7,  # 7x7 필터 (큰 영역 고려)
            padding='same',
            activation='sigmoid',
            name='spatial_conv'
        )

        # 배치 정규화
        self.bn = layers.BatchNormalization()

    def channel_attention(self, x):
        """
        채널 어텐션

        🤔 무엇을 하나?
        - 각 채널(특징 맵)의 중요도를 0~1 사이 값으로 계산
        - 중요한 채널은 1에 가깝게, 덜 중요한 채널은 0에 가깝게

        과정:
        1. Global Average Pooling: 각 채널의 평균값 계산
        2. Global Max Pooling: 각 채널의 최대값 계산
        3. MLP로 처리
        4. 두 결과를 합쳐서 중요도 계산
        """
        # 입력 크기: (batch, height, width, channels)
        batch, height, width, channels = x.shape

        # 1. Global Average Pooling
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        # 크기: (batch, 1, 1, channels)

        # 2. Global Max Pooling
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        # 크기: (batch, 1, 1, channels)

        # 3. MLP 처리
        avg_out = self.channel_fc2(self.channel_fc1(
            tf.reshape(avg_pool, [-1, channels])
        ))
        max_out = self.channel_fc2(self.channel_fc1(
            tf.reshape(max_pool, [-1, channels])
        ))

        # 4. 합치기 + Sigmoid (0~1 사이로 변환)
        channel_att = tf.nn.sigmoid(avg_out + max_out)
        channel_att = tf.reshape(channel_att, [-1, 1, 1, channels])

        return channel_att

    def spatial_attention(self, x):
        """
        공간 어텐션

        🤔 무엇을 하나?
        - 스펙트로그램의 각 위치(시간-주파수)의 중요도 계산

        과정:
        1. 채널 방향으로 평균과 최대값 계산
        2. 합성곱으로 공간 중요도 맵 생성
        """
        # 1. 채널 방향 평균
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        # 크기: (batch, height, width, 1)

        # 2. 채널 방향 최대값
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        # 크기: (batch, height, width, 1)

        # 3. 결합 + 합성곱
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        # 크기: (batch, height, width, 2)

        spatial_att = self.spatial_conv(concat)
        # 크기: (batch, height, width, 1)

        return spatial_att

    def call(self, x):
        """
        순전파 (Forward Pass)

        과정:
        1. 채널 어텐션 적용
        2. 공간 어텐션 적용
        3. 배치 정규화
        """
        # 1. 채널 어텐션
        channel_att = self.channel_attention(x)
        x = x * channel_att  # 원소별 곱셈 (중요한 채널 강조)

        # 2. 공간 어텐션
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att  # 원소별 곱셈 (중요한 위치 강조)

        # 3. 배치 정규화
        x = self.bn(x)

        return x
