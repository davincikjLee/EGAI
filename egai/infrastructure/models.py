"""
Deep Learning Models - 딥러닝 모델

TensorFlow/Keras 기반 CNN 모델
최적 모델: Simple CNN + CBAM (MAE 0.4819)
"""

from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class CBAM(layers.Layer):
    """
    CBAM (Convolutional Block Attention Module)

    채널 + 공간 어텐션으로 중요 특징 강조
    리버스 엔지니어링 결과: +14.5% 성능 향상
    """

    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        # 채널 어텐션 MLP
        self.channel_fc1 = layers.Dense(
            channels // self.reduction_ratio, activation="relu"
        )
        self.channel_fc2 = layers.Dense(channels)

        # 공간 어텐션 Conv
        self.spatial_conv = layers.Conv2D(
            1, kernel_size=7, padding="same", activation="sigmoid"
        )

        self.bn = layers.BatchNormalization()

    def call(self, x):
        # 채널 어텐션
        channel_att = self._channel_attention(x)
        x = x * channel_att

        # 공간 어텐션
        spatial_att = self._spatial_attention(x)
        x = x * spatial_att

        return self.bn(x)

    def _channel_attention(self, x):
        channels = x.shape[-1]

        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)

        avg_out = self.channel_fc2(
            self.channel_fc1(tf.reshape(avg_pool, [-1, channels]))
        )
        max_out = self.channel_fc2(
            self.channel_fc1(tf.reshape(max_pool, [-1, channels]))
        )

        att = tf.nn.sigmoid(avg_out + max_out)
        return tf.reshape(att, [-1, 1, 1, channels])

    def _spatial_attention(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return self.spatial_conv(concat)


class ModelFactory:
    """
    모델 팩토리

    사용 가능한 모델:
    - simple_cbam: Simple CNN + CBAM (최적, MAE 0.4819)
    - simple: Simple CNN (Baseline, MAE 0.5635)
    - channel_concat: 채널 결합 + CBAM
    """

    @staticmethod
    def create(
        model_type: str = "simple_cbam",
        input_shape: tuple = (128, 128, 1),
        num_outputs: int = 5,
    ) -> Model:
        """
        모델 생성

        Args:
            model_type: 모델 타입
            input_shape: 입력 형태 (H, W, C)
            num_outputs: 출력 개수 (5개 품질 점수)

        Returns:
            Keras Model
        """
        if model_type == "simple_cbam":
            return ModelFactory._build_simple_cbam(input_shape, num_outputs)
        elif model_type == "simple":
            return ModelFactory._build_simple_cnn(input_shape, num_outputs)
        elif model_type == "channel_concat":
            return ModelFactory._build_channel_concat(input_shape, num_outputs)
        elif model_type == "4channel_cbam":
            return ModelFactory._build_4channel_cbam(input_shape, num_outputs)
        elif model_type == "multihead_cbam":
            return ModelFactory._build_multihead_cbam(input_shape, num_outputs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _build_simple_cbam(
        input_shape: tuple, num_outputs: int
    ) -> Model:
        """
        Simple CNN + CBAM (최적 모델)

        MAE: 0.4819 (+14.5% 개선)
        파라미터: 455,853개
        """
        inputs = layers.Input(shape=input_shape, name="input")

        # Conv Block 1 + CBAM
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam1")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 2 + CBAM
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam2")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 3 + CBAM
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam3")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 4 + CBAM
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam4")(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Dense
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_outputs, activation="linear", name="output")(x)

        return Model(inputs=inputs, outputs=outputs, name="SimpleCBAM")

    @staticmethod
    def _build_simple_cnn(
        input_shape: tuple, num_outputs: int
    ) -> Model:
        """
        Simple CNN (Baseline)

        MAE: 0.5635
        """
        inputs = layers.Input(shape=input_shape, name="input")

        # Conv blocks without CBAM
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_outputs, activation="linear", name="output")(x)

        return Model(inputs=inputs, outputs=outputs, name="SimpleCNN")

    @staticmethod
    def _build_channel_concat(
        input_shape: tuple, num_outputs: int
    ) -> Model:
        """
        채널 결합 + CBAM

        Full + Percussive 스펙트로그램 채널 결합
        """
        input_full = layers.Input(shape=input_shape, name="input_full")
        input_percussive = layers.Input(shape=input_shape, name="input_percussive")

        # 채널 결합
        x = layers.Concatenate(axis=-1)([input_full, input_percussive])

        # Conv blocks with CBAM
        for i, filters in enumerate([32, 64, 128, 256], 1):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = CBAM(reduction_ratio=8, name=f"cbam{i}")(x)
            if i < 4:
                x = layers.MaxPooling2D(2)(x)
            else:
                x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_outputs, activation="linear", name="output")(x)

        return Model(
            inputs=[input_full, input_percussive],
            outputs=outputs,
            name="ChannelConcatCBAM",
        )

    @staticmethod
    def _build_4channel_cbam(
        input_shape: tuple, num_outputs: int
    ) -> Model:
        """
        4채널 CNN + CBAM (irregularity/regularity 개선용)

        입력 채널:
            1. Full Spectrogram - 주파수 특성
            2. Percussive Spectrogram - 충격음 성분
            3. Difference Spectrogram - 시간 변동 (급격한 변화)
            4. Variance Map - 불안정 영역

        input_shape: (128, 128, 4)
        """
        inputs = layers.Input(shape=input_shape, name="input_4ch")

        # Conv Block 1 + CBAM
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam1")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 2 + CBAM
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam2")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 3 + CBAM
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam3")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 4 + CBAM
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam4")(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Dense
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_outputs, activation="linear", name="output")(x)

        return Model(inputs=inputs, outputs=outputs, name="4ChannelCBAM")

    @staticmethod
    def _build_multihead_cbam(
        input_shape: tuple, num_outputs: int
    ) -> Model:
        """
        Multi-head CNN + CBAM

        공유 백본에서 특징 추출 후, 두 개의 전문화된 헤드로 분기:
        - Frequency Head: low_high_freq, mid_freq_score, audible_range_score (3개)
        - Regularity Head: regularity, irregularity (2개)

        4채널 입력 사용:
            1. Full Spectrogram - 주파수 특성
            2. Percussive Spectrogram - 충격음 성분
            3. Difference Spectrogram - 시간 변동
            4. Variance Map - 불안정 영역

        input_shape: (128, 128, 4)
        """
        inputs = layers.Input(shape=input_shape, name="input_4ch")

        # ========== Shared Backbone ==========
        # Conv Block 1 + CBAM
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam1")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 2 + CBAM
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam2")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 3 + CBAM
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam3")(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 4 + CBAM
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = CBAM(reduction_ratio=8, name="cbam4")(x)
        shared_features = layers.GlobalAveragePooling2D()(x)

        # ========== Frequency Head (3 outputs) ==========
        # low_high_freq, mid_freq_score, audible_range_score
        freq_x = layers.Dense(64, activation="relu", name="freq_dense1")(shared_features)
        freq_x = layers.Dropout(0.3)(freq_x)
        freq_x = layers.Dense(32, activation="relu", name="freq_dense2")(freq_x)
        freq_x = layers.Dropout(0.3)(freq_x)
        freq_output = layers.Dense(3, activation="linear", name="freq_output")(freq_x)

        # ========== Regularity Head (2 outputs) ==========
        # regularity, irregularity - 시간 변동 특징을 더 활용
        reg_x = layers.Dense(64, activation="relu", name="reg_dense1")(shared_features)
        reg_x = layers.Dropout(0.3)(reg_x)
        reg_x = layers.Dense(32, activation="relu", name="reg_dense2")(reg_x)
        reg_x = layers.Dropout(0.3)(reg_x)
        reg_output = layers.Dense(2, activation="linear", name="reg_output")(reg_x)

        # ========== Concatenate Outputs ==========
        # 순서: low_high_freq, mid_freq_score, audible_range_score, regularity, irregularity
        outputs = layers.Concatenate(name="output")([freq_output, reg_output])

        return Model(inputs=inputs, outputs=outputs, name="MultiHeadCBAM")

    @staticmethod
    def load(model_path: str) -> Model:
        """저장된 모델 로드"""
        return keras.models.load_model(
            model_path, custom_objects={"CBAM": CBAM}
        )
