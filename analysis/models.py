"""
모델 정의 모듈
역할: 3가지 모델 아키텍처 정의 (Simple CNN, Dual Input CBAM, Transfer Learning)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
try:
    from conv_blocks import ConvBlock
    from attention_modules import CBAM
except ImportError:
    from analysis.conv_blocks import ConvBlock
    from analysis.attention_modules import CBAM


def build_simple_cnn(input_shape=(128, 128, 1), num_outputs=4):
    """
    Simple CNN 모델 (Baseline)

    구조:
        Input → Conv×3 → GlobalAvgPool → Dense → Output

    매개변수:
        input_shape: 입력 형태 (H, W, C)
        num_outputs: 출력 개수 (타겟 점수 개수)

    반환값:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Conv Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output (Multi-output Regression)
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='SimpleCNN')

    return model


def build_dual_input_cbam(input_shape=(128, 128, 1), num_outputs=4):
    """
    Dual Input CNN + CBAM 모델

    구조:
        Full Spec → CNN → CBAM ─┐
                                ├→ Concat → Dense → Output
        Percussive → CNN → CBAM ┘

    매개변수:
        input_shape: 각 입력의 형태 (H, W, C)
        num_outputs: 출력 개수

    반환값:
        Keras Model
    """
    # 두 개의 입력
    input_full = layers.Input(shape=input_shape, name='input_full')
    input_percussive = layers.Input(shape=input_shape, name='input_percussive')

    def create_branch(x, name_prefix):
        """공유 구조의 CNN 브랜치"""
        # Conv Block 1
        x = ConvBlock(32, name=f'{name_prefix}_conv1')(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 2
        x = ConvBlock(64, name=f'{name_prefix}_conv2')(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv Block 3
        x = ConvBlock(128, name=f'{name_prefix}_conv3')(x)
        x = layers.MaxPooling2D(2)(x)

        # CBAM Attention
        x = CBAM(reduction_ratio=8, name=f'{name_prefix}_cbam')(x)

        # Conv Block 4
        x = ConvBlock(256, name=f'{name_prefix}_conv4')(x)
        x = layers.GlobalAveragePooling2D()(x)

        return x

    # 두 브랜치 처리
    branch_full = create_branch(input_full, 'full')
    branch_percussive = create_branch(input_percussive, 'percussive')

    # 결합
    merged = layers.Concatenate()([branch_full, branch_percussive])

    # Dense layers
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(
        inputs=[input_full, input_percussive],
        outputs=outputs,
        name='DualInputCBAM'
    )

    return model


def build_transfer_resnet(input_shape=(224, 224, 3), num_outputs=4):
    """
    Transfer Learning 모델 (ResNet50 기반)

    구조:
        Input → ResNet50 (pretrained) → Dense → Output

    매개변수:
        input_shape: 입력 형태 (224, 224, 3)
        num_outputs: 출력 개수

    반환값:
        Keras Model
    """
    # ResNet50 base (ImageNet pretrained)
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model layers (처음에는 고정)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name='input')

    # ResNet50 통과
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='TransferResNet50')

    return model


def unfreeze_resnet(model, unfreeze_layers=50):
    """
    ResNet50 모델의 일부 레이어 해제 (Fine-tuning용)

    매개변수:
        model: ResNet50 기반 모델
        unfreeze_layers: 해제할 레이어 수 (뒤에서부터)
    """
    # base_model 찾기
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    # 일부 레이어 해제
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    print(f"ResNet50: 마지막 {unfreeze_layers}개 레이어 학습 가능으로 변경")


def build_multiinput_metadata(audio_input_shape=(128, 128, 1), metadata_dim=12, num_outputs=5):
    """
    오디오 + 메타데이터 결합 모델

    구조:
        Full Spec → CNN → CBAM ─┐
                                ├→ Concat (512) ─┐
        Percussive → CNN → CBAM ┘               ├→ Concat (544) → Dense → Output
                                                │
        Metadata → Dense → Dense ───────────────┘ (32)

    매개변수:
        audio_input_shape: 오디오 스펙트로그램 형태 (H, W, C)
        metadata_dim: 메타데이터 특징 수
        num_outputs: 출력 개수

    반환값:
        Keras Model
    """
    # 오디오 입력
    input_full = layers.Input(shape=audio_input_shape, name='input_full')
    input_percussive = layers.Input(shape=audio_input_shape, name='input_percussive')

    # 메타데이터 입력
    input_metadata = layers.Input(shape=(metadata_dim,), name='input_metadata')

    def create_audio_branch(x, name_prefix):
        """오디오 CNN 브랜치"""
        x = ConvBlock(32, name=f'{name_prefix}_conv1')(x)
        x = layers.MaxPooling2D(2)(x)
        x = ConvBlock(64, name=f'{name_prefix}_conv2')(x)
        x = layers.MaxPooling2D(2)(x)
        x = ConvBlock(128, name=f'{name_prefix}_conv3')(x)
        x = layers.MaxPooling2D(2)(x)
        x = CBAM(reduction_ratio=8, name=f'{name_prefix}_cbam')(x)
        x = ConvBlock(256, name=f'{name_prefix}_conv4')(x)
        x = layers.GlobalAveragePooling2D()(x)
        return x

    # 오디오 브랜치 처리
    branch_full = create_audio_branch(input_full, 'full')
    branch_percussive = create_audio_branch(input_percussive, 'percussive')
    audio_merged = layers.Concatenate()([branch_full, branch_percussive])  # 512차원

    # 메타데이터 브랜치 처리
    meta = layers.Dense(64, activation='relu', name='meta_fc1')(input_metadata)
    meta = layers.BatchNormalization()(meta)
    meta = layers.Dropout(0.3)(meta)
    meta = layers.Dense(32, activation='relu', name='meta_fc2')(meta)  # 32차원

    # 오디오 + 메타데이터 결합
    merged = layers.Concatenate()([audio_merged, meta])  # 544차원

    # Dense layers
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(
        inputs=[input_full, input_percussive, input_metadata],
        outputs=outputs,
        name='MultiInputMetadata'
    )

    return model


def build_multiinput_physics(audio_input_shape=(128, 128, 1), physics_dim=14, num_outputs=5):
    """
    오디오 + 물리량 특징 결합 모델

    구조:
        Full Spec → CNN → CBAM ─┐
                                ├→ Concat (512) ─┐
        Percussive → CNN → CBAM ┘               ├→ Concat (576) → Dense → Output
                                                │
        Physics (14) → Dense → Dense ───────────┘ (64)

    매개변수:
        audio_input_shape: 오디오 스펙트로그램 형태 (H, W, C)
        physics_dim: 물리량 특징 수 (기본 14)
        num_outputs: 출력 개수

    반환값:
        Keras Model
    """
    # 오디오 입력
    input_full = layers.Input(shape=audio_input_shape, name='input_full')
    input_percussive = layers.Input(shape=audio_input_shape, name='input_percussive')

    # 물리량 특징 입력
    input_physics = layers.Input(shape=(physics_dim,), name='input_physics')

    def create_audio_branch(x, name_prefix):
        """오디오 CNN 브랜치"""
        x = ConvBlock(32, name=f'{name_prefix}_conv1')(x)
        x = layers.MaxPooling2D(2)(x)
        x = ConvBlock(64, name=f'{name_prefix}_conv2')(x)
        x = layers.MaxPooling2D(2)(x)
        x = ConvBlock(128, name=f'{name_prefix}_conv3')(x)
        x = layers.MaxPooling2D(2)(x)
        x = CBAM(reduction_ratio=8, name=f'{name_prefix}_cbam')(x)
        x = ConvBlock(256, name=f'{name_prefix}_conv4')(x)
        x = layers.GlobalAveragePooling2D()(x)
        return x

    # 오디오 브랜치 처리
    branch_full = create_audio_branch(input_full, 'full')
    branch_percussive = create_audio_branch(input_percussive, 'percussive')
    audio_merged = layers.Concatenate()([branch_full, branch_percussive])  # 512차원

    # 물리량 특징 브랜치 처리 (해석 가능한 특징에 더 큰 가중치)
    physics = layers.Dense(64, activation='relu', name='physics_fc1')(input_physics)
    physics = layers.BatchNormalization()(physics)
    physics = layers.Dropout(0.3)(physics)
    physics = layers.Dense(64, activation='relu', name='physics_fc2')(physics)  # 64차원

    # 오디오 + 물리량 결합
    merged = layers.Concatenate()([audio_merged, physics])  # 576차원

    # Dense layers
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(
        inputs=[input_full, input_percussive, input_physics],
        outputs=outputs,
        name='MultiInputPhysics'
    )

    return model


def build_simple_cbam(input_shape=(128, 128, 1), num_outputs=5):
    """
    Simple CNN + CBAM 모델 (논문 기반 개선)

    논문: "소음 데이터를 이용한 딥러닝 기반의 차량 진단 기술 개발" (현대자동차, 2023)
    - CBAM 추가로 회귀 태스크에서 78% → 86% (+8%) 향상

    구조:
        Input → Conv → BN → CBAM → Pool × 4 → GlobalAvgPool → Dense → Output

    매개변수:
        input_shape: 입력 형태 (H, W, C)
        num_outputs: 출력 개수 (타겟 점수 개수)

    반환값:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Conv Block 1 + CBAM
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam1')(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 2 + CBAM
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam2')(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 3 + CBAM
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam3')(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 4 + CBAM
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam4')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output (Multi-output Regression)
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='SimpleCBAM')

    return model


def build_channel_concat_cbam(input_shape=(128, 128, 1), num_outputs=5):
    """
    채널 결합 + CBAM 모델 (논문 Fig.6 회귀 모델 방식)

    논문: "소음 데이터를 이용한 딥러닝 기반의 차량 진단 기술 개발" (현대자동차, 2023)
    - 회귀 태스크에서는 두 스펙트로그램을 채널 축으로 결합하는 것이 더 효과적
    - "타음 성분에서 따로 특징을 추출하는 방법이 성능 향상을 가져오지 않는 것으로 분석"

    구조:
        [Full + Percussive] → 채널 결합 → Conv → BN → CBAM → Pool × 4~5 → Dense → Output

    매개변수:
        input_shape: 각 입력의 형태 (H, W, C)
        num_outputs: 출력 개수

    반환값:
        Keras Model
    """
    # 두 개의 입력
    input_full = layers.Input(shape=input_shape, name='input_full')
    input_percussive = layers.Input(shape=input_shape, name='input_percussive')

    # 채널 축으로 결합 (논문 방식)
    # (128, 128, 1) + (128, 128, 1) → (128, 128, 2)
    x = layers.Concatenate(axis=-1)([input_full, input_percussive])

    # Conv Block 1 + CBAM
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam1')(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 2 + CBAM
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam2')(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 3 + CBAM
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam3')(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 4 + CBAM
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam4')(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 5 + CBAM (논문처럼 더 깊게)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(reduction_ratio=8, name='cbam5')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output - Sigmoid × 5 후 스케일링 (논문 방식)
    # 논문: sigmoid × 5 → 0~5 실수 → 올림 → 1~5 정수
    # 우리는 회귀이므로 linear 유지하되, 필요시 sigmoid로 변경 가능
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(
        inputs=[input_full, input_percussive],
        outputs=outputs,
        name='ChannelConcatCBAM'
    )

    return model


def build_multiinput_modulation(audio_input_shape=(128, 128, 1), modulation_dim=20, num_outputs=5):
    """
    오디오 + 모듈레이션 특징 결합 모델 (regularity/irregularity 개선용)

    논문 참조: "소음 데이터를 이용한 딥러닝 기반의 차량 진단 기술 개발"
    - Modulation Spectrum: 음의 변동성 측정
    - Spectral Flux: 스펙트럼 변화량
    - Onset Detection: 충격음/이벤트 감지
    - Autocorrelation: 주기성 측정

    구조:
        Full Spec → CNN → CBAM ─┐
                                ├→ Concat (512) ─┐
        Percussive → CNN → CBAM ┘               ├→ Concat (576) → Dense → Output
                                                │
        Modulation (20) → Dense → Dense ────────┘ (64)

    매개변수:
        audio_input_shape: 오디오 스펙트로그램 형태 (H, W, C)
        modulation_dim: 모듈레이션 특징 수 (기본 20)
        num_outputs: 출력 개수

    반환값:
        Keras Model
    """
    # 오디오 입력
    input_full = layers.Input(shape=audio_input_shape, name='input_full')
    input_percussive = layers.Input(shape=audio_input_shape, name='input_percussive')

    # 모듈레이션 특징 입력
    input_modulation = layers.Input(shape=(modulation_dim,), name='input_modulation')

    def create_audio_branch(x, name_prefix):
        """오디오 CNN 브랜치"""
        x = ConvBlock(32, name=f'{name_prefix}_conv1')(x)
        x = layers.MaxPooling2D(2)(x)
        x = ConvBlock(64, name=f'{name_prefix}_conv2')(x)
        x = layers.MaxPooling2D(2)(x)
        x = ConvBlock(128, name=f'{name_prefix}_conv3')(x)
        x = layers.MaxPooling2D(2)(x)
        x = CBAM(reduction_ratio=8, name=f'{name_prefix}_cbam')(x)
        x = ConvBlock(256, name=f'{name_prefix}_conv4')(x)
        x = layers.GlobalAveragePooling2D()(x)
        return x

    # 오디오 브랜치 처리
    branch_full = create_audio_branch(input_full, 'full')
    branch_percussive = create_audio_branch(input_percussive, 'percussive')
    audio_merged = layers.Concatenate()([branch_full, branch_percussive])  # 512차원

    # 모듈레이션 특징 브랜치 처리
    # (regularity/irregularity 관련 특징에 더 큰 가중치)
    modulation = layers.Dense(64, activation='relu', name='modulation_fc1')(input_modulation)
    modulation = layers.BatchNormalization()(modulation)
    modulation = layers.Dropout(0.3)(modulation)
    modulation = layers.Dense(64, activation='relu', name='modulation_fc2')(modulation)  # 64차원

    # 오디오 + 모듈레이션 결합
    merged = layers.Concatenate()([audio_merged, modulation])  # 576차원

    # Dense layers
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output
    outputs = layers.Dense(num_outputs, activation='linear', name='output')(x)

    model = Model(
        inputs=[input_full, input_percussive, input_modulation],
        outputs=outputs,
        name='MultiInputModulation'
    )

    return model


def get_model(model_type, input_shape=None, num_outputs=4, metadata_dim=None, physics_dim=None, modulation_dim=None):
    """
    모델 타입에 따라 모델 반환

    매개변수:
        model_type: 'simple', 'simple_cbam', 'channel_concat', 'dual', 'resnet',
                   'multiinput_metadata', 'multiinput_physics', 'multiinput_modulation'
        input_shape: 입력 형태 (None이면 기본값)
        num_outputs: 출력 개수
        metadata_dim: 메타데이터 차원 (multiinput_metadata용)
        physics_dim: 물리량 특징 차원 (multiinput_physics용)
        modulation_dim: 모듈레이션 특징 차원 (multiinput_modulation용)

    반환값:
        Keras Model
    """
    if model_type == 'simple':
        shape = input_shape or (128, 128, 1)
        return build_simple_cnn(shape, num_outputs)

    elif model_type == 'simple_cbam':
        shape = input_shape or (128, 128, 1)
        return build_simple_cbam(shape, num_outputs)

    elif model_type == 'channel_concat':
        shape = input_shape or (128, 128, 1)
        return build_channel_concat_cbam(shape, num_outputs)

    elif model_type == 'dual':
        shape = input_shape or (128, 128, 1)
        return build_dual_input_cbam(shape, num_outputs)

    elif model_type == 'resnet':
        shape = input_shape or (224, 224, 3)
        return build_transfer_resnet(shape, num_outputs)

    elif model_type == 'multiinput_metadata':
        shape = input_shape or (128, 128, 1)
        meta_dim = metadata_dim or 12
        return build_multiinput_metadata(shape, meta_dim, num_outputs)

    elif model_type == 'multiinput_physics':
        shape = input_shape or (128, 128, 1)
        phys_dim = physics_dim or 14
        return build_multiinput_physics(shape, phys_dim, num_outputs)

    elif model_type == 'multiinput_modulation':
        shape = input_shape or (128, 128, 1)
        mod_dim = modulation_dim or 20
        return build_multiinput_modulation(shape, mod_dim, num_outputs)

    else:
        raise ValueError(f"지원하지 않는 model_type: {model_type}")


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("모델 테스트")
    print("=" * 60)

    # 1. Simple CNN
    print("\n1. Simple CNN")
    model_simple = build_simple_cnn()
    model_simple.summary()
    print(f"   파라미터: {model_simple.count_params():,}")

    # 2. Dual Input CBAM
    print("\n2. Dual Input CBAM")
    model_dual = build_dual_input_cbam()
    model_dual.summary()
    print(f"   파라미터: {model_dual.count_params():,}")

    # 3. Transfer ResNet50
    print("\n3. Transfer ResNet50")
    model_resnet = build_transfer_resnet()
    model_resnet.summary()
    print(f"   파라미터: {model_resnet.count_params():,}")

    # 더미 데이터로 테스트
    print("\n" + "=" * 60)
    print("더미 데이터 테스트")
    print("=" * 60)

    import numpy as np

    # Simple CNN
    dummy_input = np.random.randn(2, 128, 128, 1).astype(np.float32)
    output = model_simple.predict(dummy_input, verbose=0)
    print(f"\nSimple CNN 출력: {output.shape}")

    # Dual Input
    dummy_full = np.random.randn(2, 128, 128, 1).astype(np.float32)
    dummy_perc = np.random.randn(2, 128, 128, 1).astype(np.float32)
    output = model_dual.predict([dummy_full, dummy_perc], verbose=0)
    print(f"Dual Input 출력: {output.shape}")

    # ResNet
    dummy_rgb = np.random.randn(2, 224, 224, 3).astype(np.float32)
    output = model_resnet.predict(dummy_rgb, verbose=0)
    print(f"ResNet50 출력: {output.shape}")

    print("\n모델 테스트 완료!")
