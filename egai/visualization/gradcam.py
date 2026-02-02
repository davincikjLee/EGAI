"""
Grad-CAM (Gradient-weighted Class Activation Mapping)

CNN 모델이 어떤 영역을 보고 판단했는지 시각화
- 스펙트로그램의 시간-주파수 영역별 중요도 표시
- 설명가능한 AI (XAI) 제공

참고: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import platform

# 한글 폰트 설정
def _setup_korean_font():
    """matplotlib 한글 폰트 설정"""
    import matplotlib.font_manager as fm

    if platform.system() == 'Windows':
        # Windows: 맑은 고딕
        font_candidates = ['Malgun Gothic', 'NanumGothic', 'Arial Unicode MS']
    else:
        # Mac/Linux
        font_candidates = ['AppleGothic', 'NanumGothic', 'DejaVu Sans']

    for font_name in font_candidates:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            continue

    # 폰트 못 찾으면 영문으로 fallback
    return False

_setup_korean_font()


class GradCAM:
    """
    Grad-CAM 시각화 클래스

    사용법:
        gradcam = GradCAM(model, layer_name="cbam4")
        heatmap = gradcam.compute_heatmap(input_tensor, class_idx=1)
        gradcam.visualize(input_tensor, heatmap, output_path="result.png")
    """

    def __init__(
        self,
        model: keras.Model,
        layer_name: str = "cbam4",
    ):
        """
        Args:
            model: Keras 모델 (CNN + CBAM)
            layer_name: Grad-CAM 대상 레이어 이름 (마지막 conv 레이어)
        """
        self.model = model
        self.layer_name = layer_name

        # Grad-CAM 모델 생성 (입력 → 대상 레이어 출력 + 모델 출력)
        self.grad_model = self._create_grad_model()

    def _create_grad_model(self) -> keras.Model:
        """Gradient 계산용 모델 생성"""
        # 대상 레이어 찾기
        target_layer = None
        for layer in self.model.layers:
            if layer.name == self.layer_name:
                target_layer = layer
                break

        if target_layer is None:
            # CBAM 레이어는 중첩되어 있으므로 재귀 탐색
            target_layer = self._find_layer_recursive(
                self.model, self.layer_name
            )

        if target_layer is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in model")

        # Grad-CAM 모델: 입력 → [대상 레이어 출력, 모델 출력]
        return keras.Model(
            inputs=self.model.input,
            outputs=[target_layer.output, self.model.output]
        )

    def _find_layer_recursive(
        self, model: keras.Model, layer_name: str
    ) -> Optional[keras.layers.Layer]:
        """재귀적으로 레이어 탐색 (중첩 모델 지원)"""
        for layer in model.layers:
            if layer.name == layer_name:
                return layer
            if hasattr(layer, 'layers'):
                found = self._find_layer_recursive(layer, layer_name)
                if found:
                    return found
        return None

    def compute_heatmap(
        self,
        input_tensor: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Grad-CAM 히트맵 계산

        Args:
            input_tensor: 입력 (batch, H, W, C) 또는 (H, W, C)
            class_idx: 대상 클래스 인덱스 (None이면 예측 클래스)

        Returns:
            heatmap: (H, W) 정규화된 히트맵 [0, 1]
        """
        # 배치 차원 추가
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        input_tensor = tf.cast(input_tensor, tf.float32)

        # Gradient 계산
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_outputs, predictions = self.grad_model(input_tensor)

            # 이진 분류의 경우
            if predictions.shape[-1] == 1:
                if class_idx == 0:
                    loss = 1 - predictions[:, 0]  # 클래스 0 (가솔린/OK)
                else:
                    loss = predictions[:, 0]  # 클래스 1 (디젤/NG)
            else:
                # 다중 클래스
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]

        # Gradient 계산
        grads = tape.gradient(loss, conv_outputs)

        # Global Average Pooling (채널별 중요도)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 가중 합산
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads, axis=-1
        )

        # ReLU + 정규화
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

        return heatmap.numpy()

    def visualize(
        self,
        input_tensor: np.ndarray,
        heatmap: np.ndarray,
        output_path: Optional[str] = None,
        channel_idx: int = 0,
        title: str = "Grad-CAM Visualization",
        alpha: float = 0.4,
        figsize: Tuple[int, int] = (12, 5),
    ) -> Optional[plt.Figure]:
        """
        Grad-CAM 히트맵 시각화

        Args:
            input_tensor: 입력 스펙트로그램 (H, W, C) 또는 (batch, H, W, C)
            heatmap: Grad-CAM 히트맵 (H', W')
            output_path: 저장 경로 (None이면 표시만)
            channel_idx: 시각화할 입력 채널 (0=Full, 1=Percussive, 2=Diff, 3=Var)
            title: 그래프 제목
            alpha: 히트맵 투명도
            figsize: 그래프 크기

        Returns:
            matplotlib Figure (output_path가 None인 경우)
        """
        # 배치 차원 제거
        if len(input_tensor.shape) == 4:
            input_tensor = input_tensor[0]

        # 히트맵 리사이즈 (입력 크기에 맞춤)
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            (input_tensor.shape[0], input_tensor.shape[1])
        ).numpy().squeeze()

        # Figure 생성
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        channel_names = ["Full", "Percussive", "Difference", "Variance"]

        # 1. 원본 스펙트로그램
        ax = axes[0]
        spec = input_tensor[:, :, channel_idx]
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Input ({channel_names[channel_idx]})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 2. Grad-CAM 히트맵
        ax = axes[1]
        im = ax.imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet')
        ax.set_title("Grad-CAM Heatmap")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 3. 오버레이
        ax = axes[2]
        ax.imshow(spec, aspect='auto', origin='lower', cmap='gray')
        ax.imshow(heatmap_resized, aspect='auto', origin='lower',
                  cmap='jet', alpha=alpha)
        ax.set_title("Overlay (Important Regions)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {output_path}")
            return None
        else:
            return fig

    def visualize_comparison(
        self,
        samples: List[Tuple[np.ndarray, str, int]],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 8),
    ) -> Optional[plt.Figure]:
        """
        여러 샘플 비교 시각화

        Args:
            samples: [(input_tensor, label, class_idx), ...] 리스트
            output_path: 저장 경로
            figsize: 그래프 크기

        Returns:
            matplotlib Figure
        """
        n_samples = len(samples)
        fig, axes = plt.subplots(2, n_samples, figsize=figsize)

        if n_samples == 1:
            axes = axes.reshape(2, 1)

        for i, (input_tensor, label, class_idx) in enumerate(samples):
            # 배치 차원 추가/제거
            if len(input_tensor.shape) == 3:
                input_batch = np.expand_dims(input_tensor, 0)
            else:
                input_batch = input_tensor
                input_tensor = input_tensor[0]

            # 히트맵 계산
            heatmap = self.compute_heatmap(input_batch, class_idx)
            heatmap_resized = tf.image.resize(
                heatmap[..., np.newaxis],
                (input_tensor.shape[0], input_tensor.shape[1])
            ).numpy().squeeze()

            # 원본 스펙트로그램 (첫 번째 채널)
            spec = input_tensor[:, :, 0]

            # 상단: 원본
            axes[0, i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[0, i].set_title(f"{label}\n(Input)", fontsize=10)
            axes[0, i].set_ylabel("Frequency")

            # 하단: 오버레이
            axes[1, i].imshow(spec, aspect='auto', origin='lower', cmap='gray')
            axes[1, i].imshow(heatmap_resized, aspect='auto', origin='lower',
                             cmap='jet', alpha=0.5)
            axes[1, i].set_title("Attention", fontsize=10)
            axes[1, i].set_xlabel("Time")
            axes[1, i].set_ylabel("Frequency")

        plt.suptitle("Grad-CAM: Model Attention Comparison",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {output_path}")
            return None
        else:
            return fig


def create_gradcam_for_binary_model(model_path: str) -> GradCAM:
    """
    이진 분류 모델용 Grad-CAM 생성

    Args:
        model_path: BinaryCBAM 모델 경로

    Returns:
        GradCAM 인스턴스
    """
    from egai.infrastructure.models import ModelFactory, CBAM

    model = keras.models.load_model(
        model_path,
        custom_objects={"CBAM": CBAM}
    )

    return GradCAM(model, layer_name="cbam4")


def create_gradcam_for_regression_model(model_path: str) -> GradCAM:
    """
    회귀 모델용 Grad-CAM 생성

    Args:
        model_path: 4ChannelCBAM 모델 경로

    Returns:
        GradCAM 인스턴스
    """
    from egai.infrastructure.models import ModelFactory

    model = ModelFactory.load(model_path)

    return GradCAM(model, layer_name="cbam4")


def generate_gradcam_report(
    filename: str,
    image_path: str,
    heatmap: np.ndarray,
    output_path: str,
    audio_duration: float = 10.0,
) -> str:
    """
    Grad-CAM 분석 보고서 자동 생성

    Args:
        filename: 분석 파일명
        image_path: Grad-CAM 이미지 경로 (상대 경로)
        heatmap: Grad-CAM 히트맵
        output_path: 보고서 저장 경로
        audio_duration: 오디오 길이

    Returns:
        보고서 경로
    """
    from datetime import datetime

    # 히트맵 분석
    h, w = heatmap.shape
    hotspot_y, hotspot_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    # 시간/주파수 구간 추정
    time_region = f"{int(hotspot_x - w*0.15)}~{int(hotspot_x + w*0.15)} 프레임"
    freq_region = f"{int(hotspot_y - h*0.2)}~{int(hotspot_y + h*0.2)} 대역"

    report = f"""# Grad-CAM 분석 보고서

> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> 분석 파일: {filename}

---

## 1. 분석 개요

### 목적
CNN 모델이 엔진 사운드 스펙트로그램에서 **어떤 시간-주파수 영역을 보고 품질을 판단했는지** 시각화합니다.

### 분석 대상
| 항목 | 값 |
|------|-----|
| 파일명 | {filename} |
| 오디오 길이 | 약 {audio_duration:.1f}초 |
| 분석 구간 | 중앙 10초 |
| 모델 | CNN + CBAM (4채널) |

---

## 2. 시각화 결과

![Grad-CAM 시각화]({image_path})

### 패널 설명

| 패널 | 이름 | 설명 |
|------|------|------|
| 왼쪽 | **Input (Full)** | 원본 스펙트로그램 - X축: 시간, Y축: 주파수 |
| 가운데 | **Grad-CAM Heatmap** | 모델 주목 영역 - 빨강: 높은 중요도, 파랑: 낮은 중요도 |
| 오른쪽 | **Overlay** | 원본 위에 히트맵 오버레이 - 중요 영역 강조 |

---

## 3. 분석 결과 해석

### 주요 관찰 사항

**시간 축 (X축) 분석:**
- 모델이 주목한 시간 구간: **{time_region}**
- 해석: 해당 시간 구간의 음향 패턴이 품질 판단에 주요하게 작용

**주파수 축 (Y축) 분석:**
- 모델이 주목한 주파수 대역: **{freq_region}**
- 해석: 이 주파수 대역의 에너지 분포가 엔진 상태 판별에 활용됨

### 품질 판단 근거

```
모델이 판단에 사용한 주요 특징:
1. 주요 관심 영역의 에너지 패턴
2. 시간에 따른 주파수 변화의 일관성
3. 특정 주파수 대역의 강도 분포
```

---

## 4. 기술적 배경

### Grad-CAM이란?

**Grad-CAM (Gradient-weighted Class Activation Mapping)**은 CNN 모델의 판단 근거를 시각화하는 설명가능한 AI(XAI) 기법입니다.

**해석 방법:**
- **빨간색/노란색 영역**: 모델이 판단에 크게 의존한 영역
- **파란색/초록색 영역**: 판단에 덜 중요한 영역

### 스펙트로그램 채널 설명

| 채널 | 이름 | 포착하는 특성 |
|------|------|--------------|
| 1 | Full | 전체 주파수 스펙트럼 |
| 2 | Percussive | 충격음, 노킹 소리 |
| 3 | Difference | 시간에 따른 급격한 변화 |
| 4 | Variance | 불안정한 영역, 떨림 |

---

## 5. 결론

모델은 **{freq_region} 주파수 대역**의 **{time_region}** 구간을 주로 분석하여 품질 점수를 예측했습니다.

---

*이 보고서는 EGAI 시스템에 의해 자동 생성되었습니다.*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved: {output_path}")
    return output_path


class VAEVisualizer:
    """
    VAE 재구성 오차 시각화

    VAE는 CNN과 달리 Grad-CAM을 직접 적용할 수 없으므로,
    재구성 오차 맵을 통해 이상 영역을 시각화합니다.
    """

    def __init__(self, encoder, decoder):
        """
        Args:
            encoder: VAE 인코더 모델
            decoder: VAE 디코더 모델
        """
        self.encoder = encoder
        self.decoder = decoder

    def compute_reconstruction_error(
        self,
        input_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        재구성 오차 계산

        Args:
            input_tensor: 입력 (H, W, C) 또는 (batch, H, W, C)

        Returns:
            reconstruction: 재구성된 스펙트로그램
            error_map: 픽셀별 오차 맵
            total_score: VAE 총 점수
        """
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        # 인코딩
        z_mean, z_log_var, z = self.encoder.predict(input_tensor, verbose=0)

        # 디코딩 (재구성)
        reconstruction = self.decoder.predict(z, verbose=0)

        # 오차 맵 계산 (채널별 평균)
        error_map = np.mean((input_tensor - reconstruction) ** 2, axis=-1)[0]

        # VAE 점수 계산
        mse = np.mean((input_tensor - reconstruction) ** 2)
        reconstruction_error = mse * 128 * 128 * 4
        kl_div = -0.5 * np.mean(1 + z_log_var - z_mean**2 - np.exp(z_log_var))
        total_score = reconstruction_error + 0.5 * kl_div

        return reconstruction[0], error_map, float(total_score)

    def visualize(
        self,
        input_tensor: np.ndarray,
        output_path: Optional[str] = None,
        title: str = "VAE Reconstruction Analysis",
        channel_idx: int = 0,
        figsize: Tuple[int, int] = (12, 5),
    ) -> Optional[plt.Figure]:
        """
        VAE 재구성 오차 시각화

        Args:
            input_tensor: 입력 스펙트로그램
            output_path: 저장 경로
            title: 제목
            channel_idx: 시각화할 채널
            figsize: 그래프 크기

        Returns:
            Figure 또는 None
        """
        if len(input_tensor.shape) == 4:
            input_tensor = input_tensor[0]

        reconstruction, error_map, vae_score = self.compute_reconstruction_error(
            input_tensor
        )

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        channel_names = ["Full", "Percussive", "Difference", "Variance"]

        # 1. 원본
        ax = axes[0]
        spec = input_tensor[:, :, channel_idx]
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Input ({channel_names[channel_idx]})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 2. 재구성
        ax = axes[1]
        recon_spec = reconstruction[:, :, channel_idx]
        im = ax.imshow(recon_spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title("VAE Reconstruction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 3. 오차 맵 (정규화)
        ax = axes[2]
        error_normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-10)
        im = ax.imshow(error_normalized, aspect='auto', origin='lower', cmap='hot')
        ax.set_title(f"Error Map (Score: {vae_score:.1f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {output_path}")
            return None
        else:
            return fig


def create_vae_visualizer(vae_dir: str) -> VAEVisualizer:
    """
    VAE 시각화 도구 생성

    Args:
        vae_dir: VAE 모델 디렉토리 (encoder/decoder 포함)

    Returns:
        VAEVisualizer 인스턴스
    """
    from egai.infrastructure.models import Sampling

    vae_path = Path(vae_dir)
    encoder = tf.keras.models.load_model(
        vae_path / "vae_encoder.keras",
        custom_objects={"Sampling": Sampling}
    )
    decoder = tf.keras.models.load_model(vae_path / "vae_decoder.keras")

    return VAEVisualizer(encoder, decoder)


def generate_unified_visualization(
    audio_path: str,
    regression_model_path: str,
    binary_model_path: Optional[str] = None,
    vae_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    generate_report: bool = True,
) -> dict:
    """
    3개 모델의 통합 시각화 생성

    Args:
        audio_path: 오디오 파일 경로
        regression_model_path: 회귀 모델 경로
        binary_model_path: 이진 분류 모델 경로 (Optional)
        vae_dir: VAE 모델 디렉토리 (Optional)
        output_path: 출력 이미지 경로
        generate_report: 보고서 생성 여부

    Returns:
        결과 딕셔너리
    """
    import librosa
    from egai.preprocessing.audio import AudioPreprocessor

    audio_path = Path(audio_path)

    # 오디오 로드 및 스펙트로그램 생성
    print(f"[1/4] Loading audio: {audio_path.name}")
    audio, sr = librosa.load(str(audio_path), sr=22050)
    audio_duration = len(audio) / sr

    if audio_duration < 10:
        raise ValueError(f"Audio too short: {audio_duration:.1f}s (min 10s)")

    # 중앙 10초 추출
    center = len(audio) // 2
    half_segment = 5 * sr
    segment = audio[center - half_segment:center + half_segment]

    preprocessor = AudioPreprocessor()
    spectrogram = preprocessor.compute_4channel_spectrogram(segment)
    input_tensor = np.expand_dims(spectrogram, axis=0)

    results = {
        "file": audio_path.name,
        "duration": audio_duration,
        "models": {}
    }

    # 서브플롯 개수 결정
    n_models = 1  # 회귀 모델은 필수
    if binary_model_path and Path(binary_model_path).exists():
        n_models += 1
    if vae_dir and Path(vae_dir).exists():
        n_models += 1

    fig, axes = plt.subplots(n_models, 3, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    row_idx = 0
    channel_names = ["Full", "Percussive", "Difference", "Variance"]

    # 1. Regression Model Grad-CAM
    print("[2/4] Computing Regression Grad-CAM...")
    try:
        gradcam_reg = create_gradcam_for_regression_model(regression_model_path)
        heatmap_reg = gradcam_reg.compute_heatmap(input_tensor, class_idx=0)
        heatmap_resized = tf.image.resize(
            heatmap_reg[..., np.newaxis],
            (spectrogram.shape[0], spectrogram.shape[1])
        ).numpy().squeeze()

        spec = spectrogram[:, :, 0]

        # 원본
        im = axes[row_idx, 0].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axes[row_idx, 0].set_title(f"[Regression] Input ({channel_names[0]})")
        axes[row_idx, 0].set_ylabel("Frequency")
        plt.colorbar(im, ax=axes[row_idx, 0], fraction=0.046)

        # 히트맵
        im = axes[row_idx, 1].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet')
        axes[row_idx, 1].set_title("[Regression] Grad-CAM")
        plt.colorbar(im, ax=axes[row_idx, 1], fraction=0.046)

        # 오버레이
        axes[row_idx, 2].imshow(spec, aspect='auto', origin='lower', cmap='gray')
        axes[row_idx, 2].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.4)
        axes[row_idx, 2].set_title("[Regression] Overlay")

        results["models"]["regression"] = {"status": "success", "heatmap_shape": heatmap_reg.shape}
        row_idx += 1
    except Exception as e:
        results["models"]["regression"] = {"status": "error", "message": str(e)}
        print(f"  Error: {e}")

    # 2. Binary Classification Grad-CAM
    if binary_model_path and Path(binary_model_path).exists():
        print("[3/4] Computing Binary Classification Grad-CAM...")
        try:
            gradcam_bin = create_gradcam_for_binary_model(binary_model_path)
            heatmap_bin = gradcam_bin.compute_heatmap(input_tensor, class_idx=1)
            heatmap_resized = tf.image.resize(
                heatmap_bin[..., np.newaxis],
                (spectrogram.shape[0], spectrogram.shape[1])
            ).numpy().squeeze()

            # 원본
            im = axes[row_idx, 0].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[row_idx, 0].set_title(f"[Binary] Input ({channel_names[0]})")
            axes[row_idx, 0].set_ylabel("Frequency")
            plt.colorbar(im, ax=axes[row_idx, 0], fraction=0.046)

            # 히트맵
            im = axes[row_idx, 1].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet')
            axes[row_idx, 1].set_title("[Binary] Grad-CAM")
            plt.colorbar(im, ax=axes[row_idx, 1], fraction=0.046)

            # 오버레이
            axes[row_idx, 2].imshow(spec, aspect='auto', origin='lower', cmap='gray')
            axes[row_idx, 2].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.4)
            axes[row_idx, 2].set_title("[Binary] Overlay")

            results["models"]["binary"] = {"status": "success", "heatmap_shape": heatmap_bin.shape}
            row_idx += 1
        except Exception as e:
            results["models"]["binary"] = {"status": "error", "message": str(e)}
            print(f"  Error: {e}")
    else:
        print("[3/4] Binary model not found, skipping...")

    # 3. VAE Reconstruction Error
    if vae_dir and Path(vae_dir).exists():
        print("[4/4] Computing VAE Reconstruction Error...")
        try:
            vae_viz = create_vae_visualizer(vae_dir)
            reconstruction, error_map, vae_score = vae_viz.compute_reconstruction_error(spectrogram)

            error_normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-10)

            # 원본
            im = axes[row_idx, 0].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[row_idx, 0].set_title(f"[VAE] Input ({channel_names[0]})")
            axes[row_idx, 0].set_ylabel("Frequency")
            axes[row_idx, 0].set_xlabel("Time")
            plt.colorbar(im, ax=axes[row_idx, 0], fraction=0.046)

            # 재구성
            im = axes[row_idx, 1].imshow(reconstruction[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[row_idx, 1].set_title("[VAE] Reconstruction")
            axes[row_idx, 1].set_xlabel("Time")
            plt.colorbar(im, ax=axes[row_idx, 1], fraction=0.046)

            # 오차 맵
            im = axes[row_idx, 2].imshow(error_normalized, aspect='auto', origin='lower', cmap='hot')
            axes[row_idx, 2].set_title(f"[VAE] Error Map (Score: {vae_score:.1f})")
            axes[row_idx, 2].set_xlabel("Time")
            plt.colorbar(im, ax=axes[row_idx, 2], fraction=0.046)

            results["models"]["vae"] = {"status": "success", "score": vae_score}
        except Exception as e:
            results["models"]["vae"] = {"status": "error", "message": str(e)}
            print(f"  Error: {e}")
    else:
        print("[4/4] VAE model not found, skipping...")

    plt.suptitle(f"Unified Model Visualization: {audio_path.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 저장
    if output_path is None:
        output_path = audio_path.with_suffix('.unified_gradcam.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved unified visualization: {output_path}")

    results["output_image"] = str(output_path)

    # 보고서 생성
    if generate_report:
        report_path = Path(output_path).with_suffix('.md')
        _generate_unified_report(results, str(report_path), str(output_path))
        results["output_report"] = str(report_path)

    return results


def _generate_unified_report(
    results: dict,
    report_path: str,
    image_path: str,
) -> str:
    """통합 보고서 생성"""
    from datetime import datetime

    image_filename = Path(image_path).name

    # 모델별 상태 텍스트
    model_status = []
    for model_name, info in results.get("models", {}).items():
        if info.get("status") == "success":
            if model_name == "vae":
                model_status.append(f"- **{model_name.upper()}**: Score {info.get('score', 'N/A'):.1f}")
            else:
                model_status.append(f"- **{model_name.upper()}**: 분석 완료")
        else:
            model_status.append(f"- **{model_name.upper()}**: 오류 - {info.get('message', 'Unknown')}")

    model_status_text = "\n".join(model_status) if model_status else "- 모델 정보 없음"

    report = f"""# 통합 XAI 분석 보고서

> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> 분석 파일: {results.get('file', 'Unknown')}

---

## 1. 분석 개요

### 목적
3개의 딥러닝 모델이 엔진 사운드를 **어떻게 분석하는지** 시각화합니다:
- **Regression Model**: 품질 점수 예측 시 주목하는 영역
- **Binary Model**: 가솔린/디젤 분류 시 주목하는 영역
- **VAE Model**: 재구성 오차로 이상 영역 탐지

### 분석 대상
| 항목 | 값 |
|------|-----|
| 파일명 | {results.get('file', 'Unknown')} |
| 오디오 길이 | 약 {results.get('duration', 0):.1f}초 |
| 분석 구간 | 중앙 10초 |

### 모델 상태
{model_status_text}

---

## 2. 시각화 결과

![Unified Visualization]({image_filename})

---

## 3. 모델별 해석 가이드

### Regression Model (품질 점수)
- **빨간색 영역**: 품질 점수 계산에 가장 큰 영향을 주는 영역
- **해석**: 이 영역의 특성이 좋으면 높은 점수, 이상하면 낮은 점수

### Binary Model (가솔린/디젤)
- **빨간색 영역**: 연료 유형 구분에 핵심적인 주파수 대역
- **해석**: 가솔린과 디젤 엔진의 음향적 차이가 나타나는 영역

### VAE Model (이상탐지)
- **밝은 영역 (Error Map)**: 정상 패턴과 다른 이상 영역
- **해석**: 재구성이 어려운 영역 = 학습 데이터와 다른 비정상 패턴
- **Score**: 낮을수록 정상, 높을수록 이상

---

## 4. 기술적 배경

### Grad-CAM vs VAE Reconstruction Error

| 기법 | 적용 모델 | 원리 |
|------|----------|------|
| **Grad-CAM** | CNN (Regression, Binary) | 그래디언트 기반 관심 영역 시각화 |
| **Reconstruction Error** | VAE | 입력-출력 차이로 이상 탐지 |

### 스펙트로그램 4채널

| 채널 | 포착하는 특성 |
|------|--------------|
| Full | 전체 주파수 스펙트럼 |
| Percussive | 충격음, 노킹 |
| Difference | 급격한 시간 변화 |
| Variance | 불안정한 영역 |

---

## 5. 결론

이 보고서는 세 가지 관점에서 엔진 사운드를 분석합니다:
1. **품질 관점**: 어떤 특성이 품질 점수에 영향을 주는가
2. **분류 관점**: 연료 유형 구분의 핵심 특성은 무엇인가
3. **이상탐지 관점**: 정상 범위를 벗어나는 영역이 있는가

---

*이 보고서는 EGAI 시스템에 의해 자동 생성되었습니다.*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved: {report_path}")
    return report_path
