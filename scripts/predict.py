"""
EGAI 엔진 품질 예측 스크립트

MP3 또는 M4A 엔진 오디오 파일을 분석하여 품질 점수를 예측합니다.

사용법:
    py -3.12 scripts/predict.py 파일경로.mp3
    py -3.12 scripts/predict.py 파일경로.m4a
    py -3.12 scripts/predict.py sample/     # 폴더 내 모든 파일
    py -3.12 scripts/predict.py 파일.m4a --visualize  # Grad-CAM 시각화 포함

예시:
    py -3.12 scripts/predict.py sample/제네시스_엔진음.m4a
    py -3.12 scripts/predict.py sample/제네시스_엔진음.m4a --visualize
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# FFmpeg 경로 자동 설정 (M4A 지원)
try:
    import imageio_ffmpeg
    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
    os.environ["PATH"] = str(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass  # imageio-ffmpeg 없으면 시스템 FFmpeg 사용


def print_result_box(title: str, content: dict):
    """결과를 박스 형태로 출력"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    for key, value in content.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 50)


def predict_single_file(audio_path: Path, visualize: bool = False):
    """단일 파일 예측"""
    import numpy as np
    import librosa

    print(f"\n파일: {audio_path.name}")
    print("-" * 50)

    # 1. 오디오 로드
    print("[1/4] 오디오 로드 중...")
    try:
        audio, sr = librosa.load(str(audio_path), sr=22050)
        duration = len(audio) / sr
        print(f"      길이: {duration:.1f}초")
    except Exception as e:
        print(f"      오류: {e}")
        return None

    if duration < 10:
        print("      오류: 오디오가 10초 미만입니다.")
        return None

    # 2. 스펙트로그램 생성
    print("[2/4] 스펙트로그램 생성 중...")
    from egai.preprocessing.audio import AudioPreprocessor
    preprocessor = AudioPreprocessor()

    # 중앙 10초 추출
    center = len(audio) // 2
    half_segment = 5 * sr  # 5초
    segment = audio[center - half_segment:center + half_segment]
    spectrogram = preprocessor.compute_4channel_spectrogram(segment)

    # 3. Regression 모델 예측
    print("[3/4] 품질 점수 예측 중...")
    from egai.infrastructure.models import ModelFactory

    model_path = project_root / "experiments/best_model_4channel_cbam_1.keras"
    model = ModelFactory.load(str(model_path))

    x = np.expand_dims(spectrogram, axis=0)
    predictions = model.predict(x, verbose=0)[0]

    scores = {
        "저/고주파 균형": predictions[0],
        "중주파 품질": predictions[1],
        "가청범위 상태": predictions[2],
        "규칙성": predictions[3],
        "불규칙성 없음": predictions[4],
    }
    overall = sum(predictions) / 5

    # 4. VAE 이상탐지
    print("[4/4] 이상탐지 분석 중...")
    import tensorflow as tf
    from egai.infrastructure.models import Sampling

    vae_dir = project_root / "experiments/20260124_013112_530a70"
    encoder = tf.keras.models.load_model(
        vae_dir / "vae_encoder.keras",
        custom_objects={"Sampling": Sampling}
    )
    decoder = tf.keras.models.load_model(vae_dir / "vae_decoder.keras")

    z_mean, z_log_var, z = encoder.predict(x, verbose=0)
    reconstruction = decoder.predict(z, verbose=0)

    mse = np.mean((x - reconstruction) ** 2)
    reconstruction_error = mse * 128 * 128 * 4
    kl_div = -0.5 * np.mean(1 + z_log_var - z_mean**2 - np.exp(z_log_var))
    vae_score = reconstruction_error + 0.5 * kl_div

    # 판정
    if overall >= 4.0 and vae_score < 200:
        status = "양호"
        status_icon = "[OK]"
    elif overall >= 3.5:
        status = "보통"
        status_icon = "[--]"
    else:
        status = "점검권장"
        status_icon = "[NG]"

    # 결과 출력
    print_result_box(f"{status_icon} 분석 결과: {status}", {
        "종합 점수": overall,
        "저/고주파 균형": scores["저/고주파 균형"],
        "중주파 품질": scores["중주파 품질"],
        "가청범위 상태": scores["가청범위 상태"],
        "규칙성": scores["규칙성"],
        "불규칙성 없음": scores["불규칙성 없음"],
        "VAE 점수": f"{vae_score:.1f} (낮을수록 정상)",
    })

    # Grad-CAM 시각화 (옵션)
    if visualize:
        print("\n[Grad-CAM] 시각화 생성 중...")
        try:
            from egai.visualization.gradcam import create_gradcam_for_regression_model

            gradcam = create_gradcam_for_regression_model(str(model_path))
            heatmap = gradcam.compute_heatmap(x, class_idx=0)

            output_path = audio_path.with_suffix('.gradcam.png')
            gradcam.visualize(
                spectrogram,
                heatmap,
                output_path=str(output_path),
                title=f"Grad-CAM: {audio_path.name}",
            )
            print(f"[Grad-CAM] 저장 완료: {output_path}")
        except Exception as e:
            print(f"[Grad-CAM] 오류: {e}")

    return {
        "file": audio_path.name,
        "overall": float(overall),
        "scores": {k: float(v) for k, v in scores.items()},
        "vae_score": float(vae_score),
        "status": status,
    }


def predict_directory(dir_path: Path, visualize: bool = False):
    """디렉토리 내 모든 오디오 파일 예측"""
    audio_files = list(dir_path.glob("*.mp3")) + list(dir_path.glob("*.m4a"))
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"오류: {dir_path}에서 mp3/m4a 파일을 찾을 수 없습니다.")
        return

    print(f"\n총 {len(audio_files)}개 파일 발견")

    results = []
    for audio_path in audio_files:
        result = predict_single_file(audio_path, visualize=visualize)
        if result:
            results.append(result)

    # 요약
    if results:
        import numpy as np
        overall_scores = [r["overall"] for r in results]

        print("\n" + "=" * 50)
        print("  전체 요약")
        print("=" * 50)
        print(f"  분석 파일: {len(results)}개")
        print(f"  평균 점수: {np.mean(overall_scores):.2f}")
        print(f"  최저 점수: {np.min(overall_scores):.2f}")
        print(f"  최고 점수: {np.max(overall_scores):.2f}")

        status_counts = {}
        for r in results:
            s = r["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
        print(f"  판정: {status_counts}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="EGAI 엔진 품질 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=str,
        help="오디오 파일 또는 폴더 경로"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Grad-CAM 시각화 이미지 생성"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # 상대 경로면 프로젝트 루트 기준으로 변환
    if not input_path.is_absolute():
        input_path = project_root / input_path

    print("=" * 50)
    print("  EGAI 엔진 품질 분석")
    print("=" * 50)

    if input_path.is_dir():
        predict_directory(input_path, visualize=args.visualize)
    elif input_path.is_file():
        predict_single_file(input_path, visualize=args.visualize)
    else:
        print(f"오류: {input_path}를 찾을 수 없습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
