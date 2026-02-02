"""
Grad-CAM 시각화 스크립트

CNN 모델이 스펙트로그램의 어떤 영역을 보고 판단했는지 시각화합니다.

사용법:
    py -3.12 scripts/visualize_gradcam.py 파일.m4a
    py -3.12 scripts/visualize_gradcam.py 파일.m4a --output result.png
    py -3.12 scripts/visualize_gradcam.py sample/ --compare
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
    pass


def load_audio_spectrogram(audio_path: Path):
    """오디오 파일에서 4채널 스펙트로그램 생성"""
    import numpy as np
    import librosa

    # 오디오 로드
    audio, sr = librosa.load(str(audio_path), sr=22050)
    duration = len(audio) / sr

    if duration < 10:
        raise ValueError(f"Audio too short: {duration:.1f}s (min 10s)")

    # 중앙 10초 추출
    center = len(audio) // 2
    half_segment = 5 * sr
    segment = audio[center - half_segment:center + half_segment]

    # 4채널 스펙트로그램 생성
    from egai.preprocessing.audio import AudioPreprocessor
    preprocessor = AudioPreprocessor()
    spectrogram = preprocessor.compute_4channel_spectrogram(segment)

    return spectrogram


def visualize_single(
    audio_path: Path,
    output_path: Path = None,
    model_path: Path = None,
    generate_report: bool = True,
):
    """단일 파일 Grad-CAM 시각화"""
    import numpy as np
    import librosa
    from egai.visualization.gradcam import (
        create_gradcam_for_regression_model,
        generate_gradcam_report,
    )

    print(f"\n[Grad-CAM] 파일: {audio_path.name}")
    print("-" * 50)

    # 오디오 길이 확인
    audio, sr = librosa.load(str(audio_path), sr=22050)
    audio_duration = len(audio) / sr

    # 1. 스펙트로그램 생성
    print("[1/4] 스펙트로그램 생성 중...")
    spectrogram = load_audio_spectrogram(audio_path)
    print(f"      Shape: {spectrogram.shape}")

    # 2. 모델 로드 및 Grad-CAM 생성
    print("[2/4] Grad-CAM 계산 중...")
    if model_path is None:
        model_path = project_root / "experiments/best_model_4channel_cbam_1.keras"

    gradcam = create_gradcam_for_regression_model(str(model_path))

    # 히트맵 계산 (첫 번째 출력에 대해)
    input_tensor = np.expand_dims(spectrogram, axis=0)
    heatmap = gradcam.compute_heatmap(input_tensor, class_idx=0)

    # 3. 시각화
    print("[3/4] 시각화 생성 중...")
    if output_path is None:
        output_path = audio_path.with_suffix('.gradcam.png')

    gradcam.visualize(
        spectrogram,
        heatmap,
        output_path=str(output_path),
        title=f"Grad-CAM: {audio_path.name}",
        channel_idx=0,  # Full 스펙트로그램
    )

    # 4. 보고서 생성
    if generate_report:
        print("[4/4] 보고서 생성 중...")
        report_path = audio_path.with_suffix('.gradcam_report.md')
        generate_gradcam_report(
            filename=audio_path.name,
            image_path=output_path.name,
            heatmap=heatmap,
            output_path=str(report_path),
            audio_duration=audio_duration,
        )
    else:
        print("[4/4] 보고서 생성 건너뜀")

    print(f"\n[완료] 이미지: {output_path}")
    if generate_report:
        print(f"[완료] 보고서: {report_path}")
    return output_path


def visualize_comparison(
    dir_path: Path,
    output_path: Path = None,
    max_samples: int = 4,
):
    """여러 파일 비교 시각화"""
    import numpy as np
    from egai.visualization.gradcam import create_gradcam_for_regression_model

    # 오디오 파일 찾기
    audio_files = list(dir_path.glob("*.mp3")) + list(dir_path.glob("*.m4a"))
    audio_files = sorted(audio_files)[:max_samples]

    if not audio_files:
        print(f"오류: {dir_path}에서 오디오 파일을 찾을 수 없습니다.")
        return

    print(f"\n[Grad-CAM 비교] {len(audio_files)}개 파일")
    print("-" * 50)

    # 모델 로드
    model_path = project_root / "experiments/best_model_4channel_cbam_1.keras"
    gradcam = create_gradcam_for_regression_model(str(model_path))

    # 샘플 준비
    samples = []
    for audio_path in audio_files:
        print(f"  처리 중: {audio_path.name}")
        try:
            spectrogram = load_audio_spectrogram(audio_path)
            label = audio_path.stem[:15]  # 파일명 앞 15자
            samples.append((spectrogram, label, 0))
        except Exception as e:
            print(f"    오류: {e}")

    if not samples:
        print("오류: 처리 가능한 파일이 없습니다.")
        return

    # 비교 시각화
    if output_path is None:
        output_path = dir_path / "gradcam_comparison.png"

    gradcam.visualize_comparison(
        samples,
        output_path=str(output_path),
    )

    print(f"\n[완료] 저장: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM 시각화: CNN이 주목하는 영역 표시"
    )
    parser.add_argument(
        "input",
        type=str,
        help="오디오 파일 또는 폴더 경로"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="출력 이미지 경로"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="여러 파일 비교 모드 (폴더 입력 시)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="모델 경로 (기본: best_model_4channel_cbam_1.keras)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="보고서 생성 건너뛰기"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    output_path = Path(args.output) if args.output else None
    model_path = Path(args.model) if args.model else None

    print("=" * 50)
    print("  EGAI Grad-CAM Visualization")
    print("=" * 50)

    generate_report = not args.no_report

    if input_path.is_dir():
        if args.compare:
            visualize_comparison(input_path, output_path)
        else:
            # 폴더 내 모든 파일 개별 처리
            audio_files = list(input_path.glob("*.mp3")) + list(input_path.glob("*.m4a"))
            for audio_path in sorted(audio_files):
                visualize_single(audio_path, model_path=model_path, generate_report=generate_report)
    elif input_path.is_file():
        visualize_single(input_path, output_path, model_path, generate_report=generate_report)
    else:
        print(f"오류: {input_path}를 찾을 수 없습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
