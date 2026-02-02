"""
통합 XAI 시각화 스크립트

3개 모델(Regression, Binary Classification, VAE)의 분석 결과를 하나의 이미지로 시각화합니다.

사용법:
    py -3.12 scripts/visualize_unified.py 파일.m4a
    py -3.12 scripts/visualize_unified.py 파일.m4a --output result.png
    py -3.12 scripts/visualize_unified.py sample/  # 폴더 내 모든 파일
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


def get_model_paths():
    """모델 경로 반환"""
    experiments_dir = project_root / "experiments"

    # Regression 모델
    regression_path = experiments_dir / "best_model_4channel_cbam_1.keras"

    # Binary 모델 찾기
    binary_path = None
    # 1. experiments 폴더 직접 확인
    for pattern in ["best_binary_model*.keras", "best_model_binary*.keras"]:
        candidates = list(experiments_dir.glob(pattern))
        if candidates:
            binary_path = sorted(candidates, reverse=True)[0]
            break
    # 2. 실험 폴더 내부 확인
    if binary_path is None:
        for exp_dir in sorted(experiments_dir.glob("202*"), reverse=True):
            for pattern in ["best_model_binary.keras", "best_binary_model*.keras"]:
                candidates = list(exp_dir.glob(pattern))
                if candidates:
                    binary_path = candidates[0]
                    break
            if binary_path:
                break

    # VAE 모델: CLAUDE.md에 명시된 기본 경로 우선
    vae_dir = experiments_dir / "20260124_013112_530a70"
    if not (vae_dir / "vae_encoder.keras").exists():
        # 기본 경로 없으면 가장 최근 VAE 찾기
        vae_dir = None
        for exp_dir in sorted(experiments_dir.glob("202*"), reverse=True):
            encoder = exp_dir / "vae_encoder.keras"
            decoder = exp_dir / "vae_decoder.keras"
            if encoder.exists() and decoder.exists():
                vae_dir = exp_dir
                break

    return regression_path, binary_path, vae_dir


def visualize_single(
    audio_path: Path,
    output_path: Path = None,
    no_report: bool = False,
):
    """단일 파일 통합 시각화"""
    from egai.visualization.gradcam import generate_unified_visualization

    print(f"\n[Unified XAI] File: {audio_path.name}")
    print("-" * 50)

    regression_path, binary_path, vae_dir = get_model_paths()

    print(f"  Regression: {regression_path.name if regression_path else 'Not found'}")
    print(f"  Binary: {binary_path.name if binary_path else 'Not found'}")
    print(f"  VAE: {vae_dir.name if vae_dir else 'Not found'}")
    print("-" * 50)

    if output_path is None:
        output_path = audio_path.with_suffix('.unified_gradcam.png')

    results = generate_unified_visualization(
        audio_path=str(audio_path),
        regression_model_path=str(regression_path) if regression_path else None,
        binary_model_path=str(binary_path) if binary_path else None,
        vae_dir=str(vae_dir) if vae_dir else None,
        output_path=str(output_path),
        generate_report=not no_report,
    )

    print("\n[Result Summary]")
    for model_name, info in results.get("models", {}).items():
        status = info.get("status", "unknown")
        if status == "success":
            extra = f" (Score: {info['score']:.1f})" if "score" in info else ""
            print(f"  {model_name.upper()}: OK{extra}")
        else:
            print(f"  {model_name.upper()}: Error - {info.get('message', 'Unknown')}")

    return results


def visualize_directory(
    dir_path: Path,
    no_report: bool = False,
):
    """디렉토리 내 모든 파일 처리"""
    audio_files = list(dir_path.glob("*.mp3")) + list(dir_path.glob("*.m4a"))
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"Error: No audio files found in {dir_path}")
        return

    print(f"\n[Unified XAI] Found {len(audio_files)} files")
    print("=" * 50)

    for audio_path in audio_files:
        try:
            visualize_single(audio_path, no_report=no_report)
        except Exception as e:
            print(f"  Error processing {audio_path.name}: {e}")

    print("\n" + "=" * 50)
    print(f"  Completed: {len(audio_files)} files")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Unified XAI Visualization: Regression + Binary + VAE"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Audio file or directory path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output image path"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    output_path = Path(args.output) if args.output else None

    print("=" * 50)
    print("  EGAI Unified XAI Visualization")
    print("=" * 50)

    if input_path.is_dir():
        visualize_directory(input_path, no_report=args.no_report)
    elif input_path.is_file():
        visualize_single(input_path, output_path, no_report=args.no_report)
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
