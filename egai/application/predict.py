"""
Prediction Use Case - 엔진 품질 예측

오디오 파일 → 품질 점수 예측 파이프라인
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from egai.domain.entities import AudioSample, PredictionResult, QualityScore
from egai.domain.services import ScoreInterpreter
from egai.infrastructure.preprocessing import AudioPreprocessor
from egai.infrastructure.models import ModelFactory


class EngineGrader:
    """
    엔진 품질 평가기

    사용 예:
        grader = EngineGrader("models/simple_cbam.h5")
        result = grader.predict("engine_audio.mp3")
        print(result.interpretation)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "simple_cbam",
    ):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로 (None이면 새 모델 생성)
            model_type: 모델 타입 (simple_cbam, simple, channel_concat)
        """
        self.model_type = model_type
        self.preprocessor = AudioPreprocessor()
        self.interpreter = ScoreInterpreter()

        if model_path and Path(model_path).exists():
            self.model = ModelFactory.load(model_path)
        else:
            self.model = ModelFactory.create(model_type)

    def predict(
        self,
        audio_path: Union[str, Path],
        interpret: bool = True,
    ) -> PredictionResult:
        """
        엔진 오디오에서 품질 예측

        Args:
            audio_path: 오디오 파일 경로
            interpret: 해석 추가 여부

        Returns:
            예측 결과
        """
        audio_path = str(audio_path)

        # 전처리
        full_spec, percussive_spec = self.preprocessor.process(audio_path)

        # 예측
        if self.model_type == "channel_concat":
            inputs = [
                np.expand_dims(full_spec, 0),
                np.expand_dims(percussive_spec, 0),
            ]
        else:
            inputs = np.expand_dims(full_spec, 0)

        predictions = self.model.predict(inputs, verbose=0)

        # 결과 생성
        scores = QualityScore.from_array(predictions[0])
        result = PredictionResult(
            scores=scores,
            model_version=f"{self.model_type}_v1",
        )

        # 해석 추가
        if interpret:
            result = self.interpreter.interpret(result)

        return result

    def predict_batch(
        self,
        audio_paths: list,
        interpret: bool = True,
        show_progress: bool = True,
    ) -> list:
        """
        배치 예측

        Args:
            audio_paths: 오디오 파일 경로 리스트
            interpret: 해석 추가 여부
            show_progress: 진행 상황 출력

        Returns:
            예측 결과 리스트
        """
        results = []

        for i, path in enumerate(audio_paths):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  예측 중: {i + 1}/{len(audio_paths)}")

            try:
                result = self.predict(path, interpret=interpret)
                results.append(result)
            except Exception as e:
                print(f"  [경고] {path} 예측 실패: {e}")
                results.append(None)

        return results


def main():
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser(description="EGAI 엔진 품질 예측")
    parser.add_argument("audio_path", help="오디오 파일 경로")
    parser.add_argument(
        "--model", "-m", default=None, help="학습된 모델 경로"
    )
    parser.add_argument(
        "--model_type", "-t", default="simple_cbam",
        choices=["simple_cbam", "simple", "channel_concat"],
        help="모델 타입",
    )
    parser.add_argument(
        "--json", "-j", action="store_true", help="JSON 형식 출력"
    )

    args = parser.parse_args()

    # 예측
    grader = EngineGrader(
        model_path=args.model,
        model_type=args.model_type,
    )
    result = grader.predict(args.audio_path)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*50}")
        print("EGAI 엔진 품질 예측 결과")
        print(f"{'='*50}")
        print(f"\n{result.interpretation}")
        print(f"\n상세 점수:")
        for name, value in result.scores.to_dict().items():
            if name != "overall":
                print(f"  {name:20s}: {value:.2f}")
        print(f"\n종합 점수: {result.scores.overall:.2f}")
        print(f"\n권장 사항:")
        for rec in result.recommendations:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
