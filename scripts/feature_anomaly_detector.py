#!/usr/bin/env python
"""
Feature-based Anomaly Detection

기존 Regression 모델의 중간층 특징을 추출하여 Isolation Forest로 이상 탐지
- 2,233개 정상 데이터의 특징 분포 학습
- 새 샘플이 이 분포에서 벗어나면 이상으로 판단

Usage:
    py -3.12 scripts/feature_anomaly_detector.py
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import joblib

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from egai.infrastructure.models import ModelFactory, CBAM, BoundedOutput
from egai.preprocessing.audio import AudioPreprocessor


class FeatureExtractor:
    """Regression 모델에서 특징 추출기 생성"""

    REGRESSION_MODEL_PATH = "experiments/best_model_4channel_cbam_1.keras"

    def __init__(self):
        self.model = None
        self.feature_model = None
        self.preprocessor = AudioPreprocessor()

    def load_model(self):
        """모델 로드 및 특징 추출기 생성"""
        print("[1/2] Regression 모델 로드...")
        model_path = project_root / self.REGRESSION_MODEL_PATH
        self.model = ModelFactory.load(str(model_path))

        print("[2/2] Feature Extractor 생성...")
        # GlobalAveragePooling2D 출력 찾기 (256차원)
        # 모델 구조: Conv → CBAM → GlobalAvgPool → Dense → Output

        # GlobalAveragePooling2D 레이어 찾기
        gap_layer = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                gap_layer = layer
                break

        if gap_layer is None:
            raise ValueError("GlobalAveragePooling2D 레이어를 찾을 수 없습니다")

        # 특징 추출기 모델 생성
        self.feature_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=gap_layer.output,
            name="FeatureExtractor"
        )

        print(f"   특징 차원: {self.feature_model.output_shape[-1]}")
        print("모델 로드 완료!\n")

    def extract_features(self, spectrogram: np.ndarray) -> np.ndarray:
        """스펙트로그램에서 특징 추출"""
        if spectrogram.ndim == 3:
            spectrogram = np.expand_dims(spectrogram, axis=0)
        return self.feature_model.predict(spectrogram, verbose=0)

    def extract_from_audio(self, audio_path: str) -> np.ndarray:
        """오디오 파일에서 특징 추출"""
        spectrogram = self.preprocessor.process_file(audio_path)
        return self.extract_features(spectrogram)


class FeatureAnomalyDetector:
    """특징 기반 이상 탐지기"""

    def __init__(self, contamination: float = 0.01):
        """
        Args:
            contamination: 이상 비율 추정 (1% = 0.01)
        """
        self.extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

        # 학습 통계
        self.train_features = None
        self.train_scores = None

    def load_models(self):
        """모델 로드"""
        self.extractor.load_model()

    def fit_from_metadata(self, csv_path: str, audio_dir: str, min_score: float = 4.0):
        """
        정상 데이터로 학습

        Args:
            csv_path: 메타데이터 CSV 경로
            audio_dir: 오디오 파일 디렉토리
            min_score: 정상으로 간주할 최소 점수
        """
        print(f"=== 정상 데이터 특징 추출 ===")
        print(f"CSV: {csv_path}")
        print(f"Audio Dir: {audio_dir}")
        print(f"Min Score: {min_score}\n")

        # 메타데이터 로드
        df = pd.read_csv(csv_path)
        print(f"전체 데이터: {len(df)}개")

        # 정상 데이터 필터링
        df_normal = df[df['overall_score'] >= min_score].copy()
        print(f"정상 데이터 (≥{min_score}): {len(df_normal)}개\n")

        # 특징 추출
        features_list = []
        filenames = []

        for idx, row in df_normal.iterrows():
            audio_path = Path(audio_dir) / row['audio_filename']

            if not audio_path.exists():
                continue

            try:
                features = self.extractor.extract_from_audio(str(audio_path))
                features_list.append(features[0])
                filenames.append(row['audio_filename'])

                if len(features_list) % 100 == 0:
                    print(f"  진행: {len(features_list)}/{len(df_normal)}")

            except Exception as e:
                print(f"  에러: {audio_path.name} - {e}")
                continue

        print(f"\n특징 추출 완료: {len(features_list)}개")

        # 배열 변환
        self.train_features = np.array(features_list)
        print(f"특징 형상: {self.train_features.shape}")

        # 정규화
        print("\n정규화 중...")
        self.train_features = self.scaler.fit_transform(self.train_features)

        # Isolation Forest 학습
        print("Isolation Forest 학습 중...")
        self.isolation_forest.fit(self.train_features)

        # 학습 데이터 점수 계산
        self.train_scores = -self.isolation_forest.score_samples(self.train_features)

        print(f"\n=== 학습 완료 ===")
        print(f"학습 데이터 이상 점수:")
        print(f"  평균: {self.train_scores.mean():.4f}")
        print(f"  표준편차: {self.train_scores.std():.4f}")
        print(f"  최소: {self.train_scores.min():.4f}")
        print(f"  최대: {self.train_scores.max():.4f}")

        # 임계값 설정 (μ + 3σ)
        threshold = self.train_scores.mean() + 3 * self.train_scores.std()
        print(f"  권장 임계값 (μ+3σ): {threshold:.4f}")

        self.is_fitted = True

        return {
            'n_samples': len(features_list),
            'feature_dim': self.train_features.shape[1],
            'mean_score': float(self.train_scores.mean()),
            'std_score': float(self.train_scores.std()),
            'threshold': float(threshold)
        }

    def predict(self, audio_path: str) -> dict:
        """
        단일 파일 이상 탐지

        Returns:
            {
                'anomaly_score': float,
                'is_anomaly': bool,
                'percentile': float
            }
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit_from_metadata()를 먼저 호출하세요.")

        # 특징 추출
        features = self.extractor.extract_from_audio(audio_path)
        features = self.scaler.transform(features)

        # 이상 점수 계산
        anomaly_score = -self.isolation_forest.score_samples(features)[0]

        # 백분위수 계산
        percentile = (self.train_scores < anomaly_score).mean() * 100

        # 임계값 (μ + 3σ)
        threshold = self.train_scores.mean() + 3 * self.train_scores.std()
        is_anomaly = anomaly_score > threshold

        return {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'percentile': float(percentile),
            'threshold': float(threshold)
        }

    def predict_batch(self, audio_dir: str, pattern: str = "*.m4a") -> list:
        """
        폴더 내 모든 파일 이상 탐지
        """
        results = []
        audio_dir = Path(audio_dir)

        for audio_path in sorted(audio_dir.glob(pattern)):
            try:
                result = self.predict(str(audio_path))
                result['filename'] = audio_path.name
                results.append(result)

                status = "⚠️ 이상" if result['is_anomaly'] else "✓ 정상"
                print(f"  {audio_path.name}: {result['anomaly_score']:.4f} ({result['percentile']:.1f}%) {status}")

            except Exception as e:
                print(f"  {audio_path.name}: 에러 - {e}")
                results.append({
                    'filename': audio_path.name,
                    'error': str(e)
                })

        return results

    def save(self, output_dir: str):
        """모델 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scaler 저장
        joblib.dump(self.scaler, output_dir / "scaler.joblib")

        # Isolation Forest 저장
        joblib.dump(self.isolation_forest, output_dir / "isolation_forest.joblib")

        # 통계 저장
        stats = {
            'train_mean': float(self.train_scores.mean()),
            'train_std': float(self.train_scores.std()),
            'train_min': float(self.train_scores.min()),
            'train_max': float(self.train_scores.max()),
            'threshold': float(self.train_scores.mean() + 3 * self.train_scores.std())
        }
        with open(output_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"모델 저장: {output_dir}")

    def load(self, model_dir: str):
        """저장된 모델 로드"""
        model_dir = Path(model_dir)

        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.isolation_forest = joblib.load(model_dir / "isolation_forest.joblib")

        with open(model_dir / "stats.json", 'r') as f:
            stats = json.load(f)

        # train_scores 복원 (threshold 계산용)
        # 실제 scores는 없지만 통계로 계산
        self.train_scores = np.array([stats['train_mean']])  # placeholder

        self.is_fitted = True
        print(f"모델 로드: {model_dir}")
        print(f"  임계값: {stats['threshold']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Feature-based Anomaly Detection")
    parser.add_argument("--csv", default="data/car_audio_metadata.csv", help="메타데이터 CSV")
    parser.add_argument("--audio-dir", default="data/vehicle_assets", help="오디오 디렉토리")
    parser.add_argument("--sample-dir", default="sample", help="테스트 샘플 디렉토리")
    parser.add_argument("--min-score", type=float, default=4.0, help="정상 최소 점수")
    parser.add_argument("--output", default="experiments/feature_anomaly", help="출력 디렉토리")

    args = parser.parse_args()

    print("=" * 60)
    print("Feature-based Anomaly Detection")
    print("=" * 60)
    print()

    # 탐지기 생성
    detector = FeatureAnomalyDetector(contamination=0.01)
    detector.load_models()

    # 정상 데이터로 학습
    train_stats = detector.fit_from_metadata(
        csv_path=str(project_root / args.csv),
        audio_dir=str(project_root / args.audio_dir),
        min_score=args.min_score
    )

    print()
    print("=" * 60)
    print("샘플 테스트")
    print("=" * 60)
    print()

    # 샘플 테스트
    sample_dir = project_root / args.sample_dir
    if sample_dir.exists():
        results = detector.predict_batch(str(sample_dir), "*.m4a")

        # 결과 분석
        print()
        print("=== 결과 요약 ===")

        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            scores = [r['anomaly_score'] for r in valid_results]
            anomalies = [r for r in valid_results if r['is_anomaly']]

            print(f"테스트 파일: {len(valid_results)}개")
            print(f"이상 탐지: {len(anomalies)}개")
            print(f"정상: {len(valid_results) - len(anomalies)}개")
            print()
            print(f"이상 점수 통계:")
            print(f"  평균: {np.mean(scores):.4f}")
            print(f"  표준편차: {np.std(scores):.4f}")
            print(f"  최소: {np.min(scores):.4f}")
            print(f"  최대: {np.max(scores):.4f}")

            if anomalies:
                print()
                print("이상으로 탐지된 파일:")
                for r in anomalies:
                    print(f"  - {r['filename']}: {r['anomaly_score']:.4f} ({r['percentile']:.1f}%)")

    # 모델 저장
    output_dir = project_root / args.output
    detector.save(str(output_dir))

    # 결과 저장
    result_path = output_dir / "sample_results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_stats': train_stats,
            'sample_results': results if sample_dir.exists() else [],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print()
    print(f"결과 저장: {result_path}")


if __name__ == "__main__":
    main()
