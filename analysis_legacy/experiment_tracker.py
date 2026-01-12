"""
실험 추적 모듈
역할: 실험 재현성을 위한 설정, 데이터 분할, 결과 기록
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np


class ExperimentTracker:
    """
    실험 재현성 관리자
    - 데이터 분할 저장/로드
    - 실험 설정 기록
    - 캐시 매니페스트 관리
    """

    def __init__(self, experiment_dir: str = "experiments"):
        """
        초기화

        매개변수:
            experiment_dir: 실험 결과 저장 디렉토리
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 현재 실험 정보
        self.experiment_id = None
        self.config = {}

    def create_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """
        새 실험 생성

        매개변수:
            name: 실험 이름 (예: 'baseline_simple_cnn')
            config: 실험 설정

        반환값:
            experiment_id: 고유 실험 ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{name}_{timestamp}"
        self.config = config

        # 실험 디렉토리 생성
        exp_path = self.experiment_dir / self.experiment_id
        exp_path.mkdir(parents=True, exist_ok=True)

        # 설정 저장
        config_with_meta = {
            "experiment_id": self.experiment_id,
            "name": name,
            "created_at": timestamp,
            "config": config
        }

        with open(exp_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_with_meta, f, indent=2, ensure_ascii=False)

        print(f"실험 생성: {self.experiment_id}")
        return self.experiment_id

    def save_data_splits(self, train_idx: List[int], val_idx: List[int],
                         test_idx: Optional[List[int]] = None,
                         fold_splits: Optional[List[Dict]] = None):
        """
        데이터 분할 저장

        매개변수:
            train_idx: 학습 인덱스
            val_idx: 검증 인덱스
            test_idx: 테스트 인덱스 (선택)
            fold_splits: K-Fold 분할 정보 (선택)
        """
        if not self.experiment_id:
            raise ValueError("먼저 create_experiment()를 호출하세요")

        exp_path = self.experiment_dir / self.experiment_id

        splits = {
            "train_indices": train_idx,
            "val_indices": val_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx)
        }

        if test_idx is not None:
            splits["test_indices"] = test_idx
            splits["n_test"] = len(test_idx)

        if fold_splits is not None:
            splits["fold_splits"] = fold_splits

        with open(exp_path / "data_splits.json", 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)

        print(f"데이터 분할 저장: {len(train_idx)} train, {len(val_idx)} val")

    def save_cv_splits(self, cv_splits: List[Dict[str, List[int]]], random_state: int = 42):
        """
        Cross-Validation 분할 저장

        매개변수:
            cv_splits: [{"train": [...], "val": [...]}, ...] 형태의 분할 정보
            random_state: 랜덤 시드
        """
        if not self.experiment_id:
            raise ValueError("먼저 create_experiment()를 호출하세요")

        exp_path = self.experiment_dir / self.experiment_id

        splits_data = {
            "n_folds": len(cv_splits),
            "random_state": random_state,
            "folds": cv_splits
        }

        with open(exp_path / "cv_splits.json", 'w', encoding='utf-8') as f:
            json.dump(splits_data, f, indent=2)

        print(f"CV 분할 저장: {len(cv_splits)} folds")

    def save_fold_result(self, fold_num: int, metrics: Dict[str, float],
                         model_path: Optional[str] = None):
        """
        개별 Fold 결과 저장

        매개변수:
            fold_num: Fold 번호
            metrics: 평가 지표
            model_path: 모델 저장 경로 (선택)
        """
        if not self.experiment_id:
            raise ValueError("먼저 create_experiment()를 호출하세요")

        exp_path = self.experiment_dir / self.experiment_id

        result = {
            "fold": fold_num,
            "metrics": metrics,
            "model_path": model_path,
            "saved_at": datetime.now().isoformat()
        }

        # 기존 결과 로드 또는 새로 생성
        results_path = exp_path / "fold_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = {"folds": []}

        # 같은 fold 결과가 있으면 업데이트
        updated = False
        for i, r in enumerate(all_results["folds"]):
            if r["fold"] == fold_num:
                all_results["folds"][i] = result
                updated = True
                break

        if not updated:
            all_results["folds"].append(result)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    def save_final_results(self, results: Dict[str, Any]):
        """
        최종 실험 결과 저장

        매개변수:
            results: 집계된 결과 (mean, std 포함)
        """
        if not self.experiment_id:
            raise ValueError("먼저 create_experiment()를 호출하세요")

        exp_path = self.experiment_dir / self.experiment_id

        # numpy 타입을 Python 타입으로 변환
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        results_converted = convert_numpy(results)
        results_converted["completed_at"] = datetime.now().isoformat()

        with open(exp_path / "final_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)

        print(f"최종 결과 저장: {exp_path / 'final_results.json'}")

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        기존 실험 로드

        매개변수:
            experiment_id: 실험 ID

        반환값:
            실험 설정 및 결과
        """
        exp_path = self.experiment_dir / experiment_id

        if not exp_path.exists():
            raise ValueError(f"실험을 찾을 수 없음: {experiment_id}")

        self.experiment_id = experiment_id

        # 설정 로드
        with open(exp_path / "config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 데이터 분할 로드 (있으면)
        splits = None
        splits_path = exp_path / "data_splits.json"
        if splits_path.exists():
            with open(splits_path, 'r', encoding='utf-8') as f:
                splits = json.load(f)

        # CV 분할 로드 (있으면)
        cv_splits = None
        cv_splits_path = exp_path / "cv_splits.json"
        if cv_splits_path.exists():
            with open(cv_splits_path, 'r', encoding='utf-8') as f:
                cv_splits = json.load(f)

        # 결과 로드 (있으면)
        results = None
        results_path = exp_path / "final_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

        return {
            "config": config,
            "splits": splits,
            "cv_splits": cv_splits,
            "results": results
        }

    def list_experiments(self) -> List[Dict[str, str]]:
        """
        모든 실험 목록 반환

        반환값:
            실험 정보 리스트
        """
        experiments = []

        for exp_dir in sorted(self.experiment_dir.iterdir()):
            if exp_dir.is_dir():
                config_path = exp_dir / "config.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    # 결과 파일 확인
                    results_path = exp_dir / "final_results.json"
                    has_results = results_path.exists()

                    experiments.append({
                        "id": exp_dir.name,
                        "name": config.get("name", "unknown"),
                        "created_at": config.get("created_at", "unknown"),
                        "completed": has_results
                    })

        return experiments

    def import_legacy_results(self, name: str, cv_results: Dict[str, Any],
                               preprocessing_config: Dict[str, Any]) -> str:
        """
        기존 실험 결과를 새 형식으로 가져오기

        매개변수:
            name: 실험 이름
            cv_results: 기존 CV 결과 (cv_results.json 형식)
            preprocessing_config: 전처리 설정

        반환값:
            새로 생성된 experiment_id
        """
        # 설정 구성
        config = {
            "model": {
                "type": name.split('_')[0] if '_' in name else name
            },
            "training": {
                "n_folds": cv_results.get("n_folds", 5),
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "random_state": 42
            },
            "preprocessing": preprocessing_config,
            "legacy_import": True,
            "import_note": "기존 실험에서 가져옴 - 데이터 분할 정보 없음"
        }

        # 실험 생성
        exp_id = self.create_experiment(f"legacy_{name}", config)

        # 결과 저장
        self.save_final_results(cv_results)

        return exp_id


class CacheManifest:
    """
    스펙트로그램 캐시 매니페스트 관리
    - 전처리 설정과 캐시 파일 매핑
    - 캐시 무효화 감지
    """

    def __init__(self, cache_dir: str):
        """
        초기화

        매개변수:
            cache_dir: 캐시 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """매니페스트 로드"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "preprocessing_config": {},
            "files": {}
        }

    def _save_manifest(self):
        """매니페스트 저장"""
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)

    def _compute_config_hash(self, config: Dict) -> str:
        """설정의 해시값 계산"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _compute_file_hash(self, file_path: str) -> str:
        """파일 경로의 해시값 계산"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]

    def set_preprocessing_config(self, config: Dict):
        """
        전처리 설정 저장

        매개변수:
            config: 전처리 설정 (fmax, use_mel, target_shape 등)
        """
        config_hash = self._compute_config_hash(config)

        # 설정이 변경되었으면 캐시 무효화
        old_hash = self.manifest["preprocessing_config"].get("hash")
        if old_hash and old_hash != config_hash:
            print(f"전처리 설정 변경 감지 (기존: {old_hash}, 신규: {config_hash})")
            print(f"캐시 무효화됨. 기존 캐시 파일 {len(self.manifest['files'])}개")
            self.manifest["files"] = {}

        self.manifest["preprocessing_config"] = {
            "hash": config_hash,
            "config": config,
            "updated_at": datetime.now().isoformat()
        }
        self._save_manifest()

    def get_cache_path(self, audio_path: str) -> Optional[Path]:
        """
        오디오 파일의 캐시 경로 반환

        매개변수:
            audio_path: 오디오 파일 경로

        반환값:
            캐시 파일 경로 (없으면 None)
        """
        audio_hash = self._compute_file_hash(str(audio_path))

        if audio_hash in self.manifest["files"]:
            cache_file = self.manifest["files"][audio_hash]["cache_file"]
            cache_path = self.cache_dir / cache_file
            if cache_path.exists():
                return cache_path

        return None

    def register_cache(self, audio_path: str) -> Path:
        """
        캐시 파일 등록 및 경로 반환

        매개변수:
            audio_path: 원본 오디오 파일 경로

        반환값:
            캐시 파일 경로
        """
        audio_hash = self._compute_file_hash(str(audio_path))
        cache_filename = f"{audio_hash}_spec.npz"

        self.manifest["files"][audio_hash] = {
            "audio_path": str(audio_path),
            "cache_file": cache_filename,
            "created_at": datetime.now().isoformat()
        }
        self._save_manifest()

        return self.cache_dir / cache_filename

    def is_valid(self) -> bool:
        """캐시가 유효한지 확인"""
        return bool(self.manifest["preprocessing_config"].get("hash"))

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            "config_hash": self.manifest["preprocessing_config"].get("hash"),
            "n_cached_files": len(self.manifest["files"]),
            "last_updated": self.manifest["preprocessing_config"].get("updated_at")
        }


def create_experiment_config(
    model_type: str,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    target_shape: tuple = (128, 128),
    use_mel: bool = False,
    fmax: int = 6000,
    **kwargs
) -> Dict[str, Any]:
    """
    실험 설정 생성 헬퍼 함수

    반환값:
        실험 설정 딕셔너리
    """
    return {
        "model": {
            "type": model_type,
            "input_shape": list(target_shape) + [1]
        },
        "training": {
            "n_folds": n_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_state": random_state
        },
        "preprocessing": {
            "target_shape": list(target_shape),
            "use_mel": use_mel,
            "fmax": fmax
        },
        "extra": kwargs
    }


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("ExperimentTracker 테스트")
    print("=" * 60)

    # 프로젝트 루트
    project_root = Path(__file__).parent.parent
    exp_dir = project_root / "analysis" / "experiments"

    # 트래커 초기화
    tracker = ExperimentTracker(exp_dir)

    # 실험 생성
    config = create_experiment_config(
        model_type="simple",
        n_folds=5,
        epochs=50,
        batch_size=16
    )

    exp_id = tracker.create_experiment("test_experiment", config)

    # CV 분할 저장 테스트
    cv_splits = [
        {"train": [1, 2, 3, 4], "val": [5]},
        {"train": [1, 2, 3, 5], "val": [4]},
    ]
    tracker.save_cv_splits(cv_splits, random_state=42)

    # 결과 저장 테스트
    tracker.save_fold_result(0, {"MAE": 0.5, "R2": 0.3})
    tracker.save_fold_result(1, {"MAE": 0.45, "R2": 0.35})

    tracker.save_final_results({
        "overall_MAE_mean": 0.475,
        "overall_MAE_std": 0.025
    })

    # 실험 목록
    print("\n실험 목록:")
    for exp in tracker.list_experiments():
        print(f"  - {exp['id']}: {exp['name']} (완료: {exp['completed']})")

    # 실험 로드
    loaded = tracker.load_experiment(exp_id)
    print(f"\n로드된 실험: {loaded['config']['name']}")

    print("\n" + "=" * 60)
    print("CacheManifest 테스트")
    print("=" * 60)

    cache_dir = project_root / "analysis" / "cache_test"
    manifest = CacheManifest(cache_dir)

    # 전처리 설정
    manifest.set_preprocessing_config({
        "fmax": 6000,
        "use_mel": False,
        "target_shape": [128, 128]
    })

    # 캐시 등록
    test_audio = "data/audio/test.mp3"
    cache_path = manifest.register_cache(test_audio)
    print(f"캐시 경로: {cache_path}")

    # 통계
    stats = manifest.get_stats()
    print(f"캐시 통계: {stats}")

    print("\n테스트 완료!")
