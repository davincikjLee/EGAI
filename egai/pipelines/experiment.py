"""
Experiment Tracker - 실험 추적 및 버전 관리

MLOps 핵심: 실험 재현성 보장
    - 설정 저장
    - 결과 기록
    - 데이터 분할 추적
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class ExperimentTracker:
    """
    실험 추적기

    모든 실험의 설정, 결과, 데이터 분할을 기록
    """

    def __init__(self, experiments_dir: str = "experiments"):
        """
        Args:
            experiments_dir: 실험 저장 디렉토리
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.current_experiment: Optional[Dict] = None
        self.current_dir: Optional[Path] = None

    def start_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        새 실험 시작

        Args:
            name: 실험 이름
            config: 실험 설정
            tags: 태그 리스트

        Returns:
            실험 ID
        """
        # 실험 ID 생성 (타임스탬프 + 이름 해시)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
        exp_id = f"{timestamp}_{name_hash}"

        # 실험 디렉토리 생성
        self.current_dir = self.experiments_dir / exp_id
        self.current_dir.mkdir(parents=True, exist_ok=True)

        # 실험 메타데이터
        self.current_experiment = {
            "id": exp_id,
            "name": name,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "config": config,
            "tags": tags or [],
            "metrics": {},
            "fold_results": [],
        }

        # 설정 저장
        self._save_config(config)

        print(f"[Experiment] 시작: {exp_id}")
        print(f"[Experiment] 디렉토리: {self.current_dir}")

        return exp_id

    def log_fold_result(
        self,
        fold: int,
        metrics: Dict[str, float],
    ):
        """
        Fold 결과 기록

        Args:
            fold: Fold 번호
            metrics: 평가 메트릭
        """
        if not self.current_experiment:
            raise RuntimeError("실험이 시작되지 않았습니다")

        result = {
            "fold": fold,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self.current_experiment["fold_results"].append(result)

        # MAE 또는 F1 출력 (이진 분류 호환)
        mae = metrics.get('MAE')
        f1 = metrics.get('f1')
        if mae is not None:
            print(f"[Experiment] Fold {fold + 1}: MAE={mae:.4f}")
        elif f1 is not None:
            print(f"[Experiment] Fold {fold + 1}: F1={f1:.4f}")
        else:
            print(f"[Experiment] Fold {fold + 1}: 완료")

    def log_metric(self, name: str, value: float):
        """단일 메트릭 기록"""
        if not self.current_experiment:
            return

        self.current_experiment["metrics"][name] = value

    def save_data_splits(
        self,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: Optional[List[int]] = None,
        cv_splits: Optional[List[Dict]] = None,
    ):
        """
        데이터 분할 저장 (재현성)

        Args:
            train_idx: 훈련 인덱스
            val_idx: 검증 인덱스
            test_idx: 테스트 인덱스
            cv_splits: CV 분할 정보
        """
        if not self.current_dir:
            return

        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx or [],
            "cv_splits": cv_splits or [],
        }

        splits_path = self.current_dir / "data_splits.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)

    def finish_experiment(
        self,
        final_metrics: Optional[Dict] = None,
        status: str = "completed",
    ):
        """
        실험 완료

        Args:
            final_metrics: 최종 메트릭
            status: 완료 상태
        """
        if not self.current_experiment:
            return

        self.current_experiment["status"] = status
        self.current_experiment["finished_at"] = datetime.now().isoformat()

        if final_metrics:
            self.current_experiment["final_metrics"] = self._convert_to_serializable(
                final_metrics
            )

        # 결과 저장
        self._save_results()

        exp_id = self.current_experiment["id"]
        print(f"[Experiment] 완료: {exp_id}")

        if final_metrics:
            mae = final_metrics.get("overall_MAE_mean", final_metrics.get("mae_mean"))
            if mae:
                print(f"[Experiment] 최종 MAE: {mae:.4f}")

        self.current_experiment = None
        self.current_dir = None

    def _save_config(self, config: Dict):
        """설정 저장"""
        config_path = self.current_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._convert_to_serializable(config), f, indent=2)

    def _save_results(self):
        """결과 저장"""
        results_path = self.current_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(
                self._convert_to_serializable(self.current_experiment),
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _convert_to_serializable(self, obj: Any) -> Any:
        """JSON 직렬화 가능하게 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def list_experiments(self) -> List[Dict]:
        """모든 실험 목록 반환"""
        experiments = []

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            results_path = exp_dir / "results.json"
            if results_path.exists():
                with open(results_path, "r") as f:
                    exp_data = json.load(f)
                    experiments.append({
                        "id": exp_data.get("id"),
                        "name": exp_data.get("name"),
                        "status": exp_data.get("status"),
                        "started_at": exp_data.get("started_at"),
                        "final_metrics": exp_data.get("final_metrics", {}),
                    })

        return experiments

    def compare_experiments(
        self,
        exp_ids: List[str],
        metric: str = "overall_MAE_mean",
    ) -> Dict[str, float]:
        """실험 비교"""
        results = {}

        for exp_id in exp_ids:
            exp_dir = self.experiments_dir / exp_id
            results_path = exp_dir / "results.json"

            if results_path.exists():
                with open(results_path, "r") as f:
                    exp_data = json.load(f)
                    final = exp_data.get("final_metrics", {})
                    results[exp_id] = final.get(metric)

        return results
