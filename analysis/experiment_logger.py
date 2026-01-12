"""
실험 로깅 모듈
역할: ML 실험 추적 표준에 따른 재현성 보장
참조: Obsidian Vault/40_Research/ML_Experiment_Tracking_Standards.md

구조:
    experiments/
    +-- exp_001_2026-01-11_baseline/
        +-- config.yaml       # 실험 설정 (Hydra 호환)
        +-- metrics.json      # 성능 지표 (MLflow 호환)
        +-- environment.txt   # 환경 정보
        +-- data_splits.json  # Train/Val/Test 분할 인덱스
        +-- fold_results.json # CV Fold별 결과
        +-- checkpoints/      # 모델 체크포인트
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np

# YAML 지원 (없으면 JSON으로 폴백)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ExperimentLogger:
    """
    실험 로깅 클래스

    MLOps 도구 전환 대비:
    - config.yaml → Hydra 호환
    - metrics.json → MLflow/W&B 호환
    - environment.txt → DVC 호환
    """

    # 실험 번호 카운터 (클래스 변수)
    _exp_counter_file = ".exp_counter"

    def __init__(self, base_dir: str = "analysis/experiments"):
        """
        초기화

        매개변수:
            base_dir: 실험 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 현재 실험 정보
        self.exp_id: Optional[str] = None
        self.exp_dir: Optional[Path] = None
        self.config: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None

    def _get_next_exp_number(self) -> int:
        """다음 실험 번호 반환 (순차적)"""
        counter_path = self.base_dir / self._exp_counter_file

        if counter_path.exists():
            with open(counter_path, 'r') as f:
                counter = int(f.read().strip())
        else:
            # 기존 실험 폴더에서 최대 번호 찾기
            existing = list(self.base_dir.glob("exp_*"))
            if existing:
                numbers = []
                for p in existing:
                    try:
                        num = int(p.name.split("_")[1])
                        numbers.append(num)
                    except (IndexError, ValueError):
                        pass
                counter = max(numbers) if numbers else 0
            else:
                counter = 0

        # 카운터 증가 및 저장
        counter += 1
        with open(counter_path, 'w') as f:
            f.write(str(counter))

        return counter

    def start_experiment(self, name: str, config: Dict[str, Any],
                         description: str = "", tags: List[str] = None,
                         reference_exp: str = None) -> str:
        """
        새 실험 시작

        매개변수:
            name: 실험 이름 (예: 'baseline', 'batch64')
            config: 실험 설정 딕셔너리
            description: 실험 설명
            tags: 태그 리스트
            reference_exp: 비교 기준 실험 ID

        반환값:
            exp_id: 실험 ID (예: 'exp_001_2026-01-11_baseline')
        """
        self.start_time = datetime.now()
        date_str = self.start_time.strftime("%Y-%m-%d")
        exp_num = self._get_next_exp_number()

        # 실험 ID 생성 (표준 명명 규칙)
        self.exp_id = f"exp_{exp_num:03d}_{date_str}_{name}"
        self.exp_dir = self.base_dir / self.exp_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 체크포인트 디렉토리
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)

        # 설정 저장
        self.config = {
            "experiment": {
                "id": self.exp_id,
                "name": name,
                "description": description,
                "tags": tags or [],
                "reference_exp": reference_exp,
                "created_at": self.start_time.isoformat(),
            },
            **config
        }
        self._save_config()

        # 환경 정보 저장
        self._save_environment()

        # 메트릭 초기화
        self.metrics = {
            "experiment_id": self.exp_id,
            "status": "running",
            "started_at": self.start_time.isoformat(),
        }

        print(f"실험 시작: {self.exp_id}")
        print(f"저장 경로: {self.exp_dir}")

        return self.exp_id

    def _save_config(self):
        """설정 파일 저장 (YAML 또는 JSON)"""
        if YAML_AVAILABLE:
            config_path = self.exp_dir / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)
        else:
            config_path = self.exp_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _save_environment(self):
        """환경 정보 저장"""
        env_path = self.exp_dir / "environment.txt"

        env_info = []
        env_info.append(f"# Environment captured at: {datetime.now().isoformat()}")
        env_info.append(f"# Python: {sys.version}")
        env_info.append("")

        # pip freeze 실행
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=30
            )
            env_info.append("# Installed packages:")
            env_info.append(result.stdout)
        except Exception as e:
            env_info.append(f"# Failed to capture packages: {e}")

        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(env_info))

    def save_data_splits(self, train_idx: List[int], val_idx: List[int],
                         test_idx: List[int] = None,
                         cv_splits: List[Dict[str, List[int]]] = None,
                         random_state: int = 42):
        """
        데이터 분할 정보 저장 (재현성)

        매개변수:
            train_idx: 학습 데이터 인덱스
            val_idx: 검증 데이터 인덱스
            test_idx: 테스트 데이터 인덱스 (선택)
            cv_splits: CV 분할 정보 [{"train": [...], "val": [...]}, ...]
            random_state: 랜덤 시드
        """
        if not self.exp_dir:
            raise ValueError("먼저 start_experiment()를 호출하세요")

        splits = {
            "random_state": random_state,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }

        if test_idx is not None:
            splits["test_indices"] = test_idx
            splits["n_test"] = len(test_idx)

        if cv_splits is not None:
            splits["cv_splits"] = cv_splits
            splits["n_folds"] = len(cv_splits)

        splits_path = self.exp_dir / "data_splits.json"
        with open(splits_path, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)

        print(f"데이터 분할 저장: {len(train_idx)} train, {len(val_idx)} val")

    def log_fold_result(self, fold: int, metrics: Dict[str, float],
                        model_path: str = None):
        """
        CV Fold 결과 기록

        매개변수:
            fold: Fold 번호 (0-indexed)
            metrics: {"MAE": 0.5, "R2": 0.3, "epochs": 50}
            model_path: 모델 체크포인트 경로 (선택)
        """
        if not self.exp_dir:
            raise ValueError("먼저 start_experiment()를 호출하세요")

        results_path = self.exp_dir / "fold_results.json"

        # 기존 결과 로드
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = {"folds": []}

        # Fold 결과 추가/업데이트
        fold_result = {
            "fold": fold,
            "metrics": self._convert_numpy(metrics),
            "model_path": model_path,
            "logged_at": datetime.now().isoformat()
        }

        # 같은 fold가 있으면 업데이트
        updated = False
        for i, r in enumerate(all_results["folds"]):
            if r["fold"] == fold:
                all_results["folds"][i] = fold_result
                updated = True
                break

        if not updated:
            all_results["folds"].append(fold_result)
            all_results["folds"].sort(key=lambda x: x["fold"])

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """
        메트릭 기록 (MLflow/W&B 호환)

        매개변수:
            metrics: {"loss": 0.5, "mae": 0.3}
            step: 스텝/에포크 번호 (선택)
        """
        if not self.exp_dir:
            raise ValueError("먼저 start_experiment()를 호출하세요")

        # 메트릭 업데이트
        if "history" not in self.metrics:
            self.metrics["history"] = []

        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **self._convert_numpy(metrics)
        }
        self.metrics["history"].append(entry)

        # 최신 메트릭 업데이트
        self.metrics["latest"] = self._convert_numpy(metrics)

    def finish_experiment(self, final_metrics: Dict[str, Any] = None,
                          status: str = "completed"):
        """
        실험 종료 및 최종 저장

        매개변수:
            final_metrics: 최종 집계 메트릭 (CV mean/std 등)
            status: 'completed', 'failed', 'interrupted'
        """
        if not self.exp_dir:
            raise ValueError("먼저 start_experiment()를 호출하세요")

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # 최종 메트릭 업데이트
        self.metrics["status"] = status
        self.metrics["completed_at"] = end_time.isoformat()
        self.metrics["duration_seconds"] = duration

        if final_metrics:
            self.metrics["final"] = self._convert_numpy(final_metrics)

        # metrics.json 저장
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        print(f"\n실험 완료: {self.exp_id}")
        print(f"소요 시간: {duration/60:.1f}분")
        print(f"저장 경로: {self.exp_dir}")

        if final_metrics and "overall_MAE_mean" in final_metrics:
            print(f"최종 MAE: {final_metrics['overall_MAE_mean']:.4f} "
                  f"+/- {final_metrics.get('overall_MAE_std', 0):.4f}")

    def _convert_numpy(self, obj: Any) -> Any:
        """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(v) for v in obj]
        return obj

    def get_checkpoint_path(self, name: str = "best") -> Path:
        """체크포인트 저장 경로 반환"""
        if not self.exp_dir:
            raise ValueError("먼저 start_experiment()를 호출하세요")
        return self.exp_dir / "checkpoints" / f"{name}.keras"

    @classmethod
    def load_experiment(cls, exp_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        기존 실험 로드

        매개변수:
            exp_dir: 실험 디렉토리 경로

        반환값:
            실험 정보 딕셔너리
        """
        exp_dir = Path(exp_dir)

        result = {"exp_dir": str(exp_dir)}

        # config 로드
        config_yaml = exp_dir / "config.yaml"
        config_json = exp_dir / "config.json"

        if config_yaml.exists() and YAML_AVAILABLE:
            with open(config_yaml, 'r', encoding='utf-8') as f:
                result["config"] = yaml.safe_load(f)
        elif config_json.exists():
            with open(config_json, 'r', encoding='utf-8') as f:
                result["config"] = json.load(f)

        # metrics 로드
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                result["metrics"] = json.load(f)

        # data splits 로드
        splits_path = exp_dir / "data_splits.json"
        if splits_path.exists():
            with open(splits_path, 'r', encoding='utf-8') as f:
                result["data_splits"] = json.load(f)

        # fold results 로드
        folds_path = exp_dir / "fold_results.json"
        if folds_path.exists():
            with open(folds_path, 'r', encoding='utf-8') as f:
                result["fold_results"] = json.load(f)

        return result

    @classmethod
    def list_experiments(cls, base_dir: str = "analysis/experiments") -> List[Dict]:
        """
        모든 실험 목록 반환

        반환값:
            실험 정보 리스트
        """
        base_dir = Path(base_dir)
        experiments = []

        for exp_dir in sorted(base_dir.glob("exp_*")):
            if exp_dir.is_dir():
                info = {"id": exp_dir.name, "path": str(exp_dir)}

                # metrics.json에서 상태 확인
                metrics_path = exp_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    info["status"] = metrics.get("status", "unknown")
                    info["completed_at"] = metrics.get("completed_at")

                    if "final" in metrics:
                        info["mae"] = metrics["final"].get("overall_MAE_mean")
                else:
                    info["status"] = "incomplete"

                experiments.append(info)

        return experiments


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    설정 파일 로드

    매개변수:
        config_path: 설정 파일 경로

    반환값:
        설정 딕셔너리
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일 없음: {config_path}")

    if config_path.suffix in ['.yaml', '.yml'] and YAML_AVAILABLE:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"지원하지 않는 설정 파일 형식: {config_path.suffix}")


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("ExperimentLogger 테스트")
    print("=" * 60)

    # 프로젝트 루트
    project_root = Path(__file__).parent.parent

    # 설정 로드
    config = load_config(project_root / "configs" / "config.yaml")
    print(f"설정 로드 완료: {config['experiment']['name']}")

    # 로거 초기화
    logger = ExperimentLogger(project_root / "analysis" / "experiments")

    # 실험 시작
    exp_id = logger.start_experiment(
        name="test_logger",
        config=config,
        description="ExperimentLogger 테스트",
        tags=["test", "debug"]
    )

    # 데이터 분할 저장
    logger.save_data_splits(
        train_idx=[1, 2, 3, 4, 5],
        val_idx=[6, 7],
        cv_splits=[
            {"train": [1, 2, 3, 4], "val": [5, 6, 7]},
            {"train": [1, 2, 5, 6], "val": [3, 4, 7]},
        ]
    )

    # Fold 결과 기록
    logger.log_fold_result(0, {"MAE": 0.55, "R2": 0.3, "epochs": 30})
    logger.log_fold_result(1, {"MAE": 0.52, "R2": 0.35, "epochs": 28})

    # 에포크별 메트릭 기록
    for epoch in range(3):
        logger.log_metrics({"loss": 1.0 - epoch * 0.1, "mae": 0.8 - epoch * 0.1}, step=epoch)

    # 실험 종료
    logger.finish_experiment(
        final_metrics={"overall_MAE_mean": 0.535, "overall_MAE_std": 0.015}
    )

    # 실험 목록
    print("\n실험 목록:")
    for exp in ExperimentLogger.list_experiments(project_root / "analysis" / "experiments"):
        print(f"  - {exp['id']}: {exp['status']}")

    print("\n테스트 완료!")
