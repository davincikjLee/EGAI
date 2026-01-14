"""
Agent Evaluator - 서브에이전트 평가 시스템

각 서브에이전트(또는 설정)의 실행 결과를 평가하고 점수를 부여하는 시스템
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum


class AgentGrade(Enum):
    """에이전트 평가 등급"""
    A = "A"  # 우수 (80-100)
    B = "B"  # 양호 (60-79)
    C = "C"  # 보통 (40-59)
    D = "D"  # 미흡 (0-39)


@dataclass
class AgentScore:
    """
    서브에이전트 평가 점수

    Attributes:
        agent_id: 에이전트/실험 식별자
        task_type: 작업 유형 (train, evaluate, compare, deploy)
        quality_score: 결과 품질 점수 (0-40)
        stability_score: 안정성 점수 (0-30)
        completion_score: 작업 완료도 점수 (0-30)
        total_score: 총점 (0-100)
        grade: 평가 등급 (A/B/C/D)
        config_hash: 설정 해시 (재활용 식별용)
        metrics: 상세 메트릭
        evaluated_at: 평가 시간
    """
    agent_id: str
    task_type: str
    quality_score: float
    stability_score: float
    completion_score: float
    total_score: float
    grade: str
    config_hash: str
    metrics: Dict[str, Any]
    evaluated_at: str

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentScore":
        """딕셔너리에서 생성"""
        return cls(**data)


class AgentEvaluator:
    """
    서브에이전트 평가기

    실험 결과를 분석하여 평가 점수를 부여하고,
    높은 점수의 설정을 재활용할 수 있도록 관리합니다.
    """

    # 평가 기준 가중치
    QUALITY_WEIGHT = 40      # 품질 점수 가중치 (MAE 기반)
    STABILITY_WEIGHT = 30    # 안정성 점수 가중치 (STD 기반)
    COMPLETION_WEIGHT = 30   # 완료도 점수 가중치

    # MAE 기준값
    MAE_EXCELLENT = 0.3   # 우수
    MAE_GOOD = 0.4        # 양호
    MAE_ACCEPTABLE = 0.5  # 수용 가능

    # STD 기준값
    STD_EXCELLENT = 0.02  # 매우 안정적
    STD_GOOD = 0.03       # 안정적
    STD_ACCEPTABLE = 0.05 # 수용 가능

    def __init__(
        self,
        scores_dir: str = "agent_scores",
        experiments_dir: str = "experiments",
    ):
        """
        Args:
            scores_dir: 평가 점수 저장 디렉토리
            experiments_dir: 실험 결과 디렉토리
        """
        self.scores_dir = Path(scores_dir)
        self.scores_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_dir = Path(experiments_dir)

        # 점수 캐시
        self._scores_cache: Dict[str, AgentScore] = {}
        self._load_scores()

    def _load_scores(self):
        """저장된 점수 로드"""
        scores_file = self.scores_dir / "agent_scores.json"
        if scores_file.exists():
            with open(scores_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for score_data in data.get("scores", []):
                    score = AgentScore.from_dict(score_data)
                    self._scores_cache[score.agent_id] = score

    def _save_scores(self):
        """점수 저장"""
        scores_file = self.scores_dir / "agent_scores.json"
        scores_data = {
            "updated_at": datetime.now().isoformat(),
            "total_count": len(self._scores_cache),
            "scores": [s.to_dict() for s in self._scores_cache.values()]
        }

        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores_data, f, indent=2, ensure_ascii=False)

    def evaluate_experiment(
        self,
        experiment_id: str,
        task_type: str = "train",
    ) -> AgentScore:
        """
        실험 결과 평가

        Args:
            experiment_id: 실험 ID
            task_type: 작업 유형

        Returns:
            평가 점수
        """
        # 실험 결과 로드
        results_path = self.experiments_dir / experiment_id / "results.json"
        config_path = self.experiments_dir / experiment_id / "config.json"

        if not results_path.exists():
            raise FileNotFoundError(f"실험 결과를 찾을 수 없습니다: {experiment_id}")

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 메트릭 추출
        final_metrics = results.get("final_metrics", {})
        mae_mean = final_metrics.get("overall_MAE_mean", 1.0)
        mae_std = final_metrics.get("overall_MAE_std", 1.0)
        status = results.get("status", "failed")

        # 점수 계산
        quality_score = self._calculate_quality_score(mae_mean)
        stability_score = self._calculate_stability_score(mae_std)
        completion_score = self._calculate_completion_score(status)

        total_score = quality_score + stability_score + completion_score
        grade = self._determine_grade(total_score)

        # 설정 해시 생성
        config_hash = self._generate_config_hash(config)

        # 평가 점수 생성
        score = AgentScore(
            agent_id=experiment_id,
            task_type=task_type,
            quality_score=round(quality_score, 2),
            stability_score=round(stability_score, 2),
            completion_score=round(completion_score, 2),
            total_score=round(total_score, 2),
            grade=grade.value,
            config_hash=config_hash,
            metrics={
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "status": status,
                "model_type": config.get("model_type"),
                "n_folds": config.get("n_folds"),
                "epochs": config.get("epochs"),
                "learning_rate": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
            },
            evaluated_at=datetime.now().isoformat(),
        )

        # 캐시 및 저장
        self._scores_cache[experiment_id] = score
        self._save_scores()

        return score

    def _calculate_quality_score(self, mae: float) -> float:
        """
        품질 점수 계산 (MAE 기반)

        MAE가 낮을수록 높은 점수
        """
        if mae <= self.MAE_EXCELLENT:
            return self.QUALITY_WEIGHT  # 40점
        elif mae <= self.MAE_GOOD:
            # 0.3~0.4: 30~40점
            ratio = (self.MAE_GOOD - mae) / (self.MAE_GOOD - self.MAE_EXCELLENT)
            return 30 + (ratio * 10)
        elif mae <= self.MAE_ACCEPTABLE:
            # 0.4~0.5: 20~30점
            ratio = (self.MAE_ACCEPTABLE - mae) / (self.MAE_ACCEPTABLE - self.MAE_GOOD)
            return 20 + (ratio * 10)
        elif mae <= 0.7:
            # 0.5~0.7: 10~20점
            ratio = (0.7 - mae) / 0.2
            return 10 + (ratio * 10)
        else:
            # 0.7 이상: 0~10점
            return max(0, (1 - mae) * 10)

    def _calculate_stability_score(self, std: float) -> float:
        """
        안정성 점수 계산 (STD 기반)

        표준편차가 낮을수록 높은 점수
        """
        if std <= self.STD_EXCELLENT:
            return self.STABILITY_WEIGHT  # 30점
        elif std <= self.STD_GOOD:
            # 0.02~0.03: 20~30점
            ratio = (self.STD_GOOD - std) / (self.STD_GOOD - self.STD_EXCELLENT)
            return 20 + (ratio * 10)
        elif std <= self.STD_ACCEPTABLE:
            # 0.03~0.05: 10~20점
            ratio = (self.STD_ACCEPTABLE - std) / (self.STD_ACCEPTABLE - self.STD_GOOD)
            return 10 + (ratio * 10)
        else:
            # 0.05 이상: 0~10점
            return max(0, (0.1 - std) * 100)

    def _calculate_completion_score(self, status: str) -> float:
        """
        완료도 점수 계산

        완료 상태에 따른 점수
        """
        if status == "completed":
            return self.COMPLETION_WEIGHT  # 30점
        elif status == "running":
            return 15  # 진행 중
        else:  # failed
            return 0

    def _determine_grade(self, total_score: float) -> AgentGrade:
        """평가 등급 결정"""
        if total_score >= 80:
            return AgentGrade.A
        elif total_score >= 60:
            return AgentGrade.B
        elif total_score >= 40:
            return AgentGrade.C
        else:
            return AgentGrade.D

    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """설정 해시 생성"""
        import hashlib

        # 중요 설정만 추출
        key_config = {
            "model_type": config.get("model_type"),
            "n_folds": config.get("n_folds"),
            "epochs": config.get("epochs"),
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "target_shape": config.get("target_shape"),
            "fmax": config.get("fmax"),
        }

        config_str = json.dumps(key_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def get_top_scores(
        self,
        n: int = 5,
        task_type: Optional[str] = None,
        min_grade: Optional[str] = None,
    ) -> List[AgentScore]:
        """
        상위 점수 조회

        Args:
            n: 반환할 개수
            task_type: 작업 유형 필터
            min_grade: 최소 등급 필터

        Returns:
            상위 점수 리스트
        """
        scores = list(self._scores_cache.values())

        # 필터링
        if task_type:
            scores = [s for s in scores if s.task_type == task_type]

        if min_grade:
            grade_order = {"A": 4, "B": 3, "C": 2, "D": 1}
            min_order = grade_order.get(min_grade, 0)
            scores = [s for s in scores if grade_order.get(s.grade, 0) >= min_order]

        # 정렬 및 반환
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores[:n]

    def get_best_config(self, task_type: str = "train") -> Optional[Dict[str, Any]]:
        """
        최고 성능 설정 반환

        Args:
            task_type: 작업 유형

        Returns:
            최고 점수 설정
        """
        top_scores = self.get_top_scores(n=1, task_type=task_type)
        if not top_scores:
            return None

        best_score = top_scores[0]

        # 원본 설정 로드
        config_path = self.experiments_dir / best_score.agent_id / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return None

    def get_reusable_configs(
        self,
        min_score: float = 60,
        min_grade: str = "B",
    ) -> List[Dict[str, Any]]:
        """
        재활용 가능한 설정 목록

        Args:
            min_score: 최소 점수
            min_grade: 최소 등급

        Returns:
            재활용 가능한 설정 리스트
        """
        reusable = []

        for score in self._scores_cache.values():
            if score.total_score >= min_score and score.grade <= min_grade:
                config_path = self.experiments_dir / score.agent_id / "config.json"
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        reusable.append({
                            "config": config,
                            "score": score.total_score,
                            "grade": score.grade,
                            "experiment_id": score.agent_id,
                        })

        reusable.sort(key=lambda x: x["score"], reverse=True)
        return reusable

    def print_leaderboard(self, n: int = 10):
        """리더보드 출력"""
        print("\n" + "=" * 70)
        print("EGAI MLOps Agent Leaderboard")
        print("=" * 70)
        print(f"{'Rank':<5} {'ID':<25} {'Score':>8} {'Grade':>6} {'MAE':>8}")
        print("-" * 70)

        top_scores = self.get_top_scores(n=n)
        for i, score in enumerate(top_scores, 1):
            mae = score.metrics.get("mae_mean", "N/A")
            mae_str = f"{mae:.4f}" if isinstance(mae, float) else mae
            print(f"{i:<5} {score.agent_id:<25} {score.total_score:>8.1f} {score.grade:>6} {mae_str:>8}")

        print("=" * 70)
