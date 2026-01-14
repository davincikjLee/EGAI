"""
Config Registry - 설정 레지스트리

높은 점수의 설정을 저장하고 재활용하기 위한 레지스트리
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class ConfigEntry:
    """
    설정 레지스트리 항목

    Attributes:
        config_hash: 설정 해시 (고유 식별자)
        config: 설정 값
        best_score: 해당 설정의 최고 점수
        best_experiment_id: 최고 성능 실험 ID
        usage_count: 사용 횟수
        success_rate: 성공률 (completed / total)
        avg_mae: 평균 MAE
        created_at: 생성 시간
        last_used_at: 마지막 사용 시간
        tags: 태그
    """
    config_hash: str
    config: Dict[str, Any]
    best_score: float
    best_experiment_id: str
    usage_count: int
    success_rate: float
    avg_mae: float
    created_at: str
    last_used_at: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigEntry":
        return cls(**data)


class ConfigRegistry:
    """
    설정 레지스트리

    성공적인 설정을 저장하고 재활용할 수 있도록 관리합니다.
    """

    def __init__(self, registry_dir: str = "config_registry"):
        """
        Args:
            registry_dir: 레지스트리 저장 디렉토리
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, ConfigEntry] = {}
        self._load_registry()

    def _load_registry(self):
        """레지스트리 로드"""
        registry_file = self.registry_dir / "config_registry.json"
        if registry_file.exists():
            with open(registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry_data in data.get("configs", []):
                    entry = ConfigEntry.from_dict(entry_data)
                    self._registry[entry.config_hash] = entry

    def _save_registry(self):
        """레지스트리 저장"""
        registry_file = self.registry_dir / "config_registry.json"
        registry_data = {
            "updated_at": datetime.now().isoformat(),
            "total_configs": len(self._registry),
            "configs": [e.to_dict() for e in self._registry.values()]
        }

        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)

    def _generate_hash(self, config: Dict[str, Any]) -> str:
        """설정 해시 생성"""
        # 중요 설정만 추출하여 해시
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

    def register_config(
        self,
        config: Dict[str, Any],
        experiment_id: str,
        score: float,
        mae: float,
        status: str = "completed",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        설정 등록

        Args:
            config: 설정 값
            experiment_id: 실험 ID
            score: 평가 점수
            mae: MAE 값
            status: 실험 상태
            tags: 태그

        Returns:
            설정 해시
        """
        config_hash = self._generate_hash(config)
        now = datetime.now().isoformat()

        if config_hash in self._registry:
            # 기존 설정 업데이트
            entry = self._registry[config_hash]
            entry.usage_count += 1
            entry.last_used_at = now

            # 최고 점수 갱신
            if score > entry.best_score:
                entry.best_score = score
                entry.best_experiment_id = experiment_id

            # 평균 MAE 갱신 (이동 평균)
            entry.avg_mae = (entry.avg_mae * (entry.usage_count - 1) + mae) / entry.usage_count

            # 성공률 갱신
            if status == "completed":
                # 간단한 성공률 업데이트 (정확한 계산을 위해선 전체 기록 필요)
                entry.success_rate = min(1.0, entry.success_rate + 0.1)

        else:
            # 새 설정 등록
            entry = ConfigEntry(
                config_hash=config_hash,
                config=config,
                best_score=score,
                best_experiment_id=experiment_id,
                usage_count=1,
                success_rate=1.0 if status == "completed" else 0.0,
                avg_mae=mae,
                created_at=now,
                last_used_at=now,
                tags=tags or [],
            )
            self._registry[config_hash] = entry

        self._save_registry()
        return config_hash

    def get_config(self, config_hash: str) -> Optional[Dict[str, Any]]:
        """설정 조회"""
        entry = self._registry.get(config_hash)
        return entry.config if entry else None

    def get_top_configs(
        self,
        n: int = 5,
        min_score: float = 0,
        min_usage: int = 0,
    ) -> List[ConfigEntry]:
        """
        상위 성능 설정 조회

        Args:
            n: 반환할 개수
            min_score: 최소 점수
            min_usage: 최소 사용 횟수

        Returns:
            상위 설정 리스트
        """
        entries = [
            e for e in self._registry.values()
            if e.best_score >= min_score and e.usage_count >= min_usage
        ]

        entries.sort(key=lambda x: x.best_score, reverse=True)
        return entries[:n]

    def suggest_config(
        self,
        model_type: Optional[str] = None,
        min_score: float = 60,
    ) -> Optional[Dict[str, Any]]:
        """
        설정 추천

        Args:
            model_type: 선호 모델 타입
            min_score: 최소 점수

        Returns:
            추천 설정
        """
        candidates = self.get_top_configs(n=10, min_score=min_score)

        if model_type:
            # 모델 타입 필터링
            filtered = [
                e for e in candidates
                if e.config.get("model_type") == model_type
            ]
            if filtered:
                return filtered[0].config

        return candidates[0].config if candidates else None

    def get_config_history(self, config_hash: str) -> Dict[str, Any]:
        """설정 히스토리 조회"""
        entry = self._registry.get(config_hash)
        if not entry:
            return {}

        return {
            "config_hash": config_hash,
            "config": entry.config,
            "best_score": entry.best_score,
            "best_experiment_id": entry.best_experiment_id,
            "usage_count": entry.usage_count,
            "success_rate": entry.success_rate,
            "avg_mae": entry.avg_mae,
            "created_at": entry.created_at,
            "last_used_at": entry.last_used_at,
        }

    def print_registry(self, n: int = 10):
        """레지스트리 출력"""
        print("\n" + "=" * 80)
        print("EGAI Config Registry")
        print("=" * 80)
        print(f"{'Hash':<14} {'Model':<15} {'Score':>8} {'MAE':>8} {'Uses':>6} {'Success':>8}")
        print("-" * 80)

        top_configs = self.get_top_configs(n=n)
        for entry in top_configs:
            model = entry.config.get("model_type", "unknown")[:15]
            print(
                f"{entry.config_hash:<14} {model:<15} "
                f"{entry.best_score:>8.1f} {entry.avg_mae:>8.4f} "
                f"{entry.usage_count:>6} {entry.success_rate:>7.0%}"
            )

        print("=" * 80)


# JSON 스키마 정의
AGENT_SCORE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentScore",
    "description": "서브에이전트 평가 점수 스키마",
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "에이전트/실험 식별자"
        },
        "task_type": {
            "type": "string",
            "enum": ["train", "evaluate", "compare", "deploy"],
            "description": "작업 유형"
        },
        "quality_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 40,
            "description": "결과 품질 점수"
        },
        "stability_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 30,
            "description": "안정성 점수"
        },
        "completion_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 30,
            "description": "작업 완료도 점수"
        },
        "total_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "총점"
        },
        "grade": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
            "description": "평가 등급"
        },
        "config_hash": {
            "type": "string",
            "description": "설정 해시 (재활용 식별용)"
        },
        "metrics": {
            "type": "object",
            "description": "상세 메트릭",
            "properties": {
                "mae_mean": {"type": "number"},
                "mae_std": {"type": "number"},
                "status": {"type": "string"},
                "model_type": {"type": "string"},
                "n_folds": {"type": "integer"},
                "epochs": {"type": "integer"},
                "learning_rate": {"type": "number"},
                "batch_size": {"type": "integer"}
            }
        },
        "evaluated_at": {
            "type": "string",
            "format": "date-time",
            "description": "평가 시간"
        }
    },
    "required": [
        "agent_id", "task_type", "quality_score", "stability_score",
        "completion_score", "total_score", "grade", "config_hash",
        "metrics", "evaluated_at"
    ]
}

CONFIG_ENTRY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ConfigEntry",
    "description": "설정 레지스트리 항목 스키마",
    "type": "object",
    "properties": {
        "config_hash": {
            "type": "string",
            "description": "설정 해시 (고유 식별자)"
        },
        "config": {
            "type": "object",
            "description": "설정 값"
        },
        "best_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "최고 점수"
        },
        "best_experiment_id": {
            "type": "string",
            "description": "최고 성능 실험 ID"
        },
        "usage_count": {
            "type": "integer",
            "minimum": 0,
            "description": "사용 횟수"
        },
        "success_rate": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "성공률"
        },
        "avg_mae": {
            "type": "number",
            "description": "평균 MAE"
        },
        "created_at": {
            "type": "string",
            "format": "date-time"
        },
        "last_used_at": {
            "type": "string",
            "format": "date-time"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "config_hash", "config", "best_score", "best_experiment_id",
        "usage_count", "success_rate", "avg_mae", "created_at",
        "last_used_at", "tags"
    ]
}
