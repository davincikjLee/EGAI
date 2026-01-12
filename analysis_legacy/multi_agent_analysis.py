"""
멀티 에이전트 실험 분석 시스템
역할: 여러 전문가 에이전트가 실험 결과를 분석하고 개선점을 제안

전문가 에이전트:
1. 자동차 전문가: 엔진 소음 도메인 지식 기반 분석
2. 오디오 AI 전문가: 오디오 특징 추출 및 전처리 분석
3. 딥러닝 전문가: 모델 구조 및 학습 전략 분석
4. 실험 설계 전문가: 독립변수 관리 및 실험 설계 분석
5. 통계 전문가: 결과 해석 및 유의성 분석
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import hashlib


@dataclass
class ExperimentResult:
    """실험 결과 데이터 클래스"""
    exp_id: str
    model_type: str
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float
    column_metrics: Dict[str, Dict[str, float]]
    config: Dict
    avg_epochs: float = 0


@dataclass
class AgentSuggestion:
    """에이전트 제안 데이터 클래스 (MLOps 추적용)"""
    suggestion_id: str  # 고유 ID
    agent_name: str
    suggestion_text: str
    category: str  # [데이터], [모델], [전처리], [학습] 등
    created_at: str
    status: str = "pending"  # pending, implemented, success, failed, invalid
    implemented_in_exp: Optional[str] = None  # 구현된 실험 ID
    baseline_mae: Optional[float] = None
    result_mae: Optional[float] = None
    improvement_pct: Optional[float] = None
    notes: str = ""


@dataclass
class AgentPerformance:
    """에이전트 성능 추적 (MLOps 스타일)"""
    agent_name: str
    total_suggestions: int = 0
    implemented: int = 0
    successful: int = 0  # MAE 개선
    failed: int = 0  # MAE 악화
    pending: int = 0
    success_rate: float = 0.0  # successful / implemented
    avg_improvement: float = 0.0  # 평균 개선율
    trust_score: float = 0.5  # 신뢰도 점수 (0-1)


class SuggestionTracker:
    """MLOps 스타일 제안 추적 시스템"""

    def __init__(self, tracker_path: str = "agent_suggestions.json"):
        self.tracker_path = Path(tracker_path)
        self.suggestions: List[AgentSuggestion] = []
        self.agent_performance: Dict[str, AgentPerformance] = {}
        self._load()

    def _generate_id(self, text: str) -> str:
        """제안 텍스트로부터 고유 ID 생성"""
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _load(self):
        """저장된 추적 데이터 로드"""
        if self.tracker_path.exists():
            with open(self.tracker_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.suggestions = [
                    AgentSuggestion(**s) for s in data.get('suggestions', [])
                ]
                for name, perf in data.get('agent_performance', {}).items():
                    self.agent_performance[name] = AgentPerformance(**perf)

    def _save(self):
        """추적 데이터 저장"""
        data = {
            'suggestions': [
                {
                    'suggestion_id': s.suggestion_id,
                    'agent_name': s.agent_name,
                    'suggestion_text': s.suggestion_text,
                    'category': s.category,
                    'created_at': s.created_at,
                    'status': s.status,
                    'implemented_in_exp': s.implemented_in_exp,
                    'baseline_mae': s.baseline_mae,
                    'result_mae': s.result_mae,
                    'improvement_pct': s.improvement_pct,
                    'notes': s.notes
                } for s in self.suggestions
            ],
            'agent_performance': {
                name: {
                    'agent_name': perf.agent_name,
                    'total_suggestions': perf.total_suggestions,
                    'implemented': perf.implemented,
                    'successful': perf.successful,
                    'failed': perf.failed,
                    'pending': perf.pending,
                    'success_rate': perf.success_rate,
                    'avg_improvement': perf.avg_improvement,
                    'trust_score': perf.trust_score
                } for name, perf in self.agent_performance.items()
            },
            'last_updated': datetime.now().isoformat()
        }

        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracker_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_suggestion(self, agent_name: str, suggestion_text: str) -> str:
        """새 제안 추가"""
        # 카테고리 추출
        category = "기타"
        if suggestion_text.startswith("["):
            bracket_end = suggestion_text.find("]")
            if bracket_end > 0:
                category = suggestion_text[1:bracket_end]

        suggestion_id = self._generate_id(f"{agent_name}:{suggestion_text}")

        # 중복 체크
        for s in self.suggestions:
            if s.suggestion_id == suggestion_id:
                return suggestion_id

        suggestion = AgentSuggestion(
            suggestion_id=suggestion_id,
            agent_name=agent_name,
            suggestion_text=suggestion_text,
            category=category,
            created_at=datetime.now().isoformat()
        )
        self.suggestions.append(suggestion)

        # 에이전트 성능 초기화
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = AgentPerformance(agent_name=agent_name)
        self.agent_performance[agent_name].total_suggestions += 1
        self.agent_performance[agent_name].pending += 1

        self._save()
        return suggestion_id

    def mark_implemented(self, suggestion_id: str, exp_id: str, baseline_mae: float):
        """제안이 실험으로 구현됨을 기록"""
        for s in self.suggestions:
            if s.suggestion_id == suggestion_id:
                s.status = "implemented"
                s.implemented_in_exp = exp_id
                s.baseline_mae = baseline_mae

                perf = self.agent_performance.get(s.agent_name)
                if perf:
                    perf.pending -= 1
                    perf.implemented += 1

                self._save()
                return True
        return False

    def record_result(self, suggestion_id: str, result_mae: float, notes: str = ""):
        """실험 결과 기록 및 성공/실패 판정"""
        for s in self.suggestions:
            if s.suggestion_id == suggestion_id and s.baseline_mae is not None:
                s.result_mae = result_mae
                s.notes = notes

                # 개선율 계산
                s.improvement_pct = ((s.baseline_mae - result_mae) / s.baseline_mae) * 100

                # 성공/실패 판정 (MAE 개선 = 성공)
                if result_mae < s.baseline_mae:
                    s.status = "success"
                    if s.agent_name in self.agent_performance:
                        self.agent_performance[s.agent_name].successful += 1
                else:
                    s.status = "failed"
                    if s.agent_name in self.agent_performance:
                        self.agent_performance[s.agent_name].failed += 1

                # 신뢰도 점수 재계산
                self._update_trust_scores()
                self._save()
                return True
        return False

    def _update_trust_scores(self):
        """에이전트 신뢰도 점수 업데이트"""
        for name, perf in self.agent_performance.items():
            if perf.implemented > 0:
                # 성공률
                perf.success_rate = perf.successful / perf.implemented

                # 평균 개선율 계산
                improvements = []
                for s in self.suggestions:
                    if s.agent_name == name and s.improvement_pct is not None:
                        improvements.append(s.improvement_pct)

                if improvements:
                    perf.avg_improvement = sum(improvements) / len(improvements)

                # 신뢰도 점수: 성공률 * 0.7 + 평균개선율 정규화 * 0.3
                # 개선율은 -50% ~ +50% 범위를 0~1로 정규화
                normalized_improvement = (perf.avg_improvement + 50) / 100
                normalized_improvement = max(0, min(1, normalized_improvement))

                perf.trust_score = perf.success_rate * 0.7 + normalized_improvement * 0.3

    def get_agent_leaderboard(self) -> List[Dict]:
        """에이전트 리더보드 (신뢰도 순)"""
        return sorted([
            {
                'agent': name,
                'trust_score': perf.trust_score,
                'success_rate': perf.success_rate,
                'avg_improvement': perf.avg_improvement,
                'implemented': perf.implemented,
                'successful': perf.successful
            }
            for name, perf in self.agent_performance.items()
            if perf.implemented > 0
        ], key=lambda x: x['trust_score'], reverse=True)

    def get_weighted_suggestions(self) -> List[Dict]:
        """신뢰도 가중치가 적용된 미처리 제안 목록"""
        weighted = []
        for s in self.suggestions:
            if s.status == "pending":
                trust = self.agent_performance.get(s.agent_name, AgentPerformance(s.agent_name)).trust_score
                weighted.append({
                    'suggestion_id': s.suggestion_id,
                    'agent': s.agent_name,
                    'text': s.suggestion_text,
                    'category': s.category,
                    'trust_score': trust,
                    'priority': trust  # 높을수록 우선
                })

        return sorted(weighted, key=lambda x: x['priority'], reverse=True)

    def generate_mlops_report(self) -> str:
        """MLOps 스타일 보고서 생성"""
        lines = []
        lines.append("=" * 70)
        lines.append("에이전트 피드백 품질 추적 보고서 (MLOps)")
        lines.append("=" * 70)
        lines.append("")

        # 1. 에이전트 리더보드
        lines.append("## 1. 에이전트 신뢰도 리더보드")
        lines.append("")
        leaderboard = self.get_agent_leaderboard()
        if leaderboard:
            lines.append("| 순위 | 에이전트 | 신뢰도 | 성공률 | 평균개선 | 구현/성공 |")
            lines.append("|------|---------|--------|--------|---------|----------|")
            for i, entry in enumerate(leaderboard, 1):
                lines.append(
                    f"| {i} | {entry['agent'][:15]} | {entry['trust_score']:.2f} | "
                    f"{entry['success_rate']*100:.0f}% | {entry['avg_improvement']:+.1f}% | "
                    f"{entry['implemented']}/{entry['successful']} |"
                )
        else:
            lines.append("아직 구현된 제안이 없습니다.")
        lines.append("")

        # 2. 제안 상태 요약
        lines.append("## 2. 제안 상태 요약")
        status_counts = {}
        for s in self.suggestions:
            status_counts[s.status] = status_counts.get(s.status, 0) + 1

        for status, count in status_counts.items():
            lines.append(f"- {status}: {count}개")
        lines.append("")

        # 3. 최근 성공한 제안
        lines.append("## 3. 성공한 제안들")
        successes = [s for s in self.suggestions if s.status == "success"]
        for s in successes[-5:]:  # 최근 5개
            lines.append(f"- [{s.agent_name}] {s.suggestion_text[:50]}...")
            lines.append(f"  결과: MAE {s.baseline_mae:.4f} → {s.result_mae:.4f} ({s.improvement_pct:+.1f}%)")
        lines.append("")

        # 4. 우선순위 높은 미처리 제안
        lines.append("## 4. 우선순위 높은 미처리 제안 (Top 5)")
        weighted = self.get_weighted_suggestions()[:5]
        for w in weighted:
            lines.append(f"- [신뢰도 {w['trust_score']:.2f}] ({w['agent']}) {w['text'][:60]}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


class ExpertAgent(ABC):
    """전문가 에이전트 기본 클래스"""

    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.analysis_results = []

    @abstractmethod
    def analyze(self, experiments: List[ExperimentResult], baseline: ExperimentResult) -> Dict:
        """실험 결과 분석"""
        pass

    @abstractmethod
    def suggest_improvements(self, analysis: Dict) -> List[str]:
        """개선점 제안"""
        pass


class AutomotiveExpert(ExpertAgent):
    """자동차 전문가 에이전트"""

    def __init__(self):
        super().__init__(
            name="자동차 NVH 전문가",
            expertise="엔진 소음, 진동, 차량 진단"
        )

    def analyze(self, experiments: List[ExperimentResult], baseline: ExperimentResult) -> Dict:
        """자동차 도메인 관점에서 분석"""
        analysis = {
            "domain_insights": [],
            "data_quality_issues": [],
            "feature_relevance": []
        }

        # 1. 점수별 예측 난이도 분석
        if baseline.column_metrics:
            difficulty_ranking = sorted(
                baseline.column_metrics.items(),
                key=lambda x: x[1].get('MAE_mean', 0),
                reverse=True
            )

            for col, metrics in difficulty_ranking:
                mae = metrics.get('MAE_mean', 0)
                if col == 'irregularity':
                    analysis["domain_insights"].append(
                        f"irregularity(비규칙성) MAE {mae:.3f}: 엔진의 불규칙한 연소/기계적 이상 패턴 → "
                        f"스펙트로그램에서 시간적 변동성 포착이 핵심"
                    )
                elif col == 'regularity':
                    analysis["domain_insights"].append(
                        f"regularity(규칙성) MAE {mae:.3f}: 엔진 회전의 주기성 → "
                        f"RPM 기반 주파수 패턴 분석 필요"
                    )
                elif col == 'low_high_freq':
                    analysis["domain_insights"].append(
                        f"low_high_freq MAE {mae:.3f}: 저/고주파 소음 비율 → "
                        f"주파수 대역별 에너지 분포 분석"
                    )
                elif col == 'audable_range_score':
                    analysis["domain_insights"].append(
                        f"audable_range_score MAE {mae:.3f}: 94%가 5점으로 데이터 불균형 심각"
                    )

        # 2. 데이터 품질 이슈
        analysis["data_quality_issues"].append(
            "모든 데이터가 정상 엔진: 고장/이상 데이터 없이는 진정한 진단 AI 불가"
        )
        analysis["data_quality_issues"].append(
            "디젤/가솔린 혼합: 엔진 타입별 소음 특성이 다르므로 분리 학습 권장"
        )

        # 3. 특징 관련성
        analysis["feature_relevance"].append(
            "HPSS(Percussive 성분): 타음 유사 소음(녹킹, 노이즈)에 효과적 - 논문에서 입증"
        )
        analysis["feature_relevance"].append(
            "Modulation Spectrum: 엔진 RPM 변동 감지에 유용하나, 아이들 상태에서는 변동 적음"
        )

        return analysis

    def suggest_improvements(self, analysis: Dict) -> List[str]:
        """자동차 도메인 관점 개선 제안"""
        suggestions = []

        suggestions.append(
            "[데이터] 고장 엔진 오디오 수집: 오토텐셔너, 캠샤프트, 타이밍체인 등 부품별 이상 소음"
        )
        suggestions.append(
            "[데이터] 디젤/가솔린 분리 학습: 엔진 타입별 소음 특성 차이 반영"
        )
        suggestions.append(
            "[특징] RPM 정보 활용: 아이들 RPM에 맞춘 프레임 길이 설정 (논문: 150ms)"
        )
        suggestions.append(
            "[특징] 주파수 대역별 분석: 저주파(엔진 진동), 중주파(기계음), 고주파(마찰/노이즈) 분리"
        )

        return suggestions


class AudioAIExpert(ExpertAgent):
    """오디오 AI 전문가 에이전트"""

    def __init__(self):
        super().__init__(
            name="오디오 신호처리 전문가",
            expertise="오디오 특징 추출, 스펙트로그램, STFT"
        )

    def analyze(self, experiments: List[ExperimentResult], baseline: ExperimentResult) -> Dict:
        """오디오 처리 관점에서 분석"""
        analysis = {
            "preprocessing_analysis": [],
            "feature_analysis": [],
            "spectral_insights": []
        }

        # 전처리 분석
        analysis["preprocessing_analysis"].append(
            "현재 STFT: n_fft=2048, hop=512, sr=22050 → 논문: FFT=4096, Frame=150ms, sr=16000"
        )
        analysis["preprocessing_analysis"].append(
            "HPSS 적용 중: Percussive 성분 분리 (타음 유사 소음 감지용)"
        )

        # 특징 분석
        for exp in experiments:
            if 'physics' in exp.model_type:
                analysis["feature_analysis"].append(
                    f"물리량 특징 14개 추가 → MAE {exp.mae_mean:.3f} (악화): "
                    f"스펙트로그램에 이미 포함된 정보와 중복"
                )
            elif 'modulation' in exp.model_type:
                analysis["feature_analysis"].append(
                    f"모듈레이션 특징 20개 추가 → MAE {exp.mae_mean:.3f} (악화): "
                    f"아이들 상태에서 변동성 낮아 효과 미미"
                )

        # 스펙트럼 인사이트
        analysis["spectral_insights"].append(
            "스펙트로그램이 '초록색으로 비슷해 보임': 정규화 효과 + 정상 엔진만 수집"
        )
        analysis["spectral_insights"].append(
            "수치적 차이는 존재: CNN은 미세한 패턴 차이도 학습 가능"
        )

        return analysis

    def suggest_improvements(self, analysis: Dict) -> List[str]:
        """오디오 처리 관점 개선 제안"""
        suggestions = []

        suggestions.append(
            "[전처리] 논문 파라미터 적용: FFT=4096, Frame=150ms, Hop=75ms, SR=16kHz"
        )
        suggestions.append(
            "[전처리] Mel-Spectrogram 시도: 인간 청각 특성 반영, 고주파 압축"
        )
        suggestions.append(
            "[특징] 주파수 밴드별 에너지: 0-500Hz, 500-2000Hz, 2000-6000Hz 분리"
        )
        suggestions.append(
            "[특징] Temporal Modulation: 시간에 따른 에너지 변화 패턴"
        )

        return suggestions


class DeepLearningExpert(ExpertAgent):
    """딥러닝 전문가 에이전트"""

    def __init__(self):
        super().__init__(
            name="딥러닝 모델 전문가",
            expertise="CNN, Attention, 모델 아키텍처"
        )

    def analyze(self, experiments: List[ExperimentResult], baseline: ExperimentResult) -> Dict:
        """모델 구조 관점에서 분석"""
        analysis = {
            "model_comparison": [],
            "architecture_insights": [],
            "training_analysis": []
        }

        # 모델 비교
        for exp in experiments:
            improvement = ((exp.mae_mean - baseline.mae_mean) / baseline.mae_mean) * 100
            status = "개선" if improvement < 0 else "악화"
            analysis["model_comparison"].append({
                "model": exp.model_type,
                "mae": exp.mae_mean,
                "change": f"{improvement:+.1f}% {status}",
                "epochs": exp.avg_epochs
            })

        # 아키텍처 인사이트
        analysis["architecture_insights"].append(
            "Simple CNN이 최고 성능: 594개 데이터에서 복잡한 모델은 과적합"
        )
        analysis["architecture_insights"].append(
            "CBAM 효과: 논문에서 회귀 태스크 +8% 향상 → 시도 가치 있음"
        )
        analysis["architecture_insights"].append(
            "GlobalAvgPool vs Flatten: GlobalAvgPool이 과적합 방지에 효과적"
        )

        # 학습 분석
        analysis["training_analysis"].append(
            f"Early Stopping: 평균 {baseline.avg_epochs:.0f} 에포크에서 종료"
        )
        analysis["training_analysis"].append(
            "Dropout 0.3-0.4 적용 중: 적절한 정규화"
        )

        return analysis

    def suggest_improvements(self, analysis: Dict) -> List[str]:
        """딥러닝 관점 개선 제안"""
        suggestions = []

        suggestions.append(
            "[모델] Simple CNN + CBAM: 최소 변경으로 Attention 효과 검증"
        )
        suggestions.append(
            "[모델] 채널 결합 방식: 논문 Fig.6처럼 Full+Percussive 채널 결합"
        )
        suggestions.append(
            "[학습] Label Smoothing: 정수 점수 예측 시 일반화 향상"
        )
        suggestions.append(
            "[학습] Mixup/CutMix: 데이터 증강으로 일반화 향상"
        )
        suggestions.append(
            "[출력] Ordinal Regression: 순서형 점수(1-5) 예측에 최적화된 손실함수"
        )

        return suggestions


class ExperimentDesignExpert(ExpertAgent):
    """실험 설계 전문가 에이전트"""

    def __init__(self):
        super().__init__(
            name="실험 설계 전문가",
            expertise="독립변수 관리, 실험 설계, 재현성"
        )

    def analyze(self, experiments: List[ExperimentResult], baseline: ExperimentResult) -> Dict:
        """실험 설계 관점에서 분석"""
        analysis = {
            "independent_variables": [],
            "confounding_factors": [],
            "reproducibility": []
        }

        # 독립변수 분석
        analysis["independent_variables"].append({
            "variable": "모델 구조",
            "levels": [exp.model_type for exp in experiments],
            "controlled": True,
            "note": "각 실험에서 하나의 모델만 변경"
        })
        analysis["independent_variables"].append({
            "variable": "추가 특징",
            "levels": ["없음(baseline)", "물리량 14개", "메타데이터 14개", "모듈레이션 20개"],
            "controlled": True,
            "note": "특징 수와 종류 기록됨"
        })
        analysis["independent_variables"].append({
            "variable": "전처리 파라미터",
            "levels": ["기본값"],
            "controlled": False,
            "note": "아직 변경하지 않음 - 실험 필요"
        })

        # 교란 변수
        analysis["confounding_factors"].append(
            "랜덤 시드: 5-Fold CV로 완화되었으나 fold별 편차 존재 (±0.09)"
        )
        analysis["confounding_factors"].append(
            "데이터 분할: Stratified 분할 사용하나 audable_range 94%가 5점"
        )
        analysis["confounding_factors"].append(
            "엔진 타입 혼합: 디젤/가솔린 미분리 → 숨겨진 교란 변수"
        )

        # 재현성
        analysis["reproducibility"].append(
            "실험 로깅: experiment_logger.py로 모든 설정 기록"
        )
        analysis["reproducibility"].append(
            "결과 저장: experiments/ 폴더에 JSON 형태로 저장"
        )

        return analysis

    def suggest_improvements(self, analysis: Dict) -> List[str]:
        """실험 설계 관점 개선 제안"""
        suggestions = []

        suggestions.append(
            "[설계] 전처리 파라미터 실험: FFT size, Frame length를 독립변수로"
        )
        suggestions.append(
            "[설계] 엔진 타입 분리 실험: 디젤/가솔린 각각 학습 후 비교"
        )
        suggestions.append(
            "[설계] 단일 점수 예측 실험: 5개 동시 예측 vs 1개씩 예측 비교"
        )
        suggestions.append(
            "[통제] 랜덤 시드 고정: 모든 실험에서 동일 시드 사용"
        )
        suggestions.append(
            "[기록] 하이퍼파라미터 그리드: learning_rate, batch_size 등 체계적 탐색"
        )

        return suggestions


class StatisticsExpert(ExpertAgent):
    """통계 전문가 에이전트"""

    def __init__(self):
        super().__init__(
            name="통계 분석 전문가",
            expertise="결과 해석, 유의성 검정, 신뢰구간"
        )

    def analyze(self, experiments: List[ExperimentResult], baseline: ExperimentResult) -> Dict:
        """통계적 관점에서 분석"""
        analysis = {
            "significance_tests": [],
            "confidence_intervals": [],
            "effect_sizes": []
        }

        # 유의성 분석 (간단한 비교)
        for exp in experiments:
            # MAE 차이가 표준편차보다 큰지 확인
            mae_diff = abs(exp.mae_mean - baseline.mae_mean)
            combined_std = (exp.mae_std + baseline.mae_std) / 2

            if mae_diff > 2 * combined_std:
                significance = "유의미한 차이"
            elif mae_diff > combined_std:
                significance = "경계선상의 차이"
            else:
                significance = "유의미하지 않은 차이"

            analysis["significance_tests"].append({
                "comparison": f"Baseline vs {exp.model_type}",
                "mae_diff": mae_diff,
                "combined_std": combined_std,
                "conclusion": significance
            })

        # 신뢰구간
        for exp in experiments:
            ci_lower = exp.mae_mean - 1.96 * exp.mae_std
            ci_upper = exp.mae_mean + 1.96 * exp.mae_std
            analysis["confidence_intervals"].append({
                "model": exp.model_type,
                "mae_mean": exp.mae_mean,
                "ci_95": f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            })

        # 효과 크기
        analysis["effect_sizes"].append(
            f"Baseline 표준편차: {baseline.mae_std:.4f} (fold 간 변동)"
        )
        analysis["effect_sizes"].append(
            "CV 사용으로 단일 분할 대비 신뢰성 향상"
        )

        return analysis

    def suggest_improvements(self, analysis: Dict) -> List[str]:
        """통계 관점 개선 제안"""
        suggestions = []

        suggestions.append(
            "[검정] Paired t-test: 동일 fold에서 모델 간 비교"
        )
        suggestions.append(
            "[검정] Wilcoxon signed-rank: 비모수 검정으로 robustness 확인"
        )
        suggestions.append(
            "[분석] Bootstrap: 신뢰구간 추정 정확도 향상"
        )
        suggestions.append(
            "[보고] Effect Size: Cohen's d로 실질적 차이 크기 보고"
        )

        return suggestions


class MultiAgentAnalyzer:
    """멀티 에이전트 분석 조정자"""

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experts = [
            AutomotiveExpert(),
            AudioAIExpert(),
            DeepLearningExpert(),
            ExperimentDesignExpert(),
            StatisticsExpert()
        ]
        self.experiments = []
        self.baseline = None

        # MLOps 추적 시스템 초기화
        tracker_path = Path(experiments_dir).parent / "doc" / "agent_suggestions.json"
        self.tracker = SuggestionTracker(str(tracker_path))

    def load_experiments(self) -> List[ExperimentResult]:
        """실험 결과 로드"""
        experiments = []

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            metrics_file = exp_dir / "metrics.json"
            if not metrics_file.exists():
                # 구 형식 확인
                final_results = exp_dir / "final_results.json"
                if final_results.exists():
                    with open(final_results) as f:
                        data = json.load(f)
                else:
                    continue
            else:
                with open(metrics_file) as f:
                    data = json.load(f)

            # config 로드
            config_file = exp_dir / "config.json"
            config = {}
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

            exp = ExperimentResult(
                exp_id=exp_dir.name,
                model_type=config.get('model_type', exp_dir.name.split('_')[1] if '_' in exp_dir.name else 'unknown'),
                mae_mean=data.get('overall_MAE_mean', data.get('overall_MAE', 0)),
                mae_std=data.get('overall_MAE_std', 0),
                r2_mean=data.get('overall_R2_mean', data.get('overall_R2', 0)),
                r2_std=data.get('overall_R2_std', 0),
                column_metrics=data.get('column_metrics', {}),
                config=config,
                avg_epochs=data.get('avg_epochs', 0)
            )
            experiments.append(exp)

        self.experiments = experiments

        # Baseline 찾기 (simple 모델 또는 MAE가 가장 낮은 것)
        simple_exps = [e for e in experiments if 'simple' in e.model_type.lower() and 'cbam' not in e.model_type.lower()]
        if simple_exps:
            self.baseline = min(simple_exps, key=lambda x: x.mae_mean)
        elif experiments:
            self.baseline = min(experiments, key=lambda x: x.mae_mean)

        return experiments

    def run_analysis(self, track_suggestions: bool = True) -> Dict:
        """모든 전문가 에이전트의 분석 실행"""
        if not self.experiments:
            self.load_experiments()

        if not self.baseline:
            return {"error": "Baseline 실험을 찾을 수 없습니다."}

        all_analyses = {}
        all_suggestions = {}
        suggestion_ids = {}

        for expert in self.experts:
            print(f"\n분석 중: {expert.name} ({expert.expertise})")
            analysis = expert.analyze(self.experiments, self.baseline)
            suggestions = expert.suggest_improvements(analysis)

            all_analyses[expert.name] = analysis
            all_suggestions[expert.name] = suggestions

            # MLOps: 제안 추적 시스템에 등록
            if track_suggestions:
                suggestion_ids[expert.name] = []
                for suggestion in suggestions:
                    sid = self.tracker.add_suggestion(expert.name, suggestion)
                    suggestion_ids[expert.name].append(sid)

        return {
            "baseline": {
                "model": self.baseline.model_type,
                "mae": self.baseline.mae_mean,
                "std": self.baseline.mae_std
            },
            "experiments_count": len(self.experiments),
            "analyses": all_analyses,
            "suggestions": all_suggestions,
            "suggestion_ids": suggestion_ids if track_suggestions else {}
        }

    def link_experiment_to_suggestion(self, suggestion_id: str, exp_id: str):
        """실험을 특정 제안에 연결"""
        if self.baseline:
            self.tracker.mark_implemented(suggestion_id, exp_id, self.baseline.mae_mean)

    def record_experiment_result(self, suggestion_id: str, result_mae: float, notes: str = ""):
        """실험 결과를 제안에 기록"""
        self.tracker.record_result(suggestion_id, result_mae, notes)

    def get_prioritized_suggestions(self) -> List[Dict]:
        """신뢰도 기반 우선순위 제안 목록"""
        return self.tracker.get_weighted_suggestions()

    def get_agent_leaderboard(self) -> List[Dict]:
        """에이전트 신뢰도 리더보드"""
        return self.tracker.get_agent_leaderboard()

    def generate_report(self) -> str:
        """종합 분석 보고서 생성"""
        results = self.run_analysis()

        report = []
        report.append("=" * 80)
        report.append("EGAI 멀티 에이전트 실험 분석 보고서")
        report.append("=" * 80)
        report.append("")

        # 기준선 정보
        report.append("## 1. 기준선 (Baseline)")
        report.append(f"- 모델: {results['baseline']['model']}")
        report.append(f"- MAE: {results['baseline']['mae']:.4f} ± {results['baseline']['std']:.4f}")
        report.append(f"- 총 실험 수: {results['experiments_count']}")
        report.append("")

        # 실험 결과 요약
        report.append("## 2. 실험 결과 요약")
        report.append("")
        report.append("| 모델 | MAE | Baseline 대비 |")
        report.append("|------|-----|--------------|")

        for exp in sorted(self.experiments, key=lambda x: x.mae_mean):
            change = ((exp.mae_mean - self.baseline.mae_mean) / self.baseline.mae_mean) * 100
            status = "개선" if change < 0 else "악화" if change > 0 else "-"
            report.append(f"| {exp.model_type} | {exp.mae_mean:.4f} | {change:+.1f}% {status} |")
        report.append("")

        # 전문가별 분석
        report.append("## 3. 전문가별 분석")
        report.append("")

        for expert_name, analysis in results['analyses'].items():
            report.append(f"### {expert_name}")
            for category, items in analysis.items():
                if items:
                    report.append(f"\n**{category}:**")
                    for item in items:
                        if isinstance(item, dict):
                            report.append(f"- {json.dumps(item, ensure_ascii=False)}")
                        else:
                            report.append(f"- {item}")
            report.append("")

        # 개선 제안
        report.append("## 4. 종합 개선 제안")
        report.append("")

        all_suggestions = []
        for expert_name, suggestions in results['suggestions'].items():
            for s in suggestions:
                all_suggestions.append(f"({expert_name}) {s}")

        # 중복 제거 및 우선순위 정렬
        for i, suggestion in enumerate(all_suggestions, 1):
            report.append(f"{i}. {suggestion}")

        report.append("")

        # MLOps 섹션 추가
        report.append("## 5. MLOps 피드백 품질 추적")
        report.append("")
        report.append(self.tracker.generate_mlops_report())
        report.append("")

        report.append("=" * 80)
        report.append("보고서 생성 완료")
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, output_path: str = "multi_agent_analysis_report.md"):
        """보고서 저장"""
        report = self.generate_report()

        output_file = self.experiments_dir.parent / "doc" / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n보고서 저장: {output_file}")
        return str(output_file)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='EGAI 멀티 에이전트 분석 시스템')
    parser.add_argument('exp_dir', nargs='?', default='experiments', help='실험 디렉토리')
    parser.add_argument('--mlops', action='store_true', help='MLOps 피드백 보고서만 출력')
    parser.add_argument('--leaderboard', action='store_true', help='에이전트 리더보드만 출력')
    parser.add_argument('--priorities', action='store_true', help='우선순위 제안 목록만 출력')
    parser.add_argument('--link', nargs=2, metavar=('SID', 'EXP_ID'), help='제안을 실험에 연결')
    parser.add_argument('--result', nargs=3, metavar=('SID', 'MAE', 'NOTES'), help='실험 결과 기록')

    args = parser.parse_args()

    print("=" * 60)
    print("EGAI 멀티 에이전트 분석 시스템 (MLOps 확장)")
    print("=" * 60)

    analyzer = MultiAgentAnalyzer(args.exp_dir)

    # 특수 명령 처리
    if args.link:
        sid, exp_id = args.link
        analyzer.load_experiments()
        analyzer.link_experiment_to_suggestion(sid, exp_id)
        print(f"제안 {sid}를 실험 {exp_id}에 연결했습니다.")
        sys.exit(0)

    if args.result:
        sid, mae, notes = args.result
        analyzer.record_experiment_result(sid, float(mae), notes)
        print(f"제안 {sid}에 결과 MAE={mae} 기록했습니다.")
        sys.exit(0)

    # 실험 로드
    experiments = analyzer.load_experiments()
    print(f"\n로드된 실험 수: {len(experiments)}")

    if args.mlops:
        print("\n" + analyzer.tracker.generate_mlops_report())
        sys.exit(0)

    if args.leaderboard:
        print("\n에이전트 신뢰도 리더보드:")
        for i, entry in enumerate(analyzer.get_agent_leaderboard(), 1):
            print(f"{i}. {entry['agent']}: 신뢰도 {entry['trust_score']:.2f}, "
                  f"성공률 {entry['success_rate']*100:.0f}%")
        sys.exit(0)

    if args.priorities:
        print("\n우선순위 높은 제안 (신뢰도 기반):")
        for w in analyzer.get_prioritized_suggestions()[:10]:
            print(f"- [{w['trust_score']:.2f}] ({w['agent']}) {w['text'][:60]}")
        sys.exit(0)

    if experiments:
        # 분석 실행 및 보고서 생성
        report_path = analyzer.save_report()

        # 콘솔에도 출력
        print("\n" + analyzer.generate_report())
