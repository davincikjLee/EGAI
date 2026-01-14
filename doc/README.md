# EGAI 프로젝트 문서

## 문서 구조

```
doc/
├── README.md                    # 이 파일 (문서 인덱스)
├── IDENTIFIED_ISSUES.md         # 식별된 기술적 이슈
├── SERVICE_STRATEGY.md          # 서비스 전략 문서
├── PROJECT_RESPONSE_REPORT.md   # 요구사항 대응 보고서 (2025.12.23)
│
├── share/                       # 고객 전달용 문서
│   └── PROJECT_RESPONSE_REPORT.md  # 전승철 고객 전달
│
├── reports/                     # 공식 보고서
│   ├── PROGRESS_REPORT.md       # 진행 상황 보고서
│   ├── PROGRESS_REPORT.pdf      # PDF 버전
│   ├── PROJECT_CONCLUSION.md    # 프로젝트 결론
│   ├── KEY_FINDINGS.md          # 핵심 발견사항
│   └── REPORT_JSC_COMPARISON.md # JSC 비교 분석
│
├── research/                    # 연구 문서
│   ├── PAPER_SUMMARY.md         # 논문 요약 (현대차 2023)
│   ├── EXPERT_PANEL_V1.md       # AI 전문가 패널 토론 v1
│   ├── EXPERT_PANEL_V2.md       # AI 전문가 패널 토론 v2
│   ├── EXPERT_REGULARITY.md     # 규칙성 분석 전문가 토론
│   └── REVERSE_ENGINEERING.md   # 역설계 분석
│
└── archive/                     # 아카이브 (참조용)
    ├── legacy/                  # 레거시 문서
    │   ├── PROJECT_OVERVIEW_v1.md
    │   ├── CRAWLER_MODULE_v1.md
    │   └── ANALYSIS_MODULE_v1.md
    │
    └── experiment_logs/         # 실험 로그
        ├── EXPERIMENT_LOG.md
        ├── EXPERIMENT_PHASE2.md
        ├── EXPERIMENT_PHASE3.md
        └── EXPERIMENT_ANALYSIS_DEEP.md
```

## 주요 문서 안내

### 현재 활성 문서

| 문서 | 설명 | 대상 |
|------|------|------|
| [PROJECT_RESPONSE_REPORT.md](PROJECT_RESPONSE_REPORT.md) | 2025.12.23 요구사항 대응 | 현대차 담당자 |
| [SERVICE_STRATEGY.md](SERVICE_STRATEGY.md) | 서비스 모델 및 사업화 전략 | 기획팀 |
| [IDENTIFIED_ISSUES.md](IDENTIFIED_ISSUES.md) | 식별된 기술적 이슈 및 해결방안 | 개발팀 |

### 보고서 (reports/)

| 문서 | 설명 |
|------|------|
| [PROGRESS_REPORT.md](reports/PROGRESS_REPORT.md) | 전체 진행 상황 요약 |
| [PROJECT_CONCLUSION.md](reports/PROJECT_CONCLUSION.md) | 프로젝트 결론 및 권고사항 |
| [KEY_FINDINGS.md](reports/KEY_FINDINGS.md) | 핵심 발견사항 정리 |

### 연구 문서 (research/)

| 문서 | 설명 |
|------|------|
| [PAPER_SUMMARY.md](research/PAPER_SUMMARY.md) | 현대차 2023 논문 분석 |
| [EXPERT_PANEL_V2.md](research/EXPERT_PANEL_V2.md) | AI 전문가 패널 토론 (최신) |

## 빠른 참조

### 현재 모델 성능

```
Multi-head CBAM 모델 (2026.01.14)
- 전체 MAE: 0.4578 ± 0.018
- 개선율: Baseline 대비 18.8% 개선
```

### 식별된 주요 이슈

1. **출력 범위 제한 없음**: 1~5점 범위 초과 예측 가능
2. **Edge Effect**: 희소 클래스(2점) 예측 정확도 낮음
3. **데이터 불균형**: 62%가 5점으로 편중

### 서비스 전략 요약

- **포지셔닝**: "점수 예측" → "품질 보증 인증"
- **하이브리드 모델**: Autoencoder(이상탐지) + CNN(상세점수)
- **수익 모델**: 무료 스크리닝 → 유료 상세 리포트

## 문서 작성 가이드

### 새 문서 추가 시

1. **고객 전달**: `share/` 폴더에 추가
2. **보고서**: `reports/` 폴더에 추가
3. **연구/분석**: `research/` 폴더에 추가
4. **실험 로그**: `archive/experiment_logs/` 폴더에 추가
5. **더 이상 사용하지 않는 문서**: `archive/legacy/`로 이동

### 명명 규칙

- 대문자 + 언더스코어 (예: `PROJECT_REPORT.md`)
- 버전이 있는 경우: `_V1`, `_V2` 접미사
- 날짜가 중요한 경우: `YYYYMMDD_` 접두사

---

*마지막 업데이트: 2026-01-14*
