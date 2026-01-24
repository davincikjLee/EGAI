# EGAI 프로젝트 문서

**최종 업데이트**: 2026-01-24

---

## 문서 구조

```text
doc/
├── README.md                    # 이 파일 (문서 인덱스)
│
├── [핵심 문서]
├── CRAWLER_MODULE.md            # 크롤러 모듈 기술 문서
├── IDENTIFIED_ISSUES.md         # 식별된 기술적 이슈 ⭐ Phase 3 결과 추가
├── PROJECT_RESPONSE_REPORT.md   # 요구사항 대응 보고서
├── SERVICE_STRATEGY.md          # 서비스 전략 문서
│
├── share/                       # 고객 전달용 문서
│   └── PROJECT_RESPONSE_REPORT.md
│
├── reports/                     # 공식 보고서
│   └── PROGRESS_REPORT.md       # 진행 현황 보고서 (최신)
│
├── research/                    # 연구 문서
│   ├── OK_NG_MODEL_PLAN.md      # OK/NG 모델 계획 (진행중)
│   ├── SAMPLE_TEST_REPORT.md    # 샘플 테스트 결과 ⭐ 최신
│   ├── SAMPLE_TEST_PROCEDURE.md # 테스트 절차
│   ├── EXPERT_PANEL_V2.md       # AI 전문가 패널 토론
│   ├── EXPERT_REGULARITY.md     # 규칙성 분석 토론
│   ├── PAPER_SUMMARY.md         # 논문 요약 (현대차 2023)
│   └── REVERSE_ENGINEERING.md   # 역설계 분석
│
└── archive/                     # 아카이브 (참조용)
    ├── legacy/                  # 레거시 문서 (3개)
    ├── experiment_logs/         # 실험 로그 (4개)
    └── outdated_reports/        # 구버전 보고서 (4개)
```

---

## 현재 상태 요약 (2026-01-24)

### Phase 3 테스트 결과

| 차종 | 의도된 라벨 | 실제 판정 | 상태 |
|------|-------------|-----------|------|
| 디올뉴팰리세이드 | 정상 | 정상 | ✅ |
| 코나 | 정상 | 정상 | ✅ |
| **제네시스** | **이상** | **정상** | ❌ |

### 핵심 문제

```text
⚠️ 이상 차량(제네시스)이 정상으로 분류됨

원인: 학습 데이터 100%가 정상 → 이상 탐지 구조적 불가
해결: 이상 데이터 50~100개 수집 필요
```

### 다음 단계

- [ ] 의뢰자에게 이상 데이터 요청
- [ ] 제네시스 고장 유형 확인
- [ ] 이상 데이터 확보 시 OK/NG 모델 개발

---

## 주요 문서 안내

### 필수 참조 문서

| 문서 | 설명 | 최근 업데이트 |
|------|------|---------------|
| [IDENTIFIED_ISSUES.md](IDENTIFIED_ISSUES.md) | 식별된 기술적 이슈 | 01-24 |
| [research/SAMPLE_TEST_REPORT.md](research/SAMPLE_TEST_REPORT.md) | Phase 3 테스트 결과 | 01-24 |
| [research/OK_NG_MODEL_PLAN.md](research/OK_NG_MODEL_PLAN.md) | OK/NG 모델 개발 계획 | 01-14 |
| [reports/PROGRESS_REPORT.md](reports/PROGRESS_REPORT.md) | 전체 진행 현황 | 01-24 |

### 기술 문서

| 문서 | 설명 |
|------|------|
| [CRAWLER_MODULE.md](CRAWLER_MODULE.md) | 크롤러 모듈 (2,239개 데이터 수집) |
| [SERVICE_STRATEGY.md](SERVICE_STRATEGY.md) | 서비스 모델 및 사업화 전략 |
| [PROJECT_RESPONSE_REPORT.md](PROJECT_RESPONSE_REPORT.md) | 현대차 요구사항 대응 |

### 연구 문서 (research/)

| 문서 | 설명 |
|------|------|
| [PAPER_SUMMARY.md](research/PAPER_SUMMARY.md) | 현대차 2023 논문 분석 |
| [EXPERT_PANEL_V2.md](research/EXPERT_PANEL_V2.md) | AI 전문가 패널 토론 |
| [REVERSE_ENGINEERING.md](research/REVERSE_ENGINEERING.md) | 점수 체계 역설계 분석 |

---

## 모델 성능 현황

### Regression 모델 (품질 점수 예측)

```text
Multi-head CBAM 모델 (2026-01-14)
- 전체 MAE: 0.4578 ± 0.018
- 개선율: Baseline 대비 18.8% 개선
- 한계: 이상 데이터 없이 이상 탐지 불가
```

### VAE 모델 (이상탐지)

```text
VAE (2026-01-15)
- 데이터: 1,762개 (4점 이상 정상 데이터)
- VAE Score: 150.99 ± 23.92
- 한계: 녹음 환경 차이만 탐지 (도메인 갭)
```

---

## 문서 작성 가이드

### 새 문서 추가 시

| 유형 | 폴더 | 예시 |
|------|------|------|
| 고객 전달 | `share/` | 보고서, 제안서 |
| 공식 보고서 | `reports/` | 진행 현황, 결론 |
| 연구/분석 | `research/` | 실험 계획, 분석 결과 |
| 더 이상 사용 안 함 | `archive/` | 구버전 문서 |

### 명명 규칙

- 대문자 + 언더스코어 (예: `PROJECT_REPORT.md`)
- 버전 표기: `_V1`, `_V2` 접미사
- 날짜 표기: `YYYYMMDD_` 접두사 (필요시)

---

*마지막 업데이트: 2026-01-24*
