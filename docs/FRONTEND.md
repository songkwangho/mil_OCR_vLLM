# 프론트엔드 가이드

> 검토 큐 UI 설계 및 향후 구현 계획을 기술합니다.
> 현재 미구현 — 모든 추론·테스트는 CLI 스크립트로 수행.

---

## 1. 현재 상태

- 프론트엔드 UI 미구현
- 결과 출력: `data/pipeline_outputs/<YYYYMMDD_HHMMSS>/` (TESTING.md §3 참조)
- 백엔드 HTTP 서버 구현 완료 → 향후 UI는 이 엔드포인트들을 호출

---

## 2. 검토 큐 UI MVP (Phase 2-A — 최우선)

> Fine-tuning 착수 전 교정 데이터 확보를 위해 Phase 3에서 Phase 2-A로 앞당김.

### 2-1. 구현 범위

| 기능 | MVP | Phase 3 고도화 |
|------|:---:|:--------------:|
| 큐 목록 조회 (우선순위 정렬) | ✅ | |
| 개별 검토 화면 (이미지 + 필드 수정) | ✅ | |
| 승인/반려 액션 | ✅ | |
| 교정 데이터 JSONL export | ✅ | |
| Other 문서 필터링 (검토 큐에서 제외) | ✅ | |
| 대시보드/통계 | | ✅ |
| 담당자 배정 관리 | | ✅ |
| P2 bbox 오버레이 | | ✅ |
| P2.5-A 정제 결과 시각화 | | ✅ |

### 2-2. 검토 큐 적재 조건

| 조건 | 우선순위 |
|------|---------|
| CRITICAL ValidationError | 🔴 긴급 |
| LOW confidence 필드 (임계값 미달) | 🟡 일반 |
| Fallback 경로 처리 문서 | 🟡 일반 |
| VLM + Fallback 모두 불가 | 🔴 긴급 |
| `form_type = other` 문서 | ❌ 미적재 (군수 업무 대상 아님) |

### 2-3. 큐 목록 화면

- 우선순위: 🔴 긴급 > 🟡 일반
- 컬럼: 큐 ID, 문서 ID, 적재 시각, 우선순위, 사유, form_type, 처리 경로, 상태
- 필터: 상태, 우선순위, form_type, 처리 경로, 날짜 범위

### 2-4. 개별 검토 화면

- **좌측**: 원본 이미지 뷰어
- **우측**: 추출 결과 편집 패널
  - 필드별 값 + 신뢰도 (LOW confidence 🟡 강조)
  - CRITICAL 오류 🔴 인라인 표시
  - form_type + schema_id 표시
  - 담당자 수정 (JSONPath 기반 중첩 필드 지원)
  - CoT analysis 필드 표시 (VLM의 인식 과정 확인)
- **하단**: 승인 / 반려 / 메모

### 2-5. 처리 흐름

```
큐 목록 → 문서 선택 → 검토 화면 (pending → in_review)
    → 원본 이미지 + 자동 추출 결과 비교
    → 오류 필드 수정 (corrected_fields: JSONPath 형식)
    → [승인] → P5 직렬화 → P6 DB 적재 (approved)
    → 교정 데이터 자동 export (SFT/DPO/FormClassifier 형식)
```

### 2-6. API 엔드포인트 (FastAPI MVP)

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/review-queue` | 큐 목록 (other 제외, 우선순위 정렬) |
| GET | `/api/review-queue/{queue_id}` | 개별 항목 상세 |
| PATCH | `/api/review-queue/{queue_id}` | 상태 변경 + 필드 수정 |
| GET | `/api/review-queue/{queue_id}/image` | 원본 이미지 반환 |
| GET | `/api/pipeline/health` | 파이프라인 서비스 상태 |

---

## 3. 문서 처리 UI (Phase 3)

- 문서 이미지 업로드 (단일 / 배치)
- 실시간 파이프라인 진행 상태 (SSE)
- 단계별 결과 시각화:
  - P2: 원시 탐지 bbox + Fusion 모드 표시
  - P2.5-A: 정제 후 bbox (제거/병합 결과 비교)
  - P3-A: form_type + confidence + military/other 분기 표시
  - P3-B: OCR 결과 오버레이, 표 구조 렌더링, CoT analysis 표시
  - P4: 신뢰도 히트맵, 검증 오류 강조

---

## 4. 검토 큐 UI 고도화 (Phase 3)

- 대시보드: 대기 건수, 긴급 건수, 평균 대기 시간
- form_type별 처리 분포 (military 서식 유형, other 비율)
- P2 bbox 오버레이 (클릭 시 해당 영역 하이라이트)
- P2.5-A 정제 시각화 (제거된 박스, 병합된 블록)
- 담당자 배정 관리

---

## 5. 기술 스택 (후보)

- FastAPI (백엔드 API 서버)
- React 또는 Vue.js (프론트엔드)
- WebSocket / SSE (실시간 진행 상태, 검토 큐 알림)
