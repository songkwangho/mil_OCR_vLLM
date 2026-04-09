# 프론트엔드 가이드

> 현재 프론트엔드는 미구현 상태입니다.
> 향후 구현 시 이 문서에 기술합니다.

---

## 1. 현재 상태

- 프론트엔드 UI 미구현
- 모든 추론/테스트는 CLI 스크립트로 수행
- 결과 출력은 `data/pipeline_outputs/<trial_id>/` 디렉토리에 JSON + 이미지로 저장

---

## 2. 향후 계획

### 2-1. 검토 큐 UI MVP (Phase 2-A — 최우선)

> Fine-tuning 착수 전 교정 데이터 확보를 위해 Phase 3에서 Phase 2-A로 앞당김.
> 데이터 모델 및 적재 조건: `docs/BACKEND.md` §4-3 참조.

**MVP 구현 범위** (Phase 2-A):

| 기능 | MVP | Phase 3 고도화 |
|------|:---:|:--------------:|
| 큐 목록 조회 (우선순위 정렬) | ✅ | |
| 개별 검토 화면 (이미지 + 필드 수정) | ✅ | |
| 승인/반려 액션 | ✅ | |
| 교정 데이터 JSONL export | ✅ | |
| 대시보드/통계 | | ✅ |
| 담당자 배정 관리 | | ✅ |
| P2 레이아웃 bbox 오버레이 | | ✅ |
| P2.5-A 정제 결과 시각화 | | ✅ |

**적재 조건 (P4 판정)**:
- CRITICAL `ValidationError` 발생
- LOW confidence 필드 존재 (경로별 임계값 미달 — `docs/BACKEND.md` §6-2)
- Fallback 경로 처리 문서 (`processing_path == "fallback"` — 항상)
- VLM + Fallback 모두 불가 (원본 이미지만 적재)

**큐 목록 화면**:
- 우선순위: 🔴 긴급 (CRITICAL / 완전 장애) > 🟡 일반 (LOW confidence / fallback)
- 컬럼: 큐 ID, 문서 ID, 적재 시각, 우선순위, 사유, 처리 경로(vlm/fallback), 상태
- 필터: 상태, 우선순위, 사유, 처리 경로, 날짜 범위

**개별 검토 화면**:
- **좌측**: 원본 이미지 뷰어
- **우측**: 추출 결과 편집 패널
  - 필드별 값 + 신뢰도 표시 (LOW confidence 🟡 강조)
  - CRITICAL 오류 🔴 인라인 표시
  - form_type + schema_id 표시 (서식 분류 결과 확인)
  - 담당자 직접 수정 (JSONPath 기반 중첩 필드 지원)
- **하단**: 승인 / 반려 / 메모

**처리 흐름**:
```
큐 목록 → 문서 선택 → 검토 화면 (status: pending → in_review)
    → 원본 이미지 + 자동 추출 결과 비교
    → 오류 필드 수정 (corrected_fields: JSONPath 형식)
    → [승인] → P5 직렬화 → P6 DB 적재 (status: approved)
    → 교정 데이터 자동 export (SFT/DPO/FormClassifier 형식)
```

**API 엔드포인트** (FastAPI MVP):
| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/review-queue` | 큐 목록 조회 |
| GET | `/api/review-queue/{queue_id}` | 개별 항목 상세 |
| PATCH | `/api/review-queue/{queue_id}` | 상태 변경 + 필드 수정 |
| GET | `/api/review-queue/{queue_id}/image` | 원본 이미지 반환 |

### 2-2. 문서 처리 UI (Phase 3)

- 문서 이미지 업로드 (단일 / 배치)
- 실시간 파이프라인 진행 상태 표시
- 단계별 결과 시각화:
  - P1: 전처리 전/후 이미지 비교, SR 적용 여부
  - P2: 레이아웃 원시 탐지 bbox 오버레이
  - P2.5-A: 정제 후 bbox (제거/병합 결과 비교)
  - P3-A: 서식 분류 결과 (form_type + confidence)
  - P3-B: VLM OCR 결과 텍스트 오버레이, 표 구조 렌더링
  - P4: 신뢰도 히트맵, 검증 오류 강조

### 2-3. 검토 큐 UI 고도화 (Phase 3)

- 대시보드: 대기 건수, 긴급 건수, 평균 대기 시간, 처리 경로별 분포
- P2 레이아웃 bbox 오버레이 (클릭 시 해당 영역 하이라이트)
- P2.5-A 정제 결과 시각화 (제거된 박스, 병합된 블록 표시)
- 담당자 배정 관리

### 2-4. 기술 스택 (후보)

- FastAPI (백엔드 API 서버)
- React 또는 Vue.js (프론트엔드)
- WebSocket / SSE (실시간 진행 상태, 검토 큐 알림)

---

## 3. 관련 출력 형식

최종 출력은 `PipelineOutput`으로 JSON / XML / CSV 형태.
상세 스키마: `docs/BACKEND.md` §2 참조.
