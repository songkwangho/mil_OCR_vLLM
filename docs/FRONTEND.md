# 프론트엔드 가이드

> 현재 프론트엔드는 미구현 상태입니다.
> 향후 구현 시 이 문서에 기술합니다.

---

## 1. 현재 상태

- 프론트엔드 UI 미구현
- 모든 추론/테스트는 CLI 스크립트로 수행
- 결과 출력은 `data/pipeline_outputs/<trial_id>/` 디렉토리에 JSON + 이미지로 저장

---

## 2. 향후 계획 (Phase 3)

### 2-1. 문서 처리 UI

- 문서 이미지 업로드 (단일 / 배치)
- 실시간 파이프라인 진행 상태 표시
- 단계별 결과 시각화:
  - P1: 전처리 전/후 이미지 비교, SR 적용 여부
  - P2: 레이아웃 영역 bbox 오버레이
  - P3: VLM OCR 결과 텍스트 오버레이, 표 구조 렌더링
  - P4: 신뢰도 히트맵, 검증 오류 강조
  - P5: 추출된 키-값 쌍 테이블

### 2-2. 수동 검토 큐 (필수 — Phase 3 반드시 포함)

> 데이터 모델 및 적재 조건은 `docs/BACKEND.md` §3-3 참조.

**적재 조건 (P4 판정)**:
- CRITICAL `ValidationError` 발생 (산술 불일치, 코드 형식 오류 등)
- LOW confidence 필드 존재 (필드 유형별 임계값 미달 — `docs/BACKEND.md` §5-2)
- Fallback 경로 처리 문서 (`status == "partial"` — 항상 검토 대상)
- VLM + Fallback 모두 불가 (원본 이미지만 적재)

**큐 목록 화면**:
- 우선순위별 정렬: 🔴 긴급 (CRITICAL / 완전 장애) > 🟡 일반 (LOW confidence / fallback)
- 컬럼: 큐 ID, 문서 ID, 적재 시각, 우선순위, 사유, 상태, 담당자
- 필터: 상태(대기/검토 중/완료), 우선순위, 사유, 날짜 범위
- 대시보드: 대기 건수, 긴급 건수, 평균 대기 시간

**개별 검토 화면**:
- **좌측**: 원본 이미지 뷰어 (줌, 패닝, P2 레이아웃 bbox 오버레이)
- **우측**: 추출 결과 편집 패널
  - 필드별 값 + 신뢰도 표시 (LOW confidence 필드 🟡 강조)
  - CRITICAL 오류 메시지 🔴 인라인 표시
  - 필드 클릭 시 원본 이미지의 해당 영역 하이라이트
  - 담당자 직접 수정 가능 (텍스트 인풋)
- **하단**: 검토 액션
  - `승인` → 수정된 값으로 P5 직렬화 → P6 DB 적재
  - `반려` → 재스캔 요청 또는 폐기 사유 입력
  - `메모` → 검토자 코멘트 저장

**처리 흐름 (UI)**:
```
큐 목록 → 문서 선택 → 검토 화면 진입 (status: pending → in_review)
    → 원본 이미지 + 자동 추출 결과 비교
    → 담당자가 오류 필드 수정
    → [승인] 클릭 → P5 → P6 DB 적재 (status: approved)
    → 교정 데이터(원본 → 수정) 자동 축적 → VLM Fine-tuning 학습 데이터
```

**API 엔드포인트** (FastAPI):
| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/review-queue` | 큐 목록 조회 (필터/페이지네이션) |
| GET | `/api/review-queue/{queue_id}` | 개별 항목 상세 |
| PATCH | `/api/review-queue/{queue_id}` | 상태 변경 + 필드 수정 |
| GET | `/api/review-queue/stats` | 큐 통계 |
| GET | `/api/review-queue/{queue_id}/image` | 원본 이미지 반환 |

### 2-3. 학습 데이터 어노테이션

- PPOCRLabel 연동 또는 자체 어노테이션 도구
- 레이아웃 영역 bbox 수정
- OCR 정답 텍스트 입력
- 표 구조 정답 HTML 편집

### 2-4. 기술 스택 (후보)

- FastAPI (백엔드 API 서버)
- React 또는 Vue.js (프론트엔드)
- WebSocket (실시간 진행 상태)

---

## 3. 관련 출력 형식

최종 출력은 `PipelineOutput`으로 JSON / XML / CSV 형태입니다.
상세 스키마는 `docs/BACKEND.md` §2 참조.
