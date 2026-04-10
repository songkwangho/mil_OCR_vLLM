# 통합 테스트 가이드

> 통합 테스트 실행 방식, 결과 저장 구조, 타이밍 기록, 콘솔 출력 원칙을 기술합니다.

---

## 1. 실행 방식

**옵션 A — 호스트 conda 환경 직접 실행** (권장, Layout + vLLM 컨테이너만 기동):

```bash
VLLM_BASE_URL=http://localhost:8100/v1 \
VLLM_HEALTH_URL=http://localhost:8100/health \
LAYOUT_SERVICE_URL=http://localhost:8082 \
LAYOUT_MODEL_NAME=PP-DocLayoutV3 \
HF_HUB_OFFLINE=1 \
MODEL_ROOT=/home/team_gh/mil_OCR_v2/models \
FALLBACK_ENABLED=false \
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_pipeline_with_outputs.py --input-dir data/raw
```

**옵션 B — Pipeline 컨테이너 내부 실행**:

```bash
docker run --rm --gpus device=0 --network host \
  -v $(pwd):/workspace \
  -e LAYOUT_SERVICE_URL=http://localhost:8082 \
  -e VLLM_BASE_URL=http://localhost:8100/v1 \
  -e PYTHONPATH=/workspace \
  mil_ocr_v2-pipeline \
  python scripts/run_pipeline_with_outputs.py [--input-dir data/raw]
```

**환경변수**:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VLLM_BASE_URL` | `http://localhost:8100/v1` | vLLM OpenAI 호환 API |
| `VLLM_HEALTH_URL` | `http://localhost:8100/health` | vLLM 헬스체크 |
| `LAYOUT_SERVICE_URL` | — | Layout HTTP 컨테이너 URL (설정 시 HTTP 모드) |
| `LAYOUT_MODEL_NAME` | `PP-DocLayoutV3` | 레이아웃 모델 |
| `LAYOUT_FUSION_MODE` | `false` | Fusion 모드 활성화 |
| `MODEL_ROOT` | `<repo>/models` | PaddleOCR 가중치 루트 |
| `FALLBACK_ENABLED` | `false` | Fallback 경로 활성화 |
| `FALLBACK_BASE_URL` | (없음) | 설정 시 Fallback HTTP 컨테이너 사용, 미설정 시 `_DisabledFallback` |

---

## 2. 테스트 케이스

| # | 케이스 | 검증 포인트 |
|---|--------|-----------|
| T1 | Fusion OFF 기본 경로 | P1→P2→P2.5-A→P3-A→P2.5-B→P2.5-C→P3-B→P4 end-to-end |
| T2 | Fusion ON, DPI≥150 | V3+plus-L 결합, reading_order 정합성 |
| T3 | Fusion ON, DPI<150 | V3+heuristic 경로, reading_order 정합성 |
| T4 | Other 문서 입력 | `form_type=other` 분류, 범용 OCR 경로, 검토 큐 미적재 |
| T5 | Fallback 전환 | VLM 불가 시 Fallback 경로 전환 |
| T6 | military 문서 검토 큐 | P4 실패 → 큐 적재 → 담당자 수정 → P6 재적재 |

---

## 3. 결과 저장 구조

```
data/pipeline_outputs/{YYYYMMDD_HHMMSS}/
├── run_summary.json                 ← 전체 실행 요약
├── ocr_results.db                   ← P6 DB (SQLite)
├── review_queue.db                  ← 검토 큐 DB
└── {doc_id}/
    ├── summary.json                 ← 문서별 요약 (form_type, status, timings)
    ├── P1/
    │   ├── result.json              ← dpi, quality_score, sr_applied
    │   ├── preprocessed.png         ← 전처리 이미지 (RGB, 기울기 보정)
    │   └── binary.png               ← 이진화 이미지
    ├── P2/
    │   ├── result.json              ← RawLayoutResult (정제 전)
    │   └── layout_visualization.png ← P2 원시 검출 bbox 오버레이
    ├── P2.5A/
    │   ├── result.json              ← LayoutResult (removed_count, merged_count)
    │   └── layout_visualization.png ← 정제 후 bbox (P2와 비교용)
    ├── P3A/
    │   └── result.json              ← form_type, form_confidence
    ├── P2.5B/
    │   └── result.json              ← {region_id → InstructionSpec}
    │                                   (system_prompt, user_instruction, json_schema,
    │                                    pixel_budget, 1-shot 예시 포함)
    ├── P2.5C/
    │   ├── result.json              ← pixel_budget 그룹 + 영역별 crop 메타
    │   └── crops/
    │       ├── r_0001_table_b1120.png   ← 영역별 크롭 이미지 (vLLM 입력 그대로)
    │       └── ...
    ├── P3/
    │   ├── result.json              ← fields[], tables[], domain_codes[]
    │   ├── region_traces.json       ← 영역별 vLLM 호출 trace
    │   │                               (입출력/소요시간/추출 필드 — raw_vlm_responses 대체)
    │   └── raw_vlm_response.json    ← 집계 raw_json (있을 때만)
    ├── P4/
    │   └── result.json              ← overall_confidence, validation_errors
    ├── P5/
    │   ├── output.json
    │   ├── output.xml
    │   └── output.csv (선택)
    └── P6/
        └── result.json              ← db_record_ids, review_queue_id, status
```

**디버깅 흐름**:

1. P2 → P2.5A 시각화 비교 → LayoutPostProcessor 정제 효과 확인
2. P3A/result.json → form_type이 military/other로 올바르게 분류되었는지
3. P2.5B/result.json → military 경로는 CoT 지시 + 1-shot 예시 포함 여부, other 경로는 범용 OCR instruction 확인
4. P2.5C/crops/*.png → vLLM이 실제로 받은 이미지 (해상도 적절성)
5. P3/region_traces.json → 영역별 응답·소요시간·추출 결과 (JSON parse 실패 원인 추적)
6. P4/result.json → other 경로에서 룰 검증 스킵 확인, military 경로에서만 validation_errors 집계

---

## 4. 단계별 타이밍 기록 (run_summary.json)

```json
{
  "timestamp": "20260409_043928",
  "document_count": 3,
  "documents": [
    {
      "doc_id": "보급청구서_001",
      "status": "success",
      "processing_path": "vlm",
      "form_type": "supply_request",
      "total_ms": 4820.1,
      "timings": {
        "P1": 310.2,
        "P2": 48.5,
        "P2_5A": 12.3,
        "P3A": 420.7,
        "P2_5B": 5.1,
        "P2_5C": 8.4,
        "P3B": 3850.4,
        "P4": 3.1,
        "P5": 4.2,
        "P6": 18.7
      },
      "p3b_batches": [
        {"pixel_budget": 140, "region_count": 1, "ms": 380.1},
        {"pixel_budget": 560, "region_count": 2, "ms": 820.3},
        {"pixel_budget": 1120, "region_count": 3, "ms": 2650.0}
      ]
    },
    {
      "doc_id": "일반공문서_001",
      "status": "other_document",
      "processing_path": "vlm",
      "form_type": "other",
      "total_ms": 1240.3,
      "review_queue_id": null
    }
  ]
}
```

---

## 5. 콘솔 출력 원칙

통합 테스트 실행 시 콘솔에 다음을 빠짐없이 출력합니다.

1. **환경 정보**: CUDA 버전, Fusion 모드, 서비스 URL, 테스트 이미지 수
2. **P2 탐지 결과**: 모드(Fusion ON/OFF), 탐지 영역 수
3. **P2.5-A 정제 결과**: 제거된 박스 수, 병합된 블록 수
4. **P3-A 분류 결과**: form_type, form_confidence, military/other 분기 표시
5. **P3-B 배치 처리**: 그룹별 pixel_budget, 배치 크기, 추론 시간
6. **P4 검증 결과**: overall_confidence, 오류 수, 검토 큐 적재 여부
7. **실패 시**: 실패 단계, 에러 메시지, 스택 트레이스
8. **문서별 요약**: `[PASS/FAIL] {doc_id} form={form_type} total=Xms P2=Xms P3A=Xms P3B=Xms`
9. **전체 요약**: 총 처리 건수, military/other/review_queue 분포

---

## 6. 테스트 이력

| 일자 | 환경 | 케이스 | 결과 | 비고 |
|------|------|--------|------|------|
| 2026-04-08 | H100 / vLLM v0.19.0 | T1 기본 경로 | 3건 PASS | Phase 1 통합 테스트 (설계 미반영 상태, 전부 군수 오분류) |
| 2026-04-10 | H100 / vLLM v0.19.0 (구 옵션) | T1 + T4 (Other 경로) | 3건 PASS, errors=0 | Other 경로 + CoT 스키마 + 1-shot + pixel_budget 상향 반영 후 재측정 |
| (예정) | vLLM 재기동 후 | T1 + T4 | — | fp8 KV + speculative decoding 효과 측정 |
| (예정) | — | T2, T3 | — | Fusion ON DPI 분기 검증 (군수 서식 샘플 확보 필요) |
| (예정) | — | T5 | — | Fallback 전환 경로 (fallback 컨테이너 기동 필요) |
| (예정) | — | T6 | — | 검토 큐 적재 → 담당자 수정 → DB 재적재 |

### 6-1. 2026-04-10 측정치 요약

Phase 1-E 통합 테스트 (설계 반영 후):

| 문서 | form_type 분류 | status | P3-B ms | total ms | validation_errors | 검토 큐 |
|------|--------------|--------|---------|----------|-------------------|--------|
| 국회공문서 | other (conf=1.00) | other_document | 15,212 | 19,292 | 0 | 미적재 |
| 전역지원서_1 | other (conf=1.00) | other_document | 2,824 | 3,077 | 0 | 미적재 |
| 전역지원서_2 | other (conf=1.00) | other_document | 4,122 | 4,386 | 0 | 미적재 |
| **합계** | — | — | 22,158 | 26,755 | **0** | **0건** |

**이전 측정치(설계 미반영, 2026-04-10 13:27)와 비교**:

| 지표 | 이전 | 현재 | 변화 |
|------|------|------|------|
| 총 처리 시간 | 60,057 ms | 26,755 ms | **-55%** |
| form_type 오분류 | 3/3 (군수로 잘못 분류) | 0/3 | 완전 개선 |
| validation_errors | 7건 (필수 필드 누락 오탐) | 0건 | 오탐 제거 |
| 검토 큐 적재 | 3건 | 0건 | 운영 부담 제거 |

### 6-2. 잔존 이슈 — CLAUDE.md 로드맵 참조

통합 테스트에서 발견된 7개 잔존 이슈는 `CLAUDE.md` 개발 로드맵 "Phase 1 잔여 → 잔존 이슈" 섹션에 번호별로 기록되어 있습니다:

1. **P3-B JSON parse 실패** 🔴 — guided_json 반환에 코드 펜스·자연어 섞임
2. **P3-B 배치 동시 전송 미구현** 🔴 — `for cropped in regions:` 순차 호출
3. **vLLM 재기동 미적용** 🟡 — 최적화 옵션은 파일에만 반영됨
4. **Real-ESRGAN 미설치** 🟡 — `basicsr` 모듈 필요
5. **warnings 중복 전파** 🟢 — P1 경고가 P3-B에도 중복 표시
6. **빈 layout 시각화 저장** 🟢
7. **RegionType handwritten_field/signature/checkbox 확장** 🟢
