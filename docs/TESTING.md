# 통합 테스트 가이드

> 통합 테스트 실행 방식, 결과 저장 구조, 타이밍 기록, 콘솔 출력 원칙을 기술합니다.

---

## 1. 실행 방식

**옵션 A — 호스트 conda 환경 직접 실행** (권장):

```bash
VLLM_BASE_URL=http://localhost:8100/v1 \
LAYOUT_SERVICE_URL=http://localhost:8082 \
LAYOUT_FUSION_MODE=false \
HF_HUB_OFFLINE=1 \
python scripts/run_pipeline_with_outputs.py --input-dir data/raw
```

**pytest 자동화 스위트**:

```bash
# 전체 (40 pass + 1 skipped)
PYTHONPATH=/workspace pytest tests/ -v

# 개별
pytest tests/test_integration_pipeline.py          # 기본 E2E (military/other)
pytest tests/test_T2_T3_fusion.py                  # Layout Fusion (T2/T3)
pytest tests/test_T5_fallback_routing.py           # Fallback 라우팅 (T5)
pytest tests/test_T8_seal_integration.py           # 인장 처리 (T8)
pytest tests/test_T9_T10_skills_integration.py     # Skill 파이프라인 (T9/T10)
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
| `LAYOUT_SERVICE_URL` | — | Layout HTTP 컨테이너 URL |
| `LAYOUT_FUSION_MODE` | `false` | Fusion 모드 활성화 |
| `FALLBACK_ENABLED` | `false` | Fallback 경로 활성화 |
| `OCR_HINT_ENABLED` | `true` | OCR-augmented 힌트 활성화 |

---

## 2. 테스트 케이스

### 기존 군수 서식 (military 경로)

| # | 케이스 | 검증 포인트 |
|---|--------|-----------|
| T1 | Fusion OFF 기본 경로 | P1→P2→P2.5-A→P3-A→P2.5-B→P2.5-C→P3-B→P4 end-to-end |
| T2 | Fusion ON, DPI≥150 | V3+plus-L 결합, reading_order 정합성 |
| T3 | Fusion ON, DPI<150 | V3+heuristic 경로, reading_order 정합성 |
| T5 | Fallback 전환 | VLM 불가 시 Fallback 경로 전환 |
| T6 | military 문서 검토 큐 | P4 실패 → 큐 적재 → 담당자 수정 → P6 재적재 |
| T7 | OCR-augmented 힌트 | 저신뢰 영역 OCR 힌트 삽입 + 재시도 동작 확인 |

### 공문서 other 경로 (Skill Registry)

| # | 케이스 | 검증 포인트 |
|---|--------|-----------|
| T4 | Other 문서 입력 | `form_type=other` 분류, Skill Registry 디스패치, 검토 큐 미적재 |
| T8 | 인장(직인) 인식 | SealPreprocessor 허프 성공/실패 분기, VLM 인식 결과 확인 |
| T9 | 결재란 2패스 처리 | S5 패스1 구조 추출 → 패스2 셀 내용 추출, 오케스트레이터 태스크 생성 흐름 |
| T10 | 서명 탐지 | 서명 있는 셀 → `signature_present: true`, 빈 셀 → `false` |

### PDF 어댑터

| # | 케이스 | 검증 포인트 |
|---|--------|-----------|
| T11 | PDF 단일 페이지 입력 | PdfAdapter 렌더링 → `PageImage` 생성 → 기존 파이프라인 정상 통과 |
| T12 | PDF 멀티페이지 입력 | 페이지별 독립 처리, `doc_id` = `{원본}_p{N:02d}`, `PdfDocumentResult` 집계 |
| T13 | PDF 일부 페이지 실패 | `overall_status=partial`, 실패 페이지만 검토 큐 적재, 성공 페이지 DB 기록 |
| T14 | 암호화/손상 PDF | `PdfAdapterError` → `PdfDocumentResult(status=failed)`, 검토 큐 적재 |
| T15 | PDF → DB 페이지 추적 | `parent_doc_id`, `page_number`, `total_pages` 컬럼 기록 확인 |

**T12 세부 검증**:
- `pdf_summary.json` 생성 확인 (`{doc_id}/pdf_summary.json`)
- 각 페이지 결과 디렉토리 `p01/`, `p02/` 구조 확인
- `review_queue_id` 가 페이지 단위로 발급되는지 확인

**T13 세부 검증**:
- `PdfDocumentResult.overall_status = "partial"`
- 실패 페이지 `ReviewQueueItem.parent_doc_id` = 원본 PDF doc_id
- 성공 페이지 `PipelineOutput.status = "success"` + DB 레코드 존재 확인

### 전비품 확인서 작성 점검표 (equipment_checklist)

| # | 케이스 | 검증 포인트 |
|---|--------|-----------|
| T_chk1 | PDF 정상 입력 (1페이지) | `form_type=equipment_checklist`, O/X 6항목 추출, `document_date`, `writer_block` |
| T_chk2 | TemplateAugmentor 적용 확인 | PP-DocLayout 누락 영역 보완, `augmented_count > 0`, `source="template"` 영역 확인 |
| T_chk3 | form_identifier 버전 분기 | FormClassifier가 `"별지 제3-2호 서식"` 추출 → TemplateAugmentor v1.0 선택 |
| T_chk4 | O/X 판독 경계 케이스 | 흐릿한 O → `result_confidence < 0.75` → OCR 힌트 재시도 트리거 |
| T_chk5 | CHK-001 트리거 | `checklist_items` 길이 ≠ 6 → `severity=HIGH` → 검토 큐 적재 |
| T_chk6 | 서명 탐지 | `writer.signature_present = true` (서명 있음) / `false` (없음) 이진 분류 |

**T8 세부 검증**:
- 허프 성공 케이스: `unwrapped=true`, VLM이 직선화된 이미지 수신 확인
- 허프 실패 케이스: `unwrapped=false`, context_hint 포함 원본 크롭 수신 확인
- 허프 실패율이 30%+ 이면 극좌표 변환을 선택적 최적화로 격하 검토

**T9 세부 검증**:
- 패스1 결과: `TableStructure.cells` 각 셀의 `content_type` 정확도
- 패스2 결과: `SkillTask` 생성 수 = 비어있지 않은 셀 수
- DISPATCH_ORDER 순서대로 배치 실행되는지 로그 확인

---

## 3. 결과 저장 구조

```
data/pipeline_outputs/{YYYYMMDD_HHMMSS}/
├── run_summary.json
├── ocr_results.db
├── review_queue.db
└── {doc_id}/
    ├── summary.json              ← form_type, status, processing_path, timings
    ├── P1/
    │   ├── result.json           ← dpi, quality_score, sr_applied
    │   ├── preprocessed.png
    │   └── binary.png
    ├── P2/
    │   ├── result.json           ← RawLayoutResult (정제 전)
    │   └── layout_visualization.png
    ├── P2.5A/
    │   ├── result.json           ← LayoutResult (removed_count, merged_count)
    │   └── layout_visualization.png
    ├── P3A/
    │   └── result.json           ← form_type, form_confidence, form_identifier
    │
    ├── [military 경로]
    │   ├── P2.5A5/
    │   │   └── result.json       ← augmented_count, 추가된 template 영역 목록
    │   ├── P2.5B/
    │   │   └── result.json       ← {region_id → InstructionSpec}
    │   ├── P2.5C/
    │   │   ├── result.json       ← pixel_budget 그룹 + 크롭 메타
    │   │   └── crops/
    │   │       └── r_0001_table_b1120.png
    │   └── P3B/
    │       ├── result.json       ← fields[], tables[], retry_count
    │       └── raw_vlm_responses/
    │           ├── batch_{budget}_{n}.json
    │           └── retry_{region_id}.json
    │
    ├── [other 경로]
    │   ├── S1/
    │   │   └── result.json       ← 레이아웃 결과 + 결재란 휴리스틱 판단
    │   ├── S5_pass1/
    │   │   └── result.json       ← TableStructure (셀 좌표 + content_type)
    │   ├── S2_S3_S4_S6/
    │   │   ├── result.json       ← 각 Skill 결과 목록
    │   │   └── seal/
    │   │       ├── original_crop.png
    │   │       └── unwrapped.png ← 극좌표 직선화 결과 (허프 성공 시)
    │   └── S7/
    │       └── result.json       ← official_document.json 형식 최종 결과
    │
    ├── P4/
    │   └── result.json           ← overall_confidence, validation_errors
    ├── P5/
    │   ├── output.json
    │   ├── output.xml
    │   └── output.csv
    └── P6/
        └── result.json           ← db_record_ids, review_queue_id, status
```

**디버깅 흐름 (other 경로)**:

1. S1/result.json → 결재란이 `table`로 탐지되었는지, Mode B 트리거 여부
2. S5_pass1/result.json → 결재란 셀 구조와 content_type 정확도
3. S2_S3_S4_S6/seal/unwrapped.png → 허프 직선화 품질 확인
4. S7/result.json → official_document.json 스키마 준수 + low_confidence_fields

**PDF 입력 시 추가 저장 구조**:

```
data/pipeline_outputs/{YYYYMMDD_HHMMSS}/
└── {pdf_doc_id}/                      ← 원본 PDF 문서 ID
    ├── pdf_summary.json               ← 전체 페이지 집계
    │     {
    │       "doc_id": "전비품확인서_001",
    │       "total_pages": 2,
    │       "overall_status": "partial",
    │       "pages": [
    │         {"page": 1, "doc_id": "전비품확인서_001_p01", "status": "success"},
    │         {"page": 2, "doc_id": "전비품확인서_001_p02", "status": "review"}
    │       ]
    │     }
    ├── p01/                           ← 페이지별 기존 구조 그대로
    │   ├── P1/, P2/, P2.5A/, P3A/
    │   ├── [military] P2.5A5/, P2.5B/, P2.5C/, P3B/
    │   └── P4/, P5/, P6/
    └── p02/
        └── (동일 구조)
```

**디버깅 흐름 (PDF 입력)**:

1. `pdf_summary.json` → `overall_status` 및 페이지별 성공/실패 분포 확인
2. 실패 페이지 `p0N/P3A/result.json` → form_type, form_identifier 정상 추출 여부
3. `p0N/P2.5A5/result.json` → TemplateAugmentor augmented_count 및 추가 영역 확인
4. `review_queue.db` → 실패 페이지 큐 항목의 `parent_doc_id`, `page_number` 기록 확인

---

## 4. 타이밍 측정 규약 — 모델 로딩 분리 (IMPORTANT)

**원칙**: 문서별 timing은 **warm-start 기준**으로만 기록한다. 모델 로딩 시간은
`warmup_timings.json`에 별도 분리한다.

**이유**: PipelineOrchestrator는 lazy-loading 구조이며, 첫 문서 처리 시 다음이
timing에 섞여 측정값을 오염시킨다.
  - P1: Real-ESRGAN 가중치 로드 (+3~5초)
  - P2: Layout HTTP 클라이언트 첫 ping
  - P3-A / P3-B / S2~S7: vLLM 서버 멀티모달 프로세서 캐시 워밍 (+500~1000ms)

**실행 전 반드시 호출**:

```python
pipeline = PipelineOrchestrator(cfg)

# 1단계: lazy getter 초기화 (cheap)
# 2단계: dummy image 로 Real-ESRGAN + vLLM 첫 호출 강제 (heavy, ~5~10초)
warmup_timings = pipeline.warmup(run_dummy_inference=True)

# 이후 문서 timing은 warm-start 기준
for img in images:
    result = pipeline.run(doc_input)
```

`scripts/run_pipeline_with_outputs.py`는 자동으로 warmup을 수행한다.

**산출물**:
  - `{run_dir}/warmup_timings.json` — 컴포넌트별 초기화 시간
  - `{run_dir}/run_summary.json` — 문서별 warm-start timings (warmup 이후 측정)

**예외**: `warmup()` 시점에도 로드되지 않는 리소스(예: 도메인 특수 모델)가 있다면
해당 컴포넌트의 warmup 훅을 orchestrator에 추가해야 한다.

---

## 4-1. 단계별 타이밍 기록 (run_summary.json)

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
        "P1": 310.2, "P2": 48.5, "P2_5A": 12.3,
        "P3A": 420.7, "P2_5B": 5.1, "P2_5C": 8.4,
        "P3B": 3850.4, "P4": 3.1, "P5": 4.2, "P6": 18.7
      },
      "p3b_batches": [
        {"pixel_budget": 140, "region_count": 1, "ms": 380.1},
        {"pixel_budget": 560, "region_count": 2, "ms": 820.3},
        {"pixel_budget": 1120, "region_count": 3, "ms": 2650.0}
      ],
      "retry_stats": {
        "retry_count": 2,
        "retried_fields": ["nsn", "quantity"],
        "retry_ms": 950.2
      }
    },
    {
      "doc_id": "국회공문서_001",
      "status": "other_document",
      "processing_path": "skill_registry",
      "form_type": "other",
      "total_ms": 7200.5,
      "review_queue_id": null,
      "timings": {
        "P1": 280.1, "P2": 45.2, "P2_5A": 11.0,
        "P3A": 410.3,
        "S1_layout": 120.5,
        "S5_pass1": 980.2,
        "S6_batch_140": 520.1,
        "S2_batch_560": 1050.3,
        "S3_S4_S5p2_batch_1120": 2890.4,
        "S7_aggregator": 1500.8,
        "P4": 3.2, "P5": 4.1, "P6": 17.5
      },
      "skill_stats": {
        "seal_preprocessor": {
          "attempted": 1,
          "hough_success": 1,
          "hough_failed": 0
        },
        "signature_detector": {
          "attempted": 4,
          "detected": 2
        }
      }
    },
    {
      "doc_id": "전비품확인서_001",
      "status": "partial",
      "processing_path": "pdf",
      "form_type": null,
      "total_ms": 9840.2,
      "total_pages": 1,
      "pdf_pages": [
        {
          "doc_id": "전비품확인서_001_p01",
          "status": "success",
          "processing_path": "vlm",
          "form_type": "equipment_checklist",
          "form_identifier": "별지 제3-2호 서식",
          "total_ms": 9620.5,
          "timings": {
            "pdf_render": 210.3,
            "P1": 280.1, "P2": 52.4, "P2_5A": 11.2,
            "P3A": 430.8,
            "P2_5A5": 3.1,
            "P2_5B": 6.2, "P2_5C": 9.1,
            "P3B": 8310.4, "P4": 4.2, "P5": 3.8, "P6": 19.2
          },
          "template_augmentor": {
            "augmented_count": 6,
            "version_selected": "1.0",
            "form_identifier_matched": "별지 제3-2호 서식"
          },
          "p3b_batches": [
            {"pixel_budget": 1120, "region_count": 7, "ms": 8310.4}
          ],
          "retry_stats": {
            "retry_count": 1,
            "retried_fields": ["result_item_3"],
            "retry_ms": 980.1
          }
        }
      ]
    }
  ]
}
```

---

## 5. 콘솔 출력 원칙

1. **환경 정보**: CUDA 버전, Fusion 모드, vLLM 서버 상태, 테스트 이미지 수
2. **P2 탐지 결과**: 모드(Fusion ON/OFF), 탐지 영역 수
3. **P2.5-A 정제 결과**: 제거된 박스 수, 병합된 블록 수
4. **P3-A 분류 결과**: form_type, form_confidence, military/other 분기 표시
5. **[military] P3-B 배치 처리**: 그룹별 pixel_budget, 배치 크기, 추론 시간
6. **[military] P3-B 재시도**: 재시도 발생 필드, 재시도 전/후 신뢰도
7. **[other] Skill 실행 결과**: S5 패스1 셀 구조, SealPreprocessor 허프 성공/실패, 각 배치 처리 시간
8. **P4 검증 결과**: overall_confidence, 오류 수, 검토 큐 적재 여부
9. **실패 시**: 실패 단계, 에러 메시지, 스택 트레이스
10. **[military] TemplateAugmentor 결과**: augmented_count, version_selected, form_identifier_matched
11. **[PDF] 페이지별 처리**: `[PDF] {doc_id} {N}페이지 렌더링 완료 (Xms)`, 페이지별 form_type
12. **문서별 요약**:
    - military: `[PASS/FAIL] {doc_id} form={form_type} total=Xms P3B=Xms retry={N}건 aug={N}건`
    - other: `[PASS/FAIL] {doc_id} form=other total=Xms hough={success/fail} sig={N}건`
    - pdf: `[PASS/FAIL/PARTIAL] {doc_id} pages={total} success={N} review={N} failed={N} total=Xms`
13. **전체 요약**: 처리 건수, military/other/pdf/review_queue 분포, 평균 처리 시간

---

## 6. 테스트 이력

| 일자 | 환경 | 케이스 | 결과 | 비고 |
|------|------|--------|------|------|
| 2026-04-08 | H100 / vLLM v0.19.0 | T1 기본 경로 | 3건 PASS | Phase 1 초기 통합 |
| 2026-04-14 01:31 | + xgrammar enforce | T1 + T4 (other/skill_registry) | 3건 PASS, errors=0, JSON parse warning 0 | Skill Registry 경로 합류 |
| 2026-04-15 04:26 | + PdfAdapter | T1 + T11 (PDF 단일 페이지) | 4건 PASS (PDF 1건 포함), 전비품확인서 → equipment_checklist 정확 분류 | PDF→PageImage→파이프라인 통과 검증 |
| 2026-04-15 07:27 | + Sub-schema/Assembler | T_chk1 (전비품 확인서) | checklist 6항목 정확(O-O-O-X-X-O), writer 정상, CHK-001~004 모두 통과 | sub-schema 분해 + Assembler 조립 도입 |
| 2026-04-15 08:17 | + 패딩 절대 상한 + TemplateAugmentor v3 | T_chk1 안정화 | table 크롭 1920×2016(이전 2016×2160), writer 영역 침범 제거, 6항목 + writer + form_identifier 정확 | writer_block 영역 다중 PP(footer+text) 흡수 후 단일 region |
| 2026-04-17 02:26 | + 코드 클렌징 (064c1f3) | 통합 pytest + pipeline run | 40 pass / 1 skipped, pipeline 4건 errors=0 (warnings=17) | budget_config/_parsing/bbox_utils/rank_normalizer 중앙화, gemma4_engine/instruction_builder 삭제, region_id/was_retried/assembled_json 전 경로 보존 확인 |
| 2026-04-17 05:09 | + document_title 정합성 수정 | T_chk1 + 전역지원서_1/2 + 국회공문서 | 39 pass / 2 skipped, pipeline 4건 errors=0 | 전비품확인서 `assembled_json.document_title == "전비품 확인서 작성 점검표"` 첫 키 주입 + fields[] 중복 제거 확인. assembled_json None인 서식(전역지원서_1 등)은 VLM 추출 document_title 보존 |
| 2026-04-17 05:52 | + S2/S3 공통 도메인 서비스 격상 | 동일 4건 + S2/S3 위임 경로 | 39 pass / 2 skipped, pipeline 4건 errors=0, warnings=15(전회 17) | handwritten_field(r_0005, t_0001~t_0007)이 S3로 위임되어 기존 JSON parse failed 경고 소멸. O/X 판독 신뢰도 0.997~1.000. other 경로 국회공문서에서 S2가 30개 text 영역 처리. 다만 equipment_checklist `writer`/`checklist_items`가 구조적 dict→plain text로 평탄화(설계 의도: guided_json보다 정확한 텍스트 우선) |
| **2026-04-17 06:46** | + S3 옵션B (sub-schema guided_json) | T_chk1 + 전역지원서_1/2 + 국회공문서 | 39 pass / 2 skipped, pipeline 4건 errors=0, warnings=15 | SkillTask에 `field_key`/`json_schema` 추가. S3가 sub-schema 있으면 guided_json으로 구조화 JSON 반환. `_loads_relaxed_any`로 코드 펜스 정제. **assembled_json 완전 복원**: `checklist_items[6 dicts]`, `writer {team/rank/name/signature_present}` 정상 조립. overall_confidence 0.143→0.713. 국회공문서(plain text 경로) 30 fields 유지 |
| (예정) | — | T2, T3 | — | Fusion ON DPI 분기 |
| (예정) | — | T5 | — | Fallback 전환 |
| (예정) | — | T7 | — | OCR-augmented 힌트 효과 측정 |
| (예정) | — | T8 | — | 인장 허프 성공/실패 분기 |
| (예정) | — | T9 | — | 결재란 2패스 (현재 pass2_tasks=0 관찰) |
| (예정) | — | T10 | — | 서명 탐지 이진 분류 |
| (예정) | — | T12~T15 | — | PDF 멀티페이지/실패 케이스 |
| (예정) | — | T_chk2~T_chk6 | — | 전비품 확인서 경계 케이스 |

### 6-1. 잔존 이슈 (2026-04-15 기준)

| 우선순위 | 이슈 | 조치 | 상태 |
|---------|------|------|------|
| 🟡 | P4 신뢰도 산출 — assembled_json 미반영 | field_key blob 단일 FieldValue로 logprobs 평균 의미 상실. 트리 기반 sub-confidence 재산출 필요 | 미완 (CLAUDE.md 로드맵) |
| 🟡 | rank "대원"/"대위" 오인식 1회 관찰 | enum 강제 + 후처리 정규화 사전 (CLAUDE.md 로드맵) | 미완 |
| 🟡 | TableExtractor pass2_tasks=0 | S5 pass1 출력의 table_type/cells 품질 점검 | 대기 |
| 🟡 | PaddleOCR 가중치 폐쇄망 배치 | `Dockerfile.pipeline` COPY 추가 (`scripts/download_paddle_models.py`로 사전 준비) | 미완 |
| 🟢 | vLLM 변동성 N=3 반복 측정 | `scripts/measure_vllm_variance.py` 사용, 결과는 `data/variance_reports/`에 JSON+MD 저장 | ✅ 인프라 구축, 결정론 점수 20% 관측 |

### 6-2. 향후 계획 타당성 및 문제점

**[R1] vLLM 최적화 옵션 재기동 필요**
docker-compose.yml 수정 완료, 컨테이너 재기동 미수행. `docker compose restart vllm-server` 후 region_traces.json의 raw_response가 JSON 형식으로 변화하는지 확인.

**[R2] SealPreprocessor 허프 실패율 미측정**
T8에서 허프 성공/실패 분포를 측정하고, 실패율이 30%+ 이면 극좌표 변환을 선택적 최적화로 격하하고 원본 크롭 직접 VLM 전달 방식을 기본으로 변경.

**[R3] S3 HandwritingReader 수기 인식 실측치 부재**
국회공문서 샘플(또는 공개 의안 PDF 기반 합성 샘플) 10~20장으로 수기 인식률 실측 후 신뢰도 임계값 0.75를 현실화.

**[R4] 처리 시간 10초 이내 마진**
other 경로 예상 처리 시간: S1(0.5s) + S5패스1(1s) + S6배치(0.5s) + S2배치(1s) + S3/S4/S5패스2배치(3s) + S7(1.5s) = ~7.5s. Mode B 트리거 시 추가 VLM 호출로 10초 초과 가능. T4 실측 필요.

**[R5] PDF 렌더링 메모리 사용량**
멀티페이지 PDF 처리 시 PdfAdapter가 모든 페이지를 메모리에 렌더링 후 순차 처리. 100페이지 PDF의 경우 A4 300dpi 기준 약 2480×3508×3 bytes × 100 = ~2.4GB 메모리 필요. 대용량 PDF 처리 시 페이지별 스트리밍 렌더링으로 전환 필요 여부를 T12 실측 후 판단.

**[R6] equipment_checklist O/X 판독 오인식**
수기 O와 반쯤 열린 O(미완성 원)의 구분이 핵심 리스크. T_chk4에서 경계 케이스 집계 후 `result_confidence` 임계값(현재 0.75) 조정. OCR-augmented 힌트가 O/X 단일 문자 인식에 효과적인지도 함께 측정.

**[R7] form_identifier 미추출 케이스**
별지 번호가 문서 상단에서 잘리거나 인쇄 품질이 낮으면 FormClassifier가 `form_identifier=null`을 반환. TemplateAugmentor가 첫 번째 버전(v1.0)으로 폴백하므로 현재 버전에서는 문제없으나, 서식 개정 이후 버전이 2개 이상이 되면 잘못된 버전이 선택될 수 있음. T_chk3에서 form_identifier 추출 성공률 측정.
