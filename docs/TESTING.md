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
    │   └── result.json           ← form_type, form_confidence
    │
    ├── [military 경로]
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
10. **문서별 요약**:
    - military: `[PASS/FAIL] {doc_id} form={form_type} total=Xms P3B=Xms retry={N}건`
    - other: `[PASS/FAIL] {doc_id} form=other total=Xms hough={success/fail} sig={N}건`
11. **전체 요약**: 처리 건수, military/other/review_queue 분포, 평균 처리 시간

---

## 6. 테스트 이력

| 일자 | 환경 | 케이스 | 결과 | 비고 |
|------|------|--------|------|------|
| 2026-04-08 | H100 / vLLM v0.19.0 | T1 기본 경로 | 3건 PASS | Phase 1 초기 통합 |
| 2026-04-10 21:09 | H100 / vLLM v0.19.0 | T1 48px+재시도+OCR힌트 | 3건 PASS, 28.1s | guided_json 미enforce, JSON parse warning 152건 |
| 2026-04-14 00:41 | 동일 | T1 (xgrammar 플래그 인식 실패) | 3건 PASS, other 경로 | vLLM 구옵션 기동 실패 발견 |
| **2026-04-14 01:31** | H100 / vLLM v0.19.0 (`--structured-outputs-config` 적용) | **T1 + T4 (other/skill_registry)** | **3건 PASS, errors=0** | JSON parse warning 0, `skill_stats` 기록, 총 26.7~47.8s |
| (예정) | — | T2, T3 | — | Fusion ON DPI 분기 |
| (예정) | — | T5 | — | Fallback 전환 |
| (예정) | — | T7 | — | OCR-augmented 힌트 효과 |
| (예정) | — | T8 | — | 인장 인식, 허프 성공/실패 분기 |
| (예정) | — | T9 | — | 결재란 pass2 셀 실행 (현재 pass2_tasks=0 관찰) |
| (예정) | — | T10 | — | 서명 탐지 이진 분류 |

### 6-1. 2026-04-14 01:31 상세

3문서 모두 `status=other_document`, `processing_path=skill_registry`, `errors=0`:

| 문서 | total | P1 | P2 | P3A | SkillRegistry | S2 호출수 | pass1_tables | pass2_tasks |
|------|-------|------|------|------|----------------|-----------|--------------|-------------|
| 국회공문서 | 47.8s | 753ms | 6.7s | 11.8s | 28.5s | 16 | 2 | 0 |
| 전역지원서_1 | 7.5s | 53ms | 99ms | 143ms | 7.2s | 7 | 1 | 0 |
| 전역지원서_2 | 10.8s | 58ms | 122ms | 143ms | 10.5s | 6 | 1 | 0 |

결과 경로: [data/pipeline_outputs/20260414_013123/](../data/pipeline_outputs/20260414_013123/)

- `pass2_tasks=0`: S5 pass1이 `table_type=other, cells=[]`만 반환. 스키마 호환 또는 표 판정 기준 재검토 필요.
- 첫 문서의 P2/P3A 소요가 큰 것은 PP-DocLayoutV3 + vLLM 웜업. 2번째부터 ~7~10s로 정상화.

### 6-2. 검증 필요한 리스크

| # | 이슈 | 보완 계획 |
|---|------|---------|
| R1 | SealPreprocessor 허프 실패율 미측정 | T8에서 성공/실패 분포 수집, 실패율 30%+ 시 휴리스틱 격하 |
| R2 | S3 HandwritingReader 한국어 수기 실측치 부재 | 합성 샘플 10~20장으로 0-shot 인식률 측정 후 임계값 보정 |
| R3 | pass2 태스크 0건 문제 | S5 pass1 출력 분석, 실제 표(결재란 등) 샘플로 재현성 검증 |
| R4 | vLLM 출력 변동성 기준치 미수립 | 동일 문서 N=3 반복 측정 |
| R5 | PaddleOCR 가중치 폐쇄망 배치 | `Dockerfile.pipeline`에 `~/.paddlex/official_models/` COPY 추가 |
