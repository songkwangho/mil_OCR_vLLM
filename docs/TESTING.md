# 통합 테스트 가이드

> 통합 테스트 실행 방식, 결과 저장 구조, 타이밍 기록, 콘솔 출력 원칙을 기술합니다.

---

## 1. 실행 방식

**옵션 A — 호스트 conda 환경 직접 실행** (권장, Layout + vLLM 컨테이너만 기동):

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

| # | 케이스 | 검증 포인트 |
|---|--------|-----------|
| T1 | Fusion OFF 기본 경로 | P1→P2→P2.5-A→P3-A→P2.5-B→P2.5-C→P3-B→P4 end-to-end |
| T2 | Fusion ON, DPI≥150 | V3+plus-L 결합, reading_order 정합성 |
| T3 | Fusion ON, DPI<150 | V3+heuristic 경로, reading_order 정합성 |
| T4 | Other 문서 입력 | `form_type=other` 분류, 범용 OCR 경로, 검토 큐 미적재 |
| T5 | Fallback 전환 | VLM 불가 시 Fallback 경로 전환 |
| T6 | military 문서 검토 큐 | P4 실패 → 큐 적재 → 담당자 수정 → P6 재적재 |
| T7 | OCR-augmented 힌트 | 저신뢰 영역 OCR 힌트 삽입 + 재시도 동작 확인 |

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
    │   └── result.json              ← form_type, form_confidence, 추론 시간
    ├── P2.5B/
    │   └── result.json              ← {region_id → InstructionSpec}
    │                                   (system_prompt, user_instruction, json_schema,
    │                                    pixel_budget, ocr_hint 포함 여부)
    ├── P2.5C/
    │   ├── result.json              ← pixel_budget 그룹 + 영역별 crop 메타
    │   └── crops/
    │       ├── r_0001_table_b1120.png  ← 영역별 크롭 이미지 (48px 정렬 완료)
    │       └── ...
    ├── P3B/
    │   ├── result.json              ← fields[], tables[], domain_codes[], retry_count
    │   └── raw_vlm_responses/
    │       ├── batch_{budget}_{n}.json  ← pixel_budget별 배치 원본 응답
    │       └── retry_{region_id}.json  ← 재시도 호출 응답 (해당 시)
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
3. P2.5B/result.json → military 경로는 CoT 지시 + 1-shot 예시 + OCR 힌트 포함 여부
4. P2.5C/crops/*.png → VLM이 실제로 받은 이미지 (48px 정렬, 해상도 적절성)
5. P3B/result.json → retry_count 확인 → 재시도 발생 패턴 파악
6. P3B/raw_vlm_responses/retry_*.json → 재시도 효과 측정
7. P4/result.json → other 경로에서 룰 검증 스킵 확인

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
      ],
      "retry_stats": {
        "retry_count": 2,
        "retried_fields": ["nsn", "quantity"],
        "retry_ms": 950.2
      }
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

1. **환경 정보**: CUDA 버전, Fusion 모드, 서비스 URL, OCR 힌트 활성화 여부, 테스트 이미지 수
2. **P2 탐지 결과**: 모드(Fusion ON/OFF), 탐지 영역 수
3. **P2.5-A 정제 결과**: 제거된 박스 수, 병합된 블록 수
4. **P3-A 분류 결과**: form_type, form_confidence, military/other 분기 표시
5. **P3-B 배치 처리**: 그룹별 pixel_budget, 배치 크기, 추론 시간, OCR 힌트 삽입 수
6. **P3-B 재시도**: 재시도 발생 필드, 재시도 전/후 신뢰도
7. **P4 검증 결과**: overall_confidence, 오류 수, 검토 큐 적재 여부
8. **실패 시**: 실패 단계, 에러 메시지, 스택 트레이스
9. **문서별 요약**: `[PASS/FAIL] {doc_id} form={form_type} total=Xms P3B=Xms retry={N}건`
10. **전체 요약**: 총 처리 건수, military/other/review_queue 분포, 평균 재시도 비율

---

## 6. 테스트 이력

| 일자 | 환경 | 케이스 | 결과 | 비고 |
|------|------|--------|------|------|
| 2026-04-08 | H100 / vLLM v0.19.0 | T1 기본 경로 | 3건 PASS | Phase 1 통합 테스트 |
| (예정) | vLLM 재기동 후 | T1 | — | vLLM 최적화 옵션 적용 후 성능 측정 |
| (예정) | — | T2, T3 | — | Fusion ON DPI 분기 검증 |
| (예정) | — | T4 | — | Other 문서 경로 |
| (예정) | — | T5 | — | Fallback 전환 |
| (예정) | — | T7 | — | OCR-augmented 힌트 효과 측정 |

### 6-1. 향후 계획 타당성 및 문제점

**[R1] vLLM 최적화 옵션 재기동 필요**
docker-compose.yml 파일은 수정되었지만 기존 컨테이너는 구 옵션으로 기동 중입니다. `docker compose restart vllm-server` 후 동일 문서 처리 시간을 비교 측정해야 합니다.

**[R2] OCR-augmented가 모든 영역에 적용 시 처리 지연**
PaddleOCR가 영역당 ~50ms이므로, 20개 영역 문서에서 최대 1초 추가됩니다. 선택적 적용 조건(logprobs < 0.80, NSN 패턴 필드)을 통합 테스트 T7에서 실측하여 임계값을 현실화합니다.

**[R3] 재시도 로직이 피크 처리량에 미치는 영향**
MAX_RETRIES=1이어도 재시도가 집중되면 처리 지연이 발생합니다. T7 테스트에서 재시도 발생 비율을 측정하고, 5% 초과 시 임계값 조정을 검토합니다.

**보완**: run_summary.json의 retry_stats 항목으로 재시도 패턴을 2주간 수집 후 임계값 현실화.
