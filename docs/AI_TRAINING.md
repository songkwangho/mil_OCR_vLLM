# AI 재학습 (Training / Fine-tuning) 가이드

> AI 모델의 Fine-tuning 방법, 학습 데이터, Docker 환경, 가중치 출력 경로를 기술합니다.

---

## 1. Fine-tuning 대상 — 2개 모델

| 모델 | 컴포넌트 | Pretrained 출처 | Fine-tuned 출력 | 상태 |
|------|---------|----------------|----------------|------|
| PP-DocLayout_plus-L | P2 레이아웃 탐지 | PaddleOCR 3.0 | `models/t2_layout/PP-DocLayout_plus-L_finetuned/` | ❌ 미시작 (Phase 2) |
| Gemma4 26B-A4B | P3-A FormClassifier | google/gemma-4-26b-a4b-it | `models/gemma4/gemma4-mil-classify/` | ❌ 미시작 (Phase 2) |
| Gemma4 26B-A4B | P3-B StructuredExtractor | google/gemma-4-26b-a4b-it | `models/gemma4/gemma4-mil-extract/` | ❌ 미시작 (Phase 2) |

> **P3-A / P3-B adapter 분리 이유**: P3-A는 LoRA rank=8 분류 전용, P3-B는 rank=16 추출 전용으로 학습 태스크가 다르므로 adapter 경로를 분리하여 덮어쓰기 및 로드 모호성 방지.

> **Fine-tuning 선행 조건**: Phase 1 잔여에서 PP-DocLayout 원본 검출률 측정 후 어노테이션 목표 수량 확정.

---

## 2. 학습 데이터 목표

| 대상 | 데이터 유형 | 목표 규모 | 데이터 경로 | 포맷 |
|------|-----------|----------|-----------|------|
| P2 레이아웃 탐지 | 군수 서식 영역 어노테이션 | 2,000~5,000장 (검출률 기반 조정) | `data/training/layout/` | COCO |
| P3-A FormClassifier SFT | 서식 이미지 + 정답 form_type | 서식 유형당 100~200장 | `data/training/vlm_classify/` | JSONL |
| P3-B SFT (1단계) | 서식 이미지 + 정답 JSON | 유형당 300장+ | `data/training/vlm_sft/` | JSONL |
| P3-B 수기 강화 (2단계) | 수기 crop + 정답 텍스트 | 10,000장+ | `data/training/vlm_handwriting/` | JSONL |
| P3-B DPO (3단계) | Chosen/Rejected 출력 쌍 | 시스템 운영 3개월 후 | `data/training/vlm_dpo/` | JSONL |

---

## 3. PP-DocLayout Fine-tuning

| 항목 | 값 |
|------|------|
| 스크립트 | `training/layout/train_pp_doclayout.py` (Phase 2 예정) |
| 데이터 포맷 | COCO 어노테이션 |
| 카테고리 | 군수 서식 특화: text, table, header, footer, signature, seal, figure, formula, handwritten_field, checkbox |
| Pretrained | `models/t2_layout/PP-DocLayout_plus-L/` |
| Fine-tuned 출력 | `models/t2_layout/PP-DocLayout_plus-L_finetuned/` |
| Docker 서비스 | `train` (PaddlePaddle 환경) |

**Fine-tuning 목표 수량 조정 기준** (Phase 1 잔여 검출률 측정 결과):

| 검출률 | 어노테이션 목표 |
|--------|--------------|
| > 70% | 2,000장 유지 |
| 50~70% | 3,000장 이상으로 상향 |
| < 50% | 목표 재협의 + HEURISTIC fallback 비중 확대 |

---

## 4. Gemma4 VLM Fine-tuning — 분리 전략

고도화된 파이프라인에서 P3-A(FormClassifier)와 P3-B(StructuredExtractor)는 책임이 분리되었습니다. Fine-tuning도 이를 반영하여 분리 수행합니다.

### 4-1. P3-A FormClassifier Fine-tuning

**목적**: 군수 서식 6종 + unknown 분류 정확도 향상

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_classify.py` (Phase 2 예정) |
| 방식 | LoRA (rank=8, alpha=16) — 분류 태스크는 소규모 |
| pixel_budget | 140 토큰 (추론 시와 동일) |
| 목표 데이터 | 유형당 100~200장 |

**학습 데이터 포맷 (JSONL)**:
```jsonl
{"image": "form_001.png", "instruction": "이 문서의 서식 유형을 분류하세요.", "response": "{\"form_type\": \"supply_request\", \"confidence\": 0.98}"}
{"image": "form_002.png", "instruction": "이 문서의 서식 유형을 분류하세요.", "response": "{\"form_type\": \"maintenance_record\", \"confidence\": 0.95}"}
```

### 4-2. P3-B StructuredExtractor Fine-tuning — 3단계

#### 1단계 — Instruction Tuning (SFT) [필수]

**목적**: 영역별 크롭 이미지 + 도메인 instruction 조합에 대한 추출 정확도 향상

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_sft.py` (Phase 2 예정) |
| 방식 | LoRA (rank=16, alpha=32) |
| 학습 VRAM | 16~24GB (LoRA + Q4 base) |
| 목표 데이터 | 유형당 300장+ (SFT 착수 마일스톤: 유형당 100건) |

**학습 데이터 포맷** — 크롭 이미지 기반 (P3-B 추론과 동일 구조):
```jsonl
{"image": "table_crop_001.png", "pixel_budget": 1120, "instruction": "이 표의 구조를 분석하고 모든 셀을 추출하세요. (보급청구서)", "response": "{\"html\": \"<table>...</table>\", \"cells\": [...]}"}
{"image": "text_crop_001.png", "pixel_budget": 560, "instruction": "이 수기 기입란에서 NSN 코드와 수량을 인식하세요.", "response": "{\"nsn\": \"1005-01-432-1234\", \"quantity\": 10}"}
```

**학습 태스크 구성**:

| 태스크 | 비중 | instruction 유형 |
|--------|------|-----------------|
| 전체 서식 필드 추출 | 30% | 서식별 guided_json 정답 |
| 수기 OCR (텍스트 crop) | 25% | 도메인 맥락 포함 instruction |
| 표 구조 인식 (표 crop) | 20% | 고해상도 표 추출 |
| 특수 코드 인식 | 15% | NSN/K-NSN 형식 |
| 인장 인식 (seal crop) | 10% | 원형 배치 텍스트 |

#### 2단계 — 수기 인식 강화 [선택]

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_handwriting.py` (Phase 2 예정) |
| 방식 | LoRA 추가 학습 (1단계 가중치 위에 누적) |
| 조건 | 1단계 SFT 후 수기 필드 인식 오류 > 15% |

```jsonl
{"image": "handwritten_001.png", "pixel_budget": 560, "instruction": "이 수기 텍스트를 인식하세요.", "response": "{\"text\": \"1232부대\"}"}
```

#### 3단계 — DPO [선택]

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_dpo.py` |
| 방식 | DPO (Direct Preference Optimization) |
| 조건 | 시스템 운영 3개월 후 — P4 검증 실패 출력을 Rejected로 자동 축적 |
| 데이터 출처 | `scripts/export_training_data.py --format dpo` |

```jsonl
{
  "image": "supply_crop_001.png",
  "instruction": "수량, 단가, 합계를 추출하세요.",
  "chosen": "{\"quantity\": 10, \"unit_price\": 1200000, \"total\": 12000000}",
  "rejected": "{\"quantity\": 10, \"unit_price\": 1200000, \"total\": 1200000}"
}
```

### 4-3. Fine-tuning 선행 조건 및 순서

```
[Phase 1 잔여]
PP-DocLayout 검출률 측정 (군수 서식 50장 샘플)
    ↓
[Phase 2-A Track A]
PP-DocLayout Fine-tuning (P2)
    ↓ 레이아웃 품질이 VLM 크롭 입력 품질에 직결
[Phase 2-A Track B — 병행]
검토 큐 UI MVP 구현 → 교정 데이터 축적 시작
    ↓ 유형당 100건 달성 후
[Phase 2-B]
P3-A FormClassifier SFT (유형당 100~200장)
    ↓
P3-B StructuredExtractor 1단계 SFT (유형당 300장+)
    ↓ 벤치마크 후 판단
P3-B 2단계 수기 강화 [선택]
    ↓
P3-B 3단계 DPO [선택 — 운영 3개월 후]
```

---

## 5. v1 → v2 코드 이관 기록

### 5-1. 이관 완료 항목

| v1 소스 파일 | v2 대상 | 이관 방식 | 상태 |
|-------------|---------|----------|------|
| `src/interfaces/enums.py` (208줄) | `src/interfaces/enums.py` | 축소 이관 | ✅ |
| `src/interfaces/types.py` (452줄) | `src/interfaces/types.py` | 재설계 — RawLayoutResult, InstructionSpec, CroppedRegion 추가 | ✅ |
| `src/layer1_preprocess/t1_preprocessor.py` | `src/preprocess/preprocessor.py` | 그대로 이관 | ✅ |
| `src/layer1_preprocess/sr_enhancer.py` | `src/preprocess/sr_enhancer.py` | 그대로 이관 | ✅ |
| `src/layer1_preprocess/t2_layout_analyzer.py` | `src/preprocess/layout_analyzer.py` | 수정 이관 — TASK_PROMPTS 제거, RawLayoutResult 반환 | ✅ |
| `src/pipeline/orchestrator.py` | `src/pipeline/orchestrator.py` | 재작성 — P3-A→InstructionRouter→ResolutionRouter→P3-B 흐름 | ✅ |
| `src/layer3_correction/t9_confidence_scorer.py` | `src/postprocess/validator.py` | 대폭 수정 — 처리 경로별 임계값 분리 | ✅ |
| `src/layer4_transform/t11_serializer.py` | `src/postprocess/serializer.py` | 그대로 이관 | ✅ |
| `src/layer4_transform/t12_db_loader.py` | `src/postprocess/db_loader.py` | 그대로 이관 | ✅ |

### 5-2. 폐기 완료 (v2에서 대체됨)

| v1 소스 파일 | 대체된 v2 컴포넌트 |
|-------------|-------------------|
| `t3_form_classifier.py` | P3-A FormClassifier |
| `t4_handwriting_recognizer.py` | P3-B StructuredExtractor |
| `t5_structure_recognizer.py` | P3-B StructuredExtractor |
| `t6_domain_code_recognizer.py` | P3-B 도메인 코드 감지 (`_detect_domain_codes`) |
| `t7_language_model_corrector.py` | P3-B VLM 문맥 교정 |
| `t8_data_validator.py` | P4 룰 검증 |
| `t10_schema_mapper.py` | P3-B guided_json |

### 5-3. 신규 구현 (v1에 없는 것)

| v2 소스 파일 | 내용 |
|-------------|------|
| `src/preprocess/layout_postprocessor.py` | P2.5-A — 미소/중복 박스 제거, 블록 병합, `_remap_reading_order()` (제거/병합 반영 reading_order 재정렬) |
| `src/preprocess/layout_server.py` + `layout_http_client.py` | P2 PP-DocLayout HTTP 추론 서비스 — `uvicorn` 기반 컨테이너 분리 |
| `src/vlm/form_classifier.py` | P3-A — 전체 페이지 저해상도 서식 분류 |
| `src/vlm/instruction_router.py` | P2.5-B — form_type 반영 도메인 instruction (instruction_builder.py 대체 완료) |
| `src/vlm/resolution_router.py` | P2.5-C — 영역별 pixel_budget 할당 + 배치 그룹화 |
| `src/vlm/structured_extractor.py` | P3-B — 배치 병렬 VLM 추출 + 도메인 코드 감지 (gemma4_engine.py 분리) + 영역별 trace 옵션 |
| `src/vlm/vlm_client.py` | VLM 공용 클라이언트 — vLLM HTTP 통신 + 이미지 base64 인코딩 + logprobs 파싱 |
| `src/vlm/repetition_guard.py` | 반복 생성 탐지 + 절단 (운영 후 판단) |
| `src/domain/schemas/v1/*.json` | 버전 관리 도입 — schema_id: "supply_request:v1" |
| `src/domain/schema_registry.py` | form_type + version → Schema 조회 |
| `src/pipeline/orchestrator.py` | PipelineResult 확장 — `p2_5b_instructions`, `p2_5c_groups`, `p3b_trace` 필드로 단계별 진단 정보 보관 |
| `src/pipeline/server.py` | Pipeline FastAPI HTTP 래퍼 (`POST /pipeline/run`, `/pipeline/upload`, `GET /health`) — Dockerfile.pipeline 엔트리포인트 |
| `src/fallback/server.py` | Fallback FastAPI HTTP 래퍼 (`POST /fallback/process`, `GET /health`) — Dockerfile.fallback 엔트리포인트. v1/v2 enum 충돌 회피 위해 컨테이너 분리 필수 |

### 5-4. 이관 순서 (완료)

```
1순위: interfaces/ (타입 정의 — RawLayoutResult, InstructionSpec, CroppedRegion 추가) ✅
2순위: preprocess/ (P1 + P2 + P2.5-A LayoutPostProcessor + layout HTTP 서비스) ✅
3순위: vlm/ (P3-A FormClassifier + P2.5-B InstructionRouter + P2.5-C ResolutionRouter + P3-B StructuredExtractor) ✅
4순위: postprocess/ (P4 임계값 분리 + P5 + P6) ✅
5순위: pipeline/ (오케스트레이터 재작성 + PipelineResult 확장 + HTTP server.py) ✅
6순위: fallback/ (HTTP server.py + http_client) ✅
7순위: 통합 테스트 (Phase 1-E) — 2026-04-10 통과 ✅
```

---

## 6. 교정 데이터 축적 파이프라인

검토 큐 UI MVP (Phase 2-A)에서 담당자가 수정한 필드가 Fine-tuning 학습 데이터로 자동 변환됩니다.

```python
# review_queue.export_training_pairs() — 기 구현
# scripts/export_training_data.py — 기 구현

# 출력 형식:
# --format sft    → P3-B SFT JSONL (크롭 이미지 + instruction + 정답 JSON)
# --format dpo    → P3-B DPO JSONL (chosen/rejected 쌍)
# --format classify → P3-A FormClassifier JSONL (서식 이미지 + form_type)
```

**DPO 데이터 자동 축적**: P4 검증 실패 출력(산술 불일치, NSN 형식 오류 등)을 Rejected로 자동 수집. 시스템 운영 3개월 후 충분한 쌍이 축적되면 DPO 착수.

---

## 7. Docker 학습 환경

| 서비스 | Dockerfile | 프레임워크 | GPU |
|--------|-----------|-----------|-----|
| `train` | `docker/Dockerfile.train` | PyTorch + PaddlePaddle + PEFT + vLLM | 전체 |

**하드웨어 요구사항**:

| 학습 단계 | VRAM | 비고 |
|----------|------|------|
| PP-DocLayout Fine-tuning | 16GB+ | PaddlePaddle |
| P3-A FormClassifier SFT (LoRA Q4) | 8~16GB | 소규모 분류 |
| P3-B SFT 1단계 (LoRA Q4) | 16~24GB | PEFT + bitsandbytes |
| P3-B 수기 강화 | 16~24GB | 1단계와 동일 |
| P3-B DPO | 24GB+ | DPOTrainer (trl) |
| 저장 | SSD 500GB+ | 모델 가중치 + 학습 데이터 |

---

## 8. 합성 데이터 생성

| 태스크 | 스크립트 | 방법 |
|--------|---------|------|
| FormClassifier 학습 데이터 | `scripts/generate_classify_pairs.py` (Phase 2 예정) | 서식별 전체 페이지 이미지 + form_type 레이블 |
| SFT 학습 데이터 | `scripts/generate_sft_pairs.py` (Phase 2 예정) | P2.5-A 정제 + bbox 크롭 → 수동 정답 JSON |
| 표 학습 데이터 | `scripts/generate_table_pairs.py` (Phase 2 예정) | 표 crop (1120 토큰) + HTML 정답 |
| 수기 학습 데이터 | `scripts/generate_handwriting_pairs.py` (Phase 2 예정) | AI Hub 손글씨 + 군수 서식 수기 crop |
| DPO 데이터 | (자동 수집) | P4 검증 실패 출력 → Rejected 자동 축적 |
