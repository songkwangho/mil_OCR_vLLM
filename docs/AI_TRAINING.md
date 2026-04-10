# AI 재학습 (Training / Fine-tuning) 가이드

> Fine-tuning 전략, 합성 데이터 생성, 교정 데이터 파이프라인을 기술합니다.

---

## 1. Fine-tuning 대상

| 모델 | 컴포넌트 | Fine-tuned 출력 경로 | 상태 |
|------|---------|-------------------|------|
| PP-DocLayout_plus-L | P2 레이아웃 탐지 | `models/t2_layout/PP-DocLayout_plus-L_finetuned/` | ❌ 미시작 (Phase 2) |
| Gemma4 (P3-A LoRA, rank=8) | FormClassifier | `models/gemma4/adapters/form_classifier_v1/` | ❌ 미시작 (Phase 2) |
| Gemma4 (P3-B LoRA, rank=16) | StructuredExtractor | `models/gemma4/adapters/structured_extractor_v1/` | ❌ 미시작 (Phase 2) |

> **adapter 경로 분리 이유**: P3-A(분류 전용, rank=8)와 P3-B(추출 전용, rank=16)는 학습 태스크·rank가 달라 동일 경로에 저장 시 덮어쓰기 충돌 발생.

> **Fine-tuning 선행 조건**: PP-DocLayout 검출률 측정 후 어노테이션 목표 수량 확정.

---

## 2. 학습 데이터 목표

| 대상 | 데이터 유형 | 목표 규모 | 경로 | 포맷 |
|------|-----------|----------|------|------|
| P2 레이아웃 | 군수 서식 영역 어노테이션 | 2,000~5,000장 (검출률 기반) | `data/training/layout/` | COCO |
| P3-A FormClassifier | 서식 이미지 + form_type | 유형당 100~200장 | `data/training/vlm_classify/` | JSONL |
| P3-B SFT 1단계 | 크롭 이미지 + 정답 JSON | 유형당 300장+ | `data/training/vlm_sft/` | JSONL |
| P3-B 수기 강화 2단계 | 수기 crop + 정답 텍스트 | 10,000장+ | `data/training/vlm_handwriting/` | JSONL |
| P3-B DPO 3단계 | Chosen/Rejected 출력 쌍 | 운영 3개월 후 자동 축적 | `data/training/vlm_dpo/` | JSONL |

---

## 3. PP-DocLayout Fine-tuning

| 항목 | 값 |
|------|------|
| 스크립트 | `training/layout/train_pp_doclayout.py` (Phase 2 예정) |
| 데이터 포맷 | COCO 어노테이션 |
| 카테고리 | text, table, header, footer, signature, seal, figure, formula, handwritten_field, checkbox |
| Pretrained | `models/t2_layout/PP-DocLayout_plus-L/` |
| Docker | `train` (PaddlePaddle 환경) |

**목표 수량 조정 기준** (Phase 1 잔여 검출률 측정 결과):

| 검출률 | 목표 | 추가 조치 |
|--------|------|---------|
| > 70% | 2,000장 | — |
| 50~70% | 3,000장+ | Fusion ON 권장 |
| < 50% | 재협의 | HEURISTIC fallback 비중 확대 |

**PP-DocLayoutV3 대응 전략 (Fusion 모드 사용 시)**:

```
1단계 (즉시): confidence threshold 튜닝
    → 군수 서식 오검출 억제, 추가 학습 없이 가능
2단계 (검출률 < 50% 지속 시): V3 Fine-tuning 검토
    → 어노테이션 형식: Mask R-CNN (polygon 기반)
    → plus-L COCO 어노테이션 재사용 불가 — 별도 작업 필요
    → 예상 작업량: plus-L 대비 약 1.5~2배
```

---

## 4. Gemma4 VLM Fine-tuning

### 4-1. P3-A FormClassifier Fine-tuning

**목적**: 군수 서식 + other 분류 정확도 향상

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_classify.py` (Phase 2 예정) |
| 방식 | LoRA (rank=8, alpha=16) |
| pixel_budget | 140 토큰 (추론 시와 동일) |
| 목표 | 유형당 100~200장 |

```jsonl
{"image": "form_001.png", "instruction": "이 문서의 서식 유형을 분류하세요.", "response": "{\"form_type\": \"supply_request\", \"confidence\": 0.98}"}
{"image": "general_doc.png", "instruction": "이 문서의 서식 유형을 분류하세요.", "response": "{\"form_type\": \"other\", \"confidence\": 0.95}"}
```

### 4-2. P3-B StructuredExtractor Fine-tuning — 3단계

#### 1단계 — SFT (필수)

**학습 데이터 포맷** — 크롭 이미지 기반, CoT 포함:

```jsonl
{
  "image": "table_crop_001.png",
  "pixel_budget": 1120,
  "instruction": "이 표의 구조를 분석하고 모든 셀을 추출하세요. (보급청구서)",
  "response": "{\"analysis\": \"표 구조 명확, 셀 경계 뚜렷, NSN 형식 확인\", \"html\": \"<table>...</table>\", \"cells\": [...]}"
}
{
  "image": "handwritten_crop_001.png",
  "pixel_budget": 1120,
  "instruction": "이 수기 기입란에서 NSN 코드와 수량을 인식하세요.",
  "response": "{\"analysis\": \"수기 필체 보통, '8'과 '6' 구분 필요, NSN 형식 확인됨\", \"nsn\": \"1005-01-432-1234\", \"quantity\": 10}"
}
```

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_sft.py` (Phase 2 예정) |
| 방식 | LoRA (rank=16, alpha=32) |
| 학습 VRAM | 16~24GB |
| 목표 | 유형당 300장+ (착수 마일스톤: 유형당 100건) |

**학습 태스크 구성**:

| 태스크 | 비중 |
|--------|------|
| 전체 서식 필드 추출 | 30% |
| 수기 OCR (텍스트 crop) | 25% |
| 표 구조 인식 (표 crop) | 20% |
| 특수 코드 인식 (NSN/K-NSN) | 15% |
| 인장 인식 (seal crop) | 10% |

#### 2단계 — 수기 인식 강화 (선택)

| 항목 | 값 |
|------|------|
| 조건 | SFT 후 수기 필드 인식 오류 > 15% |
| 방식 | LoRA 추가 학습 (1단계 가중치 위에 누적) |

#### 3단계 — DPO (선택)

| 항목 | 값 |
|------|------|
| 조건 | 시스템 운영 3개월 후 — P4 검증 실패 출력이 Rejected로 자동 축적 |
| 방식 | DPO (Direct Preference Optimization) |
| 데이터 | `scripts/export_training_data.py --format dpo` |

```jsonl
{
  "image": "supply_crop_001.png",
  "instruction": "수량, 단가, 합계를 추출하세요.",
  "chosen": "{\"analysis\": \"숫자 명확, 산술 일치 확인\", \"quantity\": 10, \"unit_price\": 1200000, \"total\": 12000000}",
  "rejected": "{\"analysis\": \"숫자 불명확\", \"quantity\": 10, \"unit_price\": 1200000, \"total\": 1200000}"
}
```

### 4-3. Fine-tuning 순서 및 게이트 조건

```
[Phase 1 잔여]
PP-DocLayout 검출률 측정 (50장 샘플)
    ↓
[Phase 2-A Track A]
PP-DocLayout Fine-tuning → 검출률 70%+ 확인
    ↓ ← 이 게이트를 통과해야 Track B 교정 데이터를 SFT에 투입 가능
[Phase 2-A Track B — 병행]
검토 큐 UI MVP → 교정 데이터 축적 (UI 구현은 선행 가능)
    ↓ 유형당 100건 달성 후
[Phase 2-B]
P3-A FormClassifier SFT
    ↓
P3-B StructuredExtractor SFT 1단계
    ↓ 벤치마크 후 판단
P3-B 2단계 수기 강화 [선택]
    ↓
P3-B 3단계 DPO [선택 — 운영 3개월 후]
```

---

## 5. v1 → v2 코드 이관 기록

### 5-1. 이관 완료

| v1 소스 | v2 대상 | 방식 |
|---------|---------|------|
| `t1_preprocessor.py` | `src/preprocess/preprocessor.py` | 그대로 이관 |
| `sr_enhancer.py` | `src/preprocess/sr_enhancer.py` | 그대로 이관 |
| `t2_layout_analyzer.py` | `src/preprocess/layout_analyzer.py` | 수정 이관 — TASK_PROMPTS 제거 |
| `t9_confidence_scorer.py` | `src/postprocess/validator.py` | 대폭 수정 — 경로별 임계값 분리 |
| `t11_serializer.py` | `src/postprocess/serializer.py` | 그대로 이관 |
| `t12_db_loader.py` | `src/postprocess/db_loader.py` | 그대로 이관 |

### 5-2. 폐기 (v2에서 대체됨)

| v1 소스 | 대체 컴포넌트 |
|---------|-------------|
| `t3_form_classifier.py` | P3-A FormClassifier |
| `t4_handwriting_recognizer.py` | P3-B StructuredExtractor |
| `t5_structure_recognizer.py` | P3-B StructuredExtractor |
| `t6_domain_code_recognizer.py` | P3-B instruction 포함 |
| `t7_language_model_corrector.py` | P3-B VLM 문맥 교정 |
| `t8_data_validator.py` | P4 룰 검증 병합 |
| `t10_schema_mapper.py` | P3-B guided_json |

### 5-3. 신규 구현 (v1에 없음)

| v2 소스 | 내용 |
|---------|------|
| `src/preprocess/layout_postprocessor.py` | P2.5-A — 정제 + remap |
| `src/vlm/form_classifier.py` | P3-A — military/other 분류 |
| `src/vlm/instruction_router.py` | P2.5-B — 1-shot + CoT 포함 |
| `src/vlm/resolution_router.py` | P2.5-C — pixel_budget + 크롭 패딩 |
| `src/vlm/structured_extractor.py` | P3-B — 배치 병렬 추출 |
| `src/domain/schemas/v1/_general.json` | other 문서용 범용 스키마 |
| `src/domain/schema_registry.py` | 버전 관리 |
| `configs/instruction_examples/*.yaml` | 1-shot 예시 관리 |

---

## 6. 교정 데이터 → Fine-tuning 파이프라인

```python
# 검토 큐 UI에서 담당자 교정 완료 → 자동 변환
scripts/export_training_data.py

# 출력 형식
# --format classify → P3-A JSONL (서식 이미지 + form_type, other 포함)
# --format sft      → P3-B JSONL (크롭 + instruction + 정답 JSON, CoT 포함)
# --format dpo      → P3-B JSONL (chosen/rejected, P4 실패 → rejected 자동 축적)
```

---

## 7. 합성 데이터 생성

| 태스크 | 스크립트 | 방법 |
|--------|---------|------|
| FormClassifier 학습 | `scripts/generate_classify_pairs.py` (Phase 2) | 서식 이미지 + form_type 레이블 (other 포함) |
| SFT 학습 | `scripts/generate_sft_pairs.py` (Phase 2) | P2.5-A 정제 + bbox 크롭 → 정답 JSON (CoT 포함) |
| 표 학습 | `scripts/generate_table_pairs.py` (Phase 2) | 표 crop (1120토큰) + HTML 정답 |
| 수기 학습 | `scripts/generate_handwriting_pairs.py` (Phase 2) | AI Hub 손글씨 + 군수 서식 수기 crop |
| DPO | (자동 수집) | P4 검증 실패 → Rejected 자동 축적 |

---

## 8. Docker 학습 환경

| 서비스 | Dockerfile | 프레임워크 |
|--------|-----------|-----------|
| `train` | `docker/Dockerfile.train` | PyTorch + PaddlePaddle + PEFT + vLLM |

**하드웨어 요구사항**:

| 학습 단계 | VRAM |
|----------|------|
| PP-DocLayout Fine-tuning | 16GB+ |
| P3-A FormClassifier SFT (LoRA Q4) | 8~16GB |
| P3-B SFT 1단계 (LoRA Q4) | 16~24GB |
| P3-B 수기 강화 | 16~24GB |
| P3-B DPO | 24GB+ |
| 저장 | SSD 500GB+ |
