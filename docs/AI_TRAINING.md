# AI 재학습 (Training / Fine-tuning) 가이드

> AI 모델의 Fine-tuning 방법, 학습 데이터, Docker 환경, 가중치 출력 경로를 기술합니다.

---

## 1. Fine-tuning 대상 — 2개 모델

| 모델 | 컴포넌트 | Pretrained 출처 | Fine-tuned 출력 | 상태 |
|------|---------|----------------|----------------|------|
| PP-DocLayout_plus-L | P2 구조 분석 | PaddleOCR 3.0 | `models/t2_layout/PP-DocLayout_plus-L_finetuned/` | ❌ 미시작 (Phase 2) |
| Gemma4 26B-A4B | P3 VLM 통합 추론 | google/gemma-4-26b-a4b-it | `models/gemma4/gemma4-mil-finetuned/` | ❌ 미시작 (Phase 2) |

---

## 2. 학습 데이터 목표

| 대상 | 데이터 유형 | 목표 규모 | 데이터 경로 | 포맷 |
|------|-----------|----------|-----------|------|
| P2 구조 분석 | 군수 서식 영역 어노테이션 | 2,000~5,000장 | `data/training/layout/` | COCO (train.json, val.json) |
| P3-1단계 SFT | 서식 이미지 + 정답 JSON | 유형당 200~500장 | `data/training/vlm_sft/` | JSONL (image + instruction + response) |
| P3-2단계 수기 | 수기 crop + 정답 텍스트 | 10,000장+ | `data/training/vlm_handwriting/` | JSONL |
| P3-3단계 DPO | Chosen/Rejected 출력 쌍 | 5,000쌍+ | `data/training/vlm_dpo/` | JSONL (chosen + rejected) |

---

## 3. PP-DocLayout Fine-tuning

| 항목 | 값 |
|------|------|
| 스크립트 | `training/layout/train_pp_doclayout.py` (Phase 2 예정) |
| 데이터 포맷 | COCO 어노테이션 (train.json, val.json) |
| 카테고리 | 군수 서식 특화: text, table, header, footer, signature, seal, figure, formula, handwritten_field, checkbox |
| Pretrained 가중치 | `models/t2_layout/PP-DocLayout_plus-L/` (124MB) |
| Fine-tuned 출력 | `models/t2_layout/PP-DocLayout_plus-L_finetuned/` |
| Docker 서비스 | `train` (PaddlePaddle 환경) |
| 구현 상태 | v1에서 이관 예정 (부분 구현) |

---

## 4. Gemma4 VLM Fine-tuning — 3단계 전략

### 4-1. 1단계 — Instruction Tuning (SFT) [필수]

**목적**: Gemma4의 instruction following 능력을 군수 도메인에 특화

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_sft.py` (Phase 2 예정) |
| 방식 | LoRA (rank=16, alpha=32) |
| 학습 VRAM | 16~24GB (LoRA + Q4 base) |
| 프레임워크 | PyTorch + PEFT + transformers |
| 목표 데이터 | 서식 유형당 200~500장 (9종 × 200~500 = 1,800~4,500장) |

**학습 데이터 포맷 (JSONL)**:

```jsonl
{"image": "supply_001.png", "instruction": "이 보급청구서에서 모든 필드를 추출하세요.", "response": "{\"form_type\":\"supply_request\",\"unit_code\":\"1232부대\",\"items\":[{\"nsn\":\"1005-01-432-1234\",\"quantity\":10}]}"}
{"image": "table_001.png", "instruction": "Table Recognition:", "response": "{\"html\":\"<table><tr><th>품목</th><th>수량</th></tr>...</table>\"}"}
{"image": "text_001.png", "instruction": "OCR:", "response": "{\"text\":\"정비기록서\",\"corrected\":\"정비기록서\"}"}
```

**학습 태스크 구성**:

| 태스크 | 비중 | instruction 유형 |
|--------|------|-----------------|
| 서식 전체 필드 추출 | 30% | 서식별 guided_json 정답 |
| OCR (텍스트 인식) | 25% | "OCR:" + 영역 crop |
| 표 구조 인식 | 20% | "Table Recognition:" + 표 crop |
| 서식 분류 | 10% | "이 서식의 유형을 분류하세요" |
| 코드/날짜 인식 | 15% | "이 영역에서 NSN 코드를 찾으세요" |

### 4-2. 2단계 — 수기 인식 강화 [선택]

**목적**: 한국어 수기(手記) 인식 오류 개선

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_handwriting.py` (Phase 2 예정) |
| 방식 | LoRA 추가 학습 (1단계 가중치 위에 누적) |
| 조건 | 1단계 SFT 이후 수기 필드 인식 오류가 높을 때만 수행 |

**학습 데이터**:
- 군수 서식 수기 기입란 crop + 정답 텍스트
- AI Hub 한글 손글씨 데이터 활용 가능 (공개 데이터셋)
- 목표: 10,000장+

```jsonl
{"image": "handwritten_crop_001.png", "instruction": "OCR:", "response": "{\"text\":\"1232부대\"}"}
{"image": "handwritten_crop_002.png", "instruction": "OCR:", "response": "{\"text\":\"소총 K2 10정\"}"}
```

### 4-3. 3단계 — DPO / RLHF [선택]

**목적**: 산술 정합성, 코드 형식 일관성 등 규칙 준수 강화

| 항목 | 값 |
|------|------|
| 스크립트 | `training/vlm/train_gemma4_dpo.py` (Phase 2 예정) |
| 방식 | DPO (Direct Preference Optimization) |
| 조건 | 1~2단계 이후 검증 규칙 위반이 빈번할 때만 수행 |

**학습 데이터 — Chosen/Rejected 쌍**:

```jsonl
{
  "image": "supply_form.png",
  "instruction": "이 보급청구서에서 모든 필드를 추출하세요.",
  "chosen": "{\"items\":[{\"quantity\":10,\"unit_price\":1200000,\"total\":12000000}]}",
  "rejected": "{\"items\":[{\"quantity\":10,\"unit_price\":1200000,\"total\":1200000}]}"
}
```

- **Chosen**: 합계 = 수량 × 단가 (산술 정합)
- **Rejected**: 합계 오류 (P4 룰 검증 위반 출력을 자동 수집)
- 데이터 구성 방법: 파이프라인 운용 중 P4에서 검증 실패한 출력을 Rejected로 자동 축적

### 4-4. Fine-tuning 선행 조건 및 순서

```
PP-DocLayout Fine-tuning (P2)
       ↓ 레이아웃 검출 품질이 VLM 입력 품질에 직결
Gemma4 1단계 SFT (P3)
       ↓ 기본 instruction following 확보
Gemma4 2단계 수기 강화 (P3) ← 선택: 수기 오류가 많을 때만
       ↓
Gemma4 3단계 DPO (P3) ← 선택: 규칙 위반이 빈번할 때만
```

---

## 5. v1 → v2 코드 이관 계획

### 5-1. 이관 대상 분류

| v1 소스 파일 | 규모 | v2 대상 | 이관 방식 |
|-------------|------|---------|----------|
| `src/interfaces/enums.py` (208줄) | 18 Enum | `src/interfaces/enums.py` | **축소 이관** — VLM 관련 Enum 제거 (CorrectionMethod, TextType 등), 레이아웃/파이프라인 Enum만 유지 |
| `src/interfaces/types.py` (452줄) | 23 dataclass | `src/interfaces/types.py` | **재설계** — v1의 T3~T10 타입 제거, VLMResult/FieldValue/ValidatedResult 신규 정의 |
| `src/layer1_preprocess/t1_preprocessor.py` (847줄) | 화질 보정 | `src/preprocess/preprocessor.py` | **그대로 이관** — SR 포함, import 경로만 변경 |
| `src/layer1_preprocess/sr_enhancer.py` (175줄) | Real-ESRGAN | `src/preprocess/sr_enhancer.py` | **그대로 이관** |
| `src/layer1_preprocess/t2_layout_analyzer.py` (789줄) | 레이아웃 | `src/preprocess/layout_analyzer.py` | **수정 이관** — reading_order 로직 강화, TASK_PROMPTS 매핑 추가, 출력 타입을 LayoutResult로 변경 |
| `src/pipeline/orchestrator.py` (656줄) | 오케스트레이터 | `src/pipeline/orchestrator.py` | **재작성** — 12단계 → 6단계, vLLM API 호출 로직 신규 |
| `src/layer3_correction/t9_confidence_scorer.py` (280줄) | 신뢰도 | `src/postprocess/validator.py` | **대폭 수정** — logprobs 기반으로 변경, 기존 가중 평균 산식 → 토큰 확률 환산, 룰 검증 병합 |
| `src/layer4_transform/t11_serializer.py` (207줄) | 직렬화 | `src/postprocess/serializer.py` | **그대로 이관** — ExtractedDocument 입력 유지 |
| `src/layer4_transform/t12_db_loader.py` (333줄) | DB 적재 | `src/postprocess/db_loader.py` | **그대로 이관** — DB 스키마는 v2에 맞게 조정 |

### 5-2. 폐기 대상 (v2에서 불필요)

| v1 소스 파일 | 사유 |
|-------------|------|
| `t3_form_classifier.py` (513줄) | P3 VLM이 서식 분류 수행 |
| `t4_handwriting_recognizer.py` (547줄) | P3 VLM이 OCR 수행 |
| `t5_structure_recognizer.py` (564줄) | P3 VLM이 표 구조 인식 수행 |
| `t6_domain_code_recognizer.py` (7줄) | P3 VLM이 코드 인식 수행 |
| `t7_language_model_corrector.py` (440줄) | P3 VLM이 교정 수행 |
| `t8_data_validator.py` (8줄) | P4 룰 검증에 병합 |
| `t10_schema_mapper.py` (598줄) | P3 VLM guided_json이 스키마 매핑 수행 |

### 5-3. 신규 구현 (v1에 없는 것)

| v2 소스 파일 | 내용 |
|-------------|------|
| `src/vlm/gemma4_engine.py` | Gemma4 vLLM 엔진 — instruction 생성, guided_json 전달, logprobs 파싱 |
| `src/vlm/instruction_builder.py` | P2 레이블 → VLM instruction 자동 생성, JSON Schema 로드 |
| `src/vlm/logprobs_scorer.py` | logprobs → 필드별 신뢰도 환산, 임계값 비교, is_flagged 결정 |
| `src/domain/schemas/*.json` | 군수 서식별 JSON Schema (guided_json용) |
| `src/postprocess/validator.py` | 룰 검증 + logprobs 신뢰도 보정 (v1 T8+T9 병합 + 재설계) |

### 5-4. 이관 시 공통 수정 사항

| 수정 항목 | 내용 |
|----------|------|
| import 경로 | `src.layer1_preprocess.` → `src.preprocess.` 등 |
| 인터페이스 타입 | v1 dataclass → v2 dataclass (LayoutResult, VLMResult 등) |
| Enum 참조 | 불필요한 Enum 제거, 새 Enum 추가 (PipelineStatus 유지) |
| config 클래스 | v1의 개별 Config → v2 통합 Config (PipelineConfig 단순화) |

### 5-5. 이관 우선순위

```
1순위: interfaces/ (타입 정의 — 모든 컴포넌트의 기반)
2순위: preprocess/ (P1 + P2 — VLM 입력 생성)
3순위: vlm/ (P3 — 신규 구현, 이관 아님)
4순위: postprocess/ (P4~P6 — T9, T11, T12 이관 + 수정)
5순위: pipeline/ (오케스트레이터 — 전체 연결)
```

---

## 6. Docker 학습 환경

| 서비스 | Dockerfile | 프레임워크 | GPU |
|--------|-----------|-----------|-----|
| `train` | `docker/Dockerfile.train` | PyTorch + PaddlePaddle + PEFT + vLLM | 전체 |

**하드웨어 요구사항**:

| 학습 단계 | VRAM | 비고 |
|----------|------|------|
| PP-DocLayout Fine-tuning | 16GB+ | PaddlePaddle |
| Gemma4 1단계 SFT (LoRA Q4) | 16~24GB | PEFT + bitsandbytes |
| Gemma4 2단계 수기 강화 | 16~24GB | 1단계와 동일 |
| Gemma4 3단계 DPO | 24GB+ | DPOTrainer (trl) |
| 저장 | SSD 500GB+ | 모델 가중치 + 학습 데이터 |

---

## 7. 합성 데이터 생성

| 태스크 | 스크립트 | 방법 |
|--------|---------|------|
| SFT 학습 데이터 | `scripts/generate_sft_pairs.py` (Phase 2 예정) | P2로 영역 crop → 수동 정답 JSON 작성 |
| 표 학습 데이터 | `scripts/generate_table_pairs.py` (Phase 2 예정) | 표 crop + HTML 정답 생성 |
| 수기 학습 데이터 | `scripts/generate_handwriting_pairs.py` (Phase 2 예정) | AI Hub 손글씨 + 군수 서식 수기 crop |
| DPO 데이터 | (자동 수집) | 파이프라인 운용 중 P4 검증 실패 출력을 Rejected로 축적 |
