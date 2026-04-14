# 군수 OCR 시스템 v2 — 마스터 컨텍스트

> 이 파일은 프로젝트 전체 맥락의 **진입점**입니다.
> 상세 내용은 `docs/` 하위 문서를 참조하십시오.

---

## 핵심 원칙

- **폐쇄망 전용** — 외부 API 호출·모델 다운로드 절대 금지. 모든 가중치는 `models/`에 사전 배치
- **Crop-then-Infer** — PP-DocLayout bbox 크롭 이미지를 영역별로 Gemma4에 전달 (전체 페이지 입력 지양)
- **FormClassifier 선행** — 서식 분류 후 InstructionRouter가 도메인 맥락 기반 instruction 생성
- **Other 문서 분기** — 군수 서식이 아닌 문서는 `other`로 분류해 Skill Registry 기반 처리
- **Guided Decoding** — VLM 출력을 서식별 JSON Schema로 구조 보장 (xgrammar 백엔드)
- **logprobs 신뢰도** — VLM 토큰 확률 기반 필드별 정밀 신뢰도 산출
- **OCR-augmented** — 저신뢰 영역에 경량 OCR(PaddleOCR) 선행 후 텍스트 힌트를 VLM에 제공
- **오프라인 모델 로드** — `HF_HUB_OFFLINE=1`, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`

---

## 문서 구조

| 문서 | 내용 | 작업 시 읽을 것 |
|------|------|----------------|
| `docs/PIPELINE.md` | 파이프라인 구조·컴포넌트 설계·인터페이스 타입·SPOF | 파이프라인·컴포넌트 작업 시 |
| `docs/DOMAIN.md` | 서식 유형·코드 체계·JSON Schema·Other 처리 정책 | 도메인 지식·스키마 작업 시 |
| `docs/AI_INFERENCE.md` | 모델 가중치·vLLM 서빙·최적화 설정·Docker | AI 추론 작업 시 |
| `docs/AI_TRAINING.md` | Fine-tuning 전략·합성 데이터·교정 데이터 파이프라인 | AI 재학습 작업 시 |
| `docs/TESTING.md` | 통합 테스트 절차·결과 구조·타이밍 기록 | 테스트 작업 시 |
| `docs/FRONTEND.md` | 검토 큐 UI·API 엔드포인트 (현재 미구현) | 프론트엔드 작업 시 |

**작업 시 이 파일(CLAUDE.md) + 해당 영역의 docs 파일만 읽어서 진행합니다.**

---

## 시스템 파이프라인 (v2 — 고도화)

```
[Stage 1 — 전처리]
    P1:     화질 보정 + SR          → PreprocessedImage
        - LOW DPI(<150): Real-ESRGAN SR → CLAHE 대비 강화 → 기울기 보정(Deskew)
        - 이진화 금지 / RGB 유지 (SigLIP 인코더 자연 이미지 기준)
        - DPI 300 표준화 권장
    P2:     레이아웃 탐지            → RawLayoutResult
        ├── Fusion OFF: V3 단독 탐지
        └── Fusion ON (LAYOUT_FUSION_MODE=true):
            ├── DPI ≥ 150: V3(구조) + plus-L(텍스트), 50% 겹침 필터
            └── DPI < 150: V3(구조) + OpenCV heuristic(텍스트), 50% 겹침 필터
            reading_order: V3 구조 영역 순서 → 텍스트 좌상→우하 정렬
    P2.5-A: LayoutPostProcessor     → LayoutResult
        ① 미소 박스 제거 (6px 미만)
        ② 중복 박스 제거 (IoU > 0.7)
        ③ 인접 텍스트 블록 병합
        ④ reading_order 재정렬 (_remap_reading_order)
        ↓
[Stage 2 — VLM 통합 추론]              ┌─ [Fallback — VLM 불가 시]
    P3-A: FormClassifier               │   v1 T3~T5 (PP-OCRv5)
        전체 페이지 → 저해상도(140토큰) │   → 서식 분류 + 텍스트 인식 + 표 구조
        → form_type 확정               │   → 결과는 항상 검토 큐 적재
        ┌─ military: 군수 서식 경로    │
        └─ other:    Skill Registry    │
        ↓                             │
    [military 경로 — 기존 v2 유지]     │
    P2.5-B: InstructionRouter          │
        region_type + form_type        │
        → 도메인 맥락 포함 instruction  │
        → JSON Schema 선택             │
        → 1-shot 예시 포함             │
        → OCR 힌트 삽입 (저신뢰 영역)  │
        ↓                             │
    P2.5-C: ResolutionRouter           │
        배치 순서: 140→560→1120 토큰   │
        크롭 크기 48px 배수 정렬        │
        ↓                             │
    P3-B: StructuredExtractor          │
        temperature=0.0, 결정론적      │
        CoT analysis 필드 포함         │
        logprobs 신뢰도 산출           │
        저신뢰 필드 자동 재시도         │
        + S4/S5/S6 Skill 통합 호출    │
        ↓                             │
    [other 경로 — Skill Registry]      │
    S1: LayoutAnalyzer (결재란 휴리스틱 포함)
    S5-패스1: TableExtractor (표 구조 추출)
    S6(140): SignatureDetector
    S2(560): PrintedTextReader
    S3+S4+S5패스2(1120): HandwritingReader + SealReader + 셀 내용
    S7: StructuredAggregator (official_document.json Schema)
        ↓                            └───────────────┐
[Stage 3 — 후처리]                                   │
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult       │
        처리 경로별 임계값 적용 (VLM / Fallback)       │
        ├─ military 문서 + 통과 → P5 → P6 DB 적재    │
        ├─ other 문서 → P5 범용 직렬화 (검토 큐 없음) │
        └─ 실패 → 검토 큐 → 담당자 UI → P6 재적재    ↓
```

### 두 경로의 설계 사상

| 항목 | military 경로 | other 경로 (Skill Registry) |
|------|-------------|---------------------------|
| 핵심 사상 | 서식을 먼저 알고, 아는 서식에 맞춰 추출 | 문서 구조를 먼저 이해하고, 구조에 따라 추출 |
| form_type 활용 | InstructionRouter 전체를 결정 | S7 StructuredAggregator에서만 Schema 선택에 사용 |
| 표 처리 | StructuredExtractor 단일 호출 | TableExtractor 2패스 (구조→셀 내용) |
| 인장 처리 | StructuredExtractor 안에서 처리 | SealPreprocessor(극좌표 변환) + SealReader |
| 서명 처리 | 미지원 | SignatureDetector (이진 분류) |
| guided JSON | 서식별 Schema 강제 | official_document.json Schema |

### 핵심 메커니즘

- **Layout Fusion**: V3(구조) + plus-L(텍스트) 결합, DPI 기반 분기
- **Crop-then-Infer**: LayoutPostProcessor 정제 → 영역별 크롭(48px 배수) → Gemma4
- **FormClassifier 선행**: 저해상도 분류 → form_type → 경로 분기
- **military InstructionRouter**: region_type + form_type → 도메인 맥락 + 1-shot 예시
- **OCR-augmented**: 저신뢰 영역에 PaddleOCR 선행 → 텍스트 힌트로 VLM 정확도 보강
- **ResolutionRouter 배치 순서**: 140→560→1120 토큰 그룹 순차 배치 (vLLM 패딩 최소화)
- **SealPreprocessor**: HSV 적색 분리 → 허프 원 탐지 → 극좌표 직선화 (학습 불필요)
- **TableExtractor 2패스**: 패스1(구조 추출) → 패스2(셀별 Skill 라우팅)
- **SignatureDetector**: 서명 존재 여부 이진 분류 (OCR 아님)
- **저신뢰 재시도**: logprobs 임계값 미달 → pixel_budget 상향 후 재호출
- **SPOF 대비**: 수준A(헬스체크) + 수준B(fallback) + 수준C(검토 큐)

---

## 로드맵

### Phase 1 잔여 — 남은 핵심 작업

- [ ] **S2 PrintedTextReader / S3 HandwritingReader 정식 구현** — 현재는 skill_registry의 `_stub_text_skill`이 guided_json으로 대체 처리. 한국어 수기 인식률 실측(10~20장) 후 임계값 보정 필요.
- [ ] **S7 StructuredAggregator 구현** — 현재 Skill 결과를 region_id 단위 FieldValue로 평탄화. official_document 스키마에 맞춘 최종 집계가 필요.
- [ ] **OCR-augmented 힌트** — `ocr_hint_provider.py`는 구현됐으나 PaddleOCR 가중치의 오프라인 배치(`~/.paddlex/official_models/`)와 `Dockerfile.pipeline` COPY 반영 필요.
- [ ] **1-shot 예시 자산화** — `configs/instruction_examples/*.yaml` 서식별 예시 작성.
- [ ] **검출률 측정** — `scripts/evaluate_layout_detection.py --n 50` (PP-DocLayoutV3 기준치 확보).
- [ ] **T8/T9/T10 통합 테스트 확장** — 인장/결재란 2패스/서명 탐지 시나리오 결과 축적.
- [ ] **vLLM 변동성 측정** — 동일 문서 N=3 반복으로 기준치 수립 (전역지원서_2 이전 +99% 케이스).

### Phase 2 (06~07월)

**2-A. 검토 큐 UI MVP + 교정 데이터 축적 (병행)**
- [ ] 검토 큐 UI MVP (큐 목록 + 검토 + 필드 수정 + 승인/반려)
- [ ] 교정 데이터 JSONL export 파이프라인 활성화
- [ ] PP-DocLayout 어노테이션 + Fine-tuning (검출률 기반 목표)

> Track A(Fine-tuning) 검출률 70%+ 달성 후에만 Track B 교정 데이터를 SFT에 투입

**2-B. Gemma4 VLM Fine-tuning** (유형당 100건+ 확보 후)
- [ ] P3-A SFT (LoRA rank=8, adapters/form_classifier_v1/)
- [ ] P3-B SFT 1단계 (LoRA rank=16, adapters/structured_extractor_v1/)
- [ ] 벤치마크 후 2단계 수기 강화(선택) / 3단계 DPO(운영 3개월 후) 판단

### Phase 3 (08~09월)
- [ ] RepetitionGuard 도입 (반복 토큰 비율 > 15% 관측 시)
- [ ] 양자화 실험 (AWQ 4-bit + Marlin, FP8은 vllm#39049 해소 후)
- [ ] 검토 큐 UI 고도화 (대시보드, bbox 오버레이, 담당자 관리)
- [ ] 비동기 메시지 기반 통신 전환 (Redis Streams)
- [ ] Docker 프로덕션 설정 + 보안 검토 + 배포

---

## v1 → v2 컴포넌트 매핑

| v1 | v2 | 비고 |
|----|-----|------|
| T1 화질 보정 | **P1** 화질 보정 + SR | v1에서 이관 |
| T2 레이아웃 분석 | **P2** 레이아웃 탐지 (Fusion 지원) | 원시 탐지만 반환 |
| — | **P2.5-A** LayoutPostProcessor | 신규 — 정제 + remap |
| T3 서식 분류 | **P3-A** FormClassifier | military/other 분기 포함 |
| — | **P2.5-B** InstructionRouter | 신규 — 1-shot + CoT + OCR 힌트 (military) |
| — | **P2.5-C** ResolutionRouter | 신규 — pixel_budget + 48px 정렬 + 배치 순서 |
| T4~T7 | **P3-B** StructuredExtractor | 배치 병렬 + 저신뢰 재시도 (military) |
| — | **S2~S7** Skill Registry | 신규 — other 경로 전용 |
| — | **SealPreprocessor** | 신규 — 극좌표 변환 (파인튜닝 불필요) |
| T8 데이터 검증 | **P4** 룰 검증 | 경로별 임계값 분리 |
| T9 신뢰도 | **P3-B** (logprobs) | 기하평균, 한국어 임계값 보정 |
| T10 스키마 매핑 | **P3-B** (guided_json) | JSON Schema 구조화 |
| T11 직렬화 | **P5** 직렬화 | v1에서 이관 |
| T12 DB 적재 | **P6** DB 적재 | v1에서 이관 |

---

## 구현 현황

| P# | 컴포넌트 | 소스 파일 | 상태 |
|----|----------|----------|------|
| — | 공용 인터페이스 | `src/interfaces/` | ✅ Enum+dataclass (SkillTask/Result, TableStructure, SealProcessResult 포함) |
| — | 오케스트레이터 | `src/pipeline/orchestrator.py` | ✅ military/other 분기 완료 (`_process_other_document`) |
| P1 | 화질 보정 + SR | `src/preprocess/preprocessor.py` | ✅ |
| P2 | 레이아웃 탐지 (Fusion) | `src/preprocess/layout_analyzer.py` | ✅ |
| P2.5-A | LayoutPostProcessor | `src/preprocess/layout_postprocessor.py` | ✅ |
| — | SealPreprocessor | `src/preprocess/seal_preprocessor.py` | ✅ 극좌표 언래핑 + 허프 실패 폴백 |
| P3-A | FormClassifier | `src/vlm/form_classifier.py` | ✅ military/other 분기 |
| P2.5-B | InstructionRouter | `src/vlm/instruction_router.py` | ✅ 1-shot + OCR 힌트 |
| P2.5-C | ResolutionRouter | `src/vlm/resolution_router.py` | ✅ 48px 정렬 + DISPATCH_ORDER |
| P3-B | StructuredExtractor | `src/vlm/structured_extractor.py` | ✅ 저신뢰 재시도 로직 포함 |
| — | OCR 힌트 제공자 | `src/vlm/ocr_hint_provider.py` | 🟡 구현 완료, 폐쇄망 가중치 배치 필요 |
| — | Skill Registry | `src/vlm/skill_registry.py` | ✅ DISPATCH_ORDER [140,560,1120] + SkillDispatchStats |
| — | S2 PrintedTextReader | `src/vlm/skills/printed_text_reader.py` | 🔴 미구현 (Stub 대체 중) |
| — | S3 HandwritingReader | `src/vlm/skills/handwriting_reader.py` | 🔴 미구현 (Stub 대체 중) |
| — | S4 SealReader | `src/vlm/skills/seal_reader.py` | ✅ |
| — | S5 TableExtractor | `src/vlm/skills/table_extractor.py` | ✅ pass1/pass2 |
| — | S6 SignatureDetector | `src/vlm/skills/signature_detector.py` | ✅ |
| — | S7 StructuredAggregator | `src/vlm/skills/aggregator.py` | 🔴 미구현 (region_id 단위 평탄화로 임시 대체) |
| — | VLM 공용 클라이언트 | `src/vlm/vlm_client.py` | ✅ |
| P4 | 룰 검증 + 신뢰도 보정 | `src/postprocess/validator.py` | ✅ 경로별 임계값 |
| P5 | 직렬화 | `src/postprocess/serializer.py` | ✅ |
| P6 | DB 적재 | `src/postprocess/db_loader.py` | ✅ |
| — | 수동 검토 큐 | `src/postprocess/review_queue.py` | ✅ |
| — | Fallback 서비스 | `src/fallback/ocr_fallback_service.py` | ✅ |
| — | VLM 헬스 모니터 | `src/pipeline/health_monitor.py` | ✅ |
| — | Layout 추론 서비스 | `src/preprocess/layout_server.py` | ✅ |
| — | 스키마 레지스트리 | `src/domain/schema_registry.py` | ✅ v1/ 스캔 + `get()` 별칭 (`other→official_document`) |
| — | JSON Schemas v1 | `src/domain/schemas/v1/*.json` | ✅ 8종 (5 military + `_fallback` + `_general` + `official_document`) |
| — | Docker 구성 | `docker-compose.yml` + `docker/Dockerfile.*` | ✅ vLLM 0.19.0 호환 (`--structured-outputs-config`) |

> **Legacy 유지**: `instruction_builder.py` (InstructionRouter 래퍼), `gemma4_engine.py` (하위 호환용)

---

## 디렉토리 구조

```
mil_OCR_v2/
├── CLAUDE.md
├── docs/
│   ├── PIPELINE.md
│   ├── DOMAIN.md
│   ├── AI_INFERENCE.md
│   ├── AI_TRAINING.md
│   ├── TESTING.md
│   └── FRONTEND.md
├── src/
│   ├── interfaces/
│   ├── pipeline/
│   ├── preprocess/
│   │   └── seal_preprocessor.py   ← 신규 (극좌표 변환)
│   ├── vlm/
│   │   ├── skill_registry.py
│   │   ├── skills/
│   │   │   ├── seal_reader.py          (S4 ✅)
│   │   │   ├── table_extractor.py      (S5 ✅ 2패스)
│   │   │   ├── signature_detector.py   (S6 ✅)
│   │   │   ├── printed_text_reader.py  (S2 🔴 미구현)
│   │   │   ├── handwriting_reader.py   (S3 🔴 미구현)
│   │   │   └── aggregator.py           (S7 🔴 미구현)
│   │   ├── ocr_hint_provider.py
│   │   ├── form_classifier.py
│   │   ├── instruction_router.py
│   │   ├── resolution_router.py
│   │   ├── structured_extractor.py
│   │   └── vlm_client.py
│   ├── postprocess/
│   ├── fallback/
│   └── domain/
│       ├── schema_registry.py
│       └── schemas/v1/
│           ├── supply_request.json
│           ├── maintenance_record.json
│           ├── inventory_sheet.json
│           ├── handover_doc.json
│           ├── inspection_report.json
│           ├── _fallback.json
│           ├── _general.json
│           └── official_document.json
├── models/
│   ├── t1_sr/
│   ├── t2_layout/
│   ├── gemma4/
│   │   ├── gemma-4-26b-a4b-it/
│   │   └── adapters/
│   │       ├── form_classifier_v1/
│   │       └── structured_extractor_v1/
│   └── fallback/
├── data/
├── docker/
├── docker-compose.yml
├── scripts/
├── training/
├── configs/
│   └── instruction_examples/
└── tests/
```

---

## 검증 필요한 리스크

| # | 이슈 | 보완 계획 |
|---|------|---------|
| R1 | S3 HandwritingReader 한국어 수기 인식률 실측치 부재 | 국회 의안 PDF 등 합성 샘플 10~20장으로 0-shot 인식률 측정 후 임계값 보정 |
| R2 | SealPreprocessor 허프 원 탐지 실패율 미측정 | T8 테스트에서 성공/실패 케이스 집계, 실패율 30%+ 이면 휴리스틱 격하 |
| R3 | vLLM 출력 변동성 기준치 미수립 | 동일 문서 N=3 반복으로 기준 확보 (이전 전역지원서_2 +99% 변동 건) |
| R4 | pass2 셀 태스크 0건 관찰 | S5 pass1 출력 table_type/cells 품질 확인 (2026-04-14 통합 테스트 기준) |

---

## 보안 고려사항

- 군수 서식 이미지 — **로컬 추론 전용** (클라우드 API 금지)
- Gemma4 + PaddleOCR — 로컬 서버 구동 (외부 통신 차단)
- 학습 데이터·도메인 사전 — 암호화 저장 권장
- `VLMResult.raw_json` — DB 적재 제외, 파일 시스템에만 보존

---

## 하드웨어 요구사항

| 용도 | VRAM | GPU | 비고 |
|------|------|-----|------|
| Gemma4 추론 (BF16 + FP8 KV Cache) | ~48GB + KV절감 | H100 80GB | 현재 운영 |
| Gemma4 추론 (AWQ 4-bit + Marlin) | ~16GB | RTX 4090 | Phase 3 실험 |
| Gemma4 추론 (FP8 Dynamic) | ~27GB | A100 | vllm#39049 해소 후 |
| PP-DocLayout 추론 | 4GB | GPU | PaddlePaddle |
| Real-ESRGAN SR | 2GB | GPU | 타일 기반 |
| PaddleOCR (OCR 힌트) | 1GB | GPU (선택) | 경량 mobile 모델 |
| Gemma4 Fine-tuning (LoRA) | 16~24GB | GPU | PEFT |
