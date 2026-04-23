# 군수 OCR 시스템 v2 — 마스터 컨텍스트

> 이 파일은 프로젝트 전체 맥락의 **진입점**입니다.
> 상세 내용은 `docs/` 하위 문서를 참조하십시오.

---

## 핵심 원칙

- **폐쇄망 전용** — 외부 API 호출·모델 다운로드 절대 금지. 모든 가중치는 `models/`에 사전 배치
- **Crop-then-Infer** — PP-DocLayout bbox 크롭 이미지를 영역별로 Gemma4에 전달 (전체 페이지 입력 지양)
- **FormClassifier 선행** — 서식 분류 후 InstructionRouter가 도메인 맥락 기반 instruction 생성
- **TemplateAugmentor 보완** — form_type 확정 후 서식별 미리 정의된 bbox로 PP-DocLayout 누락 영역 병합 (military 전용)
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
| `docs/equipment_checklist_design.md` | 전비품 확인서(x-assembly-rules) 설계 명세 | 체크리스트 서식 작업 시 |
| `docs/pdf_adapter_design.md` | PdfAdapter + PdfDocumentResult 설계 명세 | PDF 입력·멀티페이지 집계 작업 시 |

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
    [military 경로]                    │
    P2.5-A.5: TemplateAugmentor       │  ← 신규 (form_type 확정 직후)
        form_type별 서식 템플릿 로드    │
        configs/form_templates/*.yaml  │
        PP-DocLayout 결과와 병합        │
        IoU>0.5: 탐지 결과 우선        │
        미탐지 필드: 템플릿 bbox 보완  │
        ↓                             │
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

### TemplateAugmentor 설계 사상

PP-DocLayout은 중국어/영어 중심으로 학습되어 한국어 군수 서식의 일부 고정 필드를 놓치는 경우가 있습니다. 서식이 고정되어 있다는 전제 하에, form_type이 확정된 직후 해당 서식의 미리 정의된 bbox를 LayoutResult에 병합하여 누락을 보완합니다.

```
PP-DocLayout 탐지 결과만으로는:
  - 인쇄된 고정 필드 (NSN, 수량, 단가 등): 일부 누락 가능
  - 수기로 추가된 동적 요소 (인장, 서명):  PP-DocLayout이 잘 탐지

TemplateAugmentor 병합 후:
  - 인쇄된 고정 필드: 템플릿 bbox 보장 → 누락 0
  - 수기로 추가된 동적 요소: PP-DocLayout 탐지 결과 유지
  - 충돌 시 (IoU > 0.5):  PP-DocLayout 결과 우선 (실제 bbox가 더 정확)

PP-DocLayout Fine-tuning과 상호 보완:
  - Fine-tuning으로 검출률을 높이는 것과 독립적으로 동작
  - Fine-tuning 전에도 템플릿으로 최소한의 추출 완전성 보장
  - Fine-tuning 후에는 중복 탐지 케이스가 늘어나지만 IoU 필터로 자동 처리
```

### 두 경로의 설계 사상

| 항목 | military 경로 | other 경로 (Skill Registry) |
|------|-------------|---------------------------|
| 핵심 사상 | 서식을 먼저 알고, 아는 서식에 맞춰 추출 | 문서 구조를 먼저 이해하고, 구조에 따라 추출 |
| form_type 활용 | InstructionRouter 전체 결정 + TemplateAugmentor bbox 병합 | S7 StructuredAggregator에서만 Schema 선택에 사용 |
| 표 처리 | StructuredExtractor 단일 호출 | TableExtractor 2패스 (구조→셀 내용) |
| 인장 처리 | StructuredExtractor 안에서 처리 | SealPreprocessor(극좌표 변환) + SealReader |
| 서명 처리 | 미지원 | SignatureDetector (이진 분류) |
| guided JSON | 서식별 Schema 강제 | official_document.json Schema |

### 핵심 메커니즘

- **Layout Fusion**: V3(구조) + plus-L(텍스트) 결합, DPI 기반 분기
- **Crop-then-Infer**: LayoutPostProcessor 정제 → 영역별 크롭(48px 배수) → Gemma4
- **PdfAdapter**: PDF 입력은 fitz로 페이지별 RGB 렌더링(기본 300dpi) → 페이지마다 단일 이미지로 P1~P6 통과 → `PdfDocumentResult` 집계
- **FormClassifier 선행**: 저해상도 분류 → form_type + form_identifier → 경로 분기
- **TemplateAugmentor (v4)**: 템플릿 우선 병합 + field_key 부여. PP region 포함도 ≥0.7 면 흡수, ≥0.9 역포함은 컨테이너로 보존, `seal/signature/figure/table` 보호. 단일 매칭이면 PP bbox 재사용, 다중 매칭이면 템플릿 bbox로 통합. **fixed_text 섹션 지원**: 인쇄 고정 텍스트는 VLM 추론 대상에서 제외하고 Assembler가 assembled_json에 직접 삽입 (IoU>0.5 PP region 제거, field_key 있는 region 보호).
- **Sub-schema 분해 + Assembler**: `x-assembly-rules` 정의된 스키마는 region별로 sub-schema(예: `result_item_3` → `item_number: const=3`)만 VLM에 전달 → 결과를 region_id로 역참조해 full schema dict로 조립(`VLMResult.assembled_json`)
- **CROP_PADDING_MAX_PX**: 비율 패딩에 절대 상한(예: table=30px) → 큰 bbox에서 인접 영역 오염 방지
- **military InstructionRouter**: region_type + form_type → 도메인 맥락 + 1-shot. field_key가 있으면 full-shot 대신 sub-schema 전용 짧은 instruction 사용
- **OCR-augmented**: 저신뢰 영역에 PaddleOCR 선행 → 텍스트 힌트로 VLM 정확도 보강
- **ResolutionRouter 배치 순서**: 140→560→1120 토큰 그룹 순차 배치 (vLLM 패딩 최소화)
- **SealPreprocessor**: HSV 적색 분리 → 허프 원 탐지 → 극좌표 직선화 (학습 불필요)
- **TableExtractor 2패스**: 패스1(구조 추출) → 패스2(셀별 Skill 라우팅)
- **SignatureDetector**: 서명 존재 여부 이진 분류 (OCR 아님)
- **저신뢰 재시도**: logprobs 임계값 미달 → pixel_budget 상향 후 재호출 (field_key 보존)
- **그룹 내 병렬 처리** (2026-04-22): StructuredExtractor와 SkillRegistry가 동일 pixel_budget 그룹 내 region을 `asyncio.gather`로 병렬 vLLM 호출. 그룹 간 순서(140→560→1120)는 prefix caching 유지를 위해 고정, 동시 요청 상한 16. 동기 `call()`/`run()`은 `asyncio.to_thread`로 감싸 재사용 → 단위 테스트 호환.
- **Layer 1/2 검증 엔진** (2026-04-22): P4 Validator를 범용 엔진 + YAML 규칙 파일로 분리. Layer 1(`FieldPatternValidator` + `configs/validation_rules/common.yaml`)은 form_type 무관하게 data_type/field_key 패턴 검증 + 날짜 등 자동 정규화. Layer 2(`CrossFieldValidator` + `{form_type}.yaml`)는 REGEX/NOT_EMPTY/DATE_ORDER/ARITHMETIC/NAME_MATCH/ADDR_CONTAINS/ARRAY_*를 YAML로 선언. 새 서식은 YAML만 추가. 기존 BID-001~004/CHK-001~004/ARITH-001 등 하드코딩 전면 이관 + bid_date_001/bid_person_001/bid_addr_001 등 신규 규칙 확장.
- **일관성 재추론 루프** (2026-04-23): Layer 2 실패 또는 저신뢰 필드(<0.60) 발견 시 P4가 `ConsistencyReasoningLoop`를 1회 실행해 VLM 재추론. (1) 일관성 제약 주입 — NAME_MATCH/DATE_ORDER/ADDR_CONTAINS/ARITHMETIC 위반 필드에 제약 문장 동봉, (2) 컨텍스트 주입 — 저신뢰 필드에 고신뢰(≥0.70) 관련 필드 값을 힌트로 전달. 신뢰도 개선된 경우에만 교체, 미개선이면 원본 유지. 한 문서당 최대 5개 필드 / `was_retried` 이미 True면 스킵. 재추론 후 Assembler 재조립 + Layer 1/2 재검증.
- **Layer 3 도메인 사전 교정** (2026-04-23): `DomainDictCorrector` + `configs/domain_dict/*.yaml` 폐쇄집합 정규화/교정. (1) business_location 별칭 정규화 ("서울시"→"서울특별시"), (2) address의 시/도로 business_location 덮어쓰기, (3) industry_class_number Levenshtein 최근접, (4) issuing_office Levenshtein 최근접. Layer 2 직후 + (재추론 발생 시) 재조립 직후 2단계 적용으로 재조립이 교정 결과를 덮어쓰지 않도록 보장. 교정 로그는 warnings에 누적.
- **SPOF 대비**: 수준A(헬스체크) + 수준B(fallback) + 수준C(검토 큐)

---

## 로드맵

### Phase 1 잔여 — 남은 핵심 작업

- [x] **S2 PrintedTextReader / S3 HandwritingReader 정식 구현** — 공통 도메인 서비스로 격상. StructuredExtractor(military)와 SkillRegistry(other)가 Orchestrator DI로 단일 인스턴스 공유. S3는 신뢰도<0.70 시 프롬프트 변형 재시도. 한국어 수기 인식률 실측(10~20장)은 후속.
- [x] **S7 StructuredAggregator 구현** — `src/vlm/skills/aggregator.py`. S2~S6 결과를 `[HEADER]/[TEXT]/[SEAL]/[SIGNATURE]/[TABLE]` 블록으로 직렬화해 VLM에 컨텍스트로 주입하고, `official_document.json` 스키마로 guided_json 1회 호출(pixel_budget=140). 결과는 `VLMResult.assembled_json` + region 단위 `FieldValue`로 반환. 호출 실패 시 region 평탄화로 폴백.
- [ ] **OCR-augmented 힌트** — `ocr_hint_provider.py`는 구현됐으나 PaddleOCR 가중치의 오프라인 배치(`~/.paddlex/official_models/`)와 `Dockerfile.pipeline` COPY 반영 필요.
- [ ] **1-shot 예시 자산화** — `configs/instruction_examples/*.yaml` 서식별 예시 작성.
- [ ] **검출률 측정** — `scripts/evaluate_layout_detection.py --n 50` (PP-DocLayoutV3 기준치 확보).
- [x] **T8/T9/T10 통합 테스트 확장** — `tests/test_T8_seal_integration.py` + `tests/test_T9_T10_skills_integration.py` 자동화 스위트 추가 (40 pass 확인).
- [x] **vLLM 변동성 측정** — `scripts/measure_vllm_variance.py` 구축, N=3 반복 결과는 `data/variance_reports/`에 저장. 결정론 점수 20% 관측 — R3에 연계.
- [ ] **P4 신뢰도 재산출 (assembled_json 대응)** — field_key blob 단일 FieldValue로 저장되면서 토큰 단위 logprobs 평균이 의미 없어짐. assembled_json 트리 단위로 sub-field 신뢰도 분해·집계 필요. (현재 전비품 확인서에서 overall_confidence=0.13 관찰)
  - 하위 과제: `vlm_client.extract_field_logprobs` 위치 추적 미구현으로 필드별 신뢰도가 유사 값으로 수렴하는 문제 포함 (variance 측정 20% 결정론 점수의 한 원인).
- [ ] **도메인 사전 기반 필드 교정** — 인식 모호성이 잦은 폐쇄집합 필드(계급·부대코드 등)에 대한 이중 방어.
  - **1차: 스키마 enum 강제** — `equipment_checklist.json` `writer.rank` 등에 한국군 직급 enum 추가 → xgrammar가 디코딩 단계에서 차단.
  - **2차: 후처리 정규화 사전** — `rank_normalizer.py` 1차 구현 완료 (Levenshtein 최근접). 검토 큐 플래그 연동은 P4 통합 예정.
  - **확장 후보**: `nsn/K-NSN`은 `pattern` (guidance 백엔드 필요), `unit_code`는 부대 실재 DB 조회로 보강.
  - **지식그래프는 보류** — 현 단계 단일 필드 오인식 수정엔 ROI 나쁨. Phase 3 검토 큐 대시보드와 함께 재검토.

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
| — | **P2.5-A.5** TemplateAugmentor | 신규 — PP-DocLayout 누락 보완 (military 전용) |
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
| — | 공용 인터페이스 | `src/interfaces/` | ✅ Enum+dataclass (`ValidatedResult.assembled_json`, `PdfDocumentResult.page_results`, `FieldValue.region_id/was_retried` 포함) |
| — | 오케스트레이터 | `src/pipeline/orchestrator.py` | ✅ military/other 분기 완료 (`_process_other_document`) + `_aggregate_pdf_status` OTHER_DOCUMENT 처리 |
| P1 | 화질 보정 + SR | `src/preprocess/preprocessor.py` | ✅ |
| P2 | 레이아웃 탐지 (Fusion) | `src/preprocess/layout_analyzer.py` | ✅ |
| P2.5-A | LayoutPostProcessor | `src/preprocess/layout_postprocessor.py` | ✅ |
| — | SealPreprocessor | `src/preprocess/seal_preprocessor.py` | ✅ 극좌표 언래핑 + 허프 실패 폴백 |
| — | PdfAdapter | `src/input/pdf_adapter.py` | ✅ fitz 기반 PDF→PageImage, 멀티페이지 |
| P3-A | FormClassifier | `src/vlm/form_classifier.py` | ✅ military/other 분기 + form_identifier (guided_json) |
| P2.5-A.5 | TemplateAugmentor | `src/vlm/template_augmentor.py` | ✅ v4: 포함도 기반 다중 PP 흡수 + field_key 부여 + fixed_text 처리 (반환값 `(layout, fixed_values)`) |
| P2.5-B | InstructionRouter | `src/vlm/instruction_router.py` | ✅ 1-shot + OCR 힌트 + sub-schema 분해(`_extract_sub_schema`) |
| P2.5-C | ResolutionRouter | `src/vlm/resolution_router.py` | ✅ 48px 정렬 + DISPATCH_ORDER + 패딩 절대 상한 |
| P3-B | StructuredExtractor | `src/vlm/structured_extractor.py` | ✅ 저신뢰 재시도 + field_key blob 보존 + Assembler 호출 + **asyncio.gather 병렬(MAX_CONCURRENT_REQUESTS=16)** |
| — | Assembler | `src/vlm/assembler.py` | ✅ x-assembly-rules 기반 region→full dict 조립, 누락 자동 보완 |
| — | Budget Config | `src/vlm/budget_config.py` | ✅ PIXEL_BUDGETS/FALLBACK_PIXEL_BUDGET/DISPATCH_ORDER 중앙화 |
| — | OCR 힌트 제공자 | `src/vlm/ocr_hint_provider.py` | 🟡 구현 완료, 폐쇄망 가중치 배치 필요 |
| — | Skill Registry | `src/vlm/skill_registry.py` | ✅ DISPATCH_ORDER [140,560,1120] + SkillDispatchStats |
| — | Skill JSON 파서 | `src/vlm/skills/_parsing.py` | ✅ _loads_relaxed 공용화 (seal/signature/table에서 사용) |
| — | S2 PrintedTextReader | `src/vlm/skills/printed_text_reader.py` | ✅ 공통 도메인 서비스 — other + military(unknown) text/header/footer |
| — | S3 HandwritingReader | `src/vlm/skills/handwriting_reader.py` | ✅ 공통 도메인 서비스 — 모든 form_type의 handwritten_field. `task.json_schema` 있으면 guided_json으로 구조화 JSON 반환(`_loads_relaxed_any`로 코드펜스 정제), 없으면 순수 텍스트. 신뢰도<0.70 시 프롬프트 변형 재시도 |
| — | S4 SealReader | `src/vlm/skills/seal_reader.py` | ✅ |
| — | S5 TableExtractor | `src/vlm/skills/table_extractor.py` | ✅ pass1/pass2 |
| — | S6 SignatureDetector | `src/vlm/skills/signature_detector.py` | ✅ |
| — | S7 StructuredAggregator | `src/vlm/skills/aggregator.py` | ✅ S2~S6 결과 컨텍스트 주입 → official_document.json guided_json 집계 (VLM 실패 시 region 평탄화 폴백) |
| — | VLM 공용 클라이언트 | `src/vlm/vlm_client.py` | ✅ |
| P4 | 룰 검증 + 신뢰도 보정 | `src/postprocess/validator.py` | ✅ 경로별 임계값 + region_id/was_retried 보존 + **Layer 1/2 YAML 엔진 호출** |
| — | Layer 1 공통 패턴 | `src/postprocess/field_pattern_validator.py` + `configs/validation_rules/common.yaml` | ✅ data_type/field_key 패턴 + 자동 정규화 |
| — | Layer 2 교차 검증 | `src/postprocess/cross_field_validator.py` + `configs/validation_rules/{form_type}.yaml` | ✅ REGEX/NOT_EMPTY/DATE_ORDER/ARITHMETIC/NAME_MATCH/ADDR_CONTAINS/ARRAY_* 9가지 규칙 타입 |
| — | 일관성 재추론 | `src/postprocess/consistency_reasoning_loop.py` | ✅ Layer 2 실패/저신뢰 필드에 대해 제약·컨텍스트 주입 재추론 1회, 개선 시만 교체 |
| — | Layer 3 도메인 사전 | `src/postprocess/domain_dict_corrector.py` + `configs/domain_dict/*.yaml` | ✅ 행정구역 별칭 정규화 + Levenshtein 최근접 교정 (business_location/industry_class_number/issuing_office) |
| — | 최종 결과 저장 | `scripts/run_pipeline_with_outputs.py` `save_final_output` + `save_trial_readme` | ✅ 문서 폴더 최상위에 `final_output.json` (최종 OCR 결과 단일 파일), trial 최상위에 `README.md` (디렉터리 구조 + 문서별 링크) |
| — | 계급 정규화 | `src/postprocess/rank_normalizer.py` | ✅ 한국군 계급 Levenshtein 최근접 매칭 |
| — | bbox 유틸 | `src/preprocess/bbox_utils.py` | ✅ compute_iou 공용화 (layout_postprocessor/template_augmentor 사용) |
| P5 | 직렬화 | `src/postprocess/serializer.py` | ✅ assembled_json 있는 서식에 `document_title` 정적 주입 + fields[] 중복 제거 (조건부) |
| — | FormType 라벨 매핑 | `src/domain/form_type_labels.py` | ✅ FormType → 한국어 공식 명칭, P5에서 document_title 주입 |
| P6 | DB 적재 | `src/postprocess/db_loader.py` | ✅ per-URL 캐시 + `assembled_json` Text 컬럼 |
| — | 수동 검토 큐 | `src/postprocess/review_queue.py` | ✅ per-URL 캐시 + `RQ-YYYYMMDD-HHMMSS-{doc_id}-{uuid[:6]}` 형식 |
| — | Fallback 서비스 | `src/fallback/ocr_fallback_service.py`, `fallback_http_client.py` | ✅ Docker 분리 모드 지원 |
| — | VLM 헬스 모니터 | `src/pipeline/health_monitor.py` | ✅ `record_failure(reason)` public |
| — | Layout 추론 서비스 | `src/preprocess/layout_server.py` | ✅ |
| — | 스키마 레지스트리 | `src/domain/schema_registry.py` | ✅ v1/ 스캔 + `get()` 별칭 (`other→official_document`) |
| — | JSON Schemas v1 | `src/domain/schemas/v1/*.json` | ✅ 10종 (7 military 포함 `equipment_checklist`/`bid_application` + `_fallback`/`_general`/`official_document`). `x-assembly-rules`/`x-checklist-item-schema` 지원 |
| — | Docker 구성 | `docker-compose.yml` + `docker/Dockerfile.*` | ✅ vLLM 0.19.0 호환 (`--structured-outputs-config`) |

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
│   ├── FRONTEND.md
│   ├── equipment_checklist_design.md    ← 신규 (x-assembly-rules 설계)
│   └── pdf_adapter_design.md            ← 신규 (PdfAdapter + PdfDocumentResult)
├── src/
│   ├── interfaces/
│   ├── input/
│   │   └── pdf_adapter.py               ← fitz 기반 PDF → PageImage
│   ├── pipeline/
│   ├── preprocess/
│   │   ├── seal_preprocessor.py
│   │   └── bbox_utils.py                ← 신규 (compute_iou 공용)
│   ├── vlm/
│   │   ├── template_augmentor.py        ← PP-DocLayout 누락 보완
│   │   ├── assembler.py                 ← x-assembly-rules 조립기
│   │   ├── budget_config.py             ← 신규 (PIXEL_BUDGETS/DISPATCH_ORDER)
│   │   ├── skill_registry.py
│   │   ├── skills/
│   │   │   ├── _parsing.py              ← 신규 (_loads_relaxed 공용)
│   │   │   ├── seal_reader.py          (S4 ✅)
│   │   │   ├── table_extractor.py      (S5 ✅ 2패스)
│   │   │   ├── signature_detector.py   (S6 ✅)
│   │   │   ├── printed_text_reader.py  (S2 ✅ 공통 도메인 서비스)
│   │   │   ├── handwriting_reader.py   (S3 ✅ 공통 도메인 서비스 + 재시도)
│   │   │   └── aggregator.py           (S7 🔴 미구현)
│   │   ├── ocr_hint_provider.py
│   │   ├── form_classifier.py
│   │   ├── instruction_router.py
│   │   ├── resolution_router.py
│   │   ├── structured_extractor.py
│   │   └── vlm_client.py
│   ├── postprocess/
│   │   └── rank_normalizer.py           ← 신규 (한국군 계급 정규화)
│   ├── fallback/
│   │   ├── ocr_fallback_service.py
│   │   └── fallback_http_client.py      ← Docker 분리 모드용
│   └── domain/
│       ├── schema_registry.py
│       ├── form_type_labels.py          ← 신규 (FormType → 한국어 명칭)
│       └── schemas/v1/
│           ├── supply_request.json
│           ├── maintenance_record.json
│           ├── inventory_sheet.json
│           ├── handover_doc.json
│           ├── inspection_report.json
│           ├── equipment_checklist.json (x-assembly-rules)
│           ├── bid_application.json    (x-assembly-rules, 별지 제13호)
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
│   ├── raw/                          (입력 문서)
│   ├── pipeline_outputs/             (각 단계별 산출물 타임스탬프 폴더)
│   └── variance_reports/             (scripts/measure_vllm_variance.py 출력)
├── docker/
├── docker-compose.yml
├── scripts/
│   ├── run_pipeline_with_outputs.py  (단계별 산출 저장)
│   ├── download_paddle_models.py     (OCR 힌트 가중치 오프라인 배치)
│   └── measure_vllm_variance.py      (R3 변동성 측정)
├── training/
├── configs/
│   ├── instruction_examples/
│   └── form_templates/               ← TemplateAugmentor용 서식별 bbox 정의
│       ├── supply_request.yaml
│       ├── maintenance_record.yaml
│       ├── inventory_sheet.yaml
│       ├── handover_doc.yaml
│       ├── inspection_report.yaml
│       ├── equipment_checklist.yaml
│       └── bid_application.yaml       ← 신규 (별지 제13호)
└── tests/
    ├── test_integration_pipeline.py  (기본 E2E)
    ├── test_T2_T3_fusion.py          (Layout Fusion)
    ├── test_T5_fallback_routing.py   (Fallback 라우팅)
    ├── test_T8_seal_integration.py   (인장)
    └── test_T9_T10_skills_integration.py (Skill 파이프라인)
```

---

## 검증 필요한 리스크

| # | 이슈 | 보완 계획 |
|---|------|---------|
| R1 | S3 HandwritingReader 한국어 수기 인식률 실측치 부재 | 국회 의안 PDF 등 합성 샘플 10~20장으로 0-shot 인식률 측정 후 임계값 보정 |
| R2 | SealPreprocessor 허프 원 탐지 실패율 미측정 | T8 테스트에서 성공/실패 케이스 집계, 실패율 30%+ 이면 휴리스틱 격하 |
| R3 | vLLM 출력 변동성 기준치 미수립 | 동일 문서 N=3 반복으로 기준 확보 (이전 전역지원서_2 +99% 변동 건) |
| R4 | pass2 셀 태스크 0건 관찰 | S5 pass1 출력 table_type/cells 품질 확인 (2026-04-14 통합 테스트 기준) |
| R5 | TemplateAugmentor 서식 버전 불일치 | 템플릿 yaml에 서식 버전·발효일 명시, 서식 개정 시 즉시 업데이트 프로세스 수립 |
| R6 | TemplateAugmentor 스캔 기울기로 인한 bbox 오프셋 | P1 Deskew 이후 좌표계 기준 확인, DPI·기울기 보정 후 좌표 적용 |

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

## 코드 리뷰

코드 리뷰는 `backend-code-reviewer` 서브에이전트가 담당합니다 (`.claude/agents/backend-code-reviewer.md`).

### 자동 호출 트리거

Claude Code가 아래 상황을 감지하면 자동으로 에이전트를 호출합니다.
별도 명령 불필요.

| 트리거 | 예시 |
|--------|------|
| 새 컴포넌트 구현 완료 | "structured_extractor.py 구현했어" |
| 기존 컴포넌트 수정 완료 | "assembler.py 수정 완료" |
| 신규 FormType 추가 완료 | "bid_application 추가 완료" |
| 리뷰 요청 키워드 | "리뷰해줘", "코드 검토", "review" |

### 리뷰 범위

에이전트는 아래 항목을 순서대로 점검합니다.

1. **아키텍처** — 파이프라인 책임 경계, DDD 원칙, 의존성 방향
2. **인터페이스** — dataclass 사용, 타입 힌트, 반환 타입
3. **VLM 호출** — temperature=0.0, guided_json, budget_config 사용
4. **TemplateAugmentor** — field_key, fixed_text, 보호 region
5. **Assembler** — blob 언래핑, x-assembly-rules, 유령 필드 방지
6. **P4 Validator** — 하드코딩 금지, 신뢰도 산출, fixed_content 제외
7. **재추론 루프** — 안전장치, MAX_REREASON_FIELDS, 미개선 시 원본 유지
8. **도메인 사전** — 범용성, YAML 분리, Levenshtein 임계값 상수화
9. **폐쇄망 보안** — 외부 API 호출 금지, HF_HUB_OFFLINE 준수
10. **코딩 컨벤션** — docstring, 타입 힌트, 매직 넘버 금지

신규 FormType 추가 시 8개 파일 체크리스트(enum → label → classifier → schema → template → examples → validation_rules → tests)도 함께 확인합니다.

### 보고 형식

| 심각도 | 의미 |
|--------|------|
| 🔴 CRITICAL | 즉시 수정 — 파이프라인 장애·데이터 오염·보안 위반 가능 |
| 🟡 WARNING | 수정 권장 — 설계 원칙 위반·성능 저하·유지보수성 문제 |
| 🟢 SUGGESTION | 개선 제안 — 더 나은 구현·일관성·코드 품질 향상 |
| ✅ PASS | 문제 없음 |

**자동 수정 가능** 이슈는 에이전트가 직접 파일을 수정합니다.
**수동 확인 필요** 이슈(bbox 좌표, 비즈니스 로직 판단 등)는 별도 표시됩니다.