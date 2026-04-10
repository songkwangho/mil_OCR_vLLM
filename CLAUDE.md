# 군수 OCR 시스템 v2 — 마스터 컨텍스트

> 이 파일은 프로젝트 전체 맥락의 **진입점**입니다.
> 상세 내용은 `docs/` 하위 문서를 참조하십시오.

---

## 핵심 원칙

- **폐쇄망 전용** — 외부 API 호출·모델 다운로드 절대 금지. 모든 가중치는 `models/`에 사전 배치
- **Crop-then-Infer** — PP-DocLayout bbox 크롭 이미지를 영역별로 Gemma4에 전달 (전체 페이지 입력 지양)
- **FormClassifier 선행** — 서식 분류 후 InstructionRouter가 도메인 맥락 기반 instruction 생성
- **Other 문서 분기** — 군수 서식이 아닌 문서는 `other`로 분류해 별도 경로로 처리
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
        └─ other:    범용 OCR 경로    │
        ↓                             │
    P2.5-B: InstructionRouter          │
        region_type + form_type        │
        → 도메인 맥락 포함 instruction  │
        → JSON Schema 선택             │
        → 1-shot 예시 포함             │
        → OCR 힌트 삽입 (저신뢰 영역)  │
        ↓                             │
    P2.5-C: ResolutionRouter           │
        영역 타입별 pixel_budget 할당   │
        table/handwritten: 1120토큰    │
        text/seal: 560토큰             │
        header/footer: 140토큰         │
        크롭 크기 48px 배수 정렬        │
        ↓                             │
    P3-B: StructuredExtractor          │
        bbox 크롭 + instruction        │
        → pixel_budget 기준 배치 그룹화│
        → Gemma4 병렬 배치 호출        │
          (temperature=0.0, 결정론적)  │
        → CoT analysis 필드 포함       │
        → logprobs 신뢰도 산출         │
        → 저신뢰 필드 자동 재시도       │
        ↓                            └───────────────┐
[Stage 3 — 후처리]                                   │
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult       │
        처리 경로별 임계값 적용 (VLM / Fallback)       │
        ├─ military 문서 + 통과 → P5 → P6 DB 적재    │
        ├─ other 문서 → P5 범용 직렬화 (검토 큐 없음) │
        └─ 실패 → 검토 큐 → 담당자 UI → P6 재적재    ↓
```

### 핵심 메커니즘

- **Layout Fusion**: V3(구조) + plus-L(텍스트) 결합, DPI 기반 분기 (`LAYOUT_FUSION_MODE=true`)
- **Crop-then-Infer**: LayoutPostProcessor 정제 → 영역별 크롭(48px 배수) → Gemma4
- **FormClassifier 선행**: 저해상도 분류 → form_type → InstructionRouter 피드백
- **Other 문서 분기**: `other` 분류 시 군수 룰 검증 없는 범용 OCR 경로
- **InstructionRouter**: region_type + form_type → 상세 instruction + 1-shot 예시
  - 영어 지시문 + 한국어 필드명 하이브리드 (VLM 지시 따르기 능력 + 출력 정확도 동시 확보)
- **OCR-augmented**: 저신뢰 영역에 PaddleOCR 선행 → 텍스트 힌트로 VLM 정확도 보강
- **ResolutionRouter**: 영역 타입별 pixel_budget 차등 할당 + 크롭 48px 배수 정렬
- **배치 병렬 처리**: pixel_budget 기준 그룹화 → vLLM 동시 전송
- **Guided Decoding + CoT**: JSON Schema + analysis 필드로 수기 인식 정확도 향상
- **저신뢰 재시도**: logprobs 임계값 미달 필드 → pixel_budget 상향 후 재호출
- **SPOF 대비**: 수준A(헬스체크) + 수준B(fallback) + 수준C(검토 큐)

---

## 현재까지 설계·구현 사항의 문제점

### [P1] OCR-augmented 전략 완전 누락

군수 전문 용어(부대코드, 장비 식별번호)와 의미 없는 패턴 코드(NSN, K-NSN)는 VLM이 문맥 추론으로 복원하기 어렵습니다. KLOCR 논문(2025.10)에서 OCR 텍스트를 이미지와 함께 VLM에 제공하는 OCR-augmented 접근이 Key Information Extraction에서 가장 큰 개선을 보였는데, 이 전략이 어디에도 설계되어 있지 않습니다.

### [P2] 저신뢰 필드 재시도 로직 미설계

logprobs 임계값 미달 필드가 P4에서 검토 큐로 바로 넘어가는 구조입니다. pixel_budget을 한 단계 올려서 재시도하면 인식률을 높일 수 있는데, 이 전략이 오케스트레이터와 StructuredExtractor에 없습니다.

### [P3] 크롭 이미지 48px 배수 정렬 미반영

Gemma4의 SigLIP 인코더는 이미지를 16×16 패치로 분할 후 3×3 블록 단위 average pooling합니다. 크롭 이미지 크기가 48px 배수가 아니면 패딩 픽셀에 토큰이 낭비됩니다. ResolutionRouter에 이 정렬 로직이 없습니다.

### [P4] LoRA adapter 경로 충돌

AI_TRAINING.md §1에서 P3-A FormClassifier와 P3-B StructuredExtractor의 Fine-tuned 출력 경로가 `models/gemma4/gemma4-mil-finetuned/`로 동일합니다. 두 adapter가 rank도 다르고(8 vs 16) 학습 태스크도 달라 동일 경로에 저장 시 덮어쓰기 충돌이 발생합니다.

### [P5] KV Cache FP8 vs 모델 가중치 FP8 구분 미명시

AI_INFERENCE.md의 vLLM 설정에 `--kv-cache-dtype fp8`이 있지만, 이것이 KV Cache만 FP8로 줄이는 것이고 모델 가중치는 BF16을 유지한다는 설명이 없습니다. 양자화 테이블의 FP8 Dynamic과 개념이 혼동될 수 있습니다.

### [P6] Speculative Decoding 선택 근거 미명시

n-gram 방식이 채택된 이유(MTP 헤드 미포함, 추가 VRAM 불필요, guided_json 호환)가 간략히만 언급되어 있습니다. Draft 모델 방식(gemma-4-E2B-it)과의 트레이드오프 분석이 없어 Phase 3에서 혼란이 생길 수 있습니다.

### [P7] NSN 패턴 인식에 프롬프트 힌트 전략 미반영

xgrammar가 `pattern` 제약을 미지원하여 guided_regex나 guidance 백엔드 전환으로 대응한다는 내용은 있지만, InstructionRouter에서 프롬프트 자체에 형식 힌트를 삽입하는 방법이 설계에 없습니다.

### [P8] AWQ Marlin 커널 활성화 미명시

AWQ 4-bit 적용 시 Marlin 커널 없이는 기본 속도(~68 tok/s)로 동작하고, 활성화 시 ~741 tok/s로 10배 차이가 납니다. Phase 3 양자화 실험 계획에 이 내용이 없습니다.

### [P9] 오케스트레이터 상태 표기 불일치

구현 현황에서 오케스트레이터가 `✅ 완료`로 표기되어 있지만 통합 테스트 1-E가 전부 미완입니다.

---

## 보완 계획

### 즉시 적용 (Phase 1 잔여)

**OCR-augmented 전략 추가**
- `src/vlm/ocr_hint_provider.py` 신규 — PaddleOCR(한국어) 선행 실행
- logprobs < 0.80 필드에 대해 OCR 결과를 프롬프트 힌트로 삽입
- `docs/PIPELINE.md §3-3 InstructionRouter`에 OCR 힌트 삽입 명세 추가

**저신뢰 재시도 로직 추가**
- `src/vlm/structured_extractor.py`에 필드별 재시도 조건 추가
  ```python
  RETRY_THRESHOLD = 0.60  # 이하면 pixel_budget 한 단계 상향 후 재시도
  MAX_RETRIES = 1         # 과도한 지연 방지
  ```

**ResolutionRouter 48px 배수 정렬**
- bbox 크롭 후 `(width // 48) * 48`, `(height // 48) * 48`로 리사이즈
- `docs/PIPELINE.md §3-4`에 근거 명시

**NSN 프롬프트 힌트 전략 추가**
- InstructionRouter에서 `supply_request` + `text` 영역에 형식 힌트 삽입
  ```
  user_instruction += "\nNSN 코드는 NNNN-NN-NNN-NNNN 형식 13자리입니다. 예: 1005-01-432-1234"
  ```

### Phase 2 착수 전

**LoRA adapter 경로 분리**
```
models/gemma4/adapters/
├── form_classifier_v1/      ← P3-A LoRA (rank=8)
└── structured_extractor_v1/ ← P3-B LoRA (rank=16)
```

**KV Cache FP8 vs 모델 가중치 FP8 구분 명시**
- `docs/AI_INFERENCE.md §3`에 개념 분리 설명 추가

**Speculative Decoding 트레이드오프 명세**
- n-gram vs Draft 모델(gemma-4-E2B-it) vs Suffix Decoding 비교표 추가
- `docs/AI_INFERENCE.md §3-2`에 반영

---

## v1 → v2 컴포넌트 매핑

| v1 | v2 | 비고 |
|----|-----|------|
| T1 화질 보정 | **P1** 화질 보정 + SR | v1에서 이관 |
| T2 레이아웃 분석 | **P2** 레이아웃 탐지 (Fusion 지원) | 원시 탐지만 반환 |
| — | **P2.5-A** LayoutPostProcessor | 신규 — 정제 + remap |
| T3 서식 분류 | **P3-A** FormClassifier | military/other 분기 포함 |
| — | **P2.5-B** InstructionRouter | 신규 — 1-shot + CoT + OCR 힌트 |
| — | **P2.5-C** ResolutionRouter | 신규 — pixel_budget + 48px 정렬 |
| T4~T7 | **P3-B** StructuredExtractor | 배치 병렬 + 저신뢰 재시도 |
| T8 데이터 검증 | **P4** 룰 검증 | 경로별 임계값 분리 |
| T9 신뢰도 | **P3-B** (logprobs) | 기하평균, 한국어 임계값 보정 |
| T10 스키마 매핑 | **P3-B** (guided_json) | JSON Schema 구조화 |
| T11 직렬화 | **P5** 직렬화 | v1에서 이관 |
| T12 DB 적재 | **P6** DB 적재 | v1에서 이관 |

---

## 구현 현황

| P# | 컴포넌트 | 소스 파일 | 상태 |
|----|----------|----------|------|
| — | 공용 인터페이스 | `src/interfaces/` | ✅ 완료 (14 Enum + 17 dataclass) |
| — | 오케스트레이터 | `src/pipeline/orchestrator.py` | 🟡 구현 완료, 통합 테스트 미완 |
| P1 | 화질 보정 + SR | `src/preprocess/preprocessor.py` | ✅ 완료 |
| P2 | 레이아웃 탐지 (Fusion) | `src/preprocess/layout_analyzer.py` | ✅ 완료 |
| P2.5-A | LayoutPostProcessor | `src/preprocess/layout_postprocessor.py` | ✅ 완료 |
| P3-A | FormClassifier | `src/vlm/form_classifier.py` | 🟡 other 분기 추가 필요 |
| P2.5-B | InstructionRouter | `src/vlm/instruction_router.py` | 🟡 other 경로 + 1-shot + OCR 힌트 추가 필요 |
| P2.5-C | ResolutionRouter | `src/vlm/resolution_router.py` | 🟡 48px 배수 정렬 추가 필요 |
| P3-B | StructuredExtractor | `src/vlm/structured_extractor.py` | 🟡 재시도 로직 추가 필요 |
| — | OCR 힌트 제공자 | `src/vlm/ocr_hint_provider.py` | 🔴 신규 구현 필요 |
| — | VLM 공용 클라이언트 | `src/vlm/vlm_client.py` | ✅ 완료 |
| P4 | 룰 검증 + 신뢰도 보정 | `src/postprocess/validator.py` | 🟡 other 경로 건너뜀 추가 필요 |
| P5 | 직렬화 | `src/postprocess/serializer.py` | ✅ 완료 |
| P6 | DB 적재 | `src/postprocess/db_loader.py` | ✅ 완료 (raw_json 제외 확인 필요) |
| — | 수동 검토 큐 | `src/postprocess/review_queue.py` | ✅ 완료 |
| — | Fallback 서비스 | `src/fallback/ocr_fallback_service.py` | ✅ 완료 |
| — | VLM 헬스 모니터 | `src/pipeline/health_monitor.py` | ✅ 완료 |
| — | Layout 추론 서비스 | `src/preprocess/layout_server.py` | ✅ 완료 |
| — | 스키마 레지스트리 | `src/domain/schema_registry.py` | ✅ 완료 |
| — | Docker 구성 | `docker-compose.yml` + `docker/Dockerfile.*` | ✅ 완료 (5 Dockerfile) |

> **Legacy 유지**: `instruction_builder.py` (InstructionRouter 래퍼), `gemma4_engine.py` (하위 호환용)

---

## 디렉토리 구조

```
mil_OCR_v2/
├── CLAUDE.md
├── docs/
│   ├── PIPELINE.md      ← 파이프라인·컴포넌트 설계·인터페이스 타입
│   ├── DOMAIN.md        ← 군수 도메인 지식·스키마·Other 처리
│   ├── AI_INFERENCE.md  ← 모델·vLLM 서빙·최적화·Docker
│   ├── AI_TRAINING.md   ← Fine-tuning·합성 데이터·교정 파이프라인
│   ├── TESTING.md       ← 통합 테스트 절차·결과 구조
│   └── FRONTEND.md      ← 검토 큐 UI (미구현)
├── src/
│   ├── interfaces/      ← 공용 타입, Enum
│   ├── pipeline/        ← 오케스트레이터, 헬스 모니터, Fallback 정책
│   ├── preprocess/      ← P1, P2(Fusion), P2.5-A, Layout HTTP 서비스
│   ├── vlm/             ← P3-A, P2.5-B, P2.5-C, P3-B, vlm_client,
│   │                       ocr_hint_provider (신규)
│   ├── postprocess/     ← P4, P5, P6, 검토 큐
│   ├── fallback/        ← 수준 B 경량 fallback
│   └── domain/          ← 스키마, 도메인 사전, schema_registry
├── models/
│   ├── t1_sr/
│   ├── t2_layout/
│   ├── gemma4/
│   │   ├── gemma-4-26b-a4b-it/   ← base 모델
│   │   └── adapters/
│   │       ├── form_classifier_v1/      ← P3-A LoRA (Phase 2)
│   │       └── structured_extractor_v1/ ← P3-B LoRA (Phase 2)
│   └── fallback/
├── data/
├── docker/
├── docker-compose.yml
├── scripts/
├── training/
├── configs/
│   └── instruction_examples/    ← 서식별 1-shot 예시 YAML
└── tests/
```

---

## 개발 로드맵

> 시작 기준: 2026-04 (현재)

### Phase 1 잔여 (04월 말)

**vLLM 서버 최적화 설정 (코드 변경 없음, 즉시 적용)**
- [ ] `--kv-cache-dtype fp8` (KV Cache 50% 절감, 동시처리 2배)
- [ ] `--max-num-batched-tokens 16384` (Chunked Prefill)
- [ ] `--mm-cache-preprocessor` (멀티모달 prefix caching)
- [ ] `--speculative-config '{"method":"ngram","num_speculative_tokens":5}'`
- [ ] `--guided-decoding-backend xgrammar`
- [ ] `--max-num-seqs 64` (128→64, 멀티모달 배치 안정성)

**Other 문서 처리 경로 추가**
- [ ] `FormType` enum에 `other` 추가 + `is_military()` 헬퍼
- [ ] FormClassifier guided_json enum에 `other` 포함
- [ ] InstructionRouter `other` 전용 분기 (범용 OCR instruction + `_general.json`)
- [ ] P4 Validator `other` 경로 군수 룰 검증 건너뜀
- [ ] `src/domain/schemas/v1/_general.json` 신규 작성

**pixel_budget 상향 + 48px 배수 정렬**
- [ ] `handwritten_field`: 560 → **1120**
- [ ] `text`: 280 → **560**
- [ ] ResolutionRouter 크롭 후 48px 배수 리사이즈 적용

**OCR-augmented + 저신뢰 재시도**
- [ ] `src/vlm/ocr_hint_provider.py` 신규 구현 (PaddleOCR 한국어 래퍼)
- [ ] InstructionRouter에서 저신뢰 영역 OCR 힌트 삽입
- [ ] StructuredExtractor 저신뢰 필드 재시도 로직 (RETRY_THRESHOLD=0.60, MAX_RETRIES=1)

**1-shot 예시 + CoT 스키마**
- [ ] `configs/instruction_examples/*.yaml` 서식별 예시 작성
- [ ] InstructionRouter에 예시 로드 + user_instruction 끝에 삽입
- [ ] `src/domain/schemas/v1/*.json` 최상단에 `analysis` 필드 추가

**NSN 프롬프트 힌트**
- [ ] `supply_request` + `text` 영역 instruction에 NSN 형식 힌트 삽입
- [ ] K-NSN, 부대코드 형식도 동일 적용

**검출률 측정 + 통합 테스트**
- [ ] `python scripts/evaluate_layout_detection.py --n 50`
  - >70%: Fine-tuning 2,000장, Fusion OFF 검토
  - 50~70%: Fine-tuning 3,000장+, Fusion ON 권장
  - <50%: 목표 재협의
- [ ] 통합 테스트 T1(Fusion OFF) + T2(Fusion ON DPI≥150) + T3(Fusion ON DPI<150) + T4(Other 문서) + T5(Fallback)

### Phase 2 (06~07월)

**2-A. 검토 큐 UI MVP + 교정 데이터 축적 (병행)**
- [ ] 검토 큐 UI MVP (큐 목록 + 검토 + 필드 수정 + 승인/반려)
- [ ] 교정 데이터 JSONL export 파이프라인 활성화
- [ ] PP-DocLayout 어노테이션 + Fine-tuning (검출률 기반 목표)
- [ ] PP-DocLayoutV3 confidence threshold 튜닝

> Track A(Fine-tuning) 검출률 70%+ 달성 후에만 Track B 교정 데이터를 SFT에 투입

**2-B. Gemma4 VLM Fine-tuning** (유형당 100건+ 확보 후)
- [ ] P3-A SFT (LoRA rank=8, adapters/form_classifier_v1/)
- [ ] P3-B SFT 1단계 (LoRA rank=16, adapters/structured_extractor_v1/)
- [ ] 벤치마크 후 2단계 수기 강화(선택) / 3단계 DPO(운영 3개월 후) 판단

### Phase 3 (08~09월)

- [ ] RepetitionGuard 도입 (반복 토큰 비율 > 15% 관측 시)
- [ ] 양자화 실험
  - AWQ 4-bit 우선 (Marlin 커널 활성화 필수, 한국어 혼합 캘리브레이션)
  - FP8 Dynamic — vllm#39049 해소 확인 후
- [ ] 검토 큐 UI 고도화 (대시보드, bbox 오버레이, 담당자 관리)
- [ ] 비동기 메시지 기반 통신 전환 (Redis Streams)
- [ ] Schema Registry 운영 시나리오 검증 (서식 개정 → DB 마이그레이션)
- [ ] Docker 프로덕션 설정 + 보안 검토 + 배포

---

## 향후 계획 타당성 및 문제점

### [R1] Phase 1 잔여 작업량이 과부하

현재 Phase 1 잔여에 vLLM 최적화, Other 경로, pixel_budget 조정, OCR-augmented, 1-shot 예시, NSN 힌트, 검출률 측정, 통합 테스트까지 집중되어 있습니다. 04월 말 데드라인이 현실적이지 않습니다.

**보완**: 즉시 적용 가능한 것(vLLM 설정, pixel_budget)과 구현이 필요한 것(OCR-augmented, other 경로)을 분리합니다. vLLM 최적화 설정은 하루 안에 적용 가능하고, OCR-augmented는 1~2주 작업입니다.

### [R2] OCR-augmented의 추론 지연 증가

모든 영역에 PaddleOCR을 선행 실행하면 처리 시간이 증가합니다.

**보완**: 전체 영역이 아닌 특정 조건(logprobs < 0.80, NSN 패턴 필드)에서만 OCR 힌트를 삽입합니다. PaddleOCR은 영역당 ~50ms 수준이므로 선택적 적용으로 영향을 최소화합니다.

### [R3] 저신뢰 재시도의 처리 시간 증가

재시도가 발생할 때마다 추가 VLM 호출이 생깁니다.

**보완**: `MAX_RETRIES=1` + pixel_budget 한 단계만 올리는 제한적 재시도로 최악의 경우 1회 추가 호출로 제한합니다. 재시도 통계를 TESTING.md run_summary에 기록하여 실운영 후 기준을 조정합니다.

### [R4] Phase 2 Fine-tuning 데이터 수량 게이트 조건 검증 필요

"유형당 100건"이 SFT에 충분한지 실측 근거가 없습니다.

**보완**: Phase 2-A에서 소규모 파일럿(유형당 30~50건)으로 SFT를 먼저 시도하고 성능 변화를 측정한 후 목표 수량을 조정합니다.

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
| Gemma4 Fine-tuning (LoRA) | 16~24GB | GPU | PEFT |
