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
        전체 페이지 저해상도(140토큰)    │   → 서식 분류 + 텍스트 인식 + 표 구조
        → form_type 확정               │   → 결과는 항상 검토 큐 적재
        ┌─ military: 군수 서식 경로    │
        └─ other:    범용 OCR 경로    │
        ↓                             │
    P2.5-B: InstructionRouter          │
        region_type + form_type        │
        → 도메인 맥락 포함 instruction  │
        → JSON Schema 선택             │
        → 1-shot 예시 포함 (prefix 캐싱)│
        ↓                             │
    P2.5-C: ResolutionRouter           │
        영역 타입별 pixel_budget 할당   │
        table/handwritten: 1120토큰    │
        text/seal: 560토큰             │
        header/footer: 140토큰         │
        ↓                             │
    P3-B: StructuredExtractor          │
        bbox 크롭 + instruction        │
        → pixel_budget 기준 배치 그룹화│
        → Gemma4 병렬 배치 호출        │
        → CoT analysis 필드 포함       │
        → logprobs 신뢰도 산출         │
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
- **Crop-then-Infer**: LayoutPostProcessor 정제 → 영역별 크롭 → Gemma4
- **FormClassifier 선행**: 저해상도 분류 → form_type → InstructionRouter 피드백
- **Other 문서 분기**: `other` 분류 시 군수 룰 검증 없는 범용 OCR 경로
- **InstructionRouter**: region_type + form_type → 상세 instruction + 1-shot 예시
- **ResolutionRouter**: 영역 타입별 pixel_budget 차등 할당
- **배치 병렬 처리**: pixel_budget 기준 그룹화 → vLLM 동시 전송
- **Guided Decoding + CoT**: JSON Schema + analysis 필드로 수기 인식 정확도 향상
- **SPOF 대비**: 수준A(헬스체크) + 수준B(fallback) + 수준C(검토 큐)

---

## v1 → v2 컴포넌트 매핑

| v1 | v2 | 비고 |
|----|-----|------|
| T1 화질 보정 | **P1** 화질 보정 + SR | v1에서 이관 |
| T2 레이아웃 분석 | **P2** 레이아웃 탐지 (Fusion 지원) | 원시 탐지만 반환 |
| — | **P2.5-A** LayoutPostProcessor | 신규 — 정제 + remap |
| T3 서식 분류 | **P3-A** FormClassifier | military/other 분기 포함 |
| — | **P2.5-B** InstructionRouter | 신규 — 1-shot + CoT 포함 |
| — | **P2.5-C** ResolutionRouter | 신규 — pixel_budget 차등 |
| T4~T7 | **P3-B** StructuredExtractor | 배치 병렬 VLM 호출 |
| T8 데이터 검증 | **P4** 룰 검증 | 경로별 임계값 분리 |
| T9 신뢰도 | **P3-B** (logprobs) | 기하평균 기반 |
| T10 스키마 매핑 | **P3-B** (guided_json) | JSON Schema 구조화 |
| T11 직렬화 | **P5** 직렬화 | v1에서 이관 |
| T12 DB 적재 | **P6** DB 적재 | v1에서 이관 |

---

## 구현 현황

| P# | 컴포넌트 | 소스 파일 | 상태 |
|----|----------|----------|------|
| — | 공용 인터페이스 | `src/interfaces/` | ✅ 완료 (FormType에 OTHER 추가, PipelineStatus.OTHER_DOCUMENT 추가) |
| — | 오케스트레이터 | `src/pipeline/orchestrator.py` | ✅ 완료 (Phase 1-E 통합 테스트 통과, other/military 경로 분기) |
| — | Pipeline HTTP 서버 | `src/pipeline/server.py` | ✅ 완료 (FastAPI: /pipeline/run, /pipeline/upload) |
| P1 | 화질 보정 + SR | `src/preprocess/preprocessor.py` | ✅ 완료 |
| P2 | 레이아웃 탐지 (Fusion) | `src/preprocess/layout_analyzer.py` | ✅ 완료 |
| P2.5-A | LayoutPostProcessor | `src/preprocess/layout_postprocessor.py` | ✅ 완료 |
| P3-A | FormClassifier | `src/vlm/form_classifier.py` | ✅ 완료 (military/other 분기 포함, 7개 옵션 분류) |
| P2.5-B | InstructionRouter | `src/vlm/instruction_router.py` | ✅ 완료 (other 전용 경로 + 1-shot YAML 로드 + CoT 지시 자동 부착) |
| P2.5-C | ResolutionRouter | `src/vlm/resolution_router.py` | ✅ 완료 (text 280→560 상향 적용) |
| P3-B | StructuredExtractor | `src/vlm/structured_extractor.py` | ✅ 완료 (CoT analysis 필드 인식, trace 수집 옵션) |
| — | VLM 공용 클라이언트 | `src/vlm/vlm_client.py` | ✅ 완료 |
| P4 | 룰 검증 + 신뢰도 보정 | `src/postprocess/validator.py` | ✅ 완료 (other 경로 룰 스킵 + review_required=False) |
| P5 | 직렬화 | `src/postprocess/serializer.py` | ✅ 완료 |
| P6 | DB 적재 | `src/postprocess/db_loader.py` | ✅ 완료 (raw_json DB 제외) |
| — | 수동 검토 큐 | `src/postprocess/review_queue.py` | ✅ 완료 |
| — | Fallback 서비스 (HTTP) | `src/fallback/server.py` + `fallback_http_client.py` | ✅ 완료 (컨테이너 분리 필수) |
| — | Fallback in-process | `src/fallback/ocr_fallback_service.py` | ⚠️ 사용 금지 (v1/v2 enum 충돌) |
| — | VLM 헬스 모니터 | `src/pipeline/health_monitor.py` | ✅ 완료 |
| — | Layout 추론 서비스 | `src/preprocess/layout_server.py` | ✅ 완료 |
| — | 스키마 레지스트리 | `src/domain/schema_registry.py` | ✅ 완료 |
| — | JSON Schema (CoT) | `src/domain/schemas/*.json` | ✅ 완료 (6개 군수 + `_fallback` + `_general` 모두 analysis 최상단) |
| — | 1-shot 예시 | `configs/instruction_examples/*.yaml` | ✅ 완료 (5개 군수 서식) |
| — | Docker 구성 | `docker-compose.yml` + `docker/Dockerfile.*` | ✅ 완료 (5 Dockerfile, vLLM 최적화 옵션 반영 — 재기동 시 적용) |

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
│   ├── vlm/             ← P3-A, P2.5-B, P2.5-C, P3-B, vlm_client
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
└── tests/
```

---

## 개발 로드맵

> 시작 기준: 2026-04 (현재)

### Phase 1 잔여 (04월 말)

**vLLM 서버 최적화 설정 (docker-compose.yml 반영 완료, 재기동 필요)**
- [x] `--kv-cache-dtype fp8` (KV Cache 50% 절감, 동시처리 2배)
- [x] `--max-num-batched-tokens 16384` (Chunked Prefill)
- [x] `--mm-cache-preprocessor` (멀티모달 prefix caching)
- [x] `--speculative-config '{"method":"ngram","num_speculative_tokens":5}'`
- [x] `--guided-decoding-backend xgrammar`
- [x] `--max-num-seqs 128 → 64` (멀티모달 배치 안정성)
- [x] `--gpu-memory-utilization 0.90 → 0.92`
- [ ] **vLLM 컨테이너 재기동으로 반영** — `docker compose restart vllm-server` (옵션 파일은 수정되었으나 기존 컨테이너는 구 옵션으로 기동 중)

**Other 문서 처리 경로 (완료)**
- [x] `FormType` enum에 `other` 추가 (+ `is_military()` 헬퍼)
- [x] `PipelineStatus.OTHER_DOCUMENT` 추가
- [x] FormClassifier `classify_instruction` 7옵션 분류 (+ other 유형 상세 설명)
- [x] InstructionRouter `other` 전용 분기 (`_OTHER_REGION_INSTRUCTION`, `_general` 스키마)
- [x] 오케스트레이터 스키마 분기 (other→`_general`, unknown→`_fallback`, 군수→form_type)
- [x] 오케스트레이터 status 분기 (`form_type == OTHER` → `OTHER_DOCUMENT`)
- [x] P4 Validator `is_other` 가드 (군수 룰 스킵 + `review_required=False`)
- [x] `src/domain/schemas/_general.json` 신규 작성

**pixel_budget 상향 조정 (완료)**
- [x] `text`: 280 → **560** (소형 폰트·수기 대응) — `resolution_router.py` + `instruction_router.py` 동기화
- [x] `_FALLBACK_PIXEL_BUDGET`: 280 → 560
- ~~handwritten_field 560 → 1120~~ — v2 `RegionType`에 `HANDWRITTEN_FIELD`가 없음(VLM이 내용으로 판별). `text` 상향으로 통합 대응. RegionType 확장은 별도 논의 필요

**1-shot 예시 + CoT 스키마 (완료)**
- [x] 6개 군수 스키마 + `_fallback.json`에 `analysis` 필드 최상단 + `required` 포함
- [x] `configs/instruction_examples/{supply_request,maintenance_record,inventory_sheet,handover_doc,inspection_report}.yaml` 5개 작성
- [x] InstructionRouter `_load_examples()` + `route()`에서 user_instruction 끝에 [예시] 블록 자동 부착
- [x] 군수 경로 user_instruction 끝에 `_COT_INSTRUCTION` 자동 부착 (other는 제외)
- [x] StructuredExtractor `_process_single()`에서 `analysis` 키 필드 스킵 (DB 오염 방지, trace에는 raw_response로 보존)

**Phase 1-E 통합 테스트 (2026-04-10 통과)**
- [x] P1 → P6 end-to-end 3건 PASS, errors=0
- [x] other 분기 검증: 테스트 3건 전부 `form_type=other`로 올바르게 분류, 검토 큐 미적재, 처리 시간 약 55% 단축
- [x] Layout HTTP 컨테이너(8082) + vLLM 컨테이너(8100) 경유 동작 확인
- [x] 단계별 출력 검증: P2 시각화, P2.5A/B/C, P3A, P3B region_traces.json
- [ ] PP-DocLayout 원본 검출률 측정 (군수 서식 샘플 50장) — Phase 2-A 교정 데이터 축적 후 진행
- [ ] Fusion ON DPI≥150 / DPI<150 reading_order 분리 검증 (실제 군수 서식 샘플 확보 후)
- [ ] Fallback 전환 경로 실측 — fallback 컨테이너 기동 후 진행

**잔존 이슈 (Phase 1-E 통합 테스트에서 발견, 우선순위순)**

1. **P3-B VLM 응답 JSON parse 실패** 🔴 — 통합 테스트에서 다수의 `Region r_XXXX: JSON parse failed` 경고 발생. VLM이 guided_json 형식을 따르지 않고 마크다운 코드 펜스(```json)·자연어 설명을 섞어 반환하는 경우가 있음.
   - [ ] `StructuredExtractor._process_single()` JSON 파싱 견고화:
     - 코드 펜스 제거 (```json ... ```)
     - 가장 바깥 `{...}` 블록 추출 fallback
     - 파싱 실패 시 1회 재호출 (temperature 0.1)
   - [ ] guided_json이 xgrammar 백엔드에서 실제로 enforce되는지 vLLM 재기동 후 재측정

2. **P3-B 배치 동시 전송 미구현** 🔴 — pixel_budget 그룹 내 영역들이 `for cropped in regions:`로 순차 호출됨. 국회공문서 17영역에서 P3-B에만 15초 소요 (영역당 약 0.9초, 실질적 병렬화 없음).
   - [ ] `asyncio` 전환으로 동일 budget 그룹 내 동시 전송 구현
   - [ ] vLLM `max-num-seqs=64` 제약 내에서 실측 throughput 측정
   - [ ] 예상 효과: 국회공문서 P3-B 15초 → 3~5초 (3~5배 단축)

3. **vLLM 컨테이너 재기동 미적용** 🟡 — docker-compose.yml 최적화 옵션은 반영되었으나 기존 컨테이너는 구 옵션으로 기동 중.
   - [ ] `docker compose restart vllm-server` 후 재측정 (fp8 KV + speculative decoding 효과 확인)
   - [ ] 재기동 전후 동일 문서 처리 시간 비교 기록

4. **P1 Real-ESRGAN 미설치** 🟡 — `basicsr` 모듈 미설치로 LOW DPI 입력(< 150)에서 SR 미적용. 저해상도 입력이 P3-B 정확도 저하의 주요 원인 중 하나.
   - [ ] `pip install basicsr realesrgan` 환경 정비 (pipeline 컨테이너 + dl conda)
   - [ ] Docker 이미지에 basicsr 추가 (`Dockerfile.pipeline`)

5. **`LOW_RESOLUTION_BAND` / `IMAGE_TOO_BRIGHT` 경고 이중 기록** 🟢 — P1 warnings가 P3-B warnings에도 중복 전파됨. `_run_step` 스냅샷 복사 수정 이후에도 result.warnings와 p3_result.warnings 양쪽에 "[P3B] [P1] ..." 형태로 중복 표시.
   - [ ] P3-B에 전달되는 warnings 리스트를 빈 리스트로 생성하고 종료 후 오케스트레이터가 합쳐서 prefix 붙이는 방식으로 변경

6. **layout 시각화 PNG 비어 있음 경고 없음** 🟢 — 저해상도 입력에서 P2 모델이 검출 실패 시 현재는 빈 이미지만 저장됨.
   - [ ] P2.5-A에서 `len(regions) == 0` 시 warning 발생 + 시각화 스킵

7. **RegionType 확장 논의** 🟢 — PP-DocLayout이 수기·서명·체크박스를 모두 `text` 라벨로 반환하여 pixel_budget 세분화가 불가능.
   - [ ] Phase 2 PP-DocLayout Fine-tuning 시 `handwritten_field`, `signature`, `checkbox` 카테고리 추가 검토
   - [ ] 추가 후 `RegionType` enum + `PIXEL_BUDGETS` 재조정

### Phase 2 (06~07월)

**2-A. 검토 큐 UI MVP + 교정 데이터 축적 (병행)**
- [ ] 검토 큐 UI MVP (큐 목록 + 검토 + 필드 수정 + 승인/반려)
- [ ] 교정 데이터 JSONL export 파이프라인 활성화
- [ ] PP-DocLayout 어노테이션 + Fine-tuning (검출률 기반 목표 수량)
- [ ] PP-DocLayoutV3 confidence threshold 튜닝 (Fusion 모드 오검출 억제)
- [ ] AWQ 4-bit 서빙 사전 검증 (Phase 3 리스크 분산)

> Track A(Fine-tuning) 검출률 70%+ 달성 후에만 Track B 교정 데이터를 SFT에 투입

**2-B. Gemma4 VLM Fine-tuning** (교정 데이터 유형당 100건+ 확보 후)
- [ ] P3-A SFT (LoRA rank=8, 유형당 100~200장)
- [ ] P3-B SFT 1단계 (LoRA rank=16, 유형당 300장+)
- [ ] 벤치마크 후 2단계 수기 강화(선택) / 3단계 DPO(운영 3개월 후) 판단

**2-C. Phase 1 잔존 이슈 해결 (병행)**
- [ ] P3-B JSON parse 견고화 (코드 펜스 제거 + 1회 재호출 fallback) — Phase 1-E 잔존 이슈 #1
- [ ] P3-B `asyncio` 기반 배치 동시 전송 구현 — Phase 1-E 잔존 이슈 #2
- [ ] Dockerfile.pipeline에 `basicsr`/`realesrgan` 추가 — Phase 1-E 잔존 이슈 #4
- [ ] P1 warnings 중복 전파 정리 — Phase 1-E 잔존 이슈 #5
- [ ] PP-DocLayout Fine-tuning 시 `handwritten_field`/`signature`/`checkbox` 카테고리 추가 검토 — Phase 1-E 잔존 이슈 #7

### Phase 3 (08~09월)

- [ ] RepetitionGuard 도입 (반복 토큰 비율 > 15% 관측 시)
- [ ] Gemma4 양자화 실험 (AWQ 4-bit 우선, FP8은 vllm#39049 해소 후)
- [ ] 검토 큐 UI 고도화 (대시보드, bbox 오버레이, 담당자 관리)
- [ ] 비동기 메시지 기반 통신 전환 (Redis Streams)
- [ ] Schema Registry 운영 시나리오 검증 (서식 개정 → DB 마이그레이션)
- [ ] Docker 프로덕션 설정 + 보안 검토 + 배포

---

## 보안 고려사항

- 군수 서식 이미지 — **로컬 추론 전용** (클라우드 API 금지)
- Gemma4 — 로컬 vLLM 서버 구동 (외부 통신 차단)
- 학습 데이터·도메인 사전 — 암호화 저장 권장
- `VLMResult.raw_json` — DB 적재 제외. 디버그용 영역별 raw_response는 파일 시스템 `data/pipeline_outputs/<ts>/<doc>/P3/region_traces.json`에만 보존

---

## 하드웨어 요구사항

| 용도 | VRAM | GPU | 비고 |
|------|------|-----|------|
| Gemma4 추론 (BF16 + FP8 KV Cache) | ~48GB + KV절감 | H100 80GB | 현재 운영 |
| Gemma4 추론 (AWQ 4-bit) | ~16GB | RTX 4090 | Phase 3 실험 |
| Gemma4 추론 (FP8 Dynamic) | ~27GB | A100 | vllm#39049 해소 후 |
| PP-DocLayout 추론 | 4GB | GPU | PaddlePaddle |
| Real-ESRGAN SR | 2GB | GPU | 타일 기반 |
| Gemma4 Fine-tuning (LoRA) | 16~24GB | GPU | PEFT |
