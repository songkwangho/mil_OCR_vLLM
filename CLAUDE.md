# 군수 OCR 시스템 v2 — Claude Code 마스터 컨텍스트

> 이 파일은 프로젝트 전체 맥락의 **요약본**입니다.
> 상세 내용은 `docs/` 하위 문서를 참조하십시오.

---

## 핵심 원칙

- **폐쇄망 전용** — 외부 API 호출, 모델 다운로드 절대 금지. 모든 가중치는 `models/`에 사전 배치
- **3 Stage 파이프라인** — 전처리(P1~P2.5-A) → VLM 통합 추론(P3-A~P3-B) → 후처리(P4~P6)
- **Crop-then-Infer** — PP-DocLayout bbox 크롭 이미지를 영역별로 Gemma4에 전달 (전체 페이지 입력 지양)
- **FormClassifier 선행** — 서식 분류를 InstructionRouter보다 먼저 수행하여 도메인 맥락 기반 instruction 생성
- **Guided Decoding** — VLM 출력을 군수 서식별 JSON Schema로 구조 보장
- **logprobs 신뢰도** — VLM 토큰 확률 기하평균 + 길이 편향 보정 기반 필드별 신뢰도 산출
- **오프라인 모델 로드** — `HF_HUB_OFFLINE=1`, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`

---

## v1 → v2 아키텍처 전환 배경

v1 (mil_OCR)은 T1~T12의 12개 독립 컴포넌트로 구성되어 있으나:
- PP-DocLayout이 한국어 문서에서 검출 실패 (pretrained 한계)
- 개별 AI 모델 7개를 각각 Fine-tuning해야 하는 부담
- 파이프라인 단계 간 데이터 변환 복잡도 높음

v2는 **PP-DocLayout (레이아웃) + Gemma4 VLM (통합 추론)** 구조로 재설계하여:
- AI 모델 수: 7개 → **2개** (레이아웃 + VLM)
- Fine-tuning 대상: 7개 → **2개**
- PaddleOCR-VL 파이프라인 분석을 통해 **Crop-then-Infer + FormClassifier 선행 + 배치 병렬 처리** 구조로 P2~P3 구간을 고도화

---

## 문서 구조

| 문서 | 내용 | 언제 읽을 것인가 |
|------|------|----------------|
| `docs/AI_INFERENCE.md` | AI 모델 추론 방식, 모델 가중치, Gemma4 VLM 설정, Docker 환경 | AI 추론 관련 작업 시 |
| `docs/AI_TRAINING.md` | Fine-tuning 방법, 학습 데이터, VLM 미세조정 전략 | AI 재학습 관련 작업 시 |
| `docs/BACKEND.md` | 파이프라인 구조, 인터페이스 타입, 후처리 로직, 도메인 지식 | 백엔드 로직 작업 시 |
| `docs/FRONTEND.md` | 프론트엔드 UI (현재 미구현) | 프론트엔드 작업 시 |

**작업 시 이 파일(CLAUDE.md) + 해당 영역의 docs 파일만 읽어서 진행합니다.**

---

## 시스템 파이프라인 (v2 — 고도화)

```
[Stage 1 — 전처리]
    P1: 화질 보정 + SR            → PreprocessedImage
    P2: 레이아웃 탐지              → RawLayoutResult  (PP-DocLayout 원시 탐지)
        ├── Fusion OFF: V3 단독 탐지
        └── Fusion ON (LAYOUT_FUSION_MODE=true):
            ├── DPI ≥ 150: V3(구조) + plus-L(텍스트), 50% 겹침 필터
            └── DPI < 150: V3(구조) + OpenCV heuristic(텍스트), 50% 겹침 필터
            reading_order: V3 구조 영역 순서 → 텍스트 좌상→우하 정렬
    P2.5-A: LayoutPostProcessor   → LayoutResult     (노이즈 제거 + 블록 병합)
        ① 미소 박스 제거 (6px 미만)
        ② 중복 박스 제거 (IoU > 0.7)
        ③ 인접 텍스트 블록 병합
        ↓
[Stage 2 — VLM 통합 추론]              ┌─ [Fallback — VLM 불가 시]
    P3-A: FormClassifier               │   v1 T3~T5 (PP-OCRv5)
        전체 페이지 → 저해상도(140토큰) │   → 서식 분류 + 텍스트 인식 + 표 구조
        → form_type 확정               │   → 결과는 항상 검토 큐 적재
        ↓
    P2.5-B: InstructionRouter          │
        region_type + form_type        │
        → 도메인 맥락 포함 instruction  │
        → JSON Schema 선택             │
        ↓
    P2.5-C: ResolutionRouter           │
        영역 타입별 pixel_budget 할당   │
        (표: 1120, 수기: 560, 헤더: 140토큰)
        ↓
    P3-B: StructuredExtractor          │
        bbox 크롭 이미지 + instruction │
        → pixel_budget 기준 배치 그룹화│
        → Gemma4 병렬 배치 호출        │
        → logprobs 신뢰도 산출         │
        → RepetitionGuard              │
        ↓                             └───────────────┐
[Stage 3 — 후처리]                                    │
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult        │
        ├─ ✅ 통과 → P5 직렬화 → P6 DB 적재 → PipelineOutput
        └─ ❌ 실패 (CRITICAL 오류 or LOW confidence)   │
                   → 검토 큐 적재 → 담당자 UI 확인     ↓
                   → 수정 후 P6 재적재
```

### P2~P3 구간 설계 근거 (PaddleOCR-VL 분석 기반)

**Q1. Layout detection 후 bbox 크롭 이미지를 Gemma4에 보내는 게 더 정확하지 않나?**

맞습니다. 이것이 Crop-then-Infer 방식의 핵심 근거입니다. 전체 페이지를 통째로 VLM에 입력하면 다단 컬럼, 표·수기·텍스트 혼재 환경에서 long-sequence 디코딩 지연과 누락이 발생합니다. PP-DocLayout이 영역을 분리한 뒤 각 bbox 크롭 이미지만 VLM에 전달하면 영역 집중도가 높아지고 오인식이 줄어듭니다. LayoutPostProcessor가 미소 박스·중복 박스를 제거하는 선행 정제 없이는 VLM이 의미 없는 조각 이미지를 받게 되므로, 정제 단계가 반드시 선행되어야 합니다.

**Q2. 처리속도 문제는 배치 처리로 해결 가능하지 않나?**

맞습니다. P3-B StructuredExtractor는 ResolutionRouter가 할당한 pixel_budget 기준으로 같은 해상도 그룹을 배치로 묶어 vLLM에 동시 전송합니다. 해상도가 다른 이미지가 섞이면 패딩 오버헤드가 발생하므로, 동일 pixel_budget끼리 그룹화하는 것이 vLLM 배치 처리 효율의 핵심입니다. `--max-num-seqs 128` 제약 안에서 그룹별 배치 전송으로 처리량을 최대화합니다.

**Q3. 문서 분류를 먼저 해서 InstructionRouter에 전달해야 더 정확한 instruction이 나오지 않나?**

맞습니다. 같은 `text` 라벨 영역이라도 서식 유형에 따라 instruction이 달라져야 합니다. P3-A FormClassifier가 전체 페이지를 저해상도로 먼저 분류하여 form_type을 확정하고, 이 결과를 InstructionRouter에 전달합니다. form_type 없이 영역별 instruction을 생성하면 도메인 맥락이 빠진 범용 instruction만 만들어집니다.

### 핵심 메커니즘

- **Layout Fusion**: V3(구조) + plus-L(텍스트) 결합, DPI 기반 분기 (`LAYOUT_FUSION_MODE=true`)
- **Crop-then-Infer**: PP-DocLayout bbox → LayoutPostProcessor 정제 → 영역별 크롭 → Gemma4
- **FormClassifier 선행**: 전체 페이지 저해상도 분류 → form_type → InstructionRouter 피드백
- **InstructionRouter**: region_type + form_type → 도메인 맥락 포함 상세 instruction 생성
- **ResolutionRouter**: 영역 타입별 pixel_budget 차등 할당 (표/수기 고해상도 집중)
- **배치 병렬 처리**: pixel_budget 기준 그룹화 → vLLM 동시 전송
- **Guided Decoding**: 군수 서식별 JSON Schema를 `guided_json`으로 vLLM에 전달
- **logprobs 신뢰도**: 필드별 토큰 확률 → 기하평균 + 길이 편향 보정(`length_factor`) 기반 확신도 환산
- **SPOF 대비 (3단계)**:
  - 수준 A: vLLM `/health` 헬스체크 + `restart: unless-stopped`
  - 수준 B: v1 PP-OCRv5 기반 경량 fallback 서비스 (별도 컨테이너 + HTTP API)
  - 수준 C: `fallback_base_url` 미설정 시 `_DisabledFallback` 더미 (in-process v1 import는 v2 enums와 충돌하여 사용 불가) → 검토 큐만 활성
- **수동 검토 큐**: P4 검증 실패 → 큐 적재 → 담당자 교정 → DB 재적재

### v1 → v2 컴포넌트 매핑

| v1 | v2 | 비고 |
|----|-----|------|
| T1 화질 보정 | **P1** 화질 보정 + SR | v1에서 이관 (Real-ESRGAN 포함) |
| T2 레이아웃 분석 | **P2** 레이아웃 탐지 | PP-DocLayout 원시 탐지 (정제 전) |
| — | **P2.5-A** LayoutPostProcessor | 신규 — 미소/중복 박스 제거, 블록 병합 |
| T3 서식 분류 | **P3-A** FormClassifier | 전체 페이지 저해상도 VLM 호출로 분리 |
| — | **P2.5-B** InstructionRouter | 신규 — form_type 반영 도메인 instruction |
| — | **P2.5-C** ResolutionRouter | 신규 — 영역별 pixel_budget 할당 |
| T4 수기 인식 | **P3-B** StructuredExtractor | bbox 크롭 + 배치 병렬 VLM 호출 |
| T5 구조 인식 | **P3-B** StructuredExtractor | 표 crop 고해상도 처리 |
| T6 특수 코드 | **P3-B** StructuredExtractor | 코드 패턴 instruction 포함 |
| T7 언어모델 교정 | **P3-B** StructuredExtractor | 문맥 기반 교정 |
| T8 데이터 검증 | **P4** 룰 검증 | 산술/날짜/코드 교차검증 |
| T9 신뢰도 스코어링 | **P3-B** (logprobs) | 토큰 확률 기하평균 + 길이 편향 보정 → 필드 신뢰도 |
| T10 스키마 매핑 | **P3-B** (guided_json) | JSON Schema로 구조화 추출 |
| T11 직렬화 | **P5** 직렬화 | v1에서 이관 |
| T12 DB 적재 | **P6** DB 적재 | v1에서 이관 |

---

## 구현 현황

| P# | 컴포넌트 | 소스 파일 | AI 모델 / 기술 | 상태 |
|----|----------|----------|---------------|------|
| — | 공용 인터페이스 | `src/interfaces/` | — | ✅ 완료 (14 Enum + 17 dataclass) |
| — | 오케스트레이터 | `src/pipeline/orchestrator.py` | — | ✅ 완료 (Phase 1-E 통합 테스트 통과, PipelineResult에 p2_5b_instructions/p2_5c_groups/p3b_trace 필드 보유) |
| P1 | 화질 보정 + SR | `src/preprocess/preprocessor.py` | Real-ESRGAN (LOW DPI) | ✅ 완료 (v1 이관) |
| P2 | 레이아웃 탐지 | `src/preprocess/layout_analyzer.py` | PP-DocLayout_plus-L / PP-DocLayoutV3 | ✅ 완료 (원시 탐지) |
| P2.5-A | LayoutPostProcessor | `src/preprocess/layout_postprocessor.py` | — (룰 기반) | ✅ 완료 (미소/중복 박스 제거, 블록 병합) |
| P3-A | FormClassifier | `src/vlm/form_classifier.py` | Gemma4 (저해상도 140토큰) | ✅ 완료 (서식 분류 + logprobs 신뢰도) |
| P2.5-B | InstructionRouter | `src/vlm/instruction_router.py` | — (YAML 매핑) | ✅ 완료 (form_type 반영 도메인 instruction) |
| P2.5-C | ResolutionRouter | `src/vlm/resolution_router.py` | — (룰 기반) | ✅ 완료 (pixel_budget 할당 + 배치 그룹화) |
| P3-B | StructuredExtractor | `src/vlm/structured_extractor.py` | Gemma4 + vLLM (배치 병렬) | ✅ 완료 (guided_json + logprobs 신뢰도 + 영역별 trace 옵션) |
| P3-B | RepetitionGuard | `src/vlm/repetition_guard.py` | — | 🟡 선택 구현 (운영 후 판단) |
| — | VLM 공용 클라이언트 | `src/vlm/vlm_client.py` | — | ✅ 완료 (vLLM HTTP 통신, 이미지 base64 인코딩, logprobs 파싱) |
| P4 | 룰 검증 + 신뢰도 보정 | `src/postprocess/validator.py` | ❌ (룰 엔진) | ✅ 완료 (VLM/Fallback 경로별 임계값 분리) |
| P5 | 직렬화 | `src/postprocess/serializer.py` | ❌ | ✅ 완료 (v1 이관) |
| P6 | DB 적재 | `src/postprocess/db_loader.py` | ❌ | ✅ 완료 (v1 이관 + v2 스키마) |
| — | 수동 검토 큐 | `src/postprocess/review_queue.py` | ❌ (SQLite 큐) | ✅ 완료 |
| — | Fallback 서비스 (in-process) | `src/fallback/ocr_fallback_service.py` | v1 PP-OCRv5 T3~T5 | ⚠️ 사용 금지 (v1/v2 enum 충돌, HTTP 모드만 사용) |
| — | Fallback HTTP 서버 | `src/fallback/server.py` | FastAPI | ✅ 완료 (POST /fallback/process) |
| — | Fallback HTTP 클라이언트 | `src/fallback/fallback_http_client.py` | — | ✅ 완료 |
| — | Pipeline HTTP 서버 | `src/pipeline/server.py` | FastAPI | ✅ 완료 (POST /pipeline/run, /upload) |
| — | VLM 헬스 모니터 | `src/pipeline/health_monitor.py` | ❌ (httpx polling) | ✅ 완료 |
| — | Fallback 전환 정책 | `src/pipeline/fallback_policy.py` | ❌ | ✅ 완료 |
| — | Layout 추론 서비스 | `src/preprocess/layout_server.py` + `layout_http_client.py` | PP-DocLayout | ✅ 완료 (포트 8082) |
| — | 스키마 레지스트리 | `src/domain/schema_registry.py` | ❌ | ✅ 완료 (버전 관리 지원) |
| — | Docker 구성 | `docker-compose.yml` + `docker/Dockerfile.*` | — | ✅ 완료 (5 Dockerfile, CUDA 12.6 H100 호환) |

> **Legacy 파일 (신규 코드에서 사용 금지)**:
> - `src/vlm/instruction_builder.py` — InstructionRouter 위임 래퍼. 신규 코드는 `InstructionRouter`를 직접 사용.
> - `src/vlm/gemma4_engine.py` — FormClassifier + StructuredExtractor 분리 전 monolithic 구조 (730줄). `encode_image_base64()`, `_detect_domain_codes()` 등이 vlm_client.py / structured_extractor.py와 중복. 통합 테스트 완료 후 제거 예정.

---

## 디렉토리 구조

```
mil_OCR_v2/
├── CLAUDE.md                    ← 이 파일
├── docs/
│   ├── AI_INFERENCE.md          ← AI 추론 상세
│   ├── AI_TRAINING.md           ← AI 재학습 상세
│   ├── BACKEND.md               ← 백엔드 상세
│   └── FRONTEND.md              ← 프론트엔드 상세
├── src/
│   ├── interfaces/              ← 공용 타입, Enum
│   ├── pipeline/                ← orchestrator.py + server.py (FastAPI) + health_monitor + fallback_policy
│   ├── preprocess/              ← P1 화질 보정, P2 레이아웃 (analyzer + server.py + http_client), P2.5-A LayoutPostProcessor
│   ├── vlm/                     ← P3-A FormClassifier, P2.5-B InstructionRouter,
│   │                               P2.5-C ResolutionRouter, P3-B StructuredExtractor
│   ├── postprocess/             ← P4 룰 검증, P5 직렬화, P6 DB 적재, 검토 큐
│   ├── fallback/                ← 수준 B 경량 fallback: server.py (FastAPI) + http_client + ocr_fallback_service (in-process는 사용 금지)
│   └── domain/                  ← 군수 도메인 사전/스키마
├── models/                      ← 모델 가중치 (gitignore)
│   ├── t1_sr/                   ← Real-ESRGAN
│   ├── t2_layout/               ← PP-DocLayout_plus-L / PP-DocLayoutV3 (선택)
│   ├── gemma4/                  ← Gemma4 26B-A4B
│   └── fallback/                ← PP-OCRv5 + DiT + SLANeXt (수준 B)
├── data/                        ← 원본/학습/테스트 데이터
├── docker/                      ← Dockerfile (vllm, pipeline, fallback, train)
├── docker-compose.yml           ← Docker 서비스
├── scripts/                     ← 배치/테스트 스크립트
├── training/                    ← Fine-tuning 스크립트
├── configs/                     ← 파이프라인 설정
└── tests/                       ← 단위/통합 테스트
```

---

## 개발 로드맵 (전체 6개월)

> 시작 기준: 2026-04 (현재)

### Phase 1 잔여 (04월 말) — 파이프라인 구조 고도화

**1-D. P2~P3 구간 신규 컴포넌트 구현** (최우선)
- [ ] PP-DocLayout 원본 검출률 측정 (군수 서식 샘플 50장) — Fine-tuning 목표 수량 조정 근거
- [x] `src/preprocess/layout_postprocessor.py` — LayoutPostProcessor (미소/중복 박스 제거, 블록 병합)
- [x] `src/vlm/form_classifier.py` — FormClassifier P3-A (전체 페이지 저해상도 서식 분류)
- [x] `src/vlm/instruction_router.py` — InstructionRouter (form_type 반영 도메인 instruction 생성)
- [x] `src/vlm/resolution_router.py` — ResolutionRouter (영역별 pixel_budget 할당)
- [x] `src/vlm/structured_extractor.py` — StructuredExtractor P3-B (bbox 크롭 + 배치 병렬 VLM 호출 + trace 수집 옵션)
- [x] `src/vlm/vlm_client.py` — VLM 공용 클라이언트 (이미지 인코딩, logprobs 파싱, 도메인 코드 감지)
- [x] `src/interfaces/types.py` 갱신 — RawLayoutResult, InstructionSpec, CroppedRegion 추가
- [x] P4 validator.py — 처리 경로별 신뢰도 임계값 분리 (VLM vs Fallback)
- [x] 오케스트레이터 재작성 — P3-A → InstructionRouter → ResolutionRouter → P3-B 흐름 반영
- [x] `src/pipeline/server.py` 신규 — Pipeline FastAPI HTTP 래퍼 (POST /pipeline/run, /upload)
- [x] `src/fallback/server.py` 신규 — Fallback FastAPI HTTP 래퍼 (POST /fallback/process)
- [x] PipelineResult 확장 — `p2_5b_instructions`, `p2_5c_groups`, `p3b_trace` 필드 추가 (단계별 결과 추적용)
- [x] `_run_step` 무한 루프 버그 수정 — `output.warnings`와 `result.warnings` 동일 리스트 참조 시 OOM 발생 → 스냅샷 복사 + 중복 체크
- [x] `_get_fallback_service()` 수정 — `fallback_base_url` 미설정 시 `_DisabledFallback` 더미 반환 (in-process v1 import는 v2 enums와 충돌)

**1-E. 통합 테스트** ✅ 완료 (2026-04-10)
- [x] P1 → P2 → P2.5-A → P3-A → P2.5-B → P2.5-C → P3-B → P4 → P5 → P6 end-to-end 테스트 — 3 문서 PASS, errors=0
- [x] Layout HTTP 컨테이너(8082) + vLLM 컨테이너(8100) 활용 검증 — H100 GPU 정상 동작
- [x] 단계별 출력 저장 검증 — `data/pipeline_outputs/{ts}/{doc_id}/{P1..P6,P2.5A,P2.5B,P2.5C,P3A}/` (시각화 PNG + JSON + 영역별 trace)
- [ ] 군수 서식 샘플별 단계 출력 비교 (기존 P3 단일 호출 대비 정확도 측정) — Phase 2-A 교정 데이터 축적 후 진행

### Phase 2 (3~4개월, 06~07월) — Fine-tuning + 품질 개선

**2-A. 검토 큐 UI MVP + 교정 데이터 축적 (병행 Track)**

> 검토 큐 UI는 Phase 3에서 Phase 2-A로 앞당겨 편입. SFT 학습 데이터 품질 확보를 위해 Fine-tuning 착수 전에 교정 데이터 파이프라인이 활성화되어야 함.

- [ ] 검토 큐 UI MVP 구현 (큐 목록 + 개별 검토 + 필드 수정 + 승인/반려)
- [ ] 교정 데이터 JSONL export 파이프라인 활성화
- [ ] PP-DocLayout 어노테이션 (군수 서식 영역 2,000~5,000장)
- [ ] PP-DocLayout Fine-tuning 실행 + P2 검출 품질 검증
- [ ] AWQ 4-bit 서빙 가능성 사전 검증 (Phase 3 리스크 분산)

**2-B. Gemma4 VLM Fine-tuning — 3단계** (교정 데이터 유형당 100건+ 확보 후)
- [ ] **1단계 SFT** (필수): LoRA, 유형당 300장 이상 목표
- [ ] 1단계 SFT 후 전체 파이프라인 성능 벤치마크
- [ ] **2단계 수기 강화** (선택): 수기 오류율 높을 시
- [ ] **3단계 DPO** (선택): 시스템 운영 3개월 후 자동 축적 데이터 기반

### Phase 3 (5~6개월, 08~09월) — 최적화 + 배포

- [ ] RepetitionGuard 도입 (운영 중 반복 생성 문제 관측 시)
- [ ] Gemma4 양자화 실험 — AWQ 4-bit 우선, FP8은 vllm#39049 해소 후
- [ ] 검토 큐 UI 고도화 (대시보드, 통계, bbox 오버레이)
- [ ] 비동기 메시지 기반 통신 전환 (Redis Streams)
- [ ] 도메인 스키마 버전 관리 도입 (Schema Registry)
- [ ] Docker 프로덕션 설정 + 보안 검토 + 배포

---

## 보안 고려사항

- 군수 서식 이미지는 민감 데이터 — **로컬 추론 전용** (클라우드 API 금지)
- Gemma4 모델은 로컬 vLLM 서버로 구동 (외부 통신 차단)
- 모든 모델은 폐쇄망에서 동작 가능해야 함
- 학습 데이터 및 도메인 사전은 암호화 저장 권장

---

## 하드웨어 요구사항

| 용도 | GPU | VRAM | 비고 |
|------|-----|------|------|
| Gemma4 26B-A4B 추론 (BF16) | NVIDIA GPU | **~48GB** | 현재 운영 중 (H100 80GB) |
| Gemma4 26B-A4B 추론 (AWQ 4-bit) | NVIDIA GPU | **~16GB** | Phase 3 양자화 실험 후 전환 검토 |
| Gemma4 26B-A4B 추론 (FP8 Dynamic) | NVIDIA GPU | **~27GB** | vllm#39049 해소 후 실험 |
| PP-DocLayout 추론 | NVIDIA GPU | 4GB | PaddlePaddle |
| Real-ESRGAN SR | NVIDIA GPU | 2GB | 타일 기반 처리 |
| Gemma4 Fine-tuning | NVIDIA GPU | 24GB+ | LoRA 시 16GB |
