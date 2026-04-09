# 군수 OCR 시스템 v2 — Claude Code 마스터 컨텍스트

> 이 파일은 프로젝트 전체 맥락의 **요약본**입니다.
> 상세 내용은 `docs/` 하위 문서를 참조하십시오.

---

## 핵심 원칙

- **폐쇄망 전용** — 외부 API 호출, 모델 다운로드 절대 금지. 모든 가중치는 `models/`에 사전 배치
- **3단계 파이프라인** — 전처리 → VLM 통합 추론 → 후처리 (v1의 12단계에서 6단계로 단순화)
- **Guided Decoding** — VLM 출력을 군수 서식별 JSON Schema로 구조 보장
- **logprobs 신뢰도** — VLM 토큰 확률 기반 필드별 정밀 신뢰도 산출
- **오프라인 모델 로드** — `HF_HUB_OFFLINE=1`, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`

---

## v1 → v2 아키텍처 전환 배경

v1 (mil_OCR)은 T1~T12의 12개 독립 컴포넌트로 구성되어 있으나:
- PP-DocLayout이 한국어 문서에서 검출 실패 (pretrained 한계)
- 개별 AI 모델 7개를 각각 Fine-tuning해야 하는 부담
- 파이프라인 단계 간 데이터 변환 복잡도 높음

v2는 **PP-DocLayout (레이아웃) + Gemma4 VLM (통합 추론)** 2단계 구조로 재설계하여:
- AI 모델 수: 7개 → **2개** (레이아웃 + VLM)
- Fine-tuning 대상: 7개 → **2개**
- 파이프라인 단순화: 12단계 → **6단계** (전처리2 + VLM1 + 후처리3)

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

## 시스템 파이프라인 (v2)

```
[Stage 1 — 전처리]
    P1: 화질 보정 + SR       → PreprocessedImage
    P2: 구조 분석             → LayoutResult (regions + bbox + reading_order)
        ① 페이지 분해: Dense 페이지를 관리 가능한 영역 bbox로 분리
        ② 읽기 순서: 영역 간 올바른 처리 순서 보장 (다단 컬럼 등)
        ③ Task Prompt 결정: 영역 레이블 → VLM instruction 자동 선택
        ↓
[Stage 2 — VLM 통합 추론]              ┌─ [Fallback — VLM 불가 시]
    P3: Gemma4 VLM → VLMResult         │   v1 T3~T5 (PP-OCRv5)
        - OCR + 교정                    │   → 서식 분류 + 텍스트 인식 + 표 구조
        - 표 구조 인식                   │   → 결과는 항상 검토 큐 적재
        - 서식 분류                      │
        - 특수 코드 인식                 └───────────────┐
        - 스키마 매핑 (guided_json)                      │
        - logprobs 신뢰도                                │
        ↓                                                ↓
[Stage 3 — 후처리]
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult
        ├─ ✅ 통과 → P5 직렬화 → P6 DB 적재 → PipelineOutput
        └─ ❌ 실패 (CRITICAL 오류 or LOW confidence)
                   → 검토 큐 적재 → 담당자 UI 확인 → 수정 후 P6 재적재
```

### 핵심 메커니즘

- **Guided Decoding**: 군수 서식별 JSON Schema를 `guided_json`으로 vLLM에 전달
  → VLM 출력이 항상 유효한 JSON 구조를 보장, 파싱 실패 원천 차단
- **logprobs 신뢰도**: vLLM `logprobs` 옵션으로 출력 토큰별 로그 확률 반환
  → 필드값 토큰의 확률을 확신도(0.0~1.0)로 환산, 수치 필드에 특히 정밀
- **SPOF 대비 (2단계)**:
  - 수준 A: vLLM `/health` 헬스체크 + `restart: unless-stopped` → 자동 감지·복구
  - 수준 B: v1 PP-OCRv5 기반 경량 fallback 서비스 → VLM 불가 시 기초 OCR 유지
- **수동 검토 큐**: P4 검증 실패(CRITICAL/LOW confidence) 문서 → 큐 적재 → 담당자 교정 → DB 재적재

### v1 → v2 컴포넌트 매핑

| v1 | v2 | 비고 |
|----|-----|------|
| T1 화질 보정 | **P1** 화질 보정 + SR | v1에서 이관 (Real-ESRGAN 포함) |
| T2 레이아웃 분석 | **P2** 구조 분석 | PP-DocLayout — VLM의 전처리 게이트 (분해+순서+태스크) |
| T3 서식 분류 | **P3** VLM | instruction으로 분류 |
| T4 수기 인식 | **P3** VLM | 영역별 OCR |
| T5 구조 인식 | **P3** VLM | 표 HTML 출력 |
| T6 특수 코드 | **P3** VLM | 코드 패턴 인식 |
| T7 언어모델 교정 | **P3** VLM | 문맥 기반 교정 |
| T8 데이터 검증 | **P4** 룰 검증 | 산술/날짜/코드 교차검증 |
| T9 신뢰도 스코어링 | **P3** VLM (logprobs) | VLM 토큰 확률 → 필드 신뢰도 |
| T10 스키마 매핑 | **P3** VLM (guided_json) | JSON Schema로 구조화 추출 |
| T11 직렬화 | **P5** 직렬화 | v1에서 이관 |
| T12 DB 적재 | **P6** DB 적재 | v1에서 이관 |

---

## 구현 현황

| P# | 컴포넌트 | 소스 파일 | AI 모델 / 기술 | 상태 |
|----|----------|----------|---------------|------|
| — | 공용 인터페이스 | `src/interfaces/` | — | ✅ 완료 (14 Enum + 15 dataclass) |
| — | 오케스트레이터 | `src/pipeline/orchestrator.py` | — | ✅ 완료 (6단계 재작성) |
| P1 | 화질 보정 + SR | `src/preprocess/preprocessor.py` | Real-ESRGAN (LOW DPI) | ✅ 완료 (v1 이관) |
| P2 | 구조 분석 | `src/preprocess/layout_analyzer.py` | PP-DocLayout_plus-L / PP-DocLayoutV3 선택 (분해+순서+태스크) | ✅ 완료 (v1 수정 이관 + V3 지원) |
| P3 | VLM 통합 추론 | `src/vlm/gemma4_engine.py` | Gemma4 26B-A4B + vLLM (guided_json + logprobs) | ✅ 완료 |
| P4 | 룰 검증 + 신뢰도 보정 | `src/postprocess/validator.py` | ❌ (룰 엔진) | ✅ 완료 (logprobs + 룰 병합) |
| P5 | 직렬화 | `src/postprocess/serializer.py` | ❌ | ✅ 완료 (v1 이관) |
| P6 | DB 적재 | `src/postprocess/db_loader.py` | ❌ | ✅ 완료 (v1 이관 + v2 스키마) |
| — | **수동 검토 큐** | `src/postprocess/review_queue.py` | ❌ (SQLite 큐) | ✅ 완료 |
| — | **Fallback 서비스** | `src/fallback/ocr_fallback_service.py` | v1 PP-OCRv5 T3~T5 | ✅ 완료 (v1 래핑 + v2 변환) |
| — | **VLM 헬스 모니터** | `src/pipeline/health_monitor.py` | ❌ (httpx polling) | ✅ 완료 |
| — | **Fallback 전환 정책** | `src/pipeline/fallback_policy.py` | ❌ | ✅ 완료 |
| — | **Layout 추론 서비스** | `src/preprocess/layout_server.py` + `layout_http_client.py` | PP-DocLayout (PaddlePaddle) | ✅ 완료 (CUDA 격리 컨테이너) |
| — | **Docker 구성** | `docker-compose.yml` + `docker/Dockerfile.*` | — | ✅ 완료 (5 Dockerfile) |

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
│   ├── pipeline/                ← 오케스트레이터
│   ├── preprocess/              ← P1 화질 보정, P2 레이아웃
│   ├── vlm/                     ← P3 Gemma4 VLM 엔진
│   ├── postprocess/             ← P4 룰 검증, P5 직렬화, P6 DB 적재, 검토 큐
│   ├── fallback/                ← 수준 B 경량 fallback (v1 T3~T5 래핑)
│   └── domain/                  ← 군수 도메인 사전/스키마
├── models/                      ← 모델 가중치 (gitignore)
│   ├── t1_sr/                   ← Real-ESRGAN
│   ├── t2_layout/               ← PP-DocLayout_plus-L / PP-DocLayoutV3 (선택)
│   ├── gemma4/                  ← Gemma4 26B-A4B
│   └── fallback/                ← PP-OCRv5 + DiT + SLANeXt (수준 B)
├── data/                        ← 원본/학습/테스트 데이터
├── docker/                      ← Dockerfile (vllm, pipeline, fallback, train)
├── docker-compose.yml           ← Docker 서비스 (vllm-server, pipeline, fallback, train)
├── scripts/                     ← 배치/테스트 스크립트
├── training/                    ← Fine-tuning 스크립트
├── configs/                     ← 파이프라인 설정
└── tests/                       ← 단위/통합 테스트
```

---

## 개발 로드맵 (전체 6개월)

> 시작 기준: 2026-04 (현재)

### Phase 1 (1~2개월, 04~05월) — 핵심 파이프라인 구축

**1-A. v1 이관 + 인터페이스 정의** (이관 우선순위: interfaces → preprocess → postprocess)
- [x] `src/interfaces/` 타입 재설계 (v1 축소 + VLMResult/FieldValue/ValidatedResult 신규) ✅
- [x] P1 화질 보정 + SR (v1 T1 **그대로 이관**, import 경로 변경) ✅
- [x] P1 SR (v1 sr_enhancer.py **그대로 이관**) ✅
- [x] P2 구조 분석 (v1 T2 **수정 이관** — reading_order 강화, TASK_PROMPTS 추가) ✅
- [x] Docker 통합 환경 구축 (PaddlePaddle + PyTorch + vLLM) ✅

**1-B. Gemma4 VLM 엔진 구현** (신규)
- [x] Gemma4 26B-A4B 모델 다운로드 + 로컬 배치 ✅ (`models/gemma4/gemma-4-26b-a4b-it/`, ~48GB BF16)
- [x] vLLM 서빙 설정 (오프라인 모드, guided_json, logprobs) ✅ (docker-compose + Dockerfile)
- [x] vLLM 서버 로컬 기동 검증 ✅ (v0.19.0, GPU #2 H100 80GB, `--max-num-seqs 128`)
- [x] `src/vlm/gemma4_engine.py` — vLLM API 호출, 이미지 + instruction 전달 ✅
- [x] `src/vlm/instruction_builder.py` — P2 레이블 → instruction 자동 생성 ✅
- [x] `src/vlm/logprobs_scorer.py` — logprobs → 필드별 신뢰도 환산 ✅
- [x] `src/domain/schemas/*.json` — 군수 서식별 JSON Schema 6종 (guided_json용) ✅
- [x] P2 → P3 연결 (영역별 crop + reading_order + task prompt) ✅ (gemma4_engine.process)
- [ ] P1 → P2 → P3 통합 테스트 (VLM 서버 가동 후) — 테스트 스크립트 작성 완료 (`tests/test_integration_pipeline.py`)

**1-C. 후처리 이관 + 재작성**
- [x] P4 룰 검증 + 신뢰도 보정 (v1 T9 **대폭 수정** — logprobs 기반 + T8 룰 병합) ✅
- [x] P4 → 검토 큐 적재 로직 (`src/postprocess/review_queue.py` — CRITICAL/LOW confidence 판정) ✅
- [x] P5 직렬화 (v1 T11 **이관** — ValidatedResult 입력) ✅
- [x] P6 DB ��재 (v1 T12 **이관** + DB 스키마 v2 조정: processing_path, review_queue_id) ✅
- [x] 오케스트레이터 **재작성** (12단계 → 6단계, VLM/Fallback/검토큐 분기) ✅
- [x] VLM 헬스 모니터 구현 (`src/pipeline/health_monitor.py` — 수준 A) ✅
- [x] Fallback 서비스 구현 (`src/fallback/ocr_fallback_service.py` — v1 T3~T5 래핑 + v2 인터페이스 어댑터) ✅
- [x] Fallback 전환 정책 (`FallbackPolicy` — vlm/fallback/review_queue 분기) ✅
- [ ] P1���P6 전체 파이프라인 통합 테스트 (VLM 서버 가동 후)

### Phase 2 (3~4개월, 06~07월) — Fine-tuning + 품질 개선

**2-A. PP-DocLayout Fine-tuning + 교정 데이터 축적 (병행)**
- [ ] 학습 데이터 수집 (군수 서식 영역 어노테이션 2,000~5,000장)
- [ ] PP-DocLayout Fine-tuning 실행
- [ ] P2 검출 품질 검증
- [x] 교정 데이터 → 학습 데이터 축적 파이프라인 구축 ✅
  - [x] `review_queue.export_training_pairs()` — SFT/DPO 형식 export 메서드 ✅
  - [x] `scripts/export_training_data.py` — 배치 변환 스크립트 (JSONL 출력) ✅
- [ ] 검토 큐 운영 → 교정 데이터 최소 수량 확보 (SFT 착수 전 마일스톤: 유형당 50건+)

**2-B. Gemma4 VLM Fine-tuning — 3단계** (2-A 교정 데이터 축적 후)
- [ ] **1단계 SFT** (필수): 서식 이미지 + 정답 JSON, LoRA, 유형당 200~500장
  - 학습 데이터 출처: ① 수동 어노테이션 + ② 검토 큐 교정 데이터 (`scripts/export_training_data.py`)
- [ ] 1단계 SFT 후 전체 파이프라인 성능 벤치마크
- [ ] **2단계 수기 강화** (선택): 수기 인식 오류율 높을 시 — AI Hub 손글씨 + 군수 수기 crop
- [ ] **3단계 DPO** (선택): 규칙 위반 빈번 시 — P4 검증 실패 출력을 Rejected로 자동 축적
  - DPO 데이터 출처: `scripts/export_training_data.py --format dpo`

### Phase 3 (5~6개월, 08~09월) — 최적화 + 배포

- [ ] 추론 속도 최적화 (vLLM 배치 튜닝)
- [ ] **Gemma4 양자화 실험** — BF16(현재) 대비 품질·속도·VRAM 비교
  - [ ] AWQ 4-bit (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) — 어텐션만 INT4, MoE 전문가 BF16 유지, ~16GB VRAM, vLLM v0.19.0 호환
  - [ ] FP8 Dynamic (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`) — 가중치+활성화 FP8, ~29GB VRAM, 품질 손실 ~0.3%
    - 선행 조건: `vllm-project/vllm#39049` (FP8 gibberish 버그) Close 확인
    - 확인 트리거: Phase 3 착수 시점(08월) + vLLM 마이너 릴리스마다
    - 판단 기준: issue Close + 릴리스 노트 반영 + 로컬 검증 통과
    - 미해소 시: AWQ 4-bit 우선 적용, FP8은 다음 분기로 연기
  - [ ] 양자화별 OCR 품질 벤치마크 (군수 서식 테스트셋 기준 필드 정확도, 신뢰도 분포 비교)
- [ ] 프론트엔드 UI 구현
- [ ] **수동 검토 큐 UI** (필수 — `docs/FRONTEND.md` §2-2 참조)
  - [ ] 큐 목록 화면 (우선순위 정렬, 필터, 대시보드)
  - [ ] 개별 검토 화면 (원본 이미지 + 추출 결과 비교, 필드 수정)
  - [ ] 검토 API 엔드포인트 (FastAPI)
  - [ ] 교정 데이터 → Fine-tuning 학습 데이터 자동 축적
- [ ] **서비스 통신 비동기 전환** (상세: `docs/BACKEND.md` §7-2)
  - [ ] 폐쇄망 적합 메시지 브로커 선정 (Redis 로컬 / SQLite-backed 큐)
  - [ ] pipeline → fallback: Redis Streams 비동기 전환
  - [ ] 검토 큐 알림: Redis Pub/Sub → 프론트엔드 SSE
- [ ] **서식 개정 대응 절차** (상세: `docs/AI_INFERENCE.md` §7)
  - [ ] 스키마 버전 업 → VLM 재학습 트리거 조건 정의
  - [ ] 구버전 DB 레코드 마이그레이션 정책
- [ ] Docker 프로덕션 설정
- [ ] 보안 검토 + 배포

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
| Gemma4 26B-A4B 추론 (FP8 Dynamic) | NVIDIA GPU | **~27GB** | Phase 3 양자화 실험 후 전환 검토 |
| PP-DocLayout 추론 | NVIDIA GPU | 4GB | PaddlePaddle |
| Real-ESRGAN SR | NVIDIA GPU | 2GB | 타일 기반 처리 |
| Gemma4 Fine-tuning | NVIDIA GPU | 24GB+ | LoRA 시 16GB |
