# AI 추론 (Inference) 가이드

> AI 모델을 사용한 추론 관련 설정, 모델 가중치, 동작 모드, Docker 환경을 기술합니다.
> **폐쇄망 환경 전용** — 모든 모델은 로컬 가중치를 사용하며, 외부 API/다운로드를 금지합니다.

---

## 1. 핵심 원칙

- 모든 모델 가중치는 `models/` 디렉토리에 **사전 배치**
- Gemma4 VLM은 **vLLM 로컬 서버**로 구동 (외부 통신 없음)
- **Crop-then-Infer**: PP-DocLayout bbox 크롭 이미지를 영역별로 Gemma4에 전달 (전체 페이지 입력 지양)
- **FormClassifier 선행**: 서식 분류 후 InstructionRouter가 도메인 맥락 포함 instruction 생성
- **배치 병렬 처리**: 동일 pixel_budget 영역을 그룹화하여 vLLM 동시 전송
- 모델 로드 실패 시:
  - P2 (레이아웃): HEURISTIC fallback (OpenCV 규칙 기반)
  - P3 (VLM): **수준 B fallback 전환** → v1 PP-OCRv5 기반 경량 파이프라인

---

## 2. 모델 가중치 현황

| 모델 | 위치 | 크기 | 형식 | 상태 |
|------|------|------|------|------|
| Real-ESRGAN x2plus | `models/t1_sr/RealESRGAN_x2plus.pth` | 64MB | PyTorch | ✅ 배치 완료 (v1에서 이관) |
| PP-DocLayout_plus-L | `models/t2_layout/PP-DocLayout_plus-L/` | 126MB | PaddlePaddle | ✅ 배치 완료 (v1에서 이관, fusion 교차 사용) |
| PP-DocLayoutV3 | `models/t2_layout/PP-DocLayoutV3/` | 132MB | PaddlePaddle | ✅ 배치 완료 (기본 모델) |
| Gemma4 26B-A4B | `models/gemma4/gemma-4-26b-a4b-it/` | ~48GB (BF16) | SafeTensors | ✅ 배치 완료 (2026-04-08 검증) |

---

## 3. 컴포넌트별 추론 상세

### 3-1. P1 — 화질 보정 + SR

- v1 T1에서 이관
- LOW DPI (< 150) 시 Real-ESRGAN x2 초해상도 적용
- CLAHE 대비 강화 → 이진화 → 품질 점수 산출
- Input: `DocumentInput` → Output: `PreprocessedImage`

### 3-2. P2 — 레이아웃 탐지 (`src/preprocess/layout_analyzer.py`)

**역할**: VLM 입력을 위한 영역 bbox 탐지. 정제(LayoutPostProcessor)와 분리되어 원시 탐지 결과만 반환.

- PP-DocLayoutV3 (기본) / PP-DocLayout_plus-L (호환) — 교차 사용 가능
- Input: `PreprocessedImage` → Output: `RawLayoutResult`
- `TASK_PROMPTS` 딕셔너리 제거 완료 — InstructionRouter로 일원화
- 모델 선택: `PipelineConfig.layout_model_name`
- Layout 추론은 별도 컨테이너(`layout :8082`)에서 HTTP로 수행

**Fusion 모드 (구현 완료)**:

V3의 구조 검출 강점과 plus-L의 텍스트 검출 강점을 결합합니다.
`LAYOUT_FUSION_MODE=true` 환경변수로 활성화.

| P1 출력 DPI | fusion 전략 | 근거 |
|-------------|------------|------|
| **≥ 150** | V3(구조) + plus-L(텍스트), 50% 겹침 필터 | 고해상도에서 plus-L text 검출 양호 |
| **< 150** | V3(HTTP 구조) + heuristic(로컬 텍스트), 50% 겹침 필터 | 저해상도에서 AI 모델보다 OpenCV heuristic이 text 세밀 검출 |

- Reading order: V3 구조 영역(V3 model order 유지) → 텍스트(좌상→우하 정렬)
- fusion 서버 엔드포인트: `POST /layout/analyze` (`fusion_mode: true`)

**PP-DocLayout_plus-L vs PP-DocLayoutV3 비교**:

| 항목 | PP-DocLayout_plus-L | PP-DocLayoutV3 (기본) |
|------|---------------------|----------------------|
| 아키텍처 | RT-DETR | Mask RT-DETR |
| 출력 | bbox | bbox + polygon + reading order |
| 라벨 | 20종 | 25종 (V3 신규 6종 포함) |
| 읽기 순서 | 좌표 휴리스틱 | 모델 예측 (pairwise scoring) |
| 모델 크기 | ~126MB | ~132MB |

### 3-3. P2.5-A — LayoutPostProcessor (`src/preprocess/layout_postprocessor.py`)

**역할**: PP-DocLayout 원시 탐지 결과 정제. VLM이 의미 있는 크롭 이미지를 받도록 보장.

- 미소 박스 제거 (6px 미만 — PaddleOCR-VL 기준)
- 중복 박스 제거 (IoU > 0.7, seal 영역은 0.5)
- 인접 텍스트 블록 병합 (동일 컬럼 내)
- Input: `RawLayoutResult` → Output: `LayoutResult`

**이 단계가 없을 때 발생하는 문제**:
- 6px 점 이미지를 VLM에 전달 → 무의미한 호출 낭비
- 인장+텍스트 중복 bbox → 동일 영역 이중 처리
- 단어 단위 분절 텍스트 → 문맥 없는 조각 인식

### 3-4. P3-A — FormClassifier (`src/vlm/form_classifier.py`)

**역할**: 서식 분류 전용 단일 VLM 호출. InstructionRouter의 도메인 맥락 생성을 위한 필수 선행 단계.

| 항목 | 값 |
|------|------|
| pixel_budget | 140 토큰 (저해상도 — 분류 비용 최소화) |
| 입력 | 전체 페이지 이미지 |
| 출력 | form_type, form_confidence |
| 호출 방식 | 단일 동기 호출 (배치 불필요) |

```python
# vLLM 호출 예시 (FormClassifier)
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": full_page_base64}},
            {"type": "text", "text": "이 문서의 서식 유형을 분류하세요."}
        ]
    }],
    extra_body={
        "guided_json": FORM_TYPE_SCHEMA,   # enum: supply_request | maintenance_record | ...
        "mm_processor_kwargs": {"max_pixels": 140 * 28 * 28},  # 140 토큰
    },
)
```

### 3-5. P2.5-B — InstructionRouter (`src/vlm/instruction_router.py`)

**역할**: region_type + form_type → 도메인 맥락 포함 InstructionSpec 생성.

PaddleOCR-VL의 `"OCR:"` 2단어 프리픽스는 전용 학습 모델 기반이므로 범용 VLM에 그대로 적용 불가. Gemma4는 도메인 맥락, 출력 형식, 불확실성 처리까지 포함한 상세 instruction이 필요.

```python
# 기존 (instruction_builder.py — Legacy 래퍼로 유지, 신규 코드에서 사용 금지)
# TASK_PROMPTS = {"text": "OCR:", "table": "Table Recognition:", ...}

# 현재 (instruction_router.py)
# form_type="supply_request", region_type="text" →
instruction = InstructionSpec(
    system_prompt="군수 보급청구서 필드 추출 시스템입니다.",
    user_instruction=(
        "이 수기 기입란에서 한국어 텍스트를 인식하세요. "
        "NSN 코드(NNNN-NN-NNN-NNNN) 형식이 있다면 정확히 추출하고, "
        "불확실한 글자는 [?]로 표시하세요."
    ),
    json_schema=supply_request_field_schema,
    pixel_budget=560,  # ResolutionRouter에서 결정
)
```

### 3-6. P2.5-C — ResolutionRouter (`src/vlm/resolution_router.py`)

**역할**: 영역 타입별 pixel_budget 할당 + bbox 크롭 + 배치 그룹화.

```python
PIXEL_BUDGET = {
    "table":             1120,   # 셀 경계·미세 글씨 → 최고 해상도
    "seal":               560,   # 원형 배치 텍스트
    "handwritten_field":  560,   # 군수 서식 수기 기입란 (핵심)
    "text":               280,   # 일반 텍스트
    "formula":            280,
    "chart":              280,
    "figure":             140,
    "header":             140,   # 저해상도로 충분
    "footer":             140,
}
```

**배치 그룹화 근거**: 동일 pixel_budget 영역끼리 묶어야 vLLM 내부 패딩 오버헤드가 없음. 해상도가 다른 이미지를 섞으면 최대 해상도에 맞춰 패딩 → VRAM 낭비 및 처리 지연.

### 3-7. P3-B — StructuredExtractor (`src/vlm/structured_extractor.py`)

**역할**: pixel_budget 기준 배치 그룹을 Gemma4에 병렬 전송하여 구조화 추출 수행.

```python
# pixel_budget=1120 그룹 배치 호출 예시
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": crop_base64}},
                {"type": "text", "text": instruction_spec.user_instruction}
            ]
        }
        for crop in batch  # 동일 budget 그룹 배치
    ],
    extra_body={
        "guided_json": instruction_spec.json_schema,
        "logprobs": True,
        "top_logprobs": 5,
        "mm_processor_kwargs": {"max_pixels": pixel_budget * 28 * 28},
    },
)
```

**영역별 trace 수집 (옵션)**:

`extract()`에 `trace: Optional[list[dict]] = None` 매개변수를 전달하면, 각 vLLM 호출별 메타데이터가 dict로 누적됩니다. 디버깅·통합 테스트 검증·검토 큐 시각화에 사용합니다.

```python
extractor = StructuredExtractor(cfg)
trace: list[dict] = []
result = extractor.extract(
    groups=groups,
    doc_id=doc_id,
    form_type=form_type,
    form_confidence=form_confidence,
    schema_id=schema_id,
    schema=schema,
    warnings=warnings,
    trace=trace,            # ← 여기서 영역별 메타데이터를 수집
)
# trace[i] 예시:
# {
#   "region_id": "r_0004", "region_type": "table",
#   "pixel_budget": 1120, "elapsed_ms": 3951.9, "status": "ok",
#   "field_count": 0, "table_count": 1, "domain_code_count": 0,
#   "instruction": "...", "system_prompt": "...",
#   "raw_response": "...", "fields": [...],
# }
```

오케스트레이터는 항상 `result.p3b_trace` 빈 리스트를 전달하여 trace를 수집하고, 통합 테스트 스크립트가 `data/pipeline_outputs/<ts>/<doc>/P3/region_traces.json`으로 직렬화합니다(BACKEND.md §11-2 참조).

**logprobs 기반 필드별 신뢰도 산출 (길이 편향 보정 포함)**:

```python
def calc_field_confidence(
    token_logprobs: list[float],
    field_type: str = "text",
) -> float:
    geo_mean = exp(mean(token_logprobs))
    # 긴 필드(부대명 등)는 짧은 필드 대비 구조적으로 낮은 점수를 받으므로
    # 필드 유형별 기대 토큰 수 기반 보정 계수 적용
    length_factor = min(1.0, sqrt(EXPECTED_TOKENS[field_type]
                                  / len(token_logprobs)))
    return round(geo_mean * length_factor, 4)
```

```python
EXPECTED_TOKENS = {
    "amount": 4, "code": 5, "date": 3, "quantity": 2,
    "number": 3, "text": 5, "signature": 2,
}
```

### 3-8. P4 — 룰 검증 + 신뢰도 보정 (`src/postprocess/validator.py`)

- `processing_path` 기반으로 임계값 프로파일 분기 (VLM / Fallback)
- Fallback 경로는 임계값을 낮게 설정하되 `review_required = True` 강제
- 룰 기반 교차검증 후 감점 적용

### 3-9. P5 — 직렬화 / P6 — DB 적재

- v1 T11/T12에서 이관 — JSON, XML, CSV 출력
- DB 레코드에 `schema_id` (버전 포함) 함께 기록
- **`VLMResult.raw_json`은 DB 적재 대상에서 제외** — guided_json으로 구조가 보장되므로 민감정보 이중 저장 방지. 디버깅 시에만 `data/pipeline_outputs/` 단계별 출력에서 확인

---

## 4. Gemma4 VLM 서빙 설정

| 항목 | 값 |
|------|------|
| 총 파라미터 | 26B |
| 활성 파라미터 | 4B (MoE) |
| 컨텍스트 길이 | 256K 토큰 |
| 비전 입력 | 가변 해상도 70~1120 토큰/이미지 |
| 한국어 지원 | ✅ (40+ 언어) |
| 라이선스 | Apache 2.0 |
| 추론 VRAM (BF16) | ~48GB |

```bash
# vLLM v0.19.0+ 서버 시작
vllm serve models/gemma4/gemma-4-26b-a4b-it/ \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --disable-log-requests \
    --gpu-memory-utilization 0.90
```

> **운영 환경 (2026-04-08 검증)**: vLLM v0.19.0 / H100 80GB / BF16 / 모델 로드 48.5 GiB / 기동 ~120초

---

## 5. SPOF 대비 — 추론 가용성

### 5-1. 수준 A — vLLM 헬스체크

```yaml
healthcheck:
  test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 120s
restart: unless-stopped
```

### 5-2. 수준 B — 경량 Fallback 추론 환경 (HTTP 컨테이너 분리 모드 전제)

| 모델 | 위치 | 크기 | 역할 | 상태 |
|------|------|------|------|------|
| DiT (서식 분류) | `models/fallback/t3_form_classifier/` | ~350MB | T3 서식 분류 | ⚠️ 가중치 미배치 |
| PP-OCRv5 Korean | `models/fallback/t4_handwriting/` | ~15MB | T4 텍스트 인식 | ⚠️ 가중치 미배치 |
| SLANeXt_wired | `models/fallback/t5_table_structure/` | ~30MB | T5 표 구조 인식 | ⚠️ 가중치 미배치 |

**Fallback vs VLM (고도화 후)**:

| 항목 | VLM 주 경로 (고도화) | Fallback |
|------|---------------------|----------|
| 서식 분류 | P3-A FormClassifier (저해상도) | DiT 모델 |
| 텍스트 인식 | P3-B 영역별 크롭 고해상도 | PP-OCRv5 전체 페이지 |
| 표 구조 | P3-B 표 전용 1120 토큰 | SLANeXt |
| guided_json | ✅ 구조 보장 | ❌ 규칙 기반 매핑 |
| 처리 경로 표시 | `processing_path="vlm"` | `processing_path="fallback"` |
| P4 임계값 | VLM 프로파일 | Fallback 프로파일 (낮음) |
| 결과 품질 | 높음 | 중간 (항상 검토 큐) |

**컨테이너 분리 필수 — in-process 사용 금지**:

`OCRFallbackService` (in-process)는 v1 소스(`mil_OCR`)를 sys.path에 추가하여 T3/T4/T5 모듈을 import하는데, v1과 v2가 동일한 `src.interfaces.enums` 네임스페이스를 사용하면서 v1 전용 enum (`ClassificationMode`, `Orientation`, `ContentType`)이 v2 enums에서 제거되어 import 오류 + 메모리 폭주를 유발합니다.

따라서 fallback은 **반드시 별도 컨테이너에서 기동**해야 합니다:

```bash
docker compose up -d fallback
# fallback 컨테이너가 8081 포트에서 src/fallback/server.py(FastAPI) 기동
```

오케스트레이터는 `fallback_base_url` 환경변수가 설정되어 있으면 `FallbackHTTPClient`를 사용하고, 미설정 시 `_DisabledFallback` 더미를 반환합니다(BACKEND.md §4-2 참조). 더미 상태에서는 VLM 불가 시 검토 큐만 활성화됩니다.

**Fallback HTTP 서버 엔드포인트** ([src/fallback/server.py](../src/fallback/server.py)):

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/health` | 서비스 상태 (`t3_loaded`, `t4_loaded`, `t5_loaded`) |
| `POST` | `/fallback/process` | `{doc_id, image_b64, dpi, regions, reading_order}` → `VLMResult` 호환 응답 |

---

## 6. Docker 추론 환경

| 서비스 | 포트 | 용도 | 엔트리포인트 | 프레임워크 | 헬스체크 |
|--------|------|------|--------------|-----------|---------|
| `vllm-server` | 8100→8000 | Gemma4 vLLM 서버 | `vllm serve` | vLLM v0.19.0 + transformers 5.5.0 (CUDA 12.x 내장) | `/health` (30s) |
| `layout` | 8082 | P2 레이아웃 탐지 (PP-DocLayoutV3 / plus-L, fusion 지원) | `uvicorn src.preprocess.layout_server:app` | PaddlePaddle 3.2 CUDA 12.6 | `/health` (30s) |
| `pipeline` | 8080 | P1, P2.5-A~C, P3-A~B, P4~P6 오케스트레이션 | `python -m src.pipeline.server` | PyTorch CUDA 12.6 + openai SDK + httpx (PaddleOCR 미포함, layout HTTP 호출) | `/health` (30s) |
| `fallback` | 8081 | v1 PP-OCRv5 기반 경량 파이프라인 | `uvicorn src.fallback.server:app` | PaddlePaddle + PyTorch CUDA 12.6 | `/health` (30s) |
| `train` | — | Fine-tuning 환경 | (수동 실행) | PaddlePaddle + PyTorch CUDA 12.6 + PEFT | — |

> **CUDA 11.8 → 12.6 업그레이드 이유**: 기존 PaddlePaddle CUDA 11.8 빌드는 H100 (sm_90) 커널이 누락되어 `CUDA error 209: no kernel image is available for execution on the device`가 발생합니다. 모든 GPU 컨테이너는 CUDA 12.6 베이스 이미지(`paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5`)를 사용해야 합니다.

**Pipeline HTTP 서버 엔드포인트** ([src/pipeline/server.py](../src/pipeline/server.py)):

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/health` | 서비스 상태 (`vlm_healthy`, `layout_service`, `fallback_enabled`) |
| `POST` | `/pipeline/run` | JSON 바디 (`doc_id`, `image_b64`, `file_ext`, `source_type`, `dpi_hint`, `metadata`) → `RunResponse` (status, processing_path, fields[], timings, errors, warnings) |
| `POST` | `/pipeline/upload` | 멀티파트 파일 업로드 → `RunResponse` |

> **CUDA 11.8 → 12.6 전환**: H100(sm_90) GPU에서 CUDA 11.8 PaddlePaddle 빌드의 커널 미포함(error 209) 문제로 pipeline/fallback/train 모두 CUDA 12.6으로 전환 완료.

---

## 7. 추론 파이프라인 스크립트

| 스크립트 | 범위 |
|----------|------|
| `scripts/run_pipeline.py` | P1→P6 전체 파이프라인 |
| `scripts/test_vlm.py` | P3-A/P3-B VLM 단독 테스트 |
| `scripts/run_pipeline_with_outputs.py` | 전체 파이프라인 + 단계별 출력 저장 |
| `scripts/evaluate_layout_detection.py` | PP-DocLayout 검출률 측정 (Phase 1 잔여) |

---

## 8. 하드웨어 요구사항

| 용도 | VRAM | 권장 GPU | 비고 |
|------|------|---------|------|
| Gemma4 추론 (BF16) | ~48GB | H100 80GB | 현재 운영 |
| Gemma4 추론 (AWQ 4-bit) | ~16GB | RTX 4090 24GB | Phase 3 실험 예정 |
| Gemma4 추론 (FP8 Dynamic) | ~27GB | A100 40GB | vllm#39049 해소 후 |
| PP-DocLayout 추론 | 4GB | GPU 4GB+ | PaddlePaddle |
| Real-ESRGAN SR | 2GB | GPU 4GB+ | 타일 기반 |
| Fallback (T3+T4+T5) | 2~4GB | GPU 4GB+ | v1 PP-OCRv5 기반 |

### 양자화 옵션 (Phase 3 실험 계획)

| 양자화 | VRAM | 품질 손실 | vLLM 호환 |
|--------|------|----------|----------|
| BF16 (현재) | ~48GB | 기준 | v0.19.0 ✅ |
| AWQ 4-bit | ~16GB | ~1-2% | v0.19.0 ✅ |
| FP8 Dynamic | ~27GB | ~0.3% | vllm#39049 Open → 해소 후 |

### Upstream 버그 트래킹

| 이슈 | 영향 | 상태 | 확인 트리거 | 미해소 시 대안 |
|------|------|------|-----------|--------------|
| `vllm-project/vllm#39049` (FP8 gibberish) | FP8 실험 차단 | Open (2026-04) | Phase 3 착수 + vLLM 릴리스마다 | AWQ 4-bit 우선 |
