# AI 추론 (Inference) 가이드

> AI 모델을 사용한 추론 관련 설정, 모델 가중치, 동작 모드, Docker 환경을 기술합니다.
> **폐쇄망 환경 전용** — 모든 모델은 로컬 가중치를 사용하며, 외부 API/다운로드를 금지합니다.

---

## 1. 핵심 원칙

- 모든 모델 가중치는 `models/` 디렉토리에 **사전 배치**
- Gemma4 VLM은 **vLLM 로컬 서버**로 구동 (외부 통신 없음)
- 모델 로드 실패 시:
  - P2 (레이아웃): HEURISTIC fallback (OpenCV 규칙 기반)
  - P3 (VLM): **수준 B fallback 전환** → v1 PP-OCRv5 기반 경량 파이프라인 (§5 참조)

---

## 2. 모델 가중치 현황

| 모델 | 위치 | 크기 | 형식 | 출처 | 상태 |
|------|------|------|------|------|------|
| Real-ESRGAN x2plus | `models/t1_sr/RealESRGAN_x2plus.pth` | 64MB | PyTorch (.pth) | xinntao/Real-ESRGAN | v1에서 이관 예정 |
| PP-DocLayout_plus-L | `models/t2_layout/PP-DocLayout_plus-L/` | 124MB | PaddlePaddle | PaddleOCR 3.0 | v1에서 이관 예정 |
| Gemma4 26B-A4B | `models/gemma4/gemma-4-26b-a4b-it/` | ~16GB (Q4) | GGUF / SafeTensors | google/gemma-4-26b-a4b-it | ❌ 미배치 |

---

## 3. 컴포넌트별 추론 상세

### 3-1. P1 — 화질 보정 + SR (`src/preprocess/preprocessor.py`)

- v1 T1에서 이관
- LOW DPI (< 150) 시 Real-ESRGAN x2 초해상도 적용
- 이후 CLAHE 대비 강화 → 이진화 → 품질 점수 산출
- Input: `DocumentInput` → Output: `PreprocessedImage`

### 3-2. P2 — 구조 분석 (`src/preprocess/layout_analyzer.py`)

**역할**: VLM이 처리할 입력을 관리 가능한 단위로 분해하고, 처리 순서를 보장하는 **전처리 게이트**

- v1 T2에서 이관 — PP-DocLayout_plus-L
- Input: `PreprocessedImage` → Output: `LayoutResult`

**역할 1 — 페이지 분해 (Dense 페이지 문제 해결)**
페이지 전체를 VLM에 입력하면 다단 컬럼, 표·수식·텍스트 혼재 환경에서 long-sequence 디코딩 지연과 누락이 발생합니다.
구조 분석이 페이지를 영역별 bbox로 먼저 분리하면, VLM은 각 영역 crop 이미지만 보고 element-level 인식에만 집중할 수 있습니다.

```
페이지 전체 이미지
       ↓ PP-DocLayout
[table bbox]  [text bbox]  [formula bbox]  [seal bbox]
       ↓ 각 crop 이미지 + task prompt
VLM: "Table Recognition:" → 표 구조 인식
VLM: "OCR:"              → 텍스트 인식
```

**역할 2 — 읽기 순서 보장**
구조 분석이 영역 간 읽기 순서를 예측합니다. 다단 컬럼 문서에서 "1열 상단 → 1열 하단 → 2열 상단" 같은 올바른 순서로 VLM 입력이 구성되어야 최종 JSON 출력의 구조가 올바릅니다.

**역할 3 — Task Prompt 결정**
검출된 영역의 클래스 레이블이 VLM의 task prompt를 결정합니다:
```python
TASK_PROMPTS = {
    "text":    "OCR:",
    "table":   "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart":   "Chart Recognition:",
    "seal":    "Seal Recognition:",
}
# P2가 label을 주지 않으면 VLM은 어떤 태스크를 수행할지 모름
```

- 검출 영역 유형: text, table, figure, header, footer, seal, formula, chart
- Fine-tuning 후 군수 서식 영역(수기 기입란 등) 검출 가능

### 3-3. P3 — Gemma4 VLM 통합 추론 (`src/vlm/gemma4_engine.py`)

**모델**: Gemma4 26B-A4B (MoE, 활성 4B)

| 항목 | 값 |
|------|------|
| 총 파라미터 | 26B |
| 활성 파라미터 | 4B (MoE) |
| 컨텍스트 길이 | 256K 토큰 |
| 비전 입력 | 가변 해상도, 70~1120 토큰/이미지 |
| 한국어 지원 | ✅ (40+ 언어) |
| 라이선스 | Apache 2.0 |
| 추론 VRAM | 8~14GB (Q4 양자화 시 ~8GB) |

**서빙 방식**: vLLM 로컬 서버

```python
# vLLM 서버 시작 (오프라인, guided decoding + logprobs 활성화)
vllm serve models/gemma4/gemma-4-26b-a4b-it/ \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --trust-remote-code \
    --disable-log-requests \
    --guided-decoding-backend outlines
```

**Guided Decoding — JSON Schema 기반 출력 구조 보장**

군수 서식 유형별 JSON Schema를 사전 정의하여 `guided_json` 파라미터로 전달합니다.
VLM 출력이 항상 유효한 JSON 구조임을 보장하여 **파싱 실패를 원천 차단**합니다.

```python
# vLLM API 호출 예시
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{"role": "user", "content": [image, instruction]}],
    extra_body={
        "guided_json": supply_request_schema,   # 서식별 JSON Schema
        "logprobs": True,                       # 토큰별 확률 반환
        "top_logprobs": 5,                      # 상위 5개 후보
    },
)
```

JSON Schema 예시 (`src/domain/schemas/supply_request.json`):
```json
{
  "type": "object",
  "properties": {
    "form_type": {"type": "string", "enum": ["supply_request"]},
    "unit_code": {"type": "string"},
    "request_date": {"type": "string"},
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "nsn": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{3}-\\d{4}$"},
          "item_name": {"type": "string"},
          "quantity": {"type": "integer", "minimum": 1},
          "unit_price": {"type": "integer", "minimum": 0},
          "total": {"type": "integer", "minimum": 0}
        },
        "required": ["nsn", "item_name", "quantity", "unit_price", "total"]
      }
    }
  },
  "required": ["form_type", "unit_code", "items"]
}
```

**logprobs 기반 필드별 신뢰도 산출**

vLLM은 `logprobs` 옵션으로 각 출력 토큰의 로그 확률을 반환합니다.
이를 필드값 토큰에 대한 확신도로 환산합니다.

```python
# logprobs → 필드 신뢰도 환산 로직
def calc_field_confidence(token_logprobs: list[float]) -> float:
    """필드값 토큰들의 평균 확률을 신뢰도로 환산.
    
    logprob = log(prob) → prob = exp(logprob)
    필드 신뢰도 = 토큰 확률의 기하 평균
    """
    import math
    if not token_logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in token_logprobs]
    # 기하 평균 (곱의 n-th root)
    geo_mean = math.exp(sum(token_logprobs) / len(token_logprobs))
    return round(geo_mean, 4)
```

수치 필드(수량·단가·합계)는 각 숫자 토큰의 logprob을 개별 추적하여 **자릿수 단위 정밀 신뢰도**를 산출합니다.

**VLM instruction 태스크 — P2 영역 유형별 자동 분기**

| P2 영역 유형 | VLM instruction | guided_json | logprobs 활용 |
|-------------|----------------|------------|--------------|
| `text` | "한국어 텍스트를 인식하고 오타를 교정하세요" | text_schema | 글자 단위 신뢰도 |
| `table` | "표의 구조를 분석하고 각 셀 텍스트를 추출하세요" | table_schema | 셀 단위 신뢰도 |
| `seal` | "인장/직인의 텍스트를 인식하세요" | seal_schema | 문자 단위 |
| 전체 문서 | "서식 유형을 분류하고 모든 필드를 추출하세요" | **서식별 스키마** | **수치 필드: 자릿수 단위** |

**처리 흐름**

```
P2 LayoutResult
    │
    ├─ 1. 전체 문서 → VLM (서식 분류 instruction)
    │      → form_type 결정 → 해당 서식의 JSON Schema 로드
    │
    ├─ 2. 영역별 crop 생성
    │
    ├─ 3. 전체 문서 + 서식별 guided_json → VLM (필드 추출 instruction)
    │      → guided_json으로 구조 보장
    │      → logprobs로 필드별 토큰 확률 수집
    │
    ├─ 4. 표 영역 crop → VLM (표 인식 instruction)
    │      → HTML 표 구조 + 셀 텍스트
    │
    └─ 5. VLMResult 조립
         - fields: key-value 쌍 + logprobs 기반 confidence
         - tables: HTML 표 구조
         - form_type: 서식 분류
         - domain_codes: 특수 코드
         - corrections: 교정 로그
```

### 3-4. P4 — 룰 검증 + 신뢰도 보정 (`src/postprocess/validator.py`)

- VLM logprobs 신뢰도를 1차 신뢰도로 사용
- 룰 기반 교차검증으로 2차 보정:
  - 산술 검증: `합계 == 수량 × 단가` 불일치 시 해당 필드 신뢰도 감점
  - 날짜 검증: `청구일 <= 승인일` 위반 시 감점
  - 코드 검증: NSN 형식 불일치 시 감점
- `review_required` 결정: 보정된 신뢰도가 임계값 미달 필드 존재 시
- Input: `VLMResult` → Output: `ValidatedResult`

### 3-5. P5 — 직렬화 (`src/postprocess/serializer.py`)

- v1 T11에서 이관 — JSON, XML, CSV 출력

### 3-6. P6 — DB 적재 (`src/postprocess/db_loader.py`)

- v1 T12에서 이관 — SQLAlchemy + SQLite

---

## 4. SPOF 대비 — 추론 가용성

> 상세 설계는 `docs/BACKEND.md` §3 참조. 여기서는 추론 환경 관점의 설정만 기술합니다.

### 4-1. 수준 A — vLLM 헬스체크

vLLM은 `/health` 엔드포인트를 기본 제공합니다. Docker Compose에서 이를 활용합니다.

```yaml
# docker-compose.yml (vllm-server 서비스)
healthcheck:
  test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
  interval: 30s       # 30초마다 체크
  timeout: 10s        # 10초 내 응답 없으면 실패
  retries: 3          # 3회 연속 실패 시 unhealthy → Docker 재시작
  start_period: 120s  # 모델 로드 대기 (~60-90초)
restart: unless-stopped  # unhealthy 시 자동 재시작
```

**오케스트레이터 측 감시**: 파이프라인 서비스는 매 요청 전 + 주기적으로 `VLLM_HEALTH_URL`을 polling하여 vLLM 상태를 캐싱합니다. 상세 로직은 `docs/BACKEND.md` §3-1 참조.

### 4-2. 수준 B — 경량 Fallback 추론 환경

VLM이 완전히 불가할 때 v1의 PP-OCRv5 기반 T1~T5를 별도 컨테이너(`fallback`)로 구동합니다.

| 모델 | 위치 | 크기 | 프레임워크 | 역할 |
|------|------|------|-----------|------|
| DiT (서식 분류) | `models/fallback/t3_form_classifier/` | ~350MB | PyTorch | T3 서식 분류 |
| PP-OCRv5 Korean | `models/fallback/t4_handwriting/` | ~15MB | PaddlePaddle | T4 텍스트 인식 |
| SLANeXt_wired | `models/fallback/t5_table_structure/` | ~30MB | PaddlePaddle | T5 표 구조 인식 |

**Fallback vs VLM 비교**:

| 항목 | VLM (주 경로) | Fallback |
|------|-------------|----------|
| VRAM | 12~16GB | 2~4GB |
| 추론 속도 | ~3-5초/문서 | ~1-2초/문서 |
| 텍스트 교정 | VLM 문맥 교정 | ❌ 없음 |
| 서식 분류 | instruction 기반 | DiT 모델 기반 |
| 표 구조 | VLM HTML 출력 | SLANeXt 셀 검출 |
| guided_json | ✅ 구조 보장 | ❌ 규칙 기반 매핑 |
| logprobs 신뢰도 | ✅ 토큰 확률 | ❌ rec_score 사용 |
| 결과 품질 | 높음 | 중간 (항상 검토 큐 적재) |

**Fallback 컨테이너 구성**:
- Base: PaddlePaddle 3.2.0 GPU + PyTorch (v1 unified-inference와 동일)
- API: FastAPI 서버 (`:8081`) — 파이프라인에서 HTTP로 호출
- 엔드포인트: `POST /fallback/process` (이미지 입력 → T1~T5 결과 반환)
- 헬스체크: `GET /health`

---

## 5. Docker 추론 환경

| 서비스 | 용도 | 프레임워크 | 헬스체크 |
|--------|------|-----------|---------|
| `vllm-server` | Gemma4 vLLM 서버 (guided decoding + logprobs) | vLLM + PyTorch | `/health` (30s) |
| `pipeline` | P1~P6 파이프라인 (vLLM 서버에 API 호출) | PaddlePaddle + PyTorch | `/health` (30s) |
| `fallback` | v1 PP-OCRv5 기반 경량 파이프라인 (T1~T5) | PaddlePaddle + PyTorch | `/health` (30s) |
| `train` | Fine-tuning 환경 (profiles: training) | PyTorch + PaddlePaddle + PEFT | — |

**권장 구성 — 분리 모드**:
- `vllm-server`: Gemma4 모델 로드 1회 → 상주 서빙 (OpenAI 호환 API) + `restart: unless-stopped`
- `pipeline`: vLLM API endpoint(`http://vllm-server:8000`)로 추론 요청
- `fallback`: VLM 불가 시 자동 전환 대상 — 항상 대기 상태로 유지
- 장점: 모델 로드 오버헤드 제거, 파이프라인 재시작 시에도 모델 유지, SPOF 방지

---

## 6. 추론 파이프라인 스크립트

| 스크립트 | 범위 |
|----------|------|
| `scripts/run_pipeline.py` | P1→P7 전체 파이프라인 |
| `scripts/test_vlm.py` | P3 VLM 단독 테스트 |
| `scripts/run_pipeline_with_outputs.py` | 전체 파이프라인 + 단계별 출력 저장 |

---

## 7. 하드웨어 요구사항

| 용도 | 최소 GPU | 권장 GPU |
|------|---------|---------|
| Gemma4 추론 (Q4) | RTX 3060 12GB | RTX 4090 24GB |
| Gemma4 추론 (BF16) | RTX 3090 24GB | A100 40GB |
| PP-DocLayout 추론 | CPU 가능 | GPU 4GB |
| Real-ESRGAN SR | GPU 2GB | GPU 4GB |
| Fallback (T3+T4+T5) | GPU 2GB | GPU 4GB |
