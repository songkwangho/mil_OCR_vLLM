# 백엔드 가이드

> 파이프라인 오케스트레이션, 인터페이스 타입, 후처리 로직, 도메인 지식을 기술합니다.

---

## 1. 파이프라인 구조 (6단계)

```
DocumentInput
    │
    ▼
[Stage 1 — 전처리]
    P1: 화질 보정 + SR       → PreprocessedImage
    P2: 구조 분석             → LayoutResult
        ① 페이지 분해: Dense 페이지를 영역 bbox로 분리
        ② 읽기 순서: 영역 간 reading_order 결정
        ③ Task Prompt 결정: 영역 레이블 → VLM instruction 자동 선택
    │
    ▼
[Stage 2 — VLM 통합 추론]
    P3: Gemma4 VLM           → VLMResult
        P2의 reading_order 순서대로 영역별 처리:
        ① 서식 분류 → form_type 결정 → JSON Schema 로드
        ② 영역별 crop + P2 레이블 → task prompt + guided_json → 필드 추출
        ③ logprobs → 필드별 토큰 확률 → 신뢰도 산출
    │
    ▼
[Stage 3 — 후처리]
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult
        ├─ 통과 (confidence ≥ 임계값, CRITICAL 없음)
        │      → P5 직렬화 → P6 DB 적재 → PipelineOutput
        └─ 실패 (CRITICAL 오류 or LOW confidence)
               → 검토 큐 적재 → 담당자 UI 확인 → 수정 후 P6 재적재

[Fallback 경로 — VLM 불가 시]
    P1 → P2 → Fallback(T3~T5) → P4 검증 → 검토 큐 (항상)
```

---

## 2. 인터페이스 데이터 타입

> 파일 위치: `src/interfaces/types.py`, `src/interfaces/enums.py`

### 2-1. 입력 / 전처리

```python
@dataclass
class DocumentInput:
    doc_id: str
    raw_bytes: bytes
    file_ext: FileExt          # jpg, png, tiff, pdf
    source_type: SourceType    # scan, camera, fax, pdf_export
    dpi_hint: Optional[int] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class PreprocessedImage:
    doc_id: str
    image_array: np.ndarray    # H×W×3 uint8 (RGB)
    binary_array: np.ndarray   # H×W uint8
    dpi: int
    original_dpi: int
    resolution_band: DpiResolutionBand
    quality_score: float
    sr_applied: bool = False
    warnings: list[str] = field(default_factory=list)
```

### 2-2. 레이아웃 분석

```python
@dataclass
class BoundingBox:
    x1: int; y1: int; x2: int; y2: int

@dataclass
class LayoutRegion:
    region_id: str             # r_0001, r_0002, ...
    region_type: str           # text, table, figure, header, footer, seal
    bbox: BoundingBox
    confidence: float

@dataclass
class LayoutResult:
    doc_id: str
    page_width: int
    page_height: int
    regions: list[LayoutRegion]
    reading_order: list[int]   # regions 인덱스 순서 (다단 컬럼 대응)
    analysis_mode: str         # model / heuristic
    warnings: list[str] = field(default_factory=list)

# P2가 결정하는 영역 레이블 → VLM task prompt 매핑
TASK_PROMPTS = {
    "text":    "OCR:",
    "table":   "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart":   "Chart Recognition:",
    "seal":    "Seal Recognition:",
    "header":  "OCR:",
    "footer":  "OCR:",
    "figure":  "Image Description:",
}
```

### 2-3. VLM 통합 추론 결과

```python
@dataclass
class FieldValue:
    """VLM이 추출한 개별 필드 (guided_json 출력)."""
    field_key: str             # JSON Schema의 property 이름
    raw_value: str             # VLM 출력 원본
    corrected_value: str       # VLM 교정 결과 (동일하면 교정 없음)
    data_type: str             # text, number, date, code
    confidence: float          # logprobs 기반 토큰 확률 → 신뢰도
    token_logprobs: list[float]  # 해당 필드 토큰들의 개별 logprob
    is_flagged: bool = False   # 신뢰도 < 임계값

@dataclass
class RecognizedTable:
    region_id: str
    html: str                  # 표 구조 HTML
    cells: list[dict]          # [{"row": 0, "col": 0, "text": "품목"}]
    confidence: float

@dataclass
class DomainCode:
    code_type: str             # nsn, k_nsn, unit_code, date, rank
    raw_value: str
    normalized_value: str
    confidence: float

@dataclass
class VLMResult:
    doc_id: str
    form_type: str             # supply_request, maintenance_record, ...
    form_confidence: float
    schema_id: str             # guided_json에 사용된 스키마 ID
    fields: list[FieldValue]   # guided_json으로 추출된 키-값 쌍 + logprobs 신뢰도
    tables: list[RecognizedTable]
    domain_codes: list[DomainCode]
    raw_json: str              # VLM 원본 JSON 응답
    processing_time_ms: float
    warnings: list[str] = field(default_factory=list)
```

### 2-4. 후처리 결과

```python
@dataclass
class ValidationError:
    error_id: str
    error_type: str            # arithmetic, date_logic, code_format
    severity: str              # critical, high, medium, low
    field_ref: str             # 관련 field_key
    expected: str
    actual: str
    message: str

@dataclass
class ValidatedResult:
    """P4 출력 — 룰 검증 + 신뢰도 보정 결과."""
    doc_id: str
    fields: list[FieldValue]   # 보정된 신뢰도 반영
    validation_errors: list[ValidationError]
    overall_confidence: float  # 보정 후 전체 신뢰도
    review_required: bool
    flagged_fields: list[str]

@dataclass
class PipelineOutput:
    doc_id: str
    status: PipelineStatus     # success, partial, review, failed
    json_output: Optional[str]
    xml_output: Optional[str]
    csv_rows: list[dict]
    db_record_ids: list[str]
    review_queue_id: Optional[str]
    processing_ms: float
```

---

## 3. SPOF 대비 — 가용성 설계

> Gemma4 VLM(vLLM 서버)이 단일 장애점(SPOF)이 되는 것을 방지하기 위한 2단계 대비 전략.

### 3-1. 수준 A — vLLM 자체 안정성 확보

**목표**: vLLM 프로세스가 비정상일 때 자동 감지 + 자동 복구.

| 메커니즘 | 구현 방식 | 설정 |
|----------|----------|------|
| Docker 자동 재시작 | `restart: unless-stopped` | `docker-compose.yml` |
| 헬스체크 | `curl -sf http://localhost:8000/health` | interval 30s, timeout 10s, retries 3, start_period 120s |
| 오케스트레이터 감시 | 매 요청 전 `/health` 확인 + 주기적 백그라운드 polling | `VLLM_HEALTH_CHECK_INTERVAL=30` |

**오케스트레이터 헬스체크 로직**:

```python
class VLMHealthMonitor:
    """vLLM 서버 상태를 주기적으로 확인."""

    def __init__(self, health_url: str, interval: int = 30, timeout: int = 10):
        self.health_url = health_url
        self.interval = interval
        self.timeout = timeout
        self._healthy = True
        self._last_check: float = 0.0
        self._consecutive_failures: int = 0

    def is_healthy(self) -> bool:
        """캐시된 상태 반환. interval 경과 시 실제 체크."""
        now = time.time()
        if now - self._last_check >= self.interval:
            self._check()
        return self._healthy

    def _check(self):
        try:
            resp = httpx.get(self.health_url, timeout=self.timeout)
            if resp.status_code == 200:
                self._healthy = True
                self._consecutive_failures = 0
            else:
                self._on_failure()
        except (httpx.ConnectError, httpx.TimeoutException):
            self._on_failure()
        self._last_check = time.time()

    def _on_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._healthy = False
            logger.error("vLLM 서버 비정상 — fallback 전환 대기")
```

**복구 흐름**:
1. Docker healthcheck 실패 3회 → Docker가 컨테이너 자동 재시작
2. 재시작 중(~60-90초) 오케스트레이터는 `_healthy = False` 상태
3. 이 기간의 요청은 수준 B(fallback) 또는 검토 큐로 전환
4. 복구 후 자동으로 주 경로 재개

### 3-2. 수준 B — 경량 Fallback (v1 PP-OCRv5 기반)

**목표**: VLM이 완전히 불가할 때 기초 OCR 파이프라인으로 최소 서비스 유지.

**Fallback 서비스 구성** (별도 Docker 컨테이너 `fallback`):

| v1 컴포넌트 | fallback 기능 | 정확도 | 비고 |
|------------|-------------|--------|------|
| T1 화질 보정 | 이미지 전처리 + SR | v2 P1과 동일 | 공유 |
| T2 레이아웃 | PP-DocLayout 영역 검출 | v2 P2와 동일 | 공유 |
| T3 서식 분류 | DiT 기반 분류 | 중간 | 주 경로 대비 정확도 ↓ |
| T4 수기 인식 | PP-OCRv5 텍스트 인식 | 중간 | VLM 대비 교정 능력 ↓ |
| T5 구조 인식 | SLANeXt 표 구조 | 중간 | VLM 대비 정밀도 ↓ |

**Fallback 모델 가중치** (`models/fallback/`):
```
models/fallback/
├── t3_form_classifier/dit-base-finetuned-rvlcdip/
├── t4_handwriting/korean_PP-OCRv5_mobile_rec/
└── t5_table_structure/SLANeXt_wired/
```

**전환 정책**:

```python
class FallbackPolicy:
    """VLM 불가 시 fallback 전환 정책."""

    FALLBACK_MODES = Literal["auto", "manual_queue", "disabled"]

    def decide(
        self,
        vlm_healthy: bool,
        fallback_enabled: bool,
        fallback_healthy: bool,
    ) -> str:
        """처리 경로 결정.

        Returns:
            "vlm"          — 주 경로 (VLM 정상)
            "fallback"     — 경량 fallback (VLM 불가, fallback 가용)
            "review_queue"  — 수동 검토 큐 (둘 다 불가)
        """
        if vlm_healthy:
            return "vlm"
        if fallback_enabled and fallback_healthy:
            return "fallback"
        return "review_queue"
```

**Fallback 출력 특성**:
- `PipelineOutput.status = "partial"` (VLM 대비 불완전)
- `PipelineOutput.review_required = True` (항상 검토 필요 표시)
- guided_json / logprobs 신뢰도 없음 → 필드 신뢰도 = PP-OCRv5 rec_score 사용
- 서식 분류 정확도 저하 → `_fallback.json` 스키마 사용 빈도 증가

### 3-3. 수동 검토 큐

**목표**: 자동 처리 불가/불완전 문서를 담당자가 확인·교정할 수 있는 큐 시스템.

**검토 큐 적재 조건** (P4 검증 단계에서 판정):

| 조건 | 트리거 | 우선순위 |
|------|--------|---------|
| CRITICAL ValidationError | `severity == "critical"` | 🔴 긴급 |
| LOW confidence 필드 존재 | `field.confidence < 임계값` (§4-2 참조) | 🟡 일반 |
| Fallback 경로 처리 문서 | `PipelineOutput.status == "partial"` | 🟡 일반 |
| VLM + Fallback 모두 불가 | 원본 이미지만 큐에 적재 | 🔴 긴급 |

**처리 흐름**:

```
[자동 처리 경로 — 주 경로]
DocumentInput → P1 → P2 → P3(VLM) → P4 검증 통과 → P5 직렬화 → P6 DB 적재
                                          ✅ overall_confidence ≥ 임계값
                                          ✅ CRITICAL 오류 없음

[자동 처리 경로 — Fallback]
DocumentInput → P1 → P2 → Fallback(T3~T5) → P4 검증 → 검토 큐 적재 (항상)
                                                        → 담당자 확인 후 P6 DB 적재

[수동 검토 경로]
DocumentInput → P1 → P2 → P3(VLM) → P4 검증 실패
                                          ❌ CRITICAL 오류 or LOW confidence
                                          → 검토 큐 적재
                                          → 담당자 UI 확인
                                          → 수정 후 P6 DB 재적재

[완전 장애 경로]
DocumentInput → VLM 불가 + Fallback 불가 → 원본 이미지 + 메타데이터 큐 적재
                                          → 담당자 수동 처리
```

**검토 큐 데이터 모델**:

```python
@dataclass
class ReviewQueueItem:
    """수동 검토 큐의 개별 항목."""
    queue_id: str                    # RQ-20260407-001
    doc_id: str                      # 원본 문서 ID
    enqueued_at: datetime            # 큐 적재 시각
    priority: str                    # critical, normal
    reason: str                      # validation_failed, low_confidence, fallback, total_failure
    source_path: str                 # 처리 경로 (vlm, fallback, none)

    # 자동 처리 결과 (있는 경우)
    validated_result: Optional[ValidatedResult]   # P4 출력 (부분 결과)
    validation_errors: list[ValidationError]       # CRITICAL 오류 목록
    flagged_fields: list[str]                      # LOW confidence 필드 키 목록

    # 원본 참조
    original_image_path: str         # 원본 이미지 경로
    preprocessed_image_path: str     # P1 전처리 이미지 경로

    # 검토 상태
    status: str                      # pending, in_review, approved, rejected
    reviewer: Optional[str]          # 검토 담당자
    reviewed_at: Optional[datetime]
    corrected_fields: dict           # 담당자가 수정한 필드 {field_key: corrected_value}
    reviewer_notes: str = ""

@dataclass
class ReviewQueueStats:
    """검토 큐 현황 통계."""
    total_pending: int
    total_in_review: int
    critical_count: int
    normal_count: int
    avg_wait_minutes: float
```

**검토 완료 후 처리**:
1. 담당자가 UI에서 필드 수정 → `corrected_fields` 저장
2. `status = "approved"` → P5 직렬화 → P6 DB 적재 (수정된 값 사용)
3. 교정 데이터(원본 → 수정)를 VLM Fine-tuning 학습 데이터로 자동 축적
4. `status = "rejected"` → 문서 폐기 또는 재스캔 요청

---

## 4. Guided Decoding — 서식별 JSON Schema

### 3-1. 스키마 관리

```
src/domain/schemas/
├── supply_request.json       ← 보급청구서
├── maintenance_record.json   ← 정비기록서
├── inventory_sheet.json      ← 물자현황표
├── handover_doc.json         ← 인수인계서
├── inspection_report.json    ← 검사보고서
└── _fallback.json            ← 미분류 문서용 범용 스키마
```

### 3-2. 처리 흐름

1. P3가 먼저 **서식 분류** instruction을 실행 → `form_type` 결정
2. `form_type` → 해당 JSON Schema 파일 로드
3. 전체 문서 이미지 + 서식별 Schema를 `guided_json`으로 vLLM에 전달
4. VLM은 Schema에 맞는 JSON만 생성 → **파싱 실패 원천 차단**
5. 미분류(`unknown`) 시 `_fallback.json` 사용 (범용 key-value 추출)

### 3-3. Fallback 전략

| 상황 | 처리 |
|------|------|
| VLM이 서식 유형을 `unknown`으로 분류 | `_fallback.json` 스키마 사용 |
| guided_json 에도 불구하고 빈 응답 | 영역별 OCR로 fallback (텍스트만 추출) |
| vLLM 서버 무응답 | **수준 B fallback 전환** (§3-2) → fallback도 불가 시 검토 큐 적재 (§3-3) |

---

## 5. logprobs 신뢰도 산출

### 5-1. 필드별 신뢰도

```python
# VLM 응답에서 필드값에 해당하는 토큰들의 logprob 추출
# 예: "quantity": 50 → 토큰 "50"의 logprob = -0.02 → prob = 0.98

field_confidence = geometric_mean([exp(lp) for lp in field_token_logprobs])
```

### 5-2. 필드 유형별 임계값

| 필드 유형 | 임계값 | 미달 시 |
|----------|--------|---------|
| 금액 (amount) | 0.99 | is_flagged + review |
| 코드 (code/NSN) | 0.97 | is_flagged + review |
| 날짜 (date) | 0.95 | is_flagged |
| 수량 (quantity) | 0.95 | is_flagged |
| 일반 텍스트 | 0.90 | is_flagged |
| 서명 | 0.70 | is_flagged |

### 5-3. P4 룰 검증으로 보정

logprobs 신뢰도가 높아도 룰 검증에 실패하면 감점:
- `합계 ≠ 수량 × 단가` → 관련 필드 신뢰도 **-0.30**
- NSN 형식 불일치 → 해당 필드 신뢰도 **-0.15**
- 날짜 순서 위반 → 해당 필드 신뢰도 **-0.10**

---

## 6. 군수 도메인 지식

### 6-1. 주요 서식 유형

| 서식명 | schema_id | JSON Schema | 주요 필드 |
|--------|-----------|------------|----------|
| 보급청구서 | `supply_request` | `supply_request.json` | unit_code, nsn, quantity, unit_price, total, request_date |
| 정비기록서 | `maintenance_record` | `maintenance_record.json` | equipment_id, maintenance_type, date, technician |
| 물자현황표 | `inventory_sheet` | `inventory_sheet.json` | item_name, stock_qty, location, disposal_qty |
| 인수인계서 | `handover_doc` | `handover_doc.json` | from_person, to_person, date, items |

### 6-2. 코드 체계

```
NSN (NATO Stock Number):  NNNN-NN-NNN-NNNN  (예: 1005-01-432-1234)
K-NSN (한국 물자코드):    KN-NNNNN-NNNN
부대 코드:                숫자 2~4자리 + 한글 부대명 (예: "1232부대")
```

---

## 7. 서비스 통신 아키텍처

### 7-1. 현재 구조 (Phase 1)

```
[Pipeline 컨테이너]
  ├── P1 Preprocessor ─── (in-process)
  ├── P2 LayoutAnalyzer ── (in-process, GPU)
  ├── P3 Gemma4Engine ──── HTTP POST → [vLLM 컨테이너 :8100]
  │                         └── /v1/chat/completions (동기 blocking, 120s timeout)
  ├── Fallback ──────────── HTTP POST → [Fallback 컨테이너 :8081]
  │                         └── /fallback/process (동기)
  ├── P4 Validator ──────── (in-process)
  ├── P5 Serializer ─────── (in-process)
  └── P6 DBLoader ─────── (in-process, SQLite)
```

**통신 방식**: 동기 HTTP REST
- vLLM: OpenAI 호환 API (httpx/openai SDK)
- Fallback: FastAPI REST (httpx)
- 헬스체크: HTTP GET polling (`/health`, 30초 간격)
- 검토 큐 / DB: SQLite 직접 접근

### 7-2. Phase 3 목표: 비동기 메시지 기반

폐쇄망 적합 메시지 브로커(**Redis 로컬** 또는 **SQLite-backed 경량 큐**)를 도입하여:
- pipeline → fallback: Redis Streams 비동기 전환 (스케일아웃 병목 해소)
- 검토 큐 알림: Redis Pub/Sub → 프론트엔드 SSE
- 장기 목표: Celery/Dramatiq 태스크 그래프로 P1~P6 병렬화

---

## 8. 핵심 의존성

```txt
# Stage 1 — 전처리
opencv-python>=4.9.0
scikit-image>=0.22.0
Pillow>=10.0.0
numpy>=1.26.0
realesrgan
basicsr

# Stage 2 — VLM (vLLM 서버)
vllm
torch
transformers>=5.5.0          # Gemma4 아키텍처 지원 필수

# Stage 3 — 후처리
pydantic>=2.6.0
lxml>=5.1.0
pandas>=2.2.0
sqlalchemy>=2.0.0
pyyaml

# VLM 클라이언트
openai                      # vLLM OpenAI 호환 API 클라이언트
httpx                       # vLLM 헬스체크 + fallback 통신
```

---

## 8. 코딩 컨벤션

- 모든 함수·클래스에 docstring 필수
- 컴포넌트 간 데이터 전달은 `src/interfaces/types.py`의 dataclass 사용
- 오케스트레이터(`pipeline/orchestrator.py`)만 컴포넌트를 순서대로 호출
- VLM instruction/response는 **반드시 guided_json으로 구조화**
- 신뢰도는 logprobs 기반 산출 → P4 룰 검증으로 보정
- Python 3.10+
