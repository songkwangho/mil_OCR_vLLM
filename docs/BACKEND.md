# 백엔드 가이드

> 파이프라인 오케스트레이션, 인터페이스 타입, 후처리 로직, 도메인 지식을 기술합니다.

---

## 1. 파이프라인 구조 (고도화)

```
DocumentInput
    │
    ▼
[Stage 1 — 전처리]
    P1:     화질 보정 + SR          → PreprocessedImage
    P2:     레이아웃 탐지            → RawLayoutResult  (PP-DocLayout 원시 탐지)
    P2.5-A: LayoutPostProcessor     → LayoutResult     (정제 완료)
        ① 미소 박스 제거 (6px 미만)
        ② 중복 박스 제거 (IoU > 0.7)
        ③ 인접 텍스트 블록 병합
    │
    ▼
[Stage 2 — VLM 통합 추론]
    P3-A:   FormClassifier          → FormType
        전체 페이지 저해상도(140토큰) → form_type 확정 → JSON Schema 선택

    P2.5-B: InstructionRouter       → list[InstructionSpec]
        region_type + form_type → 도메인 맥락 포함 instruction 생성
        (TASK_PROMPTS 딕셔너리 대체 — layout_analyzer/types.py에서 제거)

    P2.5-C: ResolutionRouter        → list[CroppedRegion]
        영역 타입별 pixel_budget 할당:
        table: 1120 | seal/handwritten: 560 | text: 280 | header/footer: 140

    P3-B:   StructuredExtractor     → VLMResult
        pixel_budget 기준 배치 그룹화
        → bbox 크롭 이미지 + instruction → Gemma4 병렬 배치 호출
        → guided_json + logprobs 신뢰도 산출
        → RepetitionGuard (운영 후 판단)
    │
    ▼
[Stage 3 — 후처리]
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult
        처리 경로별 임계값 적용 (VLM / Fallback 분리)
        ├─ 통과 (confidence ≥ 임계값, CRITICAL 없음)
        │      → P5 직렬화 → P6 DB 적재 → PipelineOutput
        └─ 실패 (CRITICAL 오류 or LOW confidence)
               → 검토 큐 적재 → 담당자 UI 확인 → 수정 후 P6 재적재

[Fallback 경로 — VLM 불가 시]
    P1 → P2 → P2.5-A → Fallback(T3~T5) → P4(Fallback 임계값) → 검토 큐 (항상)
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
    region_type: RegionType    # text, table, figure, header, footer, seal, formula, chart
    bbox: BoundingBox
    confidence: float
    polygon: Optional[list[tuple[float, float]]] = None  # V3 polygon

@dataclass
class RawLayoutResult:
    """P2 원시 탐지 결과 — LayoutPostProcessor 정제 전."""
    doc_id: str
    page_width: int
    page_height: int
    regions: list[LayoutRegion]
    reading_order: list[int]
    analysis_mode: AnalysisMode = AnalysisMode.HEURISTIC
    warnings: list[str] = field(default_factory=list)

@dataclass
class LayoutResult:
    """P2.5-A LayoutPostProcessor 정제 완료 결과."""
    doc_id: str
    page_width: int
    page_height: int
    regions: list[LayoutRegion]   # 미소/중복 박스 제거 + 블록 병합 완료
    reading_order: list[int]
    analysis_mode: AnalysisMode = AnalysisMode.HEURISTIC
    removed_count: int = 0        # 제거된 박스 수 (디버깅용)
    merged_count: int = 0         # 병합된 블록 수 (디버깅용)
    warnings: list[str] = field(default_factory=list)
```

### 2-3. P2.5-B InstructionRouter 출력

```python
@dataclass
class InstructionSpec:
    """InstructionRouter가 생성하는 영역별 VLM 호출 명세."""
    region_id: str
    region_type: RegionType
    form_type: Optional[FormType] = None   # P3-A FormClassifier 결과
    system_prompt: str = ""                # 도메인 맥락 (군수 서식 특화)
    user_instruction: str = ""             # 태스크 지시 (PaddleOCR-VL "OCR:" 대비 상세)
    json_schema: Optional[dict] = None     # guided_json (표/전체서식용)
    pixel_budget: int = 280                # ResolutionRouter가 할당

# InstructionRouter가 생성하는 도메인 맥락 포함 instruction 예시:
# form_type="supply_request", region_type="text" →
#   system_prompt: "군수 보급청구서 OCR 시스템입니다."
#   user_instruction: "이 수기 기입란에서 NSN 코드(NNNN-NN-NNN-NNNN)와
#                      수량을 인식하세요. 불확실한 글자는 [?]로 표시."
#
# form_type=None, region_type="text" (분류 전) →
#   user_instruction: "한국어 텍스트를 인식하세요." (도메인 맥락 없음)
# → FormClassifier 선행이 InstructionRouter 품질을 결정하는 이유
```

### 2-4. P2.5-C ResolutionRouter 출력

```python
@dataclass
class CroppedRegion:
    """ResolutionRouter가 생성하는 크롭 이미지 + 메타데이터."""
    region_id: str
    region_type: RegionType
    cropped_image: np.ndarray      # bbox 크롭 이미지
    pixel_budget: int              # Gemma4 이미지 토큰 수
    instruction_spec: InstructionSpec

# pixel_budget 기준값 (실험 후 조정 가능)
PIXEL_BUDGET = {
    "table":             1120,   # 표: 셀 경계·미세 글씨 → 최고 해상도
    "seal":               560,   # 인장: 원형 배치 텍스트
    "handwritten_field":  560,   # 수기 기입란: 군수 서식 핵심 영역
    "text":               280,   # 일반 텍스트
    "formula":            280,   # 수식
    "chart":              280,
    "figure":             140,
    "header":             140,   # 헤더/푸터: 저해상도로 충분
    "footer":             140,
}
```

### 2-5. VLM 통합 추론 결과

```python
@dataclass
class FieldValue:
    """P3-B StructuredExtractor가 추출한 개별 필드."""
    field_key: str
    raw_value: str
    corrected_value: str
    data_type: str             # text, number, date, code
    confidence: float          # logprobs 기반 (길이 편향 보정 포함)
    token_logprobs: list[float]
    is_flagged: bool = False
    region_id: Optional[str] = None   # 출처 영역 (디버깅용)

@dataclass
class RecognizedTable:
    region_id: str
    html: str
    cells: list[dict]
    confidence: float

@dataclass
class DomainCode:
    code_type: CodeType        # nsn, k_nsn, unit_code, date, rank
    raw_value: str
    normalized_value: str
    confidence: float

@dataclass
class VLMResult:
    doc_id: str
    form_type: FormType
    form_confidence: float
    schema_id: str             # "supply_request:v1" 형식 (스키마 버전 포함)
    fields: list[FieldValue]
    tables: list[RecognizedTable]
    domain_codes: list[DomainCode]
    raw_json: str = ""         # VLM 원본 JSON 응답 (디버깅용)
    processing_time_ms: float = 0.0
    processing_path: ProcessingPath = ProcessingPath.VLM
    warnings: list[str] = field(default_factory=list)
```

### 2-6. 후처리 결과

```python
@dataclass
class ValidationError:
    error_id: str              # ve_0001, ve_0002, ...
    error_type: ValidationErrorType  # arithmetic, date_logic, code_format, missing_field, format
    severity: Severity         # critical, high, medium, low
    field_ref: str
    expected: str
    actual: str
    message: str

@dataclass
class ValidatedResult:
    doc_id: str
    fields: list[FieldValue]
    tables: list[RecognizedTable]
    validation_errors: list[ValidationError]
    overall_confidence: float
    review_required: bool
    flagged_fields: list[str]
    processing_path: ProcessingPath = ProcessingPath.VLM

@dataclass
class ReviewQueueItem:
    queue_id: str                    # RQ-20260407-001
    doc_id: str
    enqueued_at: datetime
    priority: ReviewPriority         # critical, normal
    reason: ReviewReason             # validation_failed, low_confidence, fallback, total_failure
    processing_path: ProcessingPath  # vlm, fallback, none
    validated_result: Optional[ValidatedResult] = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    flagged_fields: list[str] = field(default_factory=list)
    original_image_path: str = ""
    preprocessed_image_path: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    corrected_fields: dict[str, str] = field(default_factory=dict)
    reviewer_notes: str = ""

@dataclass
class PipelineOutput:
    doc_id: str
    status: PipelineStatus     # success, partial, review, failed
    processing_path: ProcessingPath
    form_type: Optional[FormType] = None
    json_output: Optional[str] = None
    xml_output: Optional[str] = None
    csv_rows: list[dict] = field(default_factory=list)
    db_record_ids: list[str] = field(default_factory=list)
    review_queue_id: Optional[str] = None
    processing_ms: float = 0.0
```

---

## 3. P2~P3 구간 신규 컴포넌트 상세 설계

### 3-1. P2.5-A LayoutPostProcessor

**파일**: `src/preprocess/layout_postprocessor.py`

**역할**: PP-DocLayout 원시 탐지 결과의 노이즈를 제거하여 VLM 입력 품질을 보장합니다. 이 단계 없이 크롭하면 VLM이 6px짜리 점, 인장과 텍스트가 겹친 중복 박스, 단어 단위로 분절된 텍스트 조각을 받게 됩니다.

```python
class LayoutPostProcessor:
    MICRO_BOX_PX = 6          # 6px 미만 박스 제거 (PaddleOCR-VL 기준)
    OVERLAP_IOU_THRESHOLD = 0.7
    SEAL_IOU_THRESHOLD = 0.5  # 인장은 더 엄격한 기준 적용

    def process(self, raw: RawLayoutResult) -> LayoutResult:
        regions, removed_ids = self._filter_micro_boxes(raw.regions)
        regions, overlap_removed = self._filter_overlapping(regions)
        removed_ids.update(overlap_removed)
        regions, merged_map = self._merge_adjacent_text_blocks(regions)

        # 제거/병합된 region_id 반영하여 reading_order 재정렬
        reading_order = self._remap_reading_order(
            raw.reading_order,
            removed_ids=removed_ids,
            merged_map=merged_map,
        )

        return LayoutResult(
            ...,
            regions=regions,
            reading_order=reading_order,
            removed_count=len(removed_ids),
            merged_count=len(merged_map),
        )

    def _filter_micro_boxes(self, regions) -> tuple[list, set[str]]:
        """6px 미만 박스 제거.
        Returns: (남은 regions, 제거된 region_id 집합)"""

    def _filter_overlapping(self, regions) -> tuple[list, set[str]]:
        """IoU 임계값 초과 중복 박스 제거.
        seal 영역: 0.5, 그 외: 0.7
        신뢰도 낮은 쪽 제거.
        Returns: (남은 regions, 제거된 region_id 집합)"""

    def _merge_adjacent_text_blocks(self, regions) -> tuple[list, dict[str, str]]:
        """동일 컬럼 내 인접 text 블록 병합.
        x-center 차이 < 20px, y-gap < 15px 조건.
        Returns: (병합 후 regions, {흡수된 region_id → 병합 대상 region_id})"""

    def _remap_reading_order(
        self,
        original_order: list[int],
        removed_ids: set[str],
        merged_map: dict[str, str],
    ) -> list[int]:
        """제거/병합 결과를 반영하여 reading_order를 재정렬.

        처리 흐름:
          1. removed_ids에 포함된 인덱스 제거
          2. merged_map에 포함된 인덱스를 병합 대상 인덱스로 치환 (중복 제거)
          3. 남은 인덱스를 정제 후 regions 리스트 기준으로 리넘버링
             (원본 인덱스 → 정제 후 인덱스 매핑)

        예시:
          original_order = [0, 1, 2, 3, 4]
          removed_ids = {"r_0002"}        → idx=1 제거
          merged_map = {"r_0004": "r_0003"} → idx=4를 idx=3으로 치환
          → 정제 후 regions: [r_0001, r_0003(+r_0004 병합), r_0005]
          → 재넘버링: {0→0, 3→1, 5→2}
          → 결과: [0, 1, 2]
        """
```

### 3-2. P3-A FormClassifier

**파일**: `src/vlm/form_classifier.py`

**역할**: 전체 페이지를 저해상도(140 토큰)로 Gemma4에 1회 호출하여 서식 유형을 확정합니다. 이 결과가 InstructionRouter에 피드백되어 이후 모든 영역의 instruction 품질을 결정합니다. 서식 분류 실패 시 wrong schema로 전체 추출 결과가 오염되는 오류 전파를 차단하는 게이트 역할입니다.

```python
class FormClassifier:
    def classify(
        self,
        image_rgb: np.ndarray,
        warnings: Optional[list[str]] = None,
    ) -> tuple[FormType, float]:
        """
        전체 페이지 이미지(H×W×3 RGB)를 저해상도(140 토큰)로 VLM 1회 호출.

        Returns:
            form_type: FormType   # supply_request | maintenance_record | ...| unknown
            confidence: float     # logprobs 기반 분류 신뢰도
        """
        # 저해상도 인코딩 → vLLM 호출 → logprobs 기반 신뢰도 산출
        # unknown → _fallback.json 스키마 사용
```

### 3-3. P2.5-B InstructionRouter

**파일**: `src/vlm/instruction_router.py`

**역할**: region_type과 form_type을 조합하여 도메인 맥락이 포함된 상세 InstructionSpec을 생성합니다. 기존 `instruction_builder.py`와 `TASK_PROMPTS` 딕셔너리를 대체하며, 레이아웃 모델과 VLM instruction 로직의 결합을 해소합니다.

PaddleOCR-VL의 `"OCR:"` 2단어 프리픽스는 전용 학습 모델이기 때문에 가능합니다. 범용 VLM인 Gemma4는 도메인 맥락, 출력 형식, 불확실성 처리까지 포함한 상세 instruction이 필요합니다.

```python
class InstructionRouter:
    def route(
        self,
        region: LayoutRegion,
        form_type: Optional[str],
    ) -> InstructionSpec:
        """
        동일 region_type이라도 form_type에 따라 다른 instruction 생성:

        region_type="text", form_type="supply_request":
          → "이 수기 기입란에서 NSN 코드(NNNN-NN-NNN-NNNN)와 수량을 인식하세요."

        region_type="text", form_type="maintenance_record":
          → "이 수기 기입란에서 장비 ID와 정비 유형을 인식하세요."

        region_type="text", form_type=None:
          → "한국어 텍스트를 인식하세요." (FormClassifier 미수행 시)
        """
```

### 3-4. P2.5-C ResolutionRouter

**파일**: `src/vlm/resolution_router.py`

**역할**: 영역 타입별로 Gemma4 이미지 토큰 예산(pixel_budget)을 차등 할당하고 bbox 크롭 이미지를 생성합니다. 모든 영역에 동일 해상도를 적용하면 표와 수기 기입란에서 부족하고 헤더/푸터에서 낭비가 발생합니다.

```python
class ResolutionRouter:
    def route(
        self,
        layout: LayoutResult,
        preprocessed: PreprocessedImage,
        instructions: dict[str, InstructionSpec],  # {region_id → InstructionSpec}
    ) -> dict[int, list[CroppedRegion]]:
        """bbox 크롭 + pixel_budget 할당 + 배치 그룹화.
        반환: {pixel_budget → [CroppedRegion, ...]} — 동일 budget끼리 그룹화.
        해상도가 다른 이미지 혼재 시 vLLM 패딩 오버헤드 발생."""
```

### 3-5. P3-B StructuredExtractor

**파일**: `src/vlm/structured_extractor.py`

**역할**: pixel_budget 기준으로 그룹화된 CroppedRegion 배치를 Gemma4에 병렬 전송하고 결과를 조립합니다. 기존 `gemma4_engine.py`에서 추출 전용 책임만 분리한 컴포넌트입니다.

```python
class StructuredExtractor:
    def extract(
        self,
        groups: dict[int, list[CroppedRegion]],
        doc_id: str,
        form_type: FormType,
        form_confidence: float,
        schema_id: str,
        schema: Optional[dict] = None,
        warnings: Optional[list[str]] = None,
    ) -> VLMResult:
        """
        각 pixel_budget 그룹 내 영역을 순차 vLLM 호출.
        guided_json + logprobs로 필드 추출 + 도메인 코드 자동 감지.
        결과를 region_id 기준으로 재조립하여 VLMResult 구성.
        """
```

---

## 4. SPOF 대비 — 가용성 설계

### 4-1. 수준 A — vLLM 자체 안정성 확보

| 메커니즘 | 구현 방식 | 설정 |
|----------|----------|------|
| Docker 자동 재시작 | `restart: unless-stopped` | `docker-compose.yml` |
| 헬스체크 | `curl -sf http://localhost:8000/health` | interval 30s, timeout 10s, retries 3, start_period 120s |
| 오케스트레이터 감시 | 매 요청 전 `/health` 확인 + 백그라운드 polling | `VLLM_HEALTH_CHECK_INTERVAL=30` |

```python
class VLMHealthMonitor:
    def __init__(self, health_url: str, interval: int = 30, timeout: int = 10):
        self.health_url = health_url
        self.interval = interval
        self.timeout = timeout
        self._healthy = True
        self._last_check: float = 0.0
        self._consecutive_failures: int = 0

    def is_healthy(self) -> bool:
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

### 4-2. 수준 B — 경량 Fallback

**전환 정책**:

```python
class FallbackPolicy:
    def decide(
        self,
        vlm_healthy: bool,
        fallback_enabled: bool,
        fallback_healthy: bool,
    ) -> str:
        if vlm_healthy:
            return "vlm"
        if fallback_enabled and fallback_healthy:
            return "fallback"
        return "review_queue"
```

**Fallback 출력 특성**:
- `processing_path = "fallback"` → P4에서 Fallback 임계값 적용 (VLM 대비 낮게 설정)
- `review_required = True` (항상)
- guided_json / logprobs 없음 → PP-OCRv5 rec_score 사용

### 4-3. 수동 검토 큐

**검토 큐 적재 조건**:

| 조건 | 트리거 | 우선순위 |
|------|--------|---------|
| CRITICAL ValidationError | `severity == "critical"` | 🔴 긴급 |
| LOW confidence 필드 | `field.confidence < 임계값` | 🟡 일반 |
| Fallback 경로 처리 | `processing_path == "fallback"` | 🟡 일반 |
| VLM + Fallback 모두 불가 | 원본 이미지만 큐 적재 | 🔴 긴급 |

---

## 5. Guided Decoding — 서식별 JSON Schema

### 5-1. 스키마 관리 (버전 관리 도입)

```
src/domain/schemas/
├── v1/
│   ├── supply_request.json
│   ├── maintenance_record.json
│   ├── inventory_sheet.json
│   ├── handover_doc.json
│   ├── inspection_report.json
│   └── _fallback.json
└── v2/                         ← 서식 개정 시
    └── supply_request.json

schema_registry.py              ← form_type + version → Schema 조회
```

`VLMResult.schema_id` 형식: `"supply_request:v1"` (버전 포함)
DB 레코드에 스키마 버전을 함께 기록하여 서식 개정 후 마이그레이션 가능.

### 5-2. 처리 흐름

1. P3-A FormClassifier → `form_type` 결정
2. InstructionRouter → `form_type` 기반 JSON Schema 선택
3. P3-B StructuredExtractor → 영역별 크롭 + Schema를 `guided_json`으로 vLLM 전달
4. VLM은 Schema에 맞는 JSON만 생성 → **파싱 실패 원천 차단**
5. `unknown` 시 `_fallback.json` 사용

---

## 6. logprobs 신뢰도 산출

### 6-1. 필드별 신뢰도 (길이 편향 보정 포함)

```python
def calc_field_confidence(
    token_logprobs: list[float],
    field_type: str,
) -> float:
    """
    기존: geometric_mean(exp(logprob)) — 토큰 수 많을수록 구조적 낮은 점수
    개선: 필드 유형별 기대 토큰 수 기반 보정 계수 적용

    예: "부대명: 수도방위사령부 예하 1경비단" (긴 텍스트)
        vs "NSN: 1005-01-432-1234" (짧은 코드)
        → 동일 유형 임계값 적용 전 편향 보정 필요
    """
    geo_mean = exp(mean(token_logprobs))
    length_factor = min(1.0, sqrt(EXPECTED_TOKENS[field_type]
                                   / max(len(token_logprobs), 1)))
    return round(geo_mean * length_factor, 4)
```

### 6-2. 필드 유형별 임계값 — 처리 경로 분리

| 필드 유형 | VLM 임계값 | Fallback 임계값 | 비고 |
|----------|:----------:|:--------------:|------|
| 금액 (amount) | 0.99 | 0.80 | Fallback은 사실상 항상 검토 큐 |
| 코드 (code/NSN) | 0.97 | 0.75 | |
| 날짜 (date) | 0.95 | 0.70 | |
| 수량 (quantity) | 0.95 | 0.70 | |
| 일반 텍스트 | 0.90 | 0.65 | |
| 서명 | 0.70 | 0.50 | |

### 6-3. P4 룰 검증으로 보정

- `합계 ≠ 수량 × 단가` → 관련 필드 신뢰도 **-0.30**
- NSN 형식 불일치 → 해당 필드 신뢰도 **-0.15**
- 날짜 순서 위반 → 해당 필드 신뢰도 **-0.10**

---

## 7. 군수 도메인 지식

### 7-1. 주요 서식 유형

| 서식명 | schema_id | JSON Schema | 주요 필드 |
|--------|-----------|------------|----------|
| 보급청구서 | `supply_request` | `supply_request.json` | unit_code, nsn, quantity, unit_price, total, request_date |
| 정비기록서 | `maintenance_record` | `maintenance_record.json` | equipment_id, maintenance_type, date, technician |
| 물자현황표 | `inventory_sheet` | `inventory_sheet.json` | item_name, stock_qty, location, disposal_qty |
| 인수인계서 | `handover_doc` | `handover_doc.json` | from_person, to_person, date, items |

### 7-2. 코드 체계

```
NSN (NATO Stock Number):  NNNN-NN-NNN-NNNN  (예: 1005-01-432-1234)
K-NSN (한국 물자코드):    KN-NNNNN-NNNN
부대 코드:                숫자 2~4자리 + 한글 부대명 (예: "1232부대")
```

---

## 8. 서비스 통신 아키텍처

### 8-1. 현재 구조 (Phase 1)

```
[Pipeline 컨테이너]
  ├── P1 Preprocessor          (in-process)
  ├── P2 LayoutAnalyzer        HTTP POST → [Layout 컨테이너 :8082]
  │     └── PP-DocLayoutV3 추론 (PaddlePaddle CUDA 12.6)
  ├── P2.5-A LayoutPostProcessor (in-process)
  ├── P3-A FormClassifier      HTTP POST → [vLLM 컨테이너 :8100]
  │     └── 저해상도 서식 분류 호출
  ├── P2.5-B InstructionRouter (in-process)
  ├── P2.5-C ResolutionRouter  (in-process)
  ├── P3-B StructuredExtractor HTTP POST → [vLLM 컨테이너 :8100]
  │     └── 배치 병렬 추출 호출
  ├── Fallback                 HTTP POST → [Fallback 컨테이너 :8081]
  ├── P4 Validator             (in-process)
  ├── P5 Serializer            (in-process)
  └── P6 DBLoader              (in-process, SQLite)
```

**Layout 서비스 분리 사유**: PaddlePaddle과 PyTorch의 CUDA 충돌 방지.
전체 서비스 CUDA 12.6 통일 완료 (H100 sm_90 지원). Layout 컨테이너는 fusion 모드(V3+plus-L / V3+heuristic, DPI 기반 분기) 지원.

### 8-2. Phase 3 목표: 비동기 메시지 기반

폐쇄망 적합 메시지 브로커(Redis 로컬 또는 SQLite-backed 경량 큐) 도입:
- pipeline → fallback: Redis Streams 비동기 전환
- 검토 큐 알림: Redis Pub/Sub → 프론트엔드 SSE

---

## 9. 핵심 의존성

```txt
# Stage 1 — 전처리
opencv-python>=4.9.0
scikit-image>=0.22.0
Pillow>=10.0.0
numpy>=1.26.0
realesrgan
basicsr

# Stage 2 — VLM (vLLM 서버)
vllm>=0.19.0
torch
transformers>=5.5.0          # Gemma4 아키텍처 지원 필수

# Stage 3 — 후처리
pydantic>=2.6.0
lxml>=5.1.0
pandas>=2.2.0
sqlalchemy>=2.0.0
pyyaml

# VLM 클라이언트
openai
httpx
```

---

## 10. 코딩 컨벤션

- 모든 함수·클래스에 docstring 필수
- 컴포넌트 간 데이터 전달은 `src/interfaces/types.py`의 dataclass 사용
- 오케스트레이터(`pipeline/orchestrator.py`)만 컴포넌트를 순서대로 호출
- `TASK_PROMPTS` 딕셔너리를 `layout_analyzer.py` / `types.py`에 두지 않음 — `InstructionRouter`에서 일원화
- VLM instruction/response는 **반드시 guided_json으로 구조화**
- 신뢰도는 logprobs 기반 산출 (길이 편향 보정) → P4 룰 검증으로 보정
- P4는 `processing_path` 기반으로 임계값 프로파일 분기
- Python 3.10+

---

## 11. 통합 테스트 절차 및 결과 기록

### 11-1. 실행 방식

```bash
docker run --rm --gpus device=0 --network host \
  -v $(pwd):/workspace \
  -e LAYOUT_SERVICE_URL=http://localhost:8082 \
  -e VLLM_BASE_URL=http://localhost:8100/v1 \
  -e PYTHONPATH=/workspace \
  mil_ocr_v2-pipeline \
  python scripts/run_pipeline_with_outputs.py [--input-dir data/raw]
```

### 11-2. 결과 저장 구조

```
data/pipeline_outputs/{YYYYMMDD_HHMMSS}/
├── run_summary.json
├── ocr_results.db
├── review_queue.db
└── {doc_id}/
    ├── summary.json
    ├── P1/
    │   ├── result.json
    │   ├── preprocessed.png
    │   └── binary.png
    ├── P2/
    │   ├── result.json            ← RawLayoutResult (정제 전)
    │   └── layout_visualization.png
    ├── P2.5A/
    │   └── result.json            ← LayoutResult (정제 후, removed/merged 수 포함)
    ├── P3A/
    │   └── result.json            ← form_type, form_confidence, 추론 시간
    ├── P3B/
    │   ├── result.json            ← fields[], tables[], domain_codes[], logprobs
    │   └── raw_vlm_responses/
    │       ├── batch_{budget}_{n}.json   ← pixel_budget별 배치 응답
    │       └── ...
    ├── P4/
    │   └── result.json
    ├── P5/
    │   ├── output.json
    │   ├── output.xml
    │   └── output.csv
    └── P6/
        └── result.json
```

### 11-3. 단계별 타이밍 기록 (run_summary.json)

```json
{
  "timestamp": "20260409_043928",
  "document_count": 3,
  "documents": [
    {
      "doc_id": "보급청구서_001",
      "status": "success",
      "processing_path": "vlm",
      "total_ms": 4820.1,
      "timings": {
        "P1": 310.2,
        "P2": 48.5,
        "P2_5A": 12.3,
        "P3A": 420.7,
        "P3B": 3850.4,
        "P4": 3.1,
        "P5": 4.2,
        "P6": 18.7
      }
    }
  ]
}
```

### 11-4. 콘솔 출력 보고 원칙

1. **환경 정보**: CUDA 버전, 서비스 URL, 테스트 이미지 수, 컴포넌트 초기화 시간
2. **문서별 단계 결과**: 각 단계의 실행 시간, 핵심 output, 경고/폴백 여부
3. **P2.5-A 정제 결과**: 제거된 박스 수, 병합된 블록 수
4. **P3-A 분류 결과**: form_type, form_confidence
5. **P3-B 배치 처리**: 그룹별 배치 크기, 각 배치 추론 시간
6. **실패 시**: 실패 단계, 에러 메시지, 스택 트레이스
7. **전체 요약**: `[PASS/FAIL] {doc_id} total=Xms P1=Xms P2=Xms P2.5A=Xms P3A=Xms P3B=Xms ...`
