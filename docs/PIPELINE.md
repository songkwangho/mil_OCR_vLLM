# 파이프라인 설계 가이드

> 파이프라인 구조, 컴포넌트 상세 설계, 인터페이스 타입, 가용성 설계를 기술합니다.

---

## 1. 파이프라인 구조

```
DocumentInput
    │
    ▼
[Stage 1 — 전처리]
    P1: 화질 보정 + SR → PreprocessedImage
    P2: 레이아웃 탐지  → RawLayoutResult
        ├── Fusion OFF: V3 단독
        └── Fusion ON (LAYOUT_FUSION_MODE=true):
            ├── DPI ≥ 150: V3(구조) + plus-L(텍스트), 50% 겹침 필터
            └── DPI < 150: V3(구조) + OpenCV heuristic(텍스트), 50% 겹침 필터
            reading_order: V3 구조 영역 → 텍스트 좌상→우하
    P2.5-A: LayoutPostProcessor → LayoutResult
        ① 미소 박스 제거 (6px 미만)
        ② 중복 박스 제거 (IoU > 0.7, seal은 0.5)
        ③ 인접 텍스트 블록 병합
        ④ reading_order 재정렬 (_remap_reading_order)
    │
    ▼
[Stage 2 — VLM 통합 추론]
    P3-A: FormClassifier → FormType
        전체 페이지 저해상도(140토큰) → form_type 확정
        ┌─ military 서식 → 군수 경로
        └─ other 문서   → 범용 OCR 경로 (군수 룰 검증 없음)

    P2.5-B: InstructionRouter → list[InstructionSpec]
        region_type + form_type → instruction + 1-shot 예시
        military: 도메인 맥락 + 군수 JSON Schema
        other:    범용 OCR instruction + _general.json
        저신뢰 영역: OCR 힌트 삽입 (ocr_hint_provider)

    P2.5-C: ResolutionRouter → dict[int, list[CroppedRegion]]
        pixel_budget 할당 + bbox 크롭(패딩 포함) + 48px 배수 정렬 + 배치 그룹화

    P3-B: StructuredExtractor → VLMResult
        pixel_budget 기준 배치 → Gemma4 병렬 호출 (temperature=0.0)
        CoT analysis 필드 포함 → logprobs 신뢰도 산출
        저신뢰 필드(< 0.60) → pixel_budget 상향 + OCR 힌트 재시도
    │
    ▼
[Stage 3 — 후처리]
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult
        military: 산술/날짜/코드 교차검증 + 경로별 임계값
        other:    신뢰도 산출만 (군수 룰 건너뜀)
        ├─ 통과 → P5 → P6 DB 적재
        └─ 실패 → 검토 큐 → 담당자 수정 → P6 재적재

[Fallback 경로 — VLM 불가 시]
    P1 → P2 → P2.5-A → Fallback(T3~T5) → P4(Fallback 임계값) → 검토 큐(항상)
```

---

## 2. 인터페이스 데이터 타입

> 파일: `src/interfaces/types.py`, `src/interfaces/enums.py`

### 2-1. 입력 / 전처리

```python
@dataclass
class DocumentInput:
    doc_id: str
    raw_bytes: bytes
    file_ext: FileExt        # jpg, png, tiff, pdf
    source_type: SourceType  # scan, camera, fax, pdf_export
    dpi_hint: Optional[int] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class PreprocessedImage:
    doc_id: str
    image_array: np.ndarray  # H×W×3 uint8 (RGB) — 이진화 금지, RGB 유지
    binary_array: np.ndarray # H×W uint8 (기울기 보정용으로만 사용)
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
    """P2.5-A 정제 완료 결과."""
    doc_id: str
    page_width: int
    page_height: int
    regions: list[LayoutRegion]
    reading_order: list[int]    # _remap_reading_order 적용 완료
    analysis_mode: AnalysisMode = AnalysisMode.HEURISTIC
    removed_count: int = 0
    merged_count: int = 0
    warnings: list[str] = field(default_factory=list)
```

### 2-3. InstructionRouter 출력

```python
@dataclass
class InstructionSpec:
    region_id: str
    region_type: RegionType
    form_type: Optional[FormType] = None
    system_prompt: str = ""       # 도메인 맥락 (군수 서식) 또는 범용
    user_instruction: str = ""    # 태스크 지시 + 1-shot 예시 + OCR 힌트(선택)
    json_schema: Optional[dict] = None  # guided_json (CoT analysis 필드 포함)
    pixel_budget: int = 560
    ocr_hint: Optional[str] = None      # PaddleOCR 선행 결과 (저신뢰 영역)
    is_retry: bool = False              # 재시도 호출 여부
```

### 2-4. ResolutionRouter 출력

```python
@dataclass
class CroppedRegion:
    region_id: str
    region_type: RegionType
    cropped_image: np.ndarray  # bbox 크롭 + 패딩 + 48px 배수 정렬
    pixel_budget: int
    instruction_spec: InstructionSpec

# pixel_budget 기준값 (Gemma4 공식 문서 기반)
PIXEL_BUDGET = {
    "table":             1120,  # 셀 경계·미세 글씨 → 최고 해상도
    "handwritten_field": 1120,  # 수기 기입란 — 560에서 상향 (한국어 획 구분)
    "seal":               560,  # 원형 배치 텍스트
    "text":               560,  # 일반 텍스트 — 280에서 상향 (소형 폰트)
    "formula":            280,
    "chart":              280,
    "figure":             140,
    "header":             140,
    "footer":             140,
}

# 크롭 패딩 비율 (원본 이미지 맥락 포함, 공백 패딩 금지)
CROP_PADDING_RATIO = {
    "table":             0.05,
    "handwritten_field": 0.15,  # 양식 레이블·경계선 포함
    "seal":              0.10,
    "text":              0.05,
    "default":           0.05,
}
```

### 2-5. VLM 통합 추론 결과

```python
@dataclass
class FieldValue:
    field_key: str
    raw_value: str
    corrected_value: str
    data_type: str             # text, number, date, code
    confidence: float          # logprobs 기하평균
    token_logprobs: list[float]
    is_flagged: bool = False
    region_id: Optional[str] = None
    was_retried: bool = False  # 재시도 여부 (디버깅용)

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
    form_type: FormType        # military 서식 또는 other
    form_confidence: float
    schema_id: str             # "supply_request:v1" | "other:v1"
    fields: list[FieldValue]
    tables: list[RecognizedTable]
    domain_codes: list[DomainCode]
    processing_path: ProcessingPath  # vlm | fallback
    processing_time_ms: float = 0.0
    retry_count: int = 0       # 재시도 발생 횟수 (모니터링용)
    # raw_json: DB 적재 제외 — 파일 시스템(P3B/raw_vlm_responses/)에만 보존
    warnings: list[str] = field(default_factory=list)
```

### 2-6. 후처리 결과

```python
@dataclass
class ValidationError:
    error_id: str
    error_type: ValidationErrorType  # arithmetic, date_logic, code_format, missing_field, format
    severity: Severity               # critical, high, medium, low
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
    processing_path: ProcessingPath

@dataclass
class CorrectedField:
    """검토 큐 담당자 수정 — JSONPath 기반 중첩 필드 지원."""
    field_path: str        # "items[0].quantity" 형식
    original_value: str
    corrected_value: str
    corrected_by: str
    corrected_at: datetime

@dataclass
class ReviewQueueItem:
    queue_id: str                          # RQ-20260407-001
    doc_id: str
    enqueued_at: datetime
    priority: ReviewPriority               # critical, normal
    reason: ReviewReason                   # validation_failed, low_confidence, fallback, total_failure
    processing_path: ProcessingPath
    validated_result: Optional[ValidatedResult] = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    flagged_fields: list[str] = field(default_factory=list)
    original_image_path: str = ""
    preprocessed_image_path: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    corrected_fields: dict[str, CorrectedField] = field(default_factory=dict)
    reviewer_notes: str = ""

@dataclass
class PipelineOutput:
    doc_id: str
    status: PipelineStatus     # success, partial, review, failed, other_document
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

## 3. 컴포넌트 상세 설계

### 3-1. P2.5-A LayoutPostProcessor

**파일**: `src/preprocess/layout_postprocessor.py`

```python
class LayoutPostProcessor:
    MICRO_BOX_PX = 6
    OVERLAP_IOU_THRESHOLD = 0.7
    SEAL_IOU_THRESHOLD = 0.5

    def process(self, raw: RawLayoutResult) -> LayoutResult:
        regions, removed_ids = self._filter_micro_boxes(raw.regions)
        regions, overlap_removed = self._filter_overlapping(regions)
        removed_ids.update(overlap_removed)
        regions, merged_map = self._merge_adjacent_text_blocks(regions)
        reading_order = self._remap_reading_order(
            raw.reading_order, removed_ids=removed_ids, merged_map=merged_map
        )
        return LayoutResult(..., regions=regions, reading_order=reading_order,
                           removed_count=len(removed_ids), merged_count=len(merged_map))

    def _remap_reading_order(
        self, original_order: list[int],
        removed_ids: set[str], merged_map: dict[str, str],
    ) -> list[int]:
        """제거/병합 결과를 반영하여 reading_order 재정렬.
        1. removed_ids 인덱스 제거
        2. merged_map 흡수된 id → 병합 대상 id 치환 (중복 제거)
        3. 정제 후 regions 기준 리넘버링

        예시:
          original_order = [0, 1, 2, 3, 4]
          removed_ids = {"r_0002"} → idx=1 제거
          merged_map = {"r_0004": "r_0003"} → idx=4를 idx=3으로 치환
          → 결과: [0, 1, 2]
        """
```

### 3-2. P3-A FormClassifier

**파일**: `src/vlm/form_classifier.py`

**FormType 분류 체계**:

| FormType | 의미 | 처리 경로 |
|----------|------|---------|
| `supply_request` | 보급청구서 | military |
| `maintenance_record` | 정비기록서 | military |
| `inventory_sheet` | 물자현황표 | military |
| `handover_doc` | 인수인계서 | military |
| `inspection_report` | 검사보고서 | military |
| `unknown` | 군수 서식인데 유형 불명 | military (_fallback.json) |
| `other` | 군수 서식이 아님 | other → 범용 OCR, 검토 큐 없음 |

```python
class FormClassifier:
    PIXEL_BUDGET = 140

    def classify(
        self, image_rgb: np.ndarray,
        warnings: Optional[list[str]] = None,
    ) -> tuple[FormType, float]:
        """전체 페이지 이미지 → 저해상도 VLM 1회 호출.
        Returns: (form_type, confidence)
        """
        # guided_json enum: supply_request|...|unknown|other
        # temperature=0.0 — 분류 결정론적 출력
        # other → _general.json 스키마 사용, 군수 룰 검증 건너뜀
```

### 3-3. P2.5-B InstructionRouter

**파일**: `src/vlm/instruction_router.py`

**프롬프트 설계 원칙**:
- system_prompt를 **정적으로 유지** → prefix caching 효율 극대화
- 가변 콘텐츠(실제 OCR 이미지)는 프롬프트 **끝에** 배치
- **analysis 필드를 JSON Schema 최상단에** 배치 → CoT 효과
- analysis 길이를 ~30~50 토큰으로 제한하는 문구 포함
- **영어 지시문 + 한국어 필드명** 하이브리드 사용

```python
class InstructionRouter:
    def route(
        self,
        region: LayoutRegion,
        form_type: Optional[str],
        ocr_hint: Optional[str] = None,
    ) -> InstructionSpec:
        """
        military 경로:
          form_type="supply_request", region_type="text"
          → system_prompt: "군수 보급청구서 OCR 시스템입니다."
          → user_instruction: "이 수기 기입란에서 NSN 코드와 수량을 인식하세요.
                               NSN 형식: NNNN-NN-NNN-NNNN. 불확실한 글자: [?].
                               유사 문자 주의: ㄱ/ㅋ, 1/ㅣ, 0/O
                               [예시] <1-shot 이미지> → <정답 JSON>"
          → + OCR 힌트 (해당 시)

        other 경로:
          form_type="other"
          → system_prompt: "문서 OCR 시스템입니다."
          → user_instruction: "이 영역의 텍스트를 인식하세요."
          → json_schema: _general_schema (CoT 없음)
        """

    def _build_with_ocr_hint(
        self, instruction: str, ocr_hint: str
    ) -> str:
        return instruction + f"\n[OCR 힌트] 경량 OCR 인식 결과: {ocr_hint}\n위 내용을 참고하여 보다 정확하게 추출하세요."
```

### 3-4. P2.5-C ResolutionRouter

**파일**: `src/vlm/resolution_router.py`

```python
class ResolutionRouter:
    def route(
        self,
        layout: LayoutResult,
        preprocessed: PreprocessedImage,
        instructions: dict[str, InstructionSpec],
    ) -> dict[int, list[CroppedRegion]]:
        """bbox 크롭 + 패딩 + 48px 배수 정렬 + 배치 그룹화.
        반환: {pixel_budget → [CroppedRegion, ...]}
        동일 budget끼리 그룹화 → vLLM 패딩 오버헤드 없음.
        48px 배수 정렬 → SigLIP 3×3 패치 풀링 효율 최적화.
        """

    def _crop_with_padding(
        self, image: np.ndarray, bbox: BoundingBox,
        padding_ratio: float
    ) -> np.ndarray:
        """공백 패딩이 아닌 원본 이미지의 실제 주변 콘텐츠를 포함.
        경계를 넘어가는 경우 이미지 가장자리로 클리핑.
        """

    def _align_to_48px(self, image: np.ndarray) -> np.ndarray:
        """크롭 이미지를 48px 배수로 리사이즈.
        SigLIP: 16×16 패치 → 3×3 블록 average pooling = 48px 배수가 최적.
        패딩에 토큰이 낭비되지 않도록 보장.
        """
        h, w = image.shape[:2]
        new_h = max(48, round(h / 48) * 48)
        new_w = max(48, round(w / 48) * 48)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
```

### 3-5. P3-B StructuredExtractor

**파일**: `src/vlm/structured_extractor.py`

```python
class StructuredExtractor:
    RETRY_THRESHOLD = 0.60   # 이하면 재시도
    MAX_RETRIES = 1          # 과도한 지연 방지
    RETRY_BUDGET_MAP = {140: 280, 280: 560, 560: 1120, 1120: 1120}

    def extract(
        self,
        groups: dict[int, list[CroppedRegion]],
        doc_id: str, form_type: FormType,
        form_confidence: float, schema_id: str,
        schema: Optional[dict] = None,
        warnings: Optional[list[str]] = None,
    ) -> VLMResult:
        """각 pixel_budget 그룹 → vLLM 배치 호출.
        저신뢰 필드 감지 시 pixel_budget 상향 + OCR 힌트로 재시도.
        """

    def _process_batch(
        self, batch: list[CroppedRegion], pixel_budget: int
    ) -> list[dict]:
        """temperature=0.0으로 결정론적 OCR 출력 보장."""

    def _retry_low_confidence_fields(
        self, fields: list[FieldValue], region_map: dict[str, CroppedRegion]
    ) -> list[FieldValue]:
        """logprobs < RETRY_THRESHOLD 필드 → pixel_budget 상향 + OCR 힌트 재시도."""
```

**logprobs 신뢰도 산출**:

```python
def calc_field_confidence(token_logprobs: list[float]) -> float:
    """기하 평균: exp(mean(logprobs))
    한국어 subword 토크나이제이션 특성상 개별 토큰 확률이 낮게 나타날 수 있어,
    임계값을 영어 대비 5~10% 낮게 설정 (PIPELINE.md §4-1 임계값 테이블 참조).
    길이 편향 보정(length_factor)은 Phase 2 SFT 후 실측 데이터 기반으로 도입 예정.
    """
    if not token_logprobs:
        return 0.0
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    mean_logprob = max(mean_logprob, -20.0)
    return round(min(1.0, max(0.0, math.exp(mean_logprob))), 4)
```

---

## 4. logprobs 신뢰도 임계값

### 4-1. 필드 유형별 임계값 — 처리 경로 분리

| 필드 유형 | VLM 임계값 | Fallback 임계값 | 비고 |
|----------|:----------:|:--------------:|------|
| 금액 (amount) | 0.99 | 0.80 | Fallback은 사실상 항상 검토 큐 |
| 코드 (code/NSN) | 0.97 | 0.75 | |
| 날짜 (date) | 0.95 | 0.70 | |
| 수량 (quantity) | 0.95 | 0.70 | |
| 일반 텍스트 | 0.90 | 0.65 | 한국어 subword 특성상 영어 대비 5~10% 낮게 설정 |
| 서명 | 0.70 | 0.50 | |

> **재시도 임계값**: 0.60 이하 → pixel_budget 상향 + OCR 힌트로 1회 재시도
> **other 문서**: 군수 임계값 적용 안 함. 전체 신뢰도만 산출.

### 4-2. P4 룰 검증으로 보정 (military 경로만)

- `합계 ≠ 수량 × 단가` → 관련 필드 신뢰도 **-0.30**
- NSN 형식 불일치 → 해당 필드 신뢰도 **-0.15**
- 날짜 순서 위반 → 해당 필드 신뢰도 **-0.10**

### 4-3. NSN 패턴 강제 전략

```
우선순위:
1. InstructionRouter 프롬프트 힌트 (기본 — xgrammar 유지)
   user_instruction: "NSN 코드는 NNNN-NN-NNN-NNNN 형식 13자리입니다."
   
2. guided_regex (특정 필드에만 패턴 강제 필요 시)
   extra_body={"guided_regex": r"\d{4}-\d{2}-\d{3}-\d{4}"}
   
3. guidance 백엔드 전환 (NSN 형식 강제가 절대적으로 필요한 경우)
   --guided-decoding-backend guidance
   JSON Schema 내 pattern 제약 완전 지원
```

---

## 5. SPOF 대비 — 가용성 설계

### 5-1. 수준 A — vLLM 헬스체크

```python
class VLMHealthMonitor:
    def __init__(self, health_url: str, interval: int = 30, timeout: int = 10): ...

    def is_healthy(self) -> bool:
        """캐시된 상태 반환. interval 경과 시 실제 체크."""

    def _on_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._healthy = False
```

### 5-2. 수준 B — 경량 Fallback

```python
class FallbackPolicy:
    def decide(self, vlm_healthy: bool, fallback_enabled: bool,
               fallback_healthy: bool) -> str:
        if vlm_healthy:          return "vlm"
        if fallback_enabled and fallback_healthy: return "fallback"
        return "review_queue"
```

| 항목 | VLM 주 경로 | Fallback |
|------|------------|---------|
| 서식 분류 | P3-A (저해상도) | DiT 모델 |
| 텍스트 인식 | P3-B 크롭 고해상도 | PP-OCRv5 |
| 표 구조 | P3-B 1120토큰 | SLANeXt |
| guided_json | ✅ | ❌ 규칙 기반 |
| OCR-augmented | ✅ 저신뢰 영역 | ❌ |
| P4 임계값 | VLM 프로파일 | Fallback 프로파일 (낮음) |

### 5-3. 수준 C — 수동 검토 큐

| 조건 | 우선순위 |
|------|---------|
| CRITICAL ValidationError | 🔴 긴급 |
| LOW confidence 필드 (재시도 후에도 임계값 미달) | 🟡 일반 |
| Fallback 경로 처리 | 🟡 일반 |
| VLM + Fallback 모두 불가 | 🔴 긴급 |

> **other 문서는 검토 큐 미적재** — 군수 업무 대상이 아님.

---

## 6. 서비스 통신 아키텍처

### 6-1. 현재 구조 (Phase 1)

```
[Pipeline 컨테이너 :8080]
  ├── P1 Preprocessor          (in-process)
  ├── P2 LayoutAnalyzer        HTTP POST → [Layout :8082]
  ├── P2.5-A LayoutPostProcessor (in-process)
  ├── P3-A FormClassifier      HTTP POST → [vLLM :8100]
  ├── P2.5-B InstructionRouter (in-process, OCRHintProvider 포함)
  ├── P2.5-C ResolutionRouter  (in-process)
  ├── P3-B StructuredExtractor HTTP POST → [vLLM :8100]
  ├── Fallback                 HTTP POST → [Fallback :8081]
  ├── P4 Validator             (in-process)
  ├── P5 Serializer            (in-process)
  └── P6 DBLoader              (in-process, SQLite)
```

**Layout 분리 사유**: PaddlePaddle ↔ PyTorch CUDA 충돌 방지.

### 6-2. Phase 3 목표: 비동기 메시지 기반

- pipeline → fallback: Redis Streams 비동기 전환
- 검토 큐 알림: Redis Pub/Sub → 프론트엔드 SSE

---

## 7. 코딩 컨벤션

- 모든 함수·클래스에 docstring 필수
- 컴포넌트 간 데이터 전달은 `src/interfaces/types.py`의 dataclass 사용
- 오케스트레이터(`pipeline/orchestrator.py`)만 컴포넌트를 순서대로 호출
- VLM instruction/response는 **반드시 guided_json으로 구조화**
- 모든 VLM 호출에 `temperature=0.0` 설정 (결정론적 OCR)
- `VLMResult.raw_json`은 DB 적재 제외 — `P3B/raw_vlm_responses/`에만 보존
- P4는 `processing_path` + `form_type(other 여부)` 기반으로 임계값·룰 프로파일 분기
- Python 3.10+

**모듈 책임 경계**:

| 모듈 | 책임 | 하지 않는 것 |
|------|------|-------------|
| `vlm_client.py` | vLLM HTTP 통신, base64 인코딩, logprobs 파싱 | 이미지 크롭, 도메인 코드 감지 |
| `resolution_router.py` | bbox 크롭 + 패딩 + 48px 정렬 + 배치 그룹화 | VLM 통신 |
| `structured_extractor.py` | 배치 VLM 호출, 필드 추출, 재시도, VLMResult 조립 | 이미지 크롭 |
| `form_classifier.py` | 서식 분류 VLM 1회 호출 (military/other) | 필드 추출, 크롭 |
| `ocr_hint_provider.py` | PaddleOCR 선행 실행, 힌트 문자열 생성 | VLM 호출, 크롭 |

---

## 8. 핵심 의존성

```txt
opencv-python>=4.9.0
scikit-image>=0.22.0
Pillow>=10.0.0
numpy>=1.26.0
realesrgan
basicsr
paddlepaddle-gpu    # OCR-augmented (PaddleOCR)
paddleocr>=2.8.0
vllm>=0.19.0
torch
transformers>=5.5.0
pydantic>=2.6.0
lxml>=5.1.0
pandas>=2.2.0
sqlalchemy>=2.0.0
pyyaml
openai
httpx
```
