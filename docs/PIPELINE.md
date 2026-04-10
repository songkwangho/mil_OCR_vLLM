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

    P2.5-C: ResolutionRouter → dict[int, list[CroppedRegion]]
        pixel_budget 할당 + bbox 크롭 + 배치 그룹화

    P3-B: StructuredExtractor → VLMResult
        pixel_budget 기준 배치 → Gemma4 병렬 호출
        CoT analysis 필드 포함 → logprobs 신뢰도 산출
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
    user_instruction: str = ""    # 태스크 지시 + 1-shot 예시
    json_schema: Optional[dict] = None  # guided_json (CoT analysis 필드 포함)
    pixel_budget: int = 560

# form_type="supply_request", region_type="text":
#   system_prompt: "군수 보급청구서 OCR 시스템입니다."
#   user_instruction: "이 수기 기입란에서 NSN 코드(NNNN-NN-NNN-NNNN)와
#                      수량을 인식하세요. 불확실한 글자는 [?]로 표시.\n
#                      [예시]\n이미지: <example_crop>\n결과: {\"nsn\":\"1005-01-432-1234\",\"qty\":10}"
#
# form_type="other":
#   system_prompt: "문서 OCR 시스템입니다."
#   user_instruction: "이 영역의 텍스트를 인식하세요."
```

### 2-4. ResolutionRouter 출력

```python
@dataclass
class CroppedRegion:
    region_id: str
    region_type: RegionType
    cropped_image: np.ndarray  # bbox 크롭 + 5~15% 패딩 (원본 이미지 맥락 포함)
    pixel_budget: int
    instruction_spec: InstructionSpec

# pixel_budget 기준값 (Gemma4 공식 문서 기반, 2026-04-10 구현 완료)
PIXEL_BUDGET = {
    "table":    1120,   # 셀 경계·미세 글씨 → 최고 해상도
    "seal":      560,   # 원형 배치 텍스트
    "text":      560,   # 일반 텍스트 — 280에서 상향 (소형 폰트·수기 통합 대응)
    "formula":   280,
    "chart":     280,
    "figure":    140,
    "header":    140,   # 대형 텍스트, 저해상도로 충분
    "footer":    140,
}

# v2 RegionType에는 handwritten_field가 의도적으로 없음 (VLM이 내용으로 판별).
# PP-DocLayout은 수기 영역을 text 라벨로 반환하므로, text 상향(280→560)으로 간접 대응.
# Phase 2 PP-DocLayout Fine-tuning 시 handwritten_field/signature/checkbox 카테고리
# 추가 검토 (CLAUDE.md 로드맵 Phase 2-C).

# 크롭 패딩 — 단일 값 사용 (ResolutionRouterConfig.crop_padding, 기본 5px)
# 영역 타입별 차등은 Phase 2-C에서 검토 (양식 레이블·경계선 맥락 포함 필요 시)
```

### 2-5. VLM 통합 추론 결과

```python
@dataclass
class FieldValue:
    field_key: str
    raw_value: str
    corrected_value: str
    data_type: str             # text, number, date, code
    confidence: float          # logprobs 기하평균 기반
    token_logprobs: list[float]
    is_flagged: bool = False
    region_id: Optional[str] = None

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
    PIXEL_BUDGET = 140  # 분류용 저해상도, 비용 최소화

    def classify(
        self, image_rgb: np.ndarray,
        warnings: Optional[list[str]] = None,
    ) -> tuple[FormType, float]:
        """전체 페이지 이미지 → 저해상도 VLM 1회 호출.
        Returns: (form_type, confidence)
        """
        # guided_json enum: supply_request|maintenance_record|...|unknown|other
        # other → _general.json 스키마 사용, 군수 룰 검증 건너뜀
```

### 3-3. P2.5-B InstructionRouter

**파일**: `src/vlm/instruction_router.py`

```python
class InstructionRouter:
    def route(self, region: LayoutRegion, form_type: Optional[str]) -> InstructionSpec:
        """
        military 경로:
          form_type="supply_request", region_type="text"
          → system_prompt: "군수 보급청구서 OCR 시스템입니다."
          → user_instruction: "이 수기 기입란에서 NSN 코드와 수량을 인식하세요.
                               [예시] <1-shot 이미지> → <정답 JSON>"
          → json_schema: supply_request_field_schema (analysis 필드 최상단)

        other 경로:
          form_type="other"
          → system_prompt: "문서 OCR 시스템입니다."
          → user_instruction: "이 영역의 텍스트를 인식하세요."
          → json_schema: _general_schema (범용 key-value)

        form_type=None:
          → user_instruction: "한국어 텍스트를 인식하세요." (FormClassifier 미수행)
        """

# JSON Schema CoT 구조 (모든 군수 서식 스키마에 공통 적용)
# {
#   "analysis": {"type": "string",
#                "description": "영역 텍스트 품질, 모호한 문자 간략 기술 (~30~50 토큰)"},
#   ... 실제 필드들 ...
# }
# analysis 필드를 최상단에 배치 → VLM이 답변 전 사고 과정 거침 → 정확도 향상
```

**1-shot 예시 관리**:
- 서식 유형별 대표 예시 1개를 `configs/instruction_examples/{form_type}.yaml`에 저장
- system_prompt를 정적으로 유지 → prefix caching 효율 극대화
- 가변 콘텐츠(실제 OCR 이미지)는 프롬프트 끝에 배치

### 3-4. P2.5-C ResolutionRouter

**파일**: `src/vlm/resolution_router.py`

```python
class ResolutionRouter:
    def route(
        self, layout: LayoutResult, preprocessed: PreprocessedImage,
        instructions: dict[str, InstructionSpec],
    ) -> dict[int, list[CroppedRegion]]:
        """bbox 크롭 + 패딩 + pixel_budget 할당 + 배치 그룹화.
        반환: {pixel_budget → [CroppedRegion, ...]}
        동일 budget끼리 그룹화 → vLLM 패딩 오버헤드 없음.
        크롭 크기를 48px 배수로 맞춰 SigLIP 패치 효율 최적화.
        """
```

### 3-5. P3-B StructuredExtractor

**파일**: `src/vlm/structured_extractor.py`

```python
class StructuredExtractor:
    def extract(
        self, groups: dict[int, list[CroppedRegion]],
        doc_id: str, form_type: FormType,
        form_confidence: float, schema_id: str,
        schema: Optional[dict] = None,
        warnings: Optional[list[str]] = None,
    ) -> VLMResult:
        """각 pixel_budget 그룹 → vLLM 배치 호출.
        guided_json(CoT 포함) + logprobs → 필드 추출 + 신뢰도 산출.
        도메인 코드 자동 감지 (NSN/K-NSN 패턴).
        """
```

**logprobs 신뢰도 산출**:

```python
def calc_field_confidence(token_logprobs: list[float]) -> float:
    """기하 평균: exp(mean(logprobs))
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
| 일반 텍스트 | 0.90 | 0.65 | 한국어 특성상 영어 대비 5~10% 낮게 설정 |
| 서명 | 0.70 | 0.50 | |

> **other 문서**: 군수 임계값 적용 안 함. 전체 신뢰도만 산출.

### 4-2. P4 룰 검증으로 보정 (military 경로만)

- `합계 ≠ 수량 × 단가` → 관련 필드 신뢰도 **-0.30**
- NSN 형식 불일치 → 해당 필드 신뢰도 **-0.15**
- 날짜 순서 위반 → 해당 필드 신뢰도 **-0.10**

### 4-3. NSN 패턴 강제 (guidance 백엔드 전환)

xgrammar는 JSON Schema 내 `pattern` (regex) 제약을 미지원. NSN 코드 형식 강제 시:

```python
# 방법 1: 별도 guided_regex 호출
extra_body={"guided_regex": r"\d{4}-\d{2}-\d{3}-\d{4}"}

# 방법 2: guidance 백엔드 전환 (--guided-decoding-backend guidance)
# JSON Schema 내 pattern 제약 완전 지원, 복잡한 스키마 타임아웃 없음
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
            logger.error("vLLM 서버 비정상 — fallback 전환 대기")
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
| 서식 분류 | P3-A FormClassifier | DiT 모델 |
| 텍스트 인식 | P3-B 크롭 고해상도 | PP-OCRv5 전체 페이지 |
| 표 구조 | P3-B 1120 토큰 | SLANeXt |
| guided_json | ✅ 구조 보장 | ❌ 규칙 기반 매핑 |
| P4 임계값 | VLM 프로파일 | Fallback 프로파일 (낮음) |
| 결과 | 높은 품질 | 중간 (항상 검토 큐) |

### 5-3. 수준 C — 수동 검토 큐

| 조건 | 우선순위 |
|------|---------|
| CRITICAL ValidationError | 🔴 긴급 |
| LOW confidence 필드 | 🟡 일반 |
| Fallback 경로 처리 | 🟡 일반 |
| VLM + Fallback 모두 불가 | 🔴 긴급 |

> **other 문서는 검토 큐 미적재** — 군수 업무 대상이 아니므로 별도 처리.

---

## 6. 서비스 통신 아키텍처

### 6-1. 현재 구조 (Phase 1)

```
[Pipeline 컨테이너 :8080]
  ├── P1 Preprocessor          (in-process)
  ├── P2 LayoutAnalyzer        HTTP POST → [Layout :8082]
  │     └── PP-DocLayoutV3 / Fusion 모드
  ├── P2.5-A LayoutPostProcessor (in-process)
  ├── P3-A FormClassifier      HTTP POST → [vLLM :8100]
  ├── P2.5-B InstructionRouter (in-process)
  ├── P2.5-C ResolutionRouter  (in-process)
  ├── P3-B StructuredExtractor HTTP POST → [vLLM :8100]
  ├── Fallback                 HTTP POST → [Fallback :8081]
  ├── P4 Validator             (in-process)
  ├── P5 Serializer            (in-process)
  └── P6 DBLoader              (in-process, SQLite)
```

**Layout 분리 사유**: PaddlePaddle ↔ PyTorch CUDA 충돌 방지. 전체 CUDA 12.6 통일.

### 6-2. Phase 3 목표: 비동기 메시지 기반

- pipeline → fallback: Redis Streams 비동기 전환
- 검토 큐 알림: Redis Pub/Sub → 프론트엔드 SSE

---

## 7. 코딩 컨벤션

- 모든 함수·클래스에 docstring 필수
- 컴포넌트 간 데이터 전달은 `src/interfaces/types.py`의 dataclass 사용
- 오케스트레이터(`pipeline/orchestrator.py`)만 컴포넌트를 순서대로 호출
- `TASK_PROMPTS` 딕셔너리 `layout_analyzer.py`/`types.py`에 두지 않음 — `InstructionRouter` 일원화
- VLM instruction/response는 **반드시 guided_json으로 구조화**
- `VLMResult.raw_json`은 DB 적재 제외 — `P3B/raw_vlm_responses/`에만 보존
- P4는 `processing_path` + `form_type(other 여부)` 기반으로 임계값·룰 프로파일 분기
- Python 3.10+

**모듈 책임 경계**:

| 모듈 | 책임 | 하지 않는 것 |
|------|------|-------------|
| `vlm_client.py` | vLLM HTTP 통신, 이미지 base64 인코딩, logprobs 파싱 | 이미지 크롭, 도메인 코드 감지 |
| `resolution_router.py` | bbox 크롭 + 패딩, pixel_budget 할당, 배치 그룹화 | VLM 통신 |
| `structured_extractor.py` | 배치 VLM 호출, 필드 추출, 도메인 코드 감지, VLMResult 조립 | 이미지 크롭 |
| `form_classifier.py` | 서식 분류 VLM 1회 호출 (military/other 분류 포함) | 필드 추출, 크롭 |

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

# Stage 2 — VLM
vllm>=0.19.0
torch
transformers>=5.5.0

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
