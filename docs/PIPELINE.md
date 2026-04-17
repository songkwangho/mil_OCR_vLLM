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
    P2.5-A: LayoutPostProcessor → LayoutResult
        ① 미소 박스 제거 (6px 미만)
        ② 중복 박스 제거 (IoU > 0.7, seal은 0.5)
        ③ 인접 텍스트 블록 병합
        ④ reading_order 재정렬
    │
    ▼
[Stage 2 — VLM 통합 추론 — form_type 기반 분기]
    P3-A: FormClassifier → FormType
        전체 페이지 저해상도(140토큰) → form_type 확정

    ┌─────────────────────────────────────────────────────┐
    │ military 경로                                        │
    │                                                     │
    │  P2.5-A.5: TemplateAugmentor  ← 신규               │
    │    form_type별 서식 템플릿 로드                       │
    │    configs/form_templates/*.yaml                     │
    │    PP-DocLayout 결과와 병합                           │
    │    IoU > 0.5: 탐지 결과 우선                         │
    │    미탐지 필드: 템플릿 bbox 보완                      │
    │                                                     │
    │  P2.5-B: InstructionRouter                          │
    │    region_type + form_type → instruction + 1-shot  │
    │    군수 JSON Schema 선택, OCR 힌트 삽입             │
    │                                                     │
    │  P2.5-C: ResolutionRouter                           │
    │    배치 순서: 140→560→1120 토큰 그룹 순차 배치       │
    │    48px 배수 정렬, 패딩 포함 크롭                   │
    │                                                     │
    │  P3-B: StructuredExtractor                          │
    │    temperature=0.0, guided JSON, logprobs           │
    │    저신뢰 필드 재시도, S4/S6 Skill 통합 호출        │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │ other 경로 (Skill Registry)                         │
    │                                                     │
    │  S1: LayoutAnalyzer (결재란 휴리스틱 포함)           │
    │    + 조건부 Mode B (table 미탐지 시 VLM 전체 페이지)│
    │                                                     │
    │  S5-패스1: TableExtractor (표 구조만 추출)           │
    │    → 셀 좌표 + content_type 반환                    │
    │                                                     │
    │  pixel_budget 그룹별 순차 배치:                      │
    │    S6(140): SignatureDetector (이진 분류)            │
    │    S2(560): PrintedTextReader                       │
    │    S3+S4+S5패스2(1120):                             │
    │      HandwritingReader + SealReader + 셀 내용       │
    │                                                     │
    │  S7: StructuredAggregator                           │
    │    official_document.json Schema + guided JSON      │
    └─────────────────────────────────────────────────────┘
    │
    ▼
[Stage 3 — 후처리]
    P4: 룰 검증 + 신뢰도 보정 → ValidatedResult
        military: 산술/날짜/코드 교차검증
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
    raw_bytes: bytes           # PDF의 경우 PDF 원본 바이트 그대로
    file_ext: FileExt          # FileExt.pdf 포함
    source_type: SourceType
    dpi_hint: Optional[int] = None   # PDF 렌더링 DPI 힌트로 재활용
    metadata: dict = field(default_factory=dict)

@dataclass
class PreprocessedImage:
    doc_id: str
    image_array: np.ndarray  # H×W×3 uint8 (RGB) — 이진화 금지, RGB 유지
    binary_array: np.ndarray
    dpi: int
    original_dpi: int
    resolution_band: DpiResolutionBand
    quality_score: float
    sr_applied: bool = False
    warnings: list[str] = field(default_factory=list)

@dataclass
class PageImage:
    """PdfAdapter가 반환하는 단일 페이지 이미지."""
    doc_id: str
    page_number: int           # 1-based
    total_pages: int
    image_array: np.ndarray    # H×W×3 uint8 (RGB)
    render_dpi: int
    original_width_pt: float   # PDF 원본 너비 (포인트, 1pt = 1/72 inch)
    original_height_pt: float
    warnings: list[str] = field(default_factory=list)

@dataclass
class PdfDocumentResult:
    """멀티페이지 PDF 처리 결과 — 페이지별 PipelineOutput 집합.

    overall_status 집계:
        success → 모든 페이지 success
        partial → 일부 페이지 failed/review
        failed  → 전체 실패 또는 PdfAdapter 렌더링 실패
    """
    doc_id: str                      # 원본 PDF doc_id
    total_pages: int
    pages: list[PipelineOutput]      # doc_id = "{원본}_p{N:02d}"
    page_results: list[Any]          # 원시 페이지별 결과 (서버 응답 및 디버깅용)
    overall_status: PipelineStatus
    processing_ms: float
    warnings: list[str] = field(default_factory=list)
```

**PDF doc_id 명명 규칙**: `"{원본doc_id}_p{page:02d}"` — 예) `전비품확인서_001_p01`
DB에서 같은 PDF의 모든 페이지 조회: `WHERE parent_doc_id = '전비품확인서_001'`

### 2-2. 레이아웃 분석

```python
@dataclass
class BoundingBox:
    x1: int; y1: int; x2: int; y2: int

@dataclass
class LayoutRegion:
    region_id: str
    region_type: RegionType  # text, table, figure, header, footer, seal, handwritten_field, signature
    bbox: BoundingBox
    confidence: float
    polygon: Optional[list[tuple[float, float]]] = None
    source: str = "model"             # "model" | "template" | "template_matched"
    field_key: Optional[str] = None   # TemplateAugmentor 부여. Assembler가 region→schema 역참조에 사용

@dataclass
class RawLayoutResult:
    doc_id: str
    page_width: int
    page_height: int
    regions: list[LayoutRegion]
    reading_order: list[int]
    analysis_mode: AnalysisMode = AnalysisMode.HEURISTIC
    warnings: list[str] = field(default_factory=list)

@dataclass
class LayoutResult:
    doc_id: str
    page_width: int
    page_height: int
    regions: list[LayoutRegion]
    reading_order: list[int]
    analysis_mode: AnalysisMode = AnalysisMode.HEURISTIC
    removed_count: int = 0
    merged_count: int = 0
    augmented_count: int = 0   # TemplateAugmentor가 추가한 영역 수
    warnings: list[str] = field(default_factory=list)
```

> **`source` 필드**: TemplateAugmentor가 추가한 영역은 `source="template"`으로 표시합니다.
> P3-B 추출 결과에서 template 출처 영역의 신뢰도가 낮으면 해당 서식 템플릿을 재검토해야 한다는 신호입니다.

### 2-3. Skill 공통 인터페이스 (other 경로)

```python
@dataclass
class SkillTask:
    region_id: str
    region_type: RegionType
    cropped_image: np.ndarray
    pixel_budget: int
    context: str = ""
    form_type: Optional[FormType] = None

@dataclass
class SkillResult:
    region_id: str
    skill_name: str
    content: str
    confidence: float
    content_type: str
    raw_response: str = ""
    warnings: list[str] = field(default_factory=list)

@dataclass
class TableStructure:
    region_id: str
    table_type: str
    cells: list[TableCell]
    structure_confidence: float

@dataclass
class TableCell:
    row: int
    col: int
    role: str
    bbox: BoundingBox
    content_type: str

@dataclass
class SealProcessResult:
    image: np.ndarray
    unwrapped: bool
    context_hint: str
```

### 2-4. InstructionRouter 출력 (military 경로)

```python
@dataclass
class InstructionSpec:
    region_id: str
    region_type: RegionType
    form_type: Optional[FormType] = None
    system_prompt: str = ""
    user_instruction: str = ""
    json_schema: Optional[dict] = None
    pixel_budget: int = 560
    ocr_hint: Optional[str] = None
    is_retry: bool = False
    field_key: Optional[str] = None   # TemplateAugmentor 부여. sub-schema 분해/Assembler 역참조용
```

### 2-5. ResolutionRouter 출력 (military 경로)

```python
@dataclass
class CroppedRegion:
    region_id: str
    region_type: RegionType
    cropped_image: np.ndarray
    pixel_budget: int
    instruction_spec: InstructionSpec

PIXEL_BUDGET = {
    "table":             1120,
    "handwritten_field": 1120,
    "seal":               560,
    "text":               560,
    "formula":            280,
    "chart":              280,
    "figure":             140,
    "header":             140,
    "footer":             140,
    "signature":          140,
}

DISPATCH_ORDER = [140, 560, 1120]

CROP_PADDING_RATIO = {
    "table":             0.05,
    "handwritten_field": 0.15,
    "seal":              0.10,
    "text":              0.05,
    "signature":         0.10,
    "default":           0.05,
}
```

### 2-6. VLM 통합 추론 결과

```python
@dataclass
class FieldValue:
    field_key: str
    raw_value: str
    corrected_value: str
    data_type: str
    confidence: float
    token_logprobs: list[float]
    is_flagged: bool = False
    region_id: Optional[str] = None
    was_retried: bool = False

@dataclass
class VLMResult:
    doc_id: str
    form_type: FormType
    form_confidence: float
    schema_id: str
    fields: list[FieldValue]
    tables: list[RecognizedTable]
    domain_codes: list[DomainCode]
    processing_path: ProcessingPath
    processing_time_ms: float = 0.0
    retry_count: int = 0
    warnings: list[str] = field(default_factory=list)
    assembled_json: Optional[dict] = None  # Assembler 조립 결과 (x-assembly-rules 있는 서식)

@dataclass
class ValidatedResult:
    """P4 출력 — 룰 검증 + 신뢰도 보정 결과. P5가 진실의 원천으로 assembled_json을 사용."""
    doc_id: str
    fields: list[FieldValue]            # region_id/was_retried 포함된 보정 버전
    tables: list[RecognizedTable]
    validation_errors: list[ValidationError]
    overall_confidence: float
    review_required: bool
    flagged_fields: list[str]
    processing_path: ProcessingPath = ProcessingPath.VLM
    assembled_json: Optional[dict] = None  # VLMResult로부터 승계
```

### 2-7. 후처리 결과

```python
@dataclass
class CorrectedField:
    field_path: str
    original_value: str
    corrected_value: str
    corrected_by: str
    corrected_at: datetime

@dataclass
class ReviewQueueItem:
    queue_id: str
    doc_id: str                            # PDF 페이지의 경우 "{원본}_p{N:02d}"
    enqueued_at: datetime
    priority: ReviewPriority
    reason: ReviewReason
    processing_path: ProcessingPath
    validated_result: Optional[ValidatedResult] = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    flagged_fields: list[str] = field(default_factory=list)
    original_image_path: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    corrected_fields: dict[str, CorrectedField] = field(default_factory=dict)
    reviewer_notes: str = ""
    # PDF 멀티페이지 추적용 (단일 이미지 입력 시 None)
    parent_doc_id: Optional[str] = None   # 원본 PDF doc_id
    page_number: Optional[int] = None     # 페이지 번호
    total_pages: Optional[int] = None     # 전체 페이지 수

@dataclass
class PipelineOutput:
    doc_id: str
    status: PipelineStatus   # PARTIAL: PDF 멀티페이지 일부 실패/검토
    processing_path: ProcessingPath
    form_type: Optional[FormType] = None
    json_output: Optional[str] = None
    xml_output: Optional[str] = None
    csv_rows: list[dict] = field(default_factory=list)
    db_record_ids: list[str] = field(default_factory=list)
    review_queue_id: Optional[str] = None
    processing_ms: float = 0.0
```

`PipelineStatus` enum:

```python
class PipelineStatus(str, Enum):
    SUCCESS        = "success"
    PARTIAL        = "partial"        # PDF 멀티페이지 중 일부 실패/검토
    REVIEW         = "review"
    FAILED         = "failed"
    OTHER_DOCUMENT = "other_document"
```

`FileExt` enum — pdf 포함:

```python
class FileExt(str, Enum):
    jpg  = "jpg"
    jpeg = "jpeg"
    png  = "png"
    tiff = "tiff"
    pdf  = "pdf"
```

---

## 3. 컴포넌트 상세 설계 — military 경로

### 3-1. P2.5-A LayoutPostProcessor

**파일**: `src/preprocess/layout_postprocessor.py`

```python
class LayoutPostProcessor:
    MICRO_BOX_PX = 6
    OVERLAP_IOU_THRESHOLD = 0.7
    SEAL_IOU_THRESHOLD = 0.5

    def process(self, raw: RawLayoutResult) -> LayoutResult: ...

    def _remap_reading_order(
        self, original_order, removed_ids, merged_map
    ) -> list[int]:
        """제거/병합 결과 반영하여 reading_order 재정렬."""
```

### 3-2. P3-A FormClassifier

**파일**: `src/vlm/form_classifier.py`

| FormType | 의미 | 처리 경로 |
|----------|------|---------|
| `supply_request` | 보급청구서 | military |
| `maintenance_record` | 정비기록서 | military |
| `inventory_sheet` | 물자현황표 | military |
| `handover_doc` | 인수인계서 | military |
| `inspection_report` | 검사보고서 | military |
| `equipment_checklist` | 전비품 확인서 작성 점검표 | military |
| `unknown` | 군수 서식인데 유형 불명 | military (_fallback.json) |
| `other` | 군수 서식이 아님 | other → Skill Registry |

```python
class FormClassifier:
    PIXEL_BUDGET = 140

    def classify(
        self, image_rgb, warnings=None
    ) -> tuple[FormType, float, Optional[str]]:
        """Returns: (form_type, confidence, form_identifier)

        form_identifier: 서식 상단의 식별자 문자열 (예: "별지 제3-2호 서식").
        TemplateAugmentor가 버전 선택에 사용.
        식별자가 없는 서식은 None 반환.

        guided_json enum: supply_request|...|equipment_checklist|unknown|other
        form_identifier: optional string
        temperature=0.0
        """
```

### 3-3. P2.5-A.5 TemplateAugmentor (신규, military 전용)

**파일**: `src/vlm/template_augmentor.py`

**역할**: P3-A가 form_type을 확정한 직후, 서식 템플릿(`configs/form_templates/{form_type}.yaml`)의 필드 bbox를 LayoutResult에 병합하고 **각 region에 `field_key`를 부여**합니다. 이 field_key가 P2.5-B에서 sub-schema 분해를 트리거하고, P3-B 이후 Assembler가 region별 결과를 재조립하는 키가 됩니다.

**삽입 위치**: 오케스트레이터에서 P3-A → P2.5-A.5 → P2.5-B 순서로 호출.

**v3 병합 정책 (포함도 기반)** — 단순 IoU가 아닌 양방향 포함도(containment)와 보호 영역 규칙으로 PP-DocLayout 결과를 흡수·보존합니다.

```
각 템플릿 field f에 대해:
  consumable = []
  for each PP region p:
      if p.region_type in {seal, signature, figure, table}: continue   # 보호
      c_pp,  c_tpl = containment(p, f)        # = inter / area(p), inter / area(f)
      if c_tpl > 0.9: continue                # 템플릿이 PP에 내포 → PP는 컨테이너
      if c_pp > 0.7:  consumable.append(p)    # PP 70%+ 가 f 안 → 흡수

  consume(consumable)
  if len(consumable) == 1 and not _is_small_contained(p, f):
      # 1:1 정합 → PP bbox 재사용 (실측 좌표가 더 정확)
      emit LayoutRegion(bbox=p.bbox, source="template_matched", field_key=f.field_key)
  else:
      # 0개 또는 다중 매칭 → 템플릿 bbox로 단일 region (대표값)
      emit LayoutRegion(bbox=f.bbox, source="template", field_key=f.field_key)

# Step 2: 흡수되지 않은 PP 영역(인장·서명 등 동적)은 그대로 보존 (field_key=None)
```

```python
class TemplateAugmentor:
    """서식 템플릿 기반 region 재구성 — military 전용."""

    CONTAINMENT_THRESHOLD = 0.7         # PP가 템플릿에 ≥70% 포함되면 흡수
    TEMPLATE_INSIDE_PP_THRESHOLD = 0.9  # 템플릿이 PP에 ≥90% 내포되면 PP는 컨테이너
    PROTECTED = {"seal", "signature", "figure", "table"}

    def __init__(self, templates_dir: str = "configs/form_templates"):
        self._dir = Path(templates_dir)
        self._cache: dict[str, list[dict]] = {}

    def augment(
        self,
        layout: LayoutResult,
        form_type: FormType,
        form_identifier: Optional[str] = None,
        stats: Optional[TemplateAugmentorStats] = None,
    ) -> LayoutResult:
        """위 정책 박스에 따라 LayoutResult를 재구성.

        반환된 LayoutResult.regions 순서:
          ① 템플릿 field마다 region (field_key 부여, source="template" or "template_matched")
          ② 흡수되지 않은 PP-DocLayout 동적 region (field_key=None — 인장·서명·표 등)

        본문 구현은 src/vlm/template_augmentor.py 참조.
        """
        ...

    def _load_template(
        self,
        form_type: FormType,
        form_identifier: Optional[str] = None,
    ) -> list[dict]:
        """configs/form_templates/{form_type.value}.yaml 로드 후 버전 선택 (캐싱).

        versions 배열 구조:
          versions:
            - version: "1.0"
              form_identifier: "별지 제3-2호 서식"
              fields: [...]
            - version: "2.0"
              form_identifier: "별지 제3-2호의2 서식"
              fields: [...]

        form_identifier 매칭 우선, 미매칭/None 시 첫 번째 버전 사용.
        """
        cache_key = f"{form_type.value}::{form_identifier or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._dir / f"{form_type.value}.yaml"
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        versions = data.get("versions")
        if versions:
            # versions 배열 구조 — form_identifier로 버전 선택
            selected = versions[0]  # 기본: 첫 번째(최신)
            if form_identifier:
                for v in versions:
                    if v.get("form_identifier") == form_identifier:
                        selected = v
                        break
            fields = selected.get("fields", [])
        else:
            # 하위 호환: 기존 flat 구조 (supply_request 등)
            fields = data.get("fields", [])

        self._cache[cache_key] = fields
        return fields
```

**템플릿 YAML 형식** (`configs/form_templates/supply_request.yaml`):

```yaml
# supply_request.yaml
# 보급청구서 v1.3 기준 (2026-01-01 발효)
# 좌표계: P1 Deskew + DPI 300 정규화 이후 기준
# 스캔 크기: A4 세로 (2480×3508px at 300dpi)

form_type: supply_request
version: "1.3"
effective_date: "2026-01-01"
page_size: [2480, 3508]

fields:
  - field_key: unit_code
    region_type: text
    bbox: {x1: 120, y1: 340, x2: 380, y2: 390}
    description: "청구 부대 코드"

  - field_key: request_date
    region_type: text
    bbox: {x1: 400, y1: 340, x2: 700, y2: 390}
    description: "청구 일자"

  - field_key: nsn
    region_type: handwritten_field
    bbox: {x1: 120, y1: 410, x2: 520, y2: 460}
    description: "NSN 코드 기입란"

  - field_key: item_name
    region_type: text
    bbox: {x1: 120, y1: 340, x2: 860, y2: 390}
    description: "품목명"

  - field_key: quantity
    region_type: handwritten_field
    bbox: {x1: 540, y1: 410, x2: 680, y2: 460}
    description: "수량 기입란"

  - field_key: unit_price
    region_type: handwritten_field
    bbox: {x1: 690, y1: 410, x2: 860, y2: 460}
    description: "단가 기입란"

  - field_key: total
    region_type: handwritten_field
    bbox: {x1: 870, y1: 410, x2: 1060, y2: 460}
    description: "합계 기입란"
```

**주의 사항**:

```
1. 좌표 기준: P1 Deskew + DPI 300 정규화 이후 이미지 기준.
   스캔 기울기(±3~5°)가 보정된 후의 좌표여야 함.

2. 서식 버전 관리: version + effective_date 필수 명시.
   서식 개정 시 v2/ 하위 디렉토리에 신규 작성하고
   schema_registry.py 방식으로 form_type:version 매핑 추가.

3. 동적 요소 제외: 인장(seal), 서명(signature) 영역은 템플릿에 넣지 않음.
   위치가 유동적이므로 PP-DocLayout 탐지 결과를 그대로 사용.

4. unknown form_type: 템플릿 없음 → augment() 호출 시 원본 반환.
```

### 3-4. P2.5-B InstructionRouter (military 전용)

**파일**: `src/vlm/instruction_router.py`

```python
class InstructionRouter:
    def route(self, region, form_type, ocr_hint=None) -> InstructionSpec:
        """
        military + supply_request + text 예시:
          system_prompt: "군수 보급청구서 OCR 시스템입니다."
          user_instruction:
            "이 수기 기입란에서 NSN 코드와 수량을 인식하세요.
             NSN 형식: NNNN-NN-NNN-NNNN. 불확실한 글자: [?].
             유사 문자 주의: ㄱ/ㅋ, 1/ㅣ, 0/O
             [예시] <1-shot 이미지> → <정답 JSON>"
        """
```

**프롬프트 설계 원칙**:
- system_prompt 정적 유지 → prefix caching 극대화
- 가변 콘텐츠는 프롬프트 끝에 배치
- analysis 필드 JSON Schema 최상단 → CoT 효과
- 영어 지시문 + 한국어 필드명 하이브리드

#### 3-4-1. Sub-schema 분해 (`_extract_sub_schema`)

`x-assembly-rules`가 정의된 스키마(예: `equipment_checklist`)는 region마다 **full schema의 일부만** guided_json으로 전달합니다. 이로써 1-shot 예시의 full-schema 출력 패턴이 sub-schema 디코딩을 오염시키지 않고, 각 크롭이 자기 책임 영역만 채웁니다.

| field_key | 전달되는 sub-schema | 비고 |
|---|---|---|
| `result_item_3` | `x-checklist-item-schema` + `item_number: const=3` | VLM이 번호를 헷갈리지 않도록 const 주입 |
| `writer_block` | `properties.writer` (team/rank/name/signature_present) | object 그대로 |
| `document_date` | `{type: string}` | 문자열 리터럴만 출력 |
| `form_identifier` | `{type: string}` | 동상 |

field_key가 부여된 region에는 full-shot 1-shot 대신 `_build_field_key_instruction()`이 만든 **field_key 전용 짧은 지시문**(스키마 본문을 그대로 첨부)이 사용됩니다.

### 3-5. P2.5-C ResolutionRouter (military 전용)

**파일**: `src/vlm/resolution_router.py`

```python
class ResolutionRouter:
    DISPATCH_ORDER = [140, 560, 1120]

    def route(self, layout, preprocessed, instructions) -> dict[int, list[CroppedRegion]]:
        """반환: {pixel_budget → [CroppedRegion]}
        동일 budget끼리 그룹화, DISPATCH_ORDER 순서로 vLLM에 전송.
        template 출처 영역도 동일하게 처리됨.
        """

    def _align_to_48px(self, image) -> np.ndarray:
        h, w = image.shape[:2]
        new_h = max(48, round(h / 48) * 48)
        new_w = max(48, round(w / 48) * 48)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
```

**크롭 패딩 — 비율 + 절대 상한**: `pad = min(MAX_PX[rt], max(MIN_PX, int(bbox_w × RATIO[rt])))`. 큰 bbox(예: table 1854×1950)에 단순 비율(5%=92px)을 적용하면 인접 영역(writer 등)을 침범하므로 영역별 절대 상한을 둡니다. 기본값: `table/figure/chart=30px`, `seal=40px`, `text/header/footer=16px`, `default=24px`.

### 3-6. P3-B StructuredExtractor (military 전용)

**파일**: `src/vlm/structured_extractor.py`

```python
class StructuredExtractor:
    RETRY_THRESHOLD = 0.60
    MAX_RETRIES = 1
    RETRY_BUDGET_MAP = {140: 280, 280: 560, 560: 1120, 1120: 1120}

    def extract(self, groups, doc_id, form_type, ...) -> VLMResult:
        """DISPATCH_ORDER 순서로 pixel_budget 배치 전송.
        template 출처 영역도 동일 흐름으로 처리.
        저신뢰 필드 감지 시 pixel_budget 상향 + OCR 힌트 재시도.
        seal/signature 영역은 S4/S6 Skill로 위임.
        """
```

**field_key blob 보존**: `spec.field_key`가 있으면 VLM 출력 dict/list/스칼라 **전체를 단일 FieldValue로 저장**(`field_key=spec.field_key, region_id=cropped.region_id`)합니다. JSON 키 단위로 분해하지 않으므로 sub-schema 응답이 그대로 Assembler로 전달됩니다. 재시도 시 `field_key`를 신규 InstructionSpec에 그대로 전파합니다.

### 3-7. Assembler (military 전용 — `x-assembly-rules` 정의 서식)

**파일**: `src/vlm/assembler.py`

`extract()` 종료 시 `schema.x-assembly-rules`가 있으면 region별 FieldValue를 full schema dict로 조립해 `VLMResult.assembled_json`에 부착합니다.

```python
class Assembler:
    def assemble(self, fields, schema, region_field_key_map, warnings) -> dict | None:
        # 1) region_id → field_key 역참조로 field_map 구성 (confidence 높은 결과 우선)
        # 2) x-assembly-rules: {field_key → "checklist_items.0" 같은 JSON Path}
        # 3) _set_path()로 nested dict에 값 삽입
        # 4) checklist_items 6개 누락 자동 보완 + alias 매핑(item_result/handwritten_mark → result)
        ...
```

**주요 동작**:
- VLM이 `[{...}]`로 1원소 리스트 감싼 경우 자동 언래핑
- `item_number=null`은 인덱스 기반 보정
- `checklist_items` 6개 미달 시 `{"item_number": i+1, "result": "?", "result_confidence": 0.0}` 기본값 삽입
- `x-assembly-rules`가 없는 스키마는 None 반환 → 기존 `fields[]` 기반 처리 유지 (하위 호환)

P4 검증은 `assembled_json` 우선, 없으면 기존 `raw_json` 파싱 폴백.

---

## 4. 컴포넌트 상세 설계 — other 경로 (Skill Registry)

### 4-1. SealPreprocessor (양 경로 공통)

**파일**: `src/preprocess/seal_preprocessor.py`

```python
class SealPreprocessor:
    def process(self, seal_crop: np.ndarray) -> SealProcessResult:
        # 1. HSV 색공간에서 적색 마스크 추출
        # 2. 허프 원 변환으로 인장 경계 탐지
        # 허프 실패 → 원본 크롭 반환 (예외 발생 금지)
        # 3. 극좌표 → 직교좌표 변환 (곡선 텍스트 직선화)
```

### 4-2. Skill Registry

**파일**: `src/vlm/skill_registry.py`

```python
SKILL_ROUTING = {
    "text":              "S2",
    "header":            "S2",
    "footer":            "S2",
    "handwritten_field": "S3",
    "seal":              "S4",
    "table":             "S5",
    "signature":         "S6",
}
```

### 4-3. S2 PrintedTextReader

**파일**: `src/vlm/skills/printed_text_reader.py`

```python
class PrintedTextReader:
    PIXEL_BUDGET = 560
    # guided_json 없이 순수 텍스트 추출
    # 이유: 인쇄 텍스트는 스키마 강제보다 정확한 텍스트 추출이 우선
```

### 4-4. S3 HandwritingReader

**파일**: `src/vlm/skills/handwriting_reader.py`

```python
class HandwritingReader:
    PIXEL_BUDGET = 1120
    # context: S5 패스1이 제공한 셀 역할
    # 신뢰도 < 0.70 → 동일 영역 재호출 (다른 프롬프트 변형)
```

### 4-5. S4 SealReader

**파일**: `src/vlm/skills/seal_reader.py`

```python
class SealReader:
    def run(self, task: SkillTask) -> SkillResult:
        seal_result = self.seal_preprocessor.process(task.cropped_image)
        pixel_budget = 1120 if seal_result.unwrapped else 560
```

### 4-6. S5 TableExtractor (2패스)

**파일**: `src/vlm/skills/table_extractor.py`

```python
class TableExtractor:
    def pass1_structure(self, table_crop: np.ndarray) -> TableStructure:
        """패스1: 표 구조만 추출. pixel_budget=1120."""

    def pass2_route(self, structure: TableStructure,
                    original_image: np.ndarray) -> list[SkillTask]:
        """패스2: 셀별 크롭 + content_type에 맞는 Skill 태스크 반환."""
```

### 4-7. S6 SignatureDetector

**파일**: `src/vlm/skills/signature_detector.py`

```python
class SignatureDetector:
    PIXEL_BUDGET = 140
    # 서명 OCR이 아닌 존재 여부 이진 탐지
    # temperature=0.0, guided_json (이진 스키마)
```

### 4-8. S7 StructuredAggregator

**파일**: `src/vlm/skills/aggregator.py`

```python
class StructuredAggregator:
    """S2~S6 결과를 official_document.json Schema에 맞춰 최종 JSON 조립."""

    def run(self, skill_results, table_structure, form_type) -> VLMResult:
        # guided_json: official_document.json
        # low_confidence_fields: confidence < 임계값인 필드 목록
```

---

## 5. 오케스트레이터 디스패치 흐름

### 5-1. military 경로 (TemplateAugmentor 포함)

```python
# orchestrator.py (발췌)
def _process_military_document(self, layout, preprocessed, doc_id,
                                form_type, form_confidence, ...):
    # P2.5-A.5: TemplateAugmentor — PP-DocLayout 누락 보완
    augmentor = self._get_template_augmentor()
    layout = augmentor.augment(
        layout, form_type,
        form_identifier=form_identifier,   # FormClassifier 추출값 전달
        warnings=result.warnings,
    )
    # augmented_count > 0 이면 로그에 보완된 영역 수 기록

    # P2.5-B: InstructionRouter
    instructions = self._get_p2_5b().route_all(layout, form_type)

    # P2.5-C: ResolutionRouter
    groups = self._get_p2_5c().route(layout, preprocessed, instructions)

    # P3-B: StructuredExtractor
    return self._get_p3b().extract(groups, doc_id, form_type, ...)
```

### 5-2. other 경로 (Skill Registry)

```python
def _process_other_document(self, layout, preprocessed, doc_id, ...):
    registry = self._get_skill_registry()
    stats = SkillDispatchStats()

    cell_results, structures = registry.dispatch_tables(layout, preprocessed, stats)
    non_table_tasks = registry.build_tasks(layout, preprocessed)
    general_results = registry.dispatch(non_table_tasks, stats=stats)

    return VLMResult(
        form_type=FormType.OTHER,
        processing_path=ProcessingPath.SKILL_REGISTRY,
        fields=_skill_results_to_fields(general_results + cell_results),
        ...
    )
```

---

## 6. logprobs 신뢰도 임계값

### 6-1. 필드 유형별 임계값

| 필드 유형 | VLM 임계값 | Fallback 임계값 | 비고 |
|----------|:----------:|:--------------:|------|
| 금액 (amount) | 0.99 | 0.80 | |
| 코드 (code/NSN) | 0.97 | 0.75 | |
| 날짜 (date) | 0.95 | 0.70 | |
| 수량 (quantity) | 0.95 | 0.70 | |
| 일반 텍스트 | 0.90 | 0.65 | 한국어 특성상 영어 대비 5~10% 낮게 |
| 수기 (handwritten) | 0.75 | 0.50 | |
| 인장 텍스트 (seal) | 0.70 | 0.50 | |
| 서명 (signature) | — | — | 이진 탐지, confidence만 |

> **재시도 임계값**: 0.60 이하 → pixel_budget 상향 + 1회 재시도
> **template 출처 영역**: 동일 임계값 적용. 낮은 신뢰도가 반복되면 템플릿 좌표 재검토 신호.

### 6-2. P4 룰 검증 보정 (military 경로만)

- `합계 ≠ 수량 × 단가` → 관련 필드 신뢰도 **-0.30**
- NSN 형식 불일치 → 신뢰도 **-0.15**
- 날짜 순서 위반 → 신뢰도 **-0.10**

### 6-3. NSN 패턴 강제 전략

```
1순위: InstructionRouter 프롬프트 힌트 (기본)
   "NSN 코드는 NNNN-NN-NNN-NNNN 형식 13자리입니다."

2순위: guided_regex
   extra_body={"guided_regex": r"\d{4}-\d{2}-\d{3}-\d{4}"}

3순위: guidance 백엔드 전환
   --structured-outputs-config '{"backend":"guidance"}'
```

---

## 7. SPOF 대비 — 가용성 설계

### 7-1. 수준 A — vLLM 헬스체크

```python
class VLMHealthMonitor:
    def _on_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._healthy = False
```

### 7-2. 수준 B — 경량 Fallback

```python
class FallbackPolicy:
    def decide(self, vlm_healthy, fallback_enabled, fallback_healthy) -> str:
        if vlm_healthy:                              return "vlm"
        if fallback_enabled and fallback_healthy:    return "fallback"
        return "review_queue"
```

### 7-3. 수준 C — 수동 검토 큐

| 조건 | 우선순위 |
|------|---------|
| CRITICAL ValidationError | 🔴 긴급 |
| LOW confidence (재시도 후에도 임계값 미달) | 🟡 일반 |
| Fallback 경로 처리 | 🟡 일반 |
| VLM + Fallback 모두 불가 | 🔴 긴급 |

> **other 문서는 검토 큐 미적재** — 군수 업무 대상 아님.

---

## 8. 서비스 통신 아키텍처

### 8-1. 현재 구조 (Phase 1)

```
[Pipeline 컨테이너 :8080]
  ├── PdfAdapter                    (in-process) ← PDF 입력 시 P1 앞단 실행
  ├── P1 Preprocessor               (in-process)
  ├── P2 LayoutAnalyzer             HTTP POST → [Layout :8082]
  ├── P2.5-A LayoutPostProcessor    (in-process)
  ├── SealPreprocessor              (in-process)
  ├── P3-A FormClassifier           HTTP POST → [vLLM :8100]
  │
  ├── [military 경로]
  │   ├── P2.5-A.5 TemplateAugmentor  (in-process)
  │   ├── P2.5-B InstructionRouter    (in-process)
  │   ├── P2.5-C ResolutionRouter     (in-process)
  │   └── P3-B StructuredExtractor   HTTP POST → [vLLM :8100]
  │
  ├── [other 경로]
  │   ├── Skill Registry            (in-process)
  │   ├── S2~S6 Skills              HTTP POST → [vLLM :8100]
  │   └── S7 StructuredAggregator   HTTP POST → [vLLM :8100]
  │
  ├── Fallback                      HTTP POST → [Fallback :8081]
  ├── P4 Validator                  (in-process)
  ├── P5 Serializer                 (in-process)
  └── P6 DBLoader                   (in-process, SQLite)
       ↳ PDF 페이지: parent_doc_id, page_number, total_pages 컬럼 기록
```

---

## 9. 코딩 컨벤션

- 모든 함수·클래스에 docstring 필수
- 컴포넌트 간 데이터 전달은 `src/interfaces/types.py`의 dataclass 사용
- 오케스트레이터만 컴포넌트를 순서대로 호출
- **military 경로**: P3-A → TemplateAugmentor → InstructionRouter → ResolutionRouter → StructuredExtractor 순서 고정
- **other 경로**: SKILL_ROUTING으로 라우팅, S7에서만 form_type 사용
- 모든 VLM 호출에 `temperature=0.0`
- `VLMResult.raw_json`은 DB 적재 제외
- ResolutionRouter 배치 순서: `DISPATCH_ORDER = [140, 560, 1120]`
- TemplateAugmentor: unknown form_type 수신 시 원본 반환 (예외 발생 금지)
- TemplateAugmentor: `source="template"` 영역은 InstructionRouter에서 동일하게 처리
- SealPreprocessor: 허프 실패 시 원본 크롭 반환 (예외 발생 금지)
- TableExtractor: 항상 2패스
- Python 3.10+

**모듈 책임 경계**:

| 모듈 | 책임 | 하지 않는 것 |
|------|------|-------------|
| `vlm_client.py` | vLLM HTTP 통신, base64 인코딩, logprobs 파싱 | 이미지 크롭, 도메인 코드 감지 |
| `template_augmentor.py` | 서식 템플릿 로드, IoU 비교, bbox 병합 | VLM 통신, 크롭 |
| `resolution_router.py` | bbox 크롭 + 패딩 + 48px 정렬 + 배치 그룹화 | VLM 통신 |
| `structured_extractor.py` | military 배치 VLM 호출, 필드 추출, 재시도 | 이미지 크롭 |
| `form_classifier.py` | 서식 분류 (military/other 분기) | 필드 추출, 크롭 |
| `ocr_hint_provider.py` | PaddleOCR 선행 실행, 힌트 문자열 생성 | VLM 호출, 크롭 |
| `seal_preprocessor.py` | HSV 분리 + 허프 탐지 + 극좌표 변환 | VLM 호출, 크롭 |
| `skills/table_extractor.py` | 2패스 표 처리, 셀 태스크 반환 | VLM 직접 호출 |
| `skills/signature_detector.py` | 서명 이진 탐지만 | OCR, 텍스트 추출 |
| `vlm/budget_config.py` | `PIXEL_BUDGETS`, `FALLBACK_PIXEL_BUDGET`, `DISPATCH_ORDER` 중앙화 | region_type 결정 |
| `skills/_parsing.py` | VLM 응답 JSON 관대 파서(`_loads_relaxed`) 공용화 | VLM 호출 |
| `preprocess/bbox_utils.py` | `compute_iou(a, b)` 공용 IoU 계산 | bbox 변환, 크롭 |
| `postprocess/rank_normalizer.py` | 한국군 계급 Levenshtein 최근접 매칭 | 스키마 검증 |
| `domain/form_type_labels.py` | FormType → 한국어 공식 명칭 매핑 (`FORM_TYPE_LABELS`) | 스키마/룰 검증 |
| `postprocess/serializer.py` | P5 직렬화, assembled_json 있을 때 document_title 정적 주입 + fields[] 중복 제거 | assembled_json None 서식은 원본 보존 |
| `pipeline/health_monitor.py` | VLM 연속 실패 추적, `record_failure(reason)`/`is_healthy()` | Fallback 라우팅 결정 |

---

## 10. 핵심 의존성

```txt
opencv-python>=4.9.0
scikit-image>=0.22.0
Pillow>=10.0.0
numpy>=1.26.0
realesrgan
basicsr
paddlepaddle-gpu
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
