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
    │ military 경로 (기존 v2 사상 유지)                    │
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
    │ other 경로 (Skill Registry — v3 사상)               │
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
    raw_bytes: bytes
    file_ext: FileExt
    source_type: SourceType
    dpi_hint: Optional[int] = None
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
```

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
    warnings: list[str] = field(default_factory=list)
```

### 2-3. Skill 공통 인터페이스 (other 경로)

```python
@dataclass
class SkillTask:
    """오케스트레이터가 Skill에 전달하는 작업 단위."""
    region_id: str
    region_type: RegionType
    cropped_image: np.ndarray
    pixel_budget: int
    context: str = ""          # S5 패스1이 제공하는 셀 역할 등 추가 맥락
    form_type: Optional[FormType] = None

@dataclass
class SkillResult:
    region_id: str
    skill_name: str            # "S2", "S3", "S4", "S5", "S6"
    content: str               # 추출된 텍스트 또는 JSON
    confidence: float
    content_type: str          # "printed", "handwritten", "seal", "signature", "table"
    raw_response: str = ""
    warnings: list[str] = field(default_factory=list)

@dataclass
class TableStructure:
    """S5 패스1 결과 — 표 구조만."""
    region_id: str
    table_type: str            # "approval", "data", "other"
    cells: list[TableCell]
    structure_confidence: float

@dataclass
class TableCell:
    row: int
    col: int
    role: str                  # "담당", "팀장", "날짜" 등
    bbox: BoundingBox
    content_type: str          # "printed", "handwritten", "signature", "seal", "empty"

@dataclass
class SealProcessResult:
    """SealPreprocessor 결과."""
    image: np.ndarray          # 극좌표 직선화 이미지 or 원본 크롭
    unwrapped: bool            # 허프 원 탐지 성공 여부
    context_hint: str          # VLM 프롬프트에 삽입할 컨텍스트
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
```

### 2-5. ResolutionRouter 출력 (military 경로)

```python
@dataclass
class CroppedRegion:
    region_id: str
    region_type: RegionType
    cropped_image: np.ndarray  # 48px 배수 정렬 완료
    pixel_budget: int
    instruction_spec: InstructionSpec

# pixel_budget 기준값
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
    "signature":          140,  # 이진 분류용
}

# 배치 순서 (vLLM 패딩 오버헤드 최소화)
DISPATCH_ORDER = [140, 560, 1120]

# 크롭 패딩 비율
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
    # raw_json: DB 적재 제외 — 파일 시스템에만 보존
```

### 2-7. 후처리 결과

```python
@dataclass
class CorrectedField:
    field_path: str        # "items[0].quantity" — JSONPath 형식
    original_value: str
    corrected_value: str
    corrected_by: str
    corrected_at: datetime

@dataclass
class ReviewQueueItem:
    queue_id: str
    doc_id: str
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

@dataclass
class PipelineOutput:
    doc_id: str
    status: PipelineStatus  # success, partial, review, failed, other_document
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
        """제거/병합 결과 반영하여 reading_order 재정렬.
        1. removed_ids 인덱스 제거
        2. merged_map 흡수된 id 치환 (중복 제거)
        3. 정제 후 regions 기준 리넘버링
        """
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
| `unknown` | 군수 서식인데 유형 불명 | military (_fallback.json) |
| `other` | 군수 서식이 아님 | other → Skill Registry |

```python
class FormClassifier:
    PIXEL_BUDGET = 140

    def classify(self, image_rgb, warnings=None) -> tuple[FormType, float]:
        # guided_json enum: supply_request|...|unknown|other
        # temperature=0.0
```

### 3-3. P2.5-B InstructionRouter (military 전용)

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

### 3-4. P2.5-C ResolutionRouter (military 전용)

**파일**: `src/vlm/resolution_router.py`

```python
class ResolutionRouter:
    DISPATCH_ORDER = [140, 560, 1120]  # 배치 순서 — vLLM 패딩 최소화

    def route(self, layout, preprocessed, instructions) -> dict[int, list[CroppedRegion]]:
        """반환: {pixel_budget → [CroppedRegion]}
        동일 budget끼리 그룹화, DISPATCH_ORDER 순서로 vLLM에 전송.
        """

    def _align_to_48px(self, image) -> np.ndarray:
        """SigLIP 3×3 패치 풀링 효율 최적화.
        16×16 패치 → 3×3 블록 = 48px 배수가 최적.
        """
        h, w = image.shape[:2]
        new_h = max(48, round(h / 48) * 48)
        new_w = max(48, round(w / 48) * 48)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
```

### 3-5. P3-B StructuredExtractor (military 전용)

**파일**: `src/vlm/structured_extractor.py`

```python
class StructuredExtractor:
    RETRY_THRESHOLD = 0.60
    MAX_RETRIES = 1
    RETRY_BUDGET_MAP = {140: 280, 280: 560, 560: 1120, 1120: 1120}

    def extract(self, groups, doc_id, form_type, ...) -> VLMResult:
        """DISPATCH_ORDER 순서로 pixel_budget 배치 전송.
        저신뢰 필드 감지 시 pixel_budget 상향 + OCR 힌트 재시도.
        seal/signature 영역은 S4/S6 Skill로 위임.
        """
```

**logprobs 신뢰도 산출**:
```python
def calc_field_confidence(token_logprobs: list[float]) -> float:
    """기하 평균: exp(mean(logprobs))
    한국어 subword 특성상 영어 대비 5~10% 낮게 임계값 설정.
    """
    if not token_logprobs:
        return 0.0
    mean_logprob = max(sum(token_logprobs) / len(token_logprobs), -20.0)
    return round(min(1.0, max(0.0, math.exp(mean_logprob))), 4)
```

---

## 4. 컴포넌트 상세 설계 — other 경로 (Skill Registry)

### 4-1. SealPreprocessor (양 경로 공통)

**파일**: `src/preprocess/seal_preprocessor.py`

```python
class SealPreprocessor:
    """순수 기하학적 변환. 파인튜닝 대상 아님.
    military/other 양 경로에서 seal 영역에 사용.
    """

    def process(self, seal_crop: np.ndarray) -> SealProcessResult:
        # 1. HSV 색공간에서 적색 마스크 추출
        red_mask = self._extract_red(seal_crop)
        # lower_red = [0, 100, 100], upper_red = [10, 255, 255]

        # 2. 허프 원 변환으로 인장 경계 탐지
        circle = self._detect_circle(red_mask)

        if circle is None:
            # 허프 실패 → 원본 크롭 그대로 반환 (폴백)
            return SealProcessResult(
                image=seal_crop,
                unwrapped=False,
                context_hint="원형 인장 이미지입니다. 원형으로 배치된 텍스트를 인식하세요."
            )

        # 3. 극좌표 → 직교좌표 변환 (곡선 텍스트 직선화)
        unwrapped = self._polar_to_rect(seal_crop, circle)
        return SealProcessResult(
            image=unwrapped,
            unwrapped=True,
            context_hint="인장 텍스트를 직선화한 이미지입니다."
        )
```

### 4-2. Skill Registry

**파일**: `src/vlm/skill_registry.py`

```python
SKILL_ROUTING = {
    "text":              "S2",   # PrintedTextReader
    "header":            "S2",
    "footer":            "S2",
    "handwritten_field": "S3",   # HandwritingReader
    "seal":              "S4",   # SealReader
    "table":             "S5",   # TableExtractor (2패스)
    "signature":         "S6",   # SignatureDetector
}
```

### 4-3. S2 PrintedTextReader

**파일**: `src/vlm/skills/printed_text_reader.py`

```python
class PrintedTextReader:
    PIXEL_BUDGET = 560
    SYSTEM_PROMPT = """
    You are a Korean document OCR assistant.
    Extract all printed Korean text from this document region exactly as written.
    Output only the text content. Mark illegible characters as [?].
    """

    def run(self, task: SkillTask) -> SkillResult:
        # guided_json 없이 순수 텍스트 추출
        # 이유: 인쇄 텍스트는 스키마 강제보다 정확한 텍스트 추출이 우선
```

### 4-4. S3 HandwritingReader

**파일**: `src/vlm/skills/handwriting_reader.py`

```python
class HandwritingReader:
    PIXEL_BUDGET = 1120
    SYSTEM_PROMPT = """
    You are a Korean handwriting recognition specialist.

    Korean characters: 초성(initial consonant) + 중성(vowel) + 종성(final consonant).
    Analyze each component carefully.

    Common confusion pairs: ㄱ/ㅋ, ㄴ/ㄹ, ㅏ/ㅓ, 1/ㅣ, 0/O
    If ambiguous, output top candidate with [?] suffix.
    Context: {context}

    Output: {{"text": "...", "confidence": 0.0}}
    """

    def run(self, task: SkillTask) -> SkillResult:
        # context: S5 패스1이 제공한 셀 역할 (예: "담당자 서명란의 이름")
        # 첫 번째 VLM 호출 → 독립적으로 인식
        # 신뢰도 < 0.70 → 동일 영역 재호출 (다른 프롬프트 변형)
```

### 4-5. S4 SealReader

**파일**: `src/vlm/skills/seal_reader.py`

```python
class SealReader:
    SYSTEM_PROMPT = """
    This image contains a Korean official seal (직인/관인).
    {context_hint}
    Recognize the Korean text in this image.
    Common seal texts: 부대명, 부대장, 위원장, 의장, 장관 등
    Output: {{"text": "...", "confidence": 0.0}}
    """

    def run(self, task: SkillTask) -> SkillResult:
        # SealPreprocessor 먼저 실행
        seal_result = self.seal_preprocessor.process(task.cropped_image)
        pixel_budget = 1120 if seal_result.unwrapped else 560
        # 직선화 성공: 일반 텍스트 문제로 변환됨
        # 직선화 실패: context_hint로 VLM이 곡선 텍스트임을 인지
```

### 4-6. S5 TableExtractor (2패스)

**파일**: `src/vlm/skills/table_extractor.py`

```python
class TableExtractor:
    """결재란/데이터표 처리. 2패스로 순환 의존 제거."""

    PASS1_SCHEMA = {
        "analysis": "string",
        "table_type": {"enum": ["approval", "data", "other"]},
        "cells": [{
            "row": "int", "col": "int",
            "role": "string",
            "bbox": [x1, y1, x2, y2],
            "content_type": {"enum": ["printed", "handwritten", "signature", "seal", "empty"]}
        }]
    }

    def pass1_structure(self, table_crop: np.ndarray) -> TableStructure:
        """패스1: 표 구조만 추출. pixel_budget=1120.
        셀 좌표 + content_type 반환.
        오케스트레이터가 이 결과로 패스2 태스크 생성.
        """

    def pass2_route(self, structure: TableStructure,
                    original_image: np.ndarray) -> list[SkillTask]:
        """패스2: 셀별 크롭 + content_type에 맞는 Skill 태스크 반환.
        오케스트레이터가 이 태스크를 DISPATCH_ORDER 배치에 합산.
        """
        tasks = []
        skill_map = {
            "printed":     "S2",
            "handwritten": "S3",
            "signature":   "S6",
            "seal":        "S4",
        }
        for cell in structure.cells:
            if cell.content_type == "empty":
                continue
            crop = self._crop_cell(original_image, cell.bbox)
            skill = skill_map.get(cell.content_type, "S2")
            tasks.append(SkillTask(
                region_id=f"{structure.region_id}_r{cell.row}c{cell.col}",
                region_type=RegionType(cell.content_type),
                cropped_image=crop,
                pixel_budget=PIXEL_BUDGET.get(cell.content_type, 560),
                context=cell.role,
            ))
        return tasks
```

### 4-7. S6 SignatureDetector

**파일**: `src/vlm/skills/signature_detector.py`

```python
class SignatureDetector:
    """서명 OCR이 아닌 존재 여부 이진 탐지.
    military/other 양 경로에서 signature 영역에 사용.
    """
    PIXEL_BUDGET = 140  # 이진 분류는 저해상도로 충분

    SYSTEM_PROMPT = """
    Does this image region contain a handwritten signature (수기 서명)?
    A signature is a cursive personal mark — distinct from printed text or stamps.
    Output only: {{"signature_present": true/false, "confidence": 0.0}}
    """

    def run(self, task: SkillTask) -> SkillResult:
        # temperature=0.0, guided_json (이진 스키마)
```

### 4-8. S7 StructuredAggregator

**파일**: `src/vlm/skills/aggregator.py`

```python
class StructuredAggregator:
    """S2~S6 결과를 official_document.json Schema에 맞춰 최종 JSON 조립.
    form_type은 이 단계에서만 Schema 선택에 사용.
    """

    def run(self, skill_results: list[SkillResult],
            table_structure: TableStructure,
            form_type: FormType) -> VLMResult:
        # guided_json: official_document.json
        # low_confidence_fields: confidence < 임계값인 필드 목록
        # overall_confidence: 전체 필드 신뢰도 가중 평균
```

---

## 5. 오케스트레이터 디스패치 흐름 (other 경로)

**실제 구현**: `src/pipeline/orchestrator.py` `_process_other_document()` + `src/vlm/skill_registry.py` `SkillRegistry`.

```python
# orchestrator.py (발췌)
def _process_other_document(self, layout, preprocessed, doc_id, ...):
    registry = self._get_skill_registry()  # SkillRegistry(vlm_client)
    stats = SkillDispatchStats()

    # S5 2-pass (표 구조 → 셀 태스크)
    cell_results, structures = registry.dispatch_tables(layout, preprocessed, stats)

    # 비-표 영역 수집 + DISPATCH_ORDER 배치 실행
    non_table_tasks = registry.build_tasks(layout, preprocessed)
    general_results = registry.dispatch(non_table_tasks, stats)

    # SkillResult → FieldValue 평탄화 (S7 미구현 상태의 임시 집계)
    return VLMResult(
        form_type=FormType.OTHER,
        processing_path=ProcessingPath.SKILL_REGISTRY,
        fields=_skill_results_to_fields(general_results + cell_results),
        ...,
    )
```

- `SkillRegistry.dispatch()`는 태스크를 `DISPATCH_ORDER = [140, 560, 1120]` 버킷으로 나눠 순차 실행하며 `SkillDispatchStats`에 skill별 호출수·시간·배치 크기를 집계 (run_summary.json `skill_stats` 필드).
- `_skill_results_to_fields`는 region_id를 field_key로 사용하는 임시 집계 — **S7 StructuredAggregator 구현 전까지의 플레이스홀더**이며, 구현 후 `official_document.json` 스키마에 맞춘 guided_json 출력으로 대체 예정.
- S2/S3 미구현 상태에서는 `skill_registry._stub_text_skill`이 text/handwritten content_type을 모두 guided_json `{text, confidence}`로 처리.

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
> **other 문서**: 군수 임계값 적용 안 함. 전체 신뢰도만 산출.

### 6-2. P4 룰 검증 보정 (military 경로만)

- `합계 ≠ 수량 × 단가` → 관련 필드 신뢰도 **-0.30**
- NSN 형식 불일치 → 신뢰도 **-0.15**
- 날짜 순서 위반 → 신뢰도 **-0.10**

### 6-3. NSN 패턴 강제 전략

```
1순위: InstructionRouter 프롬프트 힌트 (기본 — xgrammar 유지)
   "NSN 코드는 NNNN-NN-NNN-NNNN 형식 13자리입니다."

2순위: guided_regex (필드 단위 패턴 강제 필요 시)
   extra_body={"guided_regex": r"\d{4}-\d{2}-\d{3}-\d{4}"}

3순위: guidance 백엔드 전환 (절대적 강제 필요 시)
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
  ├── P1 Preprocessor               (in-process)
  ├── P2 LayoutAnalyzer             HTTP POST → [Layout :8082]
  ├── P2.5-A LayoutPostProcessor    (in-process)
  ├── SealPreprocessor              (in-process, 극좌표 변환)
  ├── P3-A FormClassifier           HTTP POST → [vLLM :8100]
  │
  ├── [military 경로]
  │   ├── P2.5-B InstructionRouter  (in-process)
  │   ├── P2.5-C ResolutionRouter   (in-process)
  │   └── P3-B StructuredExtractor  HTTP POST → [vLLM :8100]
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
```

---

## 9. 코딩 컨벤션

- 모든 함수·클래스에 docstring 필수
- 컴포넌트 간 데이터 전달은 `src/interfaces/types.py`의 dataclass 사용
- 오케스트레이터만 컴포넌트를 순서대로 호출
- **military 경로**: form_type 기반 InstructionRouter + guided JSON Schema 유지
- **other 경로**: SKILL_ROUTING으로 라우팅, S7에서만 form_type 사용
- 모든 VLM 호출에 `temperature=0.0` (결정론적 OCR)
- `VLMResult.raw_json`은 DB 적재 제외
- ResolutionRouter 배치 순서: `DISPATCH_ORDER = [140, 560, 1120]`
- SealPreprocessor: 허프 실패 시 원본 크롭 반환 (예외 발생 금지)
- TableExtractor: 항상 2패스. 패스1 완료 후 오케스트레이터가 패스2 태스크 생성
- Python 3.10+

**모듈 책임 경계**:

| 모듈 | 책임 | 하지 않는 것 |
|------|------|-------------|
| `vlm_client.py` | vLLM HTTP 통신, base64 인코딩, logprobs 파싱 | 이미지 크롭, 도메인 코드 감지 |
| `resolution_router.py` | bbox 크롭 + 패딩 + 48px 정렬 + 배치 그룹화 | VLM 통신 |
| `structured_extractor.py` | military 배치 VLM 호출, 필드 추출, 재시도 | 이미지 크롭 |
| `form_classifier.py` | 서식 분류 (military/other 분기) | 필드 추출, 크롭 |
| `ocr_hint_provider.py` | PaddleOCR 선행 실행, 힌트 문자열 생성 | VLM 호출, 크롭 |
| `seal_preprocessor.py` | HSV 분리 + 허프 탐지 + 극좌표 변환 | VLM 호출, 크롭 |
| `skills/table_extractor.py` | 2패스 표 처리, 셀 태스크 반환 | VLM 직접 호출 (vlm_client 경유) |
| `skills/signature_detector.py` | 서명 이진 탐지만 | OCR, 텍스트 추출 |

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
