"""군수 OCR 시스템 v2 — 인터페이스 데이터 타입 정의

v1(T1~T12, 23개 dataclass)에서 v2(P1~P6, VLM 통합)로 전환하면서
Layer 2-4 전용 타입을 제거하고, VLM 출력 + 검토 큐 타입을 신규 추가합니다.

v1 → v2 변경 요약:
  유지: BoundingBox, DocumentInput, PreprocessedImage, LayoutRegion
  축소 이관: LayoutAnalysisResult → LayoutResult (reading_order 강화)
  제거: FormClassificationResult, TextLine, HandwritingRecognitionResult,
        TableCell, MergedCellRange, TableStructure, FieldGroup,
        StructureRecognitionResult, DomainCodeRecognitionResult,
        CorrectedLine, CorrectedDocument, ValidationReport,
        FieldScore, ScoredDocument, ExtractedField, ExtractedDocument
  신규: FieldValue, RecognizedTable, DomainCode(v2), VLMResult,
        ValidatedResult, ReviewQueueItem, ReviewQueueStats, PipelineOutput(v2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from .enums import (
    AnalysisMode,
    BrightnessBand,
    CodeType,
    DpiResolutionBand,
    FileExt,
    FormType,
    OutputFormat,
    PipelineStatus,
    ProcessingPath,
    RegionType,
    ReviewPriority,
    ReviewReason,
    ReviewStatus,
    Severity,
    SourceType,
    ValidationErrorType,
)

__all__ = [
    # 공용
    "BoundingBox",
    # P1 전처리
    "DocumentInput",
    "PreprocessedImage",
    # P2 구조 분석
    "LayoutRegion",
    "LayoutResult",
    # P3 VLM 통합 추론
    "FieldValue",
    "RecognizedTable",
    "DomainCode",
    "VLMResult",
    # P4 룰 검증
    "ValidationError",
    "ValidatedResult",
    # P5 직렬화 / P6 DB 적재
    "PipelineOutput",
    # 검토 큐
    "ReviewQueueItem",
    "ReviewQueueStats",
]


# ═══════════════════════════════════════════════
#  공용 타입
# ═══════════════════════════════════════════════

@dataclass
class BoundingBox:
    """바운딩 박스 — 모든 컴포넌트에서 공용 사용."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


# ═══════════════════════════════════════════════
#  P1 — 전처리
# ═══════════════════════════════════════════════

@dataclass
class DocumentInput:
    """파이프라인 최초 입력 — 원본 문서.

    INPUT → P1
    """

    doc_id: str
    raw_bytes: bytes
    file_ext: FileExt
    source_type: SourceType
    dpi_hint: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedImage:
    """P1 출력 — 전처리 완료 이미지.

    P1 → P2
    """

    doc_id: str
    image_array: np.ndarray           # H×W×3 uint8 (RGB)
    binary_array: np.ndarray          # H×W uint8
    dpi: int                          # 정규화 출력 DPI
    original_dpi: int                 # 추정 입력 DPI
    resolution_band: DpiResolutionBand
    quality_score: float
    sr_applied: bool = False
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  P2 — 구조 분석
# ═══════════════════════════════════════════════

@dataclass
class LayoutRegion:
    """레이아웃 영역 단위 (P2).

    PP-DocLayout 검출 결과. region_type에 따라
    VLM task prompt가 자동 결정됨 (TASK_PROMPTS 참조).
    """

    region_id: str                    # r_0001, r_0002, …
    region_type: RegionType
    bbox: BoundingBox
    confidence: float                 # [0.0–1.0]


# P2가 결정하는 영역 레이블 → VLM task prompt 매핑
TASK_PROMPTS: dict[str, str] = {
    "text":    "OCR:",
    "table":   "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart":   "Chart Recognition:",
    "seal":    "Seal Recognition:",
    "header":  "OCR:",
    "footer":  "OCR:",
    "figure":  "Image Description:",
}


@dataclass
class LayoutResult:
    """P2 출력 — 구조 분석 결과.

    P2 → P3 (VLM)
    reading_order 순서대로 영역별 crop + task prompt를 VLM에 전달.
    """

    doc_id: str
    page_width: int
    page_height: int
    regions: list[LayoutRegion]
    reading_order: list[int]          # regions 인덱스 순서 (다단 컬럼 대응)
    analysis_mode: AnalysisMode = AnalysisMode.HEURISTIC
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  P3 — VLM 통합 추론
# ═══════════════════════════════════════════════

@dataclass
class FieldValue:
    """VLM이 추출한 개별 필드 (guided_json 출력).

    logprobs 기반 토큰 확률 → 필드 신뢰도.
    """

    field_key: str                    # JSON Schema의 property 이름
    raw_value: str                    # VLM 출력 원본
    corrected_value: str              # VLM 교정 결과 (동일하면 교정 없음)
    data_type: str                    # text, number, date, code
    confidence: float                 # logprobs 기반 기하평균 신뢰도
    token_logprobs: list[float]       # 해당 필드 토큰들의 개별 logprob
    is_flagged: bool = False          # 신뢰도 < 필드 유형별 임계값


@dataclass
class RecognizedTable:
    """VLM이 인식한 표 구조 (P3)."""

    region_id: str                    # P2 LayoutRegion 참조
    html: str                         # 표 구조 HTML
    cells: list[dict[str, Any]]       # [{"row": 0, "col": 0, "text": "품목"}]
    confidence: float


@dataclass
class DomainCode:
    """VLM이 인식한 도메인 코드 (P3 → P4 검증)."""

    code_type: CodeType
    raw_value: str
    normalized_value: str
    confidence: float


@dataclass
class VLMResult:
    """P3 출력 — VLM 통합 추론 결과.

    P3 → P4
    서식 분류 + 필드 추출 + 표 인식 + 도메인 코드를 단일 VLM이 처리.
    guided_json으로 구조 보장, logprobs로 필드별 신뢰도 산출.
    """

    doc_id: str
    form_type: FormType
    form_confidence: float
    schema_id: str                    # guided_json에 사용된 스키마 ID
    fields: list[FieldValue]          # guided_json 추출 키-값 쌍 + logprobs 신뢰도
    tables: list[RecognizedTable]
    domain_codes: list[DomainCode]
    raw_json: str                     # VLM 원본 JSON 응답
    processing_time_ms: float
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  P4 — 룰 검증 + 신뢰도 보정
# ═══════════════════════════════════════════════

@dataclass
class ValidationError:
    """검증 오류 단위 (P4).

    룰 기반 교차검증에서 발생한 오류.
    CRITICAL severity → 검토 큐 적재 트리거.
    """

    error_id: str                     # ve_0001, ve_0002, …
    error_type: ValidationErrorType
    severity: Severity
    field_ref: str                    # 관련 field_key
    expected: str
    actual: str
    message: str


@dataclass
class ValidatedResult:
    """P4 출력 — 룰 검증 + 신뢰도 보정 결과.

    P4 → P5 (통과 시) 또는 검토 큐 (실패 시)

    판정 기준:
      통과: overall_confidence ≥ 임계값 AND CRITICAL 오류 없음
      실패: CRITICAL 오류 존재 OR flagged_fields 존재 → review_required=True
    """

    doc_id: str
    fields: list[FieldValue]          # logprobs + 룰 보정 반영된 신뢰도
    tables: list[RecognizedTable]
    validation_errors: list[ValidationError]
    overall_confidence: float         # 보정 후 전체 신뢰도
    review_required: bool
    flagged_fields: list[str]         # 임계값 미달 field_key 목록
    processing_path: ProcessingPath = ProcessingPath.VLM


# ═══════════════════════════════════════════════
#  P5 직렬화 / P6 DB 적재
# ═══════════════════════════════════════════════

@dataclass
class PipelineOutput:
    """P6 최종 출력 — 파이프라인 완료 결과.

    자동 처리 경로: P4 통과 → P5 → P6 → PipelineOutput(status=SUCCESS)
    검토 경로:      P4 실패 → 검토 큐 → 담당자 교정 → P6 → PipelineOutput(status=REVIEW)
    Fallback 경로:  Fallback → P4 → 검토 큐 → PipelineOutput(status=PARTIAL)
    """

    doc_id: str
    status: PipelineStatus
    processing_path: ProcessingPath
    form_type: Optional[FormType] = None
    json_output: Optional[str] = None
    xml_output: Optional[str] = None
    csv_rows: list[dict[str, Any]] = field(default_factory=list)
    db_record_ids: list[str] = field(default_factory=list)
    review_queue_id: Optional[str] = None
    processing_ms: float = 0.0


# ═══════════════════════════════════════════════
#  검토 큐
# ═══════════════════════════════════════════════

@dataclass
class ReviewQueueItem:
    """수동 검토 큐의 개별 항목.

    P4 검증 실패, LOW confidence, fallback 처리, 완전 장애 시 적재.
    담당자가 UI에서 확인·교정 후 P6 DB 재적재.
    교정 데이터(원본 → 수정)는 VLM Fine-tuning 학습 데이터로 자동 축적.
    """

    queue_id: str                     # RQ-20260407-001
    doc_id: str
    enqueued_at: datetime
    priority: ReviewPriority
    reason: ReviewReason
    processing_path: ProcessingPath   # vlm, fallback, none

    # 자동 처리 결과 (있는 경우)
    validated_result: Optional[ValidatedResult] = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    flagged_fields: list[str] = field(default_factory=list)

    # 원본 참조
    original_image_path: str = ""
    preprocessed_image_path: str = ""

    # 검토 상태
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    corrected_fields: dict[str, str] = field(default_factory=dict)
    reviewer_notes: str = ""


@dataclass
class ReviewQueueStats:
    """검토 큐 현황 통계."""

    total_pending: int
    total_in_review: int
    critical_count: int
    normal_count: int
    avg_wait_minutes: float
