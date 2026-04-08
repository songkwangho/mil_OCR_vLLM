"""군수 OCR 시스템 v2 — 공용 인터페이스.

사용법:
    from src.interfaces import DocumentInput, PreprocessedImage, VLMResult
    from src.interfaces.enums import FormType, Severity
"""

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
from .types import (
    BoundingBox,
    DocumentInput,
    DomainCode,
    FieldValue,
    LayoutRegion,
    LayoutResult,
    PipelineOutput,
    PreprocessedImage,
    RecognizedTable,
    ReviewQueueItem,
    ReviewQueueStats,
    ValidatedResult,
    ValidationError,
    VLMResult,
)

__all__ = [
    # enums
    "AnalysisMode",
    "BrightnessBand",
    "CodeType",
    "DpiResolutionBand",
    "FileExt",
    "FormType",
    "OutputFormat",
    "PipelineStatus",
    "ProcessingPath",
    "RegionType",
    "ReviewPriority",
    "ReviewReason",
    "ReviewStatus",
    "Severity",
    "SourceType",
    "ValidationErrorType",
    # types
    "BoundingBox",
    "DocumentInput",
    "DomainCode",
    "FieldValue",
    "LayoutRegion",
    "LayoutResult",
    "PipelineOutput",
    "PreprocessedImage",
    "RecognizedTable",
    "ReviewQueueItem",
    "ReviewQueueStats",
    "ValidatedResult",
    "ValidationError",
    "VLMResult",
]
