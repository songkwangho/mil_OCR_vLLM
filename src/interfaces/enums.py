"""군수 OCR 시스템 v2 — Enum 정의

v1(T1~T12)에서 v2(P1~P6) 전환에 따라 축소·재편된 열거형입니다.
VLM 통합 추론으로 Layer 2-3 전용 Enum(TextType, Orientation,
TableType, ContentType, CorrectionMethod, ClassificationMode)은 제거되었습니다.
"""

from enum import Enum

__all__ = [
    # 공용
    "FileExt",
    "SourceType",
    # P1 전처리
    "DpiResolutionBand",
    "BrightnessBand",
    # P2 구조 분석
    "RegionType",
    "AnalysisMode",
    # P3 VLM
    "FormType",
    "CodeType",
    # P4 룰 검증
    "Severity",
    "ValidationErrorType",
    # P5 직렬화
    "OutputFormat",
    # P6 DB 적재 / 파이프라인
    "PipelineStatus",
    "ProcessingPath",
    # 검토 큐
    "ReviewPriority",
    "ReviewStatus",
    "ReviewReason",
]


# ──────────────────────────────────────────────
#  공용
# ──────────────────────────────────────────────

class FileExt(str, Enum):
    """입력 파일 확장자."""
    JPG = "jpg"
    PNG = "png"
    TIFF = "tiff"
    PDF = "pdf"


class SourceType(str, Enum):
    """문서 입력 소스 유형."""
    SCAN = "scan"
    CAMERA = "camera"
    FAX = "fax"
    PDF_EXPORT = "pdf_export"


# ──────────────────────────────────────────────
#  P1 — 전처리
# ──────────────────────────────────────────────

class DpiResolutionBand(str, Enum):
    """DPI 해상도 대역 (P1)."""
    LOW = "low"       # < 150 dpi  → SR 적용
    MID = "mid"       # 150–249 dpi
    HIGH = "high"     # ≥ 250 dpi


class BrightnessBand(str, Enum):
    """밝기 대역 (P1)."""
    DARK = "dark"
    NORMAL = "normal"
    BRIGHT = "bright"


# ──────────────────────────────────────────────
#  P2 — 구조 분석
# ──────────────────────────────────────────────

class RegionType(str, Enum):
    """레이아웃 영역 유형 (P2).

    PP-DocLayout 검출 레이블 기반.
    v1의 HANDWRITTEN_FIELD, PRINTED_TEXT, SIGNATURE 등은
    VLM이 영역 내용으로 판별하므로 제거.
    """
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    SEAL = "seal"
    FORMULA = "formula"
    CHART = "chart"


class AnalysisMode(str, Enum):
    """레이아웃 분석 모드 (P2)."""
    MODEL = "model"          # PP-DocLayout
    HEURISTIC = "heuristic"  # OpenCV fallback


# ──────────────────────────────────────────────
#  P3 — VLM 통합 추론
# ──────────────────────────────────────────────

class FormType(str, Enum):
    """서식 유형 (P3 VLM 분류)."""
    SUPPLY_REQUEST = "supply_request"
    MAINTENANCE_RECORD = "maintenance_record"
    INVENTORY_SHEET = "inventory_sheet"
    HANDOVER_DOC = "handover_doc"
    INSPECTION_REPORT = "inspection_report"
    UNKNOWN = "unknown"


class CodeType(str, Enum):
    """도메인 코드 유형 (P3 VLM 인식 → P4 검증)."""
    NSN = "nsn"
    K_NSN = "k_nsn"
    UNIT_CODE = "unit_code"
    EQUIPMENT_ID = "equipment_id"
    DATE = "date"
    RANK = "rank"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────
#  P4 — 룰 검증
# ──────────────────────────────────────────────

class Severity(str, Enum):
    """검증 오류 심각도 (P4)."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationErrorType(str, Enum):
    """검증 오류 유형 (P4)."""
    ARITHMETIC = "arithmetic"
    DATE_LOGIC = "date_logic"
    CODE_FORMAT = "code_format"
    MISSING_FIELD = "missing_field"
    FORMAT = "format"


# ──────────────────────────────────────────────
#  P5 — 직렬화
# ──────────────────────────────────────────────

class OutputFormat(str, Enum):
    """직렬화 출력 형식 (P5)."""
    JSON = "json"
    XML = "xml"
    CSV = "csv"


# ──────────────────────────────────────────────
#  파이프라인 / 검토 큐
# ──────────────────────────────────────────────

class PipelineStatus(str, Enum):
    """파이프라인 최종 상태."""
    SUCCESS = "success"
    PARTIAL = "partial"        # fallback 경로 처리 (정확도 ↓)
    REVIEW = "review"          # 수동 검토 필요
    FAILED = "failed"


class ProcessingPath(str, Enum):
    """문서 처리 경로."""
    VLM = "vlm"                # 주 경로 (Gemma4 VLM)
    FALLBACK = "fallback"      # 수준 B (v1 PP-OCRv5 T3~T5)
    NONE = "none"              # 처리 불가 (완전 장애)


class ReviewPriority(str, Enum):
    """검토 큐 우선순위."""
    CRITICAL = "critical"      # CRITICAL 오류 / 완전 장애
    NORMAL = "normal"          # LOW confidence / fallback


class ReviewStatus(str, Enum):
    """검토 큐 항목 상태."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewReason(str, Enum):
    """검토 큐 적재 사유."""
    VALIDATION_FAILED = "validation_failed"   # P4 CRITICAL 오류
    LOW_CONFIDENCE = "low_confidence"         # 필드 신뢰도 임계값 미달
    FALLBACK = "fallback"                     # fallback 경로 처리
    TOTAL_FAILURE = "total_failure"           # VLM + fallback 모두 불가
