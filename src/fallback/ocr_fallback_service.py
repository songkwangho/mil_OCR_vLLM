"""OCR Fallback 서비스 — v1 T3(서식 분류) + T4(수기 인식) + T5(표 구조) 래핑

VLM 불가 시 v1 컴포넌트를 호출하고, v2 VLMResult 형식으로 변환합니다.
Fallback 경로 결과는 신뢰도에 할인 계수를 적용하며,
항상 검토 큐 적재 대상(review_required)으로 마킹됩니다.

의존:
  - mil_OCR (v1) 소스가 sys.path에 포함되어야 함
  - PaddlePaddle, PaddleOCR, PyTorch, Transformers
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.interfaces.enums import (
    AnalysisMode,
    CodeType,
    FormType,
    ProcessingPath,
    RegionType,
)
from src.interfaces.types import (
    DomainCode,
    FieldValue,
    LayoutRegion,
    LayoutResult,
    PreprocessedImage,
    RecognizedTable,
    VLMResult,
)

logger = logging.getLogger(__name__)

# v1 FormType → v2 FormType 매핑
_V1_FORM_TYPE_MAP: dict[str, FormType] = {
    "supply_request": FormType.SUPPLY_REQUEST,
    "maintenance_record": FormType.MAINTENANCE_RECORD,
    "inventory_sheet": FormType.INVENTORY_SHEET,
    "handover_doc": FormType.HANDOVER_DOC,
    "inspection_report": FormType.INSPECTION_REPORT,
    # v1에만 존재하는 유형 → UNKNOWN
    "transfer_order": FormType.UNKNOWN,
    "disposal_request": FormType.UNKNOWN,
    "receipt": FormType.UNKNOWN,
    "unknown": FormType.UNKNOWN,
}

# 신뢰도 할인 계수 — fallback 결과는 VLM보다 낮은 정확도
_CONFIDENCE_DISCOUNT = 0.7


# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────

@dataclass
class OCRFallbackConfig:
    """Fallback 서비스 설정."""

    # v1 소스 경로 (sys.path 추가용)
    v1_source_root: str = "/home/team_gh/mil_OCR"

    # 모델 경로 (v1 models/ 디렉토리)
    v1_model_root: str = "/home/team_gh/mil_OCR/models"

    # 디바이스
    device: str = "gpu:0"

    # 헬스체크 URL (fallback 서비스 자체 헬스)
    health_url: str = ""


# ─────────────────────────────────────────────
#  v2 ↔ v1 인터페이스 어댑터
# ─────────────────────────────────────────────

class _InterfaceAdapter:
    """v2 타입 → v1 타입 변환 유틸리티.

    v1과 v2는 동일한 개념이지만 필드가 다른 dataclass를 사용합니다.
    이 어댑터는 양 방향 변환을 담당합니다.
    """

    @staticmethod
    def layout_v2_to_v1(
        layout: LayoutResult,
    ) -> Any:
        """v2 LayoutResult → v1 LayoutAnalysisResult 변환."""
        from mil_OCR.src.interfaces.types import (
            BoundingBox as V1BBox,
            LayoutAnalysisResult as V1LayoutResult,
            LayoutRegion as V1LayoutRegion,
        )
        from mil_OCR.src.interfaces.enums import (
            AnalysisMode as V1AnalysisMode,
            RegionType as V1RegionType,
        )

        # RegionType 매핑 (v2 → v1)
        region_type_map = {
            "text": V1RegionType.TEXT if hasattr(V1RegionType, "TEXT")
            else V1RegionType.PRINTED_TEXT,
            "table": V1RegionType.TABLE,
            "figure": V1RegionType.FIGURE,
            "header": V1RegionType.HEADER if hasattr(V1RegionType, "HEADER")
            else V1RegionType.PRINTED_TEXT,
            "footer": V1RegionType.FOOTER if hasattr(V1RegionType, "FOOTER")
            else V1RegionType.PRINTED_TEXT,
            "seal": V1RegionType.SEAL if hasattr(V1RegionType, "SEAL")
            else V1RegionType.FIGURE,
            "formula": V1RegionType.FORMULA if hasattr(V1RegionType, "FORMULA")
            else V1RegionType.TEXT,
            "chart": V1RegionType.CHART if hasattr(V1RegionType, "CHART")
            else V1RegionType.FIGURE,
        }

        v1_regions = []
        for r in layout.regions:
            v1_bbox = V1BBox(
                x1=r.bbox.x1, y1=r.bbox.y1,
                x2=r.bbox.x2, y2=r.bbox.y2,
            )
            v1_rtype = region_type_map.get(
                r.region_type.value,
                V1RegionType.PRINTED_TEXT,
            )
            v1_regions.append(V1LayoutRegion(
                region_id=r.region_id,
                region_type=v1_rtype,
                bbox=v1_bbox,
                confidence=r.confidence,
            ))

        v1_mode = (
            V1AnalysisMode.MODEL
            if layout.analysis_mode == AnalysisMode.MODEL
            else V1AnalysisMode.HEURISTIC
        )

        return V1LayoutResult(
            doc_id=layout.doc_id,
            page_width=layout.page_width,
            page_height=layout.page_height,
            regions=v1_regions,
            reading_order=list(layout.reading_order),
            analysis_mode=v1_mode,
            warnings=list(layout.warnings),
        )

    @staticmethod
    def form_type_v1_to_v2(v1_form_type: Any) -> FormType:
        """v1 FormType → v2 FormType 변환."""
        return _V1_FORM_TYPE_MAP.get(v1_form_type.value, FormType.UNKNOWN)

    @staticmethod
    def text_lines_to_fields(
        text_lines: list[Any],
        discount: float = _CONFIDENCE_DISCOUNT,
    ) -> list[FieldValue]:
        """v1 TextLine 목록 → v2 FieldValue 목록 변환.

        T4의 텍스트 라인을 개별 필드로 변환합니다.
        영역 ID를 키로 사용하고, 신뢰도에 할인 계수를 적용합니다.
        """
        fields = []
        for tl in text_lines:
            confidence = tl.line_confidence * discount
            fields.append(FieldValue(
                field_key=tl.region_id,
                raw_value=tl.raw_text,
                corrected_value=tl.raw_text,  # fallback은 교정 없음
                data_type="text",
                confidence=confidence,
                token_logprobs=[],  # logprobs 없음 (PP-OCRv5)
                is_flagged=confidence < 0.5,
            ))
        return fields

    @staticmethod
    def tables_v1_to_v2(
        v1_tables: list[Any],
        discount: float = _CONFIDENCE_DISCOUNT,
    ) -> list[RecognizedTable]:
        """v1 TableStructure 목록 → v2 RecognizedTable 변환."""
        tables = []
        for t in v1_tables:
            cells = []
            for c in t.cells:
                cells.append({
                    "row": c.row,
                    "col": c.col,
                    "row_span": c.row_span,
                    "col_span": c.col_span,
                    "text": "",  # T4 결과와 연결 필요
                    "is_header": c.is_header,
                })

            tables.append(RecognizedTable(
                region_id=t.region_id,
                html=t.html,
                cells=cells,
                confidence=t.confidence * discount,
            ))
        return tables


# ─────────────────────────────────────────────
#  OCRFallbackService
# ─────────────────────────────────────────────

class OCRFallbackService:
    """v1 T3~T5 기반 Fallback 서비스.

    VLM 불가 시 v1의 서식 분류 + 수기 인식 + 표 구조를
    호출하여 v2 VLMResult로 변환합니다.

    사용법:
        service = OCRFallbackService(OCRFallbackConfig())
        vlm_result = service.process(preprocessed_image, layout_result)
    """

    def __init__(self, cfg: Optional[OCRFallbackConfig] = None):
        self.cfg = cfg or OCRFallbackConfig()
        self._v1_loaded = False
        self._t3 = None
        self._t4 = None
        self._t5 = None
        self._adapter = _InterfaceAdapter()

    # ═══════════════════════════════════════
    #  v1 모듈 로드
    # ═══════════════════════════════════════

    def _ensure_v1_loaded(self) -> None:
        """v1 소스를 sys.path에 추가하고 컴포넌트를 초기화."""
        if self._v1_loaded:
            return

        v1_root = Path(self.cfg.v1_source_root)
        if not v1_root.exists():
            raise RuntimeError(
                f"v1 소스 경로를 찾을 수 없습니다: {v1_root}"
            )

        # v1 소스를 import 경로에 추가
        v1_root_str = str(v1_root.parent)
        if v1_root_str not in sys.path:
            sys.path.insert(0, v1_root_str)
            logger.info("Fallback: v1 소스 경로 추가 — %s", v1_root_str)

        self._load_t3()
        self._load_t4()
        self._load_t5()
        self._v1_loaded = True
        logger.info("Fallback: v1 T3/T4/T5 초기화 완료")

    def _load_t3(self) -> None:
        """T3 서식 분류기 로드."""
        try:
            from mil_OCR.src.layer1_preprocess.t3_form_classifier import (
                T3FormClassifier,
                T3FormClassifierConfig,
            )
            config = T3FormClassifierConfig(device=self.cfg.device)
            self._t3 = T3FormClassifier(config)
            logger.info("Fallback: T3 (DiT) 로드 완료")
        except Exception as e:
            logger.warning("Fallback: T3 로드 실패 — %s (룰 분류 사용)", e)
            self._t3 = None

    def _load_t4(self) -> None:
        """T4 수기 인식기 로드."""
        try:
            from mil_OCR.src.layer2_ocr.t4_handwriting_recognizer import (
                T4HandwritingRecognizer,
                T4HandwritingRecognizerConfig,
            )
            config = T4HandwritingRecognizerConfig(device=self.cfg.device)
            self._t4 = T4HandwritingRecognizer(config)
            logger.info("Fallback: T4 (PP-OCRv5) 로드 완료")
        except Exception as e:
            logger.error("Fallback: T4 로드 실패 — %s", e)
            self._t4 = None

    def _load_t5(self) -> None:
        """T5 표 구조 인식기 로드."""
        try:
            from mil_OCR.src.layer2_ocr.t5_structure_recognizer import (
                T5StructureRecognizer,
                T5StructureRecognizerConfig,
            )
            config = T5StructureRecognizerConfig(device=self.cfg.device)
            self._t5 = T5StructureRecognizer(config)
            logger.info("Fallback: T5 (SLANeXt) 로드 완료")
        except Exception as e:
            logger.error("Fallback: T5 로드 실패 — %s", e)
            self._t5 = None

    # ═══════════════════════════════════════
    #  헬스체크
    # ═══════════════════════════════════════

    def is_healthy(self) -> bool:
        """Fallback 서비스 가용 여부 확인.

        T4(수기 인식)가 로드되어 있으면 최소 기능 가용으로 판단합니다.
        """
        if not self._v1_loaded:
            try:
                self._ensure_v1_loaded()
            except Exception:
                return False
        return self._t4 is not None

    # ═══════════════════════════════════════
    #  메인 처리
    # ═══════════════════════════════════════

    def process(
        self,
        preprocessed: PreprocessedImage,
        layout: LayoutResult,
    ) -> VLMResult:
        """v1 T3~T5를 호출하고 v2 VLMResult로 변환.

        Args:
            preprocessed: P1 출력 (전처리 이미지)
            layout: P2 출력 (구조 분석 결과)

        Returns:
            VLMResult — fallback 경로 결과 (신뢰도 할인 적용)

        Raises:
            RuntimeError: v1 로드 실패 또는 T4 미가용 시
        """
        t0 = time.time()
        self._ensure_v1_loaded()

        if self._t4 is None:
            raise RuntimeError("Fallback: T4(PP-OCRv5) 미가용 — fallback 불가")

        warnings: list[str] = ["[경로] Fallback (v1 PP-OCRv5 T3~T5)"]

        # v2 LayoutResult → v1 LayoutAnalysisResult
        v1_layout = self._adapter.layout_v2_to_v1(layout)
        image_rgb = preprocessed.image_array

        # ── T3: 서식 분류 ──
        form_type = FormType.UNKNOWN
        form_confidence = 0.0
        if self._t3 is not None:
            try:
                t3_result = self._t3.classify(v1_layout, image_rgb)
                form_type = self._adapter.form_type_v1_to_v2(t3_result.form_type)
                form_confidence = t3_result.confidence * _CONFIDENCE_DISCOUNT
                logger.info(
                    "Fallback T3: %s (conf=%.3f → %.3f)",
                    form_type.value, t3_result.confidence, form_confidence,
                )
            except Exception as e:
                warnings.append(f"[T3] 서식 분류 실패: {e}")
                logger.warning("Fallback T3 실패 — %s", e)
        else:
            warnings.append("[T3] DiT 미가용 — 서식 분류 생략")

        # ── T4: 수기 인식 ──
        fields: list[FieldValue] = []
        try:
            t4_result = self._t4.recognize(v1_layout, image_rgb)
            fields = self._adapter.text_lines_to_fields(t4_result.text_lines)
            logger.info(
                "Fallback T4: %d 라인 인식 (%.0fms)",
                len(t4_result.text_lines), t4_result.processing_time_ms,
            )
        except Exception as e:
            warnings.append(f"[T4] 수기 인식 실패: {e}")
            logger.error("Fallback T4 실패 — %s", e)

        # ── T5: 표 구조 인식 ──
        tables: list[RecognizedTable] = []
        if self._t5 is not None:
            try:
                t5_result = self._t5.recognize(v1_layout, image_rgb)
                tables = self._adapter.tables_v1_to_v2(t5_result.tables)
                logger.info(
                    "Fallback T5: %d 표 인식 (%.0fms)",
                    len(t5_result.tables), t5_result.processing_time_ms,
                )
            except Exception as e:
                warnings.append(f"[T5] 표 구조 인식 실패: {e}")
                logger.warning("Fallback T5 실패 — %s", e)
        else:
            warnings.append("[T5] SLANeXt 미가용 — 표 구조 인식 생략")

        # ── VLMResult 조립 ──
        elapsed_ms = (time.time() - t0) * 1000

        # 필드를 JSON으로 직렬화 (raw_json 호환)
        raw_json = json.dumps(
            {f.field_key: f.raw_value for f in fields},
            ensure_ascii=False,
        )

        return VLMResult(
            doc_id=preprocessed.doc_id,
            form_type=form_type,
            form_confidence=form_confidence,
            schema_id="fallback_v1",
            fields=fields,
            tables=tables,
            domain_codes=[],  # fallback은 도메인 코드 인식 없음
            raw_json=raw_json,
            processing_time_ms=round(elapsed_ms, 1),
            warnings=warnings,
        )
