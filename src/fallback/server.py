"""Fallback 서비스 — FastAPI HTTP 서버

v1 T3(서식 분류) + T4(수기 인식) + T5(표 구조) 기반 fallback 추론을
HTTP API로 제공합니다. Docker 컨테이너 분리 모드에서 사용됩니다.

환경변수:
  V1_SOURCE_ROOT  — v1 소스 경로 (default: /workspace/mil_OCR)
  V1_MODEL_ROOT   — v1 모델 경로 (default: /workspace/mil_OCR/models)
  FALLBACK_DEVICE  — 추론 디바이스 (default: gpu:0)

API:
  GET  /health            → 서비스 상태 확인
  POST /fallback/process  → 이미지 + 레이아웃 → VLMResult (fallback 경로)
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  환경변수 기반 설정
# ─────────────────────────────────────────────

V1_SOURCE_ROOT = os.environ.get("V1_SOURCE_ROOT", "/workspace/mil_OCR")
V1_MODEL_ROOT = os.environ.get("V1_MODEL_ROOT", "/workspace/mil_OCR/models")
FALLBACK_DEVICE = os.environ.get("FALLBACK_DEVICE", "gpu:0")

# ─────────────────────────────────────────────
#  Pydantic 스키마
# ─────────────────────────────────────────────


class RegionPayload(BaseModel):
    region_id: str
    region_type: str
    bbox: dict  # {"x1": int, "y1": int, "x2": int, "y2": int}
    confidence: float = 0.0


class ProcessRequest(BaseModel):
    doc_id: str = Field(..., description="문서 ID")
    image_b64: str = Field(..., description="Base64 인코딩된 PNG 이미지")
    dpi: int = Field(300, description="이미지 DPI")
    regions: list[RegionPayload] = Field(default_factory=list, description="레이아웃 영역")
    reading_order: list[int] = Field(default_factory=list, description="읽기 순서")


class FieldPayload(BaseModel):
    field_key: str
    raw_value: str
    corrected_value: str = ""
    data_type: str = "text"
    confidence: float = 0.0
    is_flagged: bool = False


class TablePayload(BaseModel):
    region_id: str
    html: str = ""
    cells: list[dict] = []
    confidence: float = 0.0


class ProcessResponse(BaseModel):
    doc_id: str
    form_type: str
    form_confidence: float
    schema_id: str = "_fallback"
    fields: list[FieldPayload]
    tables: list[TablePayload]
    processing_time_ms: float
    warnings: list[str]


# ─────────────────────────────────────────────
#  FastAPI 앱
# ─────────────────────────────────────────────

app = FastAPI(title="OCR Fallback Service", version="2.0.0")

_service = None


def _get_service():
    """OCRFallbackService 싱글턴 (지연 로딩)."""
    global _service
    if _service is None:
        from src.fallback.ocr_fallback_service import (
            OCRFallbackService,
            OCRFallbackConfig,
        )
        cfg = OCRFallbackConfig(
            v1_source_root=V1_SOURCE_ROOT,
            v1_model_root=V1_MODEL_ROOT,
            device=FALLBACK_DEVICE,
        )
        _service = OCRFallbackService(cfg)
        logger.info("Fallback 서비스 초기화 완료 (device=%s)", FALLBACK_DEVICE)
    return _service


@app.get("/health")
def health():
    """서비스 상태 확인."""
    try:
        svc = _get_service()
        healthy = svc.is_healthy()
        return {
            "status": "ok" if healthy else "degraded",
            "t4_loaded": svc._t4 is not None,
            "t3_loaded": svc._t3 is not None,
            "t5_loaded": svc._t5 is not None,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/fallback/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    """Fallback 추론 실행.

    이미지 + 레이아웃 정보를 받아 v1 T3~T5를 호출하고
    v2 VLMResult 호환 형식으로 반환합니다.
    """
    t0 = time.time()

    # Base64 → numpy 이미지
    try:
        img_bytes = base64.b64decode(req.image_b64)
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("이미지 디코딩 실패")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    # v2 타입 조립
    from src.interfaces.enums import AnalysisMode, RegionType, DpiResolutionBand
    from src.interfaces.types import (
        BoundingBox,
        LayoutRegion,
        LayoutResult,
        PreprocessedImage,
    )

    regions = []
    for r in req.regions:
        try:
            rtype = RegionType(r.region_type)
        except ValueError:
            rtype = RegionType.TEXT
        regions.append(LayoutRegion(
            region_id=r.region_id,
            region_type=rtype,
            bbox=BoundingBox(
                x1=r.bbox["x1"], y1=r.bbox["y1"],
                x2=r.bbox["x2"], y2=r.bbox["y2"],
            ),
            confidence=r.confidence,
        ))

    layout = LayoutResult(
        doc_id=req.doc_id,
        page_width=img_rgb.shape[1],
        page_height=img_rgb.shape[0],
        regions=regions,
        reading_order=req.reading_order,
        analysis_mode=AnalysisMode.HEURISTIC,
        warnings=[],
    )

    # DPI → 해상도 밴드
    if req.dpi < 150:
        res_band = DpiResolutionBand.LOW
    elif req.dpi < 250:
        res_band = DpiResolutionBand.MID
    else:
        res_band = DpiResolutionBand.HIGH

    preprocessed = PreprocessedImage(
        doc_id=req.doc_id,
        image_array=img_rgb,
        binary_array=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        dpi=req.dpi,
        original_dpi=req.dpi,
        resolution_band=res_band,
        quality_score=0.5,
        sr_applied=False,
        warnings=[],
    )

    # Fallback 실행
    try:
        svc = _get_service()
        result = svc.process(preprocessed, layout)
    except Exception as e:
        logger.error("Fallback 처리 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"Fallback 처리 실패: {e}")

    elapsed_ms = (time.time() - t0) * 1000

    return ProcessResponse(
        doc_id=result.doc_id,
        form_type=result.form_type.value,
        form_confidence=result.form_confidence,
        schema_id=result.schema_id,
        fields=[
            FieldPayload(
                field_key=f.field_key,
                raw_value=f.raw_value,
                corrected_value=f.corrected_value,
                data_type=f.data_type,
                confidence=f.confidence,
                is_flagged=f.is_flagged,
            )
            for f in result.fields
        ],
        tables=[
            TablePayload(
                region_id=t.region_id,
                html=t.html,
                cells=t.cells,
                confidence=t.confidence,
            )
            for t in result.tables
        ],
        processing_time_ms=round(elapsed_ms, 1),
        warnings=result.warnings,
    )
