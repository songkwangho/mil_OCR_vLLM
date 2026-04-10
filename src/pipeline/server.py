"""파이프라인 HTTP 서버 — FastAPI

P1~P6 전체 파이프라인을 HTTP API로 제공합니다.
Docker 컨테이너 모드에서 layout/vLLM/fallback 서비스를 HTTP로 호출합니다.

환경변수:
  VLLM_BASE_URL         — vLLM 서버 URL (default: http://localhost:8100/v1)
  VLLM_HEALTH_URL       — vLLM 헬스체크 URL (default: http://localhost:8100/health)
  LAYOUT_SERVICE_URL    — 레이아웃 서비스 URL (default: http://localhost:8082)
  LAYOUT_MODEL_NAME     — 레이아웃 모델명 (default: PP-DocLayoutV3)
  LAYOUT_FUSION_MODE    — 레이아웃 융합 모드 (default: false)
  FALLBACK_ENABLED      — fallback 활성화 (default: true)
  FALLBACK_BASE_URL     — fallback 서비스 URL (default: http://fallback:8081)
  REVIEW_QUEUE_ENABLED  — 검토 큐 활성화 (default: true)
  REVIEW_QUEUE_DB_PATH  — 검토 큐 DB 경로 (default: /data/review_queue.db)
  MODEL_ROOT            — 모델 가중치 루트 (default: /models)
  PIPELINE_PORT         — 서비스 포트 (default: 8080)

API:
  GET  /health          → 서비스 상태 확인
  POST /pipeline/run    → 문서 이미지 → PipelineResult
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.interfaces.enums import FileExt, PipelineStatus, SourceType
from src.interfaces.types import DocumentInput
from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator, PipelineResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  환경변수 기반 설정
# ─────────────────────────────────────────────

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8100/v1")
VLLM_HEALTH_URL = os.environ.get("VLLM_HEALTH_URL", "http://localhost:8100/health")
LAYOUT_SERVICE_URL = os.environ.get("LAYOUT_SERVICE_URL", "http://localhost:8082")
LAYOUT_MODEL_NAME = os.environ.get("LAYOUT_MODEL_NAME", "PP-DocLayoutV3")
LAYOUT_FUSION_MODE = os.environ.get("LAYOUT_FUSION_MODE", "").lower() in ("true", "1", "yes")
FALLBACK_ENABLED = os.environ.get("FALLBACK_ENABLED", "true").lower() in ("true", "1", "yes")
FALLBACK_BASE_URL = os.environ.get("FALLBACK_BASE_URL", "http://fallback:8081")
REVIEW_QUEUE_ENABLED = os.environ.get("REVIEW_QUEUE_ENABLED", "true").lower() in ("true", "1", "yes")
REVIEW_QUEUE_DB_PATH = os.environ.get("REVIEW_QUEUE_DB_PATH", "/data/review_queue.db")
MODEL_ROOT = os.environ.get("MODEL_ROOT", "/models")
DB_PATH = os.environ.get("DB_PATH", "/data/ocr_results.db")

# ─────────────────────────────────────────────
#  Pydantic 스키마
# ─────────────────────────────────────────────


class RunRequest(BaseModel):
    """파이프라인 실행 요청 (JSON 모드)."""
    doc_id: str = Field(..., description="문서 ID")
    image_b64: str = Field(..., description="Base64 인코딩된 이미지")
    file_ext: str = Field("png", description="파일 확장자 (jpg, png, tiff, pdf)")
    source_type: str = Field("scan", description="소스 유형 (scan, camera, fax, pdf_export)")
    dpi_hint: Optional[int] = Field(None, description="DPI 힌트")
    metadata: dict = Field(default_factory=dict, description="추가 메타데이터")


class FieldOutput(BaseModel):
    field_key: str
    raw_value: str
    corrected_value: str = ""
    confidence: float = 0.0
    is_flagged: bool = False


class RunResponse(BaseModel):
    doc_id: str
    status: str
    processing_path: str
    form_type: Optional[str] = None
    form_confidence: float = 0.0
    fields: list[FieldOutput] = []
    table_count: int = 0
    total_ms: float
    timings: dict[str, float] = {}
    errors: list[str] = []
    warnings: list[str] = []


# ─────────────────────────────────────────────
#  FastAPI 앱
# ─────────────────────────────────────────────

app = FastAPI(title="OCR Pipeline Service", version="2.0.0")

_pipeline: Optional[PipelineOrchestrator] = None


def _get_pipeline() -> PipelineOrchestrator:
    """PipelineOrchestrator 싱글턴 (지연 로딩)."""
    global _pipeline
    if _pipeline is None:
        cfg = PipelineConfig(
            vllm_base_url=VLLM_BASE_URL,
            vllm_health_url=VLLM_HEALTH_URL,
            layout_service_url=LAYOUT_SERVICE_URL,
            layout_model_name=LAYOUT_MODEL_NAME,
            layout_fusion_mode=LAYOUT_FUSION_MODE,
            model_root=MODEL_ROOT,
            fallback_enabled=FALLBACK_ENABLED,
            fallback_base_url=FALLBACK_BASE_URL,
            review_queue_enabled=REVIEW_QUEUE_ENABLED,
            review_queue_db_url=f"sqlite:///{REVIEW_QUEUE_DB_PATH}",
            db_url=f"sqlite:///{DB_PATH}",
        )
        _pipeline = PipelineOrchestrator(cfg)
        logger.info("파이프라인 초기화 완료")
    return _pipeline


def _result_to_response(result: PipelineResult) -> RunResponse:
    """PipelineResult → RunResponse 변환."""
    fields = []
    if result.p3_result:
        for f in result.p3_result.fields:
            fields.append(FieldOutput(
                field_key=f.field_key,
                raw_value=f.raw_value,
                corrected_value=f.corrected_value,
                confidence=f.confidence,
                is_flagged=f.is_flagged,
            ))

    return RunResponse(
        doc_id=result.doc_id,
        status=result.status.value,
        processing_path=result.processing_path.value,
        form_type=result.p3a_form_type,
        form_confidence=result.p3a_form_confidence,
        fields=fields,
        table_count=len(result.p3_result.tables) if result.p3_result else 0,
        total_ms=round(result.total_ms, 1),
        timings={k: round(v, 1) for k, v in result.timings.items()},
        errors=result.errors,
        warnings=result.warnings,
    )


@app.get("/health")
def health():
    """서비스 상태 확인."""
    try:
        pipeline = _get_pipeline()
        monitor = pipeline._get_health_monitor()
        vlm_ok = monitor.is_healthy()
        return {
            "status": "ok" if vlm_ok else "degraded",
            "vlm_healthy": vlm_ok,
            "layout_service": LAYOUT_SERVICE_URL,
            "fallback_enabled": FALLBACK_ENABLED,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/pipeline/run", response_model=RunResponse)
def run_pipeline(req: RunRequest):
    """파이프라인 실행 (JSON 요청)."""
    # Base64 → bytes
    try:
        raw_bytes = base64.b64decode(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 디코딩 실패: {e}")

    # FileExt 매핑
    ext_map = {"jpg": FileExt.JPG, "jpeg": FileExt.JPG, "png": FileExt.PNG,
               "tiff": FileExt.TIFF, "tif": FileExt.TIFF, "pdf": FileExt.PDF}
    file_ext = ext_map.get(req.file_ext.lower(), FileExt.PNG)

    # SourceType 매핑
    src_map = {"scan": SourceType.SCAN, "camera": SourceType.CAMERA,
               "fax": SourceType.FAX, "pdf_export": SourceType.PDF_EXPORT}
    source_type = src_map.get(req.source_type.lower(), SourceType.SCAN)

    doc_input = DocumentInput(
        doc_id=req.doc_id,
        raw_bytes=raw_bytes,
        file_ext=file_ext,
        source_type=source_type,
        dpi_hint=req.dpi_hint,
        metadata=req.metadata,
    )

    pipeline = _get_pipeline()
    result = pipeline.run(doc_input)

    return _result_to_response(result)


@app.post("/pipeline/upload", response_model=RunResponse)
async def run_pipeline_upload(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    source_type: str = Form("scan"),
    dpi_hint: Optional[int] = Form(None),
):
    """파이프라인 실행 (파일 업로드)."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="빈 파일")

    # 파일명에서 확장자 추출
    suffix = (file.filename or "").rsplit(".", 1)[-1].lower() if file.filename else "png"
    ext_map = {"jpg": FileExt.JPG, "jpeg": FileExt.JPG, "png": FileExt.PNG,
               "tiff": FileExt.TIFF, "tif": FileExt.TIFF, "pdf": FileExt.PDF}
    file_ext = ext_map.get(suffix, FileExt.PNG)

    src_map = {"scan": SourceType.SCAN, "camera": SourceType.CAMERA,
               "fax": SourceType.FAX, "pdf_export": SourceType.PDF_EXPORT}
    st = src_map.get(source_type.lower(), SourceType.SCAN)

    doc_input = DocumentInput(
        doc_id=doc_id or (file.filename or "unknown").rsplit(".", 1)[0],
        raw_bytes=raw_bytes,
        file_ext=file_ext,
        source_type=st,
        dpi_hint=dpi_hint,
        metadata={"filename": file.filename},
    )

    pipeline = _get_pipeline()
    result = pipeline.run(doc_input)

    return _result_to_response(result)


# ─────────────────────────────────────────────
#  엔트리포인트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PIPELINE_PORT", "8080"))
    uvicorn.run(
        "src.pipeline.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
