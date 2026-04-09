"""Fallback 서비스 HTTP 클라이언트

Docker 분리 모드에서 fallback 컨테이너(POST /fallback/process)를
HTTP로 호출합니다. OCRFallbackService와 동일한 인터페이스를 제공합니다.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional

import cv2
import httpx
import numpy as np

from src.interfaces.enums import FormType, ProcessingPath
from src.interfaces.types import (
    LayoutResult,
    PreprocessedImage,
    VLMResult,
    FieldValue,
    RecognizedTable,
    DomainCode,
)

logger = logging.getLogger(__name__)


class FallbackHTTPClient:
    """Fallback 컨테이너 HTTP 클라이언트.

    OCRFallbackService와 동일한 process()/is_healthy() 인터페이스를 제공하여
    오케스트레이터에서 투명하게 교체 가능합니다.
    """

    def __init__(self, base_url: str = "http://localhost:8081", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_healthy(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def process(
        self,
        preprocessed: PreprocessedImage,
        layout: LayoutResult,
    ) -> Optional[VLMResult]:
        """Fallback 컨테이너에 HTTP POST로 처리 요청."""
        try:
            _, buf = cv2.imencode(".png", cv2.cvtColor(preprocessed.image_array, cv2.COLOR_RGB2BGR))
            image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            regions_data = []
            for r in layout.regions:
                regions_data.append({
                    "region_id": r.region_id,
                    "region_type": r.region_type,
                    "bbox": {"x1": r.bbox.x1, "y1": r.bbox.y1, "x2": r.bbox.x2, "y2": r.bbox.y2},
                    "confidence": r.confidence,
                })

            payload = {
                "doc_id": preprocessed.doc_id,
                "image_b64": image_b64,
                "dpi": preprocessed.dpi,
                "regions": regions_data,
                "reading_order": layout.reading_order,
            }

            resp = httpx.post(
                f"{self.base_url}/fallback/process",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            fields = [
                FieldValue(
                    field_key=f.get("field_key", ""),
                    raw_value=f.get("raw_value", ""),
                    corrected_value=f.get("corrected_value", ""),
                    data_type=f.get("data_type", "text"),
                    confidence=f.get("confidence", 0.0),
                    token_logprobs=[],
                    is_flagged=f.get("is_flagged", False),
                )
                for f in data.get("fields", [])
            ]

            tables = [
                RecognizedTable(
                    region_id=t.get("region_id", ""),
                    html=t.get("html", ""),
                    cells=t.get("cells", []),
                    confidence=t.get("confidence", 0.0),
                )
                for t in data.get("tables", [])
            ]

            form_type_str = data.get("form_type", "unknown")
            try:
                form_type = FormType(form_type_str)
            except ValueError:
                form_type = FormType.UNKNOWN

            return VLMResult(
                doc_id=preprocessed.doc_id,
                form_type=form_type,
                form_confidence=data.get("form_confidence", 0.0),
                schema_id=data.get("schema_id", "_fallback"),
                fields=fields,
                tables=tables,
                domain_codes=[],
                raw_json=json.dumps(data, ensure_ascii=False),
                processing_time_ms=data.get("processing_time_ms", 0.0),
                warnings=data.get("warnings", []),
            )

        except Exception as e:
            logger.error("Fallback HTTP 호출 실패: %s", e)
            return None
