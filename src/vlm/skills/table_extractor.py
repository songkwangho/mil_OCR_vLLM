"""S5 TableExtractor — 2패스 표 처리.

PIPELINE.md §4-6 구현.
  - pass1_structure: 표 구조(셀 좌표 + content_type)만 추출. pixel_budget=1120.
  - pass2_route: 구조 기반으로 셀별 크롭 + 적합 Skill용 SkillTask 목록 반환.
  오케스트레이터가 pass2 결과 태스크를 DISPATCH_ORDER 배치에 합산.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.interfaces.enums import RegionType
from src.interfaces.types import (
    BoundingBox,
    SkillTask,
    TableCell,
    TableStructure,
)
from src.vlm.skills.seal_reader import _loads_relaxed
from src.vlm.vlm_client import VLMClient, encode_image_base64

logger = logging.getLogger(__name__)


# pass2에서 사용하는 content_type → region_type 매핑
_CONTENT_TO_REGION = {
    "printed": RegionType.TEXT,
    "handwritten": RegionType.HANDWRITTEN_FIELD,
    "signature": RegionType.SIGNATURE,
    "seal": RegionType.SEAL,
}

# content_type → pixel_budget (PIPELINE.md §2-5)
_CONTENT_BUDGET = {
    "printed": 560,
    "handwritten": 1120,
    "signature": 140,
    "seal": 560,
}


class TableExtractor:
    """표 2패스 처리 Skill."""

    SKILL_NAME = "S5"
    PASS1_BUDGET = 1120

    PASS1_SYSTEM_PROMPT = (
        "You analyze Korean document tables (결재란/데이터표).\n"
        "Return ONLY the table structure: cell coordinates and content_type per cell.\n"
        "Do NOT transcribe cell text in this pass.\n"
        "bbox is [x1, y1, x2, y2] in pixels relative to the supplied table crop.\n"
        "content_type ∈ {printed, handwritten, signature, seal, empty}."
    )

    PASS1_USER_INSTRUCTION = (
        "표의 구조를 분석해 JSON으로 출력하세요. "
        "셀별 행·열 인덱스, 역할(role, 예: '담당', '팀장'), bbox, content_type을 포함하세요."
    )

    PASS1_SCHEMA = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "table_type": {"type": "string", "enum": ["approval", "data", "other"]},
            "structure_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "cells": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer", "minimum": 0},
                        "col": {"type": "integer", "minimum": 0},
                        "role": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["printed", "handwritten", "signature", "seal", "empty"],
                        },
                    },
                    "required": ["row", "col", "bbox", "content_type"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["table_type", "cells", "structure_confidence"],
        "additionalProperties": False,
    }

    def __init__(self, vlm_client: VLMClient):
        self.vlm = vlm_client

    # ─────────────────────────────────────────────
    #  패스 1: 구조 추출
    # ─────────────────────────────────────────────
    def pass1_structure(
        self, table_crop: np.ndarray, region_id: str = "table"
    ) -> TableStructure:
        try:
            image_b64 = encode_image_base64(table_crop, max_size=self.PASS1_BUDGET)
            response = self.vlm.call(
                image_b64=image_b64,
                instruction=self.PASS1_USER_INSTRUCTION,
                system_prompt=self.PASS1_SYSTEM_PROMPT,
                guided_json=self.PASS1_SCHEMA,
                pixel_budget=self.PASS1_BUDGET,
            )
        except Exception as e:
            logger.error("[TableExtractor.pass1] VLM 호출 실패 region=%s: %s", region_id, e)
            return TableStructure(
                region_id=region_id,
                table_type="other",
                cells=[],
                structure_confidence=0.0,
            )

        raw = response.get("content", "") or ""
        return _parse_pass1_json(raw, region_id)

    # ─────────────────────────────────────────────
    #  패스 2: 셀 크롭 + 디스패치 태스크 반환
    # ─────────────────────────────────────────────
    def pass2_route(
        self,
        structure: TableStructure,
        original_image: np.ndarray,
    ) -> list[SkillTask]:
        tasks: list[SkillTask] = []
        if original_image is None or original_image.size == 0:
            return tasks

        h, w = original_image.shape[:2]
        for cell in structure.cells:
            if cell.content_type == "empty":
                continue
            crop = _crop_cell(original_image, cell.bbox, (h, w))
            if crop is None or crop.size == 0:
                continue
            region_type = _CONTENT_TO_REGION.get(cell.content_type, RegionType.TEXT)
            budget = _CONTENT_BUDGET.get(cell.content_type, 560)
            tasks.append(
                SkillTask(
                    region_id=f"{structure.region_id}_r{cell.row}c{cell.col}",
                    region_type=region_type,
                    cropped_image=crop,
                    pixel_budget=budget,
                    context=cell.role or "",
                )
            )
        return tasks


# ─────────────────────────────────────────────
#  내부 유틸
# ─────────────────────────────────────────────

def _parse_pass1_json(raw: str, region_id: str) -> TableStructure:
    """패스1 응답 JSON → TableStructure. 파싱 실패 시 빈 구조."""
    if not raw:
        return TableStructure(region_id, "other", [], 0.0)
    data = _loads_relaxed(raw)
    if data is None:
        logger.warning("[TableExtractor.pass1] JSON 파싱 실패 region=%s", region_id)
        return TableStructure(region_id, "other", [], 0.0)

    cells_raw = data.get("cells") or []
    cells: list[TableCell] = []
    for c in cells_raw:
        if not isinstance(c, dict):
            continue
        bbox_val = c.get("bbox") or [0, 0, 0, 0]
        if not (isinstance(bbox_val, list) and len(bbox_val) == 4):
            continue
        try:
            bx1, by1, bx2, by2 = (int(v) for v in bbox_val)
        except (TypeError, ValueError):
            continue
        cells.append(
            TableCell(
                row=int(c.get("row", 0) or 0),
                col=int(c.get("col", 0) or 0),
                role=str(c.get("role", "") or ""),
                bbox=BoundingBox(bx1, by1, bx2, by2),
                content_type=str(c.get("content_type", "printed") or "printed"),
            )
        )

    try:
        confidence = float(data.get("structure_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return TableStructure(
        region_id=region_id,
        table_type=str(data.get("table_type", "other") or "other"),
        cells=cells,
        structure_confidence=max(0.0, min(1.0, confidence)),
    )


def _crop_cell(
    image: np.ndarray,
    bbox: BoundingBox,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """셀 bbox 크롭. 범위 벗어나면 clip."""
    h, w = image_shape
    x1 = max(0, min(w, int(bbox.x1)))
    y1 = max(0, min(h, int(bbox.y1)))
    x2 = max(0, min(w, int(bbox.x2)))
    y2 = max(0, min(h, int(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2].copy()
