"""S6 SignatureDetector — 서명 존재 여부 이진 탐지.

PIPELINE.md §4-7 구현.
  - 서명 OCR이 아닌 존재 여부만 판정 (수기 서명 vs 인쇄/도장 구분)
  - pixel_budget=140 (이진 분류는 저해상도로 충분)
  - guided_json: {"signature_present": bool, "confidence": float}
"""

from __future__ import annotations

import json
import logging

from src.interfaces.types import SkillResult, SkillTask
from src.vlm.skills.seal_reader import _loads_relaxed
from src.vlm.vlm_client import VLMClient, encode_image_base64

logger = logging.getLogger(__name__)


class SignatureDetector:
    """서명 이진 탐지 Skill."""

    SKILL_NAME = "S6"
    CONTENT_TYPE = "signature"
    PIXEL_BUDGET = 140

    SYSTEM_PROMPT = (
        "You are a binary classifier for handwritten signatures.\n"
        "Does this image region contain a handwritten signature (수기 서명)?\n"
        "A signature is a cursive personal mark — distinct from printed text or stamps.\n"
        "Output only JSON: {\"signature_present\": true|false, \"confidence\": 0.0~1.0}."
    )

    USER_INSTRUCTION = "이 영역에 수기 서명이 있는지 판단하여 JSON으로 출력하세요."

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "signature_present": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["signature_present", "confidence"],
        "additionalProperties": False,
    }

    def __init__(self, vlm_client: VLMClient):
        self.vlm = vlm_client

    def run(self, task: SkillTask) -> SkillResult:
        pixel_budget = self.PIXEL_BUDGET
        try:
            image_b64 = encode_image_base64(task.cropped_image, max_size=pixel_budget)
            response = self.vlm.call(
                image_b64=image_b64,
                instruction=self.USER_INSTRUCTION,
                system_prompt=self.SYSTEM_PROMPT,
                guided_json=self.OUTPUT_SCHEMA,
                pixel_budget=pixel_budget,
            )
        except Exception as e:
            logger.error("[SignatureDetector] VLM 호출 실패 region=%s: %s", task.region_id, e)
            return SkillResult(
                region_id=task.region_id,
                skill_name=self.SKILL_NAME,
                content=json.dumps({"signature_present": False, "confidence": 0.0}),
                confidence=0.0,
                content_type=self.CONTENT_TYPE,
                warnings=[f"vlm_call_failed: {type(e).__name__}"],
            )

        raw = response.get("content", "") or ""
        present, conf = _parse_signature_json(raw)
        content_json = json.dumps(
            {"signature_present": present, "confidence": conf},
            ensure_ascii=False,
        )

        return SkillResult(
            region_id=task.region_id,
            skill_name=self.SKILL_NAME,
            content=content_json,
            confidence=conf,
            content_type=self.CONTENT_TYPE,
            raw_response=raw,
            warnings=[] if raw else ["empty_response"],
        )


def _parse_signature_json(raw: str) -> tuple[bool, float]:
    """VLM 출력 파싱. 실패 시 (False, 0.0)."""
    if not raw:
        return False, 0.0
    data = _loads_relaxed(raw)
    if data is not None:
        present = bool(data.get("signature_present", False))
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return present, max(0.0, min(1.0, conf))
    return False, 0.0
