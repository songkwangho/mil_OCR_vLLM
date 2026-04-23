"""S4 SealReader — 원형 인장 텍스트 VLM 인식.

PIPELINE.md §4-5 구현.
  1) SealPreprocessor 실행 → 언래핑 성공/실패에 따라 pixel_budget 조정
     - 언래핑 성공: 1120 (직선화된 텍스트를 정밀 인식)
     - 언래핑 실패: 560 (원본 원형 이미지를 저해상도로 처리)
  2) context_hint를 프롬프트에 삽입해 VLM이 이미지 특성을 인지하도록 안내
  3) guided_json: {"text": str, "confidence": float}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from src.interfaces.types import SkillResult, SkillTask
from src.preprocess.seal_preprocessor import SealPreprocessor
from src.vlm.skills._parsing import _loads_relaxed
from src.vlm.vlm_client import VLMClient, encode_image_base64

logger = logging.getLogger(__name__)


class SealReader:
    """원형 인장(직인/관인) 텍스트 인식 Skill."""

    SKILL_NAME = "S4"
    CONTENT_TYPE = "seal"

    SYSTEM_PROMPT_TEMPLATE = (
        "This image contains a Korean official seal (직인/관인).\n"
        "{context_hint}\n"
        "Recognize the Korean text in this image exactly as written.\n"
        "Common seal texts: 부대명, 부대장, 위원장, 의장, 장관 등.\n"
        "If a character is ambiguous, mark it with [?].\n"
        "Output strictly as JSON: {{\"text\": \"...\", \"confidence\": 0.0~1.0}}."
    )

    USER_INSTRUCTION = "위 이미지의 인장에 쓰여진 한국어 텍스트를 JSON으로 출력하세요."

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["text", "confidence"],
        "additionalProperties": False,
    }

    BUDGET_UNWRAPPED = 1120
    BUDGET_FALLBACK = 560

    def __init__(
        self,
        vlm_client: VLMClient,
        seal_preprocessor: Optional[SealPreprocessor] = None,
    ):
        self.vlm = vlm_client
        self.seal_preprocessor = seal_preprocessor or SealPreprocessor()

    def run(self, task: SkillTask) -> SkillResult:
        seal_result = self.seal_preprocessor.process(task.cropped_image)
        pixel_budget = (
            self.BUDGET_UNWRAPPED if seal_result.unwrapped else self.BUDGET_FALLBACK
        )

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            context_hint=seal_result.context_hint,
        )

        try:
            image_b64 = encode_image_base64(seal_result.image, max_size=pixel_budget)
            response = self.vlm.call(
                image_b64=image_b64,
                instruction=self.USER_INSTRUCTION,
                system_prompt=system_prompt,
                guided_json=self.OUTPUT_SCHEMA,
                pixel_budget=pixel_budget,
            )
        except Exception as e:
            logger.error("[SealReader] VLM 호출 실패 region=%s: %s", task.region_id, e)
            return SkillResult(
                region_id=task.region_id,
                skill_name=self.SKILL_NAME,
                content="",
                confidence=0.0,
                content_type=self.CONTENT_TYPE,
                warnings=[f"vlm_call_failed: {type(e).__name__}"],
            )

        raw = response.get("content", "") or ""
        text, conf = _parse_seal_json(raw)

        warnings: list[str] = []
        if not seal_result.unwrapped:
            warnings.append("hough_circle_detection_failed")
        if not text:
            warnings.append("empty_text_output")

        return SkillResult(
            region_id=task.region_id,
            skill_name=self.SKILL_NAME,
            content=text,
            confidence=conf,
            content_type=self.CONTENT_TYPE,
            raw_response=raw,
            warnings=warnings,
        )

    async def run_async(self, task: SkillTask) -> SkillResult:
        """동기 `run()`을 스레드풀에서 실행 — 동일 결과, 이벤트 루프 비차단."""
        return await asyncio.to_thread(self.run, task)


def _parse_seal_json(raw: str) -> tuple[str, float]:
    """VLM 출력에서 text/confidence 추출. 파싱 실패 시 (raw, 0.0)."""
    if not raw:
        return "", 0.0
    data = _loads_relaxed(raw)
    if data is not None:
        text = str(data.get("text", "") or "")
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return text, max(0.0, min(1.0, conf))
    return raw.strip(), 0.0
