"""S2 PrintedTextReader — 인쇄 텍스트 전용 도메인 서비스.

처리 경로(military/other)에 무관하게 region_type 기반으로 호출됩니다.
guided_json 없이 순수 텍스트 추출 — 스키마 강제보다 정확한 텍스트 추출 우선.

호출 시나리오:
  - other 경로 Skill Registry: region_type=text/header/footer → S2 (SKILL_ROUTING)
  - military 경로 StructuredExtractor: form_type=UNKNOWN일 때 text/header/footer → S2
"""

from __future__ import annotations

import logging

from src.interfaces.types import SkillResult, SkillTask
from src.vlm.logprobs_scorer import calc_field_confidence
from src.vlm.vlm_client import VLMClient, encode_image_base64

logger = logging.getLogger(__name__)


class PrintedTextReader:
    """인쇄 텍스트 인식 Skill (S2).

    guided_json 없이 순수 텍스트 추출. 신뢰도는 토큰 logprobs 기하평균 기반.
    """

    SKILL_NAME = "S2"
    CONTENT_TYPE = "printed"
    PIXEL_BUDGET = 560

    SYSTEM_PROMPT = (
        "You are a Korean document OCR assistant.\n"
        "Extract the printed Korean text from this region exactly as written.\n"
        "Output only the recognized text — no explanation, no JSON, no code fence.\n"
        "Preserve line breaks. Mark illegible characters as [?]."
    )

    USER_INSTRUCTION = "위 영역의 인쇄된 한국어 텍스트를 원문 그대로 출력하세요."

    def __init__(self, vlm_client: VLMClient):
        self.vlm = vlm_client

    def run(self, task: SkillTask) -> SkillResult:
        pixel_budget = task.pixel_budget or self.PIXEL_BUDGET
        instruction = self.USER_INSTRUCTION
        if task.context:
            instruction = f"{instruction}\n[참고] {task.context}"

        try:
            image_b64 = encode_image_base64(task.cropped_image, max_size=pixel_budget)
            response = self.vlm.call(
                image_b64=image_b64,
                instruction=instruction,
                system_prompt=self.SYSTEM_PROMPT,
                guided_json=None,
                pixel_budget=pixel_budget,
            )
        except Exception as e:
            logger.error("[PrintedTextReader] VLM 호출 실패 region=%s: %s", task.region_id, e)
            return SkillResult(
                region_id=task.region_id,
                skill_name=self.SKILL_NAME,
                content="",
                confidence=0.0,
                content_type=self.CONTENT_TYPE,
                warnings=[f"vlm_call_failed: {type(e).__name__}"],
            )

        raw = response.get("content", "") or ""
        text = raw.strip()
        logprobs = response.get("logprobs", []) or []
        token_lps = [
            lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
            for lp in logprobs
        ]
        confidence = calc_field_confidence(token_lps, "text") if token_lps else 0.0

        warnings: list[str] = []
        if not text:
            warnings.append("empty_text_output")

        return SkillResult(
            region_id=task.region_id,
            skill_name=self.SKILL_NAME,
            content=text,
            confidence=confidence,
            content_type=self.CONTENT_TYPE,
            raw_response=raw,
            warnings=warnings,
        )
