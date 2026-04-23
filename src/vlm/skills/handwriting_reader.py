"""S3 HandwritingReader — 수기 인식 전용 도메인 서비스.

처리 경로(military/other)에 무관하게 region_type=handwritten_field 영역에 적용됩니다.

분기:
  - task.json_schema 없음 (unknown / other): guided_json 없이 순수 텍스트 반환
  - task.json_schema 있음 (field_key가 부여된 military 서식):
      sub-schema를 guided_json으로 전달하여 구조화 JSON 반환
      (예: {"item_number":3, "result":"O", "result_confidence":0.99},
            {"team":"…","rank":"…","name":"…","signature_present":true})

저신뢰 재시도: 신뢰도 < 0.70 시 다른 프롬프트 변형으로 1회 재호출.
재시도 결과의 신뢰도가 더 높을 때만 교체.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.interfaces.types import SkillResult, SkillTask
from src.vlm.logprobs_scorer import calc_field_confidence
from src.vlm.skills._parsing import _loads_relaxed_any
from src.vlm.vlm_client import VLMClient, encode_image_base64

logger = logging.getLogger(__name__)


class HandwritingReader:
    """수기 텍스트 인식 Skill (S3)."""

    SKILL_NAME = "S3"
    CONTENT_TYPE_PLAIN = "handwritten"
    CONTENT_TYPE_STRUCTURED = "structured"
    PIXEL_BUDGET = 1120
    RETRY_CONFIDENCE_THRESHOLD = 0.70

    SYSTEM_PROMPT_PRIMARY = (
        "You are a Korean handwriting recognition specialist.\n"
        "Recognize the handwritten Korean text exactly as written.\n"
        "Common confusion pairs: ㄱ/ㅋ, ㄴ/ㄹ, ㅏ/ㅓ, 1/ㅣ, 0/O.\n"
        "Mark ambiguous characters with [?]."
    )

    SYSTEM_PROMPT_RETRY = (
        "You are an expert in Korean handwriting analysis with focus on stroke-level detail.\n"
        "Examine each stroke carefully. Pay special attention to vowel distinctions (ㅏ vs ㅓ, ㅗ vs ㅜ)\n"
        "and consonant distinctions (ㄱ vs ㅋ, ㄴ vs ㄹ, ㅁ vs ㅂ).\n"
        "Prioritize the most plausible Korean word/phrase given the stroke pattern."
    )

    USER_INSTRUCTION_PLAIN = "위 영역의 한국어 수기 글씨를 원문 그대로 출력하세요."
    USER_INSTRUCTION_STRUCTURED = "위 영역의 한국어 수기 내용을 주어진 JSON 스키마에 맞춰 출력하세요."

    def __init__(self, vlm_client: VLMClient):
        self.vlm = vlm_client

    def run(self, task: SkillTask) -> SkillResult:
        if task.json_schema is not None:
            return self._run_with_schema(task)
        return self._run_plain_text(task)

    async def run_async(self, task: SkillTask) -> SkillResult:
        """동기 `run()`을 스레드풀에서 실행 — 동일 결과, 이벤트 루프 비차단."""
        return await asyncio.to_thread(self.run, task)

    # ─────────────────────────────────────────────
    #  guided_json 분기 — 구조화 JSON 반환
    # ─────────────────────────────────────────────
    def _run_with_schema(self, task: SkillTask) -> SkillResult:
        pixel_budget = task.pixel_budget or self.PIXEL_BUDGET

        result = self._invoke(
            task, pixel_budget,
            self.SYSTEM_PROMPT_PRIMARY + "\nOutput strictly as JSON conforming to the provided schema.",
            self._build_structured_instruction(task, is_retry=False),
            guided_json=task.json_schema,
            content_type=self.CONTENT_TYPE_STRUCTURED,
            is_retry=False,
        )

        if result.confidence >= self.RETRY_CONFIDENCE_THRESHOLD or result.content == "":
            return result

        logger.info(
            "[HandwritingReader/schema] 저신뢰 재시도 region=%s field=%s (conf=%.3f < %.2f)",
            task.region_id, task.field_key, result.confidence, self.RETRY_CONFIDENCE_THRESHOLD,
        )
        retry = self._invoke(
            task, pixel_budget,
            self.SYSTEM_PROMPT_RETRY + "\nOutput strictly as JSON conforming to the provided schema.",
            self._build_structured_instruction(task, is_retry=True),
            guided_json=task.json_schema,
            content_type=self.CONTENT_TYPE_STRUCTURED,
            is_retry=True,
        )
        if retry.confidence > result.confidence:
            retry.warnings = list(retry.warnings) + ["retried_low_confidence"]
            return retry
        result.warnings = list(result.warnings) + ["retry_did_not_improve"]
        return result

    # ─────────────────────────────────────────────
    #  순수 텍스트 분기 — unknown / other 경로
    # ─────────────────────────────────────────────
    def _run_plain_text(self, task: SkillTask) -> SkillResult:
        pixel_budget = task.pixel_budget or self.PIXEL_BUDGET

        instruction = self.USER_INSTRUCTION_PLAIN
        if task.context:
            instruction = f"{instruction}\n[셀 역할] {task.context}"

        result = self._invoke(
            task, pixel_budget,
            self.SYSTEM_PROMPT_PRIMARY + "\nOutput only the recognized text — no explanation, no JSON.",
            instruction,
            guided_json=None,
            content_type=self.CONTENT_TYPE_PLAIN,
            is_retry=False,
        )

        if result.confidence >= self.RETRY_CONFIDENCE_THRESHOLD or result.content == "":
            return result

        logger.info(
            "[HandwritingReader/plain] 저신뢰 재시도 region=%s (conf=%.3f < %.2f)",
            task.region_id, result.confidence, self.RETRY_CONFIDENCE_THRESHOLD,
        )
        retry_instruction = instruction
        retry = self._invoke(
            task, pixel_budget,
            self.SYSTEM_PROMPT_RETRY + "\nOutput only the recognized text — no explanation, no JSON.",
            retry_instruction,
            guided_json=None,
            content_type=self.CONTENT_TYPE_PLAIN,
            is_retry=True,
        )
        if retry.confidence > result.confidence:
            retry.warnings = list(retry.warnings) + ["retried_low_confidence"]
            return retry
        result.warnings = list(result.warnings) + ["retry_did_not_improve"]
        return result

    # ─────────────────────────────────────────────
    #  내부 — 단일 VLM 호출
    # ─────────────────────────────────────────────
    def _invoke(
        self,
        task: SkillTask,
        pixel_budget: int,
        system_prompt: str,
        instruction: str,
        guided_json: Optional[dict],
        content_type: str,
        is_retry: bool,
    ) -> SkillResult:
        try:
            image_b64 = encode_image_base64(task.cropped_image, max_size=pixel_budget)
            response = self.vlm.call(
                image_b64=image_b64,
                instruction=instruction,
                system_prompt=system_prompt,
                guided_json=guided_json,
                pixel_budget=pixel_budget,
            )
        except Exception as e:
            logger.error("[HandwritingReader] VLM 호출 실패 region=%s: %s", task.region_id, e)
            tag = "vlm_call_failed_retry" if is_retry else "vlm_call_failed"
            return SkillResult(
                region_id=task.region_id,
                skill_name=self.SKILL_NAME,
                content="",
                confidence=0.0,
                content_type=content_type,
                warnings=[f"{tag}: {type(e).__name__}"],
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
        # 구조화 응답은 코드 펜스/주변 설명이 섞여도 Assembler가 복원할 수 있도록 JSON만 남긴다.
        if content_type == self.CONTENT_TYPE_STRUCTURED and text:
            parsed = _loads_relaxed_any(text)
            if parsed is not None:
                import json as _json
                text = (
                    _json.dumps(parsed, ensure_ascii=False)
                    if isinstance(parsed, (dict, list))
                    else str(parsed)
                )
            else:
                warnings.append("structured_json_parse_failed")
        if not text:
            warnings.append("empty_text_output")

        return SkillResult(
            region_id=task.region_id,
            skill_name=self.SKILL_NAME,
            content=text,
            confidence=confidence,
            content_type=content_type,
            raw_response=raw,
            warnings=warnings,
        )

    def _build_structured_instruction(self, task: SkillTask, is_retry: bool) -> str:
        """sub-schema 전달 경로의 instruction. field_key/context를 힌트로 포함."""
        base = self.USER_INSTRUCTION_STRUCTURED
        hints: list[str] = []
        if task.field_key:
            hints.append(f"[필드] {task.field_key}")
        if task.context:
            hints.append(f"[힌트] {task.context}")
        if is_retry:
            hints.append("[재시도] 획의 형태를 더 세밀하게 관찰하세요.")
        if hints:
            base = base + "\n" + "\n".join(hints)
        return base
