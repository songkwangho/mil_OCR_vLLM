"""P3-A — FormClassifier

전체 페이지를 저해상도(140 토큰)로 Gemma4에 1회 호출하여 서식 유형을 확정합니다.
InstructionRouter의 도메인 맥락 생성을 위한 필수 선행 단계.

서식 분류 실패 시 wrong schema로 전체 추출 결과가 오염되는
오류 전파를 차단하는 게이트 역할입니다.

Input:  PreprocessedImage + LayoutResult
Output: (FormType, float)  — form_type, form_confidence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from src.interfaces.enums import FormType

if TYPE_CHECKING:
    from src.pipeline.health_monitor import VLMHealthMonitor

logger = logging.getLogger(__name__)


FORM_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "form_type": {
            "type": "string",
            "enum": [ft.value for ft in FormType],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "form_identifier": {"type": "string"},
    },
    "required": ["form_type", "confidence"],
    "additionalProperties": False,
}


@dataclass
class FormClassifierConfig:
    """P3-A FormClassifier 설정."""

    # vLLM 서버 연결
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "/models/gemma4/gemma-4-26b-a4b-it/"
    monitor: "Optional[VLMHealthMonitor]" = field(default=None, repr=False)

    # 분류용 pixel_budget (140 토큰 = 저해상도, 추론 비용 최소화)
    classify_pixel_budget: int = 140

    # API 파라미터
    max_tokens: int = 128
    temperature: float = 0.0
    request_timeout: float = 30.0

    # 분류 instruction (other 분기 포함)
    classify_instruction: str = (
        "이 문서의 서식 유형을 분류하고 서식 식별자(있는 경우)를 함께 출력하세요.\n"
        "form_type 후보:\n"
        "- supply_request: 보급청구서\n"
        "- maintenance_record: 정비기록서\n"
        "- inventory_sheet: 물자현황표\n"
        "- handover_doc: 인수인계서\n"
        "- inspection_report: 검사보고서\n"
        "- equipment_checklist: 전비품 확인서 작성 점검표 (제목에 '전비품 확인서 작성 점검표' "
        "  또는 '별지 제3-2호'가 있고 O/X 점검결과 열이 있는 서식)\n"
        "- unknown: 군수 서식이지만 유형 불명\n"
        "- other: 군수 서식이 아닌 일반 문서 (공문서, 지시문, 개인 서류 등)\n\n"
        "form_identifier: 문서 상단의 서식 식별자 문자열 (예: '별지 제3-2호 서식'). "
        "식별자가 없으면 빈 문자열.\n\n"
        "JSON 형식으로 출력: "
        '{"form_type": "...", "confidence": 0.0~1.0, "form_identifier": "..."}'
    )


class FormClassifier:
    """P3-A — 전체 페이지 저해상도 서식 분류.

    사용:
        classifier = FormClassifier(FormClassifierConfig(
            vllm_base_url="http://vllm-server:8000/v1",
        ))
        form_type, confidence = classifier.classify(image_rgb)
    """

    def __init__(self, config: Optional[FormClassifierConfig] = None):
        self.cfg = config or FormClassifierConfig()
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 지연 초기화."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.cfg.vllm_base_url,
                api_key="not-needed",
                timeout=self.cfg.request_timeout,
            )
        return self._client

    def classify(
        self,
        image_rgb: np.ndarray,
        warnings: Optional[list[str]] = None,
    ) -> tuple[FormType, float, Optional[str]]:
        """전체 페이지 이미지로 서식 유형 분류.

        Returns:
            (form_type, confidence, form_identifier).
            form_identifier는 문서 상단 서식 식별자(예: "별지 제3-2호 서식") 또는 None.
        """
        if warnings is None:
            warnings = []

        from src.vlm.vlm_client import encode_image_base64

        # 저해상도 인코딩 (140 토큰 = ~392px 장변)
        max_px = int((self.cfg.classify_pixel_budget * 28 * 28) ** 0.5)
        b64 = encode_image_base64(image_rgb, max_size=max_px)

        try:
            response = self._call_classify(b64)

            text = response.get("text", "") or ""
            logprobs = response.get("logprobs", [])

            form_type_str, conf_from_json, form_identifier = _parse_form_json(text)

            # JSON 파싱 실패 시 기존 문자열 매칭으로 폴백
            if form_type_str is None:
                lowered = text.strip().lower()
                valid_types = {ft.value for ft in FormType}
                form_type_str = "unknown"
                for vt in valid_types:
                    if vt in lowered:
                        form_type_str = vt
                        break

            # 신뢰도: JSON 값 > logprobs 기반 기하평균
            if conf_from_json is not None:
                confidence = round(max(0.0, min(1.0, conf_from_json)), 4)
            elif logprobs:
                import math
                all_lps = [
                    lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
                    for lp in logprobs
                ]
                mean_lp = sum(all_lps) / len(all_lps) if all_lps else -1.0
                confidence = round(min(1.0, max(0.0, math.exp(max(mean_lp, -20.0)))), 4)
            else:
                confidence = 0.5

            try:
                form_type = FormType(form_type_str)
            except ValueError:
                form_type = FormType.UNKNOWN
                warnings.append(f"Unknown form_type: {form_type_str} → UNKNOWN")

            logger.info(
                "[P3-A] 서식 분류: %s (conf=%.4f, identifier=%r)",
                form_type.value, confidence, form_identifier,
            )
            return form_type, confidence, form_identifier

        except Exception as e:
            warnings.append(f"FormClassifier failed: {e}")
            logger.error("[P3-A] 서식 분류 실패: %s", e)
            if self.cfg.monitor is not None:
                self.cfg.monitor.record_failure(f"form_classifier:{type(e).__name__}")
            return FormType.UNKNOWN, 0.0, None

    def _call_classify(self, image_b64: str) -> dict:
        """vLLM 서식 분류 호출."""
        client = self._get_client()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": self.cfg.classify_instruction,
                    },
                ],
            }
        ]

        extra_body = {
            "logprobs": True,
            "top_logprobs": 5,
            "guided_json": FORM_TYPE_SCHEMA,
        }

        # pixel_budget 제한
        pixel_budget = self.cfg.classify_pixel_budget
        if pixel_budget:
            extra_body["mm_processor_kwargs"] = {
                "max_pixels": pixel_budget * 28 * 28,
            }

        response = client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            extra_body=extra_body,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        logprobs_data = []
        if hasattr(choice, "logprobs") and choice.logprobs:
            content_logprobs = getattr(choice.logprobs, "content", None)
            if content_logprobs:
                for token_info in content_logprobs:
                    logprobs_data.append({
                        "token": getattr(token_info, "token", ""),
                        "logprob": getattr(token_info, "logprob", 0.0),
                    })

        return {"text": text, "logprobs": logprobs_data}


def _parse_form_json(raw: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """VLM 응답 → (form_type, confidence, form_identifier). 실패 시 (None, None, None)."""
    if not raw:
        return None, None, None
    try:
        from src.vlm.skills.seal_reader import _loads_relaxed
        data = _loads_relaxed(raw)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return None, None, None
    ft = data.get("form_type")
    conf = data.get("confidence")
    fid = data.get("form_identifier") or None
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    if isinstance(fid, str) and not fid.strip():
        fid = None
    return (str(ft) if ft else None), conf, fid
