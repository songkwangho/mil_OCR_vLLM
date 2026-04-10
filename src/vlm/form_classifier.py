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
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.interfaces.enums import FormType

logger = logging.getLogger(__name__)


@dataclass
class FormClassifierConfig:
    """P3-A FormClassifier 설정."""

    # vLLM 서버 연결
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "/models/gemma4/gemma-4-26b-a4b-it/"

    # 분류용 pixel_budget (140 토큰 = 저해상도, 추론 비용 최소화)
    classify_pixel_budget: int = 140

    # API 파라미터
    max_tokens: int = 128
    temperature: float = 0.0
    request_timeout: float = 30.0

    # 분류 instruction (other 분기 포함)
    classify_instruction: str = (
        "이 문서의 서식 유형을 분류하세요. "
        "군수(military) 서식이면 해당 유형을, 군수 서식이 아닌 일반 문서는 'other'로 분류하세요.\n"
        "다음 중 하나로 답하세요:\n"
        "- supply_request: 보급청구서\n"
        "- maintenance_record: 정비기록서\n"
        "- inventory_sheet: 물자현황표\n"
        "- handover_doc: 인수인계서\n"
        "- inspection_report: 검사보고서\n"
        "- unknown: 군수 서식이지만 유형 불명\n"
        "- other: 군수 서식이 아닌 일반 문서 (공문서, 지시문, 개인 서류 등)"
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
    ) -> tuple[FormType, float]:
        """전체 페이지 이미지로 서식 유형 분류.

        Args:
            image_rgb: H×W×3 uint8 RGB 이미지
            warnings: 경고 수집 리스트 (None이면 무시)

        Returns:
            (form_type, confidence)
        """
        if warnings is None:
            warnings = []

        from src.vlm.vlm_client import encode_image_base64

        # 저해상도 인코딩 (140 토큰 = ~392px 장변)
        max_px = int((self.cfg.classify_pixel_budget * 28 * 28) ** 0.5)
        b64 = encode_image_base64(image_rgb, max_size=max_px)

        try:
            response = self._call_classify(b64)

            text = response.get("text", "").strip().lower()
            logprobs = response.get("logprobs", [])

            # form_type 파싱
            valid_types = {ft.value for ft in FormType}
            form_type_str = "unknown"
            for vt in valid_types:
                if vt in text:
                    form_type_str = vt
                    break

            # 분류 신뢰도
            confidence = 0.5
            if logprobs:
                import math
                all_lps = [
                    lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
                    for lp in logprobs
                ]
                if all_lps:
                    mean_lp = sum(all_lps) / len(all_lps)
                    confidence = round(min(1.0, max(0.0, math.exp(max(mean_lp, -20.0)))), 4)

            try:
                form_type = FormType(form_type_str)
            except ValueError:
                form_type = FormType.UNKNOWN
                warnings.append(f"Unknown form_type: {form_type_str} → UNKNOWN")

            logger.info(
                "[P3-A] 서식 분류: %s (conf=%.4f)",
                form_type.value, confidence,
            )
            return form_type, confidence

        except Exception as e:
            warnings.append(f"FormClassifier failed: {e}")
            logger.error("[P3-A] 서식 분류 실패: %s", e)
            return FormType.UNKNOWN, 0.0

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
