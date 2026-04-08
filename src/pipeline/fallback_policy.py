"""Fallback 전환 정책

VLM / Fallback / 검토 큐 분기 결정.

의존: 없음 (순수 Python)
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


class FallbackPolicy:
    """VLM 불가 시 fallback 전환 정책.

    사용 예시:
        policy = FallbackPolicy()
        path = policy.decide(vlm_healthy=False, fallback_enabled=True, fallback_healthy=True)
        # → "fallback"
    """

    def decide(
        self,
        vlm_healthy: bool,
        fallback_enabled: bool = True,
        fallback_healthy: bool = True,
    ) -> Literal["vlm", "fallback", "review_queue"]:
        """처리 경로 결정.

        Args:
            vlm_healthy: vLLM 서버 정상 여부
            fallback_enabled: fallback 서비스 활성화 여부
            fallback_healthy: fallback 서비스 정상 여부

        Returns:
            "vlm"          — 주 경로 (VLM 정상)
            "fallback"     — 경량 fallback (VLM 불가, fallback 가용)
            "review_queue" — 수동 검토 큐 (둘 다 불가)
        """
        if vlm_healthy:
            return "vlm"

        if fallback_enabled and fallback_healthy:
            logger.warning("FallbackPolicy: VLM 불가 → fallback 전환")
            return "fallback"

        logger.error("FallbackPolicy: VLM + fallback 모두 불가 → review_queue")
        return "review_queue"
