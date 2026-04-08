"""VLM 헬스 모니터 — 수준 A SPOF 대비

vLLM 서버 상태를 주기적으로 확인하고 캐시합니다.
연속 3회 실패 시 unhealthy 판정 → fallback 전환 트리거.

의존: httpx
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthMonitorConfig:
    """헬스 모니터 설정."""
    health_url: str = "http://localhost:8000/health"
    check_interval: int = 30    # 초
    check_timeout: int = 10     # 초
    failure_threshold: int = 3  # 연속 실패 횟수


class VLMHealthMonitor:
    """vLLM 서버 상태를 주기적으로 확인.

    사용 예시:
        monitor = VLMHealthMonitor()
        if monitor.is_healthy():
            # VLM 주 경로 사용
        else:
            # fallback 전환
    """

    def __init__(self, cfg: Optional[HealthMonitorConfig] = None):
        self.cfg = cfg or HealthMonitorConfig()
        self._healthy = True
        self._last_check: float = 0.0
        self._consecutive_failures: int = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def is_healthy(self) -> bool:
        """캐시된 상태 반환. interval 경과 시 실제 체크."""
        now = time.time()
        if now - self._last_check >= self.cfg.check_interval:
            self._check()
        return self._healthy

    def force_check(self) -> bool:
        """강제 헬스체크 실행."""
        self._check()
        return self._healthy

    def _check(self) -> None:
        """실제 HTTP 헬스체크 수행."""
        try:
            import httpx
            resp = httpx.get(
                self.cfg.health_url,
                timeout=self.cfg.check_timeout,
            )
            if resp.status_code == 200:
                if not self._healthy:
                    logger.info("VLM 헬스: 복구 (이전 실패 %d회)", self._consecutive_failures)
                self._healthy = True
                self._consecutive_failures = 0
            else:
                self._on_failure(f"status={resp.status_code}")
        except Exception as e:
            self._on_failure(str(e))

        self._last_check = time.time()

    def _on_failure(self, reason: str) -> None:
        """헬스체크 실패 처리."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.cfg.failure_threshold:
            if self._healthy:
                logger.error(
                    "VLM 헬스: 비정상 판정 (연속 %d회 실패 — %s)",
                    self._consecutive_failures, reason,
                )
            self._healthy = False
        else:
            logger.warning(
                "VLM 헬스: 실패 %d/%d (%s)",
                self._consecutive_failures, self.cfg.failure_threshold, reason,
            )
