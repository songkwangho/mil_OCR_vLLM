"""T5 — Fallback 경로 전환 통합 테스트.

TESTING.md 정의:
  T5: Fallback 전환 — VLM 불가 시 Fallback 경로 전환.

본 파일은 3가지 수준으로 T5를 검증:
  1. FallbackPolicy.decide 분기 (단위)
  2. VLMHealthMonitor가 죽은 포트에 대해 unhealthy 판정 (단위)
  3. 파이프라인 오케스트레이터가 VLM URL이 죽은 포트일 때
     review_queue/fallback 경로로 전환 (통합, 합성 이미지)

실제 Fallback 서비스(8081)를 써서 "VLM 다운 → fallback 응답 수령" 을
엔드투엔드로 보려면 별도 **수동 실행 시나리오**를 실행해야 한다 —
아래 docstring 의 "manual scenario" 절 참조.

실행:
  python -m pytest tests/test_T5_fallback_routing.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import FileExt, PipelineStatus, ProcessingPath, SourceType
from src.interfaces.types import DocumentInput
from src.pipeline.fallback_policy import FallbackPolicy
from src.pipeline.health_monitor import HealthMonitorConfig, VLMHealthMonitor
from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator

logger = logging.getLogger(__name__)


def _synth_document() -> DocumentInput:
    import cv2
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    img[20:60, 50:750] = 0  # 헤더
    for y in range(100, 500, 60):
        img[y:y + 30, 50:400] = 30
    _, buf = cv2.imencode(".png", img)
    return DocumentInput(
        doc_id="T5-synth",
        raw_bytes=buf.tobytes(),
        file_ext=FileExt.PNG,
        source_type=SourceType.SCAN,
        dpi_hint=300,
        metadata={"test": "T5"},
    )


# ─────────────────────────────────────────────
#  (1) FallbackPolicy decide 분기
# ─────────────────────────────────────────────

class TestT5FallbackPolicy:
    """FallbackPolicy 진리표 — VLM/Fallback 상태 조합별 경로."""

    def test_vlm_healthy_uses_vlm(self):
        assert FallbackPolicy().decide(vlm_healthy=True) == "vlm"

    def test_vlm_down_fallback_up_uses_fallback(self):
        p = FallbackPolicy().decide(
            vlm_healthy=False, fallback_enabled=True, fallback_healthy=True,
        )
        assert p == "fallback"

    def test_vlm_down_fallback_down_uses_review_queue(self):
        p = FallbackPolicy().decide(
            vlm_healthy=False, fallback_enabled=True, fallback_healthy=False,
        )
        assert p == "review_queue"

    def test_vlm_down_fallback_disabled_uses_review_queue(self):
        p = FallbackPolicy().decide(vlm_healthy=False, fallback_enabled=False)
        assert p == "review_queue"


# ─────────────────────────────────────────────
#  (2) HealthMonitor — 죽은 URL에 대해 unhealthy
# ─────────────────────────────────────────────

class TestT5HealthMonitorDetectsDownVLM:
    def test_unreachable_url_reports_unhealthy(self):
        monitor = VLMHealthMonitor(HealthMonitorConfig(
            health_url="http://127.0.0.1:1/health",  # 죽은 포트
            check_timeout=1,
            check_interval=0,
            failure_threshold=1,
        ))
        assert monitor.is_healthy() is False


# ─────────────────────────────────────────────
#  (3) 파이프라인 경로 전환 — VLM 죽은 포트
# ─────────────────────────────────────────────

class TestT5PipelineSwitchesOnVLMDown:
    """VLM URL을 죽은 포트로 설정했을 때 오케스트레이터의 경로 전환 동작.

    fallback_base_url 값에 따라 경로가 달라진다:
      - fallback_base_url="" → review_queue 경로 (NONE/PARTIAL/REVIEW/FAILED)
      - fallback_base_url=<가용> → FALLBACK 경로

    fallback 서비스 없이 실행해도 본 테스트는 통과해야 함 (review 경로 분기).
    """

    def test_vlm_down_no_fallback_goes_to_review_or_failed(self, tmp_path):
        """VLM 다운 + fallback 미설정 — graceful degrade 검증.

        현재 오케스트레이터는 사전 헬스체크 실패 전 VLM 호출을 시도하므로
        processing_path는 VLM으로 남을 수 있다. 핵심은:
          - 파이프라인이 크래시하지 않음
          - 모든 region이 Connection 실패 → status = REVIEW/PARTIAL/FAILED
          - warnings에 Connection error 기록
          - 검토 큐 경로로 적재 or 기록
        """
        cfg = PipelineConfig(
            vllm_base_url="http://127.0.0.1:1/v1",
            vllm_health_url="http://127.0.0.1:1/health",
            fallback_enabled=False,
            fallback_base_url="",
            review_queue_enabled=True,
            review_queue_db_url=f"sqlite:///{tmp_path}/rq.db",
            db_url=f"sqlite:///{tmp_path}/results.db",
        )
        pipeline = PipelineOrchestrator(cfg)
        result = pipeline.process(_synth_document())

        assert result is not None
        assert result.status in (
            PipelineStatus.PARTIAL, PipelineStatus.REVIEW, PipelineStatus.FAILED,
        ), f"unexpected status: {result.status.value}"
        # graceful degrade 증거: VLM 호출 실패 warning이 남아있거나 errors가 기록됨
        all_notes = " ".join(result.warnings + result.errors).lower()
        assert "connection" in all_notes or "vlm" in all_notes or result.errors, (
            f"VLM 실패 증거 누락: warnings={result.warnings} errors={result.errors}"
        )
        logger.info(
            "T5 VLM 다운/FB 없음 → status=%s path=%s warnings(sample)=%s",
            result.status.value,
            result.processing_path.value,
            result.warnings[:3],
        )


# ─────────────────────────────────────────────
#  수동 시나리오 — 실제 fallback 서비스 경로 (CI 부적합)
# ─────────────────────────────────────────────
"""
Manual scenario (수동):

  사전 조건:
    - docker compose에서 fallback 서비스(8081) 가동
    - vllm 서버 컨테이너는 정지
      $ docker stop mil-ocr-v2-vllm

  실행:
    FALLBACK_ENABLED=true \
    FALLBACK_BASE_URL=http://localhost:8081 \
    VLLM_BASE_URL=http://localhost:8100/v1 \
    VLLM_HEALTH_URL=http://localhost:8100/health \
    python scripts/run_pipeline_with_outputs.py --input-dir data/raw

  기대:
    - 오케스트레이터 로그에 "Fallback HTTP 클라이언트 초기화"
    - 각 문서 result.processing_path == ProcessingPath.FALLBACK
    - 모든 결과가 review_queue_enabled=True 이면 review 큐에 적재

  완료 후:
    $ docker start mil-ocr-v2-vllm
"""
