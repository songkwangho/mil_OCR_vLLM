"""v2 파이프라인 통합 테스트

테스트 모드:
  1. P1→P2 통합 테스트 (VLM 불필요)
  2. P1→P2→P3 통합 테스트 (VLM 서버 필요)
  3. P1→P6 전체 파이프라인 통합 테스트 (VLM 서버 필요)
  4. Fallback 경로 테스트 (VLM 불필요, v1 모델 필요)

실행:
  # P1→P2만 (VLM 없이)
  python -m pytest tests/test_integration_pipeline.py -k "test_p1_p2" -v

  # 전체 (VLM 서버 가동 후)
  python -m pytest tests/test_integration_pipeline.py -v

  # Fallback 경로만
  python -m pytest tests/test_integration_pipeline.py -k "test_fallback" -v
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import (
    FileExt,
    PipelineStatus,
    ProcessingPath,
    SourceType,
)
from src.interfaces.types import DocumentInput
from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  테스트 픽스처
# ─────────────────────────────────────────────

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_HEALTH_URL = os.environ.get("VLLM_HEALTH_URL", "http://localhost:8000/health")


def _create_test_image(width: int = 800, height: int = 600) -> bytes:
    """테스트용 합성 이미지 생성 (흰 배경 + 검은 텍스트 영역)."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 헤더 영역 (검은 바)
    img[20:60, 50:750] = 0

    # 텍스트 영역들
    for y in range(100, 500, 60):
        img[y:y+30, 50:400] = 30

    # 표 영역 (격자)
    for y in range(100, 500, 50):
        img[y:y+2, 400:750] = 0
    for x in range(400, 750, 70):
        img[100:500, x:x+2] = 0

    import cv2
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture
def test_doc_input() -> DocumentInput:
    """테스트용 DocumentInput."""
    return DocumentInput(
        doc_id="test-integ-001",
        raw_bytes=_create_test_image(),
        file_ext=FileExt.PNG,
        source_type=SourceType.SCAN,
        dpi_hint=300,
        metadata={"test": True},
    )


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """테스트용 PipelineConfig."""
    return PipelineConfig(
        vllm_base_url=VLLM_BASE_URL,
        vllm_health_url=VLLM_HEALTH_URL,
        device="gpu:0",
        fallback_enabled=True,
        review_queue_enabled=True,
        review_queue_db_url="sqlite:///data/test_review_queue.db",
        db_url="sqlite:///data/test_ocr_results.db",
    )


def _vlm_is_available() -> bool:
    """VLM 서버 가용 여부 확인."""
    try:
        import httpx
        resp = httpx.get(VLLM_HEALTH_URL, timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_vlm = pytest.mark.skipif(
    not _vlm_is_available(),
    reason="VLM 서버 미가동 — VLLM_BASE_URL/VLLM_HEALTH_URL 환경변수 확인",
)


# ─────────────────────────────────────────────
#  테스트 1: P1→P2 통합 테스트 (VLM 불필요)
# ─────────────────────────────────────────────

class TestP1P2Integration:
    """P1(화질 보정) → P2(구조 분석) 통합 테스트."""

    def test_p1_preprocessor(self, test_doc_input: DocumentInput):
        """P1 단독 실행."""
        from src.preprocess.preprocessor import P1Preprocessor

        p1 = P1Preprocessor()
        result = p1.process(test_doc_input)

        assert result is not None
        assert result.doc_id == "test-integ-001"
        assert result.image_array.shape[2] == 3  # RGB
        assert result.dpi > 0
        logger.info(
            "P1 결과: dpi=%d, quality=%.2f, sr=%s",
            result.dpi, result.quality_score, result.sr_applied,
        )

    def test_p1_p2_pipeline(self, test_doc_input: DocumentInput):
        """P1 → P2 연쇄 실행."""
        from src.preprocess.preprocessor import P1Preprocessor
        from src.preprocess.layout_analyzer import P2LayoutAnalyzer

        p1 = P1Preprocessor()
        p1_out = p1.process(test_doc_input)
        assert p1_out is not None

        p2 = P2LayoutAnalyzer()
        p2_out = p2.analyze(p1_out)

        assert p2_out is not None
        assert p2_out.doc_id == "test-integ-001"
        assert p2_out.page_width > 0
        assert p2_out.page_height > 0
        assert isinstance(p2_out.reading_order, list)
        logger.info(
            "P2 결과: %d regions, mode=%s",
            len(p2_out.regions), p2_out.analysis_mode.value,
        )


# ─────────────────────────────────────────────
#  테스트 3: P1→P6 전체 파이프라인 (VLM 필요)
# ─────────────────────────────────────────────

class TestFullPipeline:
    """P1→P6 전체 파이프라인 통합 테스트."""

    @skip_no_vlm
    def test_full_pipeline_vlm_path(
        self,
        test_doc_input: DocumentInput,
        pipeline_config: PipelineConfig,
    ):
        """VLM 주 경로 전체 실행."""
        pipeline = PipelineOrchestrator(pipeline_config)
        result = pipeline.run(test_doc_input)

        assert result is not None
        assert result.doc_id == "test-integ-001"
        assert result.processing_path == ProcessingPath.VLM
        assert result.status in (PipelineStatus.SUCCESS, PipelineStatus.REVIEW)
        assert result.p1_result is not None
        assert result.p2_result is not None
        assert result.p3_result is not None

        logger.info(
            "전체 파이프라인 결과: status=%s, path=%s, "
            "total=%.0fms, errors=%d, warnings=%d",
            result.status.value, result.processing_path.value,
            result.total_ms, len(result.errors), len(result.warnings),
        )
        logger.info("단계별 시간: %s", result.timings)


# ─────────────────────────────────────────────
#  테스트 4: Fallback 경로 테스트
# ─────────────────────────────────────────────

class TestFallbackPath:
    """VLM 불가 시 Fallback 경로 테스트."""

    def test_fallback_policy_decision(self):
        """FallbackPolicy 분기 결정 검증."""
        from src.pipeline.fallback_policy import FallbackPolicy

        policy = FallbackPolicy()

        assert policy.decide(vlm_healthy=True) == "vlm"
        assert policy.decide(vlm_healthy=False, fallback_enabled=True, fallback_healthy=True) == "fallback"
        assert policy.decide(vlm_healthy=False, fallback_enabled=True, fallback_healthy=False) == "review_queue"
        assert policy.decide(vlm_healthy=False, fallback_enabled=False) == "review_queue"

    def test_health_monitor(self):
        """VLM 헬스 모니터 기본 동작."""
        from src.pipeline.health_monitor import VLMHealthMonitor, HealthMonitorConfig

        monitor = VLMHealthMonitor(HealthMonitorConfig(
            health_url="http://localhost:99999/health",  # 존재하지 않는 포트
            check_timeout=1,
            check_interval=0,      # 매번 실제 체크
            failure_threshold=1,   # 1회 실패로 unhealthy
        ))
        # 연결 불가 → unhealthy
        assert monitor.is_healthy() is False

    def test_fallback_service_with_unavailable_vlm(
        self,
        test_doc_input: DocumentInput,
        pipeline_config: PipelineConfig,
    ):
        """VLM 불가 시 오케스트레이터가 fallback 또는 review_queue로 전환."""
        # VLM URL을 존재하지 않는 포트로 설정
        pipeline_config.vllm_base_url = "http://localhost:99999/v1"
        pipeline_config.vllm_health_url = "http://localhost:99999/health"
        pipeline_config.fallback_enabled = True

        pipeline = PipelineOrchestrator(pipeline_config)
        result = pipeline.run(test_doc_input)

        assert result is not None
        # VLM 불가 시: VLM 시도 후 실패 → REVIEW, 또는 헬스체크로 사전 감지 → FALLBACK/NONE
        assert result.status in (PipelineStatus.PARTIAL, PipelineStatus.REVIEW, PipelineStatus.FAILED)

        logger.info(
            "Fallback 결과: status=%s, path=%s, warnings=%s",
            result.status.value, result.processing_path.value,
            result.warnings,
        )


# ─────────────────────────────────────────────
#  테스트 5: 검토 큐 테스트
# ─────────────────────────────────────────────

class TestReviewQueue:
    """검토 큐 적재/조회 테스트."""

    def test_review_queue_enqueue_and_stats(self):
        """검토 큐에 항목 적재 후 통계 확인."""
        import uuid
        from src.postprocess.review_queue import ReviewQueue, ReviewQueueConfig
        from src.interfaces.enums import ReviewReason, ProcessingPath
        from src.interfaces.types import ValidatedResult

        rq = ReviewQueue(ReviewQueueConfig(
            db_url="sqlite:///data/test_review_queue_unit.db",
        ))

        unique_id = f"test-rq-{uuid.uuid4().hex[:8]}"
        validated = ValidatedResult(
            doc_id=unique_id,
            fields=[],
            tables=[],
            validation_errors=[],
            overall_confidence=0.3,
            review_required=True,
            flagged_fields=["field_a"],
            processing_path=ProcessingPath.VLM,
        )

        queue_id = rq.enqueue(validated, reason=ReviewReason.LOW_CONFIDENCE)
        assert queue_id is not None
        assert queue_id.startswith("RQ-")

        stats = rq.get_stats()
        assert stats.total_pending >= 1

        logger.info("검토 큐: queue_id=%s, stats=%s", queue_id, stats)
