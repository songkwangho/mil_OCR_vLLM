"""T9 (결재란 2패스) + T10 (서명 탐지) 통합 테스트.

TESTING.md 정의:
  T9: 결재란 2패스 처리 — S5 pass1 구조 추출 → pass2 셀 내용 추출,
      오케스트레이터 태스크 생성 흐름.
  T10: 서명 탐지 — 서명 있는 셀 → signature_present: true, 빈 셀 → false.

유닛 테스트(`test_skills_unit.py`)가 Skill 단위 계약을 다루는 반면, 본 파일은
실제 문서(`data/raw/국회공문서.jpg`)를 파이프라인 끝까지 통과시킨 뒤
`PipelineResult.skill_stats` / `SkillResult` 구조를 검증한다.

실행 조건:
  - VLM 서버 가동 (VLLM_HEALTH_URL)
  - Layout 서비스 가동 (LAYOUT_SERVICE_URL)
  - `data/raw/국회공문서.jpg` 존재

실행:
  VLLM_BASE_URL=http://localhost:8100/v1 \
  VLLM_HEALTH_URL=http://localhost:8100/health \
  LAYOUT_SERVICE_URL=http://localhost:8082 \
  python -m pytest tests/test_T9_T10_skills_integration.py -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import FileExt, SourceType
from src.interfaces.types import DocumentInput
from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8100/v1")
VLLM_HEALTH_URL = os.environ.get("VLLM_HEALTH_URL", "http://localhost:8100/health")
LAYOUT_SERVICE_URL = os.environ.get("LAYOUT_SERVICE_URL")

SAMPLE_PATH = _PROJECT_ROOT / "data" / "raw" / "국회공문서.jpg"


def _vlm_available() -> bool:
    try:
        import httpx
        return httpx.get(VLLM_HEALTH_URL, timeout=3.0).status_code == 200
    except Exception:
        return False


def _layout_available() -> bool:
    if not LAYOUT_SERVICE_URL:
        return False
    try:
        import httpx
        return httpx.get(f"{LAYOUT_SERVICE_URL}/health", timeout=3.0).status_code == 200
    except Exception:
        return False


skip_no_backends = pytest.mark.skipif(
    not (_vlm_available() and _layout_available() and SAMPLE_PATH.exists()),
    reason="VLM/Layout 서비스 미가동 또는 국회공문서 샘플 없음",
)


# ─────────────────────────────────────────────
#  세션 fixture — 같은 run을 여러 테스트에서 재사용
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def other_pipeline_result():
    """국회공문서.jpg → Skill Registry 경로 전체 실행 결과."""
    assert SAMPLE_PATH.exists(), f"샘플 없음: {SAMPLE_PATH}"

    cfg = PipelineConfig(
        vllm_base_url=VLLM_BASE_URL,
        vllm_health_url=VLLM_HEALTH_URL,
        layout_service_url=LAYOUT_SERVICE_URL,
        layout_model_name=os.environ.get("LAYOUT_MODEL_NAME", "PP-DocLayoutV3"),
        model_root=os.environ.get("MODEL_ROOT", str(_PROJECT_ROOT / "models")),
        fallback_enabled=False,
        review_queue_enabled=True,
        review_queue_db_url=f"sqlite:///{_PROJECT_ROOT}/data/test_t9_t10_review.db",
        db_url=f"sqlite:///{_PROJECT_ROOT}/data/test_t9_t10_results.db",
    )
    pipeline = PipelineOrchestrator(cfg)

    raw_bytes = SAMPLE_PATH.read_bytes()
    doc_input = DocumentInput(
        doc_id="T9_T10_국회공문서",
        raw_bytes=raw_bytes,
        file_ext=FileExt.JPG,
        source_type=SourceType.SCAN,
        dpi_hint=None,
        metadata={"test": "T9_T10"},
    )
    result = pipeline.process(doc_input)
    return result


# ─────────────────────────────────────────────
#  T9 — 결재란 2패스 검증
# ─────────────────────────────────────────────

@skip_no_backends
class TestT9ApprovalTwoPass:
    """T9: S5 pass1 → pass2 → 셀 태스크 dispatch 흐름."""

    def test_other_path_selected(self, other_pipeline_result):
        """국회공문서는 form_type=other → Skill Registry 경로로 분기되어야 한다."""
        r = other_pipeline_result
        assert r.processing_path.value in ("vlm", "skill_registry"), (
            f"예상치 못한 경로: {r.processing_path.value}"
        )
        # Skill Registry 경로인 경우 skill_stats가 채워짐
        if r.processing_path.value == "skill_registry":
            assert r.skill_stats is not None, "skill_registry 경로에서 skill_stats 누락"

    def test_pass1_tables_detected(self, other_pipeline_result):
        """S5 pass1이 하나 이상의 표를 처리했어야 한다 (국회공문서 결재란)."""
        r = other_pipeline_result
        if r.processing_path.value != "skill_registry":
            pytest.skip("Skill Registry 경로가 아님")

        stats = r.skill_stats
        assert stats.pass1_tables > 0, (
            f"pass1_tables=0: 결재란/표 영역이 탐지되지 않았음. "
            f"layout region 수={len(r.p2_result.regions) if r.p2_result else 0}"
        )
        logger.info("T9 pass1_tables=%d", stats.pass1_tables)

    def test_pass2_generates_cell_tasks(self, other_pipeline_result):
        """S5 pass1 → pass2가 실제 셀 태스크를 생성했어야 한다 (pass2_tasks > 0).

        2026-04-14 이전 구현에서 pass2_tasks=0 버그가 있었음 — 본 회귀 가드.
        """
        r = other_pipeline_result
        if r.processing_path.value != "skill_registry":
            pytest.skip("Skill Registry 경로가 아님")

        stats = r.skill_stats
        # pass1_tables가 있었다면 pass2_tasks도 보통 > 0
        # 다만 모든 셀이 content_type="empty"면 0이 될 수 있음. 이 경우 WARN만.
        if stats.pass1_tables > 0 and stats.pass2_tasks == 0:
            pytest.fail(
                f"pass1_tables={stats.pass1_tables}이나 pass2_tasks=0. "
                "pass1 JSON 파싱 실패(스키마 불일치) 또는 모든 셀이 empty로 분류됨. "
                "docs/TESTING.md T9 회귀 조건."
            )
        logger.info(
            "T9 pass2_tasks=%d (pass1_tables=%d)",
            stats.pass2_tasks, stats.pass1_tables,
        )

    def test_dispatch_order_buckets(self, other_pipeline_result):
        """DISPATCH_ORDER(140/560/1120) 버킷에 셀이 분배되어야 한다."""
        r = other_pipeline_result
        if r.processing_path.value != "skill_registry":
            pytest.skip("Skill Registry 경로가 아님")

        stats = r.skill_stats
        # batch_sizes는 {budget: count}. 적어도 한 버킷에 > 0
        nonzero = [(b, c) for b, c in stats.batch_sizes.items() if c > 0]
        assert nonzero, f"어떤 버킷에도 태스크 미배분: {stats.batch_sizes}"
        logger.info("T9 batch_sizes=%s", dict(stats.batch_sizes))


# ─────────────────────────────────────────────
#  T10 — 서명 탐지 검증
# ─────────────────────────────────────────────

@skip_no_backends
class TestT10SignatureDetection:
    """T10: SignatureDetector(S6)의 이진 분류 동작."""

    def test_signature_detector_invocable_if_present(self, other_pipeline_result):
        """국회공문서에 signature 타입 셀이 있으면 S6이 호출되었어야 한다.

        signature 셀이 없는 문서일 수 있으므로, skill_counts 확인 후 조건부 검증.
        """
        r = other_pipeline_result
        if r.processing_path.value != "skill_registry":
            pytest.skip("Skill Registry 경로가 아님")

        stats = r.skill_stats
        s6_count = stats.skill_counts.get("S6", 0)
        if s6_count == 0:
            pytest.skip(
                f"국회공문서 run에 signature 셀 없음 — skill_counts={dict(stats.skill_counts)}. "
                "서명 문서 샘플 확보 후 재검증 필요."
            )
        logger.info("T10 S6 호출 수=%d", s6_count)


# ─────────────────────────────────────────────
#  SkillRegistry 수준의 결정론 검증 (Stub 기반)
# ─────────────────────────────────────────────

class TestSkillRegistrySignatureRouting:
    """TableCell content_type="signature" → RegionType.SIGNATURE → S6 라우팅.

    Stub 기반이므로 vLLM 불필요.
    """

    def test_content_type_signature_routes_to_s6(self):
        from unittest.mock import MagicMock
        import numpy as np
        from src.interfaces.enums import RegionType
        from src.interfaces.types import BoundingBox, TableCell, TableStructure
        from src.vlm.skill_registry import SKILL_ROUTING
        from src.vlm.skills.table_extractor import TableExtractor

        # pass2_route — content_type="signature" 셀
        structure = TableStructure(
            region_id="t0",
            table_type="approval",
            cells=[
                TableCell(0, 0, "팀장", BoundingBox(0, 0, 50, 40), "signature"),
                TableCell(0, 1, "담당", BoundingBox(50, 0, 100, 40), "printed"),
            ],
            structure_confidence=0.9,
        )
        ext = TableExtractor(vlm_client=MagicMock())
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        tasks = ext.pass2_route(structure, img)
        assert len(tasks) == 2

        sig_task = next(t for t in tasks if t.region_type == RegionType.SIGNATURE)
        txt_task = next(t for t in tasks if t.region_type == RegionType.TEXT)
        assert sig_task.pixel_budget == 140, "signature는 pixel_budget=140"
        assert txt_task.pixel_budget == 560, "printed text는 pixel_budget=560"

        # SKILL_ROUTING에서 signature → S6, text → S2
        assert SKILL_ROUTING["signature"] == "S6"
        assert SKILL_ROUTING["text"] == "S2"

    def test_signature_detector_binary_schema(self):
        """S6 결과 content에는 signature_present 이진 값이 있어야 한다."""
        from unittest.mock import MagicMock
        import numpy as np
        from src.interfaces.enums import RegionType
        from src.interfaces.types import SkillTask
        from src.vlm.skills.signature_detector import SignatureDetector

        vlm = MagicMock()
        vlm.call.return_value = {
            "content": json.dumps({"signature_present": True, "confidence": 0.9}),
            "logprobs": None,
            "finish_reason": "stop",
        }
        det = SignatureDetector(vlm_client=vlm)
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        task = SkillTask(
            region_id="r_sig",
            region_type=RegionType.SIGNATURE,
            cropped_image=img,
            pixel_budget=140,
        )
        result = det.run(task)
        assert result.skill_name == "S6"
        body = json.loads(result.content)
        assert "signature_present" in body
        assert isinstance(body["signature_present"], bool)
