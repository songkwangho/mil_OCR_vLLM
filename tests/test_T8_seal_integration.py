"""T8 — 인장(직인) 인식 통합 검증.

TESTING.md §2 T8 세부 검증:
  - 허프 성공 케이스: `unwrapped=true`, VLM이 직선화된 이미지 수신 확인
  - 허프 실패 케이스: `unwrapped=false`, context_hint 포함 원본 크롭 수신 확인
  - 허프 실패율이 30%+ 이면 극좌표 변환을 선택적 최적화로 격하 검토

본 테스트는 SealPreprocessor + SealReader 통합 동작을 Stub VLM으로 검증.
실제 허프 탐지 자체(이미지 알고리즘)는 `test_seal_preprocessor.py`에서 커버.

실행:
  python -m pytest tests/test_T8_seal_integration.py -v
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import RegionType
from src.interfaces.types import SealProcessResult, SkillTask
from src.preprocess.seal_preprocessor import SealPreprocessor
from src.vlm.skills.seal_reader import SealReader

logger = logging.getLogger(__name__)

SAMPLE_PATH = _PROJECT_ROOT / "data" / "raw" / "국회공문서.jpg"


# ─────────────────────────────────────────────
#  fixture
# ─────────────────────────────────────────────

def _make_seal_task(size: int = 200) -> SkillTask:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 20, (220, 20, 20), thickness=6)
    return SkillTask(
        region_id="r_seal",
        region_type=RegionType.SEAL,
        cropped_image=img,
        pixel_budget=560,
    )


def _stub_vlm(content: str) -> MagicMock:
    client = MagicMock()
    client.call.return_value = {"content": content, "logprobs": None, "finish_reason": "stop"}
    return client


# ─────────────────────────────────────────────
#  허프 성공 케이스 — BUDGET_UNWRAPPED=1120
# ─────────────────────────────────────────────

class TestT8SealHoughSuccess:
    """허프 원 탐지 성공 → 극좌표 직선화 이미지 + pixel_budget=1120."""

    def test_unwrapped_uses_1120_budget(self):
        vlm = _stub_vlm(json.dumps({"text": "국회의장", "confidence": 0.9}))
        # preprocessor를 강제로 unwrapped=True로 mock
        preprocessor = MagicMock(spec=SealPreprocessor)
        unwrapped_img = np.full((100, 300, 3), 255, dtype=np.uint8)
        preprocessor.process.return_value = SealProcessResult(
            image=unwrapped_img,
            unwrapped=True,
            context_hint="직선화된 원형 인장 텍스트",
        )

        reader = SealReader(vlm_client=vlm, seal_preprocessor=preprocessor)
        result = reader.run(_make_seal_task())

        # 호출 시 pixel_budget=1120 (BUDGET_UNWRAPPED) 전달
        _, kwargs = vlm.call.call_args
        assert kwargs["pixel_budget"] == 1120, (
            f"unwrapped=True일 때 1120이어야 하나 {kwargs['pixel_budget']}"
        )
        assert "hough_circle_detection_failed" not in result.warnings
        assert result.content == "국회의장"
        assert result.confidence == pytest.approx(0.9)
        logger.info("T8 허프 성공 — pixel_budget=%d, text=%s", kwargs["pixel_budget"], result.content)

    def test_unwrapped_context_hint_in_system_prompt(self):
        """context_hint가 system_prompt에 주입되어야 한다."""
        vlm = _stub_vlm(json.dumps({"text": "국회의장", "confidence": 0.9}))
        preprocessor = MagicMock(spec=SealPreprocessor)
        preprocessor.process.return_value = SealProcessResult(
            image=np.full((100, 300, 3), 255, dtype=np.uint8),
            unwrapped=True,
            context_hint="UNIQUE_HINT_TOKEN_직선화",
        )
        reader = SealReader(vlm_client=vlm, seal_preprocessor=preprocessor)
        reader.run(_make_seal_task())

        _, kwargs = vlm.call.call_args
        assert "UNIQUE_HINT_TOKEN_직선화" in kwargs["system_prompt"]


# ─────────────────────────────────────────────
#  허프 실패 케이스 — BUDGET_FALLBACK=560 + warning
# ─────────────────────────────────────────────

class TestT8SealHoughFallback:
    """허프 탐지 실패 → 원본 크롭 + pixel_budget=560 + 경고 필드."""

    def test_fallback_uses_560_budget_and_warning(self):
        vlm = _stub_vlm(json.dumps({"text": "인장 텍스트", "confidence": 0.5}))
        preprocessor = MagicMock(spec=SealPreprocessor)
        original_img = np.full((200, 200, 3), 255, dtype=np.uint8)
        preprocessor.process.return_value = SealProcessResult(
            image=original_img,
            unwrapped=False,
            context_hint="원형 인장 — 원본 크롭",
        )

        reader = SealReader(vlm_client=vlm, seal_preprocessor=preprocessor)
        result = reader.run(_make_seal_task())

        _, kwargs = vlm.call.call_args
        assert kwargs["pixel_budget"] == 560, (
            f"unwrapped=False일 때 560이어야 하나 {kwargs['pixel_budget']}"
        )
        assert "hough_circle_detection_failed" in result.warnings
        logger.info("T8 허프 실패 — pixel_budget=%d, warnings=%s", kwargs["pixel_budget"], result.warnings)


# ─────────────────────────────────────────────
#  실제 이미지 기반 허프 성공/실패 비율 — 파일럿 측정
# ─────────────────────────────────────────────

class TestT8HoughSuccessRate:
    """실제 데이터로 허프 성공률 파일럿 측정.

    30% 이상 실패 시 극좌표 변환을 선택적 최적화로 격하 — TESTING.md 권고.
    현재 가진 샘플이 국회공문서 1종이어서 통계적 의미는 적지만 회귀 가드용.
    """

    @pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="국회공문서 샘플 없음")
    def test_full_document_no_exception(self):
        pre = SealPreprocessor()
        bgr = cv2.imread(str(SAMPLE_PATH))
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = pre.process(rgb)
        # 전체 페이지는 허프 실패가 정상 (seal 영역만 대상으로 해야 함)
        assert isinstance(result.unwrapped, bool)
        assert result.image.size > 0
        logger.info("T8 전체 페이지 투입 → unwrapped=%s", result.unwrapped)
