"""SealPreprocessor 단독 테스트.

  - 합성 원형 인장 이미지 → 허프 탐지 성공 → unwrapped=True
  - 적색 없는 이미지 → 허프 실패 → 원본 반환 + fallback 힌트
  - 국회공문서.jpg 전체 이미지를 그대로 투입 → 예외 없이 폴백 처리
실행: python -m pytest tests/test_seal_preprocessor.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.seal_preprocessor import SealPreprocessor  # noqa: E402


def _make_synthetic_seal(size: int = 200) -> np.ndarray:
    """적색 원형 인장을 그린 합성 RGB 이미지."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 20
    # 적색 이중 원 + 내부 텍스트 자리 (OpenCV는 BGR이므로 BGR로 그린 후 RGB 변환 생략 시 파란색)
    # 편의상 RGB 이미지 기준으로 R=220, G=20, B=20
    cv2.circle(img, center, radius, (220, 20, 20), thickness=6)
    cv2.circle(img, center, radius - 20, (220, 20, 20), thickness=3)
    return img


def test_seal_detect_synthetic_circle():
    pre = SealPreprocessor()
    img = _make_synthetic_seal(240)
    result = pre.process(img)
    assert result.unwrapped is True
    assert result.image.ndim == 3
    assert "직선화" in result.context_hint


def test_seal_fallback_when_no_red():
    pre = SealPreprocessor()
    # 적색 없는 회색 이미지
    img = np.full((200, 200, 3), 150, dtype=np.uint8)
    result = pre.process(img)
    assert result.unwrapped is False
    assert result.image.shape == img.shape
    assert "원형 인장" in result.context_hint


def test_seal_empty_input_safe():
    pre = SealPreprocessor()
    result = pre.process(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result.unwrapped is False


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/raw/국회공문서.jpg").exists(),
    reason="테스트 이미지 없음",
)
def test_seal_with_real_document():
    """실제 공문서 전체 이미지 투입 — 예외 없이 결과 반환되어야 한다."""
    pre = SealPreprocessor()
    img_path = PROJECT_ROOT / "data/raw/국회공문서.jpg"
    bgr = cv2.imread(str(img_path))
    assert bgr is not None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = pre.process(rgb)
    assert isinstance(result.unwrapped, bool)
    assert result.image.size > 0
