"""SealPreprocessor — 원형 인장(직인/관인) 기하 전처리.

PIPELINE.md §4-1 구현.
  1) HSV 적색 마스크 추출
  2) 허프 원 탐지로 인장 경계 검출
  3) 극좌표 → 직교좌표 언래핑 (원형 텍스트 직선화)
허프 실패 시 원본 크롭을 그대로 반환 (폴백, 예외 발생 금지).

파인튜닝 대상 아닌 순수 기하학적 변환이며, military/other 경로 공통 사용.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.interfaces.types import SealProcessResult

logger = logging.getLogger(__name__)


@dataclass
class SealPreprocessorConfig:
    """SealPreprocessor 설정값."""
    # HSV 적색 범위 (OpenCV H: 0-179)
    red_lower1: tuple = (0, 80, 60)
    red_upper1: tuple = (10, 255, 255)
    red_lower2: tuple = (170, 80, 60)
    red_upper2: tuple = (179, 255, 255)
    # 허프 원 탐지
    hough_dp: float = 1.0
    hough_min_dist_ratio: float = 0.5   # min(h,w) * ratio
    hough_param1: int = 100
    hough_param2: int = 25
    min_radius_ratio: float = 0.25
    max_radius_ratio: float = 0.55
    # 극좌표 언래핑
    unwrapped_height: int = 120
    unwrapped_width: int = 720


class SealPreprocessor:
    """원형 인장 극좌표 직선화 프리프로세서."""

    UNWRAPPED_HINT = "인장 텍스트를 직선화한 이미지입니다. 좌측에서 우측으로 읽으세요."
    FALLBACK_HINT = "원형 인장 이미지입니다. 원형으로 배치된 텍스트를 인식하세요."

    def __init__(self, config: Optional[SealPreprocessorConfig] = None):
        self.cfg = config or SealPreprocessorConfig()

    def process(self, seal_crop: np.ndarray) -> SealProcessResult:
        """seal_crop(RGB) → SealProcessResult."""
        if seal_crop is None or seal_crop.size == 0:
            return SealProcessResult(
                image=seal_crop if seal_crop is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                unwrapped=False,
                context_hint=self.FALLBACK_HINT,
            )

        try:
            red_mask = self._extract_red(seal_crop)
            circle = self._detect_circle(red_mask)
        except Exception as e:
            logger.warning("[SealPreprocessor] 탐지 실패, 원본 반환: %s", e)
            circle = None

        if circle is None:
            return SealProcessResult(
                image=seal_crop,
                unwrapped=False,
                context_hint=self.FALLBACK_HINT,
            )

        try:
            unwrapped = self._polar_to_rect(seal_crop, circle)
        except Exception as e:
            logger.warning("[SealPreprocessor] 언래핑 실패, 원본 반환: %s", e)
            return SealProcessResult(
                image=seal_crop,
                unwrapped=False,
                context_hint=self.FALLBACK_HINT,
            )

        return SealProcessResult(
            image=unwrapped,
            unwrapped=True,
            context_hint=self.UNWRAPPED_HINT,
        )

    def _extract_red(self, image_rgb: np.ndarray) -> np.ndarray:
        """HSV 적색 두 범위 합성 마스크."""
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array(self.cfg.red_lower1), np.array(self.cfg.red_upper1))
        m2 = cv2.inRange(hsv, np.array(self.cfg.red_lower2), np.array(self.cfg.red_upper2))
        mask = cv2.bitwise_or(m1, m2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _detect_circle(self, red_mask: np.ndarray) -> Optional[tuple[int, int, int]]:
        """허프 원 변환으로 가장 큰 적색 원 반환 (cx, cy, r)."""
        h, w = red_mask.shape[:2]
        if h < 20 or w < 20:
            return None

        # 마스크 자체에 GaussianBlur 후 Hough
        blurred = cv2.GaussianBlur(red_mask, (5, 5), 1.5)

        min_dim = min(h, w)
        min_radius = max(10, int(min_dim * self.cfg.min_radius_ratio))
        max_radius = max(min_radius + 1, int(min_dim * self.cfg.max_radius_ratio))
        min_dist = max(10, int(min_dim * self.cfg.hough_min_dist_ratio))

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.cfg.hough_dp,
            minDist=min_dist,
            param1=self.cfg.hough_param1,
            param2=self.cfg.hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None or len(circles) == 0:
            return None
        circles = np.round(circles[0]).astype(int)
        # 반지름 큰 순
        best = max(circles, key=lambda c: c[2])
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
        return cx, cy, r

    def _polar_to_rect(
        self, image_rgb: np.ndarray, circle: tuple[int, int, int]
    ) -> np.ndarray:
        """원 중심 기준 극좌표 언래핑."""
        cx, cy, r = circle
        out_w = self.cfg.unwrapped_width
        out_h = self.cfg.unwrapped_height
        # cv2.warpPolar: INVERSE_MAP=False → 직교→극좌표. 우리는 극좌표→직교를 원함.
        # WARP_POLAR_LINEAR + WARP_INVERSE_MAP 조합을 쓰면 src를 극좌표로 해석.
        # 간단히: 입력 이미지에서 중심 (cx,cy) 기준 반지름 r을 펼쳐 (out_w, out_h) 생성.
        polar = cv2.warpPolar(
            image_rgb,
            (out_w, out_h),
            (float(cx), float(cy)),
            float(r),
            cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR,
        )
        # warpPolar는 각도 축이 행(행=각도, 열=반지름)이므로 전치해 텍스트가 수평이 되도록.
        # 결과 shape: (out_h=각도, out_w=반지름). 텍스트가 가로로 읽히게 하려면
        # 각도 축을 가로로 보내고(transpose) 상하 반전 필요할 수 있음.
        # 실제로 seal 텍스트는 바깥 링에 있으므로 바깥쪽(큰 반지름)이 위로 오게 좌우 반전.
        unwrapped = cv2.transpose(polar)
        unwrapped = cv2.flip(unwrapped, 0)  # 바깥 링을 위로
        return unwrapped
