"""P2.5-C — ResolutionRouter

영역 타입별 pixel_budget 할당 + bbox 크롭 + 배치 그룹화.
모든 영역에 동일 해상도를 적용하면 표/수기 기입란에서 부족하고
헤더/푸터에서 낭비가 발생합니다.

동일 pixel_budget끼리 그룹화해야 vLLM 내부 패딩 오버헤드가 없음.
해상도가 다른 이미지를 섞으면 최대 해상도에 맞춰 패딩 → VRAM 낭비.

Input:  LayoutResult + PreprocessedImage + dict[str, InstructionSpec]
Output: dict[int, list[CroppedRegion]]  — pixel_budget별 배치 그룹
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.interfaces.enums import RegionType
from src.interfaces.types import (
    CroppedRegion,
    InstructionSpec,
    LayoutRegion,
    LayoutResult,
    PreprocessedImage,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  기본 pixel_budget 매핑
# ─────────────────────────────────────────────

DEFAULT_PIXEL_BUDGETS: dict[str, int] = {
    "table":    1120,   # 셀 경계·미세 글씨 → 최고 해상도
    "seal":      560,   # 원형 배치 텍스트
    "text":      280,   # 일반 텍스트
    "formula":   280,
    "chart":     280,
    "figure":    140,
    "header":    140,   # 저해상도로 충분
    "footer":    140,
}


@dataclass
class ResolutionRouterConfig:
    """P2.5-C 설정."""

    # 영역 타입별 pixel_budget (기본값 오버라이드 가능)
    pixel_budgets: Optional[dict[str, int]] = None

    # 크롭 패딩 (px) — bbox 주변 여유
    crop_padding: int = 5


class ResolutionRouter:
    """P2.5-C — 영역별 pixel_budget 할당 + bbox 크롭 + 배치 그룹화.

    사용:
        router = ResolutionRouter()
        groups = router.route(layout, preprocessed, instructions)
        # groups: {1120: [CroppedRegion, ...], 280: [CroppedRegion, ...], ...}
    """

    def __init__(self, config: Optional[ResolutionRouterConfig] = None):
        self.cfg = config or ResolutionRouterConfig()
        self._budgets = dict(DEFAULT_PIXEL_BUDGETS)
        if self.cfg.pixel_budgets:
            self._budgets.update(self.cfg.pixel_budgets)

    def get_pixel_budget(self, region_type: str) -> int:
        """영역 타입에 대한 pixel_budget 반환."""
        return self._budgets.get(region_type, 280)

    def route(
        self,
        layout: LayoutResult,
        preprocessed: PreprocessedImage,
        instructions: dict[str, InstructionSpec],
    ) -> dict[int, list[CroppedRegion]]:
        """영역별 크롭 + pixel_budget 할당 + 배치 그룹화.

        Args:
            layout: P2.5-A 정제 완료 결과
            preprocessed: P1 출력 (전처리 이미지)
            instructions: {region_id → InstructionSpec} (P2.5-B 출력)

        Returns:
            {pixel_budget → [CroppedRegion, ...]} 배치 그룹
        """
        image_rgb = preprocessed.image_array
        groups: dict[int, list[CroppedRegion]] = {}

        for idx in layout.reading_order:
            if idx >= len(layout.regions):
                continue

            region = layout.regions[idx]
            spec = instructions.get(region.region_id)
            if spec is None:
                logger.warning(
                    "[P2.5-C] InstructionSpec 없음: %s — 스킵", region.region_id
                )
                continue

            # bbox 크롭
            crop = self._crop_region(image_rgb, region)
            if crop.size == 0:
                logger.warning(
                    "[P2.5-C] 빈 크롭: %s — 스킵", region.region_id
                )
                continue

            # pixel_budget: InstructionSpec에 명시되어 있으면 사용, 아니면 기본값
            budget = spec.pixel_budget or self.get_pixel_budget(
                region.region_type.value
                if hasattr(region.region_type, "value")
                else str(region.region_type)
            )

            cropped = CroppedRegion(
                region_id=region.region_id,
                region_type=region.region_type,
                cropped_image=crop,
                pixel_budget=budget,
                instruction_spec=spec,
            )

            groups.setdefault(budget, []).append(cropped)

        total = sum(len(v) for v in groups.values())
        logger.info(
            "[P2.5-C] 크롭 완료: %d regions → %d groups (%s)",
            total,
            len(groups),
            ", ".join(f"{k}tok:{len(v)}" for k, v in sorted(groups.items())),
        )

        return groups

    def _crop_region(
        self, image_rgb: np.ndarray, region: LayoutRegion,
    ) -> np.ndarray:
        """bbox 크롭 (패딩 적용, 이미지 경계 클램프)."""
        h, w = image_rgb.shape[:2]
        pad = self.cfg.crop_padding
        b = region.bbox

        x1 = max(0, b.x1 - pad)
        y1 = max(0, b.y1 - pad)
        x2 = min(w, b.x2 + pad)
        y2 = min(h, b.y2 + pad)

        crop = image_rgb[y1:y2, x1:x2]
        return crop
