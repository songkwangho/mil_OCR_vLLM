"""P2.5-C — ResolutionRouter

영역 타입별 pixel_budget 할당 + bbox 크롭 + 48px 정렬 + 배치 그룹화.

핵심 메커니즘:
  1. 영역 타입별 pixel_budget 차등 할당 (table/handwritten 1120, text/seal 560, ...)
  2. CROP_PADDING_RATIO 비율 패딩 (원본 이미지 맥락 포함, 공백 패딩 금지)
  3. 48px 배수 정렬 — SigLIP 16×16 패치 + 3×3 average pooling 효율 최적화
  4. 동일 pixel_budget끼리 그룹화 → vLLM 내부 패딩 오버헤드 제거

Input:  LayoutResult + PreprocessedImage + dict[str, InstructionSpec]
Output: dict[int, list[CroppedRegion]]  — pixel_budget별 배치 그룹

Reference:
  - AI_INFERENCE.md §4-5: PIXEL_BUDGET, CROP_PADDING_RATIO, 48px 정렬 근거
  - PIPELINE.md §3-4: ResolutionRouter._align_to_48px
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
from src.vlm.budget_config import PIXEL_BUDGETS as DEFAULT_PIXEL_BUDGETS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  기본 pixel_budget 매핑 — budget_config.py에서 import
# ─────────────────────────────────────────────
# DEFAULT_PIXEL_BUDGETS는 위 import 참조 (src.vlm.budget_config.PIXEL_BUDGETS)


# 경로별 크롭 패딩 비율 (bbox 대비) — military 경로 전용.
# other 경로용은 skill_registry.CROP_PADDING_RATIO에 따로 정의되어 있다
# (TemplateAugmentor가 정확한 bbox를 제공하므로 military는 더 짧은 패딩을 사용).
DEFAULT_CROP_PADDING_RATIO: dict[str, float] = {
    "table":             0.02,   # 2% — 열/행 헤더만 포함 (상·하단 인접 영역 오염 방지)
    "handwritten_field": 0.15,   # 15% — 양식 레이블·경계선
    "seal":              0.10,
    "text":              0.05,
    "default":           0.05,
}

# 영역 타입별 패딩 절대 상한(px) — 큰 bbox에서 비율 × 면적이 과도해지는 것 방지
# 예: 1854×1950 table에 5% 적용 시 pad_x=92, pad_y=97 → 인접 writer 영역 침범
DEFAULT_CROP_PADDING_MAX_PX: dict[str, int] = {
    "table":             30,
    "figure":            30,
    "chart":             30,
    "formula":           24,
    "handwritten_field": 24,
    "seal":              40,
    "text":              16,
    "header":            16,
    "footer":            16,
    "signature":         24,
    "default":           24,
}


# SigLIP 패치 크기: 16×16 패치 → 3×3 블록 average pooling = 48px 단위
SIGLIP_PATCH_ALIGN = 48


@dataclass
class ResolutionRouterConfig:
    """P2.5-C 설정."""

    # 영역 타입별 pixel_budget (기본값 오버라이드 가능)
    pixel_budgets: Optional[dict[str, int]] = None

    # 크롭 패딩 비율 (영역 타입별, 0.0~1.0). None이면 DEFAULT 사용.
    crop_padding_ratio: Optional[dict[str, float]] = None

    # 최소 패딩 픽셀 (영역이 작아 비율 적용 시 0이 되는 경우 대비)
    min_crop_padding_px: int = 4

    # 48px 배수 정렬 활성화 (SigLIP 패치 효율)
    align_to_siglip_patch: bool = True


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
        self._padding_ratios = dict(DEFAULT_CROP_PADDING_RATIO)
        if self.cfg.crop_padding_ratio:
            self._padding_ratios.update(self.cfg.crop_padding_ratio)
        self._padding_max_px = dict(DEFAULT_CROP_PADDING_MAX_PX)

    def get_pixel_budget(self, region_type: str) -> int:
        """영역 타입에 대한 pixel_budget 반환."""
        return self._budgets.get(region_type, 280)

    def get_padding_ratio(self, region_type: str) -> float:
        """영역 타입에 대한 패딩 비율 반환."""
        return self._padding_ratios.get(
            region_type, self._padding_ratios.get("default", 0.05)
        )

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

            # bbox 크롭 + 영역 타입별 비율 패딩 + 48px 배수 정렬
            crop = self._crop_region(image_rgb, region)
            if crop.size == 0:
                logger.warning(
                    "[P2.5-C] 빈 크롭: %s — 스킵", region.region_id
                )
                continue
            if self.cfg.align_to_siglip_patch:
                crop = self._align_to_48px(crop)

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
        """bbox 크롭 + 영역 타입별 비율 패딩 (원본 이미지 맥락 포함, 경계 클램프).

        공백 패딩이 아닌 원본 이미지의 실제 주변 콘텐츠를 포함합니다.
        영역이 작은 경우 비율 패딩이 0이 되지 않도록 min_crop_padding_px 적용.
        """
        h, w = image_rgb.shape[:2]
        b = region.bbox
        bbox_w = max(1, b.x2 - b.x1)
        bbox_h = max(1, b.y2 - b.y1)

        rt = (
            region.region_type.value
            if hasattr(region.region_type, "value")
            else str(region.region_type)
        )
        ratio = self.get_padding_ratio(rt)
        max_pad = self._padding_max_px.get(rt, self._padding_max_px.get("default", 24))
        pad_x = min(max_pad, max(self.cfg.min_crop_padding_px, int(bbox_w * ratio)))
        pad_y = min(max_pad, max(self.cfg.min_crop_padding_px, int(bbox_h * ratio)))

        x1 = max(0, b.x1 - pad_x)
        y1 = max(0, b.y1 - pad_y)
        x2 = min(w, b.x2 + pad_x)
        y2 = min(h, b.y2 + pad_y)

        return image_rgb[y1:y2, x1:x2]

    def _align_to_48px(self, image: np.ndarray) -> np.ndarray:
        """크롭 이미지를 48px 배수로 리사이즈하여 SigLIP 패치 낭비 방지.

        SigLIP은 16×16 패치로 분할 후 3×3 블록 단위로 average pooling합니다.
        크롭 크기가 48px 배수가 아니면 패딩 픽셀에 토큰이 낭비됩니다.

        - 가장 가까운 48px 배수로 리사이즈 (Lanczos)
        - 최소 한 변 48px 보장
        - 원본이 작아 round 결과가 0이면 48로 보정
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return image

        align = SIGLIP_PATCH_ALIGN
        new_h = max(align, round(h / align) * align)
        new_w = max(align, round(w / align) * align)

        if new_h == h and new_w == w:
            return image

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
