"""P2.5-A — LayoutPostProcessor

PP-DocLayout 원시 탐지 결과(RawLayoutResult)를 정제하여
VLM이 의미 있는 크롭 이미지를 받도록 보장합니다.

정제 단계:
  1. 미소 박스 제거 (6px 미만 — PaddleOCR-VL 기준)
  2. 중복 박스 제거 (IoU > 0.7, seal 영역은 0.5)
  3. 인접 텍스트 블록 병합 (동일 컬럼 내)
  4. reading_order 재정렬 (제거/병합 반영)

이 단계가 없을 때 발생하는 문제:
  - 6px 점 이미지를 VLM에 전달 → 무의미한 호출 낭비
  - 인장+텍스트 중복 bbox → 동일 영역 이중 처리
  - 단어 단위 분절 텍스트 → 문맥 없는 조각 인식

Input:  RawLayoutResult (P2 출력)
Output: LayoutResult    (정제 완료)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.interfaces.enums import AnalysisMode, RegionType
from src.interfaces.types import (
    BoundingBox,
    LayoutRegion,
    LayoutResult,
    RawLayoutResult,
)
from src.preprocess.bbox_utils import compute_iou

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────

@dataclass
class LayoutPostProcessorConfig:
    """P2.5-A 설정."""

    # 미소 박스 제거 기준 (px): 가로 또는 세로가 이 값 미만이면 제거
    micro_box_px: int = 6

    # 중복 박스 IoU 임계값
    overlap_iou_threshold: float = 0.7

    # seal 영역 IoU 임계값 (인장은 다른 영역과 겹치기 쉬움)
    seal_iou_threshold: float = 0.5

    # 텍스트 블록 병합: x-center 차이 허용 (px)
    merge_x_center_tolerance: int = 20

    # 텍스트 블록 병합: y-gap 허용 (px)
    merge_y_gap_max: int = 15


# ─────────────────────────────────────────────
#  유틸
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  LayoutPostProcessor
# ─────────────────────────────────────────────

class LayoutPostProcessor:
    """P2.5-A — PP-DocLayout 원시 탐지 결과 정제.

    사용:
        postprocessor = LayoutPostProcessor()
        layout = postprocessor.process(raw_layout)
    """

    def __init__(self, config: Optional[LayoutPostProcessorConfig] = None):
        self.cfg = config or LayoutPostProcessorConfig()

    def process(self, raw: RawLayoutResult) -> LayoutResult:
        """RawLayoutResult → LayoutResult (정제 완료).

        Returns:
            LayoutResult: 미소/중복 박스 제거 + 블록 병합 + reading_order 재정렬
        """
        # 1. 미소 박스 제거
        regions, removed_ids = self._filter_micro_boxes(raw.regions)

        # 2. 중복 박스 제거
        regions, overlap_removed = self._filter_overlapping(regions)
        removed_ids.update(overlap_removed)

        # 3. 인접 텍스트 블록 병합
        regions, merged_map = self._merge_adjacent_text_blocks(regions)

        # 4. reading_order 재정렬
        reading_order = self._remap_reading_order(
            raw.reading_order,
            raw.regions,
            regions,
            removed_ids=removed_ids,
            merged_map=merged_map,
        )

        logger.info(
            "[P2.5-A][%s] 정제 완료: %d→%d regions (removed=%d, merged=%d)",
            raw.doc_id,
            len(raw.regions),
            len(regions),
            len(removed_ids),
            len(merged_map),
        )

        return LayoutResult(
            doc_id=raw.doc_id,
            page_width=raw.page_width,
            page_height=raw.page_height,
            regions=regions,
            reading_order=reading_order,
            analysis_mode=raw.analysis_mode,
            removed_count=len(removed_ids),
            merged_count=len(merged_map),
            warnings=list(raw.warnings),
        )

    # ─── 1. 미소 박스 제거 ────────────────────

    def _filter_micro_boxes(
        self, regions: list[LayoutRegion],
    ) -> tuple[list[LayoutRegion], set[str]]:
        """가로 또는 세로가 micro_box_px 미만인 박스 제거.

        Returns:
            (남은 regions, 제거된 region_id 집합)
        """
        kept: list[LayoutRegion] = []
        removed: set[str] = set()
        threshold = self.cfg.micro_box_px

        for r in regions:
            if r.bbox.width < threshold or r.bbox.height < threshold:
                removed.add(r.region_id)
            else:
                kept.append(r)

        if removed:
            logger.debug(
                "[P2.5-A] 미소 박스 제거: %d개 (<%dpx)", len(removed), threshold
            )
        return kept, removed

    # ─── 2. 중복 박스 제거 ────────────────────

    def _filter_overlapping(
        self, regions: list[LayoutRegion],
    ) -> tuple[list[LayoutRegion], set[str]]:
        """IoU 임계값 초과 중복 박스 제거.

        seal 영역: seal_iou_threshold (0.5)
        그 외: overlap_iou_threshold (0.7)
        겹치는 쌍 중 신뢰도 낮은 쪽 제거.

        Returns:
            (남은 regions, 제거된 region_id 집합)
        """
        if len(regions) <= 1:
            return regions, set()

        # 신뢰도 내림차순 정렬
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        kept: list[LayoutRegion] = []
        removed: set[str] = set()

        for region in sorted_regions:
            if region.region_id in removed:
                continue

            # 현재 region과 이전에 남긴 region들 간 IoU 검사
            suppress = False
            for kept_r in kept:
                iou = compute_iou(region.bbox, kept_r.bbox)

                # seal 영역은 더 엄격한 기준
                threshold = self.cfg.seal_iou_threshold if (
                    region.region_type == RegionType.SEAL
                    or kept_r.region_type == RegionType.SEAL
                ) else self.cfg.overlap_iou_threshold

                if iou > threshold:
                    suppress = True
                    break

            if suppress:
                removed.add(region.region_id)
            else:
                kept.append(region)

        if removed:
            logger.debug("[P2.5-A] 중복 박스 제거: %d개", len(removed))
        return kept, removed

    # ─── 3. 인접 텍스트 블록 병합 ──────────────

    def _merge_adjacent_text_blocks(
        self, regions: list[LayoutRegion],
    ) -> tuple[list[LayoutRegion], dict[str, str]]:
        """동일 컬럼 내 인접 text 블록 병합.

        조건: region_type == TEXT, x-center 차이 < tolerance, y-gap < max_gap

        Returns:
            (병합 후 regions, {흡수된 region_id → 병합 대상 region_id})
        """
        text_regions: list[LayoutRegion] = []
        other_regions: list[LayoutRegion] = []

        for r in regions:
            if r.region_type == RegionType.TEXT:
                text_regions.append(r)
            else:
                other_regions.append(r)

        if len(text_regions) <= 1:
            return regions, {}

        # y 좌표 기준 정렬
        text_regions.sort(key=lambda r: (r.bbox.y1, r.bbox.x1))

        merged_map: dict[str, str] = {}  # 흡수된 id → 병합 대상 id
        result_text: list[LayoutRegion] = [text_regions[0]]

        for i in range(1, len(text_regions)):
            curr = text_regions[i]
            prev = result_text[-1]

            # 병합 조건: x-center 근접 + y-gap 작음
            cx_diff = abs(curr.bbox.center[0] - prev.bbox.center[0])
            y_gap = curr.bbox.y1 - prev.bbox.y2

            if (cx_diff <= self.cfg.merge_x_center_tolerance
                    and 0 <= y_gap <= self.cfg.merge_y_gap_max):
                # 병합: prev bbox 확장
                new_bbox = BoundingBox(
                    x1=min(prev.bbox.x1, curr.bbox.x1),
                    y1=prev.bbox.y1,
                    x2=max(prev.bbox.x2, curr.bbox.x2),
                    y2=curr.bbox.y2,
                )
                merged_region = LayoutRegion(
                    region_id=prev.region_id,
                    region_type=RegionType.TEXT,
                    bbox=new_bbox,
                    confidence=max(prev.confidence, curr.confidence),
                    polygon=None,  # 병합 후 polygon은 소실 — 현재 downstream 소비자 없음. P2.5-B/C가 polygon을 참조하면 재검토 필요.
                )
                result_text[-1] = merged_region
                merged_map[curr.region_id] = prev.region_id
            else:
                result_text.append(curr)

        if merged_map:
            logger.debug(
                "[P2.5-A] 텍스트 블록 병합: %d개 흡수", len(merged_map)
            )
        return other_regions + result_text, merged_map

    # ─── 4. reading_order 재정렬 ──────────────

    def _remap_reading_order(
        self,
        original_order: list[int],
        original_regions: list[LayoutRegion],
        new_regions: list[LayoutRegion],
        removed_ids: set[str],
        merged_map: dict[str, str],
    ) -> list[int]:
        """제거/병합 결과를 반영하여 reading_order 재정렬.

        처리 흐름:
          1. original_order의 각 인덱스를 원본 region_id로 변환
          2. removed_ids에 포함된 region_id 제거
          3. merged_map에 포함된 region_id를 병합 대상으로 치환 (중복 제거)
          4. 남은 region_id를 new_regions 기준으로 리넘버링

        Args:
            original_order: P2 원본 reading_order (원본 regions 인덱스)
            original_regions: P2 원본 regions 리스트
            new_regions: 정제 후 regions 리스트
            removed_ids: 제거된 region_id 집합
            merged_map: {흡수된 region_id → 병합 대상 region_id}

        Returns:
            정제 후 regions 기준의 새 reading_order
        """
        # new_regions의 region_id → 새 인덱스 매핑
        new_id_to_idx: dict[str, int] = {
            r.region_id: i for i, r in enumerate(new_regions)
        }

        seen: set[str] = set()
        remapped: list[int] = []

        for orig_idx in original_order:
            if orig_idx >= len(original_regions):
                continue

            rid = original_regions[orig_idx].region_id

            # 제거된 region은 스킵
            if rid in removed_ids:
                continue

            # 병합된 region은 대상으로 치환
            if rid in merged_map:
                rid = merged_map[rid]

            # 중복 방지
            if rid in seen:
                continue
            seen.add(rid)

            # 새 인덱스로 변환
            new_idx = new_id_to_idx.get(rid)
            if new_idx is not None:
                remapped.append(new_idx)

        # reading_order에 없는 새 region 추가 (y1, x1 순)
        for i, r in enumerate(new_regions):
            if i not in remapped:
                remapped.append(i)

        return remapped
