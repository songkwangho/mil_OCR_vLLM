"""P2.5-A.5 TemplateAugmentor — 서식 템플릿 bbox로 LayoutResult 재구성.

IMPLEMENTATION_SPEC 변경 3 반영 — 템플릿 우선 병합 + field_key 부여.

병합 정책 (v3):
  1) 템플릿 field 목록을 순회하며 region 구성 (field_key 부여)
  2) 각 템플릿 field에 대해 PP-DocLayout에서 IoU 최대 region 매칭
     - IoU > 0.5 → PP-DocLayout bbox 재사용 (정밀 좌표) + 템플릿 field_key
     - 매칭 없음 → 템플릿 bbox (field_key와 함께)
  3) PP-DocLayout only 동적 영역(인장·서명 등)은 field_key=None으로 뒤에 보존

military 경로 전용.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.interfaces.enums import AnalysisMode, FormType, RegionType
from src.interfaces.types import BoundingBox, LayoutRegion, LayoutResult

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE_DIR = Path("configs/form_templates")

# PP region의 {CONTAINMENT_THRESHOLD}% 이상이 템플릿 bbox 내부에 있으면 소비
CONTAINMENT_THRESHOLD = 0.7
# 템플릿 bbox가 PP region 내부에 {TEMPLATE_INSIDE_PP_THRESHOLD}% 이상 포함되면
# 해당 PP는 구조적 컨테이너(표 등)로 간주하여 소비하지 않음
TEMPLATE_INSIDE_PP_THRESHOLD = 0.9
# 구조적/동적 영역 — 넓은 템플릿 bbox에 의해 소비되지 않도록 보호
_PROTECTED_REGION_TYPES = {"seal", "signature", "figure", "table"}


@dataclass
class TemplateAugmentorStats:
    """TemplateAugmentor 동작 집계 — run_summary.json용."""
    augmented_count: int = 0            # 템플릿 기준으로 처리된 region 수
    pp_matched_count: int = 0           # PP-DocLayout bbox 재사용 수
    version_selected: Optional[str] = None
    form_identifier_matched: Optional[str] = None
    template_loaded: bool = False
    skipped_reason: Optional[str] = None


class TemplateAugmentor:
    """form_type + form_identifier 기반 LayoutResult 재구성."""

    def __init__(self, template_dir: Optional[str] = None):
        self._dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR
        self._cache: dict[str, dict] = {}

    def augment(
        self,
        layout: LayoutResult,
        form_type: FormType,
        form_identifier: Optional[str] = None,
        stats: Optional[TemplateAugmentorStats] = None,
    ) -> LayoutResult:
        stats = stats or TemplateAugmentorStats()

        if not form_type.is_military():
            stats.skipped_reason = "not_military"
            return layout

        template = self._load_template(form_type.value)
        if template is None:
            stats.skipped_reason = "template_not_found"
            return layout
        stats.template_loaded = True

        fields, version_label = _select_version(template, form_identifier)
        if not fields:
            stats.skipped_reason = "no_fields_in_template"
            return layout

        stats.version_selected = version_label
        stats.form_identifier_matched = form_identifier

        template_size = _extract_page_size(template, version_label)
        scale_x, scale_y = _compute_scale(
            template_size, (layout.page_width, layout.page_height)
        )

        existing_regions = list(layout.regions)
        used_pp_indices: set[int] = set()
        new_regions: list[LayoutRegion] = []
        next_tmpl_idx = 1
        template_driven_count = 0
        pp_matched_count = 0

        # Step 1: 템플릿 field 기준 region 구성
        # 규칙:
        #  ① region_type이 protected(seal/signature/figure/table)이면 소비 금지
        #  ② 템플릿이 PP 내부에 0.9 이상 포함 → PP는 상위 컨테이너 → 소비 금지
        #  ③ PP의 0.7 이상이 템플릿 내부 → PP 소비 (복수 허용)
        #  소비한 PP가 하나라도 있으면 그 union 대신 템플릿 bbox를 채택
        #  (템플릿 bbox는 서식 원본 좌표이므로 여러 PP보다 일관성 있음)
        for f in fields:
            tpl_bbox = _extract_bbox(f.get("bbox"), scale_x, scale_y)
            if tpl_bbox is None:
                continue

            region_type = _parse_region_type(f.get("region_type", "text"))
            field_key: Optional[str] = f.get("field_key")

            consumable: list[int] = []
            for i, pp_region in enumerate(existing_regions):
                if i in used_pp_indices:
                    continue
                pp_rt_val = (
                    pp_region.region_type.value
                    if hasattr(pp_region.region_type, "value")
                    else str(pp_region.region_type)
                )
                if pp_rt_val in _PROTECTED_REGION_TYPES:
                    continue
                c_pp, c_tpl = _containment(pp_region.bbox, tpl_bbox)
                if c_tpl > TEMPLATE_INSIDE_PP_THRESHOLD:
                    # 템플릿이 PP 안에 내포 — PP가 컨테이너 → 건드리지 않음
                    continue
                if c_pp > CONTAINMENT_THRESHOLD:
                    consumable.append(i)

            for i in consumable:
                used_pp_indices.add(i)

            # 템플릿 bbox로 단일 region 생성 (field_key 부여)
            # PP 하나만 매칭돼서 그 bbox가 더 정확할 가능성도 있지만,
            # 여러 PP를 흡수하는 케이스의 일관성을 위해 템플릿 좌표를 기본값으로 채택.
            if len(consumable) == 1 and not _is_small_contained(
                existing_regions[consumable[0]].bbox, tpl_bbox
            ):
                # 1:1 매칭이고 PP가 작은 내포가 아니면 PP bbox 재사용 (실측 좌표 우선)
                pp_region = existing_regions[consumable[0]]
                new_regions.append(LayoutRegion(
                    region_id=pp_region.region_id,
                    region_type=region_type,
                    bbox=pp_region.bbox,
                    confidence=pp_region.confidence,
                    polygon=pp_region.polygon,
                    source="template_matched",
                    field_key=field_key,
                ))
                pp_matched_count += 1
            else:
                new_regions.append(LayoutRegion(
                    region_id=f"t_{next_tmpl_idx:04d}",
                    region_type=region_type,
                    bbox=tpl_bbox,
                    confidence=0.99,
                    source="template",
                    field_key=field_key,
                ))
                next_tmpl_idx += 1
                if consumable:
                    pp_matched_count += len(consumable)

            template_driven_count += 1

        # Step 2: PP-DocLayout only 동적 영역 보존 (인장·서명 등)
        for i, pp_region in enumerate(existing_regions):
            if i not in used_pp_indices:
                new_regions.append(pp_region)

        new_order = list(range(len(new_regions)))

        stats.augmented_count = template_driven_count
        stats.pp_matched_count = pp_matched_count

        logger.info(
            "[P2.5-A.5] TemplateAugmentor: form_type=%s version=%s "
            "template_regions=%d (pp_matched=%d, template_bbox=%d) dynamic_pp=%d",
            form_type.value, version_label,
            template_driven_count,
            pp_matched_count,
            template_driven_count - pp_matched_count,
            len(new_regions) - template_driven_count,
        )

        return LayoutResult(
            doc_id=layout.doc_id,
            page_width=layout.page_width,
            page_height=layout.page_height,
            regions=new_regions,
            reading_order=new_order,
            analysis_mode=layout.analysis_mode,
            removed_count=layout.removed_count,
            merged_count=layout.merged_count,
            augmented_count=template_driven_count,
            warnings=list(layout.warnings),
        )

    def _load_template(self, form_type_value: str) -> Optional[dict]:
        if form_type_value in self._cache:
            return self._cache[form_type_value]
        path = self._dir / f"{form_type_value}.yaml"
        if not path.exists():
            self._cache[form_type_value] = None
            return None
        try:
            import yaml  # type: ignore
        except ImportError:
            logger.warning("[TemplateAugmentor] PyYAML 미설치 — 템플릿 비활성화")
            self._cache[form_type_value] = None
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("[TemplateAugmentor] %s 로드 실패: %s", path, e)
            self._cache[form_type_value] = None
            return None
        self._cache[form_type_value] = data
        return data


def _select_version(template: dict, form_identifier: Optional[str]):
    versions = template.get("versions")
    if isinstance(versions, list) and versions:
        if form_identifier:
            for ver in versions:
                if ver.get("form_identifier") == form_identifier:
                    return list(ver.get("fields") or []), str(ver.get("version", ""))
        first = versions[0]
        return list(first.get("fields") or []), str(first.get("version", ""))
    fields = template.get("fields") or []
    version_label = str(template.get("version", "1.0"))
    return list(fields), version_label


def _extract_page_size(template: dict, version_label: Optional[str]) -> Optional[tuple[int, int]]:
    versions = template.get("versions")
    if isinstance(versions, list) and version_label:
        for ver in versions:
            if str(ver.get("version", "")) == version_label:
                ps = ver.get("page_size")
                if _is_pair(ps):
                    return int(ps[0]), int(ps[1])
    ps = template.get("page_size")
    if _is_pair(ps):
        return int(ps[0]), int(ps[1])
    return None


def _is_pair(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2


def _compute_scale(template_size, image_size) -> tuple[float, float]:
    if template_size is None or not all(template_size):
        return 1.0, 1.0
    tw, th = template_size
    iw, ih = image_size
    if tw <= 0 or th <= 0:
        return 1.0, 1.0
    return iw / tw, ih / th


def _extract_bbox(bbox_spec, scale_x, scale_y) -> Optional[BoundingBox]:
    if isinstance(bbox_spec, dict):
        try:
            x1 = int(bbox_spec["x1"])
            y1 = int(bbox_spec["y1"])
            x2 = int(bbox_spec["x2"])
            y2 = int(bbox_spec["y2"])
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(bbox_spec, (list, tuple)) and len(bbox_spec) == 4:
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox_spec)
        except (TypeError, ValueError):
            return None
    else:
        return None
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1, y1, x2, y2)


def _parse_region_type(value: str) -> RegionType:
    try:
        return RegionType(value)
    except ValueError:
        return RegionType.TEXT


def _containment(pp: BoundingBox, tpl: BoundingBox) -> tuple[float, float]:
    """두 bbox의 상호 포함도.

    Returns:
        (pp_inside_tpl, tpl_inside_pp)
        pp_inside_tpl = intersection / pp.area   (PP의 몇 %가 템플릿 안인가)
        tpl_inside_pp = intersection / tpl.area  (템플릿이 PP에 얼마나 내포되나)
    """
    inter_x1 = max(pp.x1, tpl.x1)
    inter_y1 = max(pp.y1, tpl.y1)
    inter_x2 = min(pp.x2, tpl.x2)
    inter_y2 = min(pp.y2, tpl.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0, 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    pp_area = max(1, (pp.x2 - pp.x1) * (pp.y2 - pp.y1))
    tpl_area = max(1, (tpl.x2 - tpl.x1) * (tpl.y2 - tpl.y1))
    return inter / pp_area, inter / tpl_area


def _is_small_contained(pp: BoundingBox, tpl: BoundingBox) -> bool:
    """PP가 템플릿 bbox의 작은 일부(<60%)인 경우 True — 템플릿 bbox를 우선 사용."""
    pp_area = max(1, (pp.x2 - pp.x1) * (pp.y2 - pp.y1))
    tpl_area = max(1, (tpl.x2 - tpl.x1) * (tpl.y2 - tpl.y1))
    return pp_area < 0.6 * tpl_area


