"""P2.5-A.5 TemplateAugmentor — 서식 템플릿 bbox로 PP-DocLayout 누락 영역 보완.

PIPELINE.md / equipment_checklist_design.md 설계 반영.

동작:
  1) configs/form_templates/{form_type}.yaml 로드
  2) form_identifier 매칭 (versions 배열 구조 지원, 없으면 flat 구조)
  3) 선택된 fields 각각에 대해 LayoutResult 영역과 IoU 계산
     - IoU > 0.5 → PP-DocLayout 결과 유지 (실측 bbox가 더 정확)
     - 매칭 없음 → 템플릿 bbox로 신규 LayoutRegion 추가 (augmented)

military 경로 전용. other 경로에서는 호출되지 않음.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.interfaces.enums import AnalysisMode, FormType, RegionType
from src.interfaces.types import BoundingBox, LayoutRegion, LayoutResult

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE_DIR = Path("configs/form_templates")

# IoU 임계값 — 이상이면 PP-DocLayout 탐지 우선, 이하면 템플릿 bbox 추가
IOU_THRESHOLD = 0.5


@dataclass
class TemplateAugmentorStats:
    """TemplateAugmentor 동작 집계 — run_summary.json용."""
    augmented_count: int = 0
    version_selected: Optional[str] = None
    form_identifier_matched: Optional[str] = None
    template_loaded: bool = False
    skipped_reason: Optional[str] = None


class TemplateAugmentor:
    """form_type + form_identifier 기반 LayoutResult 보강."""

    def __init__(self, template_dir: Optional[str] = None):
        self._dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR
        self._cache: dict[str, dict] = {}  # form_type → yaml dict

    # ─────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────
    def augment(
        self,
        layout: LayoutResult,
        form_type: FormType,
        form_identifier: Optional[str] = None,
        stats: Optional[TemplateAugmentorStats] = None,
    ) -> LayoutResult:
        """LayoutResult에 템플릿 누락 영역을 병합한 새 LayoutResult 반환.

        - military 서식이 아니면(`other`) 원본을 그대로 반환.
        - 템플릿 yaml이 없으면 원본을 그대로 반환.
        """
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

        # 템플릿 좌표계(page_size)와 실제 이미지 크기 비율 계산
        template_size = _extract_page_size(template, version_label)
        scale_x, scale_y = _compute_scale(
            template_size, (layout.page_width, layout.page_height)
        )

        existing_bboxes = [r.bbox for r in layout.regions]
        new_regions = list(layout.regions)
        augmented_count = 0
        next_index = len(new_regions) + 1

        for f in fields:
            tpl_bbox = _extract_bbox(f.get("bbox"), scale_x, scale_y)
            if tpl_bbox is None:
                continue
            # 기존 탐지와 IoU 계산
            if any(_iou(tpl_bbox, eb) > IOU_THRESHOLD for eb in existing_bboxes):
                continue
            # 추가 region
            region_type = _parse_region_type(f.get("region_type", "text"))
            region = LayoutRegion(
                region_id=f"t_{next_index:04d}",
                region_type=region_type,
                bbox=tpl_bbox,
                confidence=0.99,  # 템플릿 확정 좌표 — 높은 신뢰도
            )
            new_regions.append(region)
            augmented_count += 1
            next_index += 1

        stats.augmented_count = augmented_count

        if augmented_count == 0:
            return layout

        # reading_order 갱신 — 새 영역은 뒤에 추가
        new_order = list(layout.reading_order) + list(
            range(len(layout.regions), len(new_regions))
        )

        logger.info(
            "[P2.5-A.5] TemplateAugmentor: form_type=%s version=%s augmented=%d",
            form_type.value, version_label, augmented_count,
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
            warnings=list(layout.warnings),
        )

    # ─────────────────────────────────────────────
    #  내부 유틸
    # ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  헬퍼 함수 (모듈 레벨)
# ─────────────────────────────────────────────

def _select_version(
    template: dict, form_identifier: Optional[str]
) -> tuple[list[dict], Optional[str]]:
    """versions 배열 구조 우선 → 매칭 안되면 flat `fields` 구조."""
    versions = template.get("versions")
    if isinstance(versions, list) and versions:
        if form_identifier:
            for ver in versions:
                if ver.get("form_identifier") == form_identifier:
                    return list(ver.get("fields") or []), str(ver.get("version", ""))
        # 매칭 실패 시 최신(첫 번째) 버전
        first = versions[0]
        return list(first.get("fields") or []), str(first.get("version", ""))
    # flat 구조 (단일 버전)
    fields = template.get("fields") or []
    version_label = str(template.get("version", "1.0"))
    return list(fields), version_label


def _extract_page_size(
    template: dict, version_label: Optional[str]
) -> Optional[tuple[int, int]]:
    """versions 안이든 flat이든 page_size 추출."""
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


def _compute_scale(
    template_size: Optional[tuple[int, int]],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """템플릿 좌표계 → 실제 이미지 크기 비율."""
    if template_size is None or not all(template_size):
        return 1.0, 1.0
    tw, th = template_size
    iw, ih = image_size
    if tw <= 0 or th <= 0:
        return 1.0, 1.0
    return iw / tw, ih / th


def _extract_bbox(
    bbox_spec: Any, scale_x: float, scale_y: float
) -> Optional[BoundingBox]:
    """YAML bbox → BoundingBox (스케일 적용)."""
    if isinstance(bbox_spec, dict):
        try:
            x1, y1, x2, y2 = (
                int(bbox_spec["x1"]), int(bbox_spec["y1"]),
                int(bbox_spec["x2"]), int(bbox_spec["y2"]),
            )
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


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union
