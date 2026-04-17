"""공용 bbox 유틸 — IoU 등 레이아웃 파이프라인 전반에서 재사용."""

from __future__ import annotations

from src.interfaces.types import BoundingBox


def compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    """두 BoundingBox 간 IoU 계산."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
