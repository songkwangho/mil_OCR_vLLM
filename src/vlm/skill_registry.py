"""Skill Registry — other 경로 영역 유형별 Skill 디스패치.

PIPELINE.md §4-2 / §5 구현.
  - region_type → Skill 클래스 매핑 (SKILL_ROUTING)
  - Skill 인스턴스 lazy 초기화
  - DISPATCH_ORDER=[140, 560, 1120] 순 배치 실행
  - SkillTask 배열 → SkillResult 배열

실제 오케스트레이터 연결은 orchestrator._process_other_document_v3 참조.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.interfaces.enums import RegionType
from src.interfaces.types import (
    BoundingBox,
    CroppedRegion,
    LayoutRegion,
    LayoutResult,
    PreprocessedImage,
    SkillResult,
    SkillTask,
)
from src.vlm.skills.seal_reader import SealReader
from src.vlm.skills.signature_detector import SignatureDetector
from src.vlm.skills.table_extractor import TableExtractor
from src.vlm.vlm_client import VLMClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  라우팅 테이블
# ─────────────────────────────────────────────

# region_type.value → skill_name
SKILL_ROUTING: dict[str, str] = {
    "text":              "S2",
    "header":            "S2",
    "footer":            "S2",
    "handwritten_field": "S3",
    "seal":              "S4",
    "table":             "S5",
    "signature":         "S6",
    # figure/formula/chart는 스킵 (텍스트 없음)
}

# PIPELINE.md §2-5
PIXEL_BUDGET: dict[str, int] = {
    "text":              560,
    "header":            140,
    "footer":            140,
    "handwritten_field": 1120,
    "seal":              560,
    "table":             1120,
    "signature":         140,
    "figure":            140,
    "formula":           280,
    "chart":             280,
}

CROP_PADDING_RATIO: dict[str, float] = {
    "text":              0.05,
    "header":            0.05,
    "footer":            0.05,
    "handwritten_field": 0.15,
    "seal":              0.10,
    "table":             0.05,
    "signature":         0.10,
    "default":           0.05,
}

DISPATCH_ORDER: list[int] = [140, 560, 1120]


# ─────────────────────────────────────────────
#  실행 통계
# ─────────────────────────────────────────────

@dataclass
class SkillDispatchStats:
    """Skill 디스패치 집계 — run_summary/orchestrator trace용."""
    skill_counts: dict[str, int] = field(default_factory=dict)
    skill_time_ms: dict[str, float] = field(default_factory=dict)
    batch_sizes: dict[int, int] = field(default_factory=dict)  # budget → 태스크 수
    pass1_tables: int = 0
    pass2_tasks: int = 0
    total_ms: float = 0.0


# ─────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────

class SkillRegistry:
    """Skill 인스턴스 lazy 관리 + 태스크 디스패치."""

    def __init__(self, vlm_client: VLMClient):
        self.vlm = vlm_client
        self._skills: dict[str, object] = {}

    # Skill lazy 초기화 — S4/S5/S6만 구현 (S2/S3는 후속 구현, 현재는 더미 라우팅)
    def _get_skill(self, skill_name: str):
        if skill_name in self._skills:
            return self._skills[skill_name]
        if skill_name == "S4":
            inst = SealReader(vlm_client=self.vlm)
        elif skill_name == "S5":
            inst = TableExtractor(vlm_client=self.vlm)
        elif skill_name == "S6":
            inst = SignatureDetector(vlm_client=self.vlm)
        else:
            inst = None  # S2/S3 미구현 — 호출 시 텍스트 Stub 처리
        self._skills[skill_name] = inst
        return inst

    # ─────────────────────────────────────────────
    #  레이아웃 → SkillTask 생성
    # ─────────────────────────────────────────────
    def build_tasks(
        self,
        layout: LayoutResult,
        preprocessed: PreprocessedImage,
    ) -> list[SkillTask]:
        """P2.5-A 결과와 P1 원본 이미지로부터 영역별 SkillTask 목록을 생성.

        table 영역은 제외 (pass1_structure + pass2_route로 별도 처리).
        """
        tasks: list[SkillTask] = []
        image = preprocessed.image_array
        h, w = image.shape[:2]

        for region in layout.regions:
            rt = region.region_type
            rt_val = rt.value if hasattr(rt, "value") else str(rt)
            if rt_val not in SKILL_ROUTING:
                continue
            if rt_val == "table":
                # table은 별도 경로 (dispatch_table_pipeline)
                continue
            crop = _crop_with_padding(image, region.bbox, (h, w), rt_val)
            if crop is None or crop.size == 0:
                continue
            tasks.append(
                SkillTask(
                    region_id=region.region_id,
                    region_type=rt if isinstance(rt, RegionType) else RegionType(rt_val),
                    cropped_image=crop,
                    pixel_budget=PIXEL_BUDGET.get(rt_val, 560),
                )
            )
        return tasks

    # ─────────────────────────────────────────────
    #  배치 디스패치 — DISPATCH_ORDER 순
    # ─────────────────────────────────────────────
    def dispatch(self, tasks: list[SkillTask], stats: Optional[SkillDispatchStats] = None) -> list[SkillResult]:
        """DISPATCH_ORDER 순서로 배치 실행. 단일 프로세스 순차 호출."""
        stats = stats or SkillDispatchStats()
        buckets: dict[int, list[SkillTask]] = {b: [] for b in DISPATCH_ORDER}
        for t in tasks:
            b = _nearest_bucket(t.pixel_budget)
            buckets.setdefault(b, []).append(t)

        results: list[SkillResult] = []
        t_start = time.time()
        for budget in DISPATCH_ORDER:
            bucket = buckets.get(budget, [])
            stats.batch_sizes[budget] = len(bucket)
            if not bucket:
                continue
            for task in bucket:
                skill_name = self._resolve_skill(task)
                stats.skill_counts[skill_name] = stats.skill_counts.get(skill_name, 0) + 1
                t0 = time.time()
                sr = self._run_single(task, skill_name)
                elapsed_ms = (time.time() - t0) * 1000
                stats.skill_time_ms[skill_name] = (
                    stats.skill_time_ms.get(skill_name, 0.0) + elapsed_ms
                )
                results.append(sr)
        stats.total_ms = (time.time() - t_start) * 1000
        return results

    # ─────────────────────────────────────────────
    #  Table 2패스 처리 (S5)
    # ─────────────────────────────────────────────
    def dispatch_tables(
        self,
        layout: LayoutResult,
        preprocessed: PreprocessedImage,
        stats: Optional[SkillDispatchStats] = None,
    ) -> tuple[list[SkillResult], list]:
        """표 영역에 대해 pass1→pass2→dispatch 실행.

        Returns:
            (cell_skill_results, pass1_table_structures)
        """
        stats = stats or SkillDispatchStats()
        image = preprocessed.image_array
        h, w = image.shape[:2]
        s5 = self._get_skill("S5")
        if s5 is None:
            return [], []

        table_regions = [
            r for r in layout.regions
            if (r.region_type.value if hasattr(r.region_type, "value") else r.region_type) == "table"
        ]
        if not table_regions:
            return [], []

        structures = []
        cell_tasks: list[SkillTask] = []
        for region in table_regions:
            crop = _crop_with_padding(image, region.bbox, (h, w), "table")
            if crop is None or crop.size == 0:
                continue
            structure = s5.pass1_structure(crop, region_id=region.region_id)
            stats.pass1_tables += 1
            structures.append(structure)
            tasks = s5.pass2_route(structure, crop)
            cell_tasks.extend(tasks)

        stats.pass2_tasks = len(cell_tasks)
        cell_results = self.dispatch(cell_tasks, stats=stats) if cell_tasks else []
        return cell_results, structures

    # ─────────────────────────────────────────────
    #  내부 유틸
    # ─────────────────────────────────────────────

    def _resolve_skill(self, task: SkillTask) -> str:
        rt_val = task.region_type.value if hasattr(task.region_type, "value") else str(task.region_type)
        return SKILL_ROUTING.get(rt_val, "S2")

    def _run_single(self, task: SkillTask, skill_name: str) -> SkillResult:
        skill = self._get_skill(skill_name)
        if skill is None:
            # S2/S3 미구현 — PrintedTextReader/HandwritingReader 간이 Stub
            return _stub_text_skill(self.vlm, task, skill_name)
        try:
            return skill.run(task)
        except Exception as e:
            logger.error("[SkillRegistry] %s run 실패 region=%s: %s",
                         skill_name, task.region_id, e)
            return SkillResult(
                region_id=task.region_id,
                skill_name=skill_name,
                content="",
                confidence=0.0,
                content_type="error",
                warnings=[f"skill_run_failed: {type(e).__name__}"],
            )


# ─────────────────────────────────────────────
#  모듈 레벨 유틸
# ─────────────────────────────────────────────

def _nearest_bucket(budget: int) -> int:
    """pixel_budget → DISPATCH_ORDER 내 가장 가까운 버킷."""
    return min(DISPATCH_ORDER, key=lambda b: abs(b - budget))


def _crop_with_padding(
    image: np.ndarray,
    bbox: BoundingBox,
    image_shape: tuple[int, int],
    region_type_value: str,
) -> np.ndarray:
    """bbox + 유형별 패딩 비율로 크롭."""
    h, w = image_shape
    ratio = CROP_PADDING_RATIO.get(region_type_value, CROP_PADDING_RATIO["default"])
    bw = bbox.x2 - bbox.x1
    bh = bbox.y2 - bbox.y1
    pad_x = max(3, int(bw * ratio))
    pad_y = max(3, int(bh * ratio))
    x1 = max(0, bbox.x1 - pad_x)
    y1 = max(0, bbox.y1 - pad_y)
    x2 = min(w, bbox.x2 + pad_x)
    y2 = min(h, bbox.y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2].copy()


def _stub_text_skill(vlm: VLMClient, task: SkillTask, skill_name: str) -> SkillResult:
    """S2/S3 Stub — 단순 한국어 텍스트 추출 (guided_json 없이 순수 텍스트).

    TODO: S2 PrintedTextReader / S3 HandwritingReader 정식 구현 시 교체.
    """
    from src.vlm.vlm_client import encode_image_base64

    if skill_name == "S3":
        system = (
            "You are a Korean handwriting recognition specialist.\n"
            "Recognize the handwritten Korean text exactly as written.\n"
            "Common confusion pairs: ㄱ/ㅋ, ㄴ/ㄹ, ㅏ/ㅓ, 1/ㅣ, 0/O. Mark ambiguous chars with [?].\n"
            "Output JSON: {\"text\": \"...\", \"confidence\": 0.0~1.0}."
        )
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["text", "confidence"],
            "additionalProperties": False,
        }
        content_type = "handwritten"
    else:  # S2
        system = (
            "You are a Korean document OCR assistant.\n"
            "Extract all printed Korean text from this region exactly as written.\n"
            "Mark illegible characters as [?].\n"
            "Output JSON: {\"text\": \"...\", \"confidence\": 0.0~1.0}."
        )
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["text", "confidence"],
            "additionalProperties": False,
        }
        content_type = "printed"

    instruction = "위 영역의 한국어 텍스트를 JSON으로 추출하세요."
    if task.context:
        instruction += f"\n[셀 역할] {task.context}"

    try:
        b64 = encode_image_base64(task.cropped_image, max_size=task.pixel_budget)
        resp = vlm.call(
            image_b64=b64,
            instruction=instruction,
            system_prompt=system,
            guided_json=schema,
            pixel_budget=task.pixel_budget,
        )
    except Exception as e:
        return SkillResult(
            region_id=task.region_id,
            skill_name=skill_name,
            content="",
            confidence=0.0,
            content_type=content_type,
            warnings=[f"vlm_call_failed: {type(e).__name__}"],
        )

    raw = resp.get("content", "") or ""
    from src.vlm.skills.seal_reader import _loads_relaxed
    data = _loads_relaxed(raw)
    if data is not None:
        text = str(data.get("text", "") or "")
        try:
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
    else:
        text = raw.strip()
        conf = 0.0

    return SkillResult(
        region_id=task.region_id,
        skill_name=skill_name,
        content=text,
        confidence=conf,
        content_type=content_type,
        raw_response=raw,
    )
