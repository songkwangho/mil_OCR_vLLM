"""Skill Registry — other 경로 영역 유형별 Skill 디스패치.

PIPELINE.md §4-2 / §5 구현.
  - region_type → Skill 클래스 매핑 (SKILL_ROUTING)
  - Skill 인스턴스 lazy 초기화
  - DISPATCH_ORDER=[140, 560, 1120] 순 배치 실행
  - SkillTask 배열 → SkillResult 배열

실제 오케스트레이터 연결은 orchestrator._process_other_document_v3 참조.
"""

from __future__ import annotations

import asyncio
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
from src.vlm.budget_config import DISPATCH_ORDER, PIXEL_BUDGETS as PIXEL_BUDGET
from src.vlm.skills.handwriting_reader import HandwritingReader
from src.vlm.skills.printed_text_reader import PrintedTextReader
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

# PIXEL_BUDGET, DISPATCH_ORDER는 budget_config.py에서 import (위 참조)

# 경로별 크롭 패딩 비율 (bbox 대비) — other 경로 전용 (Skill Registry).
# military 경로용은 resolution_router.DEFAULT_CROP_PADDING_RATIO에 정의되어 있다
# (other 경로는 raw layout bbox를 받으므로 military보다 더 넓은 패딩이 필요).
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
    """Skill 인스턴스 lazy 관리 + 태스크 디스패치.

    S2 PrintedTextReader / S3 HandwritingReader는 공통 도메인 서비스이므로
    Orchestrator가 생성한 단일 인스턴스를 주입받습니다. 이렇게 하면
    StructuredExtractor(military 경로)와 SkillRegistry(other 경로)가 동일한
    S2/S3 인스턴스를 참조하게 됩니다.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        printed_text_reader: PrintedTextReader,
        handwriting_reader: HandwritingReader,
    ):
        self.vlm = vlm_client
        self._skills: dict[str, object] = {
            "S2": printed_text_reader,
            "S3": handwriting_reader,
        }

    # Skill lazy 초기화 — S4/S5/S6만 자체 생성, S2/S3는 주입받음
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
            inst = None
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
    #  비동기 디스패치 — 동일 budget 그룹 내 병렬 처리
    # ─────────────────────────────────────────────
    MAX_CONCURRENT_REQUESTS = 16

    async def dispatch_async(
        self,
        tasks: list[SkillTask],
        stats: Optional[SkillDispatchStats] = None,
    ) -> list[SkillResult]:
        """DISPATCH_ORDER 순 그룹 처리. 같은 budget 그룹 내 region은 asyncio.gather로 병렬 실행.

        그룹 간 순서(140→560→1120)는 유지해 동일 system_prompt prefix caching 이득을
        살린다. 동시 요청 수는 `MAX_CONCURRENT_REQUESTS`로 제한.
        """
        stats = stats or SkillDispatchStats()
        buckets: dict[int, list[SkillTask]] = {b: [] for b in DISPATCH_ORDER}
        for t in tasks:
            b = _nearest_bucket(t.pixel_budget)
            buckets.setdefault(b, []).append(t)

        # 입력 순서 복원용 index
        task_index: dict[int, int] = {id(t): i for i, t in enumerate(tasks)}
        indexed_results: list[Optional[SkillResult]] = [None] * len(tasks)

        t_start = time.time()
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        async def run_one(task: SkillTask) -> tuple[int, SkillResult, str, float]:
            skill_name = self._resolve_skill(task)
            async with semaphore:
                t0 = time.time()
                sr = await self._run_single_async(task, skill_name)
                elapsed_ms = (time.time() - t0) * 1000
            return task_index[id(task)], sr, skill_name, elapsed_ms

        for budget in DISPATCH_ORDER:
            bucket = buckets.get(budget, [])
            stats.batch_sizes[budget] = len(bucket)
            if not bucket:
                continue
            awaited = await asyncio.gather(
                *[run_one(t) for t in bucket],
                return_exceptions=True,
            )
            for item in awaited:
                if isinstance(item, Exception):
                    logger.error("[SkillRegistry] dispatch_async 중 예외: %s", item)
                    continue
                idx, sr, skill_name, elapsed_ms = item
                stats.skill_counts[skill_name] = stats.skill_counts.get(skill_name, 0) + 1
                stats.skill_time_ms[skill_name] = (
                    stats.skill_time_ms.get(skill_name, 0.0) + elapsed_ms
                )
                indexed_results[idx] = sr

        stats.total_ms = (time.time() - t_start) * 1000
        # None 은 예외 발생한 슬롯 — 드물게 전체 실패만 해당.
        return [r for r in indexed_results if r is not None]

    async def _run_single_async(
        self, task: SkillTask, skill_name: str
    ) -> SkillResult:
        """비동기 단일 Skill 호출. run_async 없는 skill은 to_thread 폴백."""
        skill = self._get_skill(skill_name)
        if skill is None:
            logger.warning(
                "[SkillRegistry] 미등록 skill=%s region=%s", skill_name, task.region_id
            )
            return SkillResult(
                region_id=task.region_id,
                skill_name=skill_name,
                content="",
                confidence=0.0,
                content_type="error",
                warnings=[f"skill_not_registered: {skill_name}"],
            )
        try:
            run_async = getattr(skill, "run_async", None)
            if run_async is not None:
                return await run_async(task)
            return await asyncio.to_thread(skill.run, task)
        except Exception as e:
            logger.error(
                "[SkillRegistry] %s run_async 실패 region=%s: %s",
                skill_name, task.region_id, e,
            )
            return SkillResult(
                region_id=task.region_id,
                skill_name=skill_name,
                content="",
                confidence=0.0,
                content_type="error",
                warnings=[f"skill_run_failed: {type(e).__name__}"],
            )

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
            logger.warning("[SkillRegistry] 미등록 skill=%s region=%s",
                           skill_name, task.region_id)
            return SkillResult(
                region_id=task.region_id,
                skill_name=skill_name,
                content="",
                confidence=0.0,
                content_type="error",
                warnings=[f"skill_not_registered: {skill_name}"],
            )
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


