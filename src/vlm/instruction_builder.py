"""P2 레이아웃 결과 → VLM instruction 자동 생성

P2가 검출한 영역 레이블(RegionType)에 따라 VLM에 전달할
instruction 텍스트를 자동 결정합니다.

처리 흐름:
  1. 서식 분류 instruction (전체 문서 이미지)
  2. 필드 추출 instruction (전체 문서 + guided_json)
  3. 영역별 개별 instruction (table, seal 등)

의존: src.interfaces
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.interfaces.enums import RegionType
from src.interfaces.types import LayoutRegion, LayoutResult

from src.vlm.instruction_router import InstructionRouter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  VLM Instruction 템플릿
# ─────────────────────────────────────────────

# 서식 분류용 instruction
CLASSIFICATION_INSTRUCTION = (
    "이 군수 서식 이미지를 분석하여 서식 유형을 분류하세요.\n"
    "가능한 유형: supply_request (보급청구서), maintenance_record (정비기록서), "
    "inventory_sheet (물자현황표), handover_doc (인수인계서), "
    "inspection_report (검사보고서), unknown (미분류)\n"
    "서식 유형만 답하세요."
)


# EXTRACTION_INSTRUCTIONS, REGION_INSTRUCTIONS, TASK_PROMPTS 는
# configs/instruction_mappings.yaml 로 이관됨 → InstructionRouter 사용


# ─────────────────────────────────────────────
#  InstructionTask — VLM 호출 단위
# ─────────────────────────────────────────────

@dataclass
class InstructionTask:
    """VLM에 전달할 단일 작업 단위.

    gemma4_engine이 이 객체를 받아 vLLM API를 호출합니다.
    """
    task_type: str          # classify, extract, region_ocr, region_table
    instruction: str        # VLM instruction 텍스트
    region: Optional[LayoutRegion] = None  # 영역 정보 (crop 대상)
    schema_id: Optional[str] = None        # guided_json용 스키마 ID
    use_guided_json: bool = False          # guided_json 사용 여부
    use_logprobs: bool = False             # logprobs 수집 여부


# ─────────────────────────────────────────────
#  InstructionBuilder
# ─────────────────────────────────────────────

class InstructionBuilder:
    """P2 LayoutResult → VLM InstructionTask 목록 생성.

    P3 처리 흐름:
      1. classify  — 전체 문서로 서식 분류
      2. extract   — 전체 문서 + guided_json으로 필드 추출
      3. region_*  — 표·인장 등 개별 영역 처리

    사용 예시:
        builder = InstructionBuilder()
        tasks = builder.build_classification_task()
        # → P3가 서식 분류 실행 → form_type 결정
        tasks = builder.build_extraction_tasks(layout_result, form_type)
        # → P3가 필드 추출 + 영역별 처리 실행
    """

    def __init__(self, router: InstructionRouter | None = None) -> None:
        self._router = router or InstructionRouter()

    def build_classification_task(self) -> InstructionTask:
        """서식 분류 task 생성 (전체 문서 이미지)."""
        return InstructionTask(
            task_type="classify",
            instruction=CLASSIFICATION_INSTRUCTION,
            use_guided_json=False,
            use_logprobs=True,
        )

    def build_extraction_tasks(
        self,
        layout: LayoutResult,
        form_type: str,
    ) -> list[InstructionTask]:
        """필드 추출 + 영역별 처리 task 목록 생성.

        Args:
            layout: P2 출력 (영역 + 읽기 순서)
            form_type: 서식 유형 (classify task의 결과)

        Returns:
            InstructionTask 목록:
              [0] = 전체 문서 필드 추출 (guided_json + logprobs)
              [1:] = 표/인장 등 개별 영역 처리
        """
        tasks: list[InstructionTask] = []

        # 1. 전체 문서 필드 추출 (guided_json)
        extraction_instruction = self._router.get_extraction_instruction(form_type)
        schema_id = form_type if form_type != "unknown" else "_fallback"

        tasks.append(InstructionTask(
            task_type="extract",
            instruction=extraction_instruction,
            schema_id=schema_id,
            use_guided_json=True,
            use_logprobs=True,
        ))

        # 2. 표 영역 개별 처리 (reading_order 순서)
        if layout is None:
            return tasks
        for idx in layout.reading_order:
            if idx >= len(layout.regions):
                continue
            region = layout.regions[idx]

            if region.region_type == RegionType.TABLE:
                tasks.append(InstructionTask(
                    task_type="region_table",
                    instruction=self._router.get_region_instruction("table"),
                    region=region,
                    use_guided_json=False,
                    use_logprobs=True,
                ))
            elif region.region_type == RegionType.SEAL:
                tasks.append(InstructionTask(
                    task_type="region_ocr",
                    instruction=self._router.get_region_instruction("seal"),
                    region=region,
                    use_guided_json=False,
                    use_logprobs=True,
                ))

        logger.info(
            "InstructionBuilder: form_type=%s, tasks=%d (extract=1, regions=%d)",
            form_type, len(tasks), len(tasks) - 1,
        )
        return tasks

    def get_region_instruction(self, region_type: RegionType) -> str:
        """영역 유형에 대응하는 instruction 텍스트 반환."""
        return self._router.get_region_instruction(region_type.value)

    def get_task_prompt(self, region_type: RegionType) -> str:
        """영역 유형에 대응하는 task prompt prefix 반환."""
        return self._router.get_task_prompt(region_type.value)
