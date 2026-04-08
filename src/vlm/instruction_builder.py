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
from src.interfaces.types import LayoutRegion, LayoutResult, TASK_PROMPTS

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

# 전체 문서 필드 추출용 instruction (서식 유형별)
EXTRACTION_INSTRUCTIONS: dict[str, str] = {
    "supply_request": (
        "이 보급청구서 이미지에서 모든 필드를 추출하세요.\n"
        "청구부대, 물자코드(NSN), 품목명, 수량, 단가, 합계, 청구일자, "
        "승인일자, 결재권자 등을 포함합니다.\n"
        "수기로 기입된 부분도 정확히 인식하고, 오타가 있으면 교정하세요.\n"
        "지정된 JSON 형식에 맞춰 응답하세요."
    ),
    "maintenance_record": (
        "이 정비기록서 이미지에서 모든 필드를 추출하세요.\n"
        "장비번호, 정비유형, 정비일자, 정비자, 정비내용, 사용부품, "
        "정비결과 등을 포함합니다.\n"
        "수기 기입 부분도 정확히 인식하고, 오타가 있으면 교정하세요.\n"
        "지정된 JSON 형식에 맞춰 응답하세요."
    ),
    "inventory_sheet": (
        "이 물자현황표 이미지에서 모든 필드를 추출하세요.\n"
        "품목명, 물자코드, 재고수량, 보관위치, 불용수량, 적부 판정 등을 포함합니다.\n"
        "수기 기입 부분도 정확히 인식하고, 오타가 있으면 교정하세요.\n"
        "지정된 JSON 형식에 맞춰 응답하세요."
    ),
    "handover_doc": (
        "이 인수인계서 이미지에서 모든 필드를 추출하세요.\n"
        "인계자, 인수자, 인수인계일자, 품목목록, 수량, 상태 등을 포함합니다.\n"
        "수기 기입 부분도 정확히 인식하고, 오타가 있으면 교정하세요.\n"
        "지정된 JSON 형식에 맞춰 응답하세요."
    ),
    "inspection_report": (
        "이 검사보고서 이미지에서 모든 필드를 추출하세요.\n"
        "검사일자, 검사자, 검사유형, 점검항목별 결과, 종합판정 등을 포함합니다.\n"
        "수기 기입 부분도 정확히 인식하고, 오타가 있으면 교정하세요.\n"
        "지정된 JSON 형식에 맞춰 응답하세요."
    ),
    "unknown": (
        "이 군수 서식 이미지에서 가능한 모든 필드를 추출하세요.\n"
        "문서 제목, 날짜, 작성자, 부대코드 및 모든 key-value 쌍과 "
        "표 데이터를 포함합니다.\n"
        "수기 기입 부분도 정확히 인식하세요.\n"
        "지정된 JSON 형식에 맞춰 응답하세요."
    ),
}

# 영역별 개별 instruction
REGION_INSTRUCTIONS: dict[str, str] = {
    "text": "이 영역의 한국어 텍스트를 인식하고 오타를 교정하세요.",
    "table": (
        "이 표의 구조를 분석하고 각 셀의 텍스트를 추출하세요.\n"
        "HTML <table> 형식으로 표 구조를 출력하세요."
    ),
    "formula": "이 수식을 인식하세요.",
    "chart": "이 차트/그래프를 분석하고 데이터를 추출하세요.",
    "seal": "이 인장/직인의 텍스트를 인식하세요.",
    "header": "이 헤더 영역의 텍스트를 인식하세요.",
    "footer": "이 푸터 영역의 텍스트를 인식하세요.",
    "figure": "이 이미지/도표의 내용을 설명하세요.",
}


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
        extraction_instruction = EXTRACTION_INSTRUCTIONS.get(
            form_type,
            EXTRACTION_INSTRUCTIONS["unknown"],
        )
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
                    instruction=REGION_INSTRUCTIONS["table"],
                    region=region,
                    use_guided_json=False,
                    use_logprobs=True,
                ))
            elif region.region_type == RegionType.SEAL:
                tasks.append(InstructionTask(
                    task_type="region_ocr",
                    instruction=REGION_INSTRUCTIONS["seal"],
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
        return REGION_INSTRUCTIONS.get(
            region_type.value,
            REGION_INSTRUCTIONS["text"],
        )

    def get_task_prompt(self, region_type: RegionType) -> str:
        """영역 유형에 대응하는 task prompt prefix 반환.

        TASK_PROMPTS 매핑 사용 (types.py에 정의).
        """
        return TASK_PROMPTS.get(region_type.value, "OCR:")
