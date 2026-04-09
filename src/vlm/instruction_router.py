"""VLM Instruction 라우터 — form_type 인식 라우팅

P2.5-B 컴포넌트. region_type + form_type 조합으로
도메인 맥락 포함 instruction을 생성합니다.

YAML 설정 기반 매핑(task_prompts, region_instructions, extraction_instructions)을
외부 YAML 설정 파일에서 로드하며, form_type에 따라 도메인 맥락(system_prompt,
json_schema, pixel_budget)을 자동으로 결합합니다.

사용 예시 (기존 호환):
    router = InstructionRouter()
    prompt = router.get_task_prompt("table")
    instr  = router.get_region_instruction("seal")

사용 예시 (form_type 인식 라우팅):
    spec = router.route(region, form_type=FormType.SUPPLY_REQUEST)
    specs = router.route_all(layout_result, form_type=FormType.SUPPLY_REQUEST)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import yaml

from src.interfaces.enums import FormType, RegionType
from src.interfaces.types import InstructionSpec, LayoutRegion, LayoutResult
from src.domain.schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  기본값 (YAML 로드 실패 시 fallback)
# ─────────────────────────────────────────────

_DEFAULT_TASK_PROMPT = "OCR:"
_DEFAULT_REGION_INSTRUCTION = "이 영역의 한국어 텍스트를 인식하고 오타를 교정하세요."
_DEFAULT_EXTRACTION_INSTRUCTION = (
    "이 군수 서식 이미지에서 가능한 모든 필드를 추출하세요.\n"
    "문서 제목, 날짜, 작성자, 부대코드 및 모든 key-value 쌍과 "
    "표 데이터를 포함합니다.\n"
    "수기 기입 부분도 정확히 인식하세요.\n"
    "지정된 JSON 형식에 맞춰 응답하세요."
)

# ─────────────────────────────────────────────
#  form_type → 시스템 프롬프트 매핑
# ─────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[str, str] = {
    "supply_request": "군수 보급청구서 필드 추출 시스템입니다.",
    "maintenance_record": "군수 정비기록서 필드 추출 시스템입니다.",
    "inventory_sheet": "군수 물자현황표 필드 추출 시스템입니다.",
    "handover_doc": "군수 인수인계서 필드 추출 시스템입니다.",
    "inspection_report": "군수 검사보고서 필드 추출 시스템입니다.",
}

_DEFAULT_SYSTEM_PROMPT = "군수 서식 OCR 시스템입니다."

# ─────────────────────────────────────────────
#  region_type → 기본 pixel_budget 매핑
# ─────────────────────────────────────────────

_DEFAULT_PIXEL_BUDGETS: dict[str, int] = {
    "table": 1120,
    "seal": 560,
    "text": 280,
    "formula": 280,
    "chart": 280,
    "figure": 140,
    "header": 140,
    "footer": 140,
}

_FALLBACK_PIXEL_BUDGET = 280


class InstructionRouter:
    """form_type 인식 VLM instruction 라우터.

    region_type + form_type 조합으로 도메인 맥락 포함 InstructionSpec을
    생성합니다. 기존 flat lookup API(get_task_prompt, get_region_instruction,
    get_extraction_instruction)도 하위 호환을 위해 유지합니다.

    Args:
        config_path: instruction_mappings.yaml 경로.
            상대 경로인 경우 프로젝트 루트 기준으로 해석합니다.
        schema_registry: SchemaRegistry 인스턴스. None이면 내부 생성.
    """

    def __init__(
        self,
        config_path: str = "configs/instruction_mappings.yaml",
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> None:
        self._task_prompts: dict[str, str] = {}
        self._region_instructions: dict[str, str] = {}
        self._extraction_instructions: dict[str, str] = {}

        resolved = self._resolve_path(config_path)
        self._load(resolved)

        self._schema_registry = schema_registry or SchemaRegistry()

    # ── form_type 인식 라우팅 API ──────────

    def route(
        self,
        region: LayoutRegion,
        form_type: Optional[FormType] = None,
    ) -> InstructionSpec:
        """영역 + 서식 유형에 대응하는 VLM 호출 명세를 생성합니다.

        Args:
            region: 레이아웃 영역 (region_id, region_type 포함).
            form_type: 서식 유형. None이면 범용 instruction 생성.

        Returns:
            InstructionSpec with system_prompt, user_instruction,
            json_schema, pixel_budget 등 모든 필드 채워짐.
        """
        region_type_value = region.region_type.value
        form_type_value = form_type.value if form_type else None

        # 1) region_instruction (YAML 기반)
        region_instruction = self._region_instructions.get(
            region_type_value, _DEFAULT_REGION_INSTRUCTION
        )

        # 2) form_type 기반 extraction context 추가
        if form_type_value:
            extraction = self._extraction_instructions.get(
                form_type_value, _DEFAULT_EXTRACTION_INSTRUCTION
            )
            user_instruction = f"{region_instruction}\n\n[도메인 맥락]\n{extraction}"
        else:
            user_instruction = region_instruction

        # 3) system_prompt
        if form_type_value:
            system_prompt = _SYSTEM_PROMPTS.get(
                form_type_value, _DEFAULT_SYSTEM_PROMPT
            )
        else:
            system_prompt = _DEFAULT_SYSTEM_PROMPT

        # 4) json_schema (SchemaRegistry)
        json_schema: Optional[dict] = None
        if form_type_value:
            json_schema = self._schema_registry.load(form_type_value)

        # 5) pixel_budget
        pixel_budget = _DEFAULT_PIXEL_BUDGETS.get(
            region_type_value, _FALLBACK_PIXEL_BUDGET
        )

        return InstructionSpec(
            region_id=region.region_id,
            region_type=region.region_type,
            form_type=form_type,
            system_prompt=system_prompt,
            user_instruction=user_instruction,
            json_schema=json_schema,
            pixel_budget=pixel_budget,
        )

    def route_all(
        self,
        layout: LayoutResult,
        form_type: Optional[FormType] = None,
    ) -> dict[str, InstructionSpec]:
        """LayoutResult의 reading_order 순서대로 모든 영역에 대해 route()를 호출합니다.

        Args:
            layout: P2.5-A 출력 (후처리 완료 레이아웃 결과).
            form_type: 서식 유형. None이면 범용 instruction 생성.

        Returns:
            {region_id: InstructionSpec} 딕셔너리. reading_order 순서 유지.
        """
        specs: dict[str, InstructionSpec] = {}
        for idx in layout.reading_order:
            region = layout.regions[idx]
            specs[region.region_id] = self.route(region, form_type)
        return specs

    # ── 기존 호환 API (flat lookup) ────────

    def get_task_prompt(self, region_type: str) -> str:
        """영역 유형에 대응하는 task prompt prefix 반환."""
        return self._task_prompts.get(region_type, _DEFAULT_TASK_PROMPT)

    def get_region_instruction(self, region_type: str) -> str:
        """영역 유형에 대응하는 region instruction 반환."""
        return self._region_instructions.get(region_type, _DEFAULT_REGION_INSTRUCTION)

    def get_extraction_instruction(self, form_type: str) -> str:
        """서식 유형에 대응하는 extraction instruction 반환."""
        return self._extraction_instructions.get(
            form_type, _DEFAULT_EXTRACTION_INSTRUCTION
        )

    # ── internal ────────────────────────────

    @staticmethod
    def _resolve_path(config_path: str) -> str:
        """상대 경로를 프로젝트 루트 기준 절대 경로로 변환."""
        if os.path.isabs(config_path):
            return config_path
        # 프로젝트 루트: src/vlm/instruction_router.py → ../../
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        return os.path.join(project_root, config_path)

    def _load(self, path: str) -> None:
        """YAML 설정 파일 로드."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(
                "InstructionRouter: config not found at %s — using defaults", path
            )
            return
        except Exception:
            logger.exception(
                "InstructionRouter: failed to load %s — using defaults", path
            )
            return

        self._task_prompts = data.get("task_prompts", {})
        self._region_instructions = data.get("region_instructions", {})
        self._extraction_instructions = data.get("extraction_instructions", {})

        logger.info(
            "InstructionRouter: loaded %d task_prompts, %d region_instructions, "
            "%d extraction_instructions from %s",
            len(self._task_prompts),
            len(self._region_instructions),
            len(self._extraction_instructions),
            path,
        )
