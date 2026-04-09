"""VLM Instruction 라우터 — YAML 설정 기반 instruction 매핑

P2 레이아웃 분석과 VLM instruction 로직을 분리하기 위해
모든 매핑(task_prompts, region_instructions, extraction_instructions)을
외부 YAML 설정 파일에서 로드합니다.

사용 예시:
    router = InstructionRouter()  # configs/instruction_mappings.yaml 기본 경로
    prompt = router.get_task_prompt("table")       # "Table Recognition:"
    instr = router.get_region_instruction("seal")  # "이 인장/직인의 텍스트를 인식하세요."
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

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


class InstructionRouter:
    """YAML 설정 기반 VLM instruction 라우터.

    Args:
        config_path: instruction_mappings.yaml 경로.
            상대 경로인 경우 프로젝트 루트 기준으로 해석합니다.
    """

    def __init__(self, config_path: str = "configs/instruction_mappings.yaml") -> None:
        self._task_prompts: dict[str, str] = {}
        self._region_instructions: dict[str, str] = {}
        self._extraction_instructions: dict[str, str] = {}

        resolved = self._resolve_path(config_path)
        self._load(resolved)

    # ── public API ──────────────────────────

    def get_task_prompt(self, region_type: str) -> str:
        """영역 유형에 대응하는 task prompt prefix 반환."""
        return self._task_prompts.get(region_type, _DEFAULT_TASK_PROMPT)

    def get_region_instruction(self, region_type: str) -> str:
        """영역 유형에 대응하는 region instruction 반환."""
        return self._region_instructions.get(region_type, _DEFAULT_REGION_INSTRUCTION)

    def get_extraction_instruction(self, form_type: str) -> str:
        """서식 유형에 대응하는 extraction instruction 반환."""
        return self._extraction_instructions.get(form_type, _DEFAULT_EXTRACTION_INSTRUCTION)

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
