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
from src.vlm.budget_config import PIXEL_BUDGETS as _DEFAULT_PIXEL_BUDGETS, FALLBACK_PIXEL_BUDGET as _FALLBACK_PIXEL_BUDGET

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
    "unknown": "군수 서식 OCR 시스템입니다.",
    "other": "문서 OCR 시스템입니다.",
}

_DEFAULT_SYSTEM_PROMPT = "군수 서식 OCR 시스템입니다."

# other 경로 전용 instruction — 군수 도메인 맥락 없음
_OTHER_REGION_INSTRUCTION = "이 영역의 텍스트를 인식하세요."

# CoT(analysis) 안내 — 군수 경로에만 추가
_COT_INSTRUCTION = (
    "\n\n[사고 과정]\n"
    "답변 JSON 최상단의 analysis 필드에 이미지 품질, 레이아웃, 모호한 문자를 "
    "30~50 토큰으로 간략히 기술한 뒤 나머지 필드를 추출하세요."
)

# 군수 도메인 패턴 힌트 — 영어 룰 + 한국어 필드명 하이브리드
# (KLOCR 논문, 다국어 프롬프팅 서베이 2025.05: VLM은 영어 지시 따르기 + 한국어 출력)
_DOMAIN_PATTERN_HINTS = {
    "supply_request": (
        "\n\n[Format Rules]\n"
        "- NSN 코드 형식: NNNN-NN-NNN-NNNN (13자리, 예: 1005-01-432-1234)\n"
        "- K-NSN 형식: KN-NNNNN-NNNN\n"
        "- quantity, unit_price, total은 정수\n"
        "- date는 YYYY-MM-DD\n"
        "- 불확실한 글자는 [?]로 표시\n"
        "- 유사 문자 주의: ㄱ/ㅋ, ㄴ/ㄹ, 1/ㅣ, 0/O, 5/S, 6/G"
    ),
    "maintenance_record": (
        "\n\n[Format Rules]\n"
        "- equipment_id 형식: 알파벳+숫자 (예: K21-001)\n"
        "- maintenance_date는 YYYY-MM-DD\n"
        "- 불확실한 글자는 [?]로 표시\n"
        "- 유사 문자 주의: 1/ㅣ, 0/O, 5/S"
    ),
    "inventory_sheet": (
        "\n\n[Format Rules]\n"
        "- NSN/K-NSN 코드: NNNN-NN-NNN-NNNN 또는 KN-NNNNN-NNNN\n"
        "- stock_qty/disposal_qty는 정수 (음수 금지)\n"
        "- 불확실한 글자는 [?]로 표시"
    ),
    "handover_doc": (
        "\n\n[Format Rules]\n"
        "- handover_date는 YYYY-MM-DD\n"
        "- from_person/to_person은 한국어 성명\n"
        "- 불확실한 글자는 [?]로 표시\n"
        "- 유사 문자 주의: 1/ㅣ, 0/O"
    ),
    "inspection_report": (
        "\n\n[Format Rules]\n"
        "- inspection_date는 YYYY-MM-DD\n"
        "- overall_result는 적합/부적합/조건부 적합 중 하나\n"
        "- 불확실한 글자는 [?]로 표시"
    ),
}

# OCR 힌트 부착 템플릿 (저신뢰 영역 보강)
_OCR_HINT_TEMPLATE = (
    "\n\n[OCR 힌트]\n"
    "경량 OCR 인식 결과: {hint}\n"
    "위 내용을 참고하여 보다 정확하게 추출하세요. "
    "단, OCR 결과가 명백히 틀린 경우에는 이미지를 보고 직접 판단하세요."
)

# bid_application 서식 — 구체적 치환 사례를 박아 반-환각 유도.
# common_rules (YAML)과 중복되지 않고 상호 보완: YAML은 광범위한 규칙 선언,
# 아래 블록은 도로명·건명 치환·날짜 누락 등 실제 관찰된 오인식 유형 차단.
BID_APPLICATION_ANTI_HALLUCINATION = (
    "\n\n[Anti-Hallucination Warning]\n"
    "CRITICAL: Extract ONLY what is physically written in the image.\n"
    "- Do NOT substitute with similar-sounding place names "
    "(e.g., do not replace 테헤란로 with 태릉로, 판교로 with 반교로).\n"
    "- Do NOT infer company names, project names, or document titles from context.\n"
    "- Do NOT omit any part of dates — year/month/day must ALL be included "
    "(e.g., \"2023년 01월 10일\" not \"2023년 01월\").\n"
    "- For unclear characters: use [?] instead of guessing.\n"
    "- Read each character stroke independently."
)


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
        examples_dir: str = "configs/instruction_examples",
    ) -> None:
        self._task_prompts: dict[str, str] = {}
        self._region_instructions: dict[str, str] = {}
        self._extraction_instructions: dict[str, str] = {}

        resolved = self._resolve_path(config_path)
        self._load(resolved)

        self._schema_registry = schema_registry or SchemaRegistry()

        # 1-shot 예시 로드 (prefix caching 효율을 위해 정적으로 캐시)
        self._examples_dir = self._resolve_path(examples_dir)
        # form_type → {description, response} (legacy flat 1-shot)
        self._examples: dict[str, dict] = {}
        # form_type → {field_key → {description, response}} (신규 per-field_key)
        self._field_key_examples: dict[str, dict[str, dict]] = {}
        # form_type → common_rules 문자열 (신규 포맷 전용)
        self._common_rules: dict[str, str] = {}
        self._load_examples()

    # ── form_type 인식 라우팅 API ──────────

    def route(
        self,
        region: LayoutRegion,
        form_type: Optional[FormType] = None,
    ) -> InstructionSpec:
        """영역 + 서식 유형에 대응하는 VLM 호출 명세를 생성합니다.

        military 경로: 도메인 맥락 + 1-shot 예시 + CoT(analysis) 지시 + 군수 스키마.
        other 경로: 범용 OCR instruction + _general 스키마 (도메인/CoT/1-shot 없음).
        form_type=None: FormClassifier 미수행 시 최소 instruction.

        Returns:
            InstructionSpec — system_prompt, user_instruction,
            json_schema, pixel_budget 등 모든 필드 채움.
        """
        region_type_value = region.region_type.value
        form_type_value = form_type.value if form_type else None
        is_other = form_type == FormType.OTHER

        # ── other 경로: 범용 OCR (도메인 맥락 없음) ──
        if is_other:
            system_prompt = _SYSTEM_PROMPTS["other"]
            user_instruction = _OTHER_REGION_INSTRUCTION
            json_schema = self._schema_registry.load("_general")
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

        # ── military 경로: 도메인 맥락 + CoT + 1-shot 예시 ──

        # 1) region_instruction (YAML 기반)
        region_instruction = self._region_instructions.get(
            region_type_value, _DEFAULT_REGION_INSTRUCTION
        )

        # 2) form_type 기반 extraction context
        if form_type_value:
            extraction = self._extraction_instructions.get(
                form_type_value, _DEFAULT_EXTRACTION_INSTRUCTION
            )
            user_instruction = f"{region_instruction}\n\n[도메인 맥락]\n{extraction}"
        else:
            user_instruction = region_instruction

        # 3) CoT(analysis) 지시 추가 (군수 경로만)
        if form_type_value and form_type_value != "other":
            user_instruction += _COT_INSTRUCTION

        # 4) 도메인 패턴 힌트 (NSN/K-NSN/유사문자 등) — 군수 경로만
        if form_type_value:
            pattern_hint = _DOMAIN_PATTERN_HINTS.get(form_type_value)
            if pattern_hint:
                user_instruction += pattern_hint

        # 5) 1-shot 예시 부착 (form_type별 예시 존재 시)
        example = self._examples.get(form_type_value) if form_type_value else None
        if example:
            user_instruction += (
                "\n\n[예시]\n"
                f"{example.get('description', '')}\n"
                "예시 결과:\n"
                f"{example.get('response', '')}"
            )

        # 6) system_prompt
        if form_type_value:
            system_prompt = _SYSTEM_PROMPTS.get(
                form_type_value, _DEFAULT_SYSTEM_PROMPT
            )
        else:
            system_prompt = _DEFAULT_SYSTEM_PROMPT

        # 7) json_schema (SchemaRegistry)
        full_schema: Optional[dict] = None
        if form_type_value:
            # unknown → _fallback, 그 외 → form_type
            schema_id = "_fallback" if form_type_value == "unknown" else form_type_value
            full_schema = self._schema_registry.load(schema_id)

        # 7-1) field_key 기반 sub-schema 분해 (x-assembly-rules 있는 서식만)
        region_field_key = getattr(region, "field_key", None)
        if region_field_key and full_schema and full_schema.get("x-assembly-rules"):
            json_schema = self._extract_sub_schema(region_field_key, full_schema)
            # field_key 전용 짧은 instruction — full-shot 예시는 오히려 혼란을 줌
            user_instruction = self._build_field_key_instruction(
                region_field_key, json_schema, form_type=form_type,
            )
        else:
            json_schema = full_schema

        # 8) pixel_budget
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
            field_key=region_field_key,
        )

    # ─── sub-schema 분해 ──────────────────────────

    def _extract_sub_schema(self, field_key: str, full_schema: dict) -> dict:
        """field_key에 해당하는 sub-schema를 full_schema에서 추출.

        - result_item_N (1~6): x-checklist-item-schema (또는 checklist_items.items) +
          item_number const=N
        - writer_block: properties.writer
        - agent_seal: seal_present boolean 단일 필드만 요청 (전체 agent object 전달 금지)
        - 그 외: x-assembly-rules 경로의 top-level properties 또는 properties[field_key]

        path가 "parent.leaf" 형태로 단일 leaf를 가리키면 `parent` 전체가 아닌
        해당 leaf 필드 schema만 담은 sub-schema를 생성해 VLM에 과한 맥락을 주지 않는다.
        """
        import copy
        import re

        properties: dict = full_schema.get("properties", {}) or {}
        assembly_rules: dict = full_schema.get("x-assembly-rules", {}) or {}

        m = re.match(r"result_item_(\d+)$", field_key)
        if m:
            item_number = int(m.group(1))
            item_schema = full_schema.get("x-checklist-item-schema")
            if item_schema is None:
                checklist_prop = properties.get("checklist_items", {}) or {}
                item_schema = checklist_prop.get("items", {})
            if item_schema:
                sub = copy.deepcopy(item_schema)
                sub.setdefault("properties", {})["item_number"] = {
                    "type": "integer",
                    "const": item_number,
                }
                return sub

        path = assembly_rules.get(field_key)
        if path:
            parts = path.split(".")
            top_key = parts[0]
            top_schema = properties.get(top_key)
            # "parent.leaf" — leaf 필드만 가진 단일 속성 sub-schema 생성
            if (
                len(parts) >= 2
                and isinstance(top_schema, dict)
                and isinstance(top_schema.get("properties"), dict)
            ):
                leaf_key = parts[-1]
                leaf_schema = top_schema["properties"].get(leaf_key)
                if isinstance(leaf_schema, dict):
                    return {
                        "type": "object",
                        "properties": {leaf_key: copy.deepcopy(leaf_schema)},
                        "required": [leaf_key],
                    }
            if top_key in properties:
                return properties[top_key]

        if field_key in properties:
            return properties[field_key]

        return full_schema

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

    # ── OCR 힌트 부착 ──────────────────────

    @staticmethod
    def with_ocr_hint(spec: InstructionSpec, hint: str) -> InstructionSpec:
        """기존 InstructionSpec의 user_instruction 끝에 OCR 힌트 블록을 부착.

        새 InstructionSpec을 반환합니다 (원본 불변).
        StructuredExtractor의 재시도 경로에서 호출됩니다.
        """
        if not hint:
            return spec
        new_instruction = spec.user_instruction + _OCR_HINT_TEMPLATE.format(hint=hint)
        return InstructionSpec(
            region_id=spec.region_id,
            region_type=spec.region_type,
            form_type=spec.form_type,
            system_prompt=spec.system_prompt,
            user_instruction=new_instruction,
            json_schema=spec.json_schema,
            pixel_budget=spec.pixel_budget,
            ocr_hint=hint,
            is_retry=spec.is_retry,
        )

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

    def _load_examples(self) -> None:
        """configs/instruction_examples/*.yaml 1-shot 예시 로드.

        두 가지 YAML 포맷을 모두 지원:
          - Legacy flat: example_description + example_response (form_type 단일 예시)
          - 신규 per-field_key: common_rules + examples: list[{field_key, ...}]

        prefix caching 최적화를 위해 정적으로 캐시합니다. 서식별 예시가
        system_prompt 이후 user_instruction에 포함되면 동일 유형 요청에서
        KV 블록이 재사용되어 TTFT가 3~10배 단축됩니다.
        """
        if not os.path.isdir(self._examples_dir):
            logger.info(
                "InstructionRouter: 예시 디렉토리 없음 — %s (1-shot 비활성)",
                self._examples_dir,
            )
            return

        for filename in sorted(os.listdir(self._examples_dir)):
            if not filename.endswith((".yaml", ".yml")):
                continue
            form_type = filename.rsplit(".", 1)[0]
            try:
                with open(
                    os.path.join(self._examples_dir, filename),
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = yaml.safe_load(f) or {}

                # 신규 포맷 감지
                examples_list = data.get("examples")
                common_rules = data.get("common_rules")

                if isinstance(examples_list, list) and examples_list:
                    fk_map: dict[str, dict] = {}
                    for entry in examples_list:
                        if not isinstance(entry, dict):
                            continue
                        fk = entry.get("field_key")
                        if not fk:
                            continue
                        fk_map[fk] = {
                            "description": entry.get("example_description", ""),
                            "response": entry.get("example_response", ""),
                        }
                    if fk_map:
                        self._field_key_examples[form_type] = fk_map
                    if isinstance(common_rules, str) and common_rules.strip():
                        self._common_rules[form_type] = common_rules.strip()
                    # legacy lookup 대비: 첫 예시를 default로도 등록
                    first = next(iter(fk_map.values()), None)
                    if first:
                        self._examples[form_type] = first
                else:
                    # Legacy flat 포맷
                    self._examples[form_type] = {
                        "description": data.get("example_description", ""),
                        "response": data.get("example_response", ""),
                    }
            except Exception as e:
                logger.warning(
                    "InstructionRouter: 예시 로드 실패 (%s): %s", filename, e
                )

        logger.info(
            "InstructionRouter: %d개 1-shot 예시 로드 완료 — "
            "field_key별 예시 %d form_type, common_rules %d form_type (%s)",
            len(self._examples),
            len(self._field_key_examples),
            len(self._common_rules),
            self._examples_dir,
        )

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


    def _build_field_key_instruction(
        self,
        field_key: str,
        sub_schema: dict,
        form_type: Optional[FormType] = None,
    ) -> str:
        """field_key 전용 VLM 지시문.

        TemplateAugmentor가 부여한 field_key에 대해, full-schema 1-shot 없이
        sub-schema만을 따르는 간결한 지시문을 생성한다.

        form_type 기반 추가 주입:
          - common_rules (YAML): bid_application 등 신규 포맷 form에서 로드된 규칙
          - field_key별 1-shot 예시 (YAML): 동일 field_key 엔트리 존재 시 부착
          - form 특화 anti-hallucination 블록: 예) bid_application
        """
        import json as _json
        import re

        form_value = form_type.value if form_type else None

        def _attach_extras(base: str) -> str:
            extras = []
            if form_value:
                common = self._common_rules.get(form_value)
                if common:
                    extras.append(f"\n\n[Common Rules]\n{common}")
                example = (self._field_key_examples
                           .get(form_value, {})
                           .get(field_key))
                if example:
                    desc = (example.get("description") or "").strip()
                    resp = (example.get("response") or "").strip()
                    parts = ["\n\n[예시]"]
                    if desc:
                        parts.append(desc)
                    parts.append("예시 결과:")
                    parts.append(resp)
                    extras.append("\n".join(parts))
            if form_type == FormType.BID_APPLICATION:
                extras.append(BID_APPLICATION_ANTI_HALLUCINATION)
            return base + "".join(extras)

        m = re.match(r"result_item_(\d+)$", field_key)
        if m:
            n = int(m.group(1))
            return _attach_extras(
                f"이 크롭 이미지는 전비품 확인서 점검항목 {n}번의 점검결과 칸(O/X) 입니다.\n"
                "수기로 표시된 기호만 판단하여 다음 JSON Schema에 맞춰 출력하세요.\n"
                "result 값은 반드시 \"O\", \"X\", \"?\" 중 하나여야 합니다.\n"
                f"item_number는 반드시 {n} 이어야 합니다.\n\n"
                "[Schema]\n"
                f"{_json.dumps(sub_schema, ensure_ascii=False, indent=2)}\n\n"
                "추가 설명, 다른 필드, 코드 블록 표시 없이 JSON 객체 하나만 출력하세요."
            )
        if field_key == "writer_block":
            return _attach_extras(
                "이 크롭 이미지는 전비품 확인서 작성자 정보(팀명·직급·성명·서명)를 포함합니다.\n"
                "다음 JSON Schema에 맞춰 필드를 추출하세요. 서명이 존재하면 signature_present=true 입니다.\n\n"
                "[Schema]\n"
                f"{_json.dumps(sub_schema, ensure_ascii=False, indent=2)}\n\n"
                "다른 키를 추가하지 말고 schema 그대로 JSON만 출력하세요."
            )
        if field_key == "document_date":
            return _attach_extras(
                "이 크롭 이미지는 전비품 확인서 작성 일자(수기)입니다.\n"
                "날짜 문자열만 JSON 문자열 리터럴로 출력하세요. 예: \"2026년 4월 15일\"\n"
                "객체·배열·코드블록 없이 문자열 하나만 반환하세요."
            )
        if field_key == "form_identifier":
            return _attach_extras(
                "이 크롭 이미지는 서식 식별자 텍스트입니다. 예: \"별지 제3-2호 서식\"\n"
                "문자열 리터럴만 JSON 형식으로 출력하세요."
            )
        return _attach_extras(
            f"이 영역의 내용을 다음 JSON Schema에 맞춰 출력하세요.\n"
            f"field_key: {field_key}\n\n"
            "[Schema]\n"
            f"{_json.dumps(sub_schema, ensure_ascii=False, indent=2)}\n\n"
            "스키마 외의 키를 추가하지 마세요."
        )
