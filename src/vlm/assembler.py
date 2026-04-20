"""Assembler — field_key 기반 VLM 결과를 schema 전체 구조로 조립.

IMPLEMENTATION_SPEC 변경 5 반영.

x-assembly-rules가 정의된 schema에 대해 region별 FieldValue를
하나의 구조화된 dict로 조립합니다.

equipment_checklist 예시:
  result_item_1 → checklist_items[0]
  result_item_2 → checklist_items[1]
  ...
  writer_block  → writer (object)
  document_date → document_date (string)

x-assembly-rules가 없는 스키마는 None 반환 → 기존 fields[] 기반 처리 유지.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.interfaces.types import FieldValue

logger = logging.getLogger(__name__)


class Assembler:
    """x-assembly-rules 기반 field_key → 전체 schema dict 조립."""

    def assemble(
        self,
        fields: list[FieldValue],
        schema: dict,
        region_field_key_map: Optional[dict[str, str]] = None,
        fixed_values: Optional[dict[str, Any]] = None,
        warnings: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """field_key별 FieldValue를 schema 전체 구조로 조립.

        Args:
            fields:               StructuredExtractor가 수집한 FieldValue 리스트
            schema:               x-assembly-rules가 포함된 full schema dict
            region_field_key_map: {region_id → field_key} 매핑
                                  None이면 FieldValue.field_key만 사용
            fixed_values:         TemplateAugmentor가 생성한 {field_key → 인쇄 고정값} dict.
                                  VLM 추출이 없거나 실패한 field_key에 한해 삽입 (VLM 결과 우선).
            warnings:             경고 수집 리스트

        Returns:
            조립된 dict, 또는 None (x-assembly-rules 없는 스키마 → 기존 처리 유지)
        """
        if warnings is None:
            warnings = []
        if fixed_values is None:
            fixed_values = {}

        assembly_rules: dict = schema.get("x-assembly-rules", {})
        if not assembly_rules:
            return None

        def _resolve_field_key(fv: FieldValue) -> Optional[str]:
            if region_field_key_map:
                fk = region_field_key_map.get(fv.region_id)
                if fk:
                    return fk
            return getattr(fv, "field_key", None)

        # field_key → FieldValue 매핑 (신뢰도 높은 것 우선 = 재시도 효과 반영)
        field_map: dict[str, FieldValue] = {}
        for fv in fields:
            fk = _resolve_field_key(fv)
            if not fk or fk not in assembly_rules:
                continue
            existing = field_map.get(fk)
            if existing is None or fv.confidence >= existing.confidence:
                field_map[fk] = fv

        result: dict[str, Any] = {}

        for field_key, path in assembly_rules.items():
            fv = field_map.get(field_key)
            value = self._extract_value(fv, field_key, warnings)
            self._set_path(result, path, value)

        # fixed_text 주입 — VLM 추출이 없었던 field_key에 한해 인쇄 고정값 삽입.
        # x-assembly-rules에 매핑된 field_key만 대상이며, VLM 결과가 이미 있으면 유지.
        fixed_injected = 0
        for field_key, value in fixed_values.items():
            if field_key not in assembly_rules:
                continue
            path = assembly_rules[field_key]
            if self._path_exists(result, path):
                continue
            self._set_path(result, path, value)
            fixed_injected += 1
        if fixed_injected:
            logger.info(
                "[Assembler] fixed_text 주입: %d개 (VLM 미추출 field_key에 한함)",
                fixed_injected,
            )

        # checklist_items 완전성 보장 — 누락 항목 기본값 채움, list로 변환
        if "checklist_items" in result and isinstance(result["checklist_items"], dict):
            items_dict: dict = result["checklist_items"]
            items_list = []
            for i in range(6):
                item = items_dict.get(str(i))
                # VLM이 [{...}] 로 감싼 경우 언래핑
                if isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict):
                    item = item[0]
                if item is None:
                    item = {
                        "item_number": i + 1,
                        "result": "?",
                        "result_confidence": 0.0,
                    }
                    warnings.append(
                        f"[Assembler] result_item_{i + 1} 추출 결과 없음 → 기본값 삽입"
                    )
                elif isinstance(item, dict):
                    # item_number 정규화: 누락/null/잘못된 값은 인덱스 기반으로 보정
                    if not isinstance(item.get("item_number"), int):
                        item["item_number"] = i + 1
                    # result 키 별칭 매핑 (item_result → result, handwritten_mark → result)
                    if "result" not in item:
                        for alias in ("item_result", "handwritten_mark"):
                            if alias in item and isinstance(item[alias], str):
                                item["result"] = item[alias]
                                break
                    item.setdefault("result", "?")
                    item.setdefault("result_confidence", 0.0)
                items_list.append(item)
            result["checklist_items"] = items_list

        logger.info(
            "[Assembler] 조립 완료: field_keys=%d matched=%d → top_keys=%s",
            len(assembly_rules), len(field_map), list(result.keys()),
        )
        return result

    def _extract_value(
        self,
        fv: Optional[FieldValue],
        field_key: str,
        warnings: list[str],
    ) -> Any:
        """FieldValue → Python 값.

        VLM이 sub-schema로 추출한 값은 raw_value에 JSON 문자열로 저장됨.
        - object/array: JSON 파싱
        - 숫자 문자열: int 변환
        - 그 외: 문자열 그대로
        None이면 None 반환.
        """
        if fv is None:
            warnings.append(f"[Assembler] {field_key} 추출 결과 없음")
            return None

        raw = (fv.corrected_value or fv.raw_value or "").strip()
        if not raw or raw.lower() == "none":
            return None

        if raw.startswith(("{", "[")):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

        if raw.lstrip("-").isdigit():
            try:
                return int(raw)
            except ValueError:
                pass

        return raw

    def _set_path(self, target: dict, path: str, value: Any) -> None:
        """'checklist_items.0' 형식의 경로로 nested dict에 값 삽입.

        배열 인덱스는 str 키로 보관하고, assemble() 마지막에 list로 변환.
        None은 삽입하지 않음 (기본값 처리는 assemble()에서).
        """
        if value is None:
            return
        parts = path.split(".")
        current = target
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _path_exists(self, target: dict, path: str) -> bool:
        """JSON path에 이미 non-None 값이 있는지 확인."""
        parts = path.split(".")
        current: Any = target
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    return False
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                except (TypeError, ValueError):
                    return False
                if idx < 0 or idx >= len(current):
                    return False
                current = current[idx]
            else:
                return False
        return current is not None
