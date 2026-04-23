"""Layer 1 — 공통 필드 패턴 검증 + 자동 교정 엔진.

configs/validation_rules/common.yaml을 로드하여 data_type 및 field_key 기반으로
모든 form_type의 FieldValue를 검증·정규화한다. form_type 무관.

반환:
  - 교정된 FieldValue 리스트 (corrected_value가 정규화되었을 수 있음)
  - ValidationError 리스트 (패턴 불일치 + 정규화 실패)

설계 원칙:
  - form_type 의존 없음
  - 규칙은 전부 YAML에서 로드, Python 코드에 도메인 상수 없음
  - field.confidence는 여기서 조정하지 않음 (P4Validator의 penalty_map이 일괄 처리)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from src.interfaces.enums import Severity, ValidationErrorType
from src.interfaces.types import FieldValue, ValidationError

logger = logging.getLogger(__name__)


# 프로젝트 루트 기준 경로 해결 (configs/는 workspace 루트에 있음)
def _default_rules_path() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent.parent, Path.cwd()):
        p = candidate / "configs" / "validation_rules" / "common.yaml"
        if p.exists():
            return p
    # 마지막 폴백 — 존재하지 않아도 Path 반환 (로드 시 빈 규칙)
    return here.parent.parent.parent / "configs" / "validation_rules" / "common.yaml"


# data_type → ValidationErrorType 분류 (Layer 1 패턴 실패 시 fallback)
_DATA_TYPE_ERROR_MAP: dict[str, ValidationErrorType] = {
    "date": ValidationErrorType.DATE_LOGIC,
    "amount": ValidationErrorType.ARITHMETIC,
    "phone": ValidationErrorType.FORMAT,
}

_FIELD_KEY_ERROR_TYPE = ValidationErrorType.CODE_FORMAT


class FieldPatternValidator:
    """Layer 1 엔진 — common.yaml 기반 공통 패턴 검증."""

    def __init__(self, rules_path: Optional[Path] = None) -> None:
        self._rules_path = rules_path or _default_rules_path()
        self._rules = self._load_rules(self._rules_path)
        logger.info(
            "[FieldPatternValidator] 규칙 로드: type=%d, field_key=%d (%s)",
            len(self._rules.get("field_type_patterns", [])),
            len(self._rules.get("field_key_patterns", [])),
            self._rules_path,
        )

    @staticmethod
    def _load_rules(path: Path) -> dict:
        if not path.exists():
            logger.warning("[FieldPatternValidator] 규칙 파일 없음: %s", path)
            return {}
        try:
            import yaml  # type: ignore
        except ImportError:
            logger.warning("[FieldPatternValidator] PyYAML 미설치 — 규칙 비활성")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("[FieldPatternValidator] %s 로드 실패: %s", path, e)
            return {}

    # ── 외부 API ─────────────────────────────────────

    def validate_and_correct(
        self,
        fields: list[FieldValue],
    ) -> tuple[list[FieldValue], list[ValidationError]]:
        """fields[]를 검증하고 정규화 가능한 값은 자동 교정.

        Returns:
            (corrected_fields, errors)
                corrected_fields: 입력 순서 유지 — 정규화된 FieldValue 대체.
                errors: 패턴 불일치 + 정규화 실패 ValidationError.
        """
        errors: list[ValidationError] = []
        corrected: list[FieldValue] = []
        for fv in fields:
            new_fv, field_errors = self._check_field(fv)
            corrected.append(new_fv)
            errors.extend(field_errors)
        return corrected, errors

    # ── 내부 ─────────────────────────────────────────

    def _check_field(
        self, fv: FieldValue
    ) -> tuple[FieldValue, list[ValidationError]]:
        errors: list[ValidationError] = []
        value = fv.corrected_value

        # 빈 값은 Layer 2에서 NOT_EMPTY 규칙이 처리. 여기선 건너뜀.
        if value is None or value in ("", "None", "{}", "[]"):
            return fv, errors

        new_value = value
        data_type = fv.data_type or "text"

        # 1) data_type 기반 검증 + 정규화
        for rule in self._rules.get("field_type_patterns", []) or []:
            if rule.get("data_type") != data_type:
                continue
            pattern = rule.get("pattern")
            if not pattern or re.match(pattern, str(new_value)):
                continue
            normalized = self._try_normalize(str(new_value), rule.get("normalize"))
            if normalized is not None and re.match(pattern, normalized):
                new_value = normalized
                continue
            errors.append(ValidationError(
                error_id=f"pat_{fv.field_key}_{data_type}",
                error_type=_DATA_TYPE_ERROR_MAP.get(data_type, ValidationErrorType.FORMAT),
                severity=Severity(rule.get("severity", "medium")),
                field_ref=fv.field_key or "",
                expected=str(pattern),
                actual=str(new_value),
                message=f"{fv.field_key}: {data_type} 형식 불일치",
            ))

        # 2) field_key 기반 코드 형식 검증
        for rule in self._rules.get("field_key_patterns", []) or []:
            if rule.get("field_key") != fv.field_key:
                continue
            pattern = rule.get("pattern")
            if not pattern or re.match(pattern, str(new_value)):
                continue
            errors.append(ValidationError(
                error_id=f"pat_{fv.field_key}",
                error_type=_FIELD_KEY_ERROR_TYPE,
                severity=Severity(rule.get("severity", "medium")),
                field_ref=fv.field_key or "",
                expected=str(pattern),
                actual=str(new_value),
                message=f"{fv.field_key}: 코드 형식 불일치",
            ))

        if new_value == value:
            return fv, errors

        # 정규화 반영된 복사본 반환
        new_fv = FieldValue(
            field_key=fv.field_key,
            raw_value=fv.raw_value,
            corrected_value=new_value,
            data_type=fv.data_type,
            confidence=fv.confidence,
            token_logprobs=fv.token_logprobs,
            is_flagged=fv.is_flagged,
            region_id=fv.region_id,
            was_retried=fv.was_retried,
        )
        return new_fv, errors

    @staticmethod
    def _try_normalize(value: str, normalize_rule: Optional[dict]) -> Optional[str]:
        """정규화 규칙 적용. 실패 시 None."""
        if not normalize_rule:
            return None
        try:
            strip_chars = normalize_rule.get("strip_chars")
            if strip_chars is not None:
                return value.replace(strip_chars, "")

            regex = normalize_rule.get("regex")
            template = normalize_rule.get("template")
            if regex and template:
                m = re.search(regex, value)
                if not m:
                    return None
                groups = m.groups()
                out = template
                for i, g in enumerate(groups, 1):
                    # {n:02d} 우선 매치
                    placeholder_pad = f"{{{i}:02d}}"
                    if placeholder_pad in out:
                        try:
                            out = out.replace(placeholder_pad, f"{int(g):02d}")
                        except (TypeError, ValueError):
                            out = out.replace(placeholder_pad, str(g))
                    out = out.replace(f"{{{i}}}", str(g))
                return out
        except Exception as e:
            logger.debug("[FieldPatternValidator] normalize 실패: %s", e)
        return None
