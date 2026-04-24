"""Layer 2 — form_type별 교차 검증 엔진.

configs/validation_rules/{form_type}.yaml을 로드하여 assembled_json에
교차 검증 규칙을 적용한다. Python 코드에 form_type별 하드코딩 없음.

지원 규칙 타입:
  REGEX_MATCH      field ↔ pattern
  NOT_EMPTY        field 비어있지 않음
  DATE_ORDER       fields 목록이 오름차순 날짜
  ARITHMETIC       result_field = operands ⊕ (+ / ×)
  NAME_MATCH       두 이름 필드 중 하나가 다른 하나를 포함
  ADDR_CONTAINS    address에 business_location 포함
  ARRAY_LENGTH     field 배열 길이 == expected_length
  ARRAY_SEQUENCE   배열 원소 key 값이 expected_sequence와 일치
  ARRAY_ENUM       배열 원소 key 값이 allowed_values에 포함

각 규칙은 `rule_id`, `error_type`, `severity`, `message`를 가져
ValidationError로 그대로 매핑된다.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.interfaces.enums import FormType, Severity, ValidationErrorType
from src.interfaces.types import ValidationError

logger = logging.getLogger(__name__)


# ARITHMETIC 허용 오차 — 수량×단가 같은 정수 연산 결과에서 부동소수/반올림
# 노이즈를 흡수하기 위한 절대 허용 오차(원 단위). yaml 규칙에 `tolerance`가
# 명시되면 그 값을 우선 사용한다.
DEFAULT_ARITHMETIC_TOLERANCE: float = 1.0


def _default_rules_dir() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent.parent, Path.cwd()):
        p = candidate / "configs" / "validation_rules"
        if p.exists():
            return p
    return here.parent.parent.parent / "configs" / "validation_rules"


# ─────────────────────────────────────────────
#  날짜 파서 (한국어 + ISO)
# ─────────────────────────────────────────────

_KR_DATE_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")


def _parse_date(value: Any) -> Optional[datetime]:
    """한국어(YYYY년 M월 D일) + ISO 형식 파싱."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = _KR_DATE_RE.search(s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%y.%m.%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────
#  CrossFieldValidator
# ─────────────────────────────────────────────

class CrossFieldValidator:
    """서식별 교차 검증 엔진."""

    def __init__(self, rules_dir: Optional[Path] = None) -> None:
        self._rules_dir = rules_dir or _default_rules_dir()
        self._cache: dict[str, list[dict]] = {}
        logger.info(
            "[CrossFieldValidator] 규칙 디렉터리: %s (exists=%s)",
            self._rules_dir, self._rules_dir.exists(),
        )

    # ── 외부 API ─────────────────────────────────

    def validate(
        self,
        assembled_json: dict,
        form_type: FormType,
    ) -> list[ValidationError]:
        """assembled_json에 form_type별 규칙 적용."""
        rules = self._load_rules(form_type)
        if not rules or not isinstance(assembled_json, dict):
            return []
        errors: list[ValidationError] = []
        for rule in rules:
            err = self._apply_rule(rule, assembled_json)
            if err is not None:
                errors.append(err)
        return errors

    # ── 규칙 로드 ────────────────────────────────

    def _load_rules(self, form_type: FormType) -> list[dict]:
        key = form_type.value
        if key in self._cache:
            return self._cache[key]
        path = self._rules_dir / f"{key}.yaml"
        if not path.exists():
            self._cache[key] = []
            return []
        try:
            import yaml  # type: ignore
        except ImportError:
            logger.warning("[CrossFieldValidator] PyYAML 미설치 — 규칙 비활성")
            self._cache[key] = []
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("[CrossFieldValidator] %s 로드 실패: %s", path, e)
            self._cache[key] = []
            return []
        rules = list(data.get("rules") or [])
        self._cache[key] = rules
        logger.info(
            "[CrossFieldValidator] %s: %d 규칙 로드 (%s)", key, len(rules), path,
        )
        return rules

    # ── 디스패치 ─────────────────────────────────

    _CHECKERS: dict[str, str] = {
        "REGEX_MATCH":    "_check_regex_match",
        "NOT_EMPTY":      "_check_not_empty",
        "DATE_ORDER":     "_check_date_order",
        "ARITHMETIC":     "_check_arithmetic",
        "NAME_MATCH":     "_check_name_match",
        "ADDR_CONTAINS":  "_check_addr_contains",
        "ARRAY_LENGTH":   "_check_array_length",
        "ARRAY_SEQUENCE": "_check_array_sequence",
        "ARRAY_ENUM":     "_check_array_enum",
    }

    def _apply_rule(self, rule: dict, data: dict) -> Optional[ValidationError]:
        rule_type = rule.get("type")
        rule_id = rule.get("rule_id", rule_type or "?")
        checker_name = self._CHECKERS.get(rule_type or "")
        if not checker_name:
            logger.debug("[CrossFieldValidator] 알 수 없는 rule type: %s", rule_type)
            return None
        try:
            passed = getattr(self, checker_name)(rule, data)
        except Exception as e:
            logger.debug("[CrossFieldValidator] %s 예외: %s", rule_id, e)
            return None
        if passed:
            return None

        # error_type 기본값: 타입별 매핑
        et_raw = rule.get("error_type") or self._default_error_type(rule_type)
        try:
            error_type = ValidationErrorType(et_raw)
        except ValueError:
            error_type = ValidationErrorType.FORMAT

        return ValidationError(
            error_id=rule_id,
            error_type=error_type,
            severity=Severity(rule.get("severity", "medium")),
            field_ref=self._field_ref(rule),
            expected=str(rule.get("pattern") or rule.get("expected_sequence") or ""),
            actual="",
            message=rule.get("message", f"{rule_id} 검증 실패"),
        )

    @staticmethod
    def _default_error_type(rule_type: Optional[str]) -> str:
        return {
            "REGEX_MATCH":    "code_format",
            "NOT_EMPTY":      "missing_field",
            "DATE_ORDER":     "date_logic",
            "ARITHMETIC":     "arithmetic",
            "NAME_MATCH":     "format",
            "ADDR_CONTAINS":  "format",
            "ARRAY_LENGTH":   "missing_field",
            "ARRAY_SEQUENCE": "format",
            "ARRAY_ENUM":     "format",
        }.get(rule_type or "", "format")

    @staticmethod
    def _field_ref(rule: dict) -> str:
        if "field" in rule:
            return str(rule["field"])
        if "fields" in rule:
            return ",".join(str(f) for f in rule["fields"])
        if "field_value" in rule and "field_contained" in rule:
            return f"{rule['field_value']},{rule['field_contained']}"
        if "result_field" in rule:
            return str(rule["result_field"])
        return ""

    # ── 값 조회 (점 표기 경로) ───────────────────

    @staticmethod
    def _get_value(data: Any, path: str) -> Any:
        if not isinstance(path, str):
            return None
        current: Any = data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if current is None:
                return None
        return current

    # ── 규칙 구현 ────────────────────────────────

    def _check_regex_match(self, rule: dict, data: dict) -> bool:
        value = self._get_value(data, rule.get("field", ""))
        if value is None or str(value).strip() == "":
            # 빈 값은 NOT_EMPTY로 별도 처리 — 여기선 통과
            return True
        return bool(re.match(rule["pattern"], str(value)))

    def _check_not_empty(self, rule: dict, data: dict) -> bool:
        value = self._get_value(data, rule.get("field", ""))
        return value is not None and str(value).strip() != ""

    def _check_date_order(self, rule: dict, data: dict) -> bool:
        paths = rule.get("fields") or []
        dates: list[datetime] = []
        for p in paths:
            d = _parse_date(self._get_value(data, p))
            if d is not None:
                dates.append(d)
        if len(dates) < 2:
            return True
        return all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def _check_arithmetic(self, rule: dict, data: dict) -> bool:
        try:
            def _num(v: Any) -> Optional[float]:
                if v is None:
                    return None
                return float(str(v).replace(",", "").strip())
            result = _num(self._get_value(data, rule.get("result_field", "")))
            operands = [_num(self._get_value(data, f)) for f in rule.get("operands") or []]
            if result is None or any(o is None for o in operands):
                return True
            op = rule.get("operator")
            if op == "multiply":
                expected = operands[0]
                for o in operands[1:]:
                    expected *= o
            elif op == "add":
                expected = sum(operands)  # type: ignore[arg-type]
            else:
                return True
            try:
                tolerance = float(rule.get("tolerance", DEFAULT_ARITHMETIC_TOLERANCE))
            except (TypeError, ValueError):
                tolerance = DEFAULT_ARITHMETIC_TOLERANCE
            return abs(result - expected) < tolerance
        except Exception:
            return True

    def _check_name_match(self, rule: dict, data: dict) -> bool:
        values: list[str] = []
        for f in rule.get("fields") or []:
            v = self._get_value(data, f)
            if v is None:
                return True
            s = str(v).replace(" ", "").strip()
            if not s:
                return True
            values.append(s)
        if len(values) < 2:
            return True
        return values[0] in values[1] or values[1] in values[0]

    def _check_addr_contains(self, rule: dict, data: dict) -> bool:
        addr = str(self._get_value(data, rule.get("field_value", "")) or "").replace(" ", "")
        loc = str(self._get_value(data, rule.get("field_contained", "")) or "").replace(" ", "")
        if not addr or not loc:
            return True
        return loc in addr

    def _check_array_length(self, rule: dict, data: dict) -> bool:
        arr = self._get_value(data, rule.get("field", ""))
        if not isinstance(arr, list):
            return False
        return len(arr) == int(rule.get("expected_length", 0))

    def _check_array_sequence(self, rule: dict, data: dict) -> bool:
        arr = self._get_value(data, rule.get("field", ""))
        if not isinstance(arr, list):
            return False
        key = rule.get("key", "")
        actual = [item.get(key) for item in arr if isinstance(item, dict)]
        return actual == list(rule.get("expected_sequence", []))

    def _check_array_enum(self, rule: dict, data: dict) -> bool:
        arr = self._get_value(data, rule.get("field", ""))
        if not isinstance(arr, list):
            return False
        key = rule.get("key", "")
        allowed = set(rule.get("allowed_values", []) or [])
        return all(
            (not isinstance(item, dict)) or item.get(key) in allowed
            for item in arr
        )
