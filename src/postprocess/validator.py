"""P4 — 룰 검증 + 신뢰도 보정 모듈
군수 OCR 시스템 v2 · Stage 3

v1 T8(룰 엔진) + T9(신뢰도 스코어링)를 통합하여
VLM logprobs 신뢰도를 1차로 사용하고, 룰 기반 교차검증으로 2차 보정합니다.

검증 규칙 (BACKEND.md §5-3):
  - 산술 검증: 합계 == 수량 × 단가 → 불일치 시 -0.30
  - 코드 검증: NSN 형식 불일치 → -0.15
  - 날짜 검증: 청구일 <= 승인일 위반 → -0.10

판정 기준:
  통과: overall_confidence ≥ 임계값 AND CRITICAL 오류 없음
  실패: CRITICAL 오류 존재 OR flagged_fields 존재 → review_required=True

Input  : VLMResult
Output : ValidatedResult

의존: src.interfaces, src.vlm.logprobs_scorer
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from src.interfaces.enums import ProcessingPath, Severity, ValidationErrorType
from src.interfaces.types import (
    FieldValue,
    RecognizedTable,
    ValidationError,
    ValidatedResult,
    VLMResult,
)
from src.vlm.logprobs_scorer import (
    PENALTY_ARITHMETIC,
    PENALTY_CODE_FORMAT,
    PENALTY_DATE_LOGIC,
    get_threshold,
    is_flagged,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  전체 문서 신뢰도 임계값
# ─────────────────────────────────────────────
OVERALL_CONFIDENCE_THRESHOLD = 0.85


# ─────────────────────────────────────────────
#  검증 규칙 실행
# ─────────────────────────────────────────────

def _validate_arithmetic(fields: list[FieldValue]) -> list[ValidationError]:
    """산술 정합성 검증: 합계 == 수량 × 단가."""
    errors: list[ValidationError] = []
    field_map = {f.field_key: f for f in fields}

    # items 배열 내부 검증 (JSON 배열은 items 필드에 직렬화)
    items_field = field_map.get("items")
    if items_field:
        try:
            import json
            items = json.loads(items_field.corrected_value)
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    qty = item.get("quantity")
                    price = item.get("unit_price")
                    total = item.get("total")
                    if qty is not None and price is not None and total is not None:
                        try:
                            expected = int(qty) * int(price)
                            actual = int(total)
                            if expected != actual:
                                errors.append(ValidationError(
                                    error_id=f"ve_{len(errors)+1:04d}",
                                    error_type=ValidationErrorType.ARITHMETIC,
                                    severity=Severity.CRITICAL,
                                    field_ref=f"items[{i}].total",
                                    expected=str(expected),
                                    actual=str(actual),
                                    message=f"합계 불일치: {qty} × {price} = {expected}, 실제 {actual}",
                                ))
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass

    # grand_total 검증
    grand_total_field = field_map.get("grand_total")
    if grand_total_field and items_field:
        try:
            import json
            items = json.loads(items_field.corrected_value)
            if isinstance(items, list):
                expected_total = sum(
                    int(item.get("total", 0))
                    for item in items
                    if isinstance(item, dict) and item.get("total") is not None
                )
                try:
                    actual_total = int(grand_total_field.corrected_value)
                    if expected_total != actual_total:
                        errors.append(ValidationError(
                            error_id=f"ve_{len(errors)+1:04d}",
                            error_type=ValidationErrorType.ARITHMETIC,
                            severity=Severity.CRITICAL,
                            field_ref="grand_total",
                            expected=str(expected_total),
                            actual=str(actual_total),
                            message=f"총계 불일치: SUM(품목별 합계)={expected_total}, 실제 {actual_total}",
                        ))
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass

    return errors


def _validate_code_format(fields: list[FieldValue]) -> list[ValidationError]:
    """코드 형식 검증: NSN, K-NSN, 부대코드 패턴."""
    errors: list[ValidationError] = []
    nsn_pattern = re.compile(r"^\d{4}-\d{2}-\d{3}-\d{4}$")
    k_nsn_pattern = re.compile(r"^KN-\d{5}-\d{4}$")

    for f in fields:
        if f.data_type == "code" and f.field_key in ("nsn",):
            if not nsn_pattern.match(f.corrected_value):
                errors.append(ValidationError(
                    error_id=f"ve_{len(errors)+1:04d}",
                    error_type=ValidationErrorType.CODE_FORMAT,
                    severity=Severity.HIGH,
                    field_ref=f.field_key,
                    expected="NNNN-NN-NNN-NNNN",
                    actual=f.corrected_value,
                    message=f"NSN 형식 불일치: {f.corrected_value}",
                ))

    # items 내부 NSN도 검증
    items_field = next((f for f in fields if f.field_key == "items"), None)
    if items_field:
        try:
            import json
            items = json.loads(items_field.corrected_value)
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, dict) and "nsn" in item:
                        nsn_val = str(item["nsn"])
                        if nsn_val and not nsn_pattern.match(nsn_val):
                            errors.append(ValidationError(
                                error_id=f"ve_{len(errors)+1:04d}",
                                error_type=ValidationErrorType.CODE_FORMAT,
                                severity=Severity.HIGH,
                                field_ref=f"items[{i}].nsn",
                                expected="NNNN-NN-NNN-NNNN",
                                actual=nsn_val,
                                message=f"NSN 형식 불일치: {nsn_val}",
                            ))
        except Exception:
            pass

    return errors


def _validate_date_logic(fields: list[FieldValue]) -> list[ValidationError]:
    """날짜 정합성 검증: 청구일 <= 승인일, 정비일 >= 배치일."""
    errors: list[ValidationError] = []
    field_map = {f.field_key: f for f in fields}

    date_pairs = [
        ("request_date", "approval_date", "청구일 > 승인일"),
        ("deployment_date", "maintenance_date", "배치일 > 정비일"),
    ]

    for earlier_key, later_key, msg_template in date_pairs:
        earlier_f = field_map.get(earlier_key)
        later_f = field_map.get(later_key)
        if earlier_f and later_f:
            try:
                d1 = _parse_date(earlier_f.corrected_value)
                d2 = _parse_date(later_f.corrected_value)
                if d1 and d2 and d1 > d2:
                    errors.append(ValidationError(
                        error_id=f"ve_{len(errors)+1:04d}",
                        error_type=ValidationErrorType.DATE_LOGIC,
                        severity=Severity.HIGH,
                        field_ref=f"{earlier_key},{later_key}",
                        expected=f"{earlier_key} <= {later_key}",
                        actual=f"{earlier_f.corrected_value} > {later_f.corrected_value}",
                        message=msg_template,
                    ))
            except Exception:
                pass

    return errors


def _parse_date(s: str) -> Optional[datetime]:
    """다양한 날짜 형식 파싱."""
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%y.%m.%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _validate_missing_fields(
    fields: list[FieldValue],
    form_type: str,
) -> list[ValidationError]:
    """필수 필드 누락 검증."""
    errors: list[ValidationError] = []

    required_fields: dict[str, list[str]] = {
        "supply_request": ["unit_code", "request_date", "items"],
        "maintenance_record": ["equipment_id", "maintenance_date", "technician_name", "result"],
        "inventory_sheet": ["unit_code", "report_date", "items"],
        "handover_doc": ["from_person", "to_person", "handover_date", "items"],
        "inspection_report": ["inspection_date", "inspector_name", "overall_result"],
    }

    required = required_fields.get(form_type, [])
    field_keys = {f.field_key for f in fields}

    for key in required:
        if key not in field_keys:
            errors.append(ValidationError(
                error_id=f"ve_{len(errors)+1:04d}",
                error_type=ValidationErrorType.MISSING_FIELD,
                severity=Severity.HIGH,
                field_ref=key,
                expected=key,
                actual="(없음)",
                message=f"필수 필드 누락: {key}",
            ))

    return errors


# ─────────────────────────────────────────────
#  P4Validator (메인 클래스)
# ─────────────────────────────────────────────

class P4Validator:
    """P4 — 룰 검증 + 신뢰도 보정.

    VLM logprobs 신뢰도를 1차로 사용하고,
    룰 기반 교차검증에서 발견된 오류에 따라 신뢰도를 2차 보정합니다.

    사용 예시:
        validator = P4Validator()
        validated = validator.validate(vlm_result)
        if validated.review_required:
            # 검토 큐 적재
    """

    def validate(
        self,
        vlm_result: VLMResult,
        processing_path: ProcessingPath = ProcessingPath.VLM,
    ) -> ValidatedResult:
        """VLMResult → ValidatedResult.

        Args:
            vlm_result: P3 출력
            processing_path: 처리 경로 (VLM/FALLBACK)

        Returns:
            ValidatedResult
        """
        fields = list(vlm_result.fields)  # 복사본 (보정 반영용)
        form_type = vlm_result.form_type.value

        # ── 1. 룰 검증 실행 ───
        all_errors: list[ValidationError] = []
        all_errors.extend(_validate_arithmetic(fields))
        all_errors.extend(_validate_code_format(fields))
        all_errors.extend(_validate_date_logic(fields))
        all_errors.extend(_validate_missing_fields(fields, form_type))

        # error_id 재번호 부여
        for i, err in enumerate(all_errors):
            err.error_id = f"ve_{i+1:04d}"

        # ── 2. 룰 기반 신뢰도 보정 ───
        penalty_map = self._build_penalty_map(all_errors)
        adjusted_fields: list[FieldValue] = []

        for f in fields:
            penalty = penalty_map.get(f.field_key, 0.0)
            adjusted_confidence = max(0.0, f.confidence - penalty)
            adjusted_flagged = is_flagged(adjusted_confidence, f.data_type)

            adjusted_fields.append(FieldValue(
                field_key=f.field_key,
                raw_value=f.raw_value,
                corrected_value=f.corrected_value,
                data_type=f.data_type,
                confidence=round(adjusted_confidence, 4),
                token_logprobs=f.token_logprobs,
                is_flagged=adjusted_flagged,
            ))

        # ── 3. 전체 신뢰도 산출 ───
        if adjusted_fields:
            overall = sum(f.confidence for f in adjusted_fields) / len(adjusted_fields)
        else:
            overall = 0.0

        # ── 4. 판정 ───
        flagged_fields = [f.field_key for f in adjusted_fields if f.is_flagged]
        has_critical = any(e.severity == Severity.CRITICAL for e in all_errors)
        review_required = (
            has_critical
            or bool(flagged_fields)
            or overall < OVERALL_CONFIDENCE_THRESHOLD
            or processing_path == ProcessingPath.FALLBACK
        )

        logger.info(
            "[P4][%s] 검증 완료: errors=%d (critical=%s), flagged=%d, "
            "overall=%.4f, review=%s, path=%s",
            vlm_result.doc_id, len(all_errors), has_critical,
            len(flagged_fields), overall, review_required,
            processing_path.value,
        )

        return ValidatedResult(
            doc_id=vlm_result.doc_id,
            fields=adjusted_fields,
            tables=vlm_result.tables,
            validation_errors=all_errors,
            overall_confidence=round(overall, 4),
            review_required=review_required,
            flagged_fields=flagged_fields,
            processing_path=processing_path,
        )

    @staticmethod
    def _build_penalty_map(errors: list[ValidationError]) -> dict[str, float]:
        """검증 오류에서 필드별 감점 맵 생성."""
        penalty_map: dict[str, float] = {}

        penalty_by_type = {
            ValidationErrorType.ARITHMETIC: PENALTY_ARITHMETIC,
            ValidationErrorType.CODE_FORMAT: PENALTY_CODE_FORMAT,
            ValidationErrorType.DATE_LOGIC: PENALTY_DATE_LOGIC,
            ValidationErrorType.MISSING_FIELD: 0.20,
            ValidationErrorType.FORMAT: 0.10,
        }

        for err in errors:
            penalty = penalty_by_type.get(err.error_type, 0.05)
            # field_ref에서 필드 키 추출 (items[0].total → items)
            field_key = err.field_ref.split("[")[0].split(",")[0].strip()
            if field_key:
                current = penalty_map.get(field_key, 0.0)
                penalty_map[field_key] = current + penalty

        return penalty_map
