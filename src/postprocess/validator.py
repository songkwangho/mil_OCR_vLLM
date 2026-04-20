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

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

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
    PENALTY_MISSING_FIELD,
    PENALTY_FORMAT,
    get_threshold,
    is_flagged,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  전체 문서 신뢰도 임계값
# ─────────────────────────────────────────────
OVERALL_CONFIDENCE_THRESHOLD = 0.85

# Fallback 경로 신뢰도 보정 상수
# Fallback lacks logprobs, so token_logprobs=[] and confidence comes from
# rec_score * 0.7. We apply stricter thresholds to prevent over-trust.
FALLBACK_CONFIDENCE_PENALTY = 0.10
FALLBACK_MAX_OVERALL_CONFIDENCE = 0.70


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


def _validate_equipment_checklist(data_or_raw) -> list[ValidationError]:
    """CHK-001~004 — 전비품 확인서 점검표 전용 룰.

    Accept either an assembled dict (preferred) or a raw JSON string (fallback).
    """
    errors: list[ValidationError] = []
    if data_or_raw is None:
        return errors
    if isinstance(data_or_raw, dict):
        data = data_or_raw
    else:
        if not data_or_raw:
            return errors
        try:
            data = json.loads(data_or_raw)
        except Exception:
            return errors
        if not isinstance(data, dict):
            return errors

    items = data.get("checklist_items")

    # CHK-001: checklist_items 배열 길이 = 6
    if not isinstance(items, list) or len(items) != 6:
        actual_len = len(items) if isinstance(items, list) else 0
        errors.append(ValidationError(
            error_id="chk_001",
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=Severity.HIGH,
            field_ref="checklist_items",
            expected="length=6",
            actual=f"length={actual_len}",
            message="CHK-001: checklist_items 배열 길이가 6이 아님",
        ))
    else:
        # CHK-002: item_number 1~6 순차
        numbers = [it.get("item_number") if isinstance(it, dict) else None for it in items]
        if numbers != [1, 2, 3, 4, 5, 6]:
            errors.append(ValidationError(
                error_id="chk_002",
                error_type=ValidationErrorType.FORMAT,
                severity=Severity.HIGH,
                field_ref="checklist_items[*].item_number",
                expected="[1,2,3,4,5,6]",
                actual=str(numbers),
                message="CHK-002: item_number가 1~6 순서대로 존재하지 않음",
            ))

        # CHK-003: result ∈ {O, X, ?}
        invalid_results = [
            (i, it.get("result") if isinstance(it, dict) else None)
            for i, it in enumerate(items, 1)
            if not (isinstance(it, dict) and it.get("result") in ("O", "X", "?"))
        ]
        for item_idx, bad in invalid_results:
            errors.append(ValidationError(
                error_id=f"chk_003_{item_idx}",
                error_type=ValidationErrorType.FORMAT,
                severity=Severity.HIGH,
                field_ref=f"checklist_items[{item_idx-1}].result",
                expected="O|X|?",
                actual=str(bad),
                message=f"CHK-003: 항목 {item_idx} 결과가 O/X/? 중 하나가 아님",
            ))

    # CHK-004: writer.name 비어있지 않음
    writer = data.get("writer")
    writer_name = writer.get("name") if isinstance(writer, dict) else None
    if not (isinstance(writer_name, str) and writer_name.strip()):
        errors.append(ValidationError(
            error_id="chk_004",
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=Severity.MEDIUM,
            field_ref="writer.name",
            expected="non-empty string",
            actual=str(writer_name),
            message="CHK-004: writer.name이 비어있음",
        ))

    return errors


def _validate_bid_application(data_or_raw) -> list[ValidationError]:
    """BID-001~004 — 입찰참가신청서 (별지 제13호 서식) 전용 룰.

    Accept either an assembled dict (preferred) or a raw JSON string (fallback).

    BID-001: applicant.business_reg_number 형식 NNN-NN-NNNNN (신뢰도 -0.15)
    BID-002: applicant.corporate_reg_number 형식 NNNNN-NNNNNNN (신뢰도 -0.15, 있을 때만)
    BID-003: submission.submission_date 비어있지 않음 (severity=MEDIUM)
    BID-004: submission.submitter_name 비어있지 않음 (severity=MEDIUM)
    """
    errors: list[ValidationError] = []
    if data_or_raw is None:
        return errors
    if isinstance(data_or_raw, dict):
        data = data_or_raw
    else:
        if not data_or_raw:
            return errors
        try:
            data = json.loads(data_or_raw)
        except Exception:
            return errors
        if not isinstance(data, dict):
            return errors

    applicant = data.get("applicant") or {}
    submission = data.get("submission") or {}
    if not isinstance(applicant, dict):
        applicant = {}
    if not isinstance(submission, dict):
        submission = {}

    # BID-001: 사업자등록번호 형식 NNN-NN-NNNNN
    biz_reg_pattern = re.compile(r"^\d{3}-\d{2}-\d{5}$")
    biz_reg = applicant.get("business_reg_number")
    if isinstance(biz_reg, str) and biz_reg.strip():
        if not biz_reg_pattern.match(biz_reg.strip()):
            errors.append(ValidationError(
                error_id="bid_001",
                error_type=ValidationErrorType.CODE_FORMAT,
                severity=Severity.HIGH,
                field_ref="applicant.business_reg_number",
                expected="NNN-NN-NNNNN",
                actual=str(biz_reg),
                message=f"BID-001: 사업자등록번호 형식 불일치: {biz_reg}",
            ))

    # BID-002: 법인등록번호 형식 NNNNN-NNNNNNN (있을 때만)
    corp_reg_pattern = re.compile(r"^\d{5}-\d{7}$")
    corp_reg = applicant.get("corporate_reg_number")
    if isinstance(corp_reg, str) and corp_reg.strip():
        if not corp_reg_pattern.match(corp_reg.strip()):
            errors.append(ValidationError(
                error_id="bid_002",
                error_type=ValidationErrorType.CODE_FORMAT,
                severity=Severity.HIGH,
                field_ref="applicant.corporate_reg_number",
                expected="NNNNN-NNNNNNN",
                actual=str(corp_reg),
                message=f"BID-002: 법인등록번호 형식 불일치: {corp_reg}",
            ))

    # BID-003: submission_date 존재
    sub_date = submission.get("submission_date")
    if not (isinstance(sub_date, str) and sub_date.strip()):
        errors.append(ValidationError(
            error_id="bid_003",
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=Severity.MEDIUM,
            field_ref="submission.submission_date",
            expected="non-empty string",
            actual=str(sub_date),
            message="BID-003: submission.submission_date가 비어있음",
        ))

    # BID-004: submitter_name 비어있지 않음
    sub_name = submission.get("submitter_name")
    if not (isinstance(sub_name, str) and sub_name.strip()):
        errors.append(ValidationError(
            error_id="bid_004",
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=Severity.MEDIUM,
            field_ref="submission.submitter_name",
            expected="non-empty string",
            actual=str(sub_name),
            message="BID-004: submission.submitter_name이 비어있음",
        ))

    return errors


def _normalize_rank_in_assembled(vlm_result: VLMResult) -> None:
    """assembled_json의 writer.rank 정규화 + fields[writer_block] blob 동기화.

    writer_block은 sub-schema 분해 경로에서 단일 FieldValue(JSON blob)로 저장된다.
    assembled_json만 수정하면 P5 output.json의 fields[]에는 원본이 남아 혼선을 주므로,
    fields blob도 정규화된 rank로 재직렬화한다.
    """
    assembled = getattr(vlm_result, "assembled_json", None)
    if not isinstance(assembled, dict):
        return
    writer = assembled.get("writer")
    if not isinstance(writer, dict):
        return
    raw_rank = writer.get("rank")
    if not isinstance(raw_rank, str) or not raw_rank.strip():
        return
    from src.postprocess.rank_normalizer import normalize_rank
    normalized, corrected, flagged = normalize_rank(raw_rank)
    if corrected:
        writer["rank"] = normalized
        logger.info("[P4] rank 정규화: '%s' → '%s'", raw_rank, normalized)
        # fields[writer_block] blob 동기화
        for f in vlm_result.fields:
            if f.field_key != "writer_block" or not isinstance(f.raw_value, str):
                continue
            try:
                blob = json.loads(f.raw_value)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(blob, dict) or blob.get("rank") != raw_rank:
                continue
            blob["rank"] = normalized
            new_blob = json.dumps(blob, ensure_ascii=False)
            f.raw_value = new_blob
            f.corrected_value = new_blob
    if flagged:
        vlm_result.warnings.append(f"[P4] rank '{raw_rank}' 정규화 실패 — 검토 필요")


# ─────────────────────────────────────────────
#  sub-field 신뢰도 트리 순회 (assembled_json 공통)
# ─────────────────────────────────────────────

# 필드 유형별 가중치 — 금액·코드가 문서 전체 신뢰도에 더 큰 영향.
FIELD_WEIGHTS: dict[str, float] = {
    "amount":      3.0,
    "code":        2.5,
    "date":        2.0,
    "quantity":    2.0,
    "text":        1.0,
    "handwritten": 1.0,
    "seal":        0.8,
    "signature":   0.5,
}

# 메타/분석 키 — 신뢰도 산출 대상에서 제외.
#  - analysis / overall_confidence / low_confidence_fields: VLM 자가 평가 메타
#  - result_confidence: checklist_items 자체 루프에서 별도 처리
#  - aggregator_blob: S7 디버깅용 blob
#  - fixed_content: TemplateAugmentor가 주입한 인쇄 고정 텍스트 — VLM 추출이 아니므로
#                   신뢰도 산출 대상에서 제외 (판독 신뢰도 개념 무관)
_TRAVERSAL_SKIP_KEYS: frozenset[str] = frozenset({
    "analysis", "result_confidence", "aggregator_blob",
    "low_confidence_fields", "overall_confidence",
    "fixed_content",
})


def _traverse_assembled(
    node: Any,
    field_map: dict[str, "FieldValue"],
    result: list[tuple[str, str, float]],
    path: str = "",
) -> None:
    """assembled_json 트리 재귀 순회 — 리프 노드별 (path, type, confidence) 추출.

    - dict에 `result`/`result_confidence` 형제 키가 있으면 VLM 자가 평가값 사용
      (equipment_checklist.checklist_items[N] 전용 패턴).
    - 일반 리프는 path → field_key 매핑 후 field_map lookup. 없으면 폴백 0.5.
    - None/빈 문자열/bool 리프는 건너뜀 (누락 필드이므로 신뢰도 산출 무의미).
    """
    if isinstance(node, dict):
        # result_confidence 형제 키 우선 처리
        if "result_confidence" in node and "result" in node:
            try:
                conf = float(node.get("result_confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            result.append((path, "handwritten", max(0.0, min(1.0, conf))))
            return

        for key, value in node.items():
            if key in _TRAVERSAL_SKIP_KEYS:
                continue
            child_path = f"{path}.{key}" if path else key
            _traverse_assembled(value, field_map, result, path=child_path)

    elif isinstance(node, list):
        for i, item in enumerate(node):
            _traverse_assembled(item, field_map, result, path=f"{path}[{i}]")

    else:
        # 리프 노드
        if node is None or node == "" or isinstance(node, bool):
            return
        field_key = _path_to_field_key(path)
        fv = field_map.get(field_key) if field_key else None
        conf = fv.confidence if fv is not None else 0.5
        field_type = _infer_field_type(path)
        result.append((path, field_type, max(0.0, min(1.0, float(conf)))))


_CHECKLIST_ITEM_RE = re.compile(r"checklist_items\[(\d+)\]")


def _path_to_field_key(path: str) -> Optional[str]:
    """JSON path → TemplateAugmentor field_key 변환.

    military (equipment_checklist):
      "checklist_items[2]"  → "result_item_3"
      "writer.name"         → "writer_block"
      "writer.rank"         → "writer_block"
      "document_date"       → "document_date"

    other 경로 (official_document):
      field_key 체계가 없으므로 None 반환 → field_map miss → 폴백 0.5.
    """
    m = _CHECKLIST_ITEM_RE.match(path)
    if m:
        return f"result_item_{int(m.group(1)) + 1}"
    if path.startswith("writer"):
        return "writer_block"
    # 평탄 키는 path 그대로 — fields[] 에 동일 field_key 있을 가능성 대비
    if "." not in path and "[" not in path:
        return path
    return None


def _infer_field_type(path: str) -> str:
    """JSON path에서 필드 유형 추론 — FIELD_WEIGHTS 가중치용."""
    p = path.lower()
    if any(k in p for k in ("nsn", "k_nsn", "unit_code", "equipment_id")):
        return "code"
    if any(k in p for k in ("total", "price", "amount")):
        return "amount"
    if "date" in p:
        return "date"
    if any(k in p for k in ("quantity", "qty", "stock")):
        return "quantity"
    if "result" in p:
        return "handwritten"
    if "seal" in p or "stamp" in p:
        return "seal"
    if "signature" in p:
        return "signature"
    if "code" in p or "number" in p or "id" in p:
        return "code"
    return "text"


def _calc_overall_from_sub(
    sub_confidences: list[tuple[str, str, float]],
) -> float:
    """sub_confidences (path, type, conf) → 가중 평균 overall_confidence."""
    if not sub_confidences:
        return 0.0
    total_w = 0.0
    weighted = 0.0
    for _, ft, c in sub_confidences:
        w = FIELD_WEIGHTS.get(ft, 1.0)
        total_w += w
        weighted += w * c
    if total_w == 0.0:
        return 0.0
    return round(max(0.0, min(1.0, weighted / total_w)), 4)


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
        # equipment_checklist: _validate_equipment_checklist()가 CHK-001~004로 별도 검증.
        #   - CHK-001: checklist_items 길이 = 6
        #   - CHK-002: item_number 순차
        #   - CHK-003: result ∈ {O, X, ?}
        #   - CHK-004: writer.name 존재
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

        Note:
            form_type == OTHER 인 경우 군수 도메인 룰 검증(산술/코드/날짜/필수 필드)을
            모두 건너뜁니다. 신뢰도 산출만 수행하고, review_required는 False로 고정.
        """
        from src.interfaces.enums import FormType

        fields = list(vlm_result.fields)  # 복사본 (보정 반영용)
        form_type = vlm_result.form_type.value
        is_other = vlm_result.form_type == FormType.OTHER

        # ── 1. 룰 검증 실행 (other는 건너뜀) ───
        all_errors: list[ValidationError] = []
        if not is_other:
            all_errors.extend(_validate_arithmetic(fields))
            all_errors.extend(_validate_code_format(fields))
            all_errors.extend(_validate_date_logic(fields))
            all_errors.extend(_validate_missing_fields(fields, form_type))
            if form_type == "equipment_checklist":
                payload = vlm_result.assembled_json or vlm_result.raw_json
                all_errors.extend(_validate_equipment_checklist(payload))
                _normalize_rank_in_assembled(vlm_result)
            elif form_type == "bid_application":
                payload = vlm_result.assembled_json or vlm_result.raw_json
                all_errors.extend(_validate_bid_application(payload))

        # error_id 재번호 부여
        for i, err in enumerate(all_errors):
            err.error_id = f"ve_{i+1:04d}"

        # ── 2. 룰 기반 신뢰도 보정 ───
        penalty_map = self._build_penalty_map(all_errors)
        adjusted_fields: list[FieldValue] = []

        # Fallback lacks logprobs, so token_logprobs=[] and confidence comes
        # from rec_score * 0.7. We apply stricter thresholds to prevent
        # over-trust in fallback results.
        is_fallback = processing_path == ProcessingPath.FALLBACK

        for f in fields:
            penalty = penalty_map.get(f.field_key, 0.0)
            adjusted_confidence = max(0.0, f.confidence - penalty)

            if is_fallback:
                # Raise the flagging threshold by FALLBACK_CONFIDENCE_PENALTY
                # so that fallback fields are more strictly flagged
                threshold = get_threshold(f.data_type) + FALLBACK_CONFIDENCE_PENALTY
                adjusted_flagged = adjusted_confidence < threshold
            else:
                adjusted_flagged = is_flagged(adjusted_confidence, f.data_type)

            adjusted_fields.append(FieldValue(
                field_key=f.field_key,
                raw_value=f.raw_value,
                corrected_value=f.corrected_value,
                data_type=f.data_type,
                confidence=round(adjusted_confidence, 4),
                token_logprobs=f.token_logprobs,
                is_flagged=adjusted_flagged,
                region_id=f.region_id,
                was_retried=f.was_retried,
            ))

        # ── 3. 전체 신뢰도 산출 ───
        # assembled_json이 있는 서식(x-assembly-rules military / other S7 공통)은
        # 트리 순회 기반 sub-field 가중 평균을 사용. 없으면 기존 fields 평균.
        assembled = getattr(vlm_result, "assembled_json", None)
        sub_confidences: list[tuple[str, str, float]] = []
        if assembled and isinstance(assembled, dict):
            adjusted_field_map = {
                f.field_key: f for f in adjusted_fields if f.field_key
            }
            _traverse_assembled(
                node=assembled,
                field_map=adjusted_field_map,
                result=sub_confidences,
            )
            overall = _calc_overall_from_sub(sub_confidences)
        elif adjusted_fields:
            overall = sum(f.confidence for f in adjusted_fields) / len(adjusted_fields)
        else:
            overall = 0.0

        # Cap overall confidence for FALLBACK path (fallback discount is 0.7x)
        if is_fallback:
            overall = min(overall, FALLBACK_MAX_OVERALL_CONFIDENCE)

        # ── 4. 판정 ───
        flagged_fields = [f.field_key for f in adjusted_fields if f.is_flagged]
        has_critical = any(e.severity == Severity.CRITICAL for e in all_errors)

        if is_other:
            # other 문서는 군수 업무 대상이 아니므로 검토 큐 미적재
            review_required = False
        else:
            review_required = (
                has_critical
                or bool(flagged_fields)
                or overall < OVERALL_CONFIDENCE_THRESHOLD
                or processing_path == ProcessingPath.FALLBACK
            )

        logger.info(
            "[P4][%s] 검증 완료: errors=%d (critical=%s), flagged=%d, "
            "overall=%.4f (sub=%d), review=%s, path=%s",
            vlm_result.doc_id, len(all_errors), has_critical,
            len(flagged_fields), overall, len(sub_confidences),
            review_required, processing_path.value,
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
            assembled_json=getattr(vlm_result, "assembled_json", None),
            sub_confidences=sub_confidences,
        )

    @staticmethod
    def _build_penalty_map(errors: list[ValidationError]) -> dict[str, float]:
        """검증 오류에서 필드별 감점 맵 생성."""
        penalty_map: dict[str, float] = {}

        penalty_by_type = {
            ValidationErrorType.ARITHMETIC: PENALTY_ARITHMETIC,
            ValidationErrorType.CODE_FORMAT: PENALTY_CODE_FORMAT,
            ValidationErrorType.DATE_LOGIC: PENALTY_DATE_LOGIC,
            ValidationErrorType.MISSING_FIELD: PENALTY_MISSING_FIELD,
            ValidationErrorType.FORMAT: PENALTY_FORMAT,
        }

        for err in errors:
            penalty = penalty_by_type.get(err.error_type, 0.05)
            # field_ref에서 필드 키 추출 (items[0].total → items)
            field_key = err.field_ref.split("[")[0].split(",")[0].strip()
            if field_key:
                current = penalty_map.get(field_key, 0.0)
                penalty_map[field_key] = current + penalty

        return penalty_map
