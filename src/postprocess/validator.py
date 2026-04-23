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

from src.interfaces.enums import FormType, ProcessingPath, Severity, ValidationErrorType
from src.interfaces.types import (
    FieldValue,
    RecognizedTable,
    ValidationError,
    ValidatedResult,
    VLMResult,
)
from src.postprocess.cross_field_validator import CrossFieldValidator
from src.postprocess.domain_dict_corrector import DomainDictCorrector
from src.postprocess.field_pattern_validator import FieldPatternValidator
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
#  검증 규칙 실행 — 하위호환 shim
#
#  Layer 1/2 엔진(FieldPatternValidator, CrossFieldValidator)이 실제 검증을 수행한다.
#  과거 호출자(단위 테스트, 외부 스크립트 등)가 직접 참조해 온 `_validate_*` 함수들은
#  엔진 API로 라우팅하는 thin wrapper로만 남긴다.
# ─────────────────────────────────────────────

# 공용 엔진 인스턴스 (프로세스 수명 동안 재사용)
_FIELD_PATTERN_VALIDATOR = FieldPatternValidator()
_CROSS_FIELD_VALIDATOR = CrossFieldValidator()
_DOMAIN_DICT_CORRECTOR = DomainDictCorrector()


def _validate_arithmetic(fields: list[FieldValue]) -> list[ValidationError]:
    """(deprecated) Layer 1 + Layer 2 엔진이 산술 검증을 담당.

    과거 호출자 호환용 shim — 엔진이 assembled_json 기반으로 처리하므로
    fields[] 전용 호출에선 아무것도 반환하지 않는다.
    """
    return []


def _validate_code_format(fields: list[FieldValue]) -> list[ValidationError]:
    """(deprecated) FieldPatternValidator(common.yaml)가 코드 형식을 검증."""
    return []


def _validate_date_logic(fields: list[FieldValue]) -> list[ValidationError]:
    """(deprecated) CrossFieldValidator DATE_ORDER 규칙이 날짜 순서를 검증."""
    return []


def _parse_date(s: str) -> Optional[datetime]:
    """다양한 날짜 형식 파싱."""
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%y.%m.%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _coerce_assembled(data_or_raw) -> Optional[dict]:
    """dict 또는 JSON 문자열 → dict. 실패 시 None."""
    if data_or_raw is None:
        return None
    if isinstance(data_or_raw, dict):
        return data_or_raw
    if not data_or_raw:
        return None
    try:
        data = json.loads(data_or_raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _validate_equipment_checklist(data_or_raw) -> list[ValidationError]:
    """하위호환 shim → CrossFieldValidator(FormType.EQUIPMENT_CHECKLIST).

    과거에는 Python 하드코딩이었으나 configs/validation_rules/equipment_checklist.yaml
    으로 이관. 이 함수는 외부 호출자(단위 테스트 등)를 위해 유지.
    """
    data = _coerce_assembled(data_or_raw)
    if data is None:
        return []
    return _CROSS_FIELD_VALIDATOR.validate(data, FormType.EQUIPMENT_CHECKLIST)


def _validate_bid_application(data_or_raw) -> list[ValidationError]:
    """하위호환 shim → CrossFieldValidator(FormType.BID_APPLICATION).

    configs/validation_rules/bid_application.yaml에서 규칙을 로드.
    """
    data = _coerce_assembled(data_or_raw)
    if data is None:
        return []
    return _CROSS_FIELD_VALIDATOR.validate(data, FormType.BID_APPLICATION)


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
    """(deprecated) CrossFieldValidator NOT_EMPTY 규칙이 필수 필드 검증 담당.

    과거 호출자 호환용 shim — 엔진이 assembled_json 기반으로 처리.
    """
    return []


# ─────────────────────────────────────────────
#  P4Validator (메인 클래스)
# ─────────────────────────────────────────────

class P4Validator:
    """P4 — 룰 검증 + 신뢰도 보정.

    VLM logprobs 신뢰도를 1차로 사용하고,
    룰 기반 교차검증에서 발견된 오류에 따라 신뢰도를 2차 보정합니다.

    ConsistencyReasoningLoop(일관성/컨텍스트 주입 재추론)은 생성 시 vlm_client가
    주입된 경우에만 활성화되며, Layer 2 실패 또는 저신뢰 필드 발생 시 1회만
    실행됩니다. region_map은 validate() 호출 시 전달받습니다.
    """

    def __init__(
        self,
        vlm_client: Optional["VLMClient"] = None,  # type: ignore[name-defined]
    ) -> None:
        self._reasoning_loop = None
        if vlm_client is not None:
            try:
                from src.postprocess.consistency_reasoning_loop import (
                    ConsistencyReasoningLoop,
                )
                self._reasoning_loop = ConsistencyReasoningLoop(vlm_client)
                logger.info("[P4Validator] ConsistencyReasoningLoop 활성화")
            except Exception as e:
                logger.warning(
                    "[P4Validator] ConsistencyReasoningLoop 초기화 실패: %s", e,
                )

    def validate(
        self,
        vlm_result: VLMResult,
        processing_path: ProcessingPath = ProcessingPath.VLM,
        region_map: Optional[dict] = None,
        schema: Optional[dict] = None,
        fixed_values: Optional[dict] = None,
    ) -> ValidatedResult:
        """VLMResult → ValidatedResult.

        Args:
            vlm_result: P3 출력
            processing_path: 처리 경로 (VLM/FALLBACK)
            region_map: {region_id → CroppedRegion} — 재추론에 필요.
            schema: 재추론 후 assembled_json 재조립에 사용 (x-assembly-rules 스키마).
            fixed_values: TemplateAugmentor가 넘긴 fixed_text 값 — 재조립 시 유지.

        Returns:
            ValidatedResult

        Note:
            form_type == OTHER 인 경우 군수 도메인 룰 검증(산술/코드/날짜/필수 필드)을
            모두 건너뜁니다. 신뢰도 산출만 수행하고, review_required는 False로 고정.
        """
        fields = list(vlm_result.fields)  # 복사본 (보정 반영용)
        form_type = vlm_result.form_type.value
        is_other = vlm_result.form_type == FormType.OTHER

        # ── 1. 룰 검증 실행 (other는 건너뜀) ───
        all_errors: list[ValidationError] = []
        if not is_other:
            # Layer 1 — common.yaml 기반 공통 패턴 검증 + 자동 정규화
            fields, pattern_errors = _FIELD_PATTERN_VALIDATOR.validate_and_correct(fields)
            all_errors.extend(pattern_errors)

            # Layer 2 — form_type별 교차 검증 (assembled_json 기반)
            assembled_for_rules = (
                vlm_result.assembled_json
                if isinstance(vlm_result.assembled_json, dict)
                else _coerce_assembled(vlm_result.raw_json)
            )
            cross_errors: list[ValidationError] = []
            if assembled_for_rules is not None:
                cross_errors = _CROSS_FIELD_VALIDATOR.validate(
                    assembled_for_rules, vlm_result.form_type,
                )

            # ── Layer 3: 도메인 사전 교정 (assembled_json이 있는 서식만) ───
            # 별칭 정규화 + 폐쇄집합 최근접 매칭. 교정이 발생하면 cross_errors를
            # 재계산해 재추론 트리거 판정을 정확히 한다.
            if (
                isinstance(assembled_for_rules, dict)
                and isinstance(vlm_result.assembled_json, dict)
            ):
                corrected_json, dict_logs = _DOMAIN_DICT_CORRECTOR.correct(
                    vlm_result.assembled_json,
                )
                if dict_logs:
                    vlm_result.warnings.extend(dict_logs)
                if corrected_json is not vlm_result.assembled_json:
                    import dataclasses as _dc
                    vlm_result = _dc.replace(
                        vlm_result, assembled_json=corrected_json,
                    )
                    assembled_for_rules = corrected_json
                    # Layer 2 재실행 — 교정된 값 기준으로 cross_errors 재산출
                    cross_errors = _CROSS_FIELD_VALIDATOR.validate(
                        assembled_for_rules, vlm_result.form_type,
                    )

            all_errors.extend(cross_errors)

            # ── 1.5 일관성/컨텍스트 재추론 (1회) ───
            if self._reasoning_loop is not None and region_map:
                has_low_conf = any(f.confidence < 0.60 for f in fields)
                if cross_errors or has_low_conf:
                    logger.info(
                        "[P4][%s] 재추론 루프 진입 (cross_errors=%d, low_conf=%s)",
                        vlm_result.doc_id, len(cross_errors), has_low_conf,
                    )
                    # reasoning_loop는 VLMResult를 새로 반환. fields/assembled_json
                    # 가 바뀔 수 있음 → Layer 1/2 재적용.
                    import dataclasses as _dc
                    staged = _dc.replace(vlm_result, fields=fields)
                    updated = self._reasoning_loop.run(
                        vlm_result=staged,
                        validation_errors=cross_errors,
                        region_map=region_map,
                        assembled_json=staged.assembled_json,
                        warnings=vlm_result.warnings,
                    )
                    if updated is not staged:
                        fields = list(updated.fields)
                        # assembled_json 재조립 (x-assembly-rules 있는 서식만)
                        if isinstance(schema, dict) and schema.get("x-assembly-rules"):
                            fields, assembled_for_rules = self._reassemble(
                                fields, schema, region_map, fixed_values or {},
                                warnings=vlm_result.warnings,
                            )
                            # 재조립 직후 Layer 3 재적용 — 재조립이 Layer 3 이전 필드
                            # 값으로 assembled_json을 복원하므로, 한 번 더 정규화해 최종
                            # 결과가 도메인 사전을 반영하도록 보장.
                            if isinstance(assembled_for_rules, dict):
                                post_json, post_logs = _DOMAIN_DICT_CORRECTOR.correct(
                                    assembled_for_rules,
                                )
                                if post_logs:
                                    vlm_result.warnings.extend(post_logs)
                                assembled_for_rules = post_json
                            updated = _dc.replace(
                                updated,
                                fields=fields,
                                assembled_json=assembled_for_rules,
                            )
                            # 재조립된 assembled_json을 vlm_result에도 반영
                            vlm_result = updated
                        # 재검증
                        fields, pattern_errors = _FIELD_PATTERN_VALIDATOR.validate_and_correct(fields)
                        cross_errors = (
                            _CROSS_FIELD_VALIDATOR.validate(
                                assembled_for_rules, vlm_result.form_type,
                            )
                            if isinstance(assembled_for_rules, dict) else []
                        )
                        all_errors = list(pattern_errors) + list(cross_errors)

            # 특수 후처리 — assembled_json rank 정규화는 Layer 2 바깥
            if form_type == "equipment_checklist":
                _normalize_rank_in_assembled(vlm_result)

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
    def _reassemble(
        fields: list[FieldValue],
        schema: dict,
        region_map: dict,
        fixed_values: dict,
        warnings: list[str],
    ) -> tuple[list[FieldValue], Optional[dict]]:
        """재추론 후 Assembler를 다시 돌려 assembled_json을 갱신."""
        try:
            from src.vlm.assembler import Assembler
            region_field_key_map: dict[str, str] = {}
            for region in region_map.values():
                fk = getattr(region.instruction_spec, "field_key", None)
                if fk:
                    region_field_key_map[region.region_id] = fk
            assembled = Assembler().assemble(
                fields=fields,
                schema=schema,
                region_field_key_map=region_field_key_map,
                fixed_values=fixed_values,
                warnings=warnings,
            )
            return fields, assembled
        except Exception as e:
            logger.warning("[P4] 재조립 실패: %s", e)
            return fields, None

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
