"""일관성 기반 재추론 + 컨텍스트 주입 재추론 루프.

P4 검증 실패 또는 저신뢰 필드에 대해 VLM 재추론을 한 번 더 수행한다. 학습 없이
추론 단계에서 인식 품질을 올리는 경량 교정 단계.

두 가지 전략:
  방법 1 — 일관성 기반: Layer 2 CrossFieldValidator가 잡아낸 교차 검증 실패
           (NAME_MATCH, DATE_ORDER, ADDR_CONTAINS 등)에 대해 관련 필드를
           제약 조건과 함께 재전송.
  방법 2 — 컨텍스트 주입: confidence가 낮은 필드를, 같은 섹션의 고신뢰 필드
           값을 힌트로 주입해 재전송.

안전 장치:
  - 한 번에 최대 MAX_REREASON_FIELDS 개 필드 (기본 5)
  - 재추론 후 신뢰도가 개선된 경우에만 교체, 그렇지 않으면 원본 유지
  - was_retried 필드에 이미 True인 필드는 스킵
  - exception 발생 시 조용히 스킵 — 재추론은 best-effort
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Optional

from src.interfaces.enums import FormType
from src.interfaces.types import (
    CroppedRegion, FieldValue, ValidationError, VLMResult,
)
from src.vlm.logprobs_scorer import calc_field_confidence
from src.vlm.vlm_client import VLMClient, encode_image_base64

logger = logging.getLogger(__name__)


CONSISTENCY_CONFIDENCE_THRESHOLD: float = 0.60   # 이하면 재추론 후보
MAX_REREASON_FIELDS: int = 5                     # 한 문서당 최대 재추론 필드 수
HIGH_CONFIDENCE_THRESHOLD: float = 0.70          # 이상이면 컨텍스트 힌트 원본


# 필드 관련성 맵 — "이 필드를 재추론할 때 힌트로 쓸 만한 관련 필드들"
# bid_application / equipment_checklist 위주. 타 서식 추가 시 확장.
_RELEVANCE_MAP: dict[str, list[str]] = {
    # 신청인 섹션
    "address":              ["phone", "business_location", "business_reg_number"],
    "business_location":    ["address"],
    "phone":                ["address"],
    "company_name":         ["representative", "corporate_reg_number", "business_reg_number"],
    "representative":       ["company_name", "submitter_name"],
    "representative_dob":   ["representative"],
    "corporate_reg_number": ["company_name", "business_reg_number"],
    "business_reg_number":  ["company_name", "corporate_reg_number"],

    # 입찰 섹션
    "bid_name":          ["announcement_number", "bid_date"],
    "announcement_number": ["bid_date", "bid_name"],
    "bid_date":          ["announcement_number", "acquisition_date"],
    "acquisition_date":  ["bid_date", "submission_date"],

    # 대리인 섹션
    "agent_name": ["agent_dob"],
    "agent_dob":  ["agent_name"],

    # 제출 섹션
    "submission_date": ["bid_date", "seal_issue_date"],
    "submitter_name":  ["representative"],

    # 인감 섹션
    "seal_issuing_office": ["seal_issue_number", "seal_issue_date"],
    "seal_issue_number":   ["seal_issuing_office", "seal_issue_date"],
    "seal_issue_date":     ["seal_issuing_office", "seal_issue_number", "submission_date"],
}


@dataclass
class ReasoningTarget:
    """재추론 대상 단위."""
    field_key: str
    current_value: str
    current_confidence: float
    cropped_region: CroppedRegion
    reason: str                                       # "consistency" | "low_confidence"
    constraint: Optional[str] = None                  # 일관성 제약 문장
    context_hints: Optional[dict[str, str]] = None    # 힌트 {field_key: value}


class ConsistencyReasoningLoop:
    """Layer 2 실패/저신뢰 필드를 한 번 더 VLM에 던져 교정."""

    def __init__(self, vlm_client: VLMClient):
        self._vlm = vlm_client

    # ────────────────────────────────────────────────────────────
    #  외부 API
    # ────────────────────────────────────────────────────────────

    def run(
        self,
        vlm_result: VLMResult,
        validation_errors: list[ValidationError],
        region_map: dict[str, CroppedRegion],
        assembled_json: Optional[dict],
        warnings: Optional[list[str]] = None,
    ) -> VLMResult:
        """재추론 실행 → 필드가 갱신된 VLMResult 반환.

        신뢰도 개선 필드만 교체하고, 나머지는 원본 유지.
        """
        if not region_map or vlm_result.form_type == FormType.OTHER:
            return vlm_result

        targets = self._collect_targets(
            vlm_result, validation_errors, region_map, assembled_json,
        )
        if not targets:
            return vlm_result

        # 우선순위: consistency 먼저, 그 다음 confidence 낮은 순
        targets.sort(
            key=lambda t: (0 if t.reason == "consistency" else 1, t.current_confidence)
        )
        targets = targets[:MAX_REREASON_FIELDS]

        logger.info(
            "[ConsistencyLoop][%s] 재추론 대상 %d개",
            vlm_result.doc_id, len(targets),
        )

        field_map: dict[str, FieldValue] = {f.field_key: f for f in vlm_result.fields}
        replaced = 0
        for target in targets:
            updated = self._rereason(target)
            if updated is None:
                continue
            field_map[target.field_key] = updated
            replaced += 1
            if warnings is not None:
                warnings.append(
                    f"[ConsistencyLoop] {target.field_key} 재추론 "
                    f"({target.reason}): {target.current_value!r} "
                    f"→ {updated.corrected_value!r}, "
                    f"conf {target.current_confidence:.3f} → {updated.confidence:.3f}"
                )

        if replaced == 0:
            return vlm_result

        # 입력 fields 순서를 유지한 채 갱신본으로 교체
        new_fields = [field_map[f.field_key] for f in vlm_result.fields]

        import dataclasses as _dc
        return _dc.replace(
            vlm_result,
            fields=new_fields,
            retry_count=vlm_result.retry_count + replaced,
        )

    # ────────────────────────────────────────────────────────────
    #  대상 수집
    # ────────────────────────────────────────────────────────────

    def _collect_targets(
        self,
        vlm_result: VLMResult,
        validation_errors: list[ValidationError],
        region_map: dict[str, CroppedRegion],
        assembled_json: Optional[dict],
    ) -> list[ReasoningTarget]:
        field_map: dict[str, FieldValue] = {f.field_key: f for f in vlm_result.fields}
        high_conf = self._extract_high_confidence_values(vlm_result.fields)

        # 1) 일관성 실패 필드
        consistency_targets = self._collect_consistency_targets(
            validation_errors, field_map, region_map, high_conf,
        )

        # 2) 저신뢰 필드 (consistency 대상 제외)
        exclude = {t.field_key for t in consistency_targets}
        context_targets = self._collect_context_targets(
            field_map, region_map, high_conf, exclude_keys=exclude,
        )

        return consistency_targets + context_targets

    def _collect_consistency_targets(
        self,
        validation_errors: list[ValidationError],
        field_map: dict[str, FieldValue],
        region_map: dict[str, CroppedRegion],
        high_conf: dict[str, str],
    ) -> list[ReasoningTarget]:
        targets: list[ReasoningTarget] = []
        seen: set[str] = set()

        for error in validation_errors:
            constraint = self._build_constraint(error)
            if not constraint:
                continue
            for key in self._extract_related_field_keys(error):
                if key in seen:
                    continue
                fv = field_map.get(key)
                region = self._find_region_by_field_key(key, region_map)
                if fv is None or region is None:
                    continue
                if fv.was_retried:
                    continue
                seen.add(key)
                targets.append(ReasoningTarget(
                    field_key=key,
                    current_value=fv.corrected_value,
                    current_confidence=fv.confidence,
                    cropped_region=region,
                    reason="consistency",
                    constraint=constraint,
                    context_hints=self._select_relevant_context(key, high_conf),
                ))
        return targets

    def _collect_context_targets(
        self,
        field_map: dict[str, FieldValue],
        region_map: dict[str, CroppedRegion],
        high_conf: dict[str, str],
        exclude_keys: set[str],
    ) -> list[ReasoningTarget]:
        targets: list[ReasoningTarget] = []
        for key, fv in field_map.items():
            if key in exclude_keys:
                continue
            if fv.confidence >= CONSISTENCY_CONFIDENCE_THRESHOLD:
                continue
            if fv.was_retried:
                continue
            region = self._find_region_by_field_key(key, region_map)
            if region is None:
                continue
            relevant = self._select_relevant_context(key, high_conf)
            if not relevant:
                continue
            targets.append(ReasoningTarget(
                field_key=key,
                current_value=fv.corrected_value,
                current_confidence=fv.confidence,
                cropped_region=region,
                reason="low_confidence",
                context_hints=relevant,
            ))
        return targets

    # ────────────────────────────────────────────────────────────
    #  재추론 실행
    # ────────────────────────────────────────────────────────────

    def _rereason(self, target: ReasoningTarget) -> Optional[FieldValue]:
        try:
            spec = target.cropped_region.instruction_spec
            instruction = self._build_rereason_instruction(target)

            max_px = int((target.cropped_region.pixel_budget * 28 * 28) ** 0.5)
            image_b64 = encode_image_base64(
                target.cropped_region.cropped_image, max_size=max_px,
            )

            response = self._vlm.call(
                image_b64=image_b64,
                instruction=instruction,
                system_prompt=spec.system_prompt,
                guided_json=spec.json_schema,
                logprobs=True,
                pixel_budget=target.cropped_region.pixel_budget,
            )
        except Exception as e:
            logger.warning(
                "[ConsistencyLoop] %s 재추론 실패: %s", target.field_key, e,
            )
            return None

        text = (response.get("content") or "").strip()
        if not text:
            return None

        logprobs = response.get("logprobs") or []
        token_lps = [
            lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
            for lp in logprobs
        ]
        new_confidence = calc_field_confidence(token_lps, "text") if token_lps else 0.0

        if new_confidence <= target.current_confidence:
            # 개선 없으면 원본 유지
            return None

        new_value = self._parse_and_unwrap(text, target.field_key)

        return FieldValue(
            field_key=target.field_key,
            raw_value=text,
            corrected_value=new_value,
            data_type="text",
            confidence=round(new_confidence, 4),
            token_logprobs=token_lps,
            is_flagged=new_confidence < CONSISTENCY_CONFIDENCE_THRESHOLD,
            region_id=target.cropped_region.region_id,
            was_retried=True,
        )

    # ────────────────────────────────────────────────────────────
    #  프롬프트 생성
    # ────────────────────────────────────────────────────────────

    def _build_rereason_instruction(self, target: ReasoningTarget) -> str:
        base = target.cropped_region.instruction_spec.user_instruction or ""
        parts: list[str] = [base, ""]
        parts.append(
            f"[이전 인식 결과] '{target.current_value}' "
            f"(신뢰도 {target.current_confidence:.2f} — 낮음)"
        )
        if target.constraint:
            parts.append(f"[일관성 제약] {target.constraint}")
        if target.context_hints:
            hints = "\n".join(
                f"  - {k}: {v}" for k, v in target.context_hints.items()
            )
            parts.append(f"[관련 필드 참고]\n{hints}")
        parts.append(
            "위 정보를 참고하여 이미지에서 해당 필드를 정확히 다시 읽으세요. "
            "반드시 이미지에 보이는 글자만 추출하고 추론하지 마세요."
        )
        return "\n".join(parts)

    @staticmethod
    def _build_constraint(error: ValidationError) -> Optional[str]:
        et = error.error_type.value if hasattr(error.error_type, "value") else str(error.error_type)
        et_upper = et.upper()
        if et_upper in ("DATE_ORDER", "DATE_LOGIC"):
            return (
                "이 필드의 날짜는 다른 관련 날짜와 논리적 순서가 맞아야 합니다. "
                f"(오류: {error.message})"
            )
        if et_upper == "NAME_MATCH":
            return (
                "이 필드의 이름은 다른 필드의 이름과 동일인이어야 합니다. "
                f"(오류: {error.message})"
            )
        if et_upper in ("ADDR_CONTAINS", "FORMAT") and "주소" in error.message:
            return (
                "사업장소재지는 주소의 앞부분(시/구)과 일치해야 합니다. "
                f"(오류: {error.message})"
            )
        if et_upper == "ARITHMETIC":
            return (
                "이 필드 값은 다른 필드들의 산술 결과와 일치해야 합니다. "
                f"(오류: {error.message})"
            )
        if et_upper in ("REGEX_MATCH", "CODE_FORMAT"):
            return (
                "이 필드 값은 정해진 형식을 따라야 합니다. "
                f"(오류: {error.message})"
            )
        return None

    @staticmethod
    def _extract_related_field_keys(error: ValidationError) -> list[str]:
        """ValidationError.field_ref → 관련 field_key 후보.

        예:
          'applicant.address'                     → ['address']
          'applicant.representative,submission.submitter_name'
                                                  → ['representative', 'submitter_name']
        field_key 기반 region_map을 탐색할 때 dotted path의 마지막 세그먼트를
        쓰고, x-assembly-rules가 `agent_name → agent.name`처럼 접두사를 다시
        붙일 수 있으므로 원본 + 변형까지 후보로 반환.
        """
        ref = error.field_ref or ""
        if not ref:
            return []
        candidates: list[str] = []
        for token in (t.strip() for t in ref.split(",")):
            if not token:
                continue
            last = token.split(".")[-1]
            if last:
                candidates.append(last)
            # agent.name → agent_name 같은 합성 field_key도 후보에 추가
            if "." in token:
                joined = token.replace(".", "_")
                if joined not in candidates:
                    candidates.append(joined)
        return candidates

    @staticmethod
    def _select_relevant_context(
        field_key: str, high_conf: dict[str, str],
    ) -> dict[str, str]:
        rel = _RELEVANCE_MAP.get(field_key, [])
        return {k: high_conf[k] for k in rel if k in high_conf}

    # ────────────────────────────────────────────────────────────
    #  유틸
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_high_confidence_values(
        fields: list[FieldValue],
        threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for fv in fields:
            if fv.confidence < threshold:
                continue
            val = fv.corrected_value
            if not val or val in ("None", "{}", "[]"):
                continue
            out[fv.field_key] = val
        return out

    @staticmethod
    def _find_region_by_field_key(
        field_key: str, region_map: dict[str, CroppedRegion],
    ) -> Optional[CroppedRegion]:
        for region in region_map.values():
            if getattr(region.instruction_spec, "field_key", None) == field_key:
                return region
        return None

    @staticmethod
    def _parse_and_unwrap(raw: str, field_key: str) -> str:
        """VLM 응답 → 순수 scalar 값. 코드 펜스·공백·래퍼 단계별 해제.

        Assembler._unwrap_blob과 동일한 필드키/단일키 언래핑 규칙.
        """
        import re as _re
        text = (raw or "").strip()
        if not text:
            return ""

        parsed = None
        # 1) 그대로 파싱
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        # 2) 코드펜스 제거 후 재시도
        if parsed is None:
            m = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, _re.DOTALL)
            if m:
                inner = m.group(1).strip()
                try:
                    parsed = json.loads(inner)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
        # 3) 가장 바깥 {...} 블록 추출
        if parsed is None and "{" in text and "}" in text:
            start = text.find("{")
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start:i + 1])
                        except (json.JSONDecodeError, TypeError):
                            parsed = None
                        break
        if parsed is None:
            return text

        # 래퍼 언래핑
        if isinstance(parsed, dict) and field_key in parsed and len(parsed) == 1:
            parsed = parsed[field_key]
        if isinstance(parsed, dict) and len(parsed) == 1:
            parsed = next(iter(parsed.values()))
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False)
        return str(parsed)
