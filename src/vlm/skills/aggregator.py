"""S7 StructuredAggregator — S2~S6 결과를 official_document.json Schema로 집계.

PIPELINE.md §4-8 구현.
  - 이미지 재호출 없이 S2~S6 텍스트 결과를 프롬프트 컨텍스트로 주입
  - 전체 페이지 이미지는 저해상도(140 토큰)로 함께 제공하여 구조 파악 보조
  - official_document.json 스키마 강제 (guided_json)
  - other 경로 전용

설계 사상:
  S7은 개별 영역을 다시 보는 것이 아니라 S2~S6가 이미 추출한 텍스트/구조 결과를
  프롬프트에 주입하고 VLM이 official_document.json 스키마로 재구성하도록 한다.
  이미지를 다시 보지 않아도 되므로 pixel_budget은 낮게 유지한다.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import numpy as np

from src.domain.schema_registry import SchemaRegistry
from src.interfaces.enums import FormType, ProcessingPath
from src.interfaces.types import (
    FieldValue,
    RecognizedTable,
    SkillResult,
    TableStructure,
    VLMResult,
)
from src.vlm.logprobs_scorer import calc_field_confidence, is_flagged as _is_flagged
from src.vlm.skills._parsing import _loads_relaxed
from src.vlm.vlm_client import VLMClient, encode_image_base64

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_AGGREGATOR = (
    "You are a Korean official document OCR assistant.\n"
    "You will be given extracted text blocks from each region of a document,\n"
    "along with a low-resolution view of the full page for structural context.\n"
    "Organize the extracted results into the structured JSON format requested.\n"
    "If a field cannot be determined from the extracted results, leave it as null.\n"
    "Do not fabricate values that are not present in the extracted blocks."
)


class StructuredAggregator:
    """S2~S6 결과를 official_document.json Schema에 맞춰 최종 JSON 조립.

    이미지 재호출 없이 S2~S6 텍스트 결과를 프롬프트에 주입하여
    VLM이 official_document.json 스키마로 집계하도록 합니다.
    other 경로 전용.
    """

    SKILL_NAME = "S7"
    PIXEL_BUDGET = 140  # 전체 페이지 저해상도 (문서 구조 파악용)
    SCHEMA_ID = "official_document"

    def __init__(
        self,
        vlm_client: VLMClient,
        schema_registry: Optional[SchemaRegistry] = None,
    ):
        self.vlm = vlm_client
        self._schema_registry = schema_registry or SchemaRegistry()

    # ─────────────────────────────────────────────
    #  메인 진입점
    # ─────────────────────────────────────────────
    def run(
        self,
        skill_results: list[SkillResult],
        table_structures: list[TableStructure],
        page_image: np.ndarray,
        form_type: FormType,
        doc_id: str,
        form_confidence: float = 0.0,
        warnings: Optional[list[str]] = None,
    ) -> VLMResult:
        """S2~S6 결과 → official_document.json 스키마 집계.

        Args:
            skill_results:    S2~S6 SkillResult 목록
            table_structures: S5 패스1 TableStructure 목록
            page_image:       전체 페이지 이미지 (H×W×3 RGB, 저해상도 컨텍스트용)
            form_type:        FormType.OTHER
            doc_id:           문서 식별자
            form_confidence:  FormClassifier 분류 신뢰도 (VLMResult에 그대로 전달)
            warnings:         호출자가 전달하는 누적 warnings (읽기 전용, 향후 컨텍스트 참조용).
                              반환 VLMResult.warnings에는 포함하지 않으므로 이중 prefix가
                              발생하지 않는다.

        Returns:
            VLMResult — assembled_json에 official_document 구조 저장, fields에
            region_id 단위 FieldValue도 함께 보존(P4/검토 큐 추적용).
        """
        # 반환 VLMResult.warnings에는 S7 자체 message만 담는다.
        # 호출자(orchestrator)가 단계 접두사("[SkillRegistry]")를 별도로 붙이므로
        # 여기서 받은 warnings(누적분)를 다시 반환하면 이중 prefix가 발생한다.
        s7_warnings: list[str] = []
        t0 = time.time()

        context = self._build_context(skill_results, table_structures)

        schema = self._schema_registry.get(self.SCHEMA_ID, "v1")
        if schema is None:
            s7_warnings.append("official_document 스키마 로드 실패")
            logger.warning("[S7] schema load failed (id=%s)", self.SCHEMA_ID)

        instruction = self._build_instruction(context)

        try:
            image_b64 = encode_image_base64(page_image, max_size=self.PIXEL_BUDGET)
            response = self.vlm.call(
                image_b64=image_b64,
                instruction=instruction,
                system_prompt=SYSTEM_PROMPT_AGGREGATOR,
                guided_json=schema,
                pixel_budget=self.PIXEL_BUDGET,
            )
        except Exception as e:
            logger.error("[S7] VLM 호출 실패 doc=%s: %s", doc_id, e)
            s7_warnings.append(f"vlm_call_failed: {type(e).__name__}")
            return self._fallback_result(
                skill_results=skill_results,
                doc_id=doc_id,
                form_type=form_type,
                form_confidence=form_confidence,
                processing_ms=(time.time() - t0) * 1000,
                warnings=s7_warnings,
                context=context,
            )

        raw = response.get("content", "") or ""
        assembled = _loads_relaxed(raw)
        if assembled is None:
            s7_warnings.append("집계 JSON 파싱 실패")
            logger.warning("[S7] JSON parse failed doc=%s raw=%s",
                           doc_id, raw[:200].replace("\n", " "))

        # 신뢰도 산출
        logprobs = response.get("logprobs", []) or []
        token_lps = [
            lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
            for lp in logprobs
        ]
        aggregator_confidence = (
            calc_field_confidence(token_lps, "text") if token_lps else 0.0
        )

        # fields: region_id 단위 FieldValue 보존 (P4/검토 큐 추적용)
        fields = _skill_results_to_fields(skill_results)

        # S7 자체 집계 결과도 aggregator_blob field로 추가
        if assembled is not None:
            fields.append(FieldValue(
                field_key="aggregator_blob",
                raw_value=json.dumps(assembled, ensure_ascii=False),
                corrected_value=json.dumps(assembled, ensure_ascii=False),
                data_type="text",
                confidence=aggregator_confidence,
                token_logprobs=token_lps,
                is_flagged=_is_flagged(aggregator_confidence, "text"),
                region_id=None,
            ))

        processing_ms = (time.time() - t0) * 1000

        logger.info(
            "[S7] 집계 완료 doc=%s, regions=%d, tables=%d, assembled=%s, %.0fms",
            doc_id, len(skill_results), len(table_structures),
            "OK" if assembled else "FAIL", processing_ms,
        )

        return VLMResult(
            doc_id=doc_id,
            form_type=form_type,
            form_confidence=form_confidence,
            schema_id=self.SCHEMA_ID,
            fields=fields,
            tables=[],
            domain_codes=[],
            raw_json=raw,
            processing_time_ms=processing_ms,
            processing_path=ProcessingPath.SKILL_REGISTRY,
            warnings=s7_warnings,
            assembled_json=assembled if isinstance(assembled, dict) else None,
        )

    # ─────────────────────────────────────────────
    #  Context 직렬화
    # ─────────────────────────────────────────────
    def _build_context(
        self,
        skill_results: list[SkillResult],
        table_structures: list[TableStructure],
    ) -> str:
        """S2~S6 결과를 VLM이 이해할 수 있는 텍스트 블록으로 직렬화.

        예시 출력:
          [HEADER] 국방부 (r_0001)
          [TEXT] 제 목: 의안 심사기간 지정의 건 (r_0010)
          [SEAL] detected=True, text="국방부장관인" (r_0022)
          [SIGNATURE] present=True (r_0023)
          [TABLE approval] (r_0030)
            row0: ['기안', '검토', '결재']
            row1: ['홍길동', '이순신', '김유신']
        """
        # table region_id → TableStructure lookup
        table_lookup = {t.region_id: t for t in table_structures}
        # table 셀 SkillResult는 "{region_id}_r{row}c{col}" 패턴 — 중첩 출력을 위해 묶어 처리
        table_cells: dict[str, list[SkillResult]] = {}

        lines: list[str] = []
        for r in skill_results:
            # 표 셀은 별도 섹션에서 처리
            parent_region_id = _parent_table_region(r.region_id, table_lookup)
            if parent_region_id is not None:
                table_cells.setdefault(parent_region_id, []).append(r)
                continue

            prefix = _skill_to_prefix(r)
            body = _format_skill_body(r)
            lines.append(f"[{prefix}] {body} ({r.region_id})")

        # 표 섹션 출력
        for region_id, structure in table_lookup.items():
            lines.append(self._serialize_table(structure, table_cells.get(region_id, [])))

        if not lines:
            return "(추출된 영역 없음)"
        return "\n".join(lines)

    def _build_instruction(self, context: str) -> str:
        return (
            "아래는 이 문서에서 영역별로 추출된 인식 결과입니다.\n"
            "이 결과를 바탕으로 official_document 스키마에 맞춰 JSON을 생성하세요.\n"
            "결과에 없는 필드는 null로 두고, 절대로 새 값을 만들어내지 마세요.\n\n"
            "=== 추출 결과 ===\n"
            f"{context}\n"
            "=== 끝 ==="
        )

    def _serialize_table(
        self,
        table: TableStructure,
        cell_results: list[SkillResult],
    ) -> str:
        """TableStructure + 셀 SkillResult를 텍스트로 직렬화.

        TableCell에는 content 필드가 없으므로, 셀 SkillResult(region_id 패턴
        "{table_region_id}_r{row}c{col}")에서 row/col/content를 역매핑한다.
        """
        lines = [f"[TABLE {table.table_type}] ({table.region_id})"]

        # "r{row}c{col}" suffix → content
        cell_content: dict[tuple[int, int], str] = {}
        for cr in cell_results:
            rc = _parse_cell_suffix(cr.region_id, table.region_id)
            if rc is None:
                continue
            row, col = rc
            cell_content[(row, col)] = _short_content(cr)

        # row/col 매트릭스 재구성
        rows: dict[int, dict[int, str]] = {}
        for cell in table.cells:
            content = cell_content.get(
                (cell.row, cell.col),
                f"({cell.content_type})" if cell.content_type != "empty" else "",
            )
            rows.setdefault(cell.row, {})[cell.col] = {
                "role": cell.role,
                "content": content,
            }

        for row_idx in sorted(rows):
            row = rows[row_idx]
            row_vals = []
            for col_idx in sorted(row):
                entry = row[col_idx]
                role = entry["role"]
                content = entry["content"]
                if role and content:
                    row_vals.append(f"{role}={content}")
                elif role:
                    row_vals.append(role)
                else:
                    row_vals.append(content)
            lines.append(f"  row{row_idx}: {row_vals}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    #  VLM 실패 시 폴백
    # ─────────────────────────────────────────────
    def _fallback_result(
        self,
        skill_results: list[SkillResult],
        doc_id: str,
        form_type: FormType,
        form_confidence: float,
        processing_ms: float,
        warnings: list[str],
        context: str,
    ) -> VLMResult:
        """S7 VLM 호출 실패 시 region_id 단위 평탄화로 폴백 (기존 동작 보존)."""
        fields = _skill_results_to_fields(skill_results)
        return VLMResult(
            doc_id=doc_id,
            form_type=form_type,
            form_confidence=form_confidence,
            schema_id=self.SCHEMA_ID,
            fields=fields,
            tables=[],
            domain_codes=[],
            raw_json="",
            processing_time_ms=processing_ms,
            processing_path=ProcessingPath.SKILL_REGISTRY,
            warnings=warnings,
            assembled_json=None,
        )


# ─────────────────────────────────────────────
#  내부 유틸
# ─────────────────────────────────────────────

def _skill_results_to_fields(skill_results: list[SkillResult]) -> list[FieldValue]:
    """SkillResult 리스트 → FieldValue 리스트 (region_id 단위 평탄화).

    orchestrator._skill_results_to_fields()와 동일한 변환을 제공.
    """
    fields: list[FieldValue] = []
    for sr in skill_results:
        data_type = "text"
        if sr.content_type == "signature":
            data_type = "flag"
        fields.append(FieldValue(
            field_key=sr.region_id,
            raw_value=sr.content,
            corrected_value=sr.content,
            data_type=data_type,
            confidence=sr.confidence,
            token_logprobs=[],
            is_flagged=_is_flagged(sr.confidence, data_type),
            region_id=sr.region_id,
        ))
    return fields


def _skill_to_prefix(r: SkillResult) -> str:
    """SkillResult.content_type → 프롬프트 블록 라벨."""
    ct = (r.content_type or "").lower()
    mapping = {
        "printed": "TEXT",
        "handwritten": "HANDWRITTEN",
        "structured": "HANDWRITTEN",
        "seal": "SEAL",
        "signature": "SIGNATURE",
    }
    return mapping.get(ct, ct.upper() or r.skill_name)


def _format_skill_body(r: SkillResult) -> str:
    """Skill 결과 본문을 사람이 읽기 쉬운 한 줄로 변환."""
    ct = (r.content_type or "").lower()
    raw = r.content or ""

    if ct == "seal":
        data = _loads_relaxed(raw)
        if data is not None:
            text = data.get("text") or ""
            conf = data.get("confidence", 0.0)
            return f'detected={bool(text)}, text="{text}", confidence={conf}'
        return f'text="{raw}"'

    if ct == "signature":
        data = _loads_relaxed(raw)
        if data is not None:
            present = bool(data.get("signature_present", False))
            return f"present={present}, confidence={data.get('confidence', 0.0)}"
        return f"raw={raw[:40]}"

    # printed / handwritten / structured / 기타 → 텍스트 요약
    return _short_content(r)


def _short_content(r: SkillResult, max_len: int = 240) -> str:
    """SkillResult.content를 프롬프트용으로 정제 (개행 제거, 길이 제한)."""
    text = (r.content or "").replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return text


def _parent_table_region(
    region_id: str, table_lookup: dict[str, TableStructure],
) -> Optional[str]:
    """셀 region_id가 특정 table region의 자식인지 확인.

    TableExtractor.pass2_route 패턴: "{table_region_id}_r{row}c{col}".
    """
    for parent_id in table_lookup:
        if region_id.startswith(parent_id + "_r"):
            return parent_id
    return None


def _parse_cell_suffix(
    cell_region_id: str, table_region_id: str,
) -> Optional[tuple[int, int]]:
    """'{table}_r{row}c{col}' → (row, col) 추출."""
    prefix = table_region_id + "_r"
    if not cell_region_id.startswith(prefix):
        return None
    tail = cell_region_id[len(prefix):]
    if "c" not in tail:
        return None
    r_str, _, c_str = tail.partition("c")
    try:
        return int(r_str), int(c_str)
    except ValueError:
        return None
