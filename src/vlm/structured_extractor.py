"""P3-B — StructuredExtractor

pixel_budget 기준 배치 그룹을 Gemma4에 전송하여 구조화 추출을 수행합니다.
기존 gemma4_engine.py의 필드 추출 + 표 인식 로직을 분리한 컴포넌트.

처리 흐름:
  1. pixel_budget별 CroppedRegion 그룹 수신 (P2.5-C ResolutionRouter 출력)
  2. 각 그룹 내 크롭 이미지 + instruction → vLLM 호출
  3. guided_json + logprobs로 필드 추출
  4. 도메인 코드 자동 감지
  5. VLMResult 조립

Input:  dict[int, list[CroppedRegion]] + FormType + schema
Output: VLMResult
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.interfaces.enums import CodeType, FormType, ProcessingPath
from src.interfaces.types import (
    CroppedRegion,
    DomainCode,
    FieldValue,
    InstructionSpec,
    RecognizedTable,
    VLMResult,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  도메인 코드 패턴 (P3-B 내부 책임)
# ─────────────────────────────────────────────

_CODE_PATTERNS: list[tuple[str, CodeType]] = [
    (r"\d{4}-\d{2}-\d{3}-\d{4}", CodeType.NSN),           # NSN
    (r"KN-\d{5}-\d{4}", CodeType.K_NSN),                   # K-NSN
    (r"\d{2,4}부대", CodeType.UNIT_CODE),                   # 부대코드
    (r"\d{4}-\d{2}-\d{2}", CodeType.DATE),                  # 날짜
]


def _safe_json_loads(text: str) -> Optional[dict]:
    """VLM 응답에서 JSON 객체를 견고하게 파싱.

    실제 운영에서 VLM은 guided_json을 따라야 하지만 종종 다음 패턴을 섞어
    반환합니다:
      1. ```json ... ``` 마크다운 코드 펜스
      2. JSON 앞·뒤에 자연어 설명
      3. 단일 따옴표 사용
      4. 끝부분에 trailing comma 또는 절단

    파싱 시도 순서:
      1. raw text 그대로 json.loads
      2. 코드 펜스(```json ... ```) 안쪽 추출 후 재시도
      3. 가장 바깥 { ... } 블록 추출 후 재시도
      4. 모두 실패 시 None
    """
    if not text:
        return None

    # 1) 그대로 시도
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) 코드 펜스 제거
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            result = json.loads(candidate)
            return result if isinstance(result, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) 가장 바깥 { ... } 블록 추출 (균형 잡힌 중괄호)
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        result = json.loads(candidate)
                        return result if isinstance(result, dict) else None
                    except (json.JSONDecodeError, TypeError):
                        break

    return None


def _detect_domain_codes(parsed_json: dict) -> list[DomainCode]:
    """VLM 출력 JSON에서 도메인 코드 패턴을 자동 감지.

    Args:
        parsed_json: VLM이 출력한 파싱된 JSON dict.

    Returns:
        감지된 DomainCode 목록.
    """
    codes: list[DomainCode] = []
    seen: set[str] = set()

    def _scan_value(val: Any) -> None:
        if isinstance(val, str):
            for pattern, code_type in _CODE_PATTERNS:
                for match in re.finditer(pattern, val):
                    raw = match.group()
                    if raw not in seen:
                        seen.add(raw)
                        codes.append(DomainCode(
                            code_type=code_type,
                            raw_value=raw,
                            normalized_value=raw,
                            confidence=0.9,
                        ))
        elif isinstance(val, dict):
            for v in val.values():
                _scan_value(v)
        elif isinstance(val, list):
            for item in val:
                _scan_value(item)

    _scan_value(parsed_json)
    return codes


@dataclass
class StructuredExtractorConfig:
    """P3-B StructuredExtractor 설정."""

    # vLLM 서버 연결
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "/models/gemma4/gemma-4-26b-a4b-it/"

    # API 파라미터
    max_tokens: int = 4096
    temperature: float = 0.0   # 결정론적 OCR 출력 — 모든 호출에 적용
    top_logprobs: int = 5
    request_timeout: float = 120.0

    # 저신뢰 필드 재시도 (PIPELINE.md §3-5)
    retry_enabled: bool = True
    retry_threshold: float = 0.60       # 이하면 재시도 대상
    max_retries: int = 1                # 영역당 최대 재시도 횟수


# pixel_budget 한 단계 상향 매핑 (재시도 시 사용)
RETRY_BUDGET_MAP: dict[int, int] = {
    140: 280,
    280: 560,
    560: 1120,
    1120: 1120,  # 이미 최대 — 동일 budget 유지
}


class StructuredExtractor:
    """P3-B — 배치 병렬 VLM 구조화 추출.

    사용:
        extractor = StructuredExtractor(StructuredExtractorConfig(
            vllm_base_url="http://vllm-server:8000/v1",
        ))
        result = extractor.extract(groups, doc_id, form_type, schema_id, schema)
    """

    def __init__(
        self,
        config: Optional[StructuredExtractorConfig] = None,
        ocr_hint_provider=None,
    ):
        self.cfg = config or StructuredExtractorConfig()
        self._client = None
        # OCR-augmented 힌트 제공자 (선택적, 저신뢰 재시도 시 활용)
        self._ocr_hint_provider = ocr_hint_provider

    def _get_client(self):
        """OpenAI 클라이언트 지연 초기화."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.cfg.vllm_base_url,
                api_key="not-needed",
                timeout=self.cfg.request_timeout,
            )
        return self._client

    def extract(
        self,
        groups: dict[int, list[CroppedRegion]],
        doc_id: str,
        form_type: FormType,
        form_confidence: float,
        schema_id: str,
        schema: Optional[dict] = None,
        warnings: Optional[list[str]] = None,
        trace: Optional[list[dict]] = None,
    ) -> VLMResult:
        """pixel_budget별 배치 그룹 → VLMResult.

        Args:
            groups: {pixel_budget → [CroppedRegion, ...]} (P2.5-C 출력)
            doc_id: 문서 ID
            form_type: P3-A FormClassifier 결과
            form_confidence: 서식 분류 신뢰도
            schema_id: guided_json 스키마 ID
            schema: JSON Schema (guided_json용)
            warnings: 경고 수집 리스트
            trace: 영역별 vLLM 호출 trace 수집 리스트 (옵션). 호출자가 빈 리스트를
                전달하면 영역별 입출력/소요시간/추출 결과가 dict로 채워집니다.
                디버깅·결과 검증용.

        Returns:
            VLMResult
        """
        if warnings is None:
            warnings = []

        t0 = time.time()

        all_fields: list[FieldValue] = []
        all_tables: list[RecognizedTable] = []
        all_domain_codes: list[DomainCode] = []
        raw_json_parts: list[str] = []

        # pixel_budget별 순차 처리 (각 그룹 내에서는 순차 호출)
        for budget, regions in sorted(groups.items(), reverse=True):
            logger.info(
                "[P3-B][%s] 배치 처리: budget=%d, regions=%d",
                doc_id, budget, len(regions),
            )

            for cropped in regions:
                region_t0 = time.time()
                try:
                    result = self._process_single(cropped, schema, warnings)
                    region_ms = (time.time() - region_t0) * 1000
                    if result is None:
                        if trace is not None:
                            trace.append({
                                "region_id": cropped.region_id,
                                "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                                "pixel_budget": cropped.pixel_budget,
                                "elapsed_ms": round(region_ms, 1),
                                "status": "skipped",
                                "field_count": 0,
                                "table_count": 0,
                                "domain_code_count": 0,
                                "raw_response": "",
                            })
                        continue

                    fields, tables, codes, raw = result
                    all_fields.extend(fields)
                    all_tables.extend(tables)
                    all_domain_codes.extend(codes)
                    if raw:
                        raw_json_parts.append(raw)

                    if trace is not None:
                        trace.append({
                            "region_id": cropped.region_id,
                            "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                            "pixel_budget": cropped.pixel_budget,
                            "elapsed_ms": round(region_ms, 1),
                            "status": "ok",
                            "field_count": len(fields),
                            "table_count": len(tables),
                            "domain_code_count": len(codes),
                            "instruction": cropped.instruction_spec.user_instruction,
                            "system_prompt": cropped.instruction_spec.system_prompt,
                            "raw_response": raw,
                            "fields": [
                                {
                                    "field_key": f.field_key,
                                    "raw_value": f.raw_value,
                                    "data_type": f.data_type,
                                    "confidence": round(f.confidence, 4),
                                    "is_flagged": f.is_flagged,
                                }
                                for f in fields
                            ],
                        })

                except Exception as e:
                    region_ms = (time.time() - region_t0) * 1000
                    warnings.append(
                        f"Region {cropped.region_id} failed: {e}"
                    )
                    logger.warning(
                        "[P3-B] 영역 처리 실패 (%s): %s",
                        cropped.region_id, e,
                    )
                    if trace is not None:
                        trace.append({
                            "region_id": cropped.region_id,
                            "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                            "pixel_budget": cropped.pixel_budget,
                            "elapsed_ms": round(region_ms, 1),
                            "status": "error",
                            "error": f"{type(e).__name__}: {e}",
                        })

        # ─── 저신뢰 필드 재시도 (OCR-augmented 강화) ───
        retry_count = 0
        if self.cfg.retry_enabled and form_type != FormType.OTHER:
            retry_count = self._retry_low_confidence_fields(
                groups=groups,
                all_fields=all_fields,
                schema=schema,
                warnings=warnings,
                trace=trace,
            )

        # ─── Assembler: x-assembly-rules 있는 스키마는 region별 결과를 full dict로 조립 ───
        assembled_json = None
        if schema and schema.get("x-assembly-rules"):
            from src.vlm.assembler import Assembler
            region_field_key_map: dict[str, str] = {}
            for regions in groups.values():
                for cr in regions:
                    fk = getattr(cr.instruction_spec, "field_key", None)
                    if fk:
                        region_field_key_map[cr.region_id] = fk
            assembled_json = Assembler().assemble(
                fields=all_fields,
                schema=schema,
                region_field_key_map=region_field_key_map,
                warnings=warnings,
            )
            if assembled_json:
                logger.info(
                    "[P3-B][%s] Assembler 조립 완료: keys=%s",
                    doc_id, list(assembled_json.keys()),
                )

        processing_time_ms = (time.time() - t0) * 1000

        logger.info(
            "[P3-B][%s] 완료: fields=%d, tables=%d, codes=%d, retries=%d, %.0fms",
            doc_id, len(all_fields), len(all_tables),
            len(all_domain_codes), retry_count, processing_time_ms,
        )

        return VLMResult(
            doc_id=doc_id,
            form_type=form_type,
            form_confidence=form_confidence,
            schema_id=schema_id,
            fields=all_fields,
            tables=all_tables,
            domain_codes=all_domain_codes,
            raw_json="",
            processing_path=ProcessingPath.VLM,
            processing_time_ms=round(processing_time_ms, 1),
            warnings=warnings,
            retry_count=retry_count,
            assembled_json=assembled_json,
        )

    def _retry_low_confidence_fields(
        self,
        groups: dict[int, list[CroppedRegion]],
        all_fields: list[FieldValue],
        schema: Optional[dict],
        warnings: list[str],
        trace: Optional[list[dict]],
    ) -> int:
        """저신뢰 필드 재시도 — pixel_budget 한 단계 상향 + OCR 힌트 부착.

        PIPELINE.md §3-5: RETRY_THRESHOLD=0.60 미만 필드 → 1회 재시도.

        Args:
            groups: pixel_budget별 그룹 (region_id 매핑용)
            all_fields: 1차 처리된 필드 목록 (제자리 갱신)
            schema: guided_json 스키마
            warnings: 경고 수집 리스트
            trace: trace 수집 리스트 (옵션)

        Returns:
            실제 재시도 발생 횟수
        """
        from src.vlm.instruction_router import InstructionRouter

        # region_id → CroppedRegion 매핑
        region_map: dict[str, CroppedRegion] = {}
        for regions in groups.values():
            for cr in regions:
                region_map[cr.region_id] = cr

        # 재시도 후보: confidence < threshold AND not yet retried AND has region_id
        retry_targets: dict[str, list[int]] = {}  # region_id → fields 인덱스 목록
        for idx, f in enumerate(all_fields):
            if (
                f.confidence < self.cfg.retry_threshold
                and not f.was_retried
                and f.region_id
                and f.region_id in region_map
            ):
                retry_targets.setdefault(f.region_id, []).append(idx)

        if not retry_targets:
            return 0

        retry_count = 0
        for region_id, field_indices in retry_targets.items():
            cropped = region_map[region_id]
            spec = cropped.instruction_spec
            old_budget = cropped.pixel_budget
            new_budget = RETRY_BUDGET_MAP.get(old_budget, old_budget)

            # OCR 힌트 생성 (가용 시)
            hint = ""
            if self._ocr_hint_provider is not None and self._ocr_hint_provider.enabled:
                try:
                    hint = self._ocr_hint_provider.get_hint(cropped.cropped_image)
                except Exception as e:
                    logger.warning(
                        "[P3-B] OCR 힌트 생성 실패 (%s): %s", region_id, e
                    )

            # 새 InstructionSpec — pixel_budget 상향 + OCR 힌트 + is_retry 표시
            retry_spec = InstructionSpec(
                region_id=spec.region_id,
                region_type=spec.region_type,
                form_type=spec.form_type,
                system_prompt=spec.system_prompt,
                user_instruction=spec.user_instruction,
                json_schema=spec.json_schema,
                pixel_budget=new_budget,
                is_retry=True,
                field_key=getattr(spec, "field_key", None),
            )
            if hint:
                retry_spec = InstructionRouter.with_ocr_hint(retry_spec, hint)

            retry_cropped = CroppedRegion(
                region_id=cropped.region_id,
                region_type=cropped.region_type,
                cropped_image=cropped.cropped_image,
                pixel_budget=new_budget,
                instruction_spec=retry_spec,
            )

            logger.info(
                "[P3-B] 재시도 %s: budget %d→%d, fields=%d, ocr_hint=%s",
                region_id, old_budget, new_budget, len(field_indices),
                "Y" if hint else "N",
            )

            region_t0 = time.time()
            try:
                result = self._process_single(retry_cropped, schema, warnings)
                region_ms = (time.time() - region_t0) * 1000
                retry_count += 1

                if result is None:
                    if trace is not None:
                        trace.append({
                            "region_id": region_id,
                            "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                            "pixel_budget": new_budget,
                            "elapsed_ms": round(region_ms, 1),
                            "status": "retry_skipped",
                            "is_retry": True,
                            "ocr_hint_used": bool(hint),
                        })
                    continue

                new_fields, _new_tables, _new_codes, new_raw = result

                # 필드 갱신: 재시도 결과가 더 높은 신뢰도면 교체, 아니면 was_retried만 표시
                new_field_map = {f.field_key: f for f in new_fields}
                for idx in field_indices:
                    old_field = all_fields[idx]
                    new_field = new_field_map.get(old_field.field_key)
                    if new_field is not None and new_field.confidence > old_field.confidence:
                        # 신뢰도 상승 — 교체 + was_retried 표시
                        all_fields[idx] = FieldValue(
                            field_key=new_field.field_key,
                            raw_value=new_field.raw_value,
                            corrected_value=new_field.corrected_value,
                            data_type=new_field.data_type,
                            confidence=new_field.confidence,
                            token_logprobs=new_field.token_logprobs,
                            is_flagged=new_field.is_flagged,
                            region_id=new_field.region_id,
                            was_retried=True,
                        )
                    else:
                        # 재시도해도 개선 없음 — 표시만
                        all_fields[idx] = FieldValue(
                            field_key=old_field.field_key,
                            raw_value=old_field.raw_value,
                            corrected_value=old_field.corrected_value,
                            data_type=old_field.data_type,
                            confidence=old_field.confidence,
                            token_logprobs=old_field.token_logprobs,
                            is_flagged=old_field.is_flagged,
                            region_id=old_field.region_id,
                            was_retried=True,
                        )

                if trace is not None:
                    trace.append({
                        "region_id": region_id,
                        "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                        "pixel_budget": new_budget,
                        "elapsed_ms": round(region_ms, 1),
                        "status": "retry_ok",
                        "is_retry": True,
                        "ocr_hint_used": bool(hint),
                        "ocr_hint_chars": len(hint),
                        "field_count": len(new_fields),
                        "instruction": retry_spec.user_instruction,
                        "raw_response": new_raw,
                    })

            except Exception as e:
                region_ms = (time.time() - region_t0) * 1000
                warnings.append(
                    f"Region {region_id} retry failed: {e}"
                )
                logger.warning(
                    "[P3-B] 재시도 실패 (%s): %s", region_id, e
                )
                if trace is not None:
                    trace.append({
                        "region_id": region_id,
                        "pixel_budget": new_budget,
                        "elapsed_ms": round(region_ms, 1),
                        "status": "retry_error",
                        "is_retry": True,
                        "error": f"{type(e).__name__}: {e}",
                    })

        return retry_count

    def _process_single(
        self,
        cropped: CroppedRegion,
        schema: Optional[dict],
        warnings: list[str],
    ) -> Optional[tuple[list[FieldValue], list[RecognizedTable], list[DomainCode], str]]:
        """단일 CroppedRegion 처리.

        Returns:
            (fields, tables, domain_codes, raw_json) 또는 None
        """
        from src.vlm.vlm_client import encode_image_base64, extract_field_logprobs
        from src.vlm.logprobs_scorer import calc_field_confidence, is_flagged

        spec = cropped.instruction_spec
        region_type = (
            cropped.region_type.value
            if hasattr(cropped.region_type, "value")
            else str(cropped.region_type)
        )

        # 이미지 인코딩 (pixel_budget 기반 최대 크기)
        max_px = int((cropped.pixel_budget * 28 * 28) ** 0.5)
        b64 = encode_image_base64(cropped.cropped_image, max_size=max_px)

        # vLLM 호출
        response = self._call_vlm(
            instruction=spec.user_instruction,
            image_b64=b64,
            system_prompt=spec.system_prompt,
            guided_json=spec.json_schema,
            pixel_budget=cropped.pixel_budget,
        )

        text = response.get("text", "").strip()
        logprobs = response.get("logprobs", [])

        # 표 영역 처리
        if region_type == "table":
            tables = self._process_table(cropped.region_id, text, logprobs, warnings)
            return [], tables, [], text

        # 필드 추출 (guided_json이 있는 경우)
        fields: list[FieldValue] = []
        domain_codes: list[DomainCode] = []

        parsed = _safe_json_loads(text) if text else None
        if parsed is None:
            warnings.append(f"Region {cropped.region_id}: JSON parse failed")
            return [], [], [], text

        # region에 template field_key가 지정된 경우:
        # VLM 출력 전체(dict/list/스칼라)를 단일 FieldValue로 보존 → Assembler가 온전히 복원
        template_field_key = getattr(spec, "field_key", None)
        if template_field_key:
            val_str = (
                json.dumps(parsed, ensure_ascii=False)
                if isinstance(parsed, (list, dict))
                else str(parsed)
            )
            # logprobs 전체 평균으로 신뢰도 산출
            all_lps = [
                lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
                for lp in (logprobs or [])
            ]
            confidence = calc_field_confidence(all_lps, "text")
            fields.append(FieldValue(
                field_key=template_field_key,
                raw_value=val_str,
                corrected_value=val_str,
                data_type="text",
                confidence=confidence,
                token_logprobs=all_lps,
                is_flagged=is_flagged(confidence, "text"),
                region_id=cropped.region_id,
            ))
            if isinstance(parsed, dict):
                domain_codes = _detect_domain_codes(parsed)
            return fields, [], domain_codes, text

        if isinstance(parsed, dict):
            # logprobs → 필드별 신뢰도
            field_logprobs = extract_field_logprobs(logprobs, text, schema)

            for key, value in parsed.items():
                # 메타 필드 스킵:
                # - form_type: 스키마 분류 결과 (이미 form_classifier가 확정)
                # - analysis: CoT 사고 과정 (값이 아닌 추론 보조 정보 — DB 저장 대상 아님)
                if key in ("form_type", "analysis"):
                    continue

                val_str = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else str(value)
                )
                data_type = "text"
                token_lps: list[float] = []

                if key in field_logprobs:
                    token_lps, data_type = field_logprobs[key]

                confidence = calc_field_confidence(token_lps, data_type)
                flagged = is_flagged(confidence, data_type)

                fields.append(FieldValue(
                    field_key=key,
                    raw_value=val_str,
                    corrected_value=val_str,
                    data_type=data_type,
                    confidence=confidence,
                    token_logprobs=token_lps,
                    is_flagged=flagged,
                    region_id=cropped.region_id,
                ))

            # 도메인 코드 감지 (P3-B 내부 책임)
            domain_codes = _detect_domain_codes(parsed)

        return fields, [], domain_codes, text

    def _process_table(
        self,
        region_id: str,
        html: str,
        logprobs: list[dict],
        warnings: list[str],
    ) -> list[RecognizedTable]:
        """표 응답 파싱."""
        from src.vlm.vlm_client import parse_html_cells
        from src.vlm.logprobs_scorer import calc_field_confidence

        confidence = 0.5
        if logprobs:
            all_lps = [
                lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
                for lp in logprobs
            ]
            if all_lps:
                confidence = calc_field_confidence(all_lps, "text")

        cells = parse_html_cells(html)

        return [RecognizedTable(
            region_id=region_id,
            html=html,
            cells=cells,
            confidence=confidence,
        )]

    def _call_vlm(
        self,
        instruction: str,
        image_b64: str,
        system_prompt: str = "",
        guided_json: Optional[dict] = None,
        pixel_budget: Optional[int] = None,
    ) -> dict[str, Any]:
        """vLLM OpenAI 호환 API 호출."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": instruction},
            ],
        })

        extra_body: dict[str, Any] = {
            "logprobs": True,
            "top_logprobs": self.cfg.top_logprobs,
        }
        if guided_json is not None:
            extra_body["guided_json"] = guided_json
        if pixel_budget:
            extra_body["mm_processor_kwargs"] = {
                "max_pixels": pixel_budget * 28 * 28,
            }

        response = client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            extra_body=extra_body,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        logprobs_data: list[dict] = []
        if hasattr(choice, "logprobs") and choice.logprobs:
            content_logprobs = getattr(choice.logprobs, "content", None)
            if content_logprobs:
                for token_info in content_logprobs:
                    logprobs_data.append({
                        "token": getattr(token_info, "token", ""),
                        "logprob": getattr(token_info, "logprob", 0.0),
                    })

        return {"text": text, "logprobs": logprobs_data}
