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
    temperature: float = 0.0
    top_logprobs: int = 5
    request_timeout: float = 120.0


class StructuredExtractor:
    """P3-B — 배치 병렬 VLM 구조화 추출.

    사용:
        extractor = StructuredExtractor(StructuredExtractorConfig(
            vllm_base_url="http://vllm-server:8000/v1",
        ))
        result = extractor.extract(groups, doc_id, form_type, schema_id, schema)
    """

    def __init__(self, config: Optional[StructuredExtractorConfig] = None):
        self.cfg = config or StructuredExtractorConfig()
        self._client = None

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

        processing_time_ms = (time.time() - t0) * 1000

        logger.info(
            "[P3-B][%s] 완료: fields=%d, tables=%d, codes=%d, %.0fms",
            doc_id, len(all_fields), len(all_tables),
            len(all_domain_codes), processing_time_ms,
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
        )

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

        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            warnings.append(f"Region {cropped.region_id}: JSON parse failed")
            return [], [], [], text

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
