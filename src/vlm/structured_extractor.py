"""P3-B — StructuredExtractor

pixel_budget 기준 배치 그룹을 Gemma4에 전송하여 구조화 추출을 수행합니다.
VLMClient에 모든 vLLM 호출을 위임하며, 필드 추출·표 인식·저신뢰 재시도를 담당합니다.

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

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.pipeline.health_monitor import VLMHealthMonitor

import numpy as np

from src.interfaces.enums import CodeType, FormType, ProcessingPath, RegionType
from src.interfaces.types import (
    CroppedRegion,
    DomainCode,
    FieldValue,
    InstructionSpec,
    RecognizedTable,
    SkillResult,
    SkillTask,
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
    """VLM 응답에서 JSON 객체(dict)를 견고하게 파싱.

    dict만 반환 — 필드 순회 경로(spec.field_key 없음)에서 사용.
    top-level string/list/scalar 허용이 필요하면 `_loads_any`를 사용.
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


def _loads_any(text: str):
    """VLM 응답에서 JSON 최상위 값을 타입 구별 없이 파싱 (dict/list/str/num/bool/None).

    sub-schema 분해 경로에서 field_key가 있는 region은 단일 값(문자열 리터럴 등)을
    반환할 수 있다. 이 경우 `_safe_json_loads`(dict 전용)는 None을 반환하므로
    별도 함수로 top-level 값을 살려 Assembler에 그대로 전달한다.

    반환:
      - 파싱 성공: 파이썬 값 (dict/list/str/int/float/bool/None)
      - 파싱 실패: 센티넬 문자열 sentinel 사용 대신 None만 쓰지 않도록
        Exception을 호출자에게 제기하지 않고, raw text를 그대로 쓰도록 상위에서 폴백.
    """
    if not text:
        return None

    # 1) 그대로
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) 코드 펜스
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) dict/list 블록 탐색
    for ob, cb in (("{", "}"), ("[", "]")):
        start = text.find(ob)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == ob:
                depth += 1
            elif text[i] == cb:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
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

    monitor: "Optional[VLMHealthMonitor]" = field(default=None, repr=False)


# pixel_budget 한 단계 상향 매핑 (재시도 시 사용)
RETRY_BUDGET_MAP: dict[int, int] = {
    140: 280,
    280: 560,
    560: 1120,
    1120: 1120,  # 이미 최대 — 동일 budget 유지
}

# 동시 처리 제한 — vLLM --max-num-seqs=64 기준 여유 있는 상한.
# 동일 이벤트 루프에서 벡터화된 I/O만 대기하므로 CPU 부담은 미미.
MAX_CONCURRENT_REQUESTS: int = 16


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
        printed_text_reader=None,
        handwriting_reader=None,
    ):
        self.cfg = config or StructuredExtractorConfig()
        # VLMClient: 모든 vLLM 호출을 위임
        from src.vlm.vlm_client import VLMClient
        self._vlm_client = VLMClient(
            base_url=self.cfg.vllm_base_url,
            model_name=self.cfg.model_name,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_logprobs=self.cfg.top_logprobs,
            timeout=self.cfg.request_timeout,
            monitor=self.cfg.monitor,
        )
        # OCR-augmented 힌트 제공자 (선택적, 저신뢰 재시도 시 활용)
        self._ocr_hint_provider = ocr_hint_provider
        # 공통 도메인 서비스 — Orchestrator가 주입. 생략 시 지연 생성.
        self._s2 = printed_text_reader
        self._s3 = handwriting_reader

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
        fixed_values: Optional[dict] = None,
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

        # pixel_budget별 그룹 처리 — 그룹 간 순서 유지(prefix caching),
        # 그룹 내 region은 asyncio.gather로 병렬 처리 (MAX_CONCURRENT_REQUESTS 상한).
        async def _gather_group(
            regions: list[CroppedRegion],
            semaphore: asyncio.Semaphore,
        ) -> list[tuple[CroppedRegion, float, Any]]:
            async def _one(cr: CroppedRegion):
                async with semaphore:
                    t0 = time.time()
                    try:
                        res = await asyncio.to_thread(
                            self._process_single, cr, schema, warnings, form_type
                        )
                        return cr, (time.time() - t0) * 1000, res
                    except Exception as e:
                        return cr, (time.time() - t0) * 1000, e
            return await asyncio.gather(*[_one(c) for c in regions])

        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def _run_all_groups():
            out: list[tuple[CroppedRegion, float, Any]] = []
            for budget, regions in sorted(groups.items(), reverse=True):
                logger.info(
                    "[P3-B][%s] 배치 처리(병렬): budget=%d, regions=%d",
                    doc_id, budget, len(regions),
                )
                out.extend(await _gather_group(regions, sem))
            return out

        try:
            gathered = asyncio.run(_run_all_groups())
        except RuntimeError:
            # 이미 이벤트 루프가 동작 중인 드문 상황: nested 실행 폴백
            loop = asyncio.new_event_loop()
            try:
                gathered = loop.run_until_complete(_run_all_groups())
            finally:
                loop.close()

        for cropped, region_ms, outcome in gathered:
            if isinstance(outcome, Exception):
                warnings.append(f"Region {cropped.region_id} failed: {outcome}")
                logger.warning(
                    "[P3-B] 영역 처리 실패 (%s): %s", cropped.region_id, outcome
                )
                if self.cfg.monitor is not None:
                    self.cfg.monitor.record_failure(
                        f"structured_extractor:{type(outcome).__name__}"
                    )
                if trace is not None:
                    trace.append({
                        "region_id": cropped.region_id,
                        "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                        "pixel_budget": cropped.pixel_budget,
                        "elapsed_ms": round(region_ms, 1),
                        "status": "error",
                        "error": f"{type(outcome).__name__}: {outcome}",
                    })
                continue

            if outcome is None:
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

            fields, tables, codes, raw = outcome
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

        # ─── 유령 필드 필터 (x-assembly-rules 있는 서식 전용) ───
        # field_key 없는 region(header/figure 등)에 full schema가 전달되면 VLM이
        # 스키마 외 필드(date/author/unit_code/fields/tables 등)까지 생성해 fields[]를
        # 오염시킴. 여기서 x-assembly-rules 키 집합 + fixed_values 키만 유지한다.
        if schema and schema.get("x-assembly-rules"):
            valid_keys = set(schema["x-assembly-rules"].keys())
            if fixed_values:
                valid_keys.update(fixed_values.keys())
            filtered = [f for f in all_fields if f.field_key in valid_keys]
            ghost_keys = sorted({
                f.field_key for f in all_fields if f.field_key not in valid_keys
            })
            removed = len(all_fields) - len(filtered)
            if removed:
                warnings.append(
                    f"[StructuredExtractor] x-assembly-rules 외 field_key {removed}개 제거: {ghost_keys}"
                )
                logger.info(
                    "[P3-B][%s] 유령 필드 %d개 제거 → %d fields 유지",
                    doc_id, removed, len(filtered),
                )
            all_fields = filtered

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
                fixed_values=fixed_values or {},
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

        # 재시도 대상들을 병렬 처리 (동시성 상한 MAX_CONCURRENT_REQUESTS)
        # 각 target은 독립적인 region이며 all_fields에 대한 동시 쓰기는
        # 서로 다른 인덱스라 단일 이벤트 루프에서 순서 없이도 안전하다.

        @dataclass
        class _RetryPlan:
            region_id: str
            field_indices: list[int]
            retry_cropped: CroppedRegion
            retry_spec: InstructionSpec
            hint: str
            old_budget: int
            new_budget: int

        plans: list[_RetryPlan] = []
        for region_id, field_indices in retry_targets.items():
            cropped = region_map[region_id]
            spec = cropped.instruction_spec
            old_budget = cropped.pixel_budget
            new_budget = RETRY_BUDGET_MAP.get(old_budget, old_budget)

            hint = ""
            if self._ocr_hint_provider is not None and self._ocr_hint_provider.enabled:
                try:
                    hint = self._ocr_hint_provider.get_hint(cropped.cropped_image)
                except Exception as e:
                    logger.warning(
                        "[P3-B] OCR 힌트 생성 실패 (%s): %s", region_id, e
                    )

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
            plans.append(_RetryPlan(
                region_id=region_id,
                field_indices=field_indices,
                retry_cropped=retry_cropped,
                retry_spec=retry_spec,
                hint=hint,
                old_budget=old_budget,
                new_budget=new_budget,
            ))

        if not plans:
            return 0

        async def _run_one(plan: "_RetryPlan"):
            t0 = time.time()
            try:
                res = await asyncio.to_thread(
                    self._process_single, plan.retry_cropped, schema, warnings
                )
                return plan, (time.time() - t0) * 1000, res
            except Exception as e:
                return plan, (time.time() - t0) * 1000, e

        async def _run_all():
            sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            async def _wrap(p):
                async with sem:
                    return await _run_one(p)
            return await asyncio.gather(*[_wrap(p) for p in plans])

        try:
            gathered = asyncio.run(_run_all())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                gathered = loop.run_until_complete(_run_all())
            finally:
                loop.close()

        retry_count = 0
        for plan, region_ms, outcome in gathered:
            if isinstance(outcome, Exception):
                warnings.append(
                    f"Region {plan.region_id} retry failed: {outcome}"
                )
                logger.warning(
                    "[P3-B] 재시도 실패 (%s): %s", plan.region_id, outcome
                )
                if trace is not None:
                    trace.append({
                        "region_id": plan.region_id,
                        "pixel_budget": plan.new_budget,
                        "elapsed_ms": round(region_ms, 1),
                        "status": "retry_error",
                        "is_retry": True,
                        "error": f"{type(outcome).__name__}: {outcome}",
                    })
                continue

            retry_count += 1

            cropped = plan.retry_cropped
            if outcome is None:
                if trace is not None:
                    trace.append({
                        "region_id": plan.region_id,
                        "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                        "pixel_budget": plan.new_budget,
                        "elapsed_ms": round(region_ms, 1),
                        "status": "retry_skipped",
                        "is_retry": True,
                        "ocr_hint_used": bool(plan.hint),
                    })
                continue

            new_fields, _new_tables, _new_codes, new_raw = outcome

            new_field_map = {f.field_key: f for f in new_fields}
            for idx in plan.field_indices:
                old_field = all_fields[idx]
                new_field = new_field_map.get(old_field.field_key)
                if new_field is not None and new_field.confidence > old_field.confidence:
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
                    "region_id": plan.region_id,
                    "region_type": cropped.region_type.value if hasattr(cropped.region_type, "value") else str(cropped.region_type),
                    "pixel_budget": plan.new_budget,
                    "elapsed_ms": round(region_ms, 1),
                    "status": "retry_ok",
                    "is_retry": True,
                    "ocr_hint_used": bool(plan.hint),
                    "ocr_hint_chars": len(plan.hint),
                    "field_count": len(new_fields),
                    "instruction": plan.retry_spec.user_instruction,
                    "raw_response": new_raw,
                })

        return retry_count

    # ─────────────────────────────────────────────
    #  S2/S3 공통 도메인 서비스 위임
    # ─────────────────────────────────────────────

    def _get_s2_s3(self, skill_name: str):
        """S2/S3 지연 생성. Orchestrator가 주입하지 않은 경우 자체 생성."""
        if skill_name == "S2":
            if self._s2 is None:
                from src.vlm.skills.printed_text_reader import PrintedTextReader
                self._s2 = PrintedTextReader(vlm_client=self._vlm_client)
            return self._s2
        if skill_name == "S3":
            if self._s3 is None:
                from src.vlm.skills.handwriting_reader import HandwritingReader
                self._s3 = HandwritingReader(vlm_client=self._vlm_client)
            return self._s3
        return None

    def _delegate_to_skill(
        self,
        cropped: CroppedRegion,
        spec: InstructionSpec,
        skill,
        skill_name: str,
        form_type: Optional[FormType],
        schema: Optional[dict] = None,
    ) -> tuple[list[FieldValue], list[RecognizedTable], list[DomainCode], str]:
        """S2/S3에 위임 실행 후 SkillResult → FieldValue 변환.

        구조화 응답은 SkillResult.content가 JSON 문자열이다. field_key가 있고
        schema에 x-assembly-rules가 정의된 경우 여기서 sub-schema 래퍼를 벗겨
        corrected_value에는 순수 값만 저장한다 (raw_value는 디버깅용 원본 유지).
        """
        spec_field_key = getattr(spec, "field_key", None) if spec else None
        spec_schema = getattr(spec, "json_schema", None) if spec else None
        task = SkillTask(
            region_id=cropped.region_id,
            region_type=cropped.region_type,
            cropped_image=cropped.cropped_image,
            pixel_budget=cropped.pixel_budget,
            context=spec.user_instruction if spec else "",
            form_type=form_type,
            field_key=spec_field_key,
            json_schema=spec_schema if skill_name == "S3" else None,
        )
        result: SkillResult = skill.run(task)
        field_key = spec_field_key or f"{skill_name.lower()}_{cropped.region_id}"

        raw_content = result.content
        corrected_content = raw_content
        # field_key + x-assembly-rules path 기반 blob 언래핑
        if spec_field_key and raw_content and isinstance(schema, dict):
            rules = schema.get("x-assembly-rules")
            if isinstance(rules, dict):
                path = rules.get(spec_field_key)
                try:
                    parsed = json.loads(raw_content)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if parsed is not None:
                    from src.vlm.assembler import Assembler
                    unwrapped = Assembler._unwrap_blob(
                        parsed, field_key=spec_field_key, path=path,
                    )
                    corrected_content = (
                        json.dumps(unwrapped, ensure_ascii=False)
                        if isinstance(unwrapped, (list, dict))
                        else ("" if unwrapped is None else str(unwrapped))
                    )

        field = FieldValue(
            field_key=field_key,
            raw_value=raw_content,
            corrected_value=corrected_content,
            data_type="text",
            confidence=result.confidence,
            token_logprobs=[],
            is_flagged=(result.confidence < 0.70 or not result.content),
            region_id=cropped.region_id,
        )
        return [field], [], [], raw_content

    def _process_single(
        self,
        cropped: CroppedRegion,
        schema: Optional[dict],
        warnings: list[str],
        form_type: Optional[FormType] = None,
    ) -> Optional[tuple[list[FieldValue], list[RecognizedTable], list[DomainCode], str]]:
        """단일 CroppedRegion 처리.

        region_type 기반 위임:
          - HANDWRITTEN_FIELD → S3 HandwritingReader (모든 form_type)
          - TEXT/HEADER/FOOTER + form_type=UNKNOWN → S2 PrintedTextReader
          - 나머지 → 기존 guided_json 경로

        Returns:
            (fields, tables, domain_codes, raw_json) 또는 None
        """
        from src.vlm.vlm_client import encode_image_base64, extract_field_logprobs
        from src.vlm.logprobs_scorer import calc_field_confidence, is_flagged

        spec = cropped.instruction_spec
        region_type_enum = cropped.region_type
        region_type = (
            region_type_enum.value
            if hasattr(region_type_enum, "value")
            else str(region_type_enum)
        )

        # ─── x-assembly-rules 서식에서 field_key 없는 region 스킵 ───
        # 필드 단위 크롭 서식(bid_application 등)은 모든 유효 region이 TemplateAugmentor를
        # 통해 field_key를 부여받는다. 여전히 field_key가 없는 region은 PP-DocLayout이
        # 전체 서식을 통째로 잡은 full-page bbox(header/figure/table)이므로 처리해도
        # 의미 있는 값이 나오지 않고, full schema 전달 시 VLM이 스키마 외 필드까지
        # 생성해 fields[]를 오염시킨다.
        template_field_key = getattr(spec, "field_key", None)
        if (
            schema
            and isinstance(schema.get("x-assembly-rules"), dict)
            and not template_field_key
        ):
            logger.info(
                "[P3-B] 스킵 — field_key 없는 region (%s/%s)",
                cropped.region_id, region_type,
            )
            return None

        # ─── S3 위임: handwritten_field (모든 form_type) ───
        if region_type_enum == RegionType.HANDWRITTEN_FIELD:
            return self._delegate_to_skill(
                cropped, spec, self._get_s2_s3("S3"), "S3", form_type,
                schema=schema,
            )

        # ─── S2 위임: text/header/footer + unknown form_type ───
        if (
            region_type_enum in (RegionType.TEXT, RegionType.HEADER, RegionType.FOOTER)
            and form_type == FormType.UNKNOWN
        ):
            return self._delegate_to_skill(
                cropped, spec, self._get_s2_s3("S2"), "S2", form_type,
                schema=schema,
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

        template_field_key = getattr(spec, "field_key", None)

        # 표 영역 처리 — field_key가 부여된 table region(예: seal_verification_block)은
        # HTML 파서 경로 대신 guided_json blob 경로로 흘려 FieldValue를 생성해야
        # Assembler가 field_map에서 찾아 assembled_json에 연결할 수 있다.
        if region_type == "table" and not template_field_key:
            tables = self._process_table(cropped.region_id, text, logprobs, warnings)
            return [], tables, [], text

        # 필드 추출 (guided_json이 있는 경우)
        fields: list[FieldValue] = []
        domain_codes: list[DomainCode] = []

        parsed = _safe_json_loads(text) if text else None

        # sub-schema 분해 경로에서는 top-level이 dict가 아닐 수 있음 (예: document_date는 문자열 리터럴).
        # dict 파서가 실패해도 template_field_key가 있으면 _loads_any로 재시도하여
        # 스칼라/리스트도 FieldValue blob으로 보존한다.
        if parsed is None and template_field_key and text:
            alt = _loads_any(text)
            if alt is not None:
                parsed = alt
            else:
                # JSON 파싱 완전 실패 — 원문 문자열을 그대로 사용 (Assembler가 검토 큐로 전달)
                parsed = text.strip()

        if parsed is None:
            warnings.append(f"Region {cropped.region_id}: JSON parse failed")
            return [], [], [], text

        # region에 template field_key가 지정된 경우:
        # sub-schema 래퍼를 여기서 해제해 corrected_value에 순수 값만 보존.
        # (raw_value는 VLM 원본 JSON 그대로 — 디버깅/trace용)
        if template_field_key:
            from src.vlm.assembler import Assembler
            raw_str = (
                json.dumps(parsed, ensure_ascii=False)
                if isinstance(parsed, (list, dict))
                else str(parsed)
            )
            # x-assembly-rules에서 path를 얻어 path-tail 래퍼도 제거
            path = None
            if schema and isinstance(schema.get("x-assembly-rules"), dict):
                path = schema["x-assembly-rules"].get(template_field_key)
            unwrapped = Assembler._unwrap_blob(
                parsed, field_key=template_field_key, path=path,
            )
            corrected_str = (
                json.dumps(unwrapped, ensure_ascii=False)
                if isinstance(unwrapped, (list, dict))
                else ("" if unwrapped is None else str(unwrapped))
            )
            # logprobs 전체 평균으로 신뢰도 산출
            all_lps = [
                lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
                for lp in (logprobs or [])
            ]
            confidence = calc_field_confidence(all_lps, "text")
            fields.append(FieldValue(
                field_key=template_field_key,
                raw_value=raw_str,
                corrected_value=corrected_str,
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
        """VLMClient에 위임하는 vLLM 호출 래퍼.

        Returns:
            {"text": str, "logprobs": list[dict], "finish_reason": str}
        """
        resp = self._vlm_client.call(
            image_b64=image_b64,
            instruction=instruction,
            system_prompt=system_prompt,
            guided_json=guided_json,
            logprobs=True,
            pixel_budget=pixel_budget,
        )
        return {
            "text": resp.get("content", ""),
            "logprobs": resp.get("logprobs") or [],
            "finish_reason": resp.get("finish_reason", "stop"),
        }
