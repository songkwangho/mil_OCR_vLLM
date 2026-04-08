"""P3 — Gemma4 VLM 통합 추론 엔진

vLLM 서버(OpenAI 호환 API)를 통해 Gemma4 26B-A4B를 호출합니다.

핵심 메커니즘:
  - guided_json: 서식별 JSON Schema로 출력 구조 보장
  - logprobs: 토큰별 확률 → 필드 신뢰도 산출
  - 영역별 crop: P2 LayoutResult의 reading_order 순서로 처리

처리 흐름:
  1. 서식 분류 (전체 문서 이미지)
  2. JSON Schema 로드 (form_type에 따라)
  3. 필드 추출 (전체 문서 + guided_json + logprobs)
  4. 표 영역 개별 처리 (crop + table instruction)
  5. VLMResult 조립

Input  : PreprocessedImage, LayoutResult
Output : VLMResult

의존: openai (vLLM OpenAI 호환 클라이언트)
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.interfaces.enums import CodeType, FormType
from src.interfaces.types import (
    DomainCode,
    FieldValue,
    LayoutRegion,
    LayoutResult,
    PreprocessedImage,
    RecognizedTable,
    VLMResult,
)

from .instruction_builder import InstructionBuilder, InstructionTask
from .logprobs_scorer import LogprobsScorer, calc_field_confidence, get_threshold, is_flagged

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────

@dataclass
class Gemma4EngineConfig:
    """Gemma4 VLM 엔진 설정."""

    # vLLM 서버 연결
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "/models/gemma4/gemma-4-26b-a4b-it/"

    # API 호출 파라미터
    max_tokens: int = 4096
    temperature: float = 0.0     # 결정론적 출력
    top_logprobs: int = 5        # 상위 5개 후보 토큰

    # 스키마 디렉토리
    schema_dir: Optional[str] = None

    # 타임아웃 (초)
    request_timeout: float = 120.0

    # 이미지 최대 크기 (VLM 입력 제한)
    max_image_size: int = 1120   # pixels (장변 기준)


# ─────────────────────────────────────────────
#  도메인 코드 패턴 (VLM 출력에서 자동 감지)
# ─────────────────────────────────────────────

_CODE_PATTERNS: list[tuple[str, CodeType]] = [
    (r"\d{4}-\d{2}-\d{3}-\d{4}", CodeType.NSN),           # NSN
    (r"KN-\d{5}-\d{4}", CodeType.K_NSN),                   # K-NSN
    (r"\d{2,4}부대", CodeType.UNIT_CODE),                   # 부대코드
    (r"\d{4}-\d{2}-\d{2}", CodeType.DATE),                  # 날짜
]


# ─────────────────────────────────────────────
#  스키마 로더
# ─────────────────────────────────────────────

class _SchemaLoader:
    """JSON Schema 로드 + 캐시."""

    def __init__(self, schema_dir: str):
        self._dir = Path(schema_dir)
        self._cache: dict[str, dict] = {}

    def load(self, schema_id: str) -> Optional[dict]:
        """스키마 ID → JSON Schema dict.

        Args:
            schema_id: 서식 유형명 (예: "supply_request") 또는 "_fallback"

        Returns:
            JSON Schema dict. 파일 없으면 None.
        """
        if schema_id in self._cache:
            return self._cache[schema_id]

        path = self._dir / f"{schema_id}.json"
        if not path.exists():
            logger.warning("Schema not found: %s", path)
            return None

        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        self._cache[schema_id] = schema
        return schema


# ─────────────────────────────────────────────
#  이미지 유틸
# ─────────────────────────────────────────────

def _encode_image_base64(image_rgb: np.ndarray, max_size: int = 1120) -> str:
    """RGB 이미지 → base64 PNG 문자열.

    장변이 max_size를 초과하면 비례 축소.
    """
    h, w = image_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _crop_region(image_rgb: np.ndarray, region: LayoutRegion) -> np.ndarray:
    """이미지에서 영역 bbox를 크롭."""
    bb = region.bbox
    h, w = image_rgb.shape[:2]
    x1 = max(0, bb.x1)
    y1 = max(0, bb.y1)
    x2 = min(w, bb.x2)
    y2 = min(h, bb.y2)
    return image_rgb[y1:y2, x1:x2]


# ─────────────────────────────────────────────
#  logprobs 파싱 유틸
# ─────────────────────────────────────────────

def _extract_field_logprobs(
    response_logprobs: list[dict],
    raw_json: str,
    schema: Optional[dict],
) -> dict[str, tuple[list[float], str]]:
    """vLLM logprobs 응답에서 필드별 토큰 logprob을 추출.

    JSON 출력의 각 필드값에 해당하는 토큰들의 logprob을 매핑합니다.

    Args:
        response_logprobs: vLLM 응답의 logprobs 리스트
            각 항목: {"token": str, "logprob": float, ...}
        raw_json: VLM 원본 JSON 응답 문자열
        schema: JSON Schema (필드 유형 참조용)

    Returns:
        {field_key: (token_logprobs, data_type)}
    """
    result: dict[str, tuple[list[float], str]] = {}

    if not response_logprobs or not raw_json:
        return result

    # JSON 파싱하여 필드값 추출
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return result

    if not isinstance(parsed, dict):
        return result

    # 스키마에서 필드 유형 정보 추출
    field_types: dict[str, str] = {}
    if schema and "properties" in schema:
        for key, prop in schema["properties"].items():
            json_type = prop.get("type", "string")
            if json_type == "integer":
                field_types[key] = "number"
            elif json_type == "number":
                field_types[key] = "number"
            elif key.endswith("_date") or key == "date":
                field_types[key] = "date"
            elif key in ("nsn", "unit_code", "equipment_id"):
                field_types[key] = "code"
            elif key in ("quantity", "stock_qty", "disposal_qty"):
                field_types[key] = "quantity"
            elif key in ("unit_price", "total", "grand_total"):
                field_types[key] = "amount"
            else:
                field_types[key] = "text"

    # 간소화된 logprobs 매핑: 필드값 문자열로 토큰 매칭
    # (정확한 토큰-필드 매핑은 토큰 오프셋이 필요하지만,
    #  여기서는 필드값 문자열 길이 기반으로 근사)
    token_logprob_values = []
    for lp_entry in response_logprobs:
        if isinstance(lp_entry, dict):
            token_logprob_values.append(lp_entry.get("logprob", 0.0))
        elif isinstance(lp_entry, (int, float)):
            token_logprob_values.append(float(lp_entry))

    if not token_logprob_values:
        return result

    # 전체 토큰의 평균 logprob을 기본값으로 사용
    avg_logprob = sum(token_logprob_values) / len(token_logprob_values)

    for key, value in parsed.items():
        if key in ("form_type",):
            continue
        if isinstance(value, (list, dict)):
            # 배열/객체 필드는 내부 토큰 전체의 평균 사용
            data_type = field_types.get(key, "text")
            result[key] = (token_logprob_values, data_type)
        elif isinstance(value, (str, int, float)):
            data_type = field_types.get(key, "text")
            # 필드값 문자열의 토큰 수 추정 (한국어: 글자당 ~1-2토큰)
            val_str = str(value)
            est_tokens = max(1, len(val_str))
            # 전체 logprobs에서 해당 분량만큼 샘플링
            sample = token_logprob_values[:est_tokens] if len(token_logprob_values) >= est_tokens else token_logprob_values
            result[key] = (sample, data_type)

    return result


# ─────────────────────────────────────────────
#  도메인 코드 자동 감지
# ─────────────────────────────────────────────

def _detect_domain_codes(parsed_json: dict) -> list[DomainCode]:
    """VLM 출력 JSON에서 도메인 코드 패턴을 자동 감지."""
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


# ─────────────────────────────────────────────
#  Gemma4Engine (메인 클래스)
# ─────────────────────────────────────────────

class Gemma4Engine:
    """P3 — Gemma4 VLM 통합 추론 엔진.

    vLLM 서버(OpenAI 호환 API)를 통해 문서 이미지에서
    서식 분류 + 필드 추출 + 표 인식 + 도메인 코드 감지를 수행합니다.

    사용 예시:
        engine = Gemma4Engine(Gemma4EngineConfig(
            vllm_base_url="http://vllm-server:8000/v1",
        ))
        result = engine.process(preprocessed_image, layout_result)
    """

    def __init__(self, config: Optional[Gemma4EngineConfig] = None):
        self.cfg = config or Gemma4EngineConfig()

        # 스키마 로더
        schema_dir = self.cfg.schema_dir
        if schema_dir is None:
            schema_dir = str(
                Path(__file__).resolve().parents[1] / "domain" / "schemas"
            )
        self._schema_loader = _SchemaLoader(schema_dir)

        # instruction builder
        self._instruction_builder = InstructionBuilder()

        # logprobs scorer
        self._scorer = LogprobsScorer()

        # OpenAI 클라이언트 (지연 초기화)
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 지연 초기화."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.cfg.vllm_base_url,
                api_key="not-needed",  # vLLM은 API 키 불필요
                timeout=self.cfg.request_timeout,
            )
        return self._client

    # ═══════════════════════════════════════
    #  메인 처리 흐름
    # ═══════════════════════════════════════

    def process(
        self,
        preprocessed: PreprocessedImage,
        layout: LayoutResult,
    ) -> VLMResult:
        """P1 + P2 출력 → VLMResult.

        처리 순서:
          1. 서식 분류
          2. JSON Schema 로드
          3. 필드 추출 (guided_json + logprobs)
          4. 표 영역 처리
          5. 도메인 코드 감지
          6. VLMResult 조립

        Args:
            preprocessed: P1 출력 (전처리 완료 이미지)
            layout: P2 출력 (영역 + 읽기 순서)

        Returns:
            VLMResult
        """
        t0 = time.time()
        warnings: list[str] = []

        image_rgb = preprocessed.image_array
        doc_id = preprocessed.doc_id

        # ── 1. 서식 분류 ───
        form_type_str, form_confidence = self._classify(image_rgb, warnings)
        try:
            form_type = FormType(form_type_str)
        except ValueError:
            form_type = FormType.UNKNOWN
            warnings.append(f"Unknown form_type: {form_type_str} → UNKNOWN")

        schema_id = form_type_str if form_type != FormType.UNKNOWN else "_fallback"

        logger.info("[P3][%s] 서식 분류: %s (conf=%.4f)",
                    doc_id, form_type_str, form_confidence)

        # ── 2. 스키마 로드 ───
        schema = self._schema_loader.load(schema_id)
        if schema is None:
            schema = self._schema_loader.load("_fallback")
            schema_id = "_fallback"
            warnings.append(f"Schema not found for {form_type_str}, using _fallback")

        # ── 3. 필드 추출 (guided_json + logprobs) ───
        fields, raw_json = self._extract_fields(
            image_rgb, form_type_str, schema, schema_id, warnings
        )

        # ── 4. 표 영역 처리 ───
        tables = self._process_tables(image_rgb, layout, warnings)

        # ── 5. 도메인 코드 감지 ───
        domain_codes: list[DomainCode] = []
        try:
            parsed = json.loads(raw_json) if raw_json else {}
            domain_codes = _detect_domain_codes(parsed)
        except json.JSONDecodeError:
            pass

        processing_time_ms = (time.time() - t0) * 1000

        logger.info(
            "[P3][%s] 완료: form=%s, fields=%d, tables=%d, codes=%d, %.0fms",
            doc_id, form_type_str, len(fields), len(tables),
            len(domain_codes), processing_time_ms,
        )

        return VLMResult(
            doc_id=doc_id,
            form_type=form_type,
            form_confidence=form_confidence,
            schema_id=schema_id,
            fields=fields,
            tables=tables,
            domain_codes=domain_codes,
            raw_json=raw_json,
            processing_time_ms=round(processing_time_ms, 1),
            warnings=warnings,
        )

    # ═══════════════════════════════════════
    #  서식 분류
    # ═══════════════════════════════════════

    def _classify(
        self,
        image_rgb: np.ndarray,
        warnings: list[str],
    ) -> tuple[str, float]:
        """전체 문서 이미지로 서식 유형 분류.

        Returns:
            (form_type_str, confidence)
        """
        task = self._instruction_builder.build_classification_task()
        b64 = _encode_image_base64(image_rgb, self.cfg.max_image_size)

        try:
            response = self._call_vlm(
                instruction=task.instruction,
                image_b64=b64,
                use_logprobs=True,
            )

            text = response["text"].strip().lower()
            logprobs = response.get("logprobs", [])

            # form_type 파싱
            valid_types = {ft.value for ft in FormType}
            form_type = "unknown"
            for vt in valid_types:
                if vt in text:
                    form_type = vt
                    break

            # 분류 신뢰도
            if logprobs:
                all_lps = [lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
                           for lp in logprobs]
                confidence = calc_field_confidence(all_lps)
            else:
                confidence = 0.5

            return form_type, confidence

        except Exception as e:
            warnings.append(f"Classification failed: {e}")
            logger.error("[P3] 서식 분류 실패: %s", e)
            return "unknown", 0.0

    # ═══════════════════════════════════════
    #  필드 추출
    # ═══════════════════════════════════════

    def _extract_fields(
        self,
        image_rgb: np.ndarray,
        form_type: str,
        schema: Optional[dict],
        schema_id: str,
        warnings: list[str],
    ) -> tuple[list[FieldValue], str]:
        """전체 문서 이미지에서 필드 추출 (guided_json + logprobs).

        Returns:
            (fields, raw_json)
        """
        tasks = self._instruction_builder.build_extraction_tasks(
            # LayoutResult는 여기서 필요 없으므로 더미 사용
            layout=None,  # type: ignore
            form_type=form_type,
        )

        # 첫 번째 task = 전체 문서 필드 추출
        extract_task = tasks[0] if tasks else None
        if extract_task is None:
            return [], ""

        b64 = _encode_image_base64(image_rgb, self.cfg.max_image_size)

        try:
            response = self._call_vlm(
                instruction=extract_task.instruction,
                image_b64=b64,
                guided_json=schema,
                use_logprobs=True,
            )

            raw_json = response["text"].strip()
            logprobs = response.get("logprobs", [])

            # JSON 파싱
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                warnings.append("Field extraction: JSON parse failed")
                return [], raw_json

            if not isinstance(parsed, dict):
                warnings.append("Field extraction: response is not a JSON object")
                return [], raw_json

            # logprobs → 필드별 신뢰도 산출
            field_logprobs = _extract_field_logprobs(logprobs, raw_json, schema)

            # FieldValue 목록 생성
            fields: list[FieldValue] = []
            for key, value in parsed.items():
                if key == "form_type":
                    continue

                val_str = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
                data_type = "text"
                token_lps: list[float] = []

                if key in field_logprobs:
                    token_lps, data_type = field_logprobs[key]

                confidence = calc_field_confidence(token_lps)
                flagged = is_flagged(confidence, data_type)

                fields.append(FieldValue(
                    field_key=key,
                    raw_value=val_str,
                    corrected_value=val_str,  # VLM이 이미 교정
                    data_type=data_type,
                    confidence=confidence,
                    token_logprobs=token_lps,
                    is_flagged=flagged,
                ))

            return fields, raw_json

        except Exception as e:
            warnings.append(f"Field extraction failed: {e}")
            logger.error("[P3] 필드 추출 실패: %s", e)
            return [], ""

    # ═══════════════════════════════════════
    #  표 영역 처리
    # ═══════════════════════════════════════

    def _process_tables(
        self,
        image_rgb: np.ndarray,
        layout: LayoutResult,
        warnings: list[str],
    ) -> list[RecognizedTable]:
        """표 영역 crop → VLM으로 표 구조 인식."""
        from src.interfaces.enums import RegionType

        tables: list[RecognizedTable] = []

        for idx in layout.reading_order:
            if idx >= len(layout.regions):
                continue
            region = layout.regions[idx]
            if region.region_type != RegionType.TABLE:
                continue

            try:
                crop = _crop_region(image_rgb, region)
                if crop.size == 0:
                    continue

                b64 = _encode_image_base64(crop, self.cfg.max_image_size)
                instruction = self._instruction_builder.get_region_instruction(
                    RegionType.TABLE
                )

                response = self._call_vlm(
                    instruction=instruction,
                    image_b64=b64,
                    use_logprobs=True,
                )

                html = response["text"].strip()
                logprobs = response.get("logprobs", [])

                # 표 신뢰도
                if logprobs:
                    all_lps = [lp.get("logprob", 0.0) if isinstance(lp, dict) else float(lp)
                               for lp in logprobs]
                    confidence = calc_field_confidence(all_lps)
                else:
                    confidence = 0.5

                # 간이 셀 파싱 (HTML에서 셀 추출)
                cells = _parse_html_cells(html)

                tables.append(RecognizedTable(
                    region_id=region.region_id,
                    html=html,
                    cells=cells,
                    confidence=confidence,
                ))

            except Exception as e:
                warnings.append(f"Table {region.region_id} failed: {e}")
                logger.warning("[P3] 표 처리 실패 (%s): %s", region.region_id, e)

        return tables

    # ═══════════════════════════════════════
    #  vLLM API 호출
    # ═══════════════════════════════════════

    def _call_vlm(
        self,
        instruction: str,
        image_b64: str,
        guided_json: Optional[dict] = None,
        use_logprobs: bool = False,
    ) -> dict[str, Any]:
        """vLLM OpenAI 호환 API 호출.

        Args:
            instruction: VLM instruction 텍스트
            image_b64: base64 인코딩된 이미지
            guided_json: JSON Schema (guided decoding용)
            use_logprobs: logprobs 수집 여부

        Returns:
            {"text": str, "logprobs": list[dict]}
        """
        client = self._get_client()

        # 메시지 구성 (이미지 + 텍스트)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ]

        # API 호출 파라미터
        kwargs: dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }

        # extra_body (vLLM 확장 파라미터)
        extra_body: dict[str, Any] = {}
        if guided_json is not None:
            extra_body["guided_json"] = guided_json
        if use_logprobs:
            extra_body["logprobs"] = True
            extra_body["top_logprobs"] = self.cfg.top_logprobs

        if extra_body:
            kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**kwargs)

        # 응답 파싱
        choice = response.choices[0]
        text = choice.message.content or ""

        # logprobs 추출
        logprobs_data: list[dict] = []
        if use_logprobs and hasattr(choice, "logprobs") and choice.logprobs:
            content_logprobs = getattr(choice.logprobs, "content", None)
            if content_logprobs:
                for token_info in content_logprobs:
                    logprobs_data.append({
                        "token": getattr(token_info, "token", ""),
                        "logprob": getattr(token_info, "logprob", 0.0),
                    })

        return {"text": text, "logprobs": logprobs_data}


# ─────────────────────────────────────────────
#  HTML 표 파싱 유틸
# ─────────────────────────────────────────────

def _parse_html_cells(html: str) -> list[dict[str, Any]]:
    """VLM 출력 HTML에서 셀 데이터를 간이 파싱.

    <tr>/<td> 태그 기반으로 row/col/text 추출.
    """
    cells: list[dict[str, Any]] = []

    # 간이 정규식 파싱 (lxml 없이)
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
    tag_strip = re.compile(r"<[^>]+>")

    for row_idx, row_match in enumerate(row_pattern.finditer(html)):
        row_html = row_match.group(1)
        for col_idx, cell_match in enumerate(cell_pattern.finditer(row_html)):
            cell_text = tag_strip.sub("", cell_match.group(1)).strip()
            cells.append({
                "row": row_idx,
                "col": col_idx,
                "text": cell_text,
            })

    return cells
