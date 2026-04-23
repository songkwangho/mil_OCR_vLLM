"""VLM 공용 유틸리티 — vLLM HTTP 통신 + 이미지 base64 인코딩 + logprobs 파싱.

책임 경계:
  - 이 모듈: vLLM HTTP 통신, 이미지 base64 인코딩, logprobs 파싱, HTML 표 파싱
  - 이미지 크롭: ResolutionRouter (resolution_router.py) 전담
  - 도메인 코드 감지: StructuredExtractor (structured_extractor.py) 내부

제공 API:
  - encode_image_base64(image_rgb, max_size) — RGB 이미지 → base64 PNG
  - extract_field_logprobs(...)              — VLM logprobs → 필드별 토큰 logprob 매핑
  - parse_html_cells(html)                   — HTML 표 → 셀 목록
  - VLMClient                                — vLLM OpenAI 호환 API 래퍼
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.pipeline.health_monitor import VLMHealthMonitor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  이미지 유틸
# ─────────────────────────────────────────────

def encode_image_base64(image_rgb: np.ndarray, max_size: int = 1120) -> str:
    """RGB 이미지 → base64 PNG 문자열.

    장변이 *max_size* 를 초과하면 비율을 유지하며 축소합니다.

    Args:
        image_rgb: (H, W, 3) RGB numpy 배열.
        max_size: 장변 최대 픽셀 수. 기본 1120.

    Returns:
        base64 인코딩된 PNG 문자열.
    """
    h, w = image_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────────
#  logprobs 파싱 유틸
# ─────────────────────────────────────────────

def extract_field_logprobs(
    response_logprobs: list[dict],
    raw_json: str,
    schema: Optional[dict],
) -> dict[str, tuple[list[float], str]]:
    """vLLM logprobs 응답에서 필드별 토큰 logprob을 추출.

    JSON 출력의 각 필드값에 해당하는 토큰들의 logprob을 매핑합니다.

    TODO(Phase 2 — CLAUDE.md "P4 신뢰도 재산출" 로드맵):
        현재 구현은 JSON 토큰 스트림에서 각 필드값의 시작/끝 위치를 추적하지 않는
        조악한 근사다:
          - array/object 필드: 전체 토큰 logprobs를 그대로 할당 → 필드 간 신뢰도 구분 불가
          - scalar 필드: `logprobs[:est_tokens]`으로 앞부분만 잘라 사용 → 위치 오정렬
        결과적으로 필드별 신뢰도가 유사 값으로 수렴하여 overall_confidence에 수평선이
        생김 (전역지원서_2 variance 측정 20% 결정론 점수의 한 원인). 위치 추적을
        도입한 정식 구현은 Phase 2에서 진행.

    Args:
        response_logprobs: vLLM 응답의 logprobs 리스트.
            각 항목: ``{"token": str, "logprob": float, ...}``
        raw_json: VLM 원본 JSON 응답 문자열.
        schema: JSON Schema (필드 유형 참조용). ``None`` 이면 기본 ``"text"`` 유형 사용.

    Returns:
        ``{field_key: (token_logprobs, data_type)}`` 매핑.
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
    token_logprob_values: list[float] = []
    for lp_entry in response_logprobs:
        if isinstance(lp_entry, dict):
            token_logprob_values.append(lp_entry.get("logprob", 0.0))
        elif isinstance(lp_entry, (int, float)):
            token_logprob_values.append(float(lp_entry))

    if not token_logprob_values:
        return result

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
            sample = (
                token_logprob_values[:est_tokens]
                if len(token_logprob_values) >= est_tokens
                else token_logprob_values
            )
            result[key] = (sample, data_type)

    return result


# ─────────────────────────────────────────────
#  VLMClient — vLLM OpenAI 호환 API 래퍼
# ─────────────────────────────────────────────

class VLMClient:
    """vLLM 서버(OpenAI 호환 API)와의 통신을 캡슐화하는 클라이언트.

    사용 예시::

        client = VLMClient(
            base_url="http://localhost:8000/v1",
            model_name="/models/gemma4/gemma-4-26b-a4b-it/",
        )
        resp = client.call(image_b64, instruction="문서에서 필드를 추출하세요.")
        print(resp["content"])
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_logprobs: int = 5,
        timeout: float = 120.0,
        monitor: "Optional[VLMHealthMonitor]" = None,
    ):
        """VLMClient 초기화.

        Args:
            base_url: vLLM 서버 OpenAI 호환 엔드포인트 (예: ``http://localhost:8000/v1``).
            model_name: vLLM에 등록된 모델 경로/이름.
            max_tokens: 최대 생성 토큰 수.
            temperature: 샘플링 온도. 0.0이면 결정론적 출력.
            top_logprobs: logprobs 활성화 시 상위 후보 토큰 수.
            timeout: API 요청 타임아웃(초).
            monitor: VLMHealthMonitor 인스턴스. 호출 실패 시 record_failure()를 호출.
        """
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.monitor = monitor

        # OpenAI 클라이언트 (지연 초기화)
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 지연 초기화."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key="not-needed",  # vLLM은 API 키 불필요
                timeout=self.timeout,
            )
        return self._client

    def call(
        self,
        image_b64: str,
        instruction: str,
        system_prompt: str = "",
        guided_json: Optional[dict] = None,
        logprobs: bool = True,
        pixel_budget: Optional[int] = None,
    ) -> dict[str, Any]:
        """단일 VLM 호출.

        이미지와 instruction을 vLLM 서버에 전송하고 응답을 반환합니다.

        Args:
            image_b64: base64 인코딩된 이미지 문자열.
            instruction: VLM에 전달할 텍스트 instruction.
            system_prompt: 시스템 프롬프트. 비어 있으면 생략.
            guided_json: JSON Schema (guided decoding용). ``None`` 이면 비활성화.
            logprobs: logprobs 수집 여부. 기본 ``True``.
            pixel_budget: 픽셀 예산. 설정 시 ``mm_processor_kwargs.max_pixels``
                = ``pixel_budget * 28 * 28`` 으로 전달.

        Returns:
            ``{"content": str, "logprobs": list[dict] | None, "finish_reason": str}``

            - ``content``: VLM 생성 텍스트.
            - ``logprobs``: 토큰별 logprob 딕셔너리 리스트 (비활성화 시 ``None``).
            - ``finish_reason``: 종료 사유 (``"stop"``, ``"length"`` 등).
        """
        client = self._get_client()

        # ── 메시지 구성 ──
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
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
        })

        # ── API 호출 파라미터 ──
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # extra_body (vLLM 확장 파라미터)
        extra_body: dict[str, Any] = {}
        if guided_json is not None:
            extra_body["guided_json"] = guided_json
        if logprobs:
            extra_body["logprobs"] = True
            extra_body["top_logprobs"] = self.top_logprobs
        if pixel_budget is not None:
            extra_body["mm_processor_kwargs"] = {
                "max_pixels": pixel_budget * 28 * 28,
            }

        if extra_body:
            kwargs["extra_body"] = extra_body

        # ── API 호출 ──
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            # 타임아웃, 연결 오류 등 graceful 처리
            error_type = type(e).__name__
            logger.error("[VLMClient] API 호출 실패 (%s): %s", error_type, e)
            if self.monitor is not None:
                self.monitor.record_failure(f"vlm_call:{error_type}")
            return {
                "content": "",
                "logprobs": None,
                "finish_reason": f"error:{error_type}",
            }

        # ── 응답 파싱 ──
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        # logprobs 추출
        logprobs_data: Optional[list[dict]] = None
        if logprobs and hasattr(choice, "logprobs") and choice.logprobs:
            content_logprobs = getattr(choice.logprobs, "content", None)
            if content_logprobs:
                logprobs_data = []
                for token_info in content_logprobs:
                    logprobs_data.append({
                        "token": getattr(token_info, "token", ""),
                        "logprob": getattr(token_info, "logprob", 0.0),
                    })

        return {
            "content": content,
            "logprobs": logprobs_data,
            "finish_reason": finish_reason,
        }

    async def call_async(
        self,
        image_b64: str,
        instruction: str,
        system_prompt: str = "",
        guided_json: Optional[dict] = None,
        logprobs: bool = True,
        pixel_budget: Optional[int] = None,
    ) -> dict[str, Any]:
        """비동기 vLLM 호출 — `call()`을 스레드풀에서 실행.

        vLLM HTTP 요청은 I/O 대기가 지배적이므로 GIL이 해제되어 동시 처리 이득이
        크다. 동기 `call()`의 모든 예외 처리·monitor 통합·logprobs 파싱을 그대로
        재사용한다.
        """
        return await asyncio.to_thread(
            self.call,
            image_b64=image_b64,
            instruction=instruction,
            system_prompt=system_prompt,
            guided_json=guided_json,
            logprobs=logprobs,
            pixel_budget=pixel_budget,
        )


# ─────────────────────────────────────────────
#  HTML 표 파싱 유틸
# ─────────────────────────────────────────────

def parse_html_cells(html: str) -> list[dict[str, Any]]:
    """VLM 출력 HTML에서 셀 데이터를 간이 파싱.

    <tr>/<td> 태그 기반으로 row/col/text 추출.
    """
    cells: list[dict[str, Any]] = []

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
