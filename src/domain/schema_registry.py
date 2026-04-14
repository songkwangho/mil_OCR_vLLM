"""Schema Registry -- 스키마 버전 관리

군수 서식별 JSON Schema를 로드하고 버전/폐기 상태를 관리합니다.

사용법:
    registry = SchemaRegistry()
    schema = registry.load("supply_request")
    version = registry.get_version("supply_request")
    all_schemas = registry.list_schemas()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
_DEFAULT_VERSION = "v1"

# form_type 별칭 → schema_id
_SCHEMA_ALIASES: dict[str, str] = {
    "other": "official_document",
    "unknown": "_fallback",
}


class SchemaRegistry:
    """JSON Schema 레지스트리.

    schemas/ 디렉토리에서 스키마를 로드하고,
    x-mil-ocr-version / x-mil-ocr-deprecated 메타데이터를 기반으로
    활성(non-deprecated) 스키마만 반환합니다.

    향후 동일 form_type의 복수 버전을 지원할 수 있도록
    내부적으로 (schema_id, version) 키로 관리합니다.
    load()는 기본적으로 최신 활성 버전을 반환합니다.
    """

    def __init__(self, schema_dir: Optional[str] = None):
        self._dir = Path(schema_dir) if schema_dir else _DEFAULT_SCHEMA_DIR
        # {schema_id: [{schema_dict, ...}, ...]}  -- 버전 목록 (로드 순)
        self._schemas: dict[str, list[dict]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """스키마 디렉토리에서 모든 JSON 파일을 지연 로드."""
        if self._loaded:
            return

        if not self._dir.is_dir():
            logger.warning("SchemaRegistry: 디렉토리 없음 — %s", self._dir)
            self._loaded = True
            return

        # schemas/ 최상위 + schemas/v1/, schemas/v2/... 모두 스캔
        scan_dirs = [self._dir]
        for sub in sorted(self._dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("v"):
                scan_dirs.append(sub)

        seen: set[tuple[str, str]] = set()
        for d in scan_dirs:
            for path in sorted(d.glob("*.json")):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        schema = json.load(f)

                    schema_id = schema.get("$id", path.stem)
                    version = schema.get("x-mil-ocr-version", "0.0.0")
                    key = (schema_id, version)
                    if key in seen:
                        continue
                    seen.add(key)
                    self._schemas.setdefault(schema_id, []).append(schema)

                except Exception as e:
                    logger.warning("SchemaRegistry: 스키마 로드 실패 (%s): %s", path.name, e)

        self._loaded = True
        logger.info(
            "SchemaRegistry: %d개 스키마 로드 완료 (%s)",
            sum(len(v) for v in self._schemas.values()),
            self._dir,
        )

    def _latest_active(self, schema_id: str) -> Optional[dict]:
        """schema_id에 대한 최신 활성(non-deprecated) 스키마 반환."""
        self._ensure_loaded()
        versions = self._schemas.get(schema_id)
        if not versions:
            return None

        # 역순으로 탐색하여 첫 번째 non-deprecated 반환
        for schema in reversed(versions):
            if not schema.get("x-mil-ocr-deprecated", False):
                return schema

        return None

    def get(self, form_type_or_id: str, version: Optional[str] = None) -> Optional[dict]:
        """form_type 또는 schema_id(+version)로 스키마 조회.

        `form_type:version` 형식(예: ``"supply_request:v1"``)도 지원합니다.
        ``"other"`` → ``"official_document"``, ``"unknown"`` → ``"_fallback"``
        별칭이 적용됩니다.

        Args:
            form_type_or_id: 서식명 / schema_id / 별칭, 또는 ``"id:version"`` 형식.
            version: ``"1.0.0"`` 또는 ``"v1"`` 등 특정 버전. None이면 최신.

        Returns:
            스키마 dict. 해당 id/version이 없으면 None.
        """
        # "form_type:version" 분리
        raw = form_type_or_id
        if ":" in raw and version is None:
            raw, version = raw.split(":", 1)

        schema_id = _SCHEMA_ALIASES.get(raw, raw)

        self._ensure_loaded()
        versions = self._schemas.get(schema_id)
        if not versions:
            return None

        if version is None:
            return self._latest_active(schema_id)

        # version 매칭 — "v1" → "1.0.0" prefix, 아니면 정확 매칭
        target = version.lstrip("v")
        for schema in reversed(versions):
            ver = schema.get("x-mil-ocr-version", "0.0.0")
            if ver == version or ver.startswith(target):
                if not schema.get("x-mil-ocr-deprecated", False):
                    return schema
        return None

    def load(self, schema_id: str) -> Optional[dict]:
        """활성(non-deprecated) 스키마를 반환.

        Args:
            schema_id: 서식 유형명 (예: "supply_request") 또는 "_fallback"

        Returns:
            JSON Schema dict. 없거나 deprecated이면 None.
        """
        return self._latest_active(schema_id)

    def get_version(self, schema_id: str) -> str:
        """schema_id의 최신 활성 버전 문자열 반환.

        Returns:
            버전 문자열 (예: "1.0.0"). 스키마가 없으면 "0.0.0".
        """
        schema = self._latest_active(schema_id)
        if schema is None:
            return "0.0.0"
        return schema.get("x-mil-ocr-version", "0.0.0")

    def list_schemas(self) -> list[dict]:
        """모든 스키마의 id/version/deprecated 상태를 반환.

        Returns:
            [{"id": str, "version": str, "deprecated": bool}, ...]
        """
        self._ensure_loaded()
        result: list[dict] = []

        for schema_id, versions in self._schemas.items():
            for schema in versions:
                result.append({
                    "id": schema.get("$id", schema_id),
                    "version": schema.get("x-mil-ocr-version", "0.0.0"),
                    "deprecated": schema.get("x-mil-ocr-deprecated", False),
                })

        return result
