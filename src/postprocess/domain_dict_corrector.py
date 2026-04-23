"""Layer 3 — 도메인 사전 기반 후처리 교정 엔진.

configs/domain_dict/*.yaml을 로드하여 VLM 인식 결과를 폐쇄집합에 맞게 교정한다.
Layer 1(형식 패턴) + Layer 2(교차 검증) 이후에 실행되며, form_type 무관.

교정 대상:
  1. applicant.business_location 별칭 정규화 ("서울시" → "서울특별시")
  2. address ↔ business_location 일관성 (address의 시/도로 덮어쓰기)
  3. bid_info.industry_class_number (유효값 Levenshtein 최근접)
  4. seal_verification.issuing_office (폐쇄집합 Levenshtein 최근접)

설계 원칙:
  - 모든 사전은 YAML에 선언. Python 코드에 도메인 상수 하드코딩 없음.
  - 교정 내역은 로그 리스트로 반환 → warnings에 누적되어 감사 추적 가능.
  - 사전이 비어 있으면 통과 (실패 없음).
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Levenshtein 거리 임계값 — 이 이하면 교정 후보로 간주. 길이 비례 적용은
# 코드/지명 길이가 짧아(2~10자) 고정 상수면 충분.
LEVENSHTEIN_THRESHOLD: int = 2


def _default_dict_dir() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent.parent, Path.cwd()):
        p = candidate / "configs" / "domain_dict"
        if p.exists():
            return p
    return here.parent.parent.parent / "configs" / "domain_dict"


class DomainDictCorrector:
    """Layer 3 엔진 — 폐쇄집합 기반 정규화/교정."""

    def __init__(self, dict_dir: Optional[Path] = None) -> None:
        self._dir = dict_dir or _default_dict_dir()
        self._address = self._load("address_regions.yaml")
        self._industry = self._load("industry_codes.yaml")
        self._offices = self._load("issuing_offices.yaml")
        logger.info(
            "[DomainDict] 사전 로드: aliases=%d, hierarchy=%d, "
            "industry_codes=%d, seal_offices=%d (%s)",
            len(self._address.get("aliases", {})),
            len(self._address.get("hierarchy", {})),
            len(self._industry.get("valid_codes", [])),
            len(self._offices.get("seal_certificate_offices", [])),
            self._dir,
        )

    # ── 로더 ────────────────────────────────────────────────

    def _load(self, filename: str) -> dict:
        path = self._dir / filename
        if not path.exists():
            return {}
        try:
            import yaml  # type: ignore
        except ImportError:
            logger.warning("[DomainDict] PyYAML 미설치 — %s 비활성", filename)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("[DomainDict] %s 로드 실패: %s", path, e)
            return {}

    # ── 외부 API ───────────────────────────────────────────

    def correct(self, assembled_json: dict) -> tuple[dict, list[str]]:
        """assembled_json을 교정해 `(교정 dict, 로그 목록)` 반환.

        입력 dict는 변경하지 않고 deep copy된 결과 반환.
        """
        if not isinstance(assembled_json, dict):
            return assembled_json, []

        data = copy.deepcopy(assembled_json)
        logs: list[str] = []

        data, log = self._normalize_business_location(data)
        logs.extend(log)

        data, log = self._correct_address_consistency(data)
        logs.extend(log)

        data, log = self._correct_industry_code(data)
        logs.extend(log)

        data, log = self._correct_issuing_office(data)
        logs.extend(log)

        return data, logs

    # ── 교정 로직 ──────────────────────────────────────────

    def _normalize_business_location(
        self, data: dict
    ) -> tuple[dict, list[str]]:
        """alias → canonical 정규화."""
        logs: list[str] = []
        aliases = self._address.get("aliases") or {}
        if not aliases:
            return data, logs

        value = self._get(data, "applicant.business_location")
        if not value or not isinstance(value, str):
            return data, logs

        stripped = value.strip()
        canonical = aliases.get(stripped)
        if canonical and canonical != stripped:
            self._set(data, "applicant.business_location", canonical)
            logs.append(
                f"[DomainDict] business_location 정규화: "
                f"'{stripped}' → '{canonical}'"
            )
        return data, logs

    def _correct_address_consistency(
        self, data: dict
    ) -> tuple[dict, list[str]]:
        """address의 시/도를 우선 진실로 간주. business_location 불일치 시 교정."""
        logs: list[str] = []
        address = self._get(data, "applicant.address")
        location = self._get(data, "applicant.business_location")
        if not address or not location or not isinstance(address, str):
            return data, logs

        city_from_address = self._extract_city(address)
        if not city_from_address:
            return data, logs

        aliases = self._address.get("aliases") or {}
        normalized = aliases.get(str(location).strip(), str(location).strip())

        if city_from_address != normalized:
            self._set(data, "applicant.business_location", city_from_address)
            logs.append(
                f"[DomainDict] business_location 교정 (address 기준): "
                f"'{location}' → '{city_from_address}'"
            )
        return data, logs

    def _correct_industry_code(
        self, data: dict
    ) -> tuple[dict, list[str]]:
        logs: list[str] = []
        valid_codes = [
            str(entry["code"])
            for entry in (self._industry.get("valid_codes") or [])
            if isinstance(entry, dict) and "code" in entry
        ]
        if not valid_codes:
            return data, logs

        code = self._get(data, "bid_info.industry_class_number")
        if code in (None, ""):
            return data, logs

        code_str = str(code).strip()
        if code_str in valid_codes:
            return data, logs

        nearest, dist = self._nearest(code_str, valid_codes)
        if nearest is not None and dist <= LEVENSHTEIN_THRESHOLD:
            self._set(data, "bid_info.industry_class_number", nearest)
            logs.append(
                f"[DomainDict] industry_class_number 교정: "
                f"'{code_str}' → '{nearest}' (거리={dist})"
            )
        else:
            logs.append(
                f"[DomainDict] industry_class_number '{code_str}' "
                f"유효값 목록에 없음 (최근접={nearest}, 거리={dist})"
            )
        return data, logs

    def _correct_issuing_office(
        self, data: dict
    ) -> tuple[dict, list[str]]:
        logs: list[str] = []
        candidates = self._offices.get("seal_certificate_offices") or []
        if not candidates:
            return data, logs

        office = self._get(data, "seal_verification.issuing_office")
        if office in (None, ""):
            return data, logs

        office_str = str(office).strip()
        if office_str in candidates:
            return data, logs

        nearest, dist = self._nearest(office_str, candidates)
        if nearest is not None and dist <= LEVENSHTEIN_THRESHOLD:
            self._set(data, "seal_verification.issuing_office", nearest)
            logs.append(
                f"[DomainDict] issuing_office 교정: "
                f"'{office_str}' → '{nearest}' (거리={dist})"
            )
        return data, logs

    # ── 유틸 ────────────────────────────────────────────────

    def _extract_city(self, address: str) -> Optional[str]:
        """주소에서 시/도(canonical) 추출."""
        hierarchy = self._address.get("hierarchy") or {}
        aliases = self._address.get("aliases") or {}

        # 1) canonical 이름으로 시작하는지
        for city in hierarchy:
            if address.startswith(city):
                return city
        # 2) 별칭으로 시작하는지
        for alias, canonical in aliases.items():
            if address.startswith(alias):
                return canonical
        # 3) 패턴 기반 시/도 단위 추출
        m = re.match(
            r"^([가-힣]+?(?:특별시|광역시|특별자치시|특별자치도|도))",
            address,
        )
        if m:
            candidate = m.group(1)
            # canonical 형태면 그대로, 아니면 alias 통한 canonical 조회
            if candidate in hierarchy:
                return candidate
            return aliases.get(candidate, candidate)
        return None

    def _nearest(
        self, value: str, candidates: list[str]
    ) -> tuple[Optional[str], int]:
        if not candidates:
            return None, 999
        best = min(candidates, key=lambda c: self._levenshtein(value, c))
        return best, self._levenshtein(value, best)

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                ))
            prev = curr
        return prev[-1]

    @staticmethod
    def _get(data: Any, path: str) -> Any:
        cur: Any = data
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    @staticmethod
    def _set(data: dict, path: str, value: Any) -> None:
        """path로 in-place 설정 — 중간 dict이 없으면 생성."""
        parts = path.split(".")
        cur = data
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = value
