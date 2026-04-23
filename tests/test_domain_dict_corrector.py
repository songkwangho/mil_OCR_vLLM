"""Layer 3 DomainDictCorrector 단위 테스트.

폐쇄집합 기반 별칭 정규화 + Levenshtein 교정 동작 검증.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess.domain_dict_corrector import (
    DomainDictCorrector,
    LEVENSHTEIN_THRESHOLD,
)


@pytest.fixture
def corrector():
    return DomainDictCorrector()


# ─────────────────────────────────────────────
#  business_location 별칭 정규화
# ─────────────────────────────────────────────

class TestBusinessLocationAliases:
    def test_seoul_alias_normalized(self, corrector):
        data = {"applicant": {"business_location": "서울시"}}
        out, logs = corrector.correct(data)
        assert out["applicant"]["business_location"] == "서울특별시"
        assert any("business_location 정규화" in l for l in logs)

    def test_gyeonggi_alias_normalized(self, corrector):
        data = {"applicant": {"business_location": "경기"}}
        out, logs = corrector.correct(data)
        assert out["applicant"]["business_location"] == "경기도"

    def test_canonical_value_passthrough(self, corrector):
        data = {"applicant": {"business_location": "서울특별시"}}
        out, logs = corrector.correct(data)
        # 이미 canonical이면 정규화 로그 없음
        assert out["applicant"]["business_location"] == "서울특별시"
        assert not any("business_location 정규화" in l for l in logs)

    def test_unknown_value_passthrough(self, corrector):
        data = {"applicant": {"business_location": "알수없음"}}
        out, logs = corrector.correct(data)
        assert out["applicant"]["business_location"] == "알수없음"
        # address가 없어 일관성 교정도 트리거 안 됨
        assert not any("business_location 교정" in l for l in logs)


# ─────────────────────────────────────────────
#  address ↔ business_location 일관성 교정
# ─────────────────────────────────────────────

class TestAddressConsistency:
    def test_mismatch_corrected_to_address_city(self, corrector):
        data = {"applicant": {
            "address": "서울특별시 강남구 테헤란로 123",
            "business_location": "경기도",
        }}
        out, logs = corrector.correct(data)
        assert out["applicant"]["business_location"] == "서울특별시"
        assert any("address 기준" in l for l in logs)

    def test_match_passthrough(self, corrector):
        data = {"applicant": {
            "address": "서울특별시 강남구 테헤란로 123",
            "business_location": "서울특별시",
        }}
        out, logs = corrector.correct(data)
        assert out["applicant"]["business_location"] == "서울특별시"
        assert not any("address 기준" in l for l in logs)

    def test_alias_in_business_location_then_address_match(self, corrector):
        """business_location='서울시'는 alias 정규화로 서울특별시가 되고, address도 서울특별시면 추가 교정 없음."""
        data = {"applicant": {
            "address": "서울특별시 서초구 반포대로 123",
            "business_location": "서울시",
        }}
        out, logs = corrector.correct(data)
        assert out["applicant"]["business_location"] == "서울특별시"
        # 정규화 로그는 있지만 address 교정은 없어야 함
        assert any("정규화" in l for l in logs)
        assert not any("address 기준" in l for l in logs)


# ─────────────────────────────────────────────
#  산업분류번호 Levenshtein 교정
# ─────────────────────────────────────────────

class TestIndustryCode:
    def test_valid_code_passthrough(self, corrector):
        data = {"bid_info": {"industry_class_number": "262"}}
        out, logs = corrector.correct(data)
        assert out["bid_info"]["industry_class_number"] == "262"

    def test_near_code_corrected(self, corrector):
        """'263' vs '262' 1글자 차이 → Levenshtein 1로 교정.
        실제로는 263도 유효값이니 '260' 같은 실제 오류로 테스트."""
        data = {"bid_info": {"industry_class_number": "260"}}
        out, logs = corrector.correct(data)
        # 260 → 거리 1 이내 후보가 있으면 교정
        assert out["bid_info"]["industry_class_number"] in {"262", "263", "264", "265"}
        assert any("industry_class_number 교정" in l for l in logs)

    def test_far_code_flagged_only(self, corrector):
        """거리 3 이상이면 교정 불가 — 로그만 기록."""
        data = {"bid_info": {"industry_class_number": "9999"}}
        out, logs = corrector.correct(data)
        assert out["bid_info"]["industry_class_number"] == "9999"  # 원본 유지
        assert any("유효값 목록에 없음" in l for l in logs)


# ─────────────────────────────────────────────
#  발급관서 Levenshtein 교정
# ─────────────────────────────────────────────

class TestIssuingOffice:
    def test_exact_match_passthrough(self, corrector):
        data = {"seal_verification": {"issuing_office": "서울특별시 강남구"}}
        out, logs = corrector.correct(data)
        assert out["seal_verification"]["issuing_office"] == "서울특별시 강남구"

    def test_close_typo_corrected(self, corrector):
        """'서울특별시 강남' (구 누락) → '서울특별시 강남구' 거리 1."""
        data = {"seal_verification": {"issuing_office": "서울특별시 강남"}}
        out, logs = corrector.correct(data)
        assert out["seal_verification"]["issuing_office"] == "서울특별시 강남구"
        assert any("issuing_office 교정" in l for l in logs)

    def test_far_value_not_corrected(self, corrector):
        data = {"seal_verification": {"issuing_office": "완전히 엉뚱한 값"}}
        out, logs = corrector.correct(data)
        # 거리 임계값 초과 → 교정 없이 원본
        assert out["seal_verification"]["issuing_office"] == "완전히 엉뚱한 값"


# ─────────────────────────────────────────────
#  유틸
# ─────────────────────────────────────────────

class TestExtractCity:
    def test_canonical_prefix(self, corrector):
        assert corrector._extract_city("서울특별시 강남구 테헤란로") == "서울특별시"

    def test_alias_prefix(self, corrector):
        assert corrector._extract_city("서울시 강남구 테헤란로") == "서울특별시"

    def test_gyeonggi(self, corrector):
        assert corrector._extract_city("경기도 성남시 분당구") == "경기도"

    def test_no_city(self, corrector):
        assert corrector._extract_city("그냥 아무 주소") is None


class TestLevenshtein:
    def test_identical(self, corrector):
        assert corrector._levenshtein("abc", "abc") == 0

    def test_one_substitution(self, corrector):
        assert corrector._levenshtein("abc", "abd") == 1

    def test_one_insertion(self, corrector):
        assert corrector._levenshtein("abc", "abcd") == 1

    def test_empty_string(self, corrector):
        assert corrector._levenshtein("", "abc") == 3
        assert corrector._levenshtein("abc", "") == 3


# ─────────────────────────────────────────────
#  입력 dict 불변성
# ─────────────────────────────────────────────

class TestImmutability:
    def test_input_not_mutated(self, corrector):
        data = {"applicant": {"business_location": "서울시"}}
        original = {"applicant": {"business_location": "서울시"}}
        out, _ = corrector.correct(data)
        # 원본 dict는 변경되지 않아야 함
        assert data == original
        # 반환 dict는 새 인스턴스
        assert out is not data

    def test_non_dict_input_returns_as_is(self, corrector):
        out, logs = corrector.correct("not a dict")  # type: ignore[arg-type]
        assert out == "not a dict"
        assert logs == []


# ─────────────────────────────────────────────
#  빈 입력 / 누락 필드
# ─────────────────────────────────────────────

class TestEmptyInput:
    def test_empty_dict(self, corrector):
        out, logs = corrector.correct({})
        assert out == {}
        assert logs == []

    def test_partial_fields(self, corrector):
        """applicant만 있고 bid_info/seal_verification 없어도 정상 처리."""
        data = {"applicant": {"business_location": "서울시"}}
        out, logs = corrector.correct(data)
        assert out["applicant"]["business_location"] == "서울특별시"
