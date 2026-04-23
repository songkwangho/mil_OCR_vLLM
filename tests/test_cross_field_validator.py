"""Layer 2 CrossFieldValidator 단위 테스트.

configs/validation_rules/{form_type}.yaml 규칙이 정확히 ValidationError로
변환되는지 확인한다. 9개 규칙 타입 전부 커버.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import FormType
from src.postprocess.cross_field_validator import CrossFieldValidator


@pytest.fixture
def validator():
    return CrossFieldValidator()


# ─────────────────────────────────────────────
#  bid_application 규칙
# ─────────────────────────────────────────────

class TestBidApplicationRules:
    def test_regex_match_pass(self, validator):
        data = {
            "applicant": {"business_reg_number": "137-63-12345"},
            "submission": {"submission_date": "2024년 05월 15일", "submitter_name": "홍길동"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert not any(e.error_id == "bid_001" for e in errors)

    def test_bid_001_business_reg_format_fail(self, validator):
        data = {
            "applicant": {"business_reg_number": "1376312345"},
            "submission": {"submission_date": "2024-05-15", "submitter_name": "홍길동"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert any(e.error_id == "bid_001" for e in errors)

    def test_bid_002_corporate_reg_format_fail(self, validator):
        data = {
            "applicant": {
                "business_reg_number": "137-63-12345",
                "corporate_reg_number": "INVALID",
            },
            "submission": {"submission_date": "2024-05-15", "submitter_name": "홍길동"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert any(e.error_id == "bid_002" for e in errors)

    def test_bid_003_submission_date_missing(self, validator):
        data = {
            "applicant": {"business_reg_number": "137-63-12345"},
            "submission": {"submitter_name": "홍길동"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert any(e.error_id == "bid_003" for e in errors)

    def test_bid_004_submitter_name_whitespace(self, validator):
        data = {
            "applicant": {"business_reg_number": "137-63-12345"},
            "submission": {"submission_date": "2024-05-15", "submitter_name": "  "},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert any(e.error_id == "bid_004" for e in errors)

    def test_bid_date_order_violation(self, validator):
        """인감 발급일(2024.06.01) > 신청일(2024.05.15) 위반."""
        data = {
            "applicant": {"business_reg_number": "137-63-12345"},
            "submission": {"submission_date": "2024년 05월 15일", "submitter_name": "홍길동"},
            "seal_verification": {"issue_date": "2024년 06월 01일"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert any(e.error_id == "bid_date_001" for e in errors)

    def test_bid_name_match_pass(self, validator):
        """대표자 성명 == 신청인 성명."""
        data = {
            "applicant": {
                "business_reg_number": "137-63-12345",
                "representative": "홍길동",
            },
            "submission": {"submission_date": "2024-05-15", "submitter_name": "홍길동"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert not any(e.error_id == "bid_person_001" for e in errors)

    def test_bid_name_match_fail(self, validator):
        data = {
            "applicant": {
                "business_reg_number": "137-63-12345",
                "representative": "김철수",
            },
            "submission": {"submission_date": "2024-05-15", "submitter_name": "박지훈"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert any(e.error_id == "bid_person_001" for e in errors)

    def test_bid_addr_contains_pass(self, validator):
        data = {
            "applicant": {
                "business_reg_number": "137-63-12345",
                "address": "서울특별시 강남구 테헤란로 123",
                "business_location": "서울특별시 강남구",
            },
            "submission": {"submission_date": "2024-05-15", "submitter_name": "홍길동"},
        }
        errors = validator.validate(data, FormType.BID_APPLICATION)
        assert not any(e.error_id == "bid_addr_001" for e in errors)


# ─────────────────────────────────────────────
#  equipment_checklist 규칙 (ARRAY_*)
# ─────────────────────────────────────────────

class TestEquipmentChecklistRules:
    def _valid(self):
        return {
            "checklist_items": [
                {"item_number": 1, "result": "O"},
                {"item_number": 2, "result": "O"},
                {"item_number": 3, "result": "O"},
                {"item_number": 4, "result": "X"},
                {"item_number": 5, "result": "X"},
                {"item_number": 6, "result": "O"},
            ],
            "writer": {"name": "김정식"},
        }

    def test_all_pass(self, validator):
        errors = validator.validate(self._valid(), FormType.EQUIPMENT_CHECKLIST)
        assert errors == []

    def test_chk_001_length_mismatch(self, validator):
        data = self._valid()
        data["checklist_items"] = data["checklist_items"][:5]
        errors = validator.validate(data, FormType.EQUIPMENT_CHECKLIST)
        assert any(e.error_id == "chk_001" for e in errors)

    def test_chk_002_sequence_mismatch(self, validator):
        data = self._valid()
        data["checklist_items"][2]["item_number"] = 99
        errors = validator.validate(data, FormType.EQUIPMENT_CHECKLIST)
        assert any(e.error_id == "chk_002" for e in errors)

    def test_chk_003_invalid_result(self, validator):
        data = self._valid()
        data["checklist_items"][0]["result"] = "△"
        errors = validator.validate(data, FormType.EQUIPMENT_CHECKLIST)
        assert any(e.error_id == "chk_003" for e in errors)

    def test_chk_004_missing_writer(self, validator):
        data = self._valid()
        data["writer"]["name"] = ""
        errors = validator.validate(data, FormType.EQUIPMENT_CHECKLIST)
        assert any(e.error_id == "chk_004" for e in errors)


# ─────────────────────────────────────────────
#  supply_request ARITHMETIC 규칙
# ─────────────────────────────────────────────

class TestArithmeticRule:
    def test_arith_001_pass(self, validator):
        data = {
            "unit_code": "123",
            "quantity": 10,
            "unit_price": 500,
            "total": 5000,
            "nsn": "1005-01-432-1234",
            "k_nsn": "KN-12345-0001",
        }
        errors = validator.validate(data, FormType.SUPPLY_REQUEST)
        assert not any(e.error_id == "arith_001" for e in errors)

    def test_arith_001_fail(self, validator):
        data = {
            "unit_code": "123",
            "quantity": 10,
            "unit_price": 500,
            "total": 9999,
            "nsn": "1005-01-432-1234",
            "k_nsn": "KN-12345-0001",
        }
        errors = validator.validate(data, FormType.SUPPLY_REQUEST)
        assert any(e.error_id == "arith_001" for e in errors)

    def test_miss_001_unit_code_missing(self, validator):
        data = {
            "quantity": 10,
            "unit_price": 500,
            "total": 5000,
            "nsn": "1005-01-432-1234",
            "k_nsn": "KN-12345-0001",
        }
        errors = validator.validate(data, FormType.SUPPLY_REQUEST)
        assert any(e.error_id == "miss_001" for e in errors)


# ─────────────────────────────────────────────
#  No rules = No errors
# ─────────────────────────────────────────────

class TestNoRules:
    def test_form_without_yaml_returns_empty(self, validator):
        # handover_doc.yaml은 구현되지 않음 → 빈 결과
        errors = validator.validate({"foo": "bar"}, FormType.HANDOVER_DOC)
        assert errors == []

    def test_non_dict_input_returns_empty(self, validator):
        errors = validator.validate("not a dict", FormType.BID_APPLICATION)  # type: ignore[arg-type]
        assert errors == []
