"""Layer 1 FieldPatternValidator 단위 테스트.

configs/validation_rules/common.yaml 기반 data_type / field_key 패턴 검증과
정규화 동작을 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.types import FieldValue
from src.postprocess.field_pattern_validator import FieldPatternValidator


@pytest.fixture
def validator():
    return FieldPatternValidator()


def _make_field(
    field_key: str, value: str, data_type: str = "text", confidence: float = 0.9
) -> FieldValue:
    return FieldValue(
        field_key=field_key,
        raw_value=value,
        corrected_value=value,
        data_type=data_type,
        confidence=confidence,
        token_logprobs=[],
        is_flagged=False,
        region_id=None,
    )


class TestFieldKeyPatterns:
    def test_corporate_reg_number_valid(self, validator):
        fv = _make_field("corporate_reg_number", "11011-2345678")
        fields, errors = validator.validate_and_correct([fv])
        assert errors == []

    def test_corporate_reg_number_invalid(self, validator):
        fv = _make_field("corporate_reg_number", "INVALID")
        fields, errors = validator.validate_and_correct([fv])
        assert len(errors) == 1
        assert errors[0].field_ref == "corporate_reg_number"

    def test_business_reg_number_invalid(self, validator):
        fv = _make_field("business_reg_number", "1376312345")
        fields, errors = validator.validate_and_correct([fv])
        assert any(e.field_ref == "business_reg_number" for e in errors)

    def test_nsn_valid(self, validator):
        fv = _make_field("nsn", "1005-01-432-1234", data_type="code")
        fields, errors = validator.validate_and_correct([fv])
        # nsn field_key pattern 자체는 통과하나 data_type=code에 대한 별도 패턴은 없음.
        assert not any(e.field_ref == "nsn" for e in errors)

    def test_k_nsn_invalid(self, validator):
        fv = _make_field("k_nsn", "KN-abc-1234")
        fields, errors = validator.validate_and_correct([fv])
        assert any(e.field_ref == "k_nsn" for e in errors)

    def test_unknown_field_key_skipped(self, validator):
        fv = _make_field("random_field", "whatever")
        fields, errors = validator.validate_and_correct([fv])
        assert errors == []


class TestDataTypePatterns:
    def test_date_valid(self, validator):
        fv = _make_field("some_date", "2024년 05월 20일", data_type="date")
        fields, errors = validator.validate_and_correct([fv])
        assert not any(e.error_type.value == "date_logic" for e in errors)

    def test_date_normalized(self, validator):
        """'2024년5월20일' → '2024년 05월 20일' 정규화."""
        fv = _make_field("some_date", "2024년5월20일", data_type="date")
        fields, errors = validator.validate_and_correct([fv])
        assert fields[0].corrected_value == "2024년 05월 20일"
        # 정규화 성공하면 에러 없음
        assert errors == []

    def test_date_invalid_after_normalize(self, validator):
        """정규화 규칙으로도 교정 불가한 값은 에러."""
        fv = _make_field("some_date", "날짜미상", data_type="date")
        fields, errors = validator.validate_and_correct([fv])
        assert any(e.field_ref == "some_date" for e in errors)


class TestEmptyValues:
    def test_empty_string_skipped(self, validator):
        fv = _make_field("corporate_reg_number", "")
        fields, errors = validator.validate_and_correct([fv])
        # 빈 값은 Layer 2 NOT_EMPTY로 처리 — 여기선 건너뜀
        assert errors == []

    def test_none_literal_skipped(self, validator):
        fv = _make_field("corporate_reg_number", "None")
        fields, errors = validator.validate_and_correct([fv])
        assert errors == []


class TestCorrectionPreservesFieldValue:
    def test_original_fields_order_preserved(self, validator):
        fvs = [
            _make_field("corporate_reg_number", "11011-2345678"),
            _make_field("business_reg_number", "137-63-12345"),
            _make_field("unknown_key", "whatever"),
        ]
        fields, _ = validator.validate_and_correct(fvs)
        assert [f.field_key for f in fields] == [
            "corporate_reg_number", "business_reg_number", "unknown_key"
        ]
