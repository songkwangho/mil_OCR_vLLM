"""입찰참가신청서 (bid_application) 신규 FormType 단위 테스트.

필드별 크롭 전환(2026-04-22) 반영 — block-level 대신 22개 field_key 기반.

검증 대상:
  - FormType.BID_APPLICATION enum 존재
  - FORM_TYPE_LABELS 한국어 매핑
  - bid_application.json 스키마 로드 가능 + 22개 1:1 x-assembly-rules
  - bid_application.yaml 템플릿 파싱 + 22개 field_key 전수 존재
  - 템플릿 field_key ∪ fixed_text.field_key == x-assembly-rules keys
  - _validate_bid_application: BID-001~004 검증
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.domain.form_type_labels import FORM_TYPE_LABELS, get_document_title
from src.domain.schema_registry import SchemaRegistry
from src.interfaces.enums import FormType, Severity, ValidationErrorType
from src.postprocess.validator import _validate_bid_application


# 22개 수기 field_key + 3개 fixed_text field_key = 25개 x-assembly-rules 키
_EXPECTED_HANDWRITTEN_FIELDS = {
    "company_name", "corporate_reg_number", "representative", "representative_dob",
    "address", "phone", "business_location", "business_reg_number",
    "announcement_number", "bid_date", "bid_name", "item_code",
    "industry_class_number", "acquisition_date", "issuing_office",
    "agent_name", "agent_dob", "agent_seal",
    "submission_date", "submitter_name",
    "seal_issuing_office", "seal_issue_number", "seal_issue_date", "seal_signature",
}
_EXPECTED_FIXED_TEXT_FIELDS = {"body_text", "seal_note", "form_title"}


# ─────────────────────────────────────────────
#  Enum + Label
# ─────────────────────────────────────────────

class TestEnumAndLabel:

    def test_form_type_enum_present(self):
        assert FormType.BID_APPLICATION.value == "bid_application"
        assert FormType.BID_APPLICATION.is_military() is True

    def test_label_mapping(self):
        assert FORM_TYPE_LABELS[FormType.BID_APPLICATION] == "입찰참가신청서"
        assert get_document_title(FormType.BID_APPLICATION) == "입찰참가신청서"


# ─────────────────────────────────────────────
#  Schema
# ─────────────────────────────────────────────

class TestSchema:

    def test_schema_loads(self):
        schema = SchemaRegistry().get("bid_application", "v1")
        assert schema is not None
        assert schema["$id"] == "bid_application"
        assert schema["title"] == "입찰참가신청서"

    def test_assembly_rules_present(self):
        schema = SchemaRegistry().get("bid_application", "v1")
        rules = schema.get("x-assembly-rules")
        assert isinstance(rules, dict)
        # 필드 단위 1:1 매핑 확인
        assert rules["company_name"] == "applicant.company_name"
        assert rules["bid_name"] == "bid_info.bid_name"
        assert rules["agent_seal"] == "agent.seal_present"
        assert rules["submission_date"] == "submission.submission_date"
        assert rules["seal_signature"] == "seal_verification.signature_present"
        assert rules["form_title"] == "fixed_content.form_title"

    def test_assembly_rules_covers_all_expected_keys(self):
        schema = SchemaRegistry().get("bid_application", "v1")
        rules_keys = set(schema["x-assembly-rules"].keys())
        expected = _EXPECTED_HANDWRITTEN_FIELDS | _EXPECTED_FIXED_TEXT_FIELDS
        assert expected == rules_keys, (
            f"missing: {expected - rules_keys}, unexpected: {rules_keys - expected}"
        )

    def test_required_top_level(self):
        schema = SchemaRegistry().get("bid_application", "v1")
        required = set(schema.get("required", []))
        assert {"applicant", "bid_info", "submission"} <= required

    def test_applicant_properties_present(self):
        schema = SchemaRegistry().get("bid_application", "v1")
        applicant = schema["properties"]["applicant"]
        assert applicant["type"] == "object"
        props = set(applicant.get("properties", {}).keys())
        assert {"company_name", "representative", "business_reg_number"} <= props


# ─────────────────────────────────────────────
#  Template YAML
# ─────────────────────────────────────────────

class TestTemplate:

    def _load(self):
        path = _PROJECT_ROOT / "configs" / "form_templates" / "bid_application.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_template_loads(self):
        tpl = self._load()
        assert tpl["form_type"] == "bid_application"
        assert isinstance(tpl.get("versions"), list) and tpl["versions"]

    def test_version_1_0_has_all_field_keys(self):
        tpl = self._load()
        v1 = tpl["versions"][0]
        assert v1["version"] == "1.0"
        assert v1["form_identifier"] == "별지 제13호 서식"
        field_keys = {f["field_key"] for f in v1["fields"]}
        assert _EXPECTED_HANDWRITTEN_FIELDS == field_keys, (
            f"missing: {_EXPECTED_HANDWRITTEN_FIELDS - field_keys}, "
            f"unexpected: {field_keys - _EXPECTED_HANDWRITTEN_FIELDS}"
        )

    def test_fixed_text_keys_present(self):
        tpl = self._load()
        v1 = tpl["versions"][0]
        fixed_keys = {f["field_key"] for f in v1.get("fixed_text", [])}
        assert _EXPECTED_FIXED_TEXT_FIELDS == fixed_keys

    def test_field_keys_match_assembly_rules(self):
        """템플릿 fields ∪ fixed_text == x-assembly-rules keys."""
        tpl = self._load()
        v1 = tpl["versions"][0]
        field_keys = {f["field_key"] for f in v1.get("fields", [])}
        field_keys |= {f["field_key"] for f in v1.get("fixed_text", [])}

        schema = SchemaRegistry().get("bid_application", "v1")
        rules_keys = set(schema["x-assembly-rules"].keys())
        missing = rules_keys - field_keys
        unexpected = field_keys - rules_keys
        assert not missing and not unexpected, (
            f"missing in template: {missing}, "
            f"missing in x-assembly-rules: {unexpected}"
        )

    def test_every_field_has_bbox(self):
        """fields 엔트리는 bbox 필수 (fixed_text는 form_title 제외 허용)."""
        tpl = self._load()
        v1 = tpl["versions"][0]
        for f in v1["fields"]:
            bbox = f.get("bbox")
            assert isinstance(bbox, dict) and {"x1", "y1", "x2", "y2"} <= bbox.keys(), (
                f"{f['field_key']}: bbox missing or malformed"
            )


# ─────────────────────────────────────────────
#  BID-001~004 validation
# ─────────────────────────────────────────────

class TestBidValidator:

    def test_all_valid(self):
        data = {
            "applicant": {
                "business_reg_number": "137-63-12345",
                "corporate_reg_number": "11011-2745676",
            },
            "submission": {
                "submission_date": "2024년 05월 15일",
                "submitter_name": "박 준형",
            },
        }
        errors = _validate_bid_application(data)
        assert errors == []

    # NOTE: business_reg_number / corporate_reg_number 형식 검증은
    # Layer 1 (FieldPatternValidator + common.yaml)에서 담당하므로
    # Layer 2 전용 shim인 _validate_bid_application 테스트에서는 제외.
    # 관련 커버리지는 tests/test_field_pattern_validator.py 참조.

    def test_bid_003_missing_submission_date(self):
        data = {
            "applicant": {"business_reg_number": "137-63-12345"},
            "submission": {"submitter_name": "홍길동"},
        }
        errors = _validate_bid_application(data)
        assert any(e.error_id == "bid_003" for e in errors)
        e = next(e for e in errors if e.error_id == "bid_003")
        assert e.severity == Severity.MEDIUM

    def test_bid_004_missing_submitter_name(self):
        data = {
            "applicant": {"business_reg_number": "137-63-12345"},
            "submission": {"submission_date": "2024-05-15", "submitter_name": "  "},  # whitespace
        }
        errors = _validate_bid_application(data)
        assert any(e.error_id == "bid_004" for e in errors)

    def test_accepts_raw_json_string(self):
        """assembled_json이 없고 raw_json만 있는 폴백 경로.

        Layer 2가 raw_json 문자열도 파싱해서 받아들이는지 확인.
        (bid_003 — submission_date 누락으로 오류 발생)
        """
        raw = json.dumps({
            "applicant": {"business_reg_number": "137-63-12345"},
            "submission": {"submitter_name": "홍길동"},  # submission_date 누락
        })
        errors = _validate_bid_application(raw)
        assert any(e.error_id == "bid_003" for e in errors)

    def test_none_input_returns_empty(self):
        assert _validate_bid_application(None) == []
        assert _validate_bid_application("") == []
        assert _validate_bid_application("not a json") == []
