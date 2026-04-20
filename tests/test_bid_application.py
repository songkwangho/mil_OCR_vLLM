"""입찰참가신청서 (bid_application) 신규 FormType 단위 테스트.

검증 대상:
  - FormType.BID_APPLICATION enum 존재
  - FORM_TYPE_LABELS 한국어 매핑
  - bid_application.json 스키마 로드 가능 + x-assembly-rules 구조
  - bid_application.yaml 템플릿 파싱 + 필수 field_key 존재
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
        # 핵심 field_key → path 매핑
        assert rules["applicant_block"] == "applicant"
        assert rules["bid_info_block"] == "bid_info"
        assert rules["submission_block"] == "submission"
        assert rules["agent_seal"] == "agent.seal_present"
        assert rules["seal_verification_block"] == "seal_verification"

    def test_required_top_level(self):
        schema = SchemaRegistry().get("bid_application", "v1")
        required = set(schema.get("required", []))
        assert {"analysis", "applicant", "bid_info", "submission"} <= required

    def test_applicant_required_fields(self):
        schema = SchemaRegistry().get("bid_application", "v1")
        applicant = schema["properties"]["applicant"]
        required = set(applicant.get("required", []))
        assert {"company_name", "representative", "business_reg_number"} <= required


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

    def test_version_1_0_fields(self):
        tpl = self._load()
        v1 = tpl["versions"][0]
        assert v1["version"] == "1.0"
        assert v1["form_identifier"] == "별지 제13호 서식"
        field_keys = {f["field_key"] for f in v1["fields"]}
        # x-assembly-rules와 일치해야 함 (키들)
        expected = {
            "doc_number", "applicant_block", "bid_info_block",
            "agent_block", "agent_seal",
            "submission_block", "seal_verification_block",
        }
        assert expected <= field_keys, f"missing: {expected - field_keys}"

    def test_field_keys_match_assembly_rules(self):
        """템플릿 fields ∪ fixed_text ∪ {form_identifier} ⊇ x-assembly-rules keys."""
        tpl = self._load()
        v1 = tpl["versions"][0]
        field_keys = {f["field_key"] for f in v1.get("fields", [])}
        field_keys |= {f["field_key"] for f in v1.get("fixed_text", [])}
        field_keys.add("form_identifier")  # P3-A에서 추출되어 assemble됨

        schema = SchemaRegistry().get("bid_application", "v1")
        rules_keys = set(schema["x-assembly-rules"].keys())
        missing = rules_keys - field_keys
        assert not missing, f"x-assembly-rules에 있으나 template에 없는 field_key: {missing}"


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

    def test_bid_001_business_reg_format_violation(self):
        data = {
            "applicant": {"business_reg_number": "1376312345"},  # 하이픈 누락
            "submission": {"submission_date": "2024년 05월 15일", "submitter_name": "홍길동"},
        }
        errors = _validate_bid_application(data)
        assert any(e.error_id == "bid_001" for e in errors)
        e = next(e for e in errors if e.error_id == "bid_001")
        assert e.error_type == ValidationErrorType.CODE_FORMAT
        assert e.severity == Severity.HIGH

    def test_bid_002_corporate_reg_format_violation(self):
        data = {
            "applicant": {
                "business_reg_number": "137-63-12345",
                "corporate_reg_number": "INVALID",
            },
            "submission": {"submission_date": "2024-05-15", "submitter_name": "홍길동"},
        }
        errors = _validate_bid_application(data)
        assert any(e.error_id == "bid_002" for e in errors)

    def test_bid_002_skip_when_empty(self):
        """법인등록번호는 있을 때만 검증 (개인사업자는 없음)."""
        data = {
            "applicant": {"business_reg_number": "137-63-12345"},  # corp_reg 없음
            "submission": {"submission_date": "2024-05-15", "submitter_name": "홍길동"},
        }
        errors = _validate_bid_application(data)
        assert not any(e.error_id == "bid_002" for e in errors)

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
        """assembled_json이 없고 raw_json만 있는 폴백 경로."""
        raw = json.dumps({
            "applicant": {"business_reg_number": "WRONG"},
            "submission": {"submission_date": "2024-05-15", "submitter_name": "홍길동"},
        })
        errors = _validate_bid_application(raw)
        assert any(e.error_id == "bid_001" for e in errors)

    def test_none_input_returns_empty(self):
        assert _validate_bid_application(None) == []
        assert _validate_bid_application("") == []
        assert _validate_bid_application("not a json") == []
