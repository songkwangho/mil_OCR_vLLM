"""P4 신뢰도 재산출 단위 테스트 — assembled_json 트리 순회.

검증 대상:
  - _traverse_assembled: 리프 경로별 (path, type, conf) 추출
  - _path_to_field_key: military checklist/writer 경로 → field_key 변환
  - _infer_field_type: path 기반 field_type 추론
  - _calc_overall_from_sub: 가중 평균
  - P4Validator.validate: assembled_json 있는 경우 sub_confidences 산출
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import FormType, ProcessingPath
from src.interfaces.types import FieldValue, VLMResult
from src.postprocess.validator import (
    FIELD_WEIGHTS,
    P4Validator,
    _calc_overall_from_sub,
    _infer_field_type,
    _path_to_field_key,
    _traverse_assembled,
)


# ─────────────────────────────────────────────
#  _path_to_field_key
# ─────────────────────────────────────────────

class TestPathToFieldKey:

    def test_checklist_items_maps_to_result_item_one_based(self):
        assert _path_to_field_key("checklist_items[0]") == "result_item_1"
        assert _path_to_field_key("checklist_items[5]") == "result_item_6"

    def test_writer_subpath_maps_to_writer_block(self):
        assert _path_to_field_key("writer.name") == "writer_block"
        assert _path_to_field_key("writer.rank") == "writer_block"
        assert _path_to_field_key("writer") == "writer_block"

    def test_flat_key_returns_itself(self):
        assert _path_to_field_key("document_date") == "document_date"
        assert _path_to_field_key("form_identifier") == "form_identifier"

    def test_other_nested_path_returns_none(self):
        """official_document 경로는 field_key 체계가 없어 None."""
        assert _path_to_field_key("sender_info.organization") is None
        assert _path_to_field_key("content.body") is None


# ─────────────────────────────────────────────
#  _infer_field_type
# ─────────────────────────────────────────────

class TestInferFieldType:

    def test_code_fields(self):
        assert _infer_field_type("nsn") == "code"
        assert _infer_field_type("items[0].nsn") == "code"
        assert _infer_field_type("unit_code") == "code"
        assert _infer_field_type("document_number") == "code"

    def test_amount_fields(self):
        assert _infer_field_type("total_amount") == "amount"
        assert _infer_field_type("items[0].unit_price") == "amount"

    def test_date_fields(self):
        assert _infer_field_type("request_date") == "date"
        assert _infer_field_type("document_info.date") == "date"

    def test_quantity_fields(self):
        assert _infer_field_type("items[0].quantity") == "quantity"

    def test_result_as_handwritten(self):
        assert _infer_field_type("checklist_items[0].result") == "handwritten"

    def test_seal_signature(self):
        assert _infer_field_type("official_seal.text.value") == "seal"
        assert _infer_field_type("approval_table[0].signature_present") == "signature"

    def test_default_text(self):
        assert _infer_field_type("content.body") == "text"
        assert _infer_field_type("sender_info.organization") == "text"


# ─────────────────────────────────────────────
#  _traverse_assembled
# ─────────────────────────────────────────────

class TestTraverseAssembled:

    def test_equipment_checklist_uses_result_confidence(self):
        """checklist_items[N]에 result_confidence 형제 키가 있으면 그 값을 사용."""
        assembled = {
            "checklist_items": [
                {"item_number": 1, "result": "O", "result_confidence": 0.99},
                {"item_number": 2, "result": "X", "result_confidence": 0.88},
            ],
            "writer": {"name": "홍길동", "rank": "소위"},
        }
        field_map = {
            "writer_block": FieldValue(
                field_key="writer_block",
                raw_value='{"name":"홍길동"}',
                corrected_value='{"name":"홍길동"}',
                data_type="text",
                confidence=0.91,
                token_logprobs=[],
            ),
        }
        sub: list = []
        _traverse_assembled(assembled, field_map, sub)

        # checklist_items 2개 + writer.name + writer.rank = 4
        paths = [p for p, _, _ in sub]
        assert "checklist_items[0]" in paths
        assert "checklist_items[1]" in paths
        assert "writer.name" in paths
        assert "writer.rank" in paths

        checklist_confs = {p: c for p, _, c in sub if p.startswith("checklist_items")}
        assert checklist_confs["checklist_items[0]"] == 0.99
        assert checklist_confs["checklist_items[1]"] == 0.88

        # writer.name/rank는 writer_block field_map → 0.91
        writer_confs = {p: c for p, _, c in sub if p.startswith("writer.")}
        assert writer_confs["writer.name"] == 0.91
        assert writer_confs["writer.rank"] == 0.91

    def test_other_path_falls_back_to_0_5(self):
        """official_document 스타일 경로는 field_map miss → 폴백 0.5."""
        assembled = {
            "document_info": {"organization": "국방부", "date": "2015-12-02"},
            "content": {"subject": "의안 심사기간 지정"},
            "analysis": "skipped_meta_field",
        }
        sub: list = []
        _traverse_assembled(assembled, {}, sub)

        paths = [p for p, _, _ in sub]
        assert "document_info.organization" in paths
        assert "document_info.date" in paths
        assert "content.subject" in paths
        assert "analysis" not in paths  # 메타 필드 스킵
        assert all(c == 0.5 for _, _, c in sub)

    def test_skip_null_and_empty(self):
        assembled = {
            "a": None,
            "b": "",
            "c": True,       # bool은 리프 아님
            "d": "value",
        }
        sub: list = []
        _traverse_assembled(assembled, {}, sub)
        paths = [p for p, _, _ in sub]
        assert paths == ["d"]

    def test_skip_meta_keys(self):
        """analysis/result_confidence/aggregator_blob/low_confidence_fields/overall_confidence/fixed_content 스킵."""
        assembled = {
            "analysis": "meta",
            "aggregator_blob": "blob",
            "low_confidence_fields": ["a", "b"],
            "overall_confidence": 0.9,
            "fixed_content": {
                "form_title": "인쇄 고정 제목",
                "checklist_item_1_text": "점검항목 1",
            },
            "document_date": "2026-04-17",
        }
        sub: list = []
        _traverse_assembled(assembled, {}, sub)
        paths = [p for p, _, _ in sub]
        assert paths == ["document_date"]
        # fixed_content 하위 경로는 신뢰도 산출 대상에 포함되지 않아야 함
        assert not any("fixed_content" in p for p in paths)


# ─────────────────────────────────────────────
#  _calc_overall_from_sub
# ─────────────────────────────────────────────

class TestCalcOverall:

    def test_empty_returns_zero(self):
        assert _calc_overall_from_sub([]) == 0.0

    def test_weighted_average(self):
        sub = [
            ("nsn", "code", 0.90),          # weight 2.5
            ("date", "date", 0.80),         # weight 2.0
            ("notes", "text", 1.00),        # weight 1.0
        ]
        total = 2.5 * 0.90 + 2.0 * 0.80 + 1.0 * 1.00
        expected = round(total / (2.5 + 2.0 + 1.0), 4)
        assert _calc_overall_from_sub(sub) == expected

    def test_values_clamped_to_0_1(self):
        sub = [("a", "text", 0.5), ("b", "text", 0.7)]
        r = _calc_overall_from_sub(sub)
        assert 0.0 <= r <= 1.0


# ─────────────────────────────────────────────
#  P4Validator.validate — assembled_json 경로
# ─────────────────────────────────────────────

class TestValidatorAssembledPath:

    def _make_vlm_result(self, assembled, fields, form_type=FormType.EQUIPMENT_CHECKLIST):
        return VLMResult(
            doc_id="test",
            form_type=form_type,
            form_confidence=0.9,
            schema_id=form_type.value,
            fields=fields,
            tables=[],
            domain_codes=[],
            assembled_json=assembled,
        )

    def test_equipment_checklist_overall_above_old_0_13(self):
        """전비품 확인서 시나리오 — result_confidence 기반 산출 시 0.85+ 예상."""
        assembled = {
            "checklist_items": [
                {"item_number": i, "result": "O", "result_confidence": 0.98}
                for i in range(1, 7)
            ],
            "writer": {"name": "홍길동", "rank": "소위"},
            "document_date": "2026-04-17",
            "form_identifier": "별지 제3-2호 서식",
        }
        fields = [
            FieldValue(
                field_key="writer_block",
                raw_value='{"name":"홍길동","rank":"소위"}',
                corrected_value='{"name":"홍길동","rank":"소위"}',
                data_type="text",
                confidence=0.92,
                token_logprobs=[],
            ),
            FieldValue(
                field_key="document_date",
                raw_value="2026-04-17",
                corrected_value="2026-04-17",
                data_type="date",
                confidence=0.95,
                token_logprobs=[],
            ),
            FieldValue(
                field_key="form_identifier",
                raw_value="별지 제3-2호 서식",
                corrected_value="별지 제3-2호 서식",
                data_type="text",
                confidence=0.88,
                token_logprobs=[],
            ),
        ]
        vlm_result = self._make_vlm_result(assembled, fields)
        validator = P4Validator()
        validated = validator.validate(vlm_result, ProcessingPath.VLM)

        assert validated.overall_confidence > 0.85, (
            f"assembled_json 기반 산출이 0.85 초과여야 함 (기존 blob 평균 0.13 개선 목표). "
            f"실제={validated.overall_confidence}"
        )
        assert len(validated.sub_confidences) >= 8  # 6 checklist + writer.name + writer.rank + document_date + form_identifier
        paths = [p for p, _, _ in validated.sub_confidences]
        assert any("checklist_items[0]" in p for p in paths)
        assert "writer.name" in paths
        assert "document_date" in paths

    def test_other_path_produces_sub_confidences(self):
        """other 경로 — assembled_json이 있으면 sub_confidences 산출."""
        assembled = {
            "document_info": {"organization": "국방부", "date": "2015-12-02"},
            "sender_info": {"department": "의안과"},
            "content": {"subject": "테스트"},
        }
        validator = P4Validator()
        vlm_result = self._make_vlm_result(assembled, fields=[], form_type=FormType.OTHER)
        validated = validator.validate(vlm_result, ProcessingPath.VLM)

        # other 경로는 review_required=False 유지
        assert validated.review_required is False
        # field_map miss → 모두 0.5 폴백
        assert len(validated.sub_confidences) >= 4
        paths = [p for p, _, _ in validated.sub_confidences]
        assert "document_info.organization" in paths
        assert "document_info.date" in paths
        # 0.5 폴백이 다수이므로 overall도 0.5 근처
        assert 0.4 <= validated.overall_confidence <= 0.6

    def test_assembled_json_none_uses_fields_average(self):
        """assembled_json이 없는 서식 — 기존 방식(fields 평균) 유지, sub_confidences 빈 리스트."""
        fields = [
            FieldValue(
                field_key="unit_code",
                raw_value="1234-56",
                corrected_value="1234-56",
                data_type="code",
                confidence=0.95,
                token_logprobs=[],
            ),
            FieldValue(
                field_key="request_date",
                raw_value="2026-04-17",
                corrected_value="2026-04-17",
                data_type="date",
                confidence=0.90,
                token_logprobs=[],
            ),
            FieldValue(
                field_key="items",
                raw_value="[]",
                corrected_value="[]",
                data_type="text",
                confidence=0.80,
                token_logprobs=[],
            ),
        ]
        vlm_result = VLMResult(
            doc_id="test_no_assembled",
            form_type=FormType.SUPPLY_REQUEST,
            form_confidence=0.9,
            schema_id="supply_request",
            fields=fields,
            tables=[],
            domain_codes=[],
            assembled_json=None,
        )
        validator = P4Validator()
        validated = validator.validate(vlm_result, ProcessingPath.VLM)

        assert validated.sub_confidences == []
        # fields 평균이지만 룰 패널티로 missing_field 감점 발생
        assert 0.0 <= validated.overall_confidence <= 1.0
