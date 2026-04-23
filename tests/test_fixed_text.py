"""fixed_text 통합 기능 단위 테스트.

검증 대상:
  - TemplateAugmentor.augment() 반환값이 (LayoutResult, fixed_values) tuple
  - fixed_text field_key → value 매핑이 fixed_values에 포함
  - bbox가 있는 fixed_text는 겹치는 PP region 제거 (field_key 없는 것만)
  - field_key가 있는 region은 fixed_text bbox와 겹쳐도 보호됨
  - Assembler가 fixed_values를 assembled_json에 주입 (VLM 결과 우선)
  - _path_exists 동작 검증
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import AnalysisMode, FormType, RegionType
from src.interfaces.types import BoundingBox, FieldValue, LayoutRegion, LayoutResult
from src.vlm.assembler import Assembler
from src.vlm.template_augmentor import TemplateAugmentor, TemplateAugmentorStats


# ─────────────────────────────────────────────
#  TemplateAugmentor — fixed_text 처리
# ─────────────────────────────────────────────

class TestTemplateAugmentorFixedText:

    def _make_layout(self, regions: list[LayoutRegion]) -> LayoutResult:
        return LayoutResult(
            doc_id="test",
            page_width=2480,
            page_height=3508,
            regions=regions,
            reading_order=list(range(len(regions))),
            analysis_mode=AnalysisMode.HEURISTIC,
        )

    def test_augment_returns_tuple(self, tmp_path: Path):
        """augment()가 (LayoutResult, fixed_values) 튜플 반환."""
        tpl_yaml = """
form_type: bid_application
versions:
  - version: "1.0"
    form_identifier: "test"
    page_size: [2480, 3508]
    fields: []
    fixed_text:
      - field_key: title
        value: "입찰참가신청서"
"""
        (tmp_path / "bid_application.yaml").write_text(tpl_yaml, encoding="utf-8")

        augmentor = TemplateAugmentor(template_dir=str(tmp_path))
        layout = self._make_layout([])
        stats = TemplateAugmentorStats()

        result = augmentor.augment(layout, FormType.BID_APPLICATION, stats=stats)
        assert isinstance(result, tuple)
        assert len(result) == 2

        aug_layout, fixed_values = result
        assert isinstance(aug_layout, LayoutResult)
        assert isinstance(fixed_values, dict)
        assert fixed_values == {"title": "입찰참가신청서"}
        assert stats.fixed_text_count == 1

    def test_fixed_text_without_bbox_no_region_removal(self, tmp_path: Path):
        """bbox 없는 fixed_text는 region 제거 없이 값만 등록."""
        tpl_yaml = """
form_type: bid_application
versions:
  - version: "1.0"
    page_size: [2480, 3508]
    fields: []
    fixed_text:
      - field_key: form_title
        value: "타이틀"
"""
        (tmp_path / "bid_application.yaml").write_text(tpl_yaml, encoding="utf-8")

        pp_region = LayoutRegion(
            region_id="r_0001",
            region_type=RegionType.TEXT,
            bbox=BoundingBox(100, 100, 500, 200),
            confidence=0.9,
        )
        layout = self._make_layout([pp_region])
        stats = TemplateAugmentorStats()

        aug_layout, fixed_values = TemplateAugmentor(template_dir=str(tmp_path)).augment(
            layout, FormType.BID_APPLICATION, stats=stats,
        )
        assert fixed_values == {"form_title": "타이틀"}
        assert stats.fixed_text_removed_count == 0
        # region은 그대로 유지됨
        assert any(r.region_id == "r_0001" for r in aug_layout.regions)

    def test_fixed_text_with_bbox_removes_overlapping_pp(self, tmp_path: Path):
        """bbox 있는 fixed_text는 IoU>0.5 PP region 제거. field_key 없는 것만."""
        tpl_yaml = """
form_type: bid_application
versions:
  - version: "1.0"
    page_size: [2480, 3508]
    fields: []
    fixed_text:
      - field_key: body_text
        bbox: {x1: 100, y1: 100, x2: 500, y2: 200}
        value: "본문"
"""
        (tmp_path / "bid_application.yaml").write_text(tpl_yaml, encoding="utf-8")

        pp_overlap = LayoutRegion(
            region_id="r_overlap",
            region_type=RegionType.TEXT,
            bbox=BoundingBox(110, 110, 490, 190),  # 거의 동일 → IoU 높음
            confidence=0.85,
        )
        pp_far = LayoutRegion(
            region_id="r_far",
            region_type=RegionType.TEXT,
            bbox=BoundingBox(1000, 1000, 1500, 1200),  # 멀리 떨어짐
            confidence=0.9,
        )
        layout = self._make_layout([pp_overlap, pp_far])
        stats = TemplateAugmentorStats()

        aug_layout, fixed_values = TemplateAugmentor(template_dir=str(tmp_path)).augment(
            layout, FormType.BID_APPLICATION, stats=stats,
        )
        region_ids = {r.region_id for r in aug_layout.regions}
        assert "r_overlap" not in region_ids, "겹치는 PP region은 제거되어야 함"
        assert "r_far" in region_ids, "멀리 떨어진 PP region은 유지되어야 함"
        assert stats.fixed_text_removed_count == 1
        assert fixed_values == {"body_text": "본문"}

    def test_fixed_text_preserves_region_with_field_key(self, tmp_path: Path):
        """field_key가 부여된 region은 fixed_text bbox와 겹쳐도 보호됨."""
        tpl_yaml = """
form_type: bid_application
versions:
  - version: "1.0"
    page_size: [2480, 3508]
    fields:
      - field_key: applicant_block
        region_type: handwritten_field
        bbox: {x1: 100, y1: 100, x2: 500, y2: 200}
    fixed_text:
      - field_key: body_text
        bbox: {x1: 100, y1: 100, x2: 500, y2: 200}  # 동일 위치
        value: "본문"
"""
        (tmp_path / "bid_application.yaml").write_text(tpl_yaml, encoding="utf-8")

        layout = self._make_layout([])  # PP 없음 → 템플릿 기반 region만 생성
        stats = TemplateAugmentorStats()

        aug_layout, fixed_values = TemplateAugmentor(template_dir=str(tmp_path)).augment(
            layout, FormType.BID_APPLICATION, stats=stats,
        )
        # applicant_block region이 보호되어야 함
        assert any(
            r.field_key == "applicant_block" for r in aug_layout.regions
        ), "field_key가 있는 region은 fixed_text bbox와 겹쳐도 제거 금지"
        assert stats.fixed_text_removed_count == 0


# ─────────────────────────────────────────────
#  Assembler — fixed_values 주입
# ─────────────────────────────────────────────

class TestAssemblerFixedValues:

    def _schema(self) -> dict:
        return {
            "x-assembly-rules": {
                "applicant_block": "applicant",
                "body_text": "fixed_content.body_text",
                "form_title": "fixed_content.form_title",
            },
            "type": "object",
            "properties": {},
        }

    def test_fixed_values_injected_when_no_vlm_result(self):
        """VLM 추출 필드가 없는 field_key에 대해 fixed_values로 삽입."""
        schema = self._schema()
        fixed_values = {
            "body_text": "본인은 위의 번호로...",
            "form_title": "입찰참가신청서",
        }
        result = Assembler().assemble(
            fields=[],
            schema=schema,
            fixed_values=fixed_values,
        )
        assert result is not None
        assert result["fixed_content"]["body_text"] == "본인은 위의 번호로..."
        assert result["fixed_content"]["form_title"] == "입찰참가신청서"

    def test_vlm_result_not_overridden_by_fixed(self):
        """VLM이 이미 추출한 field_key는 fixed_values가 덮어쓰지 않음."""
        schema = self._schema()
        # applicant_block이 VLM에 의해 이미 추출됨
        vlm_field = FieldValue(
            field_key="applicant_block",
            raw_value='{"company_name":"VLM추출"}',
            corrected_value='{"company_name":"VLM추출"}',
            data_type="text",
            confidence=0.9,
            token_logprobs=[],
        )
        fixed_values = {
            "applicant_block": "이 값은 VLM 결과가 있으므로 삽입되지 않아야 함",
        }
        result = Assembler().assemble(
            fields=[vlm_field],
            schema=schema,
            fixed_values=fixed_values,
        )
        assert result["applicant"] == {"company_name": "VLM추출"}

    def test_fixed_values_ignored_without_assembly_rule(self):
        """x-assembly-rules에 매핑되지 않은 field_key는 무시."""
        schema = self._schema()
        result = Assembler().assemble(
            fields=[],
            schema=schema,
            fixed_values={"unknown_field": "값"},
        )
        assert result is not None
        assert "unknown_field" not in result
        assert "unknown" not in result

    def test_empty_fixed_values_same_as_none(self):
        schema = self._schema()
        r1 = Assembler().assemble(fields=[], schema=schema, fixed_values={})
        r2 = Assembler().assemble(fields=[], schema=schema, fixed_values=None)
        assert r1 == r2


# ─────────────────────────────────────────────
#  _path_exists
# ─────────────────────────────────────────────

class TestPathExists:

    def test_nested_path_present(self):
        a = Assembler()
        obj = {"fixed_content": {"body_text": "hello"}}
        assert a._path_exists(obj, "fixed_content.body_text") is True

    def test_path_missing_returns_false(self):
        a = Assembler()
        obj = {"fixed_content": {}}
        assert a._path_exists(obj, "fixed_content.body_text") is False

    def test_none_value_returns_false(self):
        a = Assembler()
        obj = {"applicant": None}
        assert a._path_exists(obj, "applicant") is False

    def test_top_level_path(self):
        a = Assembler()
        assert a._path_exists({"a": 1}, "a") is True
        assert a._path_exists({"a": 1}, "b") is False


# ─────────────────────────────────────────────
#  Assembler — _unwrap_blob / sub-schema 응답 래퍼 제거
# ─────────────────────────────────────────────

class TestAssemblerUnwrap:

    def test_unwrap_field_key_wrapper_dict(self):
        """VLM이 {'applicant_block': {...}}로 감싸 반환한 것을 벗겨냄."""
        result = Assembler._unwrap_blob(
            parsed={"applicant_block": {"company_name": "(주) 인피닉"}},
            field_key="applicant_block",
        )
        assert result == {"company_name": "(주) 인피닉"}

    def test_unwrap_field_key_wrapper_scalar(self):
        """스칼라 래퍼: {'doc_number': '제 호'} → '제 호'."""
        result = Assembler._unwrap_blob(
            parsed={"doc_number": "제 호"},
            field_key="doc_number",
        )
        assert result == "제 호"

    def test_no_unwrap_when_key_absent(self):
        """field_key가 parsed에 없으면 원본 반환."""
        p = {"company_name": "A", "representative": "B"}
        result = Assembler._unwrap_blob(parsed=p, field_key="applicant_block")
        assert result == p

    def test_no_unwrap_when_multiple_keys(self):
        """키가 여러 개면 래핑이 아니므로 그대로 반환."""
        p = {"applicant_block": "A", "other": "B"}
        result = Assembler._unwrap_blob(parsed=p, field_key="applicant_block")
        assert result == p

    def test_unwrap_path_tail_for_seal_present(self):
        """path='agent.seal_present', parsed={'seal_present': false} → false."""
        result = Assembler._unwrap_blob(
            parsed={"seal_present": False},
            field_key="agent_seal",
            path="agent.seal_present",
        )
        assert result is False

    def test_unwrap_double_wrapper(self):
        """{'agent_seal': {'seal_present': true}} + path='agent.seal_present' → true."""
        result = Assembler._unwrap_blob(
            parsed={"agent_seal": {"seal_present": True}},
            field_key="agent_seal",
            path="agent.seal_present",
        )
        assert result is True

    def test_passthrough_non_dict(self):
        """dict이 아닌 경우 그대로 반환."""
        assert Assembler._unwrap_blob(parsed="string value", field_key="doc_number") == "string value"
        assert Assembler._unwrap_blob(parsed=42, field_key="x") == 42
        assert Assembler._unwrap_blob(parsed=[1, 2], field_key="x") == [1, 2]


class TestAssembledJsonStructure:
    """assemble() 최종 구조 검증 — 래퍼 없는 flat 매핑."""

    def _schema(self) -> dict:
        return {
            "x-assembly-rules": {
                "applicant_block": "applicant",
                "agent_seal": "agent.seal_present",
                "seal_verification_block": "seal_verification",
            },
            "type": "object",
            "properties": {
                "applicant": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "string"},
                        "representative": {"type": "string"},
                    },
                },
                "agent": {
                    "type": "object",
                    "properties": {
                        "seal_present": {"type": "boolean"},
                    },
                },
                "seal_verification": {
                    "type": "object",
                    "properties": {
                        "issuing_office": {"type": "string"},
                        "signature_present": {"type": "boolean"},
                    },
                },
            },
        }

    def _mk_field(self, field_key: str, value) -> FieldValue:
        import json as _json
        raw = _json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        return FieldValue(
            field_key=field_key,
            raw_value=raw,
            corrected_value=raw,
            data_type="text",
            confidence=0.9,
            token_logprobs=[],
        )

    def test_applicant_no_wrapper(self):
        """applicant_block 응답이 {'applicant_block': {...}}로 감싸도 flat하게 매핑."""
        fields = [
            self._mk_field(
                "applicant_block",
                {"applicant_block": {"company_name": "(주) 인피닉", "representative": "박 준형"}},
            ),
        ]
        result = Assembler().assemble(fields=fields, schema=self._schema())
        assert result["applicant"] == {"company_name": "(주) 인피닉", "representative": "박 준형"}
        assert "applicant_block" not in result["applicant"]

    def test_agent_seal_unwraps_to_boolean(self):
        """agent_seal이 {'seal_present': false} 반환 → agent.seal_present = false."""
        fields = [self._mk_field("agent_seal", {"seal_present": False})]
        result = Assembler().assemble(fields=fields, schema=self._schema())
        assert result["agent"]["seal_present"] is False

    def test_agent_seal_double_wrapper(self):
        """agent_seal이 {'agent_seal': {'seal_present': true}}로 감싸도 boolean 추출."""
        fields = [
            self._mk_field("agent_seal", {"agent_seal": {"seal_present": True}}),
        ]
        result = Assembler().assemble(fields=fields, schema=self._schema())
        assert result["agent"]["seal_present"] is True

    def test_table_region_field_connected(self):
        """table region field_key(seal_verification_block)가 assembled_json.seal_verification에 연결됨."""
        fields = [
            self._mk_field(
                "seal_verification_block",
                {
                    "seal_verification_block": {
                        "issuing_office": "서울특별시 강남구",
                        "signature_present": True,
                    }
                },
            ),
        ]
        result = Assembler().assemble(fields=fields, schema=self._schema())
        assert result["seal_verification"]["issuing_office"] == "서울특별시 강남구"
        assert result["seal_verification"]["signature_present"] is True
