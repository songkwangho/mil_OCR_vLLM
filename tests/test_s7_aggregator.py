"""S7 StructuredAggregator 단위 테스트.

PIPELINE.md §4-8 구현 검증.
  - _build_context: S2~S6 결과 직렬화 포맷
  - run 성공 경로: VLMResult.assembled_json 구조
  - run 실패 경로: VLM 호출 실패 시 region 평탄화 폴백

vLLM 서버 없이 MagicMock VLMClient로 검증 (SchemaRegistry는 실제 스캔).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.enums import FormType, ProcessingPath
from src.interfaces.types import (
    BoundingBox,
    SkillResult,
    TableCell,
    TableStructure,
)
from src.vlm.skills.aggregator import (
    StructuredAggregator,
    _parent_table_region,
    _parse_cell_suffix,
)


# ─────────────────────────────────────────────
#  fixture
# ─────────────────────────────────────────────

@pytest.fixture
def sample_skill_results() -> list[SkillResult]:
    """국회공문서 유형을 흉내 낸 S2~S6 결과."""
    return [
        SkillResult(
            region_id="r_0001", skill_name="S2", content="국방부",
            confidence=0.95, content_type="printed",
        ),
        SkillResult(
            region_id="r_0010", skill_name="S2",
            content="제 목: 의안 심사기간 지정의 건",
            confidence=0.92, content_type="printed",
        ),
        SkillResult(
            region_id="r_0022", skill_name="S4",
            content=json.dumps({"text": "국방부장관인", "confidence": 0.9},
                               ensure_ascii=False),
            confidence=0.9, content_type="seal",
        ),
        SkillResult(
            region_id="r_0023", skill_name="S6",
            content=json.dumps({"signature_present": True, "confidence": 0.88}),
            confidence=0.88, content_type="signature",
        ),
        # 표 셀 — S5 pass2 태스크가 dispatch되어 S2로 인식됨
        SkillResult(
            region_id="r_0030_r0c0", skill_name="S2", content="기안",
            confidence=0.95, content_type="printed",
        ),
        SkillResult(
            region_id="r_0030_r1c0", skill_name="S3", content="홍길동",
            confidence=0.82, content_type="handwritten",
        ),
    ]


@pytest.fixture
def sample_table_structure() -> TableStructure:
    return TableStructure(
        region_id="r_0030",
        table_type="approval",
        cells=[
            TableCell(0, 0, "기안", BoundingBox(0, 0, 100, 50), "printed"),
            TableCell(1, 0, "기안", BoundingBox(0, 50, 100, 100), "handwritten"),
        ],
        structure_confidence=0.9,
    )


@pytest.fixture
def dummy_page_image() -> np.ndarray:
    return np.full((200, 300, 3), 255, dtype=np.uint8)


# ─────────────────────────────────────────────
#  _build_context 포맷
# ─────────────────────────────────────────────

class TestBuildContext:
    """_build_context가 S2~S6 결과를 일관된 블록으로 직렬화한다."""

    def test_printed_header_seal_signature_all_present(
        self, sample_skill_results, sample_table_structure,
    ):
        agg = StructuredAggregator.__new__(StructuredAggregator)
        ctx = agg._build_context(sample_skill_results, [sample_table_structure])

        # 각 블록 라벨 존재
        assert "[TEXT]" in ctx
        assert "[SEAL]" in ctx
        assert "[SIGNATURE]" in ctx
        assert "[TABLE approval]" in ctx

        # 텍스트 내용 보존
        assert "국방부" in ctx
        assert "의안 심사기간 지정의 건" in ctx
        assert "국방부장관인" in ctx  # seal text 파싱됨

        # signature 값 True 직렬화
        assert "present=True" in ctx

        # table 셀은 row/col 섹션 아래로 이동
        assert "r_0030" in ctx
        assert "홍길동" in ctx

    def test_empty_inputs_returns_placeholder(self):
        agg = StructuredAggregator.__new__(StructuredAggregator)
        ctx = agg._build_context([], [])
        assert "추출된 영역 없음" in ctx

    def test_table_cells_grouped_under_table(
        self, sample_skill_results, sample_table_structure,
    ):
        """표 셀 SkillResult는 해당 TABLE 블록 하위 row 라인에 속한다."""
        agg = StructuredAggregator.__new__(StructuredAggregator)
        ctx = agg._build_context(sample_skill_results, [sample_table_structure])

        # 표 셀은 단독 [TEXT] 블록으로 나타나면 안 된다.
        for line in ctx.splitlines():
            if line.startswith("[TEXT]"):
                assert "r_0030_r" not in line, (
                    "표 셀이 단독 블록으로 중복 출력됨: " + line
                )

        # row0/row1 섹션이 존재
        assert "row0:" in ctx
        assert "row1:" in ctx


# ─────────────────────────────────────────────
#  run() — 성공 경로
# ─────────────────────────────────────────────

class TestRunSuccess:
    """VLMClient가 유효한 JSON을 반환할 때 assembled_json이 채워진다."""

    def test_run_returns_vlm_result_with_assembled_json(
        self, sample_skill_results, sample_table_structure, dummy_page_image,
    ):
        vlm = MagicMock()
        vlm.call.return_value = {
            "content": json.dumps({
                "analysis": "국회에 보낸 공문서",
                "document_type": "공문서",
                "header": {"organization": "국방부"},
                "subject": "의안 심사기간 지정의 건",
                "official_seal": {
                    "detected": True,
                    "text": {"value": "국방부장관인", "confidence": 0.9},
                },
            }, ensure_ascii=False),
            "logprobs": [{"token": "{", "logprob": -0.01}],
            "finish_reason": "stop",
        }

        agg = StructuredAggregator(vlm_client=vlm)
        result = agg.run(
            skill_results=sample_skill_results,
            table_structures=[sample_table_structure],
            page_image=dummy_page_image,
            form_type=FormType.OTHER,
            doc_id="test_doc",
        )

        assert result.doc_id == "test_doc"
        assert result.form_type == FormType.OTHER
        assert result.processing_path == ProcessingPath.SKILL_REGISTRY
        assert result.schema_id == "official_document"
        assert isinstance(result.assembled_json, dict)
        assert result.assembled_json["header"]["organization"] == "국방부"
        assert result.assembled_json["official_seal"]["detected"] is True

        # fields에 region 단위 FieldValue + aggregator_blob 포함
        field_keys = {f.field_key for f in result.fields}
        assert "r_0001" in field_keys
        assert "aggregator_blob" in field_keys

        # vlm.call이 pixel_budget=140으로 호출됨
        _, kwargs = vlm.call.call_args
        assert kwargs["pixel_budget"] == StructuredAggregator.PIXEL_BUDGET
        assert kwargs["guided_json"] is not None
        assert kwargs["guided_json"].get("$id") == "official_document"


# ─────────────────────────────────────────────
#  run() — 실패 폴백
# ─────────────────────────────────────────────

class TestRunFallback:
    """VLM 예외 / 파싱 실패 시 region 평탄화 폴백 동작."""

    def test_vlm_exception_falls_back_to_region_flatten(
        self, sample_skill_results, sample_table_structure, dummy_page_image,
    ):
        vlm = MagicMock()
        vlm.call.side_effect = RuntimeError("vllm down")

        agg = StructuredAggregator(vlm_client=vlm)
        result = agg.run(
            skill_results=sample_skill_results,
            table_structures=[sample_table_structure],
            page_image=dummy_page_image,
            form_type=FormType.OTHER,
            doc_id="test_doc_fail",
        )

        assert result.assembled_json is None
        # 경고 기록 — S7 자체 message (단계 prefix는 orchestrator가 붙임)
        assert any("vlm_call_failed" in w for w in result.warnings)
        assert not any(w.startswith("[S7]") for w in result.warnings), (
            "S7 내부 warnings는 단계 prefix를 포함하지 않아야 한다 (이중 prefix 방지)"
        )
        # region 평탄화 fields 존재
        field_keys = {f.field_key for f in result.fields}
        assert "r_0001" in field_keys
        assert "r_0022" in field_keys
        # aggregator_blob은 실패 시 추가되지 않음
        assert "aggregator_blob" not in field_keys

    def test_json_parse_failure_records_warning(
        self, sample_skill_results, sample_table_structure, dummy_page_image,
    ):
        vlm = MagicMock()
        vlm.call.return_value = {
            "content": "not a json",
            "logprobs": None,
            "finish_reason": "stop",
        }
        agg = StructuredAggregator(vlm_client=vlm)
        result = agg.run(
            skill_results=sample_skill_results,
            table_structures=[sample_table_structure],
            page_image=dummy_page_image,
            form_type=FormType.OTHER,
            doc_id="test_parse_fail",
        )
        assert result.assembled_json is None
        assert any("JSON 파싱 실패" in w for w in result.warnings)


# ─────────────────────────────────────────────
#  유틸 단위 검증
# ─────────────────────────────────────────────

class TestCellSuffixParser:

    def test_parent_match(self):
        lookup = {"r_0030": None}  # 값 무시, 키만 본다
        assert _parent_table_region("r_0030_r0c1", lookup) == "r_0030"
        assert _parent_table_region("r_0030_r15c3", lookup) == "r_0030"

    def test_non_child_returns_none(self):
        lookup = {"r_0030": None}
        assert _parent_table_region("r_0031", lookup) is None
        assert _parent_table_region("r_0001", lookup) is None

    def test_parse_cell_suffix(self):
        assert _parse_cell_suffix("r_0030_r2c3", "r_0030") == (2, 3)
        assert _parse_cell_suffix("r_0030_r10c0", "r_0030") == (10, 0)

    def test_parse_cell_suffix_invalid(self):
        assert _parse_cell_suffix("r_0031_r0c0", "r_0030") is None
        assert _parse_cell_suffix("r_0030", "r_0030") is None
        assert _parse_cell_suffix("r_0030_rXcY", "r_0030") is None
