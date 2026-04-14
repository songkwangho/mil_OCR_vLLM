"""Skills 단독 유닛 테스트 — VLMClient를 Stub으로 주입.

실제 vLLM 서버 없이 각 Skill의 입출력 계약을 검증합니다.
실행: python -m pytest tests/test_skills_unit.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.interfaces.enums import RegionType  # noqa: E402
from src.interfaces.types import SkillTask, TableStructure, TableCell, BoundingBox  # noqa: E402
from src.vlm.skills.seal_reader import SealReader  # noqa: E402
from src.vlm.skills.signature_detector import SignatureDetector  # noqa: E402
from src.vlm.skills.table_extractor import TableExtractor  # noqa: E402


def _stub_vlm(content: str) -> MagicMock:
    client = MagicMock()
    client.call.return_value = {"content": content, "logprobs": None, "finish_reason": "stop"}
    return client


def _dummy_task(region_type: RegionType = RegionType.SEAL) -> SkillTask:
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    return SkillTask(
        region_id="r_test",
        region_type=region_type,
        cropped_image=img,
        pixel_budget=560,
    )


# ─────── SealReader ───────

def test_seal_reader_parses_json():
    vlm = _stub_vlm(json.dumps({"text": "국회의장", "confidence": 0.92}))
    reader = SealReader(vlm_client=vlm)
    result = reader.run(_dummy_task())
    assert result.skill_name == "S4"
    assert result.content == "국회의장"
    assert result.confidence == pytest.approx(0.92)
    assert result.content_type == "seal"
    vlm.call.assert_called_once()


def test_seal_reader_handles_malformed_output():
    vlm = _stub_vlm("국회의장 (텍스트만)")
    reader = SealReader(vlm_client=vlm)
    result = reader.run(_dummy_task())
    assert result.content == "국회의장 (텍스트만)"
    assert result.confidence == 0.0


def test_seal_reader_accepts_code_fence():
    vlm = _stub_vlm('```json\n{"text": "국회의장", "confidence": 0.91}\n```')
    reader = SealReader(vlm_client=vlm)
    result = reader.run(_dummy_task())
    assert result.content == "국회의장"
    assert result.confidence == pytest.approx(0.91)


def test_seal_reader_vlm_failure_returns_empty():
    vlm = MagicMock()
    vlm.call.side_effect = RuntimeError("boom")
    reader = SealReader(vlm_client=vlm)
    result = reader.run(_dummy_task())
    assert result.content == ""
    assert result.confidence == 0.0
    assert any("vlm_call_failed" in w for w in result.warnings)


# ─────── SignatureDetector ───────

def test_signature_detector_true():
    vlm = _stub_vlm(json.dumps({"signature_present": True, "confidence": 0.88}))
    det = SignatureDetector(vlm_client=vlm)
    result = det.run(_dummy_task(RegionType.SIGNATURE))
    assert result.skill_name == "S6"
    body = json.loads(result.content)
    assert body["signature_present"] is True
    assert result.confidence == pytest.approx(0.88)
    # pixel_budget=140 전달 확인
    _, kwargs = vlm.call.call_args
    assert kwargs["pixel_budget"] == 140


def test_signature_detector_false():
    vlm = _stub_vlm(json.dumps({"signature_present": False, "confidence": 0.95}))
    det = SignatureDetector(vlm_client=vlm)
    result = det.run(_dummy_task(RegionType.SIGNATURE))
    assert json.loads(result.content)["signature_present"] is False


def test_signature_detector_parse_failure():
    vlm = _stub_vlm("not json")
    det = SignatureDetector(vlm_client=vlm)
    result = det.run(_dummy_task(RegionType.SIGNATURE))
    assert json.loads(result.content)["signature_present"] is False
    assert result.confidence == 0.0


# ─────── TableExtractor ───────

PASS1_RESPONSE = json.dumps(
    {
        "analysis": "결재란 3칸",
        "table_type": "approval",
        "structure_confidence": 0.9,
        "cells": [
            {"row": 0, "col": 0, "role": "담당", "bbox": [0, 0, 100, 50], "content_type": "printed"},
            {"row": 0, "col": 1, "role": "팀장", "bbox": [100, 0, 200, 50], "content_type": "signature"},
            {"row": 0, "col": 2, "role": "부서장", "bbox": [200, 0, 300, 50], "content_type": "empty"},
        ],
    }
)


def test_table_extractor_pass1_structure():
    vlm = _stub_vlm(PASS1_RESPONSE)
    ext = TableExtractor(vlm_client=vlm)
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    structure = ext.pass1_structure(img, region_id="tbl_0")
    assert structure.table_type == "approval"
    assert len(structure.cells) == 3
    assert structure.cells[1].content_type == "signature"
    assert structure.structure_confidence == pytest.approx(0.9)


def test_table_extractor_pass2_skips_empty_and_routes():
    ext = TableExtractor(vlm_client=MagicMock())
    structure = TableStructure(
        region_id="tbl_0",
        table_type="approval",
        cells=[
            TableCell(0, 0, "담당", BoundingBox(0, 0, 50, 40), "printed"),
            TableCell(0, 1, "팀장", BoundingBox(50, 0, 100, 40), "signature"),
            TableCell(0, 2, "부서장", BoundingBox(100, 0, 150, 40), "empty"),
        ],
        structure_confidence=0.9,
    )
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    tasks = ext.pass2_route(structure, img)
    # empty 셀은 건너뜀
    assert len(tasks) == 2
    assert tasks[0].region_type == RegionType.TEXT
    assert tasks[0].pixel_budget == 560
    assert tasks[1].region_type == RegionType.SIGNATURE
    assert tasks[1].pixel_budget == 140
    assert tasks[1].context == "팀장"


def test_table_extractor_pass1_handles_bad_json():
    vlm = _stub_vlm("INVALID")
    ext = TableExtractor(vlm_client=vlm)
    img = np.full((40, 40, 3), 255, dtype=np.uint8)
    structure = ext.pass1_structure(img, region_id="tbl_bad")
    assert structure.table_type == "other"
    assert structure.cells == []
    assert structure.structure_confidence == 0.0
