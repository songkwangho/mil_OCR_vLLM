"""P5 — 직렬화 모듈
군수 OCR 시스템 v2 · Stage 3

v1 T11에서 이관. 입력 타입을 ValidatedResult로 변경.

Input  : ValidatedResult (P4 출력)
Output : JSON 문자열 / XML 문자열 / CSV 행 리스트

의존: json (내장), lxml (선택), pandas (선택)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from src.interfaces.enums import OutputFormat
from src.interfaces.types import FieldValue, ValidatedResult

logger = logging.getLogger(__name__)


@dataclass
class P5SerializerConfig:
    """P5 직렬화 설정."""
    formats: list[OutputFormat] = None
    json_indent: int = 2
    xml_root_tag: str = "document"

    def __post_init__(self):
        if self.formats is None:
            self.formats = [OutputFormat.JSON]


def _serialize_json(result: ValidatedResult, indent: int = 2) -> str:
    """ValidatedResult → JSON 문자열."""
    data = {
        "doc_id": result.doc_id,
        "overall_confidence": result.overall_confidence,
        "review_required": result.review_required,
        "processing_path": result.processing_path.value,
        "fields": [
            {
                "field_key": f.field_key,
                "raw_value": f.raw_value,
                "corrected_value": f.corrected_value,
                "data_type": f.data_type,
                "confidence": f.confidence,
                "is_flagged": f.is_flagged,
            }
            for f in result.fields
        ],
        "tables": [
            {
                "region_id": t.region_id,
                "html": t.html,
                "cells": t.cells,
                "confidence": t.confidence,
            }
            for t in result.tables
        ],
        "validation_errors": [
            {
                "error_id": e.error_id,
                "error_type": e.error_type.value,
                "severity": e.severity.value,
                "field_ref": e.field_ref,
                "message": e.message,
            }
            for e in result.validation_errors
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=indent, default=str)


def _serialize_xml(result: ValidatedResult, root_tag: str = "document") -> str:
    """ValidatedResult → XML 문자열."""
    try:
        from lxml import etree
    except ImportError:
        return _serialize_xml_fallback(result, root_tag)

    root = etree.Element(root_tag)
    etree.SubElement(root, "doc_id").text = result.doc_id
    etree.SubElement(root, "overall_confidence").text = str(result.overall_confidence)
    etree.SubElement(root, "review_required").text = str(result.review_required).lower()
    etree.SubElement(root, "processing_path").text = result.processing_path.value

    fields_el = etree.SubElement(root, "fields")
    for f in result.fields:
        field_el = etree.SubElement(fields_el, "field")
        etree.SubElement(field_el, "field_key").text = f.field_key
        etree.SubElement(field_el, "raw_value").text = f.raw_value
        etree.SubElement(field_el, "corrected_value").text = f.corrected_value
        etree.SubElement(field_el, "data_type").text = f.data_type
        etree.SubElement(field_el, "confidence").text = str(f.confidence)
        etree.SubElement(field_el, "is_flagged").text = str(f.is_flagged).lower()

    xml_body = etree.tostring(root, pretty_print=True, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body


def _serialize_xml_fallback(result: ValidatedResult, root_tag: str) -> str:
    """lxml 미설치 시 간이 XML."""
    lines = ['<?xml version="1.0" encoding="utf-8"?>']
    lines.append(f"<{root_tag}>")
    lines.append(f"  <doc_id>{result.doc_id}</doc_id>")
    lines.append(f"  <overall_confidence>{result.overall_confidence}</overall_confidence>")
    lines.append(f"  <fields>")
    for f in result.fields:
        lines.append(f"    <field>")
        lines.append(f"      <field_key>{f.field_key}</field_key>")
        lines.append(f"      <corrected_value>{f.corrected_value}</corrected_value>")
        lines.append(f"      <confidence>{f.confidence}</confidence>")
        lines.append(f"    </field>")
    lines.append(f"  </fields>")
    lines.append(f"</{root_tag}>")
    return "\n".join(lines)


def _serialize_csv(result: ValidatedResult) -> list[dict[str, str]]:
    """ValidatedResult → CSV 행 리스트."""
    rows = []
    for f in result.fields:
        rows.append({
            "doc_id": result.doc_id,
            "field_key": f.field_key,
            "raw_value": f.raw_value,
            "corrected_value": f.corrected_value,
            "data_type": f.data_type,
            "confidence": str(f.confidence),
            "is_flagged": str(f.is_flagged).lower(),
        })
    return rows


class P5Serializer:
    """P5 직렬화기.

    사용법:
        serializer = P5Serializer()
        json_str, xml_str, csv_rows = serializer.serialize(validated_result)
    """

    def __init__(self, cfg: Optional[P5SerializerConfig] = None):
        self.cfg = cfg or P5SerializerConfig()

    def serialize(
        self,
        result: ValidatedResult,
    ) -> tuple[Optional[str], Optional[str], list[dict[str, str]]]:
        """ValidatedResult를 지정된 형식으로 직렬화.

        Returns:
            (json_output, xml_output, csv_rows)
        """
        json_output = None
        xml_output = None
        csv_rows: list[dict[str, str]] = []

        for fmt in self.cfg.formats:
            if fmt == OutputFormat.JSON:
                json_output = _serialize_json(result, self.cfg.json_indent)
            elif fmt == OutputFormat.XML:
                xml_output = _serialize_xml(result, self.cfg.xml_root_tag)
            elif fmt == OutputFormat.CSV:
                csv_rows = _serialize_csv(result)

        logger.info(
            "P5: 직렬화 완료 (doc_id=%s, json=%s, xml=%s, csv=%d)",
            result.doc_id,
            "Y" if json_output else "N",
            "Y" if xml_output else "N",
            len(csv_rows),
        )
        return json_output, xml_output, csv_rows
