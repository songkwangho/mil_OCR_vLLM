"""v2 파이프라인 전체 실행 + 단계별 결과 저장

v1 pipeline_outputs 구조와 동일한 형태로 P1~P6 결과를 저장합니다.

출력 구조:
  data/pipeline_outputs/{timestamp}/{doc_id}/
    ├── P1/
    │   ├── preprocessed.png
    │   ├── binary.png
    │   └── result.json
    ├── P2/
    │   ├── layout_visualization.png
    │   └── result.json
    ├── P3/
    │   ├── result.json
    │   └── raw_vlm_response.json
    ├── P4/
    │   └── result.json
    ├── P5/
    │   ├── output.json
    │   ├── output.xml
    │   └── output.csv
    ├── P6/
    │   └── result.json
    └── summary.json

실행:
  VLLM_BASE_URL=http://localhost:8100/v1 \
  VLLM_HEALTH_URL=http://localhost:8100/health \
  python scripts/run_pipeline_with_outputs.py [--input-dir data/raw]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.interfaces.enums import FileExt, SourceType
from src.interfaces.types import DocumentInput
from src.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_pipeline")

# ─────────────────────────────────────────
#  단계별 결과 저장
# ─────────────────────────────────────────

def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def save_p1(out_dir: Path, result: PipelineResult):
    """P1 결과 저장: preprocessed.png, binary.png, result.json"""
    p1 = result.p1_result
    if p1 is None:
        return
    d = out_dir / "P1"

    # 이미지
    _save_image(d / "preprocessed.png", cv2.cvtColor(p1.image_array, cv2.COLOR_RGB2BGR))
    _save_image(d / "binary.png", p1.binary_array)

    # 메타
    _save_json(d / "result.json", {
        "doc_id": p1.doc_id,
        "dpi": p1.dpi,
        "original_dpi": p1.original_dpi,
        "resolution_band": p1.resolution_band.value,
        "quality_score": round(p1.quality_score, 4),
        "sr_applied": p1.sr_applied,
        "image_shape": list(p1.image_array.shape),
        "warnings": p1.warnings,
    })


def save_p2(out_dir: Path, result: PipelineResult):
    """P2 결과 저장: layout_visualization.png, result.json"""
    p2 = result.p2_result
    p1 = result.p1_result
    if p2 is None:
        return
    d = out_dir / "P2"

    # 레이아웃 시각화 (bbox 그리기)
    if p1 is not None:
        vis = cv2.cvtColor(p1.image_array.copy(), cv2.COLOR_RGB2BGR)
        colors = {
            "text": (0, 255, 0), "table": (255, 0, 0), "figure": (0, 0, 255),
            "header": (255, 255, 0), "footer": (128, 128, 0), "seal": (0, 255, 255),
            "formula": (255, 0, 255), "chart": (128, 0, 128),
        }
        for i, region in enumerate(p2.regions):
            b = region.bbox
            color = colors.get(region.region_type, (200, 200, 200))
            cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            label = f"{region.region_type}({region.confidence:.2f})"
            cv2.putText(vis, label, (b.x1, max(b.y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # 읽기 순서 번호
            order_idx = p2.reading_order.index(i) if i in p2.reading_order else -1
            if order_idx >= 0:
                cv2.putText(vis, f"#{order_idx}", (b.x2 - 30, b.y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        _save_image(d / "layout_visualization.png", vis)

    # 메타
    regions_data = []
    for r in p2.regions:
        regions_data.append({
            "region_id": r.region_id,
            "region_type": r.region_type,
            "bbox": {"x1": r.bbox.x1, "y1": r.bbox.y1, "x2": r.bbox.x2, "y2": r.bbox.y2},
            "confidence": round(r.confidence, 4),
        })
    _save_json(d / "result.json", {
        "doc_id": p2.doc_id,
        "page_width": p2.page_width,
        "page_height": p2.page_height,
        "region_count": len(p2.regions),
        "regions": regions_data,
        "reading_order": p2.reading_order,
        "analysis_mode": p2.analysis_mode.value if hasattr(p2.analysis_mode, 'value') else str(p2.analysis_mode),
        "warnings": p2.warnings,
    })


def save_p3(out_dir: Path, result: PipelineResult):
    """P3 결과 저장: result.json, raw_vlm_response.json"""
    p3 = result.p3_result
    if p3 is None:
        return
    d = out_dir / "P3"

    # 필드 정보
    fields_data = []
    for f in p3.fields:
        fields_data.append({
            "field_key": f.field_key,
            "raw_value": f.raw_value,
            "corrected_value": f.corrected_value,
            "data_type": f.data_type,
            "confidence": round(f.confidence, 4),
            "is_flagged": f.is_flagged,
        })

    # 테이블 정보
    tables_data = []
    for t in p3.tables:
        tables_data.append({
            "region_id": t.region_id,
            "html": t.html,
            "cell_count": len(t.cells),
            "confidence": round(t.confidence, 4),
        })

    # 도메인 코드
    codes_data = []
    for c in p3.domain_codes:
        codes_data.append({
            "code_type": c.code_type,
            "raw_value": c.raw_value,
            "normalized_value": c.normalized_value,
            "confidence": round(c.confidence, 4),
        })

    _save_json(d / "result.json", {
        "doc_id": p3.doc_id,
        "form_type": p3.form_type.value if hasattr(p3.form_type, 'value') else str(p3.form_type),
        "form_confidence": round(p3.form_confidence, 4),
        "schema_id": p3.schema_id,
        "field_count": len(p3.fields),
        "fields": fields_data,
        "table_count": len(p3.tables),
        "tables": tables_data,
        "domain_codes": codes_data,
        "processing_time_ms": round(p3.processing_time_ms, 1),
        "warnings": p3.warnings,
    })

    # 원본 VLM JSON 응답
    if p3.raw_json:
        try:
            raw_parsed = json.loads(p3.raw_json)
            _save_json(d / "raw_vlm_response.json", raw_parsed)
        except json.JSONDecodeError:
            (d / "raw_vlm_response.txt").write_text(p3.raw_json, encoding="utf-8")


def save_p4(out_dir: Path, result: PipelineResult):
    """P4 결과 저장: result.json"""
    p4 = result.p4_result
    if p4 is None:
        return
    d = out_dir / "P4"

    errors_data = []
    for e in p4.validation_errors:
        errors_data.append({
            "error_id": e.error_id,
            "error_type": e.error_type,
            "severity": e.severity,
            "field_ref": e.field_ref,
            "expected": e.expected,
            "actual": e.actual,
            "message": e.message,
        })

    _save_json(d / "result.json", {
        "doc_id": p4.doc_id,
        "overall_confidence": round(p4.overall_confidence, 4),
        "review_required": p4.review_required,
        "flagged_fields": p4.flagged_fields,
        "validation_error_count": len(p4.validation_errors),
        "validation_errors": errors_data,
        "processing_path": p4.processing_path.value if hasattr(p4.processing_path, 'value') else str(p4.processing_path),
    })


def save_p5(out_dir: Path, result: PipelineResult):
    """P5 결과 저장: output.json, output.xml"""
    out = result.output
    if out is None:
        return
    d = out_dir / "P5"
    d.mkdir(parents=True, exist_ok=True)

    if out.json_output:
        (d / "output.json").write_text(out.json_output, encoding="utf-8")
    if out.xml_output:
        (d / "output.xml").write_text(out.xml_output, encoding="utf-8")
    if out.csv_rows:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=out.csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(out.csv_rows)
        (d / "output.csv").write_text(buf.getvalue(), encoding="utf-8")


def save_p6(out_dir: Path, result: PipelineResult):
    """P6 결과 저장: result.json"""
    out = result.output
    if out is None:
        return
    d = out_dir / "P6"

    _save_json(d / "result.json", {
        "doc_id": out.doc_id,
        "status": out.status.value if hasattr(out.status, 'value') else str(out.status),
        "processing_path": out.processing_path.value if hasattr(out.processing_path, 'value') else str(out.processing_path),
        "form_type": out.form_type.value if hasattr(out.form_type, 'value') else str(out.form_type),
        "db_record_ids": out.db_record_ids,
        "review_queue_id": out.review_queue_id,
        "processing_ms": round(out.processing_ms, 1),
    })


def save_summary(out_dir: Path, result: PipelineResult):
    """전체 요약 저장: summary.json"""
    summary = {
        "doc_id": result.doc_id,
        "status": result.status.value,
        "processing_path": result.processing_path.value,
        "timings": result.timings,
        "total_ms": round(result.total_ms, 1),
        "error_count": len(result.errors),
        "errors": result.errors,
        "warning_count": len(result.warnings),
        "warnings": result.warnings,
    }

    # P1 요약
    if result.p1_result:
        summary["P1"] = {
            "quality_score": round(result.p1_result.quality_score, 4),
            "dpi": result.p1_result.dpi,
            "sr_applied": result.p1_result.sr_applied,
        }
    # P2 요약
    if result.p2_result:
        summary["P2"] = {
            "region_count": len(result.p2_result.regions),
            "mode": result.p2_result.analysis_mode.value if hasattr(result.p2_result.analysis_mode, 'value') else str(result.p2_result.analysis_mode),
        }
    # P3 요약
    if result.p3_result:
        summary["P3"] = {
            "form_type": result.p3_result.form_type.value if hasattr(result.p3_result.form_type, 'value') else str(result.p3_result.form_type),
            "form_confidence": round(result.p3_result.form_confidence, 4),
            "field_count": len(result.p3_result.fields),
            "table_count": len(result.p3_result.tables),
            "processing_time_ms": round(result.p3_result.processing_time_ms, 1),
        }
    # P4 요약
    if result.p4_result:
        summary["P4"] = {
            "overall_confidence": round(result.p4_result.overall_confidence, 4),
            "review_required": result.p4_result.review_required,
            "flagged_count": len(result.p4_result.flagged_fields),
            "error_count": len(result.p4_result.validation_errors),
        }

    _save_json(out_dir / "summary.json", summary)


# ─────────────────────────────────────────
#  메인
# ─────────────────────────────────────────

def _detect_file_ext(path: Path) -> FileExt:
    suffix = path.suffix.lower()
    mapping = {".jpg": FileExt.JPG, ".jpeg": FileExt.JPG, ".png": FileExt.PNG,
               ".tiff": FileExt.TIFF, ".tif": FileExt.TIFF, ".pdf": FileExt.PDF}
    return mapping.get(suffix, FileExt.PNG)


def run_single(pipeline: PipelineOrchestrator, image_path: Path, run_dir: Path):
    """단일 문서 파이프라인 실행 + 결과 저장."""
    doc_id = image_path.stem
    logger.info("=" * 60)
    logger.info("문서 처리 시작: %s", doc_id)
    logger.info("=" * 60)

    raw_bytes = image_path.read_bytes()
    doc_input = DocumentInput(
        doc_id=doc_id,
        raw_bytes=raw_bytes,
        file_ext=_detect_file_ext(image_path),
        source_type=SourceType.SCAN,
        dpi_hint=None,
        metadata={"source_file": image_path.name},
    )

    result = pipeline.run(doc_input)

    # 결과 저장
    doc_dir = run_dir / doc_id
    save_p1(doc_dir, result)
    save_p2(doc_dir, result)
    save_p3(doc_dir, result)
    save_p4(doc_dir, result)
    save_p5(doc_dir, result)
    save_p6(doc_dir, result)
    save_summary(doc_dir, result)

    return result


def main():
    parser = argparse.ArgumentParser(description="v2 파이프라인 + 단계별 출력 저장")
    parser.add_argument("--input-dir", default="data/raw", help="입력 이미지 디렉토리")
    parser.add_argument("--output-dir", default="data/pipeline_outputs", help="출력 기본 디렉토리")
    args = parser.parse_args()

    input_dir = _ROOT / args.input_dir
    output_base = _ROOT / args.output_dir

    # 이미지 파일 수집
    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in image_exts)

    if not images:
        logger.error("입력 이미지 없음: %s", input_dir)
        sys.exit(1)

    logger.info("입력 이미지 %d개 발견: %s", len(images), input_dir)

    # 실행 디렉토리 (타임스탬프)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("출력 디렉토리: %s", run_dir)

    # 파이프라인 설정
    cfg = PipelineConfig(
        vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8100/v1"),
        vllm_health_url=os.environ.get("VLLM_HEALTH_URL", "http://localhost:8100/health"),
        fallback_enabled=True,
        review_queue_enabled=True,
        review_queue_db_url=f"sqlite:///{run_dir}/review_queue.db",
        db_url=f"sqlite:///{run_dir}/ocr_results.db",
    )

    pipeline = PipelineOrchestrator(cfg)

    # 실행
    results = []
    for img_path in images:
        result = run_single(pipeline, img_path, run_dir)
        results.append(result)

    # 전체 요약
    total_summary = {
        "timestamp": timestamp,
        "document_count": len(results),
        "documents": [],
    }
    for r in results:
        total_summary["documents"].append({
            "doc_id": r.doc_id,
            "status": r.status.value,
            "processing_path": r.processing_path.value,
            "total_ms": round(r.total_ms, 1),
            "error_count": len(r.errors),
            "warning_count": len(r.warnings),
        })

    _save_json(run_dir / "run_summary.json", total_summary)
    logger.info("=" * 60)
    logger.info("전체 완료: %d문서, 출력: %s", len(results), run_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
