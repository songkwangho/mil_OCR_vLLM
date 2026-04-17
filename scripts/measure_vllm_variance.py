"""vLLM 변동성 N회 반복 측정 — 결정론성 기준치 확보.

동일 문서를 N번 반복 처리해 VLM 출력의 변동폭을 정량화한다.
temperature=0.0 설정이어도 배치 순서/prefix caching 등으로 인해
완전 재현성이 깨질 수 있음. 본 스크립트는 기준치 수립용.

비교 단위:
  1. field (field_key 단위) — raw_value 문자열 일치율, logprob std
  2. assembled_json — leaf path별 값 변동
  3. P4 overall_confidence — 평균·표준편차·범위
  4. review_required — 분류 일관성

출력:
  data/variance_reports/{doc_id}_{ts}/
    ├── run_01/snapshot.json
    ├── run_02/snapshot.json
    ├── run_03/snapshot.json
    ├── variance_report.json
    └── variance_report.md

실행:
  VLLM_BASE_URL=http://localhost:8100/v1 \
  VLLM_HEALTH_URL=http://localhost:8100/health \
  LAYOUT_SERVICE_URL=http://localhost:8082 \
  python scripts/measure_vllm_variance.py \
      --input data/raw/전역지원서_2.jpg \
      --runs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
logger = logging.getLogger("measure_variance")


# ─────────────────────────────────────────
#  유틸
# ─────────────────────────────────────────

def _detect_file_ext(path: Path) -> FileExt:
    mapping = {
        ".jpg": FileExt.JPG, ".jpeg": FileExt.JPG, ".png": FileExt.PNG,
        ".tiff": FileExt.TIFF, ".tif": FileExt.TIFF, ".pdf": FileExt.PDF,
    }
    return mapping.get(path.suffix.lower(), FileExt.PNG)


def _save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ─────────────────────────────────────────
#  Snapshot — run별 핵심 추출
# ─────────────────────────────────────────

def extract_snapshot(result: PipelineResult, run_idx: int) -> dict:
    """PipelineResult → 비교 가능한 dict 추출."""
    snap: dict[str, Any] = {
        "run_idx": run_idx,
        "doc_id": result.doc_id,
        "status": result.status.value,
        "processing_path": result.processing_path.value,
        "total_ms": round(result.total_ms, 1),
        "timings": {k: round(v, 1) for k, v in result.timings.items()},
        "errors": result.errors,
        "warnings": result.warnings,
    }

    # P1
    if result.p1_result:
        p1 = result.p1_result
        snap["p1"] = {
            "dpi": p1.dpi,
            "quality_score": round(p1.quality_score, 4),
            "sr_applied": p1.sr_applied,
        }

    # P2.5-A
    if result.p2_result:
        snap["p2"] = {
            "region_count": len(result.p2_result.regions),
            "region_types": sorted([
                (r.region_type.value if hasattr(r.region_type, "value") else str(r.region_type))
                for r in result.p2_result.regions
            ]),
        }

    # P3-A
    if result.p3a_form_type:
        snap["p3a"] = {
            "form_type": result.p3a_form_type,
            "form_confidence": round(float(result.p3a_form_confidence or 0.0), 4),
        }

    # P3-B
    if result.p3_result:
        p3 = result.p3_result
        fields_data = []
        for f in p3.fields:
            fields_data.append({
                "field_key": f.field_key,
                "raw_value": str(getattr(f, "raw_value", "")),
                "normalized_value": getattr(f, "normalized_value", None),
                "confidence": round(float(f.confidence), 4),
                "logprob": round(float(getattr(f, "logprob", 0.0) or 0.0), 4),
                "region_id": getattr(f, "region_id", None),
                "was_retried": getattr(f, "was_retried", False),
            })
        snap["p3b"] = {
            "form_type": p3.form_type.value if hasattr(p3.form_type, "value") else str(p3.form_type),
            "field_count": len(p3.fields),
            "table_count": len(p3.tables),
            "retry_count": getattr(p3, "retry_count", 0),
            "fields": fields_data,
            "assembled_json": getattr(p3, "assembled_json", None),
        }

    # P4
    if result.p4_result:
        p4 = result.p4_result
        snap["p4"] = {
            "overall_confidence": round(float(p4.overall_confidence), 4),
            "review_required": bool(p4.review_required),
            "flagged_fields": sorted(p4.flagged_fields or []),
            "error_count": len(p4.validation_errors),
        }

    # Skill stats (other 경로만)
    stats = getattr(result, "skill_stats", None)
    if stats is not None:
        snap["skill_stats"] = {
            "skill_counts": dict(stats.skill_counts),
            "pass1_tables": stats.pass1_tables,
            "pass2_tasks": stats.pass2_tasks,
        }

    return snap


# ─────────────────────────────────────────
#  Variance Analysis
# ─────────────────────────────────────────

def _collect_leaf_paths(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    """dict/list 트리를 평탄화해 (path, leaf_value) 목록 반환."""
    leaves: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves.extend(_collect_leaf_paths(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves.extend(_collect_leaf_paths(v, f"{path}[{i}]"))
    else:
        leaves.append((path, obj))
    return leaves


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(statistics.stdev(values), 6)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.mean(values), 6)


def analyze_variance(snapshots: list[dict]) -> dict:
    """N개 snapshot 비교 → 변동 통계 산출."""
    n = len(snapshots)
    report: dict[str, Any] = {
        "run_count": n,
        "doc_id": snapshots[0].get("doc_id"),
    }

    # ── 1. timing 변동 ──
    totals = [s["total_ms"] for s in snapshots if "total_ms" in s]
    report["timing"] = {
        "total_ms": {
            "values": totals,
            "mean": _safe_mean(totals),
            "std": _safe_std(totals),
            "min": min(totals) if totals else None,
            "max": max(totals) if totals else None,
        },
    }

    # ── 2. 처리 경로 일관성 ──
    paths = [s.get("processing_path") for s in snapshots]
    report["processing_path"] = {
        "values": paths,
        "consistent": len(set(paths)) == 1,
    }

    # ── 3. P3-A form_type 변동 ──
    p3a_forms = [s.get("p3a", {}).get("form_type") for s in snapshots]
    report["p3a_form_type"] = {
        "values": p3a_forms,
        "consistent": len(set(p3a_forms)) == 1,
    }

    # ── 4. P4 overall_confidence / review_required ──
    confs = [s.get("p4", {}).get("overall_confidence") for s in snapshots if s.get("p4")]
    reviews = [s.get("p4", {}).get("review_required") for s in snapshots if s.get("p4")]
    report["p4"] = {
        "overall_confidence": {
            "values": confs,
            "mean": _safe_mean([c for c in confs if c is not None]),
            "std": _safe_std([c for c in confs if c is not None]),
            "min": min(confs) if confs else None,
            "max": max(confs) if confs else None,
            "range": (max(confs) - min(confs)) if confs else None,
        },
        "review_required": {
            "values": reviews,
            "consistent": len(set(reviews)) == 1,
        },
    }

    # ── 5. field (field_key) 단위 비교 ──
    #   field_key × run index → raw_value / logprob
    field_table: dict[str, list[dict]] = {}
    for s in snapshots:
        for f in s.get("p3b", {}).get("fields", []) or []:
            key = f["field_key"]
            field_table.setdefault(key, []).append({
                "run_idx": s["run_idx"],
                "raw_value": f.get("raw_value"),
                "logprob": f.get("logprob"),
                "confidence": f.get("confidence"),
            })

    field_analysis = []
    unstable_fields = []
    for key, entries in sorted(field_table.items()):
        # run 수가 부족한 필드 (일부 run에서 누락) → 불안정으로 표시
        run_count_for_field = len(entries)
        values = [e["raw_value"] for e in entries]
        unique_values = sorted(set(values))
        logprobs = [e["logprob"] for e in entries if e["logprob"] is not None]
        consistent = len(unique_values) == 1 and run_count_for_field == n
        entry = {
            "field_key": key,
            "appeared_in_runs": run_count_for_field,
            "unique_value_count": len(unique_values),
            "unique_values": unique_values,
            "consistent": consistent,
            "logprob_mean": _safe_mean(logprobs),
            "logprob_std": _safe_std(logprobs),
        }
        field_analysis.append(entry)
        if not consistent:
            unstable_fields.append(entry)

    report["fields"] = {
        "total_fields": len(field_analysis),
        "consistent_count": sum(1 for f in field_analysis if f["consistent"]),
        "unstable_count": len(unstable_fields),
        "unstable": unstable_fields,
        "all": field_analysis,
    }

    # ── 6. assembled_json leaf path 비교 ──
    leaf_table: dict[str, list[Any]] = {}
    for s in snapshots:
        assembled = s.get("p3b", {}).get("assembled_json")
        if not isinstance(assembled, (dict, list)):
            continue
        for path, value in _collect_leaf_paths(assembled):
            leaf_table.setdefault(path, []).append(value)

    leaf_analysis = []
    unstable_leaves = []
    for path, values in sorted(leaf_table.items()):
        unique = []
        seen = set()
        for v in values:
            key_repr = json.dumps(v, ensure_ascii=False, default=str)
            if key_repr not in seen:
                seen.add(key_repr)
                unique.append(v)
        consistent = len(unique) == 1 and len(values) == n
        entry = {
            "path": path,
            "appeared_in_runs": len(values),
            "unique_count": len(unique),
            "unique_values": unique,
            "consistent": consistent,
        }
        leaf_analysis.append(entry)
        if not consistent:
            unstable_leaves.append(entry)

    report["assembled_json"] = {
        "total_leaves": len(leaf_analysis),
        "consistent_count": sum(1 for l in leaf_analysis if l["consistent"]),
        "unstable_count": len(unstable_leaves),
        "unstable": unstable_leaves,
    }

    # ── 7. 종합 판정 ──
    determinism_score = (
        report["fields"]["consistent_count"] / max(report["fields"]["total_fields"], 1)
    )
    report["summary"] = {
        "determinism_score": round(determinism_score, 4),
        "processing_path_consistent": report["processing_path"]["consistent"],
        "form_type_consistent": report["p3a_form_type"]["consistent"],
        "review_decision_consistent": report["p4"]["review_required"]["consistent"],
        "field_consistency_rate": round(determinism_score, 4),
        "conf_range": report["p4"]["overall_confidence"].get("range"),
    }

    return report


# ─────────────────────────────────────────
#  Markdown 리포트
# ─────────────────────────────────────────

def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown_report(report: dict, snapshots: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"# vLLM 변동성 측정 리포트 — {report['doc_id']}")
    lines.append("")
    lines.append(f"- run_count: **{report['run_count']}**")
    s = report["summary"]
    lines.append(f"- 결정론 스코어 (field 일치율): **{s['determinism_score']:.2%}**")
    lines.append(f"- processing_path 일관성: **{s['processing_path_consistent']}**")
    lines.append(f"- form_type 일관성: **{s['form_type_consistent']}**")
    lines.append(f"- review_required 일관성: **{s['review_decision_consistent']}**")
    conf_range = s.get("conf_range")
    if conf_range is not None:
        lines.append(f"- overall_confidence 범위: **{conf_range:.4f}**")
    lines.append("")

    # timing
    t = report["timing"]["total_ms"]
    lines.append("## 1. 타이밍 (total_ms)")
    lines.append("")
    lines.append(_md_row(["run", "total_ms"]))
    lines.append(_md_row(["---", "---"]))
    for s_i in snapshots:
        lines.append(_md_row([str(s_i["run_idx"]), f"{s_i['total_ms']:.0f}"]))
    lines.append("")
    lines.append(f"- mean: {t['mean']:.1f}ms")
    lines.append(f"- std:  {t['std']:.1f}ms")
    lines.append(f"- min/max: {t['min']:.0f} / {t['max']:.0f}ms")
    lines.append("")

    # P4 overall
    p4 = report["p4"]["overall_confidence"]
    lines.append("## 2. P4 overall_confidence")
    lines.append("")
    lines.append(f"- mean: {p4['mean']:.4f}")
    lines.append(f"- std:  {p4['std']:.4f}")
    lines.append(f"- min/max: {p4['min']} / {p4['max']}")
    lines.append(f"- range: {p4['range']}")
    lines.append("")

    # 불안정 field
    unstable_fields = report["fields"]["unstable"]
    lines.append(f"## 3. 불안정 필드 ({len(unstable_fields)}개 / 총 {report['fields']['total_fields']})")
    lines.append("")
    if not unstable_fields:
        lines.append("_모든 필드 일관됨._")
    else:
        lines.append(_md_row(["field_key", "unique_count", "appeared_in_runs", "unique_values", "logprob_std"]))
        lines.append(_md_row(["---"] * 5))
        for f in unstable_fields[:100]:
            vals_str = " / ".join(str(v)[:30] for v in f["unique_values"][:4])
            lines.append(_md_row([
                f["field_key"],
                str(f["unique_value_count"]),
                str(f["appeared_in_runs"]),
                vals_str,
                f"{f['logprob_std']:.3f}",
            ]))
        if len(unstable_fields) > 100:
            lines.append(f"_... +{len(unstable_fields) - 100} more_")
    lines.append("")

    # 불안정 leaf path
    unstable_leaves = report["assembled_json"]["unstable"]
    lines.append(
        f"## 4. assembled_json 불안정 path ({len(unstable_leaves)}개 / 총 {report['assembled_json']['total_leaves']})"
    )
    lines.append("")
    if not unstable_leaves:
        lines.append("_모든 leaf path 일관됨._")
    else:
        lines.append(_md_row(["path", "unique_count", "values"]))
        lines.append(_md_row(["---"] * 3))
        for l in unstable_leaves[:100]:
            vals_str = " / ".join(
                json.dumps(v, ensure_ascii=False, default=str)[:40]
                for v in l["unique_values"][:4]
            )
            lines.append(_md_row([l["path"], str(l["unique_count"]), vals_str]))
        if len(unstable_leaves) > 100:
            lines.append(f"_... +{len(unstable_leaves) - 100} more_")
    lines.append("")

    # run별 상태
    lines.append("## 5. run별 간이 상태")
    lines.append("")
    lines.append(_md_row(["run", "status", "path", "form_type", "field_count", "overall_conf", "review"]))
    lines.append(_md_row(["---"] * 7))
    for s_i in snapshots:
        lines.append(_md_row([
            str(s_i["run_idx"]),
            s_i.get("status", "-"),
            s_i.get("processing_path", "-"),
            str((s_i.get("p3a") or {}).get("form_type", "-")),
            str((s_i.get("p3b") or {}).get("field_count", "-")),
            str((s_i.get("p4") or {}).get("overall_confidence", "-")),
            str((s_i.get("p4") or {}).get("review_required", "-")),
        ]))
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────
#  메인
# ─────────────────────────────────────────

def run_once(pipeline: PipelineOrchestrator, image_path: Path, run_idx: int) -> dict:
    """파이프라인 1회 실행 → snapshot 반환."""
    raw_bytes = image_path.read_bytes()
    doc_id = image_path.stem
    doc_input = DocumentInput(
        doc_id=doc_id,
        raw_bytes=raw_bytes,
        file_ext=_detect_file_ext(image_path),
        source_type=SourceType.SCAN,
        dpi_hint=None,
        metadata={"source_file": image_path.name, "variance_run_idx": run_idx},
    )
    start = time.time()
    result = pipeline.process(doc_input)
    elapsed = (time.time() - start) * 1000.0

    # PDF면 PdfDocumentResult — 지원하지 않음
    if not isinstance(result, PipelineResult):
        raise RuntimeError(
            f"PDF 입력은 본 스크립트의 비교 단위를 정의하지 못했습니다. 단일 이미지 입력을 사용하세요. (got={type(result).__name__})"
        )

    snap = extract_snapshot(result, run_idx)
    snap["wall_ms"] = round(elapsed, 1)
    return snap


def main():
    parser = argparse.ArgumentParser(description="vLLM 변동성 N회 반복 측정")
    parser.add_argument("--input", required=True, help="단일 문서 경로 (이미지)")
    parser.add_argument("--runs", type=int, default=3, help="반복 실행 횟수 (기본 3)")
    parser.add_argument(
        "--output-dir",
        default="data/variance_reports",
        help="출력 기본 디렉토리",
    )
    args = parser.parse_args()

    image_path = Path(args.input).resolve()
    if not image_path.exists():
        logger.error("입력 파일 없음: %s", image_path)
        sys.exit(1)
    if image_path.suffix.lower() == ".pdf":
        logger.error("본 스크립트는 PDF 미지원 (단일 이미지 기준). 이미지 경로를 지정하세요.")
        sys.exit(1)

    output_base = (_ROOT / args.output_dir).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"{image_path.stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("출력 디렉토리: %s", run_dir)

    # 파이프라인 설정
    cfg = PipelineConfig(
        vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8100/v1"),
        vllm_health_url=os.environ.get("VLLM_HEALTH_URL", "http://localhost:8100/health"),
        layout_service_url=os.environ.get("LAYOUT_SERVICE_URL"),
        layout_model_name=os.environ.get("LAYOUT_MODEL_NAME", "PP-DocLayoutV3"),
        layout_fusion_mode=os.environ.get("LAYOUT_FUSION_MODE", "").lower() in ("true", "1", "yes"),
        model_root=os.environ.get("MODEL_ROOT", str(_ROOT / "models")),
        fallback_enabled=os.environ.get("FALLBACK_ENABLED", "false").lower() in ("true", "1", "yes"),
        fallback_base_url=os.environ.get("FALLBACK_BASE_URL", ""),
        review_queue_enabled=True,
        review_queue_db_url=f"sqlite:///{run_dir}/review_queue.db",
        db_url=f"sqlite:///{run_dir}/ocr_results.db",
    )
    pipeline = PipelineOrchestrator(cfg)

    # N회 반복
    snapshots: list[dict] = []
    for i in range(1, args.runs + 1):
        logger.info("=" * 60)
        logger.info("Run %d/%d — %s", i, args.runs, image_path.name)
        logger.info("=" * 60)
        snap = run_once(pipeline, image_path, i)
        _save_json(run_dir / f"run_{i:02d}" / "snapshot.json", snap)
        snapshots.append(snap)
        logger.info(
            "Run %d 완료: status=%s total=%.0fms p4_conf=%s review=%s fields=%s",
            i,
            snap.get("status", "?"),
            snap.get("total_ms", 0.0),
            (snap.get("p4") or {}).get("overall_confidence"),
            (snap.get("p4") or {}).get("review_required"),
            (snap.get("p3b") or {}).get("field_count"),
        )

    # 변동 분석
    report = analyze_variance(snapshots)
    _save_json(run_dir / "variance_report.json", report)

    md = render_markdown_report(report, snapshots)
    (run_dir / "variance_report.md").write_text(md, encoding="utf-8")

    # 콘솔 요약
    s = report["summary"]
    print("\n" + "=" * 80)
    print(f"  변동성 측정 결과 — {report['doc_id']} (N={report['run_count']})")
    print("=" * 80)
    print(f"  결정론 스코어          : {s['determinism_score']:.2%}")
    print(f"  processing_path 일관   : {s['processing_path_consistent']}")
    print(f"  form_type 일관         : {s['form_type_consistent']}")
    print(f"  review_required 일관   : {s['review_decision_consistent']}")
    if s.get("conf_range") is not None:
        print(f"  overall_confidence 범위 : {s['conf_range']:.4f}")
    print(f"  불안정 필드             : {report['fields']['unstable_count']} / {report['fields']['total_fields']}")
    print(f"  불안정 leaf path        : {report['assembled_json']['unstable_count']} / {report['assembled_json']['total_leaves']}")
    print(f"  출력: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
