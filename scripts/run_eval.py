"""mil_OCR_v2 트라이얼 평가 — ground truth 없이 파이프라인 내부 지표 기반 delta + verdict.

지표 추출(P4/P6/summary/run_summary/P2.5A5/S7) → 직전 트라이얼과 per-doc/aggregate
비교 → verdict(IMPROVED/REGRESSED/MIXED/NEUTRAL) 판정 → 콘솔 리포트 + JSON/HTML 저장.

사용 예:
    # 파이프라인 실행 + 직전 트라이얼과 비교
    python scripts/run_eval.py --input-dir data/raw/

    # 특정 트라이얼과 비교
    python scripts/run_eval.py --input-dir data/raw/ --prev-trial 20260417_052600

    # 파이프라인 재실행 없이 기존 output 평가
    python scripts/run_eval.py \
        --pipeline-output-dir data/pipeline_outputs/20260418_093012

    # form_type 필터
    python scripts/run_eval.py --input-dir data/raw/ --form-type equipment_checklist

저장 경로(통합):
    data/pipeline_outputs/{ts}/
        ├── run_summary.json, P1~P6/..., S7/...      (파이프라인 산출물)
        ├── eval_summary.json                        (평가 요약)
        ├── report.html                              (평가 HTML 리포트)
        └── {doc_id}/metrics.json                    (문서별 지표)
    data/pipeline_outputs/latest  →  {ts}            (심볼릭 링크)
    data/eval_results/
        ├── latest  →  ../pipeline_outputs/{ts}      (심볼릭 링크)
        └── history.json                             (트라이얼 인덱스)

데이터 흐름:
    (--input-dir) → run_pipeline_with_outputs.py 호출 → pipeline_outputs/{ts}
        ↓
    pipeline_outputs/{ts} 또는 --pipeline-output-dir
        ↓
    per-doc extract_metrics() → per-doc deltas vs prev-trial(history.json)
        ↓
    aggregate_metrics() → aggregate deltas → verdict
        ↓
    pipeline_outputs/{ts}/ 안에 eval_summary.json + report.html + metrics.json 저장
    pipeline_outputs/latest, eval_results/latest, eval_results/history.json 갱신
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_eval")


# ─────────────────────────────────────────────
#  지표 방향 — 어느 쪽 이동이 긍정적인지
# ─────────────────────────────────────────────

METRIC_DIRECTION: dict[str, str] = {
    "overall_confidence":     "up",
    "confidence_mean":        "up",
    "validation_error_count": "down",
    "review_queued":          "down",
    "retry_count":            "down",
    "warning_count":          "down",
    "s7_fallback":            "down",
    "processing_ms":          "down",
    "low_conf_count":         "down",
    "augmented_count":        "stable",
}

# verdict 판정에 사용하는 주요 지표
PRIMARY_METRICS = ["overall_confidence", "review_queued", "validation_error_count"]


VERDICT_ICON = {"IMPROVED": "[OK]", "REGRESSED": "[!!]", "MIXED": "[~]", "NEUTRAL": "[=]"}
SIGN_ICON = {
    "positive": "[+]",
    "negative": "[-]",
    "neutral":  "[=]",
    "warning":  "[!]",
    "unknown":  "[?]",
}


# ─────────────────────────────────────────────
#  JSON 유틸
# ─────────────────────────────────────────────

def load_json_safe(path: Path) -> Optional[dict]:
    """파일 없거나 파싱 실패 시 None 반환."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _find_doc_in_run_summary(run_summary: dict, doc_id: str) -> Optional[dict]:
    """run_summary.json의 documents 배열에서 doc_id 매칭 (PDF 페이지 포함)."""
    for doc in run_summary.get("documents", []):
        if doc.get("doc_id") == doc_id:
            return doc
        for page in doc.get("pdf_pages", []) or []:
            if page.get("doc_id") == doc_id:
                return page
    return None


# ─────────────────────────────────────────────
#  지표 추출
# ─────────────────────────────────────────────

def _parse_sub_confidences(raw: list) -> list[float]:
    """P4/result.json의 sub_confidences → confidence 값 추출.

    실제 저장 형태는 list of dict:
      [{"path":..., "type":..., "confidence":...}, ...]
    list of list 형태도 방어적으로 처리.
    """
    if not raw:
        return []
    out: list[float] = []
    for item in raw:
        if isinstance(item, dict):
            v = item.get("confidence")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            v = item[2]
        else:
            v = None
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def extract_metrics(doc_output_dir: Path, run_summary_path: Path) -> dict:
    """pipeline_outputs/{ts}/{doc_id}/ 에서 지표 추출."""
    metrics: dict[str, Any] = {}

    # P4/result.json — 신뢰도, 검증 오류, sub_confidences
    p4 = load_json_safe(doc_output_dir / "P4" / "result.json") or {}
    metrics["overall_confidence"] = p4.get("overall_confidence", 0.0)
    metrics["validation_error_count"] = len(p4.get("validation_errors", []))
    metrics["review_required"] = p4.get("review_required", False)

    raw_sub = p4.get("sub_confidences", []) or []
    metrics["sub_confidence_count"] = len(raw_sub)
    confs = _parse_sub_confidences(raw_sub)
    if confs:
        metrics["confidence_min"] = round(min(confs), 4)
        metrics["confidence_mean"] = round(sum(confs) / len(confs), 4)
        metrics["low_conf_count"] = sum(1 for c in confs if c < 0.75)
    else:
        metrics["confidence_min"] = None
        metrics["confidence_mean"] = None
        metrics["low_conf_count"] = 0

    # P6/result.json — 검토 큐 적재 여부
    p6 = load_json_safe(doc_output_dir / "P6" / "result.json") or {}
    metrics["review_queued"] = p6.get("review_queue_id") is not None

    # summary.json — form_type, processing_path, total_ms, warnings
    summary = load_json_safe(doc_output_dir / "summary.json") or {}
    metrics["form_type"] = summary.get("form_type") or (
        summary.get("P3", {}).get("form_type") if isinstance(summary.get("P3"), dict) else None
    )
    metrics["processing_path"] = summary.get("processing_path")
    metrics["processing_ms"] = summary.get("total_ms", 0.0)
    metrics["warning_count"] = len(summary.get("warnings", []) or [])

    # run_summary.json에서 retry stats 조회
    run_summary = load_json_safe(run_summary_path) or {}
    doc_entry = _find_doc_in_run_summary(run_summary, doc_output_dir.name)
    retry_stats = (doc_entry or {}).get("retry_stats", {}) or {}
    metrics["retry_count"] = retry_stats.get("retry_count", 0)
    metrics["retry_ms"] = retry_stats.get("retry_ms", 0.0)

    # P2.5A5/result.json — military TemplateAugmentor (없으면 None)
    p2a5 = load_json_safe(doc_output_dir / "P2.5A5" / "result.json")
    if p2a5 is None:
        # run_pipeline_with_outputs는 P2.5A 내 augmented_count를 기록
        p2a = load_json_safe(doc_output_dir / "P2.5A" / "result.json")
        metrics["augmented_count"] = (
            p2a.get("template_augmentor", {}).get("augmented_count")
            if p2a and isinstance(p2a.get("template_augmentor"), dict) else None
        )
    else:
        metrics["augmented_count"] = p2a5.get("augmented_count")

    # S7/result.json — other 경로 집계 성공 여부
    s7 = load_json_safe(doc_output_dir / "S7" / "result.json")
    if s7 is None:
        metrics["s7_fallback"] = None
    else:
        # S7은 assembled_json이 None이면 폴백 상태로 간주
        metrics["s7_fallback"] = s7.get("assembled_json") is None

    return metrics


def _is_doc_dir(p: Path, run_summary_path: Path) -> bool:
    """doc 단위 디렉토리 판별 — summary.json 또는 P4/ 존재."""
    if not p.is_dir():
        return False
    if p.name in ("warmup_timings.json",):
        return False
    return (p / "summary.json").exists() or (p / "P4").is_dir() or (p / "p01").is_dir()


def _collect_doc_dirs(pipeline_output_dir: Path) -> list[Path]:
    """pipeline_outputs/{ts}/ 하위 doc 디렉토리 목록.

    PDF는 문서 디렉토리 + p01/p02/... 페이지 하위 디렉토리가 모두 존재.
    페이지별로 지표가 분리되어 있으므로 페이지 단위로 수집하고,
    단일 이미지 doc은 그대로 포함.
    """
    result: list[Path] = []
    for child in sorted(pipeline_output_dir.iterdir()):
        if not child.is_dir():
            continue

        # PDF: pdf_summary.json 존재 → 페이지 하위 디렉토리 수집
        if (child / "pdf_summary.json").exists():
            for page in sorted(child.iterdir()):
                if page.is_dir() and page.name.startswith("p") and (page / "P4").is_dir():
                    result.append(page)
            continue

        # 단일 이미지
        if (child / "P4").is_dir() or (child / "summary.json").exists():
            result.append(child)
    return result


def _doc_identifier(doc_dir: Path, pipeline_root: Path) -> str:
    """doc_dir → eval 상에서 고유 식별자 (PDF 페이지 포함)."""
    rel = doc_dir.relative_to(pipeline_root)
    return str(rel)


# ─────────────────────────────────────────────
#  Delta + Verdict
# ─────────────────────────────────────────────

def _to_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def calc_delta(before: dict, after: dict) -> dict:
    """지표별 before/after/delta/sign 산출."""
    deltas: dict[str, dict[str, Any]] = {}
    for key, direction in METRIC_DIRECTION.items():
        b_raw = before.get(key) if before else None
        a_raw = after.get(key) if after else None
        b = _to_number(b_raw)
        a = _to_number(a_raw)

        if b is None or a is None:
            deltas[key] = {
                "before": b_raw, "after": a_raw,
                "delta": None, "sign": "unknown",
            }
            continue

        delta = a - b
        if delta == 0:
            sign = "neutral"
        elif direction == "up":
            sign = "positive" if delta > 0 else "negative"
        elif direction == "down":
            sign = "positive" if delta < 0 else "negative"
        else:  # stable
            sign = "neutral" if abs(delta) < 1 else "warning"

        deltas[key] = {
            "before": round(b, 4) if isinstance(b_raw, float) else b_raw,
            "after": round(a, 4) if isinstance(a_raw, float) else a_raw,
            "delta": round(delta, 4),
            "sign": sign,
        }
    return deltas


def determine_verdict(deltas: dict) -> str:
    """PRIMARY_METRICS 기반 판정.

    IMPROVED  — primary 모두 positive 또는 neutral, 최소 하나는 positive
    REGRESSED — primary 전부 negative
    MIXED     — primary에 positive와 negative 혼재
    NEUTRAL   — primary 모두 neutral 또는 비교 대상 없음
    """
    signs: list[str] = []
    for m in PRIMARY_METRICS:
        s = deltas.get(m, {}).get("sign")
        if s not in (None, "unknown"):
            signs.append(s)
    if not signs:
        return "NEUTRAL"
    has_pos = "positive" in signs
    has_neg = "negative" in signs
    if has_pos and has_neg:
        return "MIXED"
    if has_neg:
        return "REGRESSED"
    if has_pos:
        return "IMPROVED"
    return "NEUTRAL"


def aggregate_metrics(per_doc_metrics: list[dict]) -> dict:
    """문서별 지표를 평균으로 집계."""
    keys = [k for k in METRIC_DIRECTION if k != "augmented_count"]
    agg: dict[str, Any] = {}
    for key in keys:
        vals = []
        for m in per_doc_metrics:
            v = m.get(key)
            n = _to_number(v)
            if n is not None:
                vals.append(n)
        agg[key] = round(sum(vals) / len(vals), 4) if vals else None
    return agg


# ─────────────────────────────────────────────
#  트라이얼 히스토리 (eval_results/history.json) + 이전 트라이얼 선택
# ─────────────────────────────────────────────

def _load_history(eval_results_dir: Path) -> list[dict]:
    """eval_results/history.json → 최신순 리스트. 없으면 빈 리스트."""
    path = eval_results_dir / "history.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _update_history(
    eval_results_dir: Path,
    summary: dict,
    pipeline_output_dir: Path,
) -> None:
    """history.json에 트라이얼 항목 추가. 동일 trial은 교체, 최신순 정렬."""
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    history = _load_history(eval_results_dir)

    entry = {
        "trial":               summary.get("trial"),
        "verdict":             summary.get("verdict"),
        "sample_count":        summary.get("sample_count"),
        "evaluated_at":        summary.get("evaluated_at"),
        "pipeline_output_dir": str(pipeline_output_dir),
    }
    history = [h for h in history if h.get("trial") != entry["trial"]]
    history.insert(0, entry)

    (eval_results_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_prev_summary(
    pipeline_outputs_dir: Path,
    eval_results_dir: Path,
    prev_trial: Optional[str],
    current_ts: str,
) -> tuple[Optional[dict], Optional[str]]:
    """이전 트라이얼 eval_summary.json 로드 + trial 이름 반환.

    prev_trial 지정(구체 타임스탬프):
        pipeline_outputs/{prev_trial}/eval_summary.json 직접 로드
    prev_trial == 'latest' 또는 None:
        eval_results/history.json에서 현재 trial보다 이전의 가장 최근 항목 탐색
    """
    if prev_trial and prev_trial != "latest":
        path = pipeline_outputs_dir / prev_trial / "eval_summary.json"
        return load_json_safe(path), prev_trial

    for entry in _load_history(eval_results_dir):
        trial = entry.get("trial")
        if not trial or trial >= current_ts:
            continue
        prev_dir = Path(entry.get("pipeline_output_dir", ""))
        summary = load_json_safe(prev_dir / "eval_summary.json")
        if summary is not None:
            return summary, trial
    return None, None


# ─────────────────────────────────────────────
#  파이프라인 실행
# ─────────────────────────────────────────────

def run_pipeline(input_dir: Path) -> Path:
    """run_pipeline_with_outputs.py 호출 → pipeline_outputs/{ts} 반환."""
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "run_pipeline_with_outputs.py"),
        "--input-dir", str(input_dir),
    ]
    logger.info("파이프라인 실행: %s", " ".join(cmd))

    # 실행 직전 타임스탬프 기록 → 실행 후 해당 이후 생성된 디렉토리 찾기
    before = {p.name for p in (_ROOT / "data" / "pipeline_outputs").iterdir()} \
        if (_ROOT / "data" / "pipeline_outputs").is_dir() else set()

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_ROOT))
    subprocess.run(cmd, check=True, env=env)

    after = {p.name for p in (_ROOT / "data" / "pipeline_outputs").iterdir()}
    new_dirs = sorted(after - before)
    if not new_dirs:
        raise RuntimeError("파이프라인 실행 후 새 출력 디렉토리가 없습니다")
    return _ROOT / "data" / "pipeline_outputs" / new_dirs[-1]


# ─────────────────────────────────────────────
#  심볼릭 링크 / 결과 저장
# ─────────────────────────────────────────────

def _update_symlink(link_path: Path, target: str) -> None:
    """심볼릭 링크 생성/갱신 — 상대 경로 문자열을 target으로 사용."""
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target)
    except OSError as e:
        logger.warning("심볼릭 링크 갱신 실패 (%s → %s): %s", link_path, target, e)


def save_eval_results(
    summary: dict,
    pipeline_output_dir: Path,
    eval_results_dir: Path,
) -> Path:
    """평가 결과를 pipeline_output_dir 안에 저장 + eval_results 메타 갱신.

    저장:
      - pipeline_outputs/{ts}/eval_summary.json
      - pipeline_outputs/{ts}/report.html
      - pipeline_outputs/{ts}/{doc_id}/metrics.json  (슬래시 포함 doc_id 그대로 유지)
    심볼릭 링크:
      - pipeline_outputs/latest       → {ts}
      - eval_results/latest           → ../pipeline_outputs/{ts}
    인덱스:
      - eval_results/history.json     (트라이얼 추가)

    Returns:
        report.html 경로.
    """
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    # eval_summary.json
    save_json(pipeline_output_dir / "eval_summary.json", summary)

    # 문서별 metrics.json — pipeline_outputs/{ts}/{doc_id}/metrics.json
    for doc in summary.get("per_doc", []):
        doc_id = doc.get("doc_id") or ""
        # PDF 페이지는 "doc/p01" 형식 — 이미 pipeline_outputs에 동일 구조 존재
        doc_dir = pipeline_output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        save_json(doc_dir / "metrics.json", doc)

    # HTML 리포트
    report_path = pipeline_output_dir / "report.html"
    report_path.write_text(
        _render_html_report(summary),
        encoding="utf-8",
    )

    # pipeline_outputs/latest → {ts}
    _update_symlink(
        pipeline_output_dir.parent / "latest",
        pipeline_output_dir.name,
    )

    # eval_results/latest → ../pipeline_outputs/{ts}
    _update_symlink(
        eval_results_dir / "latest",
        f"../pipeline_outputs/{pipeline_output_dir.name}",
    )

    # eval_results/history.json
    _update_history(eval_results_dir, summary, pipeline_output_dir)

    return report_path


# ─────────────────────────────────────────────
#  HTML 리포트
# ─────────────────────────────────────────────

def _render_html_report(summary: dict) -> str:
    """간단한 self-contained HTML 리포트 — 집계 + per-doc 표."""
    from html import escape

    verdict = summary.get("verdict", "NEUTRAL")
    prev = summary.get("prev_trial") or "없음 (첫 트라이얼)"
    trial = summary.get("trial", "")
    evaluated = summary.get("evaluated_at", "")
    sample_count = summary.get("sample_count", 0)

    verdict_color = {
        "IMPROVED": "#1b8a5a",
        "REGRESSED": "#c0392b",
        "MIXED": "#d68910",
        "NEUTRAL": "#566573",
    }.get(verdict, "#566573")

    sign_color = {
        "positive": "#1b8a5a",
        "negative": "#c0392b",
        "neutral":  "#566573",
        "warning":  "#d68910",
        "unknown":  "#95a5a6",
    }

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return f"{v:.4f}"
        return escape(str(v))

    def _delta_cell(d: dict) -> str:
        sign = d.get("sign", "unknown")
        color = sign_color.get(sign, "#95a5a6")
        dv = d.get("delta")
        if isinstance(dv, float):
            dv_str = f"{dv:+.4f}"
        elif isinstance(dv, int):
            dv_str = f"{dv:+d}"
        else:
            dv_str = _fmt(dv)
        return (
            f'<td style="color:{color}">{_fmt(d.get("before"))}</td>'
            f'<td style="color:{color}">{_fmt(d.get("after"))}</td>'
            f'<td style="color:{color};font-weight:bold">{dv_str}</td>'
            f'<td style="color:{color}">{escape(sign)}</td>'
        )

    agg_rows = "\n".join(
        f"<tr><td>{escape(k)}</td>{_delta_cell(v)}</tr>"
        for k, v in (summary.get("aggregate") or {}).items()
    )

    doc_rows = []
    for doc in summary.get("per_doc", []):
        dv = doc.get("doc_verdict", "NEUTRAL")
        dv_color = {
            "IMPROVED": "#1b8a5a",
            "REGRESSED": "#c0392b",
            "MIXED": "#d68910",
            "NEUTRAL": "#566573",
        }.get(dv, "#566573")
        conf = doc.get("deltas", {}).get("overall_confidence", {})
        b, a = conf.get("before"), conf.get("after")
        conf_str = (
            f"{_fmt(b)} → {_fmt(a)}" if b is not None or a is not None else "-"
        )
        negatives = ", ".join(
            k for k, v in (doc.get("deltas") or {}).items()
            if v.get("sign") == "negative"
        ) or "-"
        doc_rows.append(
            f'<tr><td><span style="color:{dv_color};font-weight:bold">{escape(dv)}</span></td>'
            f'<td>{escape(doc.get("doc_id", ""))}</td>'
            f'<td>{escape(doc.get("form_type") or "")}</td>'
            f'<td>{escape(doc.get("processing_path") or "")}</td>'
            f'<td>{conf_str}</td>'
            f'<td>{escape(negatives)}</td></tr>'
        )
    doc_rows_html = "\n".join(doc_rows)

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>mil_OCR_v2 평가 리포트 — {escape(trial)}</title>
<style>
  body {{ font-family: -apple-system, "Helvetica Neue", sans-serif; margin: 2em; color: #2c3e50; }}
  h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #bdc3c7; padding-bottom: 0.2em; margin-top: 2em; }}
  .verdict {{ display: inline-block; padding: 0.4em 1em; border-radius: 4px;
              color: white; font-weight: bold; background: {verdict_color}; }}
  .meta {{ color: #566573; font-size: 0.9em; margin: 0.5em 0 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5em; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 0.5em 0.8em; text-align: left;
            font-family: "Menlo", "Consolas", monospace; font-size: 0.9em; }}
  th {{ background: #ecf0f1; font-weight: 600; }}
  td:first-child {{ font-weight: 600; background: #fafafa; }}
  tbody tr:hover {{ background: #fcf6e7; }}
</style>
</head>
<body>
  <h1>mil_OCR_v2 평가 리포트</h1>
  <div class="meta">
    <span class="verdict">{escape(verdict)}</span>
    &nbsp;Trial <code>{escape(trial)}</code>
    vs <code>{escape(prev)}</code>
    · {evaluated} · 샘플 {sample_count}건
  </div>

  <h2>집계 지표 (aggregate)</h2>
  <table>
    <thead><tr><th>지표</th><th>이전</th><th>현재</th><th>delta</th><th>sign</th></tr></thead>
    <tbody>
{agg_rows}
    </tbody>
  </table>

  <h2>문서별 요약 (per_doc)</h2>
  <table>
    <thead><tr>
      <th>verdict</th><th>doc_id</th><th>form_type</th><th>path</th>
      <th>confidence</th><th>악화 지표</th>
    </tr></thead>
    <tbody>
{doc_rows_html}
    </tbody>
  </table>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  콘솔 리포트
# ─────────────────────────────────────────────

def _fmt_cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_report(summary: dict) -> None:
    prev = summary.get("prev_trial") or "없음 (첫 트라이얼)"
    bar = "━" * 72
    print(bar)
    print(f"  mil_OCR_v2 평가 리포트")
    print(f"  Trial : {summary['trial']}")
    print(f"  vs    : {prev}")
    print(f"  샘플  : {summary['sample_count']}건")
    print(bar)

    header = f"  {'지표':<26} {'이전':>9} {'현재':>9} {'delta':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, d in summary["aggregate"].items():
        icon = SIGN_ICON.get(d.get("sign"), "?")
        b = _fmt_cell(d.get("before"))
        a = _fmt_cell(d.get("after"))
        dv = _fmt_cell(d.get("delta"))
        if isinstance(d.get("delta"), (int, float)):
            dv = f"{d['delta']:+.3f}" if isinstance(d["delta"], float) else f"{d['delta']:+d}"
        print(f"  {key:<26} {b:>9} {a:>9} {dv:>10}  {icon}")

    print()
    print("  문서별 요약")
    for doc in summary["per_doc"]:
        icon = VERDICT_ICON.get(doc["doc_verdict"], "?")
        conf = doc["deltas"].get("overall_confidence", {})
        b = conf.get("before")
        a = conf.get("after")
        if isinstance(b, float) and isinstance(a, float):
            conf_str = f"conf {b:.3f}->{a:.3f}"
        elif isinstance(a, float):
            conf_str = f"conf -->{a:.3f}"
        else:
            conf_str = ""
        negatives = [
            k for k, v in doc["deltas"].items()
            if v.get("sign") == "negative"
        ]
        neg_str = f"  | 악화: {', '.join(negatives)}" if negatives else ""
        print(f"  {icon} {doc['doc_verdict']:<9} {doc['doc_id']:<40} {conf_str}{neg_str}")

    print()
    verdict = summary["verdict"]
    print(f"  verdict: {VERDICT_ICON.get(verdict)} {verdict}")
    print(bar)


# ─────────────────────────────────────────────
#  평가 수행
# ─────────────────────────────────────────────

def evaluate(
    pipeline_output_dir: Path,
    eval_results_dir: Path,
    prev_trial_arg: Optional[str],
    form_type_filter: Optional[str],
) -> tuple[dict, Path]:
    """파이프라인 출력 → 지표 추출 + 직전 트라이얼과 delta 산출.

    - trial 식별자는 pipeline_output_dir.name (예: "20260418_133650")을 사용.
    - 결과물(eval_summary/report.html/metrics.json)은 pipeline_output_dir 안에 저장.
    - eval_results/history.json + latest 심볼릭 링크 갱신.
    """
    trial_ts = pipeline_output_dir.name
    run_summary_path = pipeline_output_dir / "run_summary.json"
    doc_dirs = _collect_doc_dirs(pipeline_output_dir)
    logger.info("평가 대상 문서: %d건 (%s)", len(doc_dirs), pipeline_output_dir)

    # 문서별 지표 추출
    per_doc_metrics: list[dict] = []
    for doc_dir in doc_dirs:
        metrics = extract_metrics(doc_dir, run_summary_path)
        if form_type_filter and metrics.get("form_type") != form_type_filter:
            continue
        doc_id = _doc_identifier(doc_dir, pipeline_output_dir)
        per_doc_metrics.append({
            "doc_id": doc_id,
            "form_type": metrics.get("form_type"),
            "processing_path": metrics.get("processing_path"),
            "metrics": metrics,
        })

    # 직전 트라이얼 로드 (history.json 기반)
    prev_summary, prev_trial_name = load_prev_summary(
        pipeline_outputs_dir=pipeline_output_dir.parent,
        eval_results_dir=eval_results_dir,
        prev_trial=prev_trial_arg,
        current_ts=trial_ts,
    )

    # per-doc deltas
    prev_doc_map: dict[str, dict] = {}
    if prev_summary:
        for pd in prev_summary.get("per_doc", []):
            prev_doc_map[pd["doc_id"]] = pd

    per_doc_out: list[dict] = []
    for record in per_doc_metrics:
        prev = prev_doc_map.get(record["doc_id"], {}).get("metrics_after") if prev_doc_map else None
        deltas = calc_delta(prev or {}, record["metrics"])
        doc_verdict = determine_verdict(deltas) if prev else "NEUTRAL"
        per_doc_out.append({
            "doc_id": record["doc_id"],
            "form_type": record["form_type"],
            "processing_path": record["processing_path"],
            "doc_verdict": doc_verdict,
            "metrics_after": record["metrics"],
            "deltas": deltas,
        })

    # aggregate
    after_agg = aggregate_metrics([r["metrics"] for r in per_doc_metrics])
    before_agg = prev_summary.get("aggregate_raw", {}) if prev_summary else None
    if not before_agg and prev_summary:
        before_agg = {
            k: v.get("after") for k, v in (prev_summary.get("aggregate") or {}).items()
            if isinstance(v, dict)
        }
    aggregate_deltas = calc_delta(before_agg or {}, after_agg)
    verdict = determine_verdict(aggregate_deltas) if prev_summary else "NEUTRAL"

    summary_doc = {
        "trial": trial_ts,
        "prev_trial": prev_trial_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_output_dir": str(pipeline_output_dir),
        "sample_count": len(per_doc_metrics),
        "form_type_filter": form_type_filter,
        "verdict": verdict,
        "aggregate": aggregate_deltas,
        "aggregate_raw": after_agg,  # 다음 트라이얼이 before로 사용
        "per_doc": per_doc_out,
    }

    report_path = save_eval_results(
        summary=summary_doc,
        pipeline_output_dir=pipeline_output_dir,
        eval_results_dir=eval_results_dir,
    )
    return summary_doc, report_path


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="mil_OCR_v2 트라이얼 평가")
    parser.add_argument("--input-dir", help="파이프라인 입력 디렉토리 (raw 이미지)")
    parser.add_argument(
        "--pipeline-output-dir",
        help="기존 pipeline_outputs/{ts} 디렉토리 (재실행 생략)",
    )
    parser.add_argument(
        "--prev-trial", default="latest",
        help="비교 대상 트라이얼 타임스탬프 (기본: history.json의 직전 트라이얼)",
    )
    parser.add_argument("--form-type", default=None, help="특정 form_type 필터")
    parser.add_argument(
        "--pipeline-outputs-dir", default="data/pipeline_outputs",
        help="pipeline_outputs 루트 디렉토리 (평가 산출물도 여기에 저장)",
    )
    parser.add_argument(
        "--eval-results-dir", default="data/eval_results",
        help="eval_results 루트 — history.json + latest 심볼릭 링크만 관리",
    )
    args = parser.parse_args()

    if not args.input_dir and not args.pipeline_output_dir:
        parser.error("--input-dir 또는 --pipeline-output-dir 중 하나는 필수")

    pipeline_outputs_dir = (_ROOT / args.pipeline_outputs_dir).resolve()
    eval_results_dir = (_ROOT / args.eval_results_dir).resolve()
    pipeline_outputs_dir.mkdir(parents=True, exist_ok=True)
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    # 파이프라인 실행 또는 기존 출력 사용
    if args.pipeline_output_dir:
        pipeline_output_dir = Path(args.pipeline_output_dir).resolve()
        if not pipeline_output_dir.is_dir():
            parser.error(f"pipeline-output-dir 없음: {pipeline_output_dir}")
    else:
        input_dir = (_ROOT / args.input_dir).resolve()
        if not input_dir.is_dir():
            parser.error(f"input-dir 없음: {input_dir}")
        pipeline_output_dir = run_pipeline(input_dir)

    summary, report_path = evaluate(
        pipeline_output_dir=pipeline_output_dir,
        eval_results_dir=eval_results_dir,
        prev_trial_arg=args.prev_trial,
        form_type_filter=args.form_type,
    )

    print_report(summary)
    print(f"  report : {report_path}")
    print(f"  eval   : {pipeline_output_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
