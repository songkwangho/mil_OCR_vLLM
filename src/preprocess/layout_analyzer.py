"""
P2 — 구조 분석 모듈
군수 OCR 시스템 v2 · Stage 1

v1 T2에서 수정 이관:
  - LayoutAnalysisResult → LayoutResult (processing_notes 제거)
  - RegionType 간소화: PRINTED_TEXT/HANDWRITTEN_FIELD → TEXT,
    STAMP → SEAL, FORMULA/CHART 신규 추가
  - reading_order 강화: 다단 컬럼 감지 + 컬럼 내 상→하 순서 보장
  - TASK_PROMPTS 매핑: 영역 레이블 → VLM instruction 자동 결정

역할 (VLM 전처리 게이트):
  ① 페이지 분해: Dense 페이지를 관리 가능한 영역 bbox로 분리
  ② 읽기 순서: 영역 간 올바른 처리 순서 보장 (다단 컬럼 등)
  ③ Task Prompt 결정: 영역 레이블 → VLM instruction 자동 선택

동작 모드:
  1. MODEL     — PP-DocLayout (PaddleOCR 3.x) 기반
  2. HEURISTIC — OpenCV 기반 룰 기반 fallback (모델 미설치 시 자동 전환)

Input  : PreprocessedImage (P1 출력)
Output : LayoutResult

의존: opencv-python, numpy
선택: paddleocr (MODEL 모드)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.interfaces.enums import AnalysisMode, RegionType
from src.interfaces.types import (
    BoundingBox,
    LayoutRegion,
    LayoutResult,
    PreprocessedImage,
    TASK_PROMPTS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  P2LayoutAnalyzerConfig
# ─────────────────────────────────────────────

@dataclass
class P2LayoutAnalyzerConfig:
    """P2 구조 분석기 설정."""

    # 모델 디렉토리 (None이면 heuristic fallback)
    model_dir: Optional[str] = None

    # 모델명 (PP-DocLayout_plus-L 등)
    model_name: str = "PP-DocLayout_plus-L"

    # 추론 디바이스 ("cpu" 또는 "gpu:0" 등)
    device: str = "gpu:0"

    # 최소 영역 면적 비율 (전체 이미지 대비)
    min_region_area_ratio: float = 0.001

    # 최소 신뢰도 (모델 모드)
    min_confidence: float = 0.3

    # 표 검출: 최소 수평/수직 선 길이 (이미지 폭/높이 대비 비율)
    table_min_line_ratio: float = 0.15

    # 헤더/푸터 영역 높이 비율 (상단/하단 몇 %를 헤더/푸터로 간주)
    header_zone_ratio: float = 0.12
    footer_zone_ratio: float = 0.08

    # 텍스트 밀도 범위 (heuristic 모드에서 영역 유형 판별)
    text_density_range: tuple[float, float] = (0.02, 0.35)

    # NMS IoU 임계값
    nms_iou_threshold: float = 0.3

    # 거대 영역 판별: 전체 이미지 면적 대비 비율
    oversized_region_ratio: float = 0.85

    # 다단 컬럼 감지: 페이지 가로 분할 기준 (컬럼 간 gap이 이 비율 이상이면 다단)
    multi_column_gap_ratio: float = 0.03


# ─────────────────────────────────────────────
#  내부 유틸
# ─────────────────────────────────────────────

def _compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    """두 BoundingBox 간 IoU 계산."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _nms_regions(regions: list[LayoutRegion],
                 iou_threshold: float) -> list[LayoutRegion]:
    """Non-Maximum Suppression: 겹치는 영역 중 신뢰도 높은 것만 남김."""
    if len(regions) <= 1:
        return regions

    sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
    keep: list[LayoutRegion] = []

    for region in sorted_regions:
        is_suppressed = False
        for kept in keep:
            if _compute_iou(region.bbox, kept.bbox) > iou_threshold:
                is_suppressed = True
                break
        if not is_suppressed:
            keep.append(region)

    return keep


def _determine_reading_order(regions: list[LayoutRegion],
                             page_width: int,
                             gap_ratio: float = 0.03) -> list[int]:
    """다단 컬럼 감지 기반 읽기 순서 결정.

    1. 영역을 x 중심 좌표 기준으로 클러스터링하여 컬럼 분리
    2. 컬럼 내에서 y 좌표 상→하 정렬
    3. 컬럼 간은 좌→우 순서
    4. 헤더(상단)와 푸터(하단)는 컬럼 클러스터링에서 제외 — 항상 최상위/최하위

    군수 서식 특성: 대부분 단일 컬럼이지만, 일부 양식에서 좌우 분할 레이아웃 존재.
    """
    if not regions:
        return []

    # 유형별 우선순위 (헤더=최상위, 푸터=최하위)
    type_priority = {
        RegionType.HEADER: -1,   # 항상 최상위
        RegionType.FOOTER: 999,  # 항상 최하위
    }

    # 헤더/푸터 분리
    header_footer_indices = []
    body_indices = []
    for i, r in enumerate(regions):
        if r.region_type in (RegionType.HEADER, RegionType.FOOTER):
            header_footer_indices.append(i)
        else:
            body_indices.append(i)

    # 헤더/푸터 정렬: 헤더(y 오름차순), 푸터(y 오름차순)
    header_footer_indices.sort(key=lambda i: (
        type_priority.get(regions[i].region_type, 0),
        regions[i].bbox.y1,
    ))

    # 본문 영역: 다단 컬럼 감지
    if not body_indices:
        return header_footer_indices

    # x 중심 좌표로 컬럼 클러스터링 (gap 기반)
    gap_threshold = page_width * gap_ratio
    sorted_by_cx = sorted(body_indices, key=lambda i: regions[i].bbox.center[0])

    columns: list[list[int]] = [[sorted_by_cx[0]]]
    for idx in sorted_by_cx[1:]:
        curr_cx = regions[idx].bbox.center[0]
        # 현재 컬럼의 가장 오른쪽 x2와 비교하여 큰 gap이 있으면 새 컬럼
        last_col_max_x2 = max(regions[i].bbox.x2 for i in columns[-1])
        if curr_cx - last_col_max_x2 > gap_threshold:
            columns.append([idx])
        else:
            columns[-1].append(idx)

    # 컬럼 내 y 좌표 정렬 (상→하), 컬럼 간 좌→우 정렬
    ordered_body = []
    for col_indices in columns:
        col_indices.sort(key=lambda i: (regions[i].bbox.y1, regions[i].bbox.x1))
        ordered_body.extend(col_indices)

    # 헤더 + 본문 + 푸터 순서 조합
    headers = [i for i in header_footer_indices
               if regions[i].region_type == RegionType.HEADER]
    footers = [i for i in header_footer_indices
               if regions[i].region_type == RegionType.FOOTER]

    return headers + ordered_body + footers


# ─────────────────────────────────────────────
#  Heuristic 분석기 (OpenCV 기반)
# ─────────────────────────────────────────────

class _HeuristicAnalyzer:
    """OpenCV 기반 레이아웃 분석.

    모델 없이도 군수 서식의 기본 구조를 추출할 수 있음.

    처리 흐름:
      1. 수평/수직 직선 검출 → 표 영역 추정
      2. 거대 영역(페이지 전체 테이블) 내부 분해
      3. 표 외부 텍스트 블록 검출
      4. 위치 기반 헤더/푸터 분류
    """

    def __init__(self, cfg: P2LayoutAnalyzerConfig):
        self.cfg = cfg

    def analyze(self,
                image_rgb: np.ndarray,
                binary: np.ndarray,
                notes: list[str]) -> list[LayoutRegion]:
        """이진화 이미지 기반 레이아웃 분석."""
        h, w = binary.shape[:2]
        page_area = h * w
        min_area = int(page_area * self.cfg.min_region_area_ratio)
        regions: list[LayoutRegion] = []
        region_counter = 0

        # ── 1. 직선 검출 기반 표 영역 추출
        table_regions, region_counter = self._detect_tables(
            binary, w, h, min_area, region_counter, notes
        )

        # ── 1-1. 거대 영역 분해
        normal_tables: list[LayoutRegion] = []
        for tr in table_regions:
            if tr.bbox.area >= page_area * self.cfg.oversized_region_ratio:
                notes.append(
                    f"Oversized region {tr.region_id}: "
                    f"{tr.bbox.area / page_area:.2f} >= "
                    f"{self.cfg.oversized_region_ratio} → decomposing"
                )
                sub_regions, region_counter = self._decompose_oversized_table(
                    binary, tr.bbox, w, h, min_area, region_counter, notes
                )
                if sub_regions:
                    regions.extend(sub_regions)
                else:
                    notes.append("Oversized decomposition: no sub-regions, discarding")
            else:
                normal_tables.append(tr)
        regions.extend(normal_tables)

        # 표 영역 마스크 생성 (표 내부는 텍스트 블록 검출에서 제외)
        table_mask = np.zeros((h, w), dtype=np.uint8)
        for tr in regions:
            if tr.region_type == RegionType.TABLE:
                bb = tr.bbox
                table_mask[bb.y1:bb.y2, bb.x1:bb.x2] = 255

        # ── 2. 표 외부 텍스트 블록 검출
        text_regions, region_counter = self._detect_text_blocks(
            binary, table_mask, w, h, min_area, region_counter, notes
        )

        # ── 3. 위치 기반 헤더/푸터 분류
        classified = self._classify_by_position(text_regions, w, h, notes)
        regions.extend(classified)

        notes.append(f"Heuristic: {len(regions)} regions detected")
        return regions

    def _detect_tables(self,
                       binary: np.ndarray,
                       w: int, h: int,
                       min_area: int,
                       counter: int,
                       notes: list[str]) -> tuple[list[LayoutRegion], int]:
        """수평/수직 직선 교차 기반 표 영역 검출."""
        inv = cv2.bitwise_not(binary)

        h_kernel_len = max(20, int(w * self.cfg.table_min_line_ratio))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)

        v_kernel_len = max(20, int(h * self.cfg.table_min_line_ratio))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)

        grid = cv2.add(h_lines, v_lines)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        grid_dilated = cv2.dilate(grid, dilate_kernel, iterations=3)

        contours, _ = cv2.findContours(
            grid_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[LayoutRegion] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < min_area:
                continue
            if cw < w * 0.1 or ch < h * 0.05:
                continue

            counter += 1
            h_pixels = int(h_lines[y:y+ch, x:x+cw].sum() / 255)
            v_pixels = int(v_lines[y:y+ch, x:x+cw].sum() / 255)
            has_grid = h_pixels > cw * 0.3 and v_pixels > ch * 0.3

            if has_grid:
                region = LayoutRegion(
                    region_id=f"r_{counter:04d}",
                    region_type=RegionType.TABLE,
                    bbox=BoundingBox(x1=x, y1=y, x2=x+cw, y2=y+ch),
                    confidence=0.75,
                )
                regions.append(region)

        notes.append(f"Table detection: {len(regions)} tables found")
        return regions, counter

    def _decompose_oversized_table(self,
                                   binary: np.ndarray,
                                   parent_bbox: BoundingBox,
                                   w: int, h: int,
                                   min_area: int,
                                   counter: int,
                                   notes: list[str],
                                   ) -> tuple[list[LayoutRegion], int]:
        """거대 영역을 내부 수평선 기준으로 분해."""
        bx1, by1, bx2, by2 = parent_bbox.x1, parent_bbox.y1, parent_bbox.x2, parent_bbox.y2
        roi = binary[by1:by2, bx1:bx2]
        rh, rw = roi.shape[:2]

        inv = cv2.bitwise_not(roi)

        # 내부 수평선 검출 (외곽선 마진 제외)
        margin_x = max(3, int(rw * 0.02))
        margin_y = max(3, int(rh * 0.02))
        inner_mask = np.zeros_like(inv)
        inner_mask[margin_y:rh - margin_y, margin_x:rw - margin_x] = 255
        inv_inner = cv2.bitwise_and(inv, inner_mask)

        h_kernel_len = max(20, int(rw * 0.3))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(inv_inner, cv2.MORPH_OPEN, h_kernel)

        # 수평선 y 좌표 추출
        h_proj = h_lines.sum(axis=1) / 255
        line_threshold = rw * 0.2
        in_line = False
        split_ys: list[int] = []
        for y_idx, val in enumerate(h_proj):
            if val >= line_threshold and not in_line:
                in_line = True
                line_start = y_idx
            elif val < line_threshold and in_line:
                in_line = False
                split_ys.append((line_start + y_idx) // 2)

        if len(split_ys) < 1:
            notes.append("Oversized decomposition: no internal horizontal lines")
            return [], counter

        row_boundaries = [margin_y] + split_ys + [rh - margin_y]
        notes.append(
            f"Oversized decomposition: {len(split_ys)} splits → "
            f"{len(row_boundaries) - 1} row strips"
        )

        # 내부 수직선 검출
        v_kernel_len = max(20, int(rh * 0.1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(inv_inner, cv2.MORPH_OPEN, v_kernel)

        regions: list[LayoutRegion] = []

        for i in range(len(row_boundaries) - 1):
            ry1 = row_boundaries[i]
            ry2 = row_boundaries[i + 1]
            strip_h = ry2 - ry1
            if strip_h < min(20, rh * 0.02):
                continue

            strip_v = v_lines[ry1:ry2, :]
            v_proj = strip_v.sum(axis=0) / 255
            v_threshold = strip_h * 0.2

            in_vline = False
            col_xs: list[int] = []
            for x_idx, val in enumerate(v_proj):
                if val >= v_threshold and not in_vline:
                    in_vline = True
                    vline_start = x_idx
                elif val < v_threshold and in_vline:
                    in_vline = False
                    col_xs.append((vline_start + x_idx) // 2)

            col_boundaries = [margin_x] + col_xs + [rw - margin_x]

            for j in range(len(col_boundaries) - 1):
                cx1 = col_boundaries[j]
                cx2 = col_boundaries[j + 1]
                cell_w = cx2 - cx1
                cell_h = strip_h
                cell_area = cell_w * cell_h

                if cell_area < min_area or cell_w < 15 or cell_h < 15:
                    continue

                # 셀 내부 텍스트 밀도로 유형 판별
                cell_roi = roi[ry1:ry2, cx1:cx2]
                text_density = float((cell_roi == 0).sum()) / cell_roi.size

                if text_density < 0.005:
                    continue

                # v2: TEXT로 통합 (VLM이 수기/인쇄 판별)
                rtype = RegionType.TEXT
                conf = 0.60

                # 행 위치 기반 유형 보정
                abs_cy = by1 + ry1 + cell_h // 2
                if abs_cy < h * self.cfg.header_zone_ratio:
                    rtype = RegionType.HEADER
                    conf = 0.70
                elif abs_cy > h * (1.0 - self.cfg.footer_zone_ratio):
                    rtype = RegionType.FOOTER
                    conf = 0.60

                counter += 1
                region = LayoutRegion(
                    region_id=f"r_{counter:04d}",
                    region_type=rtype,
                    bbox=BoundingBox(
                        x1=bx1 + cx1, y1=by1 + ry1,
                        x2=bx1 + cx2, y2=by1 + ry2,
                    ),
                    confidence=conf,
                )
                regions.append(region)

        notes.append(f"Oversized decomposition: {len(regions)} sub-regions extracted")
        return regions, counter

    def _detect_text_blocks(self,
                            binary: np.ndarray,
                            table_mask: np.ndarray,
                            w: int, h: int,
                            min_area: int,
                            counter: int,
                            notes: list[str]) -> tuple[list[LayoutRegion], int]:
        """표 외부의 텍스트 블록 검출."""
        inv = cv2.bitwise_not(binary)
        inv_masked = cv2.bitwise_and(inv, cv2.bitwise_not(table_mask))

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dilated = cv2.dilate(inv_masked, kernel_h, iterations=2)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[LayoutRegion] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < min_area:
                continue

            counter += 1
            # v2: TEXT로 통합 (VLM이 수기/인쇄 판별)
            region = LayoutRegion(
                region_id=f"r_{counter:04d}",
                region_type=RegionType.TEXT,
                bbox=BoundingBox(x1=x, y1=y, x2=x+cw, y2=y+ch),
                confidence=0.60,
            )
            regions.append(region)

        notes.append(f"Text block detection: {len(regions)} blocks found")
        return regions, counter

    def _classify_by_position(self,
                              text_regions: list[LayoutRegion],
                              w: int, h: int,
                              notes: list[str]) -> list[LayoutRegion]:
        """위치 기반으로 헤더/푸터 재분류."""
        header_line = int(h * self.cfg.header_zone_ratio)
        footer_line = int(h * (1.0 - self.cfg.footer_zone_ratio))

        reclassified = 0
        for region in text_regions:
            cy = region.bbox.center[1]

            if cy < header_line:
                region.region_type = RegionType.HEADER
                region.confidence = min(region.confidence + 0.1, 1.0)
                reclassified += 1
            elif cy > footer_line:
                region.region_type = RegionType.FOOTER
                region.confidence = min(region.confidence + 0.05, 1.0)
                reclassified += 1

        if reclassified > 0:
            notes.append(f"Position classification: {reclassified} regions reclassified")
        return text_regions


# ─────────────────────────────────────────────
#  Model 분석기 (PP-DocLayout)
# ─────────────────────────────────────────────

class _ModelAnalyzer:
    """PP-DocLayout (PaddleOCR 3.x) 기반 레이아웃 분석.

    PP-DocLayout_plus-L 라벨 (20종) → v2 RegionType 매핑:
      paragraph_title → HEADER
      doc_title       → HEADER
      text            → TEXT
      content         → TEXT
      aside_text      → TEXT
      number          → TEXT
      abstract        → TEXT
      table           → TABLE
      chart           → CHART
      image           → FIGURE
      figure_title    → FIGURE
      formula         → FORMULA
      formula_number  → FORMULA
      header          → HEADER
      footer          → FOOTER
      footnote        → FOOTER
      reference       → FOOTER
      reference_content → FOOTER
      seal            → SEAL
      algorithm       → TEXT
    """

    LABEL_MAP: dict[str, RegionType] = {
        "paragraph_title":   RegionType.HEADER,
        "doc_title":         RegionType.HEADER,
        "text":              RegionType.TEXT,
        "content":           RegionType.TEXT,
        "aside_text":        RegionType.TEXT,
        "number":            RegionType.TEXT,
        "abstract":          RegionType.TEXT,
        "table":             RegionType.TABLE,
        "chart":             RegionType.CHART,
        "image":             RegionType.FIGURE,
        "figure_title":      RegionType.FIGURE,
        "formula":           RegionType.FORMULA,
        "formula_number":    RegionType.FORMULA,
        "header":            RegionType.HEADER,
        "footer":            RegionType.FOOTER,
        "footnote":          RegionType.FOOTER,
        "reference":         RegionType.FOOTER,
        "reference_content": RegionType.FOOTER,
        "seal":              RegionType.SEAL,
        "algorithm":         RegionType.TEXT,
    }

    def __init__(self, cfg: P2LayoutAnalyzerConfig):
        self.cfg = cfg
        self._engine = None

    def _load_engine(self) -> None:
        """PP-DocLayout 엔진 로드 (지연 로딩)."""
        try:
            import os
            os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
            from paddleocr import LayoutDetection

            kwargs = {
                "model_name": self.cfg.model_name,
                "threshold": self.cfg.min_confidence,
                "device": self.cfg.device,
            }
            if self.cfg.model_dir:
                kwargs["model_dir"] = self.cfg.model_dir

            self._engine = LayoutDetection(**kwargs)
            logger.info("[P2] PP-DocLayout engine loaded: %s (device=%s)",
                       self.cfg.model_name, self.cfg.device)
        except ImportError:
            raise RuntimeError(
                "paddleocr 패키지가 설치되지 않았습니다. "
                "MODEL 모드를 사용하려면 'pip install paddleocr>=3.0.0'을 실행하세요."
            )
        except Exception as e:
            raise RuntimeError(f"PP-DocLayout 모델 로드 실패: {e}")

    def analyze(self,
                image_rgb: np.ndarray,
                binary: np.ndarray,
                notes: list[str]) -> list[LayoutRegion]:
        """PP-DocLayout 모델로 레이아웃 분석."""
        if self._engine is None:
            self._load_engine()

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image_bgr)

        try:
            output = self._engine.predict(
                tmp_path, batch_size=1, layout_nms=True
            )
        finally:
            import os
            os.unlink(tmp_path)

        regions: list[LayoutRegion] = []
        counter = 0
        for res in output:
            boxes = res.get('boxes', [])
            for det in boxes:
                label = det.get("label", "text").lower()
                score = float(det.get("score", 0.0))
                coord = det.get("coordinate", [0, 0, 0, 0])

                if score < self.cfg.min_confidence:
                    continue

                counter += 1
                region_type = self.LABEL_MAP.get(label, RegionType.TEXT)
                region = LayoutRegion(
                    region_id=f"r_{counter:04d}",
                    region_type=region_type,
                    bbox=BoundingBox(
                        x1=int(coord[0]), y1=int(coord[1]),
                        x2=int(coord[2]), y2=int(coord[3]),
                    ),
                    confidence=score,
                )
                regions.append(region)

        notes.append(f"PP-DocLayout ({self.cfg.model_name}): {len(regions)} regions detected")
        return regions


# ─────────────────────────────────────────────
#  P2LayoutAnalyzer (메인 클래스)
# ─────────────────────────────────────────────

class P2LayoutAnalyzer:
    """P2 — 구조 분석 컴포넌트.

    VLM의 전처리 게이트로서:
      ① 페이지를 영역별 bbox로 분해
      ② 다단 컬럼 감지 기반 읽기 순서 결정
      ③ 영역 레이블 → TASK_PROMPTS 매핑으로 VLM instruction 자동 선택

    모델이 설정되어 있으면 PP-DocLayout을 사용하고,
    없으면 OpenCV 기반 heuristic으로 자동 전환됩니다.

    사용 예시:
        analyzer = P2LayoutAnalyzer()
        result = analyzer.analyze(preprocessed_image)
        for idx in result.reading_order:
            region = result.regions[idx]
            task = TASK_PROMPTS.get(region.region_type.value, "OCR:")
            print(f"{region.region_id} → {task}")
    """

    def __init__(self, config: Optional[P2LayoutAnalyzerConfig] = None):
        self.cfg = config or P2LayoutAnalyzerConfig()

        if self.cfg.model_dir:
            self._analyzer = _ModelAnalyzer(self.cfg)
            self._mode = AnalysisMode.MODEL
        else:
            self._analyzer = _HeuristicAnalyzer(self.cfg)
            self._mode = AnalysisMode.HEURISTIC

    @property
    def mode(self) -> AnalysisMode:
        """현재 분석 모드."""
        return self._mode

    def analyze(self, preprocessed: PreprocessedImage) -> LayoutResult:
        """PreprocessedImage → LayoutResult 전체 파이프라인.

        Args:
            preprocessed: P1 출력 (image_array: RGB, binary_array: 이진화)

        Returns:
            LayoutResult: 검출된 영역 + 읽기 순서
        """
        notes: list[str] = []
        warnings: list[str] = []

        image_rgb = preprocessed.image_array
        binary = preprocessed.binary_array
        h, w = image_rgb.shape[:2]

        notes.append(f"Mode: {self._mode.value}, image: {w}x{h}, DPI: {preprocessed.dpi}")

        # ── Step 1: 영역 검출
        try:
            raw_regions = self._analyzer.analyze(image_rgb, binary, notes)
        except RuntimeError as e:
            warnings.append(f"MODEL_FALLBACK: {e}")
            logger.warning("[P2][%s] Model failed, falling back to heuristic: %s",
                          preprocessed.doc_id, e)
            fallback = _HeuristicAnalyzer(self.cfg)
            raw_regions = fallback.analyze(image_rgb, binary, notes)
            self._mode = AnalysisMode.HEURISTIC

        # ── Step 1-1: 모델 결과가 비어있으면 heuristic fallback
        if not raw_regions and self._mode == AnalysisMode.MODEL:
            warnings.append(
                "MODEL_EMPTY_FALLBACK: 모델이 영역을 검출하지 못해 heuristic으로 전환"
            )
            logger.warning("[P2][%s] Model detected 0 regions, falling back to heuristic",
                          preprocessed.doc_id)
            fallback = _HeuristicAnalyzer(self.cfg)
            raw_regions = fallback.analyze(image_rgb, binary, notes)
            self._mode = AnalysisMode.HEURISTIC

        # ── Step 2: 최소 면적 필터링
        min_area = int(h * w * self.cfg.min_region_area_ratio)
        filtered = [r for r in raw_regions if r.bbox.area >= min_area]
        if len(filtered) < len(raw_regions):
            notes.append(
                f"Area filter: {len(raw_regions) - len(filtered)} small regions removed"
            )

        # ── Step 3: NMS
        regions = _nms_regions(filtered, self.cfg.nms_iou_threshold)
        if len(regions) < len(filtered):
            notes.append(
                f"NMS: {len(filtered) - len(regions)} overlapping regions removed"
            )

        # ── Step 4: region_id 재할당
        for idx, region in enumerate(regions):
            region.region_id = f"r_{idx + 1:04d}"

        # ── Step 5: 다단 컬럼 감지 기반 읽기 순서 결정
        reading_order = _determine_reading_order(
            regions, w, self.cfg.multi_column_gap_ratio
        )

        # ── Step 6: 경고 생성
        if not regions:
            warnings.append("NO_REGIONS_DETECTED: 검출된 영역이 없습니다")

        table_count = sum(1 for r in regions if r.region_type == RegionType.TABLE)
        text_count = sum(1 for r in regions if r.region_type == RegionType.TEXT)
        notes.append(
            f"Final: {len(regions)} regions (tables={table_count}, text={text_count})"
        )

        logger.info(
            "[P2][%s] done mode=%s regions=%d tables=%d reading_order=%s",
            preprocessed.doc_id, self._mode.value, len(regions), table_count,
            reading_order,
        )

        return LayoutResult(
            doc_id=preprocessed.doc_id,
            page_width=w,
            page_height=h,
            regions=regions,
            reading_order=reading_order,
            analysis_mode=self._mode,
            warnings=warnings,
        )
