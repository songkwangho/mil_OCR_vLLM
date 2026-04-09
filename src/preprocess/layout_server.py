"""P2 레이아웃 추론 서비스 — FastAPI 서버

PP-DocLayout_plus-L / PP-DocLayoutV3 모델 추론만 담당합니다.
label→RegionType 매핑, NMS, 읽기 순서 결정 등 후처리는
pipeline 측 _ModelAnalyzer가 수행합니다.

환경변수:
  LAYOUT_MODEL_NAME  — 기본 모델 (default: PP-DocLayoutV3)
  LAYOUT_MODEL_DIR   — 모델 가중치 경로 (default: /models/t2_layout/{model_name})
  LAYOUT_DEVICE      — 추론 디바이스 (default: gpu:0)

API:
  GET  /health          → 모델 로드 상태 확인
  POST /layout/analyze  → 이미지 → raw detections 반환
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  환경변수 기반 설정
# ─────────────────────────────────────────────

DEFAULT_MODEL_NAME = os.environ.get("LAYOUT_MODEL_NAME", "PP-DocLayoutV3")
DEFAULT_DEVICE = os.environ.get("LAYOUT_DEVICE", "gpu:0")
DEFAULT_MODEL_DIR = os.environ.get(
    "LAYOUT_MODEL_DIR",
    f"/models/t2_layout/{DEFAULT_MODEL_NAME}",
)

# ─────────────────────────────────────────────
#  Pydantic 스키마
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 인코딩된 PNG 이미지")
    model_name: Optional[str] = Field(None, description="모델명 (None이면 서버 기본값)")
    min_confidence: float = Field(0.3, description="최소 신뢰도 임계값")
    fusion_mode: bool = Field(False, description="True이면 V3(구조) + plus-L(텍스트) 융합")


class Detection(BaseModel):
    label: str
    score: float
    coordinate: list[float]
    order: Optional[int] = None
    polygon_points: Optional[list[list[float]]] = None
    source: Optional[str] = Field(None, description="융합 모드 시 출처: v3 | plus_l")


class AnalyzeResponse(BaseModel):
    detections: list[Detection]
    model_name: str
    inference_ms: float
    warnings: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
#  모델 매니저 (lazy loading)
# ─────────────────────────────────────────────

class _LayoutModelManager:
    """LayoutDetection 엔진 관리 (모델별 캐싱)."""

    def __init__(self):
        self._engines: dict[str, object] = {}
        self._healthy = False

    def get_engine(self, model_name: str, device: str, model_dir: Optional[str] = None):
        """모델 엔진 반환 (없으면 로드)."""
        cache_key = f"{model_name}@{device}"
        if cache_key not in self._engines:
            self._load(model_name, device, model_dir, cache_key)
        return self._engines[cache_key]

    def _load(self, model_name: str, device: str, model_dir: Optional[str], cache_key: str):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import LayoutDetection

        kwargs = {"model_name": model_name, "device": device}
        if model_dir:
            kwargs["model_dir"] = model_dir
        logger.info("모델 로드 시작: %s (device=%s, dir=%s)", model_name, device, model_dir)

        engine = LayoutDetection(**kwargs)
        self._engines[cache_key] = engine
        self._healthy = True
        logger.info("모델 로드 완료: %s", model_name)

    @property
    def is_healthy(self) -> bool:
        return self._healthy


_manager = _LayoutModelManager()

# ─────────────────────────────────────────────
#  FastAPI 앱
# ─────────────────────────────────────────────

app = FastAPI(title="mil_OCR_v2 Layout Service", version="1.0.0")


@app.on_event("startup")
async def _startup():
    """서버 시작 시 기본 모델 프리로드."""
    try:
        _manager.get_engine(DEFAULT_MODEL_NAME, DEFAULT_DEVICE, DEFAULT_MODEL_DIR)
        logger.info("기본 모델 프리로드 완료: %s", DEFAULT_MODEL_NAME)
    except Exception as e:
        logger.error("기본 모델 프리로드 실패: %s", e)


@app.get("/health")
async def health():
    if _manager.is_healthy:
        return {"status": "ok", "model": DEFAULT_MODEL_NAME}
    raise HTTPException(status_code=503, detail="모델 미로드")


# ─────────────────────────────────────────────
#  융합 로직
# ─────────────────────────────────────────────

# V3가 우선하는 구조 라벨 (plus-L text와 겹치면 V3를 우선)
_STRUCTURAL_LABELS = {
    "table", "chart", "figure", "image", "figure_title",
    "header", "footer", "header_image", "footer_image",
    "seal", "algorithm", "formula", "display_formula", "inline_formula",
}

# plus-L에서 가져올 텍스트 계열 라벨
_TEXT_LABELS = {
    "text", "content", "aside_text", "number", "abstract",
    "paragraph_title", "doc_title", "reference", "reference_content",
    "footnote", "vertical_text", "vision_footnote",
}


def _iou(a: list[float], b: list[float]) -> float:
    """두 bbox [x1,y1,x2,y2] 간 IoU."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(small: list[float], big: list[float]) -> float:
    """small bbox가 big bbox에 의해 덮이는 비율."""
    ix1 = max(small[0], big[0])
    iy1 = max(small[1], big[1])
    ix2 = min(small[2], big[2])
    iy2 = min(small[3], big[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_s = max(0, small[2] - small[0]) * max(0, small[3] - small[1])
    return inter / area_s if area_s > 0 else 0.0


def _run_single_model(
    engine, tmp_path: str, is_v3: bool,
    min_confidence: float, source_tag: str,
) -> list[Detection]:
    """단일 모델 추론 → Detection 리스트."""
    predict_kwargs: dict = {"batch_size": 1, "layout_nms": True}
    if is_v3:
        predict_kwargs["layout_shape_mode"] = "auto"

    output = engine.predict(tmp_path, **predict_kwargs)
    detections: list[Detection] = []
    for res in output:
        for det in res.get("boxes", []):
            label = det.get("label", "text").lower()
            score = float(det.get("score", 0.0))
            if score < min_confidence:
                continue

            coord = [float(c) for c in det.get("coordinate", [0, 0, 0, 0])]
            order = None
            polygon = None
            if is_v3:
                raw_order = det.get("order")
                if raw_order is not None:
                    order = int(raw_order)
                raw_poly = det.get("polygon_points")
                if raw_poly is not None and len(raw_poly) > 0:
                    polygon = [[float(p[0]), float(p[1])] for p in raw_poly]

            detections.append(Detection(
                label=label, score=score, coordinate=coord,
                order=order, polygon_points=polygon, source=source_tag,
            ))
    return detections


def _fuse_detections(
    v3_dets: list[Detection],
    plusl_dets: list[Detection],
    overlap_threshold: float = 0.5,
) -> tuple[list[Detection], list[str]]:
    """V3 구조 영역 + plus-L 텍스트 영역 융합.

    전략:
      1. V3의 모든 검출을 유지
      2. plus-L에서 텍스트 계열 라벨만 추출
      3. plus-L 텍스트 중 V3 영역(전체)과 50% 이상 겹치는 것은 제거
      4. reading_order: V3 구조 영역은 V3 order 유지, 텍스트는 좌상→우하 정렬
    """
    warnings: list[str] = []

    # V3 전체 유지
    fused = list(v3_dets)
    v3_boxes = [d.coordinate for d in v3_dets]

    # plus-L 텍스트 필터링
    added_text = 0
    skipped_text = 0
    for det in plusl_dets:
        if det.label not in _TEXT_LABELS:
            continue
        # V3 영역과 50% 이상 겹침 검사
        overlaps_v3 = any(
            _overlap_ratio(det.coordinate, vb) >= overlap_threshold
            for vb in v3_boxes
        )
        if overlaps_v3:
            skipped_text += 1
            continue
        fused.append(Detection(
            label=det.label, score=det.score, coordinate=det.coordinate,
            order=None, polygon_points=None, source="plus_l",
        ))
        added_text += 1

    warnings.append(
        f"fusion: V3 {len(v3_dets)} + plus-L text {added_text} "
        f"(skipped {skipped_text} overlapping>=50%) = {len(fused)} total"
    )
    return fused, warnings


@app.post("/layout/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    warnings: list[str] = []
    model_name = req.model_name or DEFAULT_MODEL_NAME
    is_v3 = "V3" in model_name.upper()

    # 이미지 디코딩
    try:
        img_bytes = base64.b64decode(req.image_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("이미지 디코딩 실패")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 오류: {e}")

    # 임시 파일
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image_bgr)

    try:
        t0 = time.time()

        if req.fusion_mode:
            # ── 융합 모드: V3(구조) + plus-L(텍스트) ──
            v3_dir = f"/models/t2_layout/PP-DocLayoutV3"
            plusl_dir = f"/models/t2_layout/PP-DocLayout_plus-L"

            try:
                v3_engine = _manager.get_engine("PP-DocLayoutV3", DEFAULT_DEVICE, v3_dir)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"V3 모델 로드 실패: {e}")
            try:
                plusl_engine = _manager.get_engine("PP-DocLayout_plus-L", DEFAULT_DEVICE, plusl_dir)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"plus-L 모델 로드 실패: {e}")

            v3_dets = _run_single_model(v3_engine, tmp_path, True, req.min_confidence, "v3")
            plusl_dets = _run_single_model(plusl_engine, tmp_path, False, req.min_confidence, "plus_l")

            detections, fusion_warnings = _fuse_detections(v3_dets, plusl_dets)
            warnings.extend(fusion_warnings)
            model_name = "fusion(V3+plus-L)"

        else:
            # ── 단일 모델 모드 ──
            model_dir = os.environ.get(
                "LAYOUT_MODEL_DIR",
                f"/models/t2_layout/{model_name}",
            )
            try:
                engine = _manager.get_engine(model_name, DEFAULT_DEVICE, model_dir)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"모델 로드 실패: {e}")

            detections = _run_single_model(engine, tmp_path, is_v3, req.min_confidence, model_name)

        inference_ms = (time.time() - t0) * 1000
    finally:
        os.unlink(tmp_path)

    return AnalyzeResponse(
        detections=detections,
        model_name=model_name,
        inference_ms=round(inference_ms, 1),
        warnings=warnings,
    )
