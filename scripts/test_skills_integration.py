"""Skills 통합 테스트 스크립트 — 국회공문서.jpg 기반.

SealPreprocessor / SealReader / SignatureDetector / TableExtractor를
실제 VLM 서버와 연동하여 실행하고, 각 단계의 결과를 저장합니다.

환경변수:
  VLLM_BASE_URL    (default: http://localhost:8100/v1)
  VLLM_MODEL_NAME  (default: /models/gemma4/gemma-4-26b-a4b-it/)

출력:
  data/skill_outputs/<timestamp>/
    ├── seal_unwrapped.png / seal_fallback.png
    ├── seal_reader.json
    ├── signature_detector.json
    ├── table_pass1.json
    └── table_pass2_tasks.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.interfaces.enums import RegionType  # noqa: E402
from src.interfaces.types import SkillTask  # noqa: E402
from src.preprocess.seal_preprocessor import SealPreprocessor  # noqa: E402
from src.vlm.skills.seal_reader import SealReader  # noqa: E402
from src.vlm.skills.signature_detector import SignatureDetector  # noqa: E402
from src.vlm.skills.table_extractor import TableExtractor  # noqa: E402
from src.vlm.vlm_client import VLMClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("skills_integration")

IMAGE_PATH = PROJECT_ROOT / "data/raw/국회공문서.jpg"
OUT_ROOT = PROJECT_ROOT / "data/skill_outputs"


def _to_serializable(obj):
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return {"_ndarray_shape": list(obj.shape), "_dtype": str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, ensure_ascii=False, indent=2)


def main() -> int:
    if not IMAGE_PATH.exists():
        logger.error("테스트 이미지 없음: %s", IMAGE_PATH)
        return 1

    bgr = cv2.imread(str(IMAGE_PATH))
    if bgr is None:
        logger.error("이미지 로드 실패: %s", IMAGE_PATH)
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    logger.info("로드 완료: %s (%d×%d)", IMAGE_PATH.name, w, h)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("출력 디렉토리: %s", out_dir)

    # ─────── 1) SealPreprocessor 단독 ───────
    pre = SealPreprocessor()
    # 이미지 하단 1/3 (직인이 주로 위치하는 영역)을 후보 크롭으로 사용
    seal_candidate = rgb[int(h * 0.55) :, :, :]
    seal_result = pre.process(seal_candidate)
    suffix = "unwrapped" if seal_result.unwrapped else "fallback"
    cv2.imwrite(
        str(out_dir / f"seal_{suffix}.png"),
        cv2.cvtColor(seal_result.image, cv2.COLOR_RGB2BGR),
    )
    logger.info(
        "[1/4] SealPreprocessor: unwrapped=%s, shape=%s",
        seal_result.unwrapped, seal_result.image.shape,
    )

    # VLM 의존 Skill 초기화
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8100/v1")
    model_name = os.environ.get("VLLM_MODEL_NAME", "/models/gemma4/gemma-4-26b-a4b-it/")
    vlm = VLMClient(base_url=base_url, model_name=model_name, timeout=120.0)
    logger.info("VLMClient: %s (%s)", base_url, model_name)

    # ─────── 2) SealReader ───────
    seal_reader = SealReader(vlm_client=vlm, seal_preprocessor=pre)
    t0 = time.time()
    seal_task = SkillTask(
        region_id="seal_candidate",
        region_type=RegionType.SEAL,
        cropped_image=seal_candidate,
        pixel_budget=560,
    )
    try:
        seal_sr = seal_reader.run(seal_task)
        logger.info(
            "[2/4] SealReader: content=%r conf=%.3f (%.2fs)",
            seal_sr.content[:40], seal_sr.confidence, time.time() - t0,
        )
        _save_json(out_dir / "seal_reader.json", seal_sr)
    except Exception as e:
        logger.exception("SealReader 실패: %s", e)
        _save_json(out_dir / "seal_reader.json", {"error": str(e)})

    # ─────── 3) SignatureDetector ───────
    # 문서 상단 헤더 영역은 대개 인쇄 텍스트 → signature_present=False 기대
    header_crop = rgb[: int(h * 0.15), :, :]
    detector = SignatureDetector(vlm_client=vlm)
    sig_task = SkillTask(
        region_id="header_probe",
        region_type=RegionType.SIGNATURE,
        cropped_image=header_crop,
        pixel_budget=140,
    )
    t0 = time.time()
    try:
        sig_sr = detector.run(sig_task)
        logger.info(
            "[3/4] SignatureDetector: %s (%.2fs)",
            sig_sr.content, time.time() - t0,
        )
        _save_json(out_dir / "signature_detector.json", sig_sr)
    except Exception as e:
        logger.exception("SignatureDetector 실패: %s", e)
        _save_json(out_dir / "signature_detector.json", {"error": str(e)})

    # ─────── 4) TableExtractor (2패스) ───────
    # 전체 페이지를 표로 투입 (실제 운영에선 P2 레이아웃이 크롭 제공).
    # 국회공문서는 본문이 문서 전체를 차지하므로 중앙 영역을 패스1 입력으로 사용.
    table_crop = rgb[int(h * 0.15) : int(h * 0.85), :, :]
    table_ext = TableExtractor(vlm_client=vlm)
    t0 = time.time()
    try:
        structure = table_ext.pass1_structure(table_crop, region_id="tbl_main")
        logger.info(
            "[4a] TableExtractor.pass1: type=%s cells=%d conf=%.3f (%.2fs)",
            structure.table_type, len(structure.cells),
            structure.structure_confidence, time.time() - t0,
        )
        _save_json(out_dir / "table_pass1.json", structure)

        tasks = table_ext.pass2_route(structure, table_crop)
        logger.info("[4b] TableExtractor.pass2: %d task(s) 생성", len(tasks))
        _save_json(
            out_dir / "table_pass2_tasks.json",
            [
                {
                    "region_id": t.region_id,
                    "region_type": t.region_type.value,
                    "pixel_budget": t.pixel_budget,
                    "context": t.context,
                    "image_shape": list(t.cropped_image.shape),
                }
                for t in tasks
            ],
        )
    except Exception as e:
        logger.exception("TableExtractor 실패: %s", e)
        _save_json(out_dir / "table_pass1.json", {"error": str(e)})

    logger.info("완료 — 결과: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
