"""온라인 환경에서 PaddleOCR 한국어 모델을 사전 다운로드하는 스크립트.

폐쇄망 배포 준비용. PaddleOCR 3.x가 초기화 시 자동 다운로드하는 모델을
`models/ocr_hint/` 하위로 수집해 `docker/Dockerfile.pipeline`의 COPY 대상으로 사용.

대상:
  - PP-OCRv5_server_det    (텍스트 검출)
  - korean_PP-OCRv5_mobile_rec (한국어 텍스트 인식)

사용:
  # 온라인 환경에서
  python scripts/download_paddle_models.py
  # → models/ocr_hint/ 아래에 모델 폴더 생성 (~100MB)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "models" / "ocr_hint"


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("ERROR: paddleocr 미설치. `pip install paddleocr`", file=sys.stderr)
        return 1

    print("[1/2] PaddleOCR 초기화 (첫 실행 시 자동 다운로드)...")
    _ = PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print("      완료 — ~/.paddlex/official_models/ 에 가중치 저장됨")

    cache_dir = Path.home() / ".paddlex" / "official_models"
    if not cache_dir.exists():
        print(f"ERROR: 기대한 캐시 경로 없음: {cache_dir}", file=sys.stderr)
        return 1

    print(f"[2/2] {cache_dir} → {TARGET_DIR}")
    copied = 0
    for item in cache_dir.iterdir():
        if not item.is_dir():
            continue
        dst = TARGET_DIR / item.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(item, dst)
        size_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / (1024 * 1024)
        print(f"      {item.name:40s} ({size_mb:.1f} MB)")
        copied += 1

    if copied == 0:
        print("WARN: 복사된 모델 없음", file=sys.stderr)
        return 1

    print(f"\n완료: {copied}개 모델을 {TARGET_DIR}에 배치")
    print("다음 단계: docker/Dockerfile.pipeline의 COPY 지시어 확인 후 이미지 재빌드")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
