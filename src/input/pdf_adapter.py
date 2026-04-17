"""PdfAdapter — PDF 바이트를 페이지별 PageImage 리스트로 변환.

pdf_adapter_design.md §6 구현.
  - 텍스트 레이어 무시, 항상 이미지 렌더링 (방침 A)
  - 기본 300 DPI (군수 서식 표준)
  - 암호화/손상 PDF는 PdfAdapterError
  - 멀티페이지: 페이지 순서 유지 (1-based page_number)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.interfaces.types import PageImage

logger = logging.getLogger(__name__)


RENDER_DPI_DEFAULT = 300   # A4 @ 300 dpi = 2480 × 3508 px
RENDER_DPI_HIGH = 400
RENDER_DPI_MIN = 150       # 이하 → P1 SR 대상 경고

SCAN_PDF_TEXT_THRESHOLD = 50


class PdfAdapterError(Exception):
    """복구 불가 PDF 처리 오류."""
    pass


class PdfAdapter:
    """PDF → list[PageImage] 변환기.

    fitz(PyMuPDF)를 지연 import하여 폐쇄망 배치에서 선택적으로 활성화.
    """

    def __init__(
        self,
        render_dpi: int = RENDER_DPI_DEFAULT,
        colorspace: str = "RGB",
    ):
        self._dpi = render_dpi
        self._colorspace_name = colorspace
        self._fitz = None

    # ─────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────
    def render(self, doc_id: str, raw_bytes: bytes) -> list[PageImage]:
        """PDF 바이트 → PageImage 리스트."""
        fitz = self._get_fitz()
        try:
            pdf = fitz.open(stream=raw_bytes, filetype="pdf")
        except Exception as e:
            raise PdfAdapterError(f"PDF 열기 실패: {e}") from e

        if pdf.is_encrypted:
            pdf.close()
            raise PdfAdapterError("암호화된 PDF — 복호화 키 없음")

        if pdf.page_count == 0:
            pdf.close()
            raise PdfAdapterError("페이지 수 0 — 빈 PDF")

        total = pdf.page_count
        cs = fitz.csRGB if self._colorspace_name == "RGB" else fitz.csGRAY
        results: list[PageImage] = []

        for i in range(total):
            page_warnings: list[str] = []
            page = pdf[i]
            self._check_scan_quality(page, page_warnings)
            image_array = self._render_page(page, self._dpi, cs, fitz)
            results.append(
                PageImage(
                    doc_id=doc_id,
                    page_number=i + 1,
                    total_pages=total,
                    image_array=image_array,
                    render_dpi=self._dpi,
                    original_width_pt=page.rect.width,
                    original_height_pt=page.rect.height,
                    warnings=page_warnings,
                )
            )

        pdf.close()
        return results

    @staticmethod
    def is_pdf(raw_bytes: bytes) -> bool:
        """매직 바이트 기반 PDF 판정 (확장자가 없거나 불확실한 경우용)."""
        return len(raw_bytes) >= 4 and raw_bytes[:4] == b"%PDF"

    # ─────────────────────────────────────────────
    #  내부 유틸
    # ─────────────────────────────────────────────
    def _get_fitz(self):
        if self._fitz is None:
            try:
                import fitz  # type: ignore
            except ImportError as e:
                raise PdfAdapterError(
                    "pymupdf 미설치 — `pip install pymupdf>=1.24.0` 필요"
                ) from e
            self._fitz = fitz
        return self._fitz

    def _check_scan_quality(self, page, warnings: list[str]) -> None:
        """텍스트/이미지 밀도 기반 품질 경고."""
        text_len = len(page.get_text("text").strip())
        img_list = page.get_images(full=False)
        is_scan = text_len < SCAN_PDF_TEXT_THRESHOLD and len(img_list) > 0

        if not is_scan and text_len < SCAN_PDF_TEXT_THRESHOLD:
            warnings.append(
                f"텍스트 레이어 없음({text_len}자), 이미지도 없음 — 빈 페이지 가능성"
            )

        if is_scan and img_list:
            xref = img_list[0][0]
            try:
                img_info = page.parent.extract_image(xref)
                img_w_px = img_info["width"]
                page_w_pt = page.rect.width
                if page_w_pt > 0:
                    estimated_dpi = int(img_w_px / (page_w_pt / 72))
                    if estimated_dpi < RENDER_DPI_MIN:
                        warnings.append(
                            f"원본 이미지 추정 DPI={estimated_dpi} (< {RENDER_DPI_MIN}) — P1 SR 대상"
                        )
            except Exception as e:
                logger.debug("페이지 이미지 DPI 추정 실패 (무시): %s", e)

    def _render_page(self, page, dpi: int, cs, fitz) -> np.ndarray:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, colorspace=cs, alpha=False)
        n = pix.n
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, n)
        if n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img.copy()
