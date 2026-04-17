"""
P1 — 이미지 전처리 모듈
군수 OCR 시스템 v2 · Stage 1

v1 T1에서 이관 (import 경로 변경 + v2 인터페이스 적응).

v2 변경사항 (v1 대비):
  - PreprocessedImage에서 brightness_band, skew_angle_corrected,
    warp_matrix, processing_notes 필드 제거
  - sr_applied 필드 추가 (LOW DPI에서 Real-ESRGAN 적용 여부)
  - SR enhancer import 경로 변경 (src.preprocess.sr_enhancer)

입력 DPI 구간(LOW / MID / HIGH)에 따라 처리 체인 분기:
  LOW  (< 150 DPI): 노이즈 제거·기울기 보정 비활성화, Real-ESRGAN SR,
                     Lanczos+CLAHE fallback → 획 보존 최우선
  MID  (150~249 DPI): 경량 노이즈 제거, 기울기 보정 활성화
  HIGH (≥ 250 DPI): 표준 노이즈 제거, 전체 기능 활성화

Input  : DocumentInput
Output : PreprocessedImage

의존: opencv-python, scikit-image, numpy, Pillow
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage import restoration
from skimage.filters import threshold_sauvola

from src.interfaces.enums import (
    BrightnessBand,
    DpiResolutionBand,
    FileExt,
    SourceType,
)
from src.interfaces.types import DocumentInput, PreprocessedImage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  DPI 구간 경계값 (하드 포인트)
# ─────────────────────────────────────────────
DPI_LOW_THRESHOLD  = 150   # LOW  / MID 경계
DPI_HIGH_THRESHOLD = 250   # MID  / HIGH 경계

# ─────────────────────────────────────────────
#  밝기 구간 경계값 (하드 포인트)
# ─────────────────────────────────────────────
BRIGHTNESS_DARK_THRESHOLD  = 80    # DARK  / NORMAL 경계 (Y-channel mean)
BRIGHTNESS_BRIGHT_THRESHOLD = 200  # NORMAL / BRIGHT 경계

# 목표 DPI (구간별)
TARGET_DPI = {
    DpiResolutionBand.LOW:  200,   # 저해상도: 과도한 업스케일 지양
    DpiResolutionBand.MID:  300,   # 중해상도: 표준 목표
    DpiResolutionBand.HIGH: 300,   # 고해상도: 표준 목표 (다운스케일 포함)
}


# ─────────────────────────────────────────────
#  P1PreprocessorConfig
# ─────────────────────────────────────────────

@dataclass
class P1PreprocessorConfig:
    """P1 전처리기 설정."""

    # 기울기 보정 최대 허용 각도 (±deg)
    max_skew_deg: float = 45.0

    # 원근 보정 활성화 (카메라 소스 전용)
    enable_perspective_correction: bool = True

    # 품질 경고 임계값
    quality_warn_threshold: float = 0.5

    # 저해상도 업스케일 시 Lanczos 사용 여부
    low_dpi_use_lanczos: bool = True

    # 저해상도에서 CLAHE(대비 강화) 적용 여부
    low_dpi_apply_clahe: bool = True

    # 팩스 원본 특화 노이즈 처리
    fax_mode: bool = False


# ─────────────────────────────────────────────
#  구간별 처리 파라미터 (핵심 하드 테이블)
# ─────────────────────────────────────────────

@dataclass
class BandParams:
    """DPI 구간별 처리 파라미터."""
    # 노이즈 제거
    apply_denoise: bool
    denoise_patch_distance: int

    # Sauvola 이진화
    # None이면 DPI 비례 동적 계산 사용
    sauvola_window_size: Optional[int]
    sauvola_k: float

    # 기울기 보정
    apply_skew_correction: bool
    hough_min_line_length_ratio: float  # image_width 대비 비율

    # 모폴로지 정리 커널 크기 (이진화 후)
    morph_kernel_size: int


BAND_PARAMS: dict[DpiResolutionBand, BandParams] = {
    DpiResolutionBand.LOW: BandParams(
        apply_denoise=False,
        denoise_patch_distance=0,
        sauvola_window_size=None,
        sauvola_k=0.30,
        apply_skew_correction=False,
        hough_min_line_length_ratio=0.5,
        morph_kernel_size=1,
    ),
    DpiResolutionBand.MID: BandParams(
        apply_denoise=True,
        denoise_patch_distance=3,
        sauvola_window_size=None,
        sauvola_k=0.20,
        apply_skew_correction=True,
        hough_min_line_length_ratio=0.3,
        morph_kernel_size=2,
    ),
    DpiResolutionBand.HIGH: BandParams(
        apply_denoise=True,
        denoise_patch_distance=5,
        sauvola_window_size=None,
        sauvola_k=0.15,
        apply_skew_correction=True,
        hough_min_line_length_ratio=0.25,
        morph_kernel_size=2,
    ),
}


# ─────────────────────────────────────────────
#  DPI 구간 분류 및 Sauvola 동적 계산
# ─────────────────────────────────────────────

def classify_dpi_band(dpi: int) -> DpiResolutionBand:
    """DPI 값에 따른 해상도 대역 분류."""
    if dpi < DPI_LOW_THRESHOLD:
        return DpiResolutionBand.LOW
    elif dpi < DPI_HIGH_THRESHOLD:
        return DpiResolutionBand.MID
    else:
        return DpiResolutionBand.HIGH


def classify_brightness_band(img_bgr: np.ndarray) -> tuple[BrightnessBand, float]:
    """BGR 이미지의 평균 밝기(grayscale)를 기준으로 밝기 구간 분류.

    Returns:
        (band, mean_brightness) — 구간 및 측정값 (진단 로깅용)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    if mean_val < BRIGHTNESS_DARK_THRESHOLD:
        return BrightnessBand.DARK, mean_val
    elif mean_val > BRIGHTNESS_BRIGHT_THRESHOLD:
        return BrightnessBand.BRIGHT, mean_val
    else:
        return BrightnessBand.NORMAL, mean_val


def compute_sauvola_window(target_dpi: int) -> int:
    """DPI에 비례하여 Sauvola window_size를 동적 계산.

    근거: 일반 군수 양식 한글 획 폭 ≈ 0.3~0.5mm
    window가 문자 한 글자(약 5mm)를 포함할 수 있어야 함.
    """
    px_per_mm = target_dpi / 25.4
    char_size_mm = 2.5
    raw = int(px_per_mm * char_size_mm)
    window = raw if raw % 2 == 1 else raw + 1
    return max(7, window)


# ─────────────────────────────────────────────
#  내부 처리 함수
# ─────────────────────────────────────────────

def _correct_brightness(img: np.ndarray,
                        band: BrightnessBand,
                        mean_brightness: float,
                        notes: list[str]) -> np.ndarray:
    """밝기 구간별 보정. Y 채널에서만 작업하여 색상 왜곡 방지."""
    if band == BrightnessBand.NORMAL:
        notes.append(f"Brightness: NORMAL (mean={mean_brightness:.1f}), no correction")
        return img

    def _make_gamma_lut(gamma: float) -> np.ndarray:
        return np.array([
            min(255, int(((i / 255.0) ** gamma) * 255.0))
            for i in range(256)
        ], dtype=np.uint8)

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    if band == BrightnessBand.DARK:
        gamma = max(0.4, mean_brightness / BRIGHTNESS_DARK_THRESHOLD)
        inv_gamma = 1.0 / gamma
        lut = _make_gamma_lut(inv_gamma)
        yuv[:, :, 0] = lut[yuv[:, :, 0]]

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

        notes.append(
            f"Brightness: DARK correction (mean={mean_brightness:.1f}, "
            f"gamma={gamma:.2f}, inv_gamma={inv_gamma:.2f}, CLAHE clipLimit=3.0)"
        )

    elif band == BrightnessBand.BRIGHT:
        gamma = max(1.1, min(1.8, mean_brightness / 170.0))
        lut = _make_gamma_lut(gamma)
        yuv[:, :, 0] = lut[yuv[:, :, 0]]

        y_after = yuv[:, :, 0]
        p2 = float(np.percentile(y_after, 2))
        p98 = float(np.percentile(y_after, 98))
        if p98 - p2 > 10:
            stretched = np.clip(
                (y_after.astype(np.float32) - p2) / (p98 - p2) * 255.0,
                0, 255,
            ).astype(np.uint8)
            yuv[:, :, 0] = stretched
            notes.append(
                f"Brightness: BRIGHT correction (mean={mean_brightness:.1f}, "
                f"gamma={gamma:.2f}, contrast stretch p2={p2:.0f} p98={p98:.0f})"
            )
        else:
            notes.append(
                f"Brightness: BRIGHT correction (mean={mean_brightness:.1f}, "
                f"gamma={gamma:.2f}, stretch skipped — range too narrow)"
            )

    corrected = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return corrected


def _bytes_to_bgr(raw_bytes: bytes) -> np.ndarray:
    """bytes → OpenCV BGR ndarray"""
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def _estimate_dpi(img: np.ndarray, dpi_hint: Optional[int]) -> int:
    """이미지 크기와 힌트를 조합해 DPI 추정."""
    if dpi_hint and dpi_hint > 0:
        return dpi_hint
    h, w = img.shape[:2]
    dpi_h = int(h / (297 / 25.4))
    dpi_w = int(w / (210 / 25.4))
    return max(72, min(600, (dpi_h + dpi_w) // 2))


def _upscale_low_dpi(img: np.ndarray,
                     current_dpi: int,
                     target_dpi: int,
                     use_lanczos: bool,
                     apply_clahe: bool,
                     notes: list[str]) -> np.ndarray:
    """저해상도 전용 업스케일 처리.

    Lanczos 보간 + CLAHE 대비 강화로 획 가시성 향상.
    과도한 업스케일(>2.0배)은 artifact 유발 → 2.0배로 클램프.
    """
    scale = target_dpi / current_dpi
    if scale > 2.0:
        scale = 2.0
        effective_dpi = int(current_dpi * 2.0)
        notes.append(
            f"LOW_DPI: upscale clamped to 2.0x "
            f"(requested {target_dpi / current_dpi:.1f}x). "
            f"Effective DPI: {effective_dpi}"
        )

    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)

    if use_lanczos:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_up  = pil_img.resize((new_w, new_h), Image.LANCZOS)
        upscaled = cv2.cvtColor(np.array(pil_up), cv2.COLOR_RGB2BGR)
        notes.append(f"LOW_DPI: Lanczos upscale {img.shape[1]}x{img.shape[0]} → {new_w}x{new_h}")
    else:
        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        notes.append(f"LOW_DPI: Cubic upscale → {new_w}x{new_h}")

    if apply_clahe:
        yuv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        upscaled = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        notes.append("LOW_DPI: CLAHE contrast enhancement applied")

    return upscaled


def _resize_standard(img: np.ndarray,
                     current_dpi: int,
                     target_dpi: int,
                     notes: list[str]) -> np.ndarray:
    """중/고해상도 표준 리사이즈."""
    if current_dpi == target_dpi:
        return img
    scale = target_dpi / current_dpi
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    notes.append(f"Resize {img.shape[1]}x{img.shape[0]} → {new_w}x{new_h} (scale={scale:.2f})")
    return resized


def _denoise(img: np.ndarray,
             source_type: SourceType,
             fax_mode: bool,
             patch_distance: int,
             notes: list[str]) -> np.ndarray:
    """노이즈 제거. 소스 유형별 최적 알고리즘 적용."""
    if source_type == SourceType.FAX or fax_mode:
        img = cv2.medianBlur(img, 3)
        notes.append("Denoise: median blur (fax mode)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if source_type == SourceType.CAMERA:
        denoised_gray = restoration.denoise_nl_means(
            gray.astype(np.float32) / 255.0,
            h=0.08,
            patch_size=5,
            patch_distance=patch_distance,
            channel_axis=None,
            fast_mode=True,
        )
        denoised_gray = (denoised_gray * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)
        notes.append(f"Denoise: NLM (camera, patch_dist={patch_distance})")
    else:
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        notes.append("Denoise: bilateral filter")

    return img


def _correct_skew(img: np.ndarray,
                  max_skew_deg: float,
                  min_line_length_ratio: float,
                  notes: list[str]) -> np.ndarray:
    """Hough 기반 기울기 감지 및 보정."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_length = int(img.shape[1] * min_line_length_ratio)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=min_length,
        maxLineGap=20,
    )

    if lines is None:
        notes.append("Skew: no lines detected, skipped")
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_skew_deg:
            angles.append(angle)

    if not angles:
        notes.append("Skew: no valid angles found, skipped")
        return img

    skew_angle = float(np.median(angles))

    if abs(skew_angle) < 0.2:
        notes.append(f"Skew: angle {skew_angle:.2f}° < threshold, skipped")
        return img

    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    corrected = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    notes.append(f"Skew: corrected {skew_angle:.2f}°")
    return corrected


def _correct_perspective(img: np.ndarray,
                         notes: list[str]) -> np.ndarray:
    """카메라 촬영 원근 왜곡 보정."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 30, 120)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        notes.append("Perspective: no contours found, skipped")
        return img

    contours    = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_contour = None
    for c in contours[:5]:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        notes.append("Perspective: 4-point contour not found, skipped")
        return img

    pts  = doc_contour.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = max(int(widthA), int(widthB))
    maxH = max(int(heightA), int(heightB))

    if maxW < 100 or maxH < 100:
        notes.append("Perspective: contour too small, skipped")
        return img

    dst = np.array([
        [0, 0], [maxW - 1, 0],
        [maxW - 1, maxH - 1], [0, maxH - 1],
    ], dtype=np.float32)

    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    notes.append(f"Perspective: corrected → {maxW}x{maxH}")
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """4점을 TL→TR→BR→BL 순서로 정렬."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _binarize_sauvola(img: np.ndarray,
                      window_size: int,
                      k: float,
                      morph_kernel_size: int,
                      notes: list[str]) -> np.ndarray:
    """Sauvola adaptive thresholding — 격자 배경·그림자에 robust."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ws   = window_size if window_size % 2 == 1 else window_size + 1
    thresh  = threshold_sauvola(gray, window_size=ws, k=k)
    binary  = (gray > thresh).astype(np.uint8) * 255

    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    notes.append(
        f"Sauvola: window={ws}, k={k:.2f}, morph_kernel={morph_kernel_size}"
    )
    return binary


def _compute_quality_score(img: np.ndarray,
                            binary: np.ndarray,
                            band: DpiResolutionBand) -> tuple[float, list[str]]:
    """품질 점수 [0.0–1.0] 산출. 저해상도 구간은 패널티 적용."""
    warnings: list[str] = []
    scores:   list[float] = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 밝기
    mean_brightness = float(gray.mean())
    if mean_brightness < 50:
        warnings.append(f"IMAGE_TOO_DARK: mean={mean_brightness:.1f}")
        scores.append(0.3)
    elif mean_brightness > 220:
        warnings.append(f"IMAGE_TOO_BRIGHT: mean={mean_brightness:.1f}")
        scores.append(0.6)
    else:
        scores.append(1.0)

    # 2. 대비
    std_dev         = float(gray.std())
    contrast_score  = min(1.0, std_dev / 60.0)
    if contrast_score < 0.4:
        warnings.append(f"LOW_CONTRAST: std_dev={std_dev:.1f}")
    scores.append(contrast_score)

    # 3. 블러 (저해상도에서는 기준 완화)
    lap_var         = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_threshold  = 200.0 if band == DpiResolutionBand.LOW else 500.0
    sharpness_score = min(1.0, lap_var / blur_threshold)
    if sharpness_score < 0.3:
        warnings.append(f"BLURRY_IMAGE: laplacian_var={lap_var:.1f}")
    scores.append(sharpness_score)

    # 4. 텍스트 밀도
    text_ratio = float((binary == 0).sum()) / binary.size
    if text_ratio < 0.01:
        warnings.append(f"NO_TEXT_DETECTED: ratio={text_ratio:.4f}")
        scores.append(0.1)
    elif text_ratio > 0.8:
        warnings.append(f"TOO_MUCH_INK: ratio={text_ratio:.4f}")
        scores.append(0.4)
    else:
        scores.append(1.0)

    # 저해상도 구간에 품질 점수에 패널티 추가
    base_score = float(np.mean(scores))
    if band == DpiResolutionBand.LOW:
        final = round(base_score * 0.85, 4)
        warnings.append("LOW_RESOLUTION_BAND: quality score penalized x0.85")
    else:
        final = round(base_score, 4)

    return final, warnings


# ─────────────────────────────────────────────
#  P1Preprocessor (메인 클래스)
# ─────────────────────────────────────────────

class P1Preprocessor:
    """P1 — DPI 적응형 이미지 전처리 컴포넌트.

    DPI 구간(LOW / MID / HIGH)에 따라 처리 체인을 자동 선택합니다.

    LOW  (< 150 DPI): 노이즈 제거·기울기 보정 비활성화, Real-ESRGAN SR,
                      Lanczos+CLAHE fallback → 획 보존 최우선
    MID  (150~249 DPI): 경량 노이즈 제거, 기울기 보정 활성화
    HIGH (≥ 250 DPI): 표준 노이즈 제거, 전체 기능 활성화

    사용 예시:
        preprocessor = P1Preprocessor()
        result = preprocessor.process(doc_input)
        print(result.resolution_band)   # DpiResolutionBand.LOW 등
    """

    def __init__(self, config: Optional[P1PreprocessorConfig] = None):
        self.cfg = config or P1PreprocessorConfig()

    # ── 메인 진입점 ──────────────────────────────

    def process(self, doc_input: DocumentInput) -> PreprocessedImage:
        """DocumentInput → PreprocessedImage 전체 파이프라인.

        Args:
            doc_input: 원본 문서 입력

        Returns:
            PreprocessedImage: 전처리 완료 이미지
        """
        notes:    list[str] = []
        warnings: list[str] = []

        # Step 1: 디코딩
        img_bgr = _bytes_to_bgr(doc_input.raw_bytes)

        # Step 1.5: 밝기 구간 분류 및 보정
        brightness_band, mean_brightness = classify_brightness_band(img_bgr)
        notes.append(
            f"Brightness: mean={mean_brightness:.1f} → band={brightness_band.value}"
        )
        logger.info("[P1][%s] brightness=%.1f band=%s",
                    doc_input.doc_id, mean_brightness, brightness_band.value)
        img_bgr = _correct_brightness(img_bgr, brightness_band, mean_brightness, notes)

        # Step 2: DPI 추정 및 구간 분류
        original_dpi = _estimate_dpi(img_bgr, doc_input.dpi_hint)
        band         = classify_dpi_band(original_dpi)
        params       = BAND_PARAMS[band]
        target_dpi   = TARGET_DPI[band]

        notes.append(
            f"DPI detected={original_dpi} → band={band.value} → target={target_dpi}"
        )
        logger.info("[P1][%s] DPI=%d band=%s", doc_input.doc_id, original_dpi, band.value)

        # Step 3: 구간별 분기 처리
        sr_applied = False
        if band == DpiResolutionBand.LOW:
            img_bgr, effective_dpi, sr_applied = self._process_low(
                img_bgr, original_dpi, target_dpi,
                doc_input.source_type, params, notes
            )
        else:
            img_bgr, effective_dpi = self._process_mid_high(
                img_bgr, original_dpi, target_dpi,
                doc_input.source_type, params, band, notes
            )

        # Step 4: 원근 보정 (카메라 + MID 이상만)
        if (self.cfg.enable_perspective_correction
                and doc_input.source_type == SourceType.CAMERA
                and band != DpiResolutionBand.LOW):
            img_bgr = _correct_perspective(img_bgr, notes)

        # Step 5: 기울기 보정 (구간 params 따름)
        if params.apply_skew_correction:
            img_bgr = _correct_skew(
                img_bgr,
                self.cfg.max_skew_deg,
                params.hough_min_line_length_ratio,
                notes,
            )
        else:
            notes.append("Skew: disabled for this resolution band")

        # Step 6: Sauvola 이진화 (window 동적 계산)
        window_size = (
            params.sauvola_window_size
            if params.sauvola_window_size is not None
            else compute_sauvola_window(effective_dpi)
        )
        binary = _binarize_sauvola(
            img_bgr,
            window_size=window_size,
            k=params.sauvola_k,
            morph_kernel_size=params.morph_kernel_size,
            notes=notes,
        )

        # Step 7: 품질 점수
        quality_score, quality_warnings = _compute_quality_score(
            img_bgr, binary, band
        )
        warnings.extend(quality_warnings)
        if quality_score < self.cfg.quality_warn_threshold:
            warnings.append(
                f"LOW_QUALITY_SCORE: {quality_score:.3f} "
                f"(threshold {self.cfg.quality_warn_threshold})"
            )

        # Step 8: BGR → RGB
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        result = PreprocessedImage(
            doc_id=doc_input.doc_id,
            image_array=image_rgb,
            binary_array=binary,
            dpi=effective_dpi,
            original_dpi=original_dpi,
            resolution_band=band,
            quality_score=quality_score,
            sr_applied=sr_applied,
            warnings=warnings,
        )

        logger.info(
            "[P1][%s] done band=%s quality=%.3f sr=%s",
            doc_input.doc_id, band.value, quality_score, sr_applied,
        )
        return result

    # ── 구간별 처리 체인 ─────────────────────────

    def _process_low(
        self,
        img: np.ndarray,
        original_dpi: int,
        target_dpi: int,
        source_type: SourceType,
        _params: BandParams,
        notes: list[str],
    ) -> tuple[np.ndarray, int, bool]:
        """LOW 구간: 노이즈 제거·기울기 보정 OFF. SR 인핸서로 업스케일.

        Returns:
            (이미지, effective_dpi, sr_applied)
        """
        notes.append("=== LOW band: stroke preservation mode ===")

        # 팩스 원본인 경우 Median blur만 허용
        if source_type == SourceType.FAX or self.cfg.fax_mode:
            img = cv2.medianBlur(img, 3)
            notes.append("LOW: fax median blur applied")

        # SR 인핸서 시도
        sr_applied = False
        try:
            from src.preprocess.sr_enhancer import SREnhancer, SREnhancerConfig
            if not hasattr(self, "_sr_enhancer"):
                self._sr_enhancer = SREnhancer(SREnhancerConfig())
            sr_result = self._sr_enhancer.enhance(img)
            notes.extend(sr_result.notes)

            if sr_result.sr_a_applied:
                img = sr_result.image
                effective_dpi = original_dpi * 2
                sr_applied = True

                # CLAHE 대비 강화
                if self.cfg.low_dpi_apply_clahe:
                    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
                    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    notes.append("LOW: CLAHE contrast enhancement applied")

                return img, effective_dpi, sr_applied

        except ImportError as e:
            logger.warning("LOW: SR 패키지 누락 — 배포 설정 확인 필요 (Lanczos fallback): %s", e)
            notes.append(f"LOW: SR 패키지 누락 → Lanczos fallback: {e}")
        except Exception as e:
            logger.warning("LOW: SR 런타임 오류 → Lanczos fallback: %s", e)
            notes.append(f"LOW: SR 인핸서 실패 → Lanczos fallback: {e}")

        # Lanczos fallback
        img = _upscale_low_dpi(
            img,
            original_dpi,
            target_dpi,
            use_lanczos=self.cfg.low_dpi_use_lanczos,
            apply_clahe=self.cfg.low_dpi_apply_clahe,
            notes=notes,
        )
        effective_dpi = int(original_dpi * min(2.0, target_dpi / original_dpi))
        return img, effective_dpi, False

    def _process_mid_high(
        self,
        img: np.ndarray,
        original_dpi: int,
        target_dpi: int,
        source_type: SourceType,
        params: BandParams,
        band: DpiResolutionBand,
        notes: list[str],
    ) -> tuple[np.ndarray, int]:
        """MID/HIGH 구간: resize + 조건부 denoise. 두 구간은 BandParams 차이로만 구분됨."""
        label = "MID" if band == DpiResolutionBand.MID else "HIGH"
        notes.append(f"=== {label} band: standard processing ===")
        img = _resize_standard(img, original_dpi, target_dpi, notes)
        if params.apply_denoise:
            img = _denoise(img, source_type, self.cfg.fax_mode,
                           params.denoise_patch_distance, notes)
        return img, target_dpi


# ─────────────────────────────────────────────
#  소스 유형별 프리셋 팩토리
# ─────────────────────────────────────────────

