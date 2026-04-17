"""T1 초해상도 인핸서 — LOW DPI (< 150) 이미지 전용

Real-ESRGAN을 사용하여 전체 이미지를 x2 업스케일합니다.
MID/HIGH DPI에서는 호출하지 않습니다 (T1 _process_low에서만 사용).

의존: torch, realesrgan, basicsr
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SREnhancerConfig:
    """SR 인핸서 설정."""

    # Real-ESRGAN 모델 경로
    esrgan_model_path: Optional[str] = None
    esrgan_scale: int = 2  # 업스케일 배율

    # 디바이스
    device: str = "cuda"


@dataclass
class SRResult:
    """SR 처리 결과."""

    image: np.ndarray                        # 최종 SR 이미지 (BGR)
    sr_a_image: Optional[np.ndarray] = None  # Real-ESRGAN 출력
    sr_a_applied: bool = False
    notes: list[str] | None = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class _RealESRGANEngine:
    """Real-ESRGAN 초해상도 엔진."""

    def __init__(self, model_path: str, scale: int, device: str):
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        logger.info("T1-SR: Real-ESRGAN 로드 중: %s", model_path)

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )

        gpu_id = 0 if device.startswith("cuda") else None

        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=256,       # 타일 기반 처리 (VRAM 절약)
            tile_pad=10,
            pre_pad=0,
            half=False,     # FP32 (안정성)
            gpu_id=gpu_id,
        )
        logger.info("T1-SR: Real-ESRGAN 로드 완료 (scale=%d, device=%s)",
                     scale, device)

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """이미지 초해상도 처리.

        Args:
            img_bgr: BGR 이미지

        Returns:
            초해상도 BGR 이미지
        """
        output, _ = self.upsampler.enhance(img_bgr, outscale=self.upsampler.scale)
        return output


class SREnhancer:
    """T1 초해상도 인핸서 — Real-ESRGAN 전체 이미지 업스케일.

    사용법:
        cfg = SREnhancerConfig()
        enhancer = SREnhancer(cfg)
        result = enhancer.enhance(img_bgr)
    """

    def __init__(self, cfg: Optional[SREnhancerConfig] = None):
        self.cfg = cfg or SREnhancerConfig()
        self._esrgan: Optional[_RealESRGANEngine] = None

        self._init_model()

    def _init_model(self) -> None:
        """모델 초기화."""
        project_root = Path(__file__).resolve().parents[2]

        esrgan_path = self.cfg.esrgan_model_path
        if esrgan_path is None:
            default = project_root / "models" / "t1_sr" / "RealESRGAN_x2plus.pth"
            if default.exists():
                esrgan_path = str(default)

        if esrgan_path:
            try:
                self._esrgan = _RealESRGANEngine(
                    esrgan_path, self.cfg.esrgan_scale, self.cfg.device,
                )
            except Exception as e:
                logger.warning("T1-SR: Real-ESRGAN 초기화 실패: %s", e)
        else:
            logger.warning("T1-SR: Real-ESRGAN 모델 미발견 → Lanczos fallback")

    @property
    def available(self) -> bool:
        return self._esrgan is not None

    def enhance(self, img_bgr: np.ndarray) -> SRResult:
        """Real-ESRGAN 전체 이미지 업스케일.

        Args:
            img_bgr: BGR 이미지

        Returns:
            SRResult
        """
        notes: list[str] = []
        sr_a_image = None

        if self._esrgan is not None:
            try:
                sr_a_image = self._esrgan.enhance(img_bgr)
                notes.append(
                    f"T1-SR: Real-ESRGAN x{self.cfg.esrgan_scale} "
                    f"({img_bgr.shape[1]}x{img_bgr.shape[0]} → "
                    f"{sr_a_image.shape[1]}x{sr_a_image.shape[0]})"
                )
                return SRResult(
                    image=sr_a_image,
                    sr_a_image=sr_a_image,
                    sr_a_applied=True,
                    notes=notes,
                )
            except Exception as e:
                logger.warning("T1-SR: Real-ESRGAN 추론 실패 → Lanczos fallback: %s", e)

        # Lanczos fallback
        return SRResult(
            image=self._lanczos_fallback(img_bgr, notes),
            sr_a_image=None,
            sr_a_applied=False,
            notes=notes,
        )

    def _lanczos_fallback(self, img_bgr: np.ndarray, notes: list[str]) -> np.ndarray:
        """SR 모델 미사용 시 Lanczos x2 fallback."""
        from PIL import Image

        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        new_w, new_h = pil_img.width * 2, pil_img.height * 2
        pil_up = pil_img.resize((new_w, new_h), Image.LANCZOS)
        notes.append(f"T1-SR: Lanczos fallback x2 → {new_w}x{new_h}")
        return cv2.cvtColor(np.array(pil_up), cv2.COLOR_RGB2BGR)
