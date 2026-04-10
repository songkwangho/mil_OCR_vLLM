"""P3-B OCR-augmented 힌트 제공자

저신뢰 영역에 경량 PaddleOCR을 선행 실행하여 텍스트 힌트를 생성합니다.
KLOCR 논문(2025.10) 기준, OCR-augmented 접근이 Key Information Extraction에서
가장 큰 정확도 향상을 보입니다. 특히 군수 전문 용어·NSN 코드처럼 VLM이
문맥으로 추론하기 어려운 필드에 효과적입니다.

적용 정책 (지연 최소화):
  - 전체 영역 일괄 적용 금지 — 영역당 ~50ms 추가
  - 트리거 1: 이전 호출 logprobs < 0.80 (저신뢰)
  - 트리거 2: NSN 패턴 후보 영역 (text/handwritten_field)
  - 트리거 3: StructuredExtractor 재시도 시 자동 부착

폐쇄망 안전 가드:
  - PaddleOCR import 실패 시 자동 비활성 (`enabled=False`)
  - PaddleOCR 모델 가중치 미배치 시 자동 비활성
  - 어떤 경우에도 메인 파이프라인을 중단시키지 않음

Reference:
  - PIPELINE.md §3-3, AI_INFERENCE.md §4-7
  - CLAUDE.md 핵심 원칙: OCR-augmented
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRHintConfig:
    """OCRHintProvider 설정."""

    # 활성화 토글 (환경변수 OCR_HINT_ENABLED로도 제어)
    enabled: bool = True

    # 힌트 트리거 임계값 — 이전 호출 신뢰도가 이 값 미만이면 힌트 생성
    hint_trigger_threshold: float = 0.80

    # NSN 패턴 후보 영역 타입 (해당 영역은 첫 호출부터 힌트 부착 가능)
    nsn_hint_region_types: tuple[str, ...] = ("text",)  # "handwritten_field"는 v2에 없음

    # 힌트 텍스트 최대 길이 (토큰 절약)
    max_hint_chars: int = 200

    # PaddleOCR 디바이스
    use_gpu: bool = True

    # PaddleOCR 언어
    lang: str = "korean"


class OCRHintProvider:
    """경량 PaddleOCR 선행 실행 → VLM에 텍스트 힌트 제공.

    사용 예시:
        provider = OCRHintProvider()
        if provider.enabled and provider.should_hint(region_type, prev_conf):
            hint = provider.get_hint(cropped_image)
            spec = InstructionRouter.with_ocr_hint(spec, hint)
    """

    def __init__(self, config: Optional[OCRHintConfig] = None):
        self.cfg = config or OCRHintConfig()

        # 환경변수 토글
        env_enabled = os.environ.get("OCR_HINT_ENABLED", "").lower()
        if env_enabled in ("false", "0", "no"):
            self.cfg.enabled = False

        self._ocr = None
        self._init_attempted = False

    @property
    def enabled(self) -> bool:
        """엔진 가용성 (cfg.enabled + 실제 import/가중치 로드 성공)."""
        if not self.cfg.enabled:
            return False
        if not self._init_attempted:
            self._try_init()
        return self._ocr is not None

    def _try_init(self) -> None:
        """PaddleOCR 3.x 지연 초기화. 실패 시 자동 비활성.

        폐쇄망 환경 가드:
          - PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True 자동 설정
          - 모델 다운로드 시도 시에는 자동 비활성으로 fallback
        """
        self._init_attempted = True
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR  # type: ignore
            # PaddleOCR 3.x 새 API: lang만 받음. 다른 옵션은 predict() 시점에 전달
            self._ocr = PaddleOCR(
                lang=self.cfg.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            logger.info(
                "OCRHintProvider: PaddleOCR 초기화 완료 (lang=%s)",
                self.cfg.lang,
            )
        except ImportError:
            logger.warning(
                "OCRHintProvider: paddleocr 미설치 — OCR-augmented 비활성"
            )
            self._ocr = None
        except Exception as e:
            logger.warning(
                "OCRHintProvider: PaddleOCR 초기화 실패 (%s) — OCR-augmented 비활성", e
            )
            self._ocr = None

    def should_hint(
        self,
        region_type: str,
        prev_confidence: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """힌트 생성 여부 결정.

        Args:
            region_type: 영역 타입 ("text", "table" 등)
            prev_confidence: 이전 호출 신뢰도 (없으면 첫 호출)
            force: 재시도 등에서 강제 적용

        Returns:
            True면 get_hint() 호출 권장.
        """
        if not self.enabled:
            return False
        if force:
            return True
        if prev_confidence is not None and prev_confidence < self.cfg.hint_trigger_threshold:
            return True
        # 첫 호출에서 NSN 패턴 후보 영역도 자동 적용 — 비용이 큰 경우
        # CLAUDE.md 보완 [R2]에 따라 기본은 보수적: 첫 호출에는 미적용.
        # 재시도 또는 명시적 force 시에만 적용.
        return False

    def get_hint(self, image_rgb: np.ndarray) -> str:
        """PaddleOCR 3.x 결과를 힌트 문자열로 변환.

        실패 시 빈 문자열 반환 (메인 파이프라인 중단 금지).

        PaddleOCR 3.x predict() 반환 형식:
          [{"rec_texts": ["foo", "bar"], "rec_scores": [0.9, 0.8], ...}, ...]
        """
        if not self.enabled:
            return ""
        if image_rgb is None or image_rgb.size == 0:
            return ""
        # PaddleOCR는 BGR 입력 기대 — RGB → BGR 변환
        try:
            import cv2
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            results = self._ocr.predict(input=bgr)
        except Exception as e:
            logger.warning("OCRHintProvider: predict() 실패 (%s)", e)
            return ""

        if not results:
            return ""

        texts: list[str] = []
        for page in results:
            if page is None:
                continue
            # PaddleOCR 3.x 반환: dict 또는 객체. rec_texts 필드 시도.
            try:
                rec_texts = None
                if isinstance(page, dict):
                    rec_texts = page.get("rec_texts")
                else:
                    rec_texts = getattr(page, "rec_texts", None)
                if rec_texts:
                    texts.extend([str(t) for t in rec_texts if t])
                    continue
            except Exception:
                pass

            # 구버전 호환: [[bbox, (text, score)], ...] 형식
            try:
                for line in page:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        text_score = line[1]
                        text = (
                            text_score[0]
                            if isinstance(text_score, (list, tuple))
                            else str(text_score)
                        )
                        if text:
                            texts.append(text)
            except Exception:
                continue

        joined = " ".join(texts).strip()
        if len(joined) > self.cfg.max_hint_chars:
            joined = joined[: self.cfg.max_hint_chars] + "…"
        return joined
