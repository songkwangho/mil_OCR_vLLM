"""P2 레이아웃 추론 서비스 HTTP 클라이언트

layout 컨테이너(POST /layout/analyze)를 호출합니다.
_ModelAnalyzer에서 in-process PaddleOCR 대신 사용됩니다.
"""

from __future__ import annotations

import base64
import logging

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class LayoutHTTPClient:
    """Layout 추론 서비스 HTTP 클라이언트.

    사용:
        client = LayoutHTTPClient("http://layout:8082")
        detections = client.predict(image_rgb, "PP-DocLayoutV3", 0.3)
    """

    def __init__(self, base_url: str = "http://localhost:8082", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_healthy(self) -> bool:
        """서비스 헬스체크."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def predict(
        self,
        image_rgb: np.ndarray,
        model_name: str,
        min_confidence: float = 0.3,
        fusion_mode: bool = False,
    ) -> list[dict]:
        """이미지를 layout 서비스에 전송하고 raw detections를 수신.

        Args:
            image_rgb: H×W×3 uint8 RGB 이미지
            model_name: 모델명 ("PP-DocLayoutV3" 등)
            min_confidence: 최소 신뢰도 임계값
            fusion_mode: V3(구조) + plus-L(텍스트) 융합 모드

        Returns:
            [{label, score, coordinate, order, polygon_points, source}, ...]

        Raises:
            RuntimeError: 서비스 호출 실패 시
        """
        # RGB → BGR → PNG → Base64
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", image_bgr)
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = {
            "image_b64": image_b64,
            "model_name": model_name,
            "min_confidence": min_confidence,
            "fusion_mode": fusion_mode,
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/layout/analyze",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(f"Layout 서비스 연결 실패 ({self.base_url}): {e}") from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Layout 서비스 타임아웃 ({self.timeout}s): {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Layout 서비스 오류 (HTTP {e.response.status_code}): {e}") from e

        data = resp.json()
        detections = data.get("detections", [])

        logger.info(
            "Layout 서비스 응답: model=%s, detections=%d, %.1fms",
            data.get("model_name", "?"),
            len(detections),
            data.get("inference_ms", 0),
        )
        return detections
