"""Fallback 서비스 — v1 PP-OCRv5 (T3~T5) 래핑

VLM 불가 시 v1의 서식 분류(T3) + 수기 인식(T4) + 표 구조(T5)를
호출하여 v2 VLMResult 형식으로 변환합니다.

Fallback 결과는 항상 검토 큐에 적재됩니다 (정확도 ↓).
"""

from .fallback_http_client import FallbackHTTPClient
from .ocr_fallback_service import OCRFallbackService, OCRFallbackConfig

__all__ = [
    "OCRFallbackService",
    "OCRFallbackConfig",
    "FallbackHTTPClient",
]
