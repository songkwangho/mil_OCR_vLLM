"""군수 OCR 시스템 v2 — Stage 2 VLM 통합 추론.

P3-A: FormClassifier (form_classifier.py) — 전체 페이지 저해상도 서식 분류
P2.5-B: InstructionRouter (instruction_router.py) — form_type 반영 도메인 instruction
P2.5-C: ResolutionRouter (resolution_router.py) — 영역별 pixel_budget 할당 + 배치 그룹화
P3-B: StructuredExtractor (structured_extractor.py) — 배치 병렬 VLM 구조화 추출

공용:
    - vlm_client.py: VLM API 래퍼 + 이미지/logprobs 유틸
    - logprobs_scorer.py: logprobs → 필드별 신뢰도

Legacy (하위 호환):
    - gemma4_engine.py: P3 통합 엔진 (FormClassifier + StructuredExtractor 분리 전)
    - instruction_builder.py: InstructionRouter 위임 래퍼
"""

from .form_classifier import FormClassifier, FormClassifierConfig
from .instruction_router import InstructionRouter
from .resolution_router import ResolutionRouter, ResolutionRouterConfig
from .structured_extractor import StructuredExtractor, StructuredExtractorConfig
from .vlm_client import VLMClient
from .logprobs_scorer import LogprobsScorer, calc_field_confidence, get_threshold, is_flagged

# Legacy (하위 호환)
from .gemma4_engine import Gemma4Engine, Gemma4EngineConfig
from .instruction_builder import InstructionBuilder, InstructionTask

__all__ = [
    # P3-A
    "FormClassifier",
    "FormClassifierConfig",
    # P2.5-B
    "InstructionRouter",
    # P2.5-C
    "ResolutionRouter",
    "ResolutionRouterConfig",
    # P3-B
    "StructuredExtractor",
    "StructuredExtractorConfig",
    # 공용
    "VLMClient",
    "LogprobsScorer",
    "calc_field_confidence",
    "get_threshold",
    "is_flagged",
    # Legacy
    "Gemma4Engine",
    "Gemma4EngineConfig",
    "InstructionBuilder",
    "InstructionTask",
]
