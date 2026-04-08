"""군수 OCR 시스템 v2 — Stage 2 VLM 통합 추론.

P3: Gemma4 VLM 엔진 (gemma4_engine.py)
    - instruction_builder.py: P2 레이블 → VLM instruction
    - logprobs_scorer.py: logprobs → 필드별 신뢰도
"""

from .gemma4_engine import Gemma4Engine, Gemma4EngineConfig
from .instruction_builder import InstructionBuilder, InstructionTask
from .logprobs_scorer import LogprobsScorer, calc_field_confidence, get_threshold, is_flagged

__all__ = [
    "Gemma4Engine",
    "Gemma4EngineConfig",
    "InstructionBuilder",
    "InstructionTask",
    "LogprobsScorer",
    "calc_field_confidence",
    "get_threshold",
    "is_flagged",
]
