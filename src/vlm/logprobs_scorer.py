"""logprobs → 필드별 신뢰도 환산 모듈

vLLM이 반환하는 토큰별 logprob을 필드 신뢰도(0.0~1.0)로 변환합니다.

산출 방식:
  - 필드값 토큰들의 logprob → 기하 평균 (geometric mean)
  - logprob = log(prob) → prob = exp(logprob)
  - geo_mean = exp(mean(logprobs))
  - 길이 편향 보정: length_factor = min(1.0, sqrt(expected_tokens / actual_tokens))
    긴 필드(부대명 등)는 토큰 수가 많아 기하평균이 구조적으로 낮아지므로,
    필드 유형별 기대 토큰 수 기반 보정 계수를 적용합니다.

필드 유형별 임계값:
  - 금액:    0.99 (미달 시 is_flagged + review)
  - 코드:    0.97
  - 날짜:    0.95
  - 수량:    0.95
  - 텍스트:  0.90
  - 서명:    0.70

의존: 없음 (순수 Python)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  필드 유형별 신뢰도 임계값
# ─────────────────────────────────────────────

FIELD_THRESHOLDS: dict[str, float] = {
    "amount":    0.99,
    "code":      0.97,
    "date":      0.95,
    "quantity":  0.95,
    "number":    0.95,
    "text":      0.90,
    "signature": 0.70,
}

DEFAULT_THRESHOLD = 0.90

# P4 룰 검증 감점 상수 (BACKEND.md §5-3)
PENALTY_ARITHMETIC = 0.30
PENALTY_CODE_FORMAT = 0.15
PENALTY_DATE_LOGIC = 0.10
PENALTY_MISSING_FIELD = 0.20
PENALTY_FORMAT = 0.10


# ─────────────────────────────────────────────
#  필드 유형별 기대 토큰 수 (길이 편향 보정용)
# ─────────────────────────────────────────────

EXPECTED_TOKENS: dict[str, int] = {
    "amount":    4,   # "12000000" → ~3-4 토큰
    "code":      5,   # "1005-01-432-1234" → ~5 토큰
    "date":      3,   # "2026-04-09" → ~3 토큰
    "quantity":  2,   # "10" → ~1-2 토큰
    "number":    3,   # 일반 숫자
    "text":      5,   # 일반 텍스트
    "signature": 2,   # 서명
}

DEFAULT_EXPECTED_TOKENS = 5


# ─────────────────────────────────────────────
#  data_type → 필드 유형 매핑
# ─────────────────────────────────────────────

_DATA_TYPE_TO_FIELD_TYPE: dict[str, str] = {
    "number": "number",
    "integer": "number",
    "amount": "amount",
    "currency": "amount",
    "code": "code",
    "nsn": "code",
    "k_nsn": "code",
    "unit_code": "code",
    "equipment_id": "code",
    "date": "date",
    "quantity": "quantity",
    "text": "text",
    "string": "text",
    "signature": "signature",
    "rank": "text",
}


# ─────────────────────────────────────────────
#  핵심 함수
# ─────────────────────────────────────────────

def calc_field_confidence(
    token_logprobs: list[float],
    field_type: str = "text",
) -> float:
    """필드값 토큰들의 logprob → 신뢰도(0.0~1.0) 환산.

    기하 평균 + 길이 편향 보정:
      geo_mean = exp(mean(logprobs))
      length_factor = min(1.0, sqrt(expected_tokens / actual_tokens))
      confidence = geo_mean * length_factor

    긴 필드(부대명 등)는 토큰 수가 많아 기하평균이 구조적으로 낮아지므로,
    필드 유형별 기대 토큰 수 기반 보정 계수를 적용합니다.

    Args:
        token_logprobs: 해당 필드 토큰들의 개별 logprob 값 목록.
            vLLM logprobs 응답에서 추출. 예: [-0.02, -0.05, -0.01]
        field_type: 필드 유형 (text, number, code, date 등).
            EXPECTED_TOKENS 매핑에서 기대 토큰 수를 참조.

    Returns:
        필드 신뢰도 (0.0~1.0)
    """
    if not token_logprobs:
        return 0.0

    mean_logprob = sum(token_logprobs) / len(token_logprobs)

    # 매우 낮은 logprob 클램핑 (-20 이하는 실질적으로 0)
    mean_logprob = max(mean_logprob, -20.0)

    geo_mean = math.exp(mean_logprob)

    # 길이 편향 보정: 긴 필드의 구조적 낮은 점수 보정
    expected = EXPECTED_TOKENS.get(field_type, DEFAULT_EXPECTED_TOKENS)
    length_factor = min(1.0, math.sqrt(expected / max(len(token_logprobs), 1)))

    return round(min(1.0, max(0.0, geo_mean * length_factor)), 4)


def get_threshold(data_type: str) -> float:
    """필드 데이터 유형에 대응하는 신뢰도 임계값 반환.

    Args:
        data_type: 필드 데이터 유형 (text, number, date, code 등)

    Returns:
        임계값 (0.0~1.0)
    """
    field_type = _DATA_TYPE_TO_FIELD_TYPE.get(data_type, "text")
    return FIELD_THRESHOLDS.get(field_type, DEFAULT_THRESHOLD)


def is_flagged(confidence: float, data_type: str) -> bool:
    """신뢰도가 해당 필드 유형의 임계값 미달인지 판정.

    Args:
        confidence: 필드 신뢰도
        data_type: 필드 데이터 유형

    Returns:
        True이면 수동 검토 필요
    """
    return confidence < get_threshold(data_type)


# ─────────────────────────────────────────────
#  LogprobsScorer — 전체 VLM 응답 신뢰도 산출
# ─────────────────────────────────────────────

@dataclass
class FieldConfidenceResult:
    """필드별 신뢰도 산출 결과."""
    field_key: str
    data_type: str
    confidence: float
    threshold: float
    flagged: bool
    token_count: int


class LogprobsScorer:
    """VLM logprobs → 필드별 신뢰도 산출기.

    vLLM 응답의 logprobs를 파싱하여 각 필드의 토큰 확률을
    기하 평균으로 환산합니다.

    사용 예시:
        scorer = LogprobsScorer()

        # 개별 필드
        conf = scorer.score_field([-0.02, -0.05, -0.01], "number")
        # → 0.9737

        # 전체 VLM 응답에서 필드별 신뢰도 산출
        results = scorer.score_response(field_logprobs_map)
    """

    def score_field(
        self,
        token_logprobs: list[float],
        data_type: str = "text",
    ) -> FieldConfidenceResult:
        """단일 필드의 신뢰도 산출.

        Args:
            token_logprobs: 필드 토큰들의 logprob 목록
            data_type: 필드 데이터 유형

        Returns:
            FieldConfidenceResult
        """
        field_type = _DATA_TYPE_TO_FIELD_TYPE.get(data_type, "text")
        confidence = calc_field_confidence(token_logprobs, field_type)
        threshold = get_threshold(data_type)
        flagged = confidence < threshold

        return FieldConfidenceResult(
            field_key="",  # 호출자가 설정
            data_type=data_type,
            confidence=confidence,
            threshold=threshold,
            flagged=flagged,
            token_count=len(token_logprobs),
        )

    def score_response(
        self,
        field_logprobs: dict[str, tuple[list[float], str]],
    ) -> list[FieldConfidenceResult]:
        """VLM 응답 전체의 필드별 신뢰도 산출.

        Args:
            field_logprobs: {field_key: (token_logprobs, data_type)}

        Returns:
            FieldConfidenceResult 목록
        """
        results: list[FieldConfidenceResult] = []

        for field_key, (logprobs, data_type) in field_logprobs.items():
            result = self.score_field(logprobs, data_type)
            result.field_key = field_key
            results.append(result)

            if result.flagged:
                logger.warning(
                    "LogprobsScorer: %s flagged (conf=%.4f < threshold=%.2f, type=%s)",
                    field_key, result.confidence, result.threshold, data_type,
                )

        return results

    def calc_overall_confidence(
        self,
        field_results: list[FieldConfidenceResult],
    ) -> float:
        """전체 문서 신뢰도 산출 (필드별 신뢰도의 가중 평균).

        금액/코드 필드에 더 높은 가중치를 부여합니다.
        """
        if not field_results:
            return 0.0

        # 필드 유형별 가중치
        type_weights = {
            "amount": 2.0,
            "code": 1.5,
            "quantity": 1.5,
            "number": 1.5,
            "date": 1.2,
            "text": 1.0,
            "signature": 0.5,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for r in field_results:
            field_type = _DATA_TYPE_TO_FIELD_TYPE.get(r.data_type, "text")
            w = type_weights.get(field_type, 1.0)
            weighted_sum += r.confidence * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        return round(weighted_sum / total_weight, 4)
