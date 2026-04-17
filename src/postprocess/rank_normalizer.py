"""한국군 직급/계급 정규화 — VLM 오인식 교정용 후처리.

Schema enum 강제(1차 방어)가 실패하거나 fallback 경로에서
VLM이 비표준 계급명을 출력한 경우, Levenshtein 최근접 매칭으로 교정.
거리 > 2 이면 교정하지 않고 flag만 부여.
"""

from __future__ import annotations

RANK_LIST: list[str] = [
    "이병", "일병", "상병", "병장",
    "하사", "중사", "상사", "원사",
    "준위",
    "소위", "중위", "대위",
    "소령", "중령", "대령",
    "준장", "소장", "중장", "대장",
]

_RANK_SET: set[str] = set(RANK_LIST)


def normalize_rank(raw: str) -> tuple[str, bool, bool]:
    """raw → (정규화된 계급, 교정 여부, flag 여부).

    Returns:
        (normalized, corrected, flagged)
        - 원본이 유효하면: (raw, False, False)
        - 거리 ≤ 2 교정 성공: (nearest, True, False)
        - 거리 > 2 교정 불가: (raw, False, True) → 검토 큐 플래그 대상
    """
    if not raw or not raw.strip():
        return raw, False, True

    cleaned = raw.strip()
    if cleaned in _RANK_SET:
        return cleaned, False, False

    best_dist = 999
    best_match = cleaned
    for rank in RANK_LIST:
        d = _levenshtein(cleaned, rank)
        if d < best_dist:
            best_dist = d
            best_match = rank

    if best_dist <= 2:
        return best_match, True, False
    return cleaned, False, True


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dlt = curr[j] + 1
            sub = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(ins, dlt, sub))
        prev = curr
    return prev[-1]
