"""Skill 내부 공용 JSON 파서.

VLM 응답이 guided_json을 따르더라도 다음 패턴을 섞어 반환할 수 있다:
  1. ```json ... ``` 마크다운 코드 펜스
  2. 자연어 설명 + JSON 블록
  3. 단일 따옴표
  4. trailing comma / 절단
각 skill의 원본에서 발췌/이동.
"""

from __future__ import annotations

import json
import re
from typing import Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _loads_relaxed(raw: str) -> Optional[dict]:
    """raw → dict (코드펜스·중괄호 블록 3단계 폴백).

    dict만 반환. list/str/scalar는 None.
    seal_reader·signature_detector가 사용.
    """
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
    return None


def _loads_relaxed_any(raw: str):
    """raw → dict/list 모두 허용 (코드펜스·혼합 텍스트 대응).

    table_extractor가 pass1 응답 파싱에서 사용 (VLM이 array of cell objects 반환).
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    for ob, cb in (("{", "}"), ("[", "]")):
        start = raw.find(ob)
        if start >= 0:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == ob:
                    depth += 1
                elif raw[i] == cb:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw[start : i + 1])
                        except json.JSONDecodeError:
                            break
    return None
