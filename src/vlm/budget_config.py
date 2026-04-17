"""영역 타입별 공용 파라미터 — pixel_budget 등.

이전에는 instruction_router, resolution_router, skill_registry 세 파일에
동일/유사 매핑이 중복됐다. 값이 갈리면 경로 간 해상도 불일치가 발생하므로
단일 정의로 통합.

- PIXEL_BUDGETS: SigLIP max_pixels 단위 (실제 max_pixels = budget × 28 × 28)
- DISPATCH_ORDER: SkillRegistry 배치 처리 순서 (작은 것부터 → prefix cache 효율)
- FALLBACK_PIXEL_BUDGET: 매핑에 없는 region_type 기본값
"""

from __future__ import annotations

PIXEL_BUDGETS: dict[str, int] = {
    # 설계 업데이트 (2026-04-10): PIPELINE.md §2-4 기준
    "table":              1120,  # 셀 경계·미세 글씨
    "seal":               560,
    "text":               560,   # 소형 폰트·수기 대응 (280 → 560 상향)
    "formula":            280,
    "chart":              280,
    "figure":             140,
    "header":             140,
    "footer":             140,
    "handwritten_field":  1120,  # other 경로 Skill Registry 전용
    "signature":          140,
}

FALLBACK_PIXEL_BUDGET: int = 560

DISPATCH_ORDER: list[int] = [140, 560, 1120]
