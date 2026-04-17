"""FormType → 한국어 공식 명칭 매핑

P5 직렬화 단계에서 assembled_json.document_title 정적 주입에 사용합니다.
form_type은 P3-A에서 확정되므로 VLM 추출에 의존하지 않고 중앙 매핑을 참조합니다.

원본 서식 목록: docs/DOMAIN.md §1-1 FormType 분류표
"""

from __future__ import annotations

from typing import Optional

from src.interfaces.enums import FormType


FORM_TYPE_LABELS: dict[FormType, str] = {
    FormType.SUPPLY_REQUEST: "보급청구서",
    FormType.MAINTENANCE_RECORD: "정비기록서",
    FormType.INVENTORY_SHEET: "물자현황표",
    FormType.HANDOVER_DOC: "인수인계서",
    FormType.INSPECTION_REPORT: "검사보고서",
    FormType.EQUIPMENT_CHECKLIST: "전비품 확인서 작성 점검표",
    FormType.UNKNOWN: "군수 서식 유형 불명",
}


def get_document_title(form_type: FormType) -> Optional[str]:
    """form_type → 문서 공식 명칭. other는 S7 StructuredAggregator 미구현으로 None."""
    return FORM_TYPE_LABELS.get(form_type)
