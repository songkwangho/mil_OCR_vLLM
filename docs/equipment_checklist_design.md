# 전비품 확인서 작성 점검표 — FormType / Schema / Template 설계

> 대상 서식: 【별지 제3-2호 서식】 전비품 확인서 작성 점검표
> 반영 파일: DOMAIN.md, PIPELINE.md (TemplateAugmentor YAML),
>           src/domain/schemas/v1/equipment_checklist.json,
>           configs/form_templates/equipment_checklist.yaml,
>           configs/instruction_examples/equipment_checklist.yaml

---

## 1. 서식 구조 분석

```
【별지 제3-2호 서식】              ← 인쇄 고정 (서식 식별자)

        전비품 확인서 작성 점검표   ← 인쇄 고정 (제목)

                           202X년 X월 X일  ← 수기

┌──────┬─────────────────────────────────────────┬──────────┐
│ 구분 │              점 검 내 용                │ 점검결과 │
├──────┼─────────────────────────────────────────┼──────────┤
│      │ 1. 해당 물품이 해당 계약번호에 ...      │    ○     │
│      ├─────────────────────────────────────────┼──────────┤
│      │ 2. 계약업체의 명칭과 주소는 ...         │    ○     │
│      ├─────────────────────────────────────────┼──────────┤
│ 점검 │ 3. (당초) 계약번호, 품명, 수량, ...     │    ○     │
│ 사항 ├─────────────────────────────────────────┼──────────┤
│      │ 4. 해당 품목이 「군수품관리법 시행령」 ..│    ×     │
│      ├─────────────────────────────────────────┼──────────┤
│      │ 5. 방위사업청이 외국으로부터 수입하는 ..│    ×     │
│      ├─────────────────────────────────────────┼──────────┤
│      │ 6. 수입대행계약을 체결하지 않음에도 ... │    ○     │
└──────┴─────────────────────────────────────────┴──────────┘

작성자: ( 군수관리팀 ) 팀   대위 (직급)   김정식 (성명)   [서명]
```

**핵심 특성**:

| 특성 | 설명 |
|------|------|
| 점검결과 값 | O / X 이진값. 수기 기입. |
| 점검항목 수 | 6개 고정 (서식 버전이 바뀌지 않는 한 불변) |
| 점검내용 텍스트 | 인쇄 고정 → 추출 불필요, 항목 번호(1~6)로 식별 |
| 날짜 | 수기. 연/월/일 분리 기재 형식 (20XX년 X월 X일) |
| 작성자 정보 | 수기. 팀명, 직급, 성명, 서명 4개 필드 |
| 서명 | 이진 탐지 대상 (OCR 아님, S6 SignatureDetector와 동일 처리) |

---

## 2. FormType 추가

### 2-1. enum 추가

```python
# src/interfaces/enums.py
class FormType(str, Enum):
    SUPPLY_REQUEST     = "supply_request"
    MAINTENANCE_RECORD = "maintenance_record"
    INVENTORY_SHEET    = "inventory_sheet"
    HANDOVER_DOC       = "handover_doc"
    INSPECTION_REPORT  = "inspection_report"
    EQUIPMENT_CHECKLIST = "equipment_checklist"   # ← 신규
    UNKNOWN            = "unknown"
    OTHER              = "other"
```

### 2-2. DOMAIN.md FormType 분류표 갱신

| FormType | 한국어 명칭 | 처리 경로 | 검토 큐 | 설명 |
|----------|-----------|---------|--------|------|
| `supply_request` | 보급청구서 | military | 조건부 | |
| `maintenance_record` | 정비기록서 | military | 조건부 | |
| `inventory_sheet` | 물자현황표 | military | 조건부 | |
| `handover_doc` | 인수인계서 | military | 조건부 | |
| `inspection_report` | 검사보고서 | military | 조건부 | |
| `equipment_checklist` | 전비품 확인서 작성 점검표 | military | 조건부 | 점검항목 O/X 결과 추출 |
| `unknown` | 군수 서식 유형 불명 | military | 조건부 | `_fallback.json` 사용 |
| `other` | 군수 서식 아님 | other → Skill Registry | 없음 | |

### 2-3. FormClassifier guided_json enum 갱신

```python
FORM_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "form_type": {
            "type": "string",
            "enum": [
                "supply_request",
                "maintenance_record",
                "inventory_sheet",
                "handover_doc",
                "inspection_report",
                "equipment_checklist",   # ← 추가
                "unknown",
                "other"
            ]
        },
        "confidence": {"type": "number"}
    },
    "required": ["form_type", "confidence"]
}
```

**분류 힌트** (P3-A InstructionRouter system_prompt에 추가):

```
- equipment_checklist: 제목에 "전비품 확인서 작성 점검표" 또는 "별지 제3-2호"가 있고,
  O/X 점검결과 열이 있는 체크리스트 서식
```

---

## 3. JSON Schema 설계

> `src/domain/schemas/v1/equipment_checklist.json`

### 3-1. 설계 원칙

점검결과(O/X)가 이 서식의 핵심 정보입니다. 다음 원칙으로 설계합니다.

- 점검항목 텍스트는 인쇄 고정이므로 **추출하지 않음** — 항목 번호(1~6)만 키로 사용
- 점검결과는 `"O"` / `"X"` / `"?"` (판독 불가) 세 가지 값만 허용
- `"?"` 허용 이유: O와 X의 수기 형태가 모호할 수 있음 (열린 O vs 닫힌 O 등)
- 날짜는 수기 전체 문자열로 추출 (`"2026년 4월 15일"` 형태)
- 작성자 정보는 팀명/직급/성명을 분리 추출
- 서명은 `signature_present: boolean` (이진 탐지)
- `analysis` 필드 최상단 배치 (CoT)

### 3-2. Schema 전문

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "equipment_checklist",
  "description": "전비품 확인서 작성 점검표 (별지 제3-2호 서식)",
  "type": "object",
  "properties": {
    "analysis": {
      "type": "string",
      "description": "이미지 품질, 필체 명확도, O/X 판독 난이도를 30~50 토큰으로 간략히 기술"
    },
    "document_date": {
      "type": "string",
      "description": "작성 일자 수기 문자열. 예: '2026년 4월 15일'"
    },
    "checklist_items": {
      "type": "array",
      "description": "점검항목 1~6번의 점검결과 목록. 항목 순서 유지.",
      "minItems": 6,
      "maxItems": 6,
      "items": {
        "type": "object",
        "properties": {
          "item_number": {
            "type": "integer",
            "description": "점검항목 번호 (1~6)"
          },
          "result": {
            "type": "string",
            "enum": ["O", "X", "?"],
            "description": "점검결과. O=이상없음, X=해당없음/부적합, ?=판독불가"
          },
          "result_confidence": {
            "type": "number",
            "description": "이 항목 결과의 인식 신뢰도 (0.0~1.0)"
          }
        },
        "required": ["item_number", "result", "result_confidence"]
      }
    },
    "writer": {
      "type": "object",
      "description": "작성자 정보",
      "properties": {
        "team": {
          "type": "string",
          "description": "소속 팀명. 예: '군수관리팀'"
        },
        "rank": {
          "type": "string",
          "description": "직급/계급. 예: '대위'"
        },
        "name": {
          "type": "string",
          "description": "성명. 예: '김정식'"
        },
        "signature_present": {
          "type": "boolean",
          "description": "서명란에 수기 서명이 존재하는지 여부"
        }
      },
      "required": ["team", "rank", "name", "signature_present"]
    },
    "form_identifier": {
      "type": "string",
      "description": "서식 식별자. 예: '별지 제3-2호 서식'"
    }
  },
  "required": ["analysis", "document_date", "checklist_items", "writer"]
}
```

---

## 4. P4 검증 룰 추가

> DOMAIN.md §6-1 military 경로 룰 테이블에 추가

| 룰 ID | 대상 서식 | 검증 내용 | 오류 시 처리 |
|-------|---------|---------|-----------|
| CHK-001 | equipment_checklist | `checklist_items` 배열 길이 = 6 | severity=HIGH, 신뢰도 -0.20 |
| CHK-002 | equipment_checklist | `item_number` 값이 1~6 순서대로 존재 | severity=HIGH, 신뢰도 -0.10 |
| CHK-003 | equipment_checklist | `result` 값이 "O" / "X" / "?" 중 하나 | severity=HIGH, 신뢰도 -0.15 |
| CHK-004 | equipment_checklist | `writer.name` 비어있지 않음 | severity=MEDIUM |
| DATE-001 | 전체 (기존) | 날짜 순서 논리 위반 | 신뢰도 -0.10 |

**CHK-001 의도**: 6개 항목 중 일부가 누락되면 레이아웃 탐지 실패일 가능성이 높으므로 검토 큐 적재.

---

## 5. TemplateAugmentor YAML

> `configs/form_templates/equipment_checklist.yaml`
>
> 좌표 기준: P1 Deskew + DPI 300 정규화 이후 이미지 (A4 세로 = 2480×3508px)
> 실측 샘플: 전비품_확인서_작성_점검표.pdf (2026-04-15) — 좌표 확정

### 5-1. 버전 관리 방식 (C3 결정: 옵션 B)

서식 개정 시 **단일 yaml 파일 내 `versions` 배열**로 관리합니다. 파일을 분기하지 않습니다.

버전 선택 기준은 **`form_identifier` 문자열**입니다. FormClassifier가 form_type 분류 시 서식 상단의 식별자(`별지 제3-2호 서식`)를 함께 추출하고, TemplateAugmentor가 이 값으로 해당 버전의 `fields`를 선택합니다. `form_identifier`가 없거나 매칭되지 않으면 배열의 첫 번째 버전(최신)을 기본으로 사용합니다.

```
FormClassifier → form_type="equipment_checklist"
                 form_identifier="별지 제3-2호 서식"   ← guided_json에 추가
                         ↓
TemplateAugmentor → versions 배열에서 form_identifier 매칭
                  → 해당 version의 fields 로드
```

이 방식은 Schema 버전(`equipment_checklist:v1`, `equipment_checklist:v2`)과도 1:1 대응됩니다.

### 5-2. FormClassifier guided_json 변경

```python
FORM_TYPE_SCHEMA = {
    "properties": {
        "form_type":       {"type": "string", "enum": [...]},
        "confidence":      {"type": "number"},
        "form_identifier": {"type": "string"}   # ← 신규 (optional)
        # 예: "별지 제3-2호 서식" | null
        # 식별자가 없는 서식(보급청구서 등)은 null 허용
    },
    "required": ["form_type", "confidence"]
}
```

### 5-3. 확정 YAML

```yaml
# configs/form_templates/equipment_checklist.yaml
#
# 전비품 확인서 작성 점검표 (별지 제3-2호 서식)
#
# 좌표 기준:
#   - P1 Deskew + DPI 300 정규화 이후 이미지
#   - A4 세로 스캔 기준: 2480×3508 px
#   - 실측 샘플: 전비품_확인서_작성_점검표.pdf (2026-04-15) — 확정
#
# 설계 원칙:
#   - 점검내용 텍스트 칸(1~6번 질문)은 지정하지 않음
#     → 추출 대상 아님 + PP-DocLayout이 인쇄 텍스트 영역으로 정상 탐지
#   - 수기 기입란(O/X 결과, 날짜, 작성자)만 지정
#   - 작성자 영역은 단일 박스로 통합
#     → 작성자마다 팀명/직급/성명/서명 기입 위치가 좌/우로 다를 수 있음

form_type: equipment_checklist

# 참고 좌표 (TemplateAugmentor 병합 대상 아님 — 구조 문서화 목적)
reference_regions:
  header_title:
    bbox: {x1: 812, y1: 440, x2: 1676, y2: 532}
    description: "제목 인쇄 텍스트"
  table_full:
    bbox: {x1: 408, y1: 748, x2: 2280, y2: 2716}
    description: "점검표 전체 테이블 경계"

# 버전별 섹션 — form_identifier 매칭으로 선택
versions:
  - version: "1.0"
    effective_date: "2026-01-01"
    form_identifier: "별지 제3-2호 서식"
    page_size: [2480, 3508]
    fields:
      # ── 날짜 ────────────────────────────────
      - field_key: document_date
        region_type: handwritten_field
        bbox: {x1: 1812, y1: 636, x2: 2272, y2: 732}
        description: "작성 일자 수기란 (예: 2026년 4월 15일)"

      # ── 점검결과 O/X (헤더행 y=748~892 제외) ──
      - field_key: result_item_1
        region_type: handwritten_field
        bbox: {x1: 2036, y1: 892, x2: 2260, y2: 1188}
        description: "점검항목 1번 결과 (O/X)"

      - field_key: result_item_2
        region_type: handwritten_field
        bbox: {x1: 2036, y1: 1188, x2: 2260, y2: 1484}
        description: "점검항목 2번 결과 (O/X)"

      - field_key: result_item_3
        region_type: handwritten_field
        bbox: {x1: 2032, y1: 1488, x2: 2256, y2: 1788}
        description: "점검항목 3번 결과 (O/X)"

      - field_key: result_item_4
        region_type: handwritten_field
        bbox: {x1: 2036, y1: 1800, x2: 2256, y2: 2088}
        description: "점검항목 4번 결과 (O/X)"

      - field_key: result_item_5
        region_type: handwritten_field
        bbox: {x1: 2032, y1: 2092, x2: 2256, y2: 2396}
        description: "점검항목 5번 결과 (O/X)"

      - field_key: result_item_6
        region_type: handwritten_field
        bbox: {x1: 2032, y1: 2400, x2: 2256, y2: 2696}
        description: "점검항목 6번 결과 (O/X)"

      # ── 작성자 전체 (단일 박스) ──────────────
      - field_key: writer_block
        region_type: handwritten_field
        bbox: {x1: 304, y1: 2748, x2: 2292, y2: 2872}
        description: "작성자 전체 수기란 (팀명·직급·성명·서명 포함)"

  # 서식 개정 시 아래 블록 추가
  # - version: "2.0"
  #   effective_date: "YYYY-MM-DD"
  #   form_identifier: "별지 제3-2호의X 서식"
  #   page_size: [2480, 3508]
  #   fields: [...]
```

---

## 6. 1-shot 예시 YAML

> `configs/instruction_examples/equipment_checklist.yaml`

```yaml
form_type: equipment_checklist
region_type: table
example_description: "전비품 확인서 작성 점검표 — O/X 점검결과 열 포함 전체 표"
example_response: |
  {
    "analysis": "표 구조 명확, 점검결과 열의 O/X 수기 판독 가능. 3번 항목 O가 다소 흐릿하나 판독 가능.",
    "document_date": "2026년 4월 15일",
    "checklist_items": [
      {"item_number": 1, "result": "O", "result_confidence": 0.97},
      {"item_number": 2, "result": "O", "result_confidence": 0.95},
      {"item_number": 3, "result": "O", "result_confidence": 0.88},
      {"item_number": 4, "result": "X", "result_confidence": 0.96},
      {"item_number": 5, "result": "X", "result_confidence": 0.98},
      {"item_number": 6, "result": "O", "result_confidence": 0.94}
    ],
    "writer": {
      "team": "군수관리팀",
      "rank": "대위",
      "name": "김정식",
      "signature_present": true
    },
    "form_identifier": "별지 제3-2호 서식"
  }
```

---

## 7. InstructionRouter 프롬프트 설계

> `equipment_checklist` + `table` 영역에 적용되는 instruction

```
System:
You are a Korean military document OCR assistant specializing in checklist forms.
Extract all check results from this equipment confirmation checklist (전비품 확인서 작성 점검표).

Rules:
- Check result values: "O" (circle, 이상없음), "X" (cross, 해당없음/부적합), "?" (illegible)
- O and X are handwritten — carefully distinguish open circle (O) from an X mark
- Extract EXACTLY 6 items in order (item_number 1 through 6)
- For writer info: extract team name inside parentheses, rank, and full name
- signature_present: true if a handwritten signature mark exists in the signature box

Similar character pairs to watch:
  - O vs 0 (zero): O is a circle mark, not a digit
  - X vs × vs ✕: all count as "X"
  - Incomplete O (open at top) still counts as "O" if clearly intended as a circle

[Example]
Image: <example_crop>
Output: <example_response from instruction_examples/equipment_checklist.yaml>
```

---

## 8. 파급 범위 정리

| 파일 | 변경 내용 |
|------|---------|
| `src/interfaces/enums.py` | `FormType.EQUIPMENT_CHECKLIST` 추가 |
| `src/domain/schemas/v1/equipment_checklist.json` | **신규 생성** |
| `configs/form_templates/equipment_checklist.yaml` | **신규 생성** (TemplateAugmentor용, versions 배열 구조) |
| `configs/instruction_examples/equipment_checklist.yaml` | **신규 생성** |
| `src/vlm/form_classifier.py` | FORM_TYPE_SCHEMA에 `equipment_checklist` 추가 + `form_identifier` 필드 추가 |
| `src/vlm/template_augmentor.py` | versions 배열 + form_identifier 매칭 로직 반영 |
| `src/postprocess/validator.py` | CHK-001~004 룰 추가 |
| `docs/DOMAIN.md` | §1-1 FormType 표, §2 필드명세, §6-1 룰 테이블 갱신 |

P1~P3-B 컴포넌트 코어 로직은 **변경 없음**.

---

## 9. 통합 테스트 케이스 추가 (TESTING.md 반영 필요)

| # | 케이스 | 검증 포인트 |
|---|--------|-----------|
| T_checklist_1 | 전비품확인서 PDF 1페이지 정상 입력 | form_type=equipment_checklist, 6개 항목 추출, O/X 정확도 |
| T_checklist_2 | O/X 판독 경계 케이스 | 흐릿한 O(item 3번 기준) → `result_confidence` < 0.80 → OCR 힌트 재시도 |
| T_checklist_3 | CHK-001 트리거 | TemplateAugmentor 미적용 시 항목 누락 → 검토 큐 적재 확인 |
| T_checklist_4 | 서명 탐지 | `writer.signature_present = true/false` 이진 분류 정확도 |

---

## 10. 미결 사항

모든 미결 사항 해결 완료 (2026-04-15).

| # | 항목 | 상태 | 해결 내용 |
|---|------|------|---------|
| C1 | TemplateAugmentor bbox 실측 | ✅ 해결 | 실제 PDF 300dpi 렌더링 후 좌표 실측 확정. §5-3 yaml 반영 완료. |
| C2 | O/X 판독 임계값 | ✅ 해결 | 기존 handwritten 임계값(0.75) 적용. 실운영 데이터 축적 후 조정. |
| C3 | 서식 버전 다양성 | ✅ 해결 | 단일 yaml + `versions` 배열 구조 채택 (옵션 B). `form_identifier` 문자열 매칭으로 버전 선택. FormClassifier guided_json에 `form_identifier` 필드 추가. §5-1~5-2 참조. |
