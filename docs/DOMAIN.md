# 도메인 지식 가이드

> 군수 서식 유형, 코드 체계, JSON Schema 관리, Other 문서 처리 정책을 기술합니다.

---

## 1. 문서 분류 체계

### 1-1. FormType 분류 및 처리 경로

| FormType | 한국어 명칭 | 처리 경로 | 검토 큐 | 설명 |
|----------|-----------|---------|--------|------|
| `supply_request` | 보급청구서 | military | 조건부 | P4 군수 룰 검증 |
| `maintenance_record` | 정비기록서 | military | 조건부 | |
| `inventory_sheet` | 물자현황표 | military | 조건부 | |
| `handover_doc` | 인수인계서 | military | 조건부 | |
| `inspection_report` | 검사보고서 | military | 조건부 | |
| `unknown` | 군수 서식 유형 불명 | military | 조건부 | `_fallback.json` 사용 |
| `other` | 군수 서식 아님 | other | **없음** | 범용 OCR, 군수 룰 미적용 |

**`unknown` vs `other` 구분**:
- `unknown` = 군수 서식인데 어떤 서식인지 모름 → 군수 처리 경로 유지
- `other` = 군수 서식이 아님 → 범용 OCR 경로, 검토 큐 미적재

### 1-2. Other 문서 처리 정책

군수 서식이 아닌 문서(공문서, 지시문, 개인 서류 등)가 입력될 경우:

```
P3-A FormClassifier → form_type = "other"
    ↓
P2.5-B InstructionRouter
    → system_prompt: "문서 OCR 시스템입니다."
    → user_instruction: "이 영역의 텍스트를 인식하세요."
    → json_schema: _general.json
    ↓
P3-B StructuredExtractor
    → 범용 key-value 추출
    ↓
P4 Validator
    → 군수 룰 검증 건너뜀 (산술/날짜/NSN 교차검증 없음)
    → 신뢰도 산출만 수행
    ↓
P5 직렬화 → P6 DB 적재
    → PipelineOutput.status = "other_document"
    → 검토 큐 미적재
```

---

## 2. 주요 군수 서식 — 필드 명세

### 2-1. 보급청구서 (supply_request)

| 필드명 | 데이터 타입 | 설명 | 검증 룰 |
|--------|-----------|------|---------|
| `unit_code` | code | 청구 부대 코드 | 2~4자리 숫자 |
| `request_date` | date | 청구 일자 | YYYY-MM-DD |
| `nsn` | code | NATO 재고번호 | NNNN-NN-NNN-NNNN |
| `item_name` | text | 품목명 | — |
| `quantity` | number | 청구 수량 | 양의 정수 |
| `unit_price` | amount | 단가 | 양의 정수 |
| `total` | amount | 합계 | quantity × unit_price |
| `requester` | text | 청구자 서명 | — |

**주요 룰**: `total = quantity × unit_price` (산술 교차검증)

### 2-2. 정비기록서 (maintenance_record)

| 필드명 | 데이터 타입 | 설명 |
|--------|-----------|------|
| `equipment_id` | code | 장비 식별번호 |
| `maintenance_type` | text | 정비 유형 |
| `date` | date | 정비 일자 |
| `technician` | text | 정비 담당자 |
| `parts_used` | text | 사용 부품 |

### 2-3. 물자현황표 (inventory_sheet)

| 필드명 | 데이터 타입 | 설명 |
|--------|-----------|------|
| `item_name` | text | 품목명 |
| `stock_qty` | number | 현재 재고 수량 |
| `location` | text | 보관 위치 |
| `disposal_qty` | number | 폐기 수량 |

### 2-4. 인수인계서 (handover_doc)

| 필드명 | 데이터 타입 | 설명 |
|--------|-----------|------|
| `from_person` | text | 인계자 |
| `to_person` | text | 인수자 |
| `date` | date | 인수인계 일자 |
| `items` | text | 인수인계 항목 목록 |

---

## 3. 코드 체계

```
NSN (NATO Stock Number):    NNNN-NN-NNN-NNNN  (예: 1005-01-432-1234)
K-NSN (한국 물자코드):      KN-NNNNN-NNNN
부대 코드:                  숫자 2~4자리 + 한글 부대명 (예: "1232부대")
장비 식별번호:              알파벳 + 숫자 조합 (예: "K21-001")
```

**유사 문자 오인식 주의** (수기 OCR):
- `ㄱ / ㅋ`, `ㄴ / ㄹ`, `1 / ㅣ`, `0 / O`, `5 / S`, `6 / G`
- InstructionRouter의 user_instruction에 명시적으로 포함: "불확실한 글자는 [?]로 표시"

---

## 4. JSON Schema 관리

### 4-1. 스키마 디렉토리 구조

```
src/domain/schemas/
├── v1/
│   ├── supply_request.json
│   ├── maintenance_record.json
│   ├── inventory_sheet.json
│   ├── handover_doc.json
│   ├── inspection_report.json
│   ├── _fallback.json         ← unknown (군수 서식 유형 불명)
│   └── _general.json          ← other (군수 서식 아님)
└── v2/                        ← 서식 개정 시
    └── supply_request.json

src/domain/schema_registry.py  ← form_type + version → Schema 조회
```

`VLMResult.schema_id` 형식: `"supply_request:v1"`, `"other:v1"` (버전 포함)

### 4-2. CoT analysis 필드 구조 (군수 서식 전용)

모든 군수 서식 스키마의 최상단에 `analysis` 필드를 배치합니다. VLM이 필드 추출 전 이미지 품질과 모호한 문자를 간략히 기술하여 정확도를 높입니다.

```json
{
  "type": "object",
  "properties": {
    "analysis": {
      "type": "string",
      "description": "이미지 영역의 텍스트 품질, 레이아웃, 모호한 문자를 30~50 토큰으로 간략히 기술"
    },
    "unit_code": {"type": "string"},
    "request_date": {"type": "string"},
    ...
  },
  "required": ["analysis", "unit_code", "request_date", ...]
}
```

**효과**: 0-shot 대비 hallucination율 ~100% → ~1.8% 감소 (IEEE 2025, Hyperscience 2025 기준). 수기 한국어 OCR에서 모호한 획 구분에 특히 유효.

### 4-3. 스키마 버전 관리 정책

```
서식 개정 발생 시:
  1. v2/ 하위에 신규 스키마 작성
  2. schema_registry.py에 form_type:v2 매핑 추가
  3. DB 레코드에 schema_id 버전 포함 → 마이그레이션 가능
  4. VLM Fine-tuning 재수행 여부 판단 (필드 추가/삭제 시)
```

### 4-4. _general.json (other 문서용)

```json
{
  "type": "object",
  "properties": {
    "text_blocks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "region_id": {"type": "string"},
          "content": {"type": "string"}
        }
      }
    }
  }
}
```

군수 도메인 특화 필드 없이 영역별 텍스트만 추출.

---

## 5. 1-shot 예시 관리

### 5-1. 예시 파일 구조

```
configs/instruction_examples/
├── supply_request.yaml
├── maintenance_record.yaml
├── inventory_sheet.yaml
├── handover_doc.yaml
├── inspection_report.yaml
└── _fallback.yaml
```

### 5-2. 예시 YAML 형식

```yaml
# supply_request.yaml
region_type: text
form_type: supply_request
example_description: "수기 NSN 코드와 수량이 있는 보급청구서 기입란"
example_response: |
  {
    "analysis": "수기 필체 선명, NSN 코드 형식 확인됨, 수량 숫자 명확",
    "nsn": "1005-01-432-1234",
    "quantity": 10
  }
```

**prefix caching 최적화**: system_prompt를 모든 요청에 동일하게 유지(정적 prefix). 첫 요청 이후 KV 블록 재사용 → TTFT 3~10배 단축.

---

## 6. P4 도메인 룰 검증 상세

### 6-1. military 경로 룰

| 룰 ID | 대상 서식 | 검증 내용 | 오류 시 처리 |
|-------|---------|---------|-----------|
| ARITH-001 | supply_request | total = quantity × unit_price | 신뢰도 -0.30 |
| DATE-001 | 전체 | 날짜 순서 논리 위반 | 신뢰도 -0.10 |
| CODE-001 | supply_request | NSN 형식 NNNN-NN-NNN-NNNN | 신뢰도 -0.15 |
| CODE-002 | 전체 | K-NSN 형식 KN-NNNNN-NNNN | 신뢰도 -0.15 |
| MISS-001 | 전체 | 필수 필드 누락 | severity=HIGH |

### 6-2. other 경로 룰

군수 룰 검증 전체 건너뜀. 신뢰도 산출만 수행.
