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
| `equipment_checklist` | 전비품 확인서 작성 점검표 | military | 조건부 | O/X 점검결과 6항목 추출 |
| `bid_application` | 입찰참가신청서 | military | 조건부 | 별지 제13호 서식(국내/국제), 방위사업청 조달 |
| `unknown` | 군수 서식 유형 불명 | military | 조건부 | `_fallback.json` 사용 |
| `other` | 군수 서식 아님 | other → Skill Registry | **없음** | 범용 OCR, 군수 룰 미적용 |

**`unknown` vs `other` 구분**:
- `unknown` = 군수 서식인데 어떤 서식인지 모름 → 군수 처리 경로 유지
- `other` = 군수 서식이 아님 → Skill Registry, 검토 큐 미적재

**한국어 명칭의 용도**: 위 표의 한국어 명칭은 `src/domain/form_type_labels.py`에 `FORM_TYPE_LABELS`로 중앙화되어 있으며, P5가 `assembled_json.document_title` 필드에 정적으로 주입합니다. form_type은 P3-A에서 확정되므로 VLM이 문서 제목을 추출하지 않으며, PP-DocLayout이 인쇄 제목 영역을 탐지해 VLM이 자의 생성한 `document_title` 키는 P5에서 제거됩니다. `other`는 S7 StructuredAggregator 미구현으로 None(주입 건너뜀).

### 1-2. Other 문서 처리 정책

군수 서식이 아닌 문서(공문서, 지시문, 개인 서류 등)가 입력될 경우:

```
P3-A FormClassifier → form_type = "other"
    ↓
Orchestrator → Skill Registry 디스패치
    S1: LayoutAnalyzer (결재란 휴리스틱 포함)
    S5-패스1: TableExtractor (표 구조 추출)
    S6(140): SignatureDetector
    S2(560): PrintedTextReader
    S3+S4+S5패스2(1120): HandwritingReader + SealReader + 셀 내용
    S7: StructuredAggregator
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

### 2-5. 전비품 확인서 작성 점검표 (equipment_checklist)

**서식 식별자**: 별지 제3-2호 서식

| 필드명 | 데이터 타입 | 설명 | 검증 룰 |
|--------|-----------|------|---------|
| `document_date` | text | 작성 일자 수기 문자열 | "YYYY년 MM월 DD일" 형태 권장 |
| `checklist_items` | array | 점검항목 1~6번 결과 배열 | 길이 = 6 (CHK-001) |
| `checklist_items[].item_number` | integer | 항목 번호 (1~6) | 1~6 순서 존재 (CHK-002) |
| `checklist_items[].result` | enum | 점검결과 | "O" / "X" / "?" (CHK-003) |
| `checklist_items[].result_confidence` | number | 항목별 인식 신뢰도 | 0.0~1.0 |
| `writer.team` | text | 소속 팀명 | — |
| `writer.rank` | text | 직급/계급 | — |
| `writer.name` | text | 성명 | 비어있지 않음 (CHK-004) |
| `writer.signature_present` | boolean | 서명 존재 여부 | 이진 탐지 |
| `form_identifier` | text | 서식 식별자 | 예: "별지 제3-2호 서식" |

**점검결과 값 의미**:
- `"O"` — 이상없음 / 해당함 (원형 수기 표시)
- `"X"` — 해당없음 / 부적합 (X자 수기 표시)
- `"?"` — 판독불가 (검토 큐 적재 대상)

### 2-6. 입찰참가신청서 (bid_application)

**서식 식별자**: 별지 제13호 서식 — 군수품조달관리규정(방위사업청) / 국내·국제 공통.

| 필드명 | 데이터 타입 | 설명 | 검증 룰 |
|--------|-----------|------|---------|
| `doc_number` | text | 제 호 — 문서번호 수기 기입란 | — |
| `applicant.company_name` | text | 상호 또는 법인명칭 | 필수 |
| `applicant.corporate_reg_number` | code | 법인등록번호 | NNNNN-NNNNNNN (BID-002) |
| `applicant.representative` | text | 대표자 성명 | 필수 |
| `applicant.representative_dob` | text | 대표자 생년월일 | — |
| `applicant.address` | text | 주소 | — |
| `applicant.phone` | text | 전화번호 | — |
| `applicant.business_location` | text | 사업장소재지 | — |
| `applicant.business_reg_number` | code | 사업자등록번호 | NNN-NN-NNNNN (BID-001, 필수) |
| `bid_info.announcement_number` | text | 입찰공고(지명)번호 | 필수 |
| `bid_info.bid_date` | date | 입찰일자 | 필수 |
| `bid_info.bid_name` | text | 입찰건명 | 필수 |
| `bid_info.item_code` | text | 품목코드 | — |
| `bid_info.industry_class_number` | text | 해당산업분류번호 | — |
| `bid_info.acquisition_date` | date | 취득일 | — |
| `bid_info.issuing_office` | text | 발행관서 | — |
| `agent.name` | text | 대리인 성명 | — |
| `agent.dob` | text | 대리인 생년월일 | — |
| `agent.seal_present` | boolean | 사용인감 날인 존재 여부 | — |
| `submission.submission_date` | date | 신청일자 | 필수 (BID-003) |
| `submission.submitter_name` | text | 신청인 대표자 성명 | 필수 (BID-004) |
| `seal_verification.issuing_office` | text | 인감증명 발급관서 | — |
| `seal_verification.issue_number` | code | 발급번호 | — |
| `seal_verification.issue_date` | date | 발급일자 | — |
| `seal_verification.verifier` | text | 화인자 | — |
| `seal_verification.signature_present` | boolean | 서명/날인 존재 여부 | — |
| `form_identifier` | text | 서식 식별자 | 예: "별지 제13호 서식" |

**블록 단위 sub-schema 분해**: TemplateAugmentor가 신청인·입찰개요·대리인·신청·인감증명확인을 블록 field_key로 부여하여 VLM이 블록 단위 구조화 JSON을 반환, Assembler가 x-assembly-rules로 재조립.

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
- InstructionRouter user_instruction에 명시: "불확실한 글자는 [?]로 표시"

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
│   ├── equipment_checklist.json ← 전비품 확인서 작성 점검표
│   ├── bid_application.json     ← 입찰참가신청서 (별지 제13호 서식)
│   ├── _fallback.json           ← unknown (군수 서식 유형 불명)
│   ├── _general.json            ← other 범용 (기존 단순 key-value)
│   └── official_document.json   ← other 공문서 전용
└── v2/                          ← 서식 개정 시
    └── supply_request.json

src/domain/schema_registry.py    ← form_type + version → Schema 조회
```

`VLMResult.schema_id` 형식: `"supply_request:v1"`, `"official_document:v1"` (버전 포함)

### 4-2. CoT analysis 필드 구조 (군수 서식 전용)

모든 군수 서식 스키마 최상단에 `analysis` 필드를 배치합니다.

```json
{
  "type": "object",
  "properties": {
    "analysis": {
      "type": "string",
      "description": "이미지 영역 텍스트 품질, 레이아웃, 모호한 문자를 30~50 토큰으로 간략히 기술"
    },
    "unit_code": {"type": "string"},
    "request_date": {"type": "string"}
  },
  "required": ["analysis", "unit_code", "request_date"]
}
```

**효과**: 0-shot 대비 hallucination율 ~100% → ~1.8% 감소 (IEEE 2025).

### 4-2-1. x-assembly-rules / x-checklist-item-schema (region 단위 추출용)

스키마 한 번을 페이지 1회 호출로 채우는 대신, **template field 단위 region별 sub-schema 호출** 후 Assembler가 재조립하는 패턴을 지원하기 위한 두 개의 비표준 키.

```json
{
  "x-assembly-rules": {
    "document_date":  "document_date",
    "result_item_1":  "checklist_items.0",
    ...
    "result_item_6":  "checklist_items.5",
    "writer_block":   "writer",
    "form_identifier": "form_identifier"
  },
  "x-checklist-item-schema": {
    "type": "object",
    "properties": {"item_number": {...}, "result": {...}, "result_confidence": {...}},
    "required": ["item_number", "result", "result_confidence"]
  }
}
```

- `x-assembly-rules`: TemplateAugmentor가 부여한 `field_key` → 최종 dict의 JSON Path 매핑. Assembler(`src/vlm/assembler.py`)가 이 규칙으로 region 결과를 `assembled_json`에 배치합니다.
- `x-checklist-item-schema`: 배열 원소 단위 sub-schema 정의 (xgrammar로 array.items에 직접 const 주입이 까다로워서 분리). InstructionRouter `_extract_sub_schema()`가 `result_item_N` field_key에 대해 이 스키마를 복사하고 `item_number: const=N` 을 주입해 사용합니다.
- `x-assembly-rules`가 없는 스키마는 Assembler가 None 반환 → 기존 fields[] 기반 처리 유지 (하위 호환).

### 4-3. official_document.json (공문서 — other 경로 S7 전용)

```json
{
  "type": "object",
  "properties": {
    "analysis": {
      "type": "string",
      "description": "문서 구조, 품질, 특이사항 간략 기술"
    },
    "document_type": {"type": "string"},
    "header": {
      "type": "object",
      "properties": {
        "organization": {"type": "string"},
        "document_number": {"type": "string"},
        "date": {"type": "string"},
        "classification": {"type": "string"}
      }
    },
    "recipient": {"type": "string"},
    "subject": {"type": "string"},
    "body": {"type": "string"},
    "attachments": {
      "type": "array",
      "items": {"type": "string"}
    },
    "approval_table": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": {"type": "string"},
          "name": {
            "type": "object",
            "properties": {
              "value": {"type": "string"},
              "confidence": {"type": "number"}
            }
          },
          "date": {
            "type": "object",
            "properties": {
              "value": {"type": "string"},
              "confidence": {"type": "number"}
            }
          },
          "signature_present": {"type": "boolean"},
          "stamp_present": {"type": "boolean"}
        }
      }
    },
    "official_seal": {
      "type": "object",
      "properties": {
        "detected": {"type": "boolean"},
        "text": {
          "type": "object",
          "properties": {
            "value": {"type": "string"},
            "confidence": {"type": "number"}
          }
        }
      }
    },
    "footer": {
      "type": "object",
      "properties": {
        "handler": {"type": "string"},
        "contact": {"type": "string"},
        "file_number": {"type": "string"}
      }
    },
    "low_confidence_fields": {
      "type": "array",
      "items": {"type": "string"}
    },
    "overall_confidence": {"type": "number"}
  },
  "required": ["analysis", "document_type"]
}
```

### 4-4. _general.json (기존 — 단순 key-value)

기존 other 경로의 단순 처리용. Skill Registry 도입 후에도 하위 호환용으로 유지.

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

### 4-5. 스키마 버전 관리 정책

```
서식 개정 발생 시:
  1. v2/ 하위에 신규 스키마 작성
  2. schema_registry.py에 form_type:v2 매핑 추가
  3. DB 레코드에 schema_id 버전 포함 → 마이그레이션 가능
  4. VLM Fine-tuning 재수행 여부 판단
```

---

## 5. 1-shot 예시 관리 (military 경로)

### 5-1. 예시 파일 구조

```
configs/instruction_examples/
├── supply_request.yaml
├── maintenance_record.yaml
├── inventory_sheet.yaml
├── handover_doc.yaml
├── inspection_report.yaml
├── equipment_checklist.yaml
├── bid_application.yaml       ← 신규 (별지 제13호 서식)
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

**prefix caching 최적화**: system_prompt 정적 유지 → 첫 요청 이후 KV 블록 재사용 → TTFT 3~10배 단축.

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
| CHK-001 | equipment_checklist | `checklist_items` 배열 길이 = 6 | severity=HIGH, 신뢰도 -0.20 |
| CHK-002 | equipment_checklist | `item_number` 값이 1~6 순서대로 존재 | severity=HIGH, 신뢰도 -0.10 |
| CHK-003 | equipment_checklist | `result` 값이 "O" / "X" / "?" 중 하나 | severity=HIGH, 신뢰도 -0.15 |
| CHK-004 | equipment_checklist | `writer.name` 비어있지 않음 | severity=MEDIUM |
| BID-001 | bid_application | `applicant.business_reg_number` 형식 NNN-NN-NNNNN | severity=HIGH, 신뢰도 -0.15 |
| BID-002 | bid_application | `applicant.corporate_reg_number` 형식 NNNNN-NNNNNNN | severity=HIGH, 신뢰도 -0.15 |
| BID-003 | bid_application | `submission.submission_date` 비어있지 않음 | severity=MEDIUM |
| BID-004 | bid_application | `submission.submitter_name` 비어있지 않음 | severity=MEDIUM |

### 6-2. other 경로 룰

군수 룰 검증 전체 건너뜀. 신뢰도 산출만 수행.
`low_confidence_fields` 목록 기반 수동 확인 권고.
