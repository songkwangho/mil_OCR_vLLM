# vLLM 변동성 측정 리포트 — 전역지원서_2

- run_count: **3**
- 결정론 스코어 (field 일치율): **20.00%**
- processing_path 일관성: **True**
- form_type 일관성: **False**
- review_required 일관성: **True**
- overall_confidence 범위: **0.0000**

## 1. 타이밍 (total_ms)

| run | total_ms |
| --- | --- |
| 1 | 25080 |
| 2 | 10589 |
| 3 | 10968 |

- mean: 15545.9ms
- std:  8259.0ms
- min/max: 10589 / 25080ms

## 2. P4 overall_confidence

- mean: 0.0000
- std:  0.0000
- min/max: 0.0 / 0.0
- range: 0.0

## 3. 불안정 필드 (12개 / 총 15)

| field_key | unique_count | appeared_in_runs | unique_values | logprob_std |
| --- | --- | --- | --- | --- |
| corrected_text | 1 | 1 | 당기 업무 보고서에 반영 요청사항을 반영하여 작성하여, | 0.000 |
| document_info | 1 | 1 | {"title": null, "date": "2017. | 0.000 |
| document_title | 3 | 3 | None / 미확인 (군수 관련 서식 추정) / 전이지원서 | 0.000 |
| fields | 4 | 4 | [{"key": "미확인", "value": "미확인" / [{"key": "안내 문구", "value": "당기 / {"date_field": "2017. 04. 11"} / {} | 0.000 |
| handwritten_content | 1 | 1 | None | 0.000 |
| handwritten_notes | 1 | 1 | [{"text": "2017. 04. 11", "con | 0.000 |
| header_text | 1 | 2 | 전이지원서 | 0.000 |
| items | 1 | 7 | [] | 0.000 |
| ocr_correction | 1 | 1 | {"original_detected": "미확인 텍스트 | 0.000 |
| request_date | 3 | 7 | 2017-04-11 / None / [?] | 0.000 |
| tables | 1 | 1 | [] | 0.000 |
| unit_code | 2 | 10 | None / [?] | 0.000 |

## 4. assembled_json 불안정 path (0개 / 총 0)

_모든 leaf path 일관됨._

## 5. run별 간이 상태

| run | status | path | form_type | field_count | overall_conf | review |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | review | vlm | unknown | 25 | 0.0 | True |
| 2 | review | vlm | supply_request | 10 | 0.0 | True |
| 3 | review | vlm | supply_request | 13 | 0.0 | True |
