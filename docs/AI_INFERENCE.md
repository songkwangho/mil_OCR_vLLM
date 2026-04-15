# AI 추론 (Inference) 가이드

> 모델 가중치, vLLM 서빙 설정, 성능 최적화, Docker 환경을 기술합니다.
> **폐쇄망 환경 전용** — 외부 API·다운로드 금지.

---

## 1. 모델 가중치 현황

| 모델 | 위치 | 크기 | 상태 |
|------|------|------|------|
| Real-ESRGAN x2plus | `models/t1_sr/RealESRGAN_x2plus.pth` | 64MB | ✅ 배치 완료 |
| PP-DocLayout_plus-L | `models/t2_layout/PP-DocLayout_plus-L/` | 126MB | ✅ 배치 완료 (fusion 교차 사용) |
| PP-DocLayoutV3 | `models/t2_layout/PP-DocLayoutV3/` | 132MB | ✅ 배치 완료 (기본 모델) |
| Gemma4 26B-A4B | `models/gemma4/gemma-4-26b-a4b-it/` | ~48GB BF16 | ✅ 배치 완료 (2026-04-08 검증) |

**Fallback 모델** (`models/fallback/`) — VLM 불가 시 사용:

| 모델 | 크기 | 역할 | 상태 |
|------|------|------|------|
| DiT | ~350MB | T3 서식 분류 | ⚠️ 가중치 미배치 |
| PP-OCRv5 Korean | ~15MB | T4 텍스트 인식 | ⚠️ 가중치 미배치 |
| SLANeXt_wired | ~30MB | T5 표 구조 인식 | ⚠️ 가중치 미배치 |

---

## 2. Gemma4 모델 특성

| 항목 | 값 |
|------|------|
| 총 파라미터 | 26B |
| 활성 파라미터 | ~4B (MoE, 레이어당 2/26 전문가 활성) |
| 컨텍스트 길이 | 256K 토큰 |
| 비전 인코더 | SigLIP (16×16 패치, 3×3 블록 average pooling) |
| 비전 입력 | 가변 해상도 70~1120 토큰/이미지 |
| 한국어 지원 | ✅ (40+ 언어) |
| 라이선스 | Apache 2.0 |
| 추론 VRAM (BF16) | ~48GB |
| MTP 헤드 | ❌ 미포함 → Speculative Decoding은 n-gram 방식 사용 |

**pixel_budget ↔ 유효 해상도 매핑** (정사각 이미지 기준):

| pixel_budget | 유효 해상도 | 권장 사용처 |
|:---:|:---:|------|
| 140 | ~189px | 서식 분류 (P3-A), 헤더/푸터 |
| 280 | ~268px | 수식, 차트 |
| 560 | ~378px | 일반 텍스트, 인장 |
| 1120 | ~535px | **표, 수기 기입란** (핵심 영역) |

---

## 3. vLLM 서빙 설정 (H100 80GB 최적화)

### 3-1. 권장 서버 시작 명령

```bash
# docker-compose.yml vllm-server 서비스에 반영
vllm serve models/gemma4/gemma-4-26b-a4b-it/ \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --mm-cache-preprocessor \
    --guided-decoding-backend xgrammar \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":1}' \
    --disable-log-requests
```

> **운영 환경**: vLLM v0.19.0 / H100 80GB / BF16 / 모델 로드 ~48.5GiB / 기동 ~120초

### 3-2. 최적화 설정 근거

| 설정 | 효과 | 비고 |
|------|------|------|
| `--kv-cache-dtype fp8` | KV Cache 메모리 50% 절감 → 동시처리 2배 | ⚠️ 모델 가중치는 BF16 유지 (§3-3 참조) |
| `--max-num-seqs 64` | 멀티모달 배치 안정성 | 128→64로 조정 |
| `--max-num-batched-tokens 16384` | Chunked Prefill 최적화 | 온라인 서빙 기준 |
| `--mm-cache-preprocessor` | 멀티모달 prefix caching 활성화 | TTFT 3~10배 단축 |
| `--guided-decoding-backend xgrammar` | 반복 스키마 캐싱 | 동일 서식 반복 사용에 유리 |
| n-gram speculative decoding | 디코드 속도 1.2~1.5배 향상 | guided_json과 완전 호환 |

### 3-3. KV Cache FP8 vs 모델 가중치 FP8 — 개념 구분

이 두 가지는 완전히 다른 개념입니다.

| 구분 | `--kv-cache-dtype fp8` (현재 적용) | FP8 Dynamic (Phase 3 실험) |
|------|----------------------------------|-----------------------------|
| 대상 | 추론 중 생성되는 KV Cache | 모델 가중치 + 활성화 함수 |
| VRAM 절감 | KV Cache ~50% 절감 | 모델 가중치 ~44GB → ~27GB |
| 품질 영향 | 거의 없음 | ~0.3% 손실 |
| vLLM 호환 | v0.19.0 ✅ | vllm#39049 Open → 해소 후 |
| 현재 상태 | ✅ **적용 중** | ❌ 미적용 |

H100 단일 GPU 환경에서 **최적 조합**: BF16 가중치 + FP8 KV Cache
→ 가중치 정확도 완전 보존 + KV Cache 메모리 절감으로 동시 처리량 2배

### 3-4. Speculative Decoding 전략 비교

Gemma4는 MTP 헤드가 없어 네이티브 추측 디코딩이 불가합니다. 세 가지 대안을 비교합니다.

| 방식 | 추가 VRAM | 속도 향상 | 추천 여부 | 근거 |
|------|----------|----------|---------|------|
| **n-gram** (현재) | 0GB | 1.2~1.5배 | ✅ 1순위 | 구조화 JSON 반복 패턴, guided_json 호환, 추가 비용 없음 |
| Draft 모델 (gemma-4-E2B-it) | ~4GB | 2~3배 | 🟡 Phase 3 | KV Cache 여유 감소, 짧은 OCR 출력(50~200토큰)에서 이점 작음 |
| Suffix Decoding | 0GB | 1.5~2배 | 🟡 Phase 3 | 반복 처리 시 acceptance rate 누적 향상 |

n-gram이 1순위인 이유: OCR 출력이 짧고(`"nsn": "1005-01-432-1234"` 같은 반복 JSON 키), guided_json으로 구조화되어 있어 n-gram acceptance rate가 높습니다.

### 3-5. Prefix Caching 전략

군수 서식 파이프라인에서 system_prompt는 서식 유형별로 동일하게 반복됩니다.

```
첫 번째 요청:  [system_prompt + 1-shot 예시] + [이미지 토큰] → KV 블록 생성
이후 동일 유형: [캐시된 KV 블록 재사용]     + [이미지 토큰] → TTFT 대폭 단축
```

효과: 10K 토큰 공유 prefix 기준 TTFT ~4.3초 → ~0.6초 (7배 단축)

**주의**: 동일 텍스트 + 다른 이미지 조합 시 캐시 키 충돌 버그(vllm#20261). `--mm-cache-preprocessor`로 이미지 해시 기반 캐시 사용 권장.

### 3-6. Guided Decoding 백엔드 선택

| 상황 | 권장 백엔드 | 이유 |
|------|-----------|------|
| 동일 서식 스키마 반복 (기본) | `xgrammar` | 스키마 캐싱 → throughput 우수 |
| NSN regex 패턴 강제 필요 | `guidance` | `pattern` 제약 완전 지원, 타임아웃 없음 |

**xgrammar 한계**: JSON Schema 내 `pattern` (regex) 제약 미지원. NSN 코드(`\d{4}-\d{2}-\d{3}-\d{4}`) 형식 강제 방법:

```python
# 방법 1: InstructionRouter에서 프롬프트 힌트로 처리 (권장 — xgrammar 유지)
user_instruction += "\nNSN 코드는 NNNN-NN-NNN-NNNN 형식 13자리입니다. 예: 1005-01-432-1234"

# 방법 2: guidance 백엔드 전환 (NSN 강제가 절대적으로 필요한 경우)
# --guided-decoding-backend guidance
# → JSON Schema 내 pattern 제약 완전 지원
```

---

## 4. 컴포넌트별 추론 설정

### 4-1. P1 — 화질 보정 + SR

**처리 순서**:
1. DPI 측정 → LOW DPI (< 150): Real-ESRGAN x2 초해상도 적용
2. DPI 300 표준화 (SR 적용 후)
3. CLAHE 대비 강화
4. 기울기 보정 (Deskew) — 2D RoPE 위치 인코딩에 직접 도움
5. 품질 점수 산출

**절대 금지 사항**:
- **이진화(Binarization) 금지**: SigLIP 인코더는 RGB 자연 이미지 기준으로 학습됨. DataUnboxed 벤치마크(2025)에서 이진화가 VLM 정확도를 오히려 저하시키거나 미미한 차이를 보임
- **그레이스케일 변환 금지**: RGB 3채널 정보 손실
- **과도한 샤프닝 금지**: 실제로 흐릿한 경우에만 가벼운 unsharp mask 적용

**VLM OCR 전처리 원칙**: "덜 하는 것이 정답". VLM은 노이즈가 있는 자연스러운 이미지에서 문맥적으로 누락 글자를 추론하므로, 인위적 처리를 최소화합니다.

### 4-2. P2 — 레이아웃 탐지

Fusion 모드 상세: `docs/PIPELINE.md §3-1` 참조.

### 4-3. P3-A — FormClassifier

```python
# vLLM 호출 (140토큰, 단일 동기 호출)
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": full_page_base64}},
        {"type": "text", "text": "이 문서의 서식 유형을 분류하세요."}
    ]}],
    extra_body={
        "guided_json": FORM_TYPE_SCHEMA,  # enum: supply_request|...|unknown|other
        "mm_processor_kwargs": {"max_soft_tokens": 140},
    },
    temperature=0.0,  # 결정론적 출력 보장
)
```

### 4-4. P2.5-B — InstructionRouter

**프롬프트 하니스 설계 원칙**:

```
✅ 영어 지시문 + 한국어 필드명 하이브리드
   이유: VLM은 영어 지시 따르기에 뛰어나지만,
         한국어 필드명이 출력 정확도를 높임
         (KLOCR 논문, 다국어 프롬프팅 서베이 2025.05)

✅ analysis 필드를 JSON Schema 최상단에 배치
   이유: VLM이 답변 전 사고 과정 → 수기 획 구분 정확도 향상
   길이 제한: "briefly note any issues" → ~30~50 토큰 유지
              ("이미지 품질, 모호한 문자를 한 문장으로 기술하세요.")

✅ 1-shot 예시를 system_prompt에 고정 삽입
   이유: 0→1-shot 전환 시 hallucination 율 ~100% → ~1.8% 감소
   구현: system_prompt를 정적으로 유지 → prefix caching 효율 극대화
         가변 콘텐츠(실제 OCR 이미지)는 프롬프트 끝에 배치

✅ 도메인 패턴 형식 힌트 명시
   예: "NSN 코드는 NNNN-NN-NNN-NNNN 형식 13자리입니다."
       "불확실한 글자는 [?]로 표시하세요."
       "유사 문자 주의: ㄱ/ㅋ, 1/ㅣ, 0/O"

✅ OCR 힌트 삽입 (저신뢰 영역 한정)
   트리거: 이전 요청 logprobs < 0.80 또는 NSN 패턴 필드
   삽입 위치: user_instruction 끝에 "[OCR 힌트] ..." 형식
```

**군수 서식 instruction 예시** (`supply_request` + `text` 영역):

```
System: You are a Korean military logistics document OCR assistant.
Extract the requested fields from the cropped document image.

Fields to extract:
- nsn (국가재고번호): NNNN-NN-NNN-NNNN format, 13 digits
- item_name (품목명): Korean item name
- quantity (수량): integer
- unit_price (단가): integer (KRW)
- requester (청구자): Korean name

Rules:
- NSN format: NNNN-NN-NNN-NNNN. Example: 1005-01-432-1234
- Unclear characters: mark as [?]
- Similar characters to watch: ㄱ/ㅋ, ㄴ/ㄹ, 1/ㅣ, 0/O, 5/S

[Example]
Image: <example_crop>
Output: {"analysis": "필체 선명, NSN 형식 확인됨", "nsn": "1005-01-432-1234", "quantity": 10}

```

**OCR 힌트 삽입 조건 (`ocr_hint_provider.py`)**:

```python
HINT_TRIGGER_THRESHOLD = 0.80  # logprobs 이하 또는 NSN 패턴 필드

# 힌트 삽입 예시
if should_insert_hint(region, prev_confidence):
    ocr_text = paddle_ocr.run(region.cropped_image)
    user_instruction += f"\n[OCR 힌트] 경량 OCR 인식 결과: {ocr_text}\n위 내용을 참고하여 보다 정확하게 추출하세요."
```

### 4-5. P2.5-C — ResolutionRouter

**pixel_budget 기준값**:

```python
PIXEL_BUDGET = {
    "table":             1120,  # 셀 경계·미세 글씨 → 최고 해상도
    "handwritten_field": 1120,  # 수기 기입란 — 560에서 상향 (한국어 획 구분 필수)
    "seal":               560,  # 원형 배치 텍스트
    "text":               560,  # 일반 텍스트 — 280에서 상향 (소형 폰트 대응)
    "formula":            280,
    "chart":              280,
    "figure":             140,
    "header":             140,  # 대형 텍스트, 저해상도로 충분
    "footer":             140,
}
```

**48px 배수 정렬 — SigLIP 패치 효율 최적화**:

SigLIP 인코더는 이미지를 16×16 패치로 분할 후 3×3 블록 단위(= 48px)로 average pooling합니다. 크롭 크기가 48px 배수가 아니면 남은 픽셀에 패딩이 발생하여 해당 토큰이 실제 콘텐츠 없는 패딩에 낭비됩니다.

```python
def align_to_48(image: np.ndarray) -> np.ndarray:
    """크롭 이미지를 48px 배수로 리사이즈하여 SigLIP 패치 낭비 방지."""
    h, w = image.shape[:2]
    new_h = max(48, (h // 48) * 48)
    new_w = max(48, (w // 48) * 48)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
```

**크롭 패딩 기준** (공백 패딩 금지, 원본 이미지 맥락 포함):

```python
CROP_PADDING_RATIO = {
    "table":             0.05,  # 열/행 헤더 포함
    "handwritten_field": 0.15,  # 양식 레이블·경계선 포함 (문맥 중요)
    "seal":              0.10,
    "text":              0.05,
    "default":           0.05,
}
```

### 4-6. P3-B — StructuredExtractor

```python
# 배치 호출 (temperature=0.0 — 결정론적 OCR)
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": crop_base64}},
            {"type": "text", "text": instruction_spec.user_instruction}
        ]
    } for crop in batch],
    extra_body={
        "guided_json": instruction_spec.json_schema,  # analysis 필드 최상단
        "logprobs": True,
        "top_logprobs": 5,
        "mm_processor_kwargs": {"max_soft_tokens": pixel_budget},
    },
    temperature=0.0,  # 결정론적 출력 — OCR 태스크 표준
)
```

**저신뢰 필드 재시도 로직**:

```python
RETRY_THRESHOLD = 0.60   # 이하면 재시도
MAX_RETRIES = 1          # 과도한 지연 방지
RETRY_BUDGET_MAP = {140: 280, 280: 560, 560: 1120, 1120: 1120}  # 한 단계 상향

def _should_retry(field: FieldValue) -> bool:
    return field.confidence < RETRY_THRESHOLD and not field.is_retry

def _retry_with_higher_budget(
    region: CroppedRegion,
    instruction_spec: InstructionSpec,
) -> FieldValue:
    higher_budget = RETRY_BUDGET_MAP[region.pixel_budget]
    # OCR 힌트도 함께 삽입하여 재시도 효과 극대화
    hint = ocr_hint_provider.run(region.cropped_image)
    return self._process_single(region, instruction_spec, pixel_budget=higher_budget, ocr_hint=hint)
```

**logprobs 기반 필드별 신뢰도 산출**:

```python
def calc_field_confidence(token_logprobs: list[float]) -> float:
    """기하 평균: exp(mean(logprobs))
    한국어 subword 토크나이제이션 특성상 개별 토큰 확률이 낮을 수 있어,
    임계값을 영어 대비 5~10% 낮게 설정.
    길이 편향 보정(length_factor)은 Phase 2 SFT 후 실측 데이터 기반으로 도입 예정.
    """
    if not token_logprobs:
        return 0.0
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    mean_logprob = max(mean_logprob, -20.0)
    return round(min(1.0, max(0.0, math.exp(mean_logprob))), 4)
```

### 4-7. OCR-augmented 전략 (`src/vlm/ocr_hint_provider.py`)

**배경**: KLOCR 논문(2025.10)에서 OCR 텍스트를 이미지와 함께 VLM에 제공하는 OCR-augmented 접근이 Key Information Extraction(군수 서식의 핵심 태스크)에서 가장 큰 개선을 보였습니다. 특히 군수 전문 용어나 의미 없는 코드(부대코드, NSN)처럼 VLM이 문맥으로 추론하기 어려운 필드에서 효과적입니다.

```python
class OCRHintProvider:
    """경량 PaddleOCR을 선행 실행하여 VLM에 텍스트 힌트 제공."""

    HINT_TRIGGER_THRESHOLD = 0.80   # logprobs 이하 영역에 힌트 삽입
    NSN_HINT_REGIONS = ["text", "handwritten_field"]  # NSN 패턴 필드 포함 영역

    def __init__(self):
        # PaddleOCR 한국어 (경량 mobile 모델, ~50ms/영역)
        self._ocr = PaddleOCR(lang="korean", use_gpu=True)

    def should_hint(
        self, region: LayoutRegion, prev_confidence: Optional[float] = None
    ) -> bool:
        """힌트 삽입 여부 결정.
        - 이전 호출 신뢰도가 낮은 경우
        - NSN 코드가 포함될 가능성이 있는 영역 타입
        """
        if prev_confidence is not None and prev_confidence < self.HINT_TRIGGER_THRESHOLD:
            return True
        if region.region_type in self.NSN_HINT_REGIONS:
            return True
        return False

    def get_hint(self, image_rgb: np.ndarray) -> str:
        """PaddleOCR 결과를 힌트 문자열로 변환."""
        results = self._ocr.ocr(image_rgb, cls=False)
        texts = [line[1][0] for res in results for line in res if res]
        return " ".join(texts)

    def build_hint_instruction(self, hint_text: str) -> str:
        return (
            f"\n[OCR 힌트] 경량 OCR 인식 결과: {hint_text}\n"
            "위 내용을 참고하여 보다 정확하게 추출하세요."
        )
```

**적용 조건 (선택적 사용으로 지연 최소화)**:
- 전체 영역 적용 시: 영역당 ~50ms 추가 → 대용량 문서에서 병목
- **선택적 적용 권장**: 저신뢰 영역(logprobs < 0.80) + NSN 패턴 필드에만 적용
- PaddleOCR이 틀려도 VLM이 참고만 하므로 오히려 오인식을 유발하지 않음

---

## 5. SPOF 대비 — 추론 가용성

### 5-1. 수준 A — vLLM 헬스체크

```yaml
healthcheck:
  test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 120s
restart: unless-stopped
```

### 5-2. 수준 B — 경량 Fallback 추론 환경

| 모델 | 역할 | 상태 |
|------|------|------|
| DiT | 서식 분류 | ⚠️ 가중치 미배치 |
| PP-OCRv5 Korean | 텍스트 인식 | ⚠️ 가중치 미배치 |
| SLANeXt_wired | 표 구조 인식 | ⚠️ 가중치 미배치 |

---

## 6. Docker 추론 환경

| 서비스 | 포트 | 용도 | 프레임워크 |
|--------|------|------|-----------|
| `vllm-server` | 8100 | Gemma4 vLLM | vLLM v0.19.0 + transformers 5.5.0 |
| `layout` | 8082 | P2 레이아웃 (Fusion 지원) | PaddlePaddle CUDA 12.6 |
| `pipeline` | 8080 | PdfAdapter, P1, P2.5-A~C, P3-A~B, P4~P6 | PyTorch CUDA 12.6 |
| `fallback` | 8081 | v1 PP-OCRv5 경량 파이프라인 | PaddlePaddle + PyTorch CUDA 12.6 |
| `train` | — | Fine-tuning | PaddlePaddle + PyTorch + PEFT |

> **CUDA 12.6**: H100(sm_90) 지원. PaddlePaddle CUDA 11.8 빌드의 커널 미포함 오류(error 209) 방지.

---

## 7. 하드웨어 요구사항

| 용도 | VRAM | 권장 GPU | 비고 |
|------|------|---------|------|
| Gemma4 추론 (BF16 + FP8 KV Cache) | ~48GB + KV절감 | H100 80GB | 현재 운영 — **최적 조합** |
| Gemma4 추론 (AWQ 4-bit + Marlin) | ~16GB | RTX 4090 24GB | Phase 3 실험 |
| Gemma4 추론 (FP8 Dynamic) | ~27GB | A100 40GB | vllm#39049 해소 후 |
| PP-DocLayout 추론 | 4GB | GPU 4GB+ | PaddlePaddle |
| Real-ESRGAN SR | 2GB | GPU 4GB+ | 타일 기반 |
| PaddleOCR (OCR 힌트) | 1GB | GPU (선택) | 경량 mobile 모델 |
| Fallback (T3+T4+T5) | 2~4GB | GPU 4GB+ | v1 PP-OCRv5 |

### 양자화 옵션 (Phase 3 실험)

| 양자화 | VRAM | 품질 손실 | vLLM 호환 | 주의 |
|--------|------|----------|----------|------|
| BF16 가중치 + FP8 KV Cache (현재) | ~48GB + KV절감 | 기준 | ✅ | **권장** |
| AWQ 4-bit | ~16GB | ~1-2% | ✅ | 한국어 비라틴 문자 불균형 손실 주의. 비전-언어 혼합 캘리브레이션 필수. **Marlin 커널 활성화 필수** (~68 tok/s → ~741 tok/s) |
| FP8 Dynamic | ~27GB | ~0.3% | vllm#39049 | 해소 후 실험 |

### Upstream 버그 트래킹

| 이슈 | 영향 | 상태 | 확인 트리거 | 대안 |
|------|------|------|-----------|------|
| `vllm#39049` (FP8 gibberish) | FP8 Dynamic 실험 차단 | Open (2026-04) | Phase 3 착수 + vLLM 릴리스마다 | AWQ 4-bit + Marlin 우선 |

---

## 8. 추론 파이프라인 스크립트

| 스크립트 | 범위 |
|----------|------|
| `scripts/run_pipeline.py` | P1→P6 전체 (PDF 입력 포함) |
| `scripts/test_vlm.py` | P3-A/P3-B VLM 단독 테스트 |
| `scripts/run_pipeline_with_outputs.py` | 전체 + 단계별 출력 저장 |
| `scripts/evaluate_layout_detection.py` | PP-DocLayout 검출률 측정 |

---

## 9. 추가 의존성 (PDF 어댑터)

```txt
# requirements.txt — pipeline 서비스 추가 항목
pymupdf>=1.24.0   # PyPI 패키지명 pymupdf, import 시 fitz 사용
                  # PDF → PageImage 변환 (PdfAdapter)
                  # C 바이너리 포함 — 별도 시스템 패키지 불필요
```

**폐쇄망 배치**:

```bash
# 온라인 환경에서 wheel 사전 다운로드 (CUDA 무관, 순수 Python/C 패키지)
pip download "pymupdf>=1.24.0" -d ./wheels/

# Dockerfile.pipeline 에 추가
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels pymupdf
```

**pymupdf 선택 이유**:

| 항목 | pymupdf | pdf2image | pypdfium2 |
|------|---------|-----------|-----------|
| 시스템 의존성 | 없음 (C 바이너리 포함) | poppler 필요 (`apt-get`) | 없음 |
| 렌더링 품질 | ✅ 최고 | 🟡 양호 | ✅ 좋음 |
| DPI 지정 | ✅ | ✅ | ✅ |
| 폐쇄망 wheel 배치 | ✅ 단순 | ❌ apt 별도 필요 | ✅ 단순 |
| 텍스트 레이어 접근 | ✅ (스캔 판정용) | ❌ | ❌ |
