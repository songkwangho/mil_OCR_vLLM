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
| 활성 파라미터 | 4B (MoE) |
| 컨텍스트 길이 | 256K 토큰 |
| 비전 입력 | 가변 해상도 70~1120 토큰/이미지 |
| 한국어 지원 | ✅ (40+ 언어) |
| 라이선스 | Apache 2.0 |
| 추론 VRAM (BF16) | ~48GB |
| MTP 헤드 | ❌ 미포함 (Speculative Decoding은 n-gram 방식 사용) |

**pixel_budget ↔ 유효 해상도 매핑** (정사각 이미지 기준):

| pixel_budget | 유효 해상도 | 권장 사용처 |
|:---:|:---:|------|
| 140 | ~189px | 서식 분류 (P3-A), 헤더/푸터 |
| 280 | ~268px | 수식, 차트 |
| 560 | ~378px | 일반 텍스트, 인장 |
| 1120 | ~535px | 표, 수기 기입란 (핵심 영역) |

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
| `--kv-cache-dtype fp8` | KV Cache 메모리 50% 절감 → 동시처리 2배 | H100 네이티브 FP8 텐서 코어 가속 |
| `--max-num-seqs 64` | 멀티모달 배치 안정성 | 128→64로 조정 |
| `--max-num-batched-tokens 16384` | Chunked Prefill 최적화 | 온라인 서빙 기준 |
| `--mm-cache-preprocessor` | 멀티모달 prefix caching 활성화 | 동일 system_prompt 재사용 시 TTFT 3~10배 단축 |
| `--guided-decoding-backend xgrammar` | 반복 스키마 캐싱 최적 | 군수 서식 동일 스키마 반복 사용에 유리 |
| n-gram speculative decoding | 디코드 속도 1.2~1.5배 향상 | 구조화 JSON의 반복 패턴 활용, guided_json과 호환 |

### 3-3. Prefix Caching 전략

군수 서식 파이프라인에서 system_prompt는 서식 유형별로 동일하게 반복됩니다. 이를 최대한 활용합니다.

```
첫 번째 요청:  [system_prompt + 1-shot 예시] + [이미지 토큰] → KV 블록 생성
이후 동일 유형: [캐시된 KV 블록 재사용]     + [이미지 토큰] → TTFT 대폭 단축
```

**주의**: 멀티모달 prefix caching에서 동일 텍스트 + 다른 이미지 조합 시 캐시 키 충돌 버그(vllm#20261) 보고됨. `--mm-cache-preprocessor` 활성화로 이미지 해시 기반 캐시 사용 권장.

### 3-4. Guided Decoding 백엔드 선택

| 상황 | 권장 백엔드 | 이유 |
|------|-----------|------|
| 동일 서식 스키마 반복 (기본) | `xgrammar` | 스키마 캐싱 → throughput 우수 |
| NSN regex 패턴 강제 필요 | `guidance` | `pattern` 제약 완전 지원, 타임아웃 없음 |

xgrammar는 JSON Schema 내 `pattern` (regex) 제약 미지원. NSN 형식 강제는 후처리 검증(P4) 또는 `guidance` 백엔드 전환으로 처리.

---

## 4. 컴포넌트별 추론 설정

### 4-1. P1 — 화질 보정 + SR

- LOW DPI (< 150): Real-ESRGAN x2 초해상도 적용
- CLAHE 대비 강화 → 기울기 보정(Deskew) → 품질 점수 산출
- **이진화 금지**: SigLIP 인코더는 RGB 자연 이미지 기준으로 학습됨
- **RGB 유지**: 그레이스케일 변환 금지

### 4-2. P2 — 레이아웃 탐지

- Fusion OFF: PP-DocLayoutV3 단독 (기본)
- Fusion ON (`LAYOUT_FUSION_MODE=true`):
  - DPI ≥ 150: V3(구조) + plus-L(텍스트), 50% 겹침 필터
  - DPI < 150: V3(구조) + OpenCV heuristic(텍스트), 50% 겹침 필터
- Layout 추론은 별도 컨테이너 (:8082) HTTP 호출

**PP-DocLayout 비교**:

| 항목 | plus-L | V3 (기본) |
|------|--------|----------|
| 아키텍처 | RT-DETR | Mask RT-DETR |
| 출력 | bbox | bbox + polygon + reading order |
| 라벨 | 20종 | 25종 |
| 읽기 순서 | 좌표 휴리스틱 | 모델 예측 |

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
        "guided_json": FORM_TYPE_SCHEMA,  # enum 포함: ...| unknown | other
        "mm_processor_kwargs": {"max_soft_tokens": 140},
    },
)
```

### 4-4. P3-B — StructuredExtractor

**현재 구현 (Phase 1)**: pixel_budget 그룹 내 영역을 **순차** 호출합니다. `for cropped in regions: self._process_single(cropped, ...)`.

```python
# 현재: 순차 호출 (Phase 1-E 잔존 이슈 #2 — Phase 2-C에서 asyncio 배치 전환 예정)
for cropped in regions:
    response = client.chat.completions.create(
        model="gemma-4-26b-a4b-it",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": crop_base64}},
                {"type": "text", "text": instruction_spec.user_instruction}
            ]
        }],
        extra_body={
            "guided_json": instruction_spec.json_schema,  # analysis 필드 최상단
            "logprobs": True,
            "top_logprobs": 5,
            "mm_processor_kwargs": {"max_pixels": pixel_budget * 28 * 28},
        },
    )
```

> **영향**: 17 영역 페이지에서 영역당 약 0.9초 × 17 = 약 15초. Phase 2-C에서 `asyncio.gather()` + vLLM `max-num-seqs=64` 제약 내 동시 전송으로 전환하면 3~5배 단축 예상.

**LoRA adapter 로드 (Fine-tuning 후)**:

```python
# P3-A 추론 시
vlm_client.load_adapter("models/gemma4/adapters/form_classifier_v1/")

# P3-B 추론 시
vlm_client.load_adapter("models/gemma4/adapters/structured_extractor_v1/")
```

---

## 5. Docker 추론 환경

| 서비스 | 포트 | 용도 | 프레임워크 | 헬스체크 |
|--------|------|------|-----------|---------|
| `vllm-server` | 8100 | Gemma4 vLLM | vLLM v0.19.0 + transformers 5.5.0 | `/health` (30s) |
| `layout` | 8082 | P2 레이아웃 (Fusion 지원) | PaddlePaddle CUDA 12.6 | `/health` (30s) |
| `pipeline` | 8080 | P1, P2.5-A~C, P3-A~B, P4~P6 | PyTorch CUDA 12.6 | `/health` (30s) |
| `fallback` | 8081 | v1 PP-OCRv5 경량 파이프라인 | PaddlePaddle + PyTorch CUDA 12.6 | `/health` (30s) |
| `train` | — | Fine-tuning | PaddlePaddle + PyTorch + PEFT | — |

> **CUDA 12.6**: H100(sm_90)에서 CUDA 11.8 빌드의 커널 미포함 오류(error 209) 방지.

```yaml
# docker-compose.yml vllm-server 서비스 (2026-04-10 최적화 옵션 반영 완료)
vllm-server:
  command: >
    vllm serve models/gemma4/gemma-4-26b-a4b-it/
    --dtype bfloat16
    --max-model-len 8192
    --gpu-memory-utilization 0.92
    --kv-cache-dtype fp8
    --max-num-seqs 64
    --max-num-batched-tokens 16384
    --mm-cache-preprocessor
    --guided-decoding-backend xgrammar
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":1}'
    --disable-log-requests
  healthcheck:
    test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 120s
  restart: unless-stopped
```

> ⚠️ **재기동 필요**: docker-compose.yml 파일 수정은 반영되었지만 실제 실행 중인 컨테이너는 구 옵션으로 기동된 상태입니다. 새 옵션을 적용하려면:
> ```bash
> docker compose restart vllm-server
> # 또는 완전 재생성
> docker compose up -d --force-recreate vllm-server
> ```
> 재기동 시 ~120초의 모델 로드 시간이 필요하며, `/health` endpoint가 200을 반환할 때까지 대기 후 사용하세요. 재기동 전후 동일 문서 처리 시간 비교는 CLAUDE.md 로드맵 Phase 1 잔존 이슈 #3에 포함되어 있습니다.

---

## 6. 하드웨어 요구사항

| 용도 | VRAM | 권장 GPU | 비고 |
|------|------|---------|------|
| Gemma4 추론 (BF16 + FP8 KV) | ~48GB + KV절감 | H100 80GB | 현재 운영 |
| Gemma4 추론 (AWQ 4-bit) | ~16GB | RTX 4090 24GB | Phase 3 실험 |
| Gemma4 추론 (FP8 Dynamic) | ~27GB | A100 40GB | vllm#39049 해소 후 |
| PP-DocLayout | 4GB | GPU 4GB+ | PaddlePaddle |
| Real-ESRGAN SR | 2GB | GPU 4GB+ | 타일 기반 |
| Fallback (T3+T4+T5) | 2~4GB | GPU 4GB+ | v1 PP-OCRv5 |

### 양자화 옵션 (Phase 3 실험)

| 양자화 | VRAM | 품질 손실 | vLLM 호환 | 주의 |
|--------|------|----------|----------|------|
| BF16 (현재) | ~48GB | 기준 | ✅ | — |
| AWQ 4-bit | ~16GB | ~1-2% | ✅ | 한국어 비라틴 문자 불균형 손실 주의. 비전-언어 혼합 캘리브레이션 필수 |
| FP8 Dynamic | ~27GB | ~0.3% | vllm#39049 | 해소 후 실험 |

> **AWQ 한국어 주의**: 영어 전용 캘리브레이션 데이터(Pile 등) 사용 시 VLM OCR 정확도 최대 10% 저하. 반드시 비전-언어 혼합 데이터로 캘리브레이션.

### Upstream 버그 트래킹

| 이슈 | 영향 | 상태 | 대안 |
|------|------|------|------|
| `vllm#39049` (FP8 gibberish) | FP8 실험 차단 | Open (2026-04) | AWQ 4-bit 우선 |

---

## 7. 추론 파이프라인 스크립트

| 스크립트 | 범위 |
|----------|------|
| `scripts/run_pipeline.py` | P1→P6 전체 |
| `scripts/test_vlm.py` | P3-A/P3-B VLM 단독 테스트 |
| `scripts/run_pipeline_with_outputs.py` | 전체 + 단계별 출력 저장 |
| `scripts/evaluate_layout_detection.py` | PP-DocLayout 검출률 측정 |
