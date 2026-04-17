# PDF 어댑터 설계

> `src/input/pdf_adapter.py`
> PDF 입력을 파이프라인이 처리할 수 있는 단위로 변환합니다.

---

## 1. 설계 원칙

### 1-1. 어댑터의 책임 범위

PdfAdapter는 **입력 포맷 변환만** 담당합니다.

```
담당:
  - PDF 바이트 → 페이지별 RGB 이미지(numpy array) 변환
  - 렌더링 DPI 결정 (PDF 내장 해상도 기반)
  - 페이지 메타데이터 추출 (페이지 수, 원본 크기)
  - 암호화 PDF 감지 → 오류 반환

담당하지 않음:
  - 이미지 화질 보정 (P1 담당)
  - 텍스트 레이어 추출 (방침 A — 이미지 렌더링 전용)
  - 서식 분류 (P3-A 담당)
  - 페이지 합산/집계 로직 (오케스트레이터 담당)
```

### 1-2. 멀티페이지 처리 방식 (결정: 1-B)

PDF 전체를 **하나의 문서**로 처리합니다. 단, 기존 P1~P6 컴포넌트는 단일 이미지(ndarray) 기반으로 설계되어 있으므로, 멀티페이지 루프는 **오케스트레이터 레벨**에서 처리합니다. 기존 컴포넌트 변경 없음.

```
orchestrator.process(doc: DocumentInput)
    │
    └── file_ext == PDF:
        PdfAdapter.render() → list[PageImage]
        for page in pages:
            _process_single_image(page → DocumentInput) → PipelineOutput
        → PdfDocumentResult 병합 후 반환
```

---

## 2. doc_id 체계 및 DB 키 설계 (M3)

### 2-1. doc_id 명명 규칙

페이지별 `doc_id`는 `{원본문서코드}_p{페이지번호:02d}` 형식을 사용합니다.

```
원본 PDF doc_id:  "전비품확인서_001"
페이지 1 doc_id:  "전비품확인서_001_p01"
페이지 2 doc_id:  "전비품확인서_001_p02"
페이지 10 doc_id: "전비품확인서_001_p10"
```

같은 PDF에서 파생된 레코드는 `doc_id` 접두사로 그룹 식별이 가능합니다.

```python
# DB에서 같은 PDF의 모든 페이지 조회 예시
SELECT * FROM ocr_results WHERE doc_id LIKE '전비품확인서_001_p%'

# 또는 parent_doc_id 칼럼으로 조회 (아래 §2-2 참조)
SELECT * FROM ocr_results WHERE parent_doc_id = '전비품확인서_001'
```

### 2-2. DB 레코드 스키마 변경 (P6 DBLoader)

페이지별 개별 레코드를 유지하되, 같은 문서임을 확인할 수 있는 `parent_doc_id`와 `page_number` 컬럼을 추가합니다.

```sql
-- 기존 ocr_results 테이블에 컬럼 추가
ALTER TABLE ocr_results ADD COLUMN parent_doc_id TEXT;
ALTER TABLE ocr_results ADD COLUMN page_number   INTEGER;
ALTER TABLE ocr_results ADD COLUMN total_pages   INTEGER;

-- 단일 이미지 입력의 경우: parent_doc_id = NULL, page_number = NULL
-- PDF 페이지 입력의 경우:
--   doc_id         = "전비품확인서_001_p01"
--   parent_doc_id  = "전비품확인서_001"
--   page_number    = 1
--   total_pages    = 2
```

`PipelineOutput.metadata`를 통해 P6 DBLoader에 전달합니다.

```python
# 오케스트레이터 _process_pdf() 내부 — 페이지 DocumentInput 생성 시
page_input = DocumentInput(
    doc_id=f"{doc.doc_id}_p{page.page_number:02d}",
    ...
    metadata={
        **doc.metadata,
        "parent_doc_id":  doc.doc_id,          # P6가 읽어 DB에 기록
        "page_number":    page.page_number,
        "total_pages":    page.total_pages,
        "pdf_source":     True,
    },
)
```

P6 DBLoader는 `metadata["pdf_source"] == True`일 때 `parent_doc_id`, `page_number`, `total_pages`를 DB에 함께 기록합니다.

---

## 3. 인터페이스 타입 (신규 추가)

> `src/interfaces/types.py`에 추가

```python
@dataclass
class PageImage:
    """PdfAdapter가 반환하는 단일 페이지 이미지."""
    doc_id: str
    page_number: int           # 1-based
    total_pages: int
    image_array: np.ndarray    # H×W×3 uint8 (RGB)
    render_dpi: int            # 실제 렌더링 DPI
    original_width_pt: float   # PDF 원본 너비 (포인트 단위, 1pt = 1/72 inch)
    original_height_pt: float  # PDF 원본 높이 (포인트 단위)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PdfDocumentResult:
    """멀티페이지 PDF 처리 결과 — 페이지별 PipelineOutput 집합."""
    doc_id: str                          # 원본 PDF doc_id
    total_pages: int
    pages: list[PipelineOutput]          # 페이지 순서 유지, doc_id = "{원본}_p{N:02d}"
    page_results: list[Any]              # 원시 페이지별 결과 (서버 응답, 디버깅용)
    overall_status: PipelineStatus       # 집계 상태 (§4 참조)
    processing_ms: float
    warnings: list[str] = field(default_factory=list)
```

> `page_results`는 구현 단계에서 추가된 필드로, `pages`가 P1~P6 `PipelineOutput`을 담는 반면 `page_results`는 오케스트레이터 내부 집계 이전의 원시 처리 결과(페이지별 raw 서버 응답, 디버그 로그 포함)를 보존합니다. 프론트엔드는 `pages`만 참조하면 되고, `page_results`는 장애 분석 및 저수준 디버깅 용도입니다.

> `src/interfaces/enums.py` — `PipelineStatus.PARTIAL` 추가 (M1):

```python
class PipelineStatus(str, Enum):
    SUCCESS        = "success"
    PARTIAL        = "partial"        # 신규 — PDF 멀티페이지 중 일부 실패/검토
    REVIEW         = "review"
    FAILED         = "failed"
    OTHER_DOCUMENT = "other_document"
```

> `FileExt` enum — `pdf` 항목 확인/추가:

```python
class FileExt(str, Enum):
    jpg  = "jpg"
    jpeg = "jpeg"
    png  = "png"
    tiff = "tiff"
    pdf  = "pdf"
```

---

## 4. 전체 상태(overall_status) 집계 정책

PDF 전체에 대한 `overall_status`는 아래 우선순위로 결정합니다.

| 조건 | overall_status |
|------|---------------|
| 모든 페이지 `success` | `success` |
| 1개 이상 `review`, 나머지 `success` | `partial` |
| 1개 이상 `failed`, 나머지 성공/검토 | `partial` |
| 모든 페이지 `failed` | `failed` |
| PdfAdapter 자체 실패 (렌더링 불가) | `failed` |

```python
def _aggregate_status(page_results: list[PipelineOutput]) -> PipelineStatus:
    statuses = {r.status for r in page_results}
    if statuses == {PipelineStatus.SUCCESS}:
        return PipelineStatus.SUCCESS
    if statuses == {PipelineStatus.FAILED}:
        return PipelineStatus.FAILED
    return PipelineStatus.PARTIAL
```

---

## 5. 검토 큐 표기 방식 (M4)

PDF 처리 시 검토 큐는 **페이지 단위로 개별 적재**합니다.

```
PDF "전비품확인서_001" (2페이지)
  ├── p01: overall_confidence=0.92 → 통과, 큐 미적재
  └── p02: overall_confidence=0.54 → 큐 적재

검토 큐 항목:
  queue_id:    "RQ-20260415-003"
  doc_id:      "전비품확인서_001_p02"   ← 페이지 식별 가능
  parent_doc_id: "전비품확인서_001"     ← 원본 PDF 추적 가능
  page_number:  2
  total_pages:  2
  priority:    "normal"
```

검토 큐 UI에서 담당자가 `전비품확인서_001_p02`를 열면 해당 페이지 이미지와 추출 결과를 보여줍니다. `parent_doc_id`로 같은 PDF의 다른 페이지 결과도 연결해서 볼 수 있도록 UI에서 링크를 제공할 수 있습니다 (Phase 3 고도화 대상).

`ReviewQueueItem`에 필드 추가:

```python
@dataclass
class ReviewQueueItem:
    queue_id: str
    doc_id: str                            # "{원본}_p{N:02d}" 형식
    parent_doc_id: Optional[str] = None   # 신규 — PDF 원본 문서 ID
    page_number: Optional[int] = None     # 신규 — 페이지 번호
    total_pages: Optional[int] = None     # 신규 — 전체 페이지 수
    # ... 기존 필드 유지
```

단일 이미지 입력의 경우 `parent_doc_id = None`, `page_number = None`으로 유지 — 기존 동작 그대로.

---

## 6. PdfAdapter 구현 설계

> `src/input/pdf_adapter.py`

```python
import fitz  # pymupdf
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.interfaces.types import PageImage
from src.interfaces.enums import FileExt


# 렌더링 DPI 기준값
RENDER_DPI_DEFAULT = 300   # 군수 서식 표준 (A4 @ 300dpi = 2480×3508px)
RENDER_DPI_HIGH    = 400   # 호출자 명시 시 상향 가능
RENDER_DPI_MIN     = 150   # 이 이하면 P1 SR 대상

# 스캔 PDF 판정: 페이지당 텍스트 문자 수가 이 이하 + 이미지 오브젝트 존재
SCAN_PDF_TEXT_THRESHOLD = 50


class PdfAdapter:
    """PDF 바이트 → PageImage 리스트 변환.

    텍스트 레이어는 무시하고 항상 이미지 렌더링 (방침 A).
    멀티페이지는 전부 렌더링하여 반환.
    암호화 PDF, 손상 PDF는 PdfAdapterError 발생.
    """

    def __init__(
        self,
        render_dpi: int = RENDER_DPI_DEFAULT,
        colorspace: str = "RGB",   # RGB 고정 (SigLIP 인코더 기준)
    ):
        self._dpi = render_dpi
        self._cs  = fitz.csRGB if colorspace == "RGB" else fitz.csGRAY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, doc_id: str, raw_bytes: bytes) -> list[PageImage]:
        """PDF 바이트를 페이지별 PageImage 리스트로 변환.

        Args:
            doc_id:    문서 식별자 (원본 PDF 단위)
            raw_bytes: PDF 원본 바이트

        Returns:
            list[PageImage] — 페이지 순서 보장 (1-based page_number)

        Raises:
            PdfAdapterError: 암호화, 손상, 빈 파일 등 복구 불가 오류
        """
        try:
            pdf = fitz.open(stream=raw_bytes, filetype="pdf")
        except Exception as e:
            raise PdfAdapterError(f"PDF 열기 실패: {e}") from e

        if pdf.is_encrypted:
            raise PdfAdapterError("암호화된 PDF — 복호화 키 없음")

        if pdf.page_count == 0:
            raise PdfAdapterError("페이지 수 0 — 빈 PDF")

        total = pdf.page_count
        results: list[PageImage] = []

        for i in range(total):
            page_warnings: list[str] = []
            page = pdf[i]

            effective_dpi = self._decide_dpi(page, page_warnings)
            image_array   = self._render_page(page, effective_dpi)

            results.append(PageImage(
                doc_id=doc_id,
                page_number=i + 1,
                total_pages=total,
                image_array=image_array,
                render_dpi=effective_dpi,
                original_width_pt=page.rect.width,
                original_height_pt=page.rect.height,
                warnings=page_warnings,
            ))

        pdf.close()
        return results

    def is_pdf(self, raw_bytes: bytes) -> bool:
        """매직 바이트로 PDF 여부 확인 (확장자 없는 경우 대비)."""
        return raw_bytes[:4] == b"%PDF"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decide_dpi(self, page: fitz.Page, warnings: list[str]) -> int:
        """페이지 특성 기반 렌더링 DPI 결정.

        스캔 PDF 여부와 무관하게 RENDER_DPI_DEFAULT(300)를 사용.
        내장 이미지 DPI가 RENDER_DPI_MIN 미만이면 P1 SR 예정임을 warnings에 기록.
        텍스트도 이미지도 없는 페이지는 빈 페이지 가능성을 경고.
        """
        text_len = len(page.get_text("text").strip())
        img_list = page.get_images(full=False)
        is_scan  = text_len < SCAN_PDF_TEXT_THRESHOLD and len(img_list) > 0

        if not is_scan and text_len < SCAN_PDF_TEXT_THRESHOLD:
            warnings.append(
                f"텍스트 레이어 없음({text_len}자), 이미지도 없음 — 빈 페이지 가능성"
            )

        # 스캔 PDF: 내장 이미지 해상도 역산으로 원본 DPI 추정
        if is_scan and img_list:
            xref = img_list[0][0]
            try:
                img_info      = page.parent.extract_image(xref)
                img_w_px      = img_info["width"]
                page_w_pt     = page.rect.width   # 포인트 (1pt = 1/72 inch)
                estimated_dpi = int(img_w_px / (page_w_pt / 72))
                if estimated_dpi < RENDER_DPI_MIN:
                    warnings.append(
                        f"원본 이미지 추정 DPI={estimated_dpi}"
                        f"(< {RENDER_DPI_MIN}) — P1 SR 대상"
                    )
            except Exception:
                pass  # DPI 역산 실패 시 무시, 렌더링 계속

        return self._dpi

    def _render_page(self, page: fitz.Page, dpi: int) -> np.ndarray:
        """fitz.Page → H×W×3 uint8 RGB numpy 배열.

        zoom factor = dpi / 72  (PDF 기본 단위: 포인트, 1pt = 1/72 inch)
        """
        zoom   = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix, colorspace=self._cs, alpha=False)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, 3)
        return img.copy()  # fitz 내부 버퍼 참조 해제


class PdfAdapterError(Exception):
    """복구 불가 PDF 처리 오류."""
    pass
```

---

## 7. DocumentInput — 변경 없음

```python
# 기존 그대로 — PDF 바이트를 raw_bytes에 그대로 담아서 전달
@dataclass
class DocumentInput:
    doc_id: str
    raw_bytes: bytes           # PDF의 경우 PDF 원본 바이트 그대로
    file_ext: FileExt          # FileExt.pdf
    source_type: SourceType
    dpi_hint: Optional[int] = None   # PDF 렌더링 DPI 힌트로 재활용
    metadata: dict = field(default_factory=dict)
```

---

## 8. 오케스트레이터 변경

> `src/pipeline/orchestrator.py`

```python
class Orchestrator:

    def process(
        self, doc: DocumentInput
    ) -> PipelineOutput | PdfDocumentResult:
        """단일 진입점.

        file_ext == pdf → _process_pdf() → PdfDocumentResult
        그 외           → _process_single_image() → PipelineOutput
        """
        if doc.file_ext == FileExt.pdf:
            return self._process_pdf(doc)
        return self._process_single_image(doc)

    # ------------------------------------------------------------------

    def _process_pdf(self, doc: DocumentInput) -> PdfDocumentResult:
        """PDF 전체를 하나의 문서로 처리.

        각 페이지는 독립적으로 파이프라인을 통과하며,
        한 페이지의 실패가 다른 페이지에 영향을 주지 않습니다.
        """
        start_ms = _now_ms()
        adapter  = PdfAdapter(render_dpi=doc.dpi_hint or RENDER_DPI_DEFAULT)

        try:
            pages = adapter.render(doc.doc_id, doc.raw_bytes)
        except PdfAdapterError as e:
            return PdfDocumentResult(
                doc_id=doc.doc_id,
                total_pages=0,
                pages=[],
                overall_status=PipelineStatus.FAILED,
                processing_ms=_now_ms() - start_ms,
                warnings=[str(e)],
            )

        page_results: list[PipelineOutput] = []

        for page in pages:
            page_doc_id = f"{doc.doc_id}_p{page.page_number:02d}"

            page_input = DocumentInput(
                doc_id=page_doc_id,
                raw_bytes=_ndarray_to_png_bytes(page.image_array),
                file_ext=FileExt.png,
                source_type=doc.source_type,
                dpi_hint=page.render_dpi,
                metadata={
                    **doc.metadata,
                    "parent_doc_id": doc.doc_id,       # M3: DB 그룹 키
                    "page_number":   page.page_number,  # M3: 페이지 번호
                    "total_pages":   page.total_pages,  # M3: 전체 페이지 수
                    "pdf_source":    True,              # P6가 PDF 처리임을 인식
                },
            )
            result = self._process_single_image(page_input)
            page_results.append(result)

        return PdfDocumentResult(
            doc_id=doc.doc_id,
            total_pages=len(pages),
            pages=page_results,
            overall_status=_aggregate_status(page_results),
            processing_ms=_now_ms() - start_ms,
        )

    def _process_single_image(self, doc: DocumentInput) -> PipelineOutput:
        """기존 단일 이미지 처리 — 변경 없음."""
        ...


# ------------------------------------------------------------------
# 헬퍼 함수
# ------------------------------------------------------------------

def _ndarray_to_png_bytes(image: np.ndarray) -> bytes:
    """numpy RGB 배열 → PNG 바이트 (메모리 내 변환, 디스크 기록 없음)."""
    bgr     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("PNG 인코딩 실패")
    return buf.tobytes()


def _aggregate_status(
    page_results: list[PipelineOutput],
) -> PipelineStatus:
    """페이지별 상태 집계 → PDF 전체 상태 결정."""
    statuses = {r.status for r in page_results}
    if statuses == {PipelineStatus.SUCCESS}:
        return PipelineStatus.SUCCESS
    if statuses == {PipelineStatus.FAILED}:
        return PipelineStatus.FAILED
    return PipelineStatus.PARTIAL


def _now_ms() -> float:
    import time
    return time.monotonic() * 1000
```

---

## 9. 결과 저장 구조

PDF 입력 시 `data/pipeline_outputs/{YYYYMMDD_HHMMSS}/` 구조:

```
{doc_id}/                          ← "전비품확인서_001" (PDF 원본)
├── pdf_summary.json               ← 전체 페이지 집계
│     {
│       "doc_id": "전비품확인서_001",
│       "total_pages": 2,
│       "overall_status": "partial",
│       "pages": [
│         {"page": 1, "doc_id": "전비품확인서_001_p01", "status": "success"},
│         {"page": 2, "doc_id": "전비품확인서_001_p02", "status": "review"}
│       ]
│     }
├── p01/                           ← 기존 단일 문서 구조 그대로
│   ├── P1/result.json
│   ├── P2/result.json
│   ├── ...
│   └── P6/result.json
└── p02/
    ├── P1/result.json
    ├── ...
    └── P6/result.json
```

---

## 10. 파급 범위 정리

| 파일 | 변경 내용 | 크기 |
|------|---------|------|
| `src/input/pdf_adapter.py` | **신규 생성**. `_render_page`는 `pix.n` 기반 동적 채널 수로 렌더링 | — |
| `src/interfaces/types.py` | `PageImage`, `PdfDocumentResult(page_results 포함)` 추가 | 소 |
| `src/interfaces/enums.py` | `PipelineStatus.PARTIAL` 추가 (M1) | 최소 |
| `src/pipeline/orchestrator.py` | `process()` 분기 + `_process_pdf()` + `_aggregate_pdf_status`(OTHER_DOCUMENT 처리 포함) | 중 |
| `src/postprocess/db_loader.py` | `parent_doc_id`, `page_number`, `total_pages` 컬럼 + `assembled_json` Text 컬럼, per-URL `_db_cache` 캐시 | 소 |
| `src/postprocess/review_queue.py` | `ReviewQueueItem`에 `parent_doc_id`, `page_number`, `total_pages` 추가. `queue_id` 형식: `RQ-YYYYMMDD-HHMMSS-{doc_id}-{uuid[:6]}`. `avg_wait_minutes` 구현 | 소 |
| `src/interfaces/__init__.py` | 신규 타입 export 추가 | 최소 |
| P1~P6 컴포넌트 전체 | **변경 없음** | — |
| `docker/Dockerfile.pipeline` | `pymupdf>=1.24.0` 추가 | 1줄 |
| DB 스키마 | `ocr_results`에 `parent_doc_id/page_number/total_pages/assembled_json` 컬럼 추가 | 소 |

---

## 11. 의존성 추가

```txt
# requirements.txt
pymupdf>=1.24.0   # PyPI 패키지명 pymupdf, import 시 fitz 사용
```

**폐쇄망 배치**:
```bash
# 온라인 환경에서 wheel 사전 다운로드
pip download pymupdf>=1.24.0 -d ./wheels/

# 폐쇄망 설치
pip install --no-index --find-links=./wheels pymupdf
```
